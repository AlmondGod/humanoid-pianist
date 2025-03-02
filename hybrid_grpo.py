import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import threading
import wandb  # Add wandb import
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, List, Any
from dm_env import TimeStep

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], min_log_std: float = -2.0):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.policy = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std = min_log_std  # Prevent log_std from getting too small
        
    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple:
        """Forward pass through the actor network
        
        Args:
            state: State tensor
            action: Optional action tensor. If provided, also returns log_prob of that action
        
        Returns:
            mean: Action mean
            std: Action standard deviation
            log_prob: Log probability of the action (if action was provided)
        """
        mean = self.policy(state)
        # Clamp log_std to prevent it from getting too small (which would make actions deterministic)
        # This helps maintain exploration and prevents action sampling collapse
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        std = torch.exp(log_std)
        
        # If action is provided, also compute log_prob
        if action is not None:
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            return mean, std, log_prob
        
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.value = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(state)

class HybridGRPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        num_samples: int = 8,  # Number of actions to sample per state
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        max_workers: int = 4,  # Maximum number of worker threads for parallel evaluation
        mini_batch_size: int = 16,  # Maximum states to process in a single mini-batch
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_log_std: float = -2.0,  # Minimum log standard deviation to prevent action space collapse
        use_lr_scheduler: bool = True  # Whether to use learning rate scheduling
    ):
        print("Initializing HybridGRPO with parallel action evaluation")
        self.actor = Actor(state_dim, action_dim, hidden_dims, min_log_std=min_log_std).to(device)
        self.critic = Critic(state_dim, hidden_dims).to(device)
        
        # Use different learning rates for actor and critic to address gradient issues
        # Reduce actor learning rate significantly to prevent gradient explosion
        actor_lr = lr / 50.0  # 50x smaller learning rate for actor (was 10x)
        critic_lr = lr  # Keep original learning rate for critic
        
        # Add weight decay to the actor optimizer to further stabilize training
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Add learning rate schedulers for more stable training
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            # Warm up learning rate for 5% of training then decay it
            warmup_steps = 5000  # Warm up over first 5000 steps
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(
                self.actor_optimizer,
                lambda steps: min(1.0, steps / warmup_steps) * max(0.1, 1.0 - (steps - warmup_steps) / 100000)
            )
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(
                self.critic_optimizer,
                lambda steps: min(1.0, steps / warmup_steps) * max(0.1, 1.0 - (steps - warmup_steps) / 100000)
            )
        
        self.gamma = gamma
        self.num_samples = num_samples
        self.clip_param = clip_param
        self.value_coef = value_coef
        # Increase entropy coefficient to encourage exploration
        self.entropy_coef = max(0.05, entropy_coef * 5)  # At least 0.05, or 5x the provided value
        # Use much stricter gradient clipping to prevent explosion
        self.max_grad_norm = min(0.1, max_grad_norm)  # Even stricter clipping, max 0.1 (was 1.0)
        self.max_workers = max_workers
        self.mini_batch_size = mini_batch_size
        self.device = device
        self._env = None  # Will store environment instance
        self._state_lock = threading.Lock()  # Lock for thread safety
        
        # Add step counter for logging
        self._update_counter = 0
        self._step_counter = 0
        
        # Log initial hyperparameters to wandb
        if wandb.run is not None:
            wandb.config.update({
                "algorithm": "hybrid_grpo",
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "gamma": gamma,
                "num_samples": num_samples,
                "clip_param": clip_param,
                "value_coef": value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "hidden_dims": hidden_dims,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "device": device,
                "min_log_std": min_log_std,
                "use_lr_scheduler": use_lr_scheduler
            }, allow_val_change=True)
            
            # Log network architecture
            wandb.log({"model/actor_params": sum(p.numel() for p in self.actor.parameters() if p.requires_grad),
                       "model/critic_params": sum(p.numel() for p in self.critic.parameters() if p.requires_grad)})

    def log_training_status(self, step: int, event: str, **kwargs):
        """Log training status events with additional details
        
        Args:
            step: Current environment step
            event: Event name/type
            **kwargs: Additional data to log
        """
        if wandb.run is not None:
            log_data = {
                "status/event": event,
                "status/step": step,
            }
            # Add any additional data
            for k, v in kwargs.items():
                log_data[f"status/{k}"] = v
                
            wandb.log(log_data)
    
    def set_env(self, env):
        """Store environment instance for state restoration"""
        self._env = env
        
        # Log environment info
        if wandb.run is not None:
            # Try to extract environment info
            env_info = {}
            try:
                if hasattr(env, 'task'):
                    env_info['task_name'] = env.task.__class__.__name__
                if hasattr(env, 'physics'):
                    env_info['physics_type'] = env.physics.__class__.__name__
            except:
                pass
                
            # Log environment info
            self.log_training_status(0, "environment_set", **env_info)

    def vectorized_evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate multiple actions from a state using vectorized operations where possible.
        
        This method tries to evaluate all actions in a batch-efficient manner by:
        1. Creating temporary copies of the environment state
        2. Executing all actions in parallel if supported, or in an optimized sequence
        
        Args:
            state: Single state tensor [state_dim]
            actions: Multiple actions to evaluate [num_samples, action_dim]
            
        Returns:
            rewards: Tensor of rewards for each action [num_samples]
        """
        if self._env is None:
            raise ValueError("Environment not set. Call set_env() first.")
            
        env = self._env
        while hasattr(env, '_environment'):
            env = env._environment
            
        # Best case: Environment supports true vectorized evaluation
        if hasattr(env, 'batch_step') or hasattr(env, 'async_step'):
            try:
                print("Using environment's native vectorized step")
                action_np = actions.cpu().numpy()
                if hasattr(env, 'batch_step'):
                    rewards = env.batch_step(action_np)
                else:
                    rewards = env.async_step(action_np)
                return torch.tensor(rewards, device=self.device)
            except (AttributeError, NotImplementedError) as e:
                print(f"Vectorized step failed: {e}. Falling back to state restoration.")
        
        # Next best: Environment supports state restoration with physics
        if hasattr(env, 'physics') and hasattr(env.physics, 'get_state') and hasattr(env.physics, 'set_state'):
            try:
                # Store initial physics state
                initial_state = env.physics.get_state()
                
                # Process actions in smaller batches to avoid memory issues
                batch_size = actions.shape[0]
                mini_batch_size = min(16, batch_size)  # Process 16 actions at a time
                all_rewards = []
                
                for i in range(0, batch_size, mini_batch_size):
                    end_idx = min(i + mini_batch_size, batch_size)
                    batch_actions = actions[i:end_idx]
                    
                    # Create a queue of (action, reward) to process
                    action_queue = [(idx, action) for idx, action in enumerate(batch_actions)]
                    rewards = [0.0] * (end_idx - i)  # Pre-allocate rewards list
                    
                    # Sequential but optimized processing with state resets
                    for idx, action in action_queue:
                        # Set environment back to initial state
                        env.physics.set_state(initial_state)
                        
                        # Take action and get reward
                        timestep = self._env.step(action.cpu().numpy())
                        rewards[idx] = 0.0 if timestep.reward is None else float(timestep.reward)
                    
                    all_rewards.extend(rewards)
                
                # Restore original state
                env.physics.set_state(initial_state)
                self._env.reset()
                
                return torch.tensor(all_rewards, device=self.device)
                
            except Exception as e:
                print(f"State restoration failed: {e}. Falling back to environment resets.")
        
        # Fallback: Standard sequential evaluation with full resets
        rewards = []
        for action in actions:
            self._env.reset()  # Reset before each action
            timestep = self._env.step(action.cpu().numpy())
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            rewards.append(reward)
        
        if not rewards:  # If rewards list is empty
            return torch.zeros(1, device=self.device)
            
        return torch.tensor(rewards, device=self.device)

    def parallel_evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor, 
                                max_workers: int = 4) -> torch.Tensor:
        """Evaluate multiple actions in parallel using a safer approach with copied environments.
        
        Instead of sharing the same environment across threads (which causes MuJoCo crashes),
        this method creates independent environment copies for each thread.
        
        Args:
            state: Single state tensor [state_dim]
            actions: Multiple actions to evaluate [num_samples, action_dim]
            max_workers: Maximum number of parallel workers
            
        Returns:
            rewards: Tensor of rewards for each action [num_samples]
        """
        if self._env is None:
            raise ValueError("Environment not set. Call set_env() first.")
            
        # Determine if parallelization is even worth it - small action counts aren't worth the overhead
        n_actions = actions.shape[0]
        if n_actions <= 3 or max_workers <= 1:
            # Use simple sequential evaluation for small action counts
            return self._sequential_evaluate_actions(state, actions)
        
        # First try state restoration without threading if possible
        # Get the unwrapped environment to access physics
        env = self._env
        while hasattr(env, '_environment'):
            env = env._environment
            
        # If we can use state restoration, try a sequential but optimized approach
        if hasattr(env, 'physics') and hasattr(env.physics, 'get_state') and hasattr(env.physics, 'set_state'):
            try:
                # Store initial physics state
                initial_state = env.physics.get_state()
                
                # Evaluate actions sequentially with state restoration
                rewards = []
                for i in range(n_actions):
                    # Reset environment first to avoid accumulating state effects
                    self._env.reset()
                    
                    # Restore to initial state
                    env.physics.set_state(initial_state)
                    
                    # Step environment with this action
                    action = actions[i].cpu().numpy()
                    timestep = self._env.step(action)
                    reward = 0.0 if timestep.reward is None else float(timestep.reward)
                    rewards.append(reward)
                
                # Restore original state
                env.physics.set_state(initial_state)
                self._env.reset()
                
                if not rewards:
                    return torch.zeros(1, device=self.device)
                
                return torch.tensor(rewards, device=self.device)
                
            except Exception as e:
                print(f"State restoration failed: {e}. Falling back to reset-based evaluation.")
        
        # Fallback to standard sequential evaluation with full resets
        return self._sequential_evaluate_actions(state, actions)

    def _sequential_evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """A simple sequential evaluation of actions without threading or state manipulation.
        This is the safest fallback that won't cause MuJoCo crashes."""
        rewards = []
        for action in actions:
            self._env.reset()  # Reset before each action
            timestep = self._env.step(action.cpu().numpy())
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            rewards.append(reward)
        
        if not rewards:
            return torch.zeros(1, device=self.device)
            
        return torch.tensor(rewards, device=self.device)

    def _get_wrappers(self, env):
        """Helper to get all wrappers in the environment chain"""
        wrappers = []
        while hasattr(env, '_environment'):
            env = env._environment
            wrappers.append(env)
        return wrappers

    def eval_actions(self, state: np.ndarray) -> np.ndarray:
        """Select deterministic action for evaluation"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device) 
            mean, _ = self.actor(state)
            return mean.cpu().numpy()

    def sample_actions(self, state: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Sample multiple actions for the same state"""
        with torch.no_grad():  # Add no_grad context to prevent gradient tracking
            # Convert numpy array to torch tensor
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
                
            if state.ndim == 1:
                state = state.unsqueeze(0)  # Add batch dimension if missing
                
            mean, std = self.actor(state)
            dist = torch.distributions.Normal(mean, std)
            
            # Sample single action for environment interaction
            action = dist.sample().squeeze(0).cpu().numpy()  # Shape: [action_dim]
            
            # Increment step counter and log occasionally
            self._step_counter += 1
            if self._step_counter % 1000 == 0 and wandb.run is not None:
                # Log action distribution statistics every 1000 steps
                wandb.log({
                    "step_stats/step": self._step_counter,
                    "step_stats/action_mean": mean.mean().item(),
                    "step_stats/action_std": std.mean().item(),
                    "step_stats/log_std": self.actor.log_std.mean().item(),
                    "step_stats/action_magnitude": float(np.linalg.norm(action)),
                })
                
            return self, action  # Return (agent, action) as expected by train.py

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Sample single action for environment interaction"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, std = self.actor(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.cpu().numpy()

    def sample_multiple_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Sample multiple actions for the same state for advantage computation"""
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = torch.distributions.Normal(mean, std)
            # Sample num_samples actions for each state in the batch
            # [num_samples, batch_size, action_dim]
            actions = dist.sample((self.num_samples,))
            return actions

    def compute_hybrid_advantage(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        discounts: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid GRPO advantage using sequential action evaluation to avoid MuJoCo threading issues"""
        # Get value estimates - batch operation
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        # Compute standard TD advantage
        td_advantage = rewards + self.gamma * discounts * next_values - values
        
        # Process in smaller mini-batches to avoid memory issues
        batch_size = states.size(0)
        mini_batch_size = min(self.mini_batch_size, batch_size)
        
        # Log advantage computation stats
        computation_start_time = time.time()
        td_advantage_stats = {
            "mean": td_advantage.mean().item(),
            "std": td_advantage.std().item(),
            "min": td_advantage.min().item(),
            "max": td_advantage.max().item()
        }
        
        # If mini-batch size is too small compared to batch size, use TD advantage only
        # This is a safeguard against spending too much time on GRPO advantage calculation
        if batch_size > 10 * mini_batch_size:
            print(f"Batch size ({batch_size}) much larger than mini-batch size ({mini_batch_size}). "
                  f"Using TD advantage only for speed.")
            
            # Log decision to use TD advantage only
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "td_advantage_only", 
                    batch_size=batch_size,
                    mini_batch_size=mini_batch_size,
                    **td_advantage_stats
                )
                
            return td_advantage
        
        # Sample actions for all states at once - efficient batch operation
        with torch.no_grad():
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            all_sampled_actions = dist.sample((self.num_samples,))  # [num_samples, batch_size, action_dim]
        
        # Process in mini-batches
        grpo_advantages = []
        
        # Debug info
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size  # Ceiling division
        # print(f"Processing {batch_size} states in {num_mini_batches} mini-batches of size {mini_batch_size}")
        
        mini_batch_times = []
        
        # Iterate through mini-batches
        for i in range(0, batch_size, mini_batch_size):
            mini_batch_start = time.time()
            end_idx = min(i + mini_batch_size, batch_size)
            current_batch_size = end_idx - i
            # print(f"Mini-batch {i//mini_batch_size + 1}/{num_mini_batches}: Processing states {i} to {end_idx-1}")
            
            mini_batch_rewards = rewards[i:end_idx]
            mini_batch_states = states[i:end_idx]
            mini_batch_sampled_actions = all_sampled_actions[:, i:end_idx]
            
            # Process each state in the mini-batch
            mini_batch_action_rewards = []
            
            start_time = time.time()
            for j in range(current_batch_size):
                # Use sequential evaluation for this state's actions
                action_rewards = self._sequential_evaluate_actions(
                    mini_batch_states[j],
                    mini_batch_sampled_actions[:, j]
                )
                mini_batch_action_rewards.append(action_rewards)
            
            eval_time = time.time() - start_time
            if eval_time > 1.0:  # Log if evaluation takes more than 1 second
                print(f"GRPO action evaluation took {eval_time:.2f}s for {self.num_samples} actions × {current_batch_size} states")
                
            # If we couldn't get action rewards, use TD advantage for this mini-batch
            if not mini_batch_action_rewards:
                print(f"Warning: No action rewards collected for mini-batch {i//mini_batch_size + 1}")
                mini_batch_grpo = td_advantage[i:end_idx]
                
                if wandb.run is not None:
                    self.log_training_status(
                        self._step_counter, 
                        "mini_batch_no_rewards", 
                        mini_batch_idx=i//mini_batch_size + 1,
                        mini_batch_size=current_batch_size
                    )
            else:
                # Stack rewards for all states in mini-batch [mini_batch_size, num_samples]
                try:
                    stacked_rewards = torch.stack(mini_batch_action_rewards)
                    
                    # Calculate baseline as mean of sampled action rewards
                    baseline_rewards = stacked_rewards.mean(dim=1, keepdim=True)
                    
                    # GRPO advantage: actual reward - baseline
                    mini_batch_grpo = mini_batch_rewards - baseline_rewards
                    
                    # Log mini-batch stats
                    if wandb.run is not None and i//mini_batch_size % 2 == 0:  # Log every other mini-batch to reduce overhead
                        try:
                            wandb.log({
                                f"grpo/mini_batch_{i//mini_batch_size}/baseline_rewards": baseline_rewards.mean().item(),
                                f"grpo/mini_batch_{i//mini_batch_size}/sampled_rewards_std": stacked_rewards.std(dim=1).mean().item(),
                                f"grpo/mini_batch_{i//mini_batch_size}/actual_rewards": mini_batch_rewards.mean().item(),
                                f"grpo/mini_batch_{i//mini_batch_size}/advantage": mini_batch_grpo.mean().item(),
                            })
                        except:
                            pass  # Ignore logging errors
                        
                except Exception as e:
                    print(f"Error calculating GRPO advantage for mini-batch: {e}")
                    print(f"Shapes - mini_batch_rewards: {mini_batch_rewards.shape}, mini_batch_action_rewards: {len(mini_batch_action_rewards)}")
                    # Fall back to TD advantage for this mini-batch
                    mini_batch_grpo = td_advantage[i:end_idx]
                    
                    if wandb.run is not None:
                        self.log_training_status(
                            self._step_counter, 
                            "mini_batch_calculation_error", 
                            mini_batch_idx=i//mini_batch_size + 1,
                            error=str(e)
                        )
            
            # Store this mini-batch's GRPO advantages
            grpo_advantages.append(mini_batch_grpo)
            
            # Track mini-batch processing time
            mini_batch_time = time.time() - mini_batch_start
            mini_batch_times.append(mini_batch_time)
            
        # Combine all mini-batch advantages
        if not grpo_advantages:
            print("No GRPO advantages calculated, using TD advantage only")
            
            if wandb.run is not None:
                self.log_training_status(self._step_counter, "no_grpo_advantages", **td_advantage_stats)
                
            return td_advantage
            
        # Calculate total processing time
        total_processing_time = time.time() - computation_start_time
            
        # Check if we have advantages for all states
        total_grpo_states = sum(adv.size(0) for adv in grpo_advantages)
        if total_grpo_states != batch_size:
            print(f"WARNING: GRPO advantage calculated for {total_grpo_states} states, but batch has {batch_size} states")
            print(f"Mini-batch sizes: {[adv.size(0) for adv in grpo_advantages]}")
            # Fall back to TD advantage to avoid size mismatch
            
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "grpo_size_mismatch", 
                    total_grpo_states=total_grpo_states,
                    batch_size=batch_size
                )
                
            return td_advantage
            
        # Concatenate all mini-batch advantages
        try:
            grpo_advantage = torch.cat(grpo_advantages, dim=0)
            
            # Log advantage stats
            grpo_advantage_stats = {
                "mean": grpo_advantage.mean().item(),
                "std": grpo_advantage.std().item(),
                "min": grpo_advantage.min().item(), 
                "max": grpo_advantage.max().item()
            }
            
            # Verify shapes match before combining
            if grpo_advantage.shape != td_advantage.shape:
                print(f"Shape mismatch: grpo_advantage {grpo_advantage.shape} vs td_advantage {td_advantage.shape}")
                
                if wandb.run is not None:
                    self.log_training_status(
                        self._step_counter, 
                        "advantage_shape_mismatch",
                        grpo_shape=str(grpo_advantage.shape),
                        td_shape=str(td_advantage.shape)
                    )
                    
                return td_advantage
                
            # Combine TD and GRPO advantages with weighted averaging (favor TD)
            # Use 80% TD and 20% GRPO to improve stability while still benefiting from GRPO
            hybrid_advantages = 0.8 * td_advantage + 0.2 * grpo_advantage
            
            # Log full advantage stats
            if wandb.run is not None:
                wandb.log({
                    "advantage_stats/td_mean": td_advantage_stats["mean"],
                    "advantage_stats/td_std": td_advantage_stats["std"],
                    "advantage_stats/grpo_mean": grpo_advantage_stats["mean"],
                    "advantage_stats/grpo_std": grpo_advantage_stats["std"],
                    "advantage_stats/hybrid_mean": hybrid_advantages.mean().item(),
                    "advantage_stats/hybrid_std": hybrid_advantages.std().item(),
                    "timing/advantage_computation_time": total_processing_time,
                    "timing/avg_mini_batch_time": sum(mini_batch_times) / len(mini_batch_times) if mini_batch_times else 0,
                    "timing/max_mini_batch_time": max(mini_batch_times) if mini_batch_times else 0,
                    "timing/num_mini_batches": len(mini_batch_times),
                })
            
            return hybrid_advantages
        except Exception as e:
            print(f"Error when concatenating GRPO advantages: {e}")
            
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "advantage_concatenation_error",
                    error=str(e),
                    processing_time=total_processing_time
                )
                
            return td_advantage

    def normalize_advantages(self, advantages):
        """Normalize advantages to have zero mean and unit variance for more stable updates.
        
        Args:
            advantages: Advantage tensor
            
        Returns:
            Normalized advantages
        """
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # Compute mean and std using a more numerically stable method
        mean = advantages.mean()
        std = advantages.std()
        
        # Normalize advantages
        if std.item() < epsilon:
            # If std is too small, don't normalize to avoid numerical issues
            return advantages - mean
        else:
            return (advantages - mean) / (std + epsilon)
            
    def update(self, transitions) -> Tuple[Any, Dict[str, float]]:
        """Update policy using the provided transitions."""
        # Convert numpy arrays to torch tensors and move to device
        states = torch.FloatTensor(transitions.state).to(self.device)
        actions = torch.FloatTensor(transitions.action).to(self.device)
        rewards = torch.FloatTensor(transitions.reward).to(self.device)
        next_states = torch.FloatTensor(transitions.next_state).to(self.device)
        discounts = torch.FloatTensor(transitions.discount).to(self.device)

        batch_size = states.shape[0]
        
        # 1. ADVANTAGE CALCULATION
        # Compute hybrid advantages using state restoration
        advantages = self.compute_hybrid_advantage(
            states, actions, rewards, next_states, discounts
        ).detach()
        
        # Additional clipping of advantages to prevent extreme values
        max_advantage_value = 10.0  # Clip advantages to reasonable range
        if advantages.abs().max() > max_advantage_value:
            print(f"Clipping extreme advantages: max before {advantages.abs().max().item()}, mean before {advantages.mean().item()}")
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "clipping_extreme_advantages", 
                    max_before=advantages.abs().max().item(),
                    mean_before=advantages.mean().item()
                )
            advantages = torch.clamp(advantages, -max_advantage_value, max_advantage_value)
            
        # Normalize advantages for more stable training
        advantages = self.normalize_advantages(advantages)
        
        # 2. POLICY LOSS
        # Get old policy distribution for PPO ratio calculation
        with torch.no_grad():
            old_action_mean, old_std = self.actor(states)
            old_dist = torch.distributions.Normal(old_action_mean, old_std)
            old_log_probs = old_dist.log_prob(actions).sum(-1)
        
        # Get current policy distribution
        action_mean, std = self.actor(states)
        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        
        # Calculate policy loss with extra safeguards against extreme values
        log_ratio = log_probs - old_log_probs
        # Clip log_ratio to prevent exp() from causing extremely large values
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
        
        # PPO-style policy loss with clipping
        ratio = torch.exp(log_ratio)
        
        # Additional ratio clipping for extra safety
        ratio = torch.clamp(ratio, max=5.0)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 3. VALUE LOSS
        value_pred = self.critic(states)  # Shape: [batch_size, 1]
        
        # Step 1: First calculate next state values separately
        next_value_pred = self.critic(next_states)  # Shape: [batch_size, 1]
        
        # Step 2: Explicit shape control before calculation
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)  # Ensure [batch_size, 1]
        if len(discounts.shape) == 1:
            discounts = discounts.unsqueeze(1)  # Ensure [batch_size, 1]
            
        # Step 3: Controlled calculation with printing for debugging
        discount_factor = self.gamma * discounts  # Should be [batch_size, 1]
        future_value = discount_factor * next_value_pred  # Should be [batch_size, 1]
        value_target = rewards + future_value  # Should be [batch_size, 1]
        
        # Step 4: Verify shapes and fix if needed
        if value_pred.shape != value_target.shape:
            print(f"Debug shapes: value_pred {value_pred.shape}, rewards {rewards.shape}, "
                  f"discounts {discounts.shape}, next_value_pred {next_value_pred.shape}, "
                  f"value_target {value_target.shape}")
            
            # Last resort reshape if shapes still don't match
            value_target = value_target.view(value_pred.shape)
            
        value_loss = self.value_coef * nn.MSELoss()(value_pred, value_target.detach())
        
        # 4. ENTROPY LOSS
        entropy = dist.entropy().mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 5. TOTAL LOSS
        total_loss = policy_loss + value_loss + entropy_loss
        
        # 6. GRADIENT COMPUTATION
        # First clear any existing gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # Compute gradients and check for NaN
        total_loss.backward()
        
        # Check for NaN gradients and handle them
        actor_has_nan = False
        critic_has_nan = False
        
        for param in self.actor.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                actor_has_nan = True
                param.grad.zero_()  # Zero out NaN gradients
                
        for param in self.critic.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                critic_has_nan = True
                param.grad.zero_()  # Zero out NaN gradients
                
        if actor_has_nan or critic_has_nan:
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "nan_gradients_detected", 
                    actor_has_nan=actor_has_nan,
                    critic_has_nan=critic_has_nan
                )
        
        # Calculate gradient norms before clipping (for logging)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        
        # Store gradient norms for later use in model summary
        self._last_actor_grad_norm = actor_grad_norm.item()
        self._last_critic_grad_norm = critic_grad_norm.item()
        
        # 7. GRADIENT HANDLING
        # Check for exploding gradients and take corrective action if needed
        exploding_gradients = False
        if actor_grad_norm > 100.0:  # Reduced from 1000.0 to be more conservative
            exploding_gradients = True
            print(f"Exploding gradients detected: Actor grad norm: {actor_grad_norm.item()}, Critic grad norm: {critic_grad_norm.item()}")
            if wandb.run is not None:
                self.log_training_status(
                    self._step_counter, 
                    "exploding_gradients", 
                    actor_grad_norm=actor_grad_norm.item(),
                    critic_grad_norm=critic_grad_norm.item()
                )
        
        # Apply different clipping strategies based on gradient magnitude
        if exploding_gradients:
            if actor_grad_norm > 1000.0:
                # Severe case: very strict clipping
                actor_clip_value = 0.01
            else:
                # Moderate case: stricter clipping
                actor_clip_value = 0.05
        else:
            # Normal case: normal clipping
            actor_clip_value = self.max_grad_norm
            
        actor_clipped_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), actor_clip_value)
        critic_clipped_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        # 8. PARAMETER UPDATES
        # Skip actor update if gradients are severely exploding, but still update critic
        if exploding_gradients and actor_grad_norm > 5000.0:  # Reduced from 10000.0
            if wandb.run is not None:
                self.log_training_status(self._step_counter, "skipping_actor_update", grad_norm=actor_grad_norm.item())
        else:
            self.actor_optimizer.step()
            
        self.critic_optimizer.step()
        
        # Update learning rate schedulers if enabled
        if self.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            # Log the current learning rates
            try:
                if wandb.run is not None:
                    wandb.log({
                        "actor_learning_rate": self.actor_optimizer.param_groups[0]['lr'],
                        "critic_learning_rate": self.critic_optimizer.param_groups[0]['lr']
                    }, commit=False)
            except ImportError:
                pass
        
        # Extended metrics
        action_std = std.mean().item()
        action_mean_abs = action_mean.abs().mean().item()
        value_mean = value_pred.mean().item()
        value_std = value_pred.std().item()
        value_target_mean = value_target.mean().item()
        value_target_std = value_target.std().item()
        
        # Calculate the fraction of values clipped by the ratio bounds
        clip_fraction = ((ratio < 1 - self.clip_param) | (ratio > 1 + self.clip_param)).float().mean().item()
        
        # Log histograms and parameter distributions if wandb is active
        if wandb.run is not None:
            # Log action distribution parameters
            wandb.log({
                "actor/log_std": self.actor.log_std.mean().item(),
                "actor/std": action_std,
                "actor/mean_abs": action_mean_abs,
                "actor/ratio_hist": wandb.Histogram(ratio.detach().cpu().numpy()),
                "actor/advantages_hist": wandb.Histogram(advantages.detach().cpu().numpy()),
                "actor/actions_mean_hist": wandb.Histogram(action_mean.detach().cpu().numpy()),
                
                # Value metrics
                "critic/value_mean": value_mean,
                "critic/value_std": value_std,
                "critic/value_target_mean": value_target_mean, 
                "critic/value_target_std": value_target_std,
                
                # Gradient metrics
                "gradients/actor_grad_norm": actor_grad_norm.item(),
                "gradients/critic_grad_norm": critic_grad_norm.item(),
                "gradients/actor_clipped_grad_norm": actor_clipped_grad_norm.item(),
                "gradients/critic_clipped_grad_norm": critic_clipped_grad_norm.item(),
                
                # Network weight stats
                "weights/actor_l2": sum(p.pow(2).sum() for p in self.actor.parameters()).sqrt().item(),
                "weights/critic_l2": sum(p.pow(2).sum() for p in self.critic.parameters()).sqrt().item(),
            })
            
            # Log parameter histograms every 100 updates to avoid slowing down training
            if hasattr(self, "_update_counter"):
                self._update_counter += 1
            else:
                self._update_counter = 0
                
            if self._update_counter % 100 == 0:
                for name, param in self.actor.named_parameters():
                    wandb.log({f"actor_params/{name}": wandb.Histogram(param.detach().cpu().numpy())})
                for name, param in self.critic.named_parameters():
                    wandb.log({f"critic_params/{name}": wandb.Histogram(param.detach().cpu().numpy())})
        
        # Return metrics that will be logged in train.py
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clip_fraction": clip_fraction,
            "entropy": entropy.item(),
            "value_mean": value_mean,
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "value_target_mean": value_target_mean,
        }
        
        return self, metrics  # Return (agent, metrics) as expected by train.py

    def log_model_summary(self, step: int):
        """Log detailed model summary information.
        
        This method logs model architecture, parameter distributions, and other
        diagnostic information. Should be called at checkpoints or important stages.
        
        Args:
            step: Current environment step
        """
        if wandb.run is None:
            return
        
        # Create model architecture summary
        actor_layers = []
        critic_layers = []
        
        # Get actor layer information
        for i, module in enumerate(self.actor.policy):
            if isinstance(module, nn.Linear):
                layer_info = {
                    "name": f"Linear_{i}",
                    "type": "Linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "parameters": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                }
                actor_layers.append(layer_info)
            elif isinstance(module, nn.Tanh):
                layer_info = {
                    "name": f"Tanh_{i}",
                    "type": "Tanh",
                    "parameters": 0
                }
                actor_layers.append(layer_info)
        
        # Add log_std parameter to actor summary
        actor_layers.append({
            "name": "log_std",
            "type": "Parameter",
            "size": self.actor.log_std.numel(),
            "parameters": self.actor.log_std.numel()
        })
        
        # Get critic layer information
        for i, module in enumerate(self.critic.value):
            if isinstance(module, nn.Linear):
                layer_info = {
                    "name": f"Linear_{i}",
                    "type": "Linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "parameters": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                }
                critic_layers.append(layer_info)
            elif isinstance(module, nn.Tanh):
                layer_info = {
                    "name": f"Tanh_{i}",
                    "type": "Tanh",
                    "parameters": 0
                }
                critic_layers.append(layer_info)
        
        # Log model summary
        wandb.log({
            "model_summary/step": step,
            "model_summary/actor_layers": wandb.Table(
                columns=["name", "type", "in_features", "out_features", "parameters"],
                data=[[layer.get("name", ""), 
                       layer.get("type", ""), 
                       layer.get("in_features", ""), 
                       layer.get("out_features", ""), 
                       layer.get("parameters", 0)] 
                      for layer in actor_layers]
            ),
            "model_summary/critic_layers": wandb.Table(
                columns=["name", "type", "in_features", "out_features", "parameters"],
                data=[[layer.get("name", ""), 
                       layer.get("type", ""), 
                       layer.get("in_features", ""), 
                       layer.get("out_features", ""), 
                       layer.get("parameters", 0)] 
                      for layer in critic_layers]
            ),
            "model_summary/total_actor_params": sum(p.numel() for p in self.actor.parameters()),
            "model_summary/total_critic_params": sum(p.numel() for p in self.critic.parameters()),
        })
        
        # Log histograms of all parameters
        histograms = {}
        for name, param in self.actor.named_parameters():
            histograms[f"model_params/actor/{name}"] = wandb.Histogram(param.detach().cpu().numpy())
        
        for name, param in self.critic.named_parameters():
            histograms[f"model_params/critic/{name}"] = wandb.Histogram(param.detach().cpu().numpy())
            
        wandb.log(histograms)
        
        # Log statistics about gradient norms if available
        if hasattr(self, "_last_actor_grad_norm") and hasattr(self, "_last_critic_grad_norm"):
            wandb.log({
                "gradients_summary/actor_norm": self._last_actor_grad_norm,
                "gradients_summary/critic_norm": self._last_critic_grad_norm
            })
            
        # Log status event
        self.log_training_status(step, "model_summary_logged")
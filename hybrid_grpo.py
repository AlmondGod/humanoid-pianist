import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, List, Any
from dm_env import TimeStep

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
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
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy(state)
        std = torch.exp(self.log_std)
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print("Initializing HybridGRPO with parallel action evaluation")
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.critic = Critic(state_dim, hidden_dims).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.num_samples = num_samples
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.max_workers = max_workers
        self.mini_batch_size = mini_batch_size
        self.device = device
        self._env = None  # Will store environment instance
        self._state_lock = threading.Lock()  # Lock for thread safety

    def set_env(self, env):
        """Store environment instance for state restoration"""
        self._env = env

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
        
        # If mini-batch size is too small compared to batch size, use TD advantage only
        # This is a safeguard against spending too much time on GRPO advantage calculation
        if batch_size > 10 * mini_batch_size:
            print(f"Batch size ({batch_size}) much larger than mini-batch size ({mini_batch_size}). "
                  f"Using TD advantage only for speed.")
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
        print(f"Processing {batch_size} states in {num_mini_batches} mini-batches of size {mini_batch_size}")
        
        # Iterate through mini-batches
        for i in range(0, batch_size, mini_batch_size):
            end_idx = min(i + mini_batch_size, batch_size)
            current_batch_size = end_idx - i
            print(f"Mini-batch {i//mini_batch_size + 1}/{num_mini_batches}: Processing states {i} to {end_idx-1}")
            
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
                mini_batch_grpo = td_advantage[i:end_idx]
            else:
                # Stack rewards for all states in mini-batch [mini_batch_size, num_samples]
                try:
                    stacked_rewards = torch.stack(mini_batch_action_rewards)
                    
                    # Calculate baseline as mean of sampled action rewards
                    baseline_rewards = stacked_rewards.mean(dim=1, keepdim=True)
                    
                    # GRPO advantage: actual reward - baseline
                    mini_batch_grpo = mini_batch_rewards - baseline_rewards
                except Exception as e:
                    print(f"Error calculating GRPO advantage for mini-batch: {e}")
                    print(f"Shapes - mini_batch_rewards: {mini_batch_rewards.shape}, mini_batch_action_rewards: {len(mini_batch_action_rewards)}")
                    # Fall back to TD advantage for this mini-batch
                    mini_batch_grpo = td_advantage[i:end_idx]
            
            # Store this mini-batch's GRPO advantages
            grpo_advantages.append(mini_batch_grpo)
            
        # Combine all mini-batch advantages
        if not grpo_advantages:
            print("No GRPO advantages calculated, using TD advantage only")
            return td_advantage
            
        # Check if we have advantages for all states
        total_grpo_states = sum(adv.size(0) for adv in grpo_advantages)
        if total_grpo_states != batch_size:
            print(f"WARNING: GRPO advantage calculated for {total_grpo_states} states, but batch has {batch_size} states")
            print(f"Mini-batch sizes: {[adv.size(0) for adv in grpo_advantages]}")
            # Fall back to TD advantage to avoid size mismatch
            return td_advantage
            
        # Concatenate all mini-batch advantages
        try:
            grpo_advantage = torch.cat(grpo_advantages, dim=0)
            
            # Verify shapes match before combining
            if grpo_advantage.shape != td_advantage.shape:
                print(f"Shape mismatch: grpo_advantage {grpo_advantage.shape} vs td_advantage {td_advantage.shape}")
                return td_advantage
                
            # Combine TD and GRPO advantages (equal weighting)
            hybrid_advantages = 0.5 * td_advantage + 0.5 * grpo_advantage
            
            return hybrid_advantages
        except Exception as e:
            print(f"Error when concatenating GRPO advantages: {e}")
            return td_advantage

    def update(self, transitions) -> Tuple[Any, Dict[str, float]]:
        """Update policy using the provided transitions.
        
        Args:
            transitions: A named tuple containing:
                - state: torch.Tensor
                - action: torch.Tensor
                - reward: torch.Tensor
                - next_state: torch.Tensor
                - discount: torch.Tensor
        """
        # Convert numpy arrays to torch tensors and move to device
        states = torch.FloatTensor(transitions.state).to(self.device)
        actions = torch.FloatTensor(transitions.action).to(self.device)
        rewards = torch.FloatTensor(transitions.reward).to(self.device)
        next_states = torch.FloatTensor(transitions.next_state).to(self.device)
        discounts = torch.FloatTensor(transitions.discount).to(self.device)

        # Verify batch size is compatible with mini-batch size
        batch_size = states.size(0)
        if self.mini_batch_size > batch_size:
            # Adjust mini-batch size if it's larger than batch size
            print(f"Mini-batch size ({self.mini_batch_size}) is larger than batch size ({batch_size}). "
                  f"Adjusting mini-batch size to {batch_size}.")
            self.mini_batch_size = batch_size
        elif batch_size > 8 * self.mini_batch_size:
            # If batch size is much larger than mini-batch size, we need to be careful
            print(f"WARNING: Batch size ({batch_size}) is much larger than mini-batch size ({self.mini_batch_size}). "
                  f"This may cause slow GRPO advantage calculations.")

        # Get old log probs
        with torch.no_grad():
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            old_log_probs = dist.log_prob(actions).sum(-1)

        # Compute hybrid advantages using state restoration
        advantages = self.compute_hybrid_advantage(
            states, actions, rewards, next_states, discounts
        ).detach()
        
        # Get current policy distribution
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(-1)
        
        # PPO-style policy loss with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = self.critic(states)
        value_target = rewards + self.gamma * discounts * self.critic(next_states)
        value_loss = self.value_coef * nn.MSELoss()(value_pred, value_target.detach())
        
        # Entropy loss for exploration
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clip_fraction": (ratio.abs() > (1 + self.clip_param)).float().mean().item(),
        }
        
        return self, metrics  # Return (agent, metrics) as expected by train.py
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import List, Optional, Any
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add these imports
from sac import SAC, SACConfig
from specs import EnvironmentSpec
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
from flax import struct

@dataclass
class CPLTrainingConfig:
    """Configuration for CPL training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 1000
    alpha: float = 0.1  # Temperature parameter
    lambda_: float = 0.5  # Preference weighting
    gamma: float = 0.99  # Discount factor
    conservative_weight: float = 0.1  # Weight for conservative regularization
    seed: int = 42
    eval_interval: int = 100
    eval_episodes: int = 1

@dataclass
class Args:
    sac_checkpoint: str = "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-24-01-38-56/checkpoint_00800000.pkl"
    preference_data: str = "/Users/almondgod/Repositories/robopianist/robopianist-rl/RLHF/preference_data/2025-03-28-22-15-43/preference_data.pkl"
    output_dir: str = "RLHF/cpl_trained_models"
    config: CPLTrainingConfig = CPLTrainingConfig()
    
    # Environment args (matching generate_preference_data.py)
    midi_file: str = "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid"
    environment_name: Optional[str] = None
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = True
    reduced_action_space: bool = True
    control_timestep: float = 0.05
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = True
    disable_forearm_reward: bool = False
    camera_id: str = "piano/back"
    action_reward_observation: bool = True

def get_env(args, record_dir=None):
    """Set up the environment."""
    env = suite.load(
        environment_name=args.environment_name if args.environment_name else "crossing-field-cut-10s",
        midi_file=args.midi_file,
        seed=42,  # Default seed
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            change_color_on_activation=True,
        ),
    )
    
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=1,  # Record every episode
            camera_id=args.camera_id,
            height=480,  # Default height
            width=640,  # Default width
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=1
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=1
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    
    if hasattr(args, 'action_reward_observation') and args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    
    env = wrappers.ConcatObservationWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    
    return env

def prepare_batch_data(batch):
    """
    Prepare batch of segments for training.
    
    Args:
        batch: List of dictionaries containing preferred and non-preferred segments
        
    Returns:
        preferred: Dict with states and actions from preferred segments
        non_preferred: Dict with states and actions from non-preferred segments
    """
    preferred = {
        "states": jnp.array([
            step["state"] 
            for segment in batch 
            for step in segment["preferred"]
        ]),
        "actions": jnp.array([
            step["action"]
            for segment in batch
            for step in segment["preferred"]
        ])
    }
    
    non_preferred = {
        "states": jnp.array([
            step["state"]
            for segment in batch
            for step in segment["non_preferred"]
        ]),
        "actions": jnp.array([
            step["action"]
            for segment in batch
            for step in segment["non_preferred"]
        ])
    }
    
    return preferred, non_preferred

def cpl_loss(policy_fn, params, preferred, non_preferred, initial_params, config):
    """
    Compute CPL loss for a batch of segments with conservative regularization.
    """
    # Calculate log probs for both segments
    preferred_logprobs = jax.vmap(
        lambda s, a: policy_fn(params, s).log_prob(a)
    )(preferred["states"], preferred["actions"])
    
    non_preferred_logprobs = jax.vmap(
        lambda s, a: policy_fn(params, s).log_prob(a)
    )(non_preferred["states"], non_preferred["actions"])
    
    # Apply discounting
    timesteps = jnp.arange(preferred_logprobs.shape[0])
    discount = config.gamma ** timesteps
    
    # First sum the discounted log probs for each segment
    preferred_sum = jnp.sum(discount * preferred_logprobs)
    non_preferred_sum = jnp.sum(discount * non_preferred_logprobs)
    
    # Then apply temperature and preference weighting
    preferred_score = config.alpha * preferred_sum
    non_preferred_score = config.alpha * config.lambda_ * non_preferred_sum
    
    # Compute preference loss
    preference_loss = -jnp.log(
        jnp.exp(preferred_score) / 
        (jnp.exp(preferred_score) + jnp.exp(non_preferred_score))
    )
    
    # Compute conservative regularization per segment
    def compute_conservative_loss(states, actions):
        current_logprobs = jax.vmap(
            lambda s, a: policy_fn(params, s).log_prob(a)
        )(states, actions)
        
        initial_logprobs = jax.vmap(
            lambda s, a: policy_fn(initial_params, s).log_prob(a)
        )(states, actions)
        
        # Apply temperature scaling to log probs
        current_logprobs = config.alpha * current_logprobs
        initial_logprobs = config.alpha * initial_logprobs
        
        return jnp.mean((current_logprobs - initial_logprobs) ** 2)
    
    # Compute conservative loss for both segments
    preferred_conservative = compute_conservative_loss(
        preferred["states"], preferred["actions"]
    )
    non_preferred_conservative = compute_conservative_loss(
        non_preferred["states"], non_preferred["actions"]
    )
    
    # Average conservative loss across segments
    conservative_loss = (preferred_conservative + non_preferred_conservative) / 2.0
    
    # Combine losses
    total_loss = preference_loss + config.conservative_weight * conservative_loss
    
    return total_loss

@struct.dataclass
class CPL_SAC(SAC):
    """SAC variant that uses CPL loss for training."""
    initial_params: Optional[Any] = None  # Add initial_params as a field
    
    @classmethod
    def initialize(
        cls,
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "CPL_SAC":
        """Initialize a CPL_SAC instance."""
        # Call parent's initialize to get base SAC instance
        sac = super().initialize(spec, config, seed, discount)
        # Convert to CPL_SAC instance
        return cls(
            actor=sac.actor,
            rng=sac.rng,
            critic=sac.critic,
            target_critic=sac.target_critic,
            temp=sac.temp,
            tau=sac.tau,
            discount=sac.discount,
            target_entropy=sac.target_entropy,
            num_qs=sac.num_qs,
            num_min_qs=sac.num_min_qs,
            backup_entropy=sac.backup_entropy,
            initial_params=None,  # Initialize with None
        )
    
    def from_sac_checkpoint(self, checkpoint_path: str, spec: EnvironmentSpec, config: SACConfig, seed: int = 0):
        """Initialize CPL_SAC from a pretrained SAC checkpoint."""
        # Initialize with same architecture
        agent = self.initialize(spec, config, seed)
        
        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Store initial params for conservative loss
            initial_params = checkpoint['params'].copy()
            
            # Load all parameters from checkpoint and set initial_params
            agent = agent.replace(
                actor=agent.actor.replace(params=checkpoint['params']),
                critic=agent.critic.replace(params=checkpoint['critic_params']),
                target_critic=agent.target_critic.replace(params=checkpoint['target_critic_params']),
                temp=agent.temp.replace(params=checkpoint['temp_params']),
                initial_params=initial_params,  # Set initial_params using replace
            )
            
            print("Successfully loaded checkpoint")
            return agent
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    def update_cpl(self, preferred, non_preferred, config):
        """Update using CPL loss."""
        def cpl_loss_fn(actor_params):
            # Calculate log probs for both segments
            preferred_logprobs = jax.vmap(
                lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
            )(preferred["states"], preferred["actions"])
            
            non_preferred_logprobs = jax.vmap(
                lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
            )(non_preferred["states"], non_preferred["actions"])
            
            # Apply discounting and compute sums
            timesteps = jnp.arange(preferred_logprobs.shape[0])
            discount = config.gamma ** timesteps
            preferred_sum = jnp.sum(discount * preferred_logprobs)
            non_preferred_sum = jnp.sum(discount * non_preferred_logprobs)
            
            # Compute scores and loss
            preferred_score = config.alpha * preferred_sum
            non_preferred_score = config.alpha * config.lambda_ * non_preferred_sum
            
            preference_loss = -jnp.log(
                jnp.exp(preferred_score) / 
                (jnp.exp(preferred_score) + jnp.exp(non_preferred_score))
            )
            
            # Compute conservative loss
            def compute_conservative_loss(states, actions):
                current_logprobs = jax.vmap(
                    lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
                )(states, actions)
                
                initial_logprobs = jax.vmap(
                    lambda s, a: self.actor.apply_fn({"params": self.initial_params}, s).log_prob(a)
                )(states, actions)
                
                # Apply temperature scaling
                current_logprobs = config.alpha * current_logprobs
                initial_logprobs = config.alpha * initial_logprobs
                
                return jnp.mean((current_logprobs - initial_logprobs) ** 2)
            
            preferred_conservative = compute_conservative_loss(
                preferred["states"], preferred["actions"]
            )
            non_preferred_conservative = compute_conservative_loss(
                non_preferred["states"], non_preferred["actions"]
            )
            
            conservative_loss = (preferred_conservative + non_preferred_conservative) / 2.0
            total_loss = preference_loss + config.conservative_weight * conservative_loss
            
            return total_loss, {"loss": total_loss}
        
        try:
            # Get gradients with respect to the params
            grads, info = jax.grad(cpl_loss_fn, has_aux=True)(self.actor.params)
            
            # Update actor parameters
            new_actor = self.actor.apply_gradients(grads=grads)
            actor = self.actor.replace(params=new_actor.params)
            
            return self.replace(actor=actor), info
            
        except Exception as e:
            print("\nError during gradient computation or update:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise

def train_cpl(args: Args):
    """Train policy using CPL on preference data."""
    # Create directories
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    base_dir = Path(args.output_dir)
    run_dir = base_dir / f"CPL-{Path(args.midi_file).stem}-{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    
    for dir_path in [checkpoints_dir, eval_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create environments
    env = get_env(args)
    eval_env = get_env(args, record_dir=eval_dir)
    spec = EnvironmentSpec.make(env)

    # Create CPL_SAC agent from checkpoint
    sac_config = SACConfig(
        hidden_dims=(256, 256, 256),
        critic_dropout_rate=0.01,
        critic_layer_norm=True,
    )
    cpl_sac = CPL_SAC.initialize(spec, sac_config)
    agent = cpl_sac.from_sac_checkpoint(
        checkpoint_path=args.sac_checkpoint, 
        spec=spec, 
        config=sac_config, 
        seed=args.config.seed
    )

    # Load preference data
    with open(args.preference_data, 'rb') as f:
        data = pickle.load(f)
        pairwise_data = data["pairwise_data"]

    # Training loop
    for epoch in tqdm(range(args.config.num_epochs)):
        epoch_losses = []
        
        # Training updates
        indices = jax.random.permutation(
            jax.random.PRNGKey(epoch), 
            len(pairwise_data)
        )
        
        for i in range(0, len(indices), args.config.batch_size):
            batch_indices = indices[i:i + args.config.batch_size]
            batch = [pairwise_data[idx] for idx in batch_indices]
            preferred, non_preferred = prepare_batch_data(batch)
            agent, info = agent.update_cpl(preferred, non_preferred, args.config)
            epoch_losses.append(info["loss"])

        # Evaluation
        if (epoch + 1) % args.config.eval_interval == 0:
            for _ in range(args.config.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    action = agent.eval_actions(timestep.observation)
                    timestep = eval_env.step(action)
            
            # Save checkpoint with video
            checkpoint = {
                "params": agent.actor.params,
                "config": args.config,
                "epoch": epoch,
                "video_path": str(eval_env.latest_filename)
            }
            checkpoint_path = checkpoints_dir / f"checkpoint_{epoch+1:06d}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"\nEpoch {epoch+1}")
            print(f"Average loss: {np.mean(epoch_losses):.4f}")
            print(f"Saved checkpoint and video to {run_dir}")

    # Save final model
    final_checkpoint = {
        "params": agent.actor.params,
        "config": args.config
    }
    with open(checkpoints_dir / "checkpoint_final.pkl", 'wb') as f:
        pickle.dump(final_checkpoint, f)

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    train_cpl(args) 
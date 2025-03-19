import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import List, Optional
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
    sac_checkpoint: str
    preference_data: str 
    output_dir: str = "RLHF/cpl_trained_models"
    config: CPLTrainingConfig = CPLTrainingConfig()
    
    # Environment args (matching generate_preference_data.py)
    midi_file: str = "/path/to/midi.mid"
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

class CPL_SAC(SAC):
    """SAC variant that uses CPL loss for training."""
    
    @classmethod
    def from_sac_checkpoint(cls, checkpoint_path: str, spec, config: SACConfig, seed: int = 0):
        """Initialize CPL_SAC from a pretrained SAC checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            sac_data = pickle.load(f)
        
        # Initialize with same architecture
        agent = cls.initialize(spec, config, seed)
        
        # Load pretrained parameters
        agent = agent.replace(
            actor=agent.actor.replace(params=sac_data['params']),
            initial_actor_params=sac_data['params']  # Store initial params for regularization
        )
        return agent
    
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
                    lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
                )(states, actions)
                
                initial_logprobs = jax.vmap(
                    lambda s, a: self.initial_actor_params[s].log_prob(a)
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
            
            return total_loss, {"loss": total_loss}
        
        grads, info = jax.grad(cpl_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        
        return self.replace(actor=actor), info

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
    agent = CPL_SAC.from_sac_checkpoint(
        args.sac_checkpoint, 
        spec, 
        sac_config, 
        args.config.seed
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
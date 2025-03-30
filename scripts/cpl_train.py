# Inspired by the original CPL implementation https://github.com/jhejna/cpl/blob/main/research/algs/cpl.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb
from dataclasses import asdict

from architecture.sac import SACConfig
from rl_dataclasses.specs import EnvironmentSpec
from architecture.cpl_sac import CPL_SAC
from scripts.eval import get_env

@dataclass
class CPLTrainingConfig:
    """Configuration for CPL training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 1000
    alpha: float = 0.1  # Temperature parameter
    lambda_: float = 0.5  # Preference weighting
    gamma: float = 0.99  # Discount factor
    conservative_weight: float = 0.01  # Weight for conservative regularization
    seed: int = 42
    eval_interval: int = 100
    eval_episodes: int = 1

@dataclass
class Args:
    seed: int = 42
    sac_checkpoint: str = None
    preference_data: str = None
    output_dir: str = "cpl_trained_models"
    config: CPLTrainingConfig = CPLTrainingConfig()
    midi_file: str = None
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
    stretch_factor: float = 1.0
    shift_factor: int = 0
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = True
    frame_stack: int = 1
    clip: bool = True
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "panning_camera"
    action_reward_observation: bool = True

def prepare_batch_data(batch):
    """
    Prepare batch of segments for training.
    
    Args:
        batch: List of dictionaries containing preferred and non-preferred segments
        
    Returns:
        preferred: Dict with states and actions from preferred segments
        non_preferred: Dict with states and actions from non-preferred segments
        segment_info: Dict with batch information
    """
    # First get segment lengths to ensure they're consistent
    preferred_lengths = [len(segment["preferred"]) for segment in batch]
    non_preferred_lengths = [len(segment["non_preferred"]) for segment in batch]
    
    if len(set(preferred_lengths + non_preferred_lengths)) != 1:
        raise ValueError(f"All segments must have same length. Found lengths: preferred={preferred_lengths}, non_preferred={non_preferred_lengths}")
    
    segment_length = preferred_lengths[0]
    batch_size = len(batch)
    
    preferred = {
        "states": jnp.array([
            step["state"] 
            for segment in batch 
            for step in segment["preferred"]
        ]).reshape(batch_size, segment_length, -1),  # Reshape to (batch_size, segment_length, state_dim)
        
        "actions": jnp.array([
            step["action"]
            for segment in batch 
            for step in segment["preferred"]
        ]).reshape(batch_size, segment_length, -1)  # Reshape to (batch_size, segment_length, action_dim)
    }
    
    non_preferred = {
        "states": jnp.array([
            step["state"]
            for segment in batch 
            for step in segment["non_preferred"]
        ]).reshape(batch_size, segment_length, -1),
        
        "actions": jnp.array([
            step["action"]
            for segment in batch
            for step in segment["non_preferred"]
        ]).reshape(batch_size, segment_length, -1)
    }
    
    segment_info = {
        "batch_size": batch_size,
        "segment_length": segment_length
    }
    
    return preferred, non_preferred, segment_info

def train_cpl(args: Args):
    """Train policy using CPL on preference data."""
    # Initialize wandb
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f"CPL-{Path(args.midi_file).stem}-{args.config.seed}-{timestamp}"
    
    wandb.init(
        project="robopianist",
        entity="almond-maj-projects",
        name=run_name,
        config=asdict(args),
        mode="online"
    )

    # Create directories
    base_dir = Path(args.output_dir)
    run_dir = base_dir / run_name
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
    start_time = time.time()
    total_steps = 0
    
    for epoch in tqdm(range(args.config.num_epochs)):
        epoch_losses = []
        epoch_metrics = {
            "train/preference_loss": 0.0,
            "train/conservative_loss": 0.0,
            "train/total_loss": 0.0,
        }
        
        # Training updates
        indices = jax.random.permutation(
            jax.random.PRNGKey(epoch), 
            len(pairwise_data)
        )
        
        num_batches = 0
        for i in range(0, len(indices), args.config.batch_size):
            batch_indices = indices[i:i + args.config.batch_size]
            batch = [pairwise_data[idx] for idx in batch_indices]
            
            preferred, non_preferred, segment_info = prepare_batch_data(batch)
            
            agent, info = agent.update_cpl(preferred, non_preferred, args.config)
            
            # Check if loss is NaN and print relevant info
            if jnp.isnan(info["loss"]):
                print("\nWARNING: NaN loss detected!")
                print(f"Full info dict: {info}")
            
            # Track detailed metrics
            epoch_metrics["train/preference_loss"] += info.get("preference_loss", 0.0)
            epoch_metrics["train/conservative_loss"] += info.get("conservative_loss", 0.0)
            epoch_metrics["train/total_loss"] += info["loss"]
            epoch_losses.append(info["loss"])
            num_batches += 1
            total_steps += 1

            # Log training metrics periodically
            if total_steps % 100 == 0:  # Adjust frequency as needed
                # Calculate average metrics
                for key in epoch_metrics:
                    epoch_metrics[key] /= num_batches
                
                # Add FPS metric
                epoch_metrics["train/fps"] = int(total_steps / (time.time() - start_time))
                
                # Log to wandb
                wandb.log(epoch_metrics, step=total_steps)
                
                # Reset metrics
                for key in epoch_metrics:
                    epoch_metrics[key] = 0.0
                num_batches = 0

        # Evaluation
        if (epoch + 1) % args.config.eval_interval == 0:
            eval_metrics = {
                "eval/episode_length": 0,
                "eval/episode_return": 0,
            }
            
            for _ in range(args.config.eval_episodes):
                timestep = eval_env.reset()
                episode_return = 0
                episode_length = 0
                
                while not timestep.last():
                    action = agent.eval_actions(timestep.observation)
                    timestep = eval_env.step(action)
                    episode_return += timestep.reward
                    episode_length += 1
                
                eval_metrics["eval/episode_length"] += episode_length
                eval_metrics["eval/episode_return"] += episode_return
            
            # Average metrics over evaluation episodes
            for key in eval_metrics:
                eval_metrics[key] /= args.config.eval_episodes
            
            # Add video to wandb
            if hasattr(eval_env, 'latest_filename'):
                video = wandb.Video(
                    str(eval_env.latest_filename),
                    fps=4,
                    format="mp4"
                )
                eval_metrics["eval/video"] = video
            
            # Log evaluation metrics
            wandb.log(eval_metrics, step=total_steps)
            
            # Save checkpoint with video
            checkpoint = {
                "params": agent.actor.params,
                "config": args.config,
                "epoch": epoch,
                "video_path": str(eval_env.latest_filename) if hasattr(eval_env, 'latest_filename') else None
            }
            checkpoint_path = checkpoints_dir / f"checkpoint_{epoch+1:06d}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"\nEpoch {epoch+1}")
            print(f"Average loss: {np.mean(epoch_losses):.4f}")
            print(f"Eval return: {eval_metrics['eval/episode_return']:.2f}")
            print(f"Saved checkpoint and video to {run_dir}")

    # Save final model
    final_checkpoint = {
        "params": agent.actor.params,
        "config": args.config
    }
    with open(checkpoints_dir / "checkpoint_final.pkl", 'wb') as f:
        pickle.dump(final_checkpoint, f)
    
    wandb.finish()

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    train_cpl(args) 
import os
import pickle
import numpy as np
import jax
import time
import random
import argparse
import wandb
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import collections
from copy import copy
from dataclasses import replace, dataclass, asdict, field

import sys
sys.path.append('.')  # Add the root directory to path

# Import necessary modules
from sac import SAC, SACConfig
from specs import EnvironmentSpec
from replay import Buffer, Transition  # Import both
from RLHF.cpl import CPL, CPLConfig

# Environment imports
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
from cpl_reward_wrapper import CPLRewardWrapper

@dataclass
class Args:
    """Arguments for fine-tuning SAC with CPL rewards."""
    # Checkpoints
    sac_checkpoint: str = "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-21-29-41/checkpoint_00840000.pkl"
    cpl_checkpoint: str = "/Users/almondgod/Repositories/robopianist/robopianist-rl/reward_model/checkpoint_latest.pkl"
    
    # Environment settings
    midi_file: str = "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid"
    environment_name: Optional[str] = None
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = True
    control_timestep: float = 0.05
    n_steps_lookahead: int = 10
    disable_fingering_reward: bool = True
    disable_forearm_reward: bool = False
    wrong_press_termination: bool = False
    action_reward_observation: bool = True
    
    # Training settings
    output_dir: str = "rlhf_models"
    seed: int = 42
    max_steps: int = 500_000
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    log_interval: int = 1000
    eval_interval: int = 1000
    eval_episodes: int = 1
    tqdm_bar: bool = True
    
    # Recording settings
    record_dir: Optional[str] = None
    record_every: int = 1
    camera_id: str = "piano/back"
    record_resolution: tuple = (480, 640)
    
    # RLHF specific settings
    reward_scale: float = 1.0
    original_reward_weight: float = 0.0
    
    # Wandb settings
    use_wandb: bool = False
    project: str = "robopianist-rlhf"
    entity: str = "almond-maj-projects"
    wandb_mode: str = "online"
    
    # Additional settings from train.py
    stretch_factor: float = 1.0
    shift_factor: int = 0
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = True
    frame_stack: int = 1
    clip: bool = True

def get_env(args, record_dir=None):
    """Set up the environment."""
    # Force reduced action space to match training
    args = copy(args)
    args.reduced_action_space = True  # Force this to match training
    
    env = suite.load(
        environment_name=args.environment_name if hasattr(args, 'environment_name') and args.environment_name else None,
        midi_file=args.midi_file,
        seed=args.seed,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,  # Now always True
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
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    
    return env


def load_cpl_model(checkpoint_path, state_dim, action_dim):
    """Load the CPL reward model from a checkpoint."""
    # Create CPL config with default values
    config = CPLConfig()
    
    # Initialize CPL model
    cpl_model = CPL(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )
    
    # Load checkpoint
    cpl_model.load_checkpoint(checkpoint_path)
    
    return cpl_model


def prefix_dict(prefix: str, d: dict) -> dict:
    """Add prefix to dictionary keys."""
    return {f"{prefix}/{k}": v for k, v in d.items()}


def finetune(args):
    """Fine-tune the SAC agent using the CPL reward model."""
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    experiment_dir = Path(args.output_dir) / f"{args.seed}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb for tracking
    if args.use_wandb:
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"RLHF-SAC-{args.midi_file.stem if args.midi_file else 'env'}-{args.seed}",
            config=vars(args),
            mode=args.wandb_mode,
        )
    
    # Create environments
    env = get_env(args)
    eval_env = get_env(args, record_dir=args.record_dir)
    
    # Add detailed observation debugging
    timestep = env.reset()
    
    # Print observation dimensions
    timestep = env.reset()
    
    # Load the checkpoint to check dimensions
    with open(args.sac_checkpoint, 'rb') as f:
        sac_checkpoint = pickle.load(f)
    
    # Get environment spec
    original_spec = EnvironmentSpec.make(env)
    
    # Create replay buffer
    replay_buffer = Buffer(
        state_dim=original_spec.observation_dim,
        action_dim=original_spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )
    
    # Load pretrained SAC agent
    print(f"Loading pretrained SAC model from {args.sac_checkpoint}")
    
    # Initialize SAC agent with the modified spec
    sac_config = SACConfig(
        hidden_dims=(256, 256, 256),
        critic_dropout_rate=0.01,
        critic_layer_norm=True,
    )
    
    agent = SAC.initialize(
        original_spec,  
        sac_config,
        seed=args.seed
    )
    
    # Replace with pretrained parameters
    agent = agent.replace(
        actor=agent.actor.replace(params=sac_checkpoint['params']),
        critic=agent.critic.replace(params=sac_checkpoint['critic_params']),
        target_critic=agent.target_critic.replace(params=sac_checkpoint['target_critic_params']),
        temp=agent.temp.replace(params=sac_checkpoint['temp_params'])
    )
    
    # Load CPL reward model
    print(f"Loading CPL reward model from {args.cpl_checkpoint}")
    
    # Get state/action dimensions from the environment
    state_dim = original_spec.observation_dim
    action_dim = original_spec.action_dim
    
    cpl_model = load_cpl_model(args.cpl_checkpoint, state_dim, action_dim)
    
    # Wrap environment with CPL reward
    env = CPLRewardWrapper(
        env=env, 
        reward_model=cpl_model,
        reward_scale=args.reward_scale,
        original_reward_weight=args.original_reward_weight
    )
    
    # Prepare for training
    timestep = env.reset()
    replay_buffer.insert(timestep, None)
    
    print(f"Starting fine-tuning for {args.max_steps} steps...")
    start_time = time.time()
    
    # Fine-tuning loop
    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        obs = timestep.observation
        
        # Act with truncated observation
        agent, action = agent.sample_actions(obs)
        
        # Step environment with CPL rewards
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)
        
        # Reset if episode ended
        if timestep.last():
            if args.use_wandb:
                wandb.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)
        
        # Train
        if replay_buffer.is_ready():
            transitions = replay_buffer.sample()
            agent, metrics = agent.update(transitions)
            if i % args.log_interval == 0 and args.use_wandb:
                wandb.log(prefix_dict("train", metrics), step=i)
        
        # Evaluate
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                eval_timestep = eval_env.reset()
                while not eval_timestep.last():
                    eval_timestep = eval_env.step(agent.eval_actions(eval_timestep.observation))
            
            if args.use_wandb:
                log_dict = prefix_dict("eval", eval_env.get_statistics())
                music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
                wandb.log(log_dict | music_dict, step=i)
                video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
                wandb.log({"video": video, "global_step": i})
            
            # Save checkpoint
            checkpoint = {
                'params': agent.actor.params,
                'critic_params': agent.critic.params,
                'target_critic_params': agent.target_critic.params,
                'temp_params': agent.temp.params
            }
            
            checkpoint_path = experiment_dir / f"checkpoint_{i:08d}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if i % args.log_interval == 0 and args.use_wandb:
            wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)
    
    # Save final checkpoint
    final_checkpoint = {
        'params': agent.actor.params,
        'critic_params': agent.critic.params,
        'target_critic_params': agent.target_critic.params,
        'temp_params': agent.temp.params
    }
    
    final_path = experiment_dir / "checkpoint_final.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    
    print(f"Fine-tuning completed. Final model saved to {final_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    
    # Convert string paths to Path objects
    if args.midi_file:
        args.midi_file = Path(args.midi_file)
    
    # Create record directory based on checkpoint
    if args.record_dir is None:
        checkpoint_name = Path(args.sac_checkpoint).parent.name
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        args.record_dir = Path("finetune_sac_videos") / f"{checkpoint_name}" / f"{timestamp}"
    else:
        args.record_dir = Path(args.record_dir)
    
    finetune(args)
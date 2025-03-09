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
from dataclasses import replace

import sys
sys.path.append('.')  # Add the root directory to path

# Import necessary modules
from sac import SAC, SACConfig
from specs import EnvironmentSpec
from replay import Transition  # Only import Transition
from RLHF.cpl import CPL, CPLConfig

# Environment imports
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

class CPLRewardWrapper:
    """Wrapper that uses the CPL reward model to modify environment rewards."""
    
    def __init__(self, env, reward_model, reward_scale=1.0, original_reward_weight=0.0):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            reward_model: The CPL reward model
            reward_scale: Scaling factor for CPL rewards
            original_reward_weight: Weight for original environment rewards (0 to 1)
                                   0 = only use CPL rewards, 1 = only use original rewards
        """
        self.env = env
        self.reward_model = reward_model
        self.reward_scale = reward_scale
        self.original_reward_weight = original_reward_weight
        self.last_state = None
        self.last_action = None
    
    def step(self, action):
        """Step the environment and modify the reward."""
        # Store last state and action for reward computation
        self.last_state = self.env.physics.get_state()
        self.last_action = action
        
        # Take a step in the environment
        timestep = self.env.step(action)
        
        if not timestep.first():  # Skip reward modification on first step
            # Get environment observation and last action
            obs = timestep.observation
            
            # Calculate reward from CPL model
            cpl_reward = self.reward_model.predict_rewards(
                np.expand_dims(obs, axis=0), 
                np.expand_dims(self.last_action, axis=0)
            )[0]
            
            # Scale CPL reward
            cpl_reward = cpl_reward * self.reward_scale
            
            # Combine rewards if needed
            if self.original_reward_weight > 0:
                combined_reward = (
                    self.original_reward_weight * timestep.reward + 
                    (1 - self.original_reward_weight) * cpl_reward
                )
            else:
                combined_reward = cpl_reward
            
            # Create new timestep with modified reward
            timestep = timestep._replace(reward=combined_reward)
        
        return timestep
    
    def reset(self):
        """Reset the environment."""
        timestep = self.env.reset()
            
        self.last_state = None
        self.last_action = None
        return timestep
    
    # Forward all other method calls to the underlying environment
    def __getattr__(self, name):
        return getattr(self.env, name)


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
    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        batch_size=args.batch_size,
        spec=original_spec
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a SAC agent with a CPL reward model")
    
    # Checkpoints
    parser.add_argument("--sac_checkpoint", type=str, 
                        default="/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-02-12-40-35/checkpoint_00680000.pkl",
                        help="Path to the pretrained SAC checkpoint")
    parser.add_argument("--cpl_checkpoint", type=str, 
                        default="/Users/almondgod/Repositories/robopianist/robopianist-rl/reward_model/checkpoint_latest.pkl",
                        help="Path to the trained CPL reward model checkpoint")
    
    # Environment settings
    parser.add_argument("--midi_file", type=str, 
                        default="/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid",
                        help="Path to MIDI file (if using a specific piece)")
    parser.add_argument("--environment_name", type=str, default=None,
                        help="Environment name (if not using MIDI file)")
    parser.add_argument("--trim_silence", action="store_true",
                        help="Trim silence from the beginning of the MIDI file")
    parser.add_argument("--gravity_compensation", action="store_true",
                        help="Use gravity compensation")
    parser.add_argument("--reduced_action_space", type=bool, default=True,
                        help="Use reduced action space")
    parser.add_argument("--control_timestep", type=float, default=0.05,
                        help="Control timestep")
    parser.add_argument("--n_steps_lookahead", type=int, default=10,
                        help="Number of steps to look ahead")
    parser.add_argument("--disable_fingering_reward", type=bool, default=True,
                        help="Disable fingering reward")
    parser.add_argument("--disable_forearm_reward", action="store_true",
                        help="Disable forearm reward")
    parser.add_argument("--wrong_press_termination", action="store_true",
                        help="Terminate episode on wrong key press")
    parser.add_argument("--action_reward_observation", type=bool, default=True,
                        help="Include action and reward in observation")
    
    # Training settings
    parser.add_argument("--output_dir", type=str, default="rlhf_models",
                        help="Directory to save fine-tuned models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_steps", type=int, default=500_000,
                        help="Maximum number of fine-tuning steps")
    parser.add_argument("--replay_capacity", type=int, default=1_000_000,
                        help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=10000,
                        help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=40000,
                        help="Evaluation interval")
    parser.add_argument("--eval_episodes", type=int, default=1,
                        help="Number of episodes for evaluation")
    parser.add_argument("--tqdm_bar", type=bool, default=True,
                        help="Show progress bar")
    
    # Recording settings
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Directory to save evaluation videos")
    parser.add_argument("--record_every", type=int, default=1,
                        help="Record every N episodes")
    parser.add_argument("--camera_id", type=str, default="piano/back",
                        help="Camera ID for recording")
    
    # RLHF specific settings
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Scale factor for CPL rewards")
    parser.add_argument("--original_reward_weight", type=float, default=0.0,
                        help="Weight for original environment rewards (0 to 1)")
    
    # Wandb settings
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--project", type=str, default="robopianist-rlhf",
                        help="Wandb project name")
    parser.add_argument("--entity", type=str, default="almond-maj-projects",
                        help="Wandb entity name")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        help="Wandb mode (online, offline, disabled)")
    
    # Add missing arguments from train.py
    parser.add_argument("--stretch_factor", type=float, default=1.0,
                       help="Time stretch factor for MIDI file")
    parser.add_argument("--shift_factor", type=int, default=0,
                       help="Time shift factor for MIDI file")
    parser.add_argument("--disable_colorization", action="store_true",
                       help="Disable piano key colorization")
    parser.add_argument("--disable_hand_collisions", action="store_true",
                       help="Disable hand collisions")
    parser.add_argument("--primitive_fingertip_collisions", action="store_true", default=True,
                       help="Use primitive fingertip collisions")
    parser.add_argument("--frame_stack", type=int, default=1,
                       help="Number of frames to stack in observations")
    parser.add_argument("--clip", type=bool, default=True,
                       help="Whether to clip observations")
    parser.add_argument("--record_resolution", type=tuple, default=(480, 640),
                       help="Resolution for recording videos")
    
    return parser.parse_args()


# Define ReplayBuffer here
class ReplayBuffer:
    """Replay buffer for off-policy reinforcement learning."""
    
    def __init__(self, capacity: int, batch_size: int, spec: EnvironmentSpec):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Batch size for sampling
            spec: Environment specification
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.spec = spec
        
        # Initialize buffer
        self.storage = collections.deque(maxlen=capacity)
        self.current_episode = []
        self.num_episodes = 0
    
    def insert(self, timestep, action):
        """Insert a transition into the replay buffer."""
        if timestep.first():
            # Start a new episode
            self.current_episode = []
            
        if action is not None:
            # Store step in current episode
            self.current_episode.append((timestep, action))
        
        if timestep.last():
            # End of episode, store the episode
            if len(self.current_episode) > 0:
                self.storage.append(self.current_episode)
                self.num_episodes += 1
            self.current_episode = []
    
    def sample(self):
        """Sample a batch of transitions for training."""
        # Sample episode indices
        episode_indices = np.random.randint(0, len(self.storage), self.batch_size)
        
        # Get batch of transitions
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        for idx in episode_indices:
            episode = self.storage[idx]
            if len(episode) <= 1:
                continue
                
            # Sample a random transition from the episode
            transition_idx = np.random.randint(0, len(episode) - 1)
            
            # Get current timestep and action
            current_timestep, action = episode[transition_idx]
            
            # Get next timestep
            next_timestep, _ = episode[transition_idx + 1]
            
            # Store transition data
            observations.append(current_timestep.observation)
            actions.append(action)
            rewards.append(next_timestep.reward)
            next_observations.append(next_timestep.observation)
            dones.append(next_timestep.last())
        
        dones = np.array(dones)
        
        # Convert to arrays
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones).astype(np.float32)
        
        # Fix: Convert to boolean before inverting
        discounts = (1.0 - dones).astype(np.float32) * 0.8  # Alternative to ~ operator
        
        # Create transition object with correct field names
        transition = Transition(
            state=observations,
            action=actions,
            reward=rewards,
            next_state=next_observations,
            discount=discounts  # Changed from terminated to discount
        )
        
        return transition
    
    def is_ready(self):
        """Check if the buffer has enough data for training."""
        return len(self.storage) >= 1 and self.num_episodes >= 1

if __name__ == "__main__":
    args = parse_args()
    
    # Convert string paths to Path objects
    if args.midi_file:
        args.midi_file = Path(args.midi_file)
    
    # Create record directory based on midi file and checkpoint
    if args.record_dir is None:
        checkpoint_name = Path(args.sac_checkpoint).parent.name
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        args.record_dir = Path("finetune_sac_videos") / f"{checkpoint_name}" / f"{timestamp}"
    else:
        args.record_dir = Path(args.record_dir)
    
    finetune(args)
# TODO: i need preference data first, created by rolling out a pretrained policy. 

# first, what data do i need to keep track of?

# also, i need you to roll out the episodes and make recording for each episode, 
# then allow me to reference them and add rankings after ive seen all the videos. 
# i want to rank the videos and then the pairwse data is created by seeing 
# which of the pair has a higher rating (1) and the lower one gets 0 in the pair. 

import os
import pickle
import numpy as np
import argparse
import random
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple, Optional
import glob
import shutil
from dataclasses import dataclass, field

import sys
sys.path.append('.')  # Add the root directory to path

from sac import SAC, SACConfig
from specs import EnvironmentSpec
# Remove the non-existent import
# from dm_env_wrappers import get_env

# Import necessary modules for environment creation
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

# Define our own version of get_env that works with our class
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

@dataclass
class Args:
    """Arguments for generating preference data."""
    # Model checkpoints
    checkpoints: List[str] = field(default_factory=lambda: [
        "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-21-29-41/checkpoint_00920000.pkl",
        "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-02-12-40-35/checkpoint_00320000.pkl",
        "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-02-28-21-38-57/checkpoint_00640000.pkl",
        "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-01-10-57-51/checkpoint_00480000.pkl",
        "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-09-53-14/checkpoint_00560000.pkl"
    ])
    
    # Noise settings
    noise_scales: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15,0.2])  # Start with no noise
    
    # Environment settings
    midi_file: str = "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid"
    environment_name: Optional[str] = None
    
    # Data collection settings
    episodes_per_config: int = 3  # Number of episodes per checkpoint-noise combination
    seed: int = 42
    data_dir: str = "RLHF/preference_data"
    algorithm: str = "sac"
    
    # Optional settings
    rankings_file: Optional[str] = None
    generate_only: bool = False

class PreferenceDataGenerator:
    def __init__(
        self,
        checkpoints: List[str],
        noise_scales: List[float],
        midi_file: str = None,
        environment_name: str = None,
        episodes_per_config: int = 3,
        seed: int = 42,
        data_dir: str = "RLHF/preference_data",
        algorithm: str = "sac"
    ):
        """Initialize the preference data generator."""
        self.checkpoints = checkpoints
        self.noise_scales = noise_scales
        self.midi_file = midi_file
        self.environment_name = environment_name
        self.episodes_per_config = episodes_per_config
        self.total_episodes = len(checkpoints) * len(noise_scales) * episodes_per_config
        self.seed = seed
        self.algorithm = algorithm
        
        # Create timestamped directories
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.base_data_dir = Path(data_dir)
        self.data_dir = self.base_data_dir / timestamp
        
        # Create subdirectories
        self.video_dir = self.data_dir / "videos"
        self.logs_dir = self.data_dir / "logs"
        
        for dir_path in [self.video_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "checkpoints": checkpoints,
            "noise_scales": noise_scales,
            "midi_file": midi_file,
            "environment_name": environment_name,
            "episodes_per_config": episodes_per_config,
            "seed": seed,
            "algorithm": algorithm,
            "timestamp": timestamp
        }
        with open(self.logs_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize environment and agent
        self._setup_env_and_agent()
        
        # Initialize data storage
        self.trajectories = []
        self.rankings = {}
        self.pairwise_preferences = []
    
    def _setup_env_and_agent(self):
        """Set up the environment and load the agent."""
        # Configure environment args
        from dataclasses import dataclass
        
        @dataclass
        class EnvArgs:
            midi_file: Optional[Path] = Path(self.midi_file) if self.midi_file else None
            environment_name: Optional[str] = self.environment_name
            trim_silence: bool = True
            gravity_compensation: bool = True
            reduced_action_space: bool = True
            control_timestep: float = 0.05
            n_steps_lookahead: int = 10
            disable_fingering_reward: bool = True
            disable_forearm_reward: bool = False
            wrong_press_termination: bool = False
            action_reward_observation: bool = True
            primitive_fingertip_collisions: bool = True
            record_dir: Optional[Path] = self.video_dir
            camera_id: str = "piano/back"
            
        self.env_args = EnvArgs()
        
        # Create environment for evaluation with recording
        self.env = get_env(self.env_args, record_dir=self.video_dir)
        
        # Get environment spec
        self.spec = EnvironmentSpec.make(self.env)
        
        # Don't load agent here anymore since we'll load per checkpoint
        self.env = get_env(self.env_args, record_dir=self.video_dir)
        self.spec = EnvironmentSpec.make(self.env)
    
    def generate_episodes(self):
        """Generate episodes from multiple checkpoints with varying noise."""
        episode_idx = 0
        self.video_paths = []
        self.trajectories = []
        self.config_info = []  # Store which checkpoint and noise level was used

        for checkpoint_path in self.checkpoints:
            print(f"\nUsing checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            with open(checkpoint_path, 'rb') as f:
                sac_checkpoint = pickle.load(f)
            
            # Initialize SAC agent
            sac_config = SACConfig(
                hidden_dims=(256, 256, 256),
                critic_dropout_rate=0.01,
                critic_layer_norm=True,
            )
            agent = SAC.initialize(self.spec, sac_config, seed=self.seed)
            
            # Load parameters
            agent = agent.replace(
                actor=agent.actor.replace(params=sac_checkpoint['params']),
                critic=agent.critic.replace(params=sac_checkpoint['critic_params']),
                target_critic=agent.target_critic.replace(params=sac_checkpoint['target_critic_params']),
                temp=agent.temp.replace(params=sac_checkpoint['temp_params'])
            )

            for noise_scale in self.noise_scales:
                print(f"\nUsing noise scale: {noise_scale}")
                
                for _ in range(self.episodes_per_config):
                    # Roll out episode with noise
                    trajectory = {
                        "observations": [],
                        "actions": [],
                        "rewards": [],
                        "next_observations": [],
                        "dones": []
                    }
                    
                    timestep = self.env.reset()
                    episode_return = 0
                    
                    while not timestep.last():
                        # Get action from policy
                        action = agent.eval_actions(timestep.observation)
                        
                        # Add noise to action
                        if noise_scale > 0:
                            noise = np.random.normal(0, noise_scale, size=action.shape)
                            action = np.clip(action + noise, -1, 1)
                        
                        # Step environment
                        next_timestep = self.env.step(action)
                        
                        # Store transition
                        trajectory["observations"].append(timestep.observation)
                        trajectory["actions"].append(action)
                        trajectory["rewards"].append(next_timestep.reward)
                        trajectory["next_observations"].append(next_timestep.observation)
                        trajectory["dones"].append(next_timestep.last())
                        
                        episode_return += next_timestep.reward
                        timestep = next_timestep

                    # Store episode info
                    self.config_info.append({
                        "episode_idx": episode_idx,
                        "checkpoint": checkpoint_path,
                        "noise_scale": noise_scale,
                        "return": episode_return
                    })
                    
                    # Get the video file path (most recent file in the directory)
                    video_files = sorted(self.video_dir.glob("*.mp4"), key=os.path.getctime)
                    if video_files:
                        latest_video = video_files[-1]
                        # Rename to include episode number
                        new_name = self.video_dir / f"episode_{episode_idx:03d}.mp4"
                        shutil.move(latest_video, new_name)
                        self.video_paths.append(str(new_name))
                    else:
                        print(f"Warning: No video found for episode {episode_idx}")
                        self.video_paths.append(None)
                    
                    # Convert trajectory lists to numpy arrays
                    for key in trajectory:
                        trajectory[key] = np.array(trajectory[key])
                    
                    # Add episode stats
                    trajectory["return"] = episode_return
                    trajectory["info"] = self.env.get_statistics()
                    trajectory["musical_metrics"] = self.env.get_musical_metrics()
                    trajectory["video_path"] = self.video_paths[-1]
                    
                    # Store trajectory
                    self.trajectories.append(trajectory)
                    
                    # Print episode stats
                    print(f"Episode {episode_idx+1} return: {episode_return:.2f}")
                    print(f"Episode stats: {trajectory['info']}")
                    print(f"Musical metrics: {trajectory['musical_metrics']}")
                    print(f"Video saved to: {self.video_paths[-1]}")
                    print("-" * 50)
                    
                    episode_idx += 1

        # Save config info
        with open(self.logs_dir / "episode_configs.json", 'w') as f:
            json.dump(self.config_info, f, indent=2)
        
        # Generate command to view videos
        print("\nTo view all videos for ranking, run:")
        if os.name == 'nt':  # Windows
            print(f"start {self.video_dir}")
        else:  # macOS or Linux
            print(f"open {self.video_dir}")
        
        print("\nAfter viewing, rank the episodes from 1 (best) to N (worst).")
        print("You can use the interactive ranking interface by running:")
        print(f"python RLHF/rank_episodes.py --data_dir {self.data_dir}")
    
    def _save_trajectories(self):
        """Save trajectories to disk."""
        trajectory_path = self.logs_dir / "trajectories.pkl"
        with open(trajectory_path, 'wb') as f:
            pickle.dump({
                "trajectories": self.trajectories,
                "video_paths": self.video_paths,
                "env_info": {
                    "midi_file": self.midi_file,
                    "environment_name": self.environment_name
                }
            }, f)
        print(f"Saved trajectories to {trajectory_path}")
    
    def load_rankings(self, rankings_file: str = None):
        """
        Load rankings from file or interactively input them.
        
        Args:
            rankings_file: Path to JSON file with rankings
        """
        if rankings_file:
            with open(rankings_file, 'r') as f:
                self.rankings = json.load(f)
        else:
            rankings_path = self.logs_dir / "rankings.json"
            if rankings_path.exists():
                with open(rankings_path, 'r') as f:
                    self.rankings = json.load(f)
                print(f"Loaded rankings from {rankings_path}")
            else:
                self._interactive_ranking()
    
    def _interactive_ranking(self):
        """Interactively input rankings for episodes."""
        print("\nPlease rank the episodes from 1 (best) to N (worst).")
        print("After viewing the videos, enter the ranking for each episode.")
        
        self.rankings = {}
        for i in range(self.total_episodes):
            rank = int(input(f"Rank for episode {i+1} (1 is best): "))
            self.rankings[str(i)] = rank
        
        # Save rankings
        rankings_path = self.logs_dir / "rankings.json"
        with open(rankings_path, 'w') as f:
            json.dump(self.rankings, f)
        print(f"Saved rankings to {rankings_path}")
    
    def generate_pairwise_preferences(self):
        """Generate pairwise preferences from rankings."""
        if not self.rankings:
            raise ValueError("Rankings must be loaded before generating preferences")
        
        # Convert rankings to list of (episode_idx, rank)
        ranked_episodes = [(int(episode_idx), rank) for episode_idx, rank in self.rankings.items()]
        
        # Sort by rank (lower rank is better)
        ranked_episodes.sort(key=lambda x: x[1])
        
        # Generate all pairs of episodes
        self.pairwise_preferences = []
        for i in range(len(ranked_episodes)):
            for j in range(i+1, len(ranked_episodes)):
                better_idx, better_rank = ranked_episodes[i]
                worse_idx, worse_rank = ranked_episodes[j]
                
                # The episode with lower rank is preferred
                self.pairwise_preferences.append({
                    "chosen_idx": better_idx,
                    "rejected_idx": worse_idx,
                    "chosen_trajectory": self.trajectories[better_idx],
                    "rejected_trajectory": self.trajectories[worse_idx],
                })
        
        # Save preferences
        preferences_path = self.logs_dir / "pairwise_preferences.pkl"
        with open(preferences_path, 'wb') as f:
            pickle.dump(self.pairwise_preferences, f)
        
        print(f"Generated {len(self.pairwise_preferences)} pairwise preferences")
        print(f"Saved preferences to {preferences_path}")
        
        # Save CPL dataset in data_dir root
        cpl_path = self.data_dir / "cpl_dataset.pkl"
        with open(cpl_path, 'wb') as f:
            pickle.dump(self.pairwise_preferences, f)
        
        print(f"Saved CPL dataset to {cpl_path}")

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    
    # Create generator
    generator = PreferenceDataGenerator(
        checkpoints=args.checkpoints,
        noise_scales=args.noise_scales,
        midi_file=args.midi_file,
        environment_name=args.environment_name,
        episodes_per_config=args.episodes_per_config,
        seed=args.seed,
        data_dir=args.data_dir,
        algorithm=args.algorithm
    )
    
    # Generate episodes
    generator.generate_episodes()
    
    # If not just generating, also handle rankings and preferences
    if not args.generate_only:
        # Load rankings
        generator.load_rankings(args.rankings_file)
        
        # Generate pairwise preferences
        generator.generate_pairwise_preferences()

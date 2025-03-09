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

class PreferenceDataGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        midi_file: str = None,
        environment_name: str = None,
        num_episodes: int = 10,
        seed: int = 42,
        video_dir: str = "preference_videos",
        data_dir: str = "preference_data",
        algorithm: str = "sac"
    ):
        """
        Initialize the preference data generator.
        
        Args:
            checkpoint_path: Path to the pretrained agent checkpoint
            midi_file: Path to MIDI file (if using a specific piece)
            environment_name: Environment name (if not using MIDI file)
            num_episodes: Number of episodes to roll out
            seed: Random seed
            video_dir: Directory to save videos
            data_dir: Directory to save preference data
            algorithm: The algorithm used for the agent (sac, qtopt, or hybrid_grpo)
        """
        self.checkpoint_path = checkpoint_path
        self.midi_file = midi_file
        self.environment_name = environment_name
        self.num_episodes = num_episodes
        self.seed = seed
        self.video_dir = Path(video_dir)
        self.data_dir = Path(data_dir)
        self.algorithm = algorithm
        
        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        with open(self.checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)
        
        # Initialize agent
        if self.algorithm == "sac":
            config = SACConfig(
                hidden_dims=(256, 256, 256),
                critic_dropout_rate=0.01,
                critic_layer_norm=True
            )
            self.agent = SAC.initialize(
                spec=self.spec,
                config=config,
                seed=self.seed,
                discount=0.8,
            )
            # Load checkpoint
            self.agent = self.agent.replace(
                actor=self.agent.actor.replace(params=self.checkpoint['params']),
                critic=self.agent.critic.replace(params=self.checkpoint['critic_params']),
                target_critic=self.agent.target_critic.replace(params=self.checkpoint['target_critic_params']),
                temp=self.agent.temp.replace(params=self.checkpoint['temp_params'])
            )
        elif self.algorithm == "qtopt":
            from qtopt import QTOpt, QTOptConfig
            config = QTOptConfig(
                hidden_dims=(256, 256, 256),
            )
            self.agent = QTOpt.initialize(
                spec=self.spec,
                config=config,
                seed=self.seed,
                discount=0.8,
            )
            # Load checkpoint (QTOpt has no actor or temp)
            self.agent = self.agent.replace(
                critic=self.agent.critic.replace(params=self.checkpoint.get('critic_params', self.checkpoint.get('params'))),
                target_critic=self.agent.target_critic.replace(params=self.checkpoint.get('target_critic_params', self.checkpoint.get('params')))
            )
        else:
            # For hybrid_grpo (PyTorch based)
            from hybrid_grpo import HybridGRPO
            self.agent = HybridGRPO(
                state_dim=self.spec.observation_dim,
                action_dim=self.spec.action_dim,
                hidden_dims=(256, 256, 256),
                lr=3e-4,
                gamma=0.8,
                num_samples=8,
                clip_param=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_workers=1,
                mini_batch_size=32,
            )
            # Load PyTorch checkpoint
            self.agent.actor.load_state_dict(self.checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(self.checkpoint['critic_state_dict'])
            self.agent.actor_optimizer.load_state_dict(self.checkpoint['actor_optimizer_state_dict'])
            self.agent.critic_optimizer.load_state_dict(self.checkpoint['critic_optimizer_state_dict'])
            self.agent.set_env(self.env)
    
    def generate_episodes(self):
        """Generate episodes and record videos."""
        print(f"Generating {self.num_episodes} episodes...")
        
        # Clear previous videos
        for video_file in self.video_dir.glob("*.mp4"):
            video_file.unlink()
        
        self.trajectories = []
        self.video_paths = []
        
        for episode_idx in range(self.num_episodes):
            print(f"Rolling out episode {episode_idx+1}/{self.num_episodes}")
            
            # Reset env and initialize trajectory storage
            timestep = self.env.reset()
            trajectory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "next_observations": [],
                "dones": [],
                "infos": []
            }
            
            # Store initial observation
            trajectory["observations"].append(timestep.observation)
            
            # Roll out episode
            episode_return = 0
            while not timestep.last():
                # Get action from agent
                if self.algorithm == "hybrid_grpo":
                    action = self.agent.select_action(timestep.observation)
                else:
                    action = self.agent.eval_actions(timestep.observation)
                
                # Take action in environment
                next_timestep = self.env.step(action)
                
                # Store transition
                trajectory["actions"].append(action)
                trajectory["rewards"].append(next_timestep.reward)
                trajectory["next_observations"].append(next_timestep.observation)
                trajectory["dones"].append(next_timestep.last())
                
                if not next_timestep.last():
                    trajectory["observations"].append(next_timestep.observation)
                
                # Update episode return
                episode_return += next_timestep.reward
                
                # Update timestep
                timestep = next_timestep
            
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
        
        # Save trajectories
        self._save_trajectories()
        
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
        trajectory_path = self.data_dir / "trajectories.pkl"
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
            rankings_path = self.data_dir / "rankings.json"
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
        for i in range(self.num_episodes):
            rank = int(input(f"Rank for episode {i} (1 is best): "))
            self.rankings[str(i)] = rank
        
        # Save rankings
        rankings_path = self.data_dir / "rankings.json"
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
        preferences_path = self.data_dir / "pairwise_preferences.pkl"
        with open(preferences_path, 'wb') as f:
            pickle.dump(self.pairwise_preferences, f)
        
        print(f"Generated {len(self.pairwise_preferences)} pairwise preferences")
        print(f"Saved preferences to {preferences_path}")
        
        # Also save in a format suitable for CPL training
        self._save_cpl_dataset()
    
    def _save_cpl_dataset(self):
        """Save dataset in a format suitable for CPL training."""
        cpl_data = []
        
        for pref in self.pairwise_preferences:
            chosen_traj = pref["chosen_trajectory"]
            rejected_traj = pref["rejected_trajectory"]
            
            # Format for CPL
            cpl_item = {
                "chosen": {
                    "observations": chosen_traj["observations"],
                    "actions": chosen_traj["actions"],
                    "rewards": chosen_traj["rewards"],
                },
                "rejected": {
                    "observations": rejected_traj["observations"],
                    "actions": rejected_traj["actions"],
                    "rewards": rejected_traj["rewards"],
                },
                "chosen_return": chosen_traj["return"],
                "rejected_return": rejected_traj["return"],
                "chosen_idx": pref["chosen_idx"],
                "rejected_idx": pref["rejected_idx"]
            }
            
            cpl_data.append(cpl_item)
        
        cpl_path = self.data_dir / "cpl_dataset.pkl"
        with open(cpl_path, 'wb') as f:
            pickle.dump(cpl_data, f)
        
        print(f"Saved CPL dataset to {cpl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate preference data for RLHF")
    parser.add_argument("--checkpoint", type=str, 
        default="/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-21-29-41/checkpoint_00920000.pkl", 
        help="Path to agent checkpoint")
    parser.add_argument("--midi_file", type=str, 
        default="/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid", 
        help="Path to MIDI file")
    parser.add_argument("--environment_name", type=str, default=None, help="Environment name (if not using MIDI)")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to roll out")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--video_dir", type=str, default="preference_videos", help="Directory to save videos")
    parser.add_argument("--data_dir", type=str, default="preference_data", help="Directory to save preference data")
    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "qtopt", "hybrid_grpo"], 
                        help="Algorithm used by the agent")
    parser.add_argument("--rankings_file", type=str, default=None, help="Path to JSON file with rankings")
    parser.add_argument("--generate_only", action="store_true", help="Only generate episodes without ranking")
    args = parser.parse_args()
    
    # Create generator
    generator = PreferenceDataGenerator(
        checkpoint_path=args.checkpoint,
        midi_file=args.midi_file,
        environment_name=args.environment_name,
        num_episodes=args.num_episodes,
        seed=args.seed,
        video_dir=args.video_dir,
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

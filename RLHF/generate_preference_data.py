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
    episodes_per_config: int = 1  # Number of episodes per checkpoint-noise combination
    seed: int = 42
    data_dir: str = "RLHF/preference_data"
    algorithm: str = "sac"
    
    # Optional settings
    rankings_file: Optional[str] = None
    generate_only: bool = False
    
    # Segment settings
    segment_length: float = 5.0  # Length of each segment in seconds
    control_timestep: float = 0.05  # Environment timestep
    steps_per_segment: int = field(init=False)
    
    # Add missing environment args
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = True
    reduced_action_space: bool = True
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = True
    disable_forearm_reward: bool = False
    camera_id: str = "piano/back"
    action_reward_observation: bool = True
    
    def __post_init__(self):
        self.steps_per_segment = int(self.segment_length / self.control_timestep)

class SegmentDataGenerator:
    def __init__(self, args):
        self.args = args
        
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Create output directories
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.base_data_dir = Path(args.data_dir)
        self.data_dir = self.base_data_dir / timestamp
        self.video_dir = self.data_dir / "videos"
        self.logs_dir = self.data_dir / "logs"
        
        for dir_path in [self.video_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = get_env(args, record_dir=self.video_dir)
        self.spec = EnvironmentSpec.make(self.env)
    
    def load_agent(self, checkpoint_path):
        """Load agent from checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Initialize SAC agent
        sac_config = SACConfig(
            hidden_dims=(256, 256, 256),
            critic_dropout_rate=0.01,
            critic_layer_norm=True,
        )
        agent = SAC.initialize(self.spec, sac_config, seed=self.args.seed)
        
        # Load parameters
        agent = agent.replace(
            actor=agent.actor.replace(params=checkpoint['params']),
            critic=agent.critic.replace(params=checkpoint['critic_params']),
            target_critic=agent.target_critic.replace(params=checkpoint['target_critic_params']),
            temp=agent.temp.replace(params=checkpoint['temp_params'])
        )
        return agent

    def generate_segments(self):
        """Generate segments from multiple checkpoints with noise."""
        segments = []
        segment_info = []
        
        for checkpoint_path in self.args.checkpoints:
            agent = self.load_agent(checkpoint_path)
            
            for noise_scale in self.args.noise_scales:
                print(f"Generating segments for checkpoint {checkpoint_path} with noise scale {noise_scale}")
                for _ in range(self.args.episodes_per_config):
                    timestep = self.env.reset()
                    current_segment = []
                    
                    while not timestep.last():
                        action = agent.eval_actions(timestep.observation)
                        if noise_scale > 0:
                            noise = np.random.normal(0, noise_scale, size=action.shape)
                            action = np.clip(action + noise, -1, 1)
                        
                        next_timestep = self.env.step(action)
                        
                        current_segment.append({
                            "state": timestep.observation,
                            "action": action,
                            "next_state": next_timestep.observation,
                        })
                        
                        if len(current_segment) == self.args.steps_per_segment:
                            segments.append(current_segment)
                            segment_info.append({
                                "checkpoint": checkpoint_path,
                                "noise_scale": noise_scale,
                                "start_time": len(segments) * self.args.segment_length
                            })
                            current_segment = []
                        
                        timestep = next_timestep
                    
                    # Handle partial segment at end of episode
                    if len(current_segment) > 0:
                        segments.append(current_segment)
                        segment_info.append({
                            "checkpoint": checkpoint_path,
                            "noise_scale": noise_scale,
                            "start_time": len(segments) * self.args.segment_length
                        })
                    
                    # Save video
                    video_files = sorted(self.video_dir.glob("*.mp4"), key=os.path.getctime)
                    if video_files:
                        latest_video = video_files[-1]
                        new_name = self.video_dir / f"episode_{len(segments):03d}_noise_{noise_scale:.2f}.mp4"
                        shutil.move(latest_video, new_name)
                        
                        for info in segment_info[-len(segments):]:
                            info["video_path"] = str(new_name)
        
        # Save all data
        data = {
            "segments": segments,
            "segment_info": segment_info,
            "args": self.args
        }
        with open(self.data_dir / "segments.pkl", "wb") as f:
            pickle.dump(data, f)
        
        return segments, segment_info
    
    def collect_ratings(self, segments, segment_info):
        """Collect ratings for each segment."""
        print("\nPlease rate each segment from 1 (worst) to 100 (best)")
        
        ratings = {}
        for i, (segment, info) in enumerate(zip(segments, segment_info)):
            print(f"\nSegment {i}:")
            print(f"From checkpoint: {info['checkpoint']}")
            print(f"Start time: {info['start_time']}s")
            print(f"Video: {info['video_path']}")
            
            while True:
                try:
                    rating = int(input(f"Rating (1-100): "))
                    if 1 <= rating <= 100:
                        ratings[i] = rating
                        break
                except ValueError:
                    print("Please enter a valid number")
        
        return ratings
    
    def generate_pairwise_data(self, segments, segment_info, ratings):
        """Generate pairwise preference data from segment ratings."""
        pairwise_data = []
        
        # Only compare segments from the same checkpoint
        for checkpoint in set(info["checkpoint"] for info in segment_info):
            # Get indices of segments from this checkpoint
            checkpoint_indices = [
                i for i, info in enumerate(segment_info) 
                if info["checkpoint"] == checkpoint
            ]
            
            # Generate all pairs from these segments
            for i in checkpoint_indices:
                for j in checkpoint_indices:
                    if i < j:  # Avoid duplicates
                        if ratings[i] != ratings[j]:  # Only include if there's a clear preference
                            # Higher rated segment is preferred
                            if ratings[i] > ratings[j]:
                                preferred = segments[i]
                                non_preferred = segments[j]
                            else:
                                preferred = segments[j]
                                non_preferred = segments[i]
                            
                            pairwise_data.append({
                                "preferred": preferred,
                                "non_preferred": non_preferred,
                                "rating_diff": abs(ratings[i] - ratings[j])
                            })
        
        return pairwise_data

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    
    print(f"Generating preference data for {args.midi_file}")
    # Create generator
    generator = SegmentDataGenerator(args)
    
    print("Generating segments")
    # Generate segments
    segments, segment_info = generator.generate_segments()

    print("Collecting ratings")
    # Collect ratings
    ratings = generator.collect_ratings(segments, segment_info)

    print("Generating pairwise data")
    # Generate pairwise data
    pairwise_data = generator.generate_pairwise_data(segments, segment_info, ratings)

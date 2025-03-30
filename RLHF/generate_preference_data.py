import os
import pickle
import numpy as np
import random
from pathlib import Path
import time
from typing import List, Optional
import shutil
from dataclasses import dataclass, field
import sys
sys.path.append('.')  # Add the root directory to path
from architecture.sac import SAC, SACConfig
from rl_dataclasses.specs import EnvironmentSpec
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
        # Add debug prints to track video recording
        if hasattr(env, '_write_frames'):
            original_write_frames = env._write_frames
            def debug_write_frames():
                print("\nAttempting to write video frames...")
                try:
                    original_write_frames()
                    print("Successfully wrote video frames")
                except Exception as e:
                    print(f"Error writing video frames: {e}")
            env._write_frames = debug_write_frames
        
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
    checkpoints: List[str] = None # Model checkpoints (use varied list)
    noise_scales: List[float] = field(default_factory=lambda: [0.0, 0.1])  # Start with no noise
    midi_file: str = None
    environment_name: Optional[str] = None
    episodes_per_config: int = 1  # Number of episodes per checkpoint-noise combination
    seed: int = 42
    data_dir: str = "RLHF/preference_data"
    algorithm: str = "sac"
    rankings_file: Optional[str] = None
    generate_only: bool = False
    segment_length: float = 5.0  # Length of each segment in seconds
    control_timestep: float = 0.05  # Environment timestep
    steps_per_segment: int = field(init=False)
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
        print(f"\nInitializing environment with video recording to: {self.video_dir}")  # Debug print
        print(f"Video directory exists: {self.video_dir.exists()}")  # Debug print
        print(f"Video directory is writable: {os.access(self.video_dir, os.W_OK)}")  # Debug print
        self.env = get_env(args, record_dir=self.video_dir)
        print(f"Environment type after wrapping: {type(self.env)}")  # Debug print
        print(f"Environment wrapper chain: {self._get_wrapper_chain(self.env)}")  # Debug print
        self.spec = EnvironmentSpec.make(self.env)
    
    def _get_wrapper_chain(self, env):
        """Helper to print the wrapper chain of an environment."""
        chain = []
        while hasattr(env, 'environment'):
            chain.append(type(env).__name__)
            env = env.environment
        chain.append(type(env).__name__)
        return ' -> '.join(chain)

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
        segments = []
        segment_info = []
        
        for checkpoint_path in self.args.checkpoints:
            agent = self.load_agent(checkpoint_path)
            
            for noise_scale in self.args.noise_scales:
                print(f"Generating segments for checkpoint {checkpoint_path} with noise scale {noise_scale}")
                for _ in range(self.args.episodes_per_config):
                    timestep = self.env.reset()
                    print(f"\nStarting new episode")  # Debug print
                    print(f"Environment type: {type(self.env)}")  # Debug print
                    if hasattr(self.env, '_record_dir'):
                        print(f"Recording directory in env: {self.env._record_dir}")  # Debug print
                    current_segment = []
                    episode_segments = []
                    episode_segment_info = []
                    
                    max_time = 15.0
                    max_steps = int(max_time / self.args.control_timestep)
                    total_steps = 0
                    
                    while not timestep.last() and total_steps < max_steps:
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
                        
                        total_steps += 1
                        
                        if len(current_segment) == self.args.steps_per_segment:
                            print(f"Current segment length: {len(current_segment)}")  # Debug print
                            if (len(episode_segments) + 1) * self.args.segment_length <= max_time:
                                episode_segments.append(current_segment)
                                episode_segment_info.append({
                                    "checkpoint": checkpoint_path,
                                    "noise_scale": noise_scale,
                                    "start_time": len(episode_segments) * self.args.segment_length
                                })
                                print(f"Episode segments count: {len(episode_segments)}")  # Debug print
                            current_segment = []
                        
                        timestep = next_timestep
                    
                    print("\nEpisode complete, checking for video...")  # Debug print
                    print(f"Total steps taken: {total_steps}")  # Debug print
                    print(f"Episode ended naturally: {timestep.last()}")  # Debug print
                    print(f"Max steps reached: {total_steps >= max_steps}")  # Debug print
                    
                    # Explicitly write frames if we hit max_steps
                    if total_steps >= max_steps and hasattr(self.env, '_write_frames'):
                        print("Max steps reached - explicitly writing video frames...")
                        try:
                            self.env._write_frames()
                            print("Successfully wrote video frames")
                        except Exception as e:
                            print(f"Error writing video frames: {e}")
                    
                    try:
                        if hasattr(self.env, 'latest_filename'):
                            print(f"Latest recorded video: {self.env.latest_filename}")  # Debug print
                    except ValueError as e:
                        print(f"Error checking latest_filename: {e}")  # Debug print
                    
                    # Save video and update segment info
                    video_files = sorted(self.video_dir.glob("*.mp4"), key=os.path.getctime)
                    print(f"\nChecking video files in {self.video_dir}:")  # Debug print
                    print(f"Found {len(video_files)} video files")  # Debug print
                    if len(video_files) > 0:
                        print(f"Latest video file: {video_files[-1]}")  # Debug print
                    
                    if video_files:
                        latest_video = video_files[-1]
                        new_name = self.video_dir / f"episode_{len(segments):03d}_noise_{noise_scale:.2f}.mp4"
                        shutil.move(latest_video, new_name)
                        print(f"Saved video: {new_name}")  # Debug print
                        
                        # Add video path to all segments from this episode
                        for info in episode_segment_info:
                            info["video_path"] = str(new_name)
                        
                        print(f"Episode segments before extend: {len(episode_segments)}")  # Debug print
                        print(f"Main segments before extend: {len(segments)}")  # Debug print
                        
                        # Add segments and info to main lists
                        segments.extend(episode_segments)
                        segment_info.extend(episode_segment_info)
                        
                        print(f"Main segments after extend: {len(segments)}")  # Debug print
                        print(f"Episode segments: {[len(s) for s in episode_segments]}")  # Debug print
                        print(f"Segment info length: {len(segment_info)}")  # Debug print
                    else:
                        print("WARNING: No video files found - segments not saved!")  # Debug print
                    
                    print(f"End of episode - episode_segments: {len(episode_segments)}")  # Debug print
        
        # Print final counts
        print(f"\nFinal segment count: {len(segments)}")
        print(f"Final segment_info count: {len(segment_info)}")
        
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
    
    # Save the data
    data = {
        "pairwise_data": pairwise_data,
        "segments": segments,
        "segment_info": segment_info,
        "ratings": ratings
    }
    
    # Save to a file in the same directory as the videos
    data_file = generator.data_dir / "preference_data.pkl"
    print(f"\nSaving preference data to: {data_file}")
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("Done!")
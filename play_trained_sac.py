import argparse
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime

from robopianist.wrappers import PianoSoundVideoWrapper
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from mujoco_utils import composer_utils

import sac
import specs
from robopianist import suite
from dm_env_wrappers import ObservationActionRewardWrapper

def create_env(midi_sequence):
    """Create a single environment instance."""
    task = PianoWithShadowHands(
        midi=midi_sequence,
        n_steps_lookahead=1,
        trim_silence=True,
        wrong_press_termination=False,
        initial_buffer_time=0.0,
        disable_fingering_reward=False,
        disable_forearm_reward=False,
        disable_colorization=False,
        disable_hand_collisions=False,
    )
    
    env = composer_utils.Environment(
        task=task,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=True
    )

    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir="videos",  # Videos will be saved here
    )

    return CanonicalSpecWrapper(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
                      help="Path to the checkpoint file (e.g., checkpoint_0010000.pkl)",
                      default="/tmp/robopianist/rl/checkpoint_0710000.pkl")
    parser.add_argument("--environment_name", type=str,
                      default="RoboPianist-debug-TwinkleTwinkleRousseau-v0",
                      help="Environment name from robopianist.music.DEBUG_MIDIS")
    args = parser.parse_args()

    # Create environment first to get initial observation
    task_kwargs = {
        'n_steps_lookahead': 10,  # Changed to match train.py
        'trim_silence': True,
        'wrong_press_termination': False,
        'gravity_compensation': True,  # Changed to match train.py
        'reduced_action_space': True,  # Changed to match train.py
        'control_timestep': 0.05,
    }

    env = suite.load(
        environment_name=args.environment_name,
        task_kwargs=task_kwargs,
    )
    
    # Add custom video path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    song_name = args.environment_name.split("-")[2]
    video_dir = Path("videos") / f"{song_name}_{timestamp}"  # e.g. "videos/NocturneRousseau_20240319_143022"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir=video_dir,
    )

    # Add action reward wrapper to match training
    env = ObservationActionRewardWrapper(env)
    env = CanonicalSpecWrapper(env)

    # Get initial observation and spec
    timestep = env.reset()
    print("\nDEBUG: Timestep observation structure:")
    print(f"Observation keys: {timestep.observation.keys()}")
    for k, v in timestep.observation.items():
        print(f"Key: {k}, Shape: {np.shape(v)}, Type: {type(v)}")

    spec = specs.EnvironmentSpec.make(env)
    print("\nDEBUG: Spec structure:")
    print(f"Observation spec: {spec.observation}")
    print(f"Action spec: {spec.action}")

    # Initialize agent without observation parameter
    agent = sac.SAC.initialize(
        spec=spec,
        config=sac.SACConfig(),
        seed=42,
        discount=0.99,
    )

    # Load checkpoint
    with open(args.checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Load parameters from checkpoint
    agent = agent.replace(
        actor=agent.actor.replace(params=checkpoint['params']),
        critic=agent.critic.replace(params=checkpoint['critic_params']),
        target_critic=agent.target_critic.replace(params=checkpoint['target_critic_params']),
        temp=agent.temp.replace(params=checkpoint['temp_params'])
    )

    # Run evaluation episode
    total_reward = 0
    
    print("Starting evaluation...")
    while not timestep.last():
        action = agent.eval_actions(timestep.observation)
        timestep = env.step(action)
        total_reward += timestep.reward
        
    print(f"Episode finished with total reward: {total_reward}")
    print(f"Video saved to: {env.latest_filename}")

if __name__ == "__main__":
    main() 
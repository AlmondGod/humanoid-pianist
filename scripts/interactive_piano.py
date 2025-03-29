#!/usr/bin/env python3

"""
Interactive RoboPianist environment with Shadow Hands and Unitree G1 humanoid.
This script provides an interactive visualization of the piano environment with 
shadow hands and an integrated Unitree G1 robot.
"""

from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass
import numpy as np
import time
from robopianist.viewer import launch
from robopianist import wrappers
from eval import load

@dataclass(frozen=True)
class Args:
    """Arguments for the interactive piano environment."""
    # Environment configuration
    seed: int = 42
    midi_file: Optional[Path] = None
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = True
    reduced_action_space: bool = True
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = True
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = True
    
    # Viewer configuration
    width: int = 1024 
    height: int = 768
    camera_id: Optional[str | int] = "piano/back"

def get_env(args: Args):
    """Create the environment based on provided arguments."""
    # Load the environment with PianoWithShadowHandsAndG1 wrapper
    env = load(
        environment_name=None,
        midi_file=args.midi_file,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    
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


def random_policy(time_step):
    """A random policy that randomly selects actions."""
    spec = env.action_spec()
    return np.random.uniform(
        low=spec.minimum, high=spec.maximum, size=spec.shape
    ).astype(spec.dtype)


def keyboard_policy(time_step):
    """
    A policy that allows keyboard control of the shadow hands.
    
    This is a simplified version that maps certain keys to specific actions.
    In a more complex implementation, you could use a full keyboard event system.
    
    For demonstration purposes, we're implementing a very basic policy.
    """
    # Get the action spec
    spec = env.action_spec()
    action = np.zeros(spec.shape, dtype=spec.dtype)
    
    # Get simulation time to create a simple oscillating pattern
    t = time.time() * 0.5
    
    # Move forearms (left and right hands)
    action[0] = 0.3 * np.sin(t)  # Left hand forearm movement
    action[24] = 0.3 * np.sin(t + np.pi)  # Right hand forearm movement (opposite phase)
    
    # Simple finger movement pattern
    # Thumb
    action[1:5] = 0.2 * np.sin(t + np.pi/4)  # Left thumb
    action[25:29] = 0.2 * np.sin(t + np.pi/4)  # Right thumb
    
    # Index
    action[5:9] = 0.25 * np.sin(t + np.pi/2)  # Left index
    action[29:33] = 0.25 * np.sin(t + np.pi/2)  # Right index
    
    # Other fingers with different phases
    action[9:13] = 0.3 * np.sin(t + 3*np.pi/4)  # Left middle
    action[13:17] = 0.3 * np.sin(t + np.pi)  # Left ring
    action[17:21] = 0.3 * np.sin(t + 5*np.pi/4)  # Left pinky
    
    action[33:37] = 0.3 * np.sin(t + 3*np.pi/4)  # Right middle
    action[37:41] = 0.3 * np.sin(t + np.pi)  # Right ring
    action[41:45] = 0.3 * np.sin(t + 5*np.pi/4)  # Right pinky
    
    # Ensure actions are within bounds
    action = np.clip(action, spec.minimum, spec.maximum)
    
    return action


def launch_viewer(env, policy=None, width=1024, height=768, camera_id=None):
    """Launch the RoboPianist viewer with the given environment and policy."""
    if camera_id:
        print(f"Note: Selected camera '{camera_id}' will be available in the viewer")
        print("You can select it from the viewer's camera menu if needed")
    
    launch(
        environment_loader=env,
        policy=policy,
        title="RoboPianist with Unitree G1",
        width=width,
        height=height
    )


def main(args: Args) -> None:
    """Main function to create and run the interactive environment."""
    # Set the random seed
    np.random.seed(args.seed)
    
    # Create the environment
    global env  # For policy access
    env = get_env(args)
    
    # Select policy (automatic keyboard emulation for demonstration)
    selected_policy = keyboard_policy
    
    # Launch the viewer with the selected policy
    launch_viewer(
        env=env,
        policy=selected_policy,
        width=args.width, 
        height=args.height,
        camera_id=args.camera_id
    )


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__)) 
#!/usr/bin/env python3

"""
Interactive RoboPianist environment with Shadow Hands and Unitree G1 humanoid.
This script provides an interactive visualization of the piano environment with 
shadow hands and an integrated Unitree G1 robot.
"""

from pathlib import Path
from typing import Optional, Tuple, Literal, List
import tyro
from dataclasses import dataclass
import numpy as np
import os
import time
from robot_descriptions import g1_mj_description
from dm_control import mujoco
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.composer import variation as base_variation

from robopianist import suite
from robopianist.music import midi_file
from robopianist.models.arenas import stage
from robopianist.suite.tasks import base
from robopianist.suite.tasks import piano_with_shadow_hands

# Import the RoboPianist viewer instead of dm_control viewer
from robopianist.viewer import launch

import robopianist.wrappers as robopianist_wrappers
import dm_env_wrappers as wrappers
from dm_control.composer import Entity


@dataclass(frozen=True)
class Args:
    """Arguments for the interactive piano environment."""
    # Environment configuration
    seed: int = 42
    midi_file: Optional[Path] = None
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    
    # Unitree G1 integration
    add_unitree_g1: bool = True
    unitree_g1_path: Optional[str] = None  # Path to Unitree G1 model, if None will use default
    unitree_position: Tuple[float, float, float] = (0.0, 0.4, 0.0)  # Behind the piano
    
    # Viewer configuration
    width: int = 1024 
    height: int = 768
    camera_id: Optional[str | int] = "piano/back"


class G1Entity(Entity):
    """Entity wrapper for Unitree G1 model.
    
    This class wraps the G1 model as a dm_control Entity, making it compatible with
    the robopianist and dm_control APIs.
    """
    
    def __init__(self, model_path):
        """Initialize G1Entity.
        
        Args:
            model_path: Path to the G1 model XML file
        """
        self._model_path = model_path
        self._attached = []  # Necessary for Entity interface
        
        # Fix the XML file by removing keyframes and fixing mesh paths
        import tempfile
        import xml.etree.ElementTree as ET
        import os
        
        # Parse the XML file
        try:
            print(f"Preprocessing G1 model XML file...")
            tree = ET.parse(model_path)
            root = tree.getroot()
            
            # Find the compiler element and update meshdir to absolute path
            original_dir = os.path.dirname(model_path)
            assets_dir = os.path.join(original_dir, "assets")
            compilers = root.findall(".//compiler")
            
            modifications_made = False
            
            if compilers:
                for compiler in compilers:
                    if 'meshdir' in compiler.attrib:
                        print(f"Updating meshdir from '{compiler.attrib['meshdir']}' to '{assets_dir}'")
                        compiler.attrib['meshdir'] = assets_dir
                        modifications_made = True
            
            # Find and remove all keyframe elements
            keyframes = root.findall(".//keyframe")
            if keyframes:
                for keyframe in keyframes:
                    print(f"Removing keyframe element to prevent qpos size mismatch")
                    # Find the parent of the keyframe and remove the keyframe
                    parent_map = {c: p for p in tree.iter() for c in p}
                    if keyframe in parent_map:
                        parent_map[keyframe].remove(keyframe)
                        modifications_made = True
            
            # Find and remove all freejoint elements
            freejoints = root.findall(".//freejoint")
            if freejoints:
                for freejoint in freejoints:
                    print(f"Removing freejoint element to prevent 'free joint can only be used on top level' error")
                    # Find the parent of the freejoint and remove the freejoint
                    parent_map = {c: p for p in tree.iter() for c in p}
                    if freejoint in parent_map:
                        parent_map[freejoint].remove(freejoint)
                        modifications_made = True
            
            if modifications_made:
                # Create a temporary file to save the modified XML
                with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    print(f"Writing modified G1 model to temporary file: {tmp_path}")
                    tree.write(tmp_path)
                    self._model_path = tmp_path
            else:
                print("No modifications needed to the G1 model file")
        except Exception as e:
            print(f"Error preprocessing G1 model XML: {e}")
            print("Will try to load the original file anyway")
        
        # Now load the modified model
        try:
            self._mjcf_model = mjcf.from_path(self._model_path)
            print("Successfully loaded G1 model with MJCF API")
        except Exception as e:
            print(f"Error loading G1 model with MJCF API: {e}")
            
            # One last attempt - try loading with direct asset path mapping
            try:
                print("Trying alternative loading method with asset overrides...")
                assets = {}
                if os.path.exists(assets_dir):
                    # Load all STL files in the assets directory
                    for filename in os.listdir(assets_dir):
                        if filename.lower().endswith('.stl'):
                            file_path = os.path.join(assets_dir, filename)
                            assets[filename] = open(file_path, 'rb').read()
                            print(f"Added asset: {filename}")
                
                self._mjcf_model = mjcf.from_path(model_path, assets=assets)
                print("Successfully loaded G1 model with asset overrides")
            except Exception as asset_error:
                print(f"All attempts to load G1 model failed: {asset_error}")
                raise
        
        # Build entity-specific attributes
        self._build()
    
    @property
    def mjcf_model(self):
        """Return the MuJoCo model."""
        return self._mjcf_model
    
    def _build(self, name=None):
        """Build the entity."""
        # Nothing to do here, the model is already loaded
        pass
    
    def initialize_episode(self, physics, random_state):
        """Initialize the entity for a new episode.
        
        Args:
            physics: The physics object
            random_state: A numpy random state object
        """
        # Nothing to do here, we'll handle initialization in the add_unitree_g1_to_env function
        pass
    
    @property
    def actuators(self):
        """Return the actuators of the model."""
        return self._mjcf_model.find_all('actuator')
        
    def attach(self, entity, attachment_site):
        """Attach another entity to this one.
        
        Args:
            entity: The entity to attach
            attachment_site: The site to attach it to
            
        Returns:
            The attached entity
        """
        self._attached.append(entity)
        return entity


def get_env(args: Args):
    """Create the environment based on provided arguments."""
    # Create basic piano environment
    env = suite.load(
        environment_name=args.environment_name,
        midi_file=args.midi_file,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            change_color_on_activation=True,
        ),
    )
    
    # Add Unitree G1 if requested
    if args.add_unitree_g1:
        add_unitree_g1_to_env(env, args.unitree_position, args.unitree_g1_path)
    
    # Add evaluation and sound wrappers for better interaction
    env = robopianist_wrappers.MidiEvaluationWrapper(
        environment=env, deque_size=1
    )
    
    # Add piano sound wrapper (if needed)
    # env = robopianist_wrappers.PianoSoundWrapper(environment=env)
    
    return env


def add_unitree_g1_to_env(env, position, model_path=None):
    """Add a Unitree G1 to the environment.
    
    Args:
        env: The environment to add the G1 to
        position: The (x, y, z) position of the G1
        model_path: Path to the G1 model XML file. If None, will try to find it
        
    Returns:
        The G1 entity if successful, None otherwise
    """
    print("Adding Unitree G1 to the environment...")
    
    # Get the arena from the environment
    task = env.task
    arena = task.arena
    print(f"Arena type: {type(arena)}")
    
    # If a model path wasn't provided, look in standard locations
    if model_path is None:
        # Try in MuJoCo Menagerie directory, which might be:
        # 1. In the root of the repository
        # 2. In a user's home directory
        # 3. In a standard system location
        potential_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_g1", "g1_modified.xml"),  # Modified G1 model in the script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mujoco_menagerie", "unitree_g1", "g1_modified.xml"),  # Repository root with modified model
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_g1", "g1.xml"),  # Original model in script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mujoco_menagerie", "unitree_g1", "g1.xml"),  # Repository root with original model
            os.path.expanduser("~/mujoco_menagerie/unitree_g1/g1_modified.xml"),  # Modified model in user's home directory
            os.path.expanduser("~/mujoco_menagerie/unitree_g1/g1.xml"),  # User's home directory
            "/usr/local/share/mujoco_menagerie/unitree_g1/g1_modified.xml",  # System location with modified model
            "/usr/local/share/mujoco_menagerie/unitree_g1/g1.xml",  # System location
        ]
        
        # Try each potential path
        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                if "g1_modified.xml" in path:
                    print(f"Found modified G1 model (without hands) at {path}")
                else:
                    print(f"Found G1 model at {path}")
                break
                
        if model_path is None:
            print("Could not find Unitree G1 model. Please clone the MuJoCo Menagerie repository")
            print("and make sure the g1.xml file is in mujoco_menagerie/unitree_g1/g1.xml")
            print("\nTo properly integrate the Unitree G1 model, please:")
            print("1. Clone the MuJoCo Menagerie repository:")
            print("   git clone https://github.com/google-deepmind/mujoco_menagerie.git")
            print("2. Specify the path to the G1 model when running the script:")
            print("   python interactive_piano.py --unitree_g1_path=/path/to/mujoco_menagerie/unitree_g1/g1.xml")
            print("\nContinuing with a simple placeholder...\n")
            
            # Try a very simple fallback: add a placeholder box
            try:
                print("Adding a placeholder box instead of G1...")
                # This should work with most MuJoCo versions
                placeholder = arena.mjcf_model.worldbody.add(
                    'geom',
                    name='g1_placeholder',
                    type='box',
                    size=[0.2, 0.3, 0.8],  # Roughly humanoid sized
                    pos=position,
                    rgba=[0.2, 0.2, 0.2, 0.8]  # Semi-transparent dark gray
                )
                print("Added placeholder box to represent G1")
                return placeholder
            except Exception as box_error:
                print(f"Even placeholder box failed: {box_error}")
                return None
    
    print(f"Loading G1 model from: {model_path}")
    
    try:
        # Load the G1 model
        g1_entity = G1Entity(model_path)
        print("Successfully created G1 Entity")
        
        # Create an attachment site on the ground
        print("Creating attachment site...")
        attachment_site = arena.mjcf_model.worldbody.add(
            'site', 
            name='g1_attachment', 
            size=[0.01, 0.01, 0.01], 
            pos=position
        )
        print(f"Attachment site created: {attachment_site}")
                
        # Attach the G1 model to the site
        print("Attaching G1 Entity to arena...")
        arena.attach(g1_entity, attachment_site)
        print("G1 attached successfully!")
        
        # Disable actuators for now as we're just visualizing
        for actuator in g1_entity.actuators:
            actuator.ctrllimited = True
            actuator.ctrlrange = [0, 0]  # Zero range to disable
        print("G1 actuators disabled")
        
        # Add a function to reset the G1 orientation on environment reset
        old_reset = env.reset
                
        def initialize_g1_pose(physics):
            try:
                # Find the pelvis body which is the root of the G1 model
                body_name = "g1_29dof_rev_1_0/pelvis"
                pelvis_id = physics.model.name2id(body_name, "body")
                
                if pelvis_id >= 0:
                    print(f"Setting G1 position and orientation for body_id: {pelvis_id} ({body_name})")
                    
                    # Since we removed the freejoint, we need to directly set the position
                    # using xpos and xquat instead of qpos
                    physics.data.xpos[pelvis_id] = position
                    physics.data.xquat[pelvis_id] = [1, 0, 0, 0]  # Identity quaternion
                else:
                    print(f"Warning: Could not find G1 pelvis body with name: {body_name}")
            except Exception as e:
                print(f"Warning: Could not initialize G1 pose: {e}")
                pass
                
        def reset_with_g1_orientation(*args, **kwargs):
            # Call the original reset
            result = old_reset(*args, **kwargs)
            
            # Now adjust the G1 orientation
            try:
                initialize_g1_pose(env.physics)
            except Exception as e:
                print(f"Error setting G1 orientation: {e}")
                
            return result
            
        # Replace the reset method with our custom one
        env.reset = reset_with_g1_orientation
        print("Added G1 orientation hook to environment reset")
        
        return g1_entity
        
    except Exception as e:
        print(f"Error loading Unitree G1 model: {e}")
        print("\nThere was an error loading the G1 model. Please check:")
        print("1. The file path is correct")
        print("2. The file is a valid MuJoCo model")
        print("3. Your MuJoCo version is compatible with the model")
        print("\nFalling back to a placeholder box...\n")
        
        # Try a simple fallback: add a placeholder box
        try:
            placeholder = arena.mjcf_model.worldbody.add(
                'geom',
                name='g1_placeholder',
                type='box',
                size=[0.2, 0.3, 0.8],  # Roughly humanoid sized
                pos=position,
                rgba=[0.2, 0.2, 0.2, 0.8]  # Semi-transparent dark gray
            )
            print("Added placeholder box to represent G1")
            return placeholder
        except Exception as box_error:
            print(f"Even placeholder box failed: {box_error}")
            return None


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
    
    # For actual keyboard control, you would need to integrate with the viewer's
    # keyboard event system. This is a simplified placeholder that just moves
    # the hands in a predefined pattern.
    
    # Get simulation time to create a simple oscillating pattern
    t = time.time() * 0.5
    
    # Example of simple automatic movements (simulating keyboard control)
    # In a real keyboard controller, these would be replaced with key mappings
    
    # Action indices might need adjustment based on your specific setup
    # Typically for shadow hands:
    # - First actions control the forearm movement
    # - Remaining actions control individual finger joints
    
    # Move forearms (left and right hands)
    action[0] = 0.3 * np.sin(t)  # Left hand forearm movement
    action[24] = 0.3 * np.sin(t + np.pi)  # Right hand forearm movement (opposite phase)
    
    # Simple finger movement pattern (adjust indices as needed)
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
    # Use the RoboPianist viewer instead of dm_control's viewer
    # Note: The RoboPianist viewer may choose its own default camera,
    # not necessarily respecting camera_id directly
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
    
    # Print environment information
    print("\n========= Interactive RoboPianist Environment =========")
    print(f"MIDI File: {args.midi_file or args.environment_name}")
    print(f"Control timestep: {args.control_timestep}")
    print(f"Fingering reward: {'Disabled' if args.disable_fingering_reward else 'Enabled'}")
    print(f"Forearm reward: {'Disabled' if args.disable_forearm_reward else 'Enabled'}")
    print(f"Colorization: {'Disabled' if args.disable_colorization else 'Enabled'}")
    print(f"Hand collisions: {'Disabled' if args.disable_hand_collisions else 'Enabled'}")
    print(f"Unitree G1 Integration: {'Enabled' if args.add_unitree_g1 else 'Disabled'}")
    print("======================================================")
    
    # Print control instructions
    print("\nControl Information:")
    print("-----------------")
    print("This demo uses an automatic finger movement pattern.")
    print("In a full implementation, you would add keyboard controls")
    print("to manipulate individual finger joints and hand positions.")
    print("\nPress ESC to quit the viewer.")
    print("-----------------")
    
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
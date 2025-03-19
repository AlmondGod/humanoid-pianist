from pathlib import Path
from typing import Optional, Tuple, Literal
import tyro
from dataclasses import dataclass, asdict
import time
import random
import numpy as np
from tqdm import tqdm
import pickle
import os

import sac
import specs
import replay
from hybrid_grpo import HybridGRPO  # Import only
from qtopt import QTOpt, QTOptConfig  # Add QTOpt import
from dm_control import mjcf
from dm_control.composer import Entity

# from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers


from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from dm_control import composer
from mujoco_utils import composer_utils

from robopianist import music
from piano_with_shadow_hands_and_g1 import PianoWithShadowHandsAndG1

# RoboPianist-repertoire-150.
_BASE_REPERTOIRE_NAME = "RoboPianist-repertoire-150-{}-v0"
REPERTOIRE_150 = [_BASE_REPERTOIRE_NAME.format(name) for name in music.PIG_MIDIS]
_REPERTOIRE_150_DICT = dict(zip(REPERTOIRE_150, music.PIG_MIDIS))

# RoboPianist-etude-12.
_BASE_ETUDE_NAME = "RoboPianist-etude-12-{}-v0"
ETUDE_12 = [_BASE_ETUDE_NAME.format(name) for name in music.ETUDE_MIDIS]
_ETUDE_12_DICT = dict(zip(ETUDE_12, music.ETUDE_MIDIS))

# RoboPianist-debug.
_DEBUG_BASE_NAME = "RoboPianist-debug-{}-v0"
DEBUG = [_DEBUG_BASE_NAME.format(name) for name in music.DEBUG_MIDIS]
_DEBUG_DICT = dict(zip(DEBUG, music.DEBUG_MIDIS))

# All valid environment names.
ALL = REPERTOIRE_150 + ETUDE_12 + DEBUG
_ALL_DICT: Dict[str, Union[Path, str]] = {
    **_REPERTOIRE_150_DICT,
    **_ETUDE_12_DICT,
    **_DEBUG_DICT,
}


def load(
    environment_name: str,
    midi_file: Optional[Path] = None,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
) -> composer.Environment:
    """Loads a RoboPianist environment.

    Args:
        environment_name: Name of the environment to load. Must be of the form
            "RoboPianist-repertoire-150-<name>-v0", where <name> is the name of a
            PIG dataset MIDI file in camel case notation.
        midi_file: Path to a MIDI file to load. If provided, this will override
            `environment_name`.
        seed: Optional random seed.
        stretch: Stretch factor for the MIDI file.
        shift: Shift factor for the MIDI file.
        recompile_physics: Whether to recompile the physics.
        legacy_step: Whether to use the legacy step function.
        task_kwargs: Additional keyword arguments to pass to the task.
    """
    if midi_file is not None:
        midi = music.load(midi_file, stretch=stretch, shift=shift)
    else:
        if environment_name not in ALL:
            raise ValueError(
                f"Unknown environment {environment_name}. "
                f"Available environments: {ALL}"
            )
        midi = music.load(_ALL_DICT[environment_name], stretch=stretch, shift=shift)

    task_kwargs = task_kwargs or {}

    return composer_utils.Environment(
        task=PianoWithShadowHandsAndG1(midi=midi, **task_kwargs),
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
    )

__all__ = [
    "ALL",
    "DEBUG",
    "ETUDE_12",
    "REPERTOIRE_150",
    "load",
]

@dataclass(frozen=True)
class Args:
    root_dir: str = "eval"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 100000
    eval_interval: int = 40000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = True
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = "almond-maj-projects"
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    environment_name: Optional[str] = None
    midi_file: Optional[Path] = Path("/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid")
    load_checkpoint: Optional[Path] = Path("/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-21-29-41/checkpoint_00920000.pkl")  # Path to checkpoint file for resuming training
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
    frame_stack: int = 1
    clip: bool = True
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "panning_camera"
    action_reward_observation: bool = True
    agent_config: sac.SACConfig = sac.SACConfig()
    algorithm: Literal["sac"] = "sac"  # Add QTOpt option
    add_unitree_g1: bool = False
    unitree_g1_path: Optional[str] = "/Users/almondgod/Repositories/robopianist/robopianist-rl/mujoco_menagerie/unitree_g1/g1_modified.xml"
    unitree_position: Tuple[float, float, float] = (0.0, 0.4, 0.7)


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_env(args: Args, record_dir: Optional[Path] = None):
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
    
    # Add G1 wrapper if specified
    if getattr(args, 'add_unitree_g1', False):
        g1_position = getattr(args, 'unitree_position', (0.0, 0.4, 0.7))
        g1_model_path = getattr(args, 'unitree_g1_path', None)
        env = UnitreeG1Wrapper(env, model_path=g1_model_path, position=g1_position)
    
    return env


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"{args.algorithm.upper()}-{args.midi_file}-{args.seed}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Set random seeds to: {args.seed}")

    save_dir = experiment_dir / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving video to {save_dir}")

    env = get_env(args)
    eval_env = get_env(args, record_dir=save_dir)

    # Add these debug lines to print observation dimensions
    print("=" * 50)
    timestep = env.reset()
    print(f"ENVIRONMENT OBSERVATION SHAPE: {timestep.observation.shape}")
    print(f"OBSERVATION FIRST 10 VALUES: {timestep.observation[:10]}")
    print("=" * 50)

    spec = specs.EnvironmentSpec.make(env)
    print(f"SPEC OBSERVATION DIM: {spec.observation_dim}")
    print("=" * 50)

    # Initialize agent based on algorithm choice
    if args.algorithm == "sac":
        print("Using SAC")
        agent = sac.SAC.initialize(
            spec=spec,
            config=args.agent_config,
            seed=args.seed,
            discount=args.discount,
        )
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Load JAX checkpoint (for SAC or QTOpt)
            agent = agent.replace(
                actor=agent.actor.replace(params=checkpoint['params']),
                critic=agent.critic.replace(params=checkpoint['critic_params']),
                target_critic=agent.target_critic.replace(params=checkpoint['target_critic_params']),
                temp=agent.temp.replace(params=checkpoint['temp_params'])
            )
            print(f"Successfully loaded JAX checkpoint for {args.algorithm}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with a fresh model")

    print(f"evaluating...")

    # Eval.
    timestep = eval_env.reset()
    while not timestep.last():
        timestep = eval_env.step(agent.eval_actions(timestep.observation))

    print(f"Evaluation complete. Stats: {eval_env.get_statistics()}")


# Add G1Entity class
class G1Entity(Entity):
    """Entity wrapper for Unitree G1 model."""

    def __init__(self, model_path):
        """Initialize G1Entity.

        Args:
            model_path: Path to the G1 model XML file
        """
        self._model_path = model_path
        self._attached = []
        self._observables = {}
        
        # Fix the XML file by removing keyframes and fixing mesh paths
        import tempfile
        import xml.etree.ElementTree as ET
        import os
        
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

    @property
    def mjcf_model(self):
        return self._mjcf_model
    
    def _build(self):
        pass
    
    def initialize_episode(self, physics, random_state):
        pass
        
    @property
    def actuators(self):
        return self._actuators
        
    @property
    def observables(self):
        class ObservablesWrapper:
            def __init__(self, obs_dict):
                self._obs_dict = obs_dict
            def as_dict(self):
                return self._obs_dict
        return ObservablesWrapper(self._observables)

# Add G1 wrapper
class UnitreeG1Wrapper(wrappers.DmControlWrapper):
    """Wrapper that adds a Unitree G1 robot to the environment."""
    
    def __init__(self, env, model_path=None, position=(0.0, 0.0, 0.0)):
        super().__init__(env)
        self._model_path = model_path
        self._position = position
        self._g1_entity = None
        self._add_g1_to_env()

    def _add_g1_to_env(self):
        """Add G1 robot to the environment."""
        if self._model_path is None:
            # Try standard locations
            potential_paths = [
                "mujoco_menagerie/unitree_g1/g1.xml",
                os.path.expanduser("~/mujoco_menagerie/unitree_g1/g1.xml"),
                "/usr/local/share/mujoco_menagerie/unitree_g1/g1.xml",
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    self._model_path = path
                    print(f"Found G1 model at {path}")
                    break
                    
            if self._model_path is None:
                raise ValueError("Could not find Unitree G1 model. Please specify model_path.")

        try:
            # Get the arena from the environment - use self.environment instead of self._env
            arena = self.environment.task.arena
            
            # Create G1 entity
            self._g1_entity = G1Entity(self._model_path)
            
            # Create attachment site
            attachment_site = arena.mjcf_model.worldbody.add(
                'site',
                name='g1_attachment',
                size=[0.01, 0.01, 0.01],
                pos=self._position
            )
            
            # Attach G1 to arena
            arena.attach(self._g1_entity, attachment_site)
            print("Successfully added G1 to environment")
            
        except Exception as e:
            print(f"Error adding G1 to environment: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            self._g1_entity = None

    def reset(self, *args, **kwargs):
        timestep = super().reset(*args, **kwargs)
        
        # Set G1 orientation on reset
        if self._g1_entity is not None:
            try:
                # Use self.environment instead of self._env
                physics = self.environment.physics
                body_name = "g1_29dof_rev_1_0/pelvis"
                pelvis_id = physics.model.name2id(body_name, "body")
                
                if pelvis_id >= 0:
                    # Set position and orientation
                    physics.data.xpos[pelvis_id] = self._position
                    # 180-degree rotation around Z axis
                    physics.data.xquat[pelvis_id] = [0, 0, 0, 1]
                    physics.forward()
            except Exception as e:
                print(f"Warning: Could not set G1 orientation: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                
        return timestep


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

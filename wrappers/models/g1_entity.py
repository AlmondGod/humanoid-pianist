"""Unitree G1 composer class."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types


@dataclass(frozen=True)
class Dof:
    """G1 degree of freedom."""
    joint_type: str
    axis: Tuple[int, int, int]
    stiffness: float
    joint_range: Tuple[float, float]
    reflect: bool = False


# Define the G1's degrees of freedom
_G1_DOFS: Dict[str, Dof] = {
    "pelvis_x": Dof(
        joint_type="slide",
        axis=(1, 0, 0),
        stiffness=1000,
        joint_range=(-1.0, 1.0),
    ),
    "pelvis_y": Dof(
        joint_type="slide",
        axis=(0, 1, 0),
        stiffness=1000,
        joint_range=(-1.0, 1.0),
    ),
    "pelvis_z": Dof(
        joint_type="slide",
        axis=(0, 0, 1),
        stiffness=1000,
        joint_range=(0.0, 1.0),
    ),
    "pelvis_roll": Dof(
        joint_type="hinge",
        axis=(1, 0, 0),
        stiffness=300,
        joint_range=(-0.5, 0.5),
    ),
    "pelvis_pitch": Dof(
        joint_type="hinge",
        axis=(0, 1, 0),
        stiffness=300,
        joint_range=(-0.5, 0.5),
    ),
    "pelvis_yaw": Dof(
        joint_type="hinge",
        axis=(0, 0, 1),
        stiffness=300,
        joint_range=(-0.5, 0.5),
    ),
}


class G1Entity(composer.Entity):
    """A Unitree G1 robot entity."""

    def __init__(self, model_path):
        """Initialize G1Entity.

        Args:
            model_path: Path to the G1 model XML file
        """
        self._model_path = model_path
        self._attached = []
        
        # Fix the XML file by removing keyframes and fixing mesh paths
        import tempfile
        import xml.etree.ElementTree as ET
        import os
        
        try:
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
                        compiler.attrib['meshdir'] = assets_dir
                        modifications_made = True
            
            # Find and remove all keyframe elements
            keyframes = root.findall(".//keyframe")
            if keyframes:
                for keyframe in keyframes:
                    parent_map = {c: p for p in tree.iter() for c in p}
                    if keyframe in parent_map:
                        parent_map[keyframe].remove(keyframe)
                        modifications_made = True
            
            # Find and remove all freejoint elements
            freejoints = root.findall(".//freejoint")
            if freejoints:
                for freejoint in freejoints:
                    parent_map = {c: p for p in tree.iter() for c in p}
                    if freejoint in parent_map:
                        parent_map[freejoint].remove(freejoint)
                        modifications_made = True
            
            if modifications_made:
                with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    tree.write(tmp_path)
                    self._model_path = tmp_path
            else:
                print("No modifications needed to the G1 model file")
                
            # Load the MJCF model here
            self._mjcf_root = mjcf.from_path(self._model_path)
            
        except Exception as e:
            print(f"Error preprocessing G1 model XML: {e}")
            raise

        # Now call the parent constructor
        super().__init__()

    def _build(self, name: Optional[str] = None) -> None:
        """Initializes a G1Entity.

        Args:
            name: Name of the robot. Used as a prefix in the MJCF name attributes.
        """
        self._name = name or "g1"
        self._joints = []
        self._actuators = []
        self._sensors = []
        
        # Parse and store important elements
        self._parse_mjcf_elements()

    def _parse_mjcf_elements(self) -> None:
        """Parse and store references to important MJCF elements."""
        # Find the pelvis (root) body
        self._pelvis = self._mjcf_root.find('body', 'pelvis')
        if self._pelvis is None:
            raise ValueError("Could not find pelvis body in G1 model")

        # Get all joints and actuators
        self._joints = tuple(self._mjcf_root.find_all('joint'))
        self._actuators = tuple(self._mjcf_root.find_all('actuator'))

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        """Returns the MJCF model of the G1."""
        return self._mjcf_root

    @property
    def name(self) -> str:
        """Returns the name of the G1."""
        return self._name

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        """Returns all joints in the G1."""
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        """Returns all actuators in the G1."""
        return self._actuators

    @property
    def sensors(self) -> Sequence[types.MjcfElement]:
        """Returns all sensors in the G1."""
        return self._sensors

    @property
    def pelvis(self) -> types.MjcfElement:
        """Returns the pelvis body."""
        return self._pelvis

    def initialize_episode(self, physics: mjcf.Physics, random_state: np.random.RandomState) -> None:
        """Initialize the G1's state at the start of each episode."""
        pass  # No special initialization needed for now

    def apply_action(self, physics: mjcf.Physics, action: np.ndarray, random_state: np.random.RandomState) -> None:
        """Apply actions to the G1's actuators."""
        del random_state  # Unused.
        physics.bind(self.actuators).ctrl = action

    def get_initial_position(self) -> np.ndarray:
        """Returns the initial position for the G1."""
        return np.array([0.0, 0.4, 0.7])  # Default position behind piano

    def get_initial_orientation(self) -> np.ndarray:
        """Returns the initial orientation quaternion for the G1."""
        return np.array([0, 0, 0, 1])  # Identity quaternion

# Interactive RoboPianist with Unitree G1

This interactive visualization demonstrates the integration of the Unitree G1 humanoid robot with the RoboPianist environment featuring shadow hands. It allows you to view and interact with the piano playing environment in real-time.

## Features

- Interactive visualization of the RoboPianist environment with shadow hands
- Integration of the Unitree G1 humanoid robot from the MuJoCo Menagerie
- Automatic finger movement patterns that demonstrate piano playing
- Customizable environment parameters

## Prerequisites

- Clone this repository and install dependencies
- Clone the MuJoCo Menagerie repository to access the Unitree G1 model:
  ```
  git clone https://github.com/google-deepmind/mujoco_menagerie.git
  ```

## Usage

Run the interactive environment with:

```bash
python interactive_piano.py
```

### Command-line Options

The script supports several command-line arguments:

```bash
# Basic usage with a specific MIDI file
python interactive_piano.py --midi_file=/path/to/your/midi_file.mid

# Change the environment parameters
python interactive_piano.py --disable_fingering_reward --disable_colorization

# Specify the Unitree G1 model path
python interactive_piano.py --unitree_g1_path=/path/to/mujoco_menagerie/unitree_g1/g1.xml

# Adjust the position of the Unitree G1 robot
python interactive_piano.py --unitree_position="0.0,0.5,0.0"

# Change visualization parameters
python interactive_piano.py --width=1280 --height=720 --camera_id="front"
```

### Full list of Options

- `--seed`: Random seed (default: 42)
- `--midi_file`: Path to a MIDI file
- `--environment_name`: Name of the environment to load (default: "RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
- `--n_steps_lookahead`: Number of timesteps to look ahead (default: 10)
- `--trim_silence`: Whether to trim silence from the beginning of the MIDI file (default: False)
- `--control_timestep`: Control timestep in seconds (default: 0.05)
- `--stretch_factor`: Stretch factor for the MIDI file (default: 1.0)
- `--shift_factor`: Shift factor for the MIDI file (default: 0)
- `--wrong_press_termination`: Whether to terminate when wrong keys are pressed (default: False)
- `--disable_fingering_reward`: Whether to disable the fingering reward (default: False)
- `--disable_forearm_reward`: Whether to disable the forearm reward (default: False)
- `--disable_colorization`: Whether to disable colorization (default: False)
- `--disable_hand_collisions`: Whether to disable hand collisions (default: False)
- `--add_unitree_g1`: Whether to add the Unitree G1 (default: True)
- `--unitree_g1_path`: Path to the Unitree G1 model XML
- `--unitree_position`: Position (x, y, z) for the G1 model (default: (0.0, 0.4, 0.0))
- `--width`: Viewer width (default: 1024)
- `--height`: Viewer height (default: 768)
- `--camera_id`: Camera ID (default: "piano/back")

## Implementation Details

This script is a simplified version of the original `humanoid_vis.py` script, with training logic removed and focusing exclusively on visualization and interaction.

### Adding the Unitree G1

The Unitree G1 is added to the environment by:
1. Loading the G1 model from the MuJoCo Menagerie repository
2. Creating an attachment site in the arena
3. Attaching the G1 to the arena with a specified position and orientation

### Keyboard Policy

The current implementation uses an automatic movement pattern to simulate keyboard control. This could be extended to use actual keyboard input in a more advanced implementation.

## Troubleshooting

### Common Errors

#### TypeError with dm_control viewer

If you see an error like:
```
TypeError: ord() expected a character, but string of length 0 found
```

This is a compatibility issue between your installed MuJoCo version and the dm_control viewer. The script has been updated to use RoboPianist's native viewer instead, which should work correctly with your setup.

#### G1 Model Not Found

If the script cannot find the Unitree G1 model:
1. Make sure you've cloned the MuJoCo Menagerie repository
2. Explicitly provide the path to the G1 model:
   ```
   python interactive_piano.py --unitree_g1_path=/absolute/path/to/mujoco_menagerie/unitree_g1/g1.xml
   ```

#### Other Issues

If you encounter other issues:
1. Ensure all dependencies are installed correctly
2. Check that your MuJoCo and dm_control versions are compatible
3. If the G1 doesn't appear in the right position, adjust the `unitree_position` parameter
4. Try disabling certain features (e.g., `--add_unitree_g1=False`) to isolate problems

## Extending the Script

To extend this script, you could:
1. Implement real keyboard controls for direct manipulation of the shadow hands
2. Add actual control for the Unitree G1 robot
3. Create interfaces between the G1 and the piano/shadow hands
4. Add recording capabilities
5. Implement alternative visualization modes 
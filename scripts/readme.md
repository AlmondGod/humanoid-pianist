# Utils

`clip_midi.py` is a script that clips a MIDI file at a specified `start_time` to a specified `duration`.

`interactive_piano.py` is a script that allows you to interact with the RoboPianist environment.

`play_song.py` is a script that plays a ground truth midi file song using robopianist self-actuated piano, without a policy or shadow hands.


# Interactive RoboPianist with Unitree G1

This interactive visualization demonstrates the integration of the Unitree G1 humanoid robot with the RoboPianist environment featuring shadow hands. It allows you to view and interact with the piano playing environment in real-time.

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
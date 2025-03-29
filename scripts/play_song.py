# @title All imports required for this tutorial
from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, required=True, help="Path to the MIDI file")
    return parser.parse_args()

def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)

def main(midi_path: str):
    task = self_actuated_piano.SelfActuatedPiano(
        midi=music.load(midi_path),
        change_color_on_activation=True,
        trim_silence=True,
        control_timestep=0.01,
    )

    env = composer_utils.Environment(
        recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
    )

    env = PianoSoundVideoWrapper(
        env,
        record_every=1,
        camera_id="piano/back",
        record_dir="./played_songs/",
    )

    action_spec = env.action_spec()
    min_ctrl = action_spec.minimum
    max_ctrl = action_spec.maximum

    timestep = env.reset()
    dim = 0
    for k, v in timestep.observation.items():
        dim += np.prod(v.shape)

    class Oracle:
        def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
            if timestep.reward is not None:
                assert timestep.reward == 0
            # Only grab the next timestep's goal state.
            goal = timestep.observation["goal"][: task.piano.n_keys]
            key_idxs = np.flatnonzero(goal)
            # For goal keys that should be pressed, set the action to the maximum
            # actuator value. For goal keys that should be released, set the action to
            # the minimum actuator value.
            action = min_ctrl.copy()
            action[key_idxs] = max_ctrl[key_idxs]
            # Grab the sustain pedal action.
            action[-1] = timestep.observation["goal"][-1]
            return action
        
    policy = Oracle()

    timestep = env.reset()
    print("Playing song, hold on...")
    while not timestep.last():
        action = policy(timestep)
        timestep = env.step(action)

    play_video(env.latest_filename)


if __name__ == "__main__":
    args = parse_args()
    main(args.midi_path)
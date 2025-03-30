from pathlib import Path
from typing import Optional, Tuple, Literal
import tyro
from dataclasses import dataclass
import time
import random
import numpy as np
import pickle
import architecture.sac as sac
import rl_dataclasses.specs as specs
from pathlib import Path
from typing import Optional
from wrappers.humanoid_env import get_env

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
    entity: str = "projects"
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    environment_name: Optional[str] = None
    midi_file: Optional[Path] = None
    load_checkpoint: Optional[Path] = None
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
    unitree_g1_path: str = "assets/unitree_g1/g1_modified.xml"
    unitree_position: Tuple[float, float, float] = (0.0, 0.4, 0.7)


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


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

    
    timestep = env.reset()

    spec = specs.EnvironmentSpec.make(env)

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

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

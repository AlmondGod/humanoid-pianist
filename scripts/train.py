from pathlib import Path
from typing import Optional, Tuple, Literal
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import pickle

import architecture.sac as sac
import rl_dataclasses.specs as specs
import rl_dataclasses.replay as replay
from architecture.hybrid_grpo import HybridGRPO  # Import only
from architecture.qtopt import QTOpt, QTOptConfig  # Add QTOpt import

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers


@dataclass(frozen=True)
class Args:
    root_dir: str = "remember to use run.sh"
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
    # environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    midi_file: Optional[Path] = None
    load_checkpoint: Optional[Path] = None  # Path to checkpoint file for resuming training
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = True
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = Path("/tmp/robopianist/twinkle-twinkle-no-fingering/videos")
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = sac.SACConfig()
    algorithm: Literal["sac", "hybrid_grpo", "qtopt"] = "sac"  # Add QTOpt option
    # Minimal GRPO args needed for initialization
    num_samples: int = 8
    clip_param: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_workers: int = 4  # Maximum number of parallel evaluation threads
    mini_batch_size: int = 16  # Maximum states to process in a single mini-batch
    # QTOpt specific parameters
    qtopt_config: QTOptConfig = QTOptConfig()  # Add QTOpt config
    cem_iterations: int = 3  # Number of CEM iterations for action optimization
    cem_population_size: int = 64  # Population size for CEM
    cem_elite_fraction: float = 0.1  # Fraction of elites to keep in CEM


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_env(args: Args, record_dir: Optional[Path] = None):
    env = suite.load(
        environment_name="crossing-field-cut-10s",
        # environment_name=args.environment_name,
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
    return env


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"{args.algorithm.upper()}-{args.midi_file}-{args.seed}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"

    print(f"\nStarting training run: {run_name}")
    print(f"Saving to directory: {args.root_dir}")

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Set random seeds to: {args.seed}")

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval")

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
    elif args.algorithm == "qtopt":
        print("Using QT-Opt with Cross-Entropy Method action optimization")
        # Update QTOpt config with command line parameters if provided
        qtopt_config = args.qtopt_config
        # Override QTOpt config with args if specified
        if args.cem_iterations != QTOptConfig().cem_iterations:
            qtopt_config = qtopt_config.replace(cem_iterations=args.cem_iterations)
        if args.cem_population_size != QTOptConfig().cem_population_size:
            qtopt_config = qtopt_config.replace(cem_population_size=args.cem_population_size)
        if args.cem_elite_fraction != QTOptConfig().cem_elite_fraction:
            qtopt_config = qtopt_config.replace(cem_elite_fraction=args.cem_elite_fraction)
        # Use the same hidden dimensions from agent_config for consistency
        qtopt_config = qtopt_config.replace(hidden_dims=args.agent_config.hidden_dims)
            
        agent = QTOpt.initialize(
            spec=spec,
            config=qtopt_config,
            seed=args.seed,
            discount=args.discount,
        )
    else:  # hybrid_grpo
        print("Using Hybrid GRPO with state restoration for multi-action evaluation")
        agent = HybridGRPO(
            state_dim=spec.observation_dim,
            action_dim=spec.action_dim,
            hidden_dims=args.agent_config.hidden_dims,
            lr=args.agent_config.actor_lr,
            gamma=args.discount,
            num_samples=args.num_samples,
            clip_param=args.clip_param,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_workers=args.max_workers,
            mini_batch_size=args.mini_batch_size,
        )
        agent.set_env(env)  # Pass environment instance for state restoration
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if args.algorithm == "hybrid_grpo":
                # Load PyTorch checkpoint
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                print("Successfully loaded PyTorch checkpoint")
            elif args.algorithm == "qtopt":
                # JAX checkpoint format for QTOpt (no actor or temp)
                agent = agent.replace(
                    critic=agent.critic.replace(params=checkpoint.get('critic_params', checkpoint.get('params'))),
                    target_critic=agent.target_critic.replace(params=checkpoint.get('target_critic_params', checkpoint.get('params')))
                )
                print(f"Successfully loaded JAX checkpoint for {args.algorithm}")
            else:
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
    
    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        # Act.
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
            if i == 1:
                print("Taking random actions for warmstart...")
        else:
            if i == args.warmstart_steps:
                print("\nWarmstart complete, starting training...")
            agent, action = agent.sample_actions(timestep.observation)

        # Observe.
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        # Reset episode.
        if timestep.last():
            wandb.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # Train.
        if i >= args.warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log(prefix_dict("train", metrics), step=i)
                    # print(f"Training metrics: {metrics}")

        # Eval.
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    timestep = eval_env.step(agent.eval_actions(timestep.observation))
            log_dict = prefix_dict("eval", eval_env.get_statistics())
            music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
            wandb.log(log_dict | music_dict, step=i)
            video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
            wandb.log({"video": video, "global_step": i})
            # eval_env.latest_filename.unlink()

            # Save checkpoint
            if args.algorithm == "hybrid_grpo":
                # PyTorch checkpoint format
                checkpoint = {
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                }
            elif args.algorithm == "qtopt":
                # JAX checkpoint format for QTOpt (no actor or temp)
                checkpoint = {
                    'critic_params': agent.critic.params,
                    'target_critic_params': agent.target_critic.params
                }
            else:
                # JAX checkpoint format (SAC)
                checkpoint = {
                    'params': agent.actor.params,
                    'critic_params': agent.critic.params,
                    'target_critic_params': agent.target_critic.params,
                    'temp_params': agent.temp.params
                }
            
            checkpoint_path = experiment_dir / f"checkpoint_{i:08d}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"Saved checkpoint to {checkpoint_path}")

        if i % args.log_interval == 0:
            wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))

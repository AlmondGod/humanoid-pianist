#!/bin/bash

#Default Twinkle Twinkle Rousseau training
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/ \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar 

# SAO Crossing Field No fingering training
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/ \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar 

# Guren no Yumiya 14s no fingering
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/GurenNoYumiya14s \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --batch_size 256

# Cruel Angel's Thesis 60s no fingering (resumed 
# 42-2025-02-24-20-25-20/checkpoint_00440000, 
#42-2025-02-25-00-44-50/checkpoint_00920000.pkl, 
#42-2025-02-25-21-49-11/checkpoint_00920000, 
#42-2025-02-26-22-12-02/checkpoint_00920000
#42-2025-02-27-10-11-26/checkpoint_00920000)
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesis60s \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut 60s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --eval-interval 40000 \
#     --load_checkpoint "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesis60s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut 60s.mid-42-2025-02-27-20-19-15/checkpoint_00920000.pkl"

# Gurenge 60s no fingering
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/Gurenge60s \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Gurenge Cut 60s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --eval-interval 40000 \
#     --batch_size 128

# Hybrid GRPO 
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/ \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --eval-interval 1000 \
#     --batch_size 64 \
#     --algorithm hybrid_grpo \
#     --num-samples 8 \
#     --clip-param 0.2 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01 \

# Resume training from a checkpoint
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/ \
#     --load-checkpoint /Users/almondgod/Repositories/robopianist/robopianist-rl/models/YOUR_MODEL_DIR/checkpoint_00040000.pkl \
#     --warmstart-steps 0 \  # Set to 0 to skip random actions phase
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --eval-interval 1000 \
#     --batch_size 64 \
#     --algorithm hybrid_grpo \
#     --num-samples 8 \
#     --clip-param 0.2 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01 \


# SAC Droq Cruel Angel's Thesis middle 15, resumed from 42-2025-02-28-21-38-57/checkpoint_00880000, 42-2025-03-01-10-57-51/checkpoint_00600000
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 40000 \
    --load-checkpoint "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CruelAngelsThesismiddle15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-03-09-53-14/checkpoint_00920000.pkl"


# Hybrid GRPO Cruel Angel's Thesis middle 15 - OPTIMIZED VERSION
# WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
#     --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/HGRPO-CruelAngelsThesismiddle15s-Optimized \
#     --warmstart-steps 5000 \
#     --max-steps 5000000 \
#     --discount 0.8 \
#     --agent-config.critic-dropout-rate 0.01 \
#     --agent-config.critic-layer-norm \
#     --agent-config.hidden-dims 256 256 256 \
#     --trim-silence \
#     --gravity-compensation \
#     --reduced-action-space \
#     --control-timestep 0.05 \
#     --n-steps-lookahead 10 \
#     --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid" \
#     --action-reward-observation \
#     --primitive-fingertip-collisions \
#     --eval-episodes 1 \
#     --camera-id "piano/back" \
#     --tqdm-bar \
#     --eval-interval 10000 \
#     --algorithm hybrid_grpo \
#     --num-samples 2 \
#     --clip-param 0.2 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01

# Hybrid GRPO - Safe Version (to avoid segmentation faults)
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/HGRPO-Safe-CruelAngelsThesismiddle15s \
    --warmstart-steps 0 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 5000 \
    --algorithm hybrid_grpo \
    --num-samples 4 \
    --clip-param 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --max-workers 1 \
    --mini-batch-size 32 \
    --batch-size 32

# QTOpt Base, loaded from 42-2025-03-01-22-39-16/checkpoint_00087000
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
  --algorithm qtopt \
  --root_dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/QTOpt-v1-CruelAngelsThesismiddle15s \
  --warmstart-steps 5000 \
  --max-steps 5000000 \
  --cem_iterations 2 \
  --cem_population_size 64 \
  --cem_elite_fraction 0.1 \
  --agent_config.hidden_dims 256 256 256 \
  --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid" \
  --trim-silence \
  --gravity-compensation \
  --reduced-action-space \
  --control-timestep 0.05 \
  --n-steps-lookahead 10 \
  --action-reward-observation \
  --primitive-fingertip-collisions \
  --tqdm-bar \
  --eval-interval 10000 \
  --load-checkpoint "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/QTOpt-v1-CruelAngelsThesismiddle15s/QTOPT-/Users/almondgod/Repositories/robopianist/midi_files_cut/Cruel Angel's Thesis Cut middle 15s.mid-42-2025-03-01-22-39-16/checkpoint_00087000.pkl"


# SAC Droq Crossing Fields Complex 15s
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/CrossingFieldsComplex15s \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Fields Complex Cut 15s.mid" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 40000 \
    # --load-checkpoint "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/CrossingFieldsComplex15s/SAC-/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Fields Complex Cut 15s.mid-42-2025-03-06-00-48-52/checkpoint_00920000.pkl"


# GUren no Yumiya 14s SAC Droq
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/GurennoYumiya14s \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 30000 

# QTOpt Cross Field Simple 10s
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
  --algorithm qtopt \
  --root_dir /Users/almondgod/Repositories/robopianist/robopianist-rl/models/QTOpt-base-CrossingFieldSimple10s \
  --warmstart-steps 5000 \
  --max-steps 5000000 \
  --cem_iterations 2 \
  --cem_population_size 64 \
  --cem_elite_fraction 0.1 \
  --agent_config.hidden_dims 256 256 256 \
  --midi-file "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid" \
  --trim-silence \
  --gravity-compensation \
  --reduced-action-space \
  --control-timestep 0.05 \
  --n-steps-lookahead 10 \
  --action-reward-observation \
  --primitive-fingertip-collisions \
  --tqdm-bar \
  --eval-interval 5000 \
  --load-checkpoint "/Users/almondgod/Repositories/robopianist/robopianist-rl/models/QTOpt-base-CrossingFieldSimple10s/QTOPT-/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid-42-2025-03-05-01-17-41/checkpoint_00090000.pkl"
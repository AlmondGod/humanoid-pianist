#!/bin/bash

#Default Twinkle Twinkle Rousseau training
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/train.py \
    --root-dir models/CrossingField10s \
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
    --midi-file "example_midis/Crossing Field 10s.mid" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 30000
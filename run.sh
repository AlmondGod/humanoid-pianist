#!/bin/bash

WANDB_DIR=/root/robopianist-rl/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir /root/robopianist-rl/models/\
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
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar

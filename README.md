![Humanoid Robot Pianist](assets/Humanoid%20Robot%20Pianist.png)

## Installation

First, install [robopianist](https://github.com/google-research/robopianist):

```bash
bash <(curl -s https://raw.githubusercontent.com/google-research/robopianist/main/scripts/install_deps.sh) --no-soundfonts

conda create -n pianist python=3.10
conda activate pianist

pip install --upgrade robopianist
```

From the same conda environment:

1. Install [JAX](https://github.com/google/jax#installation)
2. Run `pip install -r requirements.txt`

## Usage

We provide an example bash script to train an SAC policy to play Twinkle Twinkle Little Star with the task parameters used in the robopianist paper:

```bash
WANDB_DIR=/tmp/robopianist/ MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py \
    --root-dir models/ \ #add song name as a subdirectory
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
    --midi-file "TwinkleTwinkleLittleStar" \ # replace with the path to your midi file
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --eval-interval 30000
```

To evaluate a trained policy, run `python eval.py`

```bash
python eval.py \
--load_checkpoint <YOUR_MODEL_PATH> \
--midi-file <YOUR_MIDI_FILE_PATH>
```

# Troubleshooting and Debugging

The current JAX is configured to run on Apple Silicon. Please adjust the JAX METAL lines according to your system.

I encourage you not to use custom fingering, which can introduce errors and is not necessary when SAC is combined with RLHF finetuning.

## Citing robopianist

If you use this code, please cite our paper:

```bibtex
@article{zakka2023robopianist,
  author = {Zakka, Kevin and Smith, Laura and Gileadi, Nimrod and Howell, Taylor and Peng, Xue Bin and Singh, Sumeet and Tassa, Yuval and Florence, Pete and Zeng, Andy and Abbeel, Pieter},
  title = {{RoboPianist: A Benchmark for High-Dimensional Robot Control}},
  journal = {arXiv preprint arXiv:2304.04150},
  year = {2023},
}
```

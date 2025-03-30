# Humanoid Robot Pianist

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[tests-badge]: https://github.com/google-research/robopianist/actions/workflows/ci.yml/badge.svg
[docs-badge]: https://github.com/google-research/robopianist/actions/workflows/docs.yml/badge.svg
[tests]: https://github.com/google-research/robopianist/actions/workflows/ci.yml
[docs]: https://google-research.github.io/robopianist/
[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/robopianist
[pypi-badge]: https://badge.fury.io/py/robopianist.svg
[pypi]: https://pypi.org/project/robopianist/

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
```

To evaluate a trained policy, run `python eval.py`

```bash
python scripts/eval.py \
--load_checkpoint <YOUR_MODEL_PATH> \
--midi-file <YOUR_MIDI_FILE_PATH>
```

## Troubleshooting and Tuning Training

The current JAX is configured to run on Apple Silicon. Please adjust the JAX METAL lines according to your system.

I encourage you not to use custom fingering, which can introduce errors and is not necessary when SAC is combined with RLHF finetuning.



### Robopianist and CPL Citations

```bibtex
@article{zakka2023robopianist,
  author = {Zakka, Kevin and Smith, Laura and Gileadi, Nimrod and Howell, Taylor and Peng, Xue Bin and Singh, Sumeet and Tassa, Yuval and Florence, Pete and Zeng, Andy and Abbeel, Pieter},
  title = {{RoboPianist: A Benchmark for High-Dimensional Robot Control}},
  journal = {arXiv preprint arXiv:2304.04150},
  year = {2023},
}
```

```bibtex
@InProceedings{hejna23contrastive,
  title = {Contrastive Preference Learning: Learning From Human Feedback without RL},
  author = {Hejna, Joey and Rafailov, Rafael and Sikchi, Harshit and Finn, Chelsea and Niekum, Scott and Knox, W. Bradley and Sadigh, Dorsa},
  booktitle = {ArXiv preprint},
  year = {2023},
  url = {https://arxiv.org/abs/2310.13639}
}
```
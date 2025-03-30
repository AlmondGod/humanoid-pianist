# Humanoid Robot Pianist

![PyPI Python Version][pypi-versions-badge]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/robopianist

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

To train an SAC policy to play Crossing Field's first 10s with the task parameters used in the robopianist paper:

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

# Contrastive Preference Learning for Robotics
[Original Paper](papers/Contrastive%20Preference%20Learning.pdf)

## Goal
Use human feedback of recorded episodes to train a policy to act aligned with human preferences (in our case, play the piano better)

## Background
Contrastive Preference Learning (CPL) is a method for learning from human preferences without requiring explicit reward engineering. Generically, we:
1. Collect pairs of model outputs (segments)
2. Get human preferences between these pairs
3. Train a policy to maximize the likelihood of preferred behaviors while staying close to the original policy

This approach is perfect for piano playing, in which humans have subtle technique and feel preferences that are difficult to capture in reward functions. However, we want to preserve existing 'good behaviors' while making human-guided improvements.

I worked from an example implementation in the original CPL repository: https://github.com/jhejna/cpl/blob/main/research/algs/cpl.py

## Architecture
For the CPL implementation in `cpl_train_sac.py`, we have:

**Base Policy** (`scripts/train.py`, `architecture/sac.py`)
1. A Soft Actor-Critic policy to act as "pretraining" on the piano playing task
2. When finetuning with CPL, we abandon the Critic and only use the Policy. In the future, could try using the Critic as auxiliary loss to augment BC keeping the policy close to the original.

**Preference Collection** (`rlhf/generate_preference_dataset.py`)
1. Generate segments using different checkpoints and noise levels to increase the diversity of the dataset
2. Record videos of the policy playing for human evaluation, and rate the quality of each performance from 1-100
3. Create pairwise preferences based on the ratings to yield $n^2$-size preference dataset
4. The data is organized by timestamp and includes full trajectory (state + action) information

**CPL Training**: (`rlhf/cpl_train.py`, `architecture/cpl_sac.py`)
1. Uses CPL-SAC architecture to wrap SAC with CPL loss update
2. CPL loss computes log probs of preferred and non-preferred actions, computes equivalent advantages (alpha * log probs), and compute cpl loss according to:

$$
\mathcal{L}_{\text{CPL}}(\pi_\theta, \mathcal{D}_{\text{pref}}) = \mathbb{E}_{(\sigma^+,\sigma^-) \sim \mathcal{D}_{\text{pref}}}\left[-\log \frac{\exp \sum_{\sigma^+} \gamma^t \alpha \log \pi_\theta(a_t^+|s_t^+)}{\exp \sum_{\sigma^+} \gamma^t \alpha \log \pi_\theta(a_t^+|s_t^+)+\exp \sum_{\sigma^-} \gamma^t \alpha \log \pi_\theta(a_t^-|s_t^-)} \right]
$$

Essentially, we maximize the expected ratio of the advantage of preferred actions over both preferred and non-preferred actions.

3. Then we computes loss as weighted combination of CPL loss and conservative loss (MSE to original policy outputs)
4. And we finally clip gradient norms to prevent exploding gradients

## Benefits
1. No need to learn a separate reward function, only need pretrained policy
2. CPL Directly incorporates human preferences about playing style
3. Conservative Loss prevents forgetting of good behaviors learned in pretraining
4. Could learn from a relatively small number of human preferences

## Getting Training to work

1. **Hyperparameters that worked for me**:
   - Learning rate: 1e-4
   - Batch size: 32
   - Temperature (alpha): 0.1
   - Conservative weight: 0.01
   - Preference weight (lambda): 0.5

2. **Monitoring**:
   - Track preference loss
   - Monitor conservative regularization and **ensure no NaN gradients**
   - Track the videos in the `eval` folder to ensure consistent performance


# Soft Actor-Critic (with DroQ)

[Original Paper](papers/Soft%20Actor%20Critic.pdf)

## Goal

Train a policy in high-dimensional continuous actio spaces that does nto get stuck at local minima and is not as sensitive to hyperparameters as past RL algorithms

## Background

Soft Actor-Critic (SAC) is off-policy, meaning it learns from data not necessarily generated by the current policy. In practice, we often use SAC online (interacting with an environement) and store data in a replay buffer. SAC combines techniques from:
1. DDPG: offline actor-critic architecture (but has a deterministic actor)
2. Soft Q-Learning: maximizes entropy to encourage exploration

## Architecture

We use typical actor-critic architecture, but now we add to our critic:
1. Using the minimum of 2 Q functions to reduce Q overestimation bias
2. use target networks for more stable TD learning
2. Include an entropy term in the values which discourages deterministic actions and thus encourages exploration of the state space.

The `robopianist-rl` implementation (in `architecture/sac.py`) uses:

1. **Actor Network**: MLP with 3x256 hidden layers, predicts normal dist for each action dimension, `TanhMultivariateNormalDiag` distribution for bounded actions

2. **Critic Network**: Two Q-networks with 3x256 hidden layers, Layer normalization and dropout (0.01) for regularization (DroQ)

## Benefits

1. Off-policy learning allows reuse of past experience (sample efficient)
2. Double Q-learning reduces bellman-induced overestimation bias, DroQ improves generalization
3. Maximum entropy so gets stuck at local minima less

## Getting Training to work

1. **Hyperparameters that worked for me with no fingering annotations**:
   - Learning rates: ~3e-4 for all networks
   - Batch size: 256
   - Target network update rate (tau): 0.01
   - Initial temperature: 1.2

4. **Monitoring**:
   - Track Q-values to detect overestimation
   - Ensure consistent relatively high entropy and gradually increasing Qs after beginning stabilization in training


## Troubleshooting

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
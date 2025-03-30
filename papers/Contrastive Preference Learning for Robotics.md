# Contrastive Preference Learning for Robotics

## Goal
Use human feedback of recorded episodes to train a policy to act aligned with human preferences (in our case, play the piano better)

## Background
Contrastive Preference Learning (CPL) is a method for learning from human preferences without requiring explicit reward engineering. Generically, we:
1. Collect pairs of model outputs (segments)
2. Get human preferences between these pairs
3. Train a policy to maximize the likelihood of preferred behaviors while staying close to the original policy

This approach is perfect for piano playing where Humans have subtle technique and feel preferences that are difficult to capture in reward functions. However, we want to preserve existing good behaviors while making targeted improvements.

I worked from an example implementation in the original CPL repository: https://github.com/jhejna/cpl/blob/main/research/algs/cpl.py

## Architecture
For the CPL implementation in `cpl_train_sac.py`, we have:

**Base Policy**: 
1. A Soft Actor-Critic policy to act as "pretraining" on the piano playing task. 
2. When finetuning, we abandon the Critic and only use the Policy. In the future, could try using the Critic as auxiliary loss to augment BC keeping the policy close to the original.

**Preference Collection**:
1. Generate segments using different checkpoints and noise levels to increase the diversity of the dataset
2. Record videos for human evaluation, and rate the quality of each performance
3. Create pairwise preferences based on the ratings to yield $n^2$-size preference dataset
4. The data is organized by timestamp and includes full trajectory (state + action) information

**CPL Training**:
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
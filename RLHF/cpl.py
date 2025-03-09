# TODO: make cpl class to take in preference data from geenrate_preference_data
# training loop will be in train_cpl.py

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence, Any, Callable
from dataclasses import dataclass
import pickle
import random
from pathlib import Path

@dataclass
class CPLConfig:
    """Configuration for Contrastive Preference Learning."""
    # Network architecture
    hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    dropout_rate: float = 0.0
    
    # Optimization
    learning_rate: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # CPL specific
    conservative_weight: float = 0.0  # Weight for conservative regularization (Section A.4 in paper)
    normalize_rewards: bool = True    # Whether to normalize predicted rewards


class RewardModel(nn.Module):
    """Reward model for Contrastive Preference Learning."""
    hidden_dims: Sequence[int]
    dropout_rate: float = 0.0
    activation: nn = nn.gelu
    
    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        """
        Predict rewards given states and actions.
        
        Args:
            states: State representations [batch_size, state_dim]
            actions: Action vectors [batch_size, action_dim]
            training: Whether in training mode (for dropout)
            
        Returns:
            rewards: Predicted reward values [batch_size]
        """
        x = jnp.concatenate([states, actions], axis=-1)
        
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'hidden_{i}')(x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate, deterministic=not training
                )(x)
        
        rewards = nn.Dense(1, name='reward')(x)
        return rewards.squeeze(-1)


class CPL:
    """Implementation of Contrastive Preference Learning (CPL)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: CPLConfig,
        seed: int = 42,
    ):
        """
        Initialize the CPL model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration for CPL
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        self.rng = jax.random.PRNGKey(seed)
        
        # Initialize model and optimizer
        self._init_model()
    
    def _init_model(self):
        """Initialize the reward model."""
        # Create model
        self.model = RewardModel(
            hidden_dims=self.config.hidden_dims,
            dropout_rate=self.config.dropout_rate,
            activation=getattr(nn, self.config.activation),
        )
        
        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_state = jnp.zeros((1, self.state_dim))
        dummy_action = jnp.zeros((1, self.action_dim))
        params = self.model.init(init_rng, dummy_state, dummy_action)["params"]
        
        # Create optimizer with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        )
        
        # Create train state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        with open(path, 'wb') as f:
            pickle.dump({"params": self.state.params}, f)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.state = self.state.replace(params=checkpoint["params"])
    
    @staticmethod
    def cpl_loss_fn(reward_fn, params, chosen_states, chosen_actions, 
                   rejected_states, rejected_actions, training=True,
                   conservative_weight=0.0, dropout_rng=None):
        """
        Compute the Contrastive Preference Learning loss.
        
        This implements the CPL loss as described in the paper:
        r_chosen - r_rejected -> maximize
        -log(sigmoid(r_chosen - r_rejected)) -> minimize
        
        Args:
            reward_fn: Function to compute rewards
            params: Model parameters
            chosen_states: States from preferred trajectories
            chosen_actions: Actions from preferred trajectories
            rejected_states: States from rejected trajectories
            rejected_actions: Actions from rejected trajectories
            training: Whether in training mode
            conservative_weight: Weight for conservative regularization
            dropout_rng: RNG key for dropout (required when training=True)
            
        Returns:
            loss: Mean loss value
            metrics: Dictionary of evaluation metrics
        """
        # Set up rngs dict for dropout if in training mode
        rngs = {"dropout": dropout_rng} if training and dropout_rng is not None else None
        
        # Compute rewards for chosen and rejected
        chosen_rewards = reward_fn({"params": params}, chosen_states, chosen_actions, training, rngs=rngs)
        rejected_rewards = reward_fn({"params": params}, rejected_states, rejected_actions, training, rngs=rngs)
        
        # Compute preference logits (r_chosen - r_rejected)
        logits = chosen_rewards - rejected_rewards
        
        # Compute loss: -log(sigmoid(r_chosen - r_rejected))
        # This is the same as binary cross entropy with logits=r_chosen-r_rejected, labels=1
        losses = optax.sigmoid_binary_cross_entropy(logits, jnp.ones_like(logits))
        
        # Compute accuracy (how often r_chosen > r_rejected)
        accuracy = jnp.mean(logits > 0)
        
        # Mean loss
        loss = jnp.mean(losses)
        
        # Add conservative regularization if specified
        # This penalizes large reward values to prevent exploitation
        if conservative_weight > 0:
            # Penalize squared reward values
            conservative_loss = jnp.mean(chosen_rewards**2) + jnp.mean(rejected_rewards**2)
            loss = loss + conservative_weight * conservative_loss
        
        return loss, {
            "loss": loss,
            "accuracy": accuracy,
            "mean_chosen_reward": jnp.mean(chosen_rewards),
            "mean_rejected_reward": jnp.mean(rejected_rewards),
            "mean_reward_diff": jnp.mean(logits),
        }
    
    def update_step(self, chosen_states, chosen_actions, rejected_states, rejected_actions):
        """
        Perform a single update step.
        
        Args:
            chosen_states: States from preferred trajectories
            chosen_actions: Actions from preferred trajectories
            rejected_states: States from rejected trajectories
            rejected_actions: Actions from rejected trajectories
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Generate a new RNG key for this update
        self.rng, dropout_rng = jax.random.split(self.rng)
        
        # Create loss function for this update
        def loss_fn(params):
            return self.cpl_loss_fn(
                self.model.apply, 
                params, 
                chosen_states, 
                chosen_actions, 
                rejected_states, 
                rejected_actions,
                training=True,
                conservative_weight=self.config.conservative_weight,
                dropout_rng=dropout_rng  # Pass the RNG key for dropout
            )
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(self.state.params)
        
        # Update parameters
        self.state = self.state.apply_gradients(grads=grads)
        
        return metrics
    
    def predict_rewards(self, states, actions):
        """
        Predict rewards for states and actions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            rewards: Predicted rewards
        """
        rewards = self.model.apply({"params": self.state.params}, states, actions, training=False)
        return np.asarray(rewards)
    
    def preference_probability(self, states_a, actions_a, states_b, actions_b):
        """
        Compute probability that trajectory A is preferred over trajectory B.
        
        Args:
            states_a: States from trajectory A
            actions_a: Actions from trajectory A
            states_b: States from trajectory B
            actions_b: Actions from trajectory B
            
        Returns:
            prob: Probability that A is preferred over B
        """
        rewards_a = self.predict_rewards(states_a, actions_a)
        rewards_b = self.predict_rewards(states_b, actions_b)
        
        # Compute logits and apply sigmoid to get probability
        logits = np.mean(rewards_a) - np.mean(rewards_b)
        prob = 1.0 / (1.0 + np.exp(-logits))
        
        return prob



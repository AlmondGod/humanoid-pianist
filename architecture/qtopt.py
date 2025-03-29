from dataclasses import dataclass

from functools import partial
from typing import Any, Dict, Optional, Sequence

import flax.linen as nn
import jax
jax.config.update('jax_platform_name', 'METAL')
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from architecture.distributions import TanhNormal
from architecture.networks import MLP, Ensemble, StateActionValue, subsample_ensemble
from utils.rl_dataclasses.specs import EnvironmentSpec, zeros_like
from utils.rl_dataclasses.replay import Transition

LogDict = Dict[str, float]


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray, actions: np.ndarray) -> jnp.ndarray:
    """Evaluate Q-values for the given observations and actions."""
    qs = apply_fn({"params": params}, observations, actions, False)
    return qs.mean(axis=0)  # Average across ensemble


def _cem_optimize(key, critic_fn, critic_params, state, action_dim, 
                  n_iterations=3, population_size=64, elite_fraction=0.1, 
                  init_mean=None, init_var=None):
    """Cross-Entropy Method for action optimization.
    
    Args:
        key: JAX random key
        critic_fn: Q-function apply function
        critic_params: Q-function parameters
        state: Current state observation
        action_dim: Dimension of action space
        n_iterations: Number of CEM iterations
        population_size: Size of population for CEM
        elite_fraction: Fraction of elites to keep
        init_mean: Initial mean for the CEM distribution (optional)
        init_var: Initial variance for the CEM distribution (optional)
        
    Returns:
        Optimized action
    """
    # Expand the state to match the population size
    state_batch = jnp.tile(state, (population_size, 1))
    
    # Initialize mean and variance for the action distribution
    if init_mean is None:
        mean = jnp.zeros(action_dim)
    else:
        mean = init_mean
        
    if init_var is None:
        var = jnp.ones(action_dim)
    else:
        var = init_var
    
    # Number of elites to keep
    n_elites = max(1, int(population_size * elite_fraction))
    
    # CEM iterations
    for i in range(n_iterations):
        # Sample actions from the distribution
        key, subkey = jax.random.split(key)
        actions = mean + jnp.sqrt(var) * jax.random.normal(subkey, (population_size, action_dim))
        
        # Clip actions to [-1, 1] (assuming tanh bounds)
        actions = jnp.clip(actions, -1.0, 1.0)
        
        # Evaluate actions
        q_values = critic_fn({"params": critic_params}, state_batch, actions, False)
        q_values = q_values.mean(axis=0)  # Average across ensemble
        
        # Select elite actions
        elite_idxs = jnp.argsort(q_values)[-n_elites:]
        elite_actions = actions[elite_idxs]
        
        # Update distribution parameters
        mean = jnp.mean(elite_actions, axis=0)
        var = jnp.var(elite_actions, axis=0) + 1e-5  # Add small constant for stability
    
    # Return the mean of the final distribution as the optimal action
    return mean


@struct.dataclass
class QTOptConfig:
    """Configuration options for QTOpt."""

    num_qs: int = 2
    critic_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    num_min_qs: Optional[int] = None
    critic_dropout_rate: float = 0.0
    critic_layer_norm: bool = False
    tau: float = 0.005
    # CEM parameters
    cem_iterations: int = 3
    cem_population_size: int = 64
    cem_elite_fraction: float = 0.1


class QTOpt(struct.PyTreeNode):
    """QT-Opt implementation using Cross-Entropy Method for action selection."""

    critic: TrainState
    target_critic: TrainState
    rng: Any
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    # CEM parameters
    cem_iterations: int = struct.field(pytree_node=False)
    cem_population_size: int = struct.field(pytree_node=False)
    cem_elite_fraction: float = struct.field(pytree_node=False)

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: QTOptConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "QTOpt":
        """Initializes the agent from the given environment spec and config."""

        observations = zeros_like(spec.observation)
        action_dim = spec.action.shape[-1]
        actions = zeros_like(spec.action)

        rng = jax.random.PRNGKey(seed)
        rng, critic_key = jax.random.split(rng)

        # Create Q-networks (critics)
        critic_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
            dropout_rate=config.critic_dropout_rate,
            use_layer_norm=config.critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config.num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=config.num_min_qs or config.num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        return QTOpt(
            critic=critic,
            target_critic=target_critic,
            rng=rng,
            tau=config.tau,
            discount=discount,
            num_qs=config.num_qs,
            num_min_qs=config.num_min_qs,
            action_dim=action_dim,
            cem_iterations=config.cem_iterations,
            cem_population_size=config.cem_population_size,
            cem_elite_fraction=config.cem_elite_fraction,
        )

    def update_critic(self, transitions: Transition) -> tuple["QTOpt", LogDict]:
        """Update Q-function using clipped double Q-learning."""
        rng = self.rng

        # Select next actions using CEM optimization
        # Note: For efficiency in training, we use a simpler approach for selecting target actions
        # In the full QT-Opt paper, CEM is also used here, but we use random sampling for simplicity
        key, rng = jax.random.split(rng)
        next_state_batch = transitions.next_state
        
        # We'll generate multiple random actions and pick the best one (simplified CEM)
        num_action_samples = 10
        action_shape = (num_action_samples,) + transitions.next_state.shape[:-1] + (self.action_dim,)
        rand_actions = jax.random.uniform(
            key, shape=action_shape, minval=-1.0, maxval=1.0
        )
        
        # Get Q-values for all actions
        key, rng = jax.random.split(rng)
        next_q_values = []
        
        for i in range(num_action_samples):
            # Extract the i-th set of actions
            next_actions_i = rand_actions[i]
            
            # Get Q-values from target network
            next_qs_i = self.target_critic.apply_fn(
                {"params": self.target_critic.params},
                next_state_batch,
                next_actions_i,
                False,
            )
            next_q_values.append(next_qs_i)
        
        # Stack and find best action per state
        stacked_q_values = jnp.stack(next_q_values, axis=1)  # [num_qs, num_samples, batch_size]
        mean_q_values = jnp.mean(stacked_q_values, axis=0)   # [num_samples, batch_size]
        best_action_idx = jnp.argmax(mean_q_values, axis=0)  # [batch_size]
        
        # Get the best Q-values using these actions
        batch_size = transitions.next_state.shape[0]
        batch_indices = jnp.arange(batch_size)
        best_actions = rand_actions[best_action_idx, batch_indices]
        
        # Compute target Q-values with clipped double Q-learning
        next_q1, next_q2 = self.target_critic.apply_fn(
            {"params": self.target_critic.params}, 
            transitions.next_state, 
            best_actions, 
            False
        )  # Assuming we get two Q-values from the ensemble
        
        next_q = jnp.minimum(next_q1, next_q2)  # Clipped double Q-learning
        target_q = transitions.reward + self.discount * transitions.discount * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            # Get current Q-values
            qs = self.critic.apply_fn(
                {"params": critic_params},
                transitions.state,
                transitions.action,
                True,
                rngs={"dropout": key},
            )
            # Compute MSE loss for each Q-network
            critic_loss = ((qs - target_q.reshape(1, -1)) ** 2).mean(axis=1).sum()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        # Polyak averaging for target network update
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update(self, transitions: Transition) -> tuple["QTOpt", LogDict]:
        """Update the agent's parameters."""
        # In QT-Opt, we only update the critic (no actor or temperature updates)
        new_agent, critic_info = self.update_critic(transitions)
        return new_agent, critic_info

    def sample_actions(self, observations: np.ndarray) -> tuple["QTOpt", np.ndarray]:
        """Sample actions for exploration during training."""
        # During training, we use random actions with probability epsilon
        # or CEM-optimized actions otherwise
        key, new_rng = jax.random.split(self.rng)
        
        # Use a simple epsilon-greedy approach for exploration
        epsilon = 0.1
        random_action = jax.random.uniform(
            key, shape=(self.action_dim,), minval=-1.0, maxval=1.0
        )
        
        # With probability (1-epsilon), use CEM to optimize action
        key, subkey = jax.random.split(key)
        if jax.random.uniform(subkey) > epsilon:
            # Use CEM to find best action
            key, subkey = jax.random.split(key)
            opt_action = _cem_optimize(
                subkey, 
                self.critic.apply_fn, 
                self.critic.params,
                observations, 
                self.action_dim,
                n_iterations=self.cem_iterations,
                population_size=self.cem_population_size,
                elite_fraction=self.cem_elite_fraction
            )
            action = opt_action
        else:
            action = random_action
            
        return self.replace(rng=new_rng), np.asarray(action)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        """Select actions for evaluation (no exploration)."""
        # Use CEM to find best action for the given observation
        key, _ = jax.random.split(self.rng)
        opt_action = _cem_optimize(
            key, 
            self.critic.apply_fn, 
            self.critic.params,
            observations, 
            self.action_dim,
            n_iterations=self.cem_iterations,
            population_size=self.cem_population_size,
            elite_fraction=self.cem_elite_fraction
        )
        return np.asarray(opt_action)

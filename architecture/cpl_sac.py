import jax
import jax.numpy as jnp
from typing import Optional, Any
import pickle
from architecture.sac import SAC, SACConfig
from rl_dataclasses.specs import EnvironmentSpec
from flax import struct

@struct.dataclass
class CPL_SAC(SAC):
    """SAC variant that uses CPL loss for training."""
    initial_params: Optional[Any] = None  # Add initial_params as a field
    
    @classmethod
    def initialize(
        cls,
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "CPL_SAC":
        """Initialize a CPL_SAC instance."""
        # Call parent's initialize to get base SAC instance
        sac = super().initialize(spec, config, seed, discount)
        # Convert to CPL_SAC instance
        return cls(
            actor=sac.actor,
            rng=sac.rng,
            critic=sac.critic,
            target_critic=sac.target_critic,
            temp=sac.temp,
            tau=sac.tau,
            discount=sac.discount,
            target_entropy=sac.target_entropy,
            num_qs=sac.num_qs,
            num_min_qs=sac.num_min_qs,
            backup_entropy=sac.backup_entropy,
            initial_params=None,  # Initialize with None
        )
    
    @classmethod
    def from_sac_checkpoint(cls, checkpoint_path: str, spec: EnvironmentSpec, config: SACConfig, seed: int = 0):
        """Initialize CPL_SAC from a pretrained SAC checkpoint."""
        # Initialize with same architecture
        agent = cls.initialize(spec, config, seed)
        
        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Store initial params for conservative loss
            initial_params = checkpoint['params'].copy()
            
            # Load all parameters from checkpoint and set initial_params
            agent = agent.replace(
                actor=agent.actor.replace(params=checkpoint['params']),
                critic=agent.critic.replace(params=checkpoint['critic_params']),
                target_critic=agent.target_critic.replace(params=checkpoint['target_critic_params']),
                temp=agent.temp.replace(params=checkpoint['temp_params']),
                initial_params=initial_params,  # Set initial_params using replace
            )
            
            print("Successfully loaded checkpoint")
            return agent
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    def update_cpl(self, preferred, non_preferred, config):
        """Update using CPL loss."""
        def cpl_loss_fn(actor_params):
            # Get shapes from the input data
            batch_size, segment_length = preferred["states"].shape[:2]
            
            # Step 1: Compute log probabilities for both segments
            # Reshape inputs for vmap
            preferred_states = preferred["states"].reshape(-1, preferred["states"].shape[-1])
            preferred_actions = preferred["actions"].reshape(-1, preferred["actions"].shape[-1])
            non_preferred_states = non_preferred["states"].reshape(-1, non_preferred["states"].shape[-1])
            non_preferred_actions = non_preferred["actions"].reshape(-1, non_preferred["actions"].shape[-1])
            
            preferred_logprobs = jax.vmap(
                lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
            )(preferred_states, preferred_actions)
            
            non_preferred_logprobs = jax.vmap(
                lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
            )(non_preferred_states, non_preferred_actions)
            
            # Reshape logprobs back to (batch_size, segment_length)
            preferred_logprobs = preferred_logprobs.reshape(batch_size, segment_length)
            non_preferred_logprobs = non_preferred_logprobs.reshape(batch_size, segment_length)
            
            # Clip log probabilities for numerical stability
            preferred_logprobs = jnp.clip(preferred_logprobs, -10.0, 10.0)
            non_preferred_logprobs = jnp.clip(non_preferred_logprobs, -10.0, 10.0)
            
            # Step 2: Compute advantages
            # Apply discounting with numerical stability (per segment)
            timesteps = jnp.arange(segment_length)
            discount = jnp.clip(config.gamma ** timesteps, 1e-6, 1.0)
            
            # Reshape discount to match the batch dimension
            discount = discount.reshape(1, -1)  # Shape: (1, segment_length)
            preferred_advantages = config.alpha * (discount * preferred_logprobs)  # Shape: (batch_size, segment_length)
            non_preferred_advantages = config.alpha * (discount * non_preferred_logprobs)
            
            # Normalize advantages per segment
            preferred_advantages = (preferred_advantages - jnp.mean(preferred_advantages, axis=1, keepdims=True)) / (jnp.std(preferred_advantages, axis=1, keepdims=True) + 1e-8)
            non_preferred_advantages = (non_preferred_advantages - jnp.mean(non_preferred_advantages, axis=1, keepdims=True)) / (jnp.std(non_preferred_advantages, axis=1, keepdims=True) + 1e-8)
            
            # Sum advantages over time dimension (per segment)
            preferred_segment_adv = jnp.sum(preferred_advantages, axis=1)
            non_preferred_segment_adv = jnp.sum(non_preferred_advantages, axis=1)
            
            # Step 3: Compute the CPL Loss (per segment pair)
            # Compute log(exp(preferred) / (exp(preferred) + exp(non_preferred)))
            # = preferred - log(exp(preferred) + exp(non_preferred))
            # Using log-sum-exp trick to prevent overflow:
            # = preferred - (max_val + log(exp(preferred - max_val) + exp(non_preferred - max_val)))
            max_adv = jnp.maximum(preferred_segment_adv, non_preferred_segment_adv)
            exp_terms = jnp.exp(preferred_segment_adv - max_adv) + jnp.exp(non_preferred_segment_adv - max_adv)
            stable_preference_loss = -jnp.mean(preferred_segment_adv - max_adv - jnp.log(exp_terms))
            
            # Add conservative regularization with clipping
            def compute_conservative_loss(states, actions):
                current_logprobs = jax.vmap(
                    lambda s, a: self.actor.apply_fn({"params": actor_params}, s).log_prob(a)
                )(states, actions)
                
                initial_logprobs = jax.vmap(
                    lambda s, a: self.actor.apply_fn({"params": self.initial_params}, s).log_prob(a)
                )(states, actions)
                
                # Reshape to batch segments
                current_logprobs = current_logprobs.reshape(batch_size, segment_length)
                initial_logprobs = initial_logprobs.reshape(batch_size, segment_length)
                
                # Clip differences before squaring (per segment)
                diff = jnp.clip(current_logprobs - initial_logprobs, -10.0, 10.0)
                return jnp.mean(diff ** 2)
            
            preferred_conservative = compute_conservative_loss(
                preferred["states"], preferred["actions"]
            )
            non_preferred_conservative = compute_conservative_loss(
                non_preferred["states"], non_preferred["actions"]
            )
            
            conservative_loss = (preferred_conservative + non_preferred_conservative) / 2.0
            
            # Combine losses with conservative regularization
            total_loss = stable_preference_loss + config.conservative_weight * conservative_loss
            
            return total_loss, {
                "loss": total_loss,
                "preference_loss": stable_preference_loss,
                "conservative_loss": conservative_loss,
                "preferred_logprobs_mean": jnp.mean(preferred_logprobs),
                "non_preferred_logprobs_mean": jnp.mean(non_preferred_logprobs),
                "preferred_segment_adv": jnp.mean(preferred_segment_adv),
                "non_preferred_segment_adv": jnp.mean(non_preferred_segment_adv),
            }
        
        try:
            # Get gradients with respect to the params
            grads, info = jax.grad(cpl_loss_fn, has_aux=True)(self.actor.params)
            
            # Remove the manual scaling and just use the optimizer directly
            new_actor = self.actor.apply_gradients(grads=grads)  # The optimizer will handle the learning rate
            
            actor = self.actor.replace(params=new_actor.params)
            
            return self.replace(actor=actor), info
            
        except Exception as e:
            print("\nError during gradient computation or update:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise

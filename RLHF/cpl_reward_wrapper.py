import numpy as np

class CPLRewardWrapper:
    """Wrapper that uses the CPL reward model to modify environment rewards."""
    
    def __init__(self, env, reward_model, reward_scale=1.0, original_reward_weight=0.0):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            reward_model: The CPL reward model
            reward_scale: Scaling factor for CPL rewards
            original_reward_weight: Weight for original environment rewards (0 to 1)
                                   0 = only use CPL rewards, 1 = only use original rewards
        """
        self.env = env
        self.reward_model = reward_model
        self.reward_scale = reward_scale
        self.original_reward_weight = original_reward_weight
        self.last_state = None
        self.last_action = None
    
    def step(self, action):
        """Step the environment and modify the reward."""
        # Store last state and action for reward computation
        self.last_state = self.env.physics.get_state()
        self.last_action = action
        
        # Take a step in the environment
        timestep = self.env.step(action)
        
        if not timestep.first():  # Skip reward modification on first step
            # Get environment observation and last action
            obs = timestep.observation
            
            # Calculate reward from CPL model
            cpl_reward = self.reward_model.predict_rewards(
                np.expand_dims(obs, axis=0), 
                np.expand_dims(self.last_action, axis=0)
            )[0]
            
            # Scale CPL reward
            cpl_reward = cpl_reward * self.reward_scale
            
            # Combine rewards if needed
            if self.original_reward_weight > 0:
                combined_reward = (
                    self.original_reward_weight * timestep.reward + 
                    (1 - self.original_reward_weight) * cpl_reward
                )
            else:
                combined_reward = cpl_reward
            
            # Create new timestep with modified reward
            timestep = timestep._replace(reward=combined_reward)
        
        return timestep
    
    def reset(self):
        """Reset the environment."""
        timestep = self.env.reset()
            
        self.last_state = None
        self.last_action = None
        return timestep
    
    # Forward all other method calls to the underlying environment
    def __getattr__(self, name):
        return getattr(self.env, name)


# custom_wrappers.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MyNormalizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Example: scale obs in some way
        self.obs_min = getattr(env.observation_space, "low", None)
        self.obs_max = getattr(env.observation_space, "high", None)

    def observation(self, obs):
        if self.obs_min is not None and self.obs_max is not None:
            return (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-8)
        return obs

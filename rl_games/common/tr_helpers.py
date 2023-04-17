import numpy as np
from collections import defaultdict

class DefaultRewardsShaper:
    def __init__(self, scale_value = 1, shift_value = 0, min_val=-np.inf, max_val=np.inf, is_torch=True):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.is_torch = is_torch

    def __call__(self, reward):
        # print(f"scaling reward by {self.scale_value} and shifting by {self.shift_value}")
        reward = reward + self.shift_value
        reward = reward * self.scale_value
 
        if self.is_torch:
            import torch
            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward


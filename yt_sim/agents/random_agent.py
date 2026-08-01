"""
Random baseline agent.

Samples a uniformly random *valid* action from the observation's action mask at
each step. It is deliberately reward-agnostic (it ignores the reward, which is
always 0) -- a baseline for comparing against a real intrinsic-reward agent.
"""
import numpy as np


class RandomAgent:
    """Selects a uniformly random valid action given an observation."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        """Return a random action id among those marked valid in ``action_mask``."""
        valid = np.flatnonzero(observation["action_mask"])
        return int(self.rng.choice(valid))

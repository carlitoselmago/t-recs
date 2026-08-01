import numpy as np

from yt_sim import random_embeddings
from yt_sim.env import YouTubeSimEnv
from yt_sim.agents import RandomAgent


class TestRandomAgent:
    def test_only_picks_valid_actions(self):
        env = YouTubeSimEnv(random_embeddings(120, 10, seed=0), slate_size=4, max_steps=60, seed=0)
        agent = RandomAgent(seed=1)
        obs, _ = env.reset(seed=0)
        for _ in range(60):
            a = agent.act(obs)
            assert obs["action_mask"][a] == 1  # never selects a masked-out action
            obs, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break

    def test_full_episode_runs_to_truncation(self):
        env = YouTubeSimEnv(random_embeddings(120, 10, seed=0), slate_size=4, max_steps=30, seed=0)
        agent = RandomAgent(seed=2)
        obs, _ = env.reset(seed=0)
        steps = 0
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(agent.act(obs))
            steps += 1
            done = terminated or truncated
        assert steps == 30

import numpy as np
import pytest

from yt_sim import random_embeddings, actions
from yt_sim.env import YouTubeSimEnv


def _env(**kw):
    emb = random_embeddings(200, 12, seed=0)
    defaults = dict(slate_size=5, candidate_pool=40, max_steps=50, seed=0)
    defaults.update(kw)
    return YouTubeSimEnv(emb, **defaults)


class TestEnvAPI:
    def test_reset_observation_in_space(self):
        env = _env()
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert set(obs.keys()) == {"current_embedding", "slate_embeddings", "action_mask"}
        assert obs["slate_embeddings"].shape == (5, 12)
        assert info == {"step": 0}

    def test_step_reward_is_zero_and_reward_agnostic(self):
        env = _env()
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            a = int(np.flatnonzero(obs["action_mask"])[0])
            obs, reward, terminated, truncated, _ = env.step(a)
            assert reward == 0.0
            assert terminated is False

    def test_truncation_at_max_steps(self):
        env = _env(max_steps=5)
        obs, _ = env.reset(seed=0)
        truncated = False
        steps = 0
        while not truncated:
            obs, _, _, truncated, _ = env.step(1)  # skip repeatedly
            steps += 1
        assert steps == 5

    def test_like_gated_off_by_default(self):
        env = _env(like_enabled=False)
        obs, _ = env.reset(seed=0)
        assert obs["action_mask"][2] == 0  # like id not available
        env2 = _env(like_enabled=True)
        obs2, _ = env2.reset(seed=0)
        assert obs2["action_mask"][2] == 1


class TestEnvMechanics:
    def test_click_transitions_current_video(self):
        env = _env()
        env.reset(seed=1)
        target = int(env.slate[0])
        click_id = env.schema.slate_action_ids(0)[0]
        assert env.schema.decode(click_id).op == actions.CLICK
        env.step(click_id)
        assert env.current_video == target

    def test_not_interested_suppresses_item_across_future_slates(self):
        env = _env()
        env.reset(seed=2)
        victim = int(env.slate[0])
        ni_id = env.schema.slate_action_ids(0)[1]
        assert env.schema.decode(ni_id).op == actions.NOT_INTERESTED
        env.step(ni_id)
        assert victim not in env.slate.tolist()
        # exact item stays suppressed on subsequent regenerations
        for _ in range(10):
            env.step(1)  # skip -> regenerates slate
            assert victim not in env.slate.tolist()

    def test_deterministic_reset_options(self):
        env = _env()
        pref = np.ones(12) / np.sqrt(12)
        obs_a, _ = env.reset(options={"start_video": 42, "preference": pref})
        assert env.current_video == 42
        obs_b, _ = env.reset(options={"start_video": 42, "preference": pref})
        np.testing.assert_array_equal(obs_a["slate_embeddings"], obs_b["slate_embeddings"])


class TestEnvOpacity:
    def test_observation_and_info_leak_no_action_semantics(self):
        env = _env()
        obs, info = env.reset(seed=0)
        # observation is strictly embeddings + a numeric action mask
        for key, val in obs.items():
            assert isinstance(val, np.ndarray)
        # info never carries op names or effect descriptions
        leak_terms = {
            actions.WATCH_FULL,
            actions.SKIP,
            actions.LIKE,
            actions.CLICK,
            actions.NOT_INTERESTED,
            "watch_fraction",
            "preference",
        }
        for _ in range(5):
            a = int(np.flatnonzero(obs["action_mask"])[0])
            obs, _, _, _, info = env.step(a)
            assert leak_terms.isdisjoint(set(map(str, info.keys())))
            assert set(obs.keys()) == {"current_embedding", "slate_embeddings", "action_mask"}

"""
Random-agent baseline for the yt_sim YouTube-like environment.

Runs a uniformly-random policy against :class:`yt_sim.env.YouTubeSimEnv` and
prints simple rollout statistics. This is a *baseline*: the environment is
reward-agnostic (reward is always 0), so a real agent is expected to compute its
own intrinsic reward from the returned observations and beat these numbers on
whatever objective it defines.

Usage::

    python examples/random_agent_baseline.py
    python examples/random_agent_baseline.py --items 500 --dim 32 --steps 300 --episodes 5
"""
import argparse
from collections import Counter

import numpy as np

from yt_sim import random_embeddings, actions
from yt_sim.env import YouTubeSimEnv
from yt_sim.agents import RandomAgent


def run_episode(env, agent, seed):
    obs, _ = env.reset(seed=seed)
    op_counts = Counter()
    slate_churn = 0  # how often the top suggestion changes step-to-step
    prev_top = tuple(obs["slate_embeddings"][0])
    done = False
    while not done:
        action = agent.act(obs)
        op_counts[env.schema.decode(action).op] += 1
        obs, _reward, terminated, truncated, _ = env.step(action)
        top = tuple(obs["slate_embeddings"][0])
        slate_churn += int(top != prev_top)
        prev_top = top
        done = terminated or truncated
    return op_counts, slate_churn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=500, help="catalog size")
    parser.add_argument("--dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--slate", type=int, default=5, help="slate size")
    parser.add_argument("--steps", type=int, default=200, help="max steps per episode")
    parser.add_argument("--episodes", type=int, default=3, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # In a real study these would be CLIP thumbnail embeddings loaded from disk
    # (see yt_sim.embeddings.load_embeddings); here we use random unit vectors.
    embeddings = random_embeddings(args.items, args.dim, seed=args.seed)
    env = YouTubeSimEnv(
        embeddings, slate_size=args.slate, max_steps=args.steps, seed=args.seed
    )
    agent = RandomAgent(seed=args.seed)

    print(
        f"catalog={args.items} items  dim={args.dim}  slate={args.slate}  "
        f"steps={args.steps}  episodes={args.episodes}"
    )
    print("-" * 64)
    totals = Counter()
    churns = []
    for ep in range(args.episodes):
        op_counts, churn = run_episode(env, agent, seed=args.seed + ep)
        totals.update(op_counts)
        churns.append(churn)
        breakdown = "  ".join(f"{op}={op_counts.get(op, 0)}" for op in _OP_ORDER)
        print(f"episode {ep}:  {breakdown}   slate_churn={churn}/{args.steps}")

    print("-" * 64)
    total_actions = sum(totals.values())
    print("aggregate action distribution:")
    for op in _OP_ORDER:
        c = totals.get(op, 0)
        print(f"  {op:16s} {c:6d}  ({100 * c / max(total_actions, 1):5.1f}%)")
    print(f"mean slate churn: {np.mean(churns):.1f}/{args.steps}")


_OP_ORDER = (
    actions.WATCH_FULL,
    actions.SKIP,
    actions.CLICK,
    actions.NOT_INTERESTED,
    actions.LIKE,
)


if __name__ == "__main__":
    main()

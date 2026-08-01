"""
yt_sim -- a YouTube-like recommendation simulator built on top of T-RECS.

Extends T-RECS with thumbnail-embedding items, a two-stage (k-NN candidate
generation + ranking) recommendation funnel, a two-tier action space with
pluggable action effects, "not interested" neighborhood masking, a continuous
watch-simulation signal, and a reward-agnostic Gymnasium environment for
black-box RL.

The Gymnasium environment (:class:`~yt_sim.env.YouTubeSimEnv`) is imported lazily
so that importing :mod:`yt_sim` does not require ``gymnasium`` unless the env is
actually used.
"""
from .embeddings import EmbeddingItems, load_embeddings, random_embeddings
from .funnel import FunnelRecommender
from .ranking import RankingFunction, PopularityRecencySimilarityRanker
from .watch import WatchModel
from .masking import NotInterestedMask
from .action_effects import ActionEffect, DefaultActionEffect
from .actions import ActionSchema

__all__ = [
    "EmbeddingItems",
    "load_embeddings",
    "random_embeddings",
    "FunnelRecommender",
    "RankingFunction",
    "PopularityRecencySimilarityRanker",
    "WatchModel",
    "NotInterestedMask",
    "ActionEffect",
    "DefaultActionEffect",
    "ActionSchema",
    "YouTubeSimEnv",
]


def __getattr__(name):
    # lazy import so `import yt_sim` works without gymnasium installed
    if name == "YouTubeSimEnv":
        from .env import YouTubeSimEnv  # pylint: disable=import-outside-toplevel

        return YouTubeSimEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

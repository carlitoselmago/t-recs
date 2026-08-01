"""
Stage-2 ranking for the recommendation funnel.

Given the candidate set produced by Stage-1 (k-NN candidate generation), a
:class:`RankingFunction` orders the candidates and selects the top ``k`` to show
in the slate. The default ranker combines a few simple, configurable signals --
embedding similarity to the viewer's recent watches, item popularity, and item
recency -- so it is easy to extend or replace later.
"""
import numpy as np


def _minmax(x):
    """Scale an array to ``[0, 1]``; a flat array maps to all-zeros."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class RankingFunction:
    """Base class for pluggable Stage-2 ranking functions."""

    def rank(self, sim_scores, popularity, recency, k, rng=None):  # pylint: disable=unused-argument
        """
        Return the indices (into the candidate arrays) of the top ``k`` items,
        ordered best-first. Subclasses override this.
        """
        raise NotImplementedError


class PopularityRecencySimilarityRanker(RankingFunction):
    """
    Weighted-sum ranker over three min-max-normalized signals.

    Parameters
    ----------
    w_similarity, w_popularity, w_recency : float
        Non-negative weights for embedding similarity (to recent watches),
        popularity (interaction counts), and recency respectively.
    """

    def __init__(self, w_similarity=1.0, w_popularity=0.3, w_recency=0.2):
        self.w_similarity = w_similarity
        self.w_popularity = w_popularity
        self.w_recency = w_recency

    def score(self, sim_scores, popularity, recency):
        """Combined per-candidate score (higher is better)."""
        return (
            self.w_similarity * _minmax(sim_scores)
            + self.w_popularity * _minmax(popularity)
            + self.w_recency * _minmax(recency)
        )

    def rank(self, sim_scores, popularity, recency, k, rng=None):
        combined = self.score(sim_scores, popularity, recency)
        n = combined.shape[0]
        k = min(k, n)
        if rng is not None:
            # random tiebreak so equal scores do not always resolve by index
            tiebreak = rng.random(n)
            order = np.lexsort((tiebreak, combined))[::-1]
        else:
            order = np.argsort(combined)[::-1]
        return order[:k]

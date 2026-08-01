"""
Two-stage recommendation funnel.

:class:`FunnelRecommender` follows T-RECS's model-subclassing pattern (it extends
:class:`trecs.models.recommender.BaseRecommender`) so it plugs into the standard
simulation loop -- predict scores, present a slate, collect feedback, update
state, update metrics -- rather than replacing that loop. On top of that it
overrides recommendation generation with a two-stage funnel:

* **Stage 1 -- candidate generation**: k-NN over item embeddings, given the
  viewer's current preference state, respecting any active "not interested"
  masking (:mod:`yt_sim.masking`).
* **Stage 2 -- ranking**: a pluggable :class:`~yt_sim.ranking.RankingFunction`
  (default: similarity + popularity + recency) picks the slate from the
  candidate set.

Item attributes are thumbnail embeddings (:class:`~yt_sim.embeddings.EmbeddingItems`);
the recommender is assumed to know them exactly (a static catalog), so the
system's predicted item representation equals the true one.

The class works for an arbitrary number of viewers when driven by
``BaseRecommender.run()``. The Gymnasium environment
(:class:`yt_sim.env.YouTubeSimEnv`) instead drives a single viewer
(``num_users=1``) and injects the agent's actions in place of the simulated
user-choice step.
"""
import numpy as np

import trecs.matrix_ops as mo
from trecs.models.recommender import BaseRecommender

from .embeddings import EmbeddingItems
from .ranking import PopularityRecencySimilarityRanker


def _normalize_rows(mat):
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FunnelRecommender(BaseRecommender):
    """
    A k-NN-candidate-generation + ranking recommender over item embeddings.

    Parameters
    ----------
    item_embeddings : :obj:`numpy.ndarray` or :class:`~yt_sim.embeddings.EmbeddingItems`
        ``[n_items, embed_dim]`` embedding matrix (or an ``EmbeddingItems``).
    actual_user_representation : optional
        True viewer preference(s), shape ``[num_users, embed_dim]`` or a T-RECS
        ``Users`` object. If None, random preferences are generated.
    num_users : int, default 1
        Number of viewers.
    num_items_per_iter : int, default 5
        Slate size ``k`` shown each step.
    candidate_pool : int, default 50
        Stage-1 candidate pool size (number of nearest neighbors considered
        before ranking).
    ranker : :class:`~yt_sim.ranking.RankingFunction`, optional
        Stage-2 ranker. Defaults to
        :class:`~yt_sim.ranking.PopularityRecencySimilarityRanker`.
    belief_lr : float, default 0.5
        Convex-blend rate at which the system's belief about a viewer drifts
        toward embeddings of items they interact with.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        item_embeddings,
        actual_user_representation=None,
        num_users=1,
        num_items_per_iter=5,
        candidate_pool=50,
        ranker=None,
        belief_lr=0.5,
        seed=None,
        **kwargs,
    ):
        if isinstance(item_embeddings, EmbeddingItems):
            items = item_embeddings
        else:
            items = EmbeddingItems(embeddings=item_embeddings, seed=seed)
        embed_dim, n_items = items.current_state.shape

        # system belief about each viewer's preference (starts neutral)
        user_representation = np.zeros((num_users, embed_dim))
        item_representation = items.current_state  # embed_dim x n_items (known exactly)

        # funnel configuration (set before super().__init__ so it is available
        # if recommendation is triggered during construction)
        self.candidate_pool = int(candidate_pool)
        self.ranker = ranker if ranker is not None else PopularityRecencySimilarityRanker()
        self.belief_lr = float(belief_lr)
        # per-viewer "not interested" masks; populated by callers (e.g. the env)
        self.masks = {}
        # ranking signals over items
        self._popularity = np.zeros(n_items, dtype=float)
        # recency: static per-item creation order (newer -> larger); extensible
        self._created_at = np.arange(n_items, dtype=float)
        # cache normalized embeddings (n_items x embed_dim) for cheap cosine ops
        self._norm_emb = _normalize_rows(items.embeddings)

        BaseRecommender.__init__(
            self,
            user_representation,
            item_representation,
            actual_user_representation,
            item_representation.copy(),
            num_users,
            n_items,
            num_items_per_iter,
            score_fn=mo.inner_product,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Convenience accessors used by the funnel and the environment
    # ------------------------------------------------------------------ #
    @property
    def item_embeddings(self):
        """Item embeddings in ``[n_items, embed_dim]`` orientation."""
        return np.asarray(self.items_hat.value).T

    @property
    def normalized_item_embeddings(self):
        """Cached L2-normalized item embeddings, ``[n_items, embed_dim]``."""
        return self._norm_emb

    @property
    def popularity(self):
        """Per-item interaction counts used by the ranker."""
        return self._popularity

    def preference(self, user_idx=0):
        """The system's current belief vector for a viewer."""
        return np.asarray(self.users_hat.value[user_idx], dtype=float)

    def set_preference(self, vector, user_idx=0):
        """Overwrite a viewer's belief vector (used on episode reset)."""
        self.users_hat.value[user_idx] = np.asarray(vector, dtype=float)

    def cosine_to_item(self, item_id, user_idx=0):
        """Cosine similarity between a viewer's belief and an item embedding."""
        pref = self.preference(user_idx)
        norm = np.linalg.norm(pref)
        if norm == 0:
            return 0.0
        return float((pref / norm) @ self._norm_emb[item_id])

    def drift_preference(self, item_id, strength, user_idx=0):
        """
        Drift a viewer's belief toward an item's embedding by a convex blend.

        ``strength`` in ``[0, 1]`` scales how far the belief moves; the
        simulator typically sets it from the watch fraction so preference moves
        more for genuinely engaging videos.
        """
        strength = float(np.clip(strength, 0.0, 1.0))
        pref = self.preference(user_idx)
        target = self.item_embeddings[item_id]
        self.users_hat.value[user_idx] = (1.0 - strength) * pref + strength * target

    def register_interaction(self, item_id):
        """Record an interaction with an item (updates popularity)."""
        self._popularity[int(item_id)] += 1.0

    # ------------------------------------------------------------------ #
    # Two-stage funnel
    # ------------------------------------------------------------------ #
    def _similarity_scores(self, user_idx):
        """Cosine similarity of a viewer's belief to every item."""
        pref = self.preference(user_idx)
        norm = np.linalg.norm(pref)
        if norm == 0:
            # neutral belief -> rely on popularity/recency via the ranker
            return np.zeros(self.num_items)
        return self._norm_emb @ (pref / norm)

    def recommend_slate(self, user_idx=0, k=None, exclude_ids=None, allowed_ids=None):
        """
        Produce a ranked slate for a single viewer via the two-stage funnel.

        Parameters
        ----------
        user_idx : int
            Which viewer to recommend for.
        k : int, optional
            Slate size; defaults to ``num_items_per_iter``.
        exclude_ids : iterable of int, optional
            Item ids to exclude (e.g. the currently playing video).
        allowed_ids : iterable of int, optional
            If given, restrict candidates to this set (before masking).
        """
        if k is None:
            k = self.num_items_per_iter
        sim = self._similarity_scores(user_idx).copy()

        # restrict to an allowed set if provided
        if allowed_ids is not None:
            keep = np.zeros(self.num_items, dtype=bool)
            keep[np.asarray(list(allowed_ids), dtype=int)] = True
            sim[~keep] = -np.inf

        # Stage 1: apply "not interested" masking, then take the k-NN candidates
        mask = self.masks.get(user_idx)
        if mask is not None:
            masked = mask.active_mask(self.current_step)
            if masked.size:
                sim[masked] = -np.inf
        if exclude_ids is not None:
            for item_id in exclude_ids:
                sim[int(item_id)] = -np.inf

        valid = np.flatnonzero(np.isfinite(sim))
        if valid.size == 0:
            return np.array([], dtype=int)
        pool = min(self.candidate_pool, valid.size)
        # top `pool` candidates by similarity
        candidate_ids = valid[np.argpartition(-sim[valid], pool - 1)[:pool]]

        # Stage 2: rank the candidate pool and take the top k
        order = self.ranker.rank(
            sim_scores=sim[candidate_ids],
            popularity=self._popularity[candidate_ids],
            recency=self._created_at[candidate_ids],
            k=k,
            rng=self.random_state,
        )
        return candidate_ids[order]

    @property
    def current_step(self):
        """Current simulation step (drives mask expiry). Managed by the caller."""
        return getattr(self, "_current_step", 0)

    @current_step.setter
    def current_step(self, value):
        self._current_step = int(value)

    # ------------------------------------------------------------------ #
    # BaseRecommender hooks (used by the standalone run() loop)
    # ------------------------------------------------------------------ #
    def generate_recommendations(self, k=1, item_indices=None):
        if k == 0:
            return np.array([]).reshape((self.num_users, 0)).astype(int)
        recs = np.zeros((self.num_users, k), dtype=int)
        for user in range(self.num_users):
            allowed = None if item_indices is None else item_indices[user]
            slate = self.recommend_slate(user_idx=user, k=k, allowed_ids=allowed)
            if slate.size < k:
                # pad with the best remaining allowed items if the funnel came up
                # short (small catalogs / heavy masking)
                pad_pool = allowed if allowed is not None else np.arange(self.num_items)
                pad = [i for i in np.asarray(pad_pool, dtype=int) if i not in set(slate)]
                slate = np.concatenate([slate, np.array(pad[: k - slate.size], dtype=int)])
            recs[user] = slate[:k]
        return recs

    def _update_internal_state(self, interactions):
        """Drift each viewer's belief toward the item they interacted with."""
        for user in range(self.num_users):
            item_id = int(interactions[user])
            self.register_interaction(item_id)
            self.drift_preference(item_id, strength=self.belief_lr, user_idx=user)

    def process_new_items(self, new_items):
        """Static catalog by default: new items are not supported in the funnel."""
        raise NotImplementedError(
            "FunnelRecommender assumes a static catalog; dynamic item creation "
            "is not supported."
        )

    def process_new_users(self, new_users, **kwargs):
        # new viewers start with a neutral belief
        return np.zeros((new_users.shape[0], self.users_hat.num_attrs))

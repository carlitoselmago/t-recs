"""
"Not interested" neighborhood masking.

When the agent marks a suggested video as *not interested*, we suppress not only
that exact item but also its ``k`` nearest neighbors in embedding space from
future candidate generation.

Motivated by how YouTube's "Not interested" is understood to behave in practice:
the *exact* video is reliably suppressed, but it is only a weak, secondary signal
for *similar* content -- neighbors tend to resurface over time rather than being
banned forever (see the study discussion in the project README). Accordingly the
default here is:

* exact item  -> suppressed permanently,
* neighbors   -> suppressed for a configurable number of steps, then allowed back.

Set ``neighbor_suppression_steps=None`` for a permanent neighborhood ban instead.
"""
import numpy as np


class NotInterestedMask:
    """
    Tracks suppressed items for a single viewer and integrates with the funnel's
    candidate-generation stage.

    Parameters
    ----------
    normalized_embeddings : :obj:`numpy.ndarray`
        ``[n_items, embed_dim]`` matrix of L2-normalized item embeddings, used to
        find nearest neighbors by cosine similarity.
    n_neighbors : int, default 10
        Number of nearest neighbors (``k``) suppressed alongside the exact item.
    neighbor_suppression_steps : int or None, default 50
        How many steps neighbor suppression lasts. ``None`` makes it permanent.
    mask_neighbors : bool, default True
        If False, only the exact item is suppressed (neighbors are untouched).
    """

    def __init__(
        self,
        normalized_embeddings,
        n_neighbors=10,
        neighbor_suppression_steps=50,
        mask_neighbors=True,
    ):
        self._emb = np.asarray(normalized_embeddings, dtype=float)
        self.n_items = self._emb.shape[0]
        self.n_neighbors = int(n_neighbors)
        self.neighbor_suppression_steps = neighbor_suppression_steps
        self.mask_neighbors = mask_neighbors
        self.reset()

    def reset(self):
        """Clear all suppression state."""
        # exact items are suppressed permanently
        self._exact = set()
        # item_id -> expiry step (np.inf for permanent)
        self._neighbor_expiry = {}

    def _nearest_neighbors(self, item_id):
        """Return the ids of the ``n_neighbors`` items closest to ``item_id``."""
        if self.n_neighbors <= 0:
            return np.array([], dtype=int)
        sims = self._emb @ self._emb[item_id]
        sims[item_id] = -np.inf  # never count the item itself as its own neighbor
        k = min(self.n_neighbors, self.n_items - 1)
        # top-k by similarity (unordered is fine -- we only need the set)
        nn = np.argpartition(-sims, k - 1)[:k]
        return nn

    def not_interested(self, item_id, current_step):
        """
        Register a "not interested" signal on ``item_id`` at ``current_step``.

        The exact item is suppressed permanently; its neighbors are suppressed
        until they expire (unless configured permanent).
        """
        item_id = int(item_id)
        self._exact.add(item_id)
        if not self.mask_neighbors:
            return
        if self.neighbor_suppression_steps is None:
            expiry = np.inf
        else:
            expiry = current_step + int(self.neighbor_suppression_steps)
        for nb in self._nearest_neighbors(item_id):
            nb = int(nb)
            if nb in self._exact:
                continue
            # keep the latest (largest) expiry if re-triggered
            self._neighbor_expiry[nb] = max(self._neighbor_expiry.get(nb, -np.inf), expiry)

    def active_mask(self, current_step):
        """Return a sorted array of item ids currently suppressed."""
        masked = set(self._exact)
        for item_id, expiry in self._neighbor_expiry.items():
            if expiry > current_step:
                masked.add(item_id)
        return np.array(sorted(masked), dtype=int)

    def is_masked(self, item_id, current_step):
        """Whether ``item_id`` is currently suppressed."""
        if item_id in self._exact:
            return True
        expiry = self._neighbor_expiry.get(int(item_id))
        return expiry is not None and expiry > current_step

import numpy as np

from yt_sim.embeddings import random_embeddings
from yt_sim.masking import NotInterestedMask


def _norm(emb):
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


class TestNotInterestedMask:
    def test_exact_permanent_and_neighbors_temporary(self):
        emb = _norm(random_embeddings(100, 8, seed=0))
        mask = NotInterestedMask(emb, n_neighbors=5, neighbor_suppression_steps=10)
        mask.not_interested(item_id=3, current_step=0)
        active = mask.active_mask(current_step=0)
        assert 3 in active
        assert active.size == 6  # exact + 5 neighbors

        # neighbors expire, exact item stays forever
        later = mask.active_mask(current_step=11)
        assert 3 in later
        assert later.size == 1

    def test_permanent_neighbors(self):
        emb = _norm(random_embeddings(50, 8, seed=1))
        mask = NotInterestedMask(emb, n_neighbors=4, neighbor_suppression_steps=None)
        mask.not_interested(2, 0)
        assert mask.active_mask(current_step=10_000).size == 5

    def test_mask_neighbors_disabled(self):
        emb = _norm(random_embeddings(50, 8, seed=2))
        mask = NotInterestedMask(emb, n_neighbors=5, mask_neighbors=False)
        mask.not_interested(7, 0)
        active = mask.active_mask(0)
        assert active.tolist() == [7]

    def test_is_masked_and_reset(self):
        emb = _norm(random_embeddings(40, 8, seed=3))
        mask = NotInterestedMask(emb, n_neighbors=3, neighbor_suppression_steps=5)
        mask.not_interested(1, 0)
        assert mask.is_masked(1, 0)
        mask.reset()
        assert not mask.is_masked(1, 0)
        assert mask.active_mask(0).size == 0

    def test_item_is_not_its_own_neighbor(self):
        emb = _norm(random_embeddings(30, 8, seed=4))
        mask = NotInterestedMask(emb, n_neighbors=3)
        nn = mask._nearest_neighbors(5)
        assert 5 not in nn

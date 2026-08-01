import warnings

import numpy as np
import pytest

from yt_sim import random_embeddings, FunnelRecommender
from yt_sim.masking import NotInterestedMask


def _funnel(n_items=80, dim=8, num_users=1, seed=0, **kw):
    emb = random_embeddings(n_items, dim, seed=seed)
    users = np.random.default_rng(seed + 1).normal(size=(num_users, dim))
    return FunnelRecommender(
        emb, actual_user_representation=users, num_users=num_users, seed=seed, **kw
    )


class TestFunnel:
    def test_construction_shapes(self):
        rec = _funnel()
        assert rec.item_embeddings.shape == (80, 8)
        assert rec.normalized_item_embeddings.shape == (80, 8)
        assert rec.num_items == 80

    def test_recommend_slate_size_and_exclusion(self):
        rec = _funnel(num_items_per_iter=5, candidate_pool=20)
        rec.set_preference(rec.item_embeddings[0])
        slate = rec.recommend_slate(k=5, exclude_ids=[0])
        assert slate.size == 5
        assert 0 not in slate
        assert len(set(slate.tolist())) == 5  # no duplicates

    def test_recommend_slate_respects_mask(self):
        rec = _funnel(num_items_per_iter=5, candidate_pool=30)
        mask = NotInterestedMask(rec.normalized_item_embeddings, n_neighbors=5)
        rec.masks[0] = mask
        rec.set_preference(rec.item_embeddings[10])
        # suppress item 10's neighborhood and confirm none appear
        mask.not_interested(10, current_step=0)
        suppressed = set(mask.active_mask(0).tolist())
        slate = rec.recommend_slate(k=5, exclude_ids=[])
        assert suppressed.isdisjoint(set(slate.tolist()))

    def test_drift_moves_preference_toward_item(self):
        rec = _funnel()
        rec.set_preference(np.zeros(rec.item_embeddings.shape[1]))
        target = rec.item_embeddings[7]
        before = rec.cosine_to_item(7)
        rec.drift_preference(7, strength=0.5)
        after = rec.cosine_to_item(7)
        assert after > before

    def test_generate_recommendations_shape(self):
        rec = _funnel(num_users=3, num_items_per_iter=4)
        recs = rec.generate_recommendations(k=4, item_indices=rec.indices)
        assert recs.shape == (3, 4)

    def test_standalone_run_under_trecs_loop(self):
        from trecs.metrics import InteractionMeasurement

        rec = _funnel(num_users=3, num_items_per_iter=5)
        rec.add_metrics(InteractionMeasurement())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec.run(timesteps=4, disable_tqdm=True)
        hist = rec.get_measurements()["interaction_histogram"]
        assert len(hist) == 5  # initial + 4 steps

    def test_process_new_items_unsupported(self):
        rec = _funnel()
        with pytest.raises(NotImplementedError):
            rec.process_new_items(np.zeros((8, 2)))

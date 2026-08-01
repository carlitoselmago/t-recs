import numpy as np

from yt_sim.ranking import PopularityRecencySimilarityRanker, _minmax


class TestRanking:
    def test_minmax_flat_is_zero(self):
        np.testing.assert_array_equal(_minmax(np.array([2.0, 2.0, 2.0])), np.zeros(3))

    def test_similarity_dominates_by_default(self):
        ranker = PopularityRecencySimilarityRanker(w_similarity=1.0, w_popularity=0.0, w_recency=0.0)
        sim = np.array([0.1, 0.9, 0.5])
        top = ranker.rank(sim_scores=sim, popularity=np.zeros(3), recency=np.zeros(3), k=1)
        assert top.tolist() == [1]

    def test_popularity_weight(self):
        ranker = PopularityRecencySimilarityRanker(w_similarity=0.0, w_popularity=1.0, w_recency=0.0)
        pop = np.array([5.0, 1.0, 3.0])
        top = ranker.rank(sim_scores=np.zeros(3), popularity=pop, recency=np.zeros(3), k=2)
        assert top[0] == 0

    def test_k_capped_at_n(self):
        ranker = PopularityRecencySimilarityRanker()
        out = ranker.rank(
            sim_scores=np.array([1.0, 2.0]), popularity=np.zeros(2), recency=np.zeros(2), k=10
        )
        assert out.shape[0] == 2

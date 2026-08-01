import numpy as np
import pytest

from yt_sim.embeddings import EmbeddingItems, random_embeddings, load_embeddings


class TestEmbeddingItems:
    def test_orientation_matches_trecs(self):
        emb = random_embeddings(10, 4, seed=0)
        items = EmbeddingItems(embeddings=emb)
        # stored as attributes x items (embed_dim x n_items)
        assert items.current_state.shape == (4, 10)
        # natural orientation restored
        assert items.embeddings.shape == (10, 4)
        np.testing.assert_allclose(items.embeddings, emb)
        assert items.embed_dim == 4
        assert items.num_items == 10

    def test_requires_embeddings_when_enabled(self):
        with pytest.raises(ValueError):
            EmbeddingItems(use_embeddings=True)

    def test_fallback_path_generates_random(self):
        # fallback preserves the original distribution-sampled attribute path
        items = EmbeddingItems(use_embeddings=False, size=(4, 10), seed=1)
        assert items.current_state.shape == (4, 10)
        assert items.use_embeddings is False

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            EmbeddingItems(embeddings=np.zeros((3,)))

    def test_random_embeddings_unit_norm(self):
        emb = random_embeddings(20, 8, seed=3)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(20), atol=1e-6)

    def test_load_embeddings_roundtrip(self, tmp_path):
        emb = random_embeddings(5, 3, seed=4)
        path = tmp_path / "emb.npy"
        np.save(path, emb)
        np.testing.assert_allclose(load_embeddings(str(path)), emb)

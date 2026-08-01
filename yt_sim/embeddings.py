"""
Thumbnail-embedding item representation.

:class:`EmbeddingItems` extends T-RECS's :class:`trecs.components.items.Items`
so that items carry a fixed-length embedding vector (e.g. the output of a CLIP
image encoder) instead of, or alongside, the default distribution-sampled
attribute vector.

T-RECS stores items internally as an ``|A| x |I|`` matrix (attributes as rows,
items as columns). An embedding matrix is naturally ``[n_items, embed_dim]``, so
we store its transpose and expose the natural orientation via
:attr:`EmbeddingItems.embeddings`. Because the stored matrix is still just
``attributes x items``, everything downstream in T-RECS (score functions,
predicted-score plumbing) works unchanged.
"""
import numpy as np

from trecs.components.items import Items


class EmbeddingItems(Items):  # pylint: disable=too-many-ancestors
    """
    Items backed by a precomputed embedding matrix.

    Parameters
    ----------
    embeddings : array_like, optional
        Embedding matrix of shape ``[n_items, embed_dim]``. Required when
        ``use_embeddings`` is True.
    size : tuple, optional
        ``(embed_dim, n_items)`` used only for the fallback path
        (``use_embeddings=False``), where the base :class:`Items` class
        generates a random attribute matrix. Provided for testing without real
        embeddings.
    use_embeddings : bool, default True
        If True, use the supplied ``embeddings`` matrix. If False, fall back to
        T-RECS's default randomly generated attribute vectors (the original
        behaviour), controlled entirely by this flag.
    seed : int, optional
        Seed for the fallback random generation.
    """

    def __init__(
        self,
        embeddings=None,
        size=None,
        use_embeddings=True,
        verbose=False,
        seed=None,
        name="embedding_items",
    ):  # pylint: disable=too-many-arguments
        if use_embeddings:
            if embeddings is None:
                raise ValueError("embeddings must be provided when use_embeddings=True")
            emb = np.asarray(embeddings, dtype=float)
            if emb.ndim != 2:
                raise ValueError("embeddings must be a 2D array of shape [n_items, embed_dim]")
            # store as (embed_dim x n_items) to match T-RECS' attributes x items layout
            item_attributes = emb.T
        else:
            # fallback: let the base Items class generate a random matrix from `size`
            if size is None:
                raise ValueError("size must be provided when use_embeddings=False")
            item_attributes = None
        Items.__init__(
            self, item_attributes=item_attributes, size=size, verbose=verbose, seed=seed, name=name
        )
        self.use_embeddings = use_embeddings

    @property
    def embeddings(self):
        """Return item embeddings in natural ``[n_items, embed_dim]`` orientation."""
        return np.asarray(self.current_state).T

    @property
    def embed_dim(self):
        """Length of each item embedding vector."""
        return self.current_state.shape[0]


def load_embeddings(path):
    """
    Loader stub for a precomputed embedding matrix saved on disk.

    Intended to return a ``[n_items, embed_dim]`` :obj:`numpy.ndarray` (e.g. the
    output of a CLIP image encoder). The concrete loading logic is left to be
    filled in for a specific dataset; by default we support ``.npy`` files.
    """
    return np.load(path)


def random_embeddings(n_items, embed_dim, seed=None):
    """
    Convenience generator of random unit-norm embeddings, for tests and demos
    when no real thumbnail embeddings are available.
    """
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_items, embed_dim))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms

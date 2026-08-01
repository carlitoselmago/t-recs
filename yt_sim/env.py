"""
Gymnasium environment wrapping the YouTube-like recommendation funnel.

A single external RL agent acts as the *viewer*. Each step it sees only:

* the currently playing video's thumbnail embedding and which current-video
  actions are available, and
* the slate of suggested videos (their thumbnail embeddings) and which
  suggested-video actions are available.

It never sees what any action does, the internal watch fraction, the viewer's
preference vector, or the masking state -- exactly what a real viewer observes
is the *next slate*, nothing more. The environment is reward-agnostic: ``reward``
is always ``0.0`` so the agent can compute its own intrinsic reward externally.

The observation is a ``Dict``:

* ``current_embedding`` : ``Box(embed_dim,)``
* ``slate_embeddings``  : ``Box(slate_size, embed_dim)``
* ``action_mask``       : ``MultiBinary(n_actions)`` over the flat ``Discrete``
  action space -- a 1 marks an available action. The mask's layout separates the
  two tiers (current-video vs suggested-video) but attaches no meaning to any id.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yt_sim.env requires gymnasium. Install it with `pip install gymnasium` "
        "or `pip install -e .[rl]`."
    ) from exc

from .actions import ActionSchema
from .action_effects import DefaultActionEffect
from .embeddings import EmbeddingItems
from .funnel import FunnelRecommender
from .masking import NotInterestedMask
from .watch import WatchModel


class YouTubeSimEnv(gym.Env):
    """
    A reward-agnostic, black-box recommendation environment.

    Parameters
    ----------
    item_embeddings : :obj:`numpy.ndarray` or :class:`~yt_sim.embeddings.EmbeddingItems`
        ``[n_items, embed_dim]`` thumbnail embeddings.
    slate_size : int, default 5
        Number of suggested videos shown alongside the current video.
    candidate_pool : int, default 50
        Stage-1 k-NN candidate pool size.
    max_steps : int, default 200
        Episode length after which ``truncated`` becomes True.
    like_enabled : bool, default False
        Whether the ``like`` current-video action is offered (default: gated no-op).
    n_neighbors, neighbor_suppression_steps, mask_neighbors :
        "Not interested" masking configuration (see
        :class:`~yt_sim.masking.NotInterestedMask`).
    ranker, watch_model, action_effect :
        Optional plugin instances; sensible defaults are used when omitted.
    belief_lr : float, default 0.5
        Preference drift rate for the funnel's belief updates.
    seed : int, optional
        Seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        item_embeddings,
        slate_size=5,
        candidate_pool=50,
        max_steps=200,
        like_enabled=False,
        n_neighbors=10,
        neighbor_suppression_steps=50,
        mask_neighbors=True,
        ranker=None,
        watch_model=None,
        action_effect=None,
        belief_lr=0.5,
        seed=None,
    ):
        super().__init__()
        if not isinstance(item_embeddings, EmbeddingItems):
            item_embeddings = EmbeddingItems(embeddings=item_embeddings, seed=seed)

        self.funnel = FunnelRecommender(
            item_embeddings,
            num_users=1,
            num_items_per_iter=slate_size,
            candidate_pool=candidate_pool,
            ranker=ranker,
            belief_lr=belief_lr,
            seed=seed,
        )
        self.mask = NotInterestedMask(
            self.funnel.normalized_item_embeddings,
            n_neighbors=n_neighbors,
            neighbor_suppression_steps=neighbor_suppression_steps,
            mask_neighbors=mask_neighbors,
        )
        self.funnel.masks[0] = self.mask

        self.schema = ActionSchema(slate_size, like_enabled=like_enabled)
        self.watch_model = watch_model if watch_model is not None else WatchModel(seed=seed)
        self.effect = action_effect if action_effect is not None else DefaultActionEffect()

        self.slate_size = slate_size
        self.max_steps = max_steps
        self.embed_dim = item_embeddings.embed_dim
        self.n_items = self.funnel.num_items

        self.rng = np.random.default_rng(seed)
        self.current_video = 0
        self.slate = np.zeros(slate_size, dtype=int)
        self.step_count = 0

        self.action_space = spaces.Discrete(self.schema.n)
        self.observation_space = spaces.Dict(
            {
                "current_embedding": spaces.Box(
                    -np.inf, np.inf, shape=(self.embed_dim,), dtype=np.float32
                ),
                "slate_embeddings": spaces.Box(
                    -np.inf, np.inf, shape=(slate_size, self.embed_dim), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(self.schema.n),
            }
        )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        self.mask.reset()
        self.funnel.popularity[:] = 0.0
        self.step_count = 0
        self.funnel.current_step = 0

        # latent viewer taste: a random unit vector (or caller-supplied)
        pref = options.get("preference")
        if pref is None:
            pref = self.rng.normal(size=self.embed_dim)
            pref /= np.linalg.norm(pref) or 1.0
        self.funnel.set_preference(pref)

        # starting video
        start = options.get("start_video")
        self.current_video = int(start) if start is not None else int(self.rng.integers(self.n_items))

        self.regenerate_slate()
        return self._observation(), {"step": self.step_count}

    def step(self, action):
        decoded = self.schema.decode(action)
        # All action semantics live in the effect plugin, not here.
        self.effect.apply(self, decoded)

        self.step_count += 1
        self.funnel.current_step = self.step_count

        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0  # reward-agnostic: the agent computes its own intrinsic reward
        # `info` deliberately carries no action-effect information (opacity).
        return self._observation(), reward, terminated, truncated, {"step": self.step_count}

    def render(self):
        return None

    # ------------------------------------------------------------------ #
    # Helpers used by the action-effect plugin
    # ------------------------------------------------------------------ #
    def transition_to(self, item_id):
        """Make ``item_id`` the currently playing video and refresh the slate."""
        self.current_video = int(item_id)
        self.regenerate_slate()

    def regenerate_slate(self):
        """Recompute the suggested-video slate for the current context."""
        slate = self.funnel.recommend_slate(
            user_idx=0, k=self.slate_size, exclude_ids=[self.current_video]
        )
        self.slate = self._fill_slate(slate)

    def _fill_slate(self, slate):
        """Guarantee exactly ``slate_size`` ids (pad if the funnel came up short)."""
        slate = np.asarray(slate, dtype=int)
        if slate.size >= self.slate_size:
            return slate[: self.slate_size]
        chosen = set(slate.tolist()) | {self.current_video}
        # fall back to items closest to the current preference, ignoring masking
        sim = self.funnel.normalized_item_embeddings @ self.funnel.preference()
        for item_id in np.argsort(-sim):
            if slate.size >= self.slate_size:
                break
            item_id = int(item_id)
            if item_id not in chosen:
                slate = np.append(slate, item_id)
                chosen.add(item_id)
        return slate

    # ------------------------------------------------------------------ #
    # Observation construction (agent-facing; strictly embeddings + masks)
    # ------------------------------------------------------------------ #
    def _action_mask(self):
        mask = np.zeros(self.schema.n, dtype=np.int8)
        mask[self.schema.current_action_ids()] = 1
        for slot in range(self.slate_size):
            mask[self.schema.slate_action_ids(slot)] = 1
        return mask

    def _observation(self):
        embeddings = self.funnel.item_embeddings
        return {
            "current_embedding": embeddings[self.current_video].astype(np.float32),
            "slate_embeddings": embeddings[self.slate].astype(np.float32),
            "action_mask": self._action_mask(),
        }

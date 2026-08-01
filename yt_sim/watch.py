"""
Watch simulation.

Models "watching" as a *continuous* engagement signal (fraction of the video
watched in ``[0, 1]``) rather than a binary consumed/not-consumed event. The
fraction is a probabilistic function of the similarity between the viewer's
current preference state and the currently playing video's embedding.

This is an **internal** variable: it is never returned to the agent. The
simulator uses it to decide how strongly the viewer's preference state should
drift toward the current video (see :mod:`yt_sim.action_effects`).
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class WatchModel:
    """
    Sigmoid-of-cosine-similarity watch model with optional noise.

    ``frac = clip(sigmoid(steepness * (cos_sim - midpoint)), 0, 1)`` perturbed by
    zero-mean Gaussian noise. A video that closely matches the viewer's
    preference is watched for a larger fraction; a poorly matched video is mostly
    skipped -- monotonic, cheap, and tunable.

    Parameters
    ----------
    steepness : float, default 6.0
        Slope of the sigmoid; larger means a sharper watch/skip transition.
    midpoint : float, default 0.0
        Cosine similarity at which the expected watch fraction is 0.5.
    noise : float, default 0.05
        Standard deviation of additive Gaussian noise on the fraction.
    full_watch_threshold : float, default 0.9
        Watch fractions at or above this are considered a "full watch".
    seed : int, optional
        Seed for the internal random generator.
    """

    def __init__(
        self,
        steepness=6.0,
        midpoint=0.0,
        noise=0.05,
        full_watch_threshold=0.9,
        seed=None,
    ):  # pylint: disable=too-many-arguments
        self.steepness = steepness
        self.midpoint = midpoint
        self.noise = noise
        self.full_watch_threshold = full_watch_threshold
        self.rng = np.random.default_rng(seed)

    def watch_fraction(self, cos_sim):
        """Return a stochastic watch fraction in ``[0, 1]`` for a similarity."""
        base = _sigmoid(self.steepness * (cos_sim - self.midpoint))
        if self.noise > 0:
            base = base + self.rng.normal(0.0, self.noise)
        return float(np.clip(base, 0.0, 1.0))

    def is_full_watch(self, frac):
        """Whether a watch fraction counts as watching the whole video."""
        return frac >= self.full_watch_threshold

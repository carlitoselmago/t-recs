"""
Pluggable action-effect models.

This module is the **only** place where the meaning of an action lives. The
environment's ``step`` function decodes the flat action id into a structured
``(tier, op, slot)`` and hands it to an :class:`ActionEffect`, which mutates the
simulator's internal state accordingly. Swapping in a different effect model
(different preference dynamics, different masking behaviour, ...) therefore
requires no change to the core loop -- and, critically, none of this reaches the
agent: observations never carry action labels or effect descriptions.

The effect operates on a small "simulator" interface (implemented by
:class:`yt_sim.env.YouTubeSimEnv`) exposing:

* ``funnel``        -- the :class:`~yt_sim.funnel.FunnelRecommender`
* ``watch_model``   -- the :class:`~yt_sim.watch.WatchModel`
* ``mask``          -- the :class:`~yt_sim.masking.NotInterestedMask`
* ``current_video`` -- id of the currently playing video
* ``slate``         -- array of suggested-video ids
* ``step_count``    -- current step
* ``transition_to(item_id)`` / ``regenerate_slate()`` helpers
"""
from . import actions


class ActionEffect:
    """Base class for pluggable action-effect models."""

    def apply(self, sim, decoded):
        """
        Apply ``decoded`` (a :class:`yt_sim.actions.DecodedAction`) to ``sim``.

        Returns an internal info dict (not exposed to the agent verbatim).
        Subclasses override this.
        """
        raise NotImplementedError


class DefaultActionEffect(ActionEffect):
    """
    Default two-tier effect model.

    Current-video actions
        * ``watch_full`` -- the viewer watches the current video; the internal
          watch fraction (a function of true preference-item similarity) scales
          how far preference drifts toward it, and its popularity increases.
        * ``skip`` -- minimal engagement; preference barely moves.
        * ``like`` -- **gated off by default** (no-op). The branch is kept so it
          is trivial to re-enable as a strong positive signal.

    Suggested-video actions
        * ``click`` -- navigate to that video: preference drifts toward it and it
          becomes the new currently playing video, producing a fresh slate.
        * ``not_interested`` -- suppress that video and its embedding neighbors
          from future candidate generation (see :mod:`yt_sim.masking`).

    Parameters
    ----------
    skip_drift_factor : float, default 0.1
        Fraction of the normal drift applied on a skip.
    click_drift : float, default 0.4
        Drift strength applied toward a clicked video.
    like_drift : float, default 0.9
        Drift strength applied on a like (only when likes are enabled).
    """

    def __init__(self, skip_drift_factor=0.1, click_drift=0.4, like_drift=0.9):
        self.skip_drift_factor = skip_drift_factor
        self.click_drift = click_drift
        self.like_drift = like_drift

    def apply(self, sim, decoded):
        if decoded.tier == actions.CURRENT:
            return self._apply_current(sim, decoded)
        return self._apply_slate(sim, decoded)

    # -- current-video branch ------------------------------------------------
    def _apply_current(self, sim, decoded):
        item_id = sim.current_video
        # internal watch fraction from true similarity (never exposed)
        frac = sim.watch_model.watch_fraction(sim.funnel.cosine_to_item(item_id))

        if decoded.op == actions.WATCH_FULL:
            sim.funnel.register_interaction(item_id)
            sim.funnel.drift_preference(item_id, strength=frac)
            sim.regenerate_slate()
        elif decoded.op == actions.SKIP:
            sim.funnel.drift_preference(item_id, strength=frac * self.skip_drift_factor)
            sim.regenerate_slate()
        elif decoded.op == actions.LIKE:
            # Gated no-op by default. Branch kept so re-enabling is a one-liner in
            # the schema (ActionSchema(like_enabled=True)).
            if sim.schema.like_enabled:
                sim.funnel.register_interaction(item_id)
                sim.funnel.drift_preference(item_id, strength=max(frac, self.like_drift))
                sim.regenerate_slate()
        return {"watch_fraction": frac}

    # -- suggested-video branch ---------------------------------------------
    def _apply_slate(self, sim, decoded):
        if decoded.slot >= sim.slate.size:
            # slot beyond the current slate (e.g. slate shrank); no effect
            return {}
        item_id = int(sim.slate[decoded.slot])

        if decoded.op == actions.CLICK:
            sim.funnel.register_interaction(item_id)
            sim.funnel.drift_preference(item_id, strength=self.click_drift)
            sim.transition_to(item_id)
        elif decoded.op == actions.NOT_INTERESTED:
            sim.mask.not_interested(item_id, sim.step_count)
            sim.regenerate_slate()
        return {"slate_item": item_id}

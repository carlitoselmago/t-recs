"""
Internal two-tier action schema for the YouTube-like simulator.

This module defines the *mechanical* mapping between the flat integer action
space exposed to a Gymnasium agent and the structured ``(tier, op, slot)``
actions the simulator understands. It deliberately contains **no description of
what any action does** -- the *effects* live in :mod:`yt_sim.action_effects`.

The only thing that ever leaves the simulator (in an observation) is a list of
*integer* action ids that are currently valid. Their meaning is intentionally
opaque to the agent: it sees which buttons exist, never what they do.
"""
from dataclasses import dataclass

# Two tiers of actions.
CURRENT = "current"  # actions on the currently playing video
SLATE = "slate"  # actions on a suggested video in the slate

# Current-video ops.
WATCH_FULL = "watch_full"
SKIP = "skip"
LIKE = "like"  # present in the schema but gated off (no-op) by default

# Suggested-video ops.
CLICK = "click"
NOT_INTERESTED = "not_interested"

# Fixed offsets for the current-video ops at the start of the flat action space.
_CURRENT_OPS = (WATCH_FULL, SKIP, LIKE)
_SLATE_OPS = (CLICK, NOT_INTERESTED)


@dataclass(frozen=True)
class DecodedAction:
    """A flat action id decoded into its structured meaning."""

    tier: str  # CURRENT or SLATE
    op: str  # one of the op constants above
    slot: int  # slate slot index, or -1 for current-video actions


class ActionSchema:
    """
    Maps between flat integer action ids and structured actions for a fixed
    slate size.

    Layout of the flat action space (``slate_size`` == ``S``)::

        0                -> (current, watch_full)
        1                -> (current, skip)
        2                -> (current, like)          # gated no-op by default
        3 + 2*j + 0      -> (slate, click, slot=j)
        3 + 2*j + 1      -> (slate, not_interested, slot=j)

    ``n`` == ``3 + 2 * S``.
    """

    def __init__(self, slate_size, like_enabled=False):
        if slate_size < 1:
            raise ValueError("slate_size must be at least 1")
        self.slate_size = slate_size
        self.like_enabled = like_enabled

    @property
    def n(self):
        """Total number of distinct flat action ids."""
        return len(_CURRENT_OPS) + len(_SLATE_OPS) * self.slate_size

    def decode(self, action_id):
        """Turn a flat integer action id into a :class:`DecodedAction`."""
        action_id = int(action_id)
        if action_id < 0 or action_id >= self.n:
            raise ValueError(f"action id {action_id} out of range [0, {self.n})")
        if action_id < len(_CURRENT_OPS):
            return DecodedAction(CURRENT, _CURRENT_OPS[action_id], -1)
        offset = action_id - len(_CURRENT_OPS)
        slot, op_idx = divmod(offset, len(_SLATE_OPS))
        return DecodedAction(SLATE, _SLATE_OPS[op_idx], slot)

    def current_action_ids(self):
        """Valid action ids for the currently playing video.

        ``like`` is only offered when explicitly enabled; the branch still
        exists in :mod:`yt_sim.action_effects` so it is trivial to re-enable.
        """
        ids = [0, 1]  # watch_full, skip
        if self.like_enabled:
            ids.append(2)  # like
        return ids

    def slate_action_ids(self, slot):
        """Valid action ids for suggested-video ``slot``."""
        base = len(_CURRENT_OPS) + len(_SLATE_OPS) * slot
        return [base, base + 1]  # click, not_interested

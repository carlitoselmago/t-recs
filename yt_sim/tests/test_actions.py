import pytest

from yt_sim import actions
from yt_sim.actions import ActionSchema


class TestActionSchema:
    def test_size(self):
        schema = ActionSchema(slate_size=5)
        assert schema.n == 3 + 2 * 5

    def test_decode_current(self):
        schema = ActionSchema(slate_size=3)
        assert schema.decode(0) == actions.DecodedAction(actions.CURRENT, actions.WATCH_FULL, -1)
        assert schema.decode(1) == actions.DecodedAction(actions.CURRENT, actions.SKIP, -1)
        assert schema.decode(2) == actions.DecodedAction(actions.CURRENT, actions.LIKE, -1)

    def test_decode_slate_roundtrip(self):
        schema = ActionSchema(slate_size=4)
        for slot in range(4):
            click, ni = schema.slate_action_ids(slot)
            assert schema.decode(click) == actions.DecodedAction(actions.SLATE, actions.CLICK, slot)
            assert schema.decode(ni) == actions.DecodedAction(
                actions.SLATE, actions.NOT_INTERESTED, slot
            )

    def test_like_gating(self):
        assert 2 not in ActionSchema(3, like_enabled=False).current_action_ids()
        assert 2 in ActionSchema(3, like_enabled=True).current_action_ids()

    def test_out_of_range(self):
        schema = ActionSchema(slate_size=2)
        with pytest.raises(ValueError):
            schema.decode(schema.n)
        with pytest.raises(ValueError):
            schema.decode(-1)

    def test_invalid_slate_size(self):
        with pytest.raises(ValueError):
            ActionSchema(slate_size=0)

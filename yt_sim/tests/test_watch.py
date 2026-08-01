import numpy as np

from yt_sim.watch import WatchModel


class TestWatchModel:
    def test_fraction_bounds(self):
        wm = WatchModel(seed=0)
        for cos in np.linspace(-1, 1, 25):
            frac = wm.watch_fraction(float(cos))
            assert 0.0 <= frac <= 1.0

    def test_monotonic_without_noise(self):
        wm = WatchModel(noise=0.0)
        assert wm.watch_fraction(0.9) > wm.watch_fraction(0.0) > wm.watch_fraction(-0.9)

    def test_full_watch_threshold(self):
        wm = WatchModel(noise=0.0, full_watch_threshold=0.9)
        assert wm.is_full_watch(0.95)
        assert not wm.is_full_watch(0.5)

    def test_midpoint_is_half(self):
        wm = WatchModel(noise=0.0, midpoint=0.2)
        assert abs(wm.watch_fraction(0.2) - 0.5) < 1e-9

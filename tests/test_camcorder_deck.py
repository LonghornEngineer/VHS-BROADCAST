"""Camcorder and VCR deck."""
import unittest

import cv2
import numpy as np

from vhs_broadcast.core.automation import Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.camera import CamcorderEffect
from vhs_broadcast.effects.deck import BLUE, VCRDeckEffect


def make(cls, timeline=None, **values):
    params = ParameterSet()
    effect = cls(params)
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"{effect.id}.{key}").set_base(value)
    if timeline is not None:
        params.reset_motion(timeline)
    return params, effect


def grid(h=120, w=160, level=180):
    img = np.full((h, w, 3), level, np.uint8)
    img[::16] = 40; img[:, ::16] = 40
    return img


class CamcorderTests(unittest.TestCase):
    def test_defaults_leave_the_picture_alone(self):
        frame = grid()
        _, e = make(CamcorderEffect)
        np.testing.assert_array_equal(e.apply(frame, .5), frame)

    def test_fisheye_bulges_the_centre_and_pins_corners(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(frame, (70, 50), (90, 70), (255, 255, 255), -1)       # a small centre square
        _, e = make(CamcorderEffect, fisheye=1)
        out = e.apply(frame, 0)
        self.assertGreater((out[..., 0] > 128).sum(), (frame[..., 0] > 128).sum() * 1.5)
        corner = grid()
        np.testing.assert_allclose(make(CamcorderEffect, fisheye=1)[1].apply(corner, 0)[0, 0], corner[0, 0], atol=30)

    def test_auto_exposure_pumps_past_its_target_after_a_cut(self):
        _, e = make(CamcorderEffect, exposure=1, ae_speed=2)
        bright, dark = np.full((60, 80, 3), 200, np.uint8), np.full((60, 80, 3), 40, np.uint8)
        for n in range(60):
            e.apply(bright, n / 30)
        levels = [float(e.apply(dark, (60 + n) / 30).mean()) for n in range(90)]
        self.assertLess(levels[0], 60)                                      # iris still closed at the cut
        settled = levels[-1]
        self.assertGreater(max(levels), settled + 3)                        # overshoot = pumping
        self.assertAlmostEqual(settled, 115, delta=20)                      # ends near mid grey

    def test_white_balance_cast_lingers_then_corrects(self):
        _, e = make(CamcorderEffect, wb=1)
        warm = np.zeros((60, 80, 3), np.uint8); warm[:] = (90, 130, 200)   # BGR orange cast
        first = e.apply(warm, 0).reshape(-1, 3).mean(0)
        for n in range(1, 150):
            later = e.apply(warm, n / 30).reshape(-1, 3).mean(0)
        self.assertGreater(np.ptp(first), np.ptp(later) + 20)              # drifts toward neutral

    def test_shake_and_vignette_are_repeatable(self):
        frame = grid()
        _, a = make(CamcorderEffect, shake=1, vignette=1)
        _, b = make(CamcorderEffect, shake=1, vignette=1)
        np.testing.assert_array_equal(a.apply(frame, 1.3), b.apply(frame, 1.3))
        self.assertFalse(np.array_equal(a.apply(frame, 1.3), a.apply(frame, 1.6)))
        out = a.apply(frame, 1.3)
        self.assertLess(out[:8, :8].mean(), out[56:64, 76:84].mean())


def deck_timeline(times, modes):
    timeline = Timeline.from_dict({"tracks": {"deck.mode": {"t": times, "v": modes, "interpolation": "hold"}}})
    return timeline


class DeckTests(unittest.TestCase):
    def test_changed_at_reads_the_mode_keys(self):
        params, _ = make(VCRDeckEffect, deck_timeline([0, 1, 2, 3], [4, 0, 0, 1]))
        self.assertIsNone(params.changed_at("deck.mode", .5))
        self.assertEqual(params.changed_at("deck.mode", 2.5), 1)            # repeated key isn't a change
        self.assertEqual(params.changed_at("deck.mode", 3.2), 3)

    def test_stop_is_blue_and_play_after_stop_loads_the_tape(self):
        timeline = deck_timeline([0, 1], [4, 0])
        params, e = make(VCRDeckEffect, timeline)
        frame = grid()
        timeline.apply(.5, params)
        self.assertTrue((e.apply(frame, .5) == BLUE).all())
        timeline.apply(1.1, params)
        self.assertTrue((e.apply(frame, 1.1) == BLUE).all())                # still threading
        timeline.apply(1.25, params)
        snow = e.apply(frame, 1.25)
        self.assertGreater(np.abs(snow.astype(int) - frame).mean(), 30)     # drum up, no tracking yet
        timeline.apply(3.0, params)
        np.testing.assert_array_equal(e.apply(frame, 3.0), frame)           # settled: untouched
        # A scrub straight into the settle matches playing up to it.
        params2, fresh = make(VCRDeckEffect, timeline)
        timeline.apply(1.9, params2); timeline.apply(1.9, params)
        np.testing.assert_array_equal(fresh.apply(frame, 1.9), e.apply(frame, 1.9))

    def test_pause_holds_the_frame_and_play_resumes_at_once(self):
        timeline = deck_timeline([0, 1, 2], [0, 1, 0])
        params, e = make(VCRDeckEffect, timeline, noise=0, jitter=0)
        a, b = grid(level=100), grid(level=220)
        for t, img in ((.9, a), (1.0, a), (1.5, b)):
            timeline.apply(t, params)
            out = e.apply(img, t)
        np.testing.assert_array_equal(out, a)                               # still showing the paused frame
        timeline.apply(2.05, params)
        np.testing.assert_array_equal(e.apply(b, 2.05), b)

    def test_search_bars_move_with_direction(self):
        timeline = deck_timeline([0], [2])
        params, e = make(VCRDeckEffect, timeline, speed=3, bar_speed=1)     # one bar
        frame = np.full((180, 120, 3), 128, np.uint8)
        def bar_row(t):
            timeline.apply(t, params)
            out = e.apply(frame, t).astype(int)
            dev = np.abs(out - 128).mean(axis=(1, 2))
            return float((dev * np.arange(len(dev))).sum() / dev.sum())   # bar centre
        self.assertGreater(bar_row(.3), bar_row(.2))                        # fast forward runs down

    def test_rewind_shows_more_bars_than_fast_forward(self):
        # The head crosses speed-1 tracks per field in FF and speed+1 in REW.
        def bars(mode):
            timeline = deck_timeline([0], [mode])
            params, e = make(VCRDeckEffect, timeline, speed=5, bar_speed=0)
            frame = np.full((240, 160, 3), 128, np.uint8)
            timeline.apply(.5, params)
            dev = np.abs(e.apply(frame, .5).astype(int) - 128).mean(axis=(1, 2)) > 25
            return int(np.count_nonzero(np.diff(dev.astype(int)) == 1) + dev[0])
        self.assertEqual(bars(2), 2)
        self.assertEqual(bars(3), 3)


if __name__ == "__main__":
    unittest.main()

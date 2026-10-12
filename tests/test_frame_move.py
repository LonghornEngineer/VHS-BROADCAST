import unittest
import numpy as np

from vhs_broadcast.core.automation import Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import FrameMoveEffect


class FrameMoveTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effect = FrameMoveEffect(self.params)
        self.effect.enabled_param().set_base(1)
        self.frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    def set(self, name, value):
        self.params.get("frame_move." + name).set_base(value)

    def test_default_and_disabled_identity(self):
        np.testing.assert_array_equal(self.effect.apply(self.frame, 0), self.frame)
        self.set("zoom", 2)
        self.effect.enabled_param().set_base(0)
        np.testing.assert_array_equal(self.effect.apply(self.frame, 4), self.frame)

    def test_pan_wraps_in_both_directions_and_diagonally(self):
        original = self.frame.copy()
        self.set("x", .25)
        self.set("y", -.375)
        actual = self.effect.apply(self.frame, 0)
        expected = np.roll(np.roll(self.frame, 2, axis=1), -3, axis=0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.frame, original)
        self.set("x", 1)
        self.set("y", -1)
        np.testing.assert_array_equal(self.effect.apply(self.frame, 0), self.frame)

    def test_zoom_out_repeats_source_pixels_without_borders(self):
        frame = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
        self.set("zoom", .5)
        indices = (2 * np.arange(9) - 4) % 9
        expected = frame[indices[:, None], indices[None, :]]
        np.testing.assert_array_equal(self.effect.apply(frame, 0), expected)

    def test_rotated_zoom_fills_every_pixel(self):
        frame = np.full((32, 48, 3), [50, 100, 200], np.uint8)
        self.set("angle", 37)
        self.set("zoom", .3)
        self.set("x", .77)
        self.set("y", -.43)
        np.testing.assert_array_equal(self.effect.apply(frame, 3), frame)

    def test_continuous_motion_stops_in_place_and_reverses(self):
        self.set("xspeed", .25)
        self.set("yspeed", .125)
        moved = self.effect.apply(self.frame, 1)
        np.testing.assert_array_equal(moved, np.roll(np.roll(self.frame, 2, axis=1), 1, axis=0))
        self.set("xspeed", 0)
        self.set("yspeed", 0)
        np.testing.assert_array_equal(self.effect.apply(self.frame, 3), moved)
        self.set("xspeed", -.25)
        self.set("yspeed", -.125)
        np.testing.assert_array_equal(self.effect.apply(self.frame, 4), self.frame)

    def test_manual_pan_keys_and_factory_zoom_reset(self):
        timeline = Timeline()
        p = self.params.get("frame_move.x")
        timeline.set_key(0, p)
        p.set_base(.5)
        timeline.set_key(2, p)
        timeline.apply(1, self.params)
        np.testing.assert_array_equal(self.effect.apply(self.frame, 1), np.roll(self.frame, 2, axis=1))
        zoom = self.params.get("frame_move.zoom")
        zoom.set_base(3)
        zoom.reset()
        self.assertEqual(zoom.value, 1)


if __name__ == "__main__":
    unittest.main()

"""Aspect, TV Circuit Bend, Light Flicker, Bloom pixel glow and pixel-perfect Downscale."""
import unittest

import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import default_effects


def make(eid, **values):
    params = ParameterSet()
    effect = {e.id: e for e in default_effects(params)}[eid]
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"{eid}.{key}").set_base(value)
    return effect


def scene(h=120, w=160):
    rng = np.random.default_rng(0)
    img = rng.integers(40, 200, (h, w, 3), dtype=np.uint8)
    img[40:60, 60:100] = 250                                     # a light
    return img


class AspectTests(unittest.TestCase):
    def test_crop_letterboxes_and_amount_zero_is_untouched(self):
        img = scene(120, 160)                                     # 4:3
        out = make("aspect", target=1).apply(img, 0)              # 16:9 crop -> 90 lines
        self.assertTrue((out[:14] == 0).all() and (out[-14:] == 0).all())
        self.assertGreater(out[20:100].mean(), 60)
        np.testing.assert_array_equal(make("aspect", target=1, amount=0).apply(img, 0), img)

    def test_amount_slides_the_bars_in(self):
        img = scene(120, 160)
        bars = [int((make("aspect", target=2, amount=a).apply(img, 0).max(axis=(1, 2)) == 0).sum())
                for a in (.25, .5, 1)]
        self.assertTrue(bars[0] < bars[1] < bars[2])


class CircuitBendTests(unittest.TestCase):
    def test_defaults_leave_the_picture_alone(self):
        img = scene()
        np.testing.assert_array_equal(make("circuit_bend").apply(img, 1), img)

    def test_vertical_collapse_squeezes_to_a_bright_line(self):
        img = scene()
        out = make("circuit_bend", vcollapse=.95).apply(img, 1)
        lit = out.max(axis=(1, 2)) > 0
        self.assertLess(lit.mean(), .1)
        self.assertGreater(out[lit].mean(), img.mean())

    def test_guns_and_video_into_the_yoke(self):
        img = scene()
        self.assertEqual(int(make("circuit_bend", guns=1).apply(img, 1)[..., 2].max()), 0)      # red out
        self.assertGreater(int(make("circuit_bend", guns=6).apply(img, 1)[..., 0].min()), 190)  # blue stuck
        bent = make("circuit_bend", bend_h=1).apply(img, 1)
        self.assertGreater(np.abs(bent.astype(int) - img).mean(), 10)


class FlickerTests(unittest.TestCase):
    def test_ccd_pumps_the_whole_frame_and_cmos_bands_it(self):
        img = np.full((96, 64, 3), 150, np.uint8)
        ccd = make("flicker", shutter=2000, rolling=0, amount=1)
        rows = [ccd.apply(img, t)[:, :, 1].mean(axis=1) for t in (1.0, 1.003)]
        self.assertLess(rows[0].std(), 1)                         # one level per frame
        self.assertGreater(abs(rows[0].mean() - rows[1].mean()), 10)
        cmos = make("flicker", shutter=2000, rolling=1, amount=1).apply(img, 1.0)
        self.assertGreater(cmos[:, :, 1].mean(axis=1).std(), 10)  # bands down the frame

    def test_shutter_over_whole_mains_cycles_hides_it(self):
        img = np.full((96, 64, 3), 150, np.uint8)
        out = make("flicker", shutter=60, rolling=1, amount=1).apply(img, 1.0)
        self.assertLess(np.abs(out.astype(int) - 150).max(), 3)


class PixelTests(unittest.TestCase):
    def test_snap_makes_every_pixel_the_same_size(self):
        img = scene(120, 160)
        out = make("downsample", size=3, width=37, grid=1).apply(img, 0)
        k = round(160 / 37)
        blocks = out[:120 // k * k, :160 // k * k].reshape(120 // k, k, 160 // k, k, 3)
        self.assertTrue((blocks == blocks[:, :1, :, :1]).all())

    def test_bars_scale_by_a_whole_number(self):
        img = scene(120, 160)
        out = make("downsample", size=3, width=37, grid=2).apply(img, 0)
        cols = np.flatnonzero(out.max(axis=(0, 2)) > 0)
        self.assertEqual((cols[-1] - cols[0] + 1) % 37, 0)

    def test_pixel_glow_lights_whole_grid_cells_around_the_light(self):
        img = scene(120, 160)
        out = make("bloom", style=1, pixel=8, spread=1, amount=.8, threshold=.9).apply(img, 0)
        changed = (out != img).any(axis=2)
        self.assertTrue(changed[40:60, 60:100].any())
        self.assertFalse(changed[:24].any())                      # far from the light: untouched
        self.assertTrue((out.astype(int) >= img).all())           # overlay with white only brightens


if __name__ == "__main__":
    unittest.main()


class OsdEdgeTests(unittest.TestCase):
    def test_black_edge_outlines_the_text_and_defaults_off(self):
        grey = np.full((120, 320, 3), 200, np.uint8)
        plain = make("text")
        off = plain.apply(grey, 0)
        self.assertEqual(plain.params.get("text.edge").value, 0)
        edged = make("text", edge=1).apply(grey, 0)
        # The edge adds black around the letters; without it the darkest text pixels stay light.
        self.assertLess(int(edged.min()), 40)
        self.assertGreater((edged < 40).sum(), (off < 40).sum() + 50)


class TrackingRollTests(unittest.TestCase):
    def roll(self, t, **kw):
        from vhs_broadcast.core.automation import Timeline
        effect = make("tracking", **kw)
        return effect, effect._roll_at(t)

    def test_lost_sync_rolls_and_good_tracking_holds(self):
        _, good = self.roll(5, error=0, speed=0)
        self.assertEqual(good, 0)
        _, lost = self.roll(5, error=1, speed=0)              # head fully on the wrong track
        self.assertGreater(lost, .1)
        _, off = self.roll(5, error=1, speed=0, vsync=0)       # V-sync loss switched off
        self.assertEqual(off, 0)

    def test_roll_is_the_same_played_or_jumped_to_and_pulls_back_in(self):
        from vhs_broadcast.core.automation import Timeline
        timeline = Timeline.from_dict({"tracks": {"tracking.error": {
            "t": [0, 2, 3], "v": [1, 1, 0], "interpolation": "hold"}}})
        played = make("tracking", speed=0)
        played.params.reset_motion(timeline)
        path = [played._roll_at(k / 30) for k in range(0, 150)]
        jumped = make("tracking", speed=0)
        jumped.params.reset_motion(timeline)
        self.assertEqual(jumped._roll_at(149 / 30), path[-1])
        self.assertGreater(path[60], 0)                        # rolling while tracking is lost
        self.assertEqual(path[-1], 0)                          # locked again after recovering

    def test_rolled_frame_shows_the_blanking_bar(self):
        img = np.full((120, 160, 3), 150, np.uint8)
        effect = make("tracking", error=1, speed=0)
        effect._roll_at(2)
        out = effect.apply(img, 2)
        self.assertTrue((out.max(axis=(1, 2)) <= 10).any())

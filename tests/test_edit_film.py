"""LINEAR EDIT / DUBS and FILMED OFF TV."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vhs_broadcast.core.automation import Timeline, Track
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.linear_edit import bars_frame, load_edl, timecode_seconds
from vhs_broadcast.effects.tvcamera import scan_exposure
from vhs_broadcast.effects.vhs import default_effects


def model():
    params = ParameterSet()
    effects = {e.id: e for e in default_effects(params)}
    params.reset_motion(Timeline())
    return params, effects


class LinearEditTests(unittest.TestCase):
    def setUp(self):
        self.params, effects = model()
        self.edit = effects["linear_edit"]
        self.edit.enabled_param().set_base(1)
        for name, value in dict(generations=0, gen_spread=0).items():
            self.params.get(f"linear_edit.{name}").set_base(value)
        self.frame = np.full((48, 64, 3), (40, 160, 220), np.uint8)

    def test_no_cuts_leaves_the_picture_alone(self):
        np.testing.assert_array_equal(self.edit.apply(self.frame, .5), self.frame)

    def test_cut_list_flashes_then_rolls_then_settles(self):
        self.edit.restore_settings({"cuts": [1.0]})
        flash = self.edit.apply(self.frame, 1.0)
        self.assertEqual(int(flash.max()), 0)                      # one black frame
        rolling = self.edit.apply(self.frame, 1.0 + 2 / 29.97)
        self.assertFalse(np.array_equal(rolling, self.frame))
        settled = self.edit.apply(self.frame, 1.6)
        np.testing.assert_array_equal(settled, self.frame)

    def test_keyed_edit_points_are_cuts(self):
        timeline = Timeline()
        track = Track(interpolation="hold")
        for t, value in ((0, 0), (2.0, 1), (3.0, 1), (4.0, 2)):
            track.add(t, value)
        timeline.tracks["linear_edit.edit"] = track
        self.params.reset_motion(timeline)
        self.assertEqual(self.edit.edit_points(10), [2.0, 4.0])
        self.assertEqual(int(self.edit.apply(self.frame, 4.0).max()), 0)

    def test_detected_cuts(self):
        self.params.get("linear_edit.detect").set_base(.8)
        other = np.full_like(self.frame, (200, 20, 30))
        for i in range(5):
            self.edit.apply(self.frame, i / 30)
        self.edit.apply(other, 5 / 30)
        self.assertEqual(self.edit.detected, [5 / 30])

    def test_generations_soften_and_desaturate(self):
        sharp = np.zeros((48, 64, 3), np.uint8)
        sharp[:, ::4] = (0, 0, 255)
        self.params.get("linear_edit.generations").set_base(4)
        out = self.edit.apply(sharp, .1).astype(int)
        self.assertLess(np.abs(np.diff(out[..., 2], axis=1)).mean(), np.abs(np.diff(sharp[..., 2].astype(int), axis=1)).mean())

    def test_bars_head_and_audio(self):
        self.params.get("linear_edit.bars").set_base(2)
        out = self.edit.apply(self.frame, .5)
        np.testing.assert_array_equal(out, bars_frame(48, 64))
        self.edit.restore_settings({"cuts": [3.0]})
        sr = 8000
        audio = np.zeros((sr * 4, 2), np.float32)
        result = self.edit.process_audio(audio, sr, 4.0)
        self.assertGreater(np.abs(result[:sr]).max(), .05)          # 1 kHz tone under the bars
        self.assertGreater(np.abs(result[3 * sr:3 * sr + 100]).max(), .01)   # click at the cut
        self.assertLess(np.abs(result[int(2.5 * sr):3 * sr]).max(), 1e-6)

    def test_cut_list_validation(self):
        with self.assertRaises(ValueError):
            self.edit.restore_settings({"cuts": ["1.0"]})

    def test_edl_record_in_times(self):
        edl = ("TITLE: HACKER BOI\nFCM: NON-DROP FRAME\n\n"
               "001  AX       V     C        00:00:10:00 00:00:12:00 01:00:00:00 01:00:02:00\n"
               "002  AX       V     C        00:00:30:00 00:00:31:15 01:00:02:00 01:00:03:15\n"
               "003  AX       V     C        00:01:00:00 00:01:02:00 01:00:03:15 01:00:05:15\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cut.edl"
            path.write_text(edl)
            cuts = load_edl(str(path), 30)
        self.assertEqual(len(cuts), 2)
        self.assertAlmostEqual(cuts[0], 2.0)
        self.assertAlmostEqual(cuts[1], 3.5)
        self.assertAlmostEqual(timecode_seconds("00:01:00;02", 30000 / 1001), 1800 / (30000 / 1001), places=3)


class FilmedTVTests(unittest.TestCase):
    def test_matched_shutter_has_no_bar_and_a_film_shutter_does(self):
        flat = scan_exposure(240, 0, 1 / 30, 59.94, 1.5)
        self.assertLess(np.abs(flat - 1).max(), .05)
        film = scan_exposure(240, 0, 1 / 48, 59.94, 1.5)
        self.assertGreater(film[:, 1].max() - film[:, 1].min(), .5)
        # The bar sits somewhere else on the next film frame.
        later = scan_exposure(240, 1 / 24, 1 / 48, 59.94, 1.5)
        self.assertNotEqual(int(np.argmax(film[:, 1])), int(np.argmax(later[:, 1])))

    def test_fast_shutter_catches_only_part_of_the_scan(self):
        light = scan_exposure(240, 0, 1 / 500, 59.94, 1.5)[:, 1]
        self.assertGreater((light < .2).mean(), .6)

    def test_camera_effect_keeps_size_and_is_repeatable(self):
        params, effects = model()
        cam = effects["filmed_tv"]
        cam.enabled_param().set_base(1)
        frame = np.random.default_rng(3).integers(0, 255, (96, 128, 3), np.uint8)
        out = cam.apply(frame, .25)
        self.assertEqual(out.shape, frame.shape)
        np.testing.assert_array_equal(cam.apply(frame, .25), out)


if __name__ == "__main__":
    unittest.main()

"""CRT GEOMETRY / MASK beam model: electron beams on a phosphor mask at display scale."""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.pipeline import render_to_file
from vhs_broadcast.effects.crt_tube import GRILLE, SLOT, TRIADS, build_mask, draw_beams
from vhs_broadcast.effects.vhs import default_effects

FLAT = dict(curve=0, corner=0, halation=0, converge=0, soft=0)


def linear(img, gamma=2.2):
    return (img.astype(np.float64) / 255) ** gamma


class CRTBeamTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effects = {e.id: e for e in default_effects(self.params)}
        self.tube = self.effects["crt_geometry"]
        self.tube.enabled_param().set_base(1)

    def set(self, **values):
        for name, value in values.items():
            effect, _, short = name.rpartition("__")
            self.params.get(f"{effect or 'crt_geometry'}.{short}").set_base(value)

    def test_new_tube_defaults_to_beam_and_draws_at_display_scale(self):
        self.assertEqual(self.params.get("crt_geometry.model").value, 0)
        self.set(scale=3)
        out = self.tube.apply(np.full((40, 60, 3), 128, np.uint8), 0)
        self.assertEqual(out.shape, (120, 180, 3))

    def test_beam_keeps_a_flat_field_level_in_linear_light(self):
        self.set(scale=4, **FLAT)
        for mask in (TRIADS, GRILLE, SLOT):
            self.set(mask=mask)
            for level in (40, 128):
                out = self.tube.apply(np.full((60, 80, 3), level, np.uint8), 0)
                self.assertAlmostEqual(linear(out).mean(), linear(np.uint8(level)), delta=.01)

    def test_dark_lines_show_gaps_and_bright_lines_close_them(self):
        self.set(scale=5, mask=GRILLE, depth=0, gap=.8, fill=.8, **FLAT)
        def modulation(level):
            column = self.tube.apply(np.full((40, 40, 3), level, np.uint8), 0)[50:60, 100, 1].astype(float)
            return 1 - column.min() / column.max()
        dark, bright = modulation(30), modulation(250)
        self.assertGreater(dark, .3)
        self.assertLess(bright, dark / 3)

    def test_beam_energy_is_spread_not_lost(self):
        lin = np.zeros((8, 4, 3), np.float32)
        lin[4] = .5
        light = draw_beams(lin, 40, .2, .45)
        self.assertAlmostEqual(float(light[:, 0, 1].mean() * 8), .5, places=3)

    def test_mask_is_not_tied_to_source_pixels(self):
        # One source pixel covers several phosphors of every colour.
        self.set(scale=6, triad=.8, depth=1, mask=TRIADS, **FLAT)
        img = np.zeros((20, 20, 3), np.uint8)
        img[10, 10] = 255
        out = self.tube.apply(img, 0).astype(int)
        block = out[58:68, 58:68]
        for channel in range(3):
            self.assertGreater(block[..., channel].max() - block[..., channel].min(), 40)

    def test_masks_cover_one_third_per_colour(self):
        for kind in (TRIADS, GRILLE, SLOT):
            mask = build_mask(90, 120, kind, 9.0)
            share = mask.reshape(-1, 3).mean(0)
            self.assertTrue(np.all(share > .1) and np.all(share < .4), (kind, share))
            np.testing.assert_allclose(share, share.mean(), rtol=.15)

    def test_interlaced_fields_alternate_and_decay(self):
        self.set(scale=4, raster=1, fields=60, field_decay=0, mask=GRILLE, depth=0, gap=1, fill=0, **FLAT)
        img = np.full((20, 20, 3), 128, np.uint8)
        a = self.tube.apply(img, 0).astype(int)[:, 40, 1]
        b = self.tube.apply(img, 1 / 60).astype(int)[:, 40, 1]
        self.assertGreater(np.abs(a - b).max(), 40)            # different lines lit
        self.set(field_decay=.8)
        self.tube.reset_state()
        self.tube.apply(img, 0)
        held = self.tube.apply(img, 1 / 60).astype(int)[:, 40, 1]
        self.assertGreater(held.min(), b.min())                 # previous field still glows

    def test_scanlines_and_bloom_defer_to_the_beam_tube(self):
        frame = np.full((32, 32, 3), 200, np.uint8)
        scan, bloom = self.effects["scanlines"], self.effects["bloom"]
        scan.enabled_param().set_base(1)
        bloom.enabled_param().set_base(1)
        self.set(scanlines__strength=1, scanlines__vignette=0, bloom__amount=1, bloom__threshold=.1)
        np.testing.assert_array_equal(scan.apply(frame, 0), frame)
        np.testing.assert_array_equal(bloom.apply(frame, 0), frame)
        self.set(model=1)                                       # legacy overlay: filters work alone
        self.assertFalse(np.array_equal(scan.apply(frame, 0), frame))

    def test_tube_draws_bloom_from_phosphor_light(self):
        img = np.zeros((40, 40, 3), np.uint8)
        img[18:22, 18:22] = 255
        self.set(scale=3, **FLAT)
        plain = self.tube.apply(img, 0)
        self.effects["bloom"].enabled_param().set_base(1)
        self.set(bloom__amount=1)
        glowing = self.tube.apply(img, 0)
        self.assertGreater(int(glowing[45:75, 30].sum()), int(plain[45:75, 30].sum()))

    def test_live_preview_caps_display_size_but_renders_do_not(self):
        self.set(scale=8)
        img = np.full((240, 320, 3), 100, np.uint8)
        self.tube.preview = True
        self.assertLessEqual(self.tube.apply(img, 0).shape[0], self.tube.PREVIEW_MAX_ROWS)
        self.tube.preview = False
        self.assertEqual(self.tube.apply(img, 0).shape[0], 1920)

    def test_legacy_overlay_keeps_frame_size(self):
        self.set(model=1)
        out = self.tube.apply(np.full((48, 64, 3), 180, np.uint8), 0)
        self.assertEqual(out.shape, (48, 64, 3))


class CRTBeamTakeTests(unittest.TestCase):
    def load(self, data):
        params = ParameterSet()
        effects = default_effects(params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            Session(params, Timeline(), effects).load(str(path))
        return params

    def test_takes_saved_before_the_beam_model_keep_the_overlay(self):
        old = self.load({"version": 2, "parameters": {"crt_geometry.enabled": {"value": 1}}})
        self.assertEqual(old.get("crt_geometry.model").value, 1)
        new = self.load({"version": Session.VERSION, "parameters": {"crt_geometry.enabled": {"value": 1}}})
        self.assertEqual(new.get("crt_geometry.model").value, 0)
        chosen = self.load({"version": 2, "parameters": {"crt_geometry.model": {"value": 0}}})
        self.assertEqual(chosen.get("crt_geometry.model").value, 0)

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_render_writes_the_display_size(self, _probe):
        params = ParameterSet()
        effects = default_effects(params)
        tube = next(e for e in effects if e.id == "crt_geometry")
        tube.enabled_param().set_base(1)
        params.get("crt_geometry.scale").set_base(2)
        tube.preview = True                                     # the live copy; export ignores it
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.avi")
            output = str(Path(directory) / "out.mp4")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            for i in range(3):
                writer.write(np.full((48, 64, 3), 60 + 40 * i, np.uint8))
            writer.release()
            render_to_file(source, output, params, Timeline(), [tube], with_audio=False)
            capture = cv2.VideoCapture(output)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 128)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 96)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()


if __name__ == "__main__":
    unittest.main()

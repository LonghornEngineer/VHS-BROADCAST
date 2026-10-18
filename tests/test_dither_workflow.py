import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.pipeline import render_to_file
from vhs_broadcast.effects.advanced import PaletteDitherEffect

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class DitherWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effect = PaletteDitherEffect(self.params)
        self.effect.enabled_param().set_base(1)
        self.image = np.random.default_rng(42).integers(0, 256, (27, 41, 3), dtype=np.uint8)

    def set(self, **values):
        for name, value in values.items():
            self.params.get("palette." + name).set_base(value)

    def test_every_pattern_palette_scale_stays_in_palette_and_repeatable(self):
        saved = self.image.copy()
        for palette in range(12):
            colors = self.effect.colors({"palette": palette, "colors": 8}).clip(0, 255).astype(np.uint8)
            valid = {tuple(c) for c in colors}
            for mode in range(8):
                for scale in (1, 16):
                    with self.subTest(palette=palette, mode=mode, scale=scale):
                        self.set(palette=palette, mode=mode, scale=scale)
                        result = self.effect.apply(self.image, .4)
                        self.assertEqual(result.shape, self.image.shape)
                        self.assertTrue(all(tuple(c) in valid for c in result.reshape(-1, 3)))
                        np.testing.assert_array_equal(result, self.effect.apply(self.image, .4))
        np.testing.assert_array_equal(self.image, saved)

    def test_original_modes_preserve_existing_look(self):
        palette = self.effect.colors({"palette": 4, "colors": 8}).astype(np.uint8).astype(np.float32)
        h, w = self.image.shape[:2]
        for mode in range(3):
            self.set(mode=mode)
            if mode == 1:
                noise = self.effect.BAYER[np.arange(h)[:, None] % 4, np.arange(w)[None, :] % 4]
            elif mode == 2:
                noise = np.random.default_rng(12).random((h + 1, w + 1), dtype=np.float32)[:h, :w] - .5
            else:
                noise = np.zeros((h, w), np.float32)
            pixels = self.image.astype(np.float32) + noise[..., None] * (.65 * 255 / np.sqrt(8))
            distance = ((pixels[..., None, :] - palette) ** 2).sum(-1)
            expected = palette[distance.argmin(-1)].astype(np.uint8)
            np.testing.assert_array_equal(self.effect.apply(self.image, .4), expected)

    def test_noise_freeze_seed_animation_and_ordered_stability(self):
        self.set(mode=2, animate=0)
        first = self.effect.apply(self.image, 0)
        np.testing.assert_array_equal(first, self.effect.apply(self.image, 20))
        self.set(seed=17)
        self.assertFalse(np.array_equal(first, self.effect.apply(self.image, 0)))
        self.set(animate=1)
        self.assertFalse(np.array_equal(self.effect.apply(self.image, 0), self.effect.apply(self.image, 1)))
        for mode in (1, 3, 4, 5, 6, 7):
            self.set(mode=mode)
            np.testing.assert_array_equal(self.effect.apply(self.image, 0), self.effect.apply(self.image, 10))

    def test_purple_shimmer_animates_cells_and_gap_darkens_grid(self):
        self.set(palette=8, colors=5, mode=3, scale=4)
        still = self.effect.apply(self.image, 0)
        np.testing.assert_array_equal(still, self.effect.apply(self.image, 3))
        self.set(shimmer=.4, shimmer_speed=2)
        first = self.effect.apply(self.image, 0)
        np.testing.assert_array_equal(first, self.effect.apply(self.image, 0))
        self.assertFalse(np.array_equal(first, self.effect.apply(self.image, .3)))
        self.set(shimmer_speed=0)
        np.testing.assert_array_equal(self.effect.apply(self.image, 0), self.effect.apply(self.image, 5))
        self.set(shimmer=0, gap=1)
        gapped = self.effect.apply(self.image, 0)
        self.assertTrue((gapped[3::4] == 0).all() and (gapped[:, 3::4] == 0).all())
        np.testing.assert_array_equal(gapped[:3, :3], still[:3, :3])

    def test_luma_matching_blocks_grade_and_multi_hue_palettes(self):
        self.set(palette=10, colors=32, mode=0, space=1)
        field = self.effect.colors({"palette": 10, "colors": 32}).astype(np.uint8)
        self.assertEqual(len(field), 20)
        self.assertEqual(len(self.effect.colors({"palette": 4, "colors": 32})), 16)
        for colour in field[[0, 7, 19]]:
            flat = np.broadcast_to(colour, self.image.shape).copy()
            np.testing.assert_array_equal(self.effect.apply(flat, 0), flat)
        self.set(mode=3, scale=4, blocks=1)
        plain = self.effect.apply(self.image, 0)
        valid = {tuple(c) for c in field}
        self.assertTrue(all(tuple(c) in valid for c in plain.reshape(-1, 3)))
        self.set(mode=0)
        blocky = self.effect.apply(self.image, 0)
        for y in range(0, 24, 4):
            for x in range(0, 40, 4):
                self.assertEqual(len(np.unique(blocky[y:y + 4, x:x + 4].reshape(-1, 3), axis=0)), 1)
        self.set(mode=3, grade=2, curve=.5)
        graded = self.effect.apply(self.image, 0)
        self.assertFalse(np.array_equal(graded, plain))
        # Grading recolours matched entries index for index, so regions keep their shape.
        _, before = np.unique(plain.reshape(-1, 3), axis=0, return_inverse=True)
        _, after = np.unique(graded.reshape(-1, 3), axis=0, return_inverse=True)
        self.assertEqual(len(set(zip(before.ravel(), after.ravel()))), len(set(before.ravel())))

    def test_zero_dither_tones_and_tiny_frames(self):
        self.set(amount=0)
        baseline = self.effect.apply(self.image, 0)
        for mode in range(8):
            self.set(mode=mode)
            np.testing.assert_array_equal(self.effect.apply(self.image, 0), baseline)
            self.assertEqual(self.effect.apply(self.image[:1, :1], 0).shape, (1, 1, 3))
        gray = np.full((20, 30, 3), 100, np.uint8)
        self.set(palette=0, colors=16, mode=0, gamma=1)
        neutral = self.effect.apply(gray, 0).mean()
        self.set(gamma=2)
        self.assertGreater(self.effect.apply(gray, 0).mean(), neutral)
        self.set(gamma=1, bias=-.3)
        self.assertLess(self.effect.apply(gray, 0).mean(), neutral)

    def test_legacy_take_does_not_inherit_new_controls_or_modulation(self):
        session = Session(self.params, Timeline(), [self.effect])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            legacy = {"version": 2, "parameters": {"palette.mode": {"value": 2},
                      "palette.enabled": {"value": 1}}}
            path.write_text(json.dumps(legacy), encoding="utf-8")
            self.set(gamma=2, bias=.4, seed=8, animate=0)
            self.params.get("palette.gamma").mod.enabled = True
            session.load(str(path))
            self.assertEqual(self.params.get("palette.gamma").value, 1)
            self.assertFalse(self.params.get("palette.gamma").mod.enabled)
            self.assertEqual(self.params.get("palette.seed").value, 0)
            self.assertEqual(self.params.get("palette.animate").value, 1)
            fresh = ParameterSet()
            effect = PaletteDitherEffect(fresh)
            effect.enabled_param().set_base(1)
            fresh.get("palette.mode").set_base(2)
            np.testing.assert_array_equal(self.effect.apply(self.image, .4), effect.apply(self.image, .4))

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_keyed_take_roundtrip_and_repeatable_real_export(self, _probe):
        timeline = Timeline()
        self.set(mode=4, palette=5, gamma=1.4, scale=2)
        timeline.set_key(0, self.params.get("palette.mode"))
        self.set(mode=5)
        timeline.set_key(.2, self.params.get("palette.mode"))
        self.assertEqual(timeline.tracks["palette.mode"].interpolation, "hold")
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.avi")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            image = cv2.resize(self.image, (64, 48))
            for _ in range(5):
                writer.write(image)
            writer.release()
            take = str(Path(directory) / "take.json")
            session = Session(self.params, timeline, [self.effect])
            session.source_path = source
            session.save(take)
            fresh, keys = ParameterSet(), Timeline()
            effect = PaletteDitherEffect(fresh)
            Session(fresh, keys, [effect]).load(take)
            decoded = []
            for i, (p, tl, fx) in enumerate(((self.params, timeline, self.effect), (fresh, keys, effect))):
                output = str(Path(directory) / f"out{i}.mp4")
                render_to_file(source, output, p, tl, [fx], with_audio=False)
                cap = cv2.VideoCapture(output)
                frames = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                cap.release()
                decoded.append(frames)
            self.assertEqual(len(decoded[0]), 5)
            np.testing.assert_array_equal(decoded[0], decoded[1])
            self.assertFalse(np.array_equal(decoded[0][0], decoded[0][3]))

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_file_preview_edit_key_seek_play_and_take_reload(self, _probe):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtTest import QTest
        from vhs_broadcast.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])

        def wait_for(predicate):
            deadline = time.monotonic() + 4
            while not predicate() and time.monotonic() < deadline:
                QTest.qWait(30)
            self.assertTrue(predicate())

        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.avi")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            for _ in range(12):
                writer.write(cv2.resize(self.image, (64, 48)))
            writer.release()
            window = MainWindow()
            try:
                window.show()
                with patch("vhs_broadcast.ui.main_window.QFileDialog.getOpenFileName", return_value=(source, "")), patch.object(window, "_make_audio"):
                    window._load()
                wait_for(lambda: window.video._pix is not None)
                window.params.get("palette.enabled").set_base(1)
                window.filter_search.setText("dither")
                panel = next(panel for panel in window.panels if panel.effect.id == "palette")
                mode_row = next(row for row in panel.rows if row.param.id == "palette.mode")
                mode_row.choice.setCurrentIndex(5)
                self.assertFalse(window.timeline.tracks)
                window._add_parameter_key("palette.mode")
                self.assertEqual(window.timeline.tracks["palette.mode"].values, [5])
                window._seek_keyframe(.3)
                wait_for(lambda: abs(window.engine.current_time() - .3) < .01)
                self.assertTrue(window.transport.pb_btn.isChecked())
                window.transport.play_btn.setChecked(True)
                wait_for(lambda: window.engine.current_time() > .4)
                window.transport.play_btn.setChecked(False)
                mode_row.choice.setCurrentIndex(7)
                self.assertFalse(window.transport.pb_btn.isChecked())
                self.assertEqual(window.timeline.tracks["palette.mode"].values, [5])
                take = str(Path(directory) / "take.json")
                with patch("vhs_broadcast.ui.main_window.QFileDialog.getSaveFileName", return_value=(take, "")):
                    window._save_session()
                mode_row.choice.setCurrentIndex(1)
                with patch("vhs_broadcast.ui.main_window.QFileDialog.getOpenFileName", return_value=(take, "")), patch.object(window, "_make_audio"):
                    window._open_session()
                self.assertEqual(mode_row.choice.currentIndex(), 7)
                self.assertEqual(window.timeline.tracks["palette.mode"].values, [5])
            finally:
                window.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()

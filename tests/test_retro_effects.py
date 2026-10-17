import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.pipeline import render_to_file
from vhs_broadcast.effects.retro import GameBoyEffect, LEDMatrixEffect, NESCorruptionEffect, RGBTortureEffect
from vhs_broadcast.effects.advanced import MotionSmearEffect, BlockCorruptionEffect
from vhs_broadcast.effects.vhs import PixelSortEffect, default_effects


class RetroEffectsTests(unittest.TestCase):
    def make(self, cls, **values):
        params = ParameterSet()
        effect = cls(params)
        effect.enabled_param().set_base(1)
        for name, value in values.items():
            params.get(effect.id + "." + name).set_base(value)
        return params, effect

    def test_registration_defaults_and_bypass(self):
        effects = default_effects(ParameterSet())
        self.assertEqual(len({e.id for e in effects}), len(effects))
        image = np.random.default_rng(2).integers(0, 256, (25, 37, 3), dtype=np.uint8)
        original = image.copy()
        for cls in (GameBoyEffect, LEDMatrixEffect, NESCorruptionEffect, RGBTortureEffect):
            with self.subTest(effect=cls.id):
                p, e = self.make(cls)
                for i in range(4):
                    out = e.apply(image, i / 30)
                    self.assertEqual(out.shape, image.shape)
                    self.assertEqual(out.dtype, np.uint8)
                    np.testing.assert_array_equal(e.apply(image, i / 30), out)
                np.testing.assert_array_equal(image, original)
                p.get(e.id + ".amount").set_base(0)
                np.testing.assert_array_equal(e.apply(image, .3), image)
                e.enabled_param().set_base(0)
                np.testing.assert_array_equal(e.apply(image, .4), image)
                e.enabled_param().set_base(1)
                self.assertEqual(e.apply(image[:1, :1], .5).shape, (1, 1, 3))

    def test_gameboy_grid_and_four_tone_palettes(self):
        p, e = self.make(GameBoyEffect, width=32, contrast=1)
        gray = np.tile(np.arange(256, dtype=np.uint8), (144, 1))
        frame = np.repeat(gray[..., None], 3, axis=2)
        for palette in range(4):
            p.get("gameboy.palette").set_base(palette)
            for mode in range(4):
                p.get("gameboy.dither").set_base(mode)
                out = e.apply(frame, 0)
                actual = set(map(tuple, out.reshape(-1, 3)))
                expected = set(map(tuple, e.PALETTES[palette, :, ::-1]))
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(out[:, 0], out[:, 7])

    def test_lcd_history_pause_reset_and_grid(self):
        p, e = self.make(GameBoyEffect, ghost=.8, contrast=1, width=32)
        white = np.full((96, 128, 3), 255, np.uint8)
        black = white * 0
        e.apply(white, 0)
        trail = e.apply(black, 1 / 30)
        np.testing.assert_array_equal(e.apply(black, 1 / 30), trail)
        self.assertGreater(trail.mean(), e.PALETTES[0, 0].mean())
        e.request_reset()
        self.assertAlmostEqual(e.apply(black, 1 / 30).mean(), e.PALETTES[0, 0].mean())
        p.get("gameboy.ghost").set_base(0)
        plain = e.apply(white, .2)
        p.get("gameboy.grid").set_base(.5)
        self.assertLess(e.apply(white, .2).mean(), plain.mean())

    def test_led_matrix_draws_round_dots_on_a_pitch_grid(self):
        p, e = self.make(LEDMatrixEffect, pitch=8, glow=0, unlit=0, threshold=.5)
        white = np.full((32, 48, 3), 255, np.uint8)
        out = e.apply(white, 0)
        np.testing.assert_array_equal(out, e.apply(white, 3))
        cell = out[:8, :8].max(axis=2)
        self.assertGreater(cell[4, 4], 200)
        self.assertLess(cell[0, 0], 30)
        np.testing.assert_array_equal(out[:8, :8], out[8:16, 16:24])
        dark = e.apply(np.zeros_like(white), 0)
        self.assertLess(int(dark.max()), 30)
        p.get("led_matrix.invert").set_base(1)
        np.testing.assert_array_equal(e.apply(np.zeros_like(white), 0), out)
        p.get("led_matrix.invert").set_base(0)
        grey = np.full_like(white, 100)
        self.assertLess(int(e.apply(grey, 0).max()), 30)
        p.get("led_matrix.levels").set_base(4)
        p.get("led_matrix.threshold").set_base(0)
        mid = e.apply(grey, 0)[4, 4].max()
        self.assertTrue(30 < mid < 220)
        p.get("led_matrix.amount").set_base(0)
        np.testing.assert_array_equal(e.apply(grey, 0), grey)

    def test_rgb_identity_routing_wrapped_offsets_and_quantization(self):
        p, e = self.make(RGBTortureEffect)
        image = np.arange(8 * 16 * 3).reshape(8, 16, 3).astype(np.uint8)
        np.testing.assert_array_equal(e.apply(image, 0), image)
        p.get("rgb_torture.order").set_base(5)
        np.testing.assert_array_equal(e.apply(image, 0), image[..., ::-1])
        p.get("rgb_torture.order").set_base(0)
        p.get("rgb_torture.rx").set_base(.25)
        out = e.apply(image, 0)
        np.testing.assert_array_equal(out[..., 2], np.roll(image[..., 2], 4, axis=1))
        np.testing.assert_array_equal(out[..., :2], image[..., :2])
        p.get("rgb_torture.ginvert").set_base(1)
        np.testing.assert_array_equal(e.apply(image, 0)[..., 1], 255 - image[..., 1])
        p.get("rgb_torture.bbits").set_base(1)
        self.assertEqual(set(np.unique(e.apply(image, 0)[..., 0])), {0, 255})

    def test_nes_corruption_modes_change_pixels_and_hold_when_rate_zero(self):
        image = np.random.default_rng(4).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        p, e = self.make(NESCorruptionEffect, density=0, planes=0, attributes=0, tear=0, rate=0)
        clean = e.apply(image, 0)
        for name in ("density", "planes", "attributes", "tear"):
            p.get("nes_corrupt." + name).set_base(1)
            out = e.apply(image, 0)
            self.assertFalse(np.array_equal(clean, out), name)
            np.testing.assert_array_equal(out, e.apply(image, 10))
            p.get("nes_corrupt." + name).set_base(0)

    def test_sort_reorders_actual_pixels_on_both_axes_and_all_metrics(self):
        p, e = self.make(PixelSortEffect, amount=1, threshold=0, length=1)
        image = np.array([[[200, 40, 30], [10, 180, 50], [40, 30, 250], [90, 90, 90]]], np.uint8)
        for direction in range(2):
            frame = image if direction == 0 else image.transpose(1, 0, 2)
            p.get("pixel_sort.direction").set_base(direction)
            for metric in range(6):
                p.get("pixel_sort.metric").set_base(metric)
                out = e.apply(frame, 0)
                self.assertEqual(sorted(map(tuple, frame.reshape(-1, 3))), sorted(map(tuple, out.reshape(-1, 3))))
                if metric == 0:
                    values = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).ravel()
                elif metric in (1, 2):
                    values = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., metric - 1].ravel()
                else:
                    values = out[..., 5 - metric].ravel()
                self.assertTrue(np.all(np.diff(values.astype(float)) >= 0))

    def test_sort_band_preserves_boundaries(self):
        p, e = self.make(PixelSortEffect, amount=1, threshold=.2, upper=.8, length=1)
        image = np.repeat(np.array([[0, 180, 100, 255, 140, 60]], np.uint8)[..., None], 3, axis=2)
        np.testing.assert_array_equal(e.apply(image, 0)[0, :, 0], [0, 100, 180, 255, 60, 140])
        p.get("pixel_sort.outside").set_base(1)
        np.testing.assert_array_equal(e.apply(image, 0), image)

    def test_block_matching_follows_translation_and_holds_vectors(self):
        p, e = self.make(MotionSmearEffect, mode=1, block=8, search=6, hold=.5)
        first = np.random.default_rng(9).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        shifted = np.roll(first, 4, axis=1)
        e.apply(first, 0)
        out = e.apply(shifted, .1)
        flow = e._pending["flow"].copy()
        np.testing.assert_allclose(flow[8:24, 8:40, 0], -4)
        np.testing.assert_allclose(flow[8:24, 8:40, 1], 0)
        np.testing.assert_array_equal(e.apply(shifted, .1), out)
        e.apply(first, .2)
        np.testing.assert_array_equal(e._pending["flow"], flow)
        e.request_reset()
        np.testing.assert_array_equal(e.apply(first, .2), first)

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_new_effect_take_and_real_exports_match_after_reload(self, _probe):
        p = ParameterSet()
        effects = [GameBoyEffect(p), NESCorruptionEffect(p), RGBTortureEffect(p), BlockCorruptionEffect(p)]
        p.get("block_corrupt.mode").set_base(3)
        p.get("block_corrupt.pool").set_base(5)
        for effect in effects:
            effect.enabled_param().set_base(1)
        p.get("gameboy.palette").set_base(2)
        p.get("rgb_torture.rx").set_base(.1)
        timeline = Timeline()
        timeline.set_key(0, p.get("gameboy.palette"))
        p.get("gameboy.palette").set_base(0)
        timeline.set_key(.2, p.get("gameboy.palette"))
        with tempfile.TemporaryDirectory() as directory:
            take = str(Path(directory) / "take.json")
            Session(p, timeline, effects).save(take)
            fresh, keys = ParameterSet(), Timeline()
            others = [GameBoyEffect(fresh), NESCorruptionEffect(fresh), RGBTortureEffect(fresh), BlockCorruptionEffect(fresh)]
            Session(fresh, keys, others).load(take)
            self.assertEqual(fresh.snapshot(), p.snapshot())
            source = str(Path(directory) / "source.avi")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            for i in range(5):
                writer.write(np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8))
            writer.release()
            decoded = []
            for i, (params, tracks, chain) in enumerate(((p, timeline, effects), (fresh, keys, others))):
                output = str(Path(directory) / f"output{i}.mp4")
                render_to_file(source, output, params, tracks, chain, with_audio=False)
                capture = cv2.VideoCapture(output)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                capture.release()
                decoded.append(frames)
            self.assertEqual(len(decoded[0]), 5)
            np.testing.assert_array_equal(decoded[0], decoded[1])

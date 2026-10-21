import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.pipeline import RenderCancelled, render_to_file
from vhs_broadcast.effects.glitch import RasterHoldEffect, RollingShutterEffect, SyncLossEffect


class GlitchTests(unittest.TestCase):
    def make(self, cls, **values):
        params = ParameterSet()
        effect = cls(params)
        effect.enabled_param().set_base(1)
        for key, value in values.items():
            params.get(f"{effect.id}.{key}").set_base(value)
        return params, effect

    def test_raster_latches_pixels_and_preserves_input(self):
        p, effect = self.make(RasterHoldEffect, amount=1, density=1, length=8, rate=0)
        frame = np.arange(30 * 50 * 3).reshape(30, 50, 3).astype(np.uint8)
        saved = frame.copy()
        out = effect.apply(frame, 0)
        self.assertGreater(np.mean(np.all(out[:, 1:] == out[:, :-1], axis=2)), .8)
        np.testing.assert_array_equal(frame, saved)
        np.testing.assert_array_equal(effect.apply(frame, 4), out)
        for mode in range(4):
            p.get("raster_hold.direction").set_base(mode)
            actual = effect.apply(frame, 4)
            self.assertEqual(actual.shape, frame.shape)
            self.assertEqual(actual.dtype, np.uint8)
            np.testing.assert_array_equal(effect.apply(frame, 4), actual)
        p.get("raster_hold.amount").set_base(0)
        np.testing.assert_array_equal(effect.apply(frame, 4), frame)

    def test_raster_animation_and_zero_density(self):
        p, effect = self.make(RasterHoldEffect, rate=30)
        frame = np.random.default_rng(8).integers(0, 256, (40, 60, 3), dtype=np.uint8)
        first = effect.apply(frame, 0)
        second = effect.apply(frame, .1)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(effect.apply(frame, .1), second)
        p.get("raster_hold.density").set_base(0)
        np.testing.assert_array_equal(effect.apply(frame, .2), frame)

    def test_rolling_shutter_uses_subframe_history_and_direction(self):
        p, effect = self.make(RollingShutterEffect, readout=.1)
        dark = np.zeros((11, 11, 3), np.uint8)
        bright = np.full_like(dark, 200)
        effect.apply(dark, 0)
        for direction in range(4):
            p.get("rolling_shutter.direction").set_base(direction)
            out = effect.apply(bright, .1)
            sample = out[:, 0, 0] if direction < 2 else out[0, :, 0]
            self.assertEqual((int(sample[0]), int(sample[-1])),
                             (0, 200) if direction % 2 == 0 else (200, 0))
            self.assertAlmostEqual(int(sample[5]), 100, delta=1)
            np.testing.assert_array_equal(effect.apply(bright, .1), out)
        p.get("rolling_shutter.readout").set_base(0)
        np.testing.assert_array_equal(effect.apply(bright, .1), bright)

    def test_rolling_history_budget_seek_disable_and_reset(self):
        p, effect = self.make(RollingShutterEffect)
        frame = np.full((24, 32, 3), 100, np.uint8)
        effect.MAX_HISTORY_BYTES = frame.nbytes * 3
        for i in range(20):
            effect.apply(frame, i / 30)
        self.assertEqual(len(effect._pending["history"]), 3)
        effect.apply(frame, .1)
        self.assertEqual(len(effect._pending["history"]), 1)
        effect.request_reset()
        np.testing.assert_array_equal(effect.apply(frame * 0, .1), frame * 0)
        effect.enabled_param().set_base(0)
        effect.apply(frame, .2)
        self.assertFalse(effect._pending)

    def test_sync_loss_is_repeatable_and_preserves_input(self):
        p, effect = self.make(SyncLossEffect, rate=10)
        frame = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        saved = frame.copy()
        out = effect.apply(frame, .25)
        self.assertEqual((out.shape, out.dtype), (frame.shape, np.uint8))
        np.testing.assert_array_equal(frame, saved)
        np.testing.assert_array_equal(effect.apply(frame, .25), out)
        p.get("sync_loss.amount").set_base(0)
        np.testing.assert_array_equal(effect.apply(frame, .3), frame)

    def test_sync_loss_bursts_smear_whole_lines_and_quantize(self):
        p, effect = self.make(SyncLossEffect, burst=1, smear=1, split=0, dropout=0,
                              freeze=0, crush=0, dither=0, bits=1, rate=0)
        frame = np.random.default_rng(5).integers(0, 256, (60, 80, 3), dtype=np.uint8)
        out = effect.apply(frame, 0)
        self.assertTrue(set(np.unique(out)) <= {0, 255})
        flat = np.mean(np.all(out == out[:, :1], axis=(1, 2)))
        self.assertGreater(flat, .4)

    def test_sync_loss_freeze_holds_the_first_frame_of_a_tick(self):
        p, effect = self.make(SyncLossEffect, freeze=1, burst=0, smear=0, dropout=0,
                              crush=0, bits=8, rate=1)
        first = np.full((20, 20, 3), 40, np.uint8)
        second = np.full_like(first, 200)
        held = effect.apply(first, .1)
        np.testing.assert_array_equal(effect.apply(second, .2), held)
        self.assertFalse(np.array_equal(effect.apply(second, 1.2), held))



class AnalogShapeTests(unittest.TestCase):
    def make(self, cls, **values):
        params = ParameterSet()
        effect = cls(params)
        effect.enabled_param().set_base(1)
        for key, value in values.items():
            params.get(f"{effect.id}.{key}").set_base(value)
        return params, effect

    def test_fuzz_tall_and_wide_streaks_and_dash_columns(self):
        from vhs_broadcast.effects.vhs import NoiseEffect
        flat = np.full((60, 80, 3), 128, np.uint8)
        def corr(img, axis):
            g = img[..., 0].astype(float) - 128
            a, b = (g[1:], g[:-1]) if axis == 0 else (g[:, 1:], g[:, :-1])
            return np.corrcoef(a.ravel(), b.ravel())[0, 1]
        _, tall = self.make(NoiseEffect, shape=1, grain=1, dropout=0, length=12)
        out = tall.apply(flat, 1.0)
        self.assertGreater(corr(out, 0), .7)
        self.assertLess(abs(corr(out, 1)), .2)
        np.testing.assert_array_equal(tall.apply(flat, 1.0), out)
        _, wide = self.make(NoiseEffect, shape=2, grain=1, dropout=0, length=12)
        out = wide.apply(flat, 1.0)
        self.assertGreater(corr(out, 1), .7)
        self.assertLess(abs(corr(out, 0)), .2)
        _, dashes = self.make(NoiseEffect, shape=1, grain=0, dropout=1, length=10, columns=10)
        hit = np.flatnonzero((dashes.apply(flat, 2.0) == 235).any(axis=(0, 2)))
        self.assertTrue(len(hit) and (hit % 10 == 0).all())
        _, dots = self.make(NoiseEffect)
        before = NoiseEffect(ParameterSet()); before.enabled_param().set_base(1)
        np.testing.assert_array_equal(dots.apply(flat, 3.0), before.apply(flat, 3.0))

    def test_tearing_vertical_mirrors_horizontal_and_zigzag_bends_edges(self):
        from vhs_broadcast.effects.vhs import TearEffect
        frame = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        _, horizontal = self.make(TearEffect, amount=1, count=1)
        _, vertical = self.make(TearEffect, amount=1, count=1, direction=1)
        flipped = horizontal.apply(np.ascontiguousarray(frame.transpose(1, 0, 2)), .5)
        np.testing.assert_array_equal(vertical.apply(frame, .5), flipped.transpose(1, 0, 2))
        edge = np.zeros((60, 100, 3), np.uint8)
        edge[:, 50:] = 255
        _, zig = self.make(TearEffect, amount=1, count=1, shape=1, step=10)
        out = zig.apply(edge, 0)
        # Shifted rows wrap around, so find the black-to-white step near the edge only.
        first_white = 30 + (out[:, 30:80, 0] > 128).argmax(axis=1)
        self.assertGreater(np.ptp(first_white), 4)
        self.assertLessEqual(np.abs(first_white - 50).max(), round(100 * .06) + 1)
        _, half = self.make(TearEffect, amount=1, count=.5, shape=1, step=10)
        np.testing.assert_array_equal(half.apply(edge, 0)[:30], edge[:30])

    def test_bloom_jitter_shakes_ragged_lines_and_jumps(self):
        from vhs_broadcast.effects.vhs import BloomEffect
        frame = np.random.default_rng(2).integers(0, 256, (72, 96, 3), dtype=np.uint8)
        p, bloom = self.make(BloomEffect, amount=0)
        np.testing.assert_array_equal(bloom.apply(frame, 1.0), frame)
        p.get("bloom.shake").set_base(3)
        first = bloom.apply(frame, 1.0)
        np.testing.assert_array_equal(bloom.apply(frame, 1.0), first)
        self.assertFalse(np.array_equal(bloom.apply(frame, 1.04), first))
        p.get("bloom.shake").set_base(0)
        p.get("bloom.jumps").set_base(4)
        black = [bloom.apply(frame, i / 25)[:3].mean() for i in range(100)]
        self.assertTrue(any(b < 5 for b in black))     # a slip uncovers the top edge


class RenderSafetyTests(unittest.TestCase):
    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_cancel_preserves_destination_and_removes_partial_output(self, _probe):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.avi")
            output = Path(directory) / "existing.mp4"
            output.write_bytes(b"existing export")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
            for i in range(5):
                writer.write(np.full((24, 32, 3), i * 40, np.uint8))
            writer.release()
            progress = []
            with self.assertRaises(RenderCancelled):
                render_to_file(source, str(output), ParameterSet(), Timeline(), [],
                               with_audio=False, progress=progress.append,
                               cancelled=lambda: bool(progress))
            self.assertEqual(output.read_bytes(), b"existing export")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["existing.mp4", "source.avi"])
            # File handles are released and the same destination can be used again.
            render_to_file(source, str(output), ParameterSet(), Timeline(), [], with_audio=False)
            self.assertGreater(output.stat().st_size, 100)

    def test_cannot_overwrite_source(self):
        with self.assertRaisesRegex(ValueError, "source cannot be overwritten"):
            render_to_file("same.mp4", "same.mp4", ParameterSet(), Timeline(), [])


class TakeSafetyTests(unittest.TestCase):
    def test_bad_take_restores_controls_and_keeps_keys(self):
        params = ParameterSet()
        effect = RasterHoldEffect(params)
        timeline = Timeline()
        timeline.set_key(2, params.get("raster_hold.length"))
        session = Session(params, timeline, [effect])
        original = params.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.json"
            path.write_text(json.dumps({"parameters": {
                "raster_hold.length": {"value": 100},
                "raster_hold.density": {"value": "invalid"}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                session.load(str(path))
            self.assertEqual(params.snapshot(), original)
            self.assertEqual(timeline.tracks["raster_hold.length"].times, [2])
            session.save(str(path))
            session.load(str(path))
            self.assertEqual(params.snapshot(), original)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["take.json"])


if __name__ == "__main__":
    unittest.main()

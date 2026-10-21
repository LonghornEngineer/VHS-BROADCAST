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
from vhs_broadcast.effects.glitch import CRTRasterEffect, RasterHoldEffect, RollingShutterEffect, SyncLossEffect


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



class CRTRasterTests(unittest.TestCase):
    def make(self, **values):
        params = ParameterSet()
        effect = CRTRasterEffect(params)
        effect.enabled_param().set_base(1)
        for key, value in values.items():
            params.get(f"crt_raster.{key}").set_base(value)
        return params, effect

    def values(self, params, effect):
        return {p.id.split(".")[-1]: p.evaluate(0) for p in effect.own_params()}

    def test_sync_error_scales_with_noise_crawls_down_and_lock_sets_texture(self):
        p, e = self.make(churn=0, kicks=0, crawl=1, noise=6)
        h = 360
        a = e._line_error(h, 1.0, self.values(p, e))
        b = e._line_error(h, 1.0 + 1 / 30, self.values(p, e))
        np.testing.assert_allclose(a[:h - 12], b[12:], atol=1e-4)      # pattern slid 12 lines down
        for lock in (0, .6, 1):
            p.get("crt_raster.lock").set_base(lock)
            errs = np.concatenate([e._line_error(h, t, self.values(p, e)) for t in np.arange(0, 20, .5)])
            self.assertAlmostEqual(errs.std(), 6, delta=1.5)
        p.get("crt_raster.lock").set_base(0)
        smooth = e._line_error(h, 3.0, self.values(p, e))
        p.get("crt_raster.lock").set_base(1)
        rough = e._line_error(h, 3.0, self.values(p, e))
        self.assertGreater(np.corrcoef(smooth[1:], smooth[:-1])[0, 1], np.corrcoef(rough[1:], rough[:-1])[0, 1] + .5)

    def test_identity_jagged_edges_hold_roll_and_beam(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, 80:] = 200
        p, e = self.make(noise=0, bar=0)
        np.testing.assert_array_equal(e.apply(frame, 1.0), frame)
        p.get("crt_raster.noise").set_base(10)
        out = e.apply(frame, 1.0)
        np.testing.assert_array_equal(e.apply(frame, 1.0), out)
        edges = (out[:, 20:140, 0] > 100).argmax(axis=1) + 20
        self.assertGreater(np.ptp(edges), 6)                            # jagged edge
        p.get("crt_raster.noise").set_base(0)
        p.get("crt_raster.hhold").set_base(.2)
        # Lines past the right edge wrap to the left, so find the edge right of that strip.
        slant = (e.apply(frame, 1.0)[:, 40:, 0] > 100).argmax(axis=1) + 40
        self.assertEqual(slant[0], 80)
        self.assertGreater(slant[-1], slant[0] + 10)                    # later lines start later
        p.get("crt_raster.hhold").set_base(0)
        p.get("crt_raster.vhold").set_base(.5)
        p.get("crt_raster.bar").set_base(.1)
        rolled = e.apply(np.full_like(frame, 200), .5)
        dark = rolled[..., 0].mean(axis=1) < 40
        self.assertEqual(int(dark.sum()), 12)                           # blanking bar rolls into view
        p.get("crt_raster.vhold").set_base(0)
        p.get("crt_raster.beam").set_base(1)
        lit = e.apply(np.full_like(frame, 200), .25)[..., 0].mean(axis=1)
        self.assertGreater(lit.max() - lit.min(), 60)


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

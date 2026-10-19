import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.automation import Timeline, Session
from vhs_broadcast.core.pipeline import Pipeline, render_to_file
from vhs_broadcast.effects.advanced import (
    ADVANCED_EFFECTS, TemporalEffect, VideoFeedbackEffect, MotionSmearEffect,
    CompositeEffect, PaletteDitherEffect, TemporalDisplacementEffect,
    BlockCorruptionEffect, PhosphorPersistenceEffect, CRTGeometryEffect,
    RFInterferenceEffect, LumaKeyEffect, LineDitherEffect, FMVEffect,
)


class AdvancedEffectsTests(unittest.TestCase):
    def make(self, kind):
        p = ParameterSet()
        e = kind(p)
        e.enabled_param().set_base(1)
        return p, e

    def test_line_dither_spaces_bends_and_masks_lines(self):
        p, e = self.make(LineDitherEffect)
        for name, value in dict(spacing=8, thickness=1, bend=0, smooth=0, weight=0, color=1, follow=0).items():
            p.get("line_dither." + name).set_base(value)
        bright = np.full((24, 40, 3), 200, np.uint8)
        out = e.apply(bright, 0)
        columns = np.flatnonzero(out[0, :, 0] > 128)
        np.testing.assert_array_equal(columns, np.arange(0, 40, 8))
        self.assertTrue((out == out[:1]).all())
        self.assertEqual(int(e.apply(np.zeros_like(bright), 0).max()), 0)
        p.get("line_dither.bend").set_base(4)
        shifted = np.flatnonzero(e.apply(bright, 0)[0, :, 0] > 128)
        self.assertFalse(np.array_equal(shifted, columns))
        p.get("line_dither.bend").set_base(0)
        p.get("line_dither.direction").set_base(1)
        rows = np.flatnonzero(e.apply(bright, 0)[:, 0, 0] > 128)
        np.testing.assert_array_equal(rows, np.arange(0, 24, 8))
        p.get("line_dither.amount").set_base(0)
        np.testing.assert_array_equal(e.apply(bright, 0), bright)

    def test_line_dither_lines_echo_the_subject_edge(self):
        p, e = self.make(LineDitherEffect)
        for name, value in dict(spacing=6, thickness=1, bend=0, smooth=0, weight=0, color=1, threshold=.5).items():
            p.get("line_dither." + name).set_base(value)
        frame = np.zeros((20, 60, 3), np.uint8)
        for y in range(20):
            frame[y, 10 + y:] = 220  # leading edge slants one px per row
        out = e.apply(frame, 0)[..., 0] > 128
        for y in (0, 7, 19):
            np.testing.assert_array_equal(np.flatnonzero(out[y]), np.arange(10 + y, 60, 6))
        p.get("line_dither.count").set_base(2)
        first = np.flatnonzero(e.apply(frame, 0)[3, :, 0] > 128)
        np.testing.assert_array_equal(first, [13, 19])
        p.get("line_dither.count").set_base(0)
        p.get("line_dither.follow").set_base(2)
        trailing = e.apply(frame, 0)[..., 0] > 128
        np.testing.assert_array_equal(trailing[0, 29:], trailing[19, 29:])  # both rows lit from x=29
        p.get("line_dither.follow").set_base(3)
        self.assertTrue((e.apply(frame, 0)[..., 0] > 128).any())

    def test_fmv_blocks_frame_hold_and_skipped_blocks(self):
        p, e = self.make(FMVEffect)
        for name, value in dict(width=64, flatten=1, dither=0, skip=0, fps=0).items():
            p.get("fmv." + name).set_base(value)
        frame = np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        out = e.apply(frame, 0)
        for y in range(0, 48, 4):
            for x in range(0, 64, 4):
                self.assertEqual(len(np.unique(out[y:y + 4, x:x + 4].reshape(-1, 3), axis=0)), 1)
        grey = np.full_like(frame, 128)
        self.assertTrue((np.ptp(e.apply(grey, .5).astype(int), axis=2) <= 2).all())
        p.get("fmv.fps").set_base(10)
        held = e.apply(frame, 1.0)
        np.testing.assert_array_equal(e.apply(np.zeros_like(frame), 1.05), held)
        self.assertFalse(np.array_equal(e.apply(np.zeros_like(frame), 1.1), held))
        p.get("fmv.fps").set_base(0)
        p.get("fmv.skip").set_base(1)
        e.reset_state()
        first = e.apply(frame, 2.0)
        nudged = np.clip(frame.astype(int) + 3, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(e.apply(nudged, 2.04), first)
        self.assertFalse(np.array_equal(e.apply(255 - frame, 2.08), first))
        p.get("fmv.amount").set_base(0)
        np.testing.assert_array_equal(e.apply(frame, 2.12), frame)

    def test_every_effect_shape_type_no_input_mutation_and_pause(self):
        frames = [np.random.default_rng(i).integers(0, 256, (48, 64, 3), np.uint8) for i in range(3)]
        for kind in ADVANCED_EFFECTS:
            with self.subTest(effect=kind.id):
                p, effect = self.make(kind)
                for i, frame in enumerate(frames):
                    original = frame.copy()
                    out = effect.apply(frame, i / 30)
                    self.assertEqual(out.dtype, np.uint8)
                    self.assertEqual(out.shape, frame.shape)
                    np.testing.assert_array_equal(original, frame)
                    np.testing.assert_array_equal(effect.apply(frame, i / 30), out)
                effect.enabled_param().set_base(0)
                np.testing.assert_array_equal(effect.apply(frames[-1], .2), frames[-1])

    def test_feedback_trails_do_not_decay_while_paused_and_reset(self):
        p, effect = self.make(VideoFeedbackEffect)
        bright = np.full((24, 32, 3), 200, np.uint8)
        dark = np.zeros_like(bright)
        effect.apply(bright, 0)
        trail = effect.apply(dark, 1 / 30)
        self.assertGreater(trail.mean(), 0)
        for _ in range(4):
            np.testing.assert_array_equal(effect.apply(dark, 1 / 30), trail)
        decayed = effect.apply(dark, 2 / 30)
        self.assertLess(decayed.mean(), trail.mean())
        effect.request_reset()
        np.testing.assert_array_equal(effect.apply(dark, 2 / 30), dark)

    def test_keyed_reset_is_rising_edge_not_permanent_bypass(self):
        p, effect = self.make(VideoFeedbackEffect)
        bright = np.full((24, 32, 3), 180, np.uint8)
        dark = np.zeros_like(bright)
        effect.apply(bright, 0)
        p.get("feedback.reset").set_base(1)
        np.testing.assert_array_equal(effect.apply(dark, .1), dark)
        effect.apply(bright, .2)
        self.assertGreater(effect.apply(dark, .3).mean(), 0)

    def test_temporal_history_clears_on_backward_time_resize_disable_and_pipeline_seek(self):
        for kind in ADVANCED_EFFECTS:
            if not issubclass(kind, TemporalEffect):
                continue
            with self.subTest(effect=kind.id):
                p, e = self.make(kind)
                pipeline = Pipeline([e])
                bright = np.full((24, 32, 3), 220, np.uint8)
                dark = np.zeros_like(bright)
                pipeline.process(bright, 0)
                pipeline.process(dark, .1)
                pipeline.request_reset()
                np.testing.assert_array_equal(pipeline.process(dark, .1), dark)
                e.apply(bright, .2)
                np.testing.assert_array_equal(e.apply(dark, 0), dark)
                e.apply(bright, .2)
                smaller = np.zeros((16, 20, 3), np.uint8)
                np.testing.assert_array_equal(e.apply(smaller, .3), smaller)
                e.enabled_param().set_base(0)
                e.apply(bright, .4)
                e.enabled_param().set_base(1)
                np.testing.assert_array_equal(e.apply(dark, .5), dark)

    def test_smear_carries_previous_frame_and_zero_persistence_bypasses(self):
        p, e = self.make(MotionSmearEffect)
        bright = np.full((40, 60, 3), 180, np.uint8)
        dark = np.zeros_like(bright)
        e.apply(bright, 0)
        self.assertGreater(e.apply(dark, 1 / 30).mean(), 100)
        p.get("motion_smear.persistence").set_base(0)
        np.testing.assert_array_equal(e.apply(dark, 2 / 30), dark)

    def test_temporal_map_contains_different_ages_and_memory_is_bounded(self):
        p, e = self.make(TemporalDisplacementEffect)
        p.get("time_displace.speed").set_base(0)
        p.get("time_displace.depth").set_base(1)
        e.MEMORY_BYTES = 3 * 24 * 32 * 3
        for i in range(10):
            out = e.apply(np.full((24, 32, 3), i * 20, np.uint8), i / 10)
        self.assertGreater(len(np.unique(out)), 1)
        self.assertLessEqual(len(e._pending["history"]), 3)
        p.get("time_displace.depth").set_base(0)
        current = np.full((24, 32, 3), 225, np.uint8)
        np.testing.assert_array_equal(e.apply(current, 1), current)

    def test_block_freezes_then_releases_and_chroma_mode_keeps_luma(self):
        p, e = self.make(BlockCorruptionEffect)
        p.get("block_corrupt.density").set_base(1)
        p.get("block_corrupt.shift").set_base(0)
        first = np.full((24, 32, 3), 200, np.uint8)
        second = np.full_like(first, 50)
        e.apply(first, 0)
        np.testing.assert_array_equal(e.apply(second, .1), first)
        np.testing.assert_array_equal(e.apply(second, .5), second)
        e.reset_state()
        p.get("block_corrupt.chroma").set_base(1)
        e.apply(np.full_like(first, [180, 90, 120]), 0)
        out = e.apply(np.full_like(first, [120, 140, 110]), .1)
        target = cv2.cvtColor(np.full_like(first, [120, 140, 110]), cv2.COLOR_BGR2GRAY)
        np.testing.assert_allclose(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), target, atol=2)

    def test_phosphor_separate_rgb_decay(self):
        p, e = self.make(PhosphorPersistenceEffect)
        e.apply(np.full((20, 20, 3), 255, np.uint8), 0)
        out = e.apply(np.zeros((20, 20, 3), np.uint8), .3)
        blue, green, red = out[0, 0]
        self.assertGreater(green, red)
        self.assertGreater(red, blue)

    def test_palette_output_is_exact_palette_and_noise_repeats(self):
        p, e = self.make(PaletteDitherEffect)
        image = np.random.default_rng(4).integers(0, 256, (24, 32, 3), np.uint8)
        for palette in range(5):
            p.get("palette.palette").set_base(palette)
            for mode in range(3):
                p.get("palette.mode").set_base(mode)
                output = e.apply(image, .4)
                colors = e.colors({"palette": palette, "colors": 8}).clip(0, 255).astype(np.uint8)
                valid = {tuple(color) for color in colors}
                self.assertTrue(all(tuple(color) in valid for color in output.reshape(-1, 3)))
                np.testing.assert_array_equal(e.apply(image, .4), output)

    def test_signal_patterns_and_crt_mask_are_visible(self):
        frame = np.full((64, 96, 3), 180, np.uint8)
        p, e = self.make(RFInterferenceEffect)
        out = e.apply(frame, .1)
        self.assertGreater(float(out.std()), 1)
        self.assertLess(out.mean(), frame.mean())
        p, e = self.make(CRTGeometryEffect)
        out = e.apply(frame, 0)
        self.assertLess(out[0, 0].max(), 10)
        self.assertFalse(np.array_equal(out[32, 40], out[32, 41]))
        p, e = self.make(CompositeEffect)
        checker = np.tile(np.array([0, 255], np.uint8), (32, 32))
        source = np.repeat(checker[..., None], 3, axis=2)
        out = e.apply(source, .2)
        self.assertTrue(np.any(out[..., 0] != out[..., 2]))

    def test_luma_key_uses_original_feed_and_persists_background(self):
        p, e = self.make(LumaKeyEffect)
        source = np.full((16, 24, 3), 70, np.uint8)
        p.get("luma_key.fill").set_base(3)
        np.testing.assert_array_equal(Pipeline([e]).process(source, 0), source)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "background.png")
            cv2.imwrite(path, np.full((12, 20, 3), [30, 100, 180], np.uint8))
            e.load_background(path)
            p.get("luma_key.fill").set_base(4)
            out = e.apply(np.zeros_like(source), 0)
            np.testing.assert_array_equal(out, np.full_like(source, [30, 100, 180]))
            take = str(Path(directory) / "take.json")
            Session(p, Timeline(), [e]).save(take)
            other_p, other_e = self.make(LumaKeyEffect)
            Session(other_p, Timeline(), [other_e]).load(take)
            np.testing.assert_array_equal(other_e.apply(np.zeros_like(source), 0), out)

    def test_mode_keys_hold_without_automatic_keys(self):
        p, e = self.make(PaletteDitherEffect)
        timeline = Timeline()
        choice = p.get("palette.palette")
        timeline.set_key(0, choice)
        choice.set_base(1)
        timeline.set_key(1, choice)
        self.assertEqual(timeline.tracks[choice.id].interpolation, "hold")
        timeline.apply(.5, p)
        self.assertEqual(choice.value, 4)
        choice.set_base(2)
        self.assertEqual(len(timeline.tracks[choice.id].times), 2)

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_two_exports_start_clean_and_do_not_touch_preview_buffers(self, _probe):
        p, e = self.make(VideoFeedbackEffect)
        bright = np.full((48, 64, 3), 200, np.uint8)
        e.apply(bright, 0)
        saved = e._pending["output"].copy()
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.avi")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            for i in range(6):
                writer.write(bright if i == 0 else np.zeros_like(bright))
            writer.release()
            outputs = []
            for i in range(2):
                output = str(Path(directory) / f"out{i}.mp4")
                render_to_file(source, output, p, Timeline(), [e], with_audio=False)
                capture = cv2.VideoCapture(output)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                capture.release()
                outputs.append(frames)
            self.assertEqual(len(outputs[0]), 6)
            np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(e._pending["output"], saved)


if __name__ == "__main__":
    unittest.main()

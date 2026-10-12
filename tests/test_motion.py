import unittest
from unittest.mock import patch
import numpy as np

from vhs_broadcast.core.audio_react import AudioAnalysis, set_analysis
from vhs_broadcast.core.automation import Recorder, Timeline
from vhs_broadcast.core.parameters import AudioMod, Modulator, ParameterSet
from vhs_broadcast.effects.vhs import default_effects


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effects = {e.id: e for e in default_effects(self.params)}
        self.frame = np.repeat(np.arange(100, dtype=np.uint8)[:, None, None], 80, axis=1)
        self.frame = np.repeat(self.frame, 3, axis=2)

    def tearDown(self):
        set_analysis(None)

    def test_rate_edits_stop_and_reverse_without_jumping(self):
        p = self.params.get("tracking.rollspeed")
        p.set_base(.5)
        phase = lambda t: self.params.motion_phase(p.id, t, quadratic=True)
        self.assertAlmostEqual(phase(20), 5)
        p.set_base(0)
        self.assertAlmostEqual(phase(20), 5)
        self.assertAlmostEqual(phase(22), 5)
        p.set_base(-.5)
        self.assertAlmostEqual(phase(22), 5)
        self.assertAlmostEqual(phase(24), 4.5)
        self.assertAlmostEqual(phase(10), 2.5)
        self.assertAlmostEqual(phase(24), 4.5)

    def test_audio_roll_holds_and_resumes(self):
        a = AudioAnalysis(np.zeros(48000 * 5), 48000, 10, 50)
        a.level[10:20] = 1
        a.level[30:40] = 1
        set_analysis(a)
        p = self.params.get("tracking.rollspeed")
        p.set_base(.5)
        p.audio = AudioMod(enabled=True, band="level", threshold=.5, ramp=0, rest=0)
        self.params.get("tracking.jitter").set_base(0)
        effect = self.effects["tracking"]
        effect.enabled_param().set_base(1)
        before = effect.apply(self.frame.copy(), 0)
        moved = effect.apply(self.frame.copy(), 2.1)
        held = effect.apply(self.frame.copy(), 2.9)
        resumed = effect.apply(self.frame.copy(), 4.1)
        self.assertFalse(np.array_equal(before, moved))
        np.testing.assert_array_equal(moved, held)
        np.testing.assert_array_equal(moved, np.roll(self.frame, 15, axis=0))
        np.testing.assert_array_equal(resumed, np.roll(self.frame, 30, axis=0))

    def test_nonzero_rest_keeps_drifting(self):
        a = AudioAnalysis(np.zeros(48000 * 3), 48000, 10, 30)
        a.level[10:20] = 1
        set_analysis(a)
        p = self.params.get("wobble.speed")
        p.set_base(2)
        p.audio = AudioMod(enabled=True, band="level", ramp=0, rest=.5)
        self.assertAlmostEqual(self.params.motion_phase(p.id, 1), .5, places=5)
        self.assertAlmostEqual(self.params.motion_phase(p.id, 2), 2.5, places=5)
        self.assertAlmostEqual(self.params.motion_phase(p.id, 3), 3, places=5)

    def test_tracking_scroll_and_wobble_freeze_when_stopped(self):
        frame = np.random.default_rng(4).integers(0, 256, (100, 80, 3), dtype=np.uint8)
        for effect_id, rate_id in (("tracking", "tracking.speed"), ("wobble", "wobble.speed")):
            effect = self.effects[effect_id]
            effect.enabled_param().set_base(1)
            before = effect.apply(frame.copy(), 7.25)
            self.params.get(rate_id).set_base(0)
            np.testing.assert_array_equal(effect.apply(frame.copy(), 7.25), before)
            np.testing.assert_array_equal(effect.apply(frame.copy(), 9), before)

    def test_scramble_does_not_remove_accumulated_offset_at_zero_rate(self):
        effect = self.effects["scramble"]
        effect.enabled_param().set_base(1)
        for name in ("amount", "invert", "desat"):
            self.params.get("scramble." + name).set_base(0)
        self.params.get("scramble.roll").set_base(.2)
        moved = effect.apply(self.frame.copy(), 2)
        self.params.get("scramble.roll").set_base(0)
        np.testing.assert_array_equal(effect.apply(self.frame.copy(), 3), moved)
        self.assertFalse(np.array_equal(moved, self.frame))

    def test_automation_integrates_historical_rates_on_seek(self):
        p = self.params.get("tracking.rollspeed")
        timeline = Timeline()
        for t, value in ((0, .5), (2, 0), (4, -.5)):
            timeline.record(t, p.id, value)
        self.params.reset_motion(timeline)
        for t, expected in ((6, 0), (3, .5), (1, .25), (5, .25)):
            timeline.apply(t, self.params)
            self.assertAlmostEqual(self.params.motion_phase(p.id, t, quadratic=True), expected)

    def test_lfo_same_phase_with_frame_drops_and_backward_seek(self):
        p = self.params.get("wobble.speed")
        p.mod = Modulator(enabled=True, depth=1, rate=.7)
        sequential = {}
        for t in np.arange(0, 5, 1/30):
            sequential[t] = self.params.motion_phase(p.id, t)
        self.params.reset_motion()
        for t in list(sequential)[::15][::-1]:
            self.assertAlmostEqual(self.params.motion_phase(p.id, t), sequential[t], places=10)

    def test_legacy_recorder_never_creates_automatic_keys(self):
        p = self.params.get("tracking.rollspeed")
        p.set_base(.25)
        timeline = Timeline()
        recorder = Recorder(self.params, timeline)
        recorder.arm(True)
        recorder.set_time(2)
        p.set_base(.5)
        self.assertEqual(timeline.tracks, {})

    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_export_preserves_preview_position_and_controls(self, _hardware_probe):
        import cv2
        import tempfile
        from pathlib import Path
        from vhs_broadcast.core.pipeline import render_to_file
        p = self.params.get("tracking.rollspeed")
        p.set_base(.5)
        self.effects["tracking"].enabled_param().set_base(1)
        phase = self.params.motion_phase(p.id, 10, quadratic=True)
        timeline = Timeline()
        timeline.record(0, p.id, .2)
        timeline.record(.1, p.id, -.3)
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "in.avi")
            output = str(Path(directory) / "out.mp4")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 100))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(self.frame)
            writer.release()
            render_to_file(source, output, self.params, timeline,
                           list(self.effects.values()), with_audio=False)
            self.assertTrue(Path(output).stat().st_size > 0)
        self.assertEqual(p.value, .5)
        self.assertEqual(self.params.motion_phase(p.id, 10, quadratic=True), phase)


if __name__ == "__main__":
    unittest.main()

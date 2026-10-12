import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from vhs_broadcast.core.automation import Session, Timeline, Track
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import default_effects


class KeyframeModelTests(unittest.TestCase):
    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_manual_curve_renders_to_video(self, _hardware_probe):
        import cv2
        from vhs_broadcast.core.pipeline import render_to_file
        params = ParameterSet()
        effects = default_effects(params)
        for effect in effects:
            effect.enabled_param().set_base(1 if effect.id == "signal" else 0)
        params.get("signal.saturation").set_base(1)
        timeline = Timeline()
        p = params.get("signal.brightness")
        p.set_base(0)
        timeline.set_key(0, p)
        p.set_base(.5)
        timeline.set_key(1, p)
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "in.avi")
            output = str(Path(directory) / "out.mp4")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
            self.assertTrue(writer.isOpened())
            for _ in range(11):
                writer.write(np.full((60, 80, 3), 64, np.uint8))
            writer.release()
            render_to_file(source, output, params, timeline, effects, with_audio=False)
            reader = cv2.VideoCapture(output)
            means = []
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                means.append(float(frame.mean()))
            reader.release()
            self.assertEqual(len(means), 11)
            for index, expected in ((0, 64), (5, 96), (10, 128)):
                self.assertAlmostEqual(means[index], expected, delta=5)

    def test_interpolation_and_endpoints(self):
        track = Track([1, 5], [0, 1], "linear")
        self.assertEqual(track.value_at(0), 0)
        self.assertEqual(track.value_at(10), 1)
        self.assertEqual(track.value_at(2), .25)
        track.interpolation = "smooth"
        self.assertEqual(track.value_at(2), .15625)
        track.interpolation = "hold"
        self.assertEqual(track.value_at(4.9), 0)
        self.assertEqual(track.value_at(5), 1)

    def test_manual_keys_only_and_default_survives_session(self):
        params = ParameterSet()
        effects = default_effects(params)
        timeline = Timeline()
        p = params.get("signal.contrast")
        p.set_base(1.8)
        self.assertFalse(timeline.tracks)
        timeline.set_key(2, p)
        p.set_base(.3)
        p.reset()
        self.assertEqual(p.value, 1)
        self.assertEqual(timeline.tracks[p.id].values, [1.8])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "take.json")
            Session(params, timeline, effects).save(path)
            restored = ParameterSet()
            new_effects = default_effects(restored)
            new_timeline = Timeline()
            Session(restored, new_timeline, new_effects).load(path)
            self.assertEqual(new_timeline.as_dict(), timeline.as_dict())
            restored.get(p.id).set_base(.2)
            restored.get(p.id).reset()
            self.assertEqual(restored.get(p.id).value, 1)

    def test_boolean_keys_hold_and_legacy_import(self):
        params = ParameterSet()
        default_effects(params)
        p = params.get("bloom.enabled")
        timeline = Timeline()
        timeline.set_key(0, p)
        p.set_base(1)
        timeline.set_key(1, p)
        self.assertEqual(timeline.tracks[p.id].interpolation, "hold")
        timeline.apply(.9, params)
        self.assertEqual(p.value, 0)
        old = Timeline.from_dict({"tracks": {p.id: {"t": [0, 1], "v": [0, 1]}}})
        self.assertEqual(old.tracks[p.id].value_at(.5), 0)

    def test_linear_rate_track_integrates_on_seek(self):
        params = ParameterSet()
        default_effects(params)
        timeline = Timeline()
        timeline.tracks["wobble.speed"] = Track([0, 2], [0, 2], "linear")
        params.reset_motion(timeline)
        self.assertAlmostEqual(params.motion_phase("wobble.speed", 2), 2, delta=.005)
        self.assertAlmostEqual(params.motion_phase("wobble.speed", 1), .5, delta=.005)

    def test_tweaking_after_preview_preserves_motion(self):
        params = ParameterSet()
        default_effects(params)
        timeline = Timeline()
        timeline.tracks["wobble.speed"] = Track([0, 5], [2, 2], "linear")
        params.reset_motion(timeline)
        self.assertAlmostEqual(params.motion_phase("wobble.speed", 5), 10)
        params.get("wobble.speed").set_base(1)
        params.resume_live_motion()
        self.assertAlmostEqual(params.motion_phase("wobble.speed", 5), 10)
        self.assertAlmostEqual(params.motion_phase("wobble.speed", 6), 11)

    def test_deletion_and_snapshot_isolation(self):
        timeline = Timeline()
        timeline.record(2, "test", .5)
        previous = timeline.as_dict()
        timeline.record(3, "test", .9)
        self.assertEqual(previous["tracks"]["test"]["t"], [2])
        timeline.delete_key("test", 3)
        timeline.delete_key("test", 2)
        self.assertNotIn("test", timeline.tracks)


class NewEffectTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effects = {e.id: e for e in default_effects(self.params)}

    def test_pixel_sort_respects_dark_boundaries(self):
        frame = np.repeat(np.array([[[0], [200], [100], [250], [0], [240], [150]]], np.uint8), 3, axis=2)
        effect = self.effects["pixel_sort"]
        effect.enabled_param().set_base(1)
        self.params.get("pixel_sort.amount").set_base(1)
        self.params.get("pixel_sort.threshold").set_base(.3)
        self.params.get("pixel_sort.length").set_base(1)
        np.testing.assert_array_equal(effect.apply(frame, 0)[0, :, 0], [0, 100, 200, 250, 0, 150, 240])
        self.params.get("pixel_sort.reverse").set_base(1)
        np.testing.assert_array_equal(effect.apply(frame, 0)[0, :, 0], [0, 250, 200, 100, 0, 240, 150])

    def test_bloom_spreads_only_highlights(self):
        effect = self.effects["bloom"]
        effect.enabled_param().set_base(1)
        frame = np.zeros((40, 80, 3), np.uint8)
        np.testing.assert_array_equal(effect.apply(frame.copy(), 0), frame)
        frame[15:25, 35:45] = 255
        output = effect.apply(frame.copy(), 0)
        self.assertGreater(int(output[15, 34, 0]), 0)

    def test_hue_rotation_and_zero_mix(self):
        effect = self.effects["hue_shift"]
        effect.enabled_param().set_base(1)
        self.params.get("hue_shift.amount").set_base(1)
        self.params.get("hue_shift.angle").set_base(120)
        frame = np.full((4, 4, 3), [0, 0, 255], np.uint8)
        np.testing.assert_allclose(effect.apply(frame.copy(), 0), np.full_like(frame, [0, 255, 0]), atol=1)
        for eid in ("hue_shift", "bloom", "pixel_sort"):
            self.effects[eid].enabled_param().set_base(1)
            self.params.get(eid + ".amount").set_base(0)
            np.testing.assert_array_equal(self.effects[eid].apply(frame.copy(), 0), frame)


class KeyframeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont, QFontDatabase
        cls.app = QApplication.instance() or QApplication([])
        font_id = QFontDatabase.addApplicationFont(str(Path(__file__).resolve().parents[1] /
            "src/vhs_broadcast/assets/fonts/ShareTechMono-Regular.ttf"))
        cls.app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 12))

    def test_chart_add_edit_drag_delete_undo_and_double_click_reset(self):
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtTest import QTest
        from vhs_broadcast.ui.main_window import MainWindow
        from vhs_broadcast.ui.theme import STYLESHEET
        window = MainWindow()
        try:
            editor = window.keyframes
            editor.time_provider = lambda: 2.0
            editor.control.setCurrentIndex(editor.control.findData("signal.contrast"))
            param = window.params.get("signal.contrast")
            row = next(r for panel in window.panels for r in panel.rows if r.param is param)
            param.set_base(1.7)
            row.refresh()
            self.assertFalse(window.timeline.tracks)
            QTest.mouseClick(editor.add_btn, Qt.LeftButton)
            self.assertEqual(window.timeline.tracks[param.id].times, [2])
            row.slider.resize(180, 20)
            QTest.mouseDClick(row.slider, Qt.LeftButton)
            self.assertEqual(param.value, 1)
            self.assertEqual(window.timeline.tracks[param.id].values, [1.7])
            editor.time_edit.setValue(3)
            editor.value_edit.setValue(.8)
            QTest.mouseClick(editor.apply_btn, Qt.LeftButton)
            self.assertEqual(window.timeline.tracks[param.id].times, [3])
            editor.undo()
            self.assertEqual(window.timeline.tracks[param.id].times, [2])
            editor.redo()
            self.assertEqual(window.timeline.tracks[param.id].times, [3])
            window.show()
            self.app.processEvents()
            self.assertEqual(window.tabs.indexOf(editor), -1)
            self.assertTrue(editor.isVisible())
            sizes = window.left_splitter.sizes()
            self.assertLess(abs(sizes[0] - sizes[1]), 60)
            window.left_splitter.setSizes([480, 320])
            self.app.processEvents()
            adjusted = window.left_splitter.sizes()
            window.timeline_btn.click()
            self.assertFalse(editor.isVisible())
            window.transport.key_btn.click()
            self.app.processEvents()
            self.assertTrue(editor.isVisible())
            self.assertEqual(window.left_splitter.sizes(), adjusted)
            window.filter_search.setText("rolling shutter")
            self.assertEqual([p.effect.id for p in window.panels if not p.isHidden()], ["rolling_shutter"])
            window.filter_search.clear()
            self.assertGreater(editor.mapTo(window, QPoint(0, 0)).y(), window.video.mapTo(window, QPoint(0, 0)).y() + window.video.height())
            start = editor.chart.point(param.id, 3, .8).toPoint()
            end = editor.chart.point(param.id, 4, 1.2).toPoint()
            QTest.mousePress(editor.chart, Qt.LeftButton, pos=start)
            QTest.mouseMove(editor.chart, end)
            QTest.mouseRelease(editor.chart, Qt.LeftButton, pos=end)
            self.assertAlmostEqual(window.timeline.tracks[param.id].times[0], 4)
            self.assertAlmostEqual(window.timeline.tracks[param.id].values[0], 1.2, delta=.04)
            editor.delete_key()
            self.assertFalse(window.timeline.tracks)
            editor.undo()
            self.assertIn(param.id, window.timeline.tracks)
            window.transport.pb_btn.setChecked(True)
            param.set_base(1.3)
            self.assertFalse(window.transport.pb_btn.isChecked())
            self.assertEqual(len(window.timeline.tracks[param.id].times), 1)
            if os.environ.get("VHS_KEYFRAME_SCREENSHOT"):
                for pid, values in (("bloom.amount", [.1, .7, .25]), ("tracking.rollspeed", [0, .3, 0]),
                                    ("pixel_sort.amount", [0, .8, .1]), ("hue_shift.angle", [0, 120, -60])):
                    window.timeline.tracks[pid] = Track([0, 3, 7], values, "smooth")
                editor.setStyleSheet(STYLESHEET + '\n* { font-family: "Share Tech Mono"; }')
                editor.refresh()
                self.app.processEvents()
                editor.grab().save(os.environ["VHS_KEYFRAME_SCREENSHOT"])
            if os.environ.get("VHS_LAYOUT_SCREENSHOT"):
                window.setStyleSheet(STYLESHEET + '\n* { font-family: "Share Tech Mono"; }')
                self.app.processEvents()
                window.grab().save(os.environ["VHS_LAYOUT_SCREENSHOT"])
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()

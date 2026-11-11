"""Control undo, the compare split and the seamless-loop feedback."""
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vhs_broadcast.core.history import ControlHistory
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.advanced import VideoFeedbackEffect


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class HistoryTests(unittest.TestCase):
    def make(self):
        self.state = {"x": 0}
        self.clock = Clock()
        return ControlHistory(lambda: dict(self.state), lambda s: self.state.update(s), clock=self.clock)

    def change(self, history, value, dt=.05):
        self.clock.now += dt
        self.state["x"] = value
        history.note_change()

    def test_a_drag_is_one_step_and_redo_restores_it(self):
        h = self.make()
        for value in range(1, 20):                         # one drag, many changes
            self.change(h, value)
        self.clock.now += 1; h.settle()
        self.change(h, 50)                                 # a second gesture
        self.assertTrue(h.undo()); self.assertEqual(self.state["x"], 19)
        self.assertTrue(h.undo()); self.assertEqual(self.state["x"], 0)
        self.assertFalse(h.undo())
        self.assertTrue(h.redo()); self.assertEqual(self.state["x"], 19)
        self.assertTrue(h.redo()); self.assertEqual(self.state["x"], 50)
        self.assertFalse(h.redo())

    def test_new_change_clears_redo_and_reset_forgets(self):
        h = self.make()
        self.change(h, 1); self.clock.now += 1; h.settle()
        h.undo()
        self.change(h, 7)
        self.assertFalse(h.redo())
        h.reset()
        self.assertFalse(h.undo())


def feedback(**values):
    params = ParameterSet()
    effect = VideoFeedbackEffect(params)
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"feedback.{key}").set_base(value)
    return params, effect


def picture():
    img = np.zeros((90, 120, 3), np.uint8)
    img[30:60, 40:80] = (40, 200, 240)
    img[:, :10] = 255
    return img


class FeedbackTests(unittest.TestCase):
    def test_defaults_keep_the_original_tunnel(self):
        import cv2
        _, e = feedback()
        frame = picture()
        first = e.apply(frame, 0)
        np.testing.assert_array_equal(first, frame)
        second = e.apply(frame, 1 / 30)
        h, w = frame.shape[:2]
        m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 1, 1.02)
        trail = cv2.warpAffine(first, m, (w, h), borderMode=cv2.BORDER_WRAP).astype(np.float32) * .65 * .96
        f = frame.astype(np.float32)
        expected = np.clip(255 - (255 - f) * (255 - np.minimum(trail, 255)) / 255, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(second, expected)

    def test_loop_repeats_after_its_length(self):
        _, e = feedback(loop=2, drift=.1, spin=10, shake=.5, gain=.8)
        frame = picture()
        frames = [e.apply(frame, n / 30) for n in range(70)]
        diff = np.abs(frames[60].astype(int) - frames[0]).mean()
        self.assertLess(diff, 1.0)                         # t = 2 s looks like t = 0
        self.assertGreater(np.abs(frames[30].astype(int) - frames[0]).mean(), 2)   # but it moved in between

    def test_keep_subject_holds_the_bright_subject_solid(self):
        frame = picture()
        _, plain = feedback(gain=1, decay=1, blend=1)
        _, keyed = feedback(gain=1, decay=1, blend=1, keep=.3)
        for n in range(5):
            a, b = plain.apply(frame, n / 30), keyed.apply(frame, n / 30)
        white = frame[:, :10]
        self.assertLess(np.abs(b[:, :10].astype(int) - white).mean(), np.abs(a[:, :10].astype(int) - white).mean() + 1)
        np.testing.assert_array_equal(b[5:85, 2:8], frame[5:85, 2:8])


class WindowTests(unittest.TestCase):
    def test_undo_restores_a_control_and_compare_toggles(self):
        from PySide6.QtWidgets import QApplication
        from vhs_broadcast.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            param = window.params.get("scanlines.strength")
            before = param.value
            param.set_base(.9)
            window.history.settle(force=True)
            window._undo(False)
            self.assertEqual(window.params.get("scanlines.strength").value, before)
            window._undo(True)
            self.assertEqual(window.params.get("scanlines.strength").value, .9)
            window.compare_btn.setChecked(True)
            self.assertTrue(window.engine.compare and window.video.compare)
            window.compare_btn.setChecked(False)
            self.assertFalse(window.engine.compare or window.video.compare)
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from vhs_broadcast.core.audio_react import AudioAnalysis, set_analysis
from vhs_broadcast.core.parameters import AudioMod, Parameter, ParameterSet
from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.effects.vhs import default_effects


class AudioTests(unittest.TestCase):
    def tearDown(self):
        set_analysis(None)

    def test_frequency_isolation_and_silence(self):
        sr = 48000
        t = np.arange(sr * 2) / sr
        a = AudioAnalysis(np.sin(2 * np.pi * 100 * t).astype(np.float32), sr, 30, 60)
        self.assertGreater(a.selected_level("low", 1, 50, 150), .5)
        self.assertLess(a.selected_level("high", 1, 2000, 8000), .01)
        self.assertEqual(a.selected_level("high", 1, 30000, 40000), 0)
        silent = AudioAnalysis(np.zeros(sr), sr, 30, 30)
        self.assertEqual(silent.activation("low", .5, 20, 250, 0, 0), 0)

    def test_gate_ramp_and_seek_determinism(self):
        a = AudioAnalysis(np.zeros(48000), 48000, 10, 10)
        a.level = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
        values = [a.activation("level", i / 10, 0, 20000, .5, .3) for i in range(10)]
        np.testing.assert_allclose(values, [0, 0, 1/3, 2/3, 1, 1, 2/3, 1/3, 0, 0], atol=1e-6)
        for i in [8, 2, 5, 1, 3]:
            self.assertAlmostEqual(a.activation("level", i / 10, 0, 20000, .5, .3), values[i])
        self.assertEqual(a.activation("level", .2, 0, 20000, .5, 0), 1)
        set_analysis(a)
        p = Parameter("x", "X", 8, 0, 10, audio=AudioMod(enabled=True, band="level", rest=2, ramp=.3))
        self.assertAlmostEqual(p.evaluate(.3), 6, places=5)
        self.assertEqual(p.evaluate(0), 2)

    def test_no_pretrigger_and_stereo(self):
        sr = 48000
        samples = np.zeros((sr * 2, 2), np.float32)
        samples[sr:] = np.sin(2 * np.pi * 100 * np.arange(sr) / sr)[:, None]
        a = AudioAnalysis(samples, sr, 30, 60)
        self.assertEqual(a.activation("low", .99, 20, 250, .2, 0), 0)
        self.assertGreater(a.activation("low", 1.2, 20, 250, .2, 0), 0)

    def test_legacy_and_new_roundtrip(self):
        old = AudioMod.from_dict({"enabled": True, "gain": 2, "depth": .4})
        self.assertEqual(old.mode, "envelope")
        new = AudioMod(enabled=True, low_hz=80, high_hz=160, ramp=.4, rest=.1)
        self.assertEqual(AudioMod.from_dict(new.as_dict()), new)


class EffectTests(unittest.TestCase):
    @patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
    def test_headless_export(self, _hardware_probe):
        import cv2
        from vhs_broadcast.render import main
        with tempfile.TemporaryDirectory() as d:
            source = str(Path(d) / "source.avi")
            output = str(Path(d) / "output.mp4")
            take = str(Path(d) / "take.json")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(5):
                frame = np.full((120, 160, 3), 40 + i * 30, np.uint8)
                frame[:, 60:90] = 240
                writer.write(frame)
            writer.release()
            params = ParameterSet()
            effects = default_effects(params)
            for effect in effects:
                if effect.id in ("edge_ring", "solarize", "signal_fold"):
                    effect.enabled_param().set_base(1)
            effects[0].manual_clock = True
            effects[0].clock_mode = effects[0].CLOCK_DATETIME
            session = Session(params, Timeline(), effects)
            session.source_path = source
            session.order = [e.id for e in reversed(effects)]
            session.save(take)
            self.assertEqual(main(["--session", take, "--output", output]), 0)
            reader = cv2.VideoCapture(output)
            count = 0
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (120, 160, 3))
                count += 1
            reader.release()
            self.assertEqual(count, 5)

    def test_effects_deterministic_and_bypass(self):
        effects = default_effects(ParameterSet())
        frame = np.random.default_rng(12).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for effect in effects:
            if effect.id not in ("edge_ring", "solarize", "signal_fold"):
                continue
            np.testing.assert_array_equal(effect.apply(frame.copy(), 1), frame)
            effect.enabled_param().set_base(1)
            first = effect.apply(frame.copy(), 1)
            self.assertEqual(first.shape, frame.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertTrue(np.any(first != frame))
            np.testing.assert_array_equal(first, effect.apply(frame.copy(), 1))
            effect.params.get(effect.id + ".amount").set_base(0)
            np.testing.assert_array_equal(effect.apply(frame.copy(), 1), frame)

    def test_manual_clock_and_session(self):
        p = ParameterSet()
        effects = default_effects(p)
        text = effects[0]
        text.manual_clock = True
        text.manual_datetime = "1999-12-31T23:59:59"
        text.clock_mode = text.CLOCK_DATETIME
        self.assertIn("01/01/2000 00:00:01", [i["text"] for i in text.build_items(2, 720, 480)])
        text.clock_run = False
        self.assertIn("12/31/1999 23:59:59", [i["text"] for i in text.build_items(2, 720, 480)])
        p.get("solarize.amount").audio = AudioMod(enabled=True, low_hz=100, high_hz=400)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "take.json")
            session = Session(p, Timeline(), effects)
            session.order = [e.id for e in reversed(effects)]
            session.save(path)
            other = ParameterSet()
            other_effects = default_effects(other)
            restored = Session(other, Timeline(), other_effects)
            restored.load(path)
            self.assertEqual(other_effects[0].settings(), text.settings())
            self.assertEqual(restored.order, session.order)
            self.assertEqual(other.get("solarize.amount").audio, p.get("solarize.amount").audio)


class FontPortabilityTests(unittest.TestCase):
    def test_take_font_from_another_computer_keeps_face_and_size(self):
        from PIL import ImageFont
        from vhs_broadcast.effects.text_render import _load_font, available_fonts, default_font, resolve_font
        vt323 = available_fonts()["VT323"]
        for foreign in ("C:\\Users\\someone\\VHS-BROADCAST\\src\\vhs_broadcast\\assets\\fonts\\VT323-Regular.ttf",
                        "/home/other/VHS-BROADCAST/src/vhs_broadcast/assets/fonts/VT323-Regular.ttf"):
            self.assertEqual(resolve_font(foreign), vt323)
        self.assertEqual(resolve_font("/missing/NotBundled.ttf"), default_font())
        self.assertEqual(resolve_font(vt323), vt323)
        font = _load_font("/missing/NotBundled.ttf", 96)
        self.assertIsInstance(font, ImageFont.FreeTypeFont)
        self.assertEqual(font.size, 96)


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont, QFontDatabase
        cls.app = QApplication.instance() or QApplication([])
        font_id = QFontDatabase.addApplicationFont(str(Path(__file__).resolve().parents[1] /
            "src/vhs_broadcast/assets/fonts/ShareTechMono-Regular.ttf"))
        cls.app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 12))

    def test_editor_and_cpu_window(self):
        from PySide6.QtWidgets import QDoubleSpinBox, QComboBox
        from vhs_broadcast.ui.main_window import MainWindow
        window = MainWindow()
        try:
            self.assertFalse(hasattr(window.transport, "gpu_btn"))
            row = window.panels[0].rows[0]
            editor = row._build_audio()
            from vhs_broadcast.ui.theme import STYLESHEET
            editor.setStyleSheet(STYLESHEET + '\n* { font-family: "Share Tech Mono"; }')
            editor.show()
            self.app.processEvents()
            band = editor.findChild(QComboBox)
            band.setCurrentIndex(2)
            self.assertEqual(row.param.audio.low_hz, 2000)
            self.assertEqual(row.param.audio.high_hz, 16000)
            band.setCurrentIndex(0)
            self.assertEqual(row.param.audio.low_hz, 20)
            self.assertEqual(row.param.audio.high_hz, 250)
            window.manual_clock_cb.setChecked(True)
            self.assertTrue(window.manual_datetime_edit.isEnabled())
            self.assertEqual(window.text_effect.clock_mode, 4)
            if os.environ.get("VHS_QA_SCREENSHOT"):
                editor.grab().save(os.environ["VHS_QA_SCREENSHOT"])
            editor.close()
            editor.deleteLater()
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()

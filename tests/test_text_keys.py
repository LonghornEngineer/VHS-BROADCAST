import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

from vhs_broadcast.api import create_take, validate_take
from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import default_effects


def texts(effect, t):
    return [item["text"] for item in effect.build_items(t, 720, 480)]


class TextKeyTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.effects = default_effects(self.params)
        self.text = next(e for e in self.effects if e.id == "text")
        self.text.manual_clock = True
        self.text.clock_mode = self.text.CLOCK_DATE
        self.text.manual_datetime = "1987-03-14T15:42:00"

    def test_no_keys_keeps_existing_behavior(self):
        self.assertEqual(texts(self.text, 10), ["PLAY ▶", "MAR 14 1987", "SP"])

    def test_keys_hold_and_carry_unkeyed_fields(self):
        self.text.set_text_key(5.0, manual_datetime="1997-03-14T19:20:00")
        self.text.set_text_key(9.0, corner_text="", custom_text="REC")
        self.assertEqual(texts(self.text, 4.99), ["PLAY ▶", "MAR 14 1987", "SP"])
        self.assertEqual(texts(self.text, 5.0), ["PLAY ▶", "MAR 14 1997", "SP"])
        self.assertEqual(texts(self.text, 12), ["REC", "MAR 14 1997"])   # date carries over

    def test_keyed_date_runs_from_its_key_time(self):
        self.text.clock_mode = self.text.CLOCK_DATETIME
        self.text.set_text_key(276.0, manual_datetime="2026-03-14T18:05:00")
        self.assertIn("03/14/2026 18:05:00", texts(self.text, 276.0))
        self.assertIn("03/14/2026 18:05:10", texts(self.text, 286.0))
        self.assertIn("03/14/1987 15:42:10", texts(self.text, 10.0))     # base still runs from 0

    def test_set_key_replaces_same_time_and_sorts(self):
        self.text.set_text_key(8, custom_text="B")
        self.text.set_text_key(2, custom_text="A")
        self.text.set_text_key(8, custom_text="C")
        self.assertEqual([(k["t"], k["custom_text"]) for k in self.text.text_keys], [(2.0, "A"), (8.0, "C")])

    def test_keys_round_trip_through_a_take(self):
        self.text.set_text_key(3, clock_mode=self.text.CLOCK_NONE, corner_text="")
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "take.json")
            Session(self.params, Timeline(), self.effects).save(path)
            other_params = ParameterSet()
            other = default_effects(other_params)
            Session(other_params, Timeline(), other).load(path)
            restored = next(e for e in other if e.id == "text")
            self.assertEqual(restored.text_keys, self.text.text_keys)
            self.assertEqual(texts(restored, 4), ["PLAY ▶"])


class TextKeyApiTests(unittest.TestCase):
    def test_agent_spec_with_text_keys(self):
        spec = {"parameters": {"text.enabled": 1},
                "effect_settings": {"text": {"manual_clock": True, "clock_mode": 3,
                                             "manual_datetime": "1987-03-14T15:42:00",
                                             "text_keys": [{"t": 164.4, "manual_datetime": "1997-03-14T19:20:00"},
                                                           {"t": 276.0, "manual_datetime": "2026-03-14T18:05:00"}]}}}
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "take.json")
            create_take(spec, out)
            self.assertTrue(validate_take(out)["valid"])

    def test_invalid_text_keys_never_write(self):
        bad = [{"t": -1, "custom_text": "X"}, {"custom_text": "no time"}, {"t": 1, "font_path": "x"},
               {"t": 1, "manual_datetime": "not a date"}, {"t": 1, "clock_mode": 9},
               {"t": 1, "custom_text": 5}, {"t": True, "custom_text": "X"}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "take.json"
            for key in bad + ["not a list"]:
                keys = key if isinstance(key, str) else [key]
                with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                    create_take({"effect_settings": {"text": {"text_keys": keys}}}, str(out))
                self.assertFalse(out.exists())


class TextKeyUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_key_button_snapshots_fields_at_playhead(self):
        from unittest.mock import patch
        from vhs_broadcast.ui.main_window import MainWindow
        window = MainWindow()
        try:
            window.text_edit.setText("PLAY ▶")
            window.corner_edit.setText("")
            with patch.object(window.engine, "current_time", return_value=42.0):
                window.text_key_btn.click()
            key = window.text_effect.text_keys[0]
            self.assertEqual((key["t"], key["custom_text"], key["corner_text"]), (42.0, "PLAY ▶", ""))
            self.assertIn("0:42.00", window.text_keys_cb.currentText())
            window.text_key_del_btn.click()
            self.assertEqual(window.text_effect.text_keys, [])
            self.assertFalse(window.text_key_del_btn.isEnabled())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()

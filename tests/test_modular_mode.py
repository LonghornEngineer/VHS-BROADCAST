"""Modular mode: the chain as a rack of modules, sharing every parameter with Classic."""
import unittest


class ModularModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def settle(self):
        for _ in range(3):
            self.app.processEvents()

    def test_library_rack_and_classic_stay_in_sync(self):
        from vhs_broadcast.ui.main_window import MainWindow
        window = MainWindow()
        try:
            window._set_modular(True)
            self.settle()
            self.assertTrue(window.rack.isVisibleTo(window))
            self.assertIs(window.right_stack.currentWidget(), window.library)
            # Patch three modules in from the library.
            for eid in ("bloom", "camcorder", "scanlines"):
                window.library.buttons[eid].click()
            self.settle()
            shown = [card.effect.id for card in window.rack.cards]
            order = [e.id for e in window.effects if e.id in shown]
            self.assertEqual(shown, order)                      # chain order, left to right
            self.assertEqual(set(shown), {"bloom", "camcorder", "scanlines"})
            classic = next(p for p in window.panels if p.effect.id == "bloom")
            self.assertTrue(classic.enable.isChecked())
            # Move the first module later: the chain order changes and undo restores it.
            window.history.settle(force=True)                  # a pause ends the add gesture
            first = shown[0]
            window.rack._move(first, 1)
            self.settle()
            self.assertEqual([c.effect.id for c in window.rack.cards][1], first)
            window._undo(False)
            self.settle()
            self.assertEqual([c.effect.id for c in window.rack.cards], shown)
            # A dial edits the same parameter the Classic slider shows.
            card = next(c for c in window.rack.cards if c.effect.id == "bloom")
            dial = next(d for d in card.dials if d.param.id == "bloom.amount")
            dial.knob.setValue(.8)
            self.assertAlmostEqual(window.params.get("bloom.amount").value, .8)
            window._sync_panels(force=True)
            row = next(r for r in classic.rows if r.param.id == "bloom.amount")
            self.assertEqual(row.value_lbl.text(), "0.80")
            # Taking a module out turns the effect off everywhere.
            card.removeRequested.emit("bloom")
            self.settle()
            self.assertNotIn("bloom", [c.effect.id for c in window.rack.cards])
            self.assertFalse(classic.enable.isChecked())
            self.assertFalse(window.library.buttons["bloom"].isChecked())
            window._set_modular(False)
            self.assertFalse(window.rack.isVisibleTo(window))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()

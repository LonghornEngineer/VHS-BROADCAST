import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from vhs_broadcast.core.automation import Timeline, Track
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.vhs import default_effects


def item(effect, text, t=0):
    return next(i for i in effect.build_items(t, 1000, 500) if i["text"] == text)


class OsdPositionTests(unittest.TestCase):
    def setUp(self):
        self.params = ParameterSet()
        self.text = next(e for e in default_effects(self.params) if e.id == "text")
        self.text.manual_clock = True
        self.text.clock_mode = self.text.CLOCK_DATE
        self.text.manual_datetime = "1987-03-14T15:42:00"

    def test_defaults_match_original_layout(self):
        clock, tag = item(self.text, "MAR 14 1987"), item(self.text, "SP")
        self.assertEqual((clock["x"], clock["y"], clock["anchor"]), (970, 30, "ra"))
        self.assertEqual((tag["x"], tag["y"], tag["anchor"]), (30, 30, "la"))

    def test_clock_and_tag_move(self):
        self.params.get("text.clock_x").set_base(0.85)
        self.params.get("text.clock_y").set_base(0.1)
        self.params.get("text.tag_x").set_base(0.15)
        self.params.get("text.tag_y").set_base(0.9)
        clock, tag = item(self.text, "MAR 14 1987"), item(self.text, "SP")
        self.assertEqual((clock["x"], clock["y"]), (850, 50))
        self.assertEqual((tag["x"], tag["y"]), (150, 450))

    def test_positions_follow_keyframes(self):
        timeline = Timeline()
        timeline.tracks["text.clock_x"] = Track(times=[0, 10], values=[0.85, 0.97], interpolation="linear")
        timeline.apply(5, self.params)
        self.assertEqual(item(self.text, "MAR 14 1987", 5)["x"], 910)


if __name__ == "__main__":
    unittest.main()

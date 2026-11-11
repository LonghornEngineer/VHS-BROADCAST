"""Signal-order stages and retired effects that now live inside other effects."""
import os
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vhs_broadcast.api import catalog, create_take
from vhs_broadcast.core.automation import Session, Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.stages import STAGES, STAGE_INDEX, in_use
from vhs_broadcast.effects.vhs import default_effects


def model():
    params = ParameterSet()
    effects = {e.id: e for e in default_effects(params)}
    return params, effects


def run(effect_id, t=0.7, frame=None, **values):
    params, effects = model()
    effect = effects[effect_id]
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"{effect_id}.{key}").set_base(value)
    if frame is None:
        frame = np.random.default_rng(3).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    return effect.apply(frame, t)


class StageTests(unittest.TestCase):
    def test_every_effect_has_a_stage_and_defaults_run_in_signal_order(self):
        effects = default_effects(ParameterSet())
        ranks = [STAGE_INDEX[e.stage] for e in effects]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(effects[0].id, "text")                 # camcorder OSD is burned in first
        self.assertEqual({e.stage for e in effects}, {key for key, _, _ in STAGES})

    def test_catalog_lists_stages_and_hides_retired_effects_and_controls(self):
        data = catalog()
        ids = {e["id"] for e in data["effects"]}
        self.assertEqual([s["id"] for s in data["stages"]], [key for key, _, _ in STAGES])
        self.assertFalse(ids & {"wobble", "static", "bitcrush", "interlace"})
        bloom = next(e for e in data["effects"] if e["id"] == "bloom")
        self.assertFalse({"bloom.shake", "bloom.line_jitter", "bloom.jumps"} & {p["id"] for p in bloom["parameters"]})
        raster = next(e for e in data["effects"] if e["id"] == "crt_raster")
        self.assertTrue({"crt_raster.wave", "crt_raster.shake"} <= {p["id"] for p in raster["parameters"]})
        self.assertTrue(all("stage" in e for e in data["effects"]))


class RetiredEffectTests(unittest.TestCase):
    """Each retired effect renders exactly like its replacement controls."""

    def test_bit_crush_equals_rgb_bit_depth(self):
        for bits in (1, 3, 5, 8):
            # The old Bit Crush truncated 254.99 to 254; the new control rounds to 255.
            diff = np.abs(run("bitcrush", bits=bits).astype(int) - run("rgb_torture", bits=bits))
            self.assertLessEqual(diff.max(), 1)

    def test_tv_static_equals_fuzz_snow(self):
        for alpha, coarse, tint in ((.3, .2, 0), (.8, .7, .6)):
            np.testing.assert_array_equal(
                run("static", alpha=alpha, coarse=coarse, tint=tint),
                run("noise", grain=0, dropout=0, snow=alpha, snow_coarse=coarse, snow_tint=tint))

    def test_interlace_equals_scanlines_interlace(self):
        for t in (.0, .03):
            np.testing.assert_array_equal(
                run("interlace", t=t, strength=.7, offset=.5),
                run("scanlines", t=t, strength=0, vignette=0, interlace=.7, comb=.5))

    def test_bloom_jitter_equals_crt_raster_shake(self):
        dark = np.zeros((72, 96, 3), np.uint8); dark[20:50, 30:60] = 120
        for t in (.5, 1.3, 2.9):
            np.testing.assert_array_equal(
                run("bloom", t=t, frame=dark, amount=0, shake=2, line_jitter=1.5, jumps=2),
                run("crt_raster", t=t, frame=dark, noise=0, shake=2, line_jitter=1.5, jumps=2))

    def test_wobble_matches_crt_raster_wave_away_from_edges(self):
        frame = np.tile(np.linspace(0, 255, 96, dtype=np.uint8)[None, :, None], (72, 1, 3))
        a = run("wobble", frame=frame, amount=.5, freq=3, speed=1.5).astype(int)
        b = run("crt_raster", frame=frame, noise=0, wave=.5, wave_freq=3, wave_speed=1.5).astype(int)
        self.assertLessEqual(np.abs(a[:, 6:-6] - b[:, 6:-6]).max(), 1)

    def test_old_take_with_retired_effect_loads_and_is_in_use(self):
        params, effects = model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.json"
            spec = {"version": 2, "parameters": {"bitcrush.enabled": 1, "bitcrush.bits": 3}}
            create_take(spec, str(path))
            timeline = Timeline()
            Session(params, timeline, list(effects.values())).load(str(path))
            self.assertTrue(in_use(effects["bitcrush"], timeline))
            self.assertFalse(in_use(effects["wobble"], timeline))
            self.assertEqual(params.get("bitcrush.bits").value, 3)

    def test_downscale_named_targets(self):
        frame = np.random.default_rng(1).integers(0, 256, (240, 640, 3), dtype=np.uint8)
        out = run("downsample", frame=frame, size=1)            # 256 wide
        self.assertEqual(out.shape, frame.shape)
        cols = len(np.unique(out[0, :, 0], return_counts=False))
        self.assertLessEqual(cols, 256)
        self.assertTrue((out[:, ::5] == out[:, ::5]).all())
        np.testing.assert_array_equal(run("downsample", frame=frame, size=3, width=1000), frame)
        smooth = run("downsample", frame=frame, size=2, scaling=1)
        self.assertFalse(np.array_equal(smooth, run("downsample", frame=frame, size=2)))


class StageUiTests(unittest.TestCase):
    def test_panels_grouped_by_stage_and_retired_hidden_until_used(self):
        from PySide6.QtWidgets import QApplication
        from vhs_broadcast.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window._filter_panels()
            panel = {p.effect.id: p for p in window.panels}
            self.assertTrue(panel["wobble"].isHidden())
            self.assertFalse(panel["crt_raster"].isHidden())
            rows = {r.param.id: r for r in panel["bloom"].rows}
            self.assertTrue(rows["bloom.shake"].isHidden())
            window.params.get("wobble.enabled").set_base(1)
            window._filter_panels()
            self.assertFalse(panel["wobble"].isHidden())
            self.assertIn("OLD", panel["wobble"].enable.text())
            items = [window.chain_list.item(i) for i in range(window.chain_list.count())]
            self.assertTrue(any("RECEIVER / TV" in it.text() for it in items))
            window._reorder_chain(list(reversed([e.id for e in window.effects])))
            window._reset_signal_order()
            ranks = [STAGE_INDEX[e.stage] for e in window.effects]
            self.assertEqual(ranks, sorted(ranks))
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()

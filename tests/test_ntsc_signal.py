"""The simulated NTSC composite / VHS tape signal path."""
import unittest

import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects import ntsc as N
from vhs_broadcast.effects.ntsc import NTSCSignalEffect

QUIET = dict(noise=0, snow=0, head_switch=0, chroma_noise=0, luma_noise=0,
             phase_noise=0, chroma_loss=0, edge_wave=0)


def make(**values):
    params = ParameterSet()
    effect = NTSCSignalEffect(params)
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"ntsc.{key}").set_base(value)
    return params, effect


def bars(h=96, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    colours = [(192, 192, 192), (0, 192, 192), (192, 192, 0), (0, 192, 0), (192, 0, 192), (0, 0, 192), (192, 0, 0)]
    for k, c in enumerate(colours):
        img[:, k * w // 7:(k + 1) * w // 7] = c
    return img


class FilterTests(unittest.TestCase):
    def test_low_passes_keep_dc_and_notch_kills_the_subcarrier(self):
        flat = np.full((2, 256), .7, np.float32)
        for ba in (N._butterworth(1e6, N.SAMPLE_RATE), N._one_pole(1e6, N.SAMPLE_RATE), N._notch(.5, 2)):
            np.testing.assert_allclose(N.run_rows(flat, ba, start="edge")[:, 32:], .7, atol=1e-4)
        carrier = np.tile(np.array([1, 0, -1, 0], np.float32), (2, 64))
        self.assertLess(np.abs(N.run_rows(carrier, N._notch(.5, 2))[:, 64:]).max(), .02)

    def test_modulation_round_trip_recovers_colour(self):
        h, w = 4, 256
        phase = NTSCSignalEffect._phase(h, w, 0)
        y, i, q = (np.full((h, w), v, np.float32) for v in (.5, .2, -.1))
        signal = y + i * N._I_WAVE[phase] + q * N._Q_WAVE[phase]
        dy, di, dq = NTSCSignalEffect._decode(signal, 0, phase, h, w)
        mid = slice(32, -32)
        self.assertAlmostEqual(float(dy[:, mid].mean()), .5, places=2)
        self.assertAlmostEqual(float(di[:, mid].mean()), .2, places=2)
        self.assertAlmostEqual(float(dq[:, mid].mean()), -.1, places=2)


class EffectTests(unittest.TestCase):
    def test_repeatable_shape_and_mix_bypass(self):
        p, e = make()
        frame = bars()
        out = e.apply(frame, 1.0)
        self.assertEqual((out.shape, out.dtype), (frame.shape, np.uint8))
        np.testing.assert_array_equal(e.apply(frame, 1.0), out)
        self.assertFalse(np.array_equal(e.apply(frame, 1.1), out))     # noise changes per frame
        p.get("ntsc.amount").set_base(0)
        np.testing.assert_array_equal(e.apply(frame, 1.0), frame)

    def test_grey_stays_neutral_and_levels_hold(self):
        _, e = make(**QUIET)
        grey = np.full((96, 160, 3), 150, np.uint8)
        out = e.apply(grey, 0).astype(int)[:, 40:-40]
        self.assertLessEqual(np.abs(out - 150).max(), 3)

    def test_slower_tape_smears_colour_further_and_keeps_saturation(self):
        import cv2
        frame = np.zeros((8, 320, 3), np.uint8); frame[:, 160:] = (0, 0, 220)   # red block
        yiq = cv2.transform(frame[..., ::-1].astype(np.float32) / 255, N.RGB_TO_YIQ)
        def colour_edge(speed):
            params, e = make(tape=speed, vertical_blend=0, fields=2, **QUIET)
            v = {p.id.split(".")[-1]: p.evaluate(0) for p in e.own_params()}
            i = e._signal(yiq, v, 0, 1.0, 0)[4, :, 1]
            settled = i[260]
            self.assertAlmostEqual(float(settled), float(yiq[4, 260, 1]), delta=.03)
            return int(np.argmax(i > .9 * settled) - np.argmax(i > .1 * settled))
        self.assertGreater(colour_edge(3), colour_edge(1))
        self.assertGreater(colour_edge(1), colour_edge(0))

    def test_one_field_mode_doubles_lines_and_head_switch_stays_at_bottom(self):
        _, e = make(fields=1, **QUIET)
        out = e.apply(bars(), 0)
        np.testing.assert_array_equal(out[0::2], out[1::2])
        _, clean = make(fields=2, **QUIET)
        _, switched = make(fields=2, **{**QUIET, "head_switch": 8})
        frame = bars(480, 160)
        diff = np.abs(clean.apply(frame, 0).astype(int) - switched.apply(frame, 0)).max(axis=(1, 2))
        self.assertTrue((diff[:470] == 0).all())
        self.assertGreater(diff[472:].max(), 20)

    def test_tall_frames_run_at_480_lines_unless_full_frame(self):
        frame = np.random.default_rng(2).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        _, fast = make(**QUIET)
        out = fast.apply(frame, 0)
        self.assertEqual(out.shape, frame.shape)
        _, full = make(resolution=1, **QUIET)
        self.assertEqual(full.apply(frame, 0).shape, frame.shape)
        self.assertFalse(np.array_equal(out, full.apply(frame, 0)))

    def test_registered_in_tape_stage(self):
        from vhs_broadcast.effects.vhs import default_effects
        effect = next(e for e in default_effects(ParameterSet()) if e.id == "ntsc")
        self.assertEqual(effect.stage, "tape")


if __name__ == "__main__":
    unittest.main()

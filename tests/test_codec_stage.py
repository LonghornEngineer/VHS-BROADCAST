"""The real-codec Digital stage."""
import unittest

import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.codec import CodecEffect, FORMATS


def make(**values):
    params = ParameterSet()
    effect = CodecEffect(params)
    effect.enabled_param().set_base(1)
    for key, value in values.items():
        params.get(f"codec.{key}").set_base(value)
    return params, effect


def textured(seed=0, shift=0, h=120, w=160):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    return np.roll(img, shift, axis=1)


def run(effect, frames, fps=30):
    return [effect.apply(f, n / fps) for n, f in enumerate(frames)]


class CodecTests(unittest.TestCase):
    def test_every_format_round_trips_and_repeats(self):
        frames = [textured(shift=n * 2) for n in range(6)]
        for index, fmt in enumerate(FORMATS):
            with self.subTest(format=fmt.label):
                _, a = make(format=index)
                _, b = make(format=index)
                out_a, out_b = run(a, frames), run(b, frames)
                self.assertEqual((out_a[-1].shape, out_a[-1].dtype), (frames[-1].shape, np.uint8))
                for x, y in zip(out_a, out_b):
                    np.testing.assert_array_equal(x, y)
                # A still picture survives the round trip recognisably (moving frames may be held).
                _, still = make(format=index)
                kept = run(still, [frames[0]] * 6)[-1]
                self.assertLess(np.abs(kept.astype(int) - frames[0]).mean(), 35)

    def test_quality_changes_compression_damage(self):
        frames = [textured(shift=n * 2) for n in range(4)]
        err = []
        for quality in (0, 1):
            _, e = make(quality=quality, fps=30)
            err.append(np.abs(run(e, frames)[-1].astype(int) - frames[-1]).mean())
        self.assertGreater(err[0], err[1] + 1)

    def test_low_frame_rate_holds_and_backward_seek_restarts(self):
        _, e = make(fps=10)
        a = e.apply(textured(1), 0.0)
        np.testing.assert_array_equal(e.apply(textured(2), 0.05), a)           # same 1/10 s tick
        self.assertFalse(np.array_equal(e.apply(textured(2), 0.1), a))
        fresh = make(fps=10)[1].apply(textured(1), 0.0)
        np.testing.assert_array_equal(e.apply(textured(1), 0.0), fresh)        # seek back = new tape

    def test_datamosh_keeps_old_picture_through_a_cut(self):
        before = [textured(1, n) for n in range(10)]
        after = [textured(2, n) for n in range(10, 20)]
        def last(datamosh):
            _, e = make(fps=30, keyframe=10, datamosh=datamosh, quality=.8)
            return run(e, before + after)[-1].astype(int)
        target = after[-1].astype(int)
        self.assertGreater(np.abs(last(1) - target).mean(), np.abs(last(0) - target).mean() + 10)

    def test_packet_damage_is_seeded_and_mix_bypasses(self):
        frames = [textured(shift=n * 2) for n in range(8)]
        p, e = make(fps=30, errors=1, loss=.3, seed=4)
        damaged = run(e, frames)
        _, again = make(fps=30, errors=1, loss=.3, seed=4)
        np.testing.assert_array_equal(run(again, frames)[-1], damaged[-1])
        _, clean = make(fps=30)
        self.assertFalse(np.array_equal(run(clean, frames)[-1], damaged[-1]))
        p.get("codec.amount").set_base(0)
        np.testing.assert_array_equal(e.apply(frames[0], 1.0), frames[0])

    def test_registered_in_digital_stage(self):
        from vhs_broadcast.effects.vhs import default_effects
        effect = next(e for e in default_effects(ParameterSet()) if e.id == "codec")
        self.assertEqual(effect.stage, "digital")


if __name__ == "__main__":
    unittest.main()

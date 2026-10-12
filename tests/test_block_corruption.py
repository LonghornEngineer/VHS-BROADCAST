import unittest

import numpy as np

from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.effects.advanced import BlockCorruptionEffect
from vhs_broadcast.effects.vhs import default_effects


class BlockCorruptionTests(unittest.TestCase):
    def make(self, **values):
        p = ParameterSet()
        e = BlockCorruptionEffect(p)
        e.enabled_param().set_base(1)
        for name, value in values.items():
            p.get('block_corrupt.' + name).set_base(value)
        return p, e

    def test_no_duplicate_cartridge_filter_and_scramble_is_kept(self):
        effects = default_effects(ParameterSet())
        self.assertNotIn('dirty_cart', [e.id for e in effects])
        self.assertIn('TILE SCRAMBLE', [e.label for e in effects])
        self.assertFalse(any(e.label.startswith('NES') for e in effects))

    def test_tile_replace_reuses_source_pixels_and_small_pattern_bank(self):
        frame = np.random.default_rng(4).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        p, e = self.make(mode=3, block=8, density=1, pool=3, lock=1)
        output = e.apply(frame, 0)
        def tiles(image):
            return image.reshape(8, 8, 12, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 8, 8, 3)
        source = {t.tobytes() for t in tiles(frame)}
        actual = {t.tobytes() for t in tiles(output)}
        self.assertLessEqual(len(actual), 3)
        self.assertTrue(actual.issubset(source))
        np.testing.assert_array_equal(e.apply(frame, .6), output)
        # Layout is locked, but Tile Replace still uses current video pixels.
        changed = 255 - frame
        np.testing.assert_array_equal(e.apply(changed, .7), 255 - output)

    def test_chroma_preserves_clean_pixels_exactly_and_zero_density_bypasses(self):
        frame = np.random.default_rng(8).integers(0, 256, (65, 97, 3), dtype=np.uint8)
        p, e = self.make(mode=3, density=.25, block=8, pool=5, chroma=1)
        output = e.apply(frame, 0)
        self.assertGreater(np.all(output == frame, axis=2).mean(), .6)
        self.assertFalse(np.array_equal(output, frame))
        p.get('block_corrupt.density').set_base(0)
        np.testing.assert_array_equal(e.apply(frame, .1), frame)
        p.get('block_corrupt.density').set_base(1)
        p.get('block_corrupt.amount').set_base(0)
        np.testing.assert_array_equal(e.apply(frame, .2), frame)

    def test_repeat_pool_holds_content_then_refreshes_with_locked_layout(self):
        p, e = self.make(mode=1, block=8, density=1, pool=6, lock=1, hold=.4)
        frame = np.random.default_rng(11).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        output = e.apply(frame, 0)
        np.testing.assert_array_equal(e.apply(255-frame, .2), output)
        np.testing.assert_array_equal(e.apply(255-frame, .5), 255-output)
        np.testing.assert_array_equal(e.apply(255-frame, .5), 255-output)

    def test_cluster_seed_and_refresh_controls(self):
        frame = np.random.default_rng(1).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        p, e = self.make(mode=3, block=8, clusters=1, density=.5, pool=5)
        output = e.apply(frame, 0)
        changed = np.any(output != frame, axis=2).reshape(8, 8, 12, 8).any(axis=(1, 3))
        # All selected neighbors share a damage decision (apart from a copied
        # source tile landing on itself, avoided by this deterministic seed).
        np.testing.assert_array_equal(changed[::2, ::2], changed[1::2, 1::2])
        self.assertFalse(np.array_equal(e.apply(frame, .5), output))
        p.get('block_corrupt.seed').set_base(123)
        self.assertFalse(np.array_equal(e.apply(frame, .5), output))

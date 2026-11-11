"""The analog signal models behind tape and TV static."""
import unittest

import cv2
import numpy as np

from vhs_broadcast.effects.rf import fm_playback, receiver_snow, track_envelope


def picture(h=120, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :w // 2] = (40, 60, 200)                      # a saturated red half
    img[:, w // 2:] = (200, 160, 60)
    return img


class TrackTests(unittest.TestCase):
    def test_envelope_follows_overlap_with_same_azimuth_tracks(self):
        env = track_envelope(np.array([0, .5, 1, 1.5, 2, -1]))
        np.testing.assert_allclose(env, [1, .5, 0, .5, 1, 0], atol=1e-6)
        # A head wider than the track never loses its own track entirely.
        self.assertGreater(float(track_envelope(1.0, head=1.4)), .3)


class FMTests(unittest.TestCase):
    def test_full_signal_passes_the_picture_through(self):
        img = picture()
        out = fm_playback(img, np.ones(img.shape[0]), np.random.default_rng(0))
        np.testing.assert_array_equal(out, img)

    def test_noise_rises_steeply_as_the_rf_falls(self):
        img = np.full((200, 320, 3), 100, np.uint8)
        def damage(level):
            out = fm_playback(img, np.full(200, level), np.random.default_rng(1), doc=False, sync=0)
            return np.abs(out.astype(int) - 100).mean()
        clean, edge, gone = damage(.6), damage(.2), damage(0)
        self.assertLess(clean, 6)                            # fine grain only
        self.assertGreater(edge, clean * 2)                  # clicks near threshold
        self.assertGreater(gone, 30)                         # nothing but noise

    def test_colour_killer_greys_out_weak_lines(self):
        img = picture()
        env = np.ones(img.shape[0]); env[60:] = .05
        out = fm_playback(img, env, np.random.default_rng(2), doc=False, sync=0)
        sat = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 1]
        self.assertGreater(sat[:60].mean(), 100)
        self.assertLess(sat[60:].mean(), sat[:60].mean() / 2)

    def test_dropout_compensator_repeats_the_line_above(self):
        img = np.repeat(np.arange(100, dtype=np.uint8)[:, None, None] * 2, 80, axis=1).repeat(3, axis=2)
        env = np.ones((100, 80)); env[50:52] = 0
        out = fm_playback(img, env, np.random.default_rng(3), sync=0)
        self.assertGreater((out[50] == out[49]).mean(), .8)


class ReceiverTests(unittest.TestCase):
    def test_no_noise_is_untouched_and_no_station_is_grey_snow(self):
        img = picture()
        rng = np.random.default_rng(4)
        np.testing.assert_array_equal(receiver_snow(img, 0, rng), img)
        snow = receiver_snow(img, 1, rng)
        self.assertGreater(snow.std(), 40)
        sat = cv2.cvtColor(snow, cv2.COLOR_BGR2HSV)[..., 1]
        self.assertLess(sat.mean(), 20)                      # colour killer
        confetti = receiver_snow(img, 1, rng, colour=1)
        self.assertGreater(cv2.cvtColor(confetti, cv2.COLOR_BGR2HSV)[..., 1].mean(), sat.mean() + 10)

    def test_weak_signal_keeps_the_picture_under_the_snow(self):
        img = picture()
        out = receiver_snow(img, .25, np.random.default_rng(5))
        self.assertLess(np.abs(cv2.blur(out, (9, 9)).astype(int) - img).mean(), 40)


if __name__ == "__main__":
    unittest.main()

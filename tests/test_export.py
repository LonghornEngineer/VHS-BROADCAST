"""Export formats: H.264, ProRes, FFV1, interlaced masters, repeat and GIF."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from vhs_broadcast.core.automation import Timeline
from vhs_broadcast.core.parameters import ParameterSet
from vhs_broadcast.core.pipeline import render_to_file, weave_fields
from vhs_broadcast.core.video_writer import format_for
from vhs_broadcast.effects.vhs import default_effects


def write_source(path, frames=8, rate=Fraction(60000, 1001)):
    container = av.open(path, "w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for i in range(frames):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 25 * i, np.uint8), format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def video_info(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = [f for f in container.decode(stream)]
        return stream.codec_context.name, stream.codec_context.field_order, frames


@patch("vhs_broadcast.core.pipeline._AvFileSource._detect_hw", return_value=None)
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.source = str(self.root / "source.mp4")
        write_source(self.source)
        self.params = ParameterSet()
        self.effects = default_effects(self.params)
        self.params.get("text.enabled").set_base(0)

    def tearDown(self):
        self.dir.cleanup()

    def render(self, name, **options):
        out = str(self.root / name)
        render_to_file(self.source, out, self.params, Timeline(), self.effects, with_audio=False, **options)
        return out

    def test_extension_picks_the_format(self, _probe):
        self.assertEqual(format_for("a.mov"), "prores")
        self.assertEqual(format_for("a.mkv"), "ffv1")
        self.assertEqual(format_for("a.mp4"), "h264")
        with self.assertRaises(ValueError):
            format_for("a.mp4", "wmv")
        name, _, frames = video_info(self.render("master.mov"))
        self.assertEqual(name, "prores")
        self.assertEqual(len(frames), 8)

    def test_ffv1_is_lossless(self, _probe):
        out = self.render("master.mkv")
        name, _, frames = video_info(out)
        self.assertEqual(name, "ffv1")
        with av.open(self.source) as container:
            source = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
        decoded = frames[3].to_ndarray(format="bgr24")
        np.testing.assert_array_equal(decoded, source[3])

    def test_interlaced_master_weaves_pairs_and_flags_field_order(self, _probe):
        name, order, frames = video_info(self.render("i.mov", interlace="bff"))
        self.assertEqual(len(frames), 4)
        self.assertIn(order, ("bb", 3))
        name, order, frames = video_info(self.render("i.mkv", interlace="tff"))
        self.assertEqual(len(frames), 4)
        self.assertIn(order, ("tt", 2))

    def test_weave_takes_the_first_field_from_the_earlier_frame(self, _probe):
        first, second = np.zeros((4, 2, 3), np.uint8), np.full((4, 2, 3), 9, np.uint8)
        bff = weave_fields(first, second, "bff")
        self.assertTrue((bff[1::2] == 0).all() and (bff[0::2] == 9).all())
        tff = weave_fields(first, second, "tff")
        self.assertTrue((tff[0::2] == 0).all() and (tff[1::2] == 9).all())

    def test_repeat_writes_back_to_back(self, _probe):
        _, _, frames = video_info(self.render("loop.mp4", repeat=3))
        self.assertEqual(len(frames), 24)

    def test_gif(self, _probe):
        from PIL import Image
        out = self.render("clip.gif", repeat=2)
        with Image.open(out) as gif:
            self.assertEqual(gif.n_frames, 2)                 # 59.94 fps capped to 15: every 4th frame
            self.assertEqual(gif.info.get("loop"), 1)
        with self.assertRaises(ValueError):
            self.render("bad.gif", interlace="bff")


if __name__ == "__main__":
    unittest.main()

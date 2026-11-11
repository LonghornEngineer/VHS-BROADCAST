"""Renders must not shift levels (OpenCV's writer used to darken everything by 2-4 steps)."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vhs_broadcast.core.automation import Timeline
from vhs_broadcast.core.parameters import ParameterSet


class RenderLevelTests(unittest.TestCase):
    def test_render_without_effects_keeps_grey_levels(self):
        from vhs_broadcast.core.pipeline import render_to_file
        levels = [16, 64, 128, 192, 235]
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "levels.mkv")
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"FFV1"), 10, (160, 96))   # lossless source
            frame = np.zeros((96, 160, 3), np.uint8)
            for k, level in enumerate(levels):
                frame[:, k * 32:(k + 1) * 32] = level
            for _ in range(6):
                writer.write(frame)
            writer.release()
            output = str(Path(directory) / "out.mp4")
            render_to_file(source, output, ParameterSet(), Timeline(), [], with_audio=False)
            cap = cv2.VideoCapture(output)
            ok, decoded = cap.read()
            cap.release()
        self.assertTrue(ok)
        for k, level in enumerate(levels):
            patch = decoded[30:66, k * 32 + 8:(k + 1) * 32 - 8].astype(float).mean()
            self.assertAlmostEqual(patch, level, delta=1.2, msg=f"level {level} came back as {patch:.1f}")




if __name__ == "__main__":
    unittest.main()

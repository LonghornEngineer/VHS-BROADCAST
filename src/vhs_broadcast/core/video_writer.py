"""Write rendered frames without shifting their levels.

OpenCV's writer (MPEG-4 Part 2 through FFmpeg's default conversion) darkened every
level by 2-4 steps: the RGB to YUV conversion truncates instead of rounding, and
the encode added more. Here the conversion is done with correct rounding (BT.601
for SD, BT.709 for HD, studio-range levels, chroma averaged over each 2x2 block),
encoded as high-quality H.264 and tagged, so players and editors decode exactly
what was rendered.
"""
from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

# Luma weights (Kr, Kb) and the matching container tags (FFmpeg enum values).
_MATRICES = {"bt601": (.299, .114, 6), "bt709": (.2126, .0722, 1)}


def bgr_to_yuv420(bgr: np.ndarray, matrix: str = "bt601") -> np.ndarray:
    """Studio-range YUV 4:2:0 as one (h * 3/2, w) uint8 array, rounded, not truncated."""
    kr, kb, _ = _MATRICES[matrix]
    kg = 1 - kr - kb
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = kr * r + kg * g + kb * b
    u = (b - y) / (2 * (1 - kb))
    v = (r - y) / (2 * (1 - kr))
    h, w = y.shape
    Y = np.clip(np.rint(16 + y * 219 / 255), 0, 255).astype(np.uint8)
    # Average each 2x2 block before quantising, so flat colours stay exact.
    U = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    V = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    U = np.clip(np.rint(128 + U * 224 / 255), 0, 255).astype(np.uint8)
    V = np.clip(np.rint(128 + V * 224 / 255), 0, 255).astype(np.uint8)
    return np.vstack([Y, U.reshape(h // 4, w), V.reshape(h // 4, w)]) if h % 4 == 0 else \
        np.vstack([Y, np.concatenate([U.ravel(), V.ravel()]).reshape(h // 2, w)])


class VideoWriter:
    """Same interface the render used from cv2.VideoWriter: write(bgr), release(), isOpened()."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        w, h = size
        # 4:2:0 needs even dimensions; drop a trailing odd row or column.
        self.size = (w - w % 2, h - h % 2)
        self.matrix = "bt709" if self.size[1] >= 720 else "bt601"
        self._container = av.open(path, "w")
        rate = Fraction(fps).limit_denominator(1001)
        try:
            stream = self._container.add_stream("libx264", rate=rate)
            # mbtree=0: libx264's macroblock-tree pass has AVX-512 routines that
            # read uninitialised scratch memory, so the I-frame's QP offsets (and
            # every frame referencing it) varied with whatever was on the heap.
            # Without it the same frames always encode to the same bitstream.
            stream.options = {"crf": "14", "preset": "medium", "x264-params": "mbtree=0"}
        except (av.FFmpegError, ValueError):
            stream = self._container.add_stream("mpeg4", rate=rate)
            stream.options = {"qscale": "2"}
        stream.width, stream.height = self.size
        stream.pix_fmt = "yuv420p"
        tag = _MATRICES[self.matrix][2]
        ctx = stream.codec_context
        ctx.color_range = 1                       # studio (limited) range
        ctx.colorspace = ctx.color_primaries = ctx.color_trc = tag
        self._stream = stream
        self._index = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def write(self, bgr: np.ndarray) -> None:
        w, h = self.size
        planes = bgr_to_yuv420(np.ascontiguousarray(bgr[:h, :w]), self.matrix)
        frame = av.VideoFrame.from_ndarray(planes, format="yuv420p")
        frame.pts = self._index
        self._index += 1
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self) -> None:
        if not self._open:
            return
        self._open = False
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

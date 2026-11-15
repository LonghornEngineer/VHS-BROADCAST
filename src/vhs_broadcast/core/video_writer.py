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


def _ycbcr(bgr: np.ndarray, matrix: str):
    """Y in 0..255 and Cb/Cr in -127.5..127.5 (full-scale float), per pixel."""
    kr, kb, _ = _MATRICES[matrix]
    kg = 1 - kr - kb
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = kr * r + kg * g + kb * b
    return y, (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))


def _chroma_420(c: np.ndarray, interlaced: bool) -> np.ndarray:
    """Average 2x2 blocks; for interlaced pictures pair lines of the same field
    (0 with 2, 1 with 3) so the fields' colour does not bleed into each other."""
    h, w = c.shape
    if not interlaced:
        return c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    quads = c[: h - h % 4].reshape(h // 4, 2, 2, w // 2, 2)      # (group, line pair, field, x, 2)
    fields = quads.mean(axis=(1, 4))                              # (group, field, x)
    return fields.reshape(-1, w // 2)


def bgr_to_yuv420(bgr: np.ndarray, matrix: str = "bt601", interlaced: bool = False) -> np.ndarray:
    """Studio-range YUV 4:2:0 as one (h * 3/2, w) uint8 array, rounded, not truncated."""
    y, u, v = _ycbcr(bgr, matrix)
    h, w = y.shape
    Y = np.clip(np.rint(16 + y * 219 / 255), 0, 255).astype(np.uint8)
    # Average each 2x2 block before quantising, so flat colours stay exact.
    U = _chroma_420(u, interlaced and h % 4 == 0)
    V = _chroma_420(v, interlaced and h % 4 == 0)
    U = np.clip(np.rint(128 + U * 224 / 255), 0, 255).astype(np.uint8)
    V = np.clip(np.rint(128 + V * 224 / 255), 0, 255).astype(np.uint8)
    return np.vstack([Y, U.reshape(h // 4, w), V.reshape(h // 4, w)]) if h % 4 == 0 else \
        np.vstack([Y, np.concatenate([U.ravel(), V.ravel()]).reshape(h // 2, w)])


def bgr_to_yuv422p10(bgr: np.ndarray, matrix: str):
    """Studio-range 10-bit 4:2:2 planes (Y, Cb, Cr) as uint16 arrays. 4:2:2 only halves
    colour across, so interlaced fields never share colour samples."""
    y, u, v = _ycbcr(bgr, matrix)
    u = u.reshape(u.shape[0], -1, 2).mean(axis=2)
    v = v.reshape(v.shape[0], -1, 2).mean(axis=2)
    Y = np.clip(np.rint(64 + y * 876 / 255), 0, 1023).astype(np.uint16)
    U = np.clip(np.rint(512 + u * 896 / 255), 0, 1023).astype(np.uint16)
    V = np.clip(np.rint(512 + v * 896 / 255), 0, 1023).astype(np.uint16)
    return Y, U, V


def _fill_planes(frame, arrays):
    """Copy arrays into a frame's planes, padding each row to the plane's line size."""
    for plane, array in zip(frame.planes, arrays):
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(array.shape[0], -1)
        buffer = np.zeros((plane.height, plane.line_size), np.uint8)
        buffer[:, :raw.shape[1]] = raw
        plane.update(buffer)


#: Export formats: name -> (codec, container note). The extension picks one when no
#: format is given: .mp4 H.264, .mov ProRes 422 HQ, .mkv FFV1 lossless RGB.
FORMATS = ("h264", "prores", "ffv1")
EXTENSIONS = {".mp4": "h264", ".m4v": "h264", ".mov": "prores", ".mkv": "ffv1", ".avi": "h264"}


def format_for(path: str, requested: str | None = None) -> str:
    if requested:
        if requested not in FORMATS:
            raise ValueError(f"unknown export format {requested!r}; use one of {', '.join(FORMATS)}")
        return requested
    from pathlib import Path
    return EXTENSIONS.get(Path(path).suffix.lower(), "h264")


class VideoWriter:
    """Same interface the render used from cv2.VideoWriter: write(bgr), release(), isOpened().

    `codec` is "h264" (high-quality H.264, the default), "prores" (ProRes 422 HQ, 10-bit)
    or "ffv1" (lossless RGB). `interlace` "bff"/"tff" writes frames that already hold two
    woven fields and flags them interlaced with that field order."""

    def __init__(self, path: str, fps: float, size: tuple[int, int], codec: str = "h264",
                 interlace: str | None = None):
        w, h = size
        # 4:2:0 needs even dimensions; drop a trailing odd row or column.
        self.size = (w - w % 2, h - h % 2)
        self.matrix = "bt709" if self.size[1] >= 720 else "bt601"
        self.codec = codec
        self.interlace = interlace
        self._container = av.open(path, "w")
        rate = Fraction(fps).limit_denominator(1001)
        if codec == "prores":
            stream = self._container.add_stream("prores_ks", rate=rate)
            stream.options = {"profile": "3", "vendor": "apl0"}          # 422 HQ
            stream.pix_fmt = "yuv422p10le"
        elif codec == "ffv1":
            stream = self._container.add_stream("ffv1", rate=rate)
            stream.options = {"level": "3", "slicecrc": "1"}
            stream.pix_fmt = "bgr0"                                     # 8-bit RGB, exactly as rendered
        else:
            try:
                stream = self._container.add_stream("libx264", rate=rate)
                options = {"crf": "14", "preset": "medium"}
                if interlace:
                    options["x264-params"] = "interlaced=1:" + ("bff=1" if interlace == "bff" else "tff=1")
                stream.options = options
            except (av.FFmpegError, ValueError):
                stream = self._container.add_stream("mpeg4", rate=rate)
                stream.options = {"qscale": "2"}
            stream.pix_fmt = "yuv420p"
        stream.width, stream.height = self.size
        tag = _MATRICES[self.matrix][2]
        ctx = stream.codec_context
        if codec != "ffv1":
            ctx.color_range = 1                   # studio (limited) range
            ctx.colorspace = ctx.color_primaries = ctx.color_trc = tag
        else:
            ctx.color_primaries = ctx.color_trc = tag
        # AV_FIELD_PROGRESSIVE / AV_FIELD_TT / AV_FIELD_BB
        ctx.field_order = {None: 1, "tff": 2, "bff": 3}[interlace]
        self._stream = stream
        self._index = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def write(self, bgr: np.ndarray) -> None:
        w, h = self.size
        bgr = np.ascontiguousarray(bgr[:h, :w])
        if self.codec == "ffv1":
            padded = np.zeros((h, w, 4), np.uint8)
            padded[..., :3] = bgr
            frame = av.VideoFrame(w, h, "bgr0")
            _fill_planes(frame, [padded.reshape(h, w * 4)])
        elif self.codec == "prores":
            frame = av.VideoFrame(w, h, "yuv422p10le")
            _fill_planes(frame, bgr_to_yuv422p10(bgr, self.matrix))
        else:
            planes = bgr_to_yuv420(bgr, self.matrix, interlaced=bool(self.interlace))
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

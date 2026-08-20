"""Output sinks for the processed preview frames.

VirtualCamOutput streams the app's output to a system virtual camera (so OBS,
Zoom, Discord, Twitch, etc. can use the VHS look as a live source). It needs a
backend: OBS's virtual camera on Windows/macOS, or v4l2loopback on Linux.

VideoRecorder writes the processed frames straight to an .mp4 (video only) --
handy for quick clips of a live source without spinning up OBS.

Both take BGR uint8 frames (what the engine produces) and both fail gracefully.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import pyvirtualcam
    HAVE_VCAM = True
except Exception:  # noqa: BLE001
    HAVE_VCAM = False


class VirtualCamOutput:
    def __init__(self, fps: float = 30.0):
        self.fps = float(fps)
        self.cam = None
        self._size = None
        self.enabled = False
        self.error: Optional[str] = None
        self.device: Optional[str] = None

    @staticmethod
    def available() -> bool:
        return HAVE_VCAM

    def start(self) -> bool:
        """Validate the backend now (clear error if it's missing) and arm.
        The real camera is (re)created lazily to match the frame size."""
        self.error = None
        if not HAVE_VCAM:
            self.error = "pyvirtualcam is not installed"
            return False
        try:
            test = pyvirtualcam.Camera(width=1280, height=720, fps=self.fps,
                                       fmt=pyvirtualcam.PixelFormat.BGR,
                                       print_fps=False)
            self.device = test.device
            test.close()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            return False
        self.enabled = True
        return True

    def send(self, bgr: np.ndarray) -> None:
        if not self.enabled or not HAVE_VCAM:
            return
        h, w = bgr.shape[:2]
        try:
            if self.cam is None or self._size != (w, h):
                if self.cam is not None:
                    try:
                        self.cam.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.cam = pyvirtualcam.Camera(
                    width=w, height=h, fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
                self.device = self.cam.device
                self._size = (w, h)
            self.cam.send(np.ascontiguousarray(bgr))
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.enabled = False
            try:
                if self.cam:
                    self.cam.close()
            except Exception:  # noqa: BLE001
                pass
            self.cam = None

    def stop(self) -> None:
        self.enabled = False
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:  # noqa: BLE001
                pass
            self.cam = None
            self._size = None


class VideoRecorder:
    """Tees processed frames to an .mp4 (video only) via OpenCV."""

    def __init__(self):
        self.writer = None
        self.path: Optional[str] = None
        self.fps = 30.0
        self._size = None
        self._armed = False

    def start(self, path: str, fps: float) -> None:
        self.path = path
        self.fps = float(fps) or 30.0
        self.writer = None
        self._size = None
        self._armed = True   # writer created lazily on first frame (its size)

    def active(self) -> bool:
        return self._armed or self.writer is not None

    def write(self, bgr: np.ndarray) -> None:
        if not self._armed and self.writer is None:
            return
        h, w = bgr.shape[:2]
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._size = (w, h)
        if (w, h) != self._size:
            bgr = cv2.resize(bgr, self._size)
        self.writer.write(bgr)

    def stop(self) -> Optional[str]:
        self._armed = False
        w = self.writer
        self.writer = None
        if w is not None:
            try:
                w.release()
            except Exception:  # noqa: BLE001
                pass
        return self.path

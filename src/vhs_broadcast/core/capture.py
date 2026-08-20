"""Live input sources: screen / region, window, and camera / capture device.

All are `is_live = True`: infinite, non-seekable, real-time. The engine grabs
one fresh frame per tick for these (no motor time-warp, no scrubbing, no
deterministic re-render). A USB capture dongle shows up as a device source, so a
real VHS deck can be piped straight through the effect stack.

Screen/window capture uses `mss`; window enumeration uses `pygetwindow`
(Windows/macOS). Everything degrades gracefully if a backend is missing.
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np

from .pipeline import VideoSource

try:
    import mss  # screen grab
    HAVE_MSS = True
except Exception:  # noqa: BLE001
    HAVE_MSS = False

try:
    import pygetwindow as _gw   # window geometry (Windows/macOS)
    HAVE_GW = True
except Exception:  # noqa: BLE001
    _gw = None
    HAVE_GW = False


# --------------------------------------------------------------------------- #
#  Screen / region
# --------------------------------------------------------------------------- #
class ScreenSource(VideoSource):
    is_live = True

    def __init__(self, bbox: dict, fps: float = 30.0, label: str = "Screen"):
        # bbox: {"left","top","width","height"}
        self.bbox = dict(bbox)
        self.fps = float(fps)
        self.frame_count = 0
        self.width = int(bbox["width"])
        self.height = int(bbox["height"])
        self.index = 0
        self.label = label
        self._sct = None

    def _ensure(self):
        # mss instances are per-thread; create lazily in the engine's thread
        if self._sct is None:
            self._sct = mss.mss()

    def _grab_box(self) -> dict:
        return self.bbox

    def read(self) -> Optional[np.ndarray]:
        if not HAVE_MSS:
            return None
        try:
            self._ensure()
            box = self._grab_box()
            shot = self._sct.grab(box)
            arr = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height,
                                                             shot.width, 4)
            self.index += 1
            return np.ascontiguousarray(arr[:, :, :3])
        except Exception:  # noqa: BLE001
            return None

    def release(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001
                pass
            self._sct = None


# --------------------------------------------------------------------------- #
#  A specific window (tracks its geometry as it moves/resizes)
# --------------------------------------------------------------------------- #
class WindowSource(ScreenSource):
    def __init__(self, title: str, fps: float = 30.0):
        self._title = title
        box = _window_box(title) or {"left": 0, "top": 0,
                                     "width": 640, "height": 480}
        super().__init__(box, fps=fps, label=title)

    def _grab_box(self) -> dict:
        box = _window_box(self._title)
        if box and box["width"] > 0 and box["height"] > 0:
            self.bbox = box
            self.width, self.height = box["width"], box["height"]
        return self.bbox


def _window_box(title: str) -> Optional[dict]:
    if not HAVE_GW:
        return None
    try:
        wins = _gw.getWindowsWithTitle(title)
        for wobj in wins:
            if getattr(wobj, "width", 0) > 0 and getattr(wobj, "height", 0) > 0:
                return {"left": int(wobj.left), "top": int(wobj.top),
                        "width": int(wobj.width), "height": int(wobj.height)}
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
#  Camera / capture card
# --------------------------------------------------------------------------- #
class DeviceSource(VideoSource):
    is_live = True

    def __init__(self, index: int, fps: float = 30.0):
        self.dev_index = index
        self.cap = _open_device(index)
        if self.cap is None or not self.cap.isOpened():
            raise IOError(f"cannot open capture device {index}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.frame_count = 0
        self.index = 0
        self.label = f"Device {index}"

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self.index += 1
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass


def _open_device(index: int):
    # DirectShow tends to be the most reliable backend for USB cams/dongles
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


# --------------------------------------------------------------------------- #
#  Enumeration for the input picker
# --------------------------------------------------------------------------- #
def list_screens() -> list[dict]:
    """Return [{'label','bbox'}] for the full desktop and each monitor."""
    out = []
    if not HAVE_MSS:
        return out
    try:
        with mss.mss() as sct:
            mons = sct.monitors  # [0]=all, [1..]=each display
            for i, m in enumerate(mons):
                box = {"left": m["left"], "top": m["top"],
                       "width": m["width"], "height": m["height"]}
                label = "All displays" if i == 0 else f"Display {i}"
                out.append({"label": f"{label} ({m['width']}x{m['height']})",
                            "bbox": box})
    except Exception:  # noqa: BLE001
        pass
    return out


def list_windows(limit: int = 40) -> list[str]:
    """Visible window titles (Windows/macOS). Empty where unsupported."""
    if not HAVE_GW:
        return []
    titles = []
    try:
        for wobj in _gw.getAllWindows():
            t = (wobj.title or "").strip()
            if t and getattr(wobj, "width", 0) > 40 and getattr(wobj, "height", 0) > 40:
                if t not in titles:
                    titles.append(t)
            if len(titles) >= limit:
                break
    except Exception:  # noqa: BLE001
        return []
    return titles


def list_devices(max_probe: int = 5) -> list[int]:
    """Probe camera/capture indices that actually return a frame."""
    found = []
    try:
        cv2.setLogLevel(0)   # silence "camera index out of range" spam
    except Exception:  # noqa: BLE001
        pass
    for i in range(max_probe):
        cap = _open_device(i)
        try:
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
    return found

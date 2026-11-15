"""A camera pointed at the TV: the tube filmed, not just shown.

Two things give away a CRT that was filmed rather than captured.

The rolling bar. The tube paints each field top to bottom and its phosphors fade
within milliseconds. A camera only collects light while its shutter is open. When
the shutter time is not a whole number of fields, some rows of the screen get
painted one more time than others during the exposure, and that brighter or darker
band drifts up or down the picture at the beat between the camera's frame rate and
the tube's field rate. The band is worked out here from those times: when each row
is painted, how long the shutter is open, and how fast each colour's phosphor fades.

Moire. The camera's sensor samples the phosphor mask on its own grid, slightly
rotated and keystoned because the camera is never square to the glass. Fine mask
detail that the sensor cannot resolve folds into large wavy patterns. That only
happens when the mask was actually drawn (CRT GEOMETRY / MASK on its beam model),
so this belongs after the tube in the screen stage.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Effect

# How much longer than the green phosphor each colour keeps glowing (BGR).
_DECAY_RATIO = np.array([0.5, 1.0, 0.7], np.float32)


def scan_exposure(rows: int, t: float, shutter: float, field_rate: float, decay_ms: float,
                  blanking: float = 0.08) -> np.ndarray:
    """(rows, 3) light each screen row gives a global-shutter exposure starting at t,
    relative to a long exposure (1.0 = no bar).

    Row y is painted at (n + y * (1 - blanking)) / field_rate for every field n; after
    that its phosphor fades as exp(-age / tau). The shutter collects the integral of
    that glow between t and t + shutter."""
    y = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    period = 1.0 / field_rate
    tau = np.maximum(decay_ms / 1000.0 * _DECAY_RATIO.astype(np.float64), 1e-6)   # (3,)
    end = t + shutter
    paint = y * (1 - blanking) * period                                           # (rows,)
    first = math.floor((t - 12 * tau.max()) / period) - 1
    last = math.ceil(end / period) + 1
    light = np.zeros((rows, 3))
    for n in range(first, last + 1):
        when = (n * period + paint)[:, None]                                      # (rows, 1)
        start = np.maximum(t, when)
        open_ = end > start
        glow = tau * (np.exp(-(start - when) / tau) - np.exp(-(end - when) / tau))
        light += np.where(open_, glow, 0)
    expected = shutter * field_rate * tau                                         # (3,)
    return (light / expected).astype(np.float32)


class FilmedTVEffect(Effect):
    """Camera filming the screen: rolling bar, sensor moire, angle, glare and grain."""
    id = "filmed_tv"
    label = "FILMED OFF TV"
    stage = "screen"

    SHUTTERS = (30, 48, 50, 60, 100, 125, 250, 500, 1000)
    CAMERA_RATES = (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)

    def build(self, params):
        self.param("bar", "Rolling bar", 1)
        # 1/48 is a 24 fps film camera's 180-degree shutter, the classic visible bar.
        self.choice("shutter", "Shutter 1/s", tuple(f"1/{s}" for s in self.SHUTTERS), default=1)
        # The camera's own frame rate sets when each exposure starts (the bar's beat);
        # "Output rate" uses the video's frame times.
        self.choice("camera_fps", "Camera fps", ("Output rate",) + tuple(str(r) for r in self.CAMERA_RATES))
        self.param("tube_rate", "Tube fields /s", 59.94, 25, 120)
        self.param("phase", "Bar phase", 0, 0, 1)
        self.param("decay", "Phosphor ms", 1.5, .1, 20)
        self.param("sensor", "Sensor res", .32, .1, 1)
        self.param("lens", "Lens blur", .2, 0, 2)
        self.param("angle", "Angle deg", .6, -10, 10)
        self.param("tilt", "Keystone", .04, -.3, .3)
        self.param("zoom", "Zoom", 1.04, .8, 1.5)
        self.param("glare", "Glare", .15)
        self.param("grain", "Grain", .15)
        self.param("tint", "White balance", 0, -1, 1)
        self._bar_key = None
        self._bar = None

    def fork_caches(self):
        self._bar_key = None
        self._bar = None

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        f = frame.astype(np.float32) * (1 / 255)

        # Rolling bar, from the scan and shutter timing (applied in linear light).
        if v["bar"] > 1e-3:
            shutter = 1.0 / self.SHUTTERS[int(v["shutter"])]
            rate = int(v["camera_fps"])
            exposure_t = t
            if rate > 0:
                fps = self.CAMERA_RATES[rate - 1]
                exposure_t = math.floor(t * fps + 1e-6) / fps
            start = exposure_t + v["phase"] / max(1e-3, v["tube_rate"])
            key = (h, round(start, 6), shutter, round(v["tube_rate"], 4), round(v["decay"], 4))
            if key != self._bar_key:
                self._bar, self._bar_key = scan_exposure(h, start, shutter, v["tube_rate"], v["decay"]), key
            gain = 1 + v["bar"] * (self._bar - 1)
            f = np.power(np.power(f, 2.2) * gain[:, None, :], 1 / 2.2)

        # The camera's view of the glass: rotated, keystoned, zoomed, then sampled by a
        # sensor of its own resolution without an anti-alias filter (that is the moire).
        if v["lens"] > 1e-3:
            f = cv2.GaussianBlur(f, (0, 0), v["lens"] * max(1.0, w / 1440))
        sw = max(16, int(round(w * v["sensor"])))
        sh = max(16, int(round(h * v["sensor"])))
        quad = self._view(w, h, v["angle"], v["tilt"], v["zoom"])
        target = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        sensor = cv2.warpPerspective(f, cv2.getPerspectiveTransform(quad, target), (sw, sh),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
        if v["grain"] > 1e-3:
            sensor = sensor + rng.normal(0, .035 * v["grain"], sensor.shape).astype(np.float32)
        out = cv2.resize(sensor, (w, h), interpolation=cv2.INTER_CUBIC)

        # Room light reflected in the glass: a broad, soft diagonal sheen.
        if v["glare"] > 1e-3:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            band = np.exp(-(((xx / w) * .8 + (yy / h) * .6 - .35) / .18) ** 2)
            spot = np.exp(-(((xx / w - .78) / .12) ** 2 + ((yy / h - .22) / .1) ** 2))
            sheen = (band * .5 + spot) * (.18 * v["glare"])
            out = 1 - (1 - out) * (1 - sheen[..., None])
        if abs(v["tint"]) > 1e-3:
            warm = np.array([1 - .12 * v["tint"], 1, 1 + .12 * v["tint"]], np.float32)   # BGR
            out = out * warm
        return np.clip(out * 255 + .5, 0, 255).astype(np.uint8)

    @staticmethod
    def _view(w, h, angle, tilt, zoom):
        """Corners of the picture area the camera frames, in frame pixels."""
        cx, cy = w / 2, h / 2
        hw, hh = w / 2 / zoom, h / 2 / zoom
        # Keystone: the camera sits below/above centre, so one edge looks wider.
        top, bottom = 1 + tilt / 2, 1 - tilt / 2
        corners = np.array([[-hw * top, -hh], [hw * top, -hh], [hw * bottom, hh], [-hw * bottom, hh]])
        a = math.radians(angle)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return (corners @ rot.T + [cx, cy]).astype(np.float32)

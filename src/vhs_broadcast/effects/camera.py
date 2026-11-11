"""A 90s camcorder: fisheye lens, auto-exposure and white balance that lag, handheld shake.

Everything a consumer camcorder did to the picture before it ever reached tape: a
wide-angle or fisheye adapter bends straight lines, the auto-iris and auto white
balance chase the scene and overshoot, the edge enhancer draws halos, and the
operator's hands never hold still.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .advanced import TemporalEffect, _u8


def _wander(t, speed, seed):
    """Smooth, repeatable wandering in about [-1, 1]: a few incommensurate sines."""
    total = 0.0
    for k, (f, a) in enumerate(((.37, .55), (.83, .3), (1.91, .15))):
        total += a * math.sin(2 * math.pi * f * speed * t + seed * 1.7 + k * 2.1)
    return total


class CamcorderEffect(TemporalEffect):
    id = "camcorder"
    label = "CAMCORDER"
    stage = "camera"

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("fisheye", "Fisheye", 0)
        self.param("vignette", "Vignette", 0)
        self.param("exposure", "Auto exposure", 0)
        self.param("ae_speed", "AE speed Hz", 1.2, .2, 5)
        self.param("wb", "White balance", 0)
        self.param("shake", "Handheld", 0)
        self.param("shake_speed", "Handheld speed", 1, .2, 4)
        self.param("sharpen", "Edge enhance", 0)
        self.param("black", "Black level", 0, -.15, .15)

    def process_history(self, frame, t, v, previous, dt):
        if v["amount"] <= 0:
            return frame, {}
        h, w = frame.shape[:2]
        f = frame.astype(np.float32)
        state = {}
        small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)

        if v["exposure"] > 0:
            # Auto-iris: a damped spring chasing the exposure that would make the scene mid-grey,
            # so it overshoots ("pumps") whenever the scene brightness changes.
            luma = float(small.mean()) / 255
            want = math.log(min(2.5, max(.4, .45 / max(luma, 1e-3))))
            gain, speed = previous.get("gain", want), previous.get("gain_v", 0.0)
            omega = 2 * math.pi * v["ae_speed"]
            for _ in range(max(1, int(math.ceil(dt * 120)))):           # stable small steps
                step = dt / max(1, int(math.ceil(dt * 120)))
                speed += (omega * omega * (want - gain) - 2 * .3 * omega * speed) * step
                gain += speed * step
            state.update(gain=gain, gain_v=speed)
            f *= math.exp(gain * v["exposure"])

        if v["wb"] > 0:
            # Auto white balance: chases a grey-world correction slowly, so colour casts linger
            # after a cut; plus a slow warm/cool drift as the sensor heats.
            means = small.reshape(-1, 3).mean(0) + 1
            want = means.mean() / means
            # Starts from the daylight preset (no correction) and adapts, so casts linger.
            gains = previous.get("wb_gains")
            gains = np.ones(3, np.float32) if gains is None else gains
            k = 1 - math.exp(-dt * 1.2)
            gains = gains + (want - gains) * k
            state["wb_gains"] = gains
            drift = .08 * _wander(t, .15, 3)
            cast = np.array([1 - drift, 1, 1 + drift], np.float32)          # BGR: warm when drift > 0
            f *= (1 + (gains * cast - 1) * v["wb"])[None, None, :]

        if v["sharpen"] > 0:
            blur = cv2.GaussianBlur(f, (0, 0), 1.2 * w / 720)
            f = f + (f - blur) * v["sharpen"] * 2.5                         # bright/dark halos on edges
        if v["black"]:
            f = f + v["black"] * 255

        out = _u8(f)
        if v["fisheye"] > 0 or v["shake"] > 0:
            out = self._lens(out, t, v)
        if v["vignette"] > 0:
            y, x = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
            r = np.sqrt(x * x + y * y) / math.sqrt(2)
            out = _u8(out * (1 - v["vignette"] * r[..., None] ** 2.5))
        if v["amount"] < 1:
            out = cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0)
        return out, state

    @staticmethod
    def _lens(frame, t, v):
        h, w = frame.shape[:2]
        sp = v["shake_speed"]
        dx = v["shake"] * .02 * w * _wander(t, sp, 0)
        dy = v["shake"] * .02 * h * _wander(t, sp, 1)
        rot = math.radians(v["shake"] * 1.5 * _wander(t, sp, 2))
        zoom = 1 / (1 + v["shake"] * .05)                                   # crop in to hide edges
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        nx, ny = (xs - w / 2) / (w / 2), (ys - h / 2) / (w / 2)
        if v["fisheye"] > 0:
            # Barrel distortion that keeps the corners pinned: the centre bulges toward you.
            k = v["fisheye"] * 1.2
            corner = 1 + (h / w) ** 2
            scale = (1 + k * (nx * nx + ny * ny) / corner) / (1 + k)
            nx, ny = nx * scale, ny * scale
        c, s = math.cos(rot), math.sin(rot)
        sx = (nx * c - ny * s) * zoom * (w / 2) + w / 2 - dx
        sy = (nx * s + ny * c) * zoom * (w / 2) + h / 2 - dy
        return cv2.remap(frame, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

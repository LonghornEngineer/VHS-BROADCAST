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


# --------------------------------------------------------------------------- #
#  Light flicker: the scene's lighting as the camera's shutter integrates it
# --------------------------------------------------------------------------- #
LIGHTS = ("Fluorescent", "Incandescent", "Failing tube", "Candle / fire", "Strobe")


class LightFlickerEffect(TemporalEffect):
    """The lights in the scene, seen through the camera's shutter.

    Mains lamps pulse at twice the line frequency. Each row of the picture integrates
    that light over the shutter time, starting when the sensor reads that row: a CCD
    reads every row at once, a CMOS sensor rolls down the frame. So a fast shutter
    under fluorescent light pumps or bands, the bands crawl because the camera's frame
    rate isn't locked to the mains, and a slow shutter averages it all away - just as
    with a real camera. Fluorescent phosphors decay at different rates, so the colour
    flickers too.
    """
    id = "flicker"
    label = "LIGHT FLICKER"
    stage = "camera"

    def build(self, params):
        self.choice("light", "Light", LIGHTS)
        self.param("amount", "Amount", .6)
        self.choice("mains", "Mains", ("60 Hz", "50 Hz"))
        self.param("shutter", "Shutter 1/s", 250, 30, 4000, step=1)
        self.param("rolling", "CMOS readout", 1)                # 0 = CCD (all rows at once)
        self.param("rate", "Rate Hz", 6, .2, 30)                   # candle wander / strobe / tube faults
        self.param("seed", "Seed", 0, 0, 9999, step=1)

    def process_history(self, frame, t, v, previous, dt):
        if v["amount"] < 1e-3:
            return frame, {}
        h = frame.shape[0]
        rows = min(h, 96)                                          # light varies smoothly down the frame
        start = t + v["rolling"] * (np.arange(rows) / rows) * (1001 / 30000)
        exposure = 1 / max(1.0, v["shutter"])
        taps = 24
        times = start[:, None] + exposure * (np.arange(taps) + .5)[None, :] / taps
        light = self._light(times, v)                              # (rows, taps, 3)
        gain = light.mean(axis=1) / self._average(v)               # AE holds the long-run level
        gain = cv2.resize(gain[:, None, :].astype(np.float32), (1, h), interpolation=cv2.INTER_LINEAR)
        out = frame.astype(np.float32) * gain
        return _u8(out), {}

    # Light level (per BGR channel) at the given times.
    def _light(self, tau, v):
        kind = int(v["light"])
        a = float(v["amount"])
        hz = 2 * (60 if int(v["mains"]) == 0 else 50)              # lamps pulse at twice the mains
        ripple = (1 - np.cos(2 * np.pi * hz * tau)) / 2            # 0 at the zero crossings
        if kind in (0, 2):
            # Phosphors: blue decays fastest, red slowest, so blue flickers most.
            depth = a * .55 * np.array([1.25, 1.0, .75], np.float32)
            level = 1 - depth * (1 - ripple[..., None])
            if kind == 2:
                level = level * self._tube_faults(tau, v)[..., None]
            return level
        if kind == 1:
            # The filament's heat smooths it: a shallow, warm ripple.
            depth = a * .12 * np.array([.6, .85, 1.0], np.float32)
            return 1 - depth * (1 - ripple[..., None])
        if kind == 3:
            wander = self._wander(tau, v["rate"], int(v["seed"]))
            level = 1 + a * .45 * wander
            warm = np.array([.8, 1.0, 1.08], np.float32)          # dims toward orange
            return np.clip(level[..., None] ** warm, .05, None)
        # Strobe: a short flash each cycle; the shutter catches it or doesn't.
        phase = (tau * v["rate"]) % 1
        flash = (phase < .04).astype(np.float32) * 25
        return (1 - a + a * flash)[..., None] * np.ones(3, np.float32)

    def _average(self, v):
        kind = int(v["light"])
        a = float(v["amount"])
        if kind in (0, 2):
            return 1 - a * .55 * np.array([1.25, 1.0, .75], np.float32) / 2
        if kind == 1:
            return 1 - a * .12 * np.array([.6, .85, 1.0], np.float32) / 2
        if kind == 4:
            return np.full(3, 1 - a + a * 25 * .04, np.float32)
        return np.ones(3, np.float32)

    @staticmethod
    def _wander(tau, rate, seed):
        """Flame-like wander: a few incommensurate sines with random phases, ~[-1, 1]."""
        rng = np.random.default_rng(seed + 31)
        total = np.zeros_like(tau)
        for k in range(6):
            f = rate * (.25 + 1.6 * rng.random()) * (1 + k * .37)
            total += np.sin(2 * np.pi * f * tau + rng.uniform(0, 2 * np.pi)) / (1 + k * .6)
        return total / 2.2

    @staticmethod
    def _tube_faults(tau, v):
        """A dying tube: mostly lit, sometimes out (the ends still glow), sometimes
        fluttering as the starter keeps trying."""
        seg = np.floor(tau * v["rate"]).astype(np.int64)
        seed = int(v["seed"])
        out = np.ones_like(tau)
        for s in np.unique(seg):
            r = np.random.default_rng([seed, int(s) & 0x7FFFFFFF]).random(2)
            mask = seg == s
            if r[0] < .25 * v["amount"] + .05:
                out[mask] = .12                                     # out: only the ends glow
            elif r[0] < .55 * v["amount"] + .1:
                flutter = .5 + .5 * np.sign(np.sin(2 * np.pi * (8 + 20 * r[1]) * tau[mask]))
                out[mask] = .12 + .88 * flutter
        return out

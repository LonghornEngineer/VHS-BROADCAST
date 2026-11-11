"""TV Circuit Bend: a CRT set with its circuits failing, or jumpered on purpose.

Each control is a part of the set going wrong. Geometry faults act on the deflection,
so the picture is drawn where the beam actually lands; the rest act on the video amp,
the guns and the shadow mask. Sync wobble and hold loss live in CRT Raster / Sync;
this is everything after the set has locked.

- Video into H / V yoke: the circuit-bend classic. Picture signal leaks into the
  deflection current; the yoke integrates it along the line, so lines skew and tear
  following the picture's own brightness.
- Breathing: a weak high-voltage supply sags on bright scenes and the raster grows.
- V / H collapse: a dead deflection stage squeezes the raster toward one bright line.
- V / H linearity: dried-out linearity caps stretch one side and squash the other,
  folding over (doubled, brighter scan) when far gone.
- Convergence: red and blue beams land beside green, worse toward the edges.
- Purity: a magnetised shadow mask throws colour blotches over the picture.
- Retrace: screen voltage too high, so the flyback lines and a lifted black show.
- Video drive: an overdriven video amp crushes black, burns white and finally clips
  into the sync, tearing lines.
- Guns: a dead or shorted (stuck full on) electron gun.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Effect

GUNS = ("All guns", "Red out", "Green out", "Blue out", "Red stuck on", "Green stuck on", "Blue stuck on")
_CHANNEL = {"Red": 2, "Green": 1, "Blue": 0}                  # BGR


class CircuitBendEffect(Effect):
    id = "circuit_bend"
    label = "TV CIRCUIT BEND"
    stage = "receiver"

    def build(self, params):
        self.param("bend_h", "Video into H yoke", 0)
        self.param("bend_v", "Video into V yoke", 0)
        self.param("breathe", "Breathing", 0)
        self.param("vcollapse", "V collapse", 0)
        self.param("hcollapse", "H collapse", 0)
        self.param("vlin", "V linearity", 0, -1, 1)
        self.param("hlin", "H linearity", 0, -1, 1)
        self.param("convergence", "Convergence", 0)
        self.param("purity", "Purity", 0)
        self.param("retrace", "Retrace", 0)
        self.param("drive", "Video drive", 0)
        self.choice("guns", "Guns", GUNS)
        self.param("seed", "Seed", 0, 0, 9999, step=1)

    # ------------------------------------------------------------------ #
    def process(self, frame, t, v):
        out = frame
        if v["drive"] > 1e-3:
            out = self._drive(out, t, v)
        if any(abs(v[k]) > 1e-3 for k in ("bend_h", "bend_v", "breathe", "vcollapse", "hcollapse", "vlin", "hlin")):
            out = self._deflect(out, v)
        if v["convergence"] > 1e-3:
            out = self._converge(out, v["convergence"])
        if v["purity"] > 1e-3:
            out = self._purity(out, t, v)
        if int(v["guns"]):
            out = self._guns(out, int(v["guns"]))
        if v["retrace"] > 1e-3:
            out = self._retrace(out, t, v["retrace"])
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _luma(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    def _deflect(self, frame, v):
        """Where the beam lands: build one sampling map from every deflection fault."""
        h, w = frame.shape[:2]
        luma = self._luma(frame)
        ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        u = np.broadcast_to(xs, (h, w)).copy()
        s = np.broadcast_to(ys, (h, w)).copy()
        gain = np.ones((h, w), np.float32)
        # Breathing: the raster grows as the high voltage sags on a bright picture.
        if v["breathe"] > 1e-3:
            grow = 1 + v["breathe"] * .25 * max(0.0, float(luma.mean()) - .25)
            u /= grow; s /= grow
            gain *= 1 - v["breathe"] * .25 * max(0.0, float(luma.mean()) - .25)
        # Collapse: the scan shrinks toward a line; the same beam energy, so it gets brighter.
        for amount, axis in ((v["vcollapse"], "v"), (v["hcollapse"], "h")):
            if amount > 1e-3:
                size = max(.004, 1 - amount ** .6 * .996)
                if axis == "v":
                    s /= size
                else:
                    u /= size
                gain *= min(5.0, size ** -.7)
        # Linearity: scan speed varies across the screen; past ~0.6 it folds back on itself.
        for amount, axis in ((v["vlin"], "v"), (v["hlin"], "h")):
            if abs(amount) > 1e-3:
                c = s if axis == "v" else u
                c, fold = self._linearity(c, amount)
                if axis == "v":
                    s = c
                else:
                    u = c
                gain *= 1 + fold
        # Video into the yoke: the deflection current integrates the leaking video.
        if v["bend_h"] > 1e-3:
            leak = luma - luma.mean()
            drift = np.cumsum(leak, axis=1) / w
            u = u - v["bend_h"] * 2.2 * drift
        if v["bend_v"] > 1e-3:
            line = luma.mean(axis=1) - luma.mean()
            drift = np.cumsum(line) / h
            s = s - v["bend_v"] * 1.8 * drift[:, None]
        map_x = ((u + 1) * .5 * (w - 1)).astype(np.float32)
        map_y = ((s + 1) * .5 * (h - 1)).astype(np.float32)
        out = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if np.any(gain != 1):
            out = np.clip(out.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _linearity(c, amount):
        """Remap a -1..1 scan coordinate for a linearity fault; returns it and the extra
        brightness where the scan folds over (drawn twice)."""
        g = 2 ** (amount * 1.6)
        p = (c + 1) / 2
        q = np.sign(p) * np.abs(p) ** g
        fold = np.zeros_like(c)
        if abs(amount) > .6:
            # Fold the start of the scan back over itself.
            depth = (abs(amount) - .6) / .4 * .25
            start = q if amount > 0 else 1 - q
            inside = start < depth
            folded = depth + (depth - start)
            start = np.where(inside, folded, start)
            fold = np.where(inside, .6 * (1 - start / max(depth * 2, 1e-3)), 0).astype(np.float32)
            q = start if amount > 0 else 1 - start
        return q * 2 - 1, fold

    @staticmethod
    def _converge(frame, amount):
        """Red and blue land beside green: a static offset plus a radial error."""
        h, w = frame.shape[:2]
        unit = w / 640
        ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        out = frame.copy()
        for ch, sign in ((2, 1), (0, -1)):
            dx = sign * amount * unit * (2 + 6 * xs * np.abs(xs))
            dy = sign * amount * unit * (1 + 4 * ys * np.abs(ys)) * .6
            map_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :] - dx, (h, w))
            map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None] - dy, (h, w))
            out[..., ch] = cv2.remap(frame[..., ch], np.ascontiguousarray(map_x), np.ascontiguousarray(map_y),
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out

    @staticmethod
    def _purity(frame, t, v):
        """Magnetised mask: beams hit the wrong phosphor in broad patches, worst at the corners."""
        h, w = frame.shape[:2]
        rng = np.random.default_rng(int(v["seed"]) + 7)
        field = rng.normal(0, 1, (4, 5, 3)).astype(np.float32)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        corner = np.clip(.35 + .65 * (xs * xs + ys * ys) / 2, 0, 1)[..., None]
        drift = 1 + .1 * math.sin(t * .7)
        tint = field * corner * v["purity"] * .45 * drift
        f = frame.astype(np.float32) / 255
        # The wrong phosphor lights in proportion to the beam, so the blotches ride the picture.
        lum = f.mean(axis=2, keepdims=True)
        out = f + tint * (.25 + lum)
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _guns(frame, mode):
        out = frame.copy()
        colour = GUNS[mode].split()[0]
        ch = _CHANNEL[colour]
        if "out" in GUNS[mode]:
            out[..., ch] = 0
        else:
            # A shorted gun runs at full beam, with a faint trace of the picture on top.
            out[..., ch] = np.clip(200 + frame[..., ch].astype(np.int16) // 5, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _retrace(frame, t, amount):
        """Screen voltage too high: black lifts and the vertical flyback shows.

        The beam flies from the bottom back to the top in about ten line times while
        the horizontal scan keeps running, so it draws ten or so faint lines, every one
        rising left to right (the right-to-left returns stay blanked). It is stray beam
        light added to the picture: plain on dark scenes, lost on bright ones.
        """
        h, w = frame.shape[:2]
        f = frame.astype(np.float32)
        f = f * (1 - amount * .15) + amount * 38                       # lifted black
        layer = np.zeros((h, w), np.float32)
        sweeps = 10
        for k in range(sweeps):
            y0 = h - k * h / sweeps                                    # each sweep starts lower left
            y1 = y0 - h / sweeps                                       # and ends one step higher, right
            cv2.line(layer, (0, int(y0)), (w - 1, int(y1)), 1.0, 1, cv2.LINE_AA)
        # The beam is moving fast, so the lines are thin and soft, brightest where the
        # sweep starts and fading as it crosses.
        fade = np.linspace(1, .55, w, dtype=np.float32)[None, :]
        layer = cv2.GaussianBlur(layer * fade, (0, 0), max(.5, w / 1280))
        shimmer = .85 + .15 * math.sin(t * 60)
        light = layer[..., None] * 60 * amount * shimmer
        # Added light shows against what is already lit: little on white, plain on black.
        f += light * (1 - f / 255 * .8)
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _drive(frame, t, v):
        """Overdriven video amp: contrast slams up, black crushes, white burns; far enough
        and the video swings into the sync level, so the set loses lines sideways."""
        d = v["drive"]
        f = frame.astype(np.float32) / 255
        # Gain is set at black, so the picture brightens and the whites run into the
        # amp's ceiling; the setup slips a little lower, crushing the darkest detail.
        g = f * (1 + 3.5 * d) - .08 * d
        knee = .7
        g = np.where(g > knee, knee + (1 - knee) * np.tanh((g - knee) / (1 - knee)), g)
        out = np.clip(g * 255, 0, 255).astype(np.uint8)
        if d > .6:
            h, w = frame.shape[:2]
            luma = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            # Lines whose blacks crush below the sync slice lose horizontal lock.
            dark = (luma < .08).mean(axis=1)
            rng = np.random.default_rng(int(t * 30) + int(v["seed"]))
            tear = np.where(dark > .3, rng.normal(0, 1, h) * (d - .6) * 2.5 * .08 * w, 0)
            tear = np.convolve(tear, np.ones(5) / 5, mode="same").astype(np.float32)
            map_x = np.arange(w, dtype=np.float32)[None, :] - tear[:, None]
            map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out

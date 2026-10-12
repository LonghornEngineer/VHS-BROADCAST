"""Raster sample-and-hold and time-based rolling shutter, rendered on CPU."""
from __future__ import annotations

import math
import cv2
import numpy as np

from .base import Effect
from .advanced import TemporalEffect


class RasterHoldEffect(Effect):
    """Latch a pixel's colour for a run along the raster scan direction."""
    id = "raster_hold"
    label = "RASTER / PIXEL HOLD"

    def build(self, params):
        self.param("amount", "Mix", .8)
        self.param("length", "Hold px", 32, 1, 512, step=1)
        self.param("density", "Density", .5)
        self.param("lines", "Line group", 1, 1, 32, step=1)
        self.param("rate", "Refresh Hz", 8, 0, 60)
        self.choice("direction", "Scan", ("Left to Right", "Right to Left", "Top to Bottom", "Bottom to Top"))
        self.param("seed", "Seed", 1, 0, 9999, step=1)

    def process(self, frame, t, v):
        if v["amount"] == 0 or v["density"] == 0 or v["length"] <= 1:
            return frame
        direction = int(v["direction"])
        source = frame.transpose(1, 0, 2) if direction >= 2 else frame
        if direction % 2:
            source = source[:, ::-1]
        h, w = source.shape[:2]
        length, lines = max(1, round(v["length"])), max(1, round(v["lines"]))
        # Integrated refresh prevents speed edits from jumping the random sequence.
        tick = math.floor(self.params.motion_phase("raster_hold.rate", t))
        rng = np.random.default_rng((int(v["seed"]) + tick * 104729) & 0xFFFFFFFF)
        groups = math.ceil(h / lines)
        offset = rng.integers(0, length, (groups, 1))
        x = np.arange(w)[None, :]
        blocks = (x + offset) // length
        active = rng.random((groups, math.ceil(w / length) + 2)) < v["density"]
        latch = np.maximum(0, blocks * length - offset)
        mapping = np.where(np.take_along_axis(active, blocks, axis=1), latch, x)
        mapping = mapping[np.arange(h) // lines]
        held = source[np.arange(h)[:, None], mapping]
        out = cv2.addWeighted(source, 1 - v["amount"], held, v["amount"], 0)
        if direction % 2:
            out = out[:, ::-1]
        return np.ascontiguousarray(out.transpose(1, 0, 2) if direction >= 2 else out)


class RollingShutterEffect(TemporalEffect):
    """Read scanlines from progressively older moments, with temporal interpolation."""
    id = "rolling_shutter"
    label = "ROLLING SHUTTER"
    MAX_HISTORY_BYTES = 96 * 1024 * 1024
    MAX_HISTORY_FRAMES = 120

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("readout", "Readout s", .15, 0, 2)
        self.choice("direction", "Scan", ("Top to Bottom", "Bottom to Top", "Left to Right", "Right to Left"))
        self.param("bands", "Bands", 0, 0, 128, step=1)

    def process_history(self, frame, t, v, previous, dt):
        history = previous.get("history", ())
        # Keep the maximum readout window available while auditioning its value.
        history = tuple(item for item in history if item[0] >= t - 2.1)
        capacity = max(1, min(self.MAX_HISTORY_FRAMES, self.MAX_HISTORY_BYTES // frame.nbytes))
        history = (*history, (t, frame.copy()))[-capacity:]
        if v["readout"] <= 0 or v["amount"] <= 0 or len(history) == 1:
            return frame, {"history": history}
        direction = int(v["direction"])
        vertical = direction < 2
        size = frame.shape[0 if vertical else 1]
        # At the end of readout, the first scanned line is the oldest.
        ramp = np.linspace(1, 0, size)
        if direction % 2:
            ramp = 1 - ramp
        bands = int(v["bands"])
        if bands > 0:
            ramp = np.minimum(bands - 1, np.floor(ramp * bands)) / max(1, bands - 1)
        stamps = np.array([stamp for stamp, _ in history])
        target = np.clip(t - ramp * v["readout"], stamps[0], stamps[-1])
        lo = np.clip(np.searchsorted(stamps, target, side="right") - 1, 0, len(history) - 1)
        hi = np.minimum(lo + 1, len(history) - 1)
        weight = (target - stamps[lo]) / np.maximum(1e-9, stamps[hi] - stamps[lo])
        out = np.empty_like(frame)
        for index in np.unique(lo):
            rows = np.flatnonzero(lo == index)
            upper = min(index + 1, len(history) - 1)
            a, b = history[index][1], history[upper][1]
            if vertical:
                alpha = weight[rows, None, None]
                out[rows] = np.clip(a[rows] * (1 - alpha) + b[rows] * alpha, 0, 255).astype(np.uint8)
            else:
                alpha = weight[None, rows, None]
                out[:, rows] = np.clip(a[:, rows] * (1 - alpha) + b[:, rows] * alpha, 0, 255).astype(np.uint8)
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0), {"history": history}

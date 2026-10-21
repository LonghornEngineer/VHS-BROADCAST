"""Raster sample-and-hold, rolling shutter, sync loss and CRT raster sync, rendered on CPU."""
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


class SyncLossEffect(TemporalEffect):
    """Digital sync dropout: crushed low-bit colour, smeared lines, freezes and blackouts."""
    id = "sync_loss"
    label = "SYNC LOSS"
    _BAYER = np.array([[0, 2], [3, 1]], np.float32) / 4 - .375

    def build(self, params):
        self.param("amount", "Mix", 1)
        self.param("crush", "Black crush", .3, 0, .95)
        self.param("bits", "Color bits", 2, 1, 8, step=1)
        self.param("dither", "Dither", .7)
        self.param("smear", "Line smear", .5)
        self.param("split", "RGB split", .5)
        self.param("burst", "Burst", .4)
        self.param("freeze", "Freeze", .2)
        self.param("dropout", "Dropout", .15)
        self.param("rate", "Glitch Hz", 10, 0, 30)
        self.param("seed", "Seed", 1, 0, 9999, step=1)

    def process_history(self, frame, t, v, previous, dt):
        if v["amount"] <= 0:
            return frame, {}
        # Integrated refresh keeps the random sequence stable while Hz is edited.
        tick = math.floor(self.params.motion_phase("sync_loss.rate", t))
        rng = np.random.default_rng((int(v["seed"]) * 7919 + tick * 104729) & 0xFFFFFFFF)
        burst = rng.random() < v["burst"]
        dropout = rng.random() < v["dropout"]
        freeze = rng.random() < v["freeze"]
        state = {}
        source = frame
        if freeze:
            held = previous.get("frozen")
            if held is not None and held.shape == frame.shape and previous.get("frozen_tick") == tick:
                source = held
            state = {"frozen": source, "frozen_tick": tick}
        h, w = frame.shape[:2]
        f = source.astype(np.float32) / 255
        crush = min(v["crush"], .95)
        f = np.clip((f - crush) / (1 - crush), 0, 1) ** 1.4
        luma = f.mean(axis=2, keepdims=True)
        f = np.clip(luma + (f - luma) * (1 + 1.5 * crush), 0, 1)
        f = self._smear(f, rng, v, burst, dropout)
        if dropout:
            f *= .06
            lit = rng.random(h) < .04
            channel = rng.integers(1, 3, int(lit.sum()))
            f[np.flatnonzero(lit), :, channel] = np.clip(f[np.flatnonzero(lit), :, channel] * 16, 0, 1)
        levels = 2 ** int(v["bits"]) - 1
        if levels < 255:
            threshold = np.tile(self._BAYER, (h // 2 + 1, w // 2 + 1))[:h, :w, None] * v["dither"]
            f = np.floor(np.clip(f * levels + .5 + threshold, 0, levels)) / levels
        out = (f * 255).astype(np.uint8)
        return cv2.addWeighted(frame, 1 - v["amount"], out, v["amount"], 0), state

    @staticmethod
    def _smear(f, rng, v, burst, dropout):
        """Hold one sample per line run, per channel when RGB split is up."""
        h, w = f.shape[:2]
        chance = min(1, v["smear"] * (1.8 if burst else 1 if dropout else .2))
        if chance <= 0:
            return f
        # Lines arrive in short bands, like a decoder losing several scanlines at once.
        heights = rng.integers(1, 4, h)
        band = np.repeat(np.arange(h), heights)[:h]
        bands = band[-1] + 1
        x = np.arange(w)[None, :]
        out = f.copy()

        def draw():
            full = rng.random(bands) < (.6 if burst else .3)
            return (rng.random(bands), full,
                    np.where(full, 0, rng.integers(0, w, bands)),
                    np.where(full, w, rng.integers(w // 8 + 1, w + 1, bands)),
                    rng.integers(0, w, (bands, 6)))

        shared = draw()
        rows = np.arange(h)
        for c in range(3):
            own = draw()
            split = rng.random(bands) < v["split"]
            roll, full, start, length, picks = (
                np.where(split[(...,) + (None,) * (a.ndim - 1)], b, a)[band]
                for a, b in zip(shared, own))
            inside = (roll < chance)[:, None] & (x >= start[:, None]) & (x < (start + length)[:, None])
            # Full lines latch the brightest of a few samples, so lost lines glow.
            brightest = f[rows[:, None], picks, c].max(axis=1)
            held = np.where(full, brightest, f[rows, np.clip(start, 0, w - 1), c])
            out[..., c] = np.where(inside, held[:, None], f[..., c])
        return out


class CRTRasterEffect(Effect):
    """A CRT beam that keeps losing horizontal sync: jagged crawling edges, skew, roll."""
    id = "crt_raster"
    label = "CRT RASTER / SYNC"
    TABLE = 1 << 16
    WARMUP = 256                          # lines run before the top so the error has settled

    def build(self, params):
        self.param("noise", "Sync noise px", 6, 0, 48)
        self.param("lock", "Lock", .6)
        self.param("kicks", "Kicks", .2)
        self.param("crawl", "Crawl", .5, -3, 3)
        self.param("churn", "Churn", .35)
        self.param("hhold", "H-hold", 0, -1, 1)
        self.param("vhold", "V-hold", 0, -1, 1)
        self.param("bar", "Blank bar", .06, 0, .3)
        self.param("beam", "Beam fade", 0)
        self.param("beam_speed", "Beam speed", .3, -2, 2)
        self._table = np.random.default_rng(1983).normal(0, 1, self.TABLE).astype(np.float32)
        self._kick_table = np.random.default_rng(1984).random(self.TABLE).astype(np.float32)

    def process(self, frame, t, v):
        h, w = frame.shape[:2]
        unit = w / 720                    # pixel controls are measured at 720 wide
        offset = self._line_error(h, t, v) * unit
        # H-hold out of lock: every line starts a little later, so the picture slants.
        offset = offset + v["hhold"] * w * .5 * np.arange(h, dtype=np.float32) / h
        out = frame
        if np.abs(offset).max() >= .25:
            map_x = np.arange(w, dtype=np.float32)[None, :] - offset[:, None]
            map_x = np.mod(map_x, w) if v["hhold"] else map_x     # out of hold the picture wraps
            map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
            out = cv2.remap(frame, np.ascontiguousarray(map_x, np.float32), np.ascontiguousarray(map_y),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        roll = self.params.motion_phase("crt_raster.vhold", t)
        if roll or v["bar"] > 0 and v["vhold"]:
            shift = int(round((roll % 1) * h))
            out = np.roll(out, shift, axis=0)
            bar = int(v["bar"] * h)
            if bar and v["vhold"]:
                # The vertical blanking interval rolls into view as a dark bar.
                rows = (np.arange(bar) + shift - bar) % h
                out = out.copy()
                out[rows] = (out[rows] * .08).astype(np.uint8)
        if v["beam"] > 0:
            out = self._beam_fade(out, t, v)
        return out

    def _line_error(self, h, t, v):
        """Horizontal start error per line: noise integrated through a leaky sync loop."""
        if v["noise"] <= 0:
            return np.zeros(h, np.float32)
        # Positive crawl slides the jagged pattern down the screen, like the beam.
        lead = self.params.motion_phase("crt_raster.crawl", t) * h
        start = -int(np.floor(lead)) - self.WARMUP
        index = (np.arange(start, start + h + self.WARMUP) % self.TABLE)
        drive = self._table[index]
        if v["churn"] > 0:
            fresh = np.random.default_rng(int(round(t * 1000)) & 0xFFFFFFFF)
            drive = drive * (1 - v["churn"]) + fresh.normal(0, 1, drive.size).astype(np.float32) * v["churn"]
        if v["kicks"] > 0:
            # Sync spikes: rare, large impulses that leave a stepped edge behind.
            kick = self._kick_table[index] < v["kicks"] * .02
            drive = np.where(kick, drive * 12, drive)
        keep = .995 - .9 * v["lock"]                       # low lock lets errors run for many lines
        drive = drive * v["noise"] * math.sqrt(1 - keep * keep)
        error = np.empty_like(drive)
        acc = 0.0
        for i, d in enumerate(drive.tolist()):
            acc = keep * acc + d
            error[i] = acc
        return error[self.WARMUP:]

    def _beam_fade(self, frame, t, v):
        """Where the beam just passed is brightest; older lines have dimmed, as when filmed."""
        h = frame.shape[0]
        head = self.params.motion_phase("crt_raster.beam_speed", t) % 1
        since = (head - np.arange(h, dtype=np.float32) / h) % 1   # how long since each line was drawn
        gain = 1 - v["beam"] * .55 * (1 - np.exp(-since * 6))
        return np.clip(frame * gain[:, None, None], 0, 255).astype(np.uint8)

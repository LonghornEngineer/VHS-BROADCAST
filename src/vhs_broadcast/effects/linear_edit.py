"""Linear editing: the video was cut deck to deck, and every cut shows it.

On a linear edit system the record deck stops, backs up, pre-rolls and punches in at
each edit point. A cheap setup without a time-base corrector leaves tells at every
cut: a frame or two of black or deck blue, the TV losing vertical sync and rolling
the picture until it locks again, the colour killer dropping the chroma while the
burst re-locks (then the hue settles), a burst of head-switch noise and tearing at
the bottom of the frame, and a click on the audio. Material that went through more
dubs before the edit is softer, smearier and noisier: generation loss.

Edit points come from three places, combined:
- **Keys**: every change of the Edit point control's keyed value is a cut, so cuts
  are placed on the timeline like any key and scrubbing agrees with the render.
- **Cut list**: times in seconds, from an agent or an EDL (``load_edl``).
- **Detect**: cuts found in the incoming picture. Detection follows the frames as
  they play, so render from the start for exact results.
A colour-bars and 1 kHz tone head can open the tape.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

from .advanced import TemporalEffect

BLUE = (170, 0, 0)                     # BGR deck blue, as VCR DECK uses


def load_edl(path: str, fps: float = 30000 / 1001) -> list[float]:
    """Record-in times (seconds) of every event in a CMX 3600 EDL.

    Event lines read: number, reel, track, transition, source in/out, record in/out.
    Drop-frame (`;`) timecode is converted with the NTSC drop-frame rule."""
    times = []
    pattern = re.compile(r"^\s*\d+\s+\S+\s+\S+\s+\S+(?:\s+\d+)?\s+"
                         r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                times.append(timecode_seconds(match.group(3), fps))
    start = min(times) if times else 0.0
    # Record timecode often starts at 01:00:00:00; cuts are relative to the first event.
    return sorted({round(t - start, 6) for t in times if t - start > 0})


def timecode_seconds(code: str, fps: float) -> float:
    parts = re.split(r"[:;.]", code)
    if len(parts) != 4:
        raise ValueError(f"bad timecode: {code}")
    hours, minutes, seconds, frames = (int(p) for p in parts)
    nominal = round(fps)
    if ";" in code and nominal in (30, 60):
        drop = 2 * nominal // 30
        total_minutes = 60 * hours + minutes
        frame_number = ((hours * 3600 + minutes * 60 + seconds) * nominal + frames
                        - drop * (total_minutes - total_minutes // 10))
        return frame_number / fps
    return hours * 3600 + minutes * 60 + seconds + frames / nominal


def bars_frame(h: int, w: int) -> np.ndarray:
    """Colour bars: seven 75 % bars, a reversed castellation strip, then a black-level
    strip with -I, white, +Q and a PLUGE step, the usual studio layout."""
    out = np.zeros((h, w, 3), np.uint8)
    top = int(h * .67)
    mid = int(h * .75)
    bars = [(191, 191, 191), (0, 191, 191), (191, 191, 0), (0, 191, 0),
            (191, 0, 191), (0, 0, 191), (191, 0, 0)]           # BGR at 75 %
    edges = np.linspace(0, w, 8).astype(int)
    for i, color in enumerate(bars):
        out[:top, edges[i]:edges[i + 1]] = color
    castle = [(191, 0, 0), (19, 19, 19), (191, 0, 191), (19, 19, 19),
              (191, 191, 0), (19, 19, 19), (191, 191, 191)]
    for i, color in enumerate(castle):
        out[top:mid, edges[i]:edges[i + 1]] = color
    bottom = [((106, 52, 0), 1.25), ((255, 255, 255), 1.25), ((106, 0, 50), 1.25),
              ((19, 19, 19), 1.25), ((10, 10, 10), .4167), ((19, 19, 19), .4167),
              ((29, 29, 29), .4167), ((19, 19, 19), 1)]
    unit = w / sum(width for _, width in bottom)
    x = 0.0
    for color, width in bottom:
        out[mid:, int(x):int(x + width * unit)] = color
        x += width * unit
    out[mid:, int(x):] = (19, 19, 19)
    return out


class LinearEditEffect(TemporalEffect):
    id = "linear_edit"
    label = "LINEAR EDIT / DUBS"
    stage = "tape"

    def build(self, params):
        self.param("edit", "Edit point", 0, 0, 999, step=1)
        self.param("detect", "Detect cuts", 0)
        self.param("settle", "Settle s", .45, .05, 3)
        self.choice("flash", "Flash", ("None", "Black", "Deck blue"), default=1)
        self.param("flash_frames", "Flash frames", 1, 0, 6, step=1)
        self.param("roll", "Sync roll", .6, 0, 3)
        self.param("chroma", "Colour relock", .8)
        self.param("tear", "Head-switch tear", .6)
        self.param("pop", "Audio pop", .5)
        self.param("generations", "Generations", 1, 0, 6)
        self.param("gen_spread", "Dub spread", 1, 0, 3)
        self.param("bars", "Bars head s", 0, 0, 30)
        self.param("fps", "Frame rate", 29.97, 10, 60)
        self.cuts: list[float] = []
        self.detected: list[float] = []

    # ---- settings (cut list) ----------------------------------------------
    def settings(self):
        return {"cuts": list(self.cuts)}

    def restore_settings(self, data):
        cuts = data.get("cuts", self.cuts)
        if not isinstance(cuts, list) or not all(isinstance(c, (int, float)) and c >= 0 for c in cuts):
            raise ValueError("linear_edit.cuts must be a list of nonnegative seconds")
        self.cuts = sorted(float(c) for c in cuts)

    def reset_state(self):
        super().reset_state()
        self.detected = []

    # ---- cut timing --------------------------------------------------------
    def edit_points(self, t):
        """All cut times at or before t, sorted."""
        points = set(self.cuts) | set(self.params.change_times(f"{self.id}.edit")) | set(self.detected)
        return sorted(c for c in points if c <= t)

    def _detect(self, frame, previous, t, v):
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        hue = cv2.calcHist([cv2.cvtColor(cv2.resize(frame, (64, 48)), cv2.COLOR_BGR2HSV)],
                           [0], None, [16], [0, 180]).ravel()
        hue /= max(1.0, hue.sum())
        old = previous.get("thumb")
        if old is not None and v["detect"] > 1e-3:
            luma = float(np.abs(thumb - old).mean()) / 64
            colour = float(np.abs(hue - previous["hue"]).sum()) / 2
            score = max(luma, colour)
            last = self.detected[-1] if self.detected else -1e9
            if score > 1.05 - v["detect"] and t - last > .25 and t not in self.detected:
                self.detected.append(t)
        return {"thumb": thumb, "hue": hue}

    # ---- picture -----------------------------------------------------------
    def process_history(self, frame, t, v, previous, dt):
        state = self._detect(frame, previous, t, v)
        h, w = frame.shape[:2]
        if v["bars"] > 1e-3 and t < v["bars"]:
            return bars_frame(h, w), state
        points = self.edit_points(t)
        out = frame
        if v["generations"] > 1e-3 or v["gen_spread"] > 1e-3:
            out = self._generation_loss(out, self._generation(len(points), v), t)
        if not points:
            return out, state
        age = t - points[-1]
        if age < 0 or age > v["settle"] + 1e-6:
            return out, state
        fps = max(1.0, v["fps"])
        if int(v["flash"]) and age < v["flash_frames"] / fps - 1e-6:
            fill = (0, 0, 0) if int(v["flash"]) == 1 else BLUE
            return np.full_like(frame, fill), state
        progress = age / v["settle"]                          # 0 at the cut, 1 when locked
        seed = int(points[-1] * 1000) & 0xFFFFFFFF
        out = self._chroma_relock(out, progress, age, v["chroma"])
        out = self._tear(out, progress, t, seed, v["tear"])
        out = self._roll(out, progress, v["roll"])
        return out, state

    def _generation(self, segment, v):
        """Dub generations behind this segment: the base plus a fixed per-segment spread."""
        rng = np.random.default_rng(1009 + segment)
        return max(0.0, v["generations"] + v["gen_spread"] * (rng.random() - .3))

    @staticmethod
    def _generation_loss(frame, gens, t):
        """Each dub softens luma, smears and delays chroma, lowers saturation and adds noise."""
        if gens <= 1e-3:
            return frame
        h, w = frame.shape[:2]
        unit = w / 720
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[..., 0] = cv2.GaussianBlur(ycc[..., 0], (0, 0), sigmaX=.45 * gens * unit + 1e-3, sigmaY=.01)
        for c in (1, 2):
            chroma = cv2.GaussianBlur(ycc[..., c], (0, 0), sigmaX=1.6 * gens * unit + 1e-3, sigmaY=.01)
            shift = int(round(.8 * gens * unit))
            if shift:
                chroma = np.concatenate([np.repeat(chroma[:, :1], shift, axis=1), chroma[:, :-shift]], axis=1)
            ycc[..., c] = 128 + (chroma - 128) * max(0.0, 1 - .06 * gens)
        rng = np.random.default_rng(int(t * 1000) & 0xFFFFFFFF)
        ycc[..., 0] += rng.normal(0, 1.4 * gens, (h, w)).astype(np.float32)
        ycc[..., 0] = 16 + (ycc[..., 0] - 16) * (1 - .015 * gens)
        return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _chroma_relock(frame, progress, age, amount):
        """Colour killer holds the picture in black and white, then the hue swings in."""
        if amount <= 1e-3:
            return frame
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        kill = min(1.0, progress / .35)
        saturation = 1 - amount * (1 - kill * kill * (3 - 2 * kill))
        swing = math.radians(35 * amount * (1 - progress) * math.cos(2 * math.pi * 5 * age)) if kill >= 1 else 0
        cr, cb = ycc[..., 1] - 128, ycc[..., 2] - 128
        c, s = math.cos(swing), math.sin(swing)
        ycc[..., 1] = 128 + (cr * c - cb * s) * saturation
        ycc[..., 2] = 128 + (cr * s + cb * c) * saturation
        return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _tear(frame, progress, t, seed, amount):
        """Head-switch burst: the bottom lines skew sideways and break up into noise."""
        if amount <= 1e-3 or progress > .6:
            return frame
        h, w = frame.shape[:2]
        strength = amount * (1 - progress / .6)
        band = max(2, int(h * (.04 + .1 * strength)))
        rng = np.random.default_rng(seed + int(t * 1000))
        out = frame.copy()
        for i, y in enumerate(range(h - band, h)):
            skew = int((i / band) ** 1.5 * w * .12 * strength + rng.normal(0, 3) * strength)
            out[y] = np.roll(frame[y], skew, axis=0)
        noise = rng.integers(0, 255, (band, w, 1), dtype=np.uint8)
        mix = (np.linspace(0, 1, band)[:, None, None] * strength).astype(np.float32)
        out[h - band:] = (out[h - band:] * (1 - mix) + noise * mix).astype(np.uint8)
        return out

    @staticmethod
    def _roll(frame, progress, amount):
        """Vertical hold lost: the picture rolls with the blanking bar showing, slowing
        as the sync circuit pulls it back in."""
        if amount <= 1e-3:
            return frame
        h = frame.shape[0]
        travel = amount * (1 - progress) ** 2                 # in picture heights
        offset = int(round(h * (travel % 1.0)))
        if offset == 0:
            return frame
        out = np.roll(frame, offset, axis=0)
        bar = max(2, int(h * .075))                            # vertical blanking interval
        top = max(0, offset - bar)
        out[top:offset] = (out[top:offset] * .08).astype(np.uint8)
        return out

    # ---- audio -------------------------------------------------------------
    def process_audio(self, samples: np.ndarray, sr: int, duration: float):
        """Clicks at every cut and the 1 kHz line-up tone under the bars head."""
        v = {p.id.split(".")[-1]: p.value for p in self.own_params()}
        out = samples.astype(np.float32, copy=True)
        channels = out.shape[1] if out.ndim == 2 else 1
        view = out if out.ndim == 2 else out[:, None]
        if v["bars"] > 1e-3:
            n = min(len(view), int(v["bars"] * sr))
            tone = (.1 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)).astype(np.float32)   # -20 dBFS
            view[:n] = np.repeat(tone[:, None], channels, axis=1)
        if v["pop"] > 1e-3:
            length = int(.012 * sr)
            env = np.exp(-np.arange(length) / (.0025 * sr)).astype(np.float32)
            rng = np.random.default_rng(7)
            for cut in self.edit_points(duration):
                start = int(cut * sr)
                if start >= len(view):
                    continue
                n = min(length, len(view) - start)
                click = (rng.normal(0, .35, n) * env[:n] * v["pop"]).astype(np.float32)
                view[start:start + n] = view[start:start + n] * (1 - env[:n, None] * .8) + click[:, None]
        return np.clip(out, -1, 1) if np.issubdtype(samples.dtype, np.floating) else out

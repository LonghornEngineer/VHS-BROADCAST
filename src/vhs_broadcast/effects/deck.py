"""The VCR as a machine: keyframe Play, Pause, FF, REW and Stop like pressing its buttons.

Each mode has the look of the real transport: a paused tape shudders with a noise
bar, searching drags noise bands across the picture, Stop shows the blue screen,
and pressing Play after Stop runs the tape-loading settle (snow, then a picture
that rolls into vertical hold); after a search the bars fade out. Transitions are timed from the mode's
own keyframes, so scrubbing, preview and render agree.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .advanced import TemporalEffect, _u8

PLAY, PAUSE, FF, REW, STOP = range(5)
BLUE = (170, 0, 0)                     # BGR: the classic VCR blue screen


class VCRDeckEffect(TemporalEffect):
    id = "deck"
    label = "VCR DECK"
    stage = "tape"

    def build(self, params):
        self.choice("mode", "Mode", ("Play", "Pause", "Fast forward", "Rewind", "Stop (blue screen)"))
        self.param("settle", "Settle s", 1.5, 0, 6)
        self.param("bars", "Search bars", 2, 1, 4, step=1)
        self.param("bar_speed", "Bar speed", 1, .2, 4)
        self.param("noise", "Bar noise", .8)
        self.param("jitter", "Pause jitter", .6)
        self.param("seed", "Seed", 0, 0, 9999, step=1)

    def _since(self, t, mode, previous):
        """Seconds since the deck entered this mode, from keys if any, else from history."""
        changed = self.params.changed_at("deck.mode", t)
        if changed is not None:
            return t - changed, self._mode_before(changed)
        if previous.get("mode") == mode:
            return t - previous.get("since", t), previous.get("from", mode)
        if "mode" in previous:
            return 0.0, previous["mode"]
        return math.inf, mode

    def _mode_before(self, changed):
        track = self.params._motion_timeline.tracks.get("deck.mode")
        return int(round(track.value_at(changed - 1e-4)))

    def process_history(self, frame, t, v, previous, dt):
        mode = int(v["mode"])
        since, came_from = self._since(t, mode, previous)
        state = {"mode": mode, "since": t - since if math.isfinite(since) else -1e9, "from": came_from}
        rng = np.random.default_rng([int(v["seed"]), int(t * 30)])
        h, w = frame.shape[:2]
        if mode == STOP:
            return np.full_like(frame, BLUE), state
        if mode == PAUSE:
            held = previous.get("held")
            if held is None or held.shape != frame.shape:
                held = frame.copy()
            state["held"] = held
            return self._pause(held, t, v, rng), state
        if mode in (FF, REW):
            return self._search(frame, t, v, rng, mode == FF), state
        if came_from == STOP and since < v["settle"]:
            return self._settle(frame, since / max(v["settle"], 1e-3), came_from, v, rng), state
        if came_from in (FF, REW) and since < v["settle"] / 3:
            # Leaving a search: the bars fade out as the heads find the tracks again.
            fade = 1 - since / max(v["settle"] / 3, 1e-3)
            return self._search(frame, t, {**v, "noise": v["noise"] * fade}, rng, came_from == FF), state
        return frame, state   # out of pause the tape simply resumes

    # ------------------------------------------------------------------ #
    def _noise_band(self, rows, w, rng, strength):
        snow = rng.integers(0, 256, (rows, w // 2 + 1), np.uint8).repeat(2, axis=1)[:, :w]
        streaks = cv2.blur(snow, (25, 1))
        band = np.maximum(snow * .5, streaks.astype(np.float32) * 1.2) * strength
        return np.repeat(band[..., None], 3, axis=2)

    def _pause(self, held, t, v, rng):
        h, w = held.shape[:2]
        out = held.copy()
        if v["jitter"] > 0:
            # Both heads read the same track: the fields disagree and the picture shudders.
            shift = int(round(v["jitter"] * 2 * (1 if int(t * 30) % 2 else -1)))
            out[1::2] = np.roll(out[1::2], shift, axis=1)
            bounce = int(round(v["jitter"] * rng.integers(-1, 2)))
            out = np.roll(out, bounce, axis=0)
        if v["noise"] > 0:
            rows = max(2, h // 18)
            top = int(h * (.72 + .02 * math.sin(t * 3)))
            band = self._noise_band(rows, w, rng, v["noise"])
            region = out[top:top + rows].astype(np.float32)
            out[top:top + rows] = _u8(region * .35 + band)
        return out

    def _search(self, frame, t, v, rng, forward):
        h, w = frame.shape[:2]
        out = frame.copy()
        n = int(v["bars"])
        rows = max(4, h // 9)
        direction = 1 if forward else -1
        for k in range(n):
            pos = (k / n + direction * t * .35 * v["bar_speed"]) % 1
            top = int(pos * (h + rows)) - rows
            lo, hi = max(0, top), min(h, top + rows)
            if hi <= lo:
                continue
            band = self._noise_band(hi - lo, w, rng, v["noise"])
            # The picture inside a bar slips sideways as the head crosses tracks.
            slip = np.roll(frame[lo:hi], int(rng.integers(-w // 12, w // 12)), axis=1).astype(np.float32)
            out[lo:hi] = _u8(slip * (1 - .7 * v["noise"]) + band)
        return out

    def _settle(self, frame, p, came_from, v, rng):
        """Tape loading: snow (from the blue screen), then the picture rolls into vertical hold."""
        h, w = frame.shape[:2]
        if p < .25:
            snow = rng.integers(0, 256, (h, w // 2 + 1), np.uint8).repeat(2, axis=1)[:, :w]
            snow = np.repeat(snow[..., None], 3, axis=2)
            if came_from == STOP:
                base = np.full_like(frame, BLUE)
                return cv2.addWeighted(base, 1 - p * 4, snow, p * 4, 0)
            return snow
        q = (p - .25) / .75                                        # 0 -> 1 while locking in
        roll = int((1 - q) ** 2 * h * 1.3) % h
        out = np.roll(frame, roll, axis=0)
        if roll:
            bar = max(2, h // 25)
            out[max(0, roll - bar):roll] = 10                         # the vertical blanking bar
        if v["noise"] > 0 and q < .8:
            rows = max(3, int(h / 10 * (1 - q)))
            top = int(h * .8)
            band = self._noise_band(min(rows, h - top), w, rng, v["noise"] * (1 - q))
            out[top:top + band.shape[0]] = _u8(out[top:top + band.shape[0]] * .5 + band)
        return out

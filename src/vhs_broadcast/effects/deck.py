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


def analog_snow(rng, h, w, colour=.25, streak=1.4, level=105, spread=62):
    """TV snow as a receiver shows it, as float BGR around 0-255.

    Random noise is band-limited along each line (the set's video bandwidth), so it
    becomes short horizontal dashes rather than square pixels. Each line has its own
    gain, peaks clip to white sparkles, and a faint low-resolution colour noise rides
    on top, the way snow looks on a colour set with the colour killer off.
    """
    unit = max(1.0, w / 640)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=streak * unit, sigmaY=.01)
    noise /= noise.std() + 1e-6
    line_gain = 1 + .25 * rng.normal(0, 1, (h, 1)).astype(np.float32)
    luma = level + spread * noise * line_gain
    sparkle = np.maximum(noise - 2.2, 0) * 140                     # bright peaks clip toward white
    luma = luma + sparkle
    out = np.repeat(luma[..., None], 3, axis=2)
    if colour > 0:
        tint = rng.normal(0, 1, (max(2, h // 6), max(2, w // 24), 3)).astype(np.float32)
        tint = cv2.resize(tint, (w, h), interpolation=cv2.INTER_LINEAR)
        out += tint * 28 * colour
    return out


def _band_weight(rng, rows, w, soft):
    """Per-pixel snow weight for a band, 0..1. The core is solid; toward the edges the snow
    breaks up into ragged horizontal dashes and dissolves into the picture over `soft` rows,
    the way a head drifting off-track loses signal gradually instead of at a line."""
    y = np.arange(rows, dtype=np.float32) + .5
    core = np.clip(np.minimum(y, rows - y) / max(1.0, soft), 0, 1)
    core = (core * core * (3 - 2 * core))[:, None]                      # smoothstep
    # The edge wanders along the line and from line to line.
    unit = max(1.0, w / 640)
    dash = rng.normal(0, 1, (rows, w)).astype(np.float32)
    dash = cv2.GaussianBlur(dash, (0, 0), sigmaX=8 * unit, sigmaY=.6)
    dash /= dash.std() + 1e-6
    # Coverage follows the core: halfway out, about half the dashes are there. Each dash
    # also fades in over a range of core values, so nothing switches on at a hard line.
    u = .5 * (1 + np.tanh(.85 * dash))                                  # ~uniform 0..1
    weight = np.clip((core + u - 1) / .45, 0, 1) * np.sqrt(core)
    return weight[..., None]


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
    @staticmethod
    def _blend_band(out, top, rows, rng, strength, slip=0):
        """Snow over a strip of picture, in place. The band fades in and out over half its
        height on each side with ragged edges, the picture inside it slips sideways by
        `slip` pixels, and some lines still tear through as the heads cross tracks."""
        h, w = out.shape[:2]
        margin = max(2, int(rows * .8))
        lo, hi = max(0, top - margin), min(h, top + rows + margin)
        if hi <= lo:
            return
        n = (top + rows + margin) - (top - margin)
        full = _band_weight(rng, n, w, margin + rows / 3)
        weight = full[lo - (top - margin):hi - (top - margin)]
        region = out[lo:hi].astype(np.float32)
        if slip:
            # The slip follows the same soft profile, so the picture shears instead of cutting.
            shifted = np.roll(region, slip, axis=1)
            shear = weight.mean(axis=1, keepdims=True)
            region = region * (1 - shear) + shifted * shear
        # Bar noise is mostly grey with white dashes: lighter, longer streaks than full snow.
        snow = analog_snow(rng, hi - lo, w, colour=.15, streak=3.0, level=135, spread=45)
        keep = (rng.random((hi - lo, 1, 1)) < .35).astype(np.float32)
        weight = weight * strength * (1 - .6 * keep)
        out[lo:hi] = _u8(region * (1 - weight) + snow * weight)

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
            rows = max(4, h // 14)
            top = int(h * (.72 + .02 * math.sin(t * 3)))
            self._blend_band(out, top, rows, rng, v["noise"])
        return out

    def _search(self, frame, t, v, rng, forward):
        h, w = frame.shape[:2]
        out = frame.copy()
        n = int(v["bars"])
        rows = max(4, h // 9)
        direction = 1 if forward else -1
        for k in range(n):
            pos = (k / n + direction * t * .35 * v["bar_speed"]) % 1
            top = int(pos * (h + 2.6 * rows) - 1.8 * rows)   # enters and leaves fully off-screen
            # The picture inside a bar slips sideways as the head crosses tracks.
            self._blend_band(out, top, rows, rng, v["noise"], int(rng.integers(-w // 12, w // 12)))
        return out

    def _settle(self, frame, p, came_from, v, rng):
        """Tape loading: snow (from the blue screen), then the picture rolls into vertical hold."""
        h, w = frame.shape[:2]
        if p < .25:
            snow = _u8(analog_snow(rng, h, w))
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
            rows = min(max(3, int(h / 10 * (1 - q))), h - int(h * .8))
            top = int(h * .8)
            self._blend_band(out, top, rows, rng, v["noise"] * (1 - q))
        return out

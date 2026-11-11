"""The VCR as a machine: keyframe Play, Pause, FF, REW and Stop like pressing its buttons.

The noise is not painted on: it comes from where the video heads sit over the helical
tracks (see ``rf.py``). In search the tape runs at several times play speed, so each
head pass slants across neighbouring tracks and the bars sit where it is over the other
azimuth's tracks - more bars in rewind than fast forward, at the same speed. Pause scans
a stopped tape and crosses one track: one bar. Pressing Play after Stop runs the
tape-loading settle: blue while the tape threads, then FM noise as the drum spins up,
bars sweeping through as the capstan servo pulls the tracking in, and the picture
rolling into vertical hold. Transitions are timed from the mode's own keyframes, so
scrubbing, preview and render agree.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .advanced import TemporalEffect
from .rf import fm_playback, head_contact, track_envelope

PLAY, PAUSE, FF, REW, STOP = range(5)
BLUE = (170, 0, 0)                     # BGR: the classic VCR blue screen
PAUSE_PHASE = .15                      # still tracking parks the pause bar in the bottom of the picture
HEAD = 1.0                            # head width in track pitches (SP: the same as the track)


class VCRDeckEffect(TemporalEffect):
    id = "deck"
    label = "VCR DECK"
    stage = "tape"

    def build(self, params):
        self.choice("mode", "Mode", ("Play", "Pause", "Fast forward", "Rewind", "Stop (blue screen)"))
        self.param("settle", "Settle s", 1.5, 0, 6)
        self.param("speed", "Search speed x", 5, 2, 15, step=1)
        self.param("bar_speed", "Bar drift", .4, 0, 3)
        self.param("noise", "RF loss", 1)
        self.param("cnr", "Tape CNR dB", 18, 8, 40)
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
        state = {"mode": mode, "since": t - since if math.isfinite(since) else -1e9,
                 "from": came_from, "last": frame}
        rng = np.random.default_rng([int(v["seed"]), int(t * 30)])
        if mode == STOP:
            return np.full_like(frame, BLUE), state
        if mode == PAUSE:
            held = previous.get("held")
            if held is None or held.shape != frame.shape:
                held = frame.copy()
            state["held"] = held
            return self._pause(held, t, v, rng), state
        older = previous.get("last")
        if older is None or older.shape != frame.shape:
            older = frame
        if mode in (FF, REW):
            return self._search(frame, older, t, v, rng, mode == FF, 1.0), state
        if came_from == STOP and since < v["settle"]:
            return self._settle(frame, since / max(v["settle"], 1e-3), v, rng), state
        if came_from in (FF, REW) and since < v["settle"] / 3:
            # Back to play speed: the slant flattens out and the servo pulls the bars away.
            k = since / max(v["settle"] / 3, 1e-3)
            return self._search(frame, older, t, v, rng, came_from == FF, (1 - k) ** 2), state
        return frame, state   # out of pause the tape simply resumes

    # ------------------------------------------------------------------ #
    def _play(self, picture, offset, v, rng, lock=1.0):
        """Read `picture` off tape with the head `offset` track pitches off its track (per line)."""
        h = picture.shape[0]
        env = track_envelope(offset, HEAD) * head_contact(rng, h) * lock
        env = 1 - v["noise"] * (1 - np.clip(env, 0, 1))
        if env.min() > .97:
            return picture
        return fm_playback(picture, env, rng, cnr=v["cnr"])

    @staticmethod
    def _lines(h):
        return (np.arange(h, dtype=np.float32) + .5) / h

    def _pause(self, held, t, v, rng):
        h, w = held.shape[:2]
        out = held
        if v["jitter"] > 0:
            # Both heads read the same track: the fields disagree and the picture shudders.
            out = held.copy()
            shift = int(round(v["jitter"] * 2 * (1 if int(t * 30) % 2 else -1)))
            out[1::2] = np.roll(out[1::2], shift, axis=1)
            bounce = int(round(v["jitter"] * rng.integers(-1, 2)))
            out = np.roll(out, bounce, axis=0)
        # Stopped tape: the scan slants across one track pitch, so one bar.
        wander = .025 * v["jitter"] * math.sin(t * 3)
        return self._play(out, PAUSE_PHASE + wander - self._lines(h), v, rng)

    def _search(self, frame, older, t, v, rng, forward, amount):
        h, w = frame.shape[:2]
        ratio = v["speed"] if forward else -v["speed"]
        slope = (ratio - 1) * amount                        # tracks crossed per field
        phase = (v["seed"] * .618) % 2 - v["bar_speed"] * t
        offset = phase + slope * self._lines(h)
        # Each stretch between bars comes off a different track, so a different field.
        track = np.floor((offset + 1) / 2).astype(np.int64)
        picture = np.where((track % 2 == 1)[:, None, None], older, frame)
        if amount > .05:
            picture = self._flag(picture, slope)
        return self._play(picture, offset, v, rng)

    @staticmethod
    def _flag(picture, slope):
        """Off-speed tape shifts the line rate; the TV's horizontal AFC catches up only a
        few lines after each field starts, so the top of the picture bends sideways."""
        h, w = picture.shape[:2]
        top = max(4, h // 6)
        bend = slope * 1.3 * max(1.0, w / 640) * np.exp(-np.arange(top, dtype=np.float32) / (top / 4))
        map_x = np.arange(w, dtype=np.float32)[None, :] - bend[:, None]
        map_y = np.repeat(np.arange(top, dtype=np.float32)[:, None], w, axis=1)
        out = picture.copy()
        out[:top] = cv2.remap(picture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    def _settle(self, frame, p, v, rng):
        """Tape loading after Stop: blue while the tape threads, then FM noise as the drum
        spins up, bars sweeping through while the servo locks, rolling into vertical hold."""
        h = frame.shape[0]
        if p < .12:
            return np.full_like(frame, BLUE)
        q = (p - .12) / .88                                          # 0 -> 1 while locking in
        roll = int((1 - q) ** 2 * h * 1.3) % h
        picture = np.roll(frame, roll, axis=0)
        if roll:
            bar = max(2, h // 25)
            picture[max(0, roll - bar):roll] = 10                    # the vertical blanking bar
        # Capstan still hunting: the slant and the tracking phase both settle to zero.
        hunt = (1 - q) ** 1.5
        offset = 4 * hunt + (2.5 * hunt) * self._lines(h)
        return self._play(picture, offset, v, rng, lock=min(1.0, q * 4))

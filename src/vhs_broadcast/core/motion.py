"""Integrate motion rates without tying position to the current speed.

Fixed time steps make audio/LFO motion independent of preview frame drops.
Live edits start a new segment at the last displayed time, preserving travel.
Recorded tracks are sampled at integration time, including when seeking.
"""
from array import array
from bisect import bisect_right
from dataclasses import replace
import math


class _Segment:
    HZ = 240

    def __init__(self, start, origin, param, track, quadratic):
        self.start = start
        self.origin = origin
        self.param = replace(param, mod=replace(param.mod), audio=replace(param.audio),
                             on_change=None)
        self.times = tuple(track.times) if track else ()
        self.values = tuple(track.values) if track else ()
        self.track = replace(track, times=list(track.times), values=list(track.values)) if track else None
        self.quadratic = quadratic
        self.integrals = array("d", [0.0])
        self.constant = not (param.audio.enabled or param.mod.enabled or self.times)

    def rate(self, t):
        base = self.param.value
        if self.times:
            base = self.track.value_at(t)
        value = self.param.evaluate(t, base_value=base)
        return value * abs(value) if self.quadratic else value

    def at(self, t):
        elapsed = max(0.0, t - self.start)
        if self.constant:
            return self.origin + elapsed * self.rate(self.start)
        ticks = elapsed * self.HZ
        whole = int(math.floor(ticks + 1e-9))
        while len(self.integrals) <= whole:
            i = len(self.integrals) - 1
            value = self.rate(self.start + i / self.HZ)
            self.integrals.append(self.integrals[-1] + value / self.HZ)
        partial = max(0.0, elapsed - whole / self.HZ)
        return (self.origin + self.integrals[whole]
                + partial * self.rate(self.start + whole / self.HZ))


class MotionPhase:
    def __init__(self):
        self.segments = []
        self.signature = None
        self.last_time = 0.0

    def sample(self, param, t, track=None, quadratic=False):
        t = max(0.0, float(t))
        # Timeline.apply updates the current value every frame. Its track, not
        # that instantaneous value, defines the historical speed for playback.
        if track and track.times:
            param = replace(param, value=track.values[0])
        signature = (param.as_dict(), tuple(track.times) if track else (),
                     tuple(track.values) if track else (),
                     track.interpolation if track else "hold", quadratic)
        if signature != self.signature:
            start = min(t, self.last_time) if self.segments else 0.0
            origin = self._at(start) if self.segments else 0.0
            self.segments = [s for s in self.segments if s.start < start]
            self.segments.append(_Segment(start, origin, param, track, quadratic))
            self.signature = signature
        self.last_time = t
        return self._at(t)

    def _at(self, t):
        index = bisect_right([s.start for s in self.segments], t) - 1
        return self.segments[max(0, index)].at(t)

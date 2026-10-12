"""Manually authored keyframes, hold/linear/smooth interpolation, and sessions.

Changing a control never writes to a track. The editor explicitly adds, moves,
or deletes keys; playback/render sample the same curves at video time.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import Optional

from .parameters import ParameterSet, Modulator, AudioMod


@dataclass
class Track:
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    interpolation: str = "hold"

    def add(self, t: float, v: float) -> None:
        if not math.isfinite(t) or t < 0 or not math.isfinite(v):
            raise ValueError("Keyframe time and value must be finite; time cannot be negative.")
        # Keep sorted by time; overwrite if an event already sits at ~t.
        i = bisect.bisect_left(self.times, t)
        if i < len(self.times) and abs(self.times[i] - t) < 1e-6:
            self.values[i] = v
            return
        self.times.insert(i, t)
        self.values.insert(i, v)

    def value_at(self, t: float) -> Optional[float]:
        if not self.times:
            return None
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return self.values[0]
        if i == len(self.times) - 1 or self.interpolation == "hold":
            return self.values[i]
        f = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        if self.interpolation == "smooth":
            f = f * f * (3 - 2 * f)
        return self.values[i] + f * (self.values[i + 1] - self.values[i])

    def remove(self, t: float) -> None:
        i = bisect.bisect_left(self.times, t)
        if i < len(self.times) and abs(self.times[i] - t) < 1e-6:
            del self.times[i]
            del self.values[i]

    def clear_after(self, t: float) -> None:
        i = bisect.bisect_left(self.times, t)
        del self.times[i:]
        del self.values[i:]


class Timeline:
    def __init__(self) -> None:
        self.tracks: dict[str, Track] = {}

    def record(self, t: float, pid: str, value: float) -> None:
        self.tracks.setdefault(pid, Track()).add(t, value)

    def set_key(self, t: float, param) -> None:
        mode = "hold" if param.choices or (param.step == 1 and param.minimum == 0 and param.maximum == 1) else "linear"
        old = self.tracks.get(param.id)
        track = replace(old, times=list(old.times), values=list(old.values)) if old else Track(interpolation=mode)
        track.add(t, param.value)
        self.tracks = {**self.tracks, param.id: track}

    def delete_key(self, pid: str, t: float) -> None:
        track = self.tracks.get(pid)
        if track:
            track = replace(track, times=list(track.times), values=list(track.values))
            track.remove(t)
            tracks = {**self.tracks, pid: track}
            if not track.times:
                del tracks[pid]
            self.tracks = tracks

    def apply(self, t: float, params: ParameterSet) -> None:
        """Drive parameter base values from the timeline at time t."""
        for pid, track in list(self.tracks.items()):
            if pid in params:
                v = track.value_at(t)
                if v is not None:
                    params.get(pid).set_base(v, notify=False)

    def clear(self) -> None:
        self.tracks.clear()

    def duration(self) -> float:
        return max((tr.times[-1] for tr in self.tracks.values() if tr.times),
                   default=0.0)

    def as_dict(self) -> dict:
        return {
            "tracks": {
                pid: {"t": list(tr.times), "v": list(tr.values), "interpolation": tr.interpolation}
                for pid, tr in self.tracks.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        tl = cls()
        for pid, td in d.get("tracks", {}).items():
            mode = td.get("interpolation", "hold")
            tr = Track(interpolation=mode if mode in ("hold", "linear", "smooth") else "hold")
            for t, value in zip(td["t"], td["v"]):
                tr.add(float(t), float(value))
            tl.tracks[pid] = tr
        return tl


class Recorder:
    """Legacy transport clock. Keyframes are now created explicitly by the editor.

    Kept for engine compatibility; arming or changing controls never writes keys.
    """

    def __init__(self, params: ParameterSet, timeline: Timeline) -> None:
        self._params = params
        self.timeline = timeline
        self.armed = False
        self._time = 0.0

    def set_time(self, t: float) -> None:
        self._time = t

    def arm(self, on: bool) -> None:
        self.armed = False


class Session:
    """Everything needed to reproduce a take: params + timeline + source info."""

    VERSION = 2

    def __init__(self, params: ParameterSet, timeline: Timeline, effects=()) -> None:
        self.params = params
        self.timeline = timeline
        self.effects = effects
        self.source_path: Optional[str] = None
        self.order: Optional[list] = None      # effect id order (chain)

    def save(self, path: str) -> None:
        data = {
            "version": self.VERSION,
            "source": self.source_path,
            "order": self.order,
            "parameters": self.params.snapshot(),
            "effect_settings": {e.id: e.settings() for e in self.effects
                                if hasattr(e, "settings")},
            "timeline": self.timeline.as_dict(),
        }
        fd, temporary = tempfile.mkstemp(prefix=".vhs-take-", suffix=".json",
                                         dir=os.path.dirname(os.path.abspath(path)))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("This file is not a take object.")
        if data.get("version", 1) > self.VERSION:
            raise ValueError("This take was saved by a newer version of the app.")
        tracks = Timeline.from_dict(data.get("timeline", {})).tracks
        saved_params = self.params.snapshot()
        saved_settings = {e.id: e.settings() for e in self.effects if hasattr(e, "settings")}
        try:
            # A take must not inherit new controls/modulation from the previous
            # audition when loading a file saved before those controls existed.
            for param in self.params.all():
                param.set_base(param.default, notify=False)
                param.mod = Modulator()
                param.audio = AudioMod()
            self.params.restore(data.get("parameters", {}))
            for effect in self.effects:
                if hasattr(effect, "restore_settings"):
                    effect.restore_settings(data.get("effect_settings", {}).get(effect.id, {}))
        except Exception:
            self.params.restore(saved_params)
            for effect in self.effects:
                if effect.id in saved_settings:
                    effect.restore_settings(saved_settings[effect.id])
            raise
        self.source_path = data.get("source")
        self.timeline.tracks = tracks
        self.order = data.get("order")

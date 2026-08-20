"""
Automation: record control movements against video time, then replay them.

Think of it exactly like a MIDI recording:
  - While ARMED and playing, every Parameter base-value change is stored as an
    event (t_seconds, param_id, value).
  - On playback, for each parameter we look up the last event at or before the
    current time ("hold last value") -- this is the keyframe track.
  - A timeline is just dict[param_id -> sorted list of (t, value)] and is trivial
    to serialise to JSON and (later) to edit in a timeline UI.

Because Parameter.evaluate(t) is deterministic, applying the timeline at each
frame time and then re-running the pipeline reproduces the take exactly.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Optional

from .parameters import ParameterSet


@dataclass
class Track:
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(self, t: float, v: float) -> None:
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
            return None
        return self.values[i]

    def clear_after(self, t: float) -> None:
        i = bisect.bisect_left(self.times, t)
        del self.times[i:]
        del self.values[i:]


class Timeline:
    def __init__(self) -> None:
        self.tracks: dict[str, Track] = {}

    def record(self, t: float, pid: str, value: float) -> None:
        self.tracks.setdefault(pid, Track()).add(t, value)

    def apply(self, t: float, params: ParameterSet) -> None:
        """Drive parameter base values from the timeline at time t."""
        for pid, track in self.tracks.items():
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
                pid: {"t": tr.times, "v": tr.values}
                for pid, tr in self.tracks.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        tl = cls()
        for pid, td in d.get("tracks", {}).items():
            tr = Track(times=list(td["t"]), values=list(td["v"]))
            tl.tracks[pid] = tr
        return tl


class Recorder:
    """Bridges ParameterSet change events into a Timeline while armed."""

    def __init__(self, params: ParameterSet, timeline: Timeline) -> None:
        self._params = params
        self.timeline = timeline
        self.armed = False
        self._time = 0.0
        params.set_change_hook(self._on_change)

    def set_time(self, t: float) -> None:
        self._time = t

    def arm(self, on: bool) -> None:
        self.armed = on

    def _on_change(self, pid: str, value: float) -> None:
        if self.armed:
            self.timeline.record(self._time, pid, value)


class Session:
    """Everything needed to reproduce a take: params + timeline + source info."""

    VERSION = 1

    def __init__(self, params: ParameterSet, timeline: Timeline) -> None:
        self.params = params
        self.timeline = timeline
        self.source_path: Optional[str] = None
        self.order: Optional[list] = None      # effect id order (chain)

    def save(self, path: str) -> None:
        data = {
            "version": self.VERSION,
            "source": self.source_path,
            "order": self.order,
            "parameters": self.params.snapshot(),
            "timeline": self.timeline.as_dict(),
            "order": self.order,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.source_path = data.get("source")
        self.order = data.get("order")
        self.params.restore(data.get("parameters", {}))
        self.timeline.tracks = Timeline.from_dict(data.get("timeline", {})).tracks
        self.order = data.get("order")

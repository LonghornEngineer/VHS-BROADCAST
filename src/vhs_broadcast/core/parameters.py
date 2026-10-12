"""
Parameter model.

Every knob, slider and toggle in the app is a Parameter with a unique id.
That single abstraction gives us three things for free:

  1. Manual keyframes store selected base values by parameter id and time.
  2. The "wavy" tolerance -- each Parameter carries a Modulator (an LFO) whose
     output is added to the base value at evaluation time.
  3. Deterministic re-render -- evaluate(t) is a pure function of time, so the
     live preview and the offline render produce identical pixels.

Nothing in here imports Qt, numpy or cv2 -- it is plain, testable Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class ModShape(str, Enum):
    OFF = "off"
    SINE = "sine"
    TRIANGLE = "triangle"   # linear up/down ramp ("linear shift")
    RANDOM = "random"       # sample-and-hold value noise, smoothed


def _smoothstep(a: float, b: float, t: float) -> float:
    t = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * t


def _noise_sample(seed: int, index: int) -> float:
    """Deterministic value in [-1, 1] for a given (seed, integer index).

    Pure function -> identical in preview and offline render regardless of
    frame rate or when it is called.
    """
    # Mix seed and index into a single 64-bit int (splitmix-ish) so the value
    # is a pure, deterministic function of (seed, index).
    z = ((seed & 0xFFFFFFFF) * 0x9E3779B1 + (index & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0


@dataclass
class Modulator:
    """An LFO applied on top of a Parameter's base value.

    output(t) = depth * shape(rate * t + phase)   in roughly [-depth, depth]

    `enabled` is the master switch (driven by the UI '~' checkbox). When off,
    the modulator contributes nothing regardless of shape/depth.
    """
    shape: ModShape = ModShape.SINE
    depth: float = 0.0          # in the parameter's own units
    rate: float = 1.0           # Hz (cycles / second, or samples/sec for RANDOM)
    phase: float = 0.0          # 0..1
    seed: int = 0               # for RANDOM determinism
    enabled: bool = False

    def value(self, t: float) -> float:
        if not self.enabled or self.shape == ModShape.OFF or self.depth == 0.0:
            return 0.0
        if self.shape == ModShape.SINE:
            return self.depth * math.sin(2.0 * math.pi * (self.rate * t + self.phase))
        if self.shape == ModShape.TRIANGLE:
            # phase-shifted triangle wave in [-1, 1]
            x = (self.rate * t + self.phase) % 1.0
            tri = 4.0 * abs(x - 0.5) - 1.0
            return self.depth * tri
        if self.shape == ModShape.RANDOM:
            pos = self.rate * t + self.phase
            i = math.floor(pos)
            frac = pos - i
            a = _noise_sample(self.seed, i)
            b = _noise_sample(self.seed, i + 1)
            return self.depth * _smoothstep(a, b, frac)
        return 0.0

    def as_dict(self) -> dict:
        return {
            "shape": self.shape.value,
            "depth": self.depth,
            "rate": self.rate,
            "phase": self.phase,
            "seed": self.seed,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Modulator":
        return cls(
            shape=ModShape(d.get("shape", "sine")),
            depth=float(d.get("depth", 0.0)),
            rate=float(d.get("rate", 1.0)),
            phase=float(d.get("phase", 0.0)),
            seed=int(d.get("seed", 0)),
            enabled=bool(d.get("enabled", False)),
        )


# --- audio-reactive modulation ------------------------------------------- #
# The heavy audio analysis lives in core.audio_react (numpy). It registers a
# provider here so this module stays dependency-free. provider(band, t, smooth)
# returns a normalised band envelope in ~[0, 1.3], or 0 when no audio is loaded.
_audio_provider: Optional[Callable] = None


def set_audio_provider(fn: Optional[Callable]) -> None:
    global _audio_provider
    _audio_provider = fn


@dataclass
class AudioMod:
    """Trigger mode returns a 0..1 ramp from rest to the parameter's base target.
    Legacy envelope mode returns a signed depth * gain-scaled band envelope.
    """
    enabled: bool = False
    band: str = "low"
    depth: float = 0.0          # in the parameter's own units (signed)
    gain: float = 1.0           # sensitivity multiplier before clamp
    smooth: float = 0.3         # 0 = snappy (transients), 1 = smooth (swells)
    mode: str = "trigger"
    low_hz: float = 20.0
    high_hz: float = 250.0
    threshold: float = 0.35
    ramp: float = 0.15          # seconds to reach the target after activation
    rest: Optional[float] = None

    def value(self, t: float) -> float:
        if not self.enabled or _audio_provider is None:
            return 0.0
        if self.mode == "trigger":
            return _audio_provider(self.band, t, 0.0, self.low_hz,
                                   self.high_hz, self.threshold, self.ramp)
        env = _audio_provider(self.band, t, self.smooth)
        shaped = max(0.0, min(1.0, self.gain * env))
        return self.depth * shaped

    def as_dict(self) -> dict:
        return {"enabled": self.enabled, "band": self.band, "depth": self.depth,
                "gain": self.gain, "smooth": self.smooth, "mode": self.mode,
                "low_hz": self.low_hz, "high_hz": self.high_hz,
                "threshold": self.threshold, "ramp": self.ramp, "rest": self.rest}

    @classmethod
    def from_dict(cls, d: dict) -> "AudioMod":
        return cls(
            enabled=bool(d.get("enabled", False)),
            band=str(d.get("band", "low")),
            depth=float(d.get("depth", 0.0)),
            gain=float(d.get("gain", 1.0)),
            smooth=float(d.get("smooth", 0.3)),
            mode=str(d.get("mode", "envelope")),
            low_hz=float(d.get("low_hz", 20.0)),
            high_hz=float(d.get("high_hz", 250.0)),
            threshold=float(d.get("threshold", 0.35)),
            ramp=float(d.get("ramp", 0.15)),
            rest=float(d["rest"]) if d.get("rest") is not None else None,
        )


@dataclass
class Parameter:
    id: str
    label: str
    value: float                # the "base" value the user sets
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    mod: Modulator = field(default_factory=Modulator)
    audio: AudioMod = field(default_factory=AudioMod)
    # Called with (id, new_base_value) whenever the base value changes.
    # The automation recorder installs itself here.
    on_change: Optional[Callable[[str, float], None]] = None
    default: Optional[float] = None
    choices: tuple[str, ...] = ()

    def __post_init__(self):
        if self.default is None:
            self.default = self.value

    def reset(self):
        self.set_base(self.default)

    def clamp(self, v: float) -> float:
        value = max(self.minimum, min(self.maximum, v))
        return float(round(value)) if self.choices else value

    def set_base(self, v: float, *, notify: bool = True) -> None:
        v = self.clamp(v)
        if v == self.value:
            return
        self.value = v
        if notify and self.on_change is not None:
            self.on_change(self.id, v)

    def evaluate(self, t: float, *, base_value: Optional[float] = None) -> float:
        """Effective value at time t: base + LFO + audio, clamped to range."""
        value = self.value if base_value is None else base_value
        if self.audio.enabled and self.audio.mode == "trigger":
            rest = self.clamp(0.0) if self.audio.rest is None else self.audio.rest
            base = rest + (value - rest) * self.audio.value(t)
            return self.clamp(base + self.mod.value(t))
        return self.clamp(value + self.mod.value(t) + self.audio.value(t))

    def as_dict(self) -> dict:
        return {"value": self.value, "mod": self.mod.as_dict(),
                "audio": self.audio.as_dict()}


class ParameterSet:
    """Flat registry of every Parameter in the app, keyed by id."""

    def __init__(self) -> None:
        self._params: dict[str, Parameter] = {}
        self._change_hook: Optional[Callable[[str, float], None]] = None
        self._motion = {}
        self._motion_timeline = None

    def reset_motion(self, timeline=None):
        self._motion.clear()
        self._motion_timeline = timeline

    def resume_live_motion(self):
        """Leave timeline preview without discarding already accumulated travel."""
        self._motion_timeline = None

    def motion_phase(self, pid: str, t: float, *, quadratic=False) -> float:
        from .motion import MotionPhase
        key = (pid, quadratic)
        phase = self._motion.get(key)
        if phase is None:
            phase = MotionPhase()
            self._motion[key] = phase
        track = (self._motion_timeline.tracks.get(pid)
                 if self._motion_timeline is not None else None)
        return phase.sample(self.get(pid), t, track, quadratic)

    def add(self, p: Parameter) -> Parameter:
        if p.id in self._params:
            raise ValueError(f"duplicate parameter id: {p.id}")
        p.on_change = self._fire
        self._params[p.id] = p
        return p

    def get(self, pid: str) -> Parameter:
        return self._params[pid]

    def __contains__(self, pid: str) -> bool:
        return pid in self._params

    def all(self) -> list[Parameter]:
        return list(self._params.values())

    def set_change_hook(self, hook: Optional[Callable[[str, float], None]]) -> None:
        """Install the UI's manual-control change listener."""
        self._change_hook = hook

    def _fire(self, pid: str, value: float) -> None:
        if self._change_hook is not None:
            self._change_hook(pid, value)

    def snapshot(self) -> dict:
        return {pid: p.as_dict() for pid, p in self._params.items()}

    def restore(self, data: dict) -> None:
        self.reset_motion(self._motion_timeline)
        for pid, d in data.items():
            if pid in self._params:
                p = self._params[pid]
                p.set_base(float(d["value"]), notify=False)
                if "mod" in d:
                    p.mod = Modulator.from_dict(d["mod"])
                if "audio" in d:
                    p.audio = AudioMod.from_dict(d["audio"])

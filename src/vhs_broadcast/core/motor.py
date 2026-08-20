"""Motor wow/flutter: warps the mapping from output time to source time.

speed(t) is the instantaneous playback rate of a worn tape deck: a slow 'wow'
sine plus faster 'flutter' noise. The engine and renderer accumulate it to pick
source frames; the audio engine uses the same curve to resample.
"""
from __future__ import annotations

import math

from .parameters import ParameterSet, _noise_sample


def motor_enabled(params: ParameterSet) -> bool:
    return "motor.enabled" in params and params.get("motor.enabled").value >= 0.5


def motor_affects_audio(params: ParameterSet) -> bool:
    return (motor_enabled(params)
            and "motor.audio" in params
            and params.get("motor.audio").value >= 0.5)


def motor_speed(params: ParameterSet, t: float) -> float:
    """Instantaneous speed multiplier at output time t (1.0 == nominal)."""
    if not motor_enabled(params):
        return 1.0
    depth = params.get("motor.depth").evaluate(t)
    rate = params.get("motor.rate").evaluate(t)
    if depth <= 1e-4:
        return 1.0
    wow = math.sin(2 * math.pi * rate * t)
    # deterministic flutter (value noise) at ~7x the wow rate
    pos = rate * 7.0 * t
    i = math.floor(pos)
    frac = pos - i
    a = _noise_sample(0xF00D, i)
    b = _noise_sample(0xF00D, i + 1)
    flutter = a + (b - a) * (frac * frac * (3 - 2 * frac))
    speed = 1.0 + depth * (0.75 * wow + 0.25 * flutter)
    return max(0.15, min(3.0, speed))

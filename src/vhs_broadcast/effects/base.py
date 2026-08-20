"""Base class for every VHS effect.

An Effect owns a set of Parameters (registered into the shared ParameterSet so
they are automatable) and exposes apply(frame, t) which mutates/returns a BGR
uint8 frame. Every effect has an implicit `<id>.enabled` toggle parameter so it
can be switched off entirely.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.parameters import Parameter, ParameterSet

if TYPE_CHECKING:
    from ..core.parameters import Modulator


class Effect:
    #: stable, unique short id -- used as prefix for parameter ids
    id: str = "effect"
    #: human label for the UI panel
    label: str = "Effect"

    def __init__(self, params: ParameterSet) -> None:
        self.params = params
        self._enabled = params.add(Parameter(
            id=f"{self.id}.enabled", label="Enabled",
            value=0.0, minimum=0.0, maximum=1.0, step=1.0,
        ))
        self._own: list[Parameter] = []
        self.build(params)

    # ---- subclass API -----------------------------------------------------
    def build(self, params: ParameterSet) -> None:
        """Register this effect's parameters. Override in subclasses."""

    def process(self, frame: np.ndarray, t: float, values: dict[str, float]) -> np.ndarray:
        """Return the processed BGR frame. Override in subclasses.

        `values` maps this effect's short parameter names (without the id
        prefix) to their evaluated value at time t.
        """
        return frame

    # ---- helpers ----------------------------------------------------------
    def param(self, short_id: str, label: str, value: float,
              lo: float = 0.0, hi: float = 1.0, step: float = 0.01) -> Parameter:
        p = self.params.add(Parameter(
            id=f"{self.id}.{short_id}", label=label,
            value=value, minimum=lo, maximum=hi, step=step,
        ))
        self._own.append(p)
        return p

    def own_params(self) -> list[Parameter]:
        return list(self._own)

    def enabled_param(self) -> Parameter:
        return self._enabled

    # ---- runtime ----------------------------------------------------------
    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self._enabled.evaluate(t) < 0.5:
            return frame
        values = {
            p.id.split(".", 2)[-1]: p.evaluate(t)
            for p in self._own
        }
        return self.process(frame, t, values)

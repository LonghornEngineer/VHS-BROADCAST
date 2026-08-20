"""A group of controls for one effect. The enable checkbox carries the effect
name (checked = on), so there's no separate title crowding the checkbox."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QVBoxLayout

from ...effects.base import Effect
from .param_row import ParamRow


class EffectPanel(QGroupBox):
    def __init__(self, effect: Effect, parent=None, header_text: Optional[str] = None):
        super().__init__("", parent)          # no title -> checkbox names it
        self.effect = effect
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.setContentsMargins(6, 4, 6, 6)

        header = QHBoxLayout()
        self.enable = QCheckBox(header_text or effect.label)
        self.enable.setObjectName("EffectTitle")
        self.enable.setChecked(effect.enabled_param().value >= 0.5)
        self.enable.toggled.connect(self._on_enable)
        header.addWidget(self.enable)
        header.addStretch(1)
        lay.addLayout(header)

        self.rows: list[ParamRow] = []
        for p in effect.own_params():
            row = ParamRow(p)
            self.rows.append(row)
            lay.addWidget(row)

    def _on_enable(self, on: bool):
        self.effect.enabled_param().set_base(1.0 if on else 0.0)

    def refresh(self):
        self.enable.blockSignals(True)
        self.enable.setChecked(self.effect.enabled_param().value >= 0.5)
        self.enable.blockSignals(False)
        for r in self.rows:
            r.refresh()

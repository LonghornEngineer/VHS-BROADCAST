"""One control row bound to a Parameter.

- Boolean params (0/1, step 1) render as a checkbox.
- Numeric params render as a slider + value, plus two small buttons that open
  popup editors (no inline expansion, so the panel layout never reflows):
    ~   LFO modulation      (sine / linear / random)
    note  audio-reactive    (band / depth / gain / smooth)
  Both stack on the base value, mirroring Parameter.evaluate(). A button lights
  up while its modulation is enabled.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSlider,
    QVBoxLayout, QWidget, QWidgetAction,
)

from ...core.parameters import ModShape, Parameter

_SHAPES = [("SIN", ModShape.SINE), ("LIN", ModShape.TRIANGLE),
           ("RND", ModShape.RANDOM)]
_ABANDS = [("LOW", "low"), ("MID", "mid"), ("HIGH", "high"), ("LEVEL", "level")]
_STEPS = 1000


def _is_bool(p: Parameter) -> bool:
    return p.minimum == 0.0 and p.maximum == 1.0 and p.step == 1.0


class ParamRow(QWidget):
    def __init__(self, param: Parameter, parent=None):
        super().__init__(parent)
        self.param = param
        self._syncing = False
        self.bool_mode = _is_bool(param)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        if self.bool_mode:
            row = QHBoxLayout()
            self.check = QCheckBox(param.label.upper())
            self.check.toggled.connect(self._on_bool)
            row.addWidget(self.check)
            row.addStretch(1)
            root.addLayout(row)
            self.refresh()
            return

        top = QHBoxLayout()
        top.setSpacing(3)
        self.label = QLabel(param.label.upper())
        self.label.setFixedWidth(58)
        self.label.setObjectName("SectionHint")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, _STEPS)
        self.slider.valueChanged.connect(self._on_slider)
        self.value_lbl = QLabel()
        self.value_lbl.setObjectName("Value")
        self.value_lbl.setFixedWidth(34)
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.mod_btn = QPushButton("~")
        self.mod_btn.setObjectName("ModBtn")
        self.mod_btn.setFixedWidth(20)
        self.mod_btn.setToolTip("LFO modulation")
        self.mod_btn.clicked.connect(lambda: self._open_editor("lfo"))
        self.audio_btn = QPushButton("\u266a")
        self.audio_btn.setObjectName("AudioBtn")
        self.audio_btn.setFixedWidth(20)
        self.audio_btn.setToolTip("Audio-reactive modulation")
        self.audio_btn.clicked.connect(lambda: self._open_editor("audio"))
        top.addWidget(self.label)
        top.addWidget(self.slider, 1)
        top.addWidget(self.value_lbl)
        top.addWidget(self.mod_btn)
        top.addWidget(self.audio_btn)
        root.addLayout(top)
        self.refresh()

    # ---- mapping ----------------------------------------------------------
    @property
    def _span(self):
        return (self.param.maximum - self.param.minimum) or 1.0

    def _to_slider(self, v):
        return int((v - self.param.minimum) / self._span * _STEPS)

    def _from_slider(self, s):
        return self.param.minimum + (s / _STEPS) * self._span

    # ---- base value -------------------------------------------------------
    def _on_bool(self, on):
        if not self._syncing:
            self.param.set_base(1.0 if on else 0.0)

    def _on_slider(self, s):
        if self._syncing:
            return
        v = self._from_slider(s)
        self.param.set_base(v)
        self.value_lbl.setText(f"{v:.2f}")

    # ---- popup editors ----------------------------------------------------
    def _open_editor(self, kind: str):
        menu = QMenu(self)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(self._build_lfo() if kind == "lfo"
                            else self._build_audio())
        menu.addAction(wa)
        menu.aboutToHide.connect(self._refresh_btn_styles)
        btn = self.mod_btn if kind == "lfo" else self.audio_btn
        menu.exec(btn.mapToGlobal(QPoint(btn.width(), btn.height())))

    def _slider_row(self, layout, name, lo, hi, val, fmt):
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel(name)
        lbl.setFixedWidth(52)
        lbl.setObjectName("SectionHint")
        sld = QSlider(Qt.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(int(val))
        sld.setMinimumWidth(150)
        vlbl = QLabel(fmt(val))
        vlbl.setObjectName("Value")
        vlbl.setFixedWidth(46)
        vlbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sld.valueChanged.connect(lambda x: vlbl.setText(fmt(x)))
        row.addWidget(lbl)
        row.addWidget(sld, 1)
        row.addWidget(vlbl)
        layout.addLayout(row)
        return sld

    def _build_lfo(self) -> QWidget:
        w = QWidget()
        w.setObjectName("ModPanel")
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(5)
        head = QLabel(f"LFO  \u2022  {self.param.label.upper()}")
        head.setObjectName("ModHead")
        v.addWidget(head)
        en = QCheckBox("ENABLED")
        en.setChecked(self.param.mod.enabled)
        v.addWidget(en)
        r = QHBoxLayout()
        r.setSpacing(6)
        r.addWidget(QLabel("SHAPE"))
        shape = QComboBox()
        shape.setMinimumWidth(90)
        for name, _ in _SHAPES:
            shape.addItem(name)
        shape.setCurrentIndex(
            next((i for i, (_, s) in enumerate(_SHAPES)
                  if s == self.param.mod.shape), 0))
        r.addWidget(shape, 1)
        v.addLayout(r)
        depth = self._slider_row(
            v, "DEPTH", 0, _STEPS,
            (self.param.mod.depth / self._span) * _STEPS,
            lambda x: f"{(x / _STEPS) * self._span:.2f}")
        rate = self._slider_row(
            v, "RATE Hz", 1, 2000, self.param.mod.rate * 100,
            lambda x: f"{x / 100:.2f}")

        def apply(*_):
            m = self.param.mod
            m.enabled = en.isChecked()
            m.shape = _SHAPES[shape.currentIndex()][1]
            m.depth = (depth.value() / _STEPS) * self._span
            m.rate = rate.value() / 100.0
            self._refresh_btn_styles()

        en.toggled.connect(apply)
        shape.currentIndexChanged.connect(apply)
        depth.valueChanged.connect(apply)
        rate.valueChanged.connect(apply)
        return w

    def _build_audio(self) -> QWidget:
        w = QWidget()
        w.setObjectName("ModPanel")
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(5)
        head = QLabel(f"AUDIO REACT  \u2022  {self.param.label.upper()}")
        head.setObjectName("ModHead")
        v.addWidget(head)
        en = QCheckBox("ENABLED")
        en.setChecked(self.param.audio.enabled)
        v.addWidget(en)
        r = QHBoxLayout()
        r.setSpacing(6)
        r.addWidget(QLabel("BAND"))
        band = QComboBox()
        band.setMinimumWidth(100)
        for name, _ in _ABANDS:
            band.addItem(name)
        band.setCurrentIndex(
            next((i for i, (_, k) in enumerate(_ABANDS)
                  if k == self.param.audio.band), 0))
        r.addWidget(band, 1)
        v.addLayout(r)
        depth = self._slider_row(
            v, "DEPTH", -_STEPS, _STEPS,
            (self.param.audio.depth / self._span) * _STEPS,
            lambda x: f"{(x / _STEPS) * self._span:+.2f}")
        gain = self._slider_row(
            v, "GAIN", 0, 400, self.param.audio.gain * 100,
            lambda x: f"{x / 100:.2f}x")
        smooth = self._slider_row(
            v, "SMOOTH", 0, 100, self.param.audio.smooth * 100,
            lambda x: f"{x / 100:.2f}")

        def apply(*_):
            a = self.param.audio
            a.enabled = en.isChecked()
            a.band = _ABANDS[band.currentIndex()][1]
            a.depth = (depth.value() / _STEPS) * self._span
            a.gain = gain.value() / 100.0
            a.smooth = smooth.value() / 100.0
            self._refresh_btn_styles()

        en.toggled.connect(apply)
        band.currentIndexChanged.connect(apply)
        for s in (depth, gain, smooth):
            s.valueChanged.connect(apply)
        return w

    # ---- sync -------------------------------------------------------------
    def _refresh_btn_styles(self):
        for btn, on in ((self.mod_btn, self.param.mod.enabled),
                        (self.audio_btn, self.param.audio.enabled)):
            btn.setProperty("active", "true" if on else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def refresh(self):
        self._syncing = True
        if self.bool_mode:
            self.check.setChecked(self.param.value >= 0.5)
            self._syncing = False
            return
        self.slider.setValue(self._to_slider(self.param.value))
        self.value_lbl.setText(f"{self.param.value:.2f}")
        self._refresh_btn_styles()
        self._syncing = False

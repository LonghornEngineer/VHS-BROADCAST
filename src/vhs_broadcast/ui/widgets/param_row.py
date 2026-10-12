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

from PySide6.QtCore import Qt, QPoint, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSlider,
    QVBoxLayout, QWidget, QWidgetAction, QDoubleSpinBox, QProgressBar,
)

from ...core.parameters import ModShape, Parameter
from .reset_slider import ResetSlider

_SHAPES = [("SIN", ModShape.SINE), ("LIN", ModShape.TRIANGLE),
           ("RND", ModShape.RANDOM)]
_ABANDS = [("LOW", "low"), ("MID", "mid"), ("HIGH", "high"), ("LEVEL", "level")]
_STEPS = 1000
_MOTION_RATES = {"tracking.rollspeed", "tracking.speed", "wobble.speed",
                 "scramble.roll", "hue_shift.speed", "frame_move.xspeed", "frame_move.yspeed",
                 "composite.speed", "time_displace.speed", "rf_interference.speed"}


def _is_bool(p: Parameter) -> bool:
    return p.minimum == 0.0 and p.maximum == 1.0 and p.step == 1.0


class ParamRow(QWidget):
    keyframeRequested = Signal(str)

    def __init__(self, param: Parameter, parent=None):
        super().__init__(parent)
        self.param = param
        self._syncing = False
        self.bool_mode = _is_bool(param)
        self.choice_mode = bool(param.choices)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        if self.choice_mode:
            row = QHBoxLayout()
            row.addWidget(QLabel(param.label.upper()))
            self.choice = QComboBox()
            self.choice.addItems(param.choices)
            self.choice.currentIndexChanged.connect(lambda index: None if self._syncing else param.set_base(index))
            row.addWidget(self.choice, 1)
            row.addWidget(self._key_button())
            root.addLayout(row)
            self.refresh()
            return

        if self.bool_mode:
            row = QHBoxLayout()
            self.check = QCheckBox(param.label.upper())
            if param.id.endswith(".reset"):
                self.check.setToolTip("Reset on the off-to-on transition. Turn off again before another pulse. Add keys explicitly to schedule resets.")
            self.check.toggled.connect(self._on_bool)
            row.addWidget(self.check)
            row.addStretch(1)
            row.addWidget(self._key_button())
            root.addLayout(row)
            self.refresh()
            return

        top = QHBoxLayout()
        top.setSpacing(3)
        self.label = QLabel(param.label.upper())
        self.label.setToolTip(param.label)
        if param.id == "tracking.roll":
            self.setToolTip("Direct vertical position. Use Roll Rate for continuous motion that holds when audio stops.")
        elif param.id in _MOTION_RATES:
            self.setToolTip("Motion speed: changes accumulate without resetting position. Set audio REST to 0 to stop in place.")
        elif param.id in ("frame_move.x", "frame_move.y"):
            self.setToolTip("Direct pan offset, as a fraction of the frame. Use X Rate / Y Rate for continuous wrapped movement.")
        elif param.id == "frame_move.zoom":
            self.setToolTip("1 = original size. Below 1 tiles the picture; above 1 zooms in. Edges always wrap.")
        self.label.setFixedWidth(58)
        self.label.setObjectName("SectionHint")
        self.slider = ResetSlider(Qt.Horizontal)
        self.slider.setRange(0, _STEPS)
        self.slider.setToolTip(f"Double-click to reset to {param.default:g}")
        self.slider.resetRequested.connect(self._reset_value)
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
        top.addWidget(self._key_button())
        root.addLayout(top)
        self.refresh()

    # ---- mapping ----------------------------------------------------------
    def _key_button(self):
        button = QPushButton("◆")
        button.setFixedWidth(20)
        button.setToolTip("Manually add this control's current value as a keyframe at the playhead")
        button.clicked.connect(lambda: self.keyframeRequested.emit(self.param.id))
        return button

    @property
    def _span(self):
        return (self.param.maximum - self.param.minimum) or 1.0

    def _to_slider(self, v):
        return int((v - self.param.minimum) / self._span * _STEPS)

    def _from_slider(self, s):
        return self.param.minimum + (s / _STEPS) * self._span

    # ---- base value -------------------------------------------------------
    def _reset_value(self):
        self.param.reset()
        self.refresh()

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
        menu.deleteLater()

    def _slider_row(self, layout, name, lo, hi, val, fmt, default=0):
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel(name)
        lbl.setFixedWidth(52)
        lbl.setObjectName("SectionHint")
        sld = ResetSlider(Qt.Horizontal)
        sld.resetRequested.connect(lambda: sld.setValue(int(default)))
        sld.setToolTip("Double-click to reset")
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
            lambda x: f"{x / 100:.2f}", default=100)

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

    def _build_legacy_audio(self) -> QWidget:
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
            lambda x: f"{x / 100:.2f}x", default=100)
        smooth = self._slider_row(
            v, "SMOOTH", 0, 100, self.param.audio.smooth * 100,
            lambda x: f"{x / 100:.2f}", default=30)

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

    def _build_audio(self) -> QWidget:
        from ...core.audio_react import current
        a = self.param.audio
        if a.mode != "trigger":
            w = self._build_legacy_audio()
            convert = QPushButton("USE FREQUENCY / THRESHOLD MODE")
            w.layout().addWidget(convert)
            def migrate():
                a.mode = "trigger"
                a.low_hz, a.high_hz = {"low": (20, 250), "mid": (250, 2000),
                                      "high": (2000, 16000)}.get(a.band, (0, 20000))
                a.rest = self.param.value
                self.param.set_base(self.param.clamp(self.param.value + a.depth))
                convert.setText("Converted — reopen editor to tune")
                convert.setEnabled(False)
            convert.clicked.connect(migrate)
            return w
        w = QWidget()
        w.setObjectName("ModPanel")
        v = QVBoxLayout(w)
        v.addWidget(QLabel(f"AUDIO TRIGGER · {self.param.label.upper()}"))
        en = QCheckBox("ENABLED")
        en.setChecked(a.enabled)
        en.toggled.connect(lambda on: (setattr(a, "enabled", on), self._refresh_btn_styles()))
        v.addWidget(en)
        hint_text = "Above threshold: ramp from REST to TARGET.\nBelow threshold: ramp back. Target is the main slider."
        if self.param.id in _MOTION_RATES:
            hint_text += "\nThese values are speeds. REST 0 holds position; a nonzero REST keeps moving."
        elif self.param.id == "tracking.roll":
            hint_text += "\nThis is a position offset. For movement that holds between hits, use Roll Rate."
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        v.addWidget(hint)
        band = QComboBox()
        for label, key in _ABANDS:
            band.addItem(label, key)
        band.setCurrentIndex(max(0, band.findData(a.band)))
        v.addWidget(band)
        def number(label, value, lo, hi, suffix, update):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(0 if suffix == " Hz" else 2)
            spin.setSingleStep(1.0 if suffix in (" Hz", " %") else 0.01)
            spin.setSuffix(suffix)
            spin.setValue(value)
            spin.valueChanged.connect(update)
            row.addWidget(spin)
            v.addLayout(row)
            return spin
        low = number("LOW CUT", a.low_hz, 0, 19999, " Hz", lambda x: setattr(a, "low_hz", x))
        high = number("HIGH CUT", a.high_hz, 1, 20000, " Hz", lambda x: setattr(a, "high_hz", x))
        low.valueChanged.connect(lambda x: high.setMinimum(x + 1))
        high.valueChanged.connect(lambda x: low.setMaximum(x - 1))
        high.setMinimum(low.value() + 1)
        low.setMaximum(high.value() - 1)
        def select_band():
            a.band = band.currentData()
            ranges = {"low": (20, 250), "mid": (250, 2000), "high": (2000, 16000)}
            low.setEnabled(a.band != "level")
            high.setEnabled(a.band != "level")
            if a.band in ranges:
                lo, hi = ranges[a.band]
                low.setMaximum(19999)
                high.setMinimum(1)
                high.setValue(hi)
                low.setValue(lo)
        band.currentIndexChanged.connect(select_band)
        low.setEnabled(a.band != "level")
        high.setEnabled(a.band != "level")
        number("THRESHOLD", a.threshold * 100, 0, 100, " %", lambda x: setattr(a, "threshold", x / 100))
        number("RAMP", a.ramp, 0, 5, " s", lambda x: setattr(a, "ramp", x))
        number("REST", self.param.clamp(0.0) if a.rest is None else a.rest,
               self.param.minimum, self.param.maximum, "", lambda x: setattr(a, "rest", x))
        number("TARGET", self.param.value, self.param.minimum, self.param.maximum,
               "", lambda x: (self.param.set_base(x), self.refresh()))
        meter = QProgressBar()
        meter.setRange(0, 100)
        v.addWidget(meter)
        status = QLabel()
        v.addWidget(status)
        def update_meter():
            analysis = current()
            host = self.window()
            engine = getattr(host, "engine", None)
            t = engine.current_time() if engine else 0.0
            level = analysis.selected_level(a.band, t, a.low_hz, a.high_hz) if analysis else 0
            meter.setValue(round(level * 100))
            meter.setFormat(f"INPUT {level * 100:.0f}% / THRESHOLD {a.threshold * 100:.0f}%")
            status.setText("No audio loaded" if analysis is None else
                           ("TRIGGERED" if level > 0 and level >= a.threshold else "Below threshold"))
        timer = QTimer(w)
        timer.timeout.connect(update_meter)
        timer.start(50)
        update_meter()
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
        if self.choice_mode:
            self.choice.setCurrentIndex(int(round(self.param.value)))
            self._syncing = False
            return
        if self.bool_mode:
            self.check.setChecked(self.param.value >= 0.5)
            self._syncing = False
            return
        self.slider.setValue(self._to_slider(self.param.value))
        self.value_lbl.setText(f"{self.param.value:.2f}")
        self._refresh_btn_styles()
        self._syncing = False

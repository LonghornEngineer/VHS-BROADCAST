"""Transport controls: load / play / pause / manual keys / preview / scrub."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMenu, QPushButton, QSlider, QToolButton, QVBoxLayout,
    QWidget,
)
from .reset_slider import ResetSlider


class Transport(QWidget):
    loadClicked = Signal()
    playToggled = Signal(bool)
    keyframeClicked = Signal()
    playbackToggled = Signal(bool)
    renderClicked = Signal()
    saveClicked = Signal()
    openSessionClicked = Signal()
    scrubbed = Signal(float)          # 0..1 fraction

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(5)

        self.input_btn = QToolButton()
        self.input_btn.setText("INPUT \u25be")
        self.input_btn.setPopupMode(QToolButton.InstantPopup)
        self.input_menu = QMenu(self.input_btn)
        self.input_btn.setMenu(self.input_menu)

        self.output_btn = QToolButton()
        self.output_btn.setText("OUTPUT \u25be")
        self.output_btn.setPopupMode(QToolButton.InstantPopup)
        self.output_menu = QMenu(self.output_btn)
        self.output_btn.setMenu(self.output_menu)

        self.load_btn = QPushButton("LOAD")
        self.play_btn = QPushButton("\u25b6 PLAY")
        self.play_btn.setCheckable(True)
        self.key_btn = QPushButton("+ KEY")
        self.key_btn.setToolTip("Focus the manual keyframe editor below the preview")
        self.pb_btn = QPushButton("PLAY KEYS")
        self.pb_btn.setToolTip("Preview the saved keyframes; control tweaks turn this off without changing keys")
        self.pb_btn.setCheckable(True)

        self.scrub = ResetSlider(Qt.Horizontal)
        self.scrub.resetRequested.connect(lambda: self.scrubbed.emit(0.0))
        self.scrub.setToolTip("Double-click to return to the beginning")
        self.scrub.setRange(0, 1000)
        self.scrub.setObjectName("Scrub")
        self.scrub.setMinimumHeight(22)
        self.time_lbl = QLabel("00:00 / 00:00")

        self.save_btn = QPushButton("SAVE TAKE")
        self.open_btn = QPushButton("OPEN TAKE")
        self.render_btn = QPushButton("RENDER")
        self.load_btn.clicked.connect(self.loadClicked)
        self.play_btn.toggled.connect(self.playToggled)
        self.key_btn.clicked.connect(self.keyframeClicked)
        self.pb_btn.toggled.connect(self.playbackToggled)
        self.render_btn.clicked.connect(self.renderClicked)
        self.save_btn.clicked.connect(self.saveClicked)
        self.open_btn.clicked.connect(self.openSessionClicked)
        self.scrub.sliderMoved.connect(
            lambda v: self.scrubbed.emit(v / 1000.0))

        # row 1: full-width scrub bar + time
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(self.scrub, 1)
        bar.addWidget(self.time_lbl)
        outer.addLayout(bar)

        # row 2: transport buttons
        btns = QHBoxLayout()
        btns.setSpacing(6)
        for w in (self.input_btn, self.load_btn, self.play_btn,
                  self.key_btn, self.pb_btn):
            btns.addWidget(w)
        btns.addStretch(1)
        for w in (self.save_btn, self.open_btn, self.output_btn,
                  self.render_btn):
            btns.addWidget(w)
        outer.addLayout(btns)

    def set_live(self, label: str):
        """Live capture: no timeline, so disable scrubbing and deterministic
        render (there are no stored frames to re-render)."""
        self.scrub.setEnabled(False)
        self.scrub.setValue(0)
        self.render_btn.setEnabled(False)
        self.key_btn.setEnabled(False)
        self.pb_btn.setEnabled(False)
        self.time_lbl.setText(f"\u25cf LIVE \u00b7 {label}")

    def set_file(self):
        self.scrub.setEnabled(True)
        self.render_btn.setEnabled(True)
        self.key_btn.setEnabled(True)
        self.pb_btn.setEnabled(True)

    def set_position(self, t: float, duration: float):
        if duration > 0:
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(t / duration * 1000))
            self.scrub.blockSignals(False)
        self.time_lbl.setText(f"{self._fmt(t)} / {self._fmt(duration)}")

    @staticmethod
    def _fmt(s: float) -> str:
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

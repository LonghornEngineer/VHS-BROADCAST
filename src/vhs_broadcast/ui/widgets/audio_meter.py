"""A small 4-band level meter (LOW / MID / HIGH / LVL) shown under the video so
you can see what the audio is doing while dialling in audio-reactive controls.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from ... import config as C

_BANDS = [("LOW", "low"), ("MID", "mid"), ("HIGH", "high"), ("LVL", "level")]


class AudioMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setMinimumWidth(160)
        self._lv = {k: 0.0 for _, k in _BANDS}
        self._active = False

    def set_levels(self, levels: dict):
        self._lv = levels or self._lv
        self.update()

    def set_active(self, on: bool):
        self._active = on
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(C.BG))
        n = len(_BANDS)
        gap = 6
        bw = (w - gap * (n + 1)) / n
        label_h = 12
        bar_h = h - label_h - 2
        for i, (name, key) in enumerate(_BANDS):
            x = gap + i * (bw + gap)
            # track
            p.fillRect(int(x), 1, int(bw), int(bar_h), QColor("#0a1a0c"))
            val = max(0.0, min(1.0, self._lv.get(key, 0.0)))
            fh = int(bar_h * val)
            if fh > 0:
                grad = QLinearGradient(0, 1 + bar_h, 0, 1 + bar_h - fh)
                grad.setColorAt(0.0, QColor(C.GREEN_DIM if self._active else "#204020"))
                grad.setColorAt(0.7, QColor(C.GREEN if self._active else "#2c552c"))
                grad.setColorAt(1.0, QColor(C.AMBER if self._active else "#3a6a3a"))
                p.fillRect(int(x), 1 + int(bar_h) - fh, int(bw), fh, grad)
            p.setPen(QColor(C.TEXT_DIM))
            p.drawText(int(x), h - 1, name)
        p.end()

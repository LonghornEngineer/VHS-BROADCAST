"""A custom-painted rotary knob (Qt has no native knob widget)."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ... import config as C

_START = 135.0    # degrees (bottom-left)
_SWEEP = 270.0


class Knob(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum=0.0, maximum=1.0, value=0.0, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._val = value
        self._drag_y = None
        self.setFixedSize(46, 46)
        self.setCursor(Qt.SizeVerCursor)

    # ---- value ------------------------------------------------------------
    def setRange(self, lo, hi):
        self._min, self._max = lo, hi
        self.update()

    def setValue(self, v, emit=True):
        v = max(self._min, min(self._max, v))
        if v == self._val:
            return
        self._val = v
        self.update()
        if emit:
            self.valueChanged.emit(v)

    def value(self):
        return self._val

    def _frac(self):
        span = self._max - self._min
        return 0.0 if span == 0 else (self._val - self._min) / span

    # ---- interaction ------------------------------------------------------
    def mousePressEvent(self, e):
        self._drag_y = e.position().y()

    def mouseMoveEvent(self, e):
        if self._drag_y is None:
            return
        dy = self._drag_y - e.position().y()
        self._drag_y = e.position().y()
        span = self._max - self._min
        self.setValue(self._val + dy / 120.0 * span)

    def mouseReleaseEvent(self, e):
        self._drag_y = None

    def wheelEvent(self, e):
        span = self._max - self._min
        self.setValue(self._val + (e.angleDelta().y() / 120.0) * span * 0.03)

    # ---- paint ------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(4, 4, self.width() - 8, self.height() - 8)

        # track
        p.setPen(QPen(QColor(C.BORDER), 3))
        p.drawArc(rect, int(-_START * 16), int(-_SWEEP * 16))
        # value arc
        p.setPen(QPen(QColor(C.GREEN), 3))
        p.drawArc(rect, int(-_START * 16), int(-_SWEEP * self._frac() * 16))

        # indicator line
        ang = math.radians(_START + _SWEEP * self._frac())
        cx, cy = rect.center().x(), rect.center().y()
        r = rect.width() / 2
        x2 = cx + r * math.cos(-ang)
        y2 = cy + r * math.sin(-ang)
        p.setPen(QPen(QColor(C.AMBER), 2))
        p.drawLine(QPointF(cx, cy), QPointF(x2, y2))
        p.end()

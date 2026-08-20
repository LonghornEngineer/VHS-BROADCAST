"""Displays processed frames, letterboxed, with a blinking REC indicator."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from ... import config as C


class VideoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self.recording = False
        self.preview_off = False
        self._blink = False
        self.setMinimumSize(480, 320)
        self.setAutoFillBackground(True)
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(500)

    def _tick(self):
        self._blink = not self._blink
        if self.recording:
            self.update()

    def set_image(self, img: QImage):
        self._pix = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        if self.preview_off:
            p.setPen(QColor(C.GREEN))
            f = p.font(); f.setBold(True); f.setPointSize(13); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "PREVIEW OFF\n(output still running)")
            p.end()
            return
        if self._pix is not None:
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        # REC light
        if self.recording and self._blink:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C.RED))
            p.drawEllipse(16, 14, 12, 12)
            p.setPen(QColor(C.RED))
            f = p.font(); f.setBold(True); p.setFont(f)
            p.drawText(34, 25, "REC")
        p.end()

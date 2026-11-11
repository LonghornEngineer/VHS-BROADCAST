"""Displays processed frames, letterboxed, with a blinking REC indicator and a compare split."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from ... import config as C


class VideoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self._source: QPixmap | None = None
        self.compare = False
        self.split = .5               # divider position, fraction of the picture width
        self.recording = False
        self.preview_off = False
        self._blink = False
        self.setMinimumSize(400, 180)
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

    def set_source(self, img: QImage):
        self._source = QPixmap.fromImage(img)

    def set_compare(self, on: bool):
        self.compare = on
        if not on:
            self._source = None
        self.setCursor(Qt.SplitHCursor if on else Qt.ArrowCursor)
        self.update()

    def _picture_rect(self):
        if self._pix is None:
            return None
        size = self._pix.size().scaled(self.size(), Qt.KeepAspectRatio)
        return ((self.width() - size.width()) // 2, (self.height() - size.height()) // 2,
                size.width(), size.height())

    def mousePressEvent(self, e):
        self._drag_split(e)

    def mouseMoveEvent(self, e):
        self._drag_split(e)

    def _drag_split(self, e):
        rect = self._picture_rect()
        if self.compare and rect and e.buttons() & Qt.LeftButton:
            x, _, w, _ = rect
            self.split = min(1.0, max(0.0, (e.position().x() - x) / max(1, w)))
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
            if self.compare and self._source is not None:
                # Effect left of the divider, untouched source right of it.
                source = self._source.scaled(scaled.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                cut = int(scaled.width() * self.split)
                p.drawPixmap(x + cut, y, source, cut, 0, scaled.width() - cut, scaled.height())
                p.setPen(QColor(C.GREEN))
                p.drawLine(x + cut, y, x + cut, y + scaled.height())
                f = p.font(); f.setBold(True); p.setFont(f)
                p.drawText(x + cut - 34, y + 18, "FX")
                p.drawText(x + cut + 8, y + 18, "SOURCE")
        # REC light
        if self.recording and self._blink:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C.RED))
            p.drawEllipse(16, 14, 12, 12)
            p.setPen(QColor(C.RED))
            f = p.font(); f.setBold(True); p.setFont(f)
            p.drawText(34, 25, "REC")
        p.end()

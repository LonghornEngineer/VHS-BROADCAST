"""A slider whose double-click is reserved for a factory-value reset."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSlider


class ResetSlider(QSlider):
    resetRequested = Signal()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.resetRequested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

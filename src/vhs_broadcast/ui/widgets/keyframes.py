"""Manual keyframe editor: all-track time chart and precise selected-key edits."""
from __future__ import annotations

import math
from dataclasses import replace
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from ... import config as C
from ...core.automation import Timeline
from .reset_slider import ResetSlider


class KeyframeChart(QWidget):
    selected = Signal(str, float)
    moved = Signal(str, float, float, float)
    seekRequested = Signal(float)
    LEFT, TOP, ROW = 190, 32, 62

    def __init__(self, params, timeline, labels, parent=None):
        super().__init__(parent)
        self.params, self.timeline, self.labels = params, timeline, labels
        self.scale = 50.0
        self.duration = 10.0
        self.playhead = 0.0
        self.selection = None
        self.drag = None
        self.fps = 30.0
        self.snap = True
        self.lanes = []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.refresh()

    def refresh(self):
        self.lanes = [pid for pid, track in self.timeline.tracks.items()
                      if pid in self.params and track.times]
        self.setMinimumSize(int(self.LEFT + max(10, self.duration) * self.scale + 30),
                            self.TOP + max(1, len(self.lanes)) * self.ROW)
        self.update()

    def point(self, pid, t, value):
        lane = self.lanes.index(pid)
        param = self.params.get(pid)
        frac = (value - param.minimum) / (param.maximum - param.minimum or 1)
        return QPointF(self.LEFT + t * self.scale,
                       self.TOP + lane * self.ROW + 48 - frac * 34)

    def time_at(self, x):
        t = max(0, min(self.duration, (x - self.LEFT) / self.scale))
        return round(t * self.fps) / self.fps if self.snap else t

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(C.BG))
        painter.setRenderHint(QPainter.Antialiasing)
        # Adaptive ruler spacing and clipping keep long timelines inexpensive.
        step = 10 ** math.floor(math.log10(max(.01, 80 / self.scale)))
        while step * self.scale < 65:
            step *= 2
        first = max(0, int((event.rect().left() - self.LEFT) / self.scale / step))
        last = int((event.rect().right() - self.LEFT) / self.scale / step) + 1
        for tick in range(first, max(first, last) + 1):
            t = tick * step
            x = self.LEFT + t * self.scale
            painter.setPen(QColor(C.BORDER))
            painter.drawLine(QPointF(x, self.TOP), QPointF(x, self.height()))
            painter.setPen(QColor(C.TEXT_DIM))
            painter.drawText(QPointF(x + 3, 20), f"{t:.2f}s")
        for lane, pid in enumerate(self.lanes):
            y = self.TOP + lane * self.ROW
            if y + self.ROW < event.rect().top() or y > event.rect().bottom():
                continue
            painter.fillRect(0, y, self.LEFT - 4, self.ROW, QColor(C.PANEL))
            painter.setPen(QColor(C.TEXT))
            label = painter.fontMetrics().elidedText(self.labels[pid], Qt.ElideRight, self.LEFT - 16)
            painter.drawText(8, y + 24, label)
            track = self.timeline.tracks[pid]
            painter.setPen(QColor(C.TEXT_DIM))
            painter.drawText(8, y + 43, track.interpolation.upper())
            painter.setPen(QColor(C.BORDER))
            painter.drawLine(0, y + self.ROW, self.width(), y + self.ROW)
            points = []
            for i, (t, value) in enumerate(zip(track.times, track.values)):
                end = self.point(pid, t, value)
                if i:
                    if track.interpolation == "hold":
                        points.append(self.point(pid, t, track.values[i - 1]))
                    elif track.interpolation == "smooth":
                        start = track.times[i - 1]
                        for j in range(1, 16):
                            sample = start + (t - start) * j / 16
                            points.append(self.point(pid, sample, track.value_at(sample)))
                points.append(end)
            painter.setPen(QPen(QColor(C.GREEN_DIM), 1.5))
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            for t, value in zip(track.times, track.values):
                pt = self.point(pid, t, value)
                color = C.AMBER if self.selection == (pid, t) else C.GREEN
                painter.setBrush(QColor(color))
                painter.setPen(QColor(color))
                painter.drawPolygon(QPolygonF([pt + QPointF(-5, 0), pt + QPointF(0, -6),
                                               pt + QPointF(5, 0), pt + QPointF(0, 6)]))
        if not self.lanes:
            painter.setPen(QColor(C.TEXT_DIM))
            painter.drawText(12, self.TOP + 30, "No keyframes. Choose a control and press ADD KEY.")
        if self.drag:
            pid, old, t, value = self.drag
            pt = self.point(pid, t, value)
            painter.setPen(QPen(QColor(C.AMBER), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(pt, 8, 8)
        x = self.LEFT + self.playhead * self.scale
        painter.setPen(QPen(QColor(C.AMBER), 1))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    def hit(self, position):
        for pid in self.lanes:
            track = self.timeline.tracks[pid]
            for t, value in zip(track.times, track.values):
                pt = self.point(pid, t, value)
                if abs(pt.x() - position.x()) < 9 and abs(pt.y() - position.y()) < 10:
                    return pid, t, value
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        hit = self.hit(event.position())
        if hit:
            pid, t, value = hit
            self.selection = (pid, t)
            self.drag = (pid, t, t, value)
            self._press = event.position()
            self._dragging = False
            self.selected.emit(pid, t)
        elif event.position().x() >= self.LEFT:
            self.seekRequested.emit(self.time_at(event.position().x()))
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag and event.buttons() & Qt.LeftButton:
            if (event.position() - self._press).manhattanLength() < 4 and not self._dragging:
                return
            self._dragging = True
            pid, old, _, _ = self.drag
            param = self.params.get(pid)
            top = self.TOP + self.lanes.index(pid) * self.ROW
            frac = (top + 48 - event.position().y()) / 34
            value = param.clamp(param.minimum + frac * (param.maximum - param.minimum))
            if param.step == 1:
                value = round(value)
            self.drag = pid, old, self.time_at(event.position().x()), value
            self.update()
        else:
            hit = self.hit(event.position())
            self.setToolTip(f"{self.labels[hit[0]]}: {hit[1]:.3f}s / {hit[2]:.3f}\nDrag to change time and value." if hit else "Click empty time to seek. Keys are only added with ADD KEY.")

    def mouseReleaseEvent(self, event):
        if self.drag:
            move = self.drag
            self.drag = None
            if self._dragging:
                self.moved.emit(*move)
            self.update()


class KeyframeEditor(QWidget):
    changed = Signal()
    seekRequested = Signal(float)

    def __init__(self, params, timeline, effects, time_provider, parent=None):
        super().__init__(parent)
        self.params, self.timeline = params, timeline
        self.setObjectName("KeyframeEditor")
        self.time_provider = time_provider
        self.selection = None
        self.undo_stack, self.redo_stack = [], []
        effect_names = {e.id: e.label for e in effects}
        self.labels = {p.id: f"{effect_names.get(p.id.split('.')[0], '')} / {p.label}"
                       for p in params.all()}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        hint = QLabel("KEYFRAMES · Manual keys only. Drag diamonds to edit; click empty time to preview.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.control = QComboBox()
        for p in params.all():
            self.control.addItem(self.labels[p.id], p.id)
        self.control.setMinimumWidth(210)
        row.addWidget(self.control, 1)
        self.add_btn = QPushButton("ADD KEY AT PLAYHEAD")
        self.add_btn.setToolTip("Save this control's current slider/checkbox value at the playhead. An existing key at that time is replaced.")
        self.add_btn.clicked.connect(self.add_key)
        row.addWidget(self.add_btn)
        layout.addLayout(row)
        self.chart = KeyframeChart(params, timeline, self.labels)
        self.chart.selected.connect(self.select_key)
        self.chart.moved.connect(self.move_key)
        self.chart.seekRequested.connect(self.seekRequested)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.chart)
        layout.addWidget(self.scroll, 1)
        tools = QHBoxLayout()
        self.time_edit = QDoubleSpinBox()
        self.time_edit.setRange(0, 86400)
        self.time_edit.setDecimals(3)
        self.time_edit.setSuffix(" s")
        self.value_edit = QDoubleSpinBox()
        self.value_edit.setDecimals(3)
        tools.addWidget(QLabel("TIME")); tools.addWidget(self.time_edit)
        tools.addWidget(QLabel("VALUE")); tools.addWidget(self.value_edit)
        self.apply_btn = QPushButton("APPLY")
        self.apply_btn.clicked.connect(self.apply_key)
        self.delete_btn = QPushButton("DELETE KEY")
        self.delete_btn.clicked.connect(self.delete_key)
        tools.addWidget(self.apply_btn); tools.addWidget(self.delete_btn)
        layout.addLayout(tools)
        row = QHBoxLayout()
        row.addWidget(QLabel("TRANSITION"))
        self.interpolation = QComboBox()
        self.interpolation.addItems(["HOLD", "LINEAR", "SMOOTH"])
        self.interpolation.setToolTip("Interpolation for the selected control's entire track. On/off controls always hold.")
        self.interpolation.currentTextChanged.connect(self.set_interpolation)
        row.addWidget(self.interpolation)
        self.snap = QCheckBox("Snap to frames")
        self.snap.setChecked(True)
        self.snap.toggled.connect(lambda on: setattr(self.chart, "snap", on))
        row.addWidget(self.snap)
        for label, delta in (("PREV", -1), ("NEXT", 1)):
            button = QPushButton(label)
            button.clicked.connect(lambda _, d=delta: self.jump_key(d))
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_btn, self.redo_btn = QPushButton("UNDO"), QPushButton("REDO")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        row.addWidget(self.undo_btn); row.addWidget(self.redo_btn)
        row.addWidget(QLabel("ZOOM"))
        zoom = ResetSlider(Qt.Horizontal)
        zoom.setRange(5, 250); zoom.setValue(50)
        zoom.resetRequested.connect(lambda: zoom.setValue(50))
        zoom.valueChanged.connect(self.set_zoom)
        row.addWidget(zoom, 1)
        fit = QPushButton("FIT")
        fit.clicked.connect(lambda: self.set_zoom(max(.1, (self.scroll.viewport().width() - self.chart.LEFT - 30) / self.chart.duration)))
        row.addWidget(fit)
        layout.addLayout(row)
        self.control.currentIndexChanged.connect(self.control_changed)
        self.control_changed()
        self.refresh()

    def checkpoint(self):
        self.undo_stack.append(self.timeline.as_dict())
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

    def refresh(self, clear_history=False):
        if clear_history:
            self.undo_stack.clear(); self.redo_stack.clear(); self.selection = None
        if self.selection:
            pid, t = self.selection
            if pid not in self.timeline.tracks or t not in self.timeline.tracks[pid].times:
                self.selection = None
        self.chart.duration = max(self.chart.duration, self.timeline.duration(), 10)
        self.chart.selection = self.selection
        self.chart.refresh()
        for widget in (self.time_edit, self.value_edit, self.apply_btn, self.delete_btn):
            widget.setEnabled(self.selection is not None)
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self.redo_btn.setEnabled(bool(self.redo_stack))
        self.update_interpolation()

    def update_interpolation(self):
        pid = self.control.currentData()
        p = self.params.get(pid)
        track = self.timeline.tracks.get(pid)
        boolean = p.choices or (p.step == 1 and p.minimum == 0 and p.maximum == 1)
        self.interpolation.blockSignals(True)
        self.interpolation.setCurrentText(track.interpolation.upper() if track else "HOLD" if boolean else "LINEAR")
        self.interpolation.blockSignals(False)
        self.interpolation.setEnabled(bool(track) and not boolean)

    def control_changed(self):
        self.selection = None
        p = self.params.get(self.control.currentData())
        self.value_edit.setRange(p.minimum, p.maximum)
        self.value_edit.setSingleStep(p.step)
        self.value_edit.setToolTip("; ".join(f"{i}: {name}" for i, name in enumerate(p.choices)))
        self.value_edit.setValue(p.value)
        self.refresh()

    def add_key(self):
        self.checkpoint()
        param = self.params.get(self.control.currentData())
        t = max(0, self.time_provider())
        if self.chart.snap:
            t = round(t * self.chart.fps) / self.chart.fps
        self.timeline.set_key(t, param)
        self.select_key(param.id, t)
        self.changed.emit()

    def select_key(self, pid, t):
        self.control.setCurrentIndex(self.control.findData(pid))
        self.selection = (pid, t)
        self.time_edit.setValue(t)
        track = self.timeline.tracks[pid]
        self.value_edit.setValue(track.values[track.times.index(t)])
        self.refresh()

    def move_key(self, pid, old, new, value):
        self.checkpoint()
        track = self.timeline.tracks[pid]
        track = replace(track, times=list(track.times), values=list(track.values))
        param = self.params.get(pid)
        value = round(value) if param.step == 1 else value
        track.remove(old)
        track.add(max(0, new), param.clamp(value))
        self.timeline.tracks = {**self.timeline.tracks, pid: track}
        self.select_key(pid, max(0, new))
        self.changed.emit()

    def apply_key(self):
        if self.selection:
            t = self.time_edit.value()
            if self.chart.snap:
                t = round(t * self.chart.fps) / self.chart.fps
            self.move_key(*self.selection, t, self.value_edit.value())

    def delete_key(self):
        if self.selection:
            self.checkpoint()
            self.timeline.delete_key(*self.selection)
            self.selection = None
            self.refresh()
            self.changed.emit()

    def set_interpolation(self, text):
        track = self.timeline.tracks.get(self.control.currentData())
        if track and track.interpolation != text.lower():
            self.checkpoint()
            self.timeline.tracks = {**self.timeline.tracks, self.control.currentData():
                                    replace(track, interpolation=text.lower())}
            self.refresh()
            self.changed.emit()

    def jump_key(self, direction):
        times = sorted({t for track in self.timeline.tracks.values() for t in track.times})
        now = self.time_provider()
        candidates = [t for t in times if (t - now) * direction > .0001]
        if candidates:
            self.seekRequested.emit(min(candidates) if direction > 0 else max(candidates))

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.timeline.as_dict())
            self.timeline.tracks = Timeline.from_dict(self.undo_stack.pop()).tracks
            self.selection = None
            self.refresh(); self.changed.emit()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.timeline.as_dict())
            self.timeline.tracks = Timeline.from_dict(self.redo_stack.pop()).tracks
            self.selection = None
            self.refresh(); self.changed.emit()

    def set_zoom(self, value):
        self.chart.scale = value
        self.chart.refresh()

    def set_playhead(self, t, duration, fps):
        self.chart.playhead = t
        self.chart.fps = max(1, fps)
        span = max(duration, self.timeline.duration(), 10)
        if span != self.chart.duration:
            self.chart.duration = span
            self.chart.refresh()
        self.chart.update()

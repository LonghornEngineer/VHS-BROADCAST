"""A drag-reorderable list of the effect chain.

Shows every effect in processing order (OSD/TEXT included as a movable item),
with a checkbox for enabled state. Dragging reorders the chain; where OSD/TEXT
sits decides whether the text is degraded by the effects after it. Reordering
applies to both preview and offline render.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

from ...effects.stages import STAGE_LABEL


class ChainList(QListWidget):
    def __init__(self, effects, on_reorder, on_toggle, parent=None, hidden=None):
        super().__init__(parent)
        self._hidden = hidden or (lambda effect: False)
        self._on_reorder = on_reorder
        self._on_toggle = on_toggle
        self.setObjectName("ChainList")
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDefaultDropAction(Qt.MoveAction)
        self.itemChanged.connect(self._toggled)
        self.model().rowsMoved.connect(lambda *_: self._reordered())
        self.rebuild(effects)

    def rebuild(self, effects):
        self.blockSignals(True)
        self.clear()
        for e in effects:
            it = QListWidgetItem(f"{e.label}   \u00b7 {STAGE_LABEL[e.stage]}")
            it.setData(Qt.UserRole, e.id)
            it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled
                        | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            on = e.enabled_param().value >= 0.5
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.addItem(it)
            it.setHidden(self._hidden(e))
        self._renumber()
        self.blockSignals(False)

    def _renumber(self):
        """Number only the rows on show, so hidden retired effects leave no gaps."""
        n = 0
        for i in range(self.count()):
            it = self.item(i)
            label = it.text().split(".  ", 1)[-1] if ".  " in it.text()[:6] else it.text()
            if it.isHidden():
                it.setText(label)
                continue
            n += 1
            it.setText(f"{n:2d}.  {label}")

    def _order_ids(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def _toggled(self, item):
        self._on_toggle(item.data(Qt.UserRole),
                        item.checkState() == Qt.Checked)

    def _reordered(self):
        self._on_reorder(self._order_ids())
        self.blockSignals(True)
        self._renumber()
        self.blockSignals(False)

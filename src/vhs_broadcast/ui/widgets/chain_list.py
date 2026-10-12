"""A drag-reorderable list of the effect chain.

Shows every effect in processing order (OSD/TEXT included as a movable item),
with a checkbox for enabled state. Dragging reorders the chain; where OSD/TEXT
sits decides whether the text is degraded by the effects after it. Reordering
applies to both preview and offline render.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem


class ChainList(QListWidget):
    def __init__(self, effects, on_reorder, on_toggle, parent=None):
        super().__init__(parent)
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
        for i, e in enumerate(effects):
            it = QListWidgetItem(f"{i + 1:2d}.  {e.label}")
            it.setData(Qt.UserRole, e.id)
            it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled
                        | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            on = e.enabled_param().value >= 0.5
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.addItem(it)
        self.blockSignals(False)

    def _order_ids(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def _toggled(self, item):
        self._on_toggle(item.data(Qt.UserRole),
                        item.checkState() == Qt.Checked)

    def _reordered(self):
        self._on_reorder(self._order_ids())
        # renumber the labels after a move
        for i in range(self.count()):
            it = self.item(i)
            label = it.text().split(".", 1)[-1].strip()
            it.setText(f"{i + 1:2d}.  {label}")

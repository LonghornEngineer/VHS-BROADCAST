"""Modular mode: the effect chain as a rack of modules with dials.

The rack shows the enabled effects in chain order, left to right, joined by signal
arrows. Each module has the effect's mode selectors, toggles and a dial per control;
◀ ▶ move it along the chain and ✕ takes it out. The library lists every effect by
signal stage; clicking one patches it in or out. Everything edits the same
parameters as Classic mode, so the two views always agree.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from ... import config as C
from ...core.parameters import Parameter
from ...effects.stages import STAGES, param_in_use
from .flow_layout import FlowLayout
from .knob import Knob

STAGE_TAG = {"camera": "CAM", "digital": "DIG", "tape": "TAPE", "receiver": "RX",
             "screen": "SCR", "stylize": "STY", "composite": "CMP"}


def _is_bool(p: Parameter) -> bool:
    return p.minimum == 0.0 and p.maximum == 1.0 and p.step == 1.0 and not p.choices


def _fmt(p: Parameter, v: float) -> str:
    if p.step >= 1:
        return f"{v:.0f}"
    span = abs(p.maximum - p.minimum)
    return f"{v:.1f}" if span >= 20 else f"{v:.2f}"


class Dial(QWidget):
    """A knob with its label and value, bound to one parameter."""
    keyframeRequested = Signal(str)
    classicRequested = Signal(str)

    def __init__(self, param: Parameter, parent=None):
        super().__init__(parent)
        self.param = param
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.knob = Knob(param.minimum, param.maximum, param.value)
        self.knob.setToolTip(f"{param.label}: drag up/down or scroll. Double-click resets to "
                             f"{param.default:g}. Right-click for keyframe and modulation.")
        self.knob.valueChanged.connect(self._turned)
        self.knob.mouseDoubleClickEvent = lambda _e: self._reset()
        lay.addWidget(self.knob, 0, Qt.AlignHCenter)
        # The font is monospaced, so a character budget keeps every name inside its column.
        label = param.label.lower()
        self.name = QLabel(label if len(label) <= 12 else label[:11] + "…")
        self.name.setToolTip(param.label)
        self.name.setObjectName("DialName")
        self.name.setAlignment(Qt.AlignHCenter)
        self.name.setFixedWidth(84)
        lay.addWidget(self.name)
        self.value = QLabel()
        self.value.setObjectName("DialValue")
        self.value.setAlignment(Qt.AlignHCenter)
        lay.addWidget(self.value)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.refresh()

    def _turned(self, v):
        if self._syncing:
            return
        if self.param.step >= 1:
            v = round(v)
        self.param.set_base(v)
        self._show(self.param.value)

    def _reset(self):
        self.param.reset()
        self.refresh()

    def _menu(self, pos):
        menu = QMenu(self)
        menu.addAction("◆ Add keyframe at playhead", lambda: self.keyframeRequested.emit(self.param.id))
        menu.addAction("Reset to default", self._reset)
        menu.addAction("LFO / audio modulation… (Classic)", lambda: self.classicRequested.emit(self.param.id))
        menu.exec(self.mapToGlobal(pos))

    def _show(self, v):
        moving = self.param.mod.enabled or self.param.audio.enabled
        self.value.setText(_fmt(self.param, v) + (" ~" if moving else ""))

    def refresh(self):
        self._syncing = True
        self.knob.setValue(self.param.value, emit=False)
        self._syncing = False
        self._show(self.param.value)


class ModuleCard(QFrame):
    """One effect as a rack module."""
    moveRequested = Signal(str, int)
    removeRequested = Signal(str)

    def __init__(self, effect, number: int, timeline=None, parent=None):
        super().__init__(parent)
        self.effect = effect
        self.setObjectName("Module")
        self.dials: list[Dial] = []
        self._choices: list[tuple[Parameter, QComboBox]] = []
        self._toggles: list[tuple[Parameter, QCheckBox]] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(5)

        top = QHBoxLayout()
        top.setSpacing(2)
        self.tag = QLabel(f"{number:02d} [{STAGE_TAG.get(effect.stage, effect.stage.upper())}]")
        self.tag.setObjectName("ModuleTag")
        top.addWidget(self.tag, 1)
        for text, step, tip in (("◀", -1, "Move earlier in the chain"), ("▶", 1, "Move later in the chain")):
            button = QPushButton(text)
            button.setObjectName("ModuleBtn")
            button.setToolTip(tip)
            button.setFixedSize(24, 22)
            button.clicked.connect(lambda _=False, s=step: self.moveRequested.emit(effect.id, s))
            top.addWidget(button)
        out = QPushButton("✕")
        out.setObjectName("ModuleBtn")
        out.setToolTip("Take this module out of the chain (turns the effect off)")
        out.setFixedSize(24, 22)
        out.clicked.connect(lambda: self.removeRequested.emit(effect.id))
        top.addWidget(out)
        lay.addLayout(top)

        title = QLabel(effect.label)
        title.setObjectName("ModuleTitle")
        title.setWordWrap(True)
        lay.addWidget(title)

        params = [p for p in effect.own_params()
                  if not p.id.endswith(".reset") and (not p.legacy or param_in_use(p, timeline))]
        for p in params:
            if p.choices:
                box = QComboBox()
                box.addItems(p.choices)
                box.setToolTip(p.label)
                box.currentIndexChanged.connect(lambda i, par=p: None if self._syncing else par.set_base(i))
                row = QHBoxLayout()
                name = QLabel(p.label.lower())
                name.setObjectName("DialName")
                row.addWidget(name)
                row.addWidget(box, 1)
                lay.addLayout(row)
                self._choices.append((p, box))
            elif _is_bool(p):
                check = QCheckBox(p.label.lower())
                check.toggled.connect(lambda on, par=p: None if self._syncing else par.set_base(1.0 if on else 0.0))
                lay.addWidget(check)
                self._toggles.append((p, check))
        knobs = [p for p in params if not p.choices and not _is_bool(p)]
        columns = 2 if len(knobs) <= 6 else 3 if len(knobs) <= 12 else 4
        grid = QGridLayout()
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(4)
        for i, p in enumerate(knobs):
            dial = Dial(p)
            self.dials.append(dial)
            grid.addWidget(dial, i // columns, i % columns, Qt.AlignHCenter)
        lay.addLayout(grid)
        lay.addStretch(1)
        ports = QLabel("in ◉" + " " * 10 + "◉ out")
        ports.setObjectName("ModuleTag")
        ports.setAlignment(Qt.AlignHCenter)
        lay.addWidget(ports)
        self.setFixedWidth(26 + 88 * columns)
        self._syncing = False
        self.refresh()

    def refresh(self):
        self._syncing = True
        for p, box in self._choices:
            box.setCurrentIndex(int(round(p.value)))
        for p, check in self._toggles:
            check.setChecked(p.value >= .5)
        self._syncing = False
        for dial in self.dials:
            dial.refresh()


class _Strip(QWidget):
    """Holds the cards and draws the signal arrows in the gaps between them."""
    GAP = 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(10, 10, 10, 10)
        self.row.setSpacing(self.GAP)
        self.row.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def paintEvent(self, _):
        cards = [self.row.itemAt(i).widget() for i in range(self.row.count())]
        cards = [c for c in cards if isinstance(c, ModuleCard)]
        if len(cards) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(C.GREEN), 2, Qt.DashLine)
        p.setPen(pen)
        y = cards[0].y() + 70
        for a, b in zip(cards, cards[1:]):
            x1, x2 = a.geometry().right() + 3, b.x() - 3
            p.drawLine(QPointF(x1, y), QPointF(x2, y))
            p.setPen(QPen(QColor(C.GREEN), 2))
            p.drawLine(QPointF(x2 - 6, y - 5), QPointF(x2, y))
            p.drawLine(QPointF(x2 - 6, y + 5), QPointF(x2, y))
            p.setPen(pen)
        p.end()


class Rack(QScrollArea):
    """The chain as modules, left to right."""
    keyframeRequested = Signal(str)
    classicRequested = Signal(str)

    def __init__(self, get_effects: Callable, on_reorder: Callable, on_toggle: Callable,
                 timeline=None, hidden: Callable = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Rack")
        self._get_effects = get_effects
        self._on_reorder = on_reorder
        self._on_toggle = on_toggle
        self._timeline = timeline
        self._hidden = hidden or (lambda effect: False)
        self.cards: list[ModuleCard] = []
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.strip = _Strip()
        self.setWidget(self.strip)
        self.empty = QLabel("$ chain is empty — pick modules from the library")
        self.empty.setObjectName("SectionHint")
        self.rebuild()

    def modules(self):
        from ...effects.vhs import TextOverlayEffect
        return [e for e in self._get_effects()
                if e.enabled_param().value >= .5 and not isinstance(e, TextOverlayEffect)
                and not self._hidden(e)]

    def rebuild(self):
        while self.strip.row.count():
            item = self.strip.row.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.cards = []
        mods = self.modules()
        if not mods:
            self.strip.row.addWidget(self.empty)
            self.empty.show()
        for n, effect in enumerate(mods, 1):
            card = ModuleCard(effect, n, self._timeline)
            card.moveRequested.connect(self._move)
            card.removeRequested.connect(lambda eid: self._on_toggle(eid, False))
            for dial in card.dials:
                dial.keyframeRequested.connect(self.keyframeRequested)
                dial.classicRequested.connect(self.classicRequested)
            self.strip.row.addWidget(card)
            self.cards.append(card)
        self.strip.update()

    def _move(self, eid, step):
        """Swap with the neighbouring module, leaving the hidden effects where they are."""
        order = [e.id for e in self._get_effects()]
        shown = [e.id for e in self.modules()]
        i = shown.index(eid)
        j = i + step
        if not 0 <= j < len(shown):
            return
        a, b = order.index(eid), order.index(shown[j])
        order[a], order[b] = order[b], order[a]
        self._on_reorder(order)

    def refresh(self):
        for card in self.cards:
            card.refresh()


class ModuleLibrary(QScrollArea):
    """Every effect, grouped by signal stage. Click to patch it in or out."""
    toggled = Signal(str, bool)

    def __init__(self, get_effects: Callable, hidden: Callable = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Library")
        self._get_effects = get_effects
        self._hidden = hidden or (lambda effect: False)
        self.setWidgetResizable(True)
        holder = QWidget()
        self.box = QVBoxLayout(holder)
        self.box.setSpacing(6)
        self.setWidget(holder)
        self.buttons: dict[str, QPushButton] = {}
        head = QLabel("$ ls modules/      * = in the rack")
        head.setObjectName("SectionHint")
        self.box.addWidget(head)
        from ...effects.vhs import TextOverlayEffect
        for key, title, blurb in STAGES:
            members = [e for e in get_effects() if e.stage == key
                       and not isinstance(e, TextOverlayEffect)]
            if not members:
                continue
            label = QLabel(f"{key}/  — {blurb}")
            label.setObjectName("StageHeader")
            label.setWordWrap(True)
            self.box.addWidget(label)
            flow_holder = QWidget()
            flow = FlowLayout(flow_holder, spacing=4)
            for e in sorted(members, key=lambda e: e.label):
                button = QPushButton()
                button.setObjectName("LibraryItem")
                button.setCheckable(True)
                button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
                button.toggled.connect(lambda on, eid=e.id: self.toggled.emit(eid, on))
                flow.addWidget(button)
                self.buttons[e.id] = button
            self.box.addWidget(flow_holder)
        self.box.addStretch(1)
        self.refresh()

    def refresh(self):
        for e in self._get_effects():
            button = self.buttons.get(e.id)
            if button is None:
                continue
            on = e.enabled_param().value >= .5
            button.blockSignals(True)
            button.setChecked(on)
            button.setText(("* " if on else "+ ") + e.label.lower())
            button.setVisible(on or not self._hidden(e))
            button.blockSignals(False)

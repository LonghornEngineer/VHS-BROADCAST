"""Undo/redo for control changes, grouped into gestures.

A slider drag fires dozens of changes; they become one undo step. A gesture starts
with the first change after a pause and ends once changes stop for `gap` seconds.
The history stores whole snapshots (cheap: a take is a few kilobytes of numbers),
so undo restores exactly what was there, modulation and keys included.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class ControlHistory:
    def __init__(self, capture: Callable[[], Any], restore: Callable[[Any], None],
                 *, gap: float = .6, limit: int = 100, clock: Callable[[], float] = time.monotonic):
        self._capture, self._restore, self._clock = capture, restore, clock
        self.gap, self.limit = gap, limit
        self.undo_stack: list = []
        self.redo_stack: list = []
        self._stable = capture()     # state at the end of the last gesture
        self._open = False
        self._last = 0.0

    def note_change(self) -> None:
        """Call after any user change. The pre-gesture state becomes one undo step."""
        now = self._clock()
        if self._open and now - self._last >= self.gap:
            self.settle(force=True)
        if not self._open:
            self.undo_stack.append(self._stable)
            del self.undo_stack[:-self.limit]
            self.redo_stack.clear()
            self._open = True
        self._last = now

    def settle(self, force: bool = False) -> None:
        """Close the current gesture once it has been quiet for `gap` seconds."""
        if self._open and (force or self._clock() - self._last >= self.gap):
            self._stable = self._capture()
            self._open = False

    def reset(self) -> None:
        """Forget history, e.g. after opening a take."""
        self.undo_stack.clear(); self.redo_stack.clear()
        self._stable = self._capture(); self._open = False

    def undo(self) -> bool:
        self.settle(force=True)
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._capture())
        self._stable = self.undo_stack.pop()
        self._restore(self._stable)
        return True

    def redo(self) -> bool:
        self.settle(force=True)
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._capture())
        self._stable = self.redo_stack.pop()
        self._restore(self._stable)
        return True

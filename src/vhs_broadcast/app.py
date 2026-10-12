"""Application entry point."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from . import config as C
from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(C.APP_NAME)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

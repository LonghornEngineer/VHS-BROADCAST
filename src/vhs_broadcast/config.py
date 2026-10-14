"""App-wide constants and the retro-broadcast palette."""
from __future__ import annotations

APP_NAME = "VHS BROADCAST DECK"
APP_VERSION = "0.2.0"

# Preview is capped to this width for real-time performance. Final render can
# use full resolution (see render CLI / File > Render).
PREVIEW_MAX_WIDTH = 720

# --- Phosphor palette -------------------------------------------------------
BG = "#080b08"
PANEL = "#0e140e"
PANEL_HI = "#12210f"
GREEN = "#33ff66"       # primary phosphor
GREEN_DIM = "#1f8a3c"
AMBER = "#ffb000"       # secondary / values
CYAN = "#33ffff"
MAGENTA = "#ff3399"
RED = "#ff2b2b"         # REC
TEXT = "#c8ffcf"
TEXT_DIM = "#5f7a63"
BORDER = "#245c2c"

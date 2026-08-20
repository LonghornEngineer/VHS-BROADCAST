#!/usr/bin/env bash
# Launch the VHS Broadcast Deck GUI (Linux / macOS)
set -e
command -v uv >/dev/null || { echo "uv not found. Install: https://astral.sh/uv"; exit 1; }
echo "Syncing dependencies..."
uv sync
# Offer the optional VCR OSD Mono font (prompts once if missing)
uv run --no-sync python scripts/fetch_vcr_font.py --if-missing || true
uv run --no-sync vhs-broadcast

#!/usr/bin/env bash
# Launch the VHS Broadcast Deck GUI (Linux / macOS)
set -e
command -v uv >/dev/null || { echo "uv not found. Install: https://astral.sh/uv"; exit 1; }
echo "Syncing dependencies..."
cd -- "$(dirname -- "$0")"
uv sync --locked
uv run --no-sync vhs-broadcast

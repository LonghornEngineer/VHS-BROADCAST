#!/usr/bin/env bash
# Headless re-render. Usage: ./render.sh take.json out.mp4 [--full]
set -e
[ -z "$2" ] && { echo "Usage: ./render.sh take.json out.mp4 [--full]"; exit 1; }
uv run vhs-render --session "$1" --output "$2" ${3:-}

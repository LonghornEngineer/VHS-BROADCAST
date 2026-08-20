@echo off
REM Launch the VHS Broadcast Deck GUI (Windows)
where uv >nul 2>nul || (echo uv not found. Install from https://astral.sh/uv & exit /b 1)
echo Syncing dependencies...
uv sync
REM Offer the optional VCR OSD Mono font (prompts once if missing)
uv run --no-sync python scripts\fetch_vcr_font.py --if-missing
uv run --no-sync vhs-broadcast

@echo off
REM Launch the VHS Broadcast Deck GUI (Windows)
where uv >nul 2>nul || (echo uv not found. Install from https://astral.sh/uv & exit /b 1)
echo Syncing dependencies...
cd /d "%~dp0"
uv sync --locked || exit /b 1
uv run --no-sync vhs-broadcast

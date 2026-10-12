@echo off
REM Headless re-render. Usage: render.bat take.json out.mp4 [--full]
if "%~2"=="" (echo Usage: render.bat take.json out.mp4 [--full] & exit /b 1)
uv run vhs-render --session "%~1" --output "%~2" %3

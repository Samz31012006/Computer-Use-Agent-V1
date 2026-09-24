@echo off
rem Double-click to launch Buddy (no console window).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m cua.ui

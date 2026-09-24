@echo off
rem Temporary: same as start_buddy.bat but with voice debug capture enabled (saves each utterance's WAV +
rem metrics to logs\voice_debug\) so real misrecognitions can be diagnosed from actual captured audio.
cd /d "%~dp0"
set BUDDY_VOICE_DEBUG=1
start "" ".venv\Scripts\pythonw.exe" -m cua.ui

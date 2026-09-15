@echo off
title Subtitle Translator (LLM)
cd /d "%~dp0"

echo ============================================
echo   Subtitle Translator (LLM)
echo   http://localhost:8000
echo   Ctrl+C to stop the server
echo ============================================
echo.

REM Open the web app automatically after a short delay
REM (server needs a moment to start)
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8000'"

REM Run the app in the foreground (keeps logs visible, Ctrl+C stops it)
python run.py

pause
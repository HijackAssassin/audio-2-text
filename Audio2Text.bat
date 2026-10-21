@echo off
rem Launches Audio2Text. First run creates a virtual environment and installs dependencies.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First-time setup: creating virtual environment and installing dependencies...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)
start "" ".venv\Scripts\pythonw.exe" audio2text.py
exit /b 0

:error
echo Setup failed. Make sure Python 3.9+ is installed and on PATH.
pause
exit /b 1

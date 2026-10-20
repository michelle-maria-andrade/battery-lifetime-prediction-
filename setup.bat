@echo off
rem Creates .venv with Python 3.13 and installs the pinned requirements.
cd /d "%~dp0"

py -3.13 -m venv .venv
if errorlevel 1 (
    echo Python 3.13 was not found. Install it, then run this script again.
    exit /b 1
)

.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

echo Done. Activate with: .venv\Scripts\activate

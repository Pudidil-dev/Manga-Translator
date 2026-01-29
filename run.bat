@echo off
setlocal

cd /d "%~dp0"

REM Create venv with Python 3.11
py -3.11 -m venv .venv 2>nul
if errorlevel 1 goto NO_PY

call .venv\Scripts\activate.bat
if errorlevel 1 goto NO_VENV

python -m pip install --upgrade pip
pip install -r requirements.txt

set GEMINI_API_URL=http://localhost:8317/
set GEMINI_API_KEY=pudidil
set GEMINI_MODEL=gemini-2.5-flash

python app.py
pause
exit /b 0

:NO_PY
echo Python 3.11 tidak ditemukan.
echo Jalankan: winget install -e --id Python.Python.3.11
echo Atau install manual dari python.org (centang Add to PATH).
pause
exit /b 1

:NO_VENV
echo Gagal activate venv.
pause
exit /b 1

@echo off
REM START_BOT.bat — Double-click this file on Windows to start the bot
REM It will install dependencies on first run, start the server, and open the browser.

cd /d "%~dp0"

echo ==================================
echo   Weather Arb Bot — Starting...
echo ==================================
echo.

REM Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed.
    echo Install from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

python --version

REM Create venv on first run
if not exist "venv" (
    echo First run: creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Checking dependencies...
pip install -q -r requirements.txt 2>nul

echo.
echo Starting bot server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

REM Open browser after 3 seconds
start /b cmd /c "timeout /t 3 >nul && start http://localhost:8000"

cd bot
python api.py

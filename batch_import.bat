@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

rem Always cd to this script's folder (project root) so .venv is found.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [x] .venv\Scripts\python.exe not found.
    echo     Create the virtualenv and install dependencies first.
    echo.
    pause
    exit /b 2
)

".venv\Scripts\python.exe" "scripts\batch_import.py" %*
echo.
pause

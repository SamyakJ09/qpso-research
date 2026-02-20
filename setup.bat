@echo off
REM One-command project setup for Windows: creates venv, installs package + dev deps

set VENV_DIR=.venv

if exist "%VENV_DIR%\" (
    echo Virtual environment already exists at %VENV_DIR%
    echo To recreate: rmdir /s /q %VENV_DIR% ^& setup.bat
    exit /b 0
)

echo Creating virtual environment...
python -m venv %VENV_DIR%

echo Activating...
call %VENV_DIR%\Scripts\activate.bat

echo Upgrading pip...
pip install --upgrade pip

echo Installing project in editable mode with dev dependencies...
pip install -e ".[dev]"

echo.
echo Setup complete! Activate with:
echo   .venv\Scripts\activate
echo.
echo Then run:
echo   pytest                                                    # tests
echo   python experiments\run_comparison.py --benchmark sphere   # quick demo
echo.
echo For quantum modes (full/hybrid), also run:
echo   pip install -e ".[quantum]"

@echo off
REM TEMPER Validation Platform - Windows Launcher
REM Double-click to run all experiments

cd /d "%~dp0"
cd ..\..

REM Set PYTHONPATH so imports work
set PYTHONPATH=%CD%;%PYTHONPATH%

echo ==================================================
echo TEMPER VALIDATION PLATFORM
echo ==================================================
echo.

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Run experiments
echo Running experiments (this may take a few minutes)...
echo.

python temper_validation\run_validation.py --quick

echo.
echo ==================================================
echo COMPLETE!
echo ==================================================
echo.
echo Results saved in 'results' folder
echo.
pause

@echo off
REM TEMPER Verification Script (Windows)
REM Double-click to run

cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%

echo ==============================================
echo TEMPER LOCAL VERIFICATION
echo ==============================================
python verify_local.py

echo.
pause

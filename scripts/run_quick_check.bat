@echo off
cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%
echo ==============================================
echo TEMPER QUICK CHECK
echo ==============================================
echo.
echo 1. Import test...
python -c "from crucible import simulation; print('Imports OK')"
echo.
echo 2. Determinism check...
python verify_local.py
echo.
pause

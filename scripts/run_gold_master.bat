@echo off
REM TEMPER Gold Master - Comprehensive Test Suite
REM Covers: Exp 1, 3, 4, A, D

cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%

echo ==============================================
echo TEMPER GOLD MASTER - COMPREHENSIVE TEST
echo ==============================================
echo.
echo Runs Exp 1, 3, 4, A, D in one harness
echo.
python gold_master_hostile.py --seeds 10
pause

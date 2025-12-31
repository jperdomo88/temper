@echo off
cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%
echo ==============================================
echo EXP M: MACHIAVELLI TRANSFER
echo ==============================================
cd machiavelli
python run_transfer_test.py --kernel kernels/saint_kernel.json
pause

@echo off
REM Exp H: Cross-Domain Robustness Test
REM Paper Section 9.4
REM
REM WHAT IT PROVES:
REM SAINT/BRUTE separation is a genuine KERNEL property
REM Tests: 2 domains x 2 adapters = 4 conditions

cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%

echo ==============================================
echo EXP H: CROSS-DOMAIN ROBUSTNESS TEST
echo Paper Section 9.4
echo ==============================================
echo.
echo Proves SAINT/BRUTE separation is a genuine kernel property:
echo   - Not a gridworld quirk (tests INFO + COMMITMENT domains)
echo   - Not an adapter artifact (tests Adapter A + Adapter B)
echo.

python -c "from crucible.experiments.exp_h_combined import run_exp_h, ExpHConfig; config = ExpHConfig(n_seeds=10); results = run_exp_h(config)"

pause

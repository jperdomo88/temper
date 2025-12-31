@echo off
cd /d "%~dp0..\validation"
set PYTHONPATH=%CD%;%PYTHONPATH%
echo ==============================================
echo EXP 0: FITNESS ABLATION (Core Thesis)
echo ==============================================
python ablation_fitness.py --kernels 5 --evals 5
pause

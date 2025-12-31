@echo off
REM Registered Replication - Tempered RLHF Experiment
REM 
REM PURPOSE: Run identical protocol with N=15 seeds (vs prereg N=5)
REM This provides stronger statistical power while maintaining methodological integrity.
REM Report BOTH results in paper: "Prereg (N=5)" and "Replication (N=15)"
REM
REM GPT's guidance: "Add more seeds, same everything, same analysis"

echo ========================================
echo TEMPERED RLHF - REGISTERED REPLICATION
echo ========================================
echo.
echo Running N=15 seeds (vs prereg N=5)
echo Same protocol, stronger inference
echo.
echo Expected runtime: ~45-60 min on 14-core machine
echo.

cd /d "%~dp0"

REM Use 6 workers as reasonable default for Windows
set WORKERS=6

echo Using %WORKERS% parallel workers
echo.

REM Run with 15 seeds - everything else identical to prereg
python src/run_all_v2.py --seeds 15 --workers %WORKERS% %*

echo.
echo ========================================
echo Replication complete!
echo ========================================
echo.
echo Results saved to:
echo   - results/  (JSON data with N=15)
echo   - figures/  (updated plots)
echo.
echo For paper, report:
echo   - Prereg (N=5): original run_experiment results
echo   - Replication (N=15): these results
echo.
pause

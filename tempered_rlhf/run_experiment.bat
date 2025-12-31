@echo off
REM Run Tempered RLHF Experiment v2 - Complete Package (Prereg N=5)

echo ========================================
echo TEMPERED RLHF EXPERIMENT v2
echo ========================================
echo.
echo This runs ALL conditions:
echo   - Proxy (baseline)
echo   - Tempered (our mechanism)
echo   - Oracle (ceiling)
echo   - Ablation (visible fitness)
echo   - Born-gamed analysis
echo.

cd /d "%~dp0"

REM Use 6 workers as reasonable default for Windows
set WORKERS=6

echo Using %WORKERS% parallel workers
echo.

REM Default: 5 seeds (prereg)
REM Add --fast for quick test (~5 min)
REM Use run_replication.bat for N=15

python src/run_all_v2.py --workers %WORKERS% %*

echo.
echo ========================================
echo Experiment complete!
echo ========================================
echo.
echo Results saved to:
echo   - results/  (JSON data)
echo   - figures/  (plots)
echo.
pause

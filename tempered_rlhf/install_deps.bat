@echo off
REM Install dependencies for Tempered RLHF Experiment

echo ========================================
echo Installing dependencies...
echo ========================================

cd /d "%~dp0"

pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo You can now run the experiment with:
echo   Double-click run_experiment.bat (prereg N=5)
echo   Double-click run_replication.bat (replication N=15)
echo.
pause

@echo off
setlocal
cd /d "%~dp0"
echo QuTiP may take more than 12 hours at full CPU load on the 2048x2048 grid.
echo GQIS and Julia results are saved to CSV before QuTiP starts and can be plotted immediately.
rem Set GQIS_PYTHON to use a different Python environment.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
"%GQIS_PYTHON%" Benchmark_03_accuracy_timestep_sweep.py --settings "presets\Benchmark_03_two_level_2048.json"
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%

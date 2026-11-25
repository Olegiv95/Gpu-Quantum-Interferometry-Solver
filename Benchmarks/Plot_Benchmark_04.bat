@echo off
setlocal
cd /d "%~dp0"
rem Set GQIS_PYTHON to use a different Python environment.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
"%GQIS_PYTHON%" Benchmark_04_plot_from_csv.py --csv "results\Benchmark_04_hilbert_space_scaling_calibrated.csv" --no-show
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%

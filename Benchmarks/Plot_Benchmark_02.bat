@echo off
setlocal
cd /d "%~dp0"
rem Set GQIS_PYTHON to the Python environment with NumPy and Matplotlib installed.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
rem Use recorded CSV values, including points already marked as extrapolated.
"%GQIS_PYTHON%" Benchmark_01_02_plot_from_csv.py --csv "results\Benchmark_02_full_benchmark.csv" --stored-points %*
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%

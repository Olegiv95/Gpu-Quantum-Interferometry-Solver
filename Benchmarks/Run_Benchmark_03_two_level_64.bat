@echo off
setlocal
cd /d "%~dp0"
rem Set GQIS_PYTHON to use a different Python environment.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
"%GQIS_PYTHON%" Benchmark_03_accuracy_timestep_sweep.py --settings "presets\Benchmark_03_two_level_64.json"
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%


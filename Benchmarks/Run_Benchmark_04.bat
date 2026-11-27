@echo off
setlocal
cd /d "%~dp0"
rem Set GQIS_PYTHON to use a different Python environment.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
rem Saved points are kept; arguments are forwarded, e.g. --recalculate gqis_lserk4:6
"%GQIS_PYTHON%" Benchmark_04_hilbert_space_scaling.py %*
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%

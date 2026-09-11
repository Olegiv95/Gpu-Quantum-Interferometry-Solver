@echo off
setlocal
cd /d "%~dp0"
rem Set GQIS_PYTHON to use a different Python environment.
if not defined GQIS_PYTHON (
    set "GQIS_PYTHON=python"
    if exist "%USERPROFILE%\anaconda3\python.exe" set "GQIS_PYTHON=%USERPROFILE%\anaconda3\python.exe"
)
"%GQIS_PYTHON%" Benchmark_03_plot_from_csv.py --csv "results\Benchmark_03_two_level_2048_metrics.csv" --solvers gqis_rk4 gqis_ab5 gqis_anas5 gqis_tsit5 gqis_dop853 julia_gpu_fp32_fopt julia_gpu_fp32 qutip_cpu
set "benchmark_exit=%ERRORLEVEL%"
if not "%benchmark_exit%"=="0" echo Command failed with exit code %benchmark_exit%.
pause
exit /b %benchmark_exit%


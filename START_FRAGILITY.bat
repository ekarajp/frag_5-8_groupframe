@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0run.py"
) else (
  python -B -X utf8 "%~dp0run.py"
)
if errorlevel 1 pause

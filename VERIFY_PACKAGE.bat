@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0app\verify.py"
) else (
  python -B -X utf8 "%~dp0app\verify.py"
)
pause

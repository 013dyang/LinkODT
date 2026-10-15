@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=C:\ProgramData\anaconda3\envs\pytorch_gpu\python.exe"

"%PYTHON%" "tools\check_environment.py"
pause

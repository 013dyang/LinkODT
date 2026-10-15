@echo off
setlocal
cd /d "%~dp0"
if not defined LINKODT_PYTHON set "LINKODT_PYTHON=python"
if exist "%~dp0..\..\.linkodt_runtime\Scripts\python.exe" set "LINKODT_PYTHON=%~dp0..\..\.linkodt_runtime\Scripts\python.exe"
"%LINKODT_PYTHON%" main.py
if errorlevel 1 pause

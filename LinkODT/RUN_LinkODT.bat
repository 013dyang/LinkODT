@echo off
setlocal
cd /d "%~dp0"
rem LINKODT_PYTHON can select an existing scientific Python environment.
if defined LINKODT_PYTHON goto launch
if exist "%~dp0..\.linkodt_runtime\Scripts\python.exe" (
    set "LINKODT_PYTHON=%~dp0..\.linkodt_runtime\Scripts\python.exe"
    goto launch
)
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "LINKODT_PYTHON=%CONDA_PREFIX%\python.exe"
    goto launch
)
set "LINKODT_PYTHON=python"
:launch
"%LINKODT_PYTHON%" "LinkODT.py"
if errorlevel 1 (
    echo.
    echo LinkODT could not start. Activate the project environment or set LINKODT_PYTHON.
    echo See the user guide: Setting up Python.
    pause
)

@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo =============================================
echo LinkNex pretrained model downloader
echo =============================================
echo.

set "PYEXE="
set "USE_PY_LAUNCHER=0"

rem 1) Active Conda environment
if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" set "PYEXE=%CONDA_PREFIX%\python.exe"
)

rem 2) Project virtual environments
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0venv\Scripts\python.exe" set "PYEXE=%~dp0venv\Scripts\python.exe"

rem 3) Common Anaconda / Miniconda install locations
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PYEXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PYEXE=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\miniconda3\python.exe" set "PYEXE=%LOCALAPPDATA%\miniconda3\python.exe"
if not defined PYEXE if exist "C:\ProgramData\anaconda3\python.exe" set "PYEXE=C:\ProgramData\anaconda3\python.exe"
if not defined PYEXE if exist "C:\ProgramData\miniconda3\python.exe" set "PYEXE=C:\ProgramData\miniconda3\python.exe"

rem 4) Search PATH but explicitly reject the broken WindowsApps alias
if not defined PYEXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        echo %%P | findstr /I /C:"WindowsApps" >nul
        if errorlevel 1 (
            "%%P" -c "import sys" >nul 2>&1
            if !errorlevel! EQU 0 if not defined PYEXE set "PYEXE=%%P"
        )
    )
)

rem 5) Windows Python Launcher, if installed
if not defined PYEXE (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "USE_PY_LAUNCHER=1"
)

if defined PYEXE (
    echo [OK] Python found:
    echo      %PYEXE%
    echo.
    "%PYEXE%" tools\download_pretrained_models.py
    set "RC=%ERRORLEVEL%"
    echo.
    if not "%RC%"=="0" echo [ERROR] Downloader exited with code %RC%.
    pause
    exit /b %RC%
)

if "%USE_PY_LAUNCHER%"=="1" (
    echo [OK] Using Windows Python Launcher: py -3
    echo.
    py -3 tools\download_pretrained_models.py
    set "RC=%ERRORLEVEL%"
    echo.
    if not "%RC%"=="0" echo [ERROR] Downloader exited with code %RC%.
    pause
    exit /b %RC%
)

echo [ERROR] No usable Python installation was found.
echo.
echo Windows is currently resolving "python" to the Microsoft Store alias:
echo   %%LOCALAPPDATA%%\Microsoft\WindowsApps\python.exe
echo but that alias is not a real Python installation on this PC.
echo.
echo You can do ONE of the following:
echo   1. Install/repair Anaconda or Miniconda, then run this BAT again.
echo   2. Open Anaconda Prompt, activate your LinkNex environment, and run:
echo        python tools\download_pretrained_models.py
echo   3. To download only the recommended DINOv2 ViT-B/14 without Python,
echo      double-click: DOWNLOAD_DINO_VITB14_NO_PYTHON.bat

echo.
pause
exit /b 1

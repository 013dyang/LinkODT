@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo ============================================================
echo               LinkODT Windows EXE Builder
echo ============================================================
echo.

set "PYTHON=%CONDA_PREFIX%\python.exe"
if exist "%~dp0..\..\.linkodt_runtime\Scripts\python.exe" set "PYTHON=%~dp0..\..\.linkodt_runtime\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found:
    echo         %PYTHON%
    pause
    exit /b 1
)

echo [1/7] Python
"%PYTHON%" --version
if errorlevel 1 goto :fail

echo.
echo [2/7] PyInstaller
"%PYTHON%" -c "import PyInstaller; print('PyInstaller', PyInstaller.__version__)"
if errorlevel 1 (
    "%PYTHON%" -m pip install pyinstaller pyinstaller-hooks-contrib
    if errorlevel 1 goto :fail
)

echo.
echo [3/7] Syntax check
"%PYTHON%" -m py_compile "LinkODT.py"
if errorlevel 1 goto :fail

echo.
echo [4/7] Cleaning
if exist "build" rmdir /s /q "build"
if exist "dist\LinkODT" rmdir /s /q "dist\LinkODT"

echo.
echo [5/7] Building ONEDIR EXE
"%PYTHON%" -m PyInstaller --noconfirm --clean "build_windows\LinkODT_Windows.spec"
if errorlevel 1 goto :fail

echo.
echo [6/7] Copying offline DINO/model assets
if exist "models" (
    xcopy "models" "dist\LinkODT\_internal\models\" /E /I /Y >nul
)

if exist "dinov2-main" (
    xcopy "dinov2-main" "dist\LinkODT\_internal\dinov2-main\" /E /I /Y >nul
)

echo.
echo [7/7] DONE
echo.
echo Final:
echo   %CD%\dist\LinkODT\LinkODT.exe
echo.
echo Distribute the COMPLETE dist\LinkODT folder.
pause
exit /b 0

:fail
echo.
echo BUILD FAILED
pause
exit /b 1

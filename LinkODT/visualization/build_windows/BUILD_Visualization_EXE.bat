@echo off
setlocal
cd /d "%~dp0"
python -m pip install pyinstaller pyinstaller-hooks-contrib
python -m PyInstaller --noconfirm --clean LinkODT_Visualization.spec
if errorlevel 1 (
  echo.
  echo Build failed.
  pause
  exit /b 1
)
echo.
echo Built application:
echo   %CD%\dist\LinkODT_Visualization\LinkODT_Visualization.exe
echo.
echo Copy the whole LinkODT_Visualization folder into the main project's visualization folder,
echo or copy the executable plus its required bundled files as a self-contained release.
pause
endlocal

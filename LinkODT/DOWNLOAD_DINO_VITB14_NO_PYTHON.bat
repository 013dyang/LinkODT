@echo off
setlocal
cd /d "%~dp0"

set "MODEL_DIR=%~dp0models"
set "DEST=%MODEL_DIR%\dinov2_vitb14_pretrain.pth"
set "URL=https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%DEST%" (
    echo Model already exists:
    echo   %DEST%
    echo.
    choice /M "Download again and overwrite"
    if errorlevel 2 exit /b 0
    del /f /q "%DEST%"
)

echo =============================================
echo Downloading DINOv2 ViT-B/14
echo This is the recommended LinkNex backbone.
echo =============================================
echo.
echo Destination:
echo   %DEST%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -Uri '%URL%' -OutFile '%DEST%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }"

if errorlevel 1 (
    echo.
    echo [ERROR] Download failed.
    if exist "%DEST%" del /f /q "%DEST%"
    echo Check your internet connection or firewall and try again.
    pause
    exit /b 1
)

echo.
echo [OK] Download complete:
echo   %DEST%
pause

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$packageRoot = Join-Path (Split-Path $taskRoot -Parent) 'LinkODT_3.2_Windows'
$resultPath = Join-Path $taskRoot 'LinkODT/review/delivery_check.json'
# Test the bundled executable without inheriting scientific-Python search paths.
$env:PATH = "$env:SystemRoot/System32;$env:SystemRoot"
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONPATH,Env:PYTHONHOME -ErrorAction SilentlyContinue
$process = Start-Process -FilePath (Join-Path $packageRoot 'LinkODT.exe') -ArgumentList @('--delivery-check', ('"' + $resultPath + '"')) -WorkingDirectory $packageRoot -WindowStyle Hidden -PassThru
for ($attempt = 0; $attempt -lt 4; $attempt++) {
    if ($process.WaitForExit(30000)) { break }
}
if (-not $process.HasExited) {
    Stop-Process -Id $process.Id
    throw 'Executable verification did not finish within two minutes'
}
if (-not (Test-Path -LiteralPath $resultPath)) { throw 'Executable did not produce its verification record' }
Get-Content -LiteralPath $resultPath
if ($process.ExitCode -ne 0) { throw "Executable verification failed: $($process.ExitCode)" }

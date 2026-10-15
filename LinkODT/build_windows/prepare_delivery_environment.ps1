$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$envRoot = Join-Path $taskRoot '.linkodt_build'
if (-not (Test-Path -LiteralPath "$envRoot/Scripts/python.exe")) {
    & 'D:/Pycharm/Anaconda/python.exe' -m venv --without-pip $envRoot
    if ($LASTEXITCODE -ne 0) { throw 'Build environment creation failed' }
}
$destination = Join-Path $envRoot 'Lib/site-packages'
$sources = @((Join-Path $taskRoot '.linkodt_runtime/Lib/site-packages'), 'D:/Pycharm/Anaconda/Lib/site-packages')
foreach ($source in $sources) {
    foreach ($entry in Get-ChildItem -LiteralPath $source -Force) {
        if ($entry.Name -match '^(pathlib|enum34|typing-|PyQt5|PyQt6|PySide2|pip|setuptools|wheel|__pycache__)' -or $entry.Extension -eq '.pth') { continue }
        $target = Join-Path $destination $entry.Name
        if (Test-Path -LiteralPath $target) { continue }
        if ($entry.PSIsContainer) {
            New-Item -ItemType Junction -Path $target -Target $entry.FullName | Out-Null
        } elseif ($entry.Extension -in @('.py','.pyd','.dll')) {
            Copy-Item -LiteralPath $entry.FullName -Destination $target
        }
    }
}
Copy-Item -LiteralPath 'D:/Pycharm/Anaconda/Lib/site-packages/pywin32.pth' -Destination (Join-Path $destination 'pywin32.pth')
& "$envRoot/Scripts/python.exe" -m PyInstaller --version
if ($LASTEXITCODE -ne 0) { throw 'Build environment check failed' }

# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

HERE = Path(SPEC).resolve().parent
VIZ = HERE.parent
ROOT = VIZ.parent

hiddenimports = []
hiddenimports += collect_submodules('vtkmodules')
hiddenimports += collect_submodules('imageio_ffmpeg')
hiddenimports += collect_submodules('llama_cpp')

datas = [(str(ROOT / "assets"), "assets")]
datas += collect_data_files('imageio_ffmpeg')
binaries = []
binaries += collect_dynamic_libs('llama_cpp')
icon = ROOT / 'assets' / 'linkodt_app.ico'
if not icon.exists():
    icon = ROOT / 'assets' / 'linknex_app.ico'

a = Analysis(
    [str(VIZ / 'main.py')],
    pathex=[str(VIZ), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LinkODT_Visualization',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon) if icon.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LinkODT_Visualization',
)

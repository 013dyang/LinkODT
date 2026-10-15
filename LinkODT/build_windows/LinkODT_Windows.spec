# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(max(10000, sys.getrecursionlimit() * 10))

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

build_dir = Path(SPECPATH)
project_dir = build_dir.parent

datas = []
binaries = []
hiddenimports = []


def safe_collect_all(package_name):
    global datas, binaries, hiddenimports
    try:
        d, b, h = collect_all(package_name)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"[collect-all] {package_name}")
    except Exception as exc:
        print(f"[collect-all skipped] {package_name}: {exc}")


for package in [
    "pyvista",
    "pyvistaqt",
    "qtpy",
]:
    safe_collect_all(package)

try:
    d, b, h = collect_all(
        "open3d",
        filter_submodules=lambda name: (
            ".ml." not in name
            and not name.startswith("open3d.ml")
        ),
    )
    datas += d
    binaries += b
    hiddenimports += h
except TypeError:
    safe_collect_all("open3d")
except Exception as exc:
    print(f"[collect-all skipped] open3d: {exc}")

hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "numpy",
    "pandas",
    "networkx",
    "scipy",
    "scipy.ndimage",
    "scipy.spatial",
    "scipy.sparse",
    "scipy.sparse.csgraph",
    "scipy.sparse.linalg",
    "scipy.stats",
    "skimage",
    "skimage.measure",
    "skimage.morphology",
    "skimage.segmentation",
    "sklearn",
    "sklearn.cluster",
    "matplotlib",
    "matplotlib.pyplot",
    "tifffile",
    "PIL",
    "PIL.Image",
    "torch",
    "torchvision",
    "omegaconf",
    "fvcore",
    "iopath",
    "einops",
    "timm",
    "umap",
    "vtkmodules",
    "vtkmodules.qt",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.vtkRenderingOpenGL2",
    "vtkmodules.vtkInteractionStyle",
]

hiddenimports = list(dict.fromkeys(hiddenimports))
hiddenimports += ['visualization_app', 'director', 'io_utils', 'ai_director', 'imageio_ffmpeg']
datas += collect_all('imageio_ffmpeg')[0]

for filename in [
    "linknex_app.ico",
    "linknex_logo_full_transparent.png",
    "linknex_logo_round_transparent.png",
]:
    source = project_dir / filename
    if source.exists():
        datas.append((str(source), "."))

datas.append((str(project_dir / "assets"), "assets"))

a = Analysis(
    [str(project_dir / "LinkODT.py")],
    pathex=[str(project_dir), str(project_dir / 'visualization')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "tensorflow",
        "tensorflow_estimator",
        "keras",
        "cupy",
        "trame",
        "jupyter",
        "notebook",
        "pytest",
        "IPython",
        "boto3",
        "botocore",
        "dask",
        "distributed",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LinkODT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_dir / "assets" / "linkodt_app.ico")
         if (project_dir / "assets" / "linkodt_app.ico").exists()
         else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LinkODT",
)

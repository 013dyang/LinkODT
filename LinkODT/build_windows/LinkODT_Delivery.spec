# -*- mode: python ; coding: utf-8 -*-

import sys,os
sys.setrecursionlimit(max(10000, sys.getrecursionlimit() * 10))

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

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
]:
    safe_collect_all(package)
datas += collect_data_files('qtpy')

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

for filename in [
    "linknex_app.ico",
    "linknex_logo_full_transparent.png",
    "linknex_logo_round_transparent.png",
]:
    source = project_dir / filename
    if source.exists():
        datas.append((str(source), "."))

datas.append((str(project_dir / "assets"), "assets"))
for source in (project_dir / 'dinov2-main').rglob('*'):
    if not source.is_file() or '__pycache__' in source.parts:
        continue
    relative = source.relative_to(project_dir / 'dinov2-main')
    if relative.parts[0] in ('.github', 'docs', 'notebooks', 'scripts'):
        continue
    if source.suffix == '.py' or source.name.startswith('LICENSE'):
        datas.append((str(source), str(Path('dinov2-main') / relative.parent)))
hiddenimports += ['visualization_app','director','io_utils','ai_director','semantic_models',
                  'imageio_ffmpeg','torchmetrics','submitit','skimage.morphology._skeletonize']
datas += collect_all('imageio_ffmpeg')[0]
# Embedded backends import modules at runtime; include their actual imports.
import ast
tree = ast.parse((project_dir / 'LinkNex.py').read_text(encoding='utf-8'))
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        if any(isinstance(t,ast.Name) and t.id.endswith('_SOURCE') for t in node.targets):
            embedded = ast.parse(node.value.value)
            for child in ast.walk(embedded):
                if isinstance(child,ast.Import):hiddenimports.extend(a.name for a in child.names)
                elif isinstance(child,ast.ImportFrom) and child.module:hiddenimports.append(child.module)
hiddenimports = list(dict.fromkeys(hiddenimports))

if os.environ.get('LINKODT_REBUILD_EXE_ONLY'):
    from types import SimpleNamespace
    from PyInstaller.utils.misc import load_py_data_struct
    from PyInstaller.config import CONF
    cached = load_py_data_struct(Path(CONF['workpath']) / 'Analysis-00.toc')
    a = SimpleNamespace(**{key[0]:value for key,value in zip(Analysis._GUTS,cached)})
else:
    a = Analysis(
    [str(build_dir / "delivery_entry.py")],
    pathex=[str(project_dir),str(project_dir / 'visualization')],
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
        "transformers",
        "dash",
        "ipywidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "trimesh",
        "rtree",
        "plotly",
        "onnx",
        "onnxruntime",
        "bokeh",
        "tkinter",
    ],
    noarchive=False,
)

# Use the portable-build-compatible threadpool controller without changing
# the user's existing scientific environment.
studio_modules = ['studio_ui','display_controls','ai_dialog','_studio_core','_studio_io','_studio_director']
pure_names = {entry[0] for entry in a.pure}
for name in studio_modules:
    if name not in pure_names:
        extension = '.pyc' if name.startswith('_studio_') else '.py'
        a.pure.append((name,str(project_dir/'visualization'/(name+extension)),'PYMODULE'))
controller = project_dir / 'review/delivery_addons/threadpoolctl.py'
if controller.exists():
    a.pure = [(name, str(controller) if name == 'threadpoolctl' else path, kind)
              for name, path, kind in a.pure]
if os.environ.get('LINKODT_REBUILD_EXE_ONLY'):
    # Third-party modules are unchanged in an EXE-only refresh. Reuse their
    # already verified code, while compiling every current application module.
    from PyInstaller.archive.readers import ZlibArchiveReader
    import marshal
    existing_pyz = Path(CONF['workpath']) / 'PYZ-00.pyz'
    if existing_pyz.exists():
        reader = ZlibArchiveReader(str(existing_pyz))
        cache = {name:reader.extract(name) for name in reader.toc}
        for name,path,kind in a.pure:
            if path and Path(path).is_relative_to(project_dir):
                content = Path(path).read_bytes()
                cache[name] = (marshal.loads(content[16:]) if Path(path).suffix=='.pyc'
                    else compile(content,str(path),'exec'))
        CONF['code_cache'][id(a.pure)] = cache
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

import os
if not os.environ.get('LINKODT_REBUILD_EXE_ONLY'):
    coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LinkODT",
)

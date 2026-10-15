from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("tifffile", "tifffile"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("networkx", "networkx"),
    ("open3d", "open3d"),
    ("pyvista", "pyvista"),
    ("pyvistaqt", "pyvistaqt"),
    ("vtk", "vtk"),
    ("PySide6", "PySide6"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("umap", "umap-learn"),
    ("omegaconf", "omegaconf"),
    ("fvcore", "fvcore"),
    ("iopath", "iopath"),
    ("einops", "einops"),
    ("timm", "timm"),
]

print("=" * 72)
print("LinkODT environment check")
print("=" * 72)
print("Python:", sys.executable)
print("Version:", sys.version.replace("\n", " "))
print()

missing = []

for module_name, package_name in MODULES:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "OK")
        print(f"[OK]      {module_name:12s} {version}")
    except Exception as exc:
        print(f"[MISSING] {module_name:12s} -> pip/conda package: {package_name}")
        print(f"          {exc}")
        missing.append(package_name)

print()
print("Project files")
print("-------------")

checks = [
    PROJECT_ROOT / "LinkODT.py",
    PROJECT_ROOT / "semantic_models.py",
    PROJECT_ROOT / "dinov2-main" / "hubconf.py",
    PROJECT_ROOT / "tools" / "download_pretrained_models.py",
    PROJECT_ROOT / "assets/linkodt_wordmark.png",
    PROJECT_ROOT / "assets/linkodt_icon.png",
]

for path in checks:
    print(
        "[OK]     " if path.exists() else "[MISSING]",
        path,
    )

try:
    import torch
    print()
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA:", torch.version.cuda)
except Exception:
    pass

print()
if missing:
    print("Environment is NOT complete.")
    print("Missing packages:", ", ".join(sorted(set(missing))))
else:
    print("Python package import check passed.")

print()
weight_names = [
    "dinov2_vits14_pretrain.pth",
    "dinov2_vitb14_pretrain.pth",
    "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4_pretrain.pth",
]
found_weights = [
    PROJECT_ROOT / "models" / name
    for name in weight_names
    if (PROJECT_ROOT / "models" / name).exists()
]
if found_weights:
    print("DINOv2 pretrained assets found:")
    for path in found_weights:
        print("  ", path)
else:
    print("NOTE: DINOv2 weights are intentionally not bundled.")
    print("Download one/all with:")
    print("  python tools/download_pretrained_models.py")

try:
    import transformers  # noqa: F401
    print("Optional SegFormer support: transformers is installed.")
except Exception:
    print("Optional SegFormer support: not installed (only needed for SegFormer-B2).")

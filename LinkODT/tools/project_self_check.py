from pathlib import Path
import py_compile
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "LinkODT.py",
    ROOT / "semantic_models.py",
    ROOT / "dinov2-main" / "hubconf.py",
    ROOT / "ssl" / "train_dinov2_microscopy_ssl.py",
    ROOT / "tools" / "download_pretrained_models.py",
    ROOT / "assets/linkodt_wordmark.png",
    ROOT / "assets/linkodt_icon.png",
]

print("=" * 72)
print("LinkODT project self-check")
print("=" * 72)

failed = False

for path in required:
    if path.exists():
        print("[OK]", path.relative_to(ROOT))
    else:
        print("[MISSING]", path.relative_to(ROOT))
        failed = True

print()
print("Compiling project Python sources...")

for path in sorted(ROOT.rglob("*.py")):
    if "dinov2-main" in path.parts:
        continue
    try:
        py_compile.compile(str(path), doraise=True)
        print("[OK]", path.relative_to(ROOT))
    except Exception as exc:
        print("[FAIL]", path.relative_to(ROOT), exc)
        failed = True

print()
weight_names = [
    "dinov2_vits14_pretrain.pth",
    "dinov2_vitb14_pretrain.pth",
    "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4_pretrain.pth",
]
found = [ROOT / "models" / name for name in weight_names if (ROOT / "models" / name).exists()]
if found:
    for path in found:
        print("[WEIGHT OK]", path.relative_to(ROOT))
else:
    print("[WEIGHT NOTE] No official DINOv2 weight is bundled in this source package.")
    print("Download one with: python tools/download_pretrained_models.py")

segformer = ROOT / "models" / "hf_segformer_b2_ade"
if segformer.exists():
    print("[OPTIONAL OK] SegFormer-B2 local cache found.")
else:
    print("[OPTIONAL] SegFormer-B2 is not cached; only needed for that GUI baseline.")

sys.exit(1 if failed else 0)

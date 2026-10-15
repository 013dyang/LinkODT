# -*- coding: utf-8 -*-
"""Download pretrained assets used by LinkNex semantic segmentation.

Examples
--------
python tools/download_pretrained_models.py --model dinov2_vitb14
python tools/download_pretrained_models.py --model all-dino
python tools/download_pretrained_models.py --model segformer_b2
python tools/download_pretrained_models.py --model all

With no --model argument an interactive menu is shown.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

DINO_URLS = {
    "dinov2_vits14": (
        "dinov2_vits14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    ),
    "dinov2_vitb14": (
        "dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    ),
    "dinov2_vits14_reg": (
        "dinov2_vits14_reg4_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    ),
    "dinov2_vitb14_reg": (
        "dinov2_vitb14_reg4_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    ),
}

SEGFORMER_REPO = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_DIR = MODELS_DIR / "hf_segformer_b2_ade"


def _download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print(f"[SKIP] already exists: {dst}")
        return

    partial = dst.with_suffix(dst.suffix + ".part")
    if partial.exists():
        partial.unlink()

    print(f"[DOWNLOAD] {url}")
    print(f"       -> {dst}")

    class Progress:
        def __init__(self):
            self.last = -1

        def __call__(self, block_num, block_size, total_size):
            if total_size <= 0:
                return
            done = min(block_num * block_size, total_size)
            pct = int(done * 100 / total_size)
            if pct >= self.last + 5 or pct == 100:
                self.last = pct
                print(f"  {pct:3d}%  {done / 1024**2:8.1f} / {total_size / 1024**2:8.1f} MB")

    try:
        urllib.request.urlretrieve(url, partial, reporthook=Progress())
        partial.replace(dst)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def download_dino(key: str) -> None:
    if key not in DINO_URLS:
        raise ValueError(f"Unknown DINOv2 model: {key}")
    filename, url = DINO_URLS[key]
    _download_url(url, MODELS_DIR / filename)


def download_segformer() -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "SegFormer download requires huggingface_hub.\n"
            "Install it with:\n"
            "  pip install huggingface_hub transformers safetensors\n"
            "or:\n"
            "  pip install -r requirements_semantic_optional.txt"
        ) from exc

    SEGFORMER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] Hugging Face: {SEGFORMER_REPO}")
    print(f"       -> {SEGFORMER_DIR}")
    snapshot_download(
        repo_id=SEGFORMER_REPO,
        local_dir=str(SEGFORMER_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "*.json",
        ],
    )
    print("[OK] SegFormer-B2 cached locally.")


def run_selection(selection: str) -> None:
    selection = selection.strip().lower()
    if selection in DINO_URLS:
        download_dino(selection)
    elif selection == "all-dino":
        for key in DINO_URLS:
            download_dino(key)
    elif selection == "segformer_b2":
        download_segformer()
    elif selection == "all":
        for key in DINO_URLS:
            download_dino(key)
        download_segformer()
    else:
        raise ValueError(f"Unsupported selection: {selection}")


def interactive_choice() -> str:
    print("\nLinkNex pretrained model downloader")
    print("===================================")
    print("1  DINOv2 ViT-S/14            (fast)")
    print("2  DINOv2 ViT-B/14            (recommended final backbone)")
    print("3  DINOv2 ViT-S/14 + registers")
    print("4  DINOv2 ViT-B/14 + registers")
    print("5  All DINOv2 backbones")
    print("6  SegFormer-B2 external baseline")
    print("7  All assets")
    print("0  Exit")
    mapping = {
        "1": "dinov2_vits14",
        "2": "dinov2_vitb14",
        "3": "dinov2_vits14_reg",
        "4": "dinov2_vitb14_reg",
        "5": "all-dino",
        "6": "segformer_b2",
        "7": "all",
    }
    while True:
        choice = input("Choose: ").strip()
        if choice == "0":
            raise SystemExit(0)
        if choice in mapping:
            return mapping[choice]
        print("Please enter 0-7.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[*DINO_URLS.keys(), "all-dino", "segformer_b2", "all"],
        default=None,
        help="Asset to download. Without this flag an interactive menu is shown.",
    )
    args = parser.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    selection = args.model or interactive_choice()
    try:
        run_selection(selection)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    print(f"Model directory: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# LinkNex microscopy DINO SSL - canonical project script
# -*- coding: utf-8 -*-
"""
Microscopy DINO self-supervised domain adaptation on a true 1-channel DINOv2 ViT backbone.

Purpose
-------
Use a large unlabeled microscopy corpus to adapt the official DINOv2 encoder
to microscopy before supervised segmentation training.

Edit the USER CONFIG section near the bottom of this file and Run directly.
No argparse / command-line arguments are required.

This script intentionally stays OUTSIDE the LinkNex GUI.

Recommended use
---------------
official DINOv2 ViT-S/14 or ViT-B/14 checkpoint
        ↓
DINO-style self-distillation on unlabeled microscopy
        ↓
microscopy-adapted ViT-S/14 encoder
        ↓
LinkNex supervised semantic segmentation

Important
---------
This is a practical DINO-style domain-adaptation recipe using a DINOv2 backbone.
It is NOT a bit-for-bit reproduction of Meta's full DINOv2 pretraining recipe
(which additionally uses components such as iBOT / KoLeo and a distributed
training stack).

Input
-----
A directory recursively containing:
    .tif / .tiff / .png / .jpg / .jpeg / .bmp

For 3D TIFF stacks such as ZYX:
    --tiff-stack-mode random_slice   (default)
will sample one z-slice from each stack every time the file is visited.

Outputs
-------
output_dir/
    ssl_training_log.csv
    ssl_loss_curve.png
    representation_visualization/
        epoch_000/   # official pretrained DINOv2 baseline
        epoch_005/
        epoch_010/
        ...
    nearest_neighbor_retrieval/
        epoch_000/
        epoch_005/
        epoch_010/
        ...
    ssl_training_files.csv
    nearest_neighbor_retrieval_manifest.csv
    checkpoints/
        {backbone_name}_microscopy_epoch005_YYYYMMDD_HHMMSS.pth
        {backbone_name}_microscopy_epoch010_YYYYMMDD_HHMMSS.pth
        ...
        ssl_checkpoint_latest.pth
    encoders/
        {backbone_name}_microscopy_epoch010_YYYYMMDD_HHMMSS.pth
        ...
        train_dinov2_micro_epochFINAL_YYYYMMDD_HHMMSS.pth

The encoder .pth files contain ONLY the DINOv2 backbone state_dict and are
intended to be easy to reuse in LinkNex segmentation training.
"""

from __future__ import annotations

import copy
import csv
import math
import random
import re
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Training-file selection
# ---------------------------------------------------------------------

def select_training_records(
    records: Sequence[ImageRecord],
    count: int,
    seed: int,
) -> List[ImageRecord]:
    """
    Fixed random selection of training files/stacks.

    This intentionally samples FILES/stacks rather than adjacent Z slices.
    The same seed + same discovered file list gives the same selection.
    """
    records = list(records)

    if count <= 0 or count >= len(records):
        return records

    rng = random.Random(int(seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)

    selected_indices = sorted(
        indices[: int(count)]
    )

    return [
        records[index]
        for index in selected_indices
    ]


def save_training_manifest(
    path: Path,
    records: Sequence[ImageRecord],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "path",
                "is_tiff",
                "shape",
                "axes",
            ],
        )
        writer.writeheader()

        for index, record in enumerate(
            records,
            1,
        ):
            writer.writerow(
                {
                    "index": index,
                    "path": str(record.path),
                    "is_tiff": int(record.is_tiff),
                    "shape": "x".join(
                        str(v)
                        for v in record.shape
                    ),
                    "axes": record.axes,
                }
            )


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

@dataclass
class ImageRecord:
    path: Path
    is_tiff: bool
    shape: Tuple[int, ...]
    axes: str


def _safe_axes(axes: Optional[str], ndim: int) -> str:
    if axes:
        return str(axes).upper()
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "ZYX"
    return ""


def inspect_image(path: Path) -> ImageRecord:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            shape = tuple(int(v) for v in series.shape)
            axes = _safe_axes(getattr(series, "axes", ""), len(shape))
        return ImageRecord(path, True, shape, axes)

    # For non-TIFF files, shape metadata is not needed up front.
    return ImageRecord(path, False, tuple(), "")


def discover_images(
    root: Path,
    exclude_keywords: Sequence[str],
    max_files: int = 0,
) -> List[ImageRecord]:
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(root)

    exclude_keywords = [
        item.lower().strip()
        for item in exclude_keywords
        if item.strip()
    ]

    paths: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        name_lower = path.name.lower()

        if any(keyword in name_lower for keyword in exclude_keywords):
            continue

        paths.append(path)

    paths.sort()

    if max_files > 0:
        paths = paths[:max_files]

    if not paths:
        raise RuntimeError(
            f"No supported microscopy images found under:\n{root}"
        )

    print(f"Inspecting {len(paths)} input files...")

    records: List[ImageRecord] = []

    for i, path in enumerate(paths, 1):
        try:
            records.append(inspect_image(path))
        except Exception as exc:
            print(f"[SKIP] Cannot inspect {path}: {exc}")

        if i % 1000 == 0:
            print(f"  inspected {i}/{len(paths)}")

    if not records:
        raise RuntimeError("No readable input images were found.")

    return records


def _read_tiff_2d(
    record: ImageRecord,
    stack_mode: str,
) -> np.ndarray:
    path = record.path
    shape = record.shape
    axes = record.axes

    # Normal 2D TIFF
    if len(shape) == 2:
        return np.asarray(tifffile.imread(path))

    # Common LinkNex microscopy case: ZYX
    if len(shape) == 3 and axes == "ZYX":
        z_count = shape[0]

        if stack_mode == "random_slice":
            z = random.randrange(z_count)
            return np.asarray(tifffile.imread(path, key=z))

        if stack_mode == "middle_slice":
            z = z_count // 2
            return np.asarray(tifffile.imread(path, key=z))

        if stack_mode == "max_projection":
            arr = np.asarray(tifffile.imread(path))
            return np.max(arr, axis=0)

    # Some TIFFs may have CYX / TYX.
    if len(shape) == 3 and axes.endswith("YX"):
        first_dim = shape[0]

        if stack_mode == "random_slice":
            i = random.randrange(first_dim)
        else:
            i = first_dim // 2

        try:
            return np.asarray(tifffile.imread(path, key=i))
        except Exception:
            arr = np.asarray(tifffile.imread(path))
            return arr[i]

    # Generic fallback. Squeeze singleton dimensions and then select
    # from leading dimensions until a 2D image remains.
    arr = np.asarray(tifffile.imread(path))
    arr = np.squeeze(arr)

    while arr.ndim > 2:
        if stack_mode == "random_slice":
            index = random.randrange(arr.shape[0])
        elif stack_mode == "max_projection":
            arr = np.max(arr, axis=0)
            continue
        else:
            index = arr.shape[0] // 2

        arr = arr[index]

    if arr.ndim != 2:
        raise ValueError(
            f"Could not reduce TIFF to 2D: {path}, shape={record.shape}, axes={axes}"
        )

    return arr


def _read_standard_2d(record: ImageRecord, stack_mode: str) -> np.ndarray:
    if record.is_tiff:
        return _read_tiff_2d(record, stack_mode)

    from PIL import Image

    with Image.open(record.path) as image:
        arr = np.asarray(image)

    if arr.ndim == 3:
        # Convert RGB/RGBA into grayscale intensity because the microscopy
        # self-supervised pipeline is intentionally intensity-based.
        arr = arr[..., :3].astype(np.float32)
        arr = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        )

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported non-TIFF image shape {arr.shape}: {record.path}"
        )

    return arr


def percentile_normalize(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)

    finite = np.isfinite(image)

    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")

    values = image[finite].astype(np.float32)

    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.8))

    if not np.isfinite(lo):
        lo = float(np.min(values))

    if not np.isfinite(hi):
        hi = float(np.max(values))

    if hi <= lo + 1e-8:
        hi = lo + 1.0

    x = image.astype(np.float32)
    x = np.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    return torch.from_numpy(x).unsqueeze(0)


# ---------------------------------------------------------------------
# Microscopy augmentations
# ---------------------------------------------------------------------

class MicroscopyMultiCrop:
    """
    Grayscale/intensity-aware DINO augmentations.

    Deliberately avoids hue/saturation augmentation because microscopy
    intensities are not natural-image RGB colors.
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 98,
        local_crops: int = 4,
    ):
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.local_crops = int(local_crops)

        if self.global_size % 14 != 0:
            raise ValueError("global_size must be divisible by DINOv2 patch size 14.")

        if self.local_size % 14 != 0:
            raise ValueError("local_size must be divisible by DINOv2 patch size 14.")

    @staticmethod
    def _random_resized_crop(
        x: torch.Tensor,
        output_size: int,
        scale: Tuple[float, float],
    ) -> torch.Tensor:
        i, j, h, w = RandomResizedCrop.get_params(
            x,
            scale=scale,
            ratio=(0.75, 1.3333333333),
        )

        return TF.resized_crop(
            x,
            i,
            j,
            h,
            w,
            [output_size, output_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    @staticmethod
    def _intensity_augment(x: torch.Tensor) -> torch.Tensor:
        # Horizontal / vertical orientation is not semantically fixed for
        # most cellular microscopy fields.
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])

        # Exact 90-degree rotations avoid interpolation artifacts.
        k = random.randrange(4)
        if k:
            x = torch.rot90(x, k, dims=[1, 2])

        # Contrast around image mean.
        if random.random() < 0.8:
            contrast = random.uniform(0.65, 1.45)
            mean = x.mean()
            x = (x - mean) * contrast + mean

        # Brightness scaling.
        if random.random() < 0.8:
            brightness = random.uniform(0.7, 1.3)
            x = x * brightness

        # Gamma variation models different microscope/exposure display
        # characteristics without inventing color information.
        if random.random() < 0.7:
            gamma = random.uniform(0.65, 1.55)
            x = torch.clamp(x, 0.0, 1.0).pow(gamma)

        # Blur.
        if random.random() < 0.35:
            kernel = random.choice([3, 5, 7])
            sigma = random.uniform(0.2, 1.5)
            x = TF.gaussian_blur(
                x,
                kernel_size=[kernel, kernel],
                sigma=[sigma, sigma],
            )

        # Detector-like Gaussian noise.
        if random.random() < 0.45:
            sigma = random.uniform(0.0, 0.05)
            x = x + torch.randn_like(x) * sigma

        x = torch.clamp(x, 0.0, 1.0)

        # LinkNex microscopy DINO uses TRUE one-channel input. The official
        # DINOv2 RGB patch-embedding kernels are averaged to one channel when
        # the backbone is created below.
        gray_mean = (0.485 + 0.456 + 0.406) / 3.0
        gray_std = (0.229 + 0.224 + 0.225) / 3.0
        x = TF.normalize(
            x,
            mean=[gray_mean],
            std=[gray_std],
        )

        return x

    def _make_view(
        self,
        x: torch.Tensor,
        size: int,
        scale: Tuple[float, float],
    ) -> torch.Tensor:
        x = self._random_resized_crop(
            x,
            output_size=size,
            scale=scale,
        )
        x = self._intensity_augment(x)
        return x

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        views = [
            self._make_view(
                x.clone(),
                self.global_size,
                (0.40, 1.00),
            ),
            self._make_view(
                x.clone(),
                self.global_size,
                (0.40, 1.00),
            ),
        ]

        for _ in range(self.local_crops):
            views.append(
                self._make_view(
                    x.clone(),
                    self.local_size,
                    (0.08, 0.40),
                )
            )

        return views


class MicroscopySSLDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        stack_mode: str,
        transform: MicroscopyMultiCrop,
    ):
        self.records = list(records)
        self.stack_mode = stack_mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]

        image = _read_standard_2d(
            record,
            stack_mode=self.stack_mode,
        )

        x = percentile_normalize(image)
        views = self.transform(x)

        return {
            "views": views,
            "path": str(record.path),
        }


# ---------------------------------------------------------------------
# DINO model
# ---------------------------------------------------------------------

class DINOProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 8192,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.last_layer = nn.Linear(
            bottleneck_dim,
            out_dim,
            bias=False,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def _extract_backbone_feature(
    backbone: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    out = backbone(x)

    if torch.is_tensor(out):
        if out.ndim == 2:
            return out

        if out.ndim == 3:
            return out[:, 0]

    if isinstance(out, dict):
        for key in (
            "x_norm_clstoken",
            "cls_token",
            "features",
            "x",
        ):
            value = out.get(key)
            if torch.is_tensor(value):
                if value.ndim == 2:
                    return value
                if value.ndim == 3:
                    return value[:, 0]

    # Fallback to DINOv2 forward_features API.
    if hasattr(backbone, "forward_features"):
        features = backbone.forward_features(x)

        if isinstance(features, dict):
            value = features.get("x_norm_clstoken")

            if torch.is_tensor(value):
                return value

    raise RuntimeError(
        "Could not extract a CLS embedding from the DINOv2 backbone."
    )


class DINOStudentTeacher(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
    ):
        super().__init__()

        self.student_backbone = backbone

        embed_dim = int(
            getattr(
                backbone,
                "embed_dim",
                getattr(backbone, "num_features", 384),
            )
        )

        self.student_head = DINOProjectionHead(
            embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=out_dim,
        )

        self.teacher_backbone = copy.deepcopy(
            self.student_backbone
        )
        self.teacher_head = copy.deepcopy(
            self.student_head
        )

        for parameter in self.teacher_backbone.parameters():
            parameter.requires_grad_(False)

        for parameter in self.teacher_head.parameters():
            parameter.requires_grad_(False)

        self.teacher_backbone.eval()
        self.teacher_head.eval()

    def student_forward(
        self,
        crops: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs = []

        for crop in crops:
            feature = _extract_backbone_feature(
                self.student_backbone,
                crop,
            )
            outputs.append(
                self.student_head(feature)
            )

        return outputs

    @torch.no_grad()
    def teacher_forward(
        self,
        global_crops: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs = []

        for crop in global_crops:
            feature = _extract_backbone_feature(
                self.teacher_backbone,
                crop,
            )
            outputs.append(
                self.teacher_head(feature)
            )

        return outputs

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for student, teacher in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters(),
        ):
            teacher.data.mul_(momentum).add_(
                student.data,
                alpha=1.0 - momentum,
            )

        for student, teacher in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters(),
        ):
            teacher.data.mul_(momentum).add_(
                student.data,
                alpha=1.0 - momentum,
            )


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)

        self.register_buffer(
            "center",
            torch.zeros(1, out_dim),
        )

    def forward(
        self,
        student_outputs: Sequence[torch.Tensor],
        teacher_outputs: Sequence[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        student_log_probs = [
            F.log_softmax(
                output / self.student_temp,
                dim=-1,
            )
            for output in student_outputs
        ]

        teacher_probs = [
            F.softmax(
                (output - self.center) / teacher_temp,
                dim=-1,
            ).detach()
            for output in teacher_outputs
        ]

        total_loss = 0.0
        n_terms = 0

        # Teacher receives two global views.
        # Student receives the same two global views + local views.
        # Skip matching global-view pairs.
        for teacher_view_index, teacher_prob in enumerate(
            teacher_probs
        ):
            for student_view_index, student_log_prob in enumerate(
                student_log_probs
            ):
                if student_view_index == teacher_view_index:
                    continue

                loss = torch.sum(
                    -teacher_prob * student_log_prob,
                    dim=-1,
                ).mean()

                total_loss = total_loss + loss
                n_terms += 1

        if n_terms == 0:
            raise RuntimeError("No DINO cross-view loss terms were generated.")

        return total_loss / n_terms

    @torch.no_grad()
    def update_center(
        self,
        teacher_outputs: Sequence[torch.Tensor],
    ) -> None:
        batch_center = torch.cat(
            list(teacher_outputs),
            dim=0,
        ).mean(dim=0, keepdim=True)

        self.center.mul_(
            self.center_momentum
        ).add_(
            batch_center,
            alpha=1.0 - self.center_momentum,
        )


# ---------------------------------------------------------------------
# DINOv2 loading
# ---------------------------------------------------------------------

def _clean_state_dict(
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned = {}

    prefixes = (
        "module.",
        "student.backbone.",
        "teacher.backbone.",
        "backbone.",
    )

    for key, value in state.items():
        new_key = str(key)

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def _extract_possible_state_dict(
    checkpoint: object,
) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        # Plain official backbone state_dict
        if checkpoint and all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        ):
            return _clean_state_dict(checkpoint)

        for key in (
            "state_dict",
            "model",
            "backbone",
            "teacher",
            "student",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                try:
                    return _clean_state_dict(
                        _extract_possible_state_dict(value)
                    )
                except Exception:
                    pass

    raise ValueError(
        "Could not identify a DINOv2 backbone state_dict in checkpoint."
    )


def build_dinov2_backbone(
    dinov2_repo: Path,
    init_checkpoint: Optional[Path],
    backbone_name: str = "dinov2_vits14",
) -> nn.Module:
    """Build local DINOv2 and adapt its RGB patch embedding to 1 channel."""
    dinov2_repo = Path(dinov2_repo)
    if not dinov2_repo.exists():
        raise FileNotFoundError(
            f"DINOv2 repository not found:\n{dinov2_repo}"
        )

    allowed = {
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
    }
    if backbone_name not in allowed:
        raise ValueError(
            f"Unsupported BACKBONE_NAME={backbone_name}. Choose one of {sorted(allowed)}"
        )

    print(f"\nLoading local DINOv2 backbone: {backbone_name}")
    print("Repository:", dinov2_repo)

    backbone = torch.hub.load(
        str(dinov2_repo),
        backbone_name,
        source="local",
        pretrained=False,
        verbose=False,
    )

    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint)
        if not init_checkpoint.exists():
            raise FileNotFoundError(
                f"Initial DINOv2 checkpoint not found:\n{init_checkpoint}"
            )
        print("Initial checkpoint:", init_checkpoint)
        raw = torch.load(init_checkpoint, map_location="cpu")
        state = _extract_possible_state_dict(raw)

        patch_key = "patch_embed.proj.weight"
        is_gray_checkpoint = (
            patch_key in state
            and state[patch_key].ndim == 4
            and int(state[patch_key].shape[1]) == 1
        )

        if is_gray_checkpoint:
            old = backbone.patch_embed.proj
            new = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )
            backbone.patch_embed.proj = new

        result = backbone.load_state_dict(state, strict=False)
        print(
            "Initial checkpoint loaded:",
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}",
        )
    else:
        print("[WARNING] no initialization checkpoint; backbone is random.")

    # Official DINOv2 is RGB. After loading those pretrained weights, average
    # the RGB patch kernels into one microscopy intensity channel.
    if backbone.patch_embed.proj.in_channels == 3:
        old = backbone.patch_embed.proj
        new = nn.Conv2d(
            1, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        backbone.patch_embed.proj = new

    if backbone.patch_embed.proj.in_channels != 1:
        raise RuntimeError("Failed to convert DINOv2 patch embedding to one channel.")

    print("DINOv2 input channels: 1 (true grayscale)")
    return backbone
def cosine_value(
    start: float,
    end: float,
    progress: float,
) -> float:
    progress = min(max(float(progress), 0.0), 1.0)

    return end + 0.5 * (start - end) * (
        1.0 + math.cos(math.pi * progress)
    )


def teacher_temperature(
    epoch_index: int,
    warmup_epochs: int,
    warmup_start: float,
    target: float,
) -> float:
    if warmup_epochs <= 0:
        return target

    if epoch_index >= warmup_epochs:
        return target

    alpha = epoch_index / max(1, warmup_epochs - 1)

    return (
        warmup_start
        + alpha * (target - warmup_start)
    )


def set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    backbone_lr: float,
    head_lr: float,
) -> None:
    optimizer.param_groups[0]["lr"] = backbone_lr
    optimizer.param_groups[1]["lr"] = head_lr


# ---------------------------------------------------------------------
# Representation visualization
# ---------------------------------------------------------------------

def _prepare_probe_tensor(
    record: ImageRecord,
    stack_mode: str,
    size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Deterministically prepare one fixed microscopy probe image.

    For a 3D stack, visualization deliberately uses the middle slice so that
    the SAME biological plane is followed across epochs. Training can still
    use random_slice independently.
    """
    probe_stack_mode = (
        "middle_slice"
        if len(record.shape) >= 3
        else stack_mode
    )

    image = _read_standard_2d(
        record,
        stack_mode=probe_stack_mode,
    )
    x = percentile_normalize(image)

    x = TF.resize(
        x,
        [size, size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    x = torch.clamp(x, 0.0, 1.0)

    original = x.squeeze(0).cpu().numpy()

    tensor = x
    gray_mean = (0.485 + 0.456 + 0.406) / 3.0
    gray_std = (0.229 + 0.224 + 0.225) / 3.0
    tensor = TF.normalize(
        tensor,
        mean=[gray_mean],
        std=[gray_std],
    )

    return original, tensor.unsqueeze(0)


def _extract_patch_tokens(
    backbone: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return normalized DINOv2 patch tokens as [B, N, D]."""
    if not hasattr(backbone, "forward_features"):
        raise RuntimeError(
            "DINOv2 backbone does not expose forward_features()."
        )

    features = backbone.forward_features(x)

    if isinstance(features, dict):
        for key in (
            "x_norm_patchtokens",
            "x_prenorm",
            "patch_tokens",
        ):
            value = features.get(key)

            if torch.is_tensor(value):
                if value.ndim == 3:
                    # x_prenorm may include a CLS token. If the token count is
                    # one larger than a square grid, remove the first token.
                    n = value.shape[1]
                    grid = int(round(math.sqrt(n)))

                    if grid * grid != n:
                        n_without_cls = n - 1
                        grid_without_cls = int(
                            round(math.sqrt(n_without_cls))
                        )

                        if (
                            grid_without_cls
                            * grid_without_cls
                            == n_without_cls
                        ):
                            value = value[:, 1:, :]

                    return value

    raise RuntimeError(
        "Could not find DINOv2 patch tokens in forward_features() output."
    )


def _patch_tokens_to_maps(
    patch_tokens: torch.Tensor,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce two human-readable representation maps:

    1. feature_strength: L2 magnitude of each patch embedding
    2. feature_pca_rgb: first 3 PCA components of patch embeddings
    """
    tokens = (
        patch_tokens[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    n_patches, dim = tokens.shape
    grid = int(round(math.sqrt(n_patches)))

    if grid * grid != n_patches:
        raise ValueError(
            f"Patch-token count is not square: {n_patches}"
        )

    # --------------------------------------------------------------
    # Feature strength
    # --------------------------------------------------------------
    strength = np.linalg.norm(
        tokens,
        axis=1,
    ).reshape(grid, grid)

    lo = float(np.percentile(strength, 2.0))
    hi = float(np.percentile(strength, 98.0))

    if hi <= lo + 1e-8:
        hi = lo + 1.0

    strength = np.clip(
        (strength - lo) / (hi - lo),
        0.0,
        1.0,
    )

    # --------------------------------------------------------------
    # PCA RGB of patch representation
    # --------------------------------------------------------------
    centered = tokens - tokens.mean(
        axis=0,
        keepdims=True,
    )

    # SVD on N x D is small for ViT-S/14 probe maps.
    u, s, _ = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = u[:, :3] * s[:3]

    # If feature dimension/rank is unexpectedly <3, pad safely.
    if components.shape[1] < 3:
        padded = np.zeros(
            (components.shape[0], 3),
            dtype=np.float32,
        )
        padded[:, :components.shape[1]] = components
        components = padded

    pca_rgb = np.zeros_like(
        components,
        dtype=np.float32,
    )

    for channel in range(3):
        values = components[:, channel]
        c_lo = float(np.percentile(values, 2.0))
        c_hi = float(np.percentile(values, 98.0))

        if c_hi <= c_lo + 1e-8:
            c_hi = c_lo + 1.0

        pca_rgb[:, channel] = np.clip(
            (values - c_lo) / (c_hi - c_lo),
            0.0,
            1.0,
        )

    pca_rgb = pca_rgb.reshape(
        grid,
        grid,
        3,
    )

    # Upsample patch-grid maps to the displayed image size.
    strength_tensor = torch.from_numpy(
        strength
    ).float()[None, None]

    strength_up = F.interpolate(
        strength_tensor,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    pca_tensor = (
        torch.from_numpy(pca_rgb)
        .permute(2, 0, 1)
        .float()[None]
    )

    pca_up = F.interpolate(
        pca_tensor,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).numpy()

    pca_up = np.clip(
        pca_up,
        0.0,
        1.0,
    )

    return strength_up, pca_up


@torch.no_grad()
def save_representation_probe_visualizations(
    backbone: nn.Module,
    probe_records: Sequence[ImageRecord],
    output_dir: Path,
    epoch: int,
    device: torch.device,
    image_size: int,
    stack_mode: str,
) -> None:
    """
    Save fixed microscopy probe images every few epochs.

    Each PNG shows:
        Original | Patch feature strength | Patch feature PCA RGB

    This is the self-supervised analogue of keeping segmentation predictions.
    """
    if not probe_records:
        return

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epoch_dir = (
        Path(output_dir)
        / "representation_visualization"
        / f"epoch_{int(epoch):03d}"
    )
    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    was_training = bool(backbone.training)
    backbone.eval()

    try:
        for index, record in enumerate(
            probe_records,
            1,
        ):
            try:
                original, tensor = _prepare_probe_tensor(
                    record,
                    stack_mode=stack_mode,
                    size=image_size,
                )

                tensor = tensor.to(
                    device,
                    non_blocking=True,
                )

                patch_tokens = _extract_patch_tokens(
                    backbone,
                    tensor,
                )

                strength, pca_rgb = _patch_tokens_to_maps(
                    patch_tokens,
                    output_size=image_size,
                )

                safe_stem = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    record.path.stem,
                )

                png_path = (
                    epoch_dir
                    / f"{index:02d}_{safe_stem}.png"
                )

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(10.8, 3.6),
                )

                axes[0].imshow(
                    original,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[0].set_title("Original")

                axes[1].imshow(
                    original,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[1].imshow(
                    strength,
                    cmap="magma",
                    alpha=0.62,
                    vmin=0,
                    vmax=1,
                )
                axes[1].set_title(
                    "Patch feature strength"
                )

                axes[2].imshow(
                    pca_rgb
                )
                axes[2].set_title(
                    "Patch feature PCA RGB"
                )

                for ax in axes:
                    ax.axis("off")

                fig.suptitle(
                    f"Epoch {int(epoch)} · {record.path.name}",
                    fontsize=10,
                )

                fig.tight_layout()
                fig.savefig(
                    png_path,
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)

            except Exception as exc:
                print(
                    f"[VISUALIZATION SKIP] {record.path.name}: {exc}"
                )

    finally:
        if was_training:
            backbone.train()


# ---------------------------------------------------------------------
# Fixed-image nearest-neighbor retrieval
# ---------------------------------------------------------------------

@torch.no_grad()
def _extract_cls_embedding_for_record(
    backbone: nn.Module,
    record: ImageRecord,
    device: torch.device,
    image_size: int,
    stack_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic CLS embedding + display image for nearest-neighbor retrieval.

    Z stacks use the middle slice so retrieval comparisons remain fixed across
    epochs even when SSL training itself uses random Z slices.
    """
    original, tensor = _prepare_probe_tensor(
        record,
        stack_mode=stack_mode,
        size=image_size,
    )

    tensor = tensor.to(
        device,
        non_blocking=True,
    )

    feature = _extract_backbone_feature(
        backbone,
        tensor,
    )

    embedding = (
        F.normalize(
            feature,
            dim=-1,
            p=2,
        )[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return original, embedding


def _build_fixed_retrieval_sets(
    records: Sequence[ImageRecord],
    query_count: int,
    pool_count: int,
    seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Create fixed query and fixed candidate pools.

    We use a separate deterministic RNG from training order so the retrieval
    probes do not change if DataLoader shuffling changes.
    """
    records = list(records)

    if not records:
        return [], []

    rng = random.Random(
        int(seed) + 104729
    )

    indices = list(
        range(len(records))
    )
    rng.shuffle(indices)

    q_count = min(
        int(query_count),
        len(indices),
    )

    query_indices = indices[:q_count]

    if pool_count <= 0:
        pool_indices = indices
    else:
        pool_indices = indices[
            : min(
                len(indices),
                max(
                    int(pool_count),
                    q_count,
                ),
            )
        ]

    queries = [
        records[i]
        for i in query_indices
    ]

    pool = [
        records[i]
        for i in pool_indices
    ]

    return queries, pool


def save_retrieval_manifest(
    path: Path,
    query_records: Sequence[ImageRecord],
    pool_records: Sequence[ImageRecord],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_paths = {
        str(record.path)
        for record in query_records
    }

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "role",
                "path",
                "shape",
                "axes",
            ],
        )
        writer.writeheader()

        for record in pool_records:
            writer.writerow(
                {
                    "role": (
                        "query+pool"
                        if str(record.path) in query_paths
                        else "pool"
                    ),
                    "path": str(record.path),
                    "shape": "x".join(
                        str(v)
                        for v in record.shape
                    ),
                    "axes": record.axes,
                }
            )


@torch.no_grad()
def save_fixed_nearest_neighbor_retrieval(
    backbone: nn.Module,
    query_records: Sequence[ImageRecord],
    pool_records: Sequence[ImageRecord],
    output_dir: Path,
    epoch: int,
    device: torch.device,
    image_size: int,
    stack_mode: str,
    top_k: int,
) -> None:
    """
    Save nearest-neighbor retrieval contact sheets using DINO CLS embeddings.

    Each row:
        fixed Query | NN1 | NN2 | ... | NN-K

    The same query files and candidate pool are reused across epochs.
    """
    if not query_records or not pool_records:
        return

    import matplotlib
    matplotlib.use(
        "Agg",
        force=True,
    )
    import matplotlib.pyplot as plt

    was_training = bool(
        backbone.training
    )
    backbone.eval()

    epoch_dir = (
        Path(output_dir)
        / "nearest_neighbor_retrieval"
        / f"epoch_{int(epoch):03d}"
    )
    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        print(
            f"  NN retrieval: embedding fixed pool "
            f"({len(pool_records)} image(s))..."
        )

        pool_embeddings = []
        pool_images = []

        for index, record in enumerate(
            pool_records,
            1,
        ):
            try:
                original, embedding = (
                    _extract_cls_embedding_for_record(
                        backbone,
                        record,
                        device=device,
                        image_size=image_size,
                        stack_mode=stack_mode,
                    )
                )

                pool_images.append(
                    original
                )
                pool_embeddings.append(
                    embedding
                )

            except Exception as exc:
                print(
                    f"[NN POOL SKIP] {record.path.name}: {exc}"
                )
                pool_images.append(None)
                pool_embeddings.append(None)

            if index % 250 == 0:
                print(
                    f"    pool {index}/{len(pool_records)}"
                )

        valid_indices = [
            i
            for i, embedding in enumerate(
                pool_embeddings
            )
            if embedding is not None
        ]

        if not valid_indices:
            print(
                "[NN RETRIEVAL] No valid pool embeddings."
            )
            return

        valid_matrix = np.stack(
            [
                pool_embeddings[i]
                for i in valid_indices
            ],
            axis=0,
        ).astype(
            np.float32
        )

        # Already normalized, but normalize again for numerical safety.
        valid_matrix /= np.maximum(
            np.linalg.norm(
                valid_matrix,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        valid_records = [
            pool_records[i]
            for i in valid_indices
        ]
        valid_images = [
            pool_images[i]
            for i in valid_indices
        ]

        summary_rows = []

        for query_index, query_record in enumerate(
            query_records,
            1,
        ):
            try:
                query_image, query_embedding = (
                    _extract_cls_embedding_for_record(
                        backbone,
                        query_record,
                        device=device,
                        image_size=image_size,
                        stack_mode=stack_mode,
                    )
                )

                query_embedding = (
                    query_embedding
                    / max(
                        float(
                            np.linalg.norm(
                                query_embedding
                            )
                        ),
                        1e-12,
                    )
                )

                similarities = (
                    valid_matrix
                    @ query_embedding.astype(
                        np.float32
                    )
                )

                # Exclude query itself from its retrieval list if present.
                for i, record in enumerate(
                    valid_records
                ):
                    if (
                        record.path.resolve()
                        == query_record.path.resolve()
                    ):
                        similarities[i] = -np.inf

                order = np.argsort(
                    -similarities
                )

                neighbor_indices = [
                    int(i)
                    for i in order
                    if np.isfinite(
                        similarities[i]
                    )
                ][
                    : min(
                        int(top_k),
                        max(
                            0,
                            len(valid_records) - 1,
                        ),
                    )
                ]

                columns = 1 + len(
                    neighbor_indices
                )

                fig, axes = plt.subplots(
                    1,
                    columns,
                    figsize=(
                        max(
                            3.0,
                            2.25 * columns,
                        ),
                        2.8,
                    ),
                    squeeze=False,
                )
                axes = axes[0]

                axes[0].imshow(
                    query_image,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[0].set_title(
                    "Query",
                    fontsize=9,
                )
                axes[0].axis("off")

                for rank, pool_index in enumerate(
                    neighbor_indices,
                    1,
                ):
                    ax = axes[rank]
                    ax.imshow(
                        valid_images[pool_index],
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                    score = float(
                        similarities[
                            pool_index
                        ]
                    )
                    ax.set_title(
                        f"NN{rank}\ncos={score:.3f}",
                        fontsize=8,
                    )
                    ax.axis("off")

                    summary_rows.append(
                        {
                            "epoch": int(epoch),
                            "query": str(
                                query_record.path
                            ),
                            "rank": rank,
                            "neighbor": str(
                                valid_records[
                                    pool_index
                                ].path
                            ),
                            "cosine_similarity": score,
                        }
                    )

                safe_stem = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    query_record.path.stem,
                )

                fig.suptitle(
                    (
                        f"Epoch {int(epoch)} · "
                        f"{query_record.path.name}"
                    ),
                    fontsize=10,
                )
                fig.tight_layout()

                fig.savefig(
                    epoch_dir
                    / (
                        f"query_{query_index:02d}_"
                        f"{safe_stem}.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)

            except Exception as exc:
                print(
                    f"[NN QUERY SKIP] "
                    f"{query_record.path.name}: {exc}"
                )

        if summary_rows:
            with open(
                epoch_dir
                / "nearest_neighbors.csv",
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "query",
                        "rank",
                        "neighbor",
                        "cosine_similarity",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    summary_rows
                )

    finally:
        if was_training:
            backbone.train()


def make_epoch_model_filename(epoch: int, backbone_name: str = "dinov2_vits14") -> str:
    """
    Permanent saved-model naming requested by the user.

    Example:
        train_dinov2_micro_epoch005_20260904_145512.pth
    """
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_backbone = str(backbone_name).replace("/", "_").replace("\\", "_")
    return (
        f"{safe_backbone}_microscopy_"
        f"epoch{int(epoch):03d}_"
        f"{timestamp}.pth"
    )


# ---------------------------------------------------------------------
# Checkpoint / logs
# ---------------------------------------------------------------------

def save_encoder(
    path: Path,
    backbone: nn.Module,
    backbone_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_type": "LinkNex_microscopy_DINOv2_encoder",
            "backbone_name": str(backbone_name),
            "input_channels": 1,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in backbone.state_dict().items()
            },
        },
        path,
    )


def save_ssl_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: DINOStudentTeacher,
    loss_fn: DINOLoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    args,
    history: List[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "student_backbone": model.student_backbone.state_dict(),
            "student_head": model.student_head.state_dict(),
            "teacher_backbone": model.teacher_backbone.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "dino_center": loss_fn.center.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "history": history,
        },
        path,
    )


def load_ssl_checkpoint(
    path: Path,
    model: DINOStudentTeacher,
    loss_fn: DINOLoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
) -> Tuple[int, int, List[Dict[str, object]]]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    required_keys = {
        "student_backbone",
        "student_head",
        "teacher_backbone",
        "teacher_head",
        "dino_center",
        "optimizer",
    }

    if (
        not isinstance(checkpoint, dict)
        or not required_keys.issubset(
            checkpoint.keys()
        )
    ):
        raise ValueError(
            "This file is NOT a full resumable DINO SSL checkpoint.\n\n"
            f"Selected file:\n{path}\n\n"
            "To continue self-supervised training, choose a file from:\n"
            "OUTPUT_DIR\\checkpoints\\train_dinov2_micro_epochXXX_YYYYMMDD_HHMMSS.pth\n"
            "or use RESUME_MODE = 'latest'.\n\n"
            "Files under OUTPUT_DIR\\encoders\\ contain only the DINO encoder "
            "and are intended for LinkNex supervised segmentation."
        )

    model.student_backbone.load_state_dict(
        checkpoint["student_backbone"]
    )
    model.student_head.load_state_dict(
        checkpoint["student_head"]
    )
    model.teacher_backbone.load_state_dict(
        checkpoint["teacher_backbone"]
    )
    model.teacher_head.load_state_dict(
        checkpoint["teacher_head"]
    )

    loss_fn.center.copy_(
        checkpoint["dino_center"].to(
            loss_fn.center.device
        )
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    if "scaler" in checkpoint:
        try:
            scaler.load_state_dict(
                checkpoint["scaler"]
            )
        except Exception:
            pass

    start_epoch = int(
        checkpoint.get("epoch", 0)
    )
    global_step = int(
        checkpoint.get("global_step", 0)
    )
    history = list(
        checkpoint.get("history", [])
    )

    print(
        f"Resumed SSL checkpoint from epoch {start_epoch}, "
        f"global_step={global_step}"
    )

    return start_epoch, global_step, history


def write_history_csv(
    path: Path,
    history: Sequence[Dict[str, object]],
) -> None:
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(history[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(
    path: Path,
    history: Sequence[Dict[str, object]],
) -> None:
    if not history:
        return

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = [
        int(row["epoch"])
        for row in history
    ]

    losses = [
        float(row["ssl_loss"])
        for row in history
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        epochs,
        losses,
        marker="o",
        label="DINO SSL loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Microscopy DINO self-supervised training"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------

def resolve_resume_checkpoint(
    args,
    checkpoint_dir: Path,
) -> Optional[Path]:
    """
    Resolve the checkpoint used to CONTINUE DINO SSL.

    Only FULL ssl_checkpoint_*.pth files are valid for exact continuation.
    Encoder-only files under encoders/ are intentionally rejected later.
    """
    mode = str(
        getattr(
            args,
            "resume_mode",
            "none",
        )
    ).strip().lower()

    if mode == "none":
        return None

    if mode == "latest":
        candidate = (
            checkpoint_dir
            / "ssl_checkpoint_latest.pth"
        )
    elif mode == "path":
        candidate = Path(
            str(args.resume)
        ).expanduser()
    else:
        raise ValueError(
            f"Unknown resume mode: {mode}"
        )

    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(
            "Resume SSL checkpoint does not exist:\n"
            f"{candidate}\n\n"
            "For RESUME_MODE = 'latest', the expected file is:\n"
            f"{checkpoint_dir / 'ssl_checkpoint_latest.pth'}"
        )

    return candidate


def write_resume_instructions(
    output_dir: Path,
    save_every: int,
    backbone_name: str,
) -> Path:
    """
    Write a plain-text guide with the exact resume/output folders.
    """
    output_dir = Path(output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    encoder_dir = output_dir / "encoders"

    path = (
        output_dir
        / "HOW_TO_RESUME_DINO_SSL.txt"
    )

    text = f"""DINO MICROSCOPY SSL - SAVE / RESUME GUIDE
============================================================

OUTPUT ROOT
-----------
{output_dir}

FULL SSL CHECKPOINTS - USE THESE TO CONTINUE SELF-SUPERVISED TRAINING
---------------------------------------------------------------------
Folder:
{checkpoint_dir}

Every {int(save_every)} epochs:
{checkpoint_dir}\\{backbone_name}_microscopy_epoch005_YYYYMMDD_HHMMSS.pth
{checkpoint_dir}\\{backbone_name}_microscopy_epoch010_YYYYMMDD_HHMMSS.pth
{checkpoint_dir}\\{backbone_name}_microscopy_epoch015_YYYYMMDD_HHMMSS.pth
...

Also updated after EVERY completed epoch:
{checkpoint_dir}\\ssl_checkpoint_latest.pth

NOTE:
ssl_checkpoint_latest.pth is only the automatic RESUME pointer.
Permanent user-facing models use:
train_dinov2_micro_epochXXX_YYYYMMDD_HHMMSS.pth


ENCODER-ONLY MODELS - DO NOT USE THESE TO RESUME DINO SSL
----------------------------------------------------------
Folder:
{encoder_dir}

Every {int(save_every)} epochs:
{encoder_dir}\\{backbone_name}_microscopy_epoch005_YYYYMMDD_HHMMSS.pth
{encoder_dir}\\{backbone_name}_microscopy_epoch010_YYYYMMDD_HHMMSS.pth
...

These encoder-only .pth files are for:
LinkNex -> Module 2 -> Pretrained Encoder


HOW TO CONTINUE AFTER STOPPING
------------------------------

Option A - easiest: continue from the latest completed epoch

    RESUME_MODE = "latest"
    RESUME_CHECKPOINT = ""

The script automatically loads:

    {checkpoint_dir}\\ssl_checkpoint_latest.pth


Option B - continue from a specific saved epoch

    RESUME_MODE = "path"
    RESUME_CHECKPOINT = r"{checkpoint_dir}\\{backbone_name}_microscopy_epoch020_YYYYMMDD_HHMMSS.pth"


EPOCHS MEANS FINAL TOTAL EPOCH
------------------------------

Example:
Previous checkpoint = epoch 20
EPOCHS = 60

The restarted run continues:
epoch 21 -> epoch 60

It does NOT run another 60 epochs.


START A NEW SSL RUN
-------------------

    RESUME_MODE = "none"

This starts again from INIT_CHECKPOINT.
"""

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(args) -> None:
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    encoder_dir = output_dir / "encoders"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    encoder_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume_guide = write_resume_instructions(
        output_dir,
        save_every=args.save_every,
        backbone_name=args.backbone_name,
    )

    resume_path = resolve_resume_checkpoint(
        args,
        checkpoint_dir,
    )

    print("\nSave / resume locations")
    print("-----------------------")
    print("Full SSL checkpoints :", checkpoint_dir)
    print("Encoder-only models  :", encoder_dir)
    print(
        "Latest resumable SSL :",
        checkpoint_dir / "ssl_checkpoint_latest.pth",
    )
    print("Resume guide         :", resume_guide)

    if resume_path is not None:
        print("RESUMING FROM        :", resume_path)
    else:
        print(
            "RESUMING FROM        : None "
            "(new SSL run)"
        )

    all_records = discover_images(
        Path(args.data_dir),
        exclude_keywords=args.exclude_keywords,
        max_files=0,
    )

    records = select_training_records(
        all_records,
        count=args.train_image_count,
        seed=args.seed,
    )

    training_manifest = (
        output_dir
        / "ssl_training_files.csv"
    )
    save_training_manifest(
        training_manifest,
        records,
    )

    print("\nDataset summary")
    print("---------------")
    print("Discovered files     :", len(all_records))
    print("Selected train files :", len(records))
    print("Training manifest    :", training_manifest)

    stack_count = sum(
        1
        for record in records
        if record.is_tiff
        and len(record.shape) >= 3
    )

    print("3D TIFFs   :", stack_count)
    print("Stack mode :", args.tiff_stack_mode)

    # Fixed probe set for representation visualization. Keep the same files
    # across all epochs so representation evolution can be compared directly.
    probe_records = list(
        records[: min(
            len(records),
            args.visualize_probe_images,
        )]
    )

    if args.save_representation_visualization:
        print(
            "Representation probes:",
            len(probe_records),
            f"fixed image(s), every {args.visualize_every} epoch(s)",
        )

    (
        retrieval_query_records,
        retrieval_pool_records,
    ) = _build_fixed_retrieval_sets(
        records,
        query_count=args.nearest_neighbor_query_images,
        pool_count=args.nearest_neighbor_pool_images,
        seed=args.seed,
    )

    retrieval_manifest = (
        output_dir
        / "nearest_neighbor_retrieval_manifest.csv"
    )

    if args.save_nearest_neighbor_retrieval:
        save_retrieval_manifest(
            retrieval_manifest,
            retrieval_query_records,
            retrieval_pool_records,
        )
        print(
            "NN retrieval:",
            f"{len(retrieval_query_records)} fixed query image(s), "
            f"{len(retrieval_pool_records)} fixed pool image(s)",
        )
        print(
            "NN manifest:",
            retrieval_manifest,
        )

    transform = MicroscopyMultiCrop(
        global_size=args.global_size,
        local_size=args.local_size,
        local_crops=args.local_crops,
    )

    dataset = MicroscopySSLDataset(
        records,
        stack_mode=args.tiff_stack_mode,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if len(loader) == 0:
        raise RuntimeError(
            "No complete batch can be formed. "
            "Reduce --batch-size."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and not args.cpu
        else "cpu"
    )

    print("\nDevice:", device)

    init_checkpoint = (
        None
        if str(args.init_checkpoint).lower() in {
            "", "none", "null"
        }
        else Path(args.init_checkpoint)
    )

    backbone = build_dinov2_backbone(
        Path(args.dinov2_repo),
        init_checkpoint,
        backbone_name=args.backbone_name,
    )

    model = DINOStudentTeacher(
        backbone=backbone,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    loss_fn = DINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.student_backbone.parameters(),
                "lr": args.backbone_lr,
            },
            {
                "params": model.student_head.parameters(),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )

    start_epoch = 0
    global_step = 0
    history: List[Dict[str, object]] = []

    if resume_path is not None:
        (
            start_epoch,
            global_step,
            history,
        ) = load_ssl_checkpoint(
            resume_path,
            model,
            loss_fn,
            optimizer,
            scaler,
            device,
        )

        if start_epoch >= args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch}, "
                f"but EPOCHS={args.epochs}.\n"
                "EPOCHS is the FINAL TOTAL epoch number. "
                "Set EPOCHS to a value greater than the resumed epoch."
            )

        print(
            f"Continue range       : "
            f"epoch {start_epoch + 1} -> epoch {args.epochs}"
        )

    steps_per_epoch = math.ceil(
        len(loader) / args.grad_accum
    )

    total_optimizer_steps = max(
        1,
        args.epochs * steps_per_epoch,
    )

    warmup_optimizer_steps = (
        args.warmup_epochs
        * steps_per_epoch
    )

    print("\nTraining")
    print("--------")
    print("Epochs               :", args.epochs)
    print("Files/epoch          :", len(dataset))
    print("Batch size           :", args.batch_size)
    print("Grad accumulation    :", args.grad_accum)
    print(
        "Effective batch      :",
        args.batch_size * args.grad_accum,
    )
    print("Global crops/sample  : 2")
    print(
        "Local crops/sample   :",
        args.local_crops,
    )
    print(
        "Backbone LR          :",
        args.backbone_lr,
    )
    print("Head LR              :", args.head_lr)
    print("Teacher momentum     :", args.teacher_momentum)
    print("AMP                   :", amp_enabled)
    print("Output                :", output_dir)

    history_csv = (
        output_dir
        / "ssl_training_log.csv"
    )
    loss_png = (
        output_dir
        / "ssl_loss_curve.png"
    )

    # Epoch 0 = representation BEFORE microscopy SSL adaptation.
    # This gives a direct visual baseline against later epochs.
    if (
        args.save_representation_visualization
        and start_epoch == 0
    ):
        print(
            "\nSaving epoch-000 pretrained representation probes..."
        )
        save_representation_probe_visualizations(
            model.student_backbone,
            probe_records,
            output_dir,
            epoch=0,
            device=device,
            image_size=args.global_size,
            stack_mode=args.tiff_stack_mode,
        )

    if (
        args.save_nearest_neighbor_retrieval
        and start_epoch == 0
    ):
        print(
            "\nSaving epoch-000 pretrained nearest-neighbor retrieval..."
        )
        save_fixed_nearest_neighbor_retrieval(
            model.student_backbone,
            retrieval_query_records,
            retrieval_pool_records,
            output_dir,
            epoch=0,
            device=device,
            image_size=args.global_size,
            stack_mode=args.tiff_stack_mode,
            top_k=args.nearest_neighbor_top_k,
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch_index in range(
        start_epoch,
        args.epochs,
    ):
        epoch_number = epoch_index + 1
        epoch_started = time.perf_counter()

        model.student_backbone.train()
        model.student_head.train()
        model.teacher_backbone.eval()
        model.teacher_head.eval()

        running_loss = 0.0
        batch_count = 0

        teacher_temp = teacher_temperature(
            epoch_index,
            args.teacher_temp_warmup_epochs,
            args.teacher_temp_warmup_start,
            args.teacher_temp,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        for batch_index, batch in enumerate(
            loader
        ):
            views = [
                view.to(
                    device,
                    non_blocking=True,
                )
                for view in batch["views"]
            ]

            # Schedule is based on optimizer-step progress.
            optimizer_step_index = min(
                global_step,
                total_optimizer_steps - 1,
            )

            progress = (
                optimizer_step_index
                / max(
                    1,
                    total_optimizer_steps - 1,
                )
            )

            if (
                warmup_optimizer_steps > 0
                and optimizer_step_index
                < warmup_optimizer_steps
            ):
                warmup_alpha = (
                    (optimizer_step_index + 1)
                    / warmup_optimizer_steps
                )

                backbone_lr = (
                    args.backbone_lr
                    * warmup_alpha
                )
                head_lr = (
                    args.head_lr
                    * warmup_alpha
                )

            else:
                post_warmup_step = max(
                    0,
                    optimizer_step_index
                    - warmup_optimizer_steps,
                )

                post_warmup_total = max(
                    1,
                    total_optimizer_steps
                    - warmup_optimizer_steps,
                )

                post_progress = (
                    post_warmup_step
                    / post_warmup_total
                )

                backbone_lr = cosine_value(
                    args.backbone_lr,
                    args.min_backbone_lr,
                    post_progress,
                )

                head_lr = cosine_value(
                    args.head_lr,
                    args.min_head_lr,
                    post_progress,
                )

            set_optimizer_lrs(
                optimizer,
                backbone_lr,
                head_lr,
            )

            teacher_momentum = cosine_value(
                args.teacher_momentum,
                1.0,
                progress,
            )

            with torch.no_grad():
                teacher_outputs = (
                    model.teacher_forward(
                        views[:2]
                    )
                )

            try:
                autocast_context = torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                )
            except Exception:
                autocast_context = (
                    torch.cuda.amp.autocast(
                        enabled=amp_enabled
                    )
                )

            with autocast_context:
                student_outputs = (
                    model.student_forward(
                        views
                    )
                )

                loss = loss_fn(
                    student_outputs,
                    teacher_outputs,
                    teacher_temp=teacher_temp,
                )

                scaled_loss = (
                    loss
                    / args.grad_accum
                )

            scaler.scale(
                scaled_loss
            ).backward()

            running_loss += float(
                loss.detach().cpu()
            )
            batch_count += 1

            do_step = (
                (batch_index + 1)
                % args.grad_accum
                == 0
                or (batch_index + 1)
                == len(loader)
            )

            if do_step:
                scaler.unscale_(optimizer)

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(
                            model.student_backbone.parameters()
                        )
                        + list(
                            model.student_head.parameters()
                        ),
                        max_norm=args.grad_clip,
                    )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                model.ema_update(
                    teacher_momentum
                )

                global_step += 1

            loss_fn.update_center(
                teacher_outputs
            )

            if (
                batch_index == 0
                or (batch_index + 1)
                % args.print_every
                == 0
            ):
                mean_so_far = (
                    running_loss
                    / batch_count
                )

                print(
                    f"Epoch {epoch_number:03d}/{args.epochs:03d} "
                    f"batch {batch_index + 1:05d}/{len(loader):05d} "
                    f"loss={mean_so_far:.5f} "
                    f"lr_backbone={backbone_lr:.2e} "
                    f"m={teacher_momentum:.6f}"
                )

        mean_loss = (
            running_loss
            / max(1, batch_count)
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
        )

        row = {
            "epoch": epoch_number,
            "ssl_loss": mean_loss,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "teacher_temp": teacher_temp,
            "teacher_momentum": teacher_momentum,
            "center_norm": float(
                loss_fn.center.norm().detach().cpu()
            ),
            "epoch_seconds": epoch_seconds,
            "files": len(dataset),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": (
                args.batch_size
                * args.grad_accum
            ),
        }

        history.append(row)

        write_history_csv(
            history_csv,
            history,
        )
        save_loss_curve(
            loss_png,
            history,
        )

        if (
            args.save_representation_visualization
            and (
                epoch_number % args.visualize_every == 0
                or epoch_number == args.epochs
            )
        ):
            print(
                f"Saving representation probes for epoch {epoch_number}..."
            )
            save_representation_probe_visualizations(
                model.student_backbone,
                probe_records,
                output_dir,
                epoch=epoch_number,
                device=device,
                image_size=args.global_size,
                stack_mode=args.tiff_stack_mode,
            )

        if (
            args.save_nearest_neighbor_retrieval
            and (
                epoch_number
                % args.nearest_neighbor_every
                == 0
                or epoch_number == args.epochs
            )
        ):
            print(
                f"Saving fixed nearest-neighbor retrieval "
                f"for epoch {epoch_number}..."
            )
            save_fixed_nearest_neighbor_retrieval(
                model.student_backbone,
                retrieval_query_records,
                retrieval_pool_records,
                output_dir,
                epoch=epoch_number,
                device=device,
                image_size=args.global_size,
                stack_mode=args.tiff_stack_mode,
                top_k=args.nearest_neighbor_top_k,
            )

        latest_checkpoint = (
            checkpoint_dir
            / "ssl_checkpoint_latest.pth"
        )

        save_ssl_checkpoint(
            latest_checkpoint,
            epoch=epoch_number,
            global_step=global_step,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            history=history,
        )

        if (
            epoch_number
            % args.save_every
            == 0
            or epoch_number
            == args.epochs
        ):
            permanent_model_name = (
                make_epoch_model_filename(
                    epoch_number,
                    args.backbone_name,
                )
            )

            checkpoint_path = (
                checkpoint_dir
                / permanent_model_name
            )

            save_ssl_checkpoint(
                checkpoint_path,
                epoch=epoch_number,
                global_step=global_step,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                history=history,
            )

            encoder_path = (
                encoder_dir
                / permanent_model_name
            )

            # The student backbone is what we will fine-tune for supervised
            # segmentation.
            save_encoder(
                encoder_path,
                model.student_backbone,
                args.backbone_name,
            )

            print(
                "Saved resumable SSL checkpoint:"
            )
            print(
                f"  {checkpoint_path}"
            )
            print(
                "Saved encoder-only model:"
            )
            print(
                f"  {encoder_path}"
            )

        print(
            f"\nEpoch {epoch_number:03d} complete · "
            f"SSL loss={mean_loss:.5f} · "
            f"time={epoch_seconds / 60.0:.1f} min\n"
        )

    print("=" * 70)
    print("SELF-SUPERVISED TRAINING COMPLETE")
    print("=" * 70)
    print("Training log :", history_csv)
    print("Loss curve   :", loss_png)
    print(
        "Representation:",
        output_dir / "representation_visualization",
    )
    print(
        "NN retrieval  :",
        output_dir / "nearest_neighbor_retrieval",
    )
    print(
        "Train manifest:",
        output_dir / "ssl_training_files.csv",
    )
    print(
        "Final encoder folder:",
        encoder_dir,
    )
    print("Latest SSL ckpt:", checkpoint_dir / "ssl_checkpoint_latest.pth")
    print("Resume guide   :", resume_guide)
    print()
    print(
        "Use the timestamped encoder from the FINAL epoch in encoders/ "
        "as the Pretrained Encoder for supervised LinkNex segmentation."
    )


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
# Edit ONLY this section, then run this .py directly in PyCharm.
#
# Recommended workflow:
# official DINOv2 ViT-S/14 pretrained encoder
#       -> microscopy self-supervised adaptation
#       -> microscopy-adapted encoder
#       -> LinkNex supervised segmentation
# ---------------------------------------------------------------------

# ===== Paths =====
# Only DATA_DIR and OUTPUT_DIR normally need to be changed.
#
# This script lives in:
#   LinkNex_Project/ssl/train_dinov2_microscopy_ssl.py
#
# Therefore DINOv2 source and official weights are resolved from the
# project root automatically.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = r"D:\your_unlabeled_microscopy_images"

OUTPUT_DIR = r"D:\DINO_SSL_output"

DINOV2_REPO = str(
    PROJECT_ROOT / "dinov2-main"
)

# DINOv2 backbone for microscopy SSL. ViT-S is the practical first run;
# ViT-B is stronger but substantially heavier.
BACKBONE_NAME = "dinov2_vits14"  # or "dinov2_vitb14"

_DINO_WEIGHT_FILES = {
    "dinov2_vits14": "dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
}

INIT_CHECKPOINT = str(
    PROJECT_ROOT
    / "models"
    / _DINO_WEIGHT_FILES[BACKBONE_NAME]
)

# ===== Resume training =====
# Three choices:
#
#   RESUME_MODE = "none"
#       Start a NEW microscopy SSL run from INIT_CHECKPOINT.
#
#   RESUME_MODE = "latest"
#       Automatically continue from:
#       OUTPUT_DIR\checkpoints\ssl_checkpoint_latest.pth
#
#   RESUME_MODE = "path"
#       Continue from the exact checkpoint specified in RESUME_CHECKPOINT.
#
# IMPORTANT:
# To continue DINO self-supervised training, select an SSL checkpoint from:
#
#   OUTPUT_DIR\checkpoints\train_dinov2_micro_epochXXX_YYYYMMDD_HHMMSS.pth
#
# DO NOT select:
#
#   OUTPUT_DIR\encoders\dinov2_vits14_microscopy_epoch_XXX.pth
#
# because encoder files contain ONLY the backbone and are intended for
# LinkNex downstream segmentation, not for resuming teacher/student SSL.
RESUME_MODE = "none"   # "none", "latest", or "path"

# Used only when RESUME_MODE = "path".
# Example:
# RESUME_CHECKPOINT = (
#     r"D:\public_data\DINO_SSL_output"
#     r"\checkpoints\{backbone_name}_microscopy_epoch020_YYYYMMDD_HHMMSS.pth"
# )
RESUME_CHECKPOINT = ""

# ===== Input data =====
# For ZYX TIFF:
#   "random_slice"   = randomly sample one Z slice each time the stack is visited
#   "middle_slice"   = always use middle Z
#   "max_projection" = use Z max projection
TIFF_STACK_MODE = "random_slice"

# Files containing these words are ignored.
EXCLUDE_KEYWORDS = [
    "_mask",
    "mask_",
    "_label",
    "labels",
    "segmentation",
]

# ===== Training-set selection =====
# Number of image FILES/stacks randomly selected from DATA_DIR for this SSL run.
#
# Examples:
#   TRAIN_IMAGE_COUNT = 30000
#       -> randomly select 30,000 files/stacks from everything under DATA_DIR.
#
#   TRAIN_IMAGE_COUNT = 0
#       -> use every discovered file.
#
# The selected list is fixed by SEED and written to:
#   OUTPUT_DIR/ssl_training_files.csv
#
# For a ZYX TIFF stack, one FILE counts as one selected training item.
# During each visit, TIFF_STACK_MODE controls which 2D plane is used.
TRAIN_IMAGE_COUNT = 30000

# ===== Training =====
EPOCHS = 60

# Start with 16 if GPU memory allows.
# If CUDA OOM:
#   BATCH_SIZE = 8
#   GRAD_ACCUM = 4
BATCH_SIZE = 16
GRAD_ACCUM = 2

WORKERS = 4

# DINOv2 ViT-S/14 patch size is 14.
# These must be divisible by 14.
GLOBAL_SIZE = 224
LOCAL_SIZE = 98
LOCAL_CROPS = 4

# ===== DINO projection head =====
OUT_DIM = 8192
HIDDEN_DIM = 2048
BOTTLENECK_DIM = 256

# ===== Optimization =====
# Backbone uses a low LR because it starts from official DINOv2 weights.
BACKBONE_LR = 5e-5
HEAD_LR = 5e-4

MIN_BACKBONE_LR = 2e-6
MIN_HEAD_LR = 2e-5

WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.04

# ===== Teacher / Student =====
TEACHER_MOMENTUM = 0.996
STUDENT_TEMP = 0.1

TEACHER_TEMP = 0.07
TEACHER_TEMP_WARMUP_START = 0.04
TEACHER_TEMP_WARMUP_EPOCHS = 10

CENTER_MOMENTUM = 0.9
GRAD_CLIP = 3.0

# ===== Saving / logging =====
# Every 5 epochs save BOTH:
#   1) a FULL resumable SSL checkpoint in OUTPUT_DIR/checkpoints/
#   2) an encoder-only .pth in OUTPUT_DIR/encoders/
SAVE_EVERY = 5
PRINT_EVERY = 50

# ===== Representation visualization =====
# Save fixed probe-image representations every N epochs.
# These are NOT segmentation predictions. They visualize what the DINO
# encoder has learned from the same fixed microscopy images over training.
VISUALIZE_EVERY = 5
VISUALIZE_PROBE_IMAGES = 6

# Each probe PNG contains:
#   Original | Patch feature strength | Patch feature PCA RGB
SAVE_REPRESENTATION_VISUALIZATION = True

# ===== Fixed-image nearest-neighbor retrieval =====
# The SAME fixed query images and SAME fixed retrieval pool are used at every
# saved epoch, including epoch 000 before microscopy SSL adaptation.
SAVE_NEAREST_NEIGHBOR_RETRIEVAL = True

# Save retrieval panels every N epochs.
NEAREST_NEIGHBOR_EVERY = 5

# Number of fixed query images.
NEAREST_NEIGHBOR_QUERY_IMAGES = 4

# Top-K nearest neighbors shown for each query.
NEAREST_NEIGHBOR_TOP_K = 8

# Fixed candidate pool size used for retrieval.
# 0 = use the entire selected training set.
# For tens of thousands of images, 1000-3000 is usually much faster for
# routine visualization while still being informative.
NEAREST_NEIGHBOR_POOL_IMAGES = 2000

SEED = 42

# Normally keep False.
NO_AMP = False

# Normally keep False. CPU training will be extremely slow.
CPU_ONLY = False


# ---------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------

class Config:
    pass


def build_config():
    """
    Convert the editable constants above into the attribute-style object
    used by the training functions below.
    """
    cfg = Config()

    cfg.data_dir = DATA_DIR
    cfg.output_dir = OUTPUT_DIR
    cfg.dinov2_repo = DINOV2_REPO
    cfg.backbone_name = BACKBONE_NAME
    cfg.init_checkpoint = INIT_CHECKPOINT
    cfg.resume_mode = str(RESUME_MODE).strip().lower()
    cfg.resume = str(RESUME_CHECKPOINT).strip()

    cfg.tiff_stack_mode = TIFF_STACK_MODE
    cfg.exclude_keywords = list(EXCLUDE_KEYWORDS)
    cfg.train_image_count = int(TRAIN_IMAGE_COUNT)

    cfg.epochs = int(EPOCHS)
    cfg.batch_size = int(BATCH_SIZE)
    cfg.grad_accum = int(GRAD_ACCUM)
    cfg.workers = int(WORKERS)

    cfg.global_size = int(GLOBAL_SIZE)
    cfg.local_size = int(LOCAL_SIZE)
    cfg.local_crops = int(LOCAL_CROPS)

    cfg.out_dim = int(OUT_DIM)
    cfg.hidden_dim = int(HIDDEN_DIM)
    cfg.bottleneck_dim = int(BOTTLENECK_DIM)

    cfg.backbone_lr = float(BACKBONE_LR)
    cfg.head_lr = float(HEAD_LR)
    cfg.min_backbone_lr = float(MIN_BACKBONE_LR)
    cfg.min_head_lr = float(MIN_HEAD_LR)

    cfg.warmup_epochs = int(WARMUP_EPOCHS)
    cfg.weight_decay = float(WEIGHT_DECAY)

    cfg.teacher_momentum = float(TEACHER_MOMENTUM)
    cfg.student_temp = float(STUDENT_TEMP)

    cfg.teacher_temp = float(TEACHER_TEMP)
    cfg.teacher_temp_warmup_start = float(
        TEACHER_TEMP_WARMUP_START
    )
    cfg.teacher_temp_warmup_epochs = int(
        TEACHER_TEMP_WARMUP_EPOCHS
    )

    cfg.center_momentum = float(CENTER_MOMENTUM)
    cfg.grad_clip = float(GRAD_CLIP)

    cfg.save_every = int(SAVE_EVERY)
    cfg.print_every = int(PRINT_EVERY)

    cfg.visualize_every = int(VISUALIZE_EVERY)
    cfg.visualize_probe_images = int(VISUALIZE_PROBE_IMAGES)
    cfg.save_representation_visualization = bool(
        SAVE_REPRESENTATION_VISUALIZATION
    )

    cfg.save_nearest_neighbor_retrieval = bool(
        SAVE_NEAREST_NEIGHBOR_RETRIEVAL
    )
    cfg.nearest_neighbor_every = int(
        NEAREST_NEIGHBOR_EVERY
    )
    cfg.nearest_neighbor_query_images = int(
        NEAREST_NEIGHBOR_QUERY_IMAGES
    )
    cfg.nearest_neighbor_top_k = int(
        NEAREST_NEIGHBOR_TOP_K
    )
    cfg.nearest_neighbor_pool_images = int(
        NEAREST_NEIGHBOR_POOL_IMAGES
    )

    cfg.seed = int(SEED)
    cfg.no_amp = bool(NO_AMP)
    cfg.cpu = bool(CPU_ONLY)

    return cfg


def validate_config(cfg):
    if not str(cfg.data_dir).strip():
        raise ValueError("DATA_DIR is empty.")

    if not str(cfg.output_dir).strip():
        raise ValueError("OUTPUT_DIR is empty.")

    if not str(cfg.dinov2_repo).strip():
        raise ValueError("DINOV2_REPO is empty.")

    if not str(cfg.init_checkpoint).strip():
        raise ValueError(
            "INIT_CHECKPOINT is empty. "
            "For this microscopy domain-adaptation workflow, "
            "use the official DINOv2 ViT-S/14 pretrained checkpoint."
        )

    if cfg.resume_mode not in {
        "none",
        "latest",
        "path",
    }:
        raise ValueError(
            'RESUME_MODE must be "none", "latest", or "path".'
        )

    if (
        cfg.resume_mode == "path"
        and not str(cfg.resume).strip()
    ):
        raise ValueError(
            'RESUME_MODE = "path" but RESUME_CHECKPOINT is empty.'
        )

    if cfg.train_image_count < 0:
        raise ValueError("TRAIN_IMAGE_COUNT must be >= 0.")

    if cfg.batch_size < 1:
        raise ValueError("BATCH_SIZE must be >= 1.")

    if cfg.grad_accum < 1:
        raise ValueError("GRAD_ACCUM must be >= 1.")

    if cfg.epochs < 1:
        raise ValueError("EPOCHS must be >= 1.")

    if cfg.local_crops < 0:
        raise ValueError("LOCAL_CROPS must be >= 0.")

    if cfg.visualize_every < 1:
        raise ValueError("VISUALIZE_EVERY must be >= 1.")

    if cfg.visualize_probe_images < 1:
        raise ValueError("VISUALIZE_PROBE_IMAGES must be >= 1.")

    if cfg.nearest_neighbor_every < 1:
        raise ValueError("NEAREST_NEIGHBOR_EVERY must be >= 1.")

    if cfg.nearest_neighbor_query_images < 1:
        raise ValueError("NEAREST_NEIGHBOR_QUERY_IMAGES must be >= 1.")

    if cfg.nearest_neighbor_top_k < 1:
        raise ValueError("NEAREST_NEIGHBOR_TOP_K must be >= 1.")

    if cfg.nearest_neighbor_pool_images < 0:
        raise ValueError("NEAREST_NEIGHBOR_POOL_IMAGES must be >= 0.")

    if cfg.global_size % 14 != 0:
        raise ValueError("GLOBAL_SIZE must be divisible by 14.")

    if cfg.local_size % 14 != 0:
        raise ValueError("LOCAL_SIZE must be divisible by 14.")

    if cfg.tiff_stack_mode not in {
        "random_slice",
        "middle_slice",
        "max_projection",
    }:
        raise ValueError(
            "TIFF_STACK_MODE must be one of: "
            "random_slice, middle_slice, max_projection"
        )


def print_config(cfg):
    print("=" * 72)
    print("DINOv2 MICROSCOPY SELF-SUPERVISED TRAINING")
    print("=" * 72)
    print("Data directory       :", cfg.data_dir)
    print("Output directory     :", cfg.output_dir)
    print("DINOv2 repository    :", cfg.dinov2_repo)
    print("DINOv2 backbone      :", cfg.backbone_name)
    print("Input channels       : 1 (true grayscale)")
    print("Initial checkpoint   :", cfg.init_checkpoint)
    print("Resume mode          :", cfg.resume_mode)

    if cfg.resume_mode == "path":
        print("Resume checkpoint    :", cfg.resume)
    elif cfg.resume_mode == "latest":
        print(
            "Resume checkpoint    :",
            str(
                Path(cfg.output_dir)
                / "checkpoints"
                / "ssl_checkpoint_latest.pth"
            ),
        )
    else:
        print("Resume checkpoint    : None")

    print("TIFF stack mode      :", cfg.tiff_stack_mode)
    print(
        "Training image count :",
        (
            cfg.train_image_count
            if cfg.train_image_count > 0
            else "ALL"
        ),
    )
    print("Epochs               :", cfg.epochs)
    print("Batch size           :", cfg.batch_size)
    print("Gradient accumulation:", cfg.grad_accum)
    print(
        "Effective batch      :",
        cfg.batch_size * cfg.grad_accum,
    )
    print("Workers              :", cfg.workers)
    print("Global crop          :", cfg.global_size)
    print("Local crop           :", cfg.local_size)
    print("Local crops/sample   :", cfg.local_crops)
    print(
        "Representation PNG   :",
        (
            f"every {cfg.visualize_every} epoch(s), "
            f"{cfg.visualize_probe_images} fixed image(s)"
            if cfg.save_representation_visualization
            else "OFF"
        ),
    )
    print(
        "Nearest-neighbor     :",
        (
            f"every {cfg.nearest_neighbor_every} epoch(s), "
            f"{cfg.nearest_neighbor_query_images} query image(s), "
            f"top-{cfg.nearest_neighbor_top_k}, "
            f"pool={cfg.nearest_neighbor_pool_images if cfg.nearest_neighbor_pool_images > 0 else 'ALL'}"
            if cfg.save_nearest_neighbor_retrieval
            else "OFF"
        ),
    )
    print("Backbone LR          :", cfg.backbone_lr)
    print("Head LR              :", cfg.head_lr)
    print("AMP                   :", not cfg.no_amp)
    print("CPU only              :", cfg.cpu)
    print("=" * 72)
    print()


def main():
    cfg = build_config()
    validate_config(cfg)
    print_config(cfg)
    train(cfg)


if __name__ == "__main__":
    main()

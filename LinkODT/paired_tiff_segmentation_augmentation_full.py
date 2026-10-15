# -*- coding: utf-8 -*-
"""
Paired TIFF augmentation for semantic segmentation.

Expected input:
dataset_root/
├─ image/
│  ├─ sample001.tif
│  ├─ sample002.tif
│  └─ ...
└─ mask/
   ├─ sample001_mask.tif
   ├─ sample002_mask.tif
   └─ ...

Output:
dataset_root_augmentation/
├─ image/
├─ mask/
└─ augmentation_manifest.csv

Key rules:
1. All geometric transforms are applied identically to image and mask.
2. Intensity/noise/blur transforms are applied ONLY to image.
3. Masks are always converted back to binary after augmentation.
4. Works with 2D TIFF and Z-stack TIFF whose spatial axes are the last two axes (..., Y, X).
5. All parameters are defined near the top of the script so you can easily change them.

Required:
    pip install numpy scipy tifffile

Optional:
    tkinter is included with standard Python on most Windows installations.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi


# ============================================================
# USER SETTINGS
# ============================================================

# "full" produces many deterministic augmentations.
# "medium" is more conservative.
# "light" is useful for a quick test.
AUGMENTATION_PROFILE = "full"   # "light" / "medium" / "full"

# Include an unchanged copy of each original image/mask pair in output.
INCLUDE_ORIGINAL = True

# Number of additional random-combination augmentations per image.
RANDOM_COMPOSITES_PER_IMAGE = {
    "light": 3,
    "medium": 8,
    "full": 12,
}[AUGMENTATION_PROFILE]

# If True, already-existing output pairs will not be recalculated.
SKIP_EXISTING = True

# Reproducible random seed.
GLOBAL_SEED = 20260911

# Geometric padding mode:
# "reflect" avoids large black borders.
# "constant" uses zeros outside the field of view.
BORDER_MODE = "reflect"

# For constant padding only.
IMAGE_CVAL = 0.0
MASK_CVAL = 0.0

# Binary mask output values:
# None = preserve a sensible positive value based on input mask dtype.
# Set to 255 if you always want mask values 0/255.
FORCE_MASK_FOREGROUND_VALUE = None

# Supported input extensions.
IMAGE_EXTENSIONS = {".tif", ".tiff"}


# -------------------------
# Parameter banks
# -------------------------

PROFILE_PARAMS = {
    "light": {
        "rotations": [-10, 10],
        "shifts": [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)],
        "zooms": [0.90, 1.10],
        "shears": [-0.08, 0.08],
        "elastic": [(0.025, 0.035)],
        "brightness": [0.90, 1.10],
        "contrast": [0.90, 1.10],
        "gamma": [0.85, 1.15],
        "gaussian_noise": [0.015, 0.03],
        "poisson_peak": [32],
        "blur_sigma": [0.7],
        "sharpen_amount": [0.8],
        "salt_pepper": [0.002],
    },
    "medium": {
        "rotations": [-15, -8, 8, 15],
        "shifts": [
            (-0.08, 0), (0.08, 0), (0, -0.08), (0, 0.08),
            (-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05),
        ],
        "zooms": [0.88, 0.94, 1.06, 1.12],
        "shears": [-0.12, -0.06, 0.06, 0.12],
        "elastic": [(0.025, 0.035), (0.04, 0.05), (0.055, 0.06)],
        "brightness": [0.82, 0.92, 1.08, 1.18],
        "contrast": [0.80, 0.92, 1.08, 1.20],
        "gamma": [0.75, 0.88, 1.12, 1.28],
        "gaussian_noise": [0.01, 0.02, 0.04],
        "poisson_peak": [64, 32],
        "blur_sigma": [0.5, 1.0, 1.5],
        "sharpen_amount": [0.5, 1.0],
        "salt_pepper": [0.001, 0.003],
    },
    "full": {
        "rotations": [-20, -15, -10, -5, 5, 10, 15, 20],
        "shifts": [
            (-0.10, 0), (-0.05, 0), (0.05, 0), (0.10, 0),
            (0, -0.10), (0, -0.05), (0, 0.05), (0, 0.10),
            (-0.07, -0.07), (-0.07, 0.07), (0.07, -0.07), (0.07, 0.07),
        ],
        "zooms": [0.85, 0.90, 0.95, 1.05, 1.10, 1.15],
        "shears": [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15],
        "elastic": [
            (0.020, 0.030),
            (0.030, 0.040),
            (0.045, 0.050),
            (0.060, 0.065),
        ],
        "brightness": [0.80, 0.90, 1.10, 1.20],
        "contrast": [0.75, 0.90, 1.10, 1.25],
        "gamma": [0.70, 0.85, 1.15, 1.30],
        "gaussian_noise": [0.01, 0.02, 0.04, 0.06],
        "poisson_peak": [64, 32, 16],
        "blur_sigma": [0.5, 1.0, 1.5],
        "sharpen_amount": [0.5, 1.0, 1.5],
        "salt_pepper": [0.001, 0.003, 0.005],
    },
}

P = PROFILE_PARAMS[AUGMENTATION_PROFILE]


# ============================================================
# Folder selection
# ============================================================

def choose_root_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        folder = filedialog.askdirectory(
            title="Select dataset root folder containing image/ and mask/"
        )
        root.destroy()

        if not folder:
            return None
        return Path(folder)
    except Exception as e:
        print(f"[WARNING] Folder picker failed: {e}")
        typed = input("Please enter dataset root path: ").strip().strip('"')
        return Path(typed) if typed else None


# ============================================================
# Basic helpers
# ============================================================

def normalize_image01(arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Convert an image to float32 approximately in [0, 1].
    Returns normalized array and restoration metadata.
    """
    arr = np.asarray(arr)
    dtype = arr.dtype

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        x = arr.astype(np.float32)
        denom = float(info.max - info.min)
        if denom <= 0:
            x = np.zeros_like(x, dtype=np.float32)
        else:
            x = (x - info.min) / denom
        meta = {
            "dtype": dtype,
            "kind": "integer",
            "min": info.min,
            "max": info.max,
        }
        return np.clip(x, 0, 1), meta

    # Float TIFFs can have arbitrary ranges. Preserve the file's observed range.
    x = arr.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.percentile(finite, 0.1))
        hi = float(np.percentile(finite, 99.9))
        if hi <= lo:
            lo = float(finite.min())
            hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1.0

    x = (x - lo) / (hi - lo)
    meta = {
        "dtype": dtype,
        "kind": "float",
        "lo": lo,
        "hi": hi,
    }
    return np.clip(x, 0, 1), meta


def restore_image_dtype(x: np.ndarray, meta: dict) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)

    if meta["kind"] == "integer":
        lo = meta["min"]
        hi = meta["max"]
        y = x * (hi - lo) + lo
        return np.rint(y).astype(meta["dtype"])

    lo = meta["lo"]
    hi = meta["hi"]
    y = x * (hi - lo) + lo
    return y.astype(meta["dtype"])


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.float32)


def mask_foreground_value(mask: np.ndarray):
    if FORCE_MASK_FOREGROUND_VALUE is not None:
        return FORCE_MASK_FOREGROUND_VALUE

    dtype = mask.dtype
    positive = mask[mask > 0]

    if positive.size > 0:
        val = positive.max()
        # For float binary masks, 1.0 is usually preferable.
        if np.issubdtype(dtype, np.floating):
            return 1.0 if val <= 1.0 else float(val)
        return val.item()

    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1.0


def restore_mask_dtype(binary_mask: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
    fg = mask_foreground_value(original_mask)
    out = (binary_mask > 0.5).astype(np.float32) * fg
    return out.astype(original_mask.dtype)


def spatial_shape(arr: np.ndarray) -> Tuple[int, int]:
    if arr.ndim < 2:
        raise ValueError(f"Expected >=2D image, got shape {arr.shape}")
    return int(arr.shape[-2]), int(arr.shape[-1])


def ensure_same_shape(image: np.ndarray, mask: np.ndarray, image_path: Path):
    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image={image.shape}, mask={mask.shape}"
        )


def save_tiff(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def paired_output_paths(
    out_image_dir: Path,
    out_mask_dir: Path,
    stem: str,
    tag: str,
    extension: str = ".tif",
) -> Tuple[Path, Path]:
    image_out = out_image_dir / f"{stem}__{tag}{extension}"
    mask_out = out_mask_dir / f"{stem}__{tag}_mask{extension}"
    return image_out, mask_out


def already_done(image_out: Path, mask_out: Path) -> bool:
    return SKIP_EXISTING and image_out.exists() and mask_out.exists()


# ============================================================
# Geometric transforms
# ============================================================

def _shift_vector(ndim: int, dy: float, dx: float):
    v = [0.0] * ndim
    v[-2] = dy
    v[-1] = dx
    return v


def geom_flip_h(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=-1).copy()


def geom_flip_v(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=-2).copy()


def geom_flip_hv(x: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(x, axis=-1), axis=-2).copy()


def geom_rot90(x: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(x, k=k, axes=(-2, -1)).copy()


def geom_rotate(x: np.ndarray, angle: float, order: int, cval: float) -> np.ndarray:
    return ndi.rotate(
        x,
        angle=angle,
        axes=(-2, -1),
        reshape=False,
        order=order,
        mode=BORDER_MODE,
        cval=cval,
        prefilter=(order > 1),
    )


def geom_shift(
    x: np.ndarray,
    dy_fraction: float,
    dx_fraction: float,
    order: int,
    cval: float,
) -> np.ndarray:
    h, w = spatial_shape(x)
    dy = dy_fraction * h
    dx = dx_fraction * w
    return ndi.shift(
        x,
        shift=_shift_vector(x.ndim, dy, dx),
        order=order,
        mode=BORDER_MODE,
        cval=cval,
        prefilter=(order > 1),
    )


def _centered_affine(
    x: np.ndarray,
    spatial_matrix_2x2: np.ndarray,
    order: int,
    cval: float,
) -> np.ndarray:
    """
    scipy.ndimage.affine_transform uses output -> input mapping.
    spatial_matrix_2x2 therefore specifies that mapping.
    """
    n = x.ndim
    full_matrix = np.eye(n, dtype=np.float64)
    full_matrix[-2:, -2:] = spatial_matrix_2x2

    shape = np.array(x.shape, dtype=np.float64)
    center = (shape - 1.0) / 2.0

    offset = center - full_matrix @ center

    return ndi.affine_transform(
        x,
        matrix=full_matrix,
        offset=offset,
        output_shape=x.shape,
        order=order,
        mode=BORDER_MODE,
        cval=cval,
        prefilter=(order > 1),
    )


def geom_zoom(x: np.ndarray, factor: float, order: int, cval: float) -> np.ndarray:
    # Output -> input mapping for visual zoom factor:
    # factor > 1 = zoom in, factor < 1 = zoom out.
    s = 1.0 / float(factor)
    m = np.array([[s, 0.0], [0.0, s]], dtype=np.float64)
    return _centered_affine(x, m, order=order, cval=cval)


def geom_shear(x: np.ndarray, shear: float, order: int, cval: float) -> np.ndarray:
    # Moderate x/y shear in output->input coordinates.
    m = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float64)
    return _centered_affine(x, m, order=order, cval=cval)


def _elastic_displacement(
    h: int,
    w: int,
    alpha_fraction: float,
    sigma_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    min_dim = min(h, w)

    alpha = alpha_fraction * min_dim
    sigma = max(1.0, sigma_fraction * min_dim)

    dy = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dx = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)

    dy = ndi.gaussian_filter(dy, sigma=sigma, mode="reflect")
    dx = ndi.gaussian_filter(dx, sigma=sigma, mode="reflect")

    # Normalize so alpha has interpretable strength.
    dy_std = float(dy.std()) + 1e-8
    dx_std = float(dx.std()) + 1e-8
    dy = dy / dy_std * alpha * 0.15
    dx = dx / dx_std * alpha * 0.15

    return dy, dx


def _apply_elastic_2d(
    plane: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    order: int,
    cval: float,
) -> np.ndarray:
    h, w = plane.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.array([yy + dy, xx + dx])
    return ndi.map_coordinates(
        plane,
        coords,
        order=order,
        mode=BORDER_MODE,
        cval=cval,
        prefilter=(order > 1),
    )


def geom_elastic(
    x: np.ndarray,
    alpha_fraction: float,
    sigma_fraction: float,
    seed: int,
    order: int,
    cval: float,
) -> np.ndarray:
    h, w = spatial_shape(x)
    dy, dx = _elastic_displacement(
        h, w, alpha_fraction, sigma_fraction, seed
    )

    if x.ndim == 2:
        return _apply_elastic_2d(x, dy, dx, order, cval)

    # Apply the SAME XY displacement field to every leading plane/channel/Z.
    leading_shape = x.shape[:-2]
    out = np.empty_like(x, dtype=np.float32)

    for idx in np.ndindex(*leading_shape):
        out[idx] = _apply_elastic_2d(x[idx], dy, dx, order, cval)

    return out


# ============================================================
# Image-only intensity transforms
# ============================================================

def img_brightness(x: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(x * factor, 0, 1)


def img_contrast(x: np.ndarray, factor: float) -> np.ndarray:
    # Use global mean to preserve relationships within the image.
    mean = float(np.mean(x))
    return np.clip((x - mean) * factor + mean, 0, 1)


def img_gamma(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(np.power(np.clip(x, 0, 1), gamma), 0, 1)


def img_gaussian_noise(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 1)


def img_poisson_noise(x: np.ndarray, peak: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peak = max(float(peak), 1.0)
    y = rng.poisson(np.clip(x, 0, 1) * peak).astype(np.float32) / peak
    return np.clip(y, 0, 1)


def img_gaussian_blur(x: np.ndarray, sigma_xy: float) -> np.ndarray:
    # Blur spatial axes only. Do not blur across Z/leading axes.
    sigmas = [0.0] * x.ndim
    sigmas[-2] = sigma_xy
    sigmas[-1] = sigma_xy
    return np.clip(ndi.gaussian_filter(x, sigma=sigmas, mode="reflect"), 0, 1)


def img_sharpen(x: np.ndarray, amount: float, blur_sigma: float = 1.0) -> np.ndarray:
    blurred = img_gaussian_blur(x, blur_sigma)
    y = x + amount * (x - blurred)
    return np.clip(y, 0, 1)


def img_salt_pepper(x: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = x.copy()

    # Pixel-wise noise shared across leading axes if this is a stack.
    h, w = spatial_shape(x)
    r = rng.random((h, w))

    salt = r < (amount / 2.0)
    pepper = (r >= (amount / 2.0)) & (r < amount)

    if x.ndim == 2:
        y[salt] = 1.0
        y[pepper] = 0.0
    else:
        # Broadcast to every leading plane.
        y[..., salt] = 1.0
        y[..., pepper] = 0.0

    return y


# ============================================================
# Paired save helper
# ============================================================

def write_augmented_pair(
    image_aug01: np.ndarray,
    mask_aug01: np.ndarray,
    image_meta: dict,
    original_mask: np.ndarray,
    image_out: Path,
    mask_out: Path,
):
    image_saved = restore_image_dtype(image_aug01, image_meta)
    mask_saved = restore_mask_dtype(mask_aug01, original_mask)

    save_tiff(image_out, image_saved)
    save_tiff(mask_out, mask_saved)


# ============================================================
# Random composite augmentation
# ============================================================

def random_composite_pair(
    image01: np.ndarray,
    mask01: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    img = image01.copy()
    msk = mask01.copy()
    ops: List[str] = []

    # Random flips
    if rng.random() < 0.5:
        img = geom_flip_h(img)
        msk = geom_flip_h(msk)
        ops.append("flipH")

    if rng.random() < 0.5:
        img = geom_flip_v(img)
        msk = geom_flip_v(msk)
        ops.append("flipV")

    # Random rotation
    angle = float(rng.uniform(-20.0, 20.0))
    if abs(angle) > 1.0:
        img = geom_rotate(img, angle, order=1, cval=IMAGE_CVAL)
        msk = geom_rotate(msk, angle, order=0, cval=MASK_CVAL)
        ops.append(f"rot{angle:.1f}")

    # Random scale
    factor = float(rng.uniform(0.88, 1.12))
    img = geom_zoom(img, factor, order=1, cval=IMAGE_CVAL)
    msk = geom_zoom(msk, factor, order=0, cval=MASK_CVAL)
    ops.append(f"zoom{factor:.3f}")

    # Random translation
    dy = float(rng.uniform(-0.07, 0.07))
    dx = float(rng.uniform(-0.07, 0.07))
    img = geom_shift(img, dy, dx, order=1, cval=IMAGE_CVAL)
    msk = geom_shift(msk, dy, dx, order=0, cval=MASK_CVAL)
    ops.append(f"shift{dy:.3f}_{dx:.3f}")

    # Random shear
    if rng.random() < 0.5:
        shear = float(rng.uniform(-0.10, 0.10))
        img = geom_shear(img, shear, order=1, cval=IMAGE_CVAL)
        msk = geom_shear(msk, shear, order=0, cval=MASK_CVAL)
        ops.append(f"shear{shear:.3f}")

    # Elastic deformation only some of the time.
    if rng.random() < 0.35:
        alpha_fraction = float(rng.uniform(0.02, 0.05))
        sigma_fraction = float(rng.uniform(0.035, 0.060))
        elastic_seed = int(rng.integers(0, 2**31 - 1))
        img = geom_elastic(
            img, alpha_fraction, sigma_fraction,
            seed=elastic_seed, order=1, cval=IMAGE_CVAL
        )
        msk = geom_elastic(
            msk, alpha_fraction, sigma_fraction,
            seed=elastic_seed, order=0, cval=MASK_CVAL
        )
        ops.append(f"elastic{alpha_fraction:.3f}_{sigma_fraction:.3f}")

    # Image-only intensity transforms.
    brightness = float(rng.uniform(0.90, 1.10))
    contrast = float(rng.uniform(0.88, 1.12))
    gamma = float(rng.uniform(0.82, 1.18))

    img = img_brightness(img, brightness)
    img = img_contrast(img, contrast)
    img = img_gamma(img, gamma)

    ops.extend([
        f"bright{brightness:.3f}",
        f"contrast{contrast:.3f}",
        f"gamma{gamma:.3f}",
    ])

    if rng.random() < 0.65:
        sigma = float(rng.uniform(0.005, 0.030))
        img = img_gaussian_noise(
            img, sigma=sigma,
            seed=int(rng.integers(0, 2**31 - 1))
        )
        ops.append(f"gnoise{sigma:.3f}")

    if rng.random() < 0.25:
        sigma_blur = float(rng.uniform(0.4, 1.1))
        img = img_gaussian_blur(img, sigma_blur)
        ops.append(f"blur{sigma_blur:.2f}")

    if rng.random() < 0.20:
        amount = float(rng.uniform(0.3, 0.9))
        img = img_sharpen(img, amount=amount, blur_sigma=0.8)
        ops.append(f"sharp{amount:.2f}")

    msk = (msk > 0.5).astype(np.float32)

    return np.clip(img, 0, 1), msk, "+".join(ops)


# ============================================================
# Main augmentation for one pair
# ============================================================

def augment_one_pair(
    image_path: Path,
    mask_path: Path,
    out_image_dir: Path,
    out_mask_dir: Path,
    manifest_rows: List[dict],
    pair_index: int,
):
    image = tifffile.imread(str(image_path))
    mask = tifffile.imread(str(mask_path))
    ensure_same_shape(image, mask, image_path)

    image01, image_meta = normalize_image01(image)
    mask01 = mask_to_binary(mask)

    stem = image_path.stem
    ext = ".tif"

    def save_pair(tag: str, img_aug: np.ndarray, msk_aug: np.ndarray, details: str):
        image_out, mask_out = paired_output_paths(
            out_image_dir, out_mask_dir, stem, tag, ext
        )

        if not already_done(image_out, mask_out):
            write_augmented_pair(
                img_aug,
                msk_aug,
                image_meta,
                mask,
                image_out,
                mask_out,
            )

        manifest_rows.append({
            "source_image": image_path.name,
            "source_mask": mask_path.name,
            "augmentation_tag": tag,
            "details": details,
            "output_image": image_out.name,
            "output_mask": mask_out.name,
        })

    # ------------------------------------
    # Original
    # ------------------------------------
    if INCLUDE_ORIGINAL:
        save_pair(
            "original",
            image01,
            mask01,
            "unchanged copy"
        )

    # ------------------------------------
    # Exact flips
    # ------------------------------------
    save_pair("flip_h", geom_flip_h(image01), geom_flip_h(mask01), "horizontal flip")
    save_pair("flip_v", geom_flip_v(image01), geom_flip_v(mask01), "vertical flip")
    save_pair("flip_hv", geom_flip_hv(image01), geom_flip_hv(mask01), "horizontal+vertical flip")

    # ------------------------------------
    # 90-degree rotations
    # ------------------------------------
    for k, deg in [(1, 90), (2, 180), (3, 270)]:
        save_pair(
            f"rot{deg}",
            geom_rot90(image01, k),
            geom_rot90(mask01, k),
            f"rotation {deg} deg"
        )

    # ------------------------------------
    # Small arbitrary rotations
    # ------------------------------------
    for angle in P["rotations"]:
        tag = f"rotate_{angle:+d}deg".replace("+", "p").replace("-", "m")
        save_pair(
            tag,
            geom_rotate(image01, angle, order=1, cval=IMAGE_CVAL),
            geom_rotate(mask01, angle, order=0, cval=MASK_CVAL),
            f"rotation={angle} deg"
        )

    # ------------------------------------
    # XY shifts
    # ------------------------------------
    for i, (dy, dx) in enumerate(P["shifts"], 1):
        save_pair(
            f"shift_{i:02d}",
            geom_shift(image01, dy, dx, order=1, cval=IMAGE_CVAL),
            geom_shift(mask01, dy, dx, order=0, cval=MASK_CVAL),
            f"dy={dy:.3f}*H, dx={dx:.3f}*W"
        )

    # ------------------------------------
    # Zoom
    # ------------------------------------
    for factor in P["zooms"]:
        tag = f"zoom_{factor:.2f}".replace(".", "p")
        save_pair(
            tag,
            geom_zoom(image01, factor, order=1, cval=IMAGE_CVAL),
            geom_zoom(mask01, factor, order=0, cval=MASK_CVAL),
            f"zoom factor={factor:.3f}"
        )

    # ------------------------------------
    # Shear
    # ------------------------------------
    for shear in P["shears"]:
        tag = f"shear_{shear:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        save_pair(
            tag,
            geom_shear(image01, shear, order=1, cval=IMAGE_CVAL),
            geom_shear(mask01, shear, order=0, cval=MASK_CVAL),
            f"shear={shear:.3f}"
        )

    # ------------------------------------
    # Elastic deformation
    # ------------------------------------
    for i, (alpha_fraction, sigma_fraction) in enumerate(P["elastic"], 1):
        seed = GLOBAL_SEED + pair_index * 1000 + i
        save_pair(
            f"elastic_{i:02d}",
            geom_elastic(
                image01, alpha_fraction, sigma_fraction,
                seed=seed, order=1, cval=IMAGE_CVAL
            ),
            geom_elastic(
                mask01, alpha_fraction, sigma_fraction,
                seed=seed, order=0, cval=MASK_CVAL
            ),
            f"alpha_fraction={alpha_fraction:.4f}, sigma_fraction={sigma_fraction:.4f}, seed={seed}"
        )

    # ------------------------------------
    # Image-only brightness
    # ------------------------------------
    for factor in P["brightness"]:
        tag = f"brightness_{factor:.2f}".replace(".", "p")
        save_pair(
            tag,
            img_brightness(image01, factor),
            mask01,
            f"image-only brightness factor={factor:.3f}"
        )

    # ------------------------------------
    # Image-only contrast
    # ------------------------------------
    for factor in P["contrast"]:
        tag = f"contrast_{factor:.2f}".replace(".", "p")
        save_pair(
            tag,
            img_contrast(image01, factor),
            mask01,
            f"image-only contrast factor={factor:.3f}"
        )

    # ------------------------------------
    # Image-only gamma
    # ------------------------------------
    for gamma in P["gamma"]:
        tag = f"gamma_{gamma:.2f}".replace(".", "p")
        save_pair(
            tag,
            img_gamma(image01, gamma),
            mask01,
            f"image-only gamma={gamma:.3f}"
        )

    # ------------------------------------
    # Image-only Gaussian noise
    # ------------------------------------
    for i, sigma in enumerate(P["gaussian_noise"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 100 + i
        tag = f"gnoise_{sigma:.3f}".replace(".", "p")
        save_pair(
            tag,
            img_gaussian_noise(image01, sigma=sigma, seed=seed),
            mask01,
            f"image-only Gaussian noise sigma={sigma:.4f}, seed={seed}"
        )

    # ------------------------------------
    # Image-only Poisson noise
    # ------------------------------------
    for i, peak in enumerate(P["poisson_peak"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 200 + i
        save_pair(
            f"poisson_peak{int(peak)}",
            img_poisson_noise(image01, peak=peak, seed=seed),
            mask01,
            f"image-only Poisson noise peak={peak}, seed={seed}"
        )

    # ------------------------------------
    # Image-only Gaussian blur
    # ------------------------------------
    for sigma in P["blur_sigma"]:
        tag = f"blur_{sigma:.2f}".replace(".", "p")
        save_pair(
            tag,
            img_gaussian_blur(image01, sigma_xy=sigma),
            mask01,
            f"image-only Gaussian blur sigmaXY={sigma:.3f}"
        )

    # ------------------------------------
    # Image-only unsharp sharpening
    # ------------------------------------
    for amount in P["sharpen_amount"]:
        tag = f"sharpen_{amount:.2f}".replace(".", "p")
        save_pair(
            tag,
            img_sharpen(image01, amount=amount, blur_sigma=0.8),
            mask01,
            f"image-only unsharp amount={amount:.3f}"
        )

    # ------------------------------------
    # Image-only salt & pepper
    # ------------------------------------
    for i, amount in enumerate(P["salt_pepper"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 300 + i
        tag = f"saltpepper_{amount:.4f}".replace(".", "p")
        save_pair(
            tag,
            img_salt_pepper(image01, amount=amount, seed=seed),
            mask01,
            f"image-only salt-pepper amount={amount:.5f}, seed={seed}"
        )

    # ------------------------------------
    # Random combined augmentations
    # ------------------------------------
    for i in range(1, RANDOM_COMPOSITES_PER_IMAGE + 1):
        seed = GLOBAL_SEED + pair_index * 100000 + 5000 + i
        img_aug, mask_aug, details = random_composite_pair(
            image01, mask01, seed=seed
        )
        save_pair(
            f"randomcombo_{i:02d}",
            img_aug,
            mask_aug,
            f"seed={seed}; {details}"
        )


# ============================================================
# Dataset discovery / validation
# ============================================================

def discover_pairs(root: Path) -> Tuple[Path, Path, List[Tuple[Path, Path]], List[str]]:
    image_dir = root / "image"
    mask_dir = root / "mask"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {mask_dir}")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    pairs: List[Tuple[Path, Path]] = []
    missing_masks: List[str] = []

    # Build case-insensitive mask lookup.
    mask_lookup = {
        p.name.lower(): p
        for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    for image_path in images:
        candidate_names = [
            f"{image_path.stem}_mask.tif",
            f"{image_path.stem}_mask.tiff",
        ]

        mask_path = None
        for name in candidate_names:
            if name.lower() in mask_lookup:
                mask_path = mask_lookup[name.lower()]
                break

        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))

    return image_dir, mask_dir, pairs, missing_masks


def expected_outputs_per_pair() -> int:
    n = 0
    n += int(INCLUDE_ORIGINAL)
    n += 3          # flips
    n += 3          # 90 rotations
    n += len(P["rotations"])
    n += len(P["shifts"])
    n += len(P["zooms"])
    n += len(P["shears"])
    n += len(P["elastic"])
    n += len(P["brightness"])
    n += len(P["contrast"])
    n += len(P["gamma"])
    n += len(P["gaussian_noise"])
    n += len(P["poisson_peak"])
    n += len(P["blur_sigma"])
    n += len(P["sharpen_amount"])
    n += len(P["salt_pepper"])
    n += RANDOM_COMPOSITES_PER_IMAGE
    return n


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Paired TIFF Data Augmentation")
    print("=" * 72)
    print(f"Profile                 : {AUGMENTATION_PROFILE}")
    print(f"Random composites/image : {RANDOM_COMPOSITES_PER_IMAGE}")
    print(f"Skip existing           : {SKIP_EXISTING}")
    print()

    root = choose_root_folder()
    if root is None:
        print("No folder selected. Exit.")
        return

    root = root.resolve()
    print(f"Input root: {root}")

    image_dir, mask_dir, pairs, missing_masks = discover_pairs(root)

    if not pairs:
        raise RuntimeError(
            "No valid image/mask pairs found.\n"
            "Expected image: image/abc.tif\n"
            "Expected mask : mask/abc_mask.tif"
        )

    output_root = root.parent / f"{root.name}_augmentation"
    out_image_dir = output_root / "image"
    out_mask_dir = output_root / "mask"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    per_pair = expected_outputs_per_pair()
    total_expected = len(pairs) * per_pair

    print(f"Valid pairs             : {len(pairs)}")
    print(f"Augmentations per pair  : {per_pair}")
    print(f"Expected output pairs    : {total_expected}")
    print(f"Output root             : {output_root}")

    if missing_masks:
        print()
        print(f"[WARNING] {len(missing_masks)} image(s) have no matching mask and will be skipped.")
        for name in missing_masks[:20]:
            print(f"  - {name}")
        if len(missing_masks) > 20:
            print(f"  ... and {len(missing_masks) - 20} more")

    manifest_rows: List[dict] = []
    failures: List[Tuple[str, str]] = []

    print()
    for idx, (image_path, mask_path) in enumerate(pairs, 1):
        print(f"[{idx:4d}/{len(pairs):4d}] {image_path.name}")

        try:
            augment_one_pair(
                image_path=image_path,
                mask_path=mask_path,
                out_image_dir=out_image_dir,
                out_mask_dir=out_mask_dir,
                manifest_rows=manifest_rows,
                pair_index=idx,
            )
        except Exception as e:
            failures.append((image_path.name, str(e)))
            print(f"    [ERROR] {e}")

    manifest_path = output_root / "augmentation_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source_image",
            "source_mask",
            "augmentation_tag",
            "details",
            "output_image",
            "output_mask",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if failures:
        failure_path = output_root / "augmentation_failures.csv"
        with open(failure_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "error"])
            writer.writerows(failures)

    print()
    print("=" * 72)
    print("Finished.")
    print(f"Output folder : {output_root}")
    print(f"Manifest      : {manifest_path}")
    print(f"Successful rows in manifest: {len(manifest_rows)}")
    print(f"Failed source images       : {len(failures)}")
    print("=" * 72)

    # Keep console visible when double-clicked on Windows.
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL ERROR]")
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass

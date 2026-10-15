# -*- coding: utf-8 -*-
"""
Microscopy paired TIFF augmentation - FULL v2

Folder structure:
dataset_root/
├─ image/
│  ├─ sample001.tif
│  ├─ sample002.tif
│  └─ ...
└─ mask/
   ├─ sample001_mask.tif
   ├─ sample002_mask.tif
   └─ ...

Run this script, choose dataset_root in the popup window.

Output:
dataset_root_augmentation/
├─ image/
├─ mask/
└─ augmentation_manifest.csv

Main principles:
1) Image and mask always receive exactly the same GEOMETRIC transform.
2) Intensity / noise / blur / background / illumination changes affect IMAGE ONLY.
3) Mask interpolation is nearest-neighbor and converted back to binary.
4) Background/illumination augmentation NEVER uses the ground-truth mask,
   avoiding label leakage at mask boundaries.
5) Supports 2D TIFF and stacks with spatial axes in the last two dimensions (..., Y, X).

Dependencies:
    pip install numpy scipy tifffile

Authoring profile:
    "light"  = quick testing
    "medium" = moderate augmentation
    "full"   = broad microscopy augmentation
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi


# ============================================================
# USER SETTINGS
# ============================================================

AUGMENTATION_PROFILE = "full"     # "light" / "medium" / "full"

INCLUDE_ORIGINAL = True
SKIP_EXISTING = True
GLOBAL_SEED = 20260911

# Border behavior for geometric transforms.
# "reflect" is usually better for microscopy than black padding.
BORDER_MODE = "reflect"
IMAGE_CVAL = 0.0
MASK_CVAL = 0.0

# Set e.g. 255 if all masks must be 0/255.
# None = infer foreground value from each input mask.
FORCE_MASK_FOREGROUND_VALUE = None

IMAGE_EXTENSIONS = {".tif", ".tiff"}

# Random-combination images per source image.
RANDOM_COMPOSITES_PER_IMAGE = {
    "light": 5,
    "medium": 12,
    "full": 20,
}[AUGMENTATION_PROFILE]


# ============================================================
# AUGMENTATION PARAMETER BANKS
# ============================================================

PROFILE_PARAMS = {
    "light": {
        # Geometry
        "rotations": [-10, 10],
        "shifts": [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)],
        "zooms": [0.90, 1.10],
        "shears": [-0.08, 0.08],
        "elastic": [(0.025, 0.040)],

        # Intensity
        "brightness": [0.90, 1.10],
        "contrast": [0.90, 1.10],
        "gamma": [0.85, 1.15],

        # Noise / focus
        "gaussian_noise": [0.015, 0.03],
        "poisson_peak": [32],
        "salt_pepper": [0.002],
        "blur_sigma": [0.6, 1.0],
        "defocus_sigma": [1.3],
        "sharpen_amount": [0.6],

        # Background / illumination
        "background_offset": [-0.05, 0.05, 0.10],
        "gradient_strength": [0.08],
        "radial_strength": [-0.10, 0.10],
        "vignette_strength": [0.12],
        "lowfreq_strength": [0.08],
        "glow_strength": [0.10],
        "stripe_strength": [0.035],
        "multiplicative_illumination": [0.10],
    },

    "medium": {
        # Geometry
        "rotations": [-15, -8, 8, 15],
        "shifts": [
            (-0.08, 0), (0.08, 0), (0, -0.08), (0, 0.08),
            (-0.05, -0.05), (-0.05, 0.05),
            (0.05, -0.05), (0.05, 0.05),
        ],
        "zooms": [0.85, 0.92, 0.96, 1.04, 1.08, 1.15],
        "shears": [-0.12, -0.06, 0.06, 0.12],
        "elastic": [
            (0.020, 0.035),
            (0.035, 0.050),
            (0.050, 0.060),
        ],

        # Intensity
        "brightness": [0.82, 0.92, 1.08, 1.18],
        "contrast": [0.80, 0.92, 1.08, 1.20],
        "gamma": [0.75, 0.88, 1.12, 1.28],

        # Noise / focus
        "gaussian_noise": [0.01, 0.02, 0.04],
        "poisson_peak": [64, 32, 16],
        "salt_pepper": [0.001, 0.003],
        "blur_sigma": [0.4, 0.7, 1.0],
        "defocus_sigma": [1.3, 1.7],
        "sharpen_amount": [0.5, 1.0],

        # Background / illumination
        "background_offset": [-0.10, -0.05, 0.05, 0.10, 0.15],
        "gradient_strength": [0.06, 0.12],
        "radial_strength": [-0.15, -0.08, 0.08, 0.15],
        "vignette_strength": [0.08, 0.15, 0.22],
        "lowfreq_strength": [0.05, 0.10, 0.15],
        "glow_strength": [0.08, 0.15],
        "stripe_strength": [0.02, 0.04],
        "multiplicative_illumination": [0.08, 0.15],
    },

    "full": {
        # Geometry
        "rotations": [-20, -15, -10, -5, 5, 10, 15, 20],
        "shifts": [
            (-0.10, 0), (-0.05, 0), (0.05, 0), (0.10, 0),
            (0, -0.10), (0, -0.05), (0, 0.05), (0, 0.10),
            (-0.07, -0.07), (-0.07, 0.07),
            (0.07, -0.07), (0.07, 0.07),
        ],
        "zooms": [0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20],
        "shears": [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15],
        "elastic": [
            (0.015, 0.030),
            (0.025, 0.040),
            (0.035, 0.050),
            (0.050, 0.060),
        ],

        # Intensity
        "brightness": [0.75, 0.85, 0.92, 1.08, 1.15, 1.25],
        "contrast": [0.70, 0.80, 0.90, 1.10, 1.20, 1.30],
        "gamma": [0.65, 0.75, 0.85, 1.15, 1.30, 1.45],

        # Noise / focus
        "gaussian_noise": [0.008, 0.015, 0.025, 0.040, 0.060],
        "poisson_peak": [96, 64, 32, 16],
        "salt_pepper": [0.001, 0.0025, 0.005],
        "blur_sigma": [0.4, 0.7, 1.0, 1.3],
        "defocus_sigma": [1.5, 2.0],
        "sharpen_amount": [0.4, 0.8, 1.2],

        # Background / illumination
        "background_offset": [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20],

        # Applied in horizontal, vertical, diag1, diag2 directions.
        "gradient_strength": [0.05, 0.10, 0.15],

        # Positive = brighter center / darker edge.
        # Negative = darker center / brighter edge.
        "radial_strength": [-0.20, -0.12, -0.06, 0.06, 0.12, 0.20],

        # Edge darkening.
        "vignette_strength": [0.06, 0.12, 0.18, 0.25],

        # Smooth random low-frequency field.
        "lowfreq_strength": [0.04, 0.08, 0.12, 0.18],

        # Smooth local illumination hotspot.
        "glow_strength": [0.05, 0.10, 0.15, 0.20],

        # Gentle horizontal/vertical acquisition stripes.
        "stripe_strength": [0.015, 0.030, 0.050],

        # Multiplicative flat-field style illumination.
        "multiplicative_illumination": [0.05, 0.10, 0.15, 0.20],
    },
}

P = PROFILE_PARAMS[AUGMENTATION_PROFILE]


# ============================================================
# FOLDER SELECTION
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

        return Path(folder) if folder else None

    except Exception as e:
        print(f"[WARNING] Folder picker failed: {e}")
        typed = input("Enter dataset root path: ").strip().strip('"')
        return Path(typed) if typed else None


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_image01(arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    arr = np.asarray(arr)
    dtype = arr.dtype

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        x = arr.astype(np.float32)
        denom = float(info.max - info.min)
        x = (x - info.min) / denom if denom > 0 else np.zeros_like(x, dtype=np.float32)
        meta = {
            "dtype": dtype,
            "kind": "integer",
            "min": info.min,
            "max": info.max,
        }
        return np.clip(x, 0, 1), meta

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

    positive = mask[mask > 0]

    if positive.size > 0:
        val = positive.max()
        if np.issubdtype(mask.dtype, np.floating):
            return 1.0 if val <= 1.0 else float(val)
        return val.item()

    if np.issubdtype(mask.dtype, np.integer):
        return np.iinfo(mask.dtype).max

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
) -> Tuple[Path, Path]:
    image_out = out_image_dir / f"{stem}__{tag}.tif"
    mask_out = out_mask_dir / f"{stem}__{tag}_mask.tif"
    return image_out, mask_out


def already_done(image_out: Path, mask_out: Path) -> bool:
    return SKIP_EXISTING and image_out.exists() and mask_out.exists()


# ============================================================
# GEOMETRIC TRANSFORMS
# ============================================================

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

    shift_vec = [0.0] * x.ndim
    shift_vec[-2] = dy_fraction * h
    shift_vec[-1] = dx_fraction * w

    return ndi.shift(
        x,
        shift=shift_vec,
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
    s = 1.0 / float(factor)
    m = np.array([[s, 0.0], [0.0, s]], dtype=np.float64)
    return _centered_affine(x, m, order, cval)


def geom_shear(x: np.ndarray, shear: float, order: int, cval: float) -> np.ndarray:
    m = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float64)
    return _centered_affine(x, m, order, cval)


def _elastic_displacement(
    h: int,
    w: int,
    alpha_fraction: float,
    sigma_fraction: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    min_dim = min(h, w)

    alpha = alpha_fraction * min_dim
    sigma = max(1.0, sigma_fraction * min_dim)

    dy = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dx = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)

    dy = ndi.gaussian_filter(dy, sigma=sigma, mode="reflect")
    dx = ndi.gaussian_filter(dx, sigma=sigma, mode="reflect")

    dy = dy / (float(dy.std()) + 1e-8) * alpha * 0.15
    dx = dx / (float(dx.std()) + 1e-8) * alpha * 0.15

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
    dy, dx = _elastic_displacement(h, w, alpha_fraction, sigma_fraction, seed)

    if x.ndim == 2:
        return _apply_elastic_2d(x, dy, dx, order, cval)

    out = np.empty_like(x, dtype=np.float32)

    for idx in np.ndindex(*x.shape[:-2]):
        out[idx] = _apply_elastic_2d(x[idx], dy, dx, order, cval)

    return out


# ============================================================
# IMAGE-ONLY INTENSITY / NOISE / FOCUS
# ============================================================

def img_brightness(x: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(x * factor, 0, 1)


def img_contrast(x: np.ndarray, factor: float) -> np.ndarray:
    mean = float(np.mean(x))
    return np.clip((x - mean) * factor + mean, 0, 1)


def img_gamma(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(np.power(np.clip(x, 0, 1), gamma), 0, 1)


def img_gaussian_noise(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 1)


def img_poisson_noise(x: np.ndarray, peak: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peak = max(float(peak), 1.0)
    y = rng.poisson(np.clip(x, 0, 1) * peak).astype(np.float32) / peak
    return np.clip(y, 0, 1)


def img_salt_pepper(x: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = spatial_shape(x)
    r = rng.random((h, w))

    salt = r < amount / 2
    pepper = (r >= amount / 2) & (r < amount)

    y = x.copy()

    if x.ndim == 2:
        y[salt] = 1.0
        y[pepper] = 0.0
    else:
        y[..., salt] = 1.0
        y[..., pepper] = 0.0

    return y


def _xy_sigma(x: np.ndarray, sigma: float):
    sigmas = [0.0] * x.ndim
    sigmas[-2] = sigma
    sigmas[-1] = sigma
    return sigmas


def img_gaussian_blur(x: np.ndarray, sigma_xy: float) -> np.ndarray:
    return np.clip(
        ndi.gaussian_filter(x, sigma=_xy_sigma(x, sigma_xy), mode="reflect"),
        0,
        1,
    )


def img_defocus(x: np.ndarray, sigma_xy: float) -> np.ndarray:
    """
    Stronger blur to mimic mild out-of-focus acquisition.
    """
    return img_gaussian_blur(x, sigma_xy)


def img_sharpen(x: np.ndarray, amount: float, blur_sigma: float = 0.8) -> np.ndarray:
    blurred = img_gaussian_blur(x, blur_sigma)
    return np.clip(x + amount * (x - blurred), 0, 1)


# ============================================================
# MICROSCOPY BACKGROUND / ILLUMINATION AUGMENTATION
# ============================================================

def _xy_grid(x: np.ndarray):
    h, w = spatial_shape(x)

    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

    return yy, xx


def _broadcast_field(field2d: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Broadcast a 2D illumination field over all leading dimensions.
    """
    if x.ndim == 2:
        return field2d

    shape = (1,) * (x.ndim - 2) + field2d.shape
    return field2d.reshape(shape)


def img_background_offset(x: np.ndarray, offset: float) -> np.ndarray:
    """
    Additive detector/background offset.
    """
    return np.clip(x + offset, 0, 1)


def img_linear_gradient(
    x: np.ndarray,
    strength: float,
    direction: str,
) -> np.ndarray:
    """
    Add smooth additive background gradient.
    direction: horizontal / vertical / diag1 / diag2
    """
    yy, xx = _xy_grid(x)

    if direction == "horizontal":
        field = xx
    elif direction == "vertical":
        field = yy
    elif direction == "diag1":
        field = (xx + yy) / 2.0
    elif direction == "diag2":
        field = (xx - yy) / 2.0
    else:
        raise ValueError(direction)

    field = _broadcast_field(field.astype(np.float32), x)

    return np.clip(x + strength * field, 0, 1)


def img_radial_illumination(
    x: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    strength > 0: brighter center, darker edges
    strength < 0: darker center, brighter edges
    """
    yy, xx = _xy_grid(x)
    r2 = xx**2 + yy**2
    r2 /= max(float(r2.max()), 1e-8)

    field = 1.0 - r2
    field = field - float(field.mean())
    field = _broadcast_field(field.astype(np.float32), x)

    return np.clip(x + strength * field, 0, 1)


def img_vignette(
    x: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Multiplicative edge darkening.
    """
    yy, xx = _xy_grid(x)
    r2 = xx**2 + yy**2
    r2 /= max(float(r2.max()), 1e-8)

    field = 1.0 - strength * r2
    field = np.clip(field, 0.45, 1.25)
    field = _broadcast_field(field.astype(np.float32), x)

    return np.clip(x * field, 0, 1)


def img_low_frequency_background(
    x: np.ndarray,
    strength: float,
    seed: int,
) -> np.ndarray:
    """
    Smooth random illumination/autofluorescence background.
    """
    rng = np.random.default_rng(seed)
    h, w = spatial_shape(x)

    coarse_h = max(3, h // 64)
    coarse_w = max(3, w // 64)

    coarse = rng.normal(0, 1, size=(coarse_h, coarse_w)).astype(np.float32)

    zoom_y = h / coarse_h
    zoom_x = w / coarse_w

    field = ndi.zoom(coarse, zoom=(zoom_y, zoom_x), order=3)

    # Crop in case rounding makes it one pixel too large.
    field = field[:h, :w]

    # Pad if one pixel too small.
    if field.shape != (h, w):
        pad_h = h - field.shape[0]
        pad_w = w - field.shape[1]
        field = np.pad(
            field,
            ((0, max(0, pad_h)), (0, max(0, pad_w))),
            mode="reflect",
        )[:h, :w]

    field = ndi.gaussian_filter(
        field,
        sigma=max(2.0, min(h, w) * 0.03),
        mode="reflect",
    )

    field -= float(field.mean())
    field /= float(field.std()) + 1e-8
    field = np.clip(field, -2.5, 2.5) / 2.5

    field = _broadcast_field(field.astype(np.float32), x)

    return np.clip(x + strength * field, 0, 1)


def img_local_glow(
    x: np.ndarray,
    strength: float,
    seed: int,
) -> np.ndarray:
    """
    Smooth Gaussian illumination hotspot placed independently of the mask.
    """
    rng = np.random.default_rng(seed)
    h, w = spatial_shape(x)

    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]

    cy = float(rng.uniform(0.15 * h, 0.85 * h))
    cx = float(rng.uniform(0.15 * w, 0.85 * w))

    sigma = float(rng.uniform(0.18, 0.40) * min(h, w))

    field = np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)
    ).astype(np.float32)

    field -= float(field.min())
    field /= float(field.max()) + 1e-8

    # Center around zero so we do not simply increase all pixels.
    field = field - float(field.mean())
    field = _broadcast_field(field, x)

    return np.clip(x + strength * field, 0, 1)


def img_stripes(
    x: np.ndarray,
    strength: float,
    direction: str,
    seed: int,
) -> np.ndarray:
    """
    Gentle illumination/banding artifact.
    Not mask-dependent.
    """
    rng = np.random.default_rng(seed)
    h, w = spatial_shape(x)

    if direction == "horizontal":
        axis_len = h
        raw = rng.normal(0, 1, size=axis_len).astype(np.float32)
        smooth = ndi.gaussian_filter1d(raw, sigma=max(1, axis_len / 35))
        smooth -= smooth.mean()
        smooth /= smooth.std() + 1e-8
        field = np.repeat(smooth[:, None], w, axis=1)
    elif direction == "vertical":
        axis_len = w
        raw = rng.normal(0, 1, size=axis_len).astype(np.float32)
        smooth = ndi.gaussian_filter1d(raw, sigma=max(1, axis_len / 35))
        smooth -= smooth.mean()
        smooth /= smooth.std() + 1e-8
        field = np.repeat(smooth[None, :], h, axis=0)
    else:
        raise ValueError(direction)

    field = np.clip(field, -2.5, 2.5) / 2.5
    field = _broadcast_field(field.astype(np.float32), x)

    return np.clip(x + strength * field, 0, 1)


def img_multiplicative_illumination(
    x: np.ndarray,
    strength: float,
    seed: int,
) -> np.ndarray:
    """
    Smooth multiplicative flat-field variation.
    """
    rng = np.random.default_rng(seed)
    h, w = spatial_shape(x)

    coarse_h = max(3, h // 80)
    coarse_w = max(3, w // 80)

    coarse = rng.normal(0, 1, size=(coarse_h, coarse_w)).astype(np.float32)
    field = ndi.zoom(
        coarse,
        zoom=(h / coarse_h, w / coarse_w),
        order=3,
    )[:h, :w]

    if field.shape != (h, w):
        field = np.pad(
            field,
            ((0, max(0, h - field.shape[0])),
             (0, max(0, w - field.shape[1]))),
            mode="reflect",
        )[:h, :w]

    field = ndi.gaussian_filter(
        field,
        sigma=max(2.0, min(h, w) * 0.025),
        mode="reflect",
    )

    field -= float(field.mean())
    field /= float(field.std()) + 1e-8
    field = np.clip(field, -2.5, 2.5) / 2.5

    multiplier = 1.0 + strength * field
    multiplier = np.clip(multiplier, 0.55, 1.45)

    multiplier = _broadcast_field(multiplier.astype(np.float32), x)

    return np.clip(x * multiplier, 0, 1)


# ============================================================
# SAVE HELPER
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
# RANDOM COMPOSITE AUGMENTATION
# ============================================================

def random_composite_pair(
    image01: np.ndarray,
    mask01: np.ndarray,
    seed: int,
):
    rng = np.random.default_rng(seed)

    img = image01.copy()
    msk = mask01.copy()
    ops = []

    # -------------------------
    # Geometry
    # -------------------------
    if rng.random() < 0.5:
        img = geom_flip_h(img)
        msk = geom_flip_h(msk)
        ops.append("flipH")

    if rng.random() < 0.5:
        img = geom_flip_v(img)
        msk = geom_flip_v(msk)
        ops.append("flipV")

    angle = float(rng.uniform(-20, 20))
    img = geom_rotate(img, angle, 1, IMAGE_CVAL)
    msk = geom_rotate(msk, angle, 0, MASK_CVAL)
    ops.append(f"rot={angle:.1f}")

    factor = float(rng.uniform(0.82, 1.18))
    img = geom_zoom(img, factor, 1, IMAGE_CVAL)
    msk = geom_zoom(msk, factor, 0, MASK_CVAL)
    ops.append(f"zoom={factor:.3f}")

    dy = float(rng.uniform(-0.08, 0.08))
    dx = float(rng.uniform(-0.08, 0.08))
    img = geom_shift(img, dy, dx, 1, IMAGE_CVAL)
    msk = geom_shift(msk, dy, dx, 0, MASK_CVAL)
    ops.append(f"shift={dy:.3f},{dx:.3f}")

    if rng.random() < 0.45:
        shear = float(rng.uniform(-0.12, 0.12))
        img = geom_shear(img, shear, 1, IMAGE_CVAL)
        msk = geom_shear(msk, shear, 0, MASK_CVAL)
        ops.append(f"shear={shear:.3f}")

    if rng.random() < 0.30:
        alpha = float(rng.uniform(0.015, 0.045))
        sigma = float(rng.uniform(0.035, 0.060))
        e_seed = int(rng.integers(0, 2**31 - 1))

        img = geom_elastic(img, alpha, sigma, e_seed, 1, IMAGE_CVAL)
        msk = geom_elastic(msk, alpha, sigma, e_seed, 0, MASK_CVAL)

        ops.append(f"elastic={alpha:.3f},{sigma:.3f}")

    # -------------------------
    # Basic intensity
    # -------------------------
    if rng.random() < 0.80:
        b = float(rng.uniform(0.86, 1.16))
        img = img_brightness(img, b)
        ops.append(f"brightness={b:.3f}")

    if rng.random() < 0.80:
        c = float(rng.uniform(0.84, 1.18))
        img = img_contrast(img, c)
        ops.append(f"contrast={c:.3f}")

    if rng.random() < 0.65:
        g = float(rng.uniform(0.78, 1.25))
        img = img_gamma(img, g)
        ops.append(f"gamma={g:.3f}")

    # -------------------------
    # Background / illumination
    # -------------------------
    if rng.random() < 0.50:
        offset = float(rng.uniform(-0.08, 0.12))
        img = img_background_offset(img, offset)
        ops.append(f"bgOffset={offset:.3f}")

    if rng.random() < 0.45:
        strength = float(rng.uniform(0.04, 0.13))
        direction = rng.choice(["horizontal", "vertical", "diag1", "diag2"]).item()
        if rng.random() < 0.5:
            strength *= -1
        img = img_linear_gradient(img, strength, direction)
        ops.append(f"gradient={direction},{strength:.3f}")

    if rng.random() < 0.35:
        strength = float(rng.uniform(-0.16, 0.16))
        img = img_radial_illumination(img, strength)
        ops.append(f"radial={strength:.3f}")

    if rng.random() < 0.25:
        strength = float(rng.uniform(0.06, 0.20))
        img = img_vignette(img, strength)
        ops.append(f"vignette={strength:.3f}")

    if rng.random() < 0.40:
        strength = float(rng.uniform(0.04, 0.14))
        lf_seed = int(rng.integers(0, 2**31 - 1))
        img = img_low_frequency_background(img, strength, lf_seed)
        ops.append(f"lowFreq={strength:.3f}")

    if rng.random() < 0.28:
        strength = float(rng.uniform(0.05, 0.16))
        glow_seed = int(rng.integers(0, 2**31 - 1))
        img = img_local_glow(img, strength, glow_seed)
        ops.append(f"glow={strength:.3f}")

    if rng.random() < 0.20:
        strength = float(rng.uniform(0.012, 0.040))
        direction = rng.choice(["horizontal", "vertical"]).item()
        stripe_seed = int(rng.integers(0, 2**31 - 1))
        img = img_stripes(img, strength, direction, stripe_seed)
        ops.append(f"stripe={direction},{strength:.3f}")

    if rng.random() < 0.30:
        strength = float(rng.uniform(0.05, 0.16))
        mul_seed = int(rng.integers(0, 2**31 - 1))
        img = img_multiplicative_illumination(img, strength, mul_seed)
        ops.append(f"multIllum={strength:.3f}")

    # -------------------------
    # Noise / focus
    # -------------------------
    if rng.random() < 0.55:
        sigma = float(rng.uniform(0.005, 0.030))
        n_seed = int(rng.integers(0, 2**31 - 1))
        img = img_gaussian_noise(img, sigma, n_seed)
        ops.append(f"gNoise={sigma:.3f}")

    if rng.random() < 0.20:
        peak = float(rng.choice([24, 32, 48, 64]))
        p_seed = int(rng.integers(0, 2**31 - 1))
        img = img_poisson_noise(img, peak, p_seed)
        ops.append(f"poisson={peak:.0f}")

    if rng.random() < 0.25:
        sigma = float(rng.uniform(0.4, 1.2))
        img = img_gaussian_blur(img, sigma)
        ops.append(f"blur={sigma:.2f}")

    if rng.random() < 0.12:
        sigma = float(rng.uniform(1.3, 1.9))
        img = img_defocus(img, sigma)
        ops.append(f"defocus={sigma:.2f}")

    if rng.random() < 0.15:
        amount = float(rng.uniform(0.35, 0.9))
        img = img_sharpen(img, amount)
        ops.append(f"sharpen={amount:.2f}")

    msk = (msk > 0.5).astype(np.float32)

    return np.clip(img, 0, 1), msk, "; ".join(ops)


# ============================================================
# AUGMENT ONE IMAGE/MASK PAIR
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

    def save_pair(tag: str, img_aug: np.ndarray, msk_aug: np.ndarray, details: str):
        image_out, mask_out = paired_output_paths(
            out_image_dir,
            out_mask_dir,
            stem,
            tag,
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

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------
    if INCLUDE_ORIGINAL:
        save_pair("original", image01, mask01, "unchanged copy")

    # --------------------------------------------------------
    # Flip
    # --------------------------------------------------------
    save_pair("flip_h", geom_flip_h(image01), geom_flip_h(mask01), "horizontal flip")
    save_pair("flip_v", geom_flip_v(image01), geom_flip_v(mask01), "vertical flip")
    save_pair("flip_hv", geom_flip_hv(image01), geom_flip_hv(mask01), "horizontal + vertical flip")

    # --------------------------------------------------------
    # 90-degree rotation
    # --------------------------------------------------------
    for k, deg in [(1, 90), (2, 180), (3, 270)]:
        save_pair(
            f"rot{deg}",
            geom_rot90(image01, k),
            geom_rot90(mask01, k),
            f"rotation={deg} deg",
        )

    # --------------------------------------------------------
    # Arbitrary rotation
    # --------------------------------------------------------
    for angle in P["rotations"]:
        tag = f"rotate_{angle:+d}deg".replace("+", "p").replace("-", "m")

        save_pair(
            tag,
            geom_rotate(image01, angle, 1, IMAGE_CVAL),
            geom_rotate(mask01, angle, 0, MASK_CVAL),
            f"rotation={angle} deg",
        )

    # --------------------------------------------------------
    # Shift
    # --------------------------------------------------------
    for i, (dy, dx) in enumerate(P["shifts"], 1):
        save_pair(
            f"shift_{i:02d}",
            geom_shift(image01, dy, dx, 1, IMAGE_CVAL),
            geom_shift(mask01, dy, dx, 0, MASK_CVAL),
            f"dy={dy:.3f}*H, dx={dx:.3f}*W",
        )

    # --------------------------------------------------------
    # Zoom
    # --------------------------------------------------------
    for factor in P["zooms"]:
        tag = f"zoom_{factor:.2f}".replace(".", "p")

        save_pair(
            tag,
            geom_zoom(image01, factor, 1, IMAGE_CVAL),
            geom_zoom(mask01, factor, 0, MASK_CVAL),
            f"zoom={factor:.3f}",
        )

    # --------------------------------------------------------
    # Shear
    # --------------------------------------------------------
    for shear in P["shears"]:
        tag = (
            f"shear_{shear:+.2f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )

        save_pair(
            tag,
            geom_shear(image01, shear, 1, IMAGE_CVAL),
            geom_shear(mask01, shear, 0, MASK_CVAL),
            f"shear={shear:.3f}",
        )

    # --------------------------------------------------------
    # Elastic deformation
    # --------------------------------------------------------
    for i, (alpha, sigma) in enumerate(P["elastic"], 1):
        seed = GLOBAL_SEED + pair_index * 1000 + i

        save_pair(
            f"elastic_{i:02d}",
            geom_elastic(image01, alpha, sigma, seed, 1, IMAGE_CVAL),
            geom_elastic(mask01, alpha, sigma, seed, 0, MASK_CVAL),
            f"alpha_fraction={alpha:.4f}, sigma_fraction={sigma:.4f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------
    for factor in P["brightness"]:
        tag = f"brightness_{factor:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_brightness(image01, factor),
            mask01,
            f"image-only brightness={factor:.3f}",
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------
    for factor in P["contrast"]:
        tag = f"contrast_{factor:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_contrast(image01, factor),
            mask01,
            f"image-only contrast={factor:.3f}",
        )

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------
    for gamma in P["gamma"]:
        tag = f"gamma_{gamma:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_gamma(image01, gamma),
            mask01,
            f"image-only gamma={gamma:.3f}",
        )

    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------
    for i, sigma in enumerate(P["gaussian_noise"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 100 + i
        tag = f"gnoise_{sigma:.3f}".replace(".", "p")

        save_pair(
            tag,
            img_gaussian_noise(image01, sigma, seed),
            mask01,
            f"Gaussian noise sigma={sigma:.4f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Poisson noise
    # --------------------------------------------------------
    for i, peak in enumerate(P["poisson_peak"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 200 + i

        save_pair(
            f"poisson_peak{int(peak)}",
            img_poisson_noise(image01, peak, seed),
            mask01,
            f"Poisson peak={peak}, seed={seed}",
        )

    # --------------------------------------------------------
    # Salt-pepper noise
    # --------------------------------------------------------
    for i, amount in enumerate(P["salt_pepper"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 300 + i
        tag = f"saltpepper_{amount:.4f}".replace(".", "p")

        save_pair(
            tag,
            img_salt_pepper(image01, amount, seed),
            mask01,
            f"salt-pepper amount={amount:.5f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------
    for sigma in P["blur_sigma"]:
        tag = f"blur_{sigma:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_gaussian_blur(image01, sigma),
            mask01,
            f"Gaussian blur sigmaXY={sigma:.3f}",
        )

    # --------------------------------------------------------
    # Defocus
    # --------------------------------------------------------
    for sigma in P["defocus_sigma"]:
        tag = f"defocus_{sigma:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_defocus(image01, sigma),
            mask01,
            f"mild defocus sigmaXY={sigma:.3f}",
        )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------
    for amount in P["sharpen_amount"]:
        tag = f"sharpen_{amount:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_sharpen(image01, amount),
            mask01,
            f"unsharp amount={amount:.3f}",
        )

    # ========================================================
    # BACKGROUND / ILLUMINATION
    # ========================================================

    # --------------------------------------------------------
    # Global additive background offset
    # --------------------------------------------------------
    for offset in P["background_offset"]:
        tag = (
            f"bgoffset_{offset:+.2f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )

        save_pair(
            tag,
            img_background_offset(image01, offset),
            mask01,
            f"background additive offset={offset:.3f}",
        )

    # --------------------------------------------------------
    # Linear gradient illumination
    # --------------------------------------------------------
    gradient_directions = [
        "horizontal",
        "vertical",
        "diag1",
        "diag2",
    ]

    for strength in P["gradient_strength"]:
        for direction in gradient_directions:
            for sign in (-1, 1):
                s = strength * sign
                sign_tag = "p" if s > 0 else "m"
                tag = f"gradient_{direction}_{sign_tag}{abs(s):.2f}".replace(".", "p")

                save_pair(
                    tag,
                    img_linear_gradient(image01, s, direction),
                    mask01,
                    f"linear gradient direction={direction}, strength={s:.3f}",
                )

    # --------------------------------------------------------
    # Radial illumination
    # --------------------------------------------------------
    for strength in P["radial_strength"]:
        tag = (
            f"radial_{strength:+.2f}"
            .replace("+", "p")
            .replace("-", "m")
            .replace(".", "p")
        )

        save_pair(
            tag,
            img_radial_illumination(image01, strength),
            mask01,
            f"radial illumination strength={strength:.3f}",
        )

    # --------------------------------------------------------
    # Vignetting
    # --------------------------------------------------------
    for strength in P["vignette_strength"]:
        tag = f"vignette_{strength:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_vignette(image01, strength),
            mask01,
            f"vignette strength={strength:.3f}",
        )

    # --------------------------------------------------------
    # Low-frequency uneven background
    # --------------------------------------------------------
    for i, strength in enumerate(P["lowfreq_strength"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 400 + i
        tag = f"lowfreqbg_{strength:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_low_frequency_background(image01, strength, seed),
            mask01,
            f"low-frequency background strength={strength:.3f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Local smooth glow
    # --------------------------------------------------------
    for i, strength in enumerate(P["glow_strength"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 500 + i
        tag = f"localglow_{strength:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_local_glow(image01, strength, seed),
            mask01,
            f"local glow strength={strength:.3f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Horizontal / vertical banding
    # --------------------------------------------------------
    for i, strength in enumerate(P["stripe_strength"], 1):
        for j, direction in enumerate(["horizontal", "vertical"], 1):
            seed = GLOBAL_SEED + pair_index * 10000 + 600 + i * 10 + j
            tag = f"stripe_{direction}_{strength:.3f}".replace(".", "p")

            save_pair(
                tag,
                img_stripes(image01, strength, direction, seed),
                mask01,
                f"stripe direction={direction}, strength={strength:.4f}, seed={seed}",
            )

    # --------------------------------------------------------
    # Multiplicative flat-field variation
    # --------------------------------------------------------
    for i, strength in enumerate(P["multiplicative_illumination"], 1):
        seed = GLOBAL_SEED + pair_index * 10000 + 700 + i
        tag = f"multillum_{strength:.2f}".replace(".", "p")

        save_pair(
            tag,
            img_multiplicative_illumination(image01, strength, seed),
            mask01,
            f"multiplicative illumination strength={strength:.3f}, seed={seed}",
        )

    # --------------------------------------------------------
    # Random combined augmentation
    # --------------------------------------------------------
    for i in range(1, RANDOM_COMPOSITES_PER_IMAGE + 1):
        seed = GLOBAL_SEED + pair_index * 100000 + 5000 + i

        img_aug, mask_aug, details = random_composite_pair(
            image01,
            mask01,
            seed,
        )

        save_pair(
            f"randomcombo_{i:02d}",
            img_aug,
            mask_aug,
            f"seed={seed}; {details}",
        )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_pairs(root: Path):
    image_dir = root / "image"
    mask_dir = root / "mask"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {image_dir}")

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {mask_dir}")

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    mask_lookup = {
        p.name.lower(): p
        for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs = []
    missing_masks = []

    for image_path in images:
        possible = [
            f"{image_path.stem}_mask.tif",
            f"{image_path.stem}_mask.tiff",
        ]

        mask_path = None

        for name in possible:
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

    n += 3  # flips
    n += 3  # rot90 / 180 / 270

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
    n += len(P["salt_pepper"])
    n += len(P["blur_sigma"])
    n += len(P["defocus_sigma"])
    n += len(P["sharpen_amount"])

    n += len(P["background_offset"])

    # Four directions x positive/negative
    n += len(P["gradient_strength"]) * 4 * 2

    n += len(P["radial_strength"])
    n += len(P["vignette_strength"])
    n += len(P["lowfreq_strength"])
    n += len(P["glow_strength"])

    # horizontal + vertical
    n += len(P["stripe_strength"]) * 2

    n += len(P["multiplicative_illumination"])

    n += RANDOM_COMPOSITES_PER_IMAGE

    return n


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("Microscopy Paired TIFF Data Augmentation - FULL v2")
    print("=" * 78)

    print(f"Profile                  : {AUGMENTATION_PROFILE}")
    print(f"Random composites / image: {RANDOM_COMPOSITES_PER_IMAGE}")
    print(f"Skip existing            : {SKIP_EXISTING}")
    print()

    root = choose_root_folder()

    if root is None:
        print("No folder selected. Exit.")
        return

    root = root.resolve()

    print(f"Input root: {root}")

    _, _, pairs, missing_masks = discover_pairs(root)

    if not pairs:
        raise RuntimeError(
            "No valid image/mask pairs found.\n"
            "Expected:\n"
            "  image/abc.tif\n"
            "  mask/abc_mask.tif"
        )

    output_root = root.parent / f"{root.name}_augmentation"
    out_image_dir = output_root / "image"
    out_mask_dir = output_root / "mask"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    per_pair = expected_outputs_per_pair()
    total_expected = len(pairs) * per_pair

    print(f"Valid pairs              : {len(pairs)}")
    print(f"Outputs per source pair  : {per_pair}")
    print(f"Expected output pairs     : {total_expected}")
    print(f"Output root              : {output_root}")

    if missing_masks:
        print()
        print(
            f"[WARNING] {len(missing_masks)} image(s) have no matching mask "
            "and will be skipped."
        )

        for name in missing_masks[:20]:
            print(f"  - {name}")

        if len(missing_masks) > 20:
            print(f"  ... and {len(missing_masks) - 20} more")

    manifest_rows = []
    failures = []

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

    # Save manifest
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

    # Save failures if any
    if failures:
        failure_path = output_root / "augmentation_failures.csv"

        with open(failure_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "error"])
            writer.writerows(failures)

    print()
    print("=" * 78)
    print("Finished.")
    print(f"Output folder              : {output_root}")
    print(f"Manifest                   : {manifest_path}")
    print(f"Manifest rows              : {len(manifest_rows)}")
    print(f"Failed source images       : {len(failures)}")
    print("=" * 78)

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

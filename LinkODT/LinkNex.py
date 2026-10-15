# LinkNex Project v3 - canonical GUI source
# -*- coding: utf-8 -*-
"""
LinkODT - Organelle Network analysis Suite
v23 - Semantic Model Left + Annotation Right

Pipeline
--------
Image review (FIRST; Z-max multichannel composite)
    -> GOOD: continue
    -> BAD : move source TIFF to QC_Bad/
Draw 2D cell instance mask on Z-max multichannel composite
Apply Mask
    -> save Z-max composite + every channel Z-project PNG
    -> save 2D cell mask + 2D cell instance mask
Split channels (mandatory)
    -> same 2D cell mask is applied to every Z layer
    -> every cell instance is separated
    -> every channel is exported separately

2D is treated as Z=1.
"""

import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
import colorsys
import tempfile
import time
import zlib
import base64
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import tifffile

# LinkNex GUI is fixed to PySide6.
# pyvistaqt imports Qt through qtpy, so QT_API must be set BEFORE pyvistaqt.
# This prevents PyQt/PySide object-type conflicts such as:
#   unexpected type 'PySide6.QtWidgets.QFrame'
os.environ["QT_API"] = "pyside6"

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject, QThread, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None
    QtInteractor = None

from punctate_dynamics_module import PunctateDynamicsWidget
from ui_theme import configure_application, open_user_guide, show_about, DISABLED_STYLE, install_appearance_controls


# =============================================================================
# Theme
# =============================================================================

NAVY = "#202020"
DEEP_BLUE = "#243B60"
US_BLUE = "#2458A6"
CYAN = "#3265AC"
YELLOW = "#F6B965"
WHITE = "#FFFFFF"
OFFWHITE = "#F6F8FB"
TEXT = "#202020"
MUTED = "#616161"
BORDER = "#D2D2D2"
GREEN = "#1E9E5A"
RED = "#D94B4B"

SUPPORTED_SUFFIXES = {".tif", ".tiff"}
QC_DATA_SUFFIXES = {".png"}


# =============================================================================
# Utilities
# =============================================================================

def natural_sort_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", Path(path).name)
    ]


def normalize_uint8(image):
    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 0.5))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def rgb_to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def labels_to_pixmap(labels, alpha=105):
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    palette = [
        (0, 174, 239),
        (255, 209, 0),
        (0, 103, 177),
        (97, 200, 243),
        (30, 158, 90),
        (217, 75, 75),
        (180, 95, 220),
        (255, 128, 64),
    ]

    ids = np.unique(labels)
    ids = ids[ids > 0]

    for i, instance_id in enumerate(ids):
        color = palette[i % len(palette)]
        m = labels == instance_id
        rgba[m, 0] = color[0]
        rgba[m, 1] = color[1]
        rgba[m, 2] = color[2]
        rgba[m, 3] = alpha

    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def build_multichannel_composite(max_cyx):
    """Display composite only. Input is C,Y,X."""
    c, h, w = max_cyx.shape

    colors = np.array([
        [255,   0, 255],  # C1 magenta
        [  0, 255,   0],  # C2 green
        [  0, 255, 255],  # C3 cyan
        [255, 255,   0],  # C4 yellow
        [255,   0,   0],  # C5 red
        [  0, 128, 255],  # C6 blue
    ], dtype=np.float32)

    out = np.zeros((h, w, 3), dtype=np.float32)

    for ci in range(c):
        ch = normalize_uint8(max_cyx[ci]).astype(np.float32) / 255.0
        color = colors[ci % len(colors)] / 255.0
        out += ch[..., None] * color[None, None, :]

    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def unique_destination(path):
    path = Path(path)
    if not path.exists():
        return path

    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


# =============================================================================
# TIFF reader: normalize to C,Z,Y,X
# =============================================================================

class MicroscopyImage:
    def __init__(self, path):
        self.path = Path(path)
        self.original_axes = ""
        self.original_shape = ()
        self.t = 1
        self.data_czyx = None
        self._load()

    def _load(self):
        with tifffile.TiffFile(self.path) as tf:
            series = tf.series[0]
            arr = series.asarray()
            axes = (series.axes or "").upper()

        self.original_axes = axes
        self.original_shape = tuple(arr.shape)
        self.data_czyx = np.ascontiguousarray(self._to_czyx(arr, axes))
        self.c, self.z, self.y, self.x = self.data_czyx.shape

    def _to_czyx(self, arr, axes):
        # Use TIFF axes metadata whenever it is usable.
        if axes and len(axes) == arr.ndim:
            if "T" in axes:
                ti = axes.index("T")
                self.t = int(arr.shape[ti])
                if self.t != 1:
                    raise ValueError(
                        f"{self.path.name}\n\n"
                        f"This first preprocessing version supports static images (T=1).\n"
                        f"Detected T={self.t}, axes={axes}, shape={arr.shape}."
                    )
                arr = np.take(arr, 0, axis=ti)
                axes = axes.replace("T", "")

            # Drop singleton unknown dimensions (S/Q/etc.)
            changed = True
            while changed and len(axes) == arr.ndim:
                changed = False
                for i, (ax, size) in enumerate(zip(axes, arr.shape)):
                    if ax not in {"C", "Z", "Y", "X"} and size == 1:
                        arr = np.take(arr, 0, axis=i)
                        axes = axes[:i] + axes[i + 1:]
                        changed = True
                        break

            if "Y" in axes and "X" in axes and set(axes) <= {"C", "Z", "Y", "X"}:
                if "C" not in axes:
                    arr = np.expand_dims(arr, 0)
                    axes = "C" + axes

                if "Z" not in axes:
                    ci = axes.index("C")
                    insert_at = ci + 1
                    arr = np.expand_dims(arr, insert_at)
                    axes = axes[:insert_at] + "Z" + axes[insert_at:]

                order = [
                    axes.index("C"),
                    axes.index("Z"),
                    axes.index("Y"),
                    axes.index("X"),
                ]
                return np.transpose(arr, order)

        # Fallback for common TIFF layouts.
        if arr.ndim == 2:          # YX
            return arr[None, None, :, :]

        if arr.ndim == 3:
            if arr.shape[0] <= 8:  # CYX
                return arr[:, None, :, :]
            return arr[None, :, :, :]  # ZYX

        if arr.ndim == 4:
            if arr.shape[0] <= 8:  # CZYX
                return arr
            if arr.shape[1] <= 8:  # ZCYX
                return np.transpose(arr, (1, 0, 2, 3))

        raise ValueError(
            "Cannot safely interpret this TIFF.\n\n"
            f"File: {self.path.name}\n"
            f"Axes: {self.original_axes}\n"
            f"Shape: {self.original_shape}\n\n"
            "Expected YX, ZYX, CYX, CZYX, or ZCYX."
        )

    def max_projection(self):
        return np.max(self.data_czyx, axis=1)  # C,Y,X

    def dimension_text(self):
        return (
            f"File: {self.path.name}\n"
            f"TIFF axes: {self.original_axes or 'unknown'}\n"
            f"TIFF shape: {self.original_shape}\n\n"
            f"C: {self.c}\n"
            f"Z: {self.z}\n"
            f"Y: {self.y}\n"
            f"X: {self.x}\n"
            f"T: {self.t}\n\n"
            f"Working: C × Z × Y × X\n"
            f"{self.c} × {self.z} × {self.y} × {self.x}"
        )


# =============================================================================
# QC binary-classification model
# =============================================================================

QC_MODEL_INPUT_SIZE = 224


def collect_png_files(folder):
    """Recursively collect PNG files from one folder."""
    if folder is None:
        return []

    folder = Path(folder)
    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".png"
        ],
        key=natural_sort_key,
    )


def read_qc_rgb(path):
    """
    Read one PNG for Image review model training/test/predict.

    Supported:
      - grayscale PNG
      - RGB PNG
      - RGBA PNG

    Output:
      uint8 RGB, shape Y,X,3
    """
    path = Path(path)

    raw = np.fromfile(
        str(path),
        dtype=np.uint8,
    )
    arr = cv2.imdecode(
        raw,
        cv2.IMREAD_UNCHANGED,
    )

    if arr is None:
        raise ValueError(
            f"Cannot read Image review PNG:\n{path}"
        )

    arr = np.asarray(arr)

    if arr.ndim == 2:
        gray = (
            arr
            if arr.dtype == np.uint8
            else normalize_uint8(arr)
        )

        return np.repeat(
            gray[..., None],
            3,
            axis=2,
        )

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported Image review PNG dimensions:\n"
            f"{path.name}: {arr.shape}"
        )

    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(
            arr,
            cv2.COLOR_BGRA2RGB,
        )

    elif arr.shape[-1] >= 3:
        arr = cv2.cvtColor(
            arr[..., :3],
            cv2.COLOR_BGR2RGB,
        )

    else:
        # e.g. Y,X,1
        gray = np.squeeze(arr, axis=-1)

        if gray.dtype != np.uint8:
            gray = normalize_uint8(gray)

        return np.repeat(
            gray[..., None],
            3,
            axis=2,
        )

    if arr.dtype != np.uint8:
        channels = [
            normalize_uint8(arr[..., ci])
            for ci in range(3)
        ]
        arr = np.stack(
            channels,
            axis=-1,
        )

    return np.ascontiguousarray(
        arr,
        dtype=np.uint8,
    )


def qc_rgb_to_chw_float(
    rgb,
    input_size=QC_MODEL_INPUT_SIZE,
    augment=False,
):
    """
    Convert RGB uint8 to normalized CHW float32.

    Training augmentation is deliberately simple and microscopy-safe:
      - random horizontal flip
      - random vertical flip
      - random 90-degree rotation
    """
    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    resized = cv2.resize(
        rgb,
        (
            int(input_size),
            int(input_size),
        ),
        interpolation=cv2.INTER_AREA,
    )

    if augment:
        if np.random.random() < 0.5:
            resized = np.flip(
                resized,
                axis=1,
            )

        if np.random.random() < 0.5:
            resized = np.flip(
                resized,
                axis=0,
            )

        k = int(
            np.random.randint(
                0,
                4,
            )
        )

        if k:
            resized = np.rot90(
                resized,
                k=k,
                axes=(0, 1),
            )

    x = (
        resized.astype(np.float32)
        / 255.0
    )

    # ImageNet normalization for pretrained ResNet18.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    x = (
        x - mean[None, None, :]
    ) / std[None, None, :]

    x = np.transpose(
        x,
        (2, 0, 1),
    )

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


def import_torch_qc():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for Image review model Train / Test / Predict.\n\n"
            "Please run LinkODT in a Python environment with PyTorch installed."
        ) from exc

    return (
        torch,
        nn,
        DataLoader,
    )


def build_resnet18_qc_model(
    nn,
    pretrained=True,
):
    """
    LinkODT Image review binary classifier based on ResNet18.

    Input:
        224 x 224 RGB

    Classes:
        0 = BAD
        1 = GOOD

    Fresh training starts from ImageNet pretrained weights.
    Loading an existing LinkODT checkpoint uses pretrained=False and restores
    all ResNet18 weights from the checkpoint.
    """
    try:
        from torchvision.models import (
            resnet18,
            ResNet18_Weights,
        )
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for the ResNet18 Image review model.\n\n"
            "Install torchvision in the Python environment used to run LinkODT."
        ) from exc

    if pretrained:
        local_resnet_weights = next(
            (
                path
                for path in QC_RESNET18_WEIGHT_CANDIDATES
                if path.exists()
            ),
            None,
        )

        if local_resnet_weights is not None:
            try:
                model = resnet18(
                    weights=None,
                )
            except TypeError:
                model = resnet18(
                    pretrained=False,
                )

            import torch

            state_dict = torch.load(
                local_resnet_weights,
                map_location="cpu",
            )

            if (
                isinstance(state_dict, dict)
                and "state_dict" in state_dict
            ):
                state_dict = state_dict[
                    "state_dict"
                ]

            if (
                isinstance(state_dict, dict)
                and any(
                    str(key).startswith("module.")
                    for key in state_dict
                )
            ):
                state_dict = {
                    (
                        str(key)[7:]
                        if str(key).startswith("module.")
                        else str(key)
                    ): value
                    for key, value in state_dict.items()
                }

            model.load_state_dict(
                state_dict,
                strict=True,
            )

        else:
            try:
                model = resnet18(
                    weights=ResNet18_Weights.DEFAULT,
                )
            except TypeError:
                # Compatibility with older torchvision releases.
                model = resnet18(
                    pretrained=True,
                )
    else:
        try:
            model = resnet18(
                weights=None,
            )
        except TypeError:
            model = resnet18(
                pretrained=False,
            )

    in_features = int(
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features,
        2,
    )

    return model


def load_qc_checkpoint(model_path):
    torch, nn, _ = import_torch_qc()

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Image review model not found:\n"
            f"{model_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Unsupported Image review model format."
        )

    architecture = str(
        checkpoint.get(
            "architecture",
            checkpoint.get(
                "model_name",
                "",
            ),
        )
    ).lower()

    if architecture and "resnet18" not in architecture:
        raise ValueError(
            "This Image review model is not a ResNet18 LinkODT Image review checkpoint.\n\n"
            f"Checkpoint architecture: {architecture}"
        )

    model = build_resnet18_qc_model(
        nn,
        pretrained=False,
    )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    input_size = int(
        checkpoint.get(
            "input_size",
            QC_MODEL_INPUT_SIZE,
        )
    )

    return (
        torch,
        model,
        device,
        input_size,
    )


def predict_qc_loaded(
    torch,
    model,
    device,
    input_size,
    rgb,
):
    x_np = qc_rgb_to_chw_float(
        rgb,
        input_size=input_size,
        augment=False,
    )

    x = torch.from_numpy(
        x_np[None, ...]
    ).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(
            logits,
            dim=1,
        )[0].detach().cpu().numpy()

    pred_id = int(
        np.argmax(probs)
    )

    pred_name = (
        "good"
        if pred_id == 1
        else "bad"
    )

    return (
        pred_name,
        float(probs[1]),
        float(probs[0]),
    )


class QCTrainWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(str, str)
    failed = Signal(str)

    def __init__(
        self,
        output_folder,
        good_folder,
        bad_folder,
        epochs,
        batch_size,
        initial_model_path=None,
    ):
        super().__init__()

        self.output_folder = Path(
            output_folder
        )
        self.good_folder = Path(
            good_folder
        )
        self.bad_folder = Path(
            bad_folder
        )
        self.epochs = int(
            epochs
        )
        self.batch_size = int(
            batch_size
        )

        self.initial_model_path = (
            Path(initial_model_path)
            if initial_model_path
            else None
        )

    def run(self):
        try:
            (
                torch,
                nn,
                DataLoader,
            ) = import_torch_qc()

            good_paths = (
                collect_png_files(
                    self.good_folder
                )
            )
            bad_paths = (
                collect_png_files(
                    self.bad_folder
                )
            )

            if not good_paths:
                raise ValueError(
                    "No PNG images found in TRAIN GOOD folder."
                )

            if not bad_paths:
                raise ValueError(
                    "No PNG images found in TRAIN BAD folder."
                )

            samples = (
                [
                    (
                        p,
                        1,
                        "good",
                    )
                    for p in good_paths
                ]
                +
                [
                    (
                        p,
                        0,
                        "bad",
                    )
                    for p in bad_paths
                ]
            )

            # Shuffle once here; DataLoader will reshuffle each epoch.
            rng = np.random.default_rng(
                20260902
            )
            rng.shuffle(
                samples
            )

            class TrainDataset:
                def __len__(
                    self_inner,
                ):
                    return len(samples)

                def __getitem__(
                    self_inner,
                    index,
                ):
                    path, label, _ = (
                        samples[index]
                    )

                    rgb = read_qc_rgb(
                        path
                    )

                    x_np = (
                        qc_rgb_to_chw_float(
                            rgb,
                            input_size=(
                                QC_MODEL_INPUT_SIZE
                            ),
                            augment=True,
                        )
                    )

                    return (
                        torch.from_numpy(
                            x_np
                        ),
                        torch.tensor(
                            int(label),
                            dtype=torch.long,
                        ),
                    )

            dataset = TrainDataset()

            loader = DataLoader(
                dataset,
                batch_size=max(
                    1,
                    self.batch_size,
                ),
                shuffle=True,
                num_workers=0,
                pin_memory=(
                    torch.cuda.is_available()
                ),
            )

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = build_resnet18_qc_model(
                nn,
                pretrained=(
                    self.initial_model_path
                    is None
                ),
            ).to(device)

            if (
                self.initial_model_path
                is not None
            ):
                if not (
                    self.initial_model_path
                    .exists()
                ):
                    raise FileNotFoundError(
                        "Loaded model does not exist:\n"
                        f"{self.initial_model_path}"
                    )

                checkpoint = torch.load(
                    self.initial_model_path,
                    map_location=device,
                )

                if not isinstance(
                    checkpoint,
                    dict,
                ):
                    raise ValueError(
                        "Unsupported loaded Image review model format."
                    )

                architecture = str(
                    checkpoint.get(
                        "architecture",
                        checkpoint.get(
                            "model_name",
                            "",
                        ),
                    )
                ).lower()

                if architecture and "resnet18" not in architecture:
                    raise ValueError(
                        "Loaded Image review model is not a ResNet18 checkpoint.\n\n"
                        f"Checkpoint architecture: {architecture}"
                    )

                state_dict = (
                    checkpoint.get(
                        "state_dict",
                        checkpoint,
                    )
                )

                model.load_state_dict(
                    state_dict
                )

                self.progress.emit(
                    (
                        "Continue training from:\n"
                        f"{self.initial_model_path}"
                    ),
                    16,
                )

            # Balanced class loss.
            n_bad = len(
                bad_paths
            )
            n_good = len(
                good_paths
            )
            total = (
                n_bad + n_good
            )

            class_weights = torch.tensor(
                [
                    total
                    / (
                        2.0
                        * max(
                            1,
                            n_bad,
                        )
                    ),
                    total
                    / (
                        2.0
                        * max(
                            1,
                            n_good,
                        )
                    ),
                ],
                dtype=torch.float32,
                device=device,
            )

            criterion = (
                nn.CrossEntropyLoss(
                    weight=class_weights
                )
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )

            self.progress.emit(
                (
                    f"ResNet18 fine-tuning\n"
                    f"TRAIN PNG: GOOD={n_good}, BAD={n_bad}\n"
                    f"Input: {QC_MODEL_INPUT_SIZE}x{QC_MODEL_INPUT_SIZE}\n"
                    f"Device: {device}"
                ),
                0,
            )

            for epoch in range(
                1,
                self.epochs + 1,
            ):
                self._check_stop()
                model.train()

                running_loss = 0.0
                correct = 0
                seen = 0

                for x, y in loader:
                    x = x.to(
                        device,
                        non_blocking=True,
                    )
                    y = y.to(
                        device,
                        non_blocking=True,
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    logits = model(x)
                    loss = criterion(
                        logits,
                        y,
                    )

                    loss.backward()
                    optimizer.step()

                    n_batch = int(
                        y.size(0)
                    )

                    running_loss += (
                        float(
                            loss.item()
                        )
                        * n_batch
                    )

                    pred = logits.argmax(
                        dim=1
                    )

                    correct += int(
                        (
                            pred == y
                        )
                        .sum()
                        .item()
                    )

                    seen += n_batch

                epoch_loss = (
                    running_loss
                    / max(
                        1,
                        seen,
                    )
                )

                epoch_acc = (
                    correct
                    / max(
                        1,
                        seen,
                    )
                )

                progress_value = int(
                    round(
                        5
                        + 95
                        * epoch
                        / max(
                            1,
                            self.epochs,
                        )
                    )
                )

                self.progress.emit(
                    (
                        f"Epoch "
                        f"{epoch}/{self.epochs}   "
                        f"loss={epoch_loss:.4f}   "
                        f"accuracy={epoch_acc:.4f}"
                    ),
                    progress_value,
                )

            self.output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_path = (
                self.output_folder
                / "LinkNex_QC_model.pt"
            )

            torch.save(
                {
                    "model_name": (
                        "LinkNexQC_ResNet18"
                    ),
                    "architecture": (
                        "resnet18"
                    ),
                    "pretrained_source": (
                        "ImageNet"
                    ),
                    "state_dict": (
                        model.state_dict()
                    ),
                    "input_size": (
                        QC_MODEL_INPUT_SIZE
                    ),
                    "class_to_idx": {
                        "bad": 0,
                        "good": 1,
                    },
                    "train_good": (
                        n_good
                    ),
                    "train_bad": (
                        n_bad
                    ),
                },
                model_path,
            )

            self.finished.emit(
                str(model_path),
                (
                    "TRAIN COMPLETE\n"
                    f"Train pairs: {len(pairs)}\n"
                    f"Test pairs: {len(test_pairs)}\n"
                    f"Training history CSV:\n{metrics_csv}\n"
                    f"Loss curve:\n{self.dataset_root / SEMANTIC_TRAIN_LOSS_CURVE_PNG}\n"
                    f"Dice curve:\n{self.dataset_root / SEMANTIC_TRAIN_DICE_CURVE_PNG}\n"
                    f"Per-epoch Test results:\n{preview_root}\n"
                    f"Model:\n{model_path}"
                ),
            )

        except Exception as exc:
            self.failed.emit(
                str(exc)
            )


class QCTestWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        output_folder,
        good_folder,
        bad_folder,
        model_path,
    ):
        super().__init__()

        self.output_folder = Path(
            output_folder
        )
        self.good_folder = Path(
            good_folder
        )
        self.bad_folder = Path(
            bad_folder
        )
        self.model_path = Path(
            model_path
        )

    def run(self):
        try:
            (
                torch,
                model,
                device,
                input_size,
            ) = load_qc_checkpoint(
                self.model_path
            )

            good_paths = (
                collect_png_files(
                    self.good_folder
                )
            )
            bad_paths = (
                collect_png_files(
                    self.bad_folder
                )
            )

            if not good_paths:
                raise ValueError(
                    "No PNG images found in TEST GOOD folder."
                )

            if not bad_paths:
                raise ValueError(
                    "No PNG images found in TEST BAD folder."
                )

            samples = (
                [
                    (
                        p,
                        "good",
                    )
                    for p in good_paths
                ]
                +
                [
                    (
                        p,
                        "bad",
                    )
                    for p in bad_paths
                ]
            )

            rows = []

            gg = 0
            gb = 0
            bb = 0
            bg = 0

            for index, (
                path,
                true_label,
            ) in enumerate(
                samples,
                1,
            ):
                rgb = read_qc_rgb(
                    path
                )

                (
                    pred_label,
                    p_good,
                    p_bad,
                ) = predict_qc_loaded(
                    torch,
                    model,
                    device,
                    input_size,
                    rgb,
                )

                correct = (
                    pred_label
                    == true_label
                )

                if (
                    true_label
                    == "good"
                ):
                    if pred_label == "good":
                        gg += 1
                    else:
                        gb += 1
                else:
                    if pred_label == "bad":
                        bb += 1
                    else:
                        bg += 1

                rows.append(
                    {
                        "file": str(path),
                        "true_label": (
                            true_label
                        ),
                        "prediction": (
                            pred_label
                        ),
                        "prob_good": (
                            f"{p_good:.6f}"
                        ),
                        "prob_bad": (
                            f"{p_bad:.6f}"
                        ),
                        "correct": (
                            "yes"
                            if correct
                            else "no"
                        ),
                    }
                )

                self.progress.emit(
                    (
                        f"{path.name} -> "
                        f"{pred_label.upper()} "
                        f"(Pgood={p_good:.3f})"
                    ),
                    int(
                        round(
                            100
                            * index
                            / len(samples)
                        )
                    ),
                )

            correct_n = (
                gg + bb
            )

            accuracy = (
                correct_n
                / max(
                    1,
                    len(samples),
                )
            )

            self.output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            result_csv = (
                self.output_folder
                / "QC_test_results.csv"
            )

            with open(
                result_csv,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "file",
                        "true_label",
                        "prediction",
                        "prob_good",
                        "prob_bad",
                        "correct",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    rows
                )

            lines = [
                "TEST COMPLETE",
                f"Images: {len(samples)}",
                (
                    f"Accuracy: "
                    f"{accuracy:.4f}"
                ),
                "",
                "Confusion matrix:",
                (
                    f"GOOD → GOOD: "
                    f"{gg}"
                ),
                (
                    f"GOOD → BAD : "
                    f"{gb}"
                ),
                (
                    f"BAD  → BAD : "
                    f"{bb}"
                ),
                (
                    f"BAD  → GOOD: "
                    f"{bg}"
                ),
                "",
                (
                    "Results CSV:\n"
                    f"{result_csv}"
                ),
                "",
                "Predictions:",
            ]

            for row in rows:
                lines.append(
                    (
                        f"{Path(row['file']).name}: "
                        f"{row['prediction'].upper()} "
                        f"(true={row['true_label']}, "
                        f"Pgood={row['prob_good']})"
                    )
                )

            self.finished.emit(
                "\n".join(lines)
            )

        except Exception as exc:
            self.failed.emit(
                str(exc)
            )




class QCTrainingWindow(QMainWindow):
    """
    Simplified LinkODT Image review model window.

    Fixed dataset structure:
        dataset/
        ├─ train/
        │  ├─ good/
        │  └─ bad/
        ├─ test/
        │  ├─ good/
        │  └─ bad/
        └─ LinkNex_QC_model.pt

    This window only handles:
        - Browse Dataset
        - Browse Model
        - Load model
        - Train
        - Test

    Prediction of new microscopy images stays in the main window's Automatic review.
    """

    modelReady = Signal(str)

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(
            "LinkODT Image review · ResNet18"
        )
        self.resize(
            760,
            560,
        )
        self.setMinimumSize(
            680,
            500,
        )

        self.dataset_root = None
        self.model_path = None

        self.train_thread = None
        self.train_worker = None

        self.test_thread = None
        self.test_worker = None

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )
        root.setContentsMargins(
            18,
            16,
            18,
            18,
        )
        root.setSpacing(
            10
        )

        title = QLabel(
            "Image review binary classification · ResNet18"
        )
        title.setObjectName(
            "TrainerTitle"
        )
        root.addWidget(
            title
        )

        hint = QLabel(
            "PNG only; grayscale and RGB are supported.\n"
            "Fixed structure: dataset/train/good, dataset/train/bad, "
            "dataset/test/good, dataset/test/bad."
        )
        hint.setObjectName(
            "TrainerHint"
        )
        hint.setWordWrap(
            True
        )
        root.addWidget(
            hint
        )

        # --------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------
        dataset_row = QHBoxLayout()

        dataset_label = QLabel(
            "Dataset"
        )
        dataset_label.setObjectName(
            "PathLabel"
        )
        dataset_label.setFixedWidth(
            85
        )

        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "Select dataset root..."
        )

        dataset_browse = QPushButton(
            "Browse"
        )
        dataset_browse.clicked.connect(
            self.select_dataset
        )
        dataset_browse.setFixedWidth(
            95
        )

        dataset_row.addWidget(
            dataset_label
        )
        dataset_row.addWidget(
            self.dataset_edit,
            1,
        )
        dataset_row.addWidget(
            dataset_browse
        )

        root.addLayout(
            dataset_row
        )

        # --------------------------------------------------------------
        # Model: Browse -> Load Model
        # --------------------------------------------------------------
        model_row = QHBoxLayout()

        model_label = QLabel(
            "Model"
        )
        model_label.setObjectName(
            "PathLabel"
        )
        model_label.setFixedWidth(
            85
        )

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(
            "Browse an existing .pt/.pth model..."
        )

        model_browse = QPushButton(
            "Browse"
        )
        model_browse.clicked.connect(
            self.browse_model
        )
        model_browse.setFixedWidth(
            95
        )

        self.load_model_btn = QPushButton(
            "Load model"
        )
        self.load_model_btn.setObjectName(
            "LoadModelButton"
        )
        self.load_model_btn.clicked.connect(
            self.load_model
        )
        self.load_model_btn.setFixedWidth(
            110
        )

        model_row.addWidget(
            model_label
        )
        model_row.addWidget(
            self.model_edit,
            1,
        )
        model_row.addWidget(
            model_browse
        )
        model_row.addWidget(
            self.load_model_btn
        )

        root.addLayout(
            model_row
        )

        self.model_status = QLabel(
            "Loaded model: —"
        )
        self.model_status.setObjectName(
            "TrainerMetric"
        )
        self.model_status.setWordWrap(
            True
        )
        root.addWidget(
            self.model_status
        )

        # --------------------------------------------------------------
        # Train settings
        # --------------------------------------------------------------
        settings = QHBoxLayout()

        epochs_label = QLabel(
            "Epochs"
        )
        epochs_label.setObjectName(
            "PathLabel"
        )

        self.epochs_edit = QLineEdit(
            "20"
        )
        self.epochs_edit.setAlignment(
            Qt.AlignCenter
        )
        self.epochs_edit.setFixedWidth(
            80
        )

        batch_label = QLabel(
            "Batch size"
        )
        batch_label.setObjectName(
            "PathLabel"
        )

        self.batch_edit = QLineEdit(
            "16"
        )
        self.batch_edit.setAlignment(
            Qt.AlignCenter
        )
        self.batch_edit.setFixedWidth(
            80
        )

        settings.addWidget(
            epochs_label
        )
        settings.addWidget(
            self.epochs_edit
        )
        settings.addSpacing(
            12
        )
        settings.addWidget(
            batch_label
        )
        settings.addWidget(
            self.batch_edit
        )
        settings.addStretch(
            1
        )

        root.addLayout(
            settings
        )

        # --------------------------------------------------------------
        # Train / Test only
        # --------------------------------------------------------------
        actions = QHBoxLayout()

        self.train_btn = QPushButton(
            "Train"
        )
        self.train_btn.setObjectName(
            "TrainButton"
        )
        self.train_btn.setProperty(
            "state",
            "idle",
        )
        self.train_btn.clicked.connect(
            self.start_train
        )

        self.test_btn = QPushButton(
            "Test"
        )
        self.test_btn.setObjectName(
            "TestButton"
        )
        self.test_btn.setProperty(
            "state",
            "idle",
        )
        self.test_btn.clicked.connect(
            self.start_test
        )

        actions.addStretch(
            1
        )
        actions.addWidget(
            self.train_btn
        )
        actions.addWidget(
            self.test_btn
        )

        root.addLayout(
            actions
        )

        self.progress = QProgressBar()
        self.progress.setRange(
            0,
            100,
        )
        self.progress.setValue(
            0
        )
        root.addWidget(
            self.progress
        )

        result_title = QLabel(
            "RESULTS"
        )
        result_title.setObjectName(
            "Section"
        )
        root.addWidget(
            result_title
        )

        self.log = QPlainTextEdit()
        self.log.setReadOnly(
            True
        )
        self.log.setPlaceholderText(
            "Train / Test results appear here."
        )
        root.addWidget(
            self.log,
            1,
        )

    def _apply_style(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {OFFWHITE};
                color: {TEXT};
            }}

            QWidget {{
                color: {TEXT};
                font-family: "Segoe UI";
                font-size: 14px;
            }}

            QLabel#TrainerTitle {{
                color: {DEEP_BLUE};
                font-size: 20px;
                font-weight: 600;
                letter-spacing: 0px;
            }}

            QLabel#TrainerHint {{
                color: {MUTED};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 7px;
                font-weight: 600;
            }}

            QLabel#Section {{
                color: {DEEP_BLUE};
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0px;
            }}

            QLabel#PathLabel {{
                color: {MUTED};
                font-size: 13px;
                font-weight: 600;
            }}

            QLabel#TrainerMetric {{
                color: {TEXT};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 6px;
                font-weight: 700;
            }}

            QLineEdit {{
                min-height: 32px;
                padding: 2px 7px;
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                font-weight: 700;
            }}

            QLineEdit:focus {{
                border-color: {US_BLUE};
            }}

            QPushButton {{
                min-height: 34px;
                padding: 3px 11px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {NAVY};
                font-weight: 600;
            }}

            QPushButton:hover {{
                background: #EEF7FD;
                border-color: {US_BLUE};
            }}

            QPushButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
            }}

            QPushButton#LoadModelButton {{
                background: {WHITE};
                color: {DEEP_BLUE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#LoadModelButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#TrainButton {{
                min-width: 120px;
                background: {NAVY};
                color: {WHITE};
                border: 2.5px solid {NAVY};
                font-weight: 600;
            }}

            QPushButton#TrainButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#TrainButton[state="working"] {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
                font-weight: 600;
            }}

            QPushButton#TrainButton[state="done"] {{
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
                font-weight: 600;
            }}

            QPushButton#TestButton {{
                min-width: 120px;
                background: {US_BLUE};
                color: {WHITE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#TestButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#TestButton[state="working"] {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
                font-weight: 600;
            }}

            QPushButton#TestButton[state="done"] {{
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
                font-weight: 600;
            }}

            QProgressBar {{
                min-height: 20px;
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 5px;
                text-align: center;
                font-weight: 600;
            }}

            QProgressBar::chunk {{
                background: {US_BLUE};
            }}

            QPlainTextEdit {{
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 7px;
                font-family: "Consolas";
                font-size: 13px;
            }}

            QMessageBox {{
                background: {WHITE};
                color: {TEXT};
            }}

            QMessageBox QLabel {{
                color: {TEXT};
                background: transparent;
            }}
            """ + DISABLED_STYLE
        )

    # ------------------------------------------------------------------
    # Dataset / model
    # ------------------------------------------------------------------

    def select_dataset(self):
        current = (
            self.dataset_edit.text().strip()
            or str(Path.cwd())
        )

        folder = (
            QFileDialog.getExistingDirectory(
                self,
                "Select Image review dataset root",
                current,
            )
        )

        if not folder:
            return

        self.dataset_root = Path(
            folder
        )
        self.dataset_edit.setText(
            str(self.dataset_root)
        )

        # If this dataset already contains a model, show it in the model path
        # field but do NOT silently load it. User explicitly presses Load Model.
        candidate = (
            self.dataset_root
            / "LinkNex_QC_model.pt"
        )

        if candidate.exists():
            self.model_edit.setText(
                str(candidate)
            )

    def browse_model(self):
        start_dir = (
            self.dataset_edit.text().strip()
            or str(Path.cwd())
        )

        path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Browse Image review model",
                start_dir,
                (
                    "LinkODT Image review model "
                    "(*.pt *.pth);;"
                    "All files (*.*)"
                ),
            )
        )

        if path:
            self.model_edit.setText(
                path
            )

    def load_model(self):
        text = (
            self.model_edit
            .text()
            .strip()
        )

        if not text:
            QMessageBox.warning(
                self,
                "Load model",
                "Browse a model first.",
            )
            return

        candidate = Path(
            text
        )

        try:
            load_qc_checkpoint(
                candidate
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load model",
                str(exc),
            )
            return

        self.model_path = (
            candidate
        )

        self.model_status.setText(
            f"Loaded model: {candidate}"
        )

        self.log.setPlainText(
            "MODEL LOADED\n"
            f"{candidate}"
        )

        # Immediately expose loaded model to main-window Auto QC.
        self.modelReady.emit(
            str(candidate)
        )

    def _resolve_dataset_root(self):
        text = (
            self.dataset_edit
            .text()
            .strip()
        )

        if not text:
            raise ValueError(
                "Select Dataset first."
            )

        root = Path(
            text
        )

        if not root.exists():
            raise FileNotFoundError(
                f"Dataset does not exist:\n{root}"
            )

        self.dataset_root = root
        return root

    def _training_settings(self):
        try:
            epochs = int(
                self.epochs_edit
                .text()
                .strip()
            )

            batch_size = int(
                self.batch_edit
                .text()
                .strip()
            )

        except Exception:
            raise ValueError(
                "Epochs and Batch size must be positive integers."
            )

        if epochs < 1:
            raise ValueError(
                "Epochs must be >= 1."
            )

        if batch_size < 1:
            raise ValueError(
                "Batch size must be >= 1."
            )

        return (
            epochs,
            batch_size,
        )

    def _require_model(self, dataset_root):
        if (
            self.model_path
            is not None
            and Path(
                self.model_path
            ).exists()
        ):
            return Path(
                self.model_path
            )

        candidate = (
            Path(dataset_root)
            / "LinkNex_QC_model.pt"
        )

        if candidate.exists():
            # For Test, dataset's saved model may be used automatically.
            # Browse + Load is still the explicit route for an external model.
            self.model_path = (
                candidate
            )
            self.model_edit.setText(
                str(candidate)
            )
            self.model_status.setText(
                f"Loaded model: {candidate}"
            )
            self.modelReady.emit(
                str(candidate)
            )
            return candidate

        raise ValueError(
            "Load a model or Train a model first."
        )

    # ------------------------------------------------------------------
    # Button feedback
    # ------------------------------------------------------------------

    def _refresh_button_style(self, button):
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def _set_train_button_state(self, state):
        state = str(state)

        if state == "working":
            self.train_btn.setText(
                "Training..."
            )
            self.train_btn.setProperty(
                "state",
                "working",
            )

        elif state == "done":
            self.train_btn.setText(
                "Trained ✓"
            )
            self.train_btn.setProperty(
                "state",
                "done",
            )

        else:
            self.train_btn.setText(
                "Train"
            )
            self.train_btn.setProperty(
                "state",
                "idle",
            )

        self._refresh_button_style(
            self.train_btn
        )

    def _set_test_button_state(self, state):
        state = str(state)

        if state == "working":
            self.test_btn.setText(
                "Testing..."
            )
            self.test_btn.setProperty(
                "state",
                "working",
            )

        elif state == "done":
            self.test_btn.setText(
                "Tested ✓"
            )
            self.test_btn.setProperty(
                "state",
                "done",
            )

        else:
            self.test_btn.setText(
                "Test"
            )
            self.test_btn.setProperty(
                "state",
                "idle",
            )

        self._refresh_button_style(
            self.test_btn
        )

    # ------------------------------------------------------------------
    # Busy / progress
    # ------------------------------------------------------------------

    def _set_busy(
        self,
        busy,
    ):
        enabled = not busy

        self.load_model_btn.setEnabled(
            enabled
        )
        self.train_btn.setEnabled(
            enabled
        )
        self.test_btn.setEnabled(
            enabled
        )

    def _progress(
        self,
        text,
        value,
    ):
        self.log.appendPlainText(
            str(text)
        )
        self.progress.setValue(
            int(value)
        )

    def _failed(
        self,
        message,
    ):
        self._set_busy(
            False
        )
        self._set_train_button_state(
            "idle"
        )
        self._set_test_button_state(
            "idle"
        )
        self.progress.setValue(
            0
        )

        self.log.appendPlainText(
            "\nERROR:\n"
            f"{message}"
        )

        QMessageBox.critical(
            self,
            "Image review ResNet18",
            message,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def start_train(self):
        try:
            root = (
                self._resolve_dataset_root()
            )

            good_folder = (
                root
                / "train"
                / "good"
            )

            bad_folder = (
                root
                / "train"
                / "bad"
            )

            if not good_folder.exists():
                raise FileNotFoundError(
                    "Missing folder:\n"
                    f"{good_folder}"
                )

            if not bad_folder.exists():
                raise FileNotFoundError(
                    "Missing folder:\n"
                    f"{bad_folder}"
                )

            (
                epochs,
                batch_size,
            ) = self._training_settings()

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Train",
                str(exc),
            )
            return

        self.log.clear()
        self.progress.setValue(
            0
        )

        self._set_train_button_state(
            "working"
        )
        self._set_test_button_state(
            "idle"
        )

        self._set_busy(
            True
        )

        # Keep the active Train button visually enabled-looking while the
        # worker runs; state color indicates the ongoing action.
        self.train_btn.setEnabled(True)

        self.train_thread = (
            QThread(self)
        )

        self.train_worker = (
            QCTrainWorker(
                output_folder=root,
                good_folder=good_folder,
                bad_folder=bad_folder,
                epochs=epochs,
                batch_size=batch_size,
                initial_model_path=(
                    self.model_path
                    if self.model_path
                    is not None
                    else None
                ),
            )
        )

        self.train_worker.moveToThread(
            self.train_thread
        )

        self.train_thread.started.connect(
            self.train_worker.run
        )

        self.train_worker.progress.connect(
            self._progress
        )

        self.train_worker.finished.connect(
            self._train_finished
        )
        self.train_worker.stopped.connect(
            self._train_stopped
        )
        self.train_worker.failed.connect(
            self._failed
        )
        self.train_worker.finished.connect(
            self.train_thread.quit
        )
        self.train_worker.stopped.connect(
            self.train_thread.quit
        )
        self.train_worker.failed.connect(
            self.train_thread.quit
        )

        self.train_thread.finished.connect(
            self.train_worker.deleteLater
        )

        self.train_thread.finished.connect(
            self.train_thread.deleteLater
        )

        self.train_thread.start()

    def _train_finished(
        self,
        model_path,
        summary,
    ):
        self._set_busy(
            False
        )
        self._set_train_button_state(
            "done"
        )
        self._set_test_button_state(
            "idle"
        )

        self.progress.setValue(
            100
        )

        self.model_path = Path(
            model_path
        )

        self.model_edit.setText(
            str(
                self.model_path
            )
        )

        self.model_status.setText(
            f"Loaded model: {self.model_path}"
        )

        self.log.appendPlainText(
            "\n" + summary
        )

        self.modelReady.emit(
            str(
                self.model_path
            )
        )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def start_test(self):
        try:
            root = (
                self._resolve_dataset_root()
            )

            good_folder = (
                root
                / "test"
                / "good"
            )

            bad_folder = (
                root
                / "test"
                / "bad"
            )

            if not good_folder.exists():
                raise FileNotFoundError(
                    "Missing folder:\n"
                    f"{good_folder}"
                )

            if not bad_folder.exists():
                raise FileNotFoundError(
                    "Missing folder:\n"
                    f"{bad_folder}"
                )

            model_path = (
                self._require_model(
                    root
                )
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Test",
                str(exc),
            )
            return

        self.log.clear()
        self.progress.setValue(
            0
        )

        self._set_test_button_state(
            "working"
        )
        self._set_train_button_state(
            "idle"
        )

        self._set_busy(
            True
        )

        # Keep the active Test button visually enabled-looking while running.
        self.test_btn.setEnabled(True)

        self.test_thread = (
            QThread(self)
        )

        self.test_worker = (
            QCTestWorker(
                output_folder=root,
                good_folder=good_folder,
                bad_folder=bad_folder,
                model_path=model_path,
            )
        )

        self.test_worker.moveToThread(
            self.test_thread
        )

        self.test_thread.started.connect(
            self.test_worker.run
        )

        self.test_worker.progress.connect(
            self._progress
        )

        self.test_worker.finished.connect(
            self._test_finished
        )

        self.test_worker.failed.connect(
            self._failed
        )

        self.test_worker.finished.connect(
            self.test_thread.quit
        )

        self.test_worker.failed.connect(
            self.test_thread.quit
        )

        self.test_thread.finished.connect(
            self.test_worker.deleteLater
        )

        self.test_thread.finished.connect(
            self.test_thread.deleteLater
        )

        self.test_thread.start()

    def _test_finished(
        self,
        summary,
    ):
        self._set_busy(
            False
        )
        self._set_test_button_state(
            "done"
        )
        self._set_train_button_state(
            "idle"
        )

        self.progress.setValue(
            100
        )

        self.log.setPlainText(
            summary
        )




# =============================================================================
# 2D ROI canvas
# =============================================================================

class ROICanvas(QGraphicsView):
    filesDropped = Signal(object)
    clicked = Signal(int, int)
    rectangleFinished = Signal(int, int, int, int)
    brushFinished = Signal(object, int)
    polygonFinished = Signal(object)

    MODE_SELECT = "select"
    MODE_RECT = "rectangle"
    MODE_BRUSH = "brush"
    MODE_POLYGON = "polygon"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        self.base_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.preview_item = QGraphicsPathItem()
        self.preview_item.setZValue(100)
        self.preview_item.setPen(QPen(QColor(YELLOW), 2.0))

        self.scene_obj.addItem(self.base_item)
        self.scene_obj.addItem(self.overlay_item)
        self.scene_obj.addItem(self.preview_item)

        self.mode = self.MODE_SELECT
        self.has_image = False
        self.image_width = 0
        self.image_height = 0
        self.brush_diameter = 20

        self.drawing = False
        self.points = []
        self.rect_start = None
        self.path = QPainterPath()

        self.panning = False
        self.pan_last = None

        self.setBackgroundBrush(QColor("#EEF2F6"))
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [
                Path(url.toLocalFile())
                for url in event.mimeData().urls()
                if url.isLocalFile()
            ]
            if paths:
                self.filesDropped.emit(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def set_image(self, rgb):
        self.base_item.setPixmap(rgb_to_pixmap(rgb))
        self.overlay_item.setPixmap(QPixmap())
        self.image_height, self.image_width = rgb.shape[:2]
        self.scene_obj.setSceneRect(QRectF(0, 0, self.image_width, self.image_height))
        self.has_image = True
        self.fit_image()

    def clear_image(self):
        self.base_item.setPixmap(QPixmap())
        self.overlay_item.setPixmap(QPixmap())
        self.preview_item.setPath(QPainterPath())
        self.has_image = False

    def set_overlay(self, labels):
        if labels is None:
            self.overlay_item.setPixmap(QPixmap())
        else:
            self.overlay_item.setPixmap(labels_to_pixmap(labels))

    def set_mode(self, mode):
        self.mode = mode
        self.preview_item.setPath(QPainterPath())

    def set_brush_diameter(self, value):
        self.brush_diameter = int(value)

    def fit_image(self):
        if self.has_image:
            self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self.has_image:
            return super().wheelEvent(event)

        old_pos = self.mapToScene(event.position().toPoint())
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        event.accept()

    def _xy(self, event):
        p = self.mapToScene(event.position().toPoint())
        x = max(0, min(self.image_width - 1, int(round(p.x()))))
        y = max(0, min(self.image_height - 1, int(round(p.y()))))
        return x, y, p

    def mousePressEvent(self, event):
        if not self.has_image:
            return super().mousePressEvent(event)

        if event.button() == Qt.RightButton:
            self.panning = True
            self.pan_last = event.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        x, y, p = self._xy(event)

        if self.mode == self.MODE_SELECT:
            self.clicked.emit(x, y)
            event.accept()
            return

        self.drawing = True

        if self.mode == self.MODE_RECT:
            self.rect_start = (x, y)
            self.preview_item.setPath(QPainterPath())
            event.accept()
            return

        if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            self.points = [(x, y)]
            self.path = QPainterPath(p)
            self.preview_item.setPath(self.path)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.pan_last is not None:
            delta = event.position() - self.pan_last
            self.pan_last = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return

        if not self.drawing:
            return super().mouseMoveEvent(event)

        x, y, p = self._xy(event)

        if self.mode == self.MODE_RECT and self.rect_start is not None:
            x0, y0 = self.rect_start
            path = QPainterPath()
            path.addRect(QRectF(
                min(x0, x), min(y0, y),
                abs(x - x0), abs(y - y0)
            ))
            self.preview_item.setPath(path)
            event.accept()
            return

        if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            if not self.points or (x, y) != self.points[-1]:
                self.points.append((x, y))
                self.path.lineTo(p)
                self.preview_item.setPath(self.path)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.panning:
            self.panning = False
            self.pan_last = None
            self.viewport().unsetCursor()
            event.accept()
            return

        if event.button() != Qt.LeftButton or not self.drawing:
            return super().mouseReleaseEvent(event)

        x, y, _ = self._xy(event)

        if self.mode == self.MODE_RECT and self.rect_start is not None:
            x0, y0 = self.rect_start
            self.drawing = False
            self.rect_start = None
            self.preview_item.setPath(QPainterPath())
            self.rectangleFinished.emit(
                min(x0, x), min(y0, y), max(x0, x), max(y0, y)
            )
            event.accept()
            return

        if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            if not self.points or (x, y) != self.points[-1]:
                self.points.append((x, y))

            pts = list(self.points)
            finished_mode = self.mode

            self.drawing = False
            self.points = []
            self.preview_item.setPath(QPainterPath())

            if finished_mode == self.MODE_BRUSH:
                self.brushFinished.emit(pts, self.brush_diameter)
            else:
                self.polygonFinished.emit(pts)

            event.accept()
            return

        super().mouseReleaseEvent(event)


# =============================================================================
# Image Preprocessing module
# =============================================================================

class ImagePreprocessingWidget(QWidget):
    backRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_paths = []
        self.current_index = -1
        self.current_image = None

        self.max_cyx = None
        self.composite_rgb = None

        # 2D cell-instance mask
        self.cell_labels = None
        self.selected_cell_id = 0

        self.qc_status = {}
        self.qc_model_path = None
        self.qc_predict_callback = None
        self.qc_trainer_window = None

        # Z-project display state:
        # -1 = composite; 0..C-1 = individual channel project.
        self.current_projection_index = -1
        self.projection_group = None
        self.projection_buttons = []

        self.output_root = None
        self.mask_applied = False
        self.channels_split = False

        # Current Cell Mask persistence state.
        # Drawing/editing changes only RAM until Save Cell Mask is pressed.
        self.cell_mask_saved = False

        # Undo history for Module 1 cell-mask annotation.  Snapshots are
        # zlib-compressed because masks are mostly background and compress
        # very efficiently.  History is intentionally per currently loaded
        # image and is reset when another image is loaded/saved.
        self.cell_mask_undo_stack = []
        self.cell_mask_undo_limit = 50

        # Session-wide channel-folder names.
        # Switching to another image must NOT reset names edited by the user.
        # Keys are zero-based channel indices.
        self.channel_folder_names = {}

        self.setAcceptDrops(True)

        self._build_ui()
        self._update_controls()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # LEFT
        left = QFrame()
        left.setObjectName("SidePanel")
        left.setFixedWidth(330)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        home_btn = QPushButton("← Home")
        home_btn.clicked.connect(self.backRequested.emit)
        ll.addWidget(home_btn)

        sec = QLabel("Image preprocessing")
        sec.setObjectName("Section")
        ll.addWidget(sec)

        self.open_btn = QPushButton("Open / Drag TIFF Files or Folder")
        self.open_btn.setObjectName("Primary")
        self.open_btn.clicked.connect(self.open_images)
        ll.addWidget(self.open_btn)

        self.output_btn = QPushButton("Select output folder")
        self.output_btn.clicked.connect(self.select_output_folder)
        ll.addWidget(self.output_btn)

        self.output_status = QLabel("Output: composite + mask + masked")
        self.output_status.setObjectName("Muted")
        self.output_status.setWordWrap(True)
        ll.addWidget(self.output_status)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.file_list.currentRowChanged.connect(self._row_changed)
        self.file_list.itemSelectionChanged.connect(
            self._selection_changed
        )
        ll.addWidget(self.file_list, 1)

        self.selection_label = QLabel("Selected: 0")
        self.selection_label.setObjectName("Metric")
        ll.addWidget(self.selection_label)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        ll.addLayout(nav)

        ll.addSpacing(4)

        dim_title = QLabel("Image dimensions")
        dim_title.setObjectName("Section")
        ll.addWidget(dim_title)

        self.dimension_label = QLabel("—")
        self.dimension_label.setObjectName("DimensionInfo")
        self.dimension_label.setWordWrap(True)
        self.dimension_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.dimension_label.setMinimumHeight(112)
        ll.addWidget(self.dimension_label)

        batch_title = QLabel("Batch status")
        batch_title.setObjectName("Section")
        ll.addWidget(batch_title)

        self.batch_status = QPlainTextEdit()
        self.batch_status.setObjectName("BatchStatus")
        self.batch_status.setReadOnly(True)
        self.batch_status.setMinimumHeight(125)
        self.batch_status.setMaximumHeight(175)
        self.batch_status.setPlainText(
            "Select one or more images."
        )
        ll.addWidget(self.batch_status)

        root.addWidget(left)

        # CENTER
        center = QFrame()
        center.setObjectName("InteractionFrame")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(7)

        self.image_title = QLabel("Load a TIFF image")
        self.image_title.setObjectName("CanvasTitle")
        cl.addWidget(self.image_title)

        projection_bar = QFrame()
        projection_bar.setObjectName("ProjectionBar")
        self.projection_layout = QHBoxLayout(
            projection_bar
        )
        self.projection_layout.setContentsMargins(
            0, 0, 0, 0
        )
        self.projection_layout.setSpacing(6)

        projection_label = QLabel(
            "Z maximum projection"
        )
        projection_label.setObjectName(
            "SmallLabel"
        )
        self.projection_layout.addWidget(
            projection_label
        )
        self.projection_layout.addStretch(
            1
        )

        cl.addWidget(
            projection_bar
        )

        self.canvas = ROICanvas()
        self.canvas.filesDropped.connect(self._add_dropped_paths)
        self.canvas.polygonFinished.connect(self.add_polygon_cell)
        cl.addWidget(self.canvas, 1)

        root.addWidget(center, 1)

        # RIGHT
        right = QFrame()
        right.setObjectName("SidePanel")
        right.setFixedWidth(405)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        # 1 QC
        sec = QLabel("1  Quality control")
        sec.setObjectName("Section")
        rl.addWidget(sec)

        qc_row = QHBoxLayout()
        self.good_btn = QPushButton("GOOD")
        self.good_btn.setObjectName("Good")
        self.good_btn.clicked.connect(self.mark_good)
        self.bad_btn = QPushButton("BAD")
        self.bad_btn.setObjectName("Bad")
        self.bad_btn.clicked.connect(self.mark_bad)
        qc_row.addWidget(self.good_btn)
        qc_row.addWidget(self.bad_btn)
        rl.addLayout(qc_row)

        auto_row = QHBoxLayout()
        self.select_qc_model_btn = QPushButton("Select image review model")
        self.select_qc_model_btn.clicked.connect(self.select_qc_model)
        self.auto_qc_btn = QPushButton("Automatic review")
        self.auto_qc_btn.setObjectName("AutoQC")
        self.auto_qc_btn.setProperty("state", "idle")
        self.auto_qc_btn.clicked.connect(self.auto_qc_current)
        auto_row.addWidget(self.select_qc_model_btn)
        auto_row.addWidget(self.auto_qc_btn)
        rl.addLayout(auto_row)

        self.train_qc_model_btn = QPushButton(
            "Image review model: Train / test"
        )
        self.train_qc_model_btn.setObjectName("Primary")
        self.train_qc_model_btn.clicked.connect(
            self.open_qc_trainer
        )
        rl.addWidget(self.train_qc_model_btn)

        self.qc_label = QLabel("Review: —")
        self.qc_label.setObjectName("Metric")
        rl.addWidget(self.qc_label)

        # 2 mask
        sec = QLabel("2  Cell mask")
        sec.setObjectName("Section")
        rl.addWidget(sec)

        self.polygon_mode_btn = QPushButton("Polygon")
        self.polygon_mode_btn.setObjectName("Primary")
        self.polygon_mode_btn.setCheckable(True)
        self.polygon_mode_btn.setChecked(True)
        self.polygon_mode_btn.clicked.connect(
            lambda: self.canvas.set_mode(ROICanvas.MODE_POLYGON)
        )
        rl.addWidget(self.polygon_mode_btn)

        mask_edit_row = QHBoxLayout()

        self.undo_mask_btn = QPushButton("Undo")
        self.undo_mask_btn.setToolTip("Undo the last Cell Mask edit (Ctrl+Z)")
        self.undo_mask_btn.clicked.connect(self.undo_cell_mask)
        mask_edit_row.addWidget(self.undo_mask_btn)

        self.clear_mask_btn = QPushButton("Clear Mask")
        self.clear_mask_btn.clicked.connect(self.clear_mask)
        mask_edit_row.addWidget(self.clear_mask_btn)

        rl.addLayout(mask_edit_row)

        self.save_cell_mask_btn = QPushButton("Save cell mask")
        self.save_cell_mask_btn.setObjectName("SaveMask")
        self.save_cell_mask_btn.setProperty("state", "idle")
        self.save_cell_mask_btn.clicked.connect(
            self.save_current_cell_mask
        )
        rl.addWidget(self.save_cell_mask_btn)

        self.cell_count_label = QLabel("Cells: 0")
        self.cell_count_label.setObjectName("Metric")
        rl.addWidget(self.cell_count_label)

        # 3 apply
        sec = QLabel("3  Apply mask")
        sec.setObjectName("Section")
        rl.addWidget(sec)

        self.apply_btn = QPushButton("Apply cell mask")
        self.apply_btn.setObjectName("Save")
        self.apply_btn.setProperty("state", "idle")
        self.apply_btn.clicked.connect(self.apply_cell_mask)
        rl.addWidget(self.apply_btn)

        # 4 channels
        sec = QLabel("4  Split channels")
        sec.setObjectName("Section")
        rl.addWidget(sec)

        self.channel_table = QTableWidget(0, 2)
        self.channel_table.setHorizontalHeaderLabels(
            ["Channel", "Folder name (editable)"]
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.setMinimumHeight(250)
        self.channel_table.setMaximumHeight(380)
        rl.addWidget(self.channel_table, 1)

        self.split_btn = QPushButton("Split channels")
        self.split_btn.setObjectName("Primary")
        self.split_btn.setProperty("state", "idle")
        self.split_btn.clicked.connect(self.split_channels)
        rl.addWidget(self.split_btn)

        root.addWidget(right)

        # shortcuts
        good_action = QAction(self)
        good_action.setShortcut(QKeySequence("G"))
        good_action.triggered.connect(self.mark_good)
        self.addAction(good_action)

        bad_action = QAction(self)
        bad_action.setShortcut(QKeySequence("B"))
        bad_action.triggered.connect(self.mark_bad)
        self.addAction(bad_action)

        undo_mask_action = QAction(self)
        undo_mask_action.setShortcut(QKeySequence.Undo)
        undo_mask_action.triggered.connect(self.undo_cell_mask)
        self.addAction(undo_mask_action)

    # -------------------------------------------------------------- drag/drop

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [
                Path(url.toLocalFile())
                for url in event.mimeData().urls()
                if url.isLocalFile()
            ]
            if paths:
                self._add_dropped_paths(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _collect_tiff_paths(self, incoming):
        """
        Accept individual TIFF files or whole folders.
        A dropped folder contributes TIFF files directly inside that folder.
        Generated masks/composites are ignored.
        """
        paths = []

        for raw in incoming:
            p = Path(raw)

            if p.is_file():
                if (
                    p.suffix.lower() in SUPPORTED_SUFFIXES
                    and "_mask" not in p.stem.lower()
                    and "_zmax_composite" not in p.stem.lower()
                ):
                    paths.append(p)
                continue

            if p.is_dir():
                self._load_existing_qc(p)

                for q in p.iterdir():
                    if not q.is_file():
                        continue
                    if q.suffix.lower() not in SUPPORTED_SUFFIXES:
                        continue

                    low = q.stem.lower()
                    if "_mask" in low or "_zmax_composite" in low:
                        continue

                    paths.append(q)

        return sorted(set(paths), key=natural_sort_key)

    def _set_image_paths(self, paths, append=False):
        paths = [Path(p) for p in paths]
        if not paths:
            return

        if append:
            merged = list(self.image_paths)
            seen = {str(p.resolve()) for p in merged}

            for p in paths:
                key = str(p.resolve())
                if key not in seen:
                    merged.append(p)
                    seen.add(key)

            paths = sorted(merged, key=natural_sort_key)
        else:
            paths = sorted(set(paths), key=natural_sort_key)

        self.image_paths = paths
        self.current_index = -1

        self.file_list.blockSignals(True)
        self.file_list.clear()

        for p in self.image_paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.file_list.addItem(item)

            status = self.qc_status.get(str(p), "unreviewed")
            self._color_file_row(
                self.file_list.count() - 1,
                status,
            )

        self.file_list.blockSignals(False)

        if self.image_paths:
            self.file_list.setCurrentRow(0)
            self.load_image(0)
            self._update_batch_status()

    def _add_dropped_paths(self, incoming):
        paths = self._collect_tiff_paths(incoming)

        if not paths:
            QMessageBox.information(
                self,
                "No TIFF images",
                "No .tif / .tiff images were found.",
            )
            return

        for parent in {p.parent for p in paths}:
            self._load_existing_qc(parent)

        self._set_image_paths(
            paths,
            append=bool(self.image_paths),
        )

    # -------------------------------------------------------------- selection / batch

    def _selected_rows(self):
        rows = sorted({
            self.file_list.row(item)
            for item in self.file_list.selectedItems()
        })

        if (
            not rows
            and 0 <= self.current_index < len(self.image_paths)
        ):
            rows = [self.current_index]

        return [
            row
            for row in rows
            if 0 <= row < len(self.image_paths)
        ]

    def _selected_paths(self):
        return [
            self.image_paths[row]
            for row in self._selected_rows()
        ]

    def _selection_changed(self):
        self.selection_label.setText(
            f"Selected: {len(self._selected_paths())}"
        )
        self._update_batch_status()
        self._update_controls()

    def _has_saved_cell_mask(self, image_path):
        binary_path, instance_path = (
            self._existing_cell_mask_paths(image_path)
        )
        return binary_path.exists() or instance_path.exists()

    def _has_current_cell_mask(self, image_path):
        """
        Batch prerequisites are based on SAVED masks on disk only.

        An unsaved polygon visible in the current image is intentionally not
        considered ready for batch Apply.
        """
        return self._has_saved_cell_mask(
            image_path
        )


    def _pipeline_needs_for_path(self, image_path):
        """
        Return unfinished preprocessing steps for one image.

        Cell Mask is considered complete only after Save cell mask wrote it
        to mask/ on disk.
        """
        image_path = Path(image_path)
        needs = []

        if self.qc_status.get(
            str(image_path),
            "unreviewed",
        ) != "good":
            needs.append("Review: GOOD")

        if not self._has_saved_cell_mask(image_path):
            needs.append("Save cell mask")

        if (
            self.current_image is not None
            and image_path == self.current_image.path
        ):
            applied = bool(self.mask_applied)
            split = bool(self.channels_split)
        else:
            applied = self._processed_outputs_exist(
                image_path
            )
            split = self._split_outputs_exist(
                image_path
            )

        if not applied:
            needs.append("Apply Mask")

        if not split:
            needs.append("Split channels")

        return needs


    def _update_batch_status(self, extra_lines=None):
        paths = self._selected_paths()

        if not paths:
            self.batch_status.setPlainText(
                "Select one or more images."
            )
            return

        lines = [
            f"Selected images: {len(paths)}"
        ]

        for path in paths:
            needs = self._pipeline_needs_for_path(path)

            if needs:
                lines.append(
                    f"• {path.name}: "
                    + " → ".join(needs)
                )
            else:
                lines.append(
                    f"✓ {path.name}: Complete"
                )

        if extra_lines:
            lines.append("")
            lines.extend(extra_lines)

        self.batch_status.setPlainText(
            "\n".join(lines)
        )

    # -------------------------------------------------------------- loading

    def open_images(self):
        box = QMessageBox(self)
        box.setWindowTitle("Open TIFF images")
        box.setText("Open a folder or select TIFF files?")
        folder_btn = box.addButton("Folder", QMessageBox.AcceptRole)
        files_btn = box.addButton("Files", QMessageBox.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked == cancel_btn:
            return

        paths = []

        if clicked == folder_btn:
            folder = QFileDialog.getExistingDirectory(
                self, "Open TIFF folder", str(Path.cwd())
            )
            if not folder:
                return
            folder = Path(folder)
            self._load_existing_qc(folder)
            paths = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_SUFFIXES
                and "_mask" not in p.stem.lower()
            ]

        elif clicked == files_btn:
            selected, _ = QFileDialog.getOpenFileNames(
                self,
                "Select TIFF images",
                str(Path.cwd()),
                "TIFF images (*.tif *.tiff)",
            )
            paths = [Path(p) for p in selected]
            for parent in {p.parent for p in paths}:
                self._load_existing_qc(parent)

        paths = sorted(set(paths), key=natural_sort_key)
        if not paths:
            return

        self._set_image_paths(paths, append=False)

    def _row_changed(self, row):
        if row < 0 or row >= len(self.image_paths):
            return
        if row != self.current_index:
            self.load_image(row)

    def _clear_projection_buttons(self):
        for button in self.projection_buttons:
            try:
                self.projection_layout.removeWidget(
                    button
                )
                button.deleteLater()
            except Exception:
                pass

        self.projection_buttons = []

        if self.projection_group is not None:
            try:
                self.projection_group.deleteLater()
            except Exception:
                pass

        self.projection_group = None

    def _rebuild_projection_buttons(
        self,
        channel_count,
    ):
        self._clear_projection_buttons()

        self.projection_group = (
            QButtonGroup(self)
        )
        self.projection_group.setExclusive(
            True
        )

        labels = ["Composite"] + [
            f"c{i + 1}"
            for i in range(
                int(channel_count)
            )
        ]

        # Stretch was inserted at the end when UI was created.
        stretch_index = (
            self.projection_layout.count()
            - 1
        )

        for visual_index, text in enumerate(
            labels
        ):
            button = QPushButton(
                text
            )
            button.setObjectName(
                "ProjectionButton"
            )
            button.setCheckable(
                True
            )

            projection_index = (
                -1
                if visual_index == 0
                else visual_index - 1
            )

            button.clicked.connect(
                lambda checked,
                idx=projection_index:
                self._projection_button_clicked(
                    idx,
                    checked,
                )
            )

            self.projection_group.addButton(
                button
            )

            self.projection_layout.insertWidget(
                stretch_index,
                button,
            )

            self.projection_buttons.append(
                button
            )

        if self.projection_buttons:
            self.projection_buttons[
                0
            ].setChecked(True)

        self.current_projection_index = (
            -1
        )

    def _projection_button_clicked(
        self,
        projection_index,
        checked,
    ):
        if not checked:
            return

        self.show_projection(
            projection_index
        )

    def show_projection(
        self,
        projection_index,
    ):
        if self.current_image is None:
            return

        projection_index = int(
            projection_index
        )

        if projection_index < 0:
            rgb = self.composite_rgb
            self.current_projection_index = (
                -1
            )

        else:
            if (
                self.max_cyx is None
                or projection_index
                >= self.max_cyx.shape[0]
            ):
                return

            gray = normalize_uint8(
                self.max_cyx[
                    projection_index
                ]
            )

            rgb = np.repeat(
                gray[..., None],
                3,
                axis=2,
            )

            self.current_projection_index = (
                projection_index
            )

        self.canvas.set_image(
            rgb
        )
        self.canvas.set_overlay(
            self.cell_labels
        )
        self.canvas.set_mode(
            ROICanvas.MODE_POLYGON
        )

    def _save_projection_pngs(
        self,
        image,
        max_cyx=None,
        composite_rgb=None,
    ):
        """
        Save:
            <stem>_zmax_composite.png
            <stem>_zmax_c1.png
            ...
            <stem>_zmax_cn.png
        """
        if max_cyx is None:
            max_cyx = (
                image.max_projection()
            )

        if composite_rgb is None:
            composite_rgb = (
                build_multichannel_composite(
                    max_cyx
                )
            )

        output_dir = self._composite_root(
            image.path
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stem = image.path.stem
        saved = []

        composite_path = (
            output_dir
            / f"{stem}_zmax_composite.png"
        )

        ok = cv2.imwrite(
            str(composite_path),
            cv2.cvtColor(
                composite_rgb,
                cv2.COLOR_RGB2BGR,
            ),
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                6,
            ],
        )

        if not ok:
            raise RuntimeError(
                f"Could not save:\n"
                f"{composite_path}"
            )

        saved.append(
            composite_path
        )

        for ci in range(
            int(image.c)
        ):
            gray = normalize_uint8(
                max_cyx[ci]
            )

            channel_path = (
                output_dir
                / (
                    f"{stem}_zmax_"
                    f"c{ci + 1}.png"
                )
            )

            ok = cv2.imwrite(
                str(channel_path),
                gray,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    6,
                ],
            )

            if not ok:
                raise RuntimeError(
                    f"Could not save:\n"
                    f"{channel_path}"
                )

            saved.append(
                channel_path
            )

        return saved

    def load_image(self, index):
        path = self.image_paths[index]

        try:
            image = MicroscopyImage(path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open TIFF", str(exc))
            return

        self.current_index = index
        self.current_image = image
        self.max_cyx = image.max_projection()
        self.composite_rgb = build_multichannel_composite(self.max_cyx)

        binary_mask_path, instance_mask_path = (
            self._existing_cell_mask_paths(path)
        )
        saved_mask_exists = (
            binary_mask_path.exists()
            or instance_mask_path.exists()
        )

        try:
            self.cell_labels = self._load_existing_cell_mask(image)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Existing mask could not be loaded",
                str(exc),
            )
            self.cell_labels = np.zeros(
                (image.y, image.x),
                dtype=np.int32,
            )

        self.selected_cell_id = 0
        self.cell_mask_undo_stack.clear()
        self.cell_mask_saved = bool(saved_mask_exists)
        self.mask_applied = self._processed_outputs_exist(path)
        self.channels_split = self._split_outputs_exist(path)

        self._rebuild_projection_buttons(
            image.c
        )
        self.show_projection(-1)
        self.polygon_mode_btn.setChecked(True)

        self.image_title.setText(path.name)
        self.dimension_label.setText(image.dimension_text())
        print("=" * 70)
        print(image.dimension_text())

        self._populate_channel_table(image.c)
        self._set_qc_visual(self.qc_status.get(str(path), "unreviewed"))

        self._set_save_mask_button_state(
            "saved" if self.cell_mask_saved else "idle"
        )
        self._set_apply_button_state(
            "applied" if self.mask_applied else "idle"
        )
        self._set_split_button_state(
            "split" if self.channels_split else "idle"
        )

        self._update_cell_count()
        self._update_controls()
        self._update_batch_status()

    def previous_image(self):
        if self.image_paths:
            self.file_list.setCurrentRow(max(0, self.current_index - 1))

    def next_image(self):
        if self.image_paths:
            self.file_list.setCurrentRow(
                min(len(self.image_paths) - 1, self.current_index + 1)
            )

    # -------------------------------------------------------------- output

    def select_output_folder(self):
        start = str(self.output_root) if self.output_root else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(
            self, "Select preprocessing output folder", start
        )
        if folder:
            self.output_root = Path(folder)
            self.output_status.setText(f"Output: {self.output_root}")

    def _effective_output_root(self, image_path=None):
        if self.output_root is not None:
            return self.output_root

        if image_path is not None:
            return Path(image_path).parent

        if self.current_image is not None:
            return self.current_image.path.parent

        return Path.cwd()

    def _composite_root(self, image_path=None):
        return (
            self._effective_output_root(image_path)
            / "composite"
        )

    def _mask_root(self, image_path=None):
        return (
            self._effective_output_root(image_path)
            / "mask"
        )

    def _masked_root(self, image_path=None):
        return (
            self._effective_output_root(image_path)
            / "masked"
        )


    def _existing_cell_mask_paths(self, image_path):
        stem = Path(image_path).stem
        mask_dir = self._mask_root(image_path)

        return (
            mask_dir / f"{stem}_cell_mask.tif",
            mask_dir / f"{stem}_cell_mask_instance.tif",
        )

    def _load_existing_cell_mask(self, image):
        """
        Existing mask priority:
          1) mask/<stem>_cell_mask_instance.tif
          2) mask/<stem>_cell_mask.tif
          3) empty mask

        Cell masks are always 2D YX.
        """
        binary_path, instance_path = self._existing_cell_mask_paths(
            image.path
        )

        if instance_path.exists():
            arr = np.squeeze(
                np.asarray(tifffile.imread(instance_path))
            )

            if tuple(arr.shape) != (image.y, image.x):
                raise ValueError(
                    "Existing instance mask shape mismatch.\n\n"
                    f"Image: {(image.y, image.x)}\n"
                    f"Mask: {arr.shape}\n"
                    f"{instance_path}"
                )

            if not np.issubdtype(arr.dtype, np.integer):
                arr = np.rint(arr)

            labels = arr.astype(np.int32, copy=True)
            labels[labels < 0] = 0
            return labels

        if binary_path.exists():
            arr = np.squeeze(
                np.asarray(tifffile.imread(binary_path))
            )

            if tuple(arr.shape) != (image.y, image.x):
                raise ValueError(
                    "Existing binary mask shape mismatch.\n\n"
                    f"Image: {(image.y, image.x)}\n"
                    f"Mask: {arr.shape}\n"
                    f"{binary_path}"
                )

            binary = (arr > 0).astype(np.uint8)

            _, labels = cv2.connectedComponents(
                binary,
                connectivity=8,
            )
            return labels.astype(np.int32)

        return np.zeros(
            (image.y, image.x),
            dtype=np.int32,
        )

    def _processed_outputs_exist(self, image_path):
        stem = Path(image_path).stem
        masked_dir = self._masked_root(image_path)

        if not masked_dir.exists():
            return False

        return any(
            masked_dir.glob(
                f"{stem}_cell*_masked.tif"
            )
        )

    def _split_outputs_exist(self, image_path):
        """
        Find any split-channel TIFF for the source image below masked/.
        Works even when c1/c2/c3 folders were renamed.
        """
        stem = Path(image_path).stem
        masked_dir = self._masked_root(image_path)

        if not masked_dir.exists():
            return False

        for candidate in masked_dir.rglob(
            f"{stem}_cell*.tif"
        ):
            if (
                candidate.is_file()
                and candidate.parent != masked_dir
            ):
                return True

        return False

    def _delete_processed_outputs_for_image(self, image_path):
        """
        If a previously processed image is reclassified as BAD, delete all
        derived preprocessing files for that image.
        """
        image_path = Path(image_path)
        stem = image_path.stem

        composite_dir = self._composite_root(image_path)
        mask_dir = self._mask_root(image_path)
        masked_dir = self._masked_root(image_path)

        fixed_targets = [
            mask_dir / f"{stem}_cell_mask.tif",
            mask_dir / f"{stem}_cell_mask_instance.tif",
        ]

        for target in fixed_targets:
            if target.exists():
                target.unlink()

        if composite_dir.exists():
            for project_png in composite_dir.glob(
                f"{stem}_zmax_*.png"
            ):
                if project_png.is_file():
                    project_png.unlink()

        if masked_dir.exists():
            # Covers root-level masked TIFFs AND all channel folders,
            # regardless of how those folders were renamed by the user.
            for candidate in list(masked_dir.rglob("*")):
                if (
                    candidate.is_file()
                    and candidate.name.startswith(f"{stem}_")
                ):
                    candidate.unlink()

            # Remove channel folders that became empty.
            directories = sorted(
                [
                    p for p in masked_dir.rglob("*")
                    if p.is_dir()
                ],
                key=lambda p: len(p.parts),
                reverse=True,
            )

            for folder in directories:
                try:
                    folder.rmdir()
                except OSError:
                    pass

    # ------------------------------------------------------------------ QC

    def _qc_csv_path(self, parent):
        return Path(parent) / "LinkNex_QC.csv"

    def _load_existing_qc(self, parent):
        """
        If LinkNex_QC.csv can be read, show stored GOOD/BAD.
        If a row cannot be resolved, nothing is displayed for that image.
        """
        parent = Path(parent)
        path = self._qc_csv_path(
            parent
        )

        if not path.exists():
            return

        try:
            with open(
                path,
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                for row in csv.DictReader(f):
                    status = (
                        row.get(
                            "status",
                            "",
                        )
                        .strip()
                        .lower()
                    )

                    if status not in {
                        "good",
                        "bad",
                    }:
                        continue

                    raw_path = (
                        row.get(
                            "path",
                            "",
                        )
                        .strip()
                    )

                    image_name = (
                        row.get(
                            "image",
                            "",
                        )
                        .strip()
                    )

                    resolved = None

                    if raw_path:
                        candidate = Path(
                            raw_path
                        )

                        # Prefer exact current source path.
                        if candidate.parent == parent:
                            resolved = (
                                candidate
                            )

                    if (
                        resolved is None
                        and image_name
                    ):
                        resolved = (
                            parent
                            / image_name
                        )

                    if resolved is not None:
                        self.qc_status[
                            str(resolved)
                        ] = status

        except Exception:
            # User requested: if QC CSV cannot be read, simply do not show
            # a status rather than inventing one.
            return


    def _write_qc_csv(self, parent):
        path = self._qc_csv_path(parent)
        rows = []
        for p, status in self.qc_status.items():
            if Path(p).parent == Path(parent):
                rows.append({"image": Path(p).name, "path": p, "status": status})

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "path", "status"])
            writer.writeheader()
            writer.writerows(rows)

    def _set_qc_visual(self, status):
        status = str(status).lower()
        if status == "good":
            self.qc_label.setText("Review: GOOD")
            self.qc_label.setStyleSheet(f"color:{GREEN}; font-weight: 600;")
        elif status == "bad":
            self.qc_label.setText("Review: BAD")
            self.qc_label.setStyleSheet(f"color:{RED}; font-weight: 600;")
        else:
            self.qc_label.setText("Review: —")
            self.qc_label.setStyleSheet(f"color:{MUTED}; font-weight: 600;")

    def _file_item_text(
        self,
        image_path,
        status,
    ):
        image_path = Path(
            image_path
        )
        status = str(
            status
        ).lower()

        if status == "good":
            return (
                f"{image_path.name}    GOOD"
            )

        if status == "bad":
            return (
                f"{image_path.name}    BAD"
            )

        return image_path.name

    def _color_file_row(self, row, status):
        item = self.file_list.item(row)
        if item is None:
            return

        if 0 <= row < len(self.image_paths):
            item.setText(
                self._file_item_text(
                    self.image_paths[row],
                    status,
                )
            )

        if status == "good":
            item.setBackground(QColor("#E8F6EE"))
            item.setForeground(QColor(TEXT))
        elif status == "bad":
            item.setBackground(QColor("#FCECEC"))
            item.setForeground(QColor(TEXT))
        else:
            item.setBackground(QColor(WHITE))
            item.setForeground(QColor(TEXT))

    def mark_good(self):
        paths = self._selected_paths()

        if not paths:
            return

        touched_parents = set()

        for path in paths:
            path = Path(path)
            self.qc_status[str(path)] = "good"
            touched_parents.add(path.parent)

            try:
                row = self.image_paths.index(path)
                self._color_file_row(
                    row,
                    "good",
                )
            except ValueError:
                pass

        for parent in touched_parents:
            self._write_qc_csv(parent)

        if self.current_image is not None:
            current_status = self.qc_status.get(
                str(self.current_image.path),
                "unreviewed",
            )
            self._set_qc_visual(current_status)

        self._update_batch_status()
        self._update_controls()

    def _move_one_bad_image(self, old_path):
        """
        Destructive BAD action for one file, without showing a confirmation.
        Returns (success, error_message).
        """
        old_path = Path(old_path)
        parent = old_path.parent
        bad_dir = parent / "QC_Bad"

        try:
            self._delete_processed_outputs_for_image(
                old_path
            )
        except Exception as exc:
            return (
                False,
                f"{old_path.name}: cleanup failed: {exc}",
            )

        bad_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination = unique_destination(
            bad_dir / old_path.name
        )

        try:
            shutil.move(
                str(old_path),
                str(destination),
            )
        except Exception as exc:
            return (
                False,
                f"{old_path.name}: move failed: {exc}",
            )

        self.qc_status[str(old_path)] = "bad"
        return True, ""

    def mark_bad(self):
        paths = self._selected_paths()

        if not paths:
            return

        preview = "\n".join(
            f"• {p.name}"
            for p in paths[:15]
        )

        if len(paths) > 15:
            preview += (
                f"\n... and {len(paths) - 15} more"
            )

        answer = QMessageBox.question(
            self,
            "Move selected images to QC_Bad",
            (
                f"Move {len(paths)} selected image(s) to QC_Bad?\n\n"
                "All existing composite, mask, masked, and split-channel "
                "outputs for these images will also be deleted.\n\n"
                f"{preview}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        selected_set = {
            Path(p)
            for p in paths
        }

        errors = []
        successfully_removed = []
        touched_parents = set()

        for path in paths:
            path = Path(path)
            ok, error = self._move_one_bad_image(
                path
            )

            if ok:
                successfully_removed.append(path)
                touched_parents.add(path.parent)
            else:
                errors.append(error)

        for parent in touched_parents:
            self._write_qc_csv(parent)

        if successfully_removed:
            removed_set = set(
                successfully_removed
            )

            old_current_path = (
                self.current_image.path
                if self.current_image is not None
                else None
            )

            old_indices = [
                i
                for i, p in enumerate(self.image_paths)
                if p in removed_set
            ]
            preferred_index = (
                min(old_indices)
                if old_indices
                else 0
            )

            remaining = [
                p
                for p in self.image_paths
                if p not in removed_set
            ]

            self.image_paths = remaining
            self.current_index = -1
            self.current_image = None
            self.max_cyx = None
            self.composite_rgb = None
            self.cell_labels = None
            self.selected_cell_id = 0
            self.cell_mask_undo_stack.clear()
            self.cell_mask_saved = False
            self.mask_applied = False
            self.channels_split = False

            self.file_list.blockSignals(True)
            self.file_list.clear()

            for p in self.image_paths:
                item = QListWidgetItem(p.name)
                item.setToolTip(str(p))
                self.file_list.addItem(item)

                status = self.qc_status.get(
                    str(p),
                    "unreviewed",
                )
                self._color_file_row(
                    self.file_list.count() - 1,
                    status,
                )

            self.file_list.blockSignals(False)

            if self.image_paths:
                next_index = min(
                    preferred_index,
                    len(self.image_paths) - 1,
                )
                self.file_list.setCurrentRow(
                    next_index
                )
                self.load_image(next_index)
            else:
                self.canvas.clear_image()
                self.image_title.setText("No images")
                self.dimension_label.setText("—")
                self.qc_label.setText("Review: —")
                self.channel_table.setRowCount(0)
                self._set_save_mask_button_state(
                    "idle"
                )
                self._set_apply_button_state(
                    "idle"
                )
                self._set_split_button_state(
                    "idle"
                )

        if errors:
            QMessageBox.critical(
                self,
                "Some BAD images could not be moved",
                "\n".join(errors),
            )

        self._update_batch_status(
            [
                (
                    f"BAD moved: "
                    f"{len(successfully_removed)}/{len(paths)}"
                )
            ]
        )
        self._update_controls()


    def _set_auto_qc_button_state(self, state):
        state = str(state)

        if state == "working":
            self.auto_qc_btn.setText("Automatic review...")
            self.auto_qc_btn.setProperty("state", "working")
        elif state == "done":
            self.auto_qc_btn.setText("Image review Done ✓")
            self.auto_qc_btn.setProperty("state", "done")
        else:
            self.auto_qc_btn.setText("Automatic review")
            self.auto_qc_btn.setProperty("state", "idle")

        style = self.auto_qc_btn.style()
        style.unpolish(self.auto_qc_btn)
        style.polish(self.auto_qc_btn)
        self.auto_qc_btn.update()

    def open_qc_trainer(self):
        if (
            self.qc_trainer_window
            is None
        ):
            self.qc_trainer_window = (
                QCTrainingWindow(self)
            )

            self.qc_trainer_window.modelReady.connect(
                self._qc_model_trained
            )

        self.qc_trainer_window.show()
        self.qc_trainer_window.raise_()
        self.qc_trainer_window.activateWindow()

    def _qc_model_trained(
        self,
        model_path,
    ):
        self.qc_model_path = Path(
            model_path
        )

        self.select_qc_model_btn.setText(
            f"Model: "
            f"{self.qc_model_path.name}"
        )

        self.select_qc_model_btn.setToolTip(
            str(self.qc_model_path)
        )

        self._update_controls()

    def select_qc_model(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select image review model",
            str(Path.cwd()),
            "PyTorch Image review model (*.pt *.pth);;All files (*.*)",
        )

        if not p:
            return

        candidate = Path(p)

        try:
            load_qc_checkpoint(candidate)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Select image review model",
                str(exc),
            )
            return

        self.qc_model_path = candidate
        self.select_qc_model_btn.setText(
            f"Model: {self.qc_model_path.name}"
        )
        self.select_qc_model_btn.setToolTip(
            str(self.qc_model_path)
        )
        self._update_controls()

    def set_qc_predict_callback(self, callback):
        """Future model hook. callback(model_path, composite_rgb, microscopy_image)."""
        self.qc_predict_callback = callback

    def auto_qc_current(self):
        # Auto QC always classifies ALL images currently listed in the
        # left sidebar. No Ctrl/Shift selection is required.
        paths = list(self.image_paths)

        if not paths:
            QMessageBox.information(
                self,
                "Automatic review",
                "Open images first.",
            )
            return

        if self.qc_model_path is None:
            QMessageBox.information(
                self,
                "Select image review model",
                (
                    "Select an existing Image review model or "
                    "use Train / Test Image review model first."
                ),
            )
            return

        builtin = (
            self.qc_predict_callback
            is None
        )

        if builtin:
            try:
                (
                    torch,
                    model,
                    device,
                    input_size,
                ) = load_qc_checkpoint(
                    self.qc_model_path
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Image review model",
                    str(exc),
                )
                return

        self._set_auto_qc_button_state("working")
        QApplication.processEvents()

        good_paths = []
        bad_paths = []
        errors = []

        for path in paths:
            path = Path(path)

            try:
                if (
                    self.current_image
                    is not None
                    and path
                    == self.current_image.path
                ):
                    image = (
                        self.current_image
                    )
                    composite = (
                        self.composite_rgb
                    )
                else:
                    image = (
                        MicroscopyImage(
                            path
                        )
                    )
                    composite = (
                        build_multichannel_composite(
                            image.max_projection()
                        )
                    )

                if builtin:
                    result, _, _ = (
                        predict_qc_loaded(
                            torch,
                            model,
                            device,
                            input_size,
                            composite,
                        )
                    )
                else:
                    result = (
                        self.qc_predict_callback(
                            self.qc_model_path,
                            composite,
                            image,
                        )
                    )

                    if isinstance(
                        result,
                        bool,
                    ):
                        result = (
                            "good"
                            if result
                            else "bad"
                        )

                    result = str(
                        result
                    ).lower()

                if result == "good":
                    good_paths.append(path)
                elif result == "bad":
                    bad_paths.append(path)
                else:
                    errors.append(
                        (
                            f"{path.name}: "
                            f"unsupported result "
                            f"{result}"
                        )
                    )

            except Exception as exc:
                errors.append(
                    f"{path.name}: {exc}"
                )

        touched_parents = set()

        for path in good_paths:
            self.qc_status[
                str(path)
            ] = "good"
            touched_parents.add(
                path.parent
            )

            try:
                row = (
                    self.image_paths
                    .index(path)
                )
                self._color_file_row(
                    row,
                    "good",
                )
            except ValueError:
                pass

        for path in bad_paths:
            self.qc_status[
                str(path)
            ] = "bad"
            touched_parents.add(
                path.parent
            )

            try:
                row = (
                    self.image_paths
                    .index(path)
                )
                self._color_file_row(
                    row,
                    "bad",
                )
            except ValueError:
                pass

        for parent in touched_parents:
            self._write_qc_csv(
                parent
            )

        if self.current_image is not None:
            self._set_qc_visual(
                self.qc_status.get(
                    str(
                        self.current_image
                        .path
                    ),
                    "",
                )
            )

        extra = [
            (
                f"Automatic review GOOD: "
                f"{len(good_paths)}"
            ),
            (
                f"Automatic review BAD: "
                f"{len(bad_paths)}"
            ),
        ]

        if errors:
            extra.append(
                f"Automatic review errors: "
                f"{len(errors)}"
            )
            extra.extend(
                f"  {line}"
                for line in errors
            )

        self._set_auto_qc_button_state("done")
        self._update_batch_status(
            extra
        )
        self._update_controls()


    def _is_good(self):
        return (
            self.current_image is not None
            and self.qc_status.get(str(self.current_image.path), "unreviewed") == "good"
        )

    def _require_good(self):
        if self._is_good():
            return True
        QMessageBox.information(
            self,
            "Image review required",
            "Image review must be completed first.\n\nMark this image as GOOD before drawing or applying a cell mask.",
        )
        return False

    # -------------------------------------------------------------- cell mask

    def _invalidate_applied_mask(self):
        # Polygon edits exist only in RAM until Save Cell Mask is pressed.
        self.cell_mask_saved = False
        self.mask_applied = False
        self.channels_split = False

        self._set_save_mask_button_state("idle")
        self._set_apply_button_state("idle")
        self._set_split_button_state("idle")

        self._update_controls()
        self._update_batch_status()

    def _push_cell_mask_undo(self):
        """Save the current Module 1 mask state before one edit operation."""
        if self.cell_labels is None:
            return

        arr = np.ascontiguousarray(self.cell_labels, dtype=np.int32)
        payload = zlib.compress(arr.tobytes(order="C"), level=1)

        self.cell_mask_undo_stack.append(
            {
                "shape": tuple(arr.shape),
                "payload": payload,
                "selected_cell_id": int(self.selected_cell_id),
                "cell_mask_saved": bool(self.cell_mask_saved),
                "mask_applied": bool(self.mask_applied),
                "channels_split": bool(self.channels_split),
            }
        )

        if len(self.cell_mask_undo_stack) > int(self.cell_mask_undo_limit):
            del self.cell_mask_undo_stack[0]

        self._update_controls()

    def undo_cell_mask(self):
        """Undo the most recent Cell Mask annotation edit (also Ctrl+Z)."""
        if self.cell_labels is None or not self.cell_mask_undo_stack:
            return

        state = self.cell_mask_undo_stack.pop()

        raw = zlib.decompress(state["payload"])
        restored = np.frombuffer(raw, dtype=np.int32).copy()
        restored = restored.reshape(state["shape"])

        if restored.shape != self.cell_labels.shape:
            # This should never happen because history is reset on image load,
            # but fail safely rather than applying history to the wrong image.
            self.cell_mask_undo_stack.clear()
            self._update_controls()
            return

        self.cell_labels = restored
        self.selected_cell_id = int(state["selected_cell_id"])
        self.cell_mask_saved = bool(state["cell_mask_saved"])
        self.mask_applied = bool(state["mask_applied"])
        self.channels_split = bool(state["channels_split"])

        self.canvas.set_overlay(self.cell_labels)
        self._update_cell_count()

        self._set_save_mask_button_state(
            "saved" if self.cell_mask_saved else "idle"
        )
        self._set_apply_button_state(
            "applied" if self.mask_applied else "idle"
        )
        self._set_split_button_state(
            "split" if self.channels_split else "idle"
        )

        self._update_controls()
        self._update_batch_status()

    def _next_cell_id(self):
        return int(self.cell_labels.max()) + 1 if self.cell_labels is not None else 1

    def add_rectangle_cell(self, x0, y0, x1, y1):
        if not self._require_good() or self.cell_labels is None:
            return
        if x1 <= x0 or y1 <= y0:
            return

        roi = self.cell_labels[y0:y1 + 1, x0:x1 + 1]
        if not np.any(roi == 0):
            return

        self._push_cell_mask_undo()
        new_id = self._next_cell_id()
        roi[roi == 0] = new_id
        self.selected_cell_id = new_id
        self._mask_changed()

    def add_polygon_cell(self, points):
        if not self._require_good() or self.cell_labels is None or not points:
            return

        pts = np.asarray(points, dtype=np.int32)
        if len(np.unique(pts, axis=0)) < 3:
            return

        h, w = self.cell_labels.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)

        add = (mask > 0) & (self.cell_labels == 0)
        if not np.any(add):
            return

        self._push_cell_mask_undo()
        new_id = self._next_cell_id()
        self.cell_labels[add] = new_id
        self.selected_cell_id = new_id
        self._mask_changed()

    def add_brush_cell(self, points, diameter):
        if not self._require_good() or self.cell_labels is None or not points:
            return

        h, w = self.cell_labels.shape
        pts = np.asarray(points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        stroke = np.zeros((h, w), dtype=np.uint8)
        if len(pts) == 1:
            cv2.circle(stroke, tuple(pts[0]), max(1, diameter // 2), 1, -1)
        else:
            cv2.polylines(
                stroke,
                [pts.reshape(-1, 1, 2)],
                False,
                1,
                max(1, int(diameter)),
                cv2.LINE_8,
            )

        add = (stroke > 0) & (self.cell_labels == 0)
        if not np.any(add):
            return

        self._push_cell_mask_undo()
        new_id = self._next_cell_id()
        self.cell_labels[add] = new_id
        self.selected_cell_id = new_id
        self._mask_changed()

    def select_cell_at(self, x, y):
        if self.cell_labels is None:
            return
        self.selected_cell_id = int(self.cell_labels[y, x])
        self._update_cell_count()
        self._update_controls()

    def delete_selected_cell(self):
        if self.cell_labels is None or self.selected_cell_id <= 0:
            return
        if not np.any(self.cell_labels == self.selected_cell_id):
            return
        self._push_cell_mask_undo()
        self.cell_labels[self.cell_labels == self.selected_cell_id] = 0
        self.selected_cell_id = 0
        self._mask_changed()

    def clear_mask(self):
        if self.cell_labels is None or not np.any(self.cell_labels > 0):
            return
        answer = QMessageBox.question(
            self,
            "Clear Mask",
            "Clear all cell ROIs for this image?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._push_cell_mask_undo()
            self.cell_labels[:] = 0
            self.selected_cell_id = 0
            self._mask_changed()

    def _mask_changed(self):
        self.canvas.set_overlay(self.cell_labels)
        self._update_cell_count()
        self._invalidate_applied_mask()

    def _update_cell_count(self):
        count = 0
        if self.cell_labels is not None:
            count = int(np.sum(np.unique(self.cell_labels) > 0))

        self.cell_count_label.setText(f"Cells: {count}")

    # -------------------------------------------------------------- channel names

    def _populate_channel_table(self, count):
        self.channel_table.setRowCount(count)

        for ci in range(count):
            channel_id = f"c{ci + 1}"

            if ci not in self.channel_folder_names:
                self.channel_folder_names[ci] = channel_id

            fixed = QTableWidgetItem(channel_id)
            fixed.setFlags(
                fixed.flags() & ~Qt.ItemIsEditable
            )
            fixed.setTextAlignment(Qt.AlignCenter)
            self.channel_table.setItem(
                ci,
                0,
                fixed,
            )

            edit = QLineEdit(
                self.channel_folder_names[ci]
            )
            edit.setObjectName("ChannelFolderEdit")
            edit.setAlignment(Qt.AlignCenter)
            edit.setToolTip(
                f"Edit output folder name for {channel_id}"
            )
            edit.textChanged.connect(
                lambda text, index=ci:
                self._channel_name_edited(
                    index,
                    text,
                )
            )
            self.channel_table.setCellWidget(
                ci,
                1,
                edit,
            )

    def _safe_channel_folder_name(self, name, index):
        name = str(name).strip()

        if not name:
            name = f"c{index + 1}"

        safe = "".join(
            ch if ch.isalnum() or ch in "-_"
            else "_"
            for ch in name
        ).strip("_")

        return safe or f"c{index + 1}"

    def channel_names_for_count(self, count):
        names = []

        for ci in range(int(count)):
            if ci not in self.channel_folder_names:
                self.channel_folder_names[ci] = (
                    f"c{ci + 1}"
                )

            names.append(
                self._safe_channel_folder_name(
                    self.channel_folder_names[ci],
                    ci,
                )
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "Channel folder names must be unique."
            )

        return names

    def channel_names(self):
        return self.channel_names_for_count(
            self.channel_table.rowCount()
        )

    def _channel_name_edited(
        self,
        channel_index,
        text,
    ):
        self.channel_folder_names[
            int(channel_index)
        ] = str(text)

        # Folder-name settings are session-wide.
        # Existing current Split result is now considered stale.
        if self.channels_split:
            self.channels_split = False
            self._set_split_button_state("idle")

        self._update_batch_status()


    # -------------------------------------------------------------- Save Cell Mask

    def save_current_cell_mask(self):
        """
        Save only the CURRENT image's 2D cell mask immediately.

        Persistent outputs:
            mask/<stem>_cell_mask.tif
            mask/<stem>_cell_mask_instance.tif

        This action does NOT Apply Mask and does NOT Split channels.
        """
        if self.current_image is None:
            return

        if self.qc_status.get(
            str(self.current_image.path),
            "unreviewed",
        ) != "good":
            QMessageBox.information(
                self,
                "Image review required",
                "Mark the current image GOOD before saving its Cell Mask.",
            )
            return

        if (
            self.cell_labels is None
            or not np.any(self.cell_labels > 0)
        ):
            QMessageBox.information(
                self,
                "No Cell Mask",
                "Draw at least one Polygon cell ROI first.",
            )
            return

        self._set_save_mask_button_state(
            "saving"
        )
        QApplication.processEvents()

        image_path = self.current_image.path
        stem = image_path.stem

        mask_dir = self._mask_root(
            image_path
        )
        mask_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            # Save the current Z-project preview set:
            # composite + every individual channel.
            self._save_projection_pngs(
                self.current_image,
                max_cyx=self.max_cyx,
                composite_rgb=self.composite_rgb,
            )

            max_id = int(
                self.cell_labels.max()
            )

            instance_dtype = (
                np.uint16
                if max_id <= 65535
                else np.uint32
            )

            binary_path = (
                mask_dir
                / f"{stem}_cell_mask.tif"
            )
            instance_path = (
                mask_dir
                / f"{stem}_cell_mask_instance.tif"
            )

            tifffile.imwrite(
                binary_path,
                (
                    self.cell_labels > 0
                ).astype(np.uint8) * 255,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )

            tifffile.imwrite(
                instance_path,
                self.cell_labels.astype(
                    instance_dtype
                ),
                photometric="minisblack",
                metadata={"axes": "YX"},
            )

            self.cell_mask_saved = True
            # Saving establishes a new on-disk baseline. Undo is intended for
            # recent unsaved annotation mistakes, so do not cross a Save boundary.
            self.cell_mask_undo_stack.clear()

            # A newly saved/changed Cell Mask invalidates old downstream
            # products until Apply and Split are run again.
            self.mask_applied = False
            self.channels_split = False

            self._set_save_mask_button_state(
                "saved"
            )
            self._set_apply_button_state(
                "idle"
            )
            self._set_split_button_state(
                "idle"
            )

            self._update_controls()
            self._update_batch_status()

        except Exception as exc:
            self.cell_mask_saved = False
            self._set_save_mask_button_state(
                "idle"
            )

            QMessageBox.critical(
                self,
                "Save cell mask failed",
                str(exc),
            )

    # -------------------------------------------------------------- button feedback

    def _set_save_mask_button_state(self, state):
        state = str(state)

        if state == "saving":
            self.save_cell_mask_btn.setText(
                "Saving..."
            )
            self.save_cell_mask_btn.setProperty(
                "state",
                "working",
            )

        elif state == "saved":
            self.save_cell_mask_btn.setText(
                "Saved ✓"
            )
            self.save_cell_mask_btn.setProperty(
                "state",
                "done",
            )

        else:
            self.save_cell_mask_btn.setText(
                "Save cell mask"
            )
            self.save_cell_mask_btn.setProperty(
                "state",
                "idle",
            )

        style = self.save_cell_mask_btn.style()
        style.unpolish(self.save_cell_mask_btn)
        style.polish(self.save_cell_mask_btn)
        self.save_cell_mask_btn.update()

    def _set_apply_button_state(self, state):
        state = str(state)

        if state == "applying":
            self.apply_btn.setText("Applying...")
            self.apply_btn.setProperty("state", "working")

        elif state == "applied":
            self.apply_btn.setText("Applied ✓")
            self.apply_btn.setProperty("state", "done")

        else:
            self.apply_btn.setText("Apply cell mask")
            self.apply_btn.setProperty("state", "idle")

        style = self.apply_btn.style()
        style.unpolish(self.apply_btn)
        style.polish(self.apply_btn)
        self.apply_btn.update()

    def _set_split_button_state(self, state):
        state = str(state)

        if state == "splitting":
            self.split_btn.setText("Splitting...")
            self.split_btn.setProperty("state", "working")

        elif state == "split":
            self.split_btn.setText("Split ✓")
            self.split_btn.setProperty("state", "done")

        else:
            self.split_btn.setText("Split channels")
            self.split_btn.setProperty("state", "idle")

        style = self.split_btn.style()
        style.unpolish(self.split_btn)
        style.polish(self.split_btn)
        self.split_btn.update()

    # -------------------------------------------------------------- Apply Mask

    def _apply_one_image(
        self,
        image,
        labels,
        composite_rgb=None,
    ):
        """
        Apply one 2D cell-instance mask to the full C,Z,Y,X image.
        Returns the number of cell instances exported.
        """
        path = image.path
        stem = path.stem

        labels = np.asarray(labels)

        if tuple(labels.shape) != (
            image.y,
            image.x,
        ):
            raise ValueError(
                f"{path.name}: mask shape "
                f"{labels.shape} does not match "
                f"{(image.y, image.x)}"
            )

        ids = np.unique(labels)
        ids = ids[ids > 0]

        if len(ids) == 0:
            raise ValueError(
                f"{path.name}: no Cell Mask"
            )

        if composite_rgb is None:
            composite_rgb = (
                build_multichannel_composite(
                    image.max_projection()
                )
            )

        composite_dir = self._composite_root(
            path
        )
        masked_dir = self._masked_root(path)

        composite_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        masked_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._save_projection_pngs(
            image,
            max_cyx=image.max_projection(),
            composite_rgb=composite_rgb,
        )

        # Remove obsolete root-level masked cells for this source before
        # rewriting them, so deleted Cell IDs cannot leave stale files.
        for old in masked_dir.glob(
            f"{stem}_cell*_masked.tif"
        ):
            old.unlink()

        for cell_id in ids:
            cell_id = int(cell_id)

            mask2d = (
                labels == cell_id
            )

            # FULL original C,Z,Y,X dimensions.
            masked_czyx = (
                image.data_czyx
                * mask2d[None, None, :, :]
            )

            cell_path = (
                masked_dir
                / f"{stem}_cell{cell_id:03d}_masked.tif"
            )

            if image.c == 1 and image.z == 1:
                save_data = masked_czyx[0, 0]
                axes = "YX"

            elif image.c == 1:
                save_data = masked_czyx[0]
                axes = "ZYX"

            elif image.z == 1:
                save_data = masked_czyx[:, 0]
                axes = "CYX"

            else:
                save_data = masked_czyx
                axes = "CZYX"

            tifffile.imwrite(
                cell_path,
                save_data,
                photometric="minisblack",
                metadata={"axes": axes},
            )

        return len(ids)

    def apply_cell_mask(self):
        """
        Batch-aware Apply Mask.

        - Polygon editing remains current-image only.
        - Apply Mask processes every selected image that already has:
            Image review = GOOD
            Cell Mask available
        - Images missing prerequisites are skipped and listed in BATCH STATUS.
        """
        paths = self._selected_paths()

        if not paths:
            return

        self._set_apply_button_state(
            "applying"
        )
        QApplication.processEvents()

        processed = []
        skipped = []
        errors = []

        for path in paths:
            path = Path(path)

            if self.qc_status.get(
                str(path),
                "unreviewed",
            ) != "good":
                skipped.append(
                    f"{path.name}: needs Image review: GOOD"
                )
                continue

            try:
                if not self._has_saved_cell_mask(path):
                    skipped.append(
                        f"{path.name}: needs Save cell mask"
                    )
                    continue

                if (
                    self.current_image is not None
                    and path == self.current_image.path
                ):
                    image = self.current_image
                    composite = self.composite_rgb
                else:
                    image = MicroscopyImage(path)
                    composite = None

                # CRITICAL:
                # Batch Apply always reloads each image's SAVED disk mask.
                labels = self._load_existing_cell_mask(
                    image
                )

                if not np.any(labels > 0):
                    skipped.append(
                        f"{path.name}: needs Save cell mask"
                    )
                    continue

                self._apply_one_image(
                    image,
                    labels,
                    composite_rgb=composite,
                )

                processed.append(path)

                if (
                    self.current_image is not None
                    and path == self.current_image.path
                ):
                    self.mask_applied = True
                    self.channels_split = False

            except Exception as exc:
                errors.append(
                    f"{path.name}: {exc}"
                )

        if processed and len(processed) == len(paths):
            self._set_apply_button_state(
                "applied"
            )
        elif processed:
            self.apply_btn.setText(
                f"Applied {len(processed)}/{len(paths)}"
            )
            self.apply_btn.setProperty(
                "state",
                "working",
            )
            style = self.apply_btn.style()
            style.unpolish(self.apply_btn)
            style.polish(self.apply_btn)
        else:
            self._set_apply_button_state(
                "idle"
            )

        if self.current_image is not None:
            current_path = self.current_image.path
            self.mask_applied = (
                self._processed_outputs_exist(
                    current_path
                )
            )

            if self.mask_applied:
                self._set_apply_button_state(
                    "applied"
                    if len(paths) == 1
                    else (
                        "applied"
                        if len(processed) == len(paths)
                        else "idle"
                    )
                )

        extra = [
            f"Apply processed: {len(processed)}/{len(paths)}"
        ]

        if skipped:
            extra.append("Skipped:")
            extra.extend(
                f"  {line}"
                for line in skipped
            )

        if errors:
            extra.append("Errors:")
            extra.extend(
                f"  {line}"
                for line in errors
            )

        self._update_batch_status(extra)
        self._update_controls()


    # -------------------------------------------------------------- Split Channels

    def _split_one_image(
        self,
        image,
        labels,
    ):
        """
        Split one selected image using the current session-wide channel-folder
        names. Full original X/Y/Z dimensions are preserved.
        """
        path = image.path
        stem = path.stem

        labels = np.asarray(labels)
        ids = np.unique(labels)
        ids = ids[ids > 0]

        if len(ids) == 0:
            raise ValueError(
                f"{path.name}: no Cell Mask"
            )

        names = self.channel_names_for_count(
            image.c
        )

        masked_dir = self._masked_root(path)
        masked_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        channel_dirs = []

        for name in names:
            folder = masked_dir / name
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )
            channel_dirs.append(folder)

        # Remove old split TIFFs for this source across any existing channel
        # folders before writing the new folder-name configuration.
        for candidate in list(
            masked_dir.rglob(
                f"{stem}_cell*.tif"
            )
        ):
            if (
                candidate.is_file()
                and candidate.parent != masked_dir
            ):
                candidate.unlink()

        exported = 0

        for cell_id in ids:
            cell_id = int(cell_id)
            mask2d = (
                labels == cell_id
            )

            for ci in range(image.c):
                stack_zyx = (
                    image.data_czyx[ci]
                )

                masked_zyx = (
                    stack_zyx
                    * mask2d[None, :, :]
                )

                name = names[ci]

                output_path = (
                    channel_dirs[ci]
                    / (
                        f"{stem}_cell"
                        f"{cell_id:03d}_{name}.tif"
                    )
                )

                if image.z == 1:
                    tifffile.imwrite(
                        output_path,
                        masked_zyx[0],
                        photometric="minisblack",
                        metadata={"axes": "YX"},
                    )
                else:
                    tifffile.imwrite(
                        output_path,
                        masked_zyx,
                        photometric="minisblack",
                        metadata={"axes": "ZYX"},
                    )

                exported += 1

        # Clean channel directories that became empty because folder names
        # were changed during this software session.
        dirs = sorted(
            [
                d
                for d in masked_dir.iterdir()
                if d.is_dir()
            ],
            key=lambda p: p.name.lower(),
        )

        for folder in dirs:
            try:
                folder.rmdir()
            except OSError:
                pass

        return exported

    def split_channels(self):
        """
        Batch-aware Split channels.

        Each selected image must already have:
          Image review = GOOD
          Cell Mask
          Apply Mask output

        Incomplete images are skipped and their required step is shown in
        BATCH STATUS.
        """
        paths = self._selected_paths()

        if not paths:
            return

        self._set_split_button_state(
            "splitting"
        )
        QApplication.processEvents()

        processed = []
        skipped = []
        errors = []

        for path in paths:
            path = Path(path)

            if self.qc_status.get(
                str(path),
                "unreviewed",
            ) != "good":
                skipped.append(
                    f"{path.name}: needs Image review: GOOD"
                )
                continue

            if (
                self.current_image is not None
                and path == self.current_image.path
            ):
                applied = bool(
                    self.mask_applied
                )
            else:
                applied = (
                    self._processed_outputs_exist(
                        path
                    )
                )

            if not applied:
                skipped.append(
                    f"{path.name}: needs Apply Mask"
                )
                continue

            try:
                if not self._has_saved_cell_mask(path):
                    skipped.append(
                        f"{path.name}: needs Save cell mask"
                    )
                    continue

                if (
                    self.current_image is not None
                    and path == self.current_image.path
                ):
                    image = self.current_image
                else:
                    image = MicroscopyImage(path)

                # Split also uses the saved disk mask, never an unsaved edit.
                labels = self._load_existing_cell_mask(
                    image
                )

                if not np.any(labels > 0):
                    skipped.append(
                        f"{path.name}: needs Save cell mask"
                    )
                    continue

                self._split_one_image(
                    image,
                    labels,
                )

                processed.append(path)

                if (
                    self.current_image is not None
                    and path == self.current_image.path
                ):
                    self.channels_split = True

            except Exception as exc:
                errors.append(
                    f"{path.name}: {exc}"
                )

        if processed and len(processed) == len(paths):
            self._set_split_button_state(
                "split"
            )
        elif processed:
            self.split_btn.setText(
                f"Split {len(processed)}/{len(paths)}"
            )
            self.split_btn.setProperty(
                "state",
                "working",
            )
            style = self.split_btn.style()
            style.unpolish(self.split_btn)
            style.polish(self.split_btn)
        else:
            self._set_split_button_state(
                "idle"
            )

        extra = [
            f"Split processed: {len(processed)}/{len(paths)}"
        ]

        if skipped:
            extra.append("Skipped:")
            extra.extend(
                f"  {line}"
                for line in skipped
            )

        if errors:
            extra.append("Errors:")
            extra.extend(
                f"  {line}"
                for line in errors
            )

        self._update_batch_status(extra)
        self._update_controls()


    # -------------------------------------------------------------- controls

    def _update_controls(self):
        has_image = self.current_image is not None
        selected_paths = self._selected_paths()
        has_targets = bool(selected_paths)
        is_good = self._is_good()

        self.prev_btn.setEnabled(
            has_image and self.current_index > 0
        )
        self.next_btn.setEnabled(
            has_image
            and self.current_index < len(self.image_paths) - 1
        )

        # QC is batch-aware.
        self.good_btn.setEnabled(
            has_targets
        )
        self.bad_btn.setEnabled(
            has_targets
        )
        self.select_qc_model_btn.setEnabled(
            has_targets
        )
        self.auto_qc_btn.setEnabled(
            bool(self.image_paths)
            and self.qc_model_path is not None
        )

        # Polygon editing belongs only to the image currently displayed.
        self.polygon_mode_btn.setEnabled(
            has_image and is_good
        )

        # PySide6 requires a native Python bool here on some systems.
        # np.any(...) returns numpy.bool_, which can raise:
        # TypeError: 'numpy.bool' object cannot be interpreted as an integer
        # when passed directly to QWidget.setEnabled().
        has_ram_mask = bool(
            has_image
            and is_good
            and self.cell_labels is not None
            and np.any(self.cell_labels > 0)
        )

        self.undo_mask_btn.setEnabled(
            bool(has_image and self.cell_mask_undo_stack)
        )

        self.clear_mask_btn.setEnabled(
            has_ram_mask
        )

        # Save Cell Mask is current-image only.
        self.save_cell_mask_btn.setEnabled(
            has_ram_mask
        )

        # Channel names are session-wide.
        self.channel_table.setEnabled(
            has_image
        )

        # Batch Apply / Split process all selected files and report skipped
        # prerequisites in BATCH STATUS.
        self.apply_btn.setEnabled(
            has_targets
        )
        self.split_btn.setEnabled(
            has_targets
        )




# =============================================================================
# Semantic Segmentation / Annotation module
# Embedded from LinkNex 2D/3D Semantic Mask Assistant v31
# =============================================================================

# =============================================================================
# Settings
# =============================================================================

SUPPORTED_IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
}

# Prefer an icon-only LinkNex logo stored next to the script.
# Fallback to a simple text placeholder if the image is not found.
SCRIPT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)

# PyInstaller ONEDIR places bundled data in sys._MEIPASS (normally _internal).
# Source mode keeps using the .py file folder.
RESOURCE_DIR = Path(
    getattr(
        sys,
        "_MEIPASS",
        Path(__file__).resolve().parent,
    )
)

LOGO_ICON_CANDIDATES = [
    RESOURCE_DIR / "assets/linkodt_app.ico",
    RESOURCE_DIR / "assets/linkodt_icon.png",
    RESOURCE_DIR / "linknex_app.ico",
    RESOURCE_DIR / "linknex_logo_round_transparent.png",
    RESOURCE_DIR / "linknex_logo_icon_only.png",
    RESOURCE_DIR / "linknex_logo_icon_only(1).png",
    RESOURCE_DIR / "linknex_logo_round.png",
    SCRIPT_DIR / "linknex_app.ico",
    SCRIPT_DIR / "linknex_logo_round_transparent.png",
]

# Full logo used INSIDE the application header.
# It already contains both the circular mark and the "LinkNex" wordmark,
# so no extra LinkNex/title text is added inside the GUI.
LOGO_FULL_CANDIDATES = [
    RESOURCE_DIR / "assets/linkodt_wordmark.png",
    RESOURCE_DIR / "linknex_logo_full_transparent.png",
    RESOURCE_DIR / "linknex_logo_transparent.png",
    RESOURCE_DIR / "linknex_biotech_network_logo(2).png",
    SCRIPT_DIR / "linknex_logo_full_transparent.png",
]

CONNECTIVITY = 8
MIN_INSTANCE_PIXELS = 1

MASK_ALPHA = 105
SELECTED_ALPHA = 190

DEFAULT_BRUSH_DIAMETER = 20
MIN_BRUSH_DIAMETER = 1
MAX_BRUSH_DIAMETER = 300

# =============================================================================
# Palette
# =============================================================================

NAVY = "#202020"
DEEP_BLUE = "#243B60"
US_BLUE = "#2458A6"
CYAN = "#3265AC"
SKY = "#53D1CF"
YELLOW = "#F6B965"
WHITE = "#FFFFFF"
OFFWHITE = "#F6F8FB"
TEXT = "#202020"
MUTED = "#616161"
BORDER = "#D2D2D2"

GRAY_SELECTED = (145, 150, 158, SELECTED_ALPHA)


# =============================================================================
# File helpers
# =============================================================================

def natural_key(text):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def is_auxiliary_mask_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_mask") or stem.endswith("_mask_instance")


def expected_mask_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.tif")


def expected_instance_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask_instance.tif")


def collect_images_from_paths(paths):
    found = []

    for raw in paths:
        p = Path(raw)

        if p.is_dir():
            for child in p.iterdir():
                if (
                    child.is_file()
                    and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                    and not is_auxiliary_mask_file(child)
                ):
                    found.append(child)

        elif (
            p.is_file()
            and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and not is_auxiliary_mask_file(p)
        ):
            found.append(p)

    unique = {}
    for p in found:
        unique[str(p.resolve())] = p

    return sorted(unique.values(), key=lambda p: natural_key(p.name))


# =============================================================================
# Image/mask I/O
# =============================================================================


class ImageVolumeReader:
    """
    Lazy reader for:
      - 2D grayscale: Y, X
      - 2D RGB/RGBA: Y, X, C
      - 3D grayscale: Z, Y, X
      - 3D RGB/RGBA: Z, Y, X, C

    TIFF stacks are memory-mapped when possible. For compressed multi-page
    TIFF files, individual Z pages are read on demand when the TIFF layout
    permits it. Only if neither route is possible is the whole TIFF loaded.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tif = None
        self._series = None
        self._memmap = None
        self._array = None
        self._page_mode = False

        suffix = self.path.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            self._tif = tifffile.TiffFile(self.path)
            self._series = self._tif.series[0]

            shape = tuple(int(v) for v in self._series.shape)
            squeezed = tuple(v for v in shape if v != 1)

            self._set_geometry_from_shape(squeezed)

            # Best case: uncompressed/contiguous TIFF -> true lazy memory map.
            try:
                mm = tifffile.memmap(self.path, series=0)
                self._memmap = np.squeeze(mm)
            except Exception:
                self._memmap = None

            # Second-best case: compressed multi-page TIFF.
            if self._memmap is None:
                pages = self._series.pages
                if self.z_count > 1 and len(pages) == self.z_count:
                    try:
                        test = np.squeeze(np.asarray(pages[0].asarray()))
                        if self.is_color:
                            expected = (self.height, self.width, self.channels)
                        else:
                            expected = (self.height, self.width)

                        if tuple(test.shape) == tuple(expected):
                            self._page_mode = True
                    except Exception:
                        self._page_mode = False

        else:
            raw = np.fromfile(str(self.path), dtype=np.uint8)
            arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

            if arr is None:
                raise ValueError(f"Cannot read image: {self.path}")

            if arr.ndim == 3:
                if arr.shape[-1] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
                elif arr.shape[-1] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            arr = np.squeeze(np.asarray(arr))
            self._array = arr
            self._set_geometry_from_shape(arr.shape)

    def _set_geometry_from_shape(self, shape):
        shape = tuple(int(v) for v in shape)

        if len(shape) == 2:
            self.z_count = 1
            self.height, self.width = shape
            self.channels = 1
            self.is_color = False
            return

        if len(shape) == 3 and shape[-1] in (3, 4):
            self.z_count = 1
            self.height, self.width, self.channels = shape
            self.is_color = True
            return

        if len(shape) == 3:
            self.z_count, self.height, self.width = shape
            self.channels = 1
            self.is_color = False
            return

        if len(shape) == 4 and shape[-1] in (3, 4):
            self.z_count, self.height, self.width, self.channels = shape
            self.is_color = True
            return

        raise ValueError(
            "Unsupported image dimensions. Expected YX, YXC, ZYX, or ZYXC "
            f"after removing singleton dimensions, but got {shape}. "
            "For multi-channel microscopy data, export/select one image channel "
            "first unless the last dimension is RGB/RGBA."
        )

    def _squeezed_source(self):
        if self._memmap is not None:
            return self._memmap

        if self._array is None:
            self._array = np.squeeze(np.asarray(self._series.asarray()))

        return self._array

    def get_slice(self, z_index: int):
        z_index = int(z_index)

        if not (0 <= z_index < self.z_count):
            raise IndexError(z_index)

        if self._page_mode:
            arr = np.squeeze(
                np.asarray(self._series.pages[z_index].asarray())
            )
            return arr

        arr = self._squeezed_source()

        if self.z_count == 1:
            return np.asarray(arr)

        return np.asarray(arr[z_index])

    def close(self):
        self._memmap = None
        self._array = None

        if self._tif is not None:
            try:
                self._tif.close()
            except Exception:
                pass

        self._tif = None
        self._series = None


class MaskVolumeReader:
    """
    Lazy TIFF reader for a binary or instance mask volume.

    Expected geometry is always represented internally as Z, Y, X.
    A 2D mask is simply Z=1.
    """

    def __init__(self, path: Path, expected_zyx, instance=False):
        self.path = Path(path)
        self.expected_zyx = tuple(int(v) for v in expected_zyx)
        self.instance = bool(instance)

        self._tif = tifffile.TiffFile(self.path)
        self._series = self._tif.series[0]
        self._memmap = None
        self._array = None
        self._page_mode = False

        raw_shape = tuple(int(v) for v in self._series.shape)
        shape = tuple(v for v in raw_shape if v != 1)

        z_count, height, width = self.expected_zyx

        if z_count == 1:
            if tuple(shape) != (height, width):
                raise ValueError(
                    f"Image/mask shape mismatch: image={(height, width)}, "
                    f"mask={shape}"
                )
        else:
            if tuple(shape) != (z_count, height, width):
                raise ValueError(
                    f"Image/mask shape mismatch: image={self.expected_zyx}, "
                    f"mask={shape}"
                )

        try:
            self._memmap = np.squeeze(
                tifffile.memmap(self.path, series=0)
            )
        except Exception:
            self._memmap = None

        if self._memmap is None:
            pages = self._series.pages

            if z_count > 1 and len(pages) == z_count:
                try:
                    test = np.squeeze(np.asarray(pages[0].asarray()))
                    if tuple(test.shape) == (height, width):
                        self._page_mode = True
                except Exception:
                    self._page_mode = False

    def get_slice(self, z_index: int):
        z_count, height, width = self.expected_zyx
        z_index = int(z_index)

        if not (0 <= z_index < z_count):
            raise IndexError(z_index)

        if self._page_mode:
            arr = np.squeeze(
                np.asarray(self._series.pages[z_index].asarray())
            )
        else:
            if self._memmap is not None:
                src = self._memmap
            else:
                if self._array is None:
                    self._array = np.squeeze(
                        np.asarray(self._series.asarray())
                    )
                src = self._array

            arr = src if z_count == 1 else src[z_index]
            arr = np.asarray(arr)

        if tuple(arr.shape) != (height, width):
            raise ValueError(
                f"Unexpected mask slice shape {arr.shape}; "
                f"expected {(height, width)}"
            )

        if self.instance:
            if not np.issubdtype(arr.dtype, np.integer):
                arr = np.rint(arr)

            labels = arr.astype(np.int32, copy=True)
            labels[labels < 0] = 0
            return labels

        return (arr > 0).astype(np.uint8)

    def close(self):
        self._memmap = None
        self._array = None

        if self._tif is not None:
            try:
                self._tif.close()
            except Exception:
                pass

        self._tif = None
        self._series = None


def normalize_to_uint8(arr):
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        return arr.copy()

    work = arr.astype(np.float32, copy=False)
    finite = np.isfinite(work)

    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = work[finite]
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.5))

    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((work - lo) / (hi - lo), 0, 1)
    out[~finite] = 0

    return (out * 255).astype(np.uint8)


def image_to_rgb8(arr):
    if arr.ndim == 2:
        g = normalize_to_uint8(arr)
        return np.repeat(g[..., None], 3, axis=2)

    rgb = arr[..., :3]
    return rgb.copy() if rgb.dtype == np.uint8 else normalize_to_uint8(rgb)



def split_binary_mask_fast(binary_mask):
    binary = (binary_mask > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=CONNECTIVITY,
    )

    labels = labels.astype(np.int32)

    if MIN_INSTANCE_PIXELS <= 1:
        sizes = {
            int(i): int(stats[i, cv2.CC_STAT_AREA])
            for i in range(1, n_labels)
        }
        return labels, sizes

    new_labels = np.zeros_like(labels, dtype=np.int32)
    sizes = {}
    next_id = 1

    for old_id in range(1, n_labels):
        count = int(stats[old_id, cv2.CC_STAT_AREA])

        if count < MIN_INSTANCE_PIXELS:
            continue

        new_labels[labels == old_id] = next_id
        sizes[next_id] = count
        next_id += 1

    return new_labels, sizes


def sizes_from_labels(labels):
    ids, counts = np.unique(labels[labels > 0], return_counts=True)

    return {
        int(i): int(c)
        for i, c in zip(ids, counts)
    }


def save_binary_mask(path, labels):
    tifffile.imwrite(
        path,
        (labels > 0).astype(np.uint8) * 255,
        photometric="minisblack",
    )


def save_instance_mask(path, labels):
    max_label = int(labels.max()) if labels.size else 0

    dtype = np.uint16 if max_label <= 65535 else np.uint32

    tifffile.imwrite(
        path,
        labels.astype(dtype),
        photometric="minisblack",
    )


# =============================================================================
# Colors and Qt conversion
# =============================================================================

def instance_color_rgba(label_id, alpha=MASK_ALPHA):
    golden = 0.618033988749895
    hue = (0.56 + label_id * golden) % 1.0
    sat = 0.72 + 0.16 * ((label_id * 37) % 3) / 2.0
    val = 0.93

    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

    return np.array(
        [int(r * 255), int(g * 255), int(b * 255), int(alpha)],
        dtype=np.uint8,
    )


def generate_lut(max_label):
    lut = np.zeros((max_label + 1, 4), dtype=np.uint8)

    for label_id in range(1, max_label + 1):
        lut[label_id] = instance_color_rgba(label_id)

    return lut


def rgb_to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape

    qimg = QImage(
        rgb.data,
        w,
        h,
        3 * w,
        QImage.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(qimg)


def rgba_to_pixmap(rgba):
    rgba = np.ascontiguousarray(rgba.astype(np.uint8))
    h, w, _ = rgba.shape

    qimg = QImage(
        rgba.data,
        w,
        h,
        4 * w,
        QImage.Format_RGBA8888,
    ).copy()

    return QPixmap.fromImage(qimg)


# =============================================================================
# Toggle switch
# =============================================================================

class ToggleSwitch(QCheckBox):
    """Single circular hide/show toggle.

    OFF = light gray
    ON  = LinkODT blue
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Click to hide/show the current Instance or Mask overlay"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if not self.isEnabled():
            fill = QColor(225, 229, 235)
            border = QColor(190, 198, 208)
        elif self.isChecked():
            fill = QColor(0, 103, 177)
            border = QColor(0, 59, 112)
        else:
            fill = QColor(215, 222, 231)
            border = QColor(160, 173, 188)

        painter.setPen(QPen(border, 2.0))
        painter.setBrush(fill)
        painter.drawEllipse(QRectF(3.0, 3.0, 22.0, 22.0))
        painter.end()


class BrushSpinBox(QSpinBox):
    """
    Brush diameter spinbox with a clear vertical blue button column:
      top    = ▲
      bottom = ▼
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Hide Qt's native spin buttons. We draw our own explicit buttons
        # so the arrows are always clearly visible on Windows.
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.up_btn = QToolButton(self)
        self.up_btn.setObjectName("BrushSpinUp")
        self.up_btn.setText("▲")
        self.up_btn.setAutoRepeat(True)
        self.up_btn.setAutoRepeatDelay(350)
        self.up_btn.setAutoRepeatInterval(70)
        self.up_btn.clicked.connect(self.stepUp)

        self.down_btn = QToolButton(self)
        self.down_btn.setObjectName("BrushSpinDown")
        self.down_btn.setText("▼")
        self.down_btn.setAutoRepeat(True)
        self.down_btn.setAutoRepeatDelay(350)
        self.down_btn.setAutoRepeatInterval(70)
        self.down_btn.clicked.connect(self.stepDown)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        button_w = 28
        margin = 2
        available_h = max(28, self.height() - 2 * margin)
        top_h = available_h // 2
        bottom_h = available_h - top_h

        x = self.width() - button_w - margin

        self.up_btn.setGeometry(
            x,
            margin,
            button_w,
            top_h,
        )

        self.down_btn.setGeometry(
            x,
            margin + top_h,
            button_w,
            bottom_h,
        )



# =============================================================================
# Canvas
# =============================================================================

class MaskCanvas(QGraphicsView):
    filesDropped = Signal(object)
    instanceClicked = Signal(int, int, bool)
    rectangleFinished = Signal(int, int, int, int, bool)
    brushFinished = Signal(object, int)
    polygonFinished = Signal(object, int)

    MODE_SELECT = "select"
    MODE_RECT = "rectangle"
    MODE_BRUSH = "brush"
    MODE_POLYGON = "polygon"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        scene = QGraphicsScene(self)
        self.setScene(scene)

        self.base_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()

        scene.addItem(self.base_item)
        scene.addItem(self.mask_item)

        self.selection_items = {}

        self.rect_item = QGraphicsRectItem()
        self.rect_item.setPen(QPen(QColor(YELLOW), 3.0))
        self.rect_item.setBrush(QColor(255, 209, 0, 28))
        self.rect_item.setZValue(20)
        self.rect_item.hide()
        scene.addItem(self.rect_item)

        self.brush_preview_item = QGraphicsPathItem()
        brush_pen = QPen(QColor(YELLOW), DEFAULT_BRUSH_DIAMETER)
        brush_pen.setCapStyle(Qt.RoundCap)
        brush_pen.setJoinStyle(Qt.RoundJoin)
        self.brush_preview_item.setPen(brush_pen)
        self.brush_preview_item.setOpacity(0.65)
        self.brush_preview_item.setZValue(25)
        self.brush_preview_item.hide()
        scene.addItem(self.brush_preview_item)

        # Yellow brush-size cursor.
        self.brush_cursor_item = QGraphicsEllipseItem()
        self.brush_cursor_item.setPen(QPen(QColor(YELLOW), 3.0))
        self.brush_cursor_item.setBrush(QColor(255, 209, 0, 28))
        self.brush_cursor_item.setZValue(30)
        self.brush_cursor_item.hide()
        scene.addItem(self.brush_cursor_item)

        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor("#F5F7FB"))

        # Faster viewport settings.
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)

        self._has_image = False
        self.mode = self.MODE_SELECT

        self.brush_diameter = DEFAULT_BRUSH_DIAMETER

        self.drag_start = None
        self.rect_ctrl = False

        self.brushing = False
        self.brush_points = []
        self.brush_path = QPainterPath()

        # Right-mouse panning state.
        self._right_panning = False
        self._pan_last_pos = None

    # ------------------------- drag/drop -------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]

        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------- graphics -------------------------

    def set_base_pixmap(self, pixmap):
        self.base_item.setPixmap(pixmap)
        self.mask_item.setPos(0, 0)
        self.scene().setSceneRect(self.base_item.boundingRect())
        self._has_image = not pixmap.isNull()

    def set_mask_pixmap(self, pixmap):
        self.mask_item.setPixmap(pixmap)
        self.mask_item.setPos(0, 0)

    def clear_selection_items(self):
        for item in list(self.selection_items.values()):
            self.scene().removeItem(item)

        self.selection_items.clear()

    def set_selection_crop(self, label_id, pixmap, x, y):
        if label_id in self.selection_items:
            return

        item = QGraphicsPixmapItem(pixmap)
        item.setPos(float(x), float(y))
        item.setZValue(10)
        self.scene().addItem(item)
        self.selection_items[label_id] = item

    def remove_selection_crop(self, label_id):
        item = self.selection_items.pop(label_id, None)

        if item is not None:
            self.scene().removeItem(item)

    def clear_all(self):
        self.clear_selection_items()

        self.base_item.setPixmap(QPixmap())
        self.mask_item.setPixmap(QPixmap())
        self.scene().setSceneRect(0, 0, 1, 1)

        self.cancel_preview()
        self.brush_cursor_item.hide()

        self._has_image = False

    def fit_image(self):
        if self._has_image:
            self.fitInView(
                self.base_item.boundingRect(),
                Qt.KeepAspectRatio,
            )

    def set_mode(self, mode):
        self.mode = mode
        self.cancel_preview()

        if mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()
            self.brush_cursor_item.hide()

    def set_brush_diameter(self, diameter):
        self.brush_diameter = int(
            np.clip(
                diameter,
                MIN_BRUSH_DIAMETER,
                MAX_BRUSH_DIAMETER,
            )
        )

        pen = self.brush_preview_item.pen()
        pen.setWidthF(float(self.brush_diameter))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self.brush_preview_item.setPen(pen)

    def cancel_preview(self):
        self.drag_start = None
        self.rect_item.setRect(QRectF())
        self.rect_item.hide()

        self.brushing = False
        self.brush_points = []
        self.brush_path = QPainterPath()
        self.brush_preview_item.setPath(self.brush_path)
        self.brush_preview_item.hide()

    def _scene_xy(self, event):
        p = self.mapToScene(event.position().toPoint())
        return p, int(np.floor(p.x())), int(np.floor(p.y()))

    def _update_brush_cursor(self, scene_pos):
        d = float(self.brush_diameter)
        r = d / 2.0

        self.brush_cursor_item.setRect(
            scene_pos.x() - r,
            scene_pos.y() - r,
            d,
            d,
        )

        if (
            self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}
            and self._has_image
        ):
            self.brush_cursor_item.show()
        else:
            self.brush_cursor_item.hide()

    # ------------------------- mouse -------------------------

    def wheelEvent(self, event):
        if not self._has_image:
            return super().wheelEvent(event)

        # Zoom around the current mouse position rather than the view center.
        # This keeps the biological structure under the cursor nearly fixed
        # while zooming, which is much more convenient for annotation.
        mouse_pos = event.position().toPoint()
        scene_pos_before = self.mapToScene(mouse_pos)

        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

        scene_pos_after = self.mapToScene(mouse_pos)
        delta = scene_pos_after - scene_pos_before

        self.translate(delta.x(), delta.y())
        event.accept()

    def mousePressEvent(self, event):
        if not self._has_image:
            return super().mousePressEvent(event)

        # Right mouse button = pan, regardless of current annotation mode.
        if event.button() == Qt.RightButton:
            self._right_panning = True
            self._pan_last_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        scene_pos, x, y = self._scene_xy(event)
        ctrl = bool(event.modifiers() & Qt.ControlModifier)

        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        if self.mode == self.MODE_SELECT:
            self.instanceClicked.emit(x, y, ctrl)
            event.accept()
            return

        if self.mode == self.MODE_RECT:
            self.drag_start = QPointF(scene_pos)
            self.rect_ctrl = ctrl

            self.rect_item.setRect(
                QRectF(self.drag_start, self.drag_start).normalized()
            )
            self.rect_item.show()

            event.accept()
            return

        if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            self.brushing = True
            self.brush_points = [(x, y)]

            self.brush_path = QPainterPath(QPointF(scene_pos))
            self.brush_preview_item.setPath(self.brush_path)
            self.brush_preview_item.show()

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._has_image:
            return super().mouseMoveEvent(event)

        # Fast right-button panning using the view scroll bars.
        if self._right_panning and self._pan_last_pos is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._pan_last_pos

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            self._pan_last_pos = current_pos
            event.accept()
            return

        scene_pos, x, y = self._scene_xy(event)

        if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
            self._update_brush_cursor(scene_pos)

        if self.mode == self.MODE_RECT and self.drag_start is not None:
            self.rect_item.setRect(
                QRectF(self.drag_start, QPointF(scene_pos)).normalized()
            )
            event.accept()
            return

        if (
            self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}
            and self.brushing
        ):
            if not self.brush_points or (x, y) != self.brush_points[-1]:
                self.brush_points.append((x, y))
                self.brush_path.lineTo(QPointF(scene_pos))
                self.brush_preview_item.setPath(self.brush_path)

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._has_image:
            return super().mouseReleaseEvent(event)

        if event.button() == Qt.RightButton and self._right_panning:
            self._right_panning = False
            self._pan_last_pos = None

            if self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}:
                self.viewport().setCursor(Qt.CrossCursor)
            else:
                self.viewport().unsetCursor()

            event.accept()
            return

        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        scene_pos, x, y = self._scene_xy(event)

        if self.mode == self.MODE_RECT and self.drag_start is not None:
            start = self.drag_start
            ctrl = self.rect_ctrl

            self.drag_start = None
            self.rect_item.setRect(QRectF())
            self.rect_item.hide()

            self.rectangleFinished.emit(
                int(np.floor(start.x())),
                int(np.floor(start.y())),
                x,
                y,
                ctrl,
            )
            event.accept()
            return

        if (
            self.mode in {self.MODE_BRUSH, self.MODE_POLYGON}
            and self.brushing
        ):
            if not self.brush_points or (x, y) != self.brush_points[-1]:
                self.brush_points.append((x, y))

            pts = list(self.brush_points)
            diameter = int(self.brush_diameter)
            finished_mode = self.mode

            self.brushing = False
            self.brush_points = []
            self.brush_path = QPainterPath()
            self.brush_preview_item.setPath(self.brush_path)
            self.brush_preview_item.hide()

            if finished_mode == self.MODE_POLYGON:
                self.polygonFinished.emit(pts, diameter)
            else:
                self.brushFinished.emit(pts, diameter)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.brush_cursor_item.hide()
        super().leaveEvent(event)


# =============================================================================
# Main GUI
# =============================================================================


# =============================================================================
# Semantic segmentation model factory: 1-channel microscopy models
# =============================================================================

SEMANTIC_MODEL_INPUT_SIZE = 448
SEMANTIC_TILE_SIZE = 448
SEMANTIC_TILE_OVERLAP = 112
SEMANTIC_TILE_STRIDE = SEMANTIC_TILE_SIZE - SEMANTIC_TILE_OVERLAP
SEMANTIC_TRAIN_PATCHES_PER_IMAGE = 4
SEMANTIC_INFER_BATCH_SIZE = 4
SEMANTIC_DEFAULT_PAIRS_TO_USE = 2000
SEMANTIC_DEFAULT_TEST_PERCENT = 10
SEMANTIC_SPLIT_SEED = 42
SEMANTIC_SPLIT_MANIFEST = "Semantic_split_manifest.csv"
SEMANTIC_MODEL_FILENAME = "LinkNex_Semantic_Model.pt"
SEMANTIC_TRAINED_MODEL_PREFIX = "LinkNex_Semantic_"
SEMANTIC_MODELS_DIR = SCRIPT_DIR / "models"
SEMANTIC_THRESHOLD = 0.5

# Optional local ImageNet ResNet18 weights for fully offline QC training.
# If neither file exists, LinkNex falls back to torchvision's normal
# pretrained-weight cache/download behavior.
QC_RESNET18_WEIGHT_CANDIDATES = [
    SCRIPT_DIR / "models" / "resnet18-f37072fd.pth",
    SCRIPT_DIR / "models" / "resnet18_imagenet.pth",
]
SEMANTIC_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Training-time visual validation.
# Every epoch, run full tiled inference on a small FIXED subset of test
# images and save compact PNG previews for visual inspection.
SEMANTIC_TRAIN_PREVIEW_EVERY_EPOCHS = 1
SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES = 12
SEMANTIC_TRAIN_PREVIEW_PANEL_SIZE = 192
SEMANTIC_TRAIN_PREVIEW_FOLDER = "test_results"
SEMANTIC_TRAIN_METRICS_CSV = "Semantic_training_metrics.csv"
SEMANTIC_TRAIN_LOSS_CURVE_PNG = "Semantic_training_loss_curve.png"
SEMANTIC_TRAIN_DICE_CURVE_PNG = "Semantic_training_dice_curve.png"

# -----------------------------------------------------------------------------
# OFFLINE DINOv2
# -----------------------------------------------------------------------------
# LinkNex never downloads DINOv2 at runtime.
#
# Recommended folder layout beside this .py file:
#
# LinkNex_folder/
# ├─ LinkNex_v30_Offline_DINOv2.py
# ├─ dinov2/
# │  ├─ hubconf.py
# │  └─ dinov2/
# │     └─ ...
# └─ models/
#    └─ dinov2_vits14_pretrain.pth
#
# The code also accepts a GitHub ZIP extracted as "dinov2-main".
DINOV2_LOCAL_REPO_CANDIDATES = [
    SCRIPT_DIR / "dinov2",
    SCRIPT_DIR / "dinov2-main",
    RESOURCE_DIR / "dinov2-main",
]

DINOV2_WEIGHT_FILENAMES = {
    "dinov2_vits14": "dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
}

SEMANTIC_DEFAULT_ARCHITECTURE = "dinov2_dpt_hr"
SEMANTIC_DEFAULT_BACKBONE = "dinov2_vits14"
SEMANTIC_HF_SEGFORMER_DIR = (
    SCRIPT_DIR / "models" / "hf_segformer_b2_ade"
)


def _semantic_model_registry():
    # Lazy import keeps LinkNex able to open even before PyTorch is imported.
    from semantic_models import MODEL_SPECS
    return MODEL_SPECS


def _semantic_backbone_registry():
    from semantic_models import DINO_BACKBONES
    return DINO_BACKBONES


def _semantic_arch_uses_dino(architecture_key):
    from semantic_models import uses_dino
    return bool(uses_dino(str(architecture_key)))


def _find_local_dinov2_weights(backbone_name=SEMANTIC_DEFAULT_BACKBONE):
    filename = DINOV2_WEIGHT_FILENAMES.get(str(backbone_name))
    if not filename:
        raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")

    candidates = [
        SCRIPT_DIR / "models" / filename,
        SCRIPT_DIR / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    expected = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Local DINOv2 weights were not found for {backbone_name}.\n\n"
        f"Required filename: {filename}\n\nAccepted locations:\n{expected}\n\n"
        "Run tools/download_pretrained_models.py to download official weights."
    )


def _find_local_dinov2_vits14_weights():
    # Backward-compatible helper used by older code paths and SSL defaults.
    return _find_local_dinov2_weights("dinov2_vits14")



def _semantic_timestamped_model_path(architecture_key=SEMANTIC_DEFAULT_ARCHITECTURE):
    """
    New LinkODT semantic checkpoints are always stored in the program-level
    ./models folder and are never allowed to overwrite older training runs.

    Filename example:
        LinkNex_Semantic_DINOv2_UNet_20260902_191532.pt
    """
    SEMANTIC_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_arch = re.sub(r"[^a-zA-Z0-9_]+", "_", str(architecture_key)).strip("_")
    candidate = (
        SEMANTIC_MODELS_DIR
        / (
            f"{SEMANTIC_TRAINED_MODEL_PREFIX}"
            f"{safe_arch}_{timestamp}.pt"
        )
    )

    # A second-precision timestamp is the normal name.
    # In the extremely unlikely event that two saves occur in the same second,
    # wait for the next available second-style filename by adding an index only
    # as a collision safeguard rather than overwriting a checkpoint.
    if not candidate.exists():
        return (
            candidate,
            timestamp,
        )

    collision_index = 2

    while True:
        collision_candidate = (
            SEMANTIC_MODELS_DIR
            / (
                f"{SEMANTIC_TRAINED_MODEL_PREFIX}"
                f"{safe_arch}_{timestamp}_{collision_index}.pt"
            )
        )

        if not collision_candidate.exists():
            return (
                collision_candidate,
                timestamp,
            )

        collision_index += 1


def _semantic_latest_trained_model():
    """
    Return the newest timestamped LinkODT semantic checkpoint in ./models.
    The DINOv2 backbone checkpoint is excluded by filename prefix.
    """
    if not SEMANTIC_MODELS_DIR.exists():
        return None

    candidates = [
        path
        for path in SEMANTIC_MODELS_DIR.glob(
            f"{SEMANTIC_TRAINED_MODEL_PREFIX}*.pt"
        )
        if path.is_file()
    ]

    if not candidates:
        return None

    # Timestamp is embedded in the filename, so lexicographic filename order
    # is chronological for YYYYMMDD_HHMMSS.
    candidates.sort(
        key=lambda p: p.name,
        reverse=True,
    )

    return candidates[0]


def _semantic_natural_key(text):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def _semantic_collect_images(folder):
    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in SEMANTIC_IMAGE_SUFFIXES
                and not p.stem.lower().endswith("_mask")
                and not p.stem.lower().endswith("_mask_instance")
            )
        ],
        key=lambda p: _semantic_natural_key(p.name),
    )


def _semantic_find_mask(mask_folder, image_path):
    """
    Match masks by either:
      image stem == mask stem
    or
      <image_stem>_mask
    """
    mask_folder = Path(mask_folder)
    image_path = Path(image_path)

    candidates = []

    for suffix in (
        ".tif",
        ".tiff",
        ".png",
        ".bmp",
        ".jpg",
        ".jpeg",
    ):
        candidates.extend(
            [
                mask_folder / f"{image_path.stem}{suffix}",
                mask_folder / f"{image_path.stem}_mask{suffix}",
            ]
        )

    for p in candidates:
        if p.exists():
            return p

    return None


def _semantic_collect_all_pairs(
    dataset_root,
    progress_callback=None,
):
    """
    Dataset structure:
        dataset/
        ├─ image/
        └─ mask/

    Only files in image/ are enumerated.
    For each image, LinkODT searches mask/ for the corresponding mask.

    Supported matching:
        image/001.tif <-> mask/001_mask.tif
    or
        image/001.tif <-> mask/001.tif
    """
    root = Path(dataset_root)

    image_dir = root / "image"
    mask_dir = root / "mask"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing folder:\n{image_dir}"
        )

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Missing folder:\n{mask_dir}"
        )

    if progress_callback is not None:
        progress_callback(
            "[1/8] Scanning image folder...",
            1,
        )

    images = _semantic_collect_images(
        image_dir
    )

    if progress_callback is not None:
        progress_callback(
            f"[1/8] Found {len(images)} image file(s).",
            2,
        )

    pairs = []
    missing = []

    total_images = max(
        1,
        len(images),
    )

    for image_index, image_path in enumerate(
        images,
        1,
    ):
        mask_path = _semantic_find_mask(
            mask_dir,
            image_path,
        )

        if mask_path is None:
            missing.append(
                image_path.name
            )
        else:
            pairs.append(
                (
                    image_path,
                    mask_path,
                )
            )

        if (
            progress_callback is not None
            and (
                image_index == 1
                or image_index == total_images
                or image_index % max(
                    1,
                    total_images // 10,
                ) == 0
            )
        ):
            progress_callback(
                (
                    "[2/8] Matching masks... "
                    f"{image_index}/{total_images} "
                    f"(matched={len(pairs)}, missing={len(missing)})"
                ),
                min(
                    4,
                    2 + int(
                        2
                        * image_index
                        / total_images
                    ),
                ),
            )

    if not pairs:
        raise ValueError(
            "No image/mask pairs found.\n\n"
            f"Image folder:\n{image_dir}\n"
            f"Mask folder:\n{mask_dir}"
        )

    if missing:
        preview = "\n".join(
            missing[:10]
        )
        suffix = (
            "\n..."
            if len(missing) > 10
            else ""
        )

        raise ValueError(
            "Some images have no matching mask.\n\n"
            "Expected either:\n"
            "image/001.tif <-> mask/001_mask.tif\n"
            "or\n"
            "image/001.tif <-> mask/001.tif\n\n"
            f"Missing examples:\n{preview}{suffix}"
        )

    return pairs


def _semantic_manifest_path(dataset_root):
    return (
        Path(dataset_root)
        / SEMANTIC_SPLIT_MANIFEST
    )


def _semantic_write_split_manifest(
    dataset_root,
    train_pairs,
    test_pairs,
    requested_pairs,
    test_percent,
    available_pairs,
):
    path = _semantic_manifest_path(
        dataset_root
    )

    rows = [
        {
            "split": "__meta__",
            "image": "requested_pairs",
            "mask": str(int(requested_pairs)),
        },
        {
            "split": "__meta__",
            "image": "test_percent",
            "mask": str(int(test_percent)),
        },
        {
            "split": "__meta__",
            "image": "available_pairs",
            "mask": str(int(available_pairs)),
        },
        {
            "split": "__meta__",
            "image": "selected_pairs",
            "mask": str(
                int(
                    len(train_pairs)
                    + len(test_pairs)
                )
            ),
        },
        {
            "split": "__meta__",
            "image": "seed",
            "mask": str(int(SEMANTIC_SPLIT_SEED)),
        },
    ]

    for split_name, pairs in (
        ("train", train_pairs),
        ("test", test_pairs),
    ):
        for image_path, mask_path in pairs:
            rows.append(
                {
                    "split": split_name,
                    "image": image_path.name,
                    "mask": mask_path.name,
                }
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
                "split",
                "image",
                "mask",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def _semantic_read_split_manifest(
    dataset_root,
    requested_pairs,
    test_percent,
    available_pairs,
):
    root = Path(dataset_root)
    path = _semantic_manifest_path(root)

    if not path.exists():
        return None

    image_dir = root / "image"
    mask_dir = root / "mask"

    metadata = {}
    train_pairs = []
    test_pairs = []

    try:
        with open(
            path,
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                split_name = str(
                    row.get("split", "")
                ).strip().lower()

                image_name = str(
                    row.get("image", "")
                ).strip()

                mask_name = str(
                    row.get("mask", "")
                ).strip()

                if split_name == "__meta__":
                    metadata[image_name] = mask_name
                    continue

                if (
                    not image_name
                    or not mask_name
                ):
                    continue

                image_path = image_dir / image_name
                mask_path = mask_dir / mask_name

                if (
                    not image_path.exists()
                    or not mask_path.exists()
                ):
                    return None

                pair = (
                    image_path,
                    mask_path,
                )

                if split_name == "train":
                    train_pairs.append(pair)
                elif split_name == "test":
                    test_pairs.append(pair)

    except Exception:
        return None

    try:
        same_settings = (
            int(
                metadata.get(
                    "requested_pairs",
                    "-1",
                )
            )
            == int(requested_pairs)
            and int(
                metadata.get(
                    "test_percent",
                    "-1",
                )
            )
            == int(test_percent)
            and int(
                metadata.get(
                    "available_pairs",
                    "-1",
                )
            )
            == int(available_pairs)
        )
    except Exception:
        return None

    if not same_settings:
        return None

    if (
        not train_pairs
        or not test_pairs
    ):
        return None

    return (
        train_pairs,
        test_pairs,
        path,
    )


def _semantic_get_or_create_split(
    dataset_root,
    pairs_to_use=SEMANTIC_DEFAULT_PAIRS_TO_USE,
    test_percent=SEMANTIC_DEFAULT_TEST_PERCENT,
    progress_callback=None,
):
    """
    Semantic dataset preparation order:

        dataset/image + dataset/mask
                ↓
        find image/mask pairs
                ↓
        dimension check ALL pairs
                ↓
        skip invalid pairs automatically
                ↓
        randomly select up to Pairs to use from VALID pairs
                ↓
        automatic Train/Test split

    Valid pair requirements:
      - image is a supported 2D YX or RGB image after singleton squeeze
      - mask is 2D YX after singleton squeeze
      - image and mask have identical YX size

    Examples:
        (1, 1200, 1200) -> squeezed to (1200, 1200) -> VALID
        (1200, 1200)    -> VALID
        (2,)             -> INVALID and skipped
        image 1200x1200 + mask 1024x1024 -> INVALID and skipped

    If requested pairs >= number of valid pairs, all valid pairs are used.

    Dimension reports:
        Semantic_dataset_dimension_check.csv
        Semantic_skipped_bad_pairs.csv
    """
    root = Path(
        dataset_root
    )

    all_pairs = (
        _semantic_collect_all_pairs(
            root,
            progress_callback=progress_callback,
        )
    )

    total_pairs = len(
        all_pairs
    )

    if total_pairs < 2:
        raise ValueError(
            "At least 2 image/mask pairs are required."
        )

    pairs_to_use = int(
        pairs_to_use
    )
    test_percent = int(
        test_percent
    )

    if pairs_to_use < 1:
        raise ValueError(
            "Pairs to use must be >= 1."
        )

    if not (
        1 <= test_percent <= 50
    ):
        raise ValueError(
            "Test % must be between 1 and 50."
        )

    dimension_csv = (
        root
        / "Semantic_dataset_dimension_check.csv"
    )

    (
        dimension_rows,
        dimension_issues,
    ) = _semantic_validate_2d_pairs(
        all_pairs,
        csv_path=dimension_csv,
        progress_callback=progress_callback,
    )

    valid_pairs = []
    skipped_rows = []

    for pair, row in zip(
        all_pairs,
        dimension_rows,
    ):
        if str(
            row.get(
                "status",
                "",
            )
        ).upper() == "OK":
            valid_pairs.append(
                pair
            )
        else:
            skipped_rows.append(
                row
            )

    valid_count = len(
        valid_pairs
    )

    skipped_count = len(
        skipped_rows
    )

    if progress_callback is not None:
        progress_callback(
            (
                "[4/8] Dimension filtering complete. "
                f"Valid={valid_count}, skipped={skipped_count}. "
                "Preparing random subset and Train/Test split..."
            ),
            9,
        )

    skipped_csv = (
        root
        / "Semantic_skipped_bad_pairs.csv"
    )

    with open(
        skipped_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fieldnames = [
            "pair_index",
            "image",
            "image_raw_shape",
            "image_squeezed_shape",
            "image_axes",
            "mask",
            "mask_raw_shape",
            "mask_squeezed_shape",
            "mask_axes",
            "status",
            "issue",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        if skipped_rows:
            writer.writerows(
                skipped_rows
            )

    if valid_count < 2:
        raise ValueError(
            "Fewer than 2 valid image/mask pairs remain after "
            "dimension checking.\n\n"
            f"Total pairs: {total_pairs}\n"
            f"Valid pairs: {valid_count}\n"
            f"Skipped pairs: {skipped_count}\n\n"
            f"Dimension report:\n{dimension_csv}"
        )

    selected_count = min(
        pairs_to_use,
        valid_count,
    )

    # Existing manifest is reusable only when:
    # - settings match
    # - valid-pair count matches
    # - every saved pair is still part of the current valid-pair set
    existing = (
        _semantic_read_split_manifest(
            root,
            requested_pairs=pairs_to_use,
            test_percent=test_percent,
            available_pairs=valid_count,
        )
    )

    valid_key_set = {
        (
            str(
                Path(image_path)
                .resolve()
            ),
            str(
                Path(mask_path)
                .resolve()
            ),
        )
        for image_path, mask_path
        in valid_pairs
    }

    if existing is not None:
        existing_train = list(
            existing[0]
        )
        existing_test = list(
            existing[1]
        )

        existing_keys = {
            (
                str(
                    Path(image_path)
                    .resolve()
                ),
                str(
                    Path(mask_path)
                    .resolve()
                ),
            )
            for image_path, mask_path
            in (
                existing_train
                + existing_test
            )
        }

        if existing_keys.issubset(
            valid_key_set
        ):
            if progress_callback is not None:
                progress_callback(
                    "[4/8] Reusing existing valid Train/Test split manifest.",
                    10,
                )

            return (
                existing_train,
                existing_test,
                existing[2],
                total_pairs,
                valid_count,
                skipped_count,
                selected_count,
                dimension_csv,
                skipped_csv,
            )

    # Fixed seed -> reproducible random subset and split.
    indices = np.arange(
        valid_count,
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        SEMANTIC_SPLIT_SEED
    )

    rng.shuffle(
        indices
    )

    selected_indices = indices[
        :selected_count
    ]

    selected_pairs = [
        valid_pairs[
            int(i)
        ]
        for i in selected_indices
    ]

    n_test = int(
        round(
            selected_count
            * test_percent
            / 100.0
        )
    )

    n_test = max(
        1,
        min(
            selected_count - 1,
            n_test,
        ),
    )

    test_pairs = list(
        selected_pairs[
            :n_test
        ]
    )

    train_pairs = list(
        selected_pairs[
            n_test:
        ]
    )

    manifest = (
        _semantic_write_split_manifest(
            root,
            train_pairs,
            test_pairs,
            requested_pairs=pairs_to_use,
            test_percent=test_percent,
            available_pairs=valid_count,
        )
    )

    if progress_callback is not None:
        progress_callback(
            (
                "[4/8] Train/Test split ready. "
                f"Train={len(train_pairs)}, Test={len(test_pairs)}."
            ),
            10,
        )

    return (
        train_pairs,
        test_pairs,
        manifest,
        total_pairs,
        valid_count,
        skipped_count,
        selected_count,
        dimension_csv,
        skipped_csv,
    )


def _semantic_read_standard_image(path):
    path = Path(path)

    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        arr = np.squeeze(
            np.asarray(
                tifffile.imread(path)
            )
        )
    else:
        raw = np.fromfile(
            str(path),
            dtype=np.uint8,
        )
        arr = cv2.imdecode(
            raw,
            cv2.IMREAD_UNCHANGED,
        )

        if arr is None:
            raise ValueError(
                f"Cannot read image:\n{path}"
            )

        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                arr = cv2.cvtColor(
                    arr,
                    cv2.COLOR_BGRA2RGBA,
                )
            elif arr.shape[-1] == 3:
                arr = cv2.cvtColor(
                    arr,
                    cv2.COLOR_BGR2RGB,
                )

        arr = np.squeeze(
            np.asarray(arr)
        )

    return arr


def _semantic_extract_2d_image(
    arr,
    source_path=None,
):
    """
    Training/Test is 2D.

    Accepted:
      YX
      YXC/RGB(A)

    A Z stack is rejected in the trainer so the 2D model is not accidentally
    trained with ambiguous volume axes. The main Predict button supports 3D
    by running this 2D model slice-by-slice.
    """
    arr = np.squeeze(
        np.asarray(arr)
    )

    if arr.ndim == 2:
        return arr

    if (
        arr.ndim == 3
        and arr.shape[-1] in (3, 4)
    ):
        return arr

    source_text = (
        f"\nFile: {source_path}"
        if source_path is not None
        else ""
    )

    raise ValueError(
        "Semantic Train/Test currently expects 2D image/mask pairs. "
        f"Received shape {arr.shape}."
        f"{source_text}\n"
        "3D images are supported by Predict using slice-wise inference."
    )



def _semantic_probe_file_geometry(
    path,
):
    """
    Read only enough information to diagnose file dimensions.

    TIFF:
        use first-series metadata (fast; no full pixel load).

    Other supported formats:
        read the image normally because dimensions are not expensive to query.

    Returns a dict with raw shape, squeezed shape, axes and readable flag.
    """
    path = Path(
        path
    )

    result = {
        "path": str(path),
        "raw_shape": None,
        "squeezed_shape": None,
        "axes": "",
        "error": "",
    }

    try:
        suffix = path.suffix.lower()

        if suffix in {
            ".tif",
            ".tiff",
        }:
            with tifffile.TiffFile(
                path
            ) as tf:
                if not tf.series:
                    raise ValueError(
                        "TIFF contains no readable series."
                    )

                series = tf.series[0]
                raw_shape = tuple(
                    int(v)
                    for v in series.shape
                )
                axes = str(
                    series.axes
                    or ""
                )

        else:
            raw = np.fromfile(
                str(path),
                dtype=np.uint8,
            )

            arr = cv2.imdecode(
                raw,
                cv2.IMREAD_UNCHANGED,
            )

            if arr is None:
                raise ValueError(
                    "Cannot decode image."
                )

            raw_shape = tuple(
                int(v)
                for v in arr.shape
            )
            axes = ""

        squeezed_shape = tuple(
            int(v)
            for v in raw_shape
            if int(v) != 1
        )

        result[
            "raw_shape"
        ] = raw_shape

        result[
            "squeezed_shape"
        ] = squeezed_shape

        result[
            "axes"
        ] = axes

        return result

    except Exception as exc:
        result[
            "error"
        ] = str(exc)

        return result


def _semantic_valid_2d_image_shape(
    squeezed_shape,
):
    shape = tuple(
        squeezed_shape
        or ()
    )

    if len(shape) == 2:
        return True

    if (
        len(shape) == 3
        and shape[-1] in (
            3,
            4,
        )
    ):
        return True

    return False


def _semantic_valid_2d_mask_shape(
    squeezed_shape,
):
    shape = tuple(
        squeezed_shape
        or ()
    )

    return (
        len(shape) == 2
    )


def _semantic_xy_from_squeezed_shape(
    squeezed_shape,
    is_mask=False,
):
    shape = tuple(
        squeezed_shape
        or ()
    )

    if is_mask:
        if len(shape) == 2:
            return (
                int(shape[0]),
                int(shape[1]),
            )
        return None

    if len(shape) == 2:
        return (
            int(shape[0]),
            int(shape[1]),
        )

    if (
        len(shape) == 3
        and shape[-1] in (
            3,
            4,
        )
    ):
        return (
            int(shape[0]),
            int(shape[1]),
        )

    return None


def _semantic_validate_2d_pairs(
    pairs,
    csv_path=None,
    progress_callback=None,
):
    """
    Check every selected image/mask pair BEFORE Train/Test.

    Diagnoses:
      - unreadable files
      - non-2D image geometry
      - non-2D mask geometry
      - image/mask YX mismatch

    Singleton dimensions are accepted because the actual training loader
    uses np.squeeze(). Thus:
        (1, 1200, 1200) -> valid 1200x1200
    while:
        (2,) -> invalid

    Returns:
        rows, issues
    """
    rows = []
    issues = []

    total_pairs = max(
        1,
        len(pairs),
    )

    if progress_callback is not None:
        progress_callback(
            (
                "[3/8] Checking image/mask dimensions... "
                f"0/{len(pairs)}"
            ),
            5,
        )

    for pair_index, (
        image_path,
        mask_path,
    ) in enumerate(
        pairs,
        1,
    ):
        image_info = (
            _semantic_probe_file_geometry(
                image_path
            )
        )

        mask_info = (
            _semantic_probe_file_geometry(
                mask_path
            )
        )

        pair_issues = []

        if image_info[
            "error"
        ]:
            pair_issues.append(
                "IMAGE unreadable: "
                + image_info[
                    "error"
                ]
            )

        elif not (
            _semantic_valid_2d_image_shape(
                image_info[
                    "squeezed_shape"
                ]
            )
        ):
            pair_issues.append(
                "IMAGE is not a supported 2D YX/RGB image: "
                f"raw={image_info['raw_shape']}, "
                f"squeezed={image_info['squeezed_shape']}, "
                f"axes={image_info['axes'] or 'unknown'}"
            )

        if mask_info[
            "error"
        ]:
            pair_issues.append(
                "MASK unreadable: "
                + mask_info[
                    "error"
                ]
            )

        elif not (
            _semantic_valid_2d_mask_shape(
                mask_info[
                    "squeezed_shape"
                ]
            )
        ):
            pair_issues.append(
                "MASK is not a 2D YX image: "
                f"raw={mask_info['raw_shape']}, "
                f"squeezed={mask_info['squeezed_shape']}, "
                f"axes={mask_info['axes'] or 'unknown'}"
            )

        image_xy = (
            _semantic_xy_from_squeezed_shape(
                image_info[
                    "squeezed_shape"
                ],
                is_mask=False,
            )
        )

        mask_xy = (
            _semantic_xy_from_squeezed_shape(
                mask_info[
                    "squeezed_shape"
                ],
                is_mask=True,
            )
        )

        if (
            image_xy is not None
            and mask_xy is not None
            and image_xy != mask_xy
        ):
            pair_issues.append(
                "IMAGE/MASK YX mismatch: "
                f"image={image_xy}, "
                f"mask={mask_xy}"
            )

        status = (
            "BAD"
            if pair_issues
            else "OK"
        )

        issue_text = (
            " | ".join(
                pair_issues
            )
        )

        row = {
            "pair_index": (
                pair_index
            ),
            "image": str(
                image_path
            ),
            "image_raw_shape": str(
                image_info[
                    "raw_shape"
                ]
            ),
            "image_squeezed_shape": str(
                image_info[
                    "squeezed_shape"
                ]
            ),
            "image_axes": (
                image_info[
                    "axes"
                ]
            ),
            "mask": str(
                mask_path
            ),
            "mask_raw_shape": str(
                mask_info[
                    "raw_shape"
                ]
            ),
            "mask_squeezed_shape": str(
                mask_info[
                    "squeezed_shape"
                ]
            ),
            "mask_axes": (
                mask_info[
                    "axes"
                ]
            ),
            "status": status,
            "issue": issue_text,
        }

        rows.append(
            row
        )

        if pair_issues:
            issues.append(
                row
            )

        if (
            progress_callback is not None
            and (
                pair_index == 1
                or pair_index == total_pairs
                or pair_index % max(
                    1,
                    total_pairs // 20,
                ) == 0
            )
        ):
            progress_callback(
                (
                    "[3/8] Checking dimensions... "
                    f"{pair_index}/{total_pairs} "
                    f"(OK={len(rows) - len(issues)}, BAD={len(issues)})"
                ),
                min(
                    8,
                    5 + int(
                        3
                        * pair_index
                        / total_pairs
                    ),
                ),
            )

    if csv_path is not None:
        csv_path = Path(
            csv_path
        )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_index",
                    "image",
                    "image_raw_shape",
                    "image_squeezed_shape",
                    "image_axes",
                    "mask",
                    "mask_raw_shape",
                    "mask_squeezed_shape",
                    "mask_axes",
                    "status",
                    "issue",
                ],
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    return (
        rows,
        issues,
    )


def _semantic_dimension_issue_text(
    issues,
    csv_path,
):
    lines = [
        (
            f"DIMENSION CHECK FAILED: "
            f"{len(issues)} problematic pair(s)"
        ),
        "",
    ]

    for index, row in enumerate(
        issues,
        1,
    ):
        lines.extend(
            [
                (
                    f"[{index}] "
                    f"Image: {row['image']}"
                ),
                (
                    "    image raw: "
                    f"{row['image_raw_shape']}  "
                    "squeezed: "
                    f"{row['image_squeezed_shape']}  "
                    "axes: "
                    f"{row['image_axes'] or 'unknown'}"
                ),
                (
                    f"    Mask: {row['mask']}"
                ),
                (
                    "    mask raw: "
                    f"{row['mask_raw_shape']}  "
                    "squeezed: "
                    f"{row['mask_squeezed_shape']}  "
                    "axes: "
                    f"{row['mask_axes'] or 'unknown'}"
                ),
                (
                    f"    Problem: "
                    f"{row['issue']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Full dimension report:",
            str(
                csv_path
            ),
        ]
    )

    return "\\n".join(
        lines
    )



def _semantic_image_to_rgb_float(image):
    """Return ONE grayscale uint8 channel for semantic segmentation.

    The legacy function name is retained for compatibility with the rest of
    LinkODT, but semantic models now consume true Bx1xHxW microscopy input.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        gray = normalize_to_uint8(image)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        arr = image[..., :3].astype(np.float32)
        gray_float = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        )
        gray = normalize_to_uint8(gray_float)
    elif image.ndim == 3 and image.shape[-1] == 1:
        gray = normalize_to_uint8(image[..., 0])
    else:
        raise ValueError(
            f"Unsupported semantic image shape: {image.shape}"
        )

    return np.ascontiguousarray(gray, dtype=np.uint8)


def _semantic_prepare_image_tensor(
    rgb,
    size=SEMANTIC_MODEL_INPUT_SIZE,
    resize=False,
):
    """Convert microscopy image to normalized 1xHxW float32.

    DINOv2 RGB patch-embedding weights are averaged to one input channel in
    semantic_models.py. The scalar normalization below is the mean ImageNet
    normalization used as an initialization bridge; microscopy SSL/fine-tuning
    adapts it to the fluorescence domain.
    """
    arr = np.asarray(rgb)

    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = (
            0.299 * arr[..., 0].astype(np.float32)
            + 0.587 * arr[..., 1].astype(np.float32)
            + 0.114 * arr[..., 2].astype(np.float32)
        )
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Semantic input must reduce to one channel, got {arr.shape}")

    if resize:
        arr = cv2.resize(
            arr,
            (int(size), int(size)),
            interpolation=cv2.INTER_AREA,
        )

    # Inputs generated by LinkNex are usually uint8. Preserve float images in
    # 0..1 when already normalized.
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0

    mean = np.float32((0.485 + 0.456 + 0.406) / 3.0)
    std = np.float32((0.229 + 0.224 + 0.225) / 3.0)
    arr = (arr - mean) / std
    arr = arr[None, ...]

    return np.ascontiguousarray(arr, dtype=np.float32)


def _semantic_prepare_mask(
    mask,
    size=SEMANTIC_MODEL_INPUT_SIZE,
    resize=False,
):
    mask = np.squeeze(
        np.asarray(mask)
    )

    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D, got {mask.shape}"
        )

    binary = (
        mask > 0
    ).astype(np.uint8)

    if resize:
        binary = cv2.resize(
            binary,
            (int(size), int(size)),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.ascontiguousarray(
        binary[None, ...].astype(
            np.float32
        )
    )


def _semantic_pad_image_mask(
    rgb,
    mask=None,
    tile_size=SEMANTIC_TILE_SIZE,
):
    """
    Pad image/mask only when Y or X is smaller than tile_size.

    Image uses REFLECT_101 when possible; mask always uses constant 0.
    """
    rgb = np.asarray(
        rgb
    )

    h, w = rgb.shape[:2]

    pad_bottom = max(
        0,
        int(tile_size) - h,
    )
    pad_right = max(
        0,
        int(tile_size) - w,
    )

    if pad_bottom == 0 and pad_right == 0:
        return (
            np.ascontiguousarray(rgb),
            None
            if mask is None
            else np.ascontiguousarray(mask),
        )

    border_type = (
        cv2.BORDER_REFLECT_101
        if h > 1 and w > 1
        else cv2.BORDER_REPLICATE
    )

    padded_rgb = cv2.copyMakeBorder(
        rgb,
        0,
        pad_bottom,
        0,
        pad_right,
        border_type,
    )

    padded_mask = None

    if mask is not None:
        padded_mask = cv2.copyMakeBorder(
            np.asarray(mask),
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    return (
        np.ascontiguousarray(
            padded_rgb
        ),
        None
        if padded_mask is None
        else np.ascontiguousarray(
            padded_mask
        ),
    )


def _semantic_random_patch(
    rgb,
    mask,
    tile_size=SEMANTIC_TILE_SIZE,
):
    """
    Random aligned image/mask crop.

    Images smaller than 448 in either dimension are padded first.
    Larger images retain native pixel size; nothing is globally resized.
    """
    rgb, mask = _semantic_pad_image_mask(
        rgb,
        mask,
        tile_size=tile_size,
    )

    h, w = rgb.shape[:2]

    max_y = max(
        0,
        h - int(tile_size),
    )
    max_x = max(
        0,
        w - int(tile_size),
    )

    y0 = (
        int(
            np.random.randint(
                0,
                max_y + 1,
            )
        )
        if max_y > 0
        else 0
    )

    x0 = (
        int(
            np.random.randint(
                0,
                max_x + 1,
            )
        )
        if max_x > 0
        else 0
    )

    y1 = y0 + int(
        tile_size
    )
    x1 = x0 + int(
        tile_size
    )

    return (
        np.ascontiguousarray(
            rgb[
                y0:y1,
                x0:x1,
            ]
        ),
        np.ascontiguousarray(
            mask[
                y0:y1,
                x0:x1,
            ]
        ),
    )


def _semantic_augment_pair(
    rgb,
    mask,
):
    """Microscopy-aware paired augmentation.

    Geometry is applied identically to image and mask. Intensity/noise/blur/
    resolution perturbations are applied only to the grayscale image so the
    model is less dependent on one microscope, exposure level or PSF.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)

    # ---- paired geometry ----
    if np.random.random() < 0.5:
        rgb = np.flip(rgb, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random() < 0.5:
        rgb = np.flip(rgb, axis=0)
        mask = np.flip(mask, axis=0)

    k = int(np.random.randint(0, 4))
    if k:
        rgb = np.rot90(rgb, k=k, axes=(0, 1))
        mask = np.rot90(mask, k=k, axes=(0, 1))

    # ---- microscopy appearance only ----
    image = np.asarray(rgb, dtype=np.float32)
    max_value = 255.0 if float(np.nanmax(image)) > 1.5 else 1.0
    image = np.clip(image / max_value, 0.0, 1.0)

    # Exposure / brightness variation.
    if np.random.random() < 0.55:
        image *= float(np.random.uniform(0.75, 1.25))

    # Detector / display contrast variation.
    if np.random.random() < 0.45:
        gamma = float(np.random.uniform(0.75, 1.35))
        image = np.clip(image, 0.0, 1.0) ** gamma

    # Background offset / autofluorescence variation.
    if np.random.random() < 0.30:
        image += float(np.random.uniform(-0.06, 0.10))

    # Modest focus / PSF variation.
    if np.random.random() < 0.25:
        sigma = float(np.random.uniform(0.2, 1.2))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Detector noise.
    if np.random.random() < 0.35:
        sigma = float(np.random.uniform(0.0, 0.04))
        image += np.random.normal(0.0, sigma, image.shape).astype(np.float32)

    # Mild resolution degradation followed by restoration to the same size.
    # This simulates different pixel sizes/objectives without changing masks.
    if np.random.random() < 0.25:
        h, w = image.shape[:2]
        scale = float(np.random.uniform(0.70, 0.95))
        small_w = max(16, int(round(w * scale)))
        small_h = max(16, int(round(h * scale)))
        image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    image = np.clip(image, 0.0, 1.0)
    if max_value > 1.5:
        image = np.rint(image * 255.0).astype(np.uint8)
    else:
        image = image.astype(np.float32)

    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(mask),
    )


def _semantic_tile_starts(
    length,
    tile_size=SEMANTIC_TILE_SIZE,
    stride=SEMANTIC_TILE_STRIDE,
):
    """
    Generate starts that cover the full dimension and force the final tile
    to touch the image boundary.
    """
    length = int(
        length
    )
    tile_size = int(
        tile_size
    )
    stride = int(
        stride
    )

    if length <= tile_size:
        return [0]

    starts = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    last = (
        length - tile_size
    )

    if starts[-1] != last:
        starts.append(
            last
        )

    return starts


def _semantic_weight_window(
    tile_size=SEMANTIC_TILE_SIZE,
):
    """
    Smooth positive 2D blending window for overlapping tiles.

    Values never reach zero, so border pixels remain numerically stable.
    """
    n = int(
        tile_size
    )

    w1 = np.hanning(
        n
    ).astype(
        np.float32
    )

    w1 = np.maximum(
        w1,
        0.05,
    )

    w2 = np.outer(
        w1,
        w1,
    ).astype(
        np.float32
    )

    return w2


def _semantic_autocast_context(
    torch,
    device,
):
    enabled = (
        getattr(
            device,
            "type",
            str(device),
        )
        == "cuda"
    )

    return torch.autocast(
        device_type="cuda"
        if enabled
        else "cpu",
        dtype=torch.float16
        if enabled
        else torch.bfloat16,
        enabled=enabled,
    )


def _semantic_make_grad_scaler(
    torch,
    device,
):
    enabled = (
        getattr(
            device,
            "type",
            str(device),
        )
        == "cuda"
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def semantic_predict_tiled_rgb(
    torch,
    model,
    device,
    input_size,
    threshold,
    rgb,
    tile_size=SEMANTIC_TILE_SIZE,
    overlap=SEMANTIC_TILE_OVERLAP,
    infer_batch_size=SEMANTIC_INFER_BATCH_SIZE,
):
    """
    Sliding-window inference that preserves the exact original Y/X dimensions.

    - no global resize
    - 448x448 tiles
    - 112 px overlap
    - overlap blended using a smooth weight window
    - output binary/probability maps match the original image exactly
    """
    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    original_h, original_w = (
        rgb.shape[:2]
    )

    padded_rgb, _ = (
        _semantic_pad_image_mask(
            rgb,
            None,
            tile_size=tile_size,
        )
    )

    h, w = padded_rgb.shape[:2]

    stride = (
        int(tile_size)
        - int(overlap)
    )

    ys = _semantic_tile_starts(
        h,
        tile_size=tile_size,
        stride=stride,
    )

    xs = _semantic_tile_starts(
        w,
        tile_size=tile_size,
        stride=stride,
    )

    coords = [
        (
            y0,
            x0,
        )
        for y0 in ys
        for x0 in xs
    ]

    prob_sum = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    weight = (
        _semantic_weight_window(
            tile_size
        )
    )

    batch_size = max(
        1,
        int(
            infer_batch_size
        ),
    )

    for batch_start in range(
        0,
        len(coords),
        batch_size,
    ):
        batch_coords = coords[
            batch_start:
            batch_start + batch_size
        ]

        batch_np = []

        for y0, x0 in batch_coords:
            tile = padded_rgb[
                y0:y0 + tile_size,
                x0:x0 + tile_size,
            ]

            x_np = (
                _semantic_prepare_image_tensor(
                    tile,
                    size=input_size,
                    resize=False,
                )
            )

            batch_np.append(
                x_np
            )

        x = torch.from_numpy(
            np.stack(
                batch_np,
                axis=0,
            )
        ).to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            with _semantic_autocast_context(
                torch,
                device,
            ):
                logits = model(
                    x
                )
                probs = torch.sigmoid(
                    logits
                )

        probs_np = (
            probs[:, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        for (
            y0,
            x0,
        ), prob_tile in zip(
            batch_coords,
            probs_np,
        ):
            if prob_tile.shape != (
                tile_size,
                tile_size,
            ):
                prob_tile = cv2.resize(
                    prob_tile.astype(
                        np.float32
                    ),
                    (
                        tile_size,
                        tile_size,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            prob_sum[
                y0:y0 + tile_size,
                x0:x0 + tile_size,
            ] += (
                prob_tile
                * weight
            )

            weight_sum[
                y0:y0 + tile_size,
                x0:x0 + tile_size,
            ] += weight

    probability = (
        prob_sum
        / np.maximum(
            weight_sum,
            1e-6,
        )
    )

    probability = probability[
        :original_h,
        :original_w,
    ]

    binary = (
        probability
        >= float(
            threshold
        )
    ).astype(
        np.uint8
    )

    return (
        binary,
        probability.astype(
            np.float32,
            copy=False,
        ),
    )



def import_semantic_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import (
            Dataset,
            DataLoader,
        )
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for Semantic Train/Test/Predict.\n\n"
            "Run LinkODT in your pytorch_gpu environment."
        ) from exc

    return (
        torch,
        nn,
        F,
        Dataset,
        DataLoader,
    )



def _find_local_dinov2_repo():
    """
    Return a local DINOv2 repository containing hubconf.py.
    No network access is attempted.
    """
    for candidate in DINOV2_LOCAL_REPO_CANDIDATES:
        candidate = Path(candidate)

        if (
            candidate.exists()
            and candidate.is_dir()
            and (candidate / "hubconf.py").exists()
        ):
            return candidate

    expected = "\n".join(
        f"  - {p}"
        for p in DINOV2_LOCAL_REPO_CANDIDATES
    )

    raise FileNotFoundError(
        "Local DINOv2 repository was not found.\n\n"
        "Download/extract the official facebookresearch/dinov2 repository "
        "beside the LinkODT .py file.\n\n"
        "Accepted locations:\n"
        f"{expected}\n\n"
        "The selected folder must contain hubconf.py."
    )


def _find_local_dinov2_vits14_weights():
    """Backward-compatible alias for the ViT-S/14 official checkpoint."""
    return _find_local_dinov2_weights("dinov2_vits14")


def _load_dinov2_backbone_weights(
    torch,
    encoder,
    weight_path,
):
    """
    Load the official DINOv2 ViT-S/14 backbone checkpoint from disk.

    Supports:
      - raw state_dict
      - {'state_dict': ...}
      - {'model': ...}

    Common wrapper prefixes are stripped when present.
    """
    weight_path = Path(
        weight_path
    )

    checkpoint = torch.load(
        weight_path,
        map_location="cpu",
    )

    state_dict = checkpoint

    if isinstance(
        checkpoint,
        dict,
    ):
        if (
            "state_dict" in checkpoint
            and isinstance(
                checkpoint["state_dict"],
                dict,
            )
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        elif (
            "model" in checkpoint
            and isinstance(
                checkpoint["model"],
                dict,
            )
        ):
            state_dict = checkpoint[
                "model"
            ]

    if not isinstance(
        state_dict,
        dict,
    ):
        raise ValueError(
            "Unsupported DINOv2 backbone checkpoint format:\n"
            f"{weight_path}"
        )

    cleaned = {}

    prefixes = (
        "module.",
        "backbone.",
        "encoder.",
    )

    for key, value in state_dict.items():
        new_key = str(
            key
        )

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(
                    prefix
                ):
                    new_key = new_key[
                        len(prefix):
                    ]
                    changed = True
                    break

        cleaned[
            new_key
        ] = value

    missing, unexpected = (
        encoder.load_state_dict(
            cleaned,
            strict=False,
        )
    )

    # Official ViT-S/14 backbone weights should match almost completely.
    # Allow only harmless wrapper/version differences; reject a clearly wrong
    # checkpoint such as ViT-B/L/g or a segmentation-head checkpoint.
    model_key_count = len(
        encoder.state_dict()
    )

    matched_count = (
        model_key_count
        - len(missing)
    )

    if (
        model_key_count <= 0
        or matched_count
        < int(
            0.90
            * model_key_count
        )
    ):
        raise ValueError(
            "The selected local DINOv2 checkpoint does not match "
            "dinov2_vits14.\n\n"
            f"Checkpoint:\n{weight_path}\n\n"
            f"Matched keys: {matched_count}/{model_key_count}\n"
            f"Missing keys: {len(missing)}\n"
            f"Unexpected keys: {len(unexpected)}"
        )

    return (
        list(missing),
        list(unexpected),
    )


def build_semantic_model_by_key(
    torch,
    nn,
    F,
    architecture_key=SEMANTIC_DEFAULT_ARCHITECTURE,
    backbone_name=SEMANTIC_DEFAULT_BACKBONE,
    pretrained_encoder_path=None,
):
    """Build any LinkODT semantic architecture with true 1-channel input."""
    from semantic_models import build_semantic_model

    repo_dir = None
    if _semantic_arch_uses_dino(architecture_key):
        repo_dir = _find_local_dinov2_repo()

    model, load_info = build_semantic_model(
        architecture_key,
        torch,
        nn,
        F,
        repo_dir=repo_dir,
        backbone_name=backbone_name,
        pretrained_path=pretrained_encoder_path,
        hf_segformer_dir=SEMANTIC_HF_SEGFORMER_DIR,
    )
    return model, load_info


def build_dinov2_unet_model(
    torch,
    nn,
    F,
    pretrained=True,
):
    """Backward-compatible builder for older LinkODT checkpoints/scripts."""
    pretrained_path = (
        _find_local_dinov2_vits14_weights()
        if pretrained
        else None
    )
    model, _ = build_semantic_model_by_key(
        torch,
        nn,
        F,
        architecture_key="dinov2_unet",
        backbone_name="dinov2_vits14",
        pretrained_encoder_path=pretrained_path,
    )
    return model


def _semantic_dice_loss(torch, logits, target):
    probs = torch.sigmoid(
        logits
    )

    dims = (
        1,
        2,
        3,
    )

    inter = (
        probs * target
    ).sum(
        dim=dims
    )

    denom = (
        probs.sum(dim=dims)
        + target.sum(dim=dims)
    )

    dice = (
        2.0 * inter + 1.0
    ) / (
        denom + 1.0
    )

    return (
        1.0 - dice.mean()
    )


def _semantic_binary_dice_from_logits(
    torch,
    logits,
    target,
    threshold=SEMANTIC_THRESHOLD,
):
    """
    Thresholded binary Dice used only as a human-readable training metric.
    Loss remains BCE + soft Dice loss.
    """
    with torch.no_grad():
        pred = (
            torch.sigmoid(logits)
            >= float(threshold)
        )

        truth = (
            target >= 0.5
        )

        dims = (
            1,
            2,
            3,
        )

        inter = (
            pred & truth
        ).sum(
            dim=dims
        ).float()

        pred_sum = (
            pred.sum(
                dim=dims
            ).float()
        )

        truth_sum = (
            truth.sum(
                dim=dims
            ).float()
        )

        dice = (
            2.0 * inter + 1.0
        ) / (
            pred_sum
            + truth_sum
            + 1.0
        )

        return float(
            dice.mean().item()
        )


def _semantic_binary_dice_iou(
    pred,
    truth,
):
    pred_b = (
        np.asarray(pred) > 0
    )
    truth_b = (
        np.asarray(truth) > 0
    )

    intersection = float(
        np.logical_and(
            pred_b,
            truth_b,
        ).sum()
    )

    pred_sum = float(
        pred_b.sum()
    )

    truth_sum = float(
        truth_b.sum()
    )

    union = float(
        np.logical_or(
            pred_b,
            truth_b,
        ).sum()
    )

    dice = (
        (
            2.0 * intersection
            + 1.0
        )
        / (
            pred_sum
            + truth_sum
            + 1.0
        )
    )

    iou = (
        (
            intersection
            + 1.0
        )
        / (
            union
            + 1.0
        )
    )

    return (
        float(dice),
        float(iou),
    )


def _semantic_fit_preview_panel(
    image_rgb,
    panel_size=SEMANTIC_TRAIN_PREVIEW_PANEL_SIZE,
):
    """
    Resize an RGB image to fit inside a small square white panel.
    No stretching; aspect ratio is preserved.
    """
    panel_size = max(
        64,
        int(panel_size),
    )

    image_rgb = np.asarray(
        image_rgb,
        dtype=np.uint8,
    )

    if image_rgb.ndim == 2:
        image_rgb = np.repeat(
            image_rgb[..., None],
            3,
            axis=2,
        )

    if (
        image_rgb.ndim != 3
        or image_rgb.shape[2] < 3
    ):
        raise ValueError(
            "Preview panel requires RGB/YX image."
        )

    image_rgb = image_rgb[
        ...,
        :3,
    ]

    h, w = image_rgb.shape[
        :2
    ]

    scale = min(
        panel_size / max(1, w),
        panel_size / max(1, h),
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image_rgb,
        (
            new_w,
            new_h,
        ),
        interpolation=interpolation,
    )

    panel = np.full(
        (
            panel_size,
            panel_size,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    y0 = (
        panel_size
        - new_h
    ) // 2

    x0 = (
        panel_size
        - new_w
    ) // 2

    panel[
        y0:y0 + new_h,
        x0:x0 + new_w,
    ] = resized

    return panel


def _semantic_save_training_preview_png(
    save_path,
    rgb,
    truth,
    pred,
    epoch,
    dice,
    iou,
):
    """
    Compact PNG montage:
        Original | Ground Truth | Prediction

    Formal Predict output remains TIFF; this helper is ONLY for small
    training-time visual checkpoints.
    """
    panel_size = (
        SEMANTIC_TRAIN_PREVIEW_PANEL_SIZE
    )

    original_panel = (
        _semantic_fit_preview_panel(
            rgb,
            panel_size=panel_size,
        )
    )

    truth_rgb = np.repeat(
        (
            (np.asarray(truth) > 0)
            .astype(np.uint8)
            * 255
        )[..., None],
        3,
        axis=2,
    )

    pred_rgb = np.repeat(
        (
            (np.asarray(pred) > 0)
            .astype(np.uint8)
            * 255
        )[..., None],
        3,
        axis=2,
    )

    truth_panel = (
        _semantic_fit_preview_panel(
            truth_rgb,
            panel_size=panel_size,
        )
    )

    pred_panel = (
        _semantic_fit_preview_panel(
            pred_rgb,
            panel_size=panel_size,
        )
    )

    header_h = 34

    canvas = np.full(
        (
            panel_size
            + header_h,
            panel_size * 3,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    canvas[
        header_h:,
        0:panel_size,
    ] = original_panel

    canvas[
        header_h:,
        panel_size:panel_size * 2,
    ] = truth_panel

    canvas[
        header_h:,
        panel_size * 2:panel_size * 3,
    ] = pred_panel

    cv2.putText(
        canvas,
        "Original",
        (
            8,
            21,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (
            20,
            20,
            20,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Ground Truth",
        (
            panel_size + 8,
            21,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (
            20,
            20,
            20,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"Pred E{epoch:03d} "
            f"D={dice:.3f} I={iou:.3f}"
        ),
        (
            panel_size * 2 + 8,
            21,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (
            20,
            20,
            20,
        ),
        1,
        cv2.LINE_AA,
    )

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # cv2.imwrite can fail on some Windows Unicode paths.
    # imencode + tofile is safer.
    bgr = cv2.cvtColor(
        canvas,
        cv2.COLOR_RGB2BGR,
    )

    ok, encoded = cv2.imencode(
        ".png",
        bgr,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            7,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Could not encode preview PNG:\n{save_path}"
        )

    encoded.tofile(
        str(save_path)
    )


def _semantic_run_epoch_preview(
    torch,
    model,
    device,
    test_pairs,
    dataset_root,
    epoch,
    stop_callback=None,
):
    """
    Every epoch, evaluate a fixed small subset of the Test split using
    the SAME full-resolution tiled inference path as formal Test/Predict.

    Saves compact PNG montages under:
        dataset/test_results/epoch_001/
        dataset/test_results/epoch_002/
        dataset/test_results/epoch_003/
        ...

    To keep training practical, visual preview is capped at
    SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES. The formal Test button still evaluates
    the complete test set.
    """
    preview_pairs = list(
        test_pairs[
            :min(
                len(test_pairs),
                SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES,
            )
        ]
    )

    if not preview_pairs:
        return (
            float("nan"),
            float("nan"),
            0,
            None,
        )

    root = (
        Path(dataset_root)
        / SEMANTIC_TRAIN_PREVIEW_FOLDER
    )

    epoch_dir = (
        root
        / f"epoch_{int(epoch):03d}"
    )

    # Avoid stale PNGs when a training run is repeated.
    if epoch_dir.exists():
        shutil.rmtree(
            epoch_dir
        )

    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    dice_values = []
    iou_values = []

    was_training = bool(
        model.training
    )

    model.eval()

    try:
        for index, (
            image_path,
            mask_path,
        ) in enumerate(
            preview_pairs,
            1,
        ):
            if stop_callback is not None:
                stop_callback()
            image = (
                _semantic_extract_2d_image(
                    _semantic_read_standard_image(
                        image_path
                    ),
                    source_path=image_path,
                )
            )

            mask = (
                _semantic_extract_2d_image(
                    _semantic_read_standard_image(
                        mask_path
                    ),
                    source_path=mask_path,
                )
            )

            rgb = (
                _semantic_image_to_rgb_float(
                    image
                )
            )

            truth = (
                mask > 0
            ).astype(
                np.uint8
            )

            if rgb.shape[:2] != truth.shape:
                raise ValueError(
                    "Preview image/mask size mismatch:\n"
                    f"Image: {image_path.name} {rgb.shape[:2]}\n"
                    f"Mask:  {mask_path.name} {truth.shape}"
                )

            pred, _ = (
                semantic_predict_tiled_rgb(
                    torch,
                    model,
                    device,
                    SEMANTIC_MODEL_INPUT_SIZE,
                    SEMANTIC_THRESHOLD,
                    rgb,
                    tile_size=(
                        SEMANTIC_TILE_SIZE
                    ),
                    overlap=(
                        SEMANTIC_TILE_OVERLAP
                    ),
                    infer_batch_size=(
                        SEMANTIC_INFER_BATCH_SIZE
                    ),
                )
            )

            dice, iou = (
                _semantic_binary_dice_iou(
                    pred,
                    truth,
                )
            )

            dice_values.append(
                dice
            )

            iou_values.append(
                iou
            )

            safe_stem = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                image_path.stem,
            )

            png_path = (
                epoch_dir
                / (
                    f"{index:03d}_"
                    f"{safe_stem}.png"
                )
            )

            _semantic_save_training_preview_png(
                png_path,
                rgb,
                truth,
                pred,
                epoch=epoch,
                dice=dice,
                iou=iou,
            )

            rows.append(
                {
                    "epoch": int(
                        epoch
                    ),
                    "image": (
                        image_path.name
                    ),
                    "mask": (
                        mask_path.name
                    ),
                    "dice": (
                        f"{dice:.6f}"
                    ),
                    "iou": (
                        f"{iou:.6f}"
                    ),
                    "preview_png": (
                        str(png_path)
                    ),
                }
            )

        metrics_csv = (
            epoch_dir
            / "preview_metrics.csv"
        )

        with open(
            metrics_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "image",
                    "mask",
                    "dice",
                    "iou",
                    "preview_png",
                ],
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        return (
            float(
                np.mean(
                    dice_values
                )
            ),
            float(
                np.mean(
                    iou_values
                )
            ),
            len(
                preview_pairs
            ),
            epoch_dir,
        )

    finally:
        if was_training:
            model.train()



def _semantic_fullres_loss_from_probability(probability, truth):
    """
    Deterministic full-resolution Test loss for training history.

    Objective: BCE + soft Dice loss, matching the conceptual training loss.
    Train loss is measured on random training patches; Test loss is measured
    on the reconstructed full-resolution tiled probability map.
    """
    probability = np.asarray(probability, dtype=np.float64)
    truth = (np.asarray(truth) > 0).astype(np.float64)

    if probability.shape != truth.shape:
        raise ValueError(
            f"Probability/truth shape mismatch: {probability.shape} vs {truth.shape}"
        )

    eps = 1e-7
    p = np.clip(probability, eps, 1.0 - eps)

    bce = -np.mean(
        truth * np.log(p)
        + (1.0 - truth) * np.log(1.0 - p)
    )

    intersection = float(np.sum(probability * truth))
    denominator = float(np.sum(probability) + np.sum(truth))
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - soft_dice

    return float(bce + dice_loss)


def _semantic_evaluate_full_test_epoch(
    torch,
    model,
    device,
    test_pairs,
    dataset_root,
    epoch,
    stop_callback=None,
):
    """
    Evaluate the COMPLETE Test split after one training epoch.

    All Test images contribute to Test loss/Dice/IoU. Visual PNG previews are
    limited to SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES to keep disk output compact.
    A per-image test_metrics.csv is saved inside each epoch folder.
    """
    test_pairs = list(test_pairs)
    if not test_pairs:
        return float("nan"), float("nan"), float("nan"), 0, None

    root = Path(dataset_root) / SEMANTIC_TRAIN_PREVIEW_FOLDER
    epoch_dir = root / f"epoch_{int(epoch):03d}"

    if epoch_dir.exists():
        shutil.rmtree(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    losses = []
    dices = []
    ious = []

    was_training = bool(model.training)
    model.eval()

    try:
        for index, (image_path, mask_path) in enumerate(test_pairs, 1):
            if stop_callback is not None:
                stop_callback()

            image = _semantic_extract_2d_image(
                _semantic_read_standard_image(image_path),
                source_path=image_path,
            )
            mask = _semantic_extract_2d_image(
                _semantic_read_standard_image(mask_path),
                source_path=mask_path,
            )

            rgb = _semantic_image_to_rgb_float(image)
            truth = (mask > 0).astype(np.uint8)

            if rgb.shape[:2] != truth.shape:
                raise ValueError(
                    "Test image/mask size mismatch:\n"
                    f"Image: {image_path.name} {rgb.shape[:2]}\n"
                    f"Mask:  {mask_path.name} {truth.shape}"
                )

            pred, probability = semantic_predict_tiled_rgb(
                torch,
                model,
                device,
                SEMANTIC_MODEL_INPUT_SIZE,
                SEMANTIC_THRESHOLD,
                rgb,
                tile_size=SEMANTIC_TILE_SIZE,
                overlap=SEMANTIC_TILE_OVERLAP,
                infer_batch_size=SEMANTIC_INFER_BATCH_SIZE,
            )

            test_loss = _semantic_fullres_loss_from_probability(
                probability,
                truth,
            )
            dice, iou = _semantic_binary_dice_iou(pred, truth)

            losses.append(test_loss)
            dices.append(dice)
            ious.append(iou)

            preview_png = ""
            if index <= SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES:
                safe_stem = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    image_path.stem,
                )
                png_path = epoch_dir / f"{index:03d}_{safe_stem}.png"
                _semantic_save_training_preview_png(
                    png_path,
                    rgb,
                    truth,
                    pred,
                    epoch=epoch,
                    dice=dice,
                    iou=iou,
                )
                preview_png = str(png_path)

            rows.append(
                {
                    "epoch": int(epoch),
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "test_loss": f"{test_loss:.6f}",
                    "dice": f"{dice:.6f}",
                    "iou": f"{iou:.6f}",
                    "preview_png": preview_png,
                }
            )

        per_image_csv = epoch_dir / "test_metrics.csv"
        with open(
            per_image_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "image",
                    "mask",
                    "test_loss",
                    "dice",
                    "iou",
                    "preview_png",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return (
            float(np.mean(losses)),
            float(np.mean(dices)),
            float(np.mean(ious)),
            len(test_pairs),
            epoch_dir,
        )

    finally:
        if was_training:
            model.train()


def _semantic_save_training_curves(metrics_rows, dataset_root):
    """Save Train/Test Loss and Dice curves after every completed epoch."""
    if not metrics_rows:
        return None, None

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dataset_root = Path(dataset_root)
    loss_path = dataset_root / SEMANTIC_TRAIN_LOSS_CURVE_PNG
    dice_path = dataset_root / SEMANTIC_TRAIN_DICE_CURVE_PNG

    epochs = [int(row["epoch"]) for row in metrics_rows]
    train_loss = [float(row["train_loss"]) for row in metrics_rows]
    test_loss = [float(row["test_loss"]) for row in metrics_rows]
    train_dice = [float(row["train_dice"]) for row in metrics_rows]
    test_dice = [float(row["test_dice"]) for row in metrics_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, test_loss, marker="o", label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Semantic segmentation training loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(epochs, train_dice, marker="o", label="Train Dice")
    ax.plot(epochs, test_dice, marker="o", label="Test Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Semantic segmentation Dice")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dice_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return loss_path, dice_path


def load_semantic_checkpoint(model_path):
    (
        torch,
        nn,
        F,
        _,
        _,
    ) = import_semantic_torch()

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Semantic model not found:\n{model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Unsupported semantic model format.")

    architecture = str(
        checkpoint.get("architecture", "dinov2_unet")
    ).lower()
    # Backward compatibility with old architecture strings.
    if architecture in {"dinov2_vits14_unet_decoder", "linknex_semantic_dinov2_unet"}:
        architecture = "dinov2_unet"

    backbone_name = str(
        checkpoint.get("backbone_name", "dinov2_vits14")
    )

    model, _ = build_semantic_model_by_key(
        torch,
        nn,
        F,
        architecture_key=architecture,
        backbone_name=backbone_name,
        pretrained_encoder_path=None,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    input_size = int(checkpoint.get("input_size", SEMANTIC_MODEL_INPUT_SIZE))
    threshold = float(checkpoint.get("threshold", SEMANTIC_THRESHOLD))

    return torch, model, device, input_size, threshold



class SemanticTrainWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(str, str)
    stopped = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        dataset_root,
        pairs_to_use,
        test_percent,
        epochs,
        batch_size,
        initial_model_path=None,
        initial_encoder_path=None,
        architecture_key=SEMANTIC_DEFAULT_ARCHITECTURE,
        backbone_name=SEMANTIC_DEFAULT_BACKBONE,
    ):
        super().__init__()

        self.dataset_root = Path(
            dataset_root
        )
        self.pairs_to_use = int(
            pairs_to_use
        )
        self.test_percent = int(
            test_percent
        )
        self.epochs = int(
            epochs
        )
        self.batch_size = int(
            batch_size
        )

        self.initial_model_path = (
            Path(
                initial_model_path
            )
            if initial_model_path
            else None
        )

        self.initial_encoder_path = (
            Path(
                initial_encoder_path
            )
            if initial_encoder_path
            else None
        )
        self.architecture_key = str(architecture_key)
        self.backbone_name = str(backbone_name)

        if (
            self.initial_model_path is not None
            and self.initial_encoder_path is not None
        ):
            raise ValueError(
                "Choose either a full LinkODT checkpoint OR a microscopy SSL "
                "encoder, not both."
            )

    def _check_stop(self):
        if QThread.currentThread().isInterruptionRequested():
            raise InterruptedError(
                "Training stopped by user."
            )

    def _progress_checked(self, text, value):
        self._check_stop()
        self.progress.emit(str(text), int(value))

    def run(self):
        try:
            # IMPORTANT:
            # Dataset discovery, TIFF dimension checks and Train/Test splitting
            # all happen inside this worker thread. The Qt GUI thread remains
            # responsive immediately after the user clicks Train.
            self._check_stop()
            self._progress_checked(
                "Checking image/mask pairs and dimensions...",
                1,
            )

            (
                train_pairs,
                test_pairs,
                manifest,
                total_pairs,
                valid_count,
                skipped_count,
                selected_count,
                dimension_csv,
                skipped_csv,
            ) = _semantic_get_or_create_split(
                self.dataset_root,
                pairs_to_use=self.pairs_to_use,
                test_percent=self.test_percent,
                progress_callback=(
                    self._progress_checked
                ),
            )

            pairs = list(
                train_pairs
            )

            if not pairs:
                raise ValueError(
                    "No valid training pairs were selected."
                )

            use_text = (
                "ALL VALID"
                if self.pairs_to_use >= valid_count
                else str(selected_count)
            )

            self.progress.emit(
                (
                    "DATASET READY\n"
                    f"Total image/mask pairs: {total_pairs}\n"
                    f"Valid pairs: {valid_count}\n"
                    f"Bad pairs skipped: {skipped_count}\n"
                    f"Requested pairs: {self.pairs_to_use}\n"
                    f"Pairs used: {use_text}\n"
                    f"Train pairs: {len(train_pairs)}\n"
                    f"Test pairs: {len(test_pairs)}\n"
                    f"Dimension report:\n{dimension_csv}\n"
                    f"Skipped report:\n{skipped_csv}\n"
                    f"Manifest:\n{manifest}"
                ),
                3,
            )

            self._progress_checked(
                "[5/8] Importing PyTorch / CUDA components...",
                11,
            )

            (
                torch,
                nn,
                F,
                Dataset,
                DataLoader,
            ) = import_semantic_torch()

            self.progress.emit(
                (
                    "[5/8] PyTorch ready. "
                    f"CUDA available={torch.cuda.is_available()}."
                ),
                12,
            )

            class PairDataset(Dataset):
                def __len__(
                    self_inner,
                ):
                    return (
                        len(pairs)
                        * SEMANTIC_TRAIN_PATCHES_PER_IMAGE
                    )

                def __getitem__(
                    self_inner,
                    index,
                ):
                    pair_index = int(
                        index
                        % len(pairs)
                    )

                    image_path, mask_path = (
                        pairs[pair_index]
                    )

                    image = (
                        _semantic_extract_2d_image(
                            _semantic_read_standard_image(
                                image_path
                            ),
                            source_path=image_path,
                        )
                    )

                    mask = (
                        _semantic_extract_2d_image(
                            _semantic_read_standard_image(
                                mask_path
                            ),
                            source_path=mask_path,
                        )
                    )

                    if mask.ndim != 2:
                        raise ValueError(
                            f"Mask must be grayscale 2D:\n{mask_path}"
                        )

                    rgb = (
                        _semantic_image_to_rgb_float(
                            image
                        )
                    )

                    binary = (
                        mask > 0
                    ).astype(
                        np.uint8
                    )

                    if rgb.shape[:2] != binary.shape:
                        raise ValueError(
                            "Image/mask size mismatch:\n"
                            f"Image: {image_path.name} {rgb.shape[:2]}\n"
                            f"Mask:  {mask_path.name} {binary.shape}"
                        )

                    rgb, binary = (
                        _semantic_random_patch(
                            rgb,
                            binary,
                            tile_size=(
                                SEMANTIC_TILE_SIZE
                            ),
                        )
                    )

                    rgb, binary = (
                        _semantic_augment_pair(
                            rgb,
                            binary,
                        )
                    )

                    x_np = (
                        _semantic_prepare_image_tensor(
                            rgb,
                            size=(
                                SEMANTIC_TILE_SIZE
                            ),
                            resize=False,
                        )
                    )

                    y_np = (
                        _semantic_prepare_mask(
                            binary,
                            size=(
                                SEMANTIC_TILE_SIZE
                            ),
                            resize=False,
                        )
                    )

                    return (
                        torch.from_numpy(
                            x_np
                        ),
                        torch.from_numpy(
                            y_np
                        ),
                    )

            self.progress.emit(
                (
                    "[6/8] Building DataLoader... "
                    f"{len(pairs)} training image(s), "
                    f"{SEMANTIC_TRAIN_PATCHES_PER_IMAGE} patch(es)/image/epoch."
                ),
                13,
            )

            loader = DataLoader(
                PairDataset(),
                batch_size=max(
                    1,
                    self.batch_size,
                ),
                shuffle=True,
                num_workers=0,
                pin_memory=(
                    torch.cuda.is_available()
                ),
            )

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            from semantic_models import (
                optimizer_parameter_groups,
                model_label,
            )

            selected_architecture = self.architecture_key
            selected_backbone = self.backbone_name

            self.progress.emit(
                (
                    "[7/8] Building semantic architecture...\n"
                    f"Model: {model_label(selected_architecture)}\n"
                    f"Backbone: {selected_backbone}"
                ),
                14,
            )

            self._check_stop()

            if self.initial_model_path is not None:
                # Continue mode restores architecture + backbone from the
                # checkpoint itself, so GUI selection cannot accidentally
                # reconstruct the wrong model.
                checkpoint_meta = torch.load(
                    self.initial_model_path,
                    map_location="cpu",
                )
                selected_architecture = str(
                    checkpoint_meta.get("architecture", "dinov2_unet")
                ).lower()
                if selected_architecture in {
                    "dinov2_vits14_unet_decoder",
                    "linknex_semantic_dinov2_unet",
                }:
                    selected_architecture = "dinov2_unet"
                selected_backbone = str(
                    checkpoint_meta.get("backbone_name", "dinov2_vits14")
                )

                _, model, _, _, _ = load_semantic_checkpoint(
                    self.initial_model_path
                )
                model = model.to(device)
                self.progress.emit(
                    (
                        "Continue training from complete checkpoint:\n"
                        f"{self.initial_model_path}\n"
                        f"Architecture: {selected_architecture}\n"
                        f"Backbone: {selected_backbone}"
                    ),
                    16,
                )
            else:
                encoder_path = self.initial_encoder_path
                if _semantic_arch_uses_dino(selected_architecture):
                    if encoder_path is None:
                        encoder_path = _find_local_dinov2_weights(
                            selected_backbone
                        )
                    self.progress.emit(
                        (
                            "DINOv2 encoder initialization:\n"
                            f"{encoder_path}"
                        ),
                        15,
                    )
                else:
                    encoder_path = None

                model, load_info = build_semantic_model_by_key(
                    torch,
                    nn,
                    F,
                    architecture_key=selected_architecture,
                    backbone_name=selected_backbone,
                    pretrained_encoder_path=encoder_path,
                )
                model = model.to(device)

                if _semantic_arch_uses_dino(selected_architecture):
                    self.progress.emit(
                        (
                            "Pretrained DINOv2 encoder loaded.\n"
                            f"Missing keys: {len(load_info.get('missing', []))} · "
                            f"Unexpected keys: {len(load_info.get('unexpected', []))}\n"
                            "Input channels: 1 (true grayscale)"
                        ),
                        16,
                    )

            self.architecture_key = selected_architecture
            self.backbone_name = selected_backbone

            optimizer = torch.optim.AdamW(
                optimizer_parameter_groups(
                    model,
                    encoder_lr=1e-5,
                    decoder_lr=1e-4,
                    weight_decay=1e-4,
                )
            )

            bce = (
                nn.BCEWithLogitsLoss()
            )

            scaler = (
                _semantic_make_grad_scaler(
                    torch,
                    device,
                )
            )

            self.progress.emit(
                (
                    f"Architecture: {model_label(self.architecture_key)}\n"
                    f"Backbone: {self.backbone_name}\n"
                    "Input: true 1-channel grayscale\n"
                    + (
                        f"Encoder init: Selected pretrained encoder\n"
                        f"Encoder: {self.initial_encoder_path}\n"
                        if self.initial_encoder_path is not None
                        else (
                            f"Encoder init: Full LinkNex checkpoint\n"
                            f"Checkpoint: {self.initial_model_path}\n"
                            if self.initial_model_path is not None
                            else (
                                (
                                    f"Encoder init: Official DINOv2\n"
                                    f"Weights: {_find_local_dinov2_weights(self.backbone_name)}\n"
                                )
                                if _semantic_arch_uses_dino(self.architecture_key)
                                else "Encoder init: architecture-native initialization\n"
                            )
                        )
                    )
                    + f"Training images: {len(pairs)}\n"
                    f"Patches/image/epoch: "
                    f"{SEMANTIC_TRAIN_PATCHES_PER_IMAGE}\n"
                    f"Patch size: {SEMANTIC_TILE_SIZE} x "
                    f"{SEMANTIC_TILE_SIZE}\n"
                    f"AMP: {'ON' if device.type == 'cuda' else 'OFF'}\n"
                    f"Device: {device}"
                ),
                17,
            )

            self.progress.emit(
                "[8/8] Optimizer/AMP ready. Entering training loop...",
                18,
            )

            total_batches = max(
                1,
                len(loader),
            )

            log_every = max(
                1,
                total_batches // 20,
            )

            preview_pairs = list(
                test_pairs[
                    :min(
                        len(test_pairs),
                        SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES,
                    )
                ]
            )

            preview_root = (
                self.dataset_root
                / SEMANTIC_TRAIN_PREVIEW_FOLDER
            )

            preview_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            metrics_csv = (
                self.dataset_root
                / SEMANTIC_TRAIN_METRICS_CSV
            )

            metrics_rows = []

            self.progress.emit(
                (
                    "Training history recording:\n"
                    "Every epoch: Train loss/Dice + COMPLETE Test loss/Dice/IoU\n"
                    f"Test images/epoch: {len(test_pairs)}\n"
                    f"Visual PNG previews/epoch: up to {len(preview_pairs)}\n"
                    f"Metrics CSV:\n{metrics_csv}\n"
                    f"Preview folder:\n{preview_root}"
                ),
                18,
            )

            for epoch in range(
                1,
                self.epochs + 1,
            ):
                epoch_started_at = time.perf_counter()
                model.train()

                loss_sum = 0.0
                dice_sum = 0.0
                n_seen = 0

                self.progress.emit(
                    (
                        f"Epoch {epoch}/{self.epochs} START · "
                        f"{total_batches} batch(es)"
                    ),
                    int(
                        18
                        + 80
                        * (
                            (epoch - 1)
                            / max(1, self.epochs)
                        )
                    ),
                )

                for batch_index, (x, y) in enumerate(
                    loader,
                    1,
                ):
                    self._check_stop()
                    if batch_index == 1:
                        self.progress.emit(
                            (
                                f"Epoch {epoch}/{self.epochs} · "
                                "first batch loaded; running GPU forward/backward..."
                            ),
                            int(
                                18
                                + 80
                                * (
                                    (
                                        (epoch - 1)
                                        * total_batches
                                        + batch_index
                                    )
                                    / max(
                                        1,
                                        self.epochs
                                        * total_batches,
                                    )
                                )
                            ),
                        )

                    x = x.to(
                        device,
                        non_blocking=True,
                    )
                    y = y.to(
                        device,
                        non_blocking=True,
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    with _semantic_autocast_context(
                        torch,
                        device,
                    ):
                        logits = model(
                            x
                        )

                        loss_bce = bce(
                            logits,
                            y,
                        )

                        loss_dice = (
                            _semantic_dice_loss(
                                torch,
                                logits,
                                y,
                            )
                        )

                        loss = (
                            loss_bce
                            + loss_dice
                        )

                    batch_dice = (
                        _semantic_binary_dice_from_logits(
                            torch,
                            logits.detach(),
                            y,
                            threshold=(
                                SEMANTIC_THRESHOLD
                            ),
                        )
                    )

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.step(
                        optimizer
                    )

                    scaler.update()
                    self._check_stop()

                    n = int(
                        x.size(0)
                    )

                    loss_sum += (
                        float(
                            loss.item()
                        )
                        * n
                    )

                    dice_sum += (
                        float(
                            batch_dice
                        )
                        * n
                    )

                    n_seen += n

                    if (
                        batch_index == 1
                        or batch_index == total_batches
                        or batch_index % log_every == 0
                    ):
                        running_loss = (
                            loss_sum
                            / max(
                                1,
                                n_seen,
                            )
                        )

                        running_dice = (
                            dice_sum
                            / max(
                                1,
                                n_seen,
                            )
                        )

                        global_fraction = (
                            (
                                (epoch - 1)
                                * total_batches
                                + batch_index
                            )
                            / max(
                                1,
                                self.epochs
                                * total_batches,
                            )
                        )

                        progress_value = min(
                            98,
                            int(
                                round(
                                    18
                                    + 80
                                    * global_fraction
                                )
                            ),
                        )

                        self.progress.emit(
                            (
                                f"Epoch {epoch}/{self.epochs} · "
                                f"Batch {batch_index}/{total_batches} · "
                                f"loss={running_loss:.4f} · "
                                f"Dice={running_dice:.4f}"
                            ),
                            progress_value,
                        )

                mean_loss = (
                    loss_sum
                    / max(
                        1,
                        n_seen,
                    )
                )

                mean_train_dice = (
                    dice_sum
                    / max(
                        1,
                        n_seen,
                    )
                )

                progress_value = min(
                    98,
                    int(
                        round(
                            18
                            + 80
                            * epoch
                            / max(
                                1,
                                self.epochs,
                            )
                        )
                    ),
                )

                self.progress.emit(
                    (
                        f"Epoch {epoch}/{self.epochs} COMPLETE · "
                        f"loss={mean_loss:.4f} · "
                        f"Train Dice={mean_train_dice:.4f}"
                    ),
                    progress_value,
                )

                # Full Test evaluation after EVERY completed epoch.
                self.progress.emit(
                    (
                        f"Epoch {epoch}: evaluating COMPLETE Test split "
                        f"({len(test_pairs)} image(s))..."
                    ),
                    progress_value,
                )

                (
                    mean_test_loss,
                    mean_test_dice,
                    mean_test_iou,
                    test_count,
                    epoch_dir,
                ) = _semantic_evaluate_full_test_epoch(
                    torch,
                    model,
                    device,
                    test_pairs,
                    self.dataset_root,
                    epoch,
                    stop_callback=self._check_stop,
                )

                epoch_seconds = float(
                    time.perf_counter() - epoch_started_at
                )

                encoder_lr = float(
                    optimizer.param_groups[0].get("lr", 0.0)
                )
                decoder_lr = float(
                    optimizer.param_groups[1].get("lr", encoder_lr)
                    if len(optimizer.param_groups) > 1
                    else encoder_lr
                )

                self.progress.emit(
                    (
                        f"Epoch {epoch}/{self.epochs} HISTORY · "
                        f"Train loss={mean_loss:.4f} · "
                        f"Train Dice={mean_train_dice:.4f} · "
                        f"Test loss={mean_test_loss:.4f} · "
                        f"Test Dice={mean_test_dice:.4f} · "
                        f"Test IoU={mean_test_iou:.4f} · "
                        f"{epoch_seconds:.1f}s"
                    ),
                    progress_value,
                )

                metrics_rows.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(mean_loss),
                        "train_dice": float(mean_train_dice),
                        "test_loss": float(mean_test_loss),
                        "test_dice": float(mean_test_dice),
                        "test_iou": float(mean_test_iou),
                        "encoder_lr": encoder_lr,
                        "decoder_lr": decoder_lr,
                        "epoch_seconds": epoch_seconds,
                        "train_patch_samples": int(n_seen),
                        "test_images": int(test_count),
                        "test_results_folder": (
                            str(epoch_dir) if epoch_dir is not None else ""
                        ),
                    }
                )

                # Rewrite after every epoch so interrupted training still keeps
                # every completed epoch on disk.
                with open(
                    metrics_csv,
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "train_loss",
                            "train_dice",
                            "test_loss",
                            "test_dice",
                            "test_iou",
                            "encoder_lr",
                            "decoder_lr",
                            "epoch_seconds",
                            "train_patch_samples",
                            "test_images",
                            "test_results_folder",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(metrics_rows)

                _semantic_save_training_curves(
                    metrics_rows,
                    self.dataset_root,
                )

            self.progress.emit(
                "Training finished. Saving LinkODT semantic checkpoint...",
                99,
            )

            self.dataset_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                model_path,
                trained_timestamp,
            ) = (
                _semantic_timestamped_model_path(self.architecture_key)
            )

            torch.save(
                {
                    "model_name": (
                        "LinkNex_Semantic_Model"
                    ),
                    "trained_at": (
                        trained_timestamp
                    ),
                    "architecture": self.architecture_key,
                    "backbone_name": self.backbone_name,
                    "input_channels": 1,
                    "training_mode": (
                        "continue_existing_model"
                        if self.initial_model_path is not None
                        else "pretrained_encoder_new_segmentation_head"
                    ),
                    "resume_from": (
                        str(self.initial_model_path)
                        if self.initial_model_path is not None
                        else None
                    ),
                    "encoder_init_from": (
                        str(self.initial_encoder_path)
                        if self.initial_encoder_path is not None
                        else (
                            str(_find_local_dinov2_weights(self.backbone_name))
                            if (
                                self.initial_model_path is None
                                and _semantic_arch_uses_dino(self.architecture_key)
                            )
                            else None
                        )
                    ),
                    "state_dict": (
                        model.state_dict()
                    ),
                    "input_size": (
                        SEMANTIC_MODEL_INPUT_SIZE
                    ),
                    "tile_size": (
                        SEMANTIC_TILE_SIZE
                    ),
                    "tile_overlap": (
                        SEMANTIC_TILE_OVERLAP
                    ),
                    "train_patches_per_image": (
                        SEMANTIC_TRAIN_PATCHES_PER_IMAGE
                    ),
                    "train_preview_every_epochs": (
                        SEMANTIC_TRAIN_PREVIEW_EVERY_EPOCHS
                    ),
                    "train_preview_max_images": (
                        SEMANTIC_TRAIN_PREVIEW_MAX_IMAGES
                    ),
                    "threshold": (
                        SEMANTIC_THRESHOLD
                    ),
                    "classes": {
                        "background": 0,
                        "foreground": 1,
                    },
                    "training_history": metrics_rows,
                    "training_metrics_csv": str(metrics_csv),
                    "training_loss_curve_png": str(
                        self.dataset_root / SEMANTIC_TRAIN_LOSS_CURVE_PNG
                    ),
                    "training_dice_curve_png": str(
                        self.dataset_root / SEMANTIC_TRAIN_DICE_CURVE_PNG
                    ),
                },
                model_path,
            )

            self.progress.emit(
                (
                    "Checkpoint saved successfully:\n"
                    f"{model_path}"
                ),
                100,
            )

            self.finished.emit(
                str(model_path),
                (
                    "TRAIN COMPLETE\n"
                    + (
                        (
                            f"Training mode: CONTINUE EXISTING LINKNEX MODEL\n"
                            f"Complete model: {self.initial_model_path}\n"
                        )
                        if self.initial_model_path is not None
                        else (
                            f"Training mode: PRETRAINED ENCODER + NEW SEGMENTATION HEAD\n"
                            f"Encoder init: {self.initial_encoder_path}\n"
                        )
                    )
                    + f"Train pairs: {len(pairs)}\n"
                    f"Training metrics:\n{metrics_csv}\n"
                    f"Epoch test PNGs:\n{preview_root}\n"
                    f"Model:\n{model_path}"
                ),
            )

        except InterruptedError:
            try:
                if "torch" in locals() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.stopped.emit(
                "TRAINING STOPPED BY USER\n"
                "Training ended safely after the active batch/operation.\n"
                "No new partial model checkpoint was saved.\n"
                "Completed epoch metrics and test_results remain on disk."
            )

        except Exception as exc:
            self.failed.emit(
                str(exc)
            )


class SemanticTestWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        dataset_root,
        pairs_to_use,
        test_percent,
        model_path,
    ):
        super().__init__()

        self.dataset_root = Path(
            dataset_root
        )
        self.pairs_to_use = int(
            pairs_to_use
        )
        self.test_percent = int(
            test_percent
        )
        self.model_path = Path(
            model_path
        )

    def run(self):
        try:
            self.progress.emit(
                "Checking image/mask pairs and dimensions...",
                1,
            )

            (
                train_pairs,
                test_pairs,
                manifest,
                total_pairs,
                valid_count,
                skipped_count,
                selected_count,
                dimension_csv,
                skipped_csv,
            ) = _semantic_get_or_create_split(
                self.dataset_root,
                pairs_to_use=self.pairs_to_use,
                test_percent=self.test_percent,
                progress_callback=lambda text, value: (
                    self.progress.emit(
                        str(text),
                        int(value),
                    )
                ),
            )

            pairs = list(
                test_pairs
            )

            if not pairs:
                raise ValueError(
                    "No valid test pairs were selected."
                )

            use_text = (
                "ALL VALID"
                if self.pairs_to_use >= valid_count
                else str(selected_count)
            )

            self.progress.emit(
                (
                    "TEST DATASET READY\n"
                    f"Total image/mask pairs: {total_pairs}\n"
                    f"Valid pairs: {valid_count}\n"
                    f"Bad pairs skipped: {skipped_count}\n"
                    f"Pairs used: {use_text}\n"
                    f"Test pairs: {len(test_pairs)}\n"
                    f"Dimension report:\n{dimension_csv}\n"
                    f"Skipped report:\n{skipped_csv}\n"
                    f"Manifest:\n{manifest}"
                ),
                3,
            )

            self.progress.emit(
                "Loading semantic model...",
                4,
            )

            (
                torch,
                model,
                device,
                input_size,
                threshold,
            ) = load_semantic_checkpoint(
                self.model_path
            )

            rows = []

            dice_values = []
            iou_values = []

            for index, (
                image_path,
                mask_path,
            ) in enumerate(
                pairs,
                1,
            ):
                image = (
                    _semantic_extract_2d_image(
                        _semantic_read_standard_image(
                            image_path
                        )
                    )
                )

                mask = (
                    _semantic_extract_2d_image(
                        _semantic_read_standard_image(
                            mask_path
                        )
                    )
                )

                if mask.ndim != 2:
                    raise ValueError(
                        f"Mask must be 2D:\n{mask_path}"
                    )

                rgb = (
                    _semantic_image_to_rgb_float(
                        image
                    )
                )

                truth = (
                    mask > 0
                ).astype(
                    np.uint8
                )

                if rgb.shape[:2] != truth.shape:
                    raise ValueError(
                        "Image/mask size mismatch:\n"
                        f"Image: {image_path.name} {rgb.shape[:2]}\n"
                        f"Mask:  {mask_path.name} {truth.shape}"
                    )

                pred, _ = (
                    semantic_predict_tiled_rgb(
                        torch,
                        model,
                        device,
                        input_size,
                        threshold,
                        rgb,
                        tile_size=(
                            SEMANTIC_TILE_SIZE
                        ),
                        overlap=(
                            SEMANTIC_TILE_OVERLAP
                        ),
                        infer_batch_size=(
                            SEMANTIC_INFER_BATCH_SIZE
                        ),
                    )
                )

                intersection = float(
                    np.logical_and(
                        pred > 0,
                        truth > 0,
                    ).sum()
                )

                pred_sum = float(
                    (pred > 0).sum()
                )

                truth_sum = float(
                    (truth > 0).sum()
                )

                union = float(
                    np.logical_or(
                        pred > 0,
                        truth > 0,
                    ).sum()
                )

                dice = (
                    (
                        2.0 * intersection
                        + 1.0
                    )
                    / (
                        pred_sum
                        + truth_sum
                        + 1.0
                    )
                )

                iou = (
                    (
                        intersection + 1.0
                    )
                    / (
                        union + 1.0
                    )
                )

                dice_values.append(
                    dice
                )
                iou_values.append(
                    iou
                )

                rows.append(
                    {
                        "image": (
                            image_path.name
                        ),
                        "mask": (
                            mask_path.name
                        ),
                        "dice": (
                            f"{dice:.6f}"
                        ),
                        "iou": (
                            f"{iou:.6f}"
                        ),
                    }
                )

                self.progress.emit(
                    (
                        f"{image_path.name}   "
                        f"Dice={dice:.4f}   "
                        f"IoU={iou:.4f}"
                    ),
                    int(
                        round(
                            100
                            * index
                            / len(pairs)
                        )
                    ),
                )

            result_csv = (
                self.dataset_root
                / "Semantic_test_results.csv"
            )

            with open(
                result_csv,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image",
                        "mask",
                        "dice",
                        "iou",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    rows
                )

            mean_dice = float(
                np.mean(
                    dice_values
                )
            )

            mean_iou = float(
                np.mean(
                    iou_values
                )
            )

            self.finished.emit(
                (
                    "TEST COMPLETE\n"
                    f"Pairs: {len(pairs)}\n"
                    f"Mean Dice: {mean_dice:.4f}\n"
                    f"Mean IoU: {mean_iou:.4f}\n"
                    f"Tile: {SEMANTIC_TILE_SIZE} x "
                    f"{SEMANTIC_TILE_SIZE}, "
                    f"overlap={SEMANTIC_TILE_OVERLAP}\n\n"
                    f"Results CSV:\n{result_csv}"
                )
            )

        except Exception as exc:
            self.failed.emit(
                str(exc)
            )


class SemanticBatchPredictWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image_paths,
        model_path,
        predict_threshold=SEMANTIC_THRESHOLD,
    ):
        super().__init__()

        self.image_paths = [
            Path(p)
            for p in image_paths
        ]

        self.model_path = Path(
            model_path
        )

        self.predict_threshold = float(
            predict_threshold
        )

        if not (
            0.0 < self.predict_threshold < 1.0
        ):
            raise ValueError(
                "Predict threshold must be between 0 and 1."
            )

    def _predict_one_slice(
        self,
        torch,
        model,
        device,
        input_size,
        threshold,
        rgb,
    ):
        requested_batch = max(
            1,
            int(
                SEMANTIC_INFER_BATCH_SIZE
            ),
        )

        batch_candidates = []

        for value in (
            requested_batch,
            2,
            1,
        ):
            value = max(
                1,
                min(
                    requested_batch,
                    int(value),
                ),
            )

            if value not in batch_candidates:
                batch_candidates.append(
                    value
                )

        last_error = None

        for infer_batch_size in batch_candidates:
            try:
                binary, probability = (
                    semantic_predict_tiled_rgb(
                        torch,
                        model,
                        device,
                        input_size,
                        threshold,
                        rgb,
                        tile_size=(
                            SEMANTIC_TILE_SIZE
                        ),
                        overlap=(
                            SEMANTIC_TILE_OVERLAP
                        ),
                        infer_batch_size=(
                            infer_batch_size
                        ),
                    )
                )

                del probability

                return (
                    binary,
                    infer_batch_size,
                )

            except RuntimeError as exc:
                last_error = exc

                text = str(
                    exc
                ).lower()

                if "out of memory" not in text:
                    raise

                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        raise RuntimeError(
            "CUDA out of memory even after retrying semantic inference "
            f"with batch sizes {batch_candidates}.\n\n"
            f"Last CUDA error:\n{last_error}"
        )

    def run(self):
        current_image_path = None
        current_binary_mask_path = None
        current_instance_mask_path = None
        current_binary_partial_path = None
        current_instance_partial_path = None
        current_z = None

        try:
            if not self.image_paths:
                raise ValueError(
                    "No images selected for prediction."
                )

            self.progress.emit(
                (
                    "Loading semantic model in background...\n"
                    f"{self.model_path}"
                ),
                0,
            )

            (
                torch,
                model,
                device,
                input_size,
                checkpoint_threshold,
            ) = load_semantic_checkpoint(
                self.model_path
            )

            # Prediction threshold is intentionally user-adjustable from the
            # Semantic main window. It changes only binary post-processing;
            # it does NOT change model weights or training.
            threshold = float(
                self.predict_threshold
            )

            self.progress.emit(
                (
                    "Semantic model ready. "
                    f"Device={device} · tile={SEMANTIC_TILE_SIZE} · "
                    f"overlap={SEMANTIC_TILE_OVERLAP} · "
                    f"threshold={threshold:.2f} "
                    f"({threshold * 100:.0f}%)"
                ),
                1,
            )

            saved_masks = []
            saved_instance_masks = []

            total_images = len(
                self.image_paths
            )

            for image_index, image_path in enumerate(
                self.image_paths,
                1,
            ):
                reader = None
                binary_memmap = None
                instance_memmap = None

                current_image_path = Path(
                    image_path
                )

                current_binary_mask_path = image_path.with_name(
                    f"{image_path.stem}_mask.tif"
                )

                current_instance_mask_path = image_path.with_name(
                    f"{image_path.stem}_mask_instance.tif"
                )

                current_binary_partial_path = image_path.with_name(
                    f".{image_path.stem}_mask.partial.tif"
                )

                current_instance_partial_path = image_path.with_name(
                    f".{image_path.stem}_mask_instance.partial.tif"
                )

                current_z = None

                try:
                    for p in (
                        current_binary_partial_path,
                        current_instance_partial_path,
                    ):
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass

                    reader = ImageVolumeReader(
                        image_path
                    )

                    self.progress.emit(
                        (
                            f"Image {image_index}/{total_images}: "
                            f"{image_path.name}\n"
                            f"Source size: "
                            f"{reader.width} x {reader.height} x {reader.z_count}\n"
                            f"Internal inference: "
                            f"{SEMANTIC_TILE_SIZE}x{SEMANTIC_TILE_SIZE} tiles\n"
                            f"Predict threshold: {threshold:.2f} "
                            f"({threshold * 100:.0f}%)\n"
                            f"Outputs:\n"
                            f"  {current_binary_mask_path.name}\n"
                            f"  {current_instance_mask_path.name}"
                        ),
                        int(
                            round(
                                100
                                * (
                                    image_index - 1
                                )
                                / max(
                                    1,
                                    total_images,
                                )
                            )
                        ),
                    )

                    if reader.z_count == 1:
                        output_shape = (
                            int(reader.height),
                            int(reader.width),
                        )
                        output_axes = "YX"
                    else:
                        output_shape = (
                            int(reader.z_count),
                            int(reader.height),
                            int(reader.width),
                        )
                        output_axes = "ZYX"

                    binary_memmap = tifffile.memmap(
                        current_binary_partial_path,
                        shape=output_shape,
                        dtype=np.uint8,
                        photometric="minisblack",
                        metadata={"axes": output_axes},
                    )

                    instance_memmap = tifffile.memmap(
                        current_instance_partial_path,
                        shape=output_shape,
                        dtype=np.uint32,
                        photometric="minisblack",
                        metadata={"axes": output_axes},
                    )

                    for z_index in range(
                        reader.z_count
                    ):
                        current_z = int(
                            z_index
                        )

                        raw = reader.get_slice(
                            z_index
                        )

                        rgb = image_to_rgb8(
                            raw
                        )

                        binary, used_batch = (
                            self._predict_one_slice(
                                torch,
                                model,
                                device,
                                input_size,
                                threshold,
                                rgb,
                            )
                        )

                        labels, _ = split_binary_mask_fast(
                            binary
                        )

                        binary_slice = (
                            binary.astype(
                                np.uint8,
                                copy=False,
                            )
                            * 255
                        )

                        instance_slice = labels.astype(
                            np.uint32,
                            copy=False,
                        )

                        if reader.z_count == 1:
                            binary_memmap[...] = binary_slice
                            instance_memmap[...] = instance_slice
                        else:
                            binary_memmap[
                                z_index,
                                ...,
                            ] = binary_slice
                            instance_memmap[
                                z_index,
                                ...,
                            ] = instance_slice

                        binary_memmap.flush()
                        instance_memmap.flush()

                        image_fraction = (
                            (
                                z_index + 1
                            )
                            / max(
                                1,
                                reader.z_count,
                            )
                        )

                        overall = int(
                            round(
                                100
                                * (
                                    (
                                        image_index - 1
                                    )
                                    + image_fraction
                                )
                                / total_images
                            )
                        )

                        self.progress.emit(
                            (
                                f"{image_path.name} · "
                                f"Z {z_index + 1}/{reader.z_count} · "
                                f"infer batch={used_batch} · "
                                f"binary + instance in progress"
                            ),
                            overall,
                        )

                    for mem in (
                        binary_memmap,
                        instance_memmap,
                    ):
                        mem.flush()

                        try:
                            underlying_mmap = getattr(
                                mem,
                                "_mmap",
                                None,
                            )
                            if underlying_mmap is not None:
                                underlying_mmap.close()
                        except Exception:
                            pass

                    del binary_memmap
                    del instance_memmap
                    binary_memmap = None
                    instance_memmap = None

                    current_binary_partial_path.replace(
                        current_binary_mask_path
                    )
                    current_instance_partial_path.replace(
                        current_instance_mask_path
                    )

                    if not current_binary_mask_path.exists():
                        raise RuntimeError(
                            "Prediction finished but final *_mask.tif "
                            "was not found after saving."
                        )

                    if not current_instance_mask_path.exists():
                        raise RuntimeError(
                            "Prediction finished but final *_mask_instance.tif "
                            "was not found after saving."
                        )

                    saved_masks.append(
                        str(
                            current_binary_mask_path
                        )
                    )

                    saved_instance_masks.append(
                        str(
                            current_instance_mask_path
                        )
                    )

                    self.progress.emit(
                        (
                            "Saved successfully:\n"
                            f"{current_binary_mask_path.name}\n"
                            f"{current_instance_mask_path.name}\n"
                            f"Output size: "
                            f"{reader.width} x {reader.height} x {reader.z_count}"
                        ),
                        int(
                            round(
                                100
                                * image_index
                                / total_images
                            )
                        ),
                    )

                except Exception as image_exc:
                    for mem in (
                        binary_memmap,
                        instance_memmap,
                    ):
                        try:
                            if mem is not None:
                                mem.flush()

                                underlying_mmap = getattr(
                                    mem,
                                    "_mmap",
                                    None,
                                )

                                if underlying_mmap is not None:
                                    underlying_mmap.close()
                        except Exception:
                            pass

                    binary_memmap = None
                    instance_memmap = None

                    for p in (
                        current_binary_partial_path,
                        current_instance_partial_path,
                    ):
                        try:
                            if p is not None and p.exists():
                                p.unlink()
                        except Exception:
                            pass

                    if reader is None:
                        size_text = "unknown"
                    else:
                        size_text = (
                            f"{reader.width} x "
                            f"{reader.height} x "
                            f"{reader.z_count}"
                        )

                    if current_z is None:
                        stage_text = (
                            "before Z inference"
                        )
                    elif reader is not None:
                        stage_text = (
                            f"while processing "
                            f"Z {current_z + 1}/{reader.z_count}"
                        )
                    else:
                        stage_text = (
                            f"while processing Z {current_z + 1}"
                        )

                    raise RuntimeError(
                        "Semantic prediction failed.\n\n"
                        f"Image:\n{current_image_path}\n\n"
                        f"Image size:\n{size_text}\n\n"
                        f"Stage:\n{stage_text}\n\n"
                        f"Model:\n{self.model_path}\n\n"
                        f"Intended outputs:\n"
                        f"{current_binary_mask_path}\n"
                        f"{current_instance_mask_path}\n\n"
                        f"Original error:\n{image_exc}"
                    ) from image_exc

                finally:
                    if reader is not None:
                        try:
                            reader.close()
                        except Exception:
                            pass

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            self.finished.emit(
                {
                    "image_paths": [
                        str(p)
                        for p in self.image_paths
                    ],
                    "saved_masks": saved_masks,
                    "saved_instance_masks": saved_instance_masks,
                }
            )

        except Exception as exc:
            self.failed.emit(
                str(exc)
            )


class SemanticTrainingWindow(QMainWindow):
    modelReady = Signal(str)

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self.setWindowTitle(
            "LinkODT Semantic Model · Multi-Architecture"
        )

        self.resize(
            760,
            640,
        )
        self.setMinimumSize(
            690,
            560,
        )

        self.dataset_root = None
        self.model_path = None

        self.train_thread = None
        self.train_worker = None

        self.test_thread = None
        self.test_worker = None

        # Tracks which operation is currently running so a failure resets
        # only that button. A previously completed Train/Test button remains
        # green instead of losing its completed state.
        self.active_action = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )
        root.setContentsMargins(
            18,
            16,
            18,
            18,
        )
        root.setSpacing(
            10
        )

        title = QLabel(
            "Semantic segmentation · single-channel microscopy"
        )
        title.setObjectName(
            "SemanticTrainerTitle"
        )
        root.addWidget(
            title
        )

        dataset_row = QHBoxLayout()

        dataset_label = QLabel(
            "Dataset"
        )
        dataset_label.setObjectName(
            "SemanticPathLabel"
        )
        dataset_label.setFixedWidth(
            80
        )

        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "Select semantic dataset root..."
        )

        dataset_browse = QPushButton(
            "Browse"
        )
        dataset_browse.clicked.connect(
            self.select_dataset
        )

        dataset_row.addWidget(
            dataset_label
        )
        dataset_row.addWidget(
            self.dataset_edit,
            1,
        )
        dataset_row.addWidget(
            dataset_browse
        )

        root.addLayout(
            dataset_row
        )

        architecture_row = QHBoxLayout()
        architecture_label = QLabel("Model")
        architecture_label.setObjectName("SemanticPathLabel")
        architecture_label.setFixedWidth(80)
        self.architecture_combo = QComboBox()
        for key, spec in _semantic_model_registry().items():
            self.architecture_combo.addItem(spec["label"], key)
        default_index = self.architecture_combo.findData(
            SEMANTIC_DEFAULT_ARCHITECTURE
        )
        if default_index >= 0:
            self.architecture_combo.setCurrentIndex(default_index)
        self.architecture_combo.setMinimumWidth(360)
        self.architecture_combo.currentIndexChanged.connect(
            self._architecture_changed
        )
        architecture_row.addWidget(architecture_label)
        architecture_row.addWidget(self.architecture_combo)
        architecture_row.addStretch(1)
        root.addLayout(architecture_row)

        backbone_row = QHBoxLayout()
        backbone_label = QLabel("Backbone")
        backbone_label.setObjectName("SemanticPathLabel")
        backbone_label.setFixedWidth(80)
        self.backbone_combo = QComboBox()
        for key, spec in _semantic_backbone_registry().items():
            self.backbone_combo.addItem(spec["label"], key)
        default_backbone_index = self.backbone_combo.findData(
            SEMANTIC_DEFAULT_BACKBONE
        )
        if default_backbone_index >= 0:
            self.backbone_combo.setCurrentIndex(default_backbone_index)
        self.backbone_combo.setMinimumWidth(300)
        self.backbone_combo.currentIndexChanged.connect(
            self._backbone_changed
        )
        input_label = QLabel("Input: 2D · 1 grayscale channel")
        input_label.setObjectName("SemanticMetric")
        backbone_row.addWidget(backbone_label)
        backbone_row.addWidget(self.backbone_combo)
        backbone_row.addWidget(input_label)
        backbone_row.addStretch(1)
        root.addLayout(backbone_row)

        model_row = QHBoxLayout()

        model_label = QLabel(
            "Continue Model"
        )
        model_label.setObjectName(
            "SemanticPathLabel"
        )
        model_label.setFixedWidth(
            125
        )

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(
            "Browse existing semantic model..."
        )

        self.model_browse_btn = QPushButton(
            "Browse"
        )
        self.model_browse_btn.clicked.connect(
            self.browse_model
        )

        self.load_model_btn = QPushButton(
            "Load model"
        )
        self.load_model_btn.setObjectName(
            "SemanticLoadModel"
        )
        self.load_model_btn.clicked.connect(
            self.load_model
        )

        model_row.addWidget(
            model_label
        )
        model_row.addWidget(
            self.model_edit,
            1,
        )
        model_row.addWidget(
            self.model_browse_btn
        )
        model_row.addWidget(
            self.load_model_btn
        )

        root.addLayout(
            model_row
        )

        self.model_status = QLabel(
            "Loaded model: —"
        )
        self.model_status.setObjectName(
            "SemanticMetric"
        )
        self.model_status.setWordWrap(
            True
        )
        root.addWidget(
            self.model_status
        )

        encoder_row = QHBoxLayout()

        encoder_label = QLabel(
            "Pretrained Encoder"
        )
        encoder_label.setObjectName(
            "SemanticPathLabel"
        )
        encoder_label.setFixedWidth(
            145
        )

        self.ssl_encoder_edit = QLineEdit()
        self.ssl_encoder_edit.setPlaceholderText(
            "Select encoder-only DINOv2 .pt/.pth..."
        )
        self.ssl_encoder_edit.setMaximumWidth(
            430
        )

        try:
            _default_pretrained_encoder = (
                _find_local_dinov2_weights(
                    self.backbone_combo.currentData()
                )
            )
            if (
                _default_pretrained_encoder is not None
                and Path(_default_pretrained_encoder).exists()
            ):
                self.ssl_encoder_edit.setText(
                    str(_default_pretrained_encoder)
                )
        except Exception:
            pass

        self.ssl_encoder_browse_btn = QPushButton(
            "Browse"
        )
        self.ssl_encoder_browse_btn.clicked.connect(
            self.browse_ssl_encoder
        )

        self.weights_help_btn = QPushButton(
            "Weights Help"
        )
        self.weights_help_btn.clicked.connect(
            self.show_weights_help
        )

        encoder_row.addWidget(
            encoder_label
        )
        encoder_row.addWidget(
            self.ssl_encoder_edit
        )
        encoder_row.addWidget(
            self.ssl_encoder_browse_btn
        )
        encoder_row.addWidget(
            self.weights_help_btn
        )
        encoder_row.addStretch(
            1
        )

        root.addLayout(
            encoder_row
        )

        training_mode_row = QHBoxLayout()

        training_mode_label = QLabel(
            "Training mode"
        )
        training_mode_label.setObjectName(
            "SemanticPathLabel"
        )
        training_mode_label.setFixedWidth(
            120
        )

        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItem(
            "Pretrained encoder · new segmentation head",
            "pretrained",
        )
        self.training_mode_combo.addItem(
            "Continue existing LinkODT model",
            "continue",
        )
        self.training_mode_combo.setCurrentIndex(0)
        self.training_mode_combo.setMaximumWidth(
            390
        )
        self.training_mode_combo.currentIndexChanged.connect(
            self._training_mode_changed
        )

        training_mode_row.addWidget(
            training_mode_label
        )
        training_mode_row.addWidget(
            self.training_mode_combo
        )
        training_mode_row.addStretch(
            1
        )
        root.addLayout(
            training_mode_row
        )

        self._architecture_changed()
        self._training_mode_changed()

        settings = QHBoxLayout()

        epoch_label = QLabel(
            "Epochs"
        )
        epoch_label.setObjectName(
            "SemanticPathLabel"
        )

        self.epochs_edit = QLineEdit(
            "20"
        )
        self.epochs_edit.setAlignment(
            Qt.AlignCenter
        )
        self.epochs_edit.setFixedWidth(
            70
        )

        batch_label = QLabel(
            "Batch"
        )
        batch_label.setObjectName(
            "SemanticPathLabel"
        )

        self.batch_edit = QLineEdit(
            "4"
        )
        self.batch_edit.setAlignment(
            Qt.AlignCenter
        )
        self.batch_edit.setFixedWidth(
            65
        )

        pairs_label = QLabel(
            "Pairs to use"
        )
        pairs_label.setObjectName(
            "SemanticPathLabel"
        )

        self.pairs_to_use_edit = QLineEdit(
            str(
                SEMANTIC_DEFAULT_PAIRS_TO_USE
            )
        )
        self.pairs_to_use_edit.setAlignment(
            Qt.AlignCenter
        )
        self.pairs_to_use_edit.setFixedWidth(
            85
        )

        test_label = QLabel(
            "Test %"
        )
        test_label.setObjectName(
            "SemanticPathLabel"
        )

        self.test_percent_edit = QLineEdit(
            str(
                SEMANTIC_DEFAULT_TEST_PERCENT
            )
        )
        self.test_percent_edit.setAlignment(
            Qt.AlignCenter
        )
        self.test_percent_edit.setFixedWidth(
            65
        )

        settings.addWidget(
            epoch_label
        )
        settings.addWidget(
            self.epochs_edit
        )
        settings.addSpacing(
            8
        )
        settings.addWidget(
            batch_label
        )
        settings.addWidget(
            self.batch_edit
        )
        settings.addSpacing(
            8
        )
        settings.addWidget(
            pairs_label
        )
        settings.addWidget(
            self.pairs_to_use_edit
        )
        settings.addSpacing(
            8
        )
        settings.addWidget(
            test_label
        )
        settings.addWidget(
            self.test_percent_edit
        )
        settings.addStretch(
            1
        )

        root.addLayout(
            settings
        )

        action_row = QHBoxLayout()
        action_row.addStretch(
            1
        )

        self.train_btn = QPushButton(
            "Train"
        )
        self.train_btn.setObjectName(
            "SemanticTrain"
        )
        self.train_btn.setProperty(
            "state",
            "idle",
        )
        self.train_btn.clicked.connect(
            self.start_train
        )

        self.test_btn = QPushButton(
            "Test"
        )
        self.test_btn.setObjectName(
            "SemanticTest"
        )
        self.test_btn.setProperty(
            "state",
            "idle",
        )
        self.test_btn.clicked.connect(
            self.start_test
        )

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("SemanticStop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)

        action_row.addWidget(self.train_btn)
        action_row.addWidget(self.test_btn)
        action_row.addWidget(self.stop_btn)

        root.addLayout(
            action_row
        )

        self.progress = QProgressBar()
        self.progress.setRange(
            0,
            100,
        )
        self.progress.setValue(
            0
        )
        root.addWidget(
            self.progress
        )

        result_label = QLabel(
            "RESULTS"
        )
        result_label.setObjectName(
            "SemanticSection"
        )
        root.addWidget(
            result_label
        )

        self.log = QPlainTextEdit()
        self.log.setMinimumHeight(
            270
        )
        self.log.setReadOnly(
            True
        )
        root.addWidget(
            self.log,
            1,
        )

    def _apply_style(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {OFFWHITE};
                color: {TEXT};
            }}

            QWidget {{
                color: {TEXT};
                font-family: "Segoe UI";
                font-size: 14px;
            }}

            QLabel#SemanticTrainerTitle {{
                color: {DEEP_BLUE};
                font-size: 16px;
                font-weight: 600;
            }}

            QLabel#SemanticTrainerHint {{
                color: {MUTED};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 7px;
                font-weight: 600;
            }}

            QLabel#SemanticPathLabel {{
                color: {MUTED};
                font-size: 13px;
                font-weight: 600;
            }}

            QLabel#SemanticMetric {{
                color: {TEXT};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 6px;
                font-weight: 700;
            }}

            QLabel#SemanticSection {{
                color: {DEEP_BLUE};
                font-size: 14px;
                font-weight: 600;
            }}

            QLineEdit {{
                min-height: 32px;
                padding: 2px 7px;
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                font-weight: 700;
            }}

            QComboBox {{
                min-height: 32px;
                padding: 2px 7px;
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                font-weight: 600;
            }}

            QComboBox QAbstractItemView {{
                background: {WHITE};
                color: {TEXT};
                selection-background-color: #DCEAF7;
                selection-color: {NAVY};
            }}

            QPushButton {{
                min-height: 34px;
                padding: 3px 11px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {NAVY};
                font-weight: 600;
            }}

            QPushButton:hover {{
                background: #EEF7FD;
                border-color: {US_BLUE};
            }}

            QPushButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
            }}

            QPushButton#SemanticLoadModel {{
                color: {DEEP_BLUE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#SemanticTrain {{
                min-width: 120px;
                background: {NAVY};
                color: {WHITE};
                border: 2.5px solid {NAVY};
                font-weight: 600;
            }}

            QPushButton#SemanticTest {{
                min-width: 120px;
                background: {US_BLUE};
                color: {WHITE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#SemanticTrain[state="working"],
            QPushButton#SemanticTest[state="working"],
            QPushButton#SemanticTrain:disabled[state="working"],
            QPushButton#SemanticTest:disabled[state="working"] {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#SemanticStop {{
                min-width: 90px;
                background: #D94B4B;
                color: {WHITE};
                border: 2.5px solid #AE3232;
                font-weight: 600;
            }}
            QPushButton#SemanticStop:disabled {{
                background: #E5E9EF;
                color: #9AA7B4;
                border-color: {BORDER};
            }}
            QPushButton#SemanticTrain[state="stopped"] {{
                background: #D94B4B;
                color: {WHITE};
                border-color: #AE3232;
            }}

            QPushButton#SemanticTrain[state="done"],
            QPushButton#SemanticTest[state="done"] {{
                background: #1E9E5A;
                color: {WHITE};
                border-color: #167B45;
            }}

            QProgressBar {{
                min-height: 20px;
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 5px;
                text-align: center;
                font-weight: 600;
            }}

            QProgressBar::chunk {{
                background: {US_BLUE};
            }}

            QPlainTextEdit {{
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 7px;
                font-family: "Consolas";
                font-size: 13px;
            }}
            """ + DISABLED_STYLE
        )

    def _refresh_button(
        self,
        button,
    ):
        style = button.style()
        style.unpolish(
            button
        )
        style.polish(
            button
        )
        button.update()

    def _set_train_state(
        self,
        state,
    ):
        if state == "working":
            self.train_btn.setText(
                "Training..."
            )
        elif state == "done":
            self.train_btn.setText(
                "Trained ✓"
            )
        elif state == "stopped":
            self.train_btn.setText("Stopped")
        else:
            state = "idle"
            self.train_btn.setText(
                "Train"
            )

        self.train_btn.setProperty(
            "state",
            state,
        )
        self._refresh_button(
            self.train_btn
        )

    def _set_test_state(
        self,
        state,
    ):
        if state == "working":
            self.test_btn.setText(
                "Testing..."
            )
        elif state == "done":
            self.test_btn.setText(
                "Tested ✓"
            )
        else:
            state = "idle"
            self.test_btn.setText(
                "Test"
            )

        self.test_btn.setProperty(
            "state",
            state,
        )
        self._refresh_button(
            self.test_btn
        )

    def _architecture_changed(self):
        architecture = str(
            self.architecture_combo.currentData()
            or SEMANTIC_DEFAULT_ARCHITECTURE
        )
        is_dino = _semantic_arch_uses_dino(architecture)
        continue_mode = (
            hasattr(self, "training_mode_combo")
            and self.training_mode_combo.currentData() == "continue"
        )
        self.backbone_combo.setEnabled(is_dino and not continue_mode)

        if architecture == "segformer_b2":
            self.ssl_encoder_edit.setPlaceholderText(
                "SegFormer-B2 uses models/hf_segformer_b2_ade (download tool)"
            )
        elif architecture == "unet":
            self.ssl_encoder_edit.setPlaceholderText(
                "U-Net does not require pretrained weights"
            )
        else:
            self.ssl_encoder_edit.setPlaceholderText(
                "Official DINOv2 or microscopy-SSL encoder .pt/.pth..."
            )

        self._training_mode_changed()

    def _backbone_changed(self):
        architecture = str(
            self.architecture_combo.currentData()
            or SEMANTIC_DEFAULT_ARCHITECTURE
        )
        if not _semantic_arch_uses_dino(architecture):
            return
        backbone = str(
            self.backbone_combo.currentData()
            or SEMANTIC_DEFAULT_BACKBONE
        )
        current = self.ssl_encoder_edit.text().strip()
        current_name = Path(current).name.lower() if current else ""
        # Do not overwrite a user-selected microscopy SSL encoder. Only auto-
        # update empty fields or an official dinov2_*_pretrain file.
        should_auto = (
            not current
            or (
                current_name.startswith("dinov2_")
                and current_name.endswith("_pretrain.pth")
            )
        )
        if should_auto:
            try:
                self.ssl_encoder_edit.setText(
                    str(_find_local_dinov2_weights(backbone))
                )
            except Exception:
                if not current:
                    self.ssl_encoder_edit.clear()

    def _training_mode_changed(self):
        mode = self.training_mode_combo.currentData()
        use_continue = mode == "continue"
        architecture = str(
            self.architecture_combo.currentData()
            or SEMANTIC_DEFAULT_ARCHITECTURE
        )
        use_dino_encoder = (
            mode == "pretrained"
            and _semantic_arch_uses_dino(architecture)
        )

        self.architecture_combo.setEnabled(not use_continue)
        self.backbone_combo.setEnabled(
            (not use_continue) and _semantic_arch_uses_dino(architecture)
        )
        self.ssl_encoder_edit.setEnabled(use_dino_encoder)
        self.ssl_encoder_browse_btn.setEnabled(use_dino_encoder)

        self.model_edit.setEnabled(use_continue)
        self.model_browse_btn.setEnabled(use_continue)
        self.load_model_btn.setEnabled(use_continue)

    def _resolve_training_initial_model(self):
        """Return a complete LinkODT checkpoint only for Continue mode."""
        mode = self.training_mode_combo.currentData()

        if mode != "continue":
            return None

        text = self.model_edit.text().strip()

        candidate = (
            Path(text)
            if text
            else (
                Path(self.model_path)
                if self.model_path is not None
                else None
            )
        )

        if candidate is None:
            raise ValueError(
                "Continue existing LinkODT model is selected.\n\n"
                "Browse a complete LinkODT segmentation .pt/.pth model first."
            )

        if not candidate.exists():
            raise FileNotFoundError(
                f"Existing LinkNex model does not exist:\n{candidate}"
            )

        # A Continue checkpoint must be a full LinkNex semantic checkpoint.
        load_semantic_checkpoint(
            candidate
        )

        self.model_path = candidate
        self.model_edit.setText(
            str(candidate)
        )
        self.model_status.setText(
            f"Loaded complete LinkNex model: {candidate}"
        )

        return candidate

    def _resolve_training_ssl_encoder(self):
        """Resolve DINO encoder-only initialization for the selected model."""
        mode = self.training_mode_combo.currentData()
        if mode != "pretrained":
            return None

        architecture = str(
            self.architecture_combo.currentData()
            or SEMANTIC_DEFAULT_ARCHITECTURE
        )
        if not _semantic_arch_uses_dino(architecture):
            # U-Net needs no pretrained encoder. SegFormer loads its own local
            # Hugging Face pretrained directory through semantic_models.py.
            return None

        backbone = str(
            self.backbone_combo.currentData()
            or SEMANTIC_DEFAULT_BACKBONE
        )
        text = self.ssl_encoder_edit.text().strip()
        if not text:
            candidate = _find_local_dinov2_weights(backbone)
            self.ssl_encoder_edit.setText(str(candidate))
            return candidate

        candidate = Path(text)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Pretrained DINOv2 encoder does not exist:\n{candidate}\n\n"
                "Run tools/download_pretrained_models.py for official weights, "
                "or browse a microscopy SSL encoder."
            )
        if candidate.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Pretrained DINOv2 encoder must be .pt or .pth.")
        return candidate

    def show_weights_help(self):
        QMessageBox.information(
            self,
            "Semantic pretrained weights",
            (
                "Download helper included with LinkODT:\n\n"
                "  DOWNLOAD_MODELS.bat\n"
                "or\n"
                "  python tools/download_pretrained_models.py\n\n"
                "Recommended final backbone:\n"
                "  dinov2_vitb14\n\n"
                "Microscopy SSL encoders can also be selected directly with Browse.\n\n"
                "See models/README_MODELS.md for official download sources and "
                "offline folder locations."
            ),
        )

    def browse_ssl_encoder(self):
        start = str(
            SEMANTIC_MODELS_DIR
            if SEMANTIC_MODELS_DIR.exists()
            else Path.cwd()
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Browse pretrained DINOv2 encoder",
            start,
            (
                "DINOv2 Encoder (*.pt *.pth);;"
                "All files (*.*)"
            ),
        )

        if path:
            self.ssl_encoder_edit.setText(
                path
            )

    def select_dataset(self):
        current = (
            self.dataset_edit
            .text()
            .strip()
            or str(Path.cwd())
        )

        folder = (
            QFileDialog
            .getExistingDirectory(
                self,
                "Select semantic dataset root",
                current,
            )
        )

        if not folder:
            return

        self.dataset_root = Path(
            folder
        )

        self.dataset_edit.setText(
            str(
                self.dataset_root
            )
        )

    def browse_model(self):
        SEMANTIC_MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        start = str(
            SEMANTIC_MODELS_DIR
        )

        path, _ = (
            QFileDialog
            .getOpenFileName(
                self,
                "Browse semantic model",
                start,
                (
                    "LinkODT Semantic Model "
                    "(*.pt *.pth);;"
                    "All files (*.*)"
                ),
            )
        )

        if path:
            self.model_edit.setText(
                path
            )

    def load_model(self):
        text = (
            self.model_edit
            .text()
            .strip()
        )

        if not text:
            QMessageBox.warning(
                self,
                "Load model",
                "Browse a model first.",
            )
            return

        candidate = Path(
            text
        )

        try:
            load_semantic_checkpoint(
                candidate
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load model",
                str(exc),
            )
            return

        self.model_path = (
            candidate
        )

        try:
            torch_meta = __import__("torch")
            meta = torch_meta.load(candidate, map_location="cpu")
            arch = str(meta.get("architecture", "dinov2_unet"))
            if arch in {"dinov2_vits14_unet_decoder", "linknex_semantic_dinov2_unet"}:
                arch = "dinov2_unet"
            i = self.architecture_combo.findData(arch)
            if i >= 0:
                self.architecture_combo.setCurrentIndex(i)
            backbone = str(meta.get("backbone_name", "dinov2_vits14"))
            j = self.backbone_combo.findData(backbone)
            if j >= 0:
                self.backbone_combo.setCurrentIndex(j)
            model_desc = f"{arch} · {backbone}"
        except Exception:
            model_desc = "metadata unavailable"

        self.model_status.setText(
            f"Loaded model: {candidate}\n{model_desc}"
        )

        self.log.setPlainText(
            "MODEL LOADED\n"
            f"{candidate}"
        )

        self.modelReady.emit(
            str(candidate)
        )

    def _resolve_dataset(self):
        text = (
            self.dataset_edit
            .text()
            .strip()
        )

        if not text:
            raise ValueError(
                "Select Dataset first."
            )

        root = Path(
            text
        )

        if not root.exists():
            raise FileNotFoundError(
                f"Dataset does not exist:\n{root}"
            )

        self.dataset_root = root

        return root

    def _settings(self):
        try:
            epochs = int(
                self.epochs_edit
                .text()
                .strip()
            )
            batch_size = int(
                self.batch_edit
                .text()
                .strip()
            )
            pairs_to_use = int(
                self.pairs_to_use_edit
                .text()
                .strip()
            )
            test_percent = int(
                self.test_percent_edit
                .text()
                .strip()
            )
        except Exception:
            raise ValueError(
                "Epochs, Batch, Pairs to use and Test % "
                "must be integers."
            )

        if epochs < 1:
            raise ValueError(
                "Epochs must be >= 1."
            )

        if batch_size < 1:
            raise ValueError(
                "Batch must be >= 1."
            )

        if pairs_to_use < 1:
            raise ValueError(
                "Pairs to use must be >= 1."
            )

        if not (
            1 <= test_percent <= 50
        ):
            raise ValueError(
                "Test % must be between 1 and 50."
            )

        return (
            epochs,
            batch_size,
            pairs_to_use,
            test_percent,
        )


    def _require_model(
        self,
        root,
    ):
        if (
            self.model_path is not None
            and self.model_path.exists()
        ):
            return self.model_path

        candidate = (
            _semantic_latest_trained_model()
        )

        if (
            candidate is not None
            and candidate.exists()
        ):
            self.model_path = (
                candidate
            )
            self.model_edit.setText(
                str(candidate)
            )
            self.model_status.setText(
                f"Loaded model: {candidate}"
            )
            self.modelReady.emit(
                str(candidate)
            )
            return candidate

        raise ValueError(
            "Load a semantic model or Train first.\n\n"
            f"No trained LinkNex semantic model was found in:\n"
            f"{SEMANTIC_MODELS_DIR}"
        )

    def _set_busy(
        self,
        busy,
    ):
        enabled = not busy

        self.training_mode_combo.setEnabled(
            enabled
        )
        self.architecture_combo.setEnabled(enabled)
        self.backbone_combo.setEnabled(enabled)
        self.weights_help_btn.setEnabled(enabled)

        if enabled:
            self._training_mode_changed()
        else:
            self.load_model_btn.setEnabled(False)
            self.model_browse_btn.setEnabled(False)
            self.model_edit.setEnabled(False)
            self.ssl_encoder_edit.setEnabled(False)
            self.ssl_encoder_browse_btn.setEnabled(False)

        self.train_btn.setEnabled(
            enabled
        )
        self.test_btn.setEnabled(
            enabled
        )
        self.stop_btn.setEnabled(
            bool(busy and self.active_action == "train")
        )

    def _progress(
        self,
        text,
        value,
    ):
        self.log.appendPlainText(
            str(text)
        )

        # Always keep the newest progress message visible.
        scrollbar = (
            self.log.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

        self.progress.setValue(
            max(
                0,
                min(
                    100,
                    int(value),
                ),
            )
        )

    def _failed(
        self,
        message,
    ):
        self._set_busy(
            False
        )

        if self.active_action == "train":
            self._set_train_state(
                "idle"
            )

        elif self.active_action == "test":
            self._set_test_state(
                "idle"
            )

        self.active_action = None
        self.stop_btn.setText("Stop")
        self.stop_btn.setEnabled(False)

        self.progress.setValue(
            0
        )

        self.log.appendPlainText(
            "\nERROR:\n"
            f"{message}"
        )

        QMessageBox.critical(
            self,
            "Semantic Model",
            str(message),
        )


    def stop_training(self):
        if (
            self.active_action != "train"
            or self.train_thread is None
            or not self.train_thread.isRunning()
        ):
            return
        self.stop_btn.setText("Stopping...")
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText(
            "\nSTOP REQUESTED...\n"
            "The active batch/operation will finish, then training will stop safely."
        )
        self.train_thread.requestInterruption()


    def start_train(self):
        # Keep the GUI-thread work deliberately tiny.
        # Heavy dataset scanning/dimension checking happens in
        # SemanticTrainWorker.run() after the QThread starts.
        try:
            root = (
                self._resolve_dataset()
            )

            (
                epochs,
                batch_size,
                pairs_to_use,
                test_percent,
            ) = self._settings()

            initial_model_path = (
                self._resolve_training_initial_model()
            )
            initial_encoder_path = (
                self._resolve_training_ssl_encoder()
            )

            mode = self.training_mode_combo.currentData()
            architecture_key = str(
                self.architecture_combo.currentData()
                or SEMANTIC_DEFAULT_ARCHITECTURE
            )
            backbone_name = str(
                self.backbone_combo.currentData()
                or SEMANTIC_DEFAULT_BACKBONE
            )

            if mode == "continue":
                training_mode_text = (
                    "Continue existing LinkODT model"
                )
            else:
                training_mode_text = (
                    "Pretrained encoder · new segmentation head"
                )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Train",
                str(exc),
            )
            return

        self.log.clear()
        self.log.setPlainText(
            "TRAIN STARTING...\n"
            f"Mode: {training_mode_text}\n"
            + (
                f"Continue full LinkNex model: {initial_model_path}\n"
                if initial_model_path is not None
                else (
                    f"Pretrained encoder: {initial_encoder_path}\n"
                    "Segmentation decoder/head: NEW initialization\n"
                )
            )
            + "Dataset checking will run in the background."
        )

        self.progress.setValue(
            0
        )

        # Immediately show visible feedback BEFORE any heavy operation.
        self._set_train_state(
            "working"
        )

        self.active_action = (
            "train"
        )

        self._set_busy(
            True
        )

        self.stop_btn.setText("Stop")

        self.train_thread = (
            QThread(self)
        )

        self.train_worker = (
            SemanticTrainWorker(
                dataset_root=root,
                pairs_to_use=pairs_to_use,
                test_percent=test_percent,
                epochs=epochs,
                batch_size=batch_size,
                initial_model_path=(
                    initial_model_path
                ),
                initial_encoder_path=(
                    initial_encoder_path
                ),
                architecture_key=architecture_key,
                backbone_name=backbone_name,
            )
        )

        self.train_worker.moveToThread(
            self.train_thread
        )

        self.train_thread.started.connect(
            self.train_worker.run
        )

        self.train_worker.progress.connect(
            self._progress
        )

        self.train_worker.finished.connect(
            self._train_finished
        )

        self.train_worker.failed.connect(
            self._failed
        )
        self.train_worker.stopped.connect(
            self._train_stopped
        )

        self.train_worker.finished.connect(
            self.train_thread.quit
        )

        self.train_worker.failed.connect(
            self.train_thread.quit
        )
        self.train_worker.stopped.connect(
            self.train_thread.quit
        )

        self.train_thread.finished.connect(
            self.train_worker.deleteLater
        )

        self.train_thread.finished.connect(
            self.train_thread.deleteLater
        )

        self.train_thread.start()


    def _train_stopped(self, summary):
        self._set_busy(False)
        self._set_train_state("stopped")
        self.active_action = None
        self.stop_btn.setText("Stop")
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText("\n" + str(summary))


    def _train_finished(
        self,
        model_path,
        summary,
    ):
        self._set_busy(
            False
        )
        self.stop_btn.setText("Stop")
        self._set_train_state(
            "done"
        )
        self.active_action = None
        self.progress.setValue(
            100
        )

        self.model_path = Path(
            model_path
        )

        self.model_edit.setText(
            str(
                self.model_path
            )
        )

        self.model_status.setText(
            f"Loaded model: {self.model_path}"
        )

        self.log.appendPlainText(
            "\n" + summary
        )

        self.modelReady.emit(
            str(
                self.model_path
            )
        )

    def start_test(self):
        # Heavy dataset checking/splitting happens in SemanticTestWorker.
        try:
            root = (
                self._resolve_dataset()
            )

            (
                _,
                _,
                pairs_to_use,
                test_percent,
            ) = self._settings()

            model_path = (
                self._require_model(
                    root
                )
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Test",
                str(exc),
            )
            return

        self.log.clear()
        self.log.setPlainText(
            "TEST STARTING...\n"
            "Dataset checking will run in the background."
        )

        self.progress.setValue(
            0
        )

        self._set_test_state(
            "working"
        )

        self.active_action = (
            "test"
        )

        self._set_busy(
            True
        )

        self.test_thread = (
            QThread(self)
        )

        self.test_worker = (
            SemanticTestWorker(
                dataset_root=root,
                pairs_to_use=pairs_to_use,
                test_percent=test_percent,
                model_path=model_path,
            )
        )

        self.test_worker.moveToThread(
            self.test_thread
        )

        self.test_thread.started.connect(
            self.test_worker.run
        )

        self.test_worker.progress.connect(
            self._progress
        )

        self.test_worker.finished.connect(
            self._test_finished
        )

        self.test_worker.failed.connect(
            self._failed
        )

        self.test_worker.finished.connect(
            self.test_thread.quit
        )

        self.test_worker.failed.connect(
            self.test_thread.quit
        )

        self.test_thread.finished.connect(
            self.test_worker.deleteLater
        )

        self.test_thread.finished.connect(
            self.test_thread.deleteLater
        )

        self.test_thread.start()


    def _test_finished(
        self,
        summary,
    ):
        self._set_busy(
            False
        )
        self._set_test_state(
            "done"
        )
        self.active_action = None
        self.progress.setValue(
            100
        )
        self.log.setPlainText(
            summary
        )


class SemanticSegmentationWidget(QWidget):
    backRequested = Signal()

    def __init__(self):
        super().__init__()

        # Embedded inside the LinkNex application.
        # Window sizing is controlled by LinkNexMainWindow.

        # Backup drag/drop path on the whole semantic page.
        self.setAcceptDrops(True)

        self.image_paths = []
        self.current_index = -1

        self.current_image_path = None
        self.binary_mask_path = None
        self.instance_mask_path = None

        # 2D/3D volume state. A 2D image is represented as Z=1.
        self.image_reader = None
        self.binary_reader = None
        self.instance_reader = None

        self.z_count = 1
        self.current_z = 0
        self.volume_height = 0
        self.volume_width = 0

        # Only visited/editable label slices are cached in RAM.
        self.slice_label_cache = {}

        # A small RGB cache keeps Z scrolling responsive without retaining
        # the entire normalized 3D image in memory.
        self.rgb_cache = OrderedDict()
        self.rgb_cache_limit = 9

        # Live Z-slider state. The slider updates the image while dragging,
        # but refreshes are coalesced so fast mouse motion does not create
        # a long queue of obsolete Z-slice loads.
        self.pending_z = None
        self.z_refresh_interval_ms = 30

        self.image_rgb = None
        self.labels = None
        self.instance_sizes = {}

        # Display-only image adjustment; does not modify source TIFF/masks.
        self.display_brightness = 0      # -100 .. +100
        self.display_contrast = 100      # 50 .. 200 percent

        # Brightness / contrast live-preview state.
        # Rendering is throttled so fast slider movement never creates
        # a backlog of obsolete full-image redraws.
        self.image_adjust_pending = False
        self.image_adjust_interval_ms = 24

        # Save-button visual feedback state.
        self.save_feedback_state = "idle"
        self.delete_feedback_state = "idle"
        self.instance_slices = {}

        self.instance_lut = None
        self.base_overlay_rgba = None

        self.selected_ids = set()
        self.dirty = False
        self.current_source = "none"

        # Semantic model state.
        self.semantic_model_path = None
        self.semantic_training_window = None
        self.semantic_predict_thread = None
        self.semantic_predict_worker = None

        # Overlay content mode:
        #   instance = multicolor instance overlay
        #   mask     = single-color binary mask overlay
        # Visibility is controlled independently by hide_overlay.
        self.display_mode = "instance"
        self.hide_overlay = False

        self._build_ui()
        self.brush_mode_btn.setToolTip("Left-drag to add foreground. Existing labels are protected.")
        self.polygon_mode_btn.setToolTip("Left-drag a continuous outline; release to close and fill it.")
        self.rect_mode_btn.setToolTip("Select whole labels touched by the rectangle. Ctrl adds to selection.")
        self.delete_btn.setToolTip("Delete entire selected label regions in the current Z slice.")
        self.save_btn.setToolTip("Save and overwrite the binary mask and instance labels for the full stack (Ctrl+S).")
        self.semantic_predict_all_btn.setToolTip("Predict all loaded images. Existing prediction masks will be overwritten.")

        self._build_shortcuts()
        self._apply_style()
        self._update_controls()

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")

        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 6, 16, 6)
        hl.setSpacing(12)

        # Internal GUI header:
        # use the FULL user-provided LinkNex logo only.
        # Do not add separate "LinkNex" or "2D Semantic Mask Assistant" text.
        self.logo_label = QLabel()
        self.logo_label.setObjectName("LogoImage")
        self.logo_label.setFixedSize(250, 72)
        self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        full_logo_pix = QPixmap()
        for _candidate in LOGO_FULL_CANDIDATES:
            if _candidate.exists():
                full_logo_pix = QPixmap(str(_candidate))
                if not full_logo_pix.isNull():
                    break

        if not full_logo_pix.isNull():
            self.logo_label.setPixmap(
                full_logo_pix.scaled(
                    235,
                    66,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            # Minimal fallback only if the image file is missing.
            self.logo_label.setText("LinkODT")

        hl.addWidget(self.logo_label)
        hl.addStretch(1)

        # Dedicated top header removed to avoid wasting vertical space.
        # The full LinkNex logo is shown in the left column instead.

        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(12, 8, 12, 12)
        body_l.setSpacing(12)

        # left
        left = QFrame()
        left.setObjectName("Panel")
        left.setFixedWidth(335)

        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 10, 14, 14)
        ll.setSpacing(10)

        # Put the full LinkNex logo at the INSIDE top-left corner
        # of the semantic page. The round icon in the title bar remains the
        # window-level top-left mark.
        self.sidebar_logo_label = QLabel()
        self.sidebar_logo_label.setObjectName("LogoImage")
        self.sidebar_logo_label.setFixedSize(230, 62)
        self.sidebar_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        sidebar_logo_pix = QPixmap()
        for _candidate in LOGO_FULL_CANDIDATES:
            if _candidate.exists():
                sidebar_logo_pix = QPixmap(str(_candidate))
                if not sidebar_logo_pix.isNull():
                    break

        if not sidebar_logo_pix.isNull():
            self.sidebar_logo_label.setPixmap(
                sidebar_logo_pix.scaled(
                    220,
                    55,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.sidebar_logo_label.setText("LinkODT")

        ll.addWidget(self.sidebar_logo_label)

        self.back_btn = QPushButton("← Home")
        self.back_btn.setObjectName("SemanticBack")
        self.back_btn.clicked.connect(
            self.backRequested.emit
        )
        ll.addWidget(self.back_btn)

        ll.addSpacing(4)

        sec = QLabel("Images")
        sec.setObjectName("Section")
        ll.addWidget(sec)

        self.open_btn = QPushButton("Open folder / files")
        self.open_btn.setObjectName("Primary")
        self.open_btn.clicked.connect(self.open_files_or_folder)
        ll.addWidget(self.open_btn)

        drop_hint = QLabel(
            "Drag image files or a folder directly onto the image area."
        )
        drop_hint.setWordWrap(True)
        drop_hint.setObjectName("Muted")
        ll.addWidget(drop_hint)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.file_list.currentRowChanged.connect(
            self._on_row_changed
        )
        self.file_list.itemSelectionChanged.connect(
            self._update_controls
        )
        ll.addWidget(self.file_list, 1)

        ll.addSpacing(4)

        model_sec = QLabel("Semantic model")
        model_sec.setObjectName("Section")
        ll.addWidget(model_sec)

        self.semantic_train_test_btn = QPushButton(
            "Train / test model"
        )
        self.semantic_train_test_btn.setObjectName(
            "SemanticModelTool"
        )
        self.semantic_train_test_btn.clicked.connect(
            self.open_semantic_training_window
        )
        ll.addWidget(
            self.semantic_train_test_btn
        )

        self.semantic_model_load_btn = QPushButton(
            "Load model"
        )
        self.semantic_model_load_btn.setObjectName(
            "SemanticModelTool"
        )
        self.semantic_model_load_btn.clicked.connect(
            self.load_semantic_model
        )
        ll.addWidget(
            self.semantic_model_load_btn
        )

        self.semantic_model_status = QLabel(
            "Model: —"
        )
        self.semantic_model_status.setObjectName(
            "Metric"
        )
        self.semantic_model_status.setWordWrap(
            True
        )
        ll.addWidget(
            self.semantic_model_status
        )

        threshold_row = QHBoxLayout()
        threshold_row.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        threshold_row.setSpacing(
            8
        )

        threshold_label = QLabel(
            "Predict threshold"
        )
        threshold_label.setObjectName(
            "Metric"
        )

        self.semantic_threshold_spin = QSpinBox()
        self.semantic_threshold_spin.setRange(
            1,
            99,
        )
        self.semantic_threshold_spin.setSingleStep(
            5
        )
        self.semantic_threshold_spin.setValue(
            50
        )
        self.semantic_threshold_spin.setSuffix(
            "%"
        )
        self.semantic_threshold_spin.setToolTip(
            "Higher threshold = stricter foreground. "
            "Try 60-65% when nearby objects are incorrectly connected. "
            "Too high can remove weak/thin structures."
        )

        threshold_row.addWidget(
            threshold_label,
            1,
        )
        threshold_row.addWidget(
            self.semantic_threshold_spin
        )

        ll.addLayout(
            threshold_row
        )

        self.semantic_predict_btn = QPushButton(
            "Predict"
        )
        self.semantic_predict_btn.setToolTip(
            "Predict selected image(s). Existing *_mask.tif and "
            "*_mask_instance.tif are overwritten."
        )
        self.semantic_predict_btn.setObjectName(
            "SemanticPredict"
        )
        self.semantic_predict_btn.setProperty(
            "state",
            "idle",
        )
        self.semantic_predict_btn.clicked.connect(
            self.predict_semantic_current
        )
        self.semantic_predict_all_btn = QPushButton("Predict all")
        self.semantic_predict_all_btn.setToolTip(
            "Predict all loaded images and overwrite existing mask outputs."
        )
        self.semantic_predict_all_btn.setObjectName("SemanticPredictAll")
        self.semantic_predict_all_btn.setProperty("state", "idle")
        self.semantic_predict_all_btn.clicked.connect(self.predict_semantic_all)

        predict_row = QHBoxLayout()
        predict_row.setContentsMargins(0, 0, 0, 0)
        predict_row.setSpacing(6)
        predict_row.addWidget(self.semantic_predict_btn, 1)
        predict_row.addWidget(self.semantic_predict_all_btn)
        ll.addLayout(predict_row)


        body_l.addWidget(left)

        # center
        center = QFrame()
        center.setObjectName("CanvasPanel")

        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self.canvas = MaskCanvas()
        self.canvas.filesDropped.connect(self.add_input_paths)
        self.canvas.instanceClicked.connect(self.select_at)
        self.canvas.rectangleFinished.connect(self.select_by_rectangle)
        self.canvas.brushFinished.connect(self.add_brush_stroke)
        self.canvas.polygonFinished.connect(self.add_polygon_region)
        cl.addWidget(self.canvas, 1)

        # Z navigation lives INSIDE the interaction frame.
        # Z=1 is the normal 2D case.
        zbar = QFrame()
        zbar.setObjectName("BottomBar")
        zl = QHBoxLayout(zbar)
        zl.setContentsMargins(10, 6, 10, 2)
        zl.setSpacing(8)

        z_title = QLabel("Z")
        z_title.setObjectName("SmallLabel")
        z_title.setFixedWidth(14)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(1, 1)
        self.z_slider.setValue(1)

        # Live preview while dragging.
        # setTracking(True) lets the slider value follow the handle immediately.
        # Actual image refresh is throttled below with a short QTimer so the
        # GUI always renders the newest requested Z instead of processing every
        # intermediate slider pixel.
        self.z_slider.setTracking(True)
        self.z_slider.sliderMoved.connect(self._queue_live_z)
        self.z_slider.valueChanged.connect(self._z_value_changed)
        self.z_slider.sliderReleased.connect(self._flush_live_z)

        self.z_refresh_timer = QTimer(self)
        self.z_refresh_timer.setSingleShot(True)
        self.z_refresh_timer.timeout.connect(self._render_pending_z)

        self.z_value_label = QLabel("1 / 1")
        self.z_value_label.setObjectName("Metric")
        self.z_value_label.setFixedWidth(72)
        self.z_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        zl.addWidget(z_title)
        zl.addWidget(self.z_slider, 1)
        zl.addWidget(self.z_value_label)

        cl.addWidget(zbar)

        nav = QFrame()
        nav.setObjectName("BottomBar")

        nl = QHBoxLayout(nav)
        nl.setContentsMargins(10, 8, 10, 8)
        nl.setSpacing(8)

        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_image)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.canvas.fit_image)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_image)

        nl.addWidget(self.prev_btn)
        nl.addWidget(self.fit_btn)
        nl.addWidget(self.next_btn)
        nl.addStretch(1)

        self.image_info = QLabel("Drag an image here to begin")
        self.image_info.setObjectName("Muted")
        nl.addWidget(self.image_info)

        cl.addWidget(nav)
        body_l.addWidget(center, 1)

        # right
        right = QFrame()
        right.setObjectName("Panel")
        right.setFixedWidth(335)

        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(8)

        sec2 = QLabel("Instance mask")
        sec2.setObjectName("Section")
        rl.addWidget(sec2)

        display_label = QLabel("Display")
        display_label.setObjectName("SmallLabel")
        rl.addWidget(display_label)

        display_row = QHBoxLayout()
        display_row.setSpacing(6)

        self.display_instance_btn = QPushButton("Instance")
        self.display_instance_btn.setCheckable(True)
        self.display_instance_btn.setChecked(True)

        self.display_mask_btn = QPushButton("Mask")
        self.display_mask_btn.setCheckable(True)

        self.display_group = QButtonGroup(self)
        self.display_group.setExclusive(True)
        self.display_group.addButton(self.display_instance_btn)
        self.display_group.addButton(self.display_mask_btn)

        self.display_instance_btn.clicked.connect(
            lambda: self.set_display_mode("instance")
        )
        self.display_mask_btn.clicked.connect(
            lambda: self.set_display_mode("mask")
        )

        display_row.addWidget(self.display_instance_btn)
        display_row.addWidget(self.display_mask_btn)
        rl.addLayout(display_row)

        # Hide is an independent ON/OFF switch, not a third display mode.
        hide_row = QHBoxLayout()
        hide_row.setSpacing(10)

        hide_label = QLabel("Hide")
        hide_label.setObjectName("SmallLabel")

        self.hide_overlay_switch = ToggleSwitch()
        self.hide_overlay_switch.setChecked(False)
        self.hide_overlay_switch.toggled.connect(self.set_hide_overlay)

        hide_row.addWidget(hide_label)
        hide_row.addStretch(1)
        hide_row.addWidget(self.hide_overlay_switch)
        rl.addLayout(hide_row)

        image_display_label = QLabel("Image appearance")
        image_display_label.setObjectName("SmallLabel")
        rl.addWidget(image_display_label)

        brightness_row = QHBoxLayout()
        brightness_row.setSpacing(8)

        brightness_label = QLabel("Brightness")
        brightness_label.setObjectName("SmallLabel")
        brightness_label.setFixedWidth(64)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setObjectName("ImageAdjustSlider")
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTracking(True)
        self.brightness_slider.valueChanged.connect(
            self._brightness_changed
        )

        self.brightness_value = QLabel("0")
        self.brightness_value.setObjectName("Metric")
        self.brightness_value.setFixedWidth(34)
        self.brightness_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        brightness_row.addWidget(brightness_label)
        brightness_row.addWidget(self.brightness_slider, 1)
        brightness_row.addWidget(self.brightness_value)
        rl.addLayout(brightness_row)

        contrast_row = QHBoxLayout()
        contrast_row.setSpacing(8)

        contrast_label = QLabel("Contrast")
        contrast_label.setObjectName("SmallLabel")
        contrast_label.setFixedWidth(64)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setObjectName("ImageAdjustSlider")
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTracking(True)
        self.contrast_slider.valueChanged.connect(
            self._contrast_changed
        )

        # One timer is shared by both brightness and contrast.
        # ~24 ms corresponds to a maximum of about 40 display updates/sec.
        self.image_adjust_timer = QTimer(self)
        self.image_adjust_timer.setSingleShot(True)
        self.image_adjust_timer.timeout.connect(
            self._render_pending_image_adjustment
        )

        self.brightness_slider.sliderReleased.connect(
            self._flush_image_adjustment
        )
        self.contrast_slider.sliderReleased.connect(
            self._flush_image_adjustment
        )

        self.contrast_value = QLabel("100%")
        self.contrast_value.setObjectName("Metric")
        self.contrast_value.setFixedWidth(44)
        self.contrast_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        contrast_row.addWidget(contrast_label)
        contrast_row.addWidget(self.contrast_slider, 1)
        contrast_row.addWidget(self.contrast_value)
        rl.addLayout(contrast_row)

        self.reset_display_btn = QPushButton("Reset brightness / contrast")
        self.reset_display_btn.setObjectName("DisplayReset")
        self.reset_display_btn.clicked.connect(self._reset_image_adjustments)
        rl.addWidget(self.reset_display_btn)

        self.source_status = QLabel("Source: —")
        self.source_status.setObjectName("Metric")
        self.source_status.setWordWrap(True)
        rl.addWidget(self.source_status)

        self.instance_status = QLabel("Instances: —")
        self.instance_status.setObjectName("Metric")
        rl.addWidget(self.instance_status)

        self.selected_status = QLabel("Selected: 0")
        self.selected_status.setObjectName("Selected")
        rl.addWidget(self.selected_status)

        mode_label = QLabel("Interaction mode")
        mode_label.setObjectName("SmallLabel")
        rl.addWidget(mode_label)

        mode_row_top = QHBoxLayout()
        mode_row_top.setSpacing(6)

        self.select_mode_btn = QPushButton("Select")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.setChecked(True)

        self.rect_mode_btn = QPushButton("Rectangle")
        self.rect_mode_btn.setCheckable(True)

        mode_row_bottom = QHBoxLayout()
        mode_row_bottom.setSpacing(6)

        self.brush_mode_btn = QPushButton("Brush")
        self.brush_mode_btn.setCheckable(True)

        self.polygon_mode_btn = QPushButton("Polygon")
        self.polygon_mode_btn.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.select_mode_btn)
        self.mode_group.addButton(self.rect_mode_btn)
        self.mode_group.addButton(self.brush_mode_btn)
        self.mode_group.addButton(self.polygon_mode_btn)

        self.select_mode_btn.clicked.connect(
            lambda: self.set_interaction_mode(MaskCanvas.MODE_SELECT)
        )
        self.rect_mode_btn.clicked.connect(
            lambda: self.set_interaction_mode(MaskCanvas.MODE_RECT)
        )
        self.brush_mode_btn.clicked.connect(
            lambda: self.set_interaction_mode(MaskCanvas.MODE_BRUSH)
        )
        self.polygon_mode_btn.clicked.connect(
            lambda: self.set_interaction_mode(MaskCanvas.MODE_POLYGON)
        )

        mode_row_top.addWidget(self.select_mode_btn)
        mode_row_top.addWidget(self.rect_mode_btn)

        mode_row_bottom.addWidget(self.brush_mode_btn)
        mode_row_bottom.addWidget(self.polygon_mode_btn)

        rl.addLayout(mode_row_top)
        rl.addLayout(mode_row_bottom)

        brush_label = QLabel("Brush / Polygon diameter (pixels)")
        brush_label.setObjectName("SmallLabel")
        rl.addWidget(brush_label)

        brush_row = QHBoxLayout()
        brush_row.setSpacing(8)

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(
            MIN_BRUSH_DIAMETER,
            MAX_BRUSH_DIAMETER,
        )
        self.brush_slider.setValue(DEFAULT_BRUSH_DIAMETER)

        self.brush_spin = BrushSpinBox()
        self.brush_spin.setRange(
            MIN_BRUSH_DIAMETER,
            MAX_BRUSH_DIAMETER,
        )
        self.brush_spin.setValue(DEFAULT_BRUSH_DIAMETER)
        self.brush_spin.setFixedWidth(108)
        self.brush_spin.setFixedHeight(40)

        self.brush_slider.valueChanged.connect(self._brush_slider_changed)
        self.brush_spin.valueChanged.connect(self._brush_spin_changed)

        brush_row.addWidget(self.brush_slider, 1)
        brush_row.addWidget(self.brush_spin)
        rl.addLayout(brush_row)

        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.setObjectName("Delete")
        self.delete_btn.setProperty("state", "idle")
        self.delete_btn.clicked.connect(self.delete_selected)
        rl.addWidget(self.delete_btn)

        self.delete_feedback_timer = QTimer(self)
        self.delete_feedback_timer.setSingleShot(True)
        self.delete_feedback_timer.timeout.connect(
            self._restore_delete_button_idle
        )

        self.clear_selection_btn = QPushButton("Clear selection")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        rl.addWidget(self.clear_selection_btn)

        self.save_btn = QPushButton("Save masks")
        self.save_btn.setObjectName("Save")
        self.save_btn.setProperty("state", "idle")
        self.save_btn.clicked.connect(self.save_masks)
        rl.addWidget(self.save_btn)

        self.save_feedback_timer = QTimer(self)
        self.save_feedback_timer.setSingleShot(True)
        self.save_feedback_timer.timeout.connect(
            self._restore_save_button_idle
        )

        self.remove_current_btn = QPushButton("Remove current from list")
        self.remove_current_btn.setObjectName("RemoveCurrent")
        self.remove_current_btn.clicked.connect(self.remove_current_from_list)
        rl.addWidget(self.remove_current_btn)

        rl.addStretch(1)

        body_l.addWidget(right)

        root.addWidget(body, 1)

        # Status bar removed to avoid occupying an extra line.

    def _build_shortcuts(self):
        save_action = QAction(self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_masks)
        self.addAction(save_action)

        delete_action = QAction(self)
        delete_action.setShortcut(QKeySequence(Qt.Key_Delete))
        delete_action.triggered.connect(self.delete_selected)
        self.addAction(delete_action)

        left_action = QAction(self)
        left_action.setShortcut(QKeySequence(Qt.Key_Left))
        left_action.triggered.connect(self.previous_image)
        self.addAction(left_action)

        right_action = QAction(self)
        right_action.setShortcut(QKeySequence(Qt.Key_Right))
        right_action.triggered.connect(self.next_image)
        self.addAction(right_action)

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {OFFWHITE};
                color: {TEXT};
            }}

            QFrame#Header {{
                background: #F8FAFD;
                border: none;
            }}

            QLabel#LogoImage {{
                background: transparent;
                border: none;
                min-width: 250px;
                max-width: 250px;
                min-height: 72px;
                max-height: 72px;
            }}

            QFrame#Panel {{
                background: transparent;
                border: none;
                border-radius: 0px;
            }}

            QFrame#CanvasPanel {{
                background: #F5F7FB;
                border: 2px solid {BORDER};
                border-radius: 8px;
            }}

            QFrame#BottomBar {{
                background: transparent;
                border-top: none;
            }}

            QLabel#Section {{
                color: {DEEP_BLUE};
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0px;
            }}

            QLabel#SmallLabel {{
                color: {MUTED};
                font-size: 13px;
                font-weight: 700;
            }}

            QLabel#Muted {{
                color: {MUTED};
                font-size: 14px;
            }}

            QLabel#Metric {{
                color: {TEXT};
                font-size: 14px;
                font-weight: 600;
            }}

            QLabel#Selected {{
                color: {TEXT};
                background: #E6E8EB;
                border: 1px solid #B8BDC5;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                font-weight: 600;
            }}

            QListWidget {{
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {TEXT};
                outline: none;
            }}

            QListWidget::item {{
                padding: 7px 6px;
                border-radius: 4px;
                color: {TEXT};
                background: {WHITE};
            }}

            QListWidget::item:hover {{
                background: #EAF5FC;
                color: {NAVY};
            }}

            QListWidget::item:selected {{
                background: {US_BLUE};
                color: {WHITE};
            }}

            QListWidget::item:selected:!active {{
                background: {US_BLUE};
                color: {WHITE};
            }}

            QComboBox {{
                min-height: 32px;
                padding: 2px 7px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {TEXT};
            }}

            QSpinBox {{
                min-height: 36px;
                padding: 2px 36px 2px 10px;
                border: 2px solid {BORDER};
                border-radius: 7px;
                background: {WHITE};
                color: {TEXT};
                font-size: 14px;
                font-weight: 700;
            }}

            QToolButton#BrushSpinUp,
            QToolButton#BrushSpinDown {{
                background: {US_BLUE};
                color: {WHITE};
                border: none;
                font-size: 13px;
                font-weight: 600;
                padding: 0px;
                margin: 0px;
            }}

            QToolButton#BrushSpinUp {{
                border-left: 2px solid {DEEP_BLUE};
                border-bottom: 1px solid {DEEP_BLUE};
                border-top-right-radius: 5px;
            }}

            QToolButton#BrushSpinDown {{
                border-left: 2px solid {DEEP_BLUE};
                border-top: 1px solid {DEEP_BLUE};
                border-bottom-right-radius: 5px;
            }}

            QToolButton#BrushSpinUp:hover,
            QToolButton#BrushSpinDown:hover {{
                background: {DEEP_BLUE};
            }}

            QToolButton#BrushSpinUp:pressed,
            QToolButton#BrushSpinDown:pressed {{
                background: {NAVY};
            }}

            QPushButton {{
                min-height: 34px;
                padding: 3px 9px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {NAVY};
                font-weight: 700;
            }}

            QPushButton:hover {{
                border-color: {US_BLUE};
                background: #EEF7FD;
            }}

            QPushButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
            }}

            QPushButton:disabled {{
                color: #A9B4C0;
                background: #F1F3F6;
                border-color: #C9D4E2;
            }}

            QSlider#ImageAdjustSlider {{
                min-height: 22px;
            }}

            QSlider#ImageAdjustSlider::groove:horizontal {{
                height: 5px;
                background: #D5DEE8;
                border-radius: 2px;
            }}

            QSlider#ImageAdjustSlider::sub-page:horizontal {{
                background: {US_BLUE};
                border-radius: 2px;
            }}

            QSlider#ImageAdjustSlider::handle:horizontal {{
                background: {WHITE};
                border: 2px solid {US_BLUE};
                width: 15px;
                height: 15px;
                margin: -6px 0px;
                border-radius: 8px;
            }}

            QSlider#ImageAdjustSlider::handle:horizontal:hover {{
                background: #EAF6FC;
                border-color: {DEEP_BLUE};
            }}

            QPushButton#DisplayReset {{
                min-height: 28px;
                padding: 2px 8px;
                background: transparent;
                color: {MUTED};
                border: 1px solid {BORDER};
                font-size: 13px;
                font-weight: 700;
            }}

            QPushButton#DisplayReset:hover {{
                color: {DEEP_BLUE};
                border-color: {US_BLUE};
                background: #EEF7FD;
            }}

            QPushButton#Primary {{
                background: {US_BLUE};
                color: {WHITE};
                border-color: {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#SemanticModelTool {{
                background: {WHITE};
                color: {DEEP_BLUE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#SemanticPredict,
            QPushButton#SemanticPredictAll {{
                min-height: 40px;
                background: {NAVY};
                color: {WHITE};
                border: 2.5px solid {NAVY};
                font-weight: 600;
            }}
            QPushButton#SemanticPredictAll {{
                min-width: 54px;
                max-width: 70px;
            }}

            QPushButton#SemanticPredict:pressed,
            QPushButton#SemanticPredictAll:pressed,
            QPushButton#SemanticPredict[state="working"],
            QPushButton#SemanticPredictAll[state="working"] {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#SemanticPredict[state="done"],
            QPushButton#SemanticPredictAll[state="done"] {{
                background: #1E9E5A;
                color: {WHITE};
                border-color: #167B45;
            }}

            QPushButton#SemanticPredict[state="error"],
            QPushButton#SemanticPredictAll[state="error"] {{
                background: #D94B4B;
                color: {WHITE};
                border-color: #AE3232;
            }}

            QPushButton#SemanticBack {{
                min-height: 32px;
                background: {WHITE};
                color: {DEEP_BLUE};
                border: 2px solid {BORDER};
                font-weight: 600;
            }}

            QPushButton#SemanticBack:hover {{
                background: #EEF7FD;
                border-color: {US_BLUE};
            }}

            QPushButton#SemanticBack:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#Delete {{
                min-height: 42px;
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #D7B000;
                font-weight: 600;
            }}

            QPushButton#Delete:hover {{
                background: #FFE15A;
                border-color: #C79F00;
            }}

            QPushButton#Delete:pressed,
            QPushButton#Delete[state="deleting"] {{
                background: #F3A712;
                color: {NAVY};
                border: 2.5px solid #B87500;
            }}

            QPushButton#Delete[state="deleted"] {{
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
            }}

            QPushButton#Save {{
                min-height: 42px;
                background: {NAVY};
                color: {WHITE};
                border: 2.5px solid {NAVY};
                font-weight: 600;
            }}

            QPushButton#Save:hover {{
                background: {DEEP_BLUE};
                border-color: {US_BLUE};
            }}

            QPushButton#Save:pressed,
            QPushButton#Save[state="saving"] {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
            }}

            QPushButton#Save[state="saved"] {{
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
            }}

            QPushButton#Save[state="error"] {{
                background: #D94B4B;
                color: {WHITE};
                border: 2.5px solid #AE3232;
            }}

            QPushButton#RemoveCurrent {{
                min-height: 38px;
                background: {WHITE};
                color: {DEEP_BLUE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#RemoveCurrent:hover {{
                background: #EEF7FD;
                border-color: {DEEP_BLUE};
            }}

            QPushButton:checked {{
                background: {CYAN};
                color: {WHITE};
                border-color: {US_BLUE};
            }}

            QStatusBar {{
                background: {WHITE};
                color: {MUTED};
                border-top: 2px solid {BORDER};
            }}

        """ + DISABLED_STYLE)

    def _selected_semantic_paths(self):
        rows = sorted({
            self.file_list.row(item)
            for item in self.file_list.selectedItems()
        })

        if (
            not rows
            and 0 <= self.current_index < len(self.image_paths)
        ):
            rows = [
                self.current_index
            ]

        return [
            self.image_paths[row]
            for row in rows
            if 0 <= row < len(self.image_paths)
        ]

    # -------------------------------------------------------------------------
    # Semantic model Train / Test / Predict
    # -------------------------------------------------------------------------

    def _set_semantic_predict_state(
        self,
        state,
    ):
        if not hasattr(self, "semantic_predict_btn"):
            return
        state = str(state)
        mode = getattr(self, "semantic_predict_mode", "selected")
        selected_state = state if mode == "selected" else "idle"
        all_state = state if mode == "all" else "idle"

        self.semantic_predict_btn.setText(
            "Predicting..." if selected_state == "working" else
            "Predicted ✓" if selected_state == "done" else
            "Predict failed" if selected_state == "error" else "Predict"
        )
        self.semantic_predict_btn.setProperty("state", selected_state)

        if hasattr(self, "semantic_predict_all_btn"):
            self.semantic_predict_all_btn.setText(
                "All..." if all_state == "working" else
                "All ✓" if all_state == "done" else
                "All failed" if all_state == "error" else "Predict all"
            )
            self.semantic_predict_all_btn.setProperty("state", all_state)

        for button in (self.semantic_predict_btn, getattr(self, "semantic_predict_all_btn", None)):
            if button is None:
                continue
            style = button.style()
            style.unpolish(button)
            style.polish(button)
            button.update()

    def open_semantic_training_window(self):
        if (
            self.semantic_training_window
            is None
        ):
            self.semantic_training_window = (
                SemanticTrainingWindow(
                    self
                )
            )

            self.semantic_training_window.modelReady.connect(
                self._semantic_model_ready
            )

        self.semantic_training_window.show()
        self.semantic_training_window.raise_()
        self.semantic_training_window.activateWindow()

    def _semantic_model_ready(
        self,
        path,
    ):
        candidate = Path(
            path
        )

        self.semantic_model_path = (
            candidate
        )

        self.semantic_model_status.setText(
            f"Model: {candidate.name}"
        )
        self.semantic_model_status.setToolTip(
            str(candidate)
        )

        self._update_controls()

    def load_semantic_model(self):
        start = (
            str(
                self.semantic_model_path.parent
            )
            if self.semantic_model_path
            is not None
            else str(Path.cwd())
        )

        path, _ = (
            QFileDialog
            .getOpenFileName(
                self,
                "Load semantic model",
                start,
                (
                    "LinkODT Semantic Model "
                    "(*.pt *.pth);;"
                    "All files (*.*)"
                ),
            )
        )

        if not path:
            return

        candidate = Path(
            path
        )

        try:
            load_semantic_checkpoint(
                candidate
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load model",
                str(exc),
            )
            return

        self.semantic_model_path = (
            candidate
        )

        self.semantic_model_status.setText(
            f"Model: {candidate.name}"
        )
        self.semantic_model_status.setToolTip(
            str(candidate)
        )

        if (
            self.semantic_training_window
            is not None
        ):
            self.semantic_training_window.model_path = (
                candidate
            )
            self.semantic_training_window.model_edit.setText(
                str(candidate)
            )
            self.semantic_training_window.model_status.setText(
                f"Loaded model: {candidate}"
            )

        self._update_controls()

    def predict_semantic_current(self):
        paths = self._selected_semantic_paths()
        if not paths:
            QMessageBox.warning(self, "Predict", "Open and select one or more images first.")
            return
        self.semantic_predict_mode = "selected"
        self._start_semantic_prediction(paths)

    def predict_semantic_all(self):
        paths = list(self.image_paths)
        if not paths:
            QMessageBox.warning(self, "Predict All", "Open images in the left list first.")
            return
        self.semantic_predict_mode = "all"
        self._start_semantic_prediction(paths)

    def _start_semantic_prediction(self, paths):
        paths = [Path(p) for p in paths]
        if self.semantic_model_path is None or not self.semantic_model_path.exists():
            QMessageBox.warning(self, "Predict", "Load a semantic model first.")
            return

        current_path = self.current_image_path if self.current_image_path is not None else None
        current_will_be_overwritten = bool(current_path is not None and current_path in paths)

        if self.dirty and current_will_be_overwritten:
            answer = QMessageBox.question(
                self,
                "Replace current mask?",
                "The current image has unsaved mask edits.\n\n"
                "Predict will overwrite its saved *_mask.tif and *_mask_instance.tif and replace the current in-memory mask. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return


        # Explicit overwrite support: on Windows an opened TIFF/memmap can block
        # os.replace().  Release only the current mask readers before the worker
        # writes the new *_mask.tif and *_mask_instance.tif.  The source image
        # reader stays open; the current item is reloaded after prediction.
        self._semantic_prediction_reload_row = None
        if current_will_be_overwritten:
            self._semantic_prediction_reload_row = int(self.current_index)
            self._close_mask_readers_for_overwrite()

        self._set_semantic_predict_state("working")
        self.semantic_predict_btn.setEnabled(False)
        if hasattr(self, "semantic_predict_all_btn"):
            self.semantic_predict_all_btn.setEnabled(False)

        predict_threshold = (
            float(
                self.semantic_threshold_spin.value()
            )
            / 100.0
        )

        self.semantic_threshold_spin.setEnabled(
            False
        )

        self.semantic_predict_thread = QThread(self)
        self.semantic_predict_worker = SemanticBatchPredictWorker(
            image_paths=paths,
            model_path=self.semantic_model_path,
            predict_threshold=predict_threshold,
        )
        self.semantic_predict_worker.moveToThread(self.semantic_predict_thread)
        self.semantic_predict_thread.started.connect(self.semantic_predict_worker.run)
        self.semantic_predict_worker.progress.connect(self._semantic_predict_progress)
        self.semantic_predict_worker.finished.connect(self._semantic_batch_predict_finished)
        self.semantic_predict_worker.failed.connect(self._semantic_predict_failed)
        self.semantic_predict_worker.finished.connect(self.semantic_predict_thread.quit)
        self.semantic_predict_worker.failed.connect(self.semantic_predict_thread.quit)
        self.semantic_predict_thread.finished.connect(self.semantic_predict_worker.deleteLater)
        self.semantic_predict_thread.finished.connect(self.semantic_predict_thread.deleteLater)
        self.semantic_predict_thread.start()

    def _semantic_predict_progress(
        self,
        text,
        value,
    ):
        active_button = (
            self.semantic_predict_all_btn
            if getattr(self, "semantic_predict_mode", "selected") == "all"
            else self.semantic_predict_btn
        )
        active_button.setToolTip(
            f"{text}  ({int(value)}%)"
        )

    def _semantic_batch_predict_finished(
        self,
        result,
    ):
        result = dict(
            result
        )

        saved_paths = [
            Path(p)
            for p in result.get(
                "saved_masks",
                []
            )
        ]

        saved_instance_paths = [
            Path(p)
            for p in result.get(
                "saved_instance_masks",
                []
            )
        ]

        predicted_images = {
            Path(p)
            for p in result.get(
                "image_paths",
                []
            )
        }

        self._set_semantic_predict_state(
            "done"
        )

        active_button = (
            self.semantic_predict_all_btn
            if getattr(
                self,
                "semantic_predict_mode",
                "selected",
            ) == "all"
            else self.semantic_predict_btn
        )

        active_button.setToolTip(
            (
                f"Predicted {len(saved_paths)} image(s). "
                "Saved *_mask.tif and *_mask_instance.tif beside each source image."
            )
        )

        reload_error = None

        try:
            if (
                self.current_image_path
                is not None
                and Path(
                    self.current_image_path
                ) in predicted_images
            ):
                current_row = (
                    self.current_index
                )

                if (
                    0 <= current_row < len(self.image_paths)
                ):
                    self.load_image(
                        current_row
                    )

        except Exception as exc:
            reload_error = str(
                exc
            )

        self._semantic_prediction_reload_row = None

        if hasattr(
            self,
            "semantic_threshold_spin",
        ):
            self.semantic_threshold_spin.setEnabled(
                True
            )

        self._update_controls()

        # Completion feedback is temporary. Show the green check briefly,
        # then restore the normal Predict / All button text automatically.
        QTimer.singleShot(
            1500,
            self._restore_semantic_predict_idle,
        )

        if reload_error:
            QMessageBox.warning(
                self,
                "Prediction saved",
                (
                    "Prediction completed successfully and the files were saved, "
                    "but the GUI could not automatically refresh the overlay.\n\n"
                    f"Saved binary masks: {len(saved_paths)}\n"
                    f"Saved instance masks: {len(saved_instance_paths)}\n\n"
                    f"GUI reload error:\n{reload_error}"
                ),
            )

    def _restore_semantic_predict_idle(self):
        self._set_semantic_predict_state(
            "idle"
        )

        if hasattr(
            self,
            "semantic_threshold_spin",
        ):
            self.semantic_threshold_spin.setEnabled(
                True
            )

        self._update_controls()


    def _semantic_predict_failed(
        self,
        message,
    ):
        self._set_semantic_predict_state(
            "error"
        )

        if hasattr(
            self,
            "semantic_threshold_spin",
        ):
            self.semantic_threshold_spin.setEnabled(
                True
            )

        # If current mask readers were released for overwrite, reopen whatever
        # final mask is now on disk (old file if prediction failed early, or the
        # newly replaced file if another later image caused the batch failure).
        reload_row = getattr(self, "_semantic_prediction_reload_row", None)
        if reload_row is not None:
            try:
                if 0 <= int(reload_row) < len(self.image_paths):
                    self.load_image(int(reload_row))
            except Exception:
                pass
            self._semantic_prediction_reload_row = None

        QMessageBox.critical(
            self,
            "Predict failed",
            str(message),
        )

        self._update_controls()

    # -------------------------------------------------------------------------
    # Backup drag/drop on whole window
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]

        if paths:
            self.add_input_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    def open_files_or_folder(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open 2D / 3D images",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
        )

        if files:
            self.add_input_paths([Path(f) for f in files])
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Open image folder",
            "",
        )

        if folder:
            self.add_input_paths([Path(folder)])

    def add_input_paths(self, paths):
        images = collect_images_from_paths(paths)

        if not images:
            QMessageBox.information(
                self,
                "No images",
                "No supported source images were found.",
            )
            return

        if not self._confirm_discard():
            return

        self.image_paths = images

        self.file_list.blockSignals(True)
        self.file_list.clear()

        for p in images:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.file_list.addItem(item)

        self.file_list.blockSignals(False)
        self.file_list.setCurrentRow(0)

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self.image_paths):
            return

        if not self._confirm_discard():
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_index)
            self.file_list.blockSignals(False)
            return

        self.load_image(row)

    # -------------------------------------------------------------------------
    # Loading / caches
    # -------------------------------------------------------------------------

    def reload_current(self):
        if self.current_index >= 0:
            self.load_image(self.current_index)

    def _build_instance_slices(self):
        self.instance_slices = {}

        if self.labels is None:
            return

        labels = self.labels
        h, w = labels.shape

        yy, xx = np.nonzero(labels > 0)
        if yy.size == 0:
            return

        ids = labels[yy, xx].astype(np.int64, copy=False)
        max_id = int(ids.max())

        min_y = np.full(max_id + 1, h, dtype=np.int64)
        max_y = np.full(max_id + 1, -1, dtype=np.int64)
        min_x = np.full(max_id + 1, w, dtype=np.int64)
        max_x = np.full(max_id + 1, -1, dtype=np.int64)

        np.minimum.at(min_y, ids, yy)
        np.maximum.at(max_y, ids, yy)
        np.minimum.at(min_x, ids, xx)
        np.maximum.at(max_x, ids, xx)

        for label_id in np.unique(ids):
            label_id = int(label_id)
            self.instance_slices[label_id] = (
                slice(
                    int(min_y[label_id]),
                    int(max_y[label_id]) + 1,
                ),
                slice(
                    int(min_x[label_id]),
                    int(max_x[label_id]) + 1,
                ),
            )


    def _close_mask_readers_for_overwrite(self):
        """Release existing *_mask TIFF handles before overwrite on Windows."""
        for reader_name in ("binary_reader", "instance_reader"):
            reader = getattr(self, reader_name, None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
            setattr(self, reader_name, None)

        # Cached slices may point at memmapped mask data; discard them too.
        self.slice_label_cache = {}

    def _close_volume_readers(self):
        for reader_name in (
            "image_reader",
            "binary_reader",
            "instance_reader",
        ):
            reader = getattr(self, reader_name, None)

            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

            setattr(self, reader_name, None)

    def _get_rgb_for_z(self, z_index):
        z_index = int(z_index)

        if z_index in self.rgb_cache:
            rgb = self.rgb_cache.pop(z_index)
            self.rgb_cache[z_index] = rgb
            return rgb

        raw = self.image_reader.get_slice(z_index)
        rgb = image_to_rgb8(raw)

        self.rgb_cache[z_index] = rgb

        while len(self.rgb_cache) > self.rgb_cache_limit:
            self.rgb_cache.popitem(last=False)

        return rgb

    def _get_labels_for_z(self, z_index, cache=True):
        z_index = int(z_index)

        if z_index in self.slice_label_cache:
            return self.slice_label_cache[z_index]

        if self.instance_reader is not None:
            labels = self.instance_reader.get_slice(z_index)

        elif self.binary_reader is not None:
            binary = self.binary_reader.get_slice(z_index)
            labels, _ = split_binary_mask_fast(binary)

        else:
            labels = np.zeros(
                (self.volume_height, self.volume_width),
                dtype=np.int32,
            )

        if cache:
            self.slice_label_cache[z_index] = labels

        return labels

    def _queue_live_z(self, value):
        """
        Called continuously while the user drags the Z slider.

        The numeric label updates immediately, while image rendering is
        throttled to roughly 30 ms intervals. If the mouse moves across many
        slices quickly, only the newest requested Z is rendered.
        """
        value = int(value)

        self.z_value_label.setText(
            f"{value} / {self.z_count}"
        )

        self.pending_z = value - 1

        if not self.z_refresh_timer.isActive():
            self.z_refresh_timer.start(
                self.z_refresh_interval_ms
            )

    def _z_value_changed(self, value):
        """
        Handles keyboard arrows, page-step clicks, or programmatic changes.

        During mouse dragging, sliderMoved() already queues the live preview,
        so valueChanged() does not trigger a second refresh.
        """
        if self.current_image_path is None:
            return

        value = int(value)

        self.z_value_label.setText(
            f"{value} / {self.z_count}"
        )

        if self.z_slider.isSliderDown():
            return

        self.pending_z = value - 1
        self._render_pending_z()

    def _render_pending_z(self):
        """
        Render only the newest requested Z slice.

        No backlog is accumulated: pending_z is consumed once per refresh.
        """
        if self.current_image_path is None:
            self.pending_z = None
            return

        if self.pending_z is None:
            return

        new_z = int(self.pending_z)
        self.pending_z = None

        if new_z == self.current_z:
            return

        old_z = self.current_z

        try:
            self._show_z_slice(new_z, fit=False)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Z slice load failed",
                str(exc),
            )

            self.pending_z = None

            self.z_slider.blockSignals(True)
            self.z_slider.setValue(old_z + 1)
            self.z_slider.blockSignals(False)

            self.z_value_label.setText(
                f"{old_z + 1} / {self.z_count}"
            )

    def _flush_live_z(self):
        """
        When the mouse is released, immediately render the final slider Z.

        This guarantees that even if the drag ended between timer ticks, the
        image finishes exactly on the layer under the slider handle.
        """
        if self.current_image_path is None:
            return

        if self.z_refresh_timer.isActive():
            self.z_refresh_timer.stop()

        self.pending_z = int(self.z_slider.value()) - 1
        self._render_pending_z()


    def _show_z_slice(self, z_index, fit=False):
        z_index = int(z_index)

        if not (0 <= z_index < self.z_count):
            return

        self.current_z = z_index
        self.image_rgb = self._get_rgb_for_z(z_index)
        self.labels = self._get_labels_for_z(z_index, cache=True)

        self.instance_sizes = sizes_from_labels(self.labels)
        self.selected_ids.clear()
        self.canvas.clear_selection_items()

        self._build_instance_slices()
        self._rebuild_base_overlay()

        self.canvas.set_base_pixmap(
            rgb_to_pixmap(
                self._adjust_display_rgb(self.image_rgb)
            )
        )
        self.apply_display_mode()

        if fit:
            self.canvas.fit_image()

        # Do not move the slider handle from code while the user is dragging;
        # otherwise the rendering loop can visually "fight" the mouse.
        if not self.z_slider.isSliderDown():
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(z_index + 1)
            self.z_slider.blockSignals(False)

        self.z_value_label.setText(
            f"{z_index + 1} / {self.z_count}"
        )

        self.image_info.setText(
            f"{self.current_image_path.name}  |  "
            f"{self.volume_width} × {self.volume_height} × {self.z_count}  |  "
            f"Z {z_index + 1}/{self.z_count}"
        )

        self._update_status()
        self._update_controls()

    def load_image(self, index):
        if index < 0 or index >= len(self.image_paths):
            return

        path = self.image_paths[index]

        # Release file handles from the previous volume first.
        self._close_volume_readers()

        try:
            image_reader = ImageVolumeReader(path)

            z_count = int(image_reader.z_count)
            h = int(image_reader.height)
            w = int(image_reader.width)

            binary_path = expected_mask_path(path)
            instance_path = expected_instance_path(path)

            binary_reader = None
            instance_reader = None
            source = "none"

            expected_zyx = (z_count, h, w)

            # Automatic source priority:
            # 1) existing *_mask_instance.tif
            # 2) otherwise existing *_mask.tif
            # 3) otherwise empty mask
            #
            # For a 3D binary mask, connected components are generated only
            # for the Z slice that is actually visited. The complete instance
            # stack is generated when Save is pressed.
            if instance_path.exists():
                instance_reader = MaskVolumeReader(
                    instance_path,
                    expected_zyx,
                    instance=True,
                )
                source = "instance"

            elif binary_path.exists():
                binary_reader = MaskVolumeReader(
                    binary_path,
                    expected_zyx,
                    instance=False,
                )
                source = "binary (slice-wise instances)"

        except Exception as exc:
            try:
                image_reader.close()
            except Exception:
                pass

            QMessageBox.critical(
                self,
                "Load failed",
                f"{path.name}\n\n{exc}",
            )
            return

        self.current_index = index
        self.current_image_path = path
        self.binary_mask_path = binary_path
        self.instance_mask_path = instance_path

        self.image_reader = image_reader
        self.binary_reader = binary_reader
        self.instance_reader = instance_reader

        self.z_count = z_count
        self.current_z = 0
        self.volume_height = h
        self.volume_width = w

        self.slice_label_cache = {}
        self.rgb_cache.clear()

        self.pending_z = None
        if self.z_refresh_timer.isActive():
            self.z_refresh_timer.stop()

        self.image_adjust_pending = False
        if self.image_adjust_timer.isActive():
            self.image_adjust_timer.stop()

        self.image_rgb = None
        self.labels = None
        self.instance_sizes = {}
        self.instance_slices = {}
        self.selected_ids = set()

        self.dirty = False
        self.current_source = source
        self._set_save_button_state("idle")
        self._set_delete_button_state("idle")

        self.z_slider.blockSignals(True)
        self.z_slider.setRange(1, max(1, self.z_count))
        self.z_slider.setValue(1)
        self.z_slider.setEnabled(self.z_count > 1)
        self.z_slider.blockSignals(False)

        self._show_z_slice(0, fit=True)

        self._refresh_file_prefix()


    def _rebuild_base_overlay(self):
        max_label = int(self.labels.max()) if self.labels.size else 0

        self.instance_lut = generate_lut(max_label)
        self.base_overlay_rgba = self.instance_lut[self.labels].copy()

        self.apply_display_mode()

    # -------------------------------------------------------------------------
    # Display-only brightness / contrast
    # -------------------------------------------------------------------------

    def _adjust_display_rgb(self, rgb):
        """
        Fast display-only brightness / contrast using an 8-bit LUT.

        Only 256 values are calculated in Python/NumPy; OpenCV applies the LUT
        to the whole RGB image in compiled code. This is substantially faster
        than converting every pixel to float32 for every slider movement.
        """
        if rgb is None:
            return None

        # Fast path for the default view.
        if (
            self.display_brightness == 0
            and self.display_contrast == 100
        ):
            return rgb

        contrast = float(self.display_contrast) / 100.0
        brightness = float(self.display_brightness)

        values = np.arange(256, dtype=np.float32)
        lut = np.clip(
            (values - 127.5) * contrast
            + 127.5
            + brightness,
            0,
            255,
        ).astype(np.uint8)

        # cv2.LUT applies the same 256-entry LUT to all RGB channels.
        return cv2.LUT(
            np.ascontiguousarray(rgb),
            lut,
        )


    def _refresh_image_appearance(self):
        """Immediately render the current display settings."""
        if self.image_rgb is None:
            return

        adjusted = self._adjust_display_rgb(self.image_rgb)

        self.canvas.set_base_pixmap(
            rgb_to_pixmap(adjusted)
        )

    def _queue_image_adjustment_refresh(self):
        """
        Queue only the newest brightness/contrast state.

        While dragging, slider signals can arrive much faster than a large
        QPixmap can be rebuilt. This timer coalesces those signals so the UI
        stays responsive instead of processing stale intermediate states.
        """
        self.image_adjust_pending = True

        if not self.image_adjust_timer.isActive():
            self.image_adjust_timer.start(
                self.image_adjust_interval_ms
            )

    def _render_pending_image_adjustment(self):
        if not self.image_adjust_pending:
            return

        self.image_adjust_pending = False
        self._refresh_image_appearance()

        # If another slider event arrived while the image was rendering,
        # schedule one more refresh for the newest value only.
        if self.image_adjust_pending:
            self.image_adjust_timer.start(
                self.image_adjust_interval_ms
            )

    def _flush_image_adjustment(self):
        """On mouse release, render the exact final slider state immediately."""
        if self.image_adjust_timer.isActive():
            self.image_adjust_timer.stop()

        self.image_adjust_pending = False
        self._refresh_image_appearance()

    def _brightness_changed(self, value):
        self.display_brightness = int(value)
        self.brightness_value.setText(
            f"+{value}" if value > 0 else str(value)
        )

        if self.brightness_slider.isSliderDown():
            self._queue_image_adjustment_refresh()
        else:
            self._refresh_image_appearance()

    def _contrast_changed(self, value):
        self.display_contrast = int(value)
        self.contrast_value.setText(f"{value}%")

        if self.contrast_slider.isSliderDown():
            self._queue_image_adjustment_refresh()
        else:
            self._refresh_image_appearance()


    def _reset_image_adjustments(self):
        if self.image_adjust_timer.isActive():
            self.image_adjust_timer.stop()
        self.image_adjust_pending = False

        self.brightness_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)

        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)

        self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)

        self.display_brightness = 0
        self.display_contrast = 100
        self.brightness_value.setText("0")
        self.contrast_value.setText("100%")
        self._refresh_image_appearance()


    # -------------------------------------------------------------------------
    # Overlay display mode
    # -------------------------------------------------------------------------

    def set_display_mode(self, mode):
        if mode not in {"instance", "mask"}:
            return

        self.display_mode = mode
        self.apply_display_mode()

    def set_hide_overlay(self, checked):
        self.hide_overlay = bool(checked)
        self.hide_overlay_switch.update()
        self.apply_display_mode()


    def apply_display_mode(self):
        if self.labels is None:
            self.canvas.set_mask_pixmap(QPixmap())
            self.canvas.clear_selection_items()
            return

        if self.hide_overlay:
            self.canvas.set_mask_pixmap(QPixmap())
            self.canvas.clear_selection_items()
            return

        if self.display_mode == "mask":
            h, w = self.labels.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)

            foreground = self.labels > 0

            # Single semi-transparent cyan/blue for the semantic mask.
            rgba[foreground, 0] = 0
            rgba[foreground, 1] = 174
            rgba[foreground, 2] = 239
            rgba[foreground, 3] = MASK_ALPHA

            self.canvas.set_mask_pixmap(
                rgba_to_pixmap(rgba)
            )

        else:
            # Multicolor instance view.
            if self.base_overlay_rgba is None:
                self.canvas.set_mask_pixmap(QPixmap())
            else:
                self.canvas.set_mask_pixmap(
                    rgba_to_pixmap(self.base_overlay_rgba)
                )

        # Selected instances remain gray in both visible display modes.
        self.canvas.clear_selection_items()
        self._sync_selection_visuals()

    # -------------------------------------------------------------------------
    # Fast selected-instance crops
    # -------------------------------------------------------------------------

    def _make_gray_crop(self, label_id):
        slc = self.instance_slices.get(label_id)

        if slc is None:
            return None

        crop_labels = self.labels[slc]
        mask = crop_labels == label_id

        if not np.any(mask):
            return None

        hh, ww = mask.shape
        rgba = np.zeros((hh, ww, 4), dtype=np.uint8)

        rgba[mask, 0] = GRAY_SELECTED[0]
        rgba[mask, 1] = GRAY_SELECTED[1]
        rgba[mask, 2] = GRAY_SELECTED[2]
        rgba[mask, 3] = GRAY_SELECTED[3]

        y_slice, x_slice = slc

        return (
            rgba_to_pixmap(rgba),
            int(x_slice.start),
            int(y_slice.start),
        )

    def _select_id_visual(self, label_id):
        if label_id <= 0:
            return

        if self.hide_overlay:
            return

        if label_id in self.canvas.selection_items:
            return

        result = self._make_gray_crop(label_id)

        if result is None:
            return

        pixmap, x, y = result
        self.canvas.set_selection_crop(label_id, pixmap, x, y)

    def _deselect_id_visual(self, label_id):
        self.canvas.remove_selection_crop(label_id)

    def _sync_selection_visuals(self):
        current_visual = set(self.canvas.selection_items.keys())

        for label_id in current_visual - self.selected_ids:
            self._deselect_id_visual(label_id)

        for label_id in self.selected_ids - current_visual:
            self._select_id_visual(label_id)

    # -------------------------------------------------------------------------
    # Modes / brush size
    # -------------------------------------------------------------------------

    def set_interaction_mode(self, mode):
        self.canvas.set_mode(mode)

        if mode == MaskCanvas.MODE_SELECT:
            pass

        elif mode == MaskCanvas.MODE_RECT:
            pass

        elif mode == MaskCanvas.MODE_BRUSH:
            pass

        elif mode == MaskCanvas.MODE_POLYGON:
            pass

    def _brush_slider_changed(self, value):
        if self.brush_spin.value() != value:
            self.brush_spin.blockSignals(True)
            self.brush_spin.setValue(value)
            self.brush_spin.blockSignals(False)

        self.canvas.set_brush_diameter(value)

    def _brush_spin_changed(self, value):
        if self.brush_slider.value() != value:
            self.brush_slider.blockSignals(True)
            self.brush_slider.setValue(value)
            self.brush_slider.blockSignals(False)

        self.canvas.set_brush_diameter(value)

    # -------------------------------------------------------------------------
    # Select / rectangle
    # -------------------------------------------------------------------------

    def select_at(self, x, y, ctrl):
        if self.labels is None:
            return

        h, w = self.labels.shape

        if not (0 <= x < w and 0 <= y < h):
            return

        label_id = int(self.labels[y, x])

        if ctrl:
            if label_id > 0:
                if label_id in self.selected_ids:
                    self.selected_ids.remove(label_id)
                    self._deselect_id_visual(label_id)
                else:
                    self.selected_ids.add(label_id)
                    self._select_id_visual(label_id)

        else:
            old_ids = set(self.selected_ids)

            if label_id > 0:
                self.selected_ids = {label_id}
            else:
                self.selected_ids.clear()

            for old_id in old_ids - self.selected_ids:
                self._deselect_id_visual(old_id)

            for new_id in self.selected_ids - old_ids:
                self._select_id_visual(new_id)

        self._update_status()
        self._update_controls()

    def select_by_rectangle(self, x1, y1, x2, y2, ctrl):
        if self.labels is None:
            return

        h, w = self.labels.shape

        left = max(0, min(x1, x2))
        right = min(w - 1, max(x1, x2))
        top = max(0, min(y1, y2))
        bottom = min(h - 1, max(y1, y2))

        if right < left or bottom < top:
            return

        touched = np.unique(
            self.labels[top:bottom + 1, left:right + 1]
        )

        touched_ids = {
            int(v)
            for v in touched
            if int(v) > 0
        }

        if ctrl:
            self.selected_ids.update(touched_ids)
        else:
            self.selected_ids = touched_ids

        self._sync_selection_visuals()
        self._update_status()
        self._update_controls()

        # status bar removed

    def clear_selection(self):
        self.selected_ids.clear()
        self.canvas.clear_selection_items()
        self._update_status()
        self._update_controls()

    # -------------------------------------------------------------------------
    # Brush
    # -------------------------------------------------------------------------

    def add_brush_stroke(self, points, diameter):
        if self.labels is None or not points:
            return

        h, w = self.labels.shape

        pts = np.asarray(points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        # Restrict raster work to a local bounding box around the stroke.
        radius = max(1, int(np.ceil(diameter / 2.0)))

        x_min = max(0, int(pts[:, 0].min()) - radius - 2)
        x_max = min(w - 1, int(pts[:, 0].max()) + radius + 2)
        y_min = max(0, int(pts[:, 1].min()) - radius - 2)
        y_max = min(h - 1, int(pts[:, 1].max()) + radius + 2)

        local_w = x_max - x_min + 1
        local_h = y_max - y_min + 1

        local_pts = pts.copy()
        local_pts[:, 0] -= x_min
        local_pts[:, 1] -= y_min

        stroke = np.zeros((local_h, local_w), dtype=np.uint8)

        if len(local_pts) == 1:
            cv2.circle(
                stroke,
                tuple(local_pts[0]),
                radius,
                1,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        else:
            cv2.polylines(
                stroke,
                [local_pts.reshape(-1, 1, 2)],
                isClosed=False,
                color=1,
                thickness=max(1, int(diameter)),
                lineType=cv2.LINE_8,
            )

            cv2.circle(
                stroke,
                tuple(local_pts[0]),
                radius,
                1,
                thickness=-1,
                lineType=cv2.LINE_8,
            )

            cv2.circle(
                stroke,
                tuple(local_pts[-1]),
                radius,
                1,
                thickness=-1,
                lineType=cv2.LINE_8,
            )

        # Brush is intentionally stroke-only.
        # Even if the stroke accidentally closes into a loop, its interior
        # is NOT filled. Filled regions belong exclusively to Polygon mode.

        local_labels = self.labels[
            y_min:y_max + 1,
            x_min:x_max + 1,
        ]

        add_mask = (stroke > 0) & (local_labels == 0)

        added_pixels = int(np.sum(add_mask))

        if added_pixels == 0:
            return

        new_id = int(self.labels.max()) + 1

        local_labels[add_mask] = new_id
        self.instance_sizes[new_id] = added_pixels

        yy, xx = np.nonzero(add_mask)

        self.instance_slices[new_id] = (
            slice(y_min + int(yy.min()), y_min + int(yy.max()) + 1),
            slice(x_min + int(xx.min()), x_min + int(xx.max()) + 1),
        )

        self.dirty = True

        # Rebuild base colored overlay only after a completed stroke.
        self._rebuild_base_overlay()

        self.selected_ids = {new_id}
        self.canvas.clear_selection_items()
        self._select_id_visual(new_id)

        self._update_status()
        self._update_controls()

        # status bar removed

    def add_polygon_region(self, points, diameter):
        """
        Add one filled polygon/freehand closed region as ONE new instance.

        The user traces the boundary with the mouse. On release, the last
        point is automatically connected back to the first point, so the
        Polygon tool does not depend on the user landing exactly on the
        starting pixel.

        Brush/Polygon share the same diameter control. For Polygon, diameter
        controls the outline thickness while the enclosed area is filled.
        Existing instances are protected and never overwritten.
        """
        if self.labels is None or not points:
            return

        h, w = self.labels.shape

        pts = np.asarray(points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        # A filled polygon needs at least three distinct coordinates.
        if len(np.unique(pts, axis=0)) < 3:
            return

        radius = max(1, int(np.ceil(diameter / 2.0)))

        x_min = max(0, int(pts[:, 0].min()) - radius - 2)
        x_max = min(w - 1, int(pts[:, 0].max()) + radius + 2)
        y_min = max(0, int(pts[:, 1].min()) - radius - 2)
        y_max = min(h - 1, int(pts[:, 1].max()) + radius + 2)

        local_w = x_max - x_min + 1
        local_h = y_max - y_min + 1

        local_pts = pts.copy()
        local_pts[:, 0] -= x_min
        local_pts[:, 1] -= y_min

        polygon_mask = np.zeros(
            (local_h, local_w),
            dtype=np.uint8,
        )

        contour = local_pts.reshape(-1, 1, 2)

        # Fill the traced region.
        cv2.fillPoly(
            polygon_mask,
            [contour],
            color=1,
            lineType=cv2.LINE_8,
        )

        # Add the same shared diameter as an explicit closed outline.
        cv2.polylines(
            polygon_mask,
            [contour],
            isClosed=True,
            color=1,
            thickness=max(1, int(diameter)),
            lineType=cv2.LINE_8,
        )

        local_labels = self.labels[
            y_min:y_max + 1,
            x_min:x_max + 1,
        ]

        add_mask = (
            (polygon_mask > 0)
            & (local_labels == 0)
        )

        added_pixels = int(np.sum(add_mask))

        if added_pixels == 0:
            return

        new_id = int(self.labels.max()) + 1

        local_labels[add_mask] = new_id
        self.instance_sizes[new_id] = added_pixels

        yy, xx = np.nonzero(add_mask)

        self.instance_slices[new_id] = (
            slice(
                y_min + int(yy.min()),
                y_min + int(yy.max()) + 1,
            ),
            slice(
                x_min + int(xx.min()),
                x_min + int(xx.max()) + 1,
            ),
        )

        self.dirty = True

        self._rebuild_base_overlay()

        self.selected_ids = {new_id}
        self.canvas.clear_selection_items()
        self._select_id_visual(new_id)

        self._update_status()
        self._update_controls()

        # status bar removed

    # -------------------------------------------------------------------------
    # Delete button feedback
    # -------------------------------------------------------------------------

    def _set_delete_button_state(self, state):
        if not hasattr(self, "delete_btn"):
            return

        state = str(state)
        self.delete_feedback_state = state

        if state == "deleting":
            text = "Deleting..."
            if self.delete_feedback_timer.isActive():
                self.delete_feedback_timer.stop()
        elif state == "deleted":
            text = "Deleted ✓"
            self.delete_feedback_timer.start(900)
        else:
            state = "idle"
            self.delete_feedback_state = state
            text = "Delete selected"
            if self.delete_feedback_timer.isActive():
                self.delete_feedback_timer.stop()

        self.delete_btn.setText(text)
        self.delete_btn.setProperty("state", state)

        style = self.delete_btn.style()
        style.unpolish(self.delete_btn)
        style.polish(self.delete_btn)
        self.delete_btn.update()

    def _restore_delete_button_idle(self):
        self._set_delete_button_state("idle")


    # -------------------------------------------------------------------------
    # Save button feedback
    # -------------------------------------------------------------------------

    def _set_save_button_state(self, state):
        """
        Give clear visual feedback for Save:
        idle   = default appearance
        saving = immediate active feedback after click
        saved  = successful save
        error  = save failed
        """
        if not hasattr(self, "save_btn"):
            return

        state = str(state)
        self.save_feedback_state = state

        if state == "saving":
            text = "Saving..."
            if self.save_feedback_timer.isActive():
                self.save_feedback_timer.stop()
        elif state == "saved":
            text = "Saved ✓"
            self.save_feedback_timer.start(1400)
        elif state == "error":
            text = "Save failed"
            self.save_feedback_timer.start(1800)
        else:
            state = "idle"
            self.save_feedback_state = state
            text = "Save masks"
            if self.save_feedback_timer.isActive():
                self.save_feedback_timer.stop()

        self.save_btn.setText(text)
        self.save_btn.setProperty("state", state)

        style = self.save_btn.style()
        style.unpolish(self.save_btn)
        style.polish(self.save_btn)
        self.save_btn.update()

    def _restore_save_button_idle(self):
        self._set_save_button_state("idle")


    # -------------------------------------------------------------------------
    # Delete / save
    # -------------------------------------------------------------------------

    def delete_selected(self):
        if self.labels is None or not self.selected_ids:
            return

        self._set_delete_button_state("deleting")
        QApplication.processEvents()

        deleted_pixels = 0
        deleted_instances = len(self.selected_ids)

        for label_id in list(self.selected_ids):
            slc = self.instance_slices.get(label_id)

            if slc is None:
                continue

            crop = self.labels[slc]
            mask = crop == label_id

            deleted_pixels += int(np.sum(mask))
            crop[mask] = 0

            self.instance_sizes.pop(label_id, None)
            self.instance_slices.pop(label_id, None)

        self.selected_ids.clear()
        self.canvas.clear_selection_items()

        self.dirty = True

        # One full overlay rebuild after the delete action only.
        self._rebuild_base_overlay()

        self._update_status()
        self._update_controls()
        self._set_delete_button_state("deleted")

        # status bar removed

    def save_masks(self):
        """
        Save the complete mask volume.

        The temporary disk-backed array is stored in the operating system's
        temp directory, never beside the user's microscopy images.

        Persistent outputs are ONLY:
          <stem>_mask.tif
          <stem>_mask_instance.tif

        Reopening an image depends only on those TIFF files.
        """
        if self.labels is None or self.current_image_path is None:
            return

        self._set_save_button_state("saving")
        QApplication.processEvents()

        temp_path = None
        temp_labels = None
        labels_2d = None
        binary_iter = None
        instance_iter = None

        try:
            shape_zyx = (
                self.z_count,
                self.volume_height,
                self.volume_width,
            )

            # IMPORTANT:
            # Use the OS temporary directory. Do not create .dat files in the
            # user's image folder.
            fd, temp_name = tempfile.mkstemp(
                prefix="linknex_mask_",
                suffix=".tmp",
            )

            import os
            os.close(fd)

            temp_path = Path(temp_name)

            temp_labels = np.memmap(
                temp_path,
                mode="w+",
                dtype=np.uint32,
                shape=shape_zyx,
            )

            max_label = 0

            for z_index in range(self.z_count):
                labels_z = self._get_labels_for_z(
                    z_index,
                    cache=False,
                )

                labels_z = np.asarray(labels_z)

                if labels_z.shape != (
                    self.volume_height,
                    self.volume_width,
                ):
                    raise ValueError(
                        f"Unexpected label shape at Z={z_index + 1}: "
                        f"{labels_z.shape}"
                    )

                local_max = (
                    int(labels_z.max())
                    if labels_z.size
                    else 0
                )
                max_label = max(max_label, local_max)

                temp_labels[z_index] = labels_z.astype(
                    np.uint32,
                    copy=False,
                )

            temp_labels.flush()

            instance_dtype = (
                np.uint16
                if max_label <= 65535
                else np.uint32
            )

            # Close old readers before overwriting TIFFs on Windows.
            if self.binary_reader is not None:
                self.binary_reader.close()
                self.binary_reader = None

            if self.instance_reader is not None:
                self.instance_reader.close()
                self.instance_reader = None

            if self.z_count == 1:
                # Make a true RAM copy so no later object keeps the temp mmap
                # file open on Windows.
                labels_2d = np.array(
                    temp_labels[0],
                    dtype=np.uint32,
                    copy=True,
                )

                tifffile.imwrite(
                    self.binary_mask_path,
                    (labels_2d > 0).astype(np.uint8) * 255,
                    photometric="minisblack",
                )

                tifffile.imwrite(
                    self.instance_mask_path,
                    labels_2d.astype(instance_dtype),
                    photometric="minisblack",
                )

            else:
                binary_iter = (
                    (np.asarray(temp_labels[z]) > 0)
                    .astype(np.uint8) * 255
                    for z in range(self.z_count)
                )

                instance_iter = (
                    np.asarray(
                        temp_labels[z],
                        dtype=instance_dtype,
                    )
                    for z in range(self.z_count)
                )

                binary_bytes = (
                    self.z_count
                    * self.volume_height
                    * self.volume_width
                )
                instance_bytes = (
                    binary_bytes
                    * np.dtype(instance_dtype).itemsize
                )

                tifffile.imwrite(
                    self.binary_mask_path,
                    data=binary_iter,
                    shape=shape_zyx,
                    dtype=np.uint8,
                    photometric="minisblack",
                    bigtiff=binary_bytes >= 4_000_000_000,
                    metadata={"axes": "ZYX"},
                )

                tifffile.imwrite(
                    self.instance_mask_path,
                    data=instance_iter,
                    shape=shape_zyx,
                    dtype=instance_dtype,
                    photometric="minisblack",
                    bigtiff=instance_bytes >= 4_000_000_000,
                    metadata={"axes": "ZYX"},
                )

                binary_iter = None
                instance_iter = None

            # Verify persistent outputs before reporting success.
            if not self.binary_mask_path.exists():
                raise RuntimeError(
                    f"Binary mask was not created:\n"
                    f"{self.binary_mask_path}"
                )

            if not self.instance_mask_path.exists():
                raise RuntimeError(
                    f"Instance mask was not created:\n"
                    f"{self.instance_mask_path}"
                )

            # Verify that the saved instance TIFF can be reopened independently
            # of the temporary file.
            verify_reader = MaskVolumeReader(
                self.instance_mask_path,
                shape_zyx,
                instance=True,
            )
            _ = verify_reader.get_slice(self.current_z)
            verify_reader.close()
            del verify_reader

            # Persistent instance TIFF is now the source of truth.
            self.instance_reader = MaskVolumeReader(
                self.instance_mask_path,
                shape_zyx,
                instance=True,
            )
            self.binary_reader = None
            self.current_source = "instance"

            current_z = self.current_z
            self.slice_label_cache = {}
            self.labels = self._get_labels_for_z(
                current_z,
                cache=True,
            )

            self.instance_sizes = sizes_from_labels(self.labels)
            self._build_instance_slices()
            self._rebuild_base_overlay()

            self.selected_ids.clear()
            self.canvas.clear_selection_items()
            self.dirty = False

            self._refresh_file_prefix()
            self._update_status()
            self._update_controls()
            self._set_save_button_state("saved")

        except Exception as exc:
            self._set_save_button_state("error")

            QMessageBox.critical(
                self,
                "Save failed",
                str(exc),
            )

            if (
                self.instance_reader is None
                and self.instance_mask_path is not None
                and self.instance_mask_path.exists()
            ):
                try:
                    self.instance_reader = MaskVolumeReader(
                        self.instance_mask_path,
                        (
                            self.z_count,
                            self.volume_height,
                            self.volume_width,
                        ),
                        instance=True,
                    )
                except Exception:
                    pass

        finally:
            # Release every object that could still hold a view on the memmap.
            labels_2d = None
            binary_iter = None
            instance_iter = None

            if temp_labels is not None:
                try:
                    temp_labels.flush()
                except Exception:
                    pass

                # Explicitly close the OS mmap handle before deleting.
                try:
                    mmap_obj = getattr(temp_labels, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                except Exception:
                    pass

                temp_labels = None

            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


    def remove_current_from_list(self):
        """
        Remove the current source image from the GUI working list only.
        The image file and masks on disk are NOT deleted.

        After removal:
          - if there is a next image at the same list position, load it;
          - otherwise load the previous remaining image;
          - if the list becomes empty, clear the canvas.
        """
        if (
            self.current_index < 0
            or self.current_index >= len(self.image_paths)
        ):
            return

        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved edits",
                "The current image has unsaved mask edits.\n\n"
                "Remove it from the GUI list WITHOUT saving these edits?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        remove_index = self.current_index
        removed_path = self.image_paths[remove_index]

        # Remove from model and visible list without triggering row-change logic.
        self.file_list.blockSignals(True)
        self.image_paths.pop(remove_index)
        self.file_list.takeItem(remove_index)
        self.file_list.blockSignals(False)

        self.selected_ids.clear()
        self.canvas.clear_selection_items()
        self.dirty = False

        if not self.image_paths:
            self.current_index = -1
            self.current_image_path = None
            self.binary_mask_path = None
            self.instance_mask_path = None
            self._close_volume_readers()

            self.z_count = 1
            self.current_z = 0
            self.volume_height = 0
            self.volume_width = 0
            self.slice_label_cache = {}
            self.rgb_cache.clear()

            self.pending_z = None
            if self.z_refresh_timer.isActive():
                self.z_refresh_timer.stop()

            self.image_adjust_pending = False
            if self.image_adjust_timer.isActive():
                self.image_adjust_timer.stop()

            self.image_rgb = None
            self.labels = None
            self.instance_sizes = {}
            self.instance_slices = {}
            self.instance_lut = None
            self.base_overlay_rgba = None

            self.z_slider.blockSignals(True)
            self.z_slider.setRange(1, 1)
            self.z_slider.setValue(1)
            self.z_slider.setEnabled(False)
            self.z_slider.blockSignals(False)
            self.z_value_label.setText("1 / 1")

            self._set_save_button_state("idle")
            self._set_delete_button_state("idle")

            self.canvas.clear_all()
            self.image_info.setText("Drag an image here to begin")

            self._update_status()
            self._update_controls()

            # status bar removed
            return

        # Prefer the next image occupying the same index.
        next_index = min(remove_index, len(self.image_paths) - 1)

        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(next_index)
        self.file_list.blockSignals(False)

        self.load_image(next_index)

        # status bar removed

    # -------------------------------------------------------------------------
    # Navigation / status
    # -------------------------------------------------------------------------

    def previous_image(self):
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if (
            self.current_index >= 0
            and self.current_index < len(self.image_paths) - 1
        ):
            self.file_list.setCurrentRow(self.current_index + 1)

    def _update_status(self):
        if self.current_image_path is None:
            self.source_status.setText("Source: —")
            self.instance_status.setText("Instances: —")
            self.selected_status.setText("Selected: 0")
            return

        dirty = " • unsaved" if self.dirty else ""

        self.source_status.setText(
            f"Source: {self.current_source}{dirty}"
        )

        self.instance_status.setText(
            f"Instances in current Z: {len(self.instance_sizes)}"
        )

        selected_pixels = sum(
            self.instance_sizes.get(i, 0)
            for i in self.selected_ids
        )

        self.selected_status.setText(
            f"Selected: {len(self.selected_ids)} instances\n"
            f"Pixels: {selected_pixels:,}"
        )

    def _update_controls(self):
        has_image = self.current_image_path is not None
        has_instances = bool(self.instance_sizes)
        has_selection = bool(self.selected_ids)

        self.delete_btn.setEnabled(has_image and has_selection)
        self.clear_selection_btn.setEnabled(has_image and has_selection)
        self.save_btn.setEnabled(has_image and self.labels is not None)
        self.remove_current_btn.setEnabled(has_image)

        self.z_slider.setEnabled(
            has_image and self.z_count > 1
        )

        self.display_instance_btn.setEnabled(has_image)
        self.display_mask_btn.setEnabled(has_image)
        self.hide_overlay_switch.setEnabled(has_image)

        self.brightness_slider.setEnabled(has_image)
        self.contrast_slider.setEnabled(has_image)
        self.reset_display_btn.setEnabled(has_image)

        self.select_mode_btn.setEnabled(has_image)
        self.rect_mode_btn.setEnabled(has_image and has_instances)

        # Brush and Polygon also work on an image with no existing mask.
        self.brush_mode_btn.setEnabled(has_image)
        self.polygon_mode_btn.setEnabled(has_image)
        self.brush_slider.setEnabled(has_image)
        self.brush_spin.setEnabled(has_image)

        self.fit_btn.setEnabled(has_image)

        model_ready = bool(
            self.semantic_model_path is not None
            and Path(self.semantic_model_path).exists()
        )
        self.semantic_predict_btn.setEnabled(
            bool(self._selected_semantic_paths()) and model_ready
        )
        if hasattr(self, "semantic_predict_all_btn"):
            self.semantic_predict_all_btn.setEnabled(
                bool(self.image_paths) and model_ready
            )

        self.prev_btn.setEnabled(
            has_image and self.current_index > 0
        )

        self.next_btn.setEnabled(
            has_image
            and self.current_index < len(self.image_paths) - 1
        )

    def _refresh_file_prefix(self):
        # Keep this method name for compatibility with the existing save/load
        # workflow, but the list now displays only the plain source filename.
        if (
            self.current_index < 0
            or self.current_index >= self.file_list.count()
        ):
            return

        p = self.current_image_path
        self.file_list.item(self.current_index).setText(p.name)

    def _confirm_discard(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved mask edits",
            "Save your mask edits before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Save:
            self.save_masks()
            return not self.dirty
        return answer == QMessageBox.Discard

    def closeEvent(self, event):
        if self._confirm_discard():
            if self.z_refresh_timer.isActive():
                self.z_refresh_timer.stop()
            if self.image_adjust_timer.isActive():
                self.image_adjust_timer.stop()
            if self.save_feedback_timer.isActive():
                self.save_feedback_timer.stop()
            if self.delete_feedback_timer.isActive():
                self.delete_feedback_timer.stop()
            self._close_volume_readers()
            event.accept()
        else:
            event.ignore()






# =============================================================================
# Embedded Instance Segmentation algorithms
# =============================================================================
#
# v44 distribution goal:
#   1) LinkNex main .py
#   2) PCD editor .py
#
# The original instance algorithm 01/02/03 sources are stored below as Python
# strings and loaded in-memory when the Instance module runs. No extra backend
# or algorithm .py files are required beside LinkNex.
#

LINKNEX_INSTANCE_ALGO01_SOURCE = 'from pathlib import Path\nimport colorsys\nimport os\n\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nfrom scipy import ndimage as ndi\nfrom scipy.sparse import csr_matrix\nfrom scipy.sparse.csgraph import minimum_spanning_tree\nfrom scipy.spatial import cKDTree\nfrom sklearn.cluster import DBSCAN\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import RobustScaler\n\n\ndef load_txt_array(path):\n    """读取逗号分隔、无表头的 txt 数据。"""\n    data = np.loadtxt(path, delimiter=",", dtype=np.float32)\n\n    if data.ndim == 1:\n        data = data.reshape(1, -1)\n\n    return data\n\n\ndef split_stem_from_pcd(path):\n    """从 xxx_pcd.txt 得到 xxx。"""\n    suffix = "_pcd.txt"\n\n    if not path.name.endswith(suffix):\n        raise ValueError(f"Unsupported PCD file: {path.name}")\n\n    return path.name[:-len(suffix)]\n\n\ndef find_pcd_files(processed_dir):\n    """寻找 processed_output 下所有原始 *_pcd.txt。"""\n    return sorted(\n        processed_dir.glob("*_pcd.txt"),\n        key=lambda p: p.name.lower(),\n    )\n\n\ndef find_skeleton_path(processed_dir, stem):\n    """\n    兼容两种拼写：\n        *_skelenton.txt\n        *_skeleton.txt\n    """\n    candidate_paths = [\n        processed_dir / f"{stem}_skelenton.txt",\n        processed_dir / f"{stem}_skeleton.txt",\n    ]\n\n    for path in candidate_paths:\n        if path.exists():\n            return path\n\n    # 如果都不存在，返回当前旧代码使用的拼写，便于报错时看清预期路径。\n    return candidate_paths[0]\n\n\ndef find_refined_skeleton_path(processed_dir, stem):\n    """\n    兼容两种拼写：\n        *_refinedskelenton.txt\n        *_refinedskeleton.txt\n    """\n    candidate_paths = [\n        processed_dir / f"{stem}_refinedskelenton.txt",\n        processed_dir / f"{stem}_refinedskeleton.txt",\n    ]\n\n    for path in candidate_paths:\n        if path.exists():\n            return path\n\n    return candidate_paths[0]\n\n\ndef print_file_information(label, path, data, column_format):\n    """打印文件路径、shape和当前约定的列格式。"""\n    print(f"{label} path   : {path}")\n    print(f"{label} shape  : {data.shape}")\n    print(f"{label} format : {column_format}")\n\n\ndef check_expected_columns(label, data, expected_columns):\n    """只检查列数，不修改数据。"""\n    actual_columns = data.shape[1]\n\n    if actual_columns != expected_columns:\n        print(\n            f"[WARNING] {label} expected {expected_columns} columns, "\n            f"but found {actual_columns} columns."\n        )\n\n\ndef add_missing_refined_skeleton_from_small_pcd_components(\n    pcd,\n    refined_skeleton,\n    dbscan_eps=2.0,\n    dbscan_min_samples=10,\n    min_component_points=20,\n    max_component_points=1000,\n    solid_grid_spacing=1.0,\n    local_padding=3,\n    surface_dilation_iterations=1,\n    surface_closing_iterations=1,\n):\n    """\n    逐个检查小 PCD instance 内部是否存在 refined skeleton。\n\n    处理流程\n    --------\n    1. 对 PCD xyz 做 DBSCAN，得到一个个 PCD instance；\n    2. 只检查指定点数范围内的小 PCD instance；\n    3. 将每个小 instance 的空心PCD表面转换成局部三维binary mask；\n    4. 封闭表面小缝并填充内部，得到实心占据范围；\n    5. 检查该实心范围内是否存在 skeleton；\n    6. 内部已有 skeleton：不处理；\n    7. 内部没有 skeleton：添加该 PCD instance 的质心。\n\n    输入\n    ----\n    pcd : ndarray, shape (N, >=3)\n        PCD 数据，前三列为 x,y,z。\n\n    refined_skeleton : ndarray, shape (M, >=4)\n        refined skeleton 数据，前四列为 x,y,z,score。\n\n    dbscan_eps : float\n        DBSCAN 的邻域半径。\n\n    dbscan_min_samples : int\n        DBSCAN 的 min_samples。\n\n    min_component_points, max_component_points : int\n        只检查点数位于这个范围内的 PCD component：\n            min_component_points <= component points < max_component_points\n\n    solid_grid_spacing : float\n        将局部PCD表面栅格化时的网格间距。当前坐标为voxel单位，默认1.0。\n\n    local_padding : int\n        局部三维网格四周增加的空白层数，避免封闭表面接触网格边界。\n\n    surface_dilation_iterations : int\n        PCD表面栅格化后轻微膨胀的次数，用于连接相邻表面采样点。\n\n    surface_closing_iterations : int\n        填充前进行binary closing的次数，用于封闭表面小缝。\n\n    返回\n    ----\n    new_refined_skeleton : ndarray, shape (M + K, 4)\n        补充后的 refined skeleton，格式为 x,y,z,score。\n\n    added_points : ndarray, shape (K, 4)\n        本次新添加的 x,y,z,score 骨架点。\n\n    labels : ndarray, shape (N,)\n        每个 PCD 点的 DBSCAN component label。\n\n    component_information : list of dict\n        被检查的小 component 信息。\n\n    说明\n    ----\n    包围盒只用于建立每个PCD instance自己的局部三维网格。\n    最终是否存在skeleton，由填充后的solid_mask决定，不由包围盒决定。\n\n    新增骨架点的 xyz 为 component 的点云质心。\n    新增点的 score 为质心到该 component 表面点云的最近距离，\n    与当前 refined skeleton 第四列的距离型 score 保持一致。\n    """\n    pcd = np.asarray(pcd, dtype=np.float32)\n    refined_skeleton = np.asarray(refined_skeleton, dtype=np.float32)\n\n    if pcd.ndim != 2 or pcd.shape[1] < 3:\n        raise ValueError("pcd must contain at least 3 columns: x,y,z")\n\n    if solid_grid_spacing <= 0:\n        raise ValueError("solid_grid_spacing must be larger than 0.")\n\n    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 4:\n        raise ValueError(\n            "refined_skeleton must contain at least 4 columns: x,y,z,score"\n        )\n\n    if min_component_points > max_component_points:\n        raise ValueError(\n            "min_component_points cannot be larger than max_component_points."\n        )\n\n    pcd_xyz = pcd[:, :3]\n    refined_skeleton_xyzi = refined_skeleton[:, :4].copy()\n    skeleton_xyz = refined_skeleton_xyzi[:, :3]\n\n    print("Running DBSCAN for small PCD components...")\n    print(f"DBSCAN eps              : {dbscan_eps}")\n    print(f"DBSCAN min_samples      : {dbscan_min_samples}")\n    print(\n        "Checked component size : "\n        f"{min_component_points} to {max_component_points} points"\n    )\n\n    labels = DBSCAN(\n        eps=dbscan_eps,\n        min_samples=dbscan_min_samples,\n        n_jobs=-1,\n    ).fit_predict(pcd_xyz)\n\n    component_labels = [\n        int(label)\n        for label in np.unique(labels)\n        if label != -1\n    ]\n\n    added_rows = []\n    component_information = []\n    checked_component_count = 0\n    components_with_skeleton_count = 0\n    components_without_skeleton_count = 0\n\n    surface_structure = ndi.generate_binary_structure(\n        rank=3,\n        connectivity=2,\n    )\n\n    for component_label in component_labels:\n        component_mask = labels == component_label\n        component_xyz = pcd_xyz[component_mask]\n        component_point_count = len(component_xyz)\n\n        if component_point_count < min_component_points:\n            continue\n\n        if component_point_count >= max_component_points:\n            continue\n\n        checked_component_count += 1\n\n        # 使用该PCD instance自己的坐标范围建立局部三维网格。\n        grid_min_index = (\n            np.floor(component_xyz.min(axis=0) / solid_grid_spacing)\n            .astype(np.int64)\n            - local_padding\n        )\n        grid_max_index = (\n            np.ceil(component_xyz.max(axis=0) / solid_grid_spacing)\n            .astype(np.int64)\n            + local_padding\n        )\n        grid_shape = tuple(\n            (grid_max_index - grid_min_index + 1).astype(np.int64)\n        )\n\n        surface_mask = np.zeros(grid_shape, dtype=bool)\n        surface_indices = (\n            np.rint(component_xyz / solid_grid_spacing).astype(np.int64)\n            - grid_min_index[None, :]\n        )\n        surface_mask[\n            surface_indices[:, 0],\n            surface_indices[:, 1],\n            surface_indices[:, 2],\n        ] = True\n\n        closed_surface_mask = surface_mask\n\n        if surface_dilation_iterations > 0:\n            closed_surface_mask = ndi.binary_dilation(\n                closed_surface_mask,\n                structure=surface_structure,\n                iterations=surface_dilation_iterations,\n            )\n\n        if surface_closing_iterations > 0:\n            closed_surface_mask = ndi.binary_closing(\n                closed_surface_mask,\n                structure=surface_structure,\n                iterations=surface_closing_iterations,\n            )\n\n        solid_mask = ndi.binary_fill_holes(closed_surface_mask)\n        solid_mask_point_count = int(np.sum(solid_mask))\n\n        # 将所有refined skeleton转换到这个局部网格中。\n        skeleton_indices = (\n            np.rint(skeleton_xyz / solid_grid_spacing).astype(np.int64)\n            - grid_min_index[None, :]\n        )\n        valid_skeleton_mask = np.all(\n            (skeleton_indices >= 0)\n            & (skeleton_indices < np.asarray(grid_shape)[None, :]),\n            axis=1,\n        )\n\n        has_internal_skeleton = False\n\n        if np.any(valid_skeleton_mask):\n            valid_indices = skeleton_indices[valid_skeleton_mask]\n            skeleton_inside_component = solid_mask[\n                valid_indices[:, 0],\n                valid_indices[:, 1],\n                valid_indices[:, 2],\n            ]\n            has_internal_skeleton = bool(\n                np.any(skeleton_inside_component)\n            )\n\n        if has_internal_skeleton:\n            components_with_skeleton_count += 1\n            component_information.append(\n                {\n                    "component_label": component_label,\n                    "component_points": int(component_point_count),\n                    "solid_grid_shape": grid_shape,\n                    "solid_mask_points": solid_mask_point_count,\n                    "status": "has_internal_skeleton",\n                }\n            )\n            continue\n\n        components_without_skeleton_count += 1\n\n        centroid = component_xyz.mean(axis=0)\n        component_tree = cKDTree(component_xyz)\n        centroid_score, _ = component_tree.query(centroid, k=1)\n\n        added_row = np.asarray(\n            [\n                centroid[0],\n                centroid[1],\n                centroid[2],\n                centroid_score,\n            ],\n            dtype=np.float32,\n        )\n        added_rows.append(added_row)\n\n        component_information.append(\n            {\n                "component_label": component_label,\n                "component_points": int(component_point_count),\n                "solid_grid_shape": grid_shape,\n                "solid_mask_points": solid_mask_point_count,\n                "status": "centroid_skeleton_added",\n                "centroid": centroid.tolist(),\n                "score": float(centroid_score),\n            }\n        )\n\n    if len(added_rows) > 0:\n        added_points = np.vstack(added_rows).astype(np.float32)\n        new_refined_skeleton = np.vstack(\n            [refined_skeleton_xyzi, added_points]\n        ).astype(np.float32)\n    else:\n        added_points = np.empty((0, 4), dtype=np.float32)\n        new_refined_skeleton = refined_skeleton_xyzi\n\n    print(f"PCD components          : {len(component_labels)}")\n    print(f"Small components checked: {checked_component_count}")\n    print(f"Small with skeleton     : {components_with_skeleton_count}")\n    print(f"Small without skeleton  : {components_without_skeleton_count}")\n    print(f"Skeleton points before  : {len(refined_skeleton_xyzi)}")\n    print(f"Skeleton points added   : {len(added_points)}")\n    print(f"Skeleton points after   : {len(new_refined_skeleton)}")\n\n    return (\n        new_refined_skeleton,\n        added_points,\n        labels,\n        component_information,\n    )\n\n\ndef segment_refined_skeleton_edges(\n    refined_skeleton,\n    neighbor_radius=1.8,\n    random_seed=42,\n    prune_shortcut_edges=True,\n):\n    """\n    将 refined skeleton 按照拓扑分叉点切分成边。\n\n    输入\n    ----\n    refined_skeleton : ndarray, shape (N, >=3)\n        refined skeleton 数据，前三列必须是 x,y,z。\n\n    neighbor_radius : float\n        两个 skeleton 点建立邻接关系的最大距离。\n        与 refined skeleton 生成代码的 connect_radius 保持一致，第一版使用 1.8。\n\n    random_seed : int\n        控制边颜色起始位置，保证同一次设置下颜色可复现。\n\n    prune_shortcut_edges : bool\n        True 时使用相对邻域规则删除急弯两侧、同一骨架跨点之间的\n        空间捷径，再计算 degree，减少弯折被误判为分叉。\n\n    返回\n    ----\n    skeleton_edges : ndarray, shape (N, 7)\n        列格式：x,y,z,r,g,b,edge_id。\n\n        edge_id == 0：分叉点区域，即计算得到的 degree > 2 的点。\n        edge_id >= 1：去掉分叉点后得到的各个连通骨架边。\n\n    说明\n    ----\n    这一版只做拓扑边切分：\n        1. 根据 refined skeleton 的 xyz 建立半径邻接图；\n        2. 删除可由更近中间点替代的空间捷径；\n        3. 计算每个点的 degree；\n        4. 将 degree > 2 的点作为分叉点；\n        5. 暂时移除分叉点；\n        6. 剩余图的每个连通分量作为一条边。\n\n    分叉点统一保存为红色、edge_id=0；不同边使用不同颜色。\n    当前函数不进行 GNN 分类，也不判断某条边是否为错误粘连边。\n    """\n    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 3:\n        raise ValueError(\n            "refined_skeleton must be a 2D array with at least 3 columns."\n        )\n\n    xyz = refined_skeleton[:, :3].astype(np.float64, copy=False)\n    point_count = len(xyz)\n\n    if point_count == 0:\n        return np.empty((0, 7), dtype=np.float64)\n\n    # 使用 KD-tree 查找 neighbor_radius 内的邻居。\n    # query_pairs 不包含点自身，因此 degree 就是邻接表长度。\n    tree = cKDTree(xyz)\n    neighbor_pairs = tree.query_pairs(\n        r=neighbor_radius,\n        output_type="ndarray",\n    )\n\n    candidate_adjacency = [set() for _ in range(point_count)]\n    for point_a, point_b in neighbor_pairs:\n        point_a = int(point_a)\n        point_b = int(point_b)\n        candidate_adjacency[point_a].add(point_b)\n        candidate_adjacency[point_b].add(point_a)\n\n    adjacency = [[] for _ in range(point_count)]\n    kept_pair_count = 0\n\n    for point_a, point_b in neighbor_pairs:\n        point_a = int(point_a)\n        point_b = int(point_b)\n        keep_pair = True\n\n        if prune_shortcut_edges:\n            distance_ab = float(np.linalg.norm(xyz[point_a] - xyz[point_b]))\n            common_neighbors = (\n                candidate_adjacency[point_a]\n                & candidate_adjacency[point_b]\n            )\n\n            # Relative-neighborhood pruning: 如果存在点 k，它到 a和b\n            # 都比 a-b 更近，则 a-b 是跨过中间点的捷径。\n            for point_k in common_neighbors:\n                distance_ak = float(\n                    np.linalg.norm(xyz[point_a] - xyz[point_k])\n                )\n                distance_bk = float(\n                    np.linalg.norm(xyz[point_b] - xyz[point_k])\n                )\n                if (\n                    distance_ak < distance_ab - 1e-8\n                    and distance_bk < distance_ab - 1e-8\n                ):\n                    keep_pair = False\n                    break\n\n        if keep_pair:\n            adjacency[point_a].append(point_b)\n            adjacency[point_b].append(point_a)\n            kept_pair_count += 1\n\n    degree = np.asarray(\n        [len(neighbors) for neighbors in adjacency],\n        dtype=np.int32,\n    )\n\n    junction_mask = degree > 2\n    edge_id = np.full(point_count, -1, dtype=np.int32)\n    edge_id[junction_mask] = 0\n\n    # 去掉分叉点后，对剩余骨架点进行连通分量搜索。\n    current_edge_id = 0\n\n    for start_index in range(point_count):\n        if edge_id[start_index] != -1:\n            continue\n\n        current_edge_id += 1\n        edge_id[start_index] = current_edge_id\n        stack = [start_index]\n\n        while stack:\n            current_index = stack.pop()\n\n            for neighbor_index in adjacency[current_index]:\n                if junction_mask[neighbor_index]:\n                    continue\n\n                if edge_id[neighbor_index] == -1:\n                    edge_id[neighbor_index] = current_edge_id\n                    stack.append(neighbor_index)\n\n    # 为不同 edge_id 生成稳定、容易区分的颜色。\n    rgb = np.zeros((point_count, 3), dtype=np.int32)\n    rgb[edge_id == 0] = [255, 0, 0]\n\n    golden_ratio = 0.618033988749895\n    hue_offset = (random_seed * golden_ratio) % 1.0\n\n    for one_edge_id in range(1, current_edge_id + 1):\n        hue = (hue_offset + one_edge_id * golden_ratio) % 1.0\n        color = colorsys.hsv_to_rgb(hue, 0.75, 1.0)\n        color = np.rint(np.asarray(color) * 255).astype(np.int32)\n        rgb[edge_id == one_edge_id] = color\n\n    skeleton_edges = np.column_stack(\n        [\n            xyz,\n            rgb,\n            edge_id,\n        ]\n    )\n\n    print(f"Skeleton points       : {point_count}")\n    print(f"Neighbor radius       : {neighbor_radius}")\n    print(f"Candidate graph edges : {len(neighbor_pairs)}")\n    print(f"Pruned graph edges    : {kept_pair_count}")\n    print(f"Endpoint points       : {np.sum(degree == 1)}")\n    print(f"Degree-2 points       : {np.sum(degree == 2)}")\n    print(f"Junction points       : {np.sum(junction_mask)}")\n    print(f"Segmented edges       : {current_edge_id}")\n\n    return skeleton_edges\n\n\ndef _safe_numeric_features(values, prefix):\n    """计算一组一维数值的稳定统计量。"""\n    values = np.asarray(values, dtype=np.float64)\n    values = values[np.isfinite(values)]\n\n    feature_names = [\n        "mean",\n        "median",\n        "std",\n        "cv",\n        "min",\n        "q10",\n        "q90",\n    ]\n\n    if len(values) == 0:\n        return {\n            f"{prefix}_{name}": np.nan\n            for name in feature_names\n        }\n\n    mean_value = float(np.mean(values))\n    median_value = float(np.median(values))\n    std_value = float(np.std(values))\n\n    return {\n        f"{prefix}_mean": mean_value,\n        f"{prefix}_median": median_value,\n        f"{prefix}_std": std_value,\n        f"{prefix}_cv": std_value / (abs(mean_value) + 1e-8),\n        f"{prefix}_min": float(np.min(values)),\n        f"{prefix}_q10": float(np.quantile(values, 0.10)),\n        f"{prefix}_q90": float(np.quantile(values, 0.90)),\n    }\n\n\ndef _indices_grouped_by_positive_label(labels):\n    """将正整数label对应的数组下标分组。"""\n    labels = np.asarray(labels, dtype=np.int32)\n    valid_indices = np.where(labels > 0)[0]\n\n    if len(valid_indices) == 0:\n        return {}\n\n    order = valid_indices[np.argsort(labels[valid_indices], kind="stable")]\n    ordered_labels = labels[order]\n    unique_labels, starts, counts = np.unique(\n        ordered_labels,\n        return_index=True,\n        return_counts=True,\n    )\n\n    return {\n        int(label): order[start:start + count]\n        for label, start, count in zip(unique_labels, starts, counts)\n    }\n\n\ndef _compute_edge_geometry(edge_xyz, neighbor_radius=1.8):\n    """\n    使用局部邻接图的最小生成树估计边长，并提取主路径、曲率和主轴。\n    """\n    edge_xyz = np.asarray(edge_xyz, dtype=np.float64)\n    point_count = len(edge_xyz)\n\n    if point_count == 0:\n        raise ValueError("edge_xyz cannot be empty.")\n\n    if point_count == 1:\n        return {\n            "ordered_local_indices": np.asarray([0], dtype=np.int32),\n            "arc_length": 0.0,\n            "chord_length": 0.0,\n            "tortuosity": 1.0,\n            "straightness": 1.0,\n            "turn_angle_mean": 0.0,\n            "turn_angle_std": 0.0,\n            "turn_angle_max": 0.0,\n            "pca_linearity": 0.0,\n            "principal_axis": np.asarray([1.0, 0.0, 0.0]),\n            "normalized_position": np.asarray([0.5]),\n        }\n\n    tree = cKDTree(edge_xyz)\n    pairs = tree.query_pairs(\n        r=neighbor_radius,\n        output_type="ndarray",\n    )\n\n    if len(pairs) == 0:\n        # 极少数不连续短边：使用最近邻顺序作为保底。\n        remaining = set(range(1, point_count))\n        ordered = [0]\n\n        while remaining:\n            current = ordered[-1]\n            next_index = min(\n                remaining,\n                key=lambda idx: np.linalg.norm(\n                    edge_xyz[idx] - edge_xyz[current]\n                ),\n            )\n            ordered.append(next_index)\n            remaining.remove(next_index)\n\n        ordered = np.asarray(ordered, dtype=np.int32)\n        segment_lengths = np.linalg.norm(\n            np.diff(edge_xyz[ordered], axis=0),\n            axis=1,\n        )\n        arc_length = float(np.sum(segment_lengths))\n    else:\n        pair_distances = np.linalg.norm(\n            edge_xyz[pairs[:, 0]] - edge_xyz[pairs[:, 1]],\n            axis=1,\n        )\n        graph = csr_matrix(\n            (\n                np.concatenate([pair_distances, pair_distances]),\n                (\n                    np.concatenate([pairs[:, 0], pairs[:, 1]]),\n                    np.concatenate([pairs[:, 1], pairs[:, 0]]),\n                ),\n            ),\n            shape=(point_count, point_count),\n        )\n        mst = minimum_spanning_tree(graph).tocoo()\n        arc_length = float(np.sum(mst.data))\n\n        adjacency = [[] for _ in range(point_count)]\n        for point_a, point_b, weight in zip(mst.row, mst.col, mst.data):\n            point_a = int(point_a)\n            point_b = int(point_b)\n            weight = float(weight)\n            adjacency[point_a].append((point_b, weight))\n            adjacency[point_b].append((point_a, weight))\n\n        def farthest_from(start_index):\n            distances = np.full(point_count, -np.inf, dtype=np.float64)\n            parents = np.full(point_count, -1, dtype=np.int32)\n            distances[start_index] = 0.0\n            stack = [start_index]\n\n            while stack:\n                current = stack.pop()\n                for neighbor, weight in adjacency[current]:\n                    if neighbor == parents[current]:\n                        continue\n                    parents[neighbor] = current\n                    distances[neighbor] = distances[current] + weight\n                    stack.append(neighbor)\n\n            farthest = int(np.nanargmax(distances))\n            return farthest, distances, parents\n\n        first_endpoint, _, _ = farthest_from(0)\n        second_endpoint, _, parents = farthest_from(first_endpoint)\n\n        path = [second_endpoint]\n        while path[-1] != first_endpoint and parents[path[-1]] != -1:\n            path.append(int(parents[path[-1]]))\n        ordered = np.asarray(path[::-1], dtype=np.int32)\n\n    ordered_xyz = edge_xyz[ordered]\n    chord_length = float(\n        np.linalg.norm(ordered_xyz[-1] - ordered_xyz[0])\n    )\n    tortuosity = arc_length / (chord_length + 1e-8)\n    straightness = chord_length / (arc_length + 1e-8)\n\n    turn_angles = []\n    if len(ordered_xyz) >= 3:\n        vector_a = ordered_xyz[1:-1] - ordered_xyz[:-2]\n        vector_b = ordered_xyz[2:] - ordered_xyz[1:-1]\n        norm_a = np.linalg.norm(vector_a, axis=1)\n        norm_b = np.linalg.norm(vector_b, axis=1)\n        valid = (norm_a > 1e-8) & (norm_b > 1e-8)\n\n        if np.any(valid):\n            cosine = np.sum(vector_a[valid] * vector_b[valid], axis=1)\n            cosine = cosine / (norm_a[valid] * norm_b[valid])\n            cosine = np.clip(cosine, -1.0, 1.0)\n            turn_angles = np.degrees(np.arccos(cosine))\n\n    if len(turn_angles) == 0:\n        turn_angle_mean = 0.0\n        turn_angle_std = 0.0\n        turn_angle_max = 0.0\n    else:\n        turn_angle_mean = float(np.mean(turn_angles))\n        turn_angle_std = float(np.std(turn_angles))\n        turn_angle_max = float(np.max(turn_angles))\n\n    centered = edge_xyz - edge_xyz.mean(axis=0, keepdims=True)\n    covariance = centered.T @ centered / max(point_count - 1, 1)\n    eigenvalues, eigenvectors = np.linalg.eigh(covariance)\n    order_eigen = np.argsort(eigenvalues)[::-1]\n    eigenvalues = np.maximum(eigenvalues[order_eigen], 0.0)\n    principal_axis = eigenvectors[:, order_eigen[0]]\n    principal_axis = principal_axis / (\n        np.linalg.norm(principal_axis) + 1e-8\n    )\n    pca_linearity = float(\n        (eigenvalues[0] - eigenvalues[1])\n        / (eigenvalues[0] + 1e-8)\n    )\n\n    projected_position = centered @ principal_axis\n    projected_min = float(np.min(projected_position))\n    projected_range = float(\n        np.max(projected_position) - projected_min\n    )\n\n    if projected_range <= 1e-8:\n        normalized_position = np.full(point_count, 0.5)\n    else:\n        normalized_position = (\n            projected_position - projected_min\n        ) / projected_range\n\n    return {\n        "ordered_local_indices": ordered,\n        "arc_length": arc_length,\n        "chord_length": chord_length,\n        "tortuosity": float(tortuosity),\n        "straightness": float(straightness),\n        "turn_angle_mean": turn_angle_mean,\n        "turn_angle_std": turn_angle_std,\n        "turn_angle_max": turn_angle_max,\n        "pca_linearity": pca_linearity,\n        "principal_axis": principal_axis,\n        "normalized_position": normalized_position,\n    }\n\n\ndef _extract_first_shell_cross_section(\n    center_xyz,\n    tangent,\n    candidate_global_indices,\n    pcd_xyz_all,\n    cross_section_half_thickness,\n    cross_section_max_radius,\n    angular_bins=36,\n    points_per_angle_bin=2,\n    radial_outlier_factor=1.75,\n):\n    """\n    从候选PCD中提取当前骨架中心在法平面内看到的第一层外壳。\n\n    每个角度分箱只保留径向距离最小的若干个点，并删除明显\n    远于主要外壳的角度分箱，以排除相邻线粒体表面。\n    """\n    candidate_global_indices = np.asarray(\n        candidate_global_indices,\n        dtype=np.int64,\n    )\n    if len(candidate_global_indices) == 0:\n        return None\n\n    tangent = np.asarray(tangent, dtype=np.float64)\n    tangent_norm = np.linalg.norm(tangent)\n    if tangent_norm <= 1e-8:\n        return None\n    tangent = tangent / tangent_norm\n\n    # 构造法平面内的两个正交基向量。\n    reference_axis = np.asarray([1.0, 0.0, 0.0])\n    if abs(float(np.dot(reference_axis, tangent))) > 0.90:\n        reference_axis = np.asarray([0.0, 1.0, 0.0])\n    plane_axis_u = np.cross(tangent, reference_axis)\n    plane_axis_u = plane_axis_u / (np.linalg.norm(plane_axis_u) + 1e-8)\n    plane_axis_v = np.cross(tangent, plane_axis_u)\n    plane_axis_v = plane_axis_v / (np.linalg.norm(plane_axis_v) + 1e-8)\n\n    candidate_xyz = pcd_xyz_all[candidate_global_indices]\n    relative_xyz = candidate_xyz - center_xyz\n    axial_distance = relative_xyz @ tangent\n    radial_vectors = (\n        relative_xyz - axial_distance[:, None] * tangent[None, :]\n    )\n    radial_distance = np.linalg.norm(radial_vectors, axis=1)\n\n    slab_mask = (\n        (np.abs(axial_distance) <= cross_section_half_thickness)\n        & (radial_distance <= cross_section_max_radius)\n        & (radial_distance > 1e-8)\n    )\n    if not np.any(slab_mask):\n        return None\n\n    slab_global_indices = candidate_global_indices[slab_mask]\n    slab_radial_vectors = radial_vectors[slab_mask]\n    slab_radial_distance = radial_distance[slab_mask]\n    slab_axial_distance = axial_distance[slab_mask]\n\n    coordinate_u = slab_radial_vectors @ plane_axis_u\n    coordinate_v = slab_radial_vectors @ plane_axis_v\n    angle = np.mod(np.arctan2(coordinate_v, coordinate_u), 2.0 * np.pi)\n    angle_bin = np.floor(\n        angle / (2.0 * np.pi) * angular_bins\n    ).astype(np.int32)\n    angle_bin = np.clip(angle_bin, 0, angular_bins - 1)\n\n    selected_local_indices = []\n    nearest_radius_by_bin = []\n    selected_by_bin = {}\n\n    for one_bin in np.unique(angle_bin):\n        one_bin_indices = np.where(angle_bin == one_bin)[0]\n        order = one_bin_indices[\n            np.argsort(slab_radial_distance[one_bin_indices])\n        ]\n        retained = order[:points_per_angle_bin]\n        selected_by_bin[int(one_bin)] = retained\n        nearest_radius_by_bin.append(float(slab_radial_distance[order[0]]))\n\n    if len(nearest_radius_by_bin) == 0:\n        return None\n\n    typical_first_shell_radius = float(np.median(nearest_radius_by_bin))\n    maximum_first_shell_radius = (\n        typical_first_shell_radius * radial_outlier_factor\n    )\n\n    retained_bin_count = 0\n    for retained in selected_by_bin.values():\n        nearest_radius = float(slab_radial_distance[retained[0]])\n        if nearest_radius <= maximum_first_shell_radius + 1e-8:\n            selected_local_indices.extend(retained.tolist())\n            retained_bin_count += 1\n\n    if len(selected_local_indices) == 0:\n        return None\n\n    selected_local_indices = np.asarray(\n        selected_local_indices,\n        dtype=np.int64,\n    )\n    selected_global_indices = slab_global_indices[selected_local_indices]\n    selected_radial_distance = slab_radial_distance[selected_local_indices]\n\n    # 保存时投影回零厚度法平面。\n    selected_candidate_xyz = pcd_xyz_all[selected_global_indices]\n    selected_axial_distance = slab_axial_distance[selected_local_indices]\n    projected_xyz = (\n        selected_candidate_xyz\n        - selected_axial_distance[:, None] * tangent[None, :]\n    )\n\n    return {\n        "global_indices": selected_global_indices,\n        "projected_xyz": projected_xyz,\n        "radial_distance": selected_radial_distance,\n        "occupied_bin_count": retained_bin_count,\n        "angular_coverage": retained_bin_count / float(angular_bins),\n        "typical_first_shell_radius": typical_first_shell_radius,\n    }\n\n\ndef _signed_offsets_from_pcd_surface(query_xyz, pcd, pcd_tree=None):\n    """\n    利用最近 PCD 表面点及其法向，判断 query 点位于表面内侧还是外侧。\n\n    返回的 signed_offset > 0 表示位于 PCD 外侧，< 0 表示位于内侧。\n    PCD 法向如果整体朝内，会根据绝大多数 skeleton 点应在内部这一事实\n    自动整体翻转。该函数只服务于后置黄色 candidate 判断，不参与前面\n    的截面尺度计算或红色 edge 判定。\n    """\n    query_xyz = np.asarray(query_xyz, dtype=np.float64)\n    pcd = np.asarray(pcd, dtype=np.float64)\n\n    if pcd.ndim != 2 or pcd.shape[1] < 6:\n        raise ValueError(\n            "The outside-PCD edge check requires PCD columns "\n            "x,y,z,nx,ny,nz."\n        )\n\n    pcd_xyz = pcd[:, :3]\n    pcd_normals = pcd[:, 3:6].copy()\n    normal_length = np.linalg.norm(pcd_normals, axis=1)\n    valid_normal = np.isfinite(normal_length) & (normal_length > 1e-8)\n    if not np.any(valid_normal):\n        raise ValueError(\n            "The outside-PCD edge check found no valid PCD surface normals."\n        )\n\n    pcd_normals[valid_normal] /= normal_length[valid_normal, None]\n    pcd_normals[~valid_normal] = np.nan\n\n    if pcd_tree is None:\n        pcd_tree = cKDTree(pcd_xyz)\n    _, nearest_pcd_rows = pcd_tree.query(\n        query_xyz,\n        k=1,\n        workers=-1,\n    )\n\n    nearest_surface_xyz = pcd_xyz[nearest_pcd_rows]\n    nearest_surface_normal = pcd_normals[nearest_pcd_rows]\n    signed_offset = np.einsum(\n        "ij,ij->i",\n        query_xyz - nearest_surface_xyz,\n        nearest_surface_normal,\n    )\n\n    finite_offset = np.isfinite(signed_offset)\n    if not np.any(finite_offset):\n        raise ValueError(\n            "The outside-PCD edge check could not calculate signed offsets."\n        )\n\n    # outward normal: 大多数内部骨架点的 signed offset 应为负。\n    # 如果中位数为正，说明输入法向整体朝内，统一翻转符号。\n    normals_were_flipped = bool(\n        np.median(signed_offset[finite_offset]) > 0.0\n    )\n    if normals_were_flipped:\n        signed_offset = -signed_offset\n\n    return signed_offset, normals_were_flipped\n\n\ndef detect_edges_with_abrupt_cross_section_narrowing(\n    skeleton_edges,\n    pcd,\n    save_path,\n    section_points_save_path=None,\n    deletion_points_save_path=None,\n    neighbor_radius=1.8,\n    tangent_window=2,\n    cross_section_half_thickness=1.0,\n    cross_section_max_radius=8.0,\n    min_section_points=1,\n    section_radius_quantile=0.75,\n    smoothing_window=1,\n    endpoint_margin=4,\n    side_window=4,\n    guard_sections=0,\n    min_side_sections=1,\n    narrow_ratio_threshold=0.65,\n    visualization_section_stride=4,\n    angular_bins=36,\n    points_per_angle_bin=2,\n    radial_outlier_factor=1.75,\n    min_owned_angular_coverage=0.20,\n    short_junction_edge_max_arc_length=8.0,\n    junction_attachment_radius=None,\n    local_cut_padding_points=0,\n    local_cut_recovery_ratio=0.85,\n    local_cut_recovery_stable_sections=2,\n    local_cut_recovery_reference_quantile=0.75,\n    short_single_junction_max_points=2,\n):\n    """\n    沿每条骨架 edge 的法平面切 PCD 外壳，寻找相对左右两侧急剧缩小的截面。\n\n    判定逻辑\n    --------\n    1. 使用 MST 主路径排列每条 edge 的 skeleton 点；\n    2. 在每个骨架点用前后邻域估计局部切向量；\n    3. 以切向量为法向建立薄截面，从属于该 edge 的 PCD 外壳点中\n       提取该截面；\n    4. 使用截面内 PCD 点到骨架中心的平面径向距离 q75 作为截面尺度；\n    5. 排除 edge 两端 endpoint_margin 个截面；它们仍保存为\n       蓝色可视化截面，但不参与窄颈判定。\n    6. 对每个内部截面计算：\n           local_ratio = 当前截面 / min(左侧中位数, 右侧中位数)\n       guard_sections=0 时，紧邻当前位置的左右截面直接参与计算。\n    7. 取整条 edge 中最小的 local_ratio。只有当它不大于\n       narrow_ratio_threshold 时，才认为出现“两侧较粗、中间急降”的窄颈。\n\n    输出格式\n    --------\n    x,y,z,r,g,b\n\n    颜色\n    ----\n    普通 edge       : 蓝色 [0, 102, 255]\n    疑似粘连假 edge : 红色 [255, 0, 0]\n    分叉点 edge_id=0: 绿色 [0, 255, 0]\n\n    截面点云\n    ----------\n    如果传入 section_points_save_path，另外保存 x,y,z,r,g,b：\n    普通采样截面 : 蓝色 [0, 102, 255]\n    疑似窄颈的局部急降截面 : 红色 [255, 0, 0]\n\n    建议删除点可视化\n    ----------------\n    如果传入 deletion_points_save_path，另外保存 x,y,z,r,g,b：\n    正常 edge                         : 蓝色 [0, 102, 255]\n    检测到急降的 edge               : 红色 [255, 0, 0]\n    建议删除的 skeleton 点            : 黄色 [255, 255, 0]\n    junction edge_id=0                 : 绿色 [0, 255, 0]\n    """\n    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)\n    pcd = np.asarray(pcd, dtype=np.float64)\n\n    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:\n        raise ValueError(\n            "skeleton_edges must contain x,y,z,r,g,b,edge_id."\n        )\n    if pcd.ndim != 2 or pcd.shape[1] < 3:\n        raise ValueError("pcd must contain at least x,y,z.")\n    if not 0.0 < narrow_ratio_threshold < 1.0:\n        raise ValueError("narrow_ratio_threshold must be between 0 and 1.")\n    if smoothing_window < 1 or smoothing_window % 2 == 0:\n        raise ValueError("smoothing_window must be a positive odd integer.")\n    if side_window < 1:\n        raise ValueError("side_window must be >= 1.")\n    if guard_sections < 0:\n        raise ValueError("guard_sections must be >= 0.")\n    if min_side_sections < 1:\n        raise ValueError("min_side_sections must be >= 1.")\n    if angular_bins < 8:\n        raise ValueError("angular_bins must be >= 8.")\n    if points_per_angle_bin < 1:\n        raise ValueError("points_per_angle_bin must be >= 1.")\n    if radial_outlier_factor <= 1.0:\n        raise ValueError("radial_outlier_factor must be > 1.")\n    if not 0.0 <= min_owned_angular_coverage <= 1.0:\n        raise ValueError(\n            "min_owned_angular_coverage must be between 0 and 1."\n        )\n    if short_junction_edge_max_arc_length <= 0:\n        raise ValueError(\n            "short_junction_edge_max_arc_length must be > 0."\n        )\n    if local_cut_padding_points < 0:\n        raise ValueError("local_cut_padding_points must be >= 0.")\n    if not 0.0 < local_cut_recovery_ratio <= 1.0:\n        raise ValueError(\n            "local_cut_recovery_ratio must be between 0 and 1."\n        )\n    if local_cut_recovery_stable_sections < 1:\n        raise ValueError(\n            "local_cut_recovery_stable_sections must be >= 1."\n        )\n    if not 0.5 <= local_cut_recovery_reference_quantile <= 1.0:\n        raise ValueError(\n            "local_cut_recovery_reference_quantile must be between 0.5 and 1."\n        )\n    if (\n        short_single_junction_max_points < 1\n        or int(short_single_junction_max_points)\n        != short_single_junction_max_points\n    ):\n        raise ValueError(\n            "short_single_junction_max_points must be a positive integer."\n        )\n    short_single_junction_max_points = int(\n        short_single_junction_max_points\n    )\n    if junction_attachment_radius is None:\n        junction_attachment_radius = neighbor_radius\n    if junction_attachment_radius <= 0:\n        raise ValueError("junction_attachment_radius must be > 0.")\n    if visualization_section_stride < 1:\n        raise ValueError("visualization_section_stride must be >= 1.")\n\n    edge_xyz_all = skeleton_edges[:, :3]\n    pcd_xyz_all = pcd[:, :3]\n    edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int32)\n    positive_rows = np.where(edge_ids > 0)[0]\n\n    if len(positive_rows) == 0:\n        raise ValueError("No edge_id >= 1 was found.")\n\n    edge_row_groups = _indices_grouped_by_positive_label(edge_ids)\n    junction_xyz = edge_xyz_all[edge_ids == 0]\n    junction_tree = cKDTree(junction_xyz) if len(junction_xyz) > 0 else None\n    if len(junction_xyz) > 0:\n        # 一个实际 junction 区域可能包含多个相邻的 edge_id=0 点。\n        # 先把这些点聚成独立 junction 区域，避免单点 edge 因为靠近\n        # 同一 junction 的多个点而被误认为连接了多个 junction。\n        junction_region_labels = DBSCAN(\n            eps=junction_attachment_radius,\n            min_samples=1,\n        ).fit_predict(junction_xyz)\n    else:\n        junction_region_labels = np.empty(0, dtype=np.int32)\n\n    # 第一级过滤：每个 PCD 点归属于最近的非分叉 skeleton edge。\n    # 截面优先使用当前 edge 自己的 PCD，减少相邻线粒体混入。\n    positive_tree = cKDTree(edge_xyz_all[positive_rows])\n    _, nearest_positive_local = positive_tree.query(\n        pcd_xyz_all,\n        k=1,\n        workers=-1,\n    )\n    nearest_positive_rows = positive_rows[nearest_positive_local]\n    pcd_owner_edge = edge_ids[nearest_positive_rows]\n    owned_pcd_groups = _indices_grouped_by_positive_label(pcd_owner_edge)\n\n    # 全局 PCD 只在当前 edge 所属 PCD 的角度覆盖不足时用作后备。\n    pcd_tree = cKDTree(pcd_xyz_all)\n\n    # 独立的后置 candidate 条件所需：判断每个 skeleton 点是否位于\n    # PCD 表面的外侧。它不参与截面计算，也不会改变红色 edge 集合。\n    skeleton_signed_surface_offset, pcd_normals_were_flipped = (\n        _signed_offsets_from_pcd_surface(\n            query_xyz=edge_xyz_all,\n            pcd=pcd,\n            pcd_tree=pcd_tree,\n        )\n    )\n    skeleton_outside_pcd = (\n        np.isfinite(skeleton_signed_surface_offset)\n        & (skeleton_signed_surface_offset > 1e-6)\n    )\n\n    suspected_edge_ids = set()\n    diagnostic_rows = []\n    visualization_blue_sections = []\n    visualization_red_sections = []\n    deletion_row_indices = set()\n    whole_edge_deletion_ids = set()\n    local_cut_deletion_ids = set()\n    outside_pcd_whole_edge_deletion_ids = set()\n    short_one_junction_edge_deletion_ids = set()\n    analyzed_edge_count = 0\n    insufficient_edge_count = 0\n    fallback_section_total = 0\n    smoothing_half_window = smoothing_window // 2\n\n    required_path_points = 1\n\n    for edge_id in sorted(edge_row_groups):\n        rows = edge_row_groups[edge_id]\n        geometry = _compute_edge_geometry(\n            edge_xyz_all[rows],\n            neighbor_radius=neighbor_radius,\n        )\n        ordered_rows = rows[geometry["ordered_local_indices"]]\n        ordered_xyz = edge_xyz_all[ordered_rows]\n        if junction_tree is None:\n            start_is_junction = False\n            end_is_junction = False\n        else:\n            start_is_junction = bool(\n                junction_tree.query(ordered_xyz[0], k=1)[0]\n                <= junction_attachment_radius\n            )\n            end_is_junction = bool(\n                junction_tree.query(ordered_xyz[-1], k=1)[0]\n                <= junction_attachment_radius\n            )\n        if len(ordered_xyz) < required_path_points:\n            insufficient_edge_count += 1\n            diagnostic_rows.append(\n                {\n                    "edge_id": edge_id,\n                    "status": "insufficient_data",\n                    "path_point_count": len(ordered_xyz),\n                    "assigned_pcd_count": 0,\n                    "fallback_section_count": 0,\n                    "start_is_junction": int(start_is_junction),\n                    "end_is_junction": int(end_is_junction),\n                    "edge_arc_length": float(geometry["arc_length"]),\n                    "deletion_mode": "none",\n                    "delete_point_count": 0,\n                    "minimum_ratio": np.nan,\n                    "minimum_scale": np.nan,\n                    "maximum_scale": np.nan,\n                    "left_reference": np.nan,\n                    "right_reference": np.nan,\n                    "neck_x": np.nan,\n                    "neck_y": np.nan,\n                    "neck_z": np.nan,\n                }\n            )\n            continue\n\n        analyzed_edge_count += 1\n        section_scale = np.full(len(ordered_xyz), np.nan, dtype=np.float64)\n        section_global_pcd_indices = [None] * len(ordered_xyz)\n        section_projected_xyz = [None] * len(ordered_xyz)\n        edge_used_pcd_indices = set()\n        fallback_section_count = 0\n\n        owned_pcd_indices = owned_pcd_groups.get(\n            edge_id,\n            np.empty(0, dtype=np.int64),\n        )\n        if len(owned_pcd_indices) > 0:\n            owned_pcd_tree = cKDTree(pcd_xyz_all[owned_pcd_indices])\n        else:\n            owned_pcd_tree = None\n\n        search_radius = float(\n            np.hypot(\n                cross_section_max_radius,\n                cross_section_half_thickness,\n            )\n        )\n\n        for path_index, center_xyz in enumerate(ordered_xyz):\n            global_near_indices = pcd_tree.query_ball_point(\n                center_xyz,\n                r=search_radius,\n            )\n\n            if len(global_near_indices) < min_section_points:\n                continue\n\n            global_near_indices = np.asarray(\n                global_near_indices,\n                dtype=np.int64,\n            )\n\n            if len(ordered_xyz) >= 2:\n                left_index = max(0, path_index - tangent_window)\n                right_index = min(\n                    len(ordered_xyz) - 1,\n                    path_index + tangent_window,\n                )\n                tangent = (\n                    ordered_xyz[right_index] - ordered_xyz[left_index]\n                )\n            else:\n                # 单点 edge 没有骨架走向，用附近 PCD 点云的\n                # 第一主轴作为截面法向，保证它也会被计算。\n                local_surface_xyz = pcd_xyz_all[global_near_indices]\n                centered_surface = (\n                    local_surface_xyz\n                    - np.mean(local_surface_xyz, axis=0, keepdims=True)\n                )\n                if len(centered_surface) >= 2:\n                    _, _, vh = np.linalg.svd(\n                        centered_surface,\n                        full_matrices=False,\n                    )\n                    tangent = vh[0]\n                else:\n                    tangent = np.asarray([1.0, 0.0, 0.0])\n\n            tangent_norm = np.linalg.norm(tangent)\n            if tangent_norm <= 1e-8:\n                tangent = np.asarray([1.0, 0.0, 0.0])\n            else:\n                tangent = tangent / tangent_norm\n\n            owned_result = None\n            if owned_pcd_tree is not None:\n                owned_near_local = owned_pcd_tree.query_ball_point(\n                    center_xyz,\n                    r=search_radius,\n                )\n                if len(owned_near_local) > 0:\n                    owned_near_global = owned_pcd_indices[\n                        np.asarray(owned_near_local, dtype=np.int64)\n                    ]\n                    owned_result = _extract_first_shell_cross_section(\n                        center_xyz=center_xyz,\n                        tangent=tangent,\n                        candidate_global_indices=owned_near_global,\n                        pcd_xyz_all=pcd_xyz_all,\n                        cross_section_half_thickness=(\n                            cross_section_half_thickness\n                        ),\n                        cross_section_max_radius=cross_section_max_radius,\n                        angular_bins=angular_bins,\n                        points_per_angle_bin=points_per_angle_bin,\n                        radial_outlier_factor=radial_outlier_factor,\n                    )\n\n            use_fallback = (\n                owned_result is None\n                or len(owned_result["global_indices"]) < min_section_points\n                or owned_result["angular_coverage"]\n                < min_owned_angular_coverage\n            )\n\n            if use_fallback:\n                fallback_result = _extract_first_shell_cross_section(\n                    center_xyz=center_xyz,\n                    tangent=tangent,\n                    candidate_global_indices=global_near_indices,\n                    pcd_xyz_all=pcd_xyz_all,\n                    cross_section_half_thickness=cross_section_half_thickness,\n                    cross_section_max_radius=cross_section_max_radius,\n                    angular_bins=angular_bins,\n                    points_per_angle_bin=points_per_angle_bin,\n                    radial_outlier_factor=radial_outlier_factor,\n                )\n                if fallback_result is not None:\n                    section_result = fallback_result\n                    fallback_section_count += 1\n                else:\n                    section_result = owned_result\n            else:\n                section_result = owned_result\n\n            if (\n                section_result is None\n                or len(section_result["global_indices"])\n                < min_section_points\n            ):\n                continue\n\n            section_global_pcd_indices[path_index] = section_result[\n                "global_indices"\n            ]\n            section_projected_xyz[path_index] = section_result[\n                "projected_xyz"\n            ]\n            edge_used_pcd_indices.update(\n                section_global_pcd_indices[path_index].tolist()\n            )\n            section_scale[path_index] = float(\n                np.quantile(\n                    section_result["radial_distance"],\n                    section_radius_quantile,\n                )\n            )\n\n        # 小窗口滚动中位数，降低单个 PCD 点缺失造成的假急降。\n        fallback_section_total += fallback_section_count\n        smoothed_scale = np.full_like(section_scale, np.nan)\n        for path_index in range(len(section_scale)):\n            start = max(0, path_index - smoothing_half_window)\n            stop = min(\n                len(section_scale),\n                path_index + smoothing_half_window + 1,\n            )\n            local_values = section_scale[start:stop]\n            local_values = local_values[np.isfinite(local_values)]\n            if len(local_values) > 0:\n                smoothed_scale[path_index] = float(np.median(local_values))\n\n        # 短的假连接 edge 不能因固定 endpoint_margin 而完全失去\n        # 可检验位置。左右至少需要 min_side_sections，中间还需要\n        # guard_sections，在此基础上自适应缩小端点保护范围。\n        minimum_required_without_margin = (\n            2 * (guard_sections + min_side_sections) + 1\n        )\n        maximum_feasible_margin = max(\n            (\n                len(smoothed_scale)\n                - minimum_required_without_margin\n            ) // 2,\n            0,\n        )\n        effective_endpoint_margin = min(\n            endpoint_margin,\n            maximum_feasible_margin,\n        )\n        first_test_index = effective_endpoint_margin\n        last_test_index = len(smoothed_scale) - effective_endpoint_margin\n        candidate_necks = []\n\n        for path_index in range(first_test_index, last_test_index):\n            center_scale = smoothed_scale[path_index]\n            if (\n                not np.isfinite(center_scale)\n                or not np.isfinite(section_scale[path_index])\n                or section_projected_xyz[path_index] is None\n            ):\n                continue\n\n            # endpoint_margin 只禁止把端点本身标红；端点区域的\n            # 正常截面仍可作为左右参考，否则短 edge 没有足够上下文。\n            left_start = max(\n                0,\n                path_index - guard_sections - side_window,\n            )\n            left_stop = path_index - guard_sections\n            right_start = path_index + guard_sections + 1\n            right_stop = min(\n                len(smoothed_scale),\n                path_index + guard_sections + side_window + 1,\n            )\n\n            left_values = smoothed_scale[left_start:left_stop]\n            right_values = smoothed_scale[right_start:right_stop]\n            left_values = left_values[np.isfinite(left_values)]\n            right_values = right_values[np.isfinite(right_values)]\n\n            if (\n                len(left_values) < min_side_sections\n                or len(right_values) < min_side_sections\n            ):\n                continue\n\n            left_reference = float(np.median(left_values))\n            right_reference = float(np.median(right_values))\n            two_sided_reference = min(left_reference, right_reference)\n\n            if two_sided_reference <= 1e-8:\n                continue\n\n            local_ratio = float(center_scale / two_sided_reference)\n            candidate_necks.append(\n                (\n                    local_ratio,\n                    path_index,\n                    float(center_scale),\n                    left_reference,\n                    right_reference,\n                    float(\n                        np.quantile(\n                            left_values,\n                            local_cut_recovery_reference_quantile,\n                        )\n                    ),\n                    float(\n                        np.quantile(\n                            right_values,\n                            local_cut_recovery_reference_quantile,\n                        )\n                    ),\n                )\n            )\n\n        if candidate_necks:\n            candidate_necks.sort(key=lambda item: item[0])\n            (\n                minimum_ratio,\n                neck_path_index,\n                minimum_scale,\n                left_reference,\n                right_reference,\n                left_recovery_reference,\n                right_recovery_reference,\n            ) = candidate_necks[0]\n            maximum_scale = min(left_reference, right_reference)\n            neck_xyz = ordered_xyz[neck_path_index]\n\n            if minimum_ratio <= narrow_ratio_threshold:\n                suspected_edge_ids.add(edge_id)\n                status = "suspected_narrow_neck"\n                red_section_xyz = section_projected_xyz[neck_path_index]\n                visualization_red_sections.append(red_section_xyz)\n            else:\n                status = "normal"\n        else:\n            minimum_ratio = np.nan\n            minimum_scale = np.nan\n            maximum_scale = np.nan\n            left_reference = np.nan\n            right_reference = np.nan\n            left_recovery_reference = np.nan\n            right_recovery_reference = np.nan\n            neck_xyz = np.full(3, np.nan)\n            status = "insufficient_local_context"\n\n        deletion_mode = "none"\n        delete_path_indices = []\n        delete_start_path_index = np.nan\n        delete_end_path_index = np.nan\n        outside_pcd_point_count = int(\n            np.count_nonzero(skeleton_outside_pcd[ordered_rows])\n        )\n        finite_edge_offsets = skeleton_signed_surface_offset[ordered_rows]\n        finite_edge_offsets = finite_edge_offsets[\n            np.isfinite(finite_edge_offsets)\n        ]\n        maximum_signed_surface_offset = (\n            float(np.max(finite_edge_offsets))\n            if len(finite_edge_offsets) > 0\n            else np.nan\n        )\n        attached_junction_region_count = 0\n        if len(ordered_rows) == 1 and junction_tree is not None:\n            nearby_junction_rows = junction_tree.query_ball_point(\n                ordered_xyz[0],\n                r=junction_attachment_radius,\n            )\n            if len(nearby_junction_rows) > 0:\n                attached_junction_region_count = int(\n                    len(\n                        np.unique(\n                            junction_region_labels[\n                                np.asarray(\n                                    nearby_junction_rows,\n                                    dtype=np.int64,\n                                )\n                            ]\n                        )\n                    )\n                )\n        if len(ordered_rows) == 1:\n            attached_junction_side_count = attached_junction_region_count\n        else:\n            attached_junction_side_count = int(start_is_junction) + int(\n                end_is_junction\n            )\n\n        # 只处理当前已经被截面算法标红的 edge。\n        if status == "suspected_narrow_neck":\n            if len(ordered_rows) == 1:\n                both_ends_are_junctions = (\n                    attached_junction_region_count >= 2\n                )\n            else:\n                both_ends_are_junctions = (\n                    start_is_junction and end_is_junction\n                )\n            short_whole_edge = (\n                geometry["arc_length"]\n                <= short_junction_edge_max_arc_length\n            )\n\n            if both_ends_are_junctions and short_whole_edge:\n                # 短的 junction-junction 假连接：删整条正 edge，\n                # 但 edge_id=0 的共享 junction 点本来就不在 rows 中。\n                delete_path_indices = list(range(len(ordered_rows)))\n                delete_start_path_index = 0\n                delete_end_path_index = len(ordered_rows) - 1\n                deletion_mode = "whole_short_junction_junction_edge"\n                whole_edge_deletion_ids.add(edge_id)\n            else:\n                # 其余红色 edge：从最强急降点向左右扩展，直到截面\n                # 分别恢复至左、右正常参考尺寸的指定比例，并且连续\n                # local_cut_recovery_stable_sections 个截面都维持恢复。\n                stable_count = local_cut_recovery_stable_sections\n                left_recovery_threshold = (\n                    local_cut_recovery_ratio * left_recovery_reference\n                )\n                right_recovery_threshold = (\n                    local_cut_recovery_ratio * right_recovery_reference\n                )\n\n                left_recovery_near_index = None\n                for distance_from_neck in range(1, len(ordered_rows)):\n                    block_end = neck_path_index - distance_from_neck\n                    block_start = block_end - stable_count + 1\n                    if block_start < 0:\n                        break\n                    recovery_block = smoothed_scale[\n                        block_start:block_end + 1\n                    ]\n                    if (\n                        len(recovery_block) == stable_count\n                        and np.all(np.isfinite(recovery_block))\n                        and np.all(\n                            recovery_block >= left_recovery_threshold\n                        )\n                    ):\n                        # 这一组恢复截面保留，删除从它右侧开始。\n                        left_recovery_near_index = block_end\n                        break\n\n                right_recovery_near_index = None\n                for distance_from_neck in range(1, len(ordered_rows)):\n                    block_start = neck_path_index + distance_from_neck\n                    block_end = block_start + stable_count - 1\n                    if block_end >= len(ordered_rows):\n                        break\n                    recovery_block = smoothed_scale[\n                        block_start:block_end + 1\n                    ]\n                    if (\n                        len(recovery_block) == stable_count\n                        and np.all(np.isfinite(recovery_block))\n                        and np.all(\n                            recovery_block >= right_recovery_threshold\n                        )\n                    ):\n                        # 这一组恢复截面保留，删除在它左侧停止。\n                        right_recovery_near_index = block_start\n                        break\n\n                if left_recovery_near_index is None:\n                    delete_start = 0\n                else:\n                    delete_start = left_recovery_near_index + 1\n\n                if right_recovery_near_index is None:\n                    delete_end = len(ordered_rows) - 1\n                else:\n                    delete_end = right_recovery_near_index - 1\n\n                delete_start = max(\n                    0,\n                    delete_start - local_cut_padding_points,\n                )\n                delete_end = min(\n                    len(ordered_rows) - 1,\n                    delete_end + local_cut_padding_points,\n                )\n\n                # 如果删除后两侧骨架点仍在建图半径内，重新建图\n                # 时会再次连上。仅向能更快增大断口的一侧扩展。\n                while delete_start > 0 and delete_end < len(ordered_rows) - 1:\n                    left_remaining = delete_start - 1\n                    right_remaining = delete_end + 1\n                    current_gap = float(\n                        np.linalg.norm(\n                            ordered_xyz[left_remaining]\n                            - ordered_xyz[right_remaining]\n                        )\n                    )\n                    if current_gap > neighbor_radius:\n                        break\n\n                    gap_if_expand_left = -np.inf\n                    if left_remaining - 1 >= 0:\n                        gap_if_expand_left = float(\n                            np.linalg.norm(\n                                ordered_xyz[left_remaining - 1]\n                                - ordered_xyz[right_remaining]\n                            )\n                        )\n\n                    gap_if_expand_right = -np.inf\n                    if right_remaining + 1 < len(ordered_rows):\n                        gap_if_expand_right = float(\n                            np.linalg.norm(\n                                ordered_xyz[left_remaining]\n                                - ordered_xyz[right_remaining + 1]\n                            )\n                        )\n\n                    if gap_if_expand_left >= gap_if_expand_right:\n                        delete_start -= 1\n                    else:\n                        delete_end += 1\n\n                delete_path_indices = list(\n                    range(delete_start, delete_end + 1)\n                )\n                delete_start_path_index = delete_start\n                delete_end_path_index = delete_end\n                deletion_mode = "local_cut_until_section_recovery"\n                local_cut_deletion_ids.add(edge_id)\n\n            deletion_row_indices.update(\n                ordered_rows[\n                    np.asarray(delete_path_indices, dtype=np.int64)\n                ].tolist()\n            )\n\n        # 独立的后置黄色 candidate 条件：不改变前面的截面计算，也不把\n        # edge 加入红色集合。只检查未标红的 edge；如果其中至少一个\n        # skeleton 点位于 PCD 外侧、两端均为 junction，且弧长严格小于\n        # 10，则在 deletion candidate 文件中将整条 edge 标黄。\n        outside_pcd_whole_edge_candidate = (\n            status != "suspected_narrow_neck"\n            and outside_pcd_point_count > 0\n            and (\n                (\n                    len(ordered_rows) == 1\n                    and attached_junction_region_count >= 2\n                )\n                or (\n                    len(ordered_rows) > 1\n                    and start_is_junction\n                    and end_is_junction\n                )\n            )\n            and geometry["arc_length"] < 10.0\n        )\n        if outside_pcd_whole_edge_candidate:\n            delete_path_indices = list(range(len(ordered_rows)))\n            delete_start_path_index = 0\n            delete_end_path_index = len(ordered_rows) - 1\n            deletion_mode = (\n                "whole_nonred_junction_junction_edge_with_outside_pcd_point"\n            )\n            outside_pcd_whole_edge_deletion_ids.add(edge_id)\n            deletion_row_indices.update(ordered_rows.tolist())\n\n        # 统一的短 edge 点数规则。参数为2时，包含1或2个 skeleton 点\n        # 的 edge 都使用同一判断；改成3时则覆盖1、2、3点。\n        # 只有恰好一侧连接 junction 才整条标黄。单点 edge 没有两个\n        # 不同几何端点，因此使用它连接的独立 junction 区域数量；\n        # 多点 edge 使用首尾端点是否连接 junction。\n        short_one_junction_edge_candidate = (\n            len(ordered_rows) <= short_single_junction_max_points\n            and attached_junction_side_count == 1\n        )\n        if short_one_junction_edge_candidate:\n            delete_path_indices = list(range(len(ordered_rows)))\n            delete_start_path_index = 0\n            delete_end_path_index = len(ordered_rows) - 1\n            deletion_mode = "whole_short_one_junction_edge_by_point_count"\n            short_one_junction_edge_deletion_ids.add(edge_id)\n            deletion_row_indices.update(ordered_rows.tolist())\n\n        # 每隔若干个骨架点保存一个蓝色截面，留出空隙，\n        # 便于在三维点云中看出一片片法平面。\n        visualization_indices = set(\n            range(\n                0,\n                len(section_projected_xyz),\n                visualization_section_stride,\n            )\n        )\n        if len(section_projected_xyz) > 0:\n            visualization_indices.add(len(section_projected_xyz) - 1)\n\n        for path_index in sorted(visualization_indices):\n            if (\n                status == "suspected_narrow_neck"\n                and path_index == neck_path_index\n            ):\n                continue\n            one_section_xyz = section_projected_xyz[path_index]\n            if one_section_xyz is not None:\n                visualization_blue_sections.append(one_section_xyz)\n\n        diagnostic_rows.append(\n            {\n                "edge_id": edge_id,\n                "status": status,\n                "path_point_count": len(ordered_xyz),\n                "assigned_pcd_count": len(edge_used_pcd_indices),\n                "fallback_section_count": fallback_section_count,\n                "start_is_junction": int(start_is_junction),\n                "end_is_junction": int(end_is_junction),\n                "edge_arc_length": float(geometry["arc_length"]),\n                "deletion_mode": deletion_mode,\n                "delete_point_count": len(delete_path_indices),\n                "delete_start_path_index": delete_start_path_index,\n                "delete_end_path_index": delete_end_path_index,\n                "outside_pcd_point_count": outside_pcd_point_count,\n                "maximum_signed_surface_offset": (\n                    maximum_signed_surface_offset\n                ),\n                "attached_junction_region_count": (\n                    attached_junction_region_count\n                ),\n                "attached_junction_side_count": (\n                    attached_junction_side_count\n                ),\n                "minimum_ratio": minimum_ratio,\n                "minimum_scale": minimum_scale,\n                "maximum_scale": maximum_scale,\n                "left_reference": left_reference,\n                "right_reference": right_reference,\n                "left_recovery_reference": left_recovery_reference,\n                "right_recovery_reference": right_recovery_reference,\n                "neck_x": neck_xyz[0],\n                "neck_y": neck_xyz[1],\n                "neck_z": neck_xyz[2],\n            }\n        )\n\n    output = np.zeros((len(skeleton_edges), 6), dtype=np.float64)\n    output[:, :3] = edge_xyz_all\n    output[:, 3:6] = [0, 102, 255]\n    output[edge_ids == 0, 3:6] = [0, 255, 0]\n\n    if suspected_edge_ids:\n        suspected_mask = np.isin(\n            edge_ids,\n            np.asarray(sorted(suspected_edge_ids), dtype=np.int32),\n        )\n        output[suspected_mask, 3:6] = [255, 0, 0]\n\n    np.savetxt(\n        save_path,\n        output,\n        delimiter=",",\n        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],\n    )\n\n    if deletion_points_save_path is not None:\n        deletion_output = output.copy()\n        if deletion_row_indices:\n            deletion_rows_array = np.asarray(\n                sorted(deletion_row_indices),\n                dtype=np.int64,\n            )\n            deletion_output[deletion_rows_array, 3:6] = [255, 255, 0]\n\n        np.savetxt(\n            deletion_points_save_path,\n            deletion_output,\n            delimiter=",",\n            fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],\n        )\n\n    if section_points_save_path is not None:\n        if visualization_blue_sections:\n            blue_xyz = np.vstack(visualization_blue_sections)\n        else:\n            blue_xyz = np.empty((0, 3), dtype=np.float64)\n        if visualization_red_sections:\n            red_xyz = np.vstack(visualization_red_sections)\n        else:\n            red_xyz = np.empty((0, 3), dtype=np.float64)\n\n        section_output_parts = []\n        if len(blue_xyz) > 0:\n            blue_output = np.zeros((len(blue_xyz), 6), dtype=np.float64)\n            blue_output[:, :3] = blue_xyz\n            blue_output[:, 3:6] = [0, 102, 255]\n            section_output_parts.append(blue_output)\n        if len(red_xyz) > 0:\n            red_output = np.zeros((len(red_xyz), 6), dtype=np.float64)\n            red_output[:, :3] = red_xyz\n            red_output[:, 3:6] = [255, 0, 0]\n            section_output_parts.append(red_output)\n\n        if section_output_parts:\n            section_output = np.vstack(section_output_parts)\n        else:\n            section_output = np.empty((0, 6), dtype=np.float64)\n\n        np.savetxt(\n            section_points_save_path,\n            section_output,\n            delimiter=",",\n            fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],\n        )\n\n    diagnostic_table = pd.DataFrame(diagnostic_rows)\n\n    print(f"Cross-section edges total      : {len(edge_row_groups)}")\n    print(f"Cross-section edges analyzed   : {analyzed_edge_count}")\n    print(f"Edges with insufficient data   : {insufficient_edge_count}")\n    print(f"Sections using global fallback : {fallback_section_total}")\n    print(f"Suspected narrow-neck edges    : {len(suspected_edge_ids)}")\n    print(f"Whole-edge deletion candidates : {len(whole_edge_deletion_ids)}")\n    print(f"Local-cut deletion candidates  : {len(local_cut_deletion_ids)}")\n    print(\n        "Outside-PCD whole-edge candidates: "\n        f"{len(outside_pcd_whole_edge_deletion_ids)}"\n    )\n    print(\n        "Short one-junction edge candidates: "\n        f"{len(short_one_junction_edge_deletion_ids)}"\n    )\n    print(\n        "Short-edge maximum skeleton points: "\n        f"{short_single_junction_max_points}"\n    )\n    print(f"Skeleton points marked yellow  : {len(deletion_row_indices)}")\n    print(f"Local-drop ratio threshold     : {narrow_ratio_threshold}")\n    print(f"PCD normals auto-flipped       : {pcd_normals_were_flipped}")\n\n    if outside_pcd_whole_edge_deletion_ids:\n        outside_candidate_rows = diagnostic_table[\n            diagnostic_table["deletion_mode"]\n            == "whole_nonred_junction_junction_edge_with_outside_pcd_point"\n        ].sort_values("edge_arc_length")\n        for _, row in outside_candidate_rows.iterrows():\n            print(\n                f"    outside-PCD edge_id={int(row[\'edge_id\'])} | "\n                f"arc_length={row[\'edge_arc_length\']:.3f} | "\n                f"outside_points={int(row[\'outside_pcd_point_count\'])} | "\n                f"max_signed_offset="\n                f"{row[\'maximum_signed_surface_offset\']:.3f}"\n            )\n\n    if short_one_junction_edge_deletion_ids:\n        short_one_junction_rows = diagnostic_table[\n            diagnostic_table["deletion_mode"]\n            == "whole_short_one_junction_edge_by_point_count"\n        ].sort_values(["path_point_count", "edge_id"])\n        for _, row in short_one_junction_rows.iterrows():\n            print(\n                f"    short edge_id={int(row[\'edge_id\'])} | "\n                f"skeleton_points={int(row[\'path_point_count\'])} | "\n                f"junction_sides="\n                f"{int(row[\'attached_junction_side_count\'])}"\n            )\n\n    if suspected_edge_ids:\n        suspected_rows = diagnostic_table[\n            diagnostic_table["status"] == "suspected_narrow_neck"\n        ].sort_values("minimum_ratio")\n        for _, row in suspected_rows.iterrows():\n            print(\n                f"    edge_id={int(row[\'edge_id\'])} | "\n                f"ratio={row[\'minimum_ratio\']:.3f} | "\n                f"neck=({row[\'neck_x\']:.2f},"\n                f"{row[\'neck_y\']:.2f},{row[\'neck_z\']:.2f})"\n            )\n\n    print("Cross-section edge result saved:")\n    print(f"    {save_path}")\n    print("Cross-section edge format: x,y,z,r,g,b")\n    print("Blue=ordinary edge, red=suspected narrow-neck edge, green=junction")\n\n    if deletion_points_save_path is not None:\n        print("Deletion-candidate skeleton visualization saved:")\n        print(f"    {deletion_points_save_path}")\n        print("Deletion visualization format: x,y,z,r,g,b")\n        print(\n            "Blue=normal, red=suspected edge, yellow=delete candidate, "\n            "green=junction"\n        )\n\n    if section_points_save_path is not None:\n        print("Cross-section PCD visualization saved:")\n        print(f"    {section_points_save_path}")\n        print("Cross-section PCD format: x,y,z,r,g,b")\n        print(\n            f"Blue section points={sum(map(len, visualization_blue_sections))}, "\n            f"red local-drop section points={sum(map(len, visualization_red_sections))}"\n        )\n\n    return (\n        output,\n        diagnostic_table,\n        np.asarray(sorted(deletion_row_indices), dtype=np.int64),\n    )\n\n\ndef rebuild_skeleton_edges_after_yellow_deletion(\n    skeleton_edges,\n    deletion_row_indices,\n    neighbor_radius=1.8,\n    random_seed=42,\n):\n    """\n    删除截面算法标黄的骨架点，并基于剩余点重新判断 junction 和 edge。\n\n    参数\n    ----\n    skeleton_edges : ndarray, shape (N, >=7)\n        删除前的 x,y,z,r,g,b,edge_id 骨架边点云。\n\n    deletion_row_indices : array-like\n        ``detect_edges_with_abrupt_cross_section_narrowing`` 返回的黄色点行号。\n\n    返回\n    ----\n    final_skeleton_edges : ndarray, shape (M, 7)\n        黄色点删除后重新分割的最终 x,y,z,r,g,b,edge_id。\n\n    说明\n    ----\n    不保留旧的 edge_id 或旧 junction 标签。黄色点删除后，直接使用所有\n    剩余 xyz 重新建邻接图、重新计算 degree、重新识别 junction，并重新\n    编号 edge。因此，原先由错误粘连造成的假分叉会随连接被切断而消失。\n    """\n    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)\n    deletion_row_indices = np.asarray(\n        deletion_row_indices,\n        dtype=np.int64,\n    ).reshape(-1)\n\n    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 3:\n        raise ValueError(\n            "skeleton_edges must be a 2D array with at least 3 columns."\n        )\n\n    point_count = len(skeleton_edges)\n    if len(deletion_row_indices) > 0:\n        deletion_row_indices = np.unique(deletion_row_indices)\n        if (\n            deletion_row_indices[0] < 0\n            or deletion_row_indices[-1] >= point_count\n        ):\n            raise IndexError(\n                "deletion_row_indices contains an out-of-range row index."\n            )\n\n    keep_mask = np.ones(point_count, dtype=bool)\n    keep_mask[deletion_row_indices] = False\n    remaining_xyz = skeleton_edges[keep_mask, :3]\n\n    final_skeleton_edges = segment_refined_skeleton_edges(\n        refined_skeleton=remaining_xyz,\n        neighbor_radius=neighbor_radius,\n        random_seed=random_seed,\n    )\n\n    print("Final edge topology rebuilt after yellow-point deletion:")\n    print(f"    Skeleton points before : {point_count}")\n    print(f"    Yellow points deleted  : {len(deletion_row_indices)}")\n    print(f"    Skeleton points after  : {len(final_skeleton_edges)}")\n\n    return final_skeleton_edges\n\n\ndef _assign_spatially_separated_instance_colors(\n    xyz,\n    instance_id,\n    random_seed=42,\n):\n    """让空间邻近的 instance 优先获得色相距离尽可能大的颜色。"""\n    xyz = np.asarray(xyz, dtype=np.float64)\n    instance_id = np.asarray(instance_id, dtype=np.int32)\n    unique_ids = np.unique(instance_id[instance_id > 0])\n    instance_count = len(unique_ids)\n    rgb = np.zeros((len(xyz), 3), dtype=np.int32)\n\n    if instance_count == 0:\n        return rgb, np.nan\n\n    id_to_position = {\n        int(one_id): position\n        for position, one_id in enumerate(unique_ids)\n    }\n    spatial_neighbors = [set() for _ in range(instance_count)]\n\n    # 质心近邻保证体积较小或点数很少的 instance 也有颜色约束。\n    centroids = np.vstack([\n        np.mean(xyz[instance_id == one_id], axis=0)\n        for one_id in unique_ids\n    ])\n    if instance_count > 1:\n        centroid_tree = cKDTree(centroids)\n        centroid_k = min(9, instance_count)\n        _, centroid_neighbor_rows = centroid_tree.query(\n            centroids,\n            k=centroid_k,\n            workers=-1,\n        )\n        if centroid_neighbor_rows.ndim == 1:\n            centroid_neighbor_rows = centroid_neighbor_rows[:, None]\n        for position, neighbor_positions in enumerate(\n            centroid_neighbor_rows\n        ):\n            for neighbor_position in neighbor_positions:\n                neighbor_position = int(neighbor_position)\n                if neighbor_position == position:\n                    continue\n                spatial_neighbors[position].add(neighbor_position)\n                spatial_neighbors[neighbor_position].add(position)\n\n    # 再从骨架点边界寻找近邻，避免两个细长 instance 的质心较远，\n    # 但局部末端实际上紧挨在一起时被分到相近颜色。\n    if len(xyz) > 1 and instance_count > 1:\n        point_tree = cKDTree(xyz)\n        point_k = min(32, len(xyz))\n        _, point_neighbor_rows = point_tree.query(\n            xyz,\n            k=point_k,\n            workers=-1,\n        )\n        if point_neighbor_rows.ndim == 1:\n            point_neighbor_rows = point_neighbor_rows[:, None]\n\n        for point_row, neighbor_rows in enumerate(point_neighbor_rows):\n            source_id = int(instance_id[point_row])\n            source_position = id_to_position[source_id]\n            nearby_other_ids = []\n            for neighbor_row in neighbor_rows:\n                neighbor_id = int(instance_id[int(neighbor_row)])\n                if (\n                    neighbor_id != source_id\n                    and neighbor_id not in nearby_other_ids\n                ):\n                    nearby_other_ids.append(neighbor_id)\n                    if len(nearby_other_ids) >= 3:\n                        break\n            for neighbor_id in nearby_other_ids:\n                neighbor_position = id_to_position[neighbor_id]\n                spatial_neighbors[source_position].add(neighbor_position)\n                spatial_neighbors[neighbor_position].add(source_position)\n\n    # 高度饱和的候选色相；颜色只要求邻近 instance 互相远离，\n    # 因而远处 instance 可以安全复用颜色。\n    golden_ratio = 0.618033988749895\n    hue_offset = (random_seed * golden_ratio) % 1.0\n    candidate_hues = (\n        hue_offset + np.arange(64, dtype=np.float64) / 64.0\n    ) % 1.0\n    candidate_use_count = np.zeros(len(candidate_hues), dtype=np.int32)\n    assigned_candidate = np.full(instance_count, -1, dtype=np.int32)\n\n    # 先处理空间邻居最多的 instance，让最拥挤区域优先获得最大色差。\n    coloring_order = sorted(\n        range(instance_count),\n        key=lambda position: (\n            -len(spatial_neighbors[position]),\n            int(unique_ids[position]),\n        ),\n    )\n\n    for position in coloring_order:\n        colored_neighbors = [\n            assigned_candidate[neighbor_position]\n            for neighbor_position in spatial_neighbors[position]\n            if assigned_candidate[neighbor_position] >= 0\n        ]\n\n        if len(colored_neighbors) == 0:\n            best_candidate = int(np.argmin(candidate_use_count))\n        else:\n            neighbor_hues = candidate_hues[\n                np.asarray(colored_neighbors, dtype=np.int32)\n            ]\n            hue_difference = np.abs(\n                candidate_hues[:, None] - neighbor_hues[None, :]\n            )\n            circular_difference = np.minimum(\n                hue_difference,\n                1.0 - hue_difference,\n            )\n            minimum_neighbor_difference = np.min(\n                circular_difference,\n                axis=1,\n            )\n            best_score = np.max(minimum_neighbor_difference)\n            best_candidates = np.where(\n                np.isclose(minimum_neighbor_difference, best_score)\n            )[0]\n            best_candidate = int(\n                best_candidates[\n                    np.argmin(candidate_use_count[best_candidates])\n                ]\n            )\n\n        assigned_candidate[position] = best_candidate\n        candidate_use_count[best_candidate] += 1\n\n    assigned_hues = candidate_hues[assigned_candidate]\n    for position, one_instance_id in enumerate(unique_ids):\n        color = colorsys.hsv_to_rgb(\n            float(assigned_hues[position]),\n            0.90,\n            1.0,\n        )\n        color = np.rint(np.asarray(color) * 255).astype(np.int32)\n        rgb[instance_id == one_instance_id] = color\n\n    neighbor_hue_distances = []\n    for position, neighbors in enumerate(spatial_neighbors):\n        for neighbor_position in neighbors:\n            if neighbor_position <= position:\n                continue\n            difference = abs(\n                assigned_hues[position]\n                - assigned_hues[neighbor_position]\n            )\n            neighbor_hue_distances.append(min(difference, 1.0 - difference))\n    minimum_neighbor_hue_distance = (\n        float(np.min(neighbor_hue_distances))\n        if neighbor_hue_distances\n        else np.nan\n    )\n\n    return rgb, minimum_neighbor_hue_distance\n\n\ndef segment_skeleton_social_instances(\n    skeleton_edges,\n    neighbor_radius=1.8,\n    random_seed=42,\n    prune_shortcut_edges=True,\n):\n    """\n    将最终 physical edges 按完整骨架连通性合并为 social instances。\n\n    与 physical edge 分割不同，这里不会移除 junction，也不会把\n    junction 当作 edge 边界。所有能够沿骨架经过 junction 相互到达的\n    点都属于同一个 social instance；不连通的骨架属于不同 instance。\n\n    返回格式：x,y,z,r,g,b,instance_id，其中 instance_id 从1开始。\n    """\n    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)\n    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 3:\n        raise ValueError(\n            "skeleton_edges must be a 2D array with at least 3 columns."\n        )\n\n    xyz = skeleton_edges[:, :3]\n    point_count = len(xyz)\n    if point_count == 0:\n        return np.empty((0, 7), dtype=np.float64)\n\n    tree = cKDTree(xyz)\n    neighbor_pairs = tree.query_pairs(\n        r=neighbor_radius,\n        output_type="ndarray",\n    )\n\n    candidate_adjacency = [set() for _ in range(point_count)]\n    for point_a, point_b in neighbor_pairs:\n        point_a = int(point_a)\n        point_b = int(point_b)\n        candidate_adjacency[point_a].add(point_b)\n        candidate_adjacency[point_b].add(point_a)\n\n    adjacency = [[] for _ in range(point_count)]\n    kept_pair_count = 0\n    for point_a, point_b in neighbor_pairs:\n        point_a = int(point_a)\n        point_b = int(point_b)\n        keep_pair = True\n\n        if prune_shortcut_edges:\n            distance_ab = float(np.linalg.norm(xyz[point_a] - xyz[point_b]))\n            common_neighbors = (\n                candidate_adjacency[point_a]\n                & candidate_adjacency[point_b]\n            )\n            for point_k in common_neighbors:\n                distance_ak = float(\n                    np.linalg.norm(xyz[point_a] - xyz[point_k])\n                )\n                distance_bk = float(\n                    np.linalg.norm(xyz[point_b] - xyz[point_k])\n                )\n                if (\n                    distance_ak < distance_ab - 1e-8\n                    and distance_bk < distance_ab - 1e-8\n                ):\n                    keep_pair = False\n                    break\n\n        if keep_pair:\n            adjacency[point_a].append(point_b)\n            adjacency[point_b].append(point_a)\n            kept_pair_count += 1\n\n    instance_id = np.zeros(point_count, dtype=np.int32)\n    current_instance_id = 0\n    for start_index in range(point_count):\n        if instance_id[start_index] != 0:\n            continue\n\n        current_instance_id += 1\n        instance_id[start_index] = current_instance_id\n        stack = [start_index]\n        while stack:\n            current_index = stack.pop()\n            for neighbor_index in adjacency[current_index]:\n                if instance_id[neighbor_index] == 0:\n                    instance_id[neighbor_index] = current_instance_id\n                    stack.append(neighbor_index)\n\n    rgb, minimum_neighbor_hue_distance = (\n        _assign_spatially_separated_instance_colors(\n            xyz=xyz,\n            instance_id=instance_id,\n            random_seed=random_seed,\n        )\n    )\n\n    social_instances = np.column_stack([xyz, rgb, instance_id])\n\n    print("Social skeleton instances generated:")\n    print(f"    Skeleton points       : {point_count}")\n    print(f"    Candidate graph edges : {len(neighbor_pairs)}")\n    print(f"    Pruned graph edges    : {kept_pair_count}")\n    print(f"    Social instances      : {current_instance_id}")\n    if np.isfinite(minimum_neighbor_hue_distance):\n        print(\n            "    Minimum neighbor hue gap: "\n            f"{minimum_neighbor_hue_distance:.3f}"\n        )\n\n    return social_instances\n\n\ndef _profile_features(values, positions, prefix, profile_bins=10):\n    """计算沿骨架方向分箱后的profile变化。"""\n    values = np.asarray(values, dtype=np.float64)\n    positions = np.asarray(positions, dtype=np.float64)\n    valid = np.isfinite(values) & np.isfinite(positions)\n    values = values[valid]\n    positions = np.clip(positions[valid], 0.0, 1.0)\n\n    output = {\n        f"{prefix}_bin_std": np.nan,\n        f"{prefix}_bin_min_median_ratio": np.nan,\n        f"{prefix}_profile_max_drop": np.nan,\n        f"{prefix}_endpoint_center_ratio": np.nan,\n        f"{prefix}_low_bin_fraction": np.nan,\n    }\n\n    if len(values) == 0:\n        return output\n\n    bin_indices = np.minimum(\n        (positions * profile_bins).astype(np.int32),\n        profile_bins - 1,\n    )\n    bin_means = np.full(profile_bins, np.nan, dtype=np.float64)\n\n    for bin_index in range(profile_bins):\n        one_bin = values[bin_indices == bin_index]\n        if len(one_bin) > 0:\n            bin_means[bin_index] = np.mean(one_bin)\n\n    valid_bin_means = bin_means[np.isfinite(bin_means)]\n    if len(valid_bin_means) == 0:\n        return output\n\n    median_bin = float(np.median(valid_bin_means))\n    output[f"{prefix}_bin_std"] = float(np.std(valid_bin_means))\n    output[f"{prefix}_bin_min_median_ratio"] = float(\n        np.min(valid_bin_means) / (abs(median_bin) + 1e-8)\n    )\n    output[f"{prefix}_low_bin_fraction"] = float(\n        np.mean(valid_bin_means < 0.5 * median_bin)\n    )\n\n    adjacent_valid = np.isfinite(bin_means[:-1]) & np.isfinite(bin_means[1:])\n    if np.any(adjacent_valid):\n        output[f"{prefix}_profile_max_drop"] = float(\n            np.max(\n                np.abs(np.diff(bin_means)[adjacent_valid])\n            ) / (abs(median_bin) + 1e-8)\n        )\n\n    endpoint_values = bin_means[[0, -1]]\n    endpoint_values = endpoint_values[np.isfinite(endpoint_values)]\n    center_values = bin_means[1:-1]\n    center_values = center_values[np.isfinite(center_values)]\n\n    if len(endpoint_values) > 0 and len(center_values) > 0:\n        output[f"{prefix}_endpoint_center_ratio"] = float(\n            np.mean(endpoint_values)\n            / (abs(np.median(center_values)) + 1e-8)\n        )\n\n    return output\n\n\ndef compute_skeleton_edge_features(\n    skeleton_edges,\n    refined_skeleton,\n    voxel,\n    pcd,\n    neighbor_radius=1.8,\n    voxel_assignment_max_distance=8.0,\n    pcd_assignment_max_distance=8.0,\n    junction_search_radius=2.5,\n    profile_bins=10,\n):\n    """\n    为每条 edge_id>=1 的骨架边计算几何、score、intensity、PCD和局部拓扑feature。\n    """\n    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)\n    refined_skeleton = np.asarray(refined_skeleton, dtype=np.float64)\n    voxel = np.asarray(voxel, dtype=np.float64)\n    pcd = np.asarray(pcd, dtype=np.float64)\n\n    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:\n        raise ValueError(\n            "skeleton_edges must contain x,y,z,r,g,b,edge_id."\n        )\n    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 4:\n        raise ValueError(\n            "refined_skeleton must contain x,y,z,score."\n        )\n\n    # 最终 skeleton_edges 已经删除黄色点，行数通常少于原始\n    # refined_skeleton。不能再假设二者逐行对应，而应按照 xyz 将每个\n    # 最终骨架点重新匹配回原始 refined skeleton，取回它的 score。\n    edge_xyz_for_alignment = skeleton_edges[:, :3]\n    refined_xyz_for_alignment = refined_skeleton[:, :3]\n    rows_already_aligned = (\n        len(skeleton_edges) == len(refined_skeleton)\n        and np.allclose(\n            edge_xyz_for_alignment,\n            refined_xyz_for_alignment,\n            atol=1e-3,\n            rtol=0.0,\n        )\n    )\n\n    if not rows_already_aligned:\n        refined_tree = cKDTree(refined_xyz_for_alignment)\n        alignment_distance, alignment_rows = refined_tree.query(\n            edge_xyz_for_alignment,\n            k=1,\n            workers=-1,\n        )\n        maximum_alignment_distance = float(np.max(alignment_distance))\n        if maximum_alignment_distance > 0.01:\n            raise ValueError(\n                "Final skeleton edge points could not be matched back to "\n                "refined_skeleton within 0.01 coordinate units. "\n                f"Maximum distance: {maximum_alignment_distance:.6f}."\n            )\n        refined_skeleton = refined_skeleton[alignment_rows]\n        print(\n            "Aligned refined-skeleton scores to final skeleton edges: "\n            f"{len(skeleton_edges)} rows, "\n            f"max distance={maximum_alignment_distance:.6f}"\n        )\n\n    edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int32)\n    edge_xyz_all = skeleton_edges[:, :3]\n    score_all = refined_skeleton[:, 3]\n    positive_rows = np.where(edge_ids > 0)[0]\n\n    if len(positive_rows) == 0:\n        raise ValueError("No edge_id >= 1 was found.")\n\n    unique_edge_ids = np.unique(edge_ids[positive_rows])\n    edge_row_groups = _indices_grouped_by_positive_label(edge_ids)\n\n    edge_cache = {}\n    principal_axis_by_row = np.full((len(edge_ids), 3), np.nan)\n    normalized_position_by_row = np.full(len(edge_ids), np.nan)\n\n    for edge_id in unique_edge_ids:\n        rows = edge_row_groups[int(edge_id)]\n        edge_xyz = edge_xyz_all[rows]\n        geometry = _compute_edge_geometry(\n            edge_xyz,\n            neighbor_radius=neighbor_radius,\n        )\n        ordered_rows = rows[geometry["ordered_local_indices"]]\n        edge_cache[int(edge_id)] = {\n            "rows": rows,\n            "xyz": edge_xyz,\n            "geometry": geometry,\n            "ordered_rows": ordered_rows,\n            "centroid": edge_xyz.mean(axis=0),\n            "start_xyz": edge_xyz_all[ordered_rows[0]],\n            "end_xyz": edge_xyz_all[ordered_rows[-1]],\n        }\n        principal_axis_by_row[rows] = geometry["principal_axis"]\n        normalized_position_by_row[rows] = geometry["normalized_position"]\n\n    # 将voxel和PCD分别分配给最近的非分叉骨架边。\n    positive_tree = cKDTree(edge_xyz_all[positive_rows])\n\n    voxel_distance, voxel_nearest_local = positive_tree.query(\n        voxel[:, :3],\n        k=1,\n        workers=-1,\n    )\n    voxel_nearest_rows = positive_rows[voxel_nearest_local]\n    voxel_assigned_edge = edge_ids[voxel_nearest_rows].copy()\n    voxel_assigned_edge[\n        voxel_distance > voxel_assignment_max_distance\n    ] = -1\n    voxel_groups = _indices_grouped_by_positive_label(voxel_assigned_edge)\n\n    pcd_distance, pcd_nearest_local = positive_tree.query(\n        pcd[:, :3],\n        k=1,\n        workers=-1,\n    )\n    pcd_nearest_rows = positive_rows[pcd_nearest_local]\n    pcd_assigned_edge = edge_ids[pcd_nearest_rows].copy()\n    pcd_assigned_edge[\n        pcd_distance > pcd_assignment_max_distance\n    ] = -1\n    pcd_groups = _indices_grouped_by_positive_label(pcd_assigned_edge)\n\n    junction_xyz = edge_xyz_all[edge_ids == 0]\n    junction_tree = cKDTree(junction_xyz) if len(junction_xyz) > 0 else None\n\n    def endpoint_context(edge_id, endpoint_xyz):\n        if junction_tree is None:\n            junction_distance = np.nan\n        else:\n            junction_distance = float(junction_tree.query(endpoint_xyz, k=1)[0])\n\n        nearby_local_rows = positive_tree.query_ball_point(\n            endpoint_xyz,\n            r=junction_search_radius,\n        )\n        nearby_global_rows = positive_rows[\n            np.asarray(nearby_local_rows, dtype=np.int64)\n        ]\n        nearby_edge_ids = set(\n            int(value)\n            for value in edge_ids[nearby_global_rows]\n            if int(value) > 0 and int(value) != int(edge_id)\n        )\n\n        current_centroid = edge_cache[int(edge_id)]["centroid"]\n        current_vector = current_centroid - endpoint_xyz\n        current_norm = np.linalg.norm(current_vector)\n        continuation_angles = []\n\n        if current_norm > 1e-8:\n            current_vector = current_vector / current_norm\n            for neighbor_edge_id in nearby_edge_ids:\n                neighbor_vector = (\n                    edge_cache[neighbor_edge_id]["centroid"] - endpoint_xyz\n                )\n                neighbor_norm = np.linalg.norm(neighbor_vector)\n                if neighbor_norm <= 1e-8:\n                    continue\n                neighbor_vector = neighbor_vector / neighbor_norm\n                cosine = np.clip(\n                    np.dot(-current_vector, neighbor_vector),\n                    -1.0,\n                    1.0,\n                )\n                continuation_angles.append(\n                    float(np.degrees(np.arccos(cosine)))\n                )\n\n        best_angle = (\n            float(np.min(continuation_angles))\n            if continuation_angles\n            else np.nan\n        )\n\n        return {\n            "junction_distance": junction_distance,\n            "is_junction": int(\n                np.isfinite(junction_distance)\n                and junction_distance <= junction_search_radius\n            ),\n            "neighbor_edge_count": len(nearby_edge_ids),\n            "best_continuation_angle": best_angle,\n        }\n\n    feature_rows = []\n\n    for edge_id in unique_edge_ids:\n        edge_id = int(edge_id)\n        cache = edge_cache[edge_id]\n        rows = cache["rows"]\n        geometry = cache["geometry"]\n        edge_xyz = cache["xyz"]\n        ordered_rows = cache["ordered_rows"]\n        point_count = len(rows)\n\n        rgb = np.rint(skeleton_edges[rows[0], 3:6]).astype(np.int32)\n        one_feature = {\n            "edge_id": edge_id,\n            "color_r": int(rgb[0]),\n            "color_g": int(rgb[1]),\n            "color_b": int(rgb[2]),\n            "skeleton_point_count": point_count,\n            "is_single_point_edge": int(point_count == 1),\n            "is_short_edge": int(point_count < 3),\n            "edge_arc_length": geometry["arc_length"],\n            "edge_chord_length": geometry["chord_length"],\n            "tortuosity": geometry["tortuosity"],\n            "straightness": geometry["straightness"],\n            "turn_angle_mean": geometry["turn_angle_mean"],\n            "turn_angle_std": geometry["turn_angle_std"],\n            "turn_angle_max": geometry["turn_angle_max"],\n            "pca_linearity": geometry["pca_linearity"],\n        }\n\n        edge_scores = score_all[rows]\n        one_feature.update(_safe_numeric_features(edge_scores, "score"))\n        score_median = np.nanmedian(edge_scores)\n        one_feature["score_min_median_ratio"] = float(\n            np.nanmin(edge_scores) / (abs(score_median) + 1e-8)\n        )\n        one_feature.update(\n            _profile_features(\n                score_all[rows],\n                normalized_position_by_row[rows],\n                prefix="score",\n                profile_bins=profile_bins,\n            )\n        )\n\n        voxel_indices = voxel_groups.get(edge_id, np.empty(0, dtype=np.int64))\n        one_feature["assigned_voxel_count"] = len(voxel_indices)\n\n        if len(voxel_indices) > 0 and voxel.shape[1] >= 4:\n            intensity = voxel[voxel_indices, 3]\n            one_feature.update(\n                _safe_numeric_features(intensity, "intensity")\n            )\n            intensity_positions = normalized_position_by_row[\n                voxel_nearest_rows[voxel_indices]\n            ]\n            one_feature.update(\n                _profile_features(\n                    intensity,\n                    intensity_positions,\n                    prefix="intensity",\n                    profile_bins=profile_bins,\n                )\n            )\n        else:\n            one_feature.update(_safe_numeric_features([], "intensity"))\n            one_feature.update(\n                _profile_features([], [], "intensity", profile_bins)\n            )\n\n        pcd_indices = pcd_groups.get(edge_id, np.empty(0, dtype=np.int64))\n        one_feature["assigned_pcd_count"] = len(pcd_indices)\n        one_feature["surface_points_per_length"] = float(\n            len(pcd_indices) / (geometry["arc_length"] + 1e-8)\n        )\n\n        if len(pcd_indices) > 0:\n            one_feature.update(\n                _safe_numeric_features(\n                    pcd_distance[pcd_indices],\n                    "surface_distance",\n                )\n            )\n        else:\n            one_feature.update(\n                _safe_numeric_features([], "surface_distance")\n            )\n\n        one_feature["normal_tangent_abs_dot_mean"] = np.nan\n        one_feature["normal_tangent_abs_dot_std"] = np.nan\n        one_feature["normal_tangent_abs_dot_q90"] = np.nan\n        one_feature["normal_resultant_length"] = np.nan\n        one_feature["normal_dispersion"] = np.nan\n        one_feature["cross_section_ellipticity"] = np.nan\n\n        if len(pcd_indices) >= 3 and pcd.shape[1] >= 6:\n            normals = pcd[pcd_indices, 3:6]\n            normal_lengths = np.linalg.norm(normals, axis=1)\n            valid_normals = normal_lengths > 1e-8\n\n            if np.any(valid_normals):\n                normals = normals[valid_normals] / normal_lengths[\n                    valid_normals, None\n                ]\n                tangent = principal_axis_by_row[\n                    pcd_nearest_rows[pcd_indices][valid_normals]\n                ]\n                abs_dot = np.abs(np.sum(normals * tangent, axis=1))\n                one_feature["normal_tangent_abs_dot_mean"] = float(\n                    np.mean(abs_dot)\n                )\n                one_feature["normal_tangent_abs_dot_std"] = float(\n                    np.std(abs_dot)\n                )\n                one_feature["normal_tangent_abs_dot_q90"] = float(\n                    np.quantile(abs_dot, 0.90)\n                )\n                resultant = float(np.linalg.norm(np.mean(normals, axis=0)))\n                one_feature["normal_resultant_length"] = resultant\n                one_feature["normal_dispersion"] = 1.0 - resultant\n\n            axis = geometry["principal_axis"]\n            reference = np.asarray([1.0, 0.0, 0.0])\n            if abs(np.dot(axis, reference)) > 0.90:\n                reference = np.asarray([0.0, 1.0, 0.0])\n            basis_u = np.cross(axis, reference)\n            basis_u = basis_u / (np.linalg.norm(basis_u) + 1e-8)\n            basis_v = np.cross(axis, basis_u)\n            centered_surface = pcd[pcd_indices, :3] - cache["centroid"]\n            projected_surface = np.column_stack(\n                [\n                    centered_surface @ basis_u,\n                    centered_surface @ basis_v,\n                ]\n            )\n            cross_covariance = np.cov(projected_surface, rowvar=False)\n            cross_eigenvalues = np.linalg.eigvalsh(cross_covariance)\n            cross_eigenvalues = np.maximum(cross_eigenvalues, 0.0)\n            one_feature["cross_section_ellipticity"] = float(\n                (cross_eigenvalues[-1] - cross_eigenvalues[0])\n                / (cross_eigenvalues[-1] + cross_eigenvalues[0] + 1e-8)\n            )\n\n        start_context = endpoint_context(edge_id, cache["start_xyz"])\n        end_context = endpoint_context(edge_id, cache["end_xyz"])\n        one_feature["start_junction_distance"] = start_context[\n            "junction_distance"\n        ]\n        one_feature["end_junction_distance"] = end_context[\n            "junction_distance"\n        ]\n        one_feature["start_is_junction"] = start_context["is_junction"]\n        one_feature["end_is_junction"] = end_context["is_junction"]\n        one_feature["junction_junction_edge"] = int(\n            start_context["is_junction"] and end_context["is_junction"]\n        )\n        one_feature["endpoint_junction_edge"] = int(\n            start_context["is_junction"] != end_context["is_junction"]\n        )\n        one_feature["endpoint_endpoint_edge"] = int(\n            not start_context["is_junction"]\n            and not end_context["is_junction"]\n        )\n        one_feature["start_neighbor_edge_count"] = start_context[\n            "neighbor_edge_count"\n        ]\n        one_feature["end_neighbor_edge_count"] = end_context[\n            "neighbor_edge_count"\n        ]\n        one_feature["best_continuation_angle_start"] = start_context[\n            "best_continuation_angle"\n        ]\n        one_feature["best_continuation_angle_end"] = end_context[\n            "best_continuation_angle"\n        ]\n\n        one_feature["eligible_for_umap"] = int(point_count >= 3)\n        feature_rows.append(one_feature)\n\n    feature_table = pd.DataFrame(feature_rows).sort_values(\n        "edge_id"\n    ).reset_index(drop=True)\n\n    print(f"Feature rows           : {len(feature_table)}")\n    print(f"Feature columns        : {len(feature_table.columns)}")\n    print(\n        "Eligible for UMAP      : "\n        f"{int(feature_table[\'eligible_for_umap\'].sum())}"\n    )\n\n    return feature_table\n\n\ndef run_edge_umap_and_clustering(\n    feature_table,\n    save_umap_csv_path,\n    save_umap_png_path,\n    random_seed=42,\n    umap_n_neighbors=15,\n    umap_min_dist=0.10,\n    hdbscan_min_cluster_size=10,\n):\n    """对边feature做稳健标准化、无监督聚类和UMAP可视化。"""\n    try:\n        import umap\n    except ImportError as error:\n        raise ImportError(\n            "UMAP is not installed. Run: pip install umap-learn"\n        ) from error\n\n    metadata_columns = {\n        "edge_id",\n        "color_r",\n        "color_g",\n        "color_b",\n        "eligible_for_umap",\n    }\n    feature_columns = [\n        column\n        for column in feature_table.columns\n        if column not in metadata_columns\n        and pd.api.types.is_numeric_dtype(feature_table[column])\n    ]\n\n    eligible_mask = feature_table["eligible_for_umap"].astype(bool).to_numpy()\n    eligible_table = feature_table.loc[eligible_mask].copy()\n\n    if len(eligible_table) < 3:\n        raise ValueError("At least 3 eligible edges are required for UMAP.")\n\n    feature_matrix = eligible_table[feature_columns].replace(\n        [np.inf, -np.inf],\n        np.nan,\n    )\n\n    # 删除全缺失或没有变化的feature，避免影响UMAP。\n    usable_columns = []\n    for column in feature_columns:\n        values = feature_matrix[column]\n        if values.notna().sum() == 0:\n            continue\n        if values.nunique(dropna=True) <= 1:\n            continue\n        usable_columns.append(column)\n\n    feature_matrix = feature_matrix[usable_columns].copy()\n\n    log1p_columns = {\n        "skeleton_point_count",\n        "edge_arc_length",\n        "edge_chord_length",\n        "assigned_voxel_count",\n        "assigned_pcd_count",\n        "surface_points_per_length",\n    }\n\n    for column in usable_columns:\n        if column in log1p_columns:\n            feature_matrix[column] = np.log1p(\n                np.maximum(feature_matrix[column], 0.0)\n            )\n\n    imputer = SimpleImputer(strategy="median")\n    scaler = RobustScaler()\n    feature_matrix_imputed = imputer.fit_transform(feature_matrix)\n    feature_matrix_scaled = scaler.fit_transform(feature_matrix_imputed)\n\n    effective_neighbors = min(\n        umap_n_neighbors,\n        len(eligible_table) - 1,\n    )\n    reducer = umap.UMAP(\n        n_components=2,\n        n_neighbors=effective_neighbors,\n        min_dist=umap_min_dist,\n        metric="euclidean",\n        random_state=random_seed,\n    )\n    embedding = reducer.fit_transform(feature_matrix_scaled)\n\n    cluster_method = "HDBSCAN"\n    try:\n        from sklearn.cluster import HDBSCAN\n\n        clusterer = HDBSCAN(\n            min_cluster_size=min(\n                hdbscan_min_cluster_size,\n                max(3, len(eligible_table) // 3),\n            ),\n            min_samples=5,\n        )\n        cluster_labels = clusterer.fit_predict(feature_matrix_scaled)\n        cluster_probability = getattr(\n            clusterer,\n            "probabilities_",\n            np.full(len(eligible_table), np.nan),\n        )\n    except ImportError:\n        cluster_method = "DBSCAN fallback"\n        fallback_clusterer = DBSCAN(eps=1.5, min_samples=5)\n        cluster_labels = fallback_clusterer.fit_predict(embedding)\n        cluster_probability = np.full(len(eligible_table), np.nan)\n\n    result_table = feature_table.copy()\n    result_table["umap_x"] = np.nan\n    result_table["umap_y"] = np.nan\n    result_table["cluster"] = -99\n    result_table["cluster_probability"] = np.nan\n    result_table.loc[eligible_mask, "umap_x"] = embedding[:, 0]\n    result_table.loc[eligible_mask, "umap_y"] = embedding[:, 1]\n    result_table.loc[eligible_mask, "cluster"] = cluster_labels\n    result_table.loc[eligible_mask, "cluster_probability"] = cluster_probability\n    result_table.to_csv(save_umap_csv_path, index=False)\n\n    plt.figure(figsize=(8, 7))\n    unique_clusters = np.unique(cluster_labels)\n    color_map = plt.get_cmap("tab20")\n\n    for cluster_index, cluster_label in enumerate(unique_clusters):\n        cluster_mask = cluster_labels == cluster_label\n\n        if cluster_label == -1:\n            color = "#9e9e9e"\n            label = "Noise / outlier"\n        else:\n            color = color_map(cluster_index % 20)\n            label = f"Cluster {cluster_label}"\n\n        plt.scatter(\n            embedding[cluster_mask, 0],\n            embedding[cluster_mask, 1],\n            s=24,\n            alpha=0.80,\n            c=[color],\n            edgecolors="none",\n            label=label,\n        )\n\n    plt.xlabel("UMAP 1")\n    plt.ylabel("UMAP 2")\n    plt.title(\n        "Mitochondrial skeleton-edge features\\n"\n        f"{cluster_method}, n={len(eligible_table)} edges"\n    )\n    plt.legend(frameon=False, fontsize=8, loc="best")\n    plt.tight_layout()\n    plt.savefig(save_umap_png_path, dpi=300, bbox_inches="tight")\n    plt.close()\n\n    print(f"UMAP feature columns   : {len(usable_columns)}")\n    print(f"UMAP edges             : {len(eligible_table)}")\n    print(f"Clustering method      : {cluster_method}")\n    print(f"UMAP CSV saved         : {save_umap_csv_path}")\n    print(f"UMAP PNG saved         : {save_umap_png_path}")\n\n    return result_table\n\n\ndef save_clustered_skeleton_edges_xyzrgb(\n    skeleton_edges,\n    clustering_table,\n    save_path,\n):\n    """\n    将 edge 的无监督聚类标签映射回 skeleton 点，并保存为 x,y,z,r,g,b。\n\n    颜色规则：\n        cluster >= 0 : 每个 cluster 使用一种固定的高区分度颜色；\n        cluster == -1: HDBSCAN/DBSCAN noise，灰色；\n        cluster == -99: 不满足 UMAP 条件的短边，深灰色；\n        edge_id == 0 : 分叉点，红色。\n    """\n    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:\n        raise ValueError(\n            "skeleton_edges must contain x,y,z,r,g,b,edge_id."\n        )\n\n    required_columns = {"edge_id", "cluster"}\n    missing_columns = required_columns.difference(clustering_table.columns)\n    if missing_columns:\n        raise ValueError(\n            "Clustering table is missing columns: "\n            + ", ".join(sorted(missing_columns))\n        )\n\n    edge_to_cluster = {}\n    for _, row in clustering_table[["edge_id", "cluster"]].dropna().iterrows():\n        edge_id = int(round(float(row["edge_id"])))\n        cluster_label = int(round(float(row["cluster"])))\n        edge_to_cluster[edge_id] = cluster_label\n\n    valid_cluster_labels = sorted(\n        {\n            cluster_label\n            for cluster_label in edge_to_cluster.values()\n            if cluster_label >= 0\n        }\n    )\n\n    # 使用 HSV 色环生成确定性的高对比颜色，不受原 edge 随机颜色影响。\n    cluster_colors = {}\n    number_of_clusters = len(valid_cluster_labels)\n    for color_index, cluster_label in enumerate(valid_cluster_labels):\n        if number_of_clusters == 1:\n            color_position = 0.58\n        else:\n            # 整体错开纯红色，避免与红色分叉点混淆。\n            color_position = (\n                0.08 + color_index / number_of_clusters\n            ) % 1.0\n\n        rgba = plt.get_cmap("hsv")(color_position)\n        cluster_colors[cluster_label] = np.rint(\n            np.asarray(rgba[:3]) * 255.0\n        ).astype(np.int32)\n\n    output = np.zeros((len(skeleton_edges), 6), dtype=np.float64)\n    output[:, :3] = skeleton_edges[:, :3]\n\n    skeleton_edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int64)\n    output_rgb = np.zeros((len(skeleton_edges), 3), dtype=np.int32)\n\n    for row_index, edge_id in enumerate(skeleton_edge_ids):\n        if edge_id == 0:\n            output_rgb[row_index] = [255, 0, 0]\n            continue\n\n        cluster_label = edge_to_cluster.get(int(edge_id), -99)\n\n        if cluster_label >= 0:\n            output_rgb[row_index] = cluster_colors[cluster_label]\n        elif cluster_label == -1:\n            output_rgb[row_index] = [160, 160, 160]\n        else:\n            output_rgb[row_index] = [80, 80, 80]\n\n    output[:, 3:6] = output_rgb\n\n    np.savetxt(\n        save_path,\n        output,\n        delimiter=",",\n        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],\n    )\n\n    cluster_edge_counts = {\n        cluster_label: sum(\n            1\n            for value in edge_to_cluster.values()\n            if value == cluster_label\n        )\n        for cluster_label in valid_cluster_labels\n    }\n    noise_edge_count = sum(\n        1 for value in edge_to_cluster.values() if value == -1\n    )\n    ineligible_edge_count = sum(\n        1 for value in edge_to_cluster.values() if value == -99\n    )\n\n    print(f"Valid clusters          : {number_of_clusters}")\n    for cluster_label in valid_cluster_labels:\n        rgb = cluster_colors[cluster_label]\n        print(\n            f"    Cluster {cluster_label}: "\n            f"{cluster_edge_counts[cluster_label]} edges | "\n            f"RGB=({rgb[0]},{rgb[1]},{rgb[2]})"\n        )\n    print(f"Noise / outlier edges   : {noise_edge_count}")\n    print(f"Ineligible short edges  : {ineligible_edge_count}")\n    print("Clustered skeleton saved:")\n    print(f"    {save_path}")\n    print("Clustered skeleton format: x,y,z,r,g,b")\n\n    return output\n\n\n###############################################################################\n# 后续准备逐步添加的函数\n###############################################################################\n\n\ndef segment_mito_instances(voxel, pcd, skeleton, refined_skeleton):\n    """\n    TODO：精细线粒体实例分割。\n\n    以后再逐步加入：\n        1. 分离由于分辨率原因黏在一起的线粒体；\n        2. 分离平行、同向、相邻的线粒体；\n        3. 使用 voxel intensity、PCD normal 和 skeleton 信息；\n        4. 返回 voxel、PCD 和 skeleton 的实例分割结果。\n\n    第一版暂时不调用这个函数。\n    """\n    raise NotImplementedError("Mitochondrial instance segmentation is not written yet.")\n\n\ndef split_instance_skeleton_by_branch_points(skeleton_instance):\n    """\n    TODO：在线粒体实例内部，按照端点和分叉点切分骨架。\n\n    第一版暂时不调用这个函数。\n    """\n    raise NotImplementedError("Skeleton branch segmentation is not written yet.")\n\n\ndef main():\n\n    # file_name_list = [\n    #\n    #                         # ====================================================\n    #     # MCF-7\n    #     # # ====================================================\n    #     "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # AC16\n    #     # # ====================================================\n    #     "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # MDA\n    #     # # ====================================================\n    #     "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # Fibroblast\n    #     # # ====================================================\n    #     "20260207_fibro_mito488_lyso640_nuc405_tif",\n    #     # "20251216_fibro_mito488tracker_ld561bodipy_nuc405hurst",\n    #     #\n    #     # # ====================================================\n    #     # # H28\n    #     # # ====================================================\n    #     "20250917_h28_mitotracker488_lysotracker640_nuc405",\n    #     # "20251213_h28_mito488_ld561_nuc405",\n    #     #\n    #     # # ====================================================\n    #     # # HeLa\n    #     # # ====================================================\n    #     "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",\n    #     # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",\n    #\n    #     # ====================================================\n    #     # U2OS\n    #     # ====================================================\n    #     # "20251201_u2os_ld488_mito640_nuc405",\n    #     "20250909_u2os_mitotracker488_lysotracker640_nuc405",\n    #\n    #     # ====================================================\n    #     # MSTO\n    #     # ====================================================\n    #     # "20251127_msto_ld488_mito640_nuc405",\n    #     "20250905_msto_mitotracekr488_lysotracker640_nuc405",\n    #\n    #                 ]\n\n    file_name_list = [\n\n        # ====================================================\n        # GLUC\n        # ====================================================\n        # "gluc_mito-mChGFP_LTRDR50nM_0031",\n        # "gluc_mito-mChGFP_LTRDR50nM_0032",\n        # "gluc_mito-mChGFP_LTRDR50nM_0051",\n        # "gluc_mito-mChGFP_LTRDR50nM_0061",\n        # "gluc_mito-mChGFP_LTRDR50nM_0071",\n        # "gluc_mito-mChGFP_LTRDR50nM_0072",\n        # "gluc_mito-mChGFP_LTRDR50nM_0081",\n        # "gluc_mito-mChGFP_LTRDR50nM_0082",\n        # "gluc_mito-mChGFP_LTRDR50nM_0091",\n\n        # ====================================================\n        # Non-GLUC\n        # ====================================================\n\n        # "mito-mChGFP_LTRDR50nM_0021",\n        # "mito-mChGFP_LTRDR50nM_0022",\n        # "mito-mChGFP_LTRDR50nM_0031",\n        # "mito-mChGFP_LTRDR50nM_0032",\n        # "mito-mChGFP_LTRDR50nM_0041",\n        # "mito-mChGFP_LTRDR50nM_0042",\n        # "mito-mChGFP_LTRDR50nM_0051",\n        # "mito-mChGFP_LTRDR50nM_0061",\n        # "mito-mChGFP_LTRDR50nM_0062",\n        # "mito-mChGFP_LTRDR50nM_0071",\n        # "mito-mChGFP_LTRDR50nM_0072",\n        "mito-mChGFP_LTRDR50nM_0081",\n\n    ]\n\n    total_done = 0\n\n    for file_name in file_name_list:\n        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/{file_name}/mito")\n        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")\n        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")\n\n        INPUT_PATH = BASE_DIR / "processed_output"\n        OUTPUT_PATH = BASE_DIR / "processed_output"\n\n        if not os.path.exists(INPUT_PATH):\n            raise FileNotFoundError(\n                f"processed_output folder not found: {INPUT_PATH}"\n            )\n\n        if not os.path.exists(OUTPUT_PATH):\n            os.makedirs(OUTPUT_PATH)\n\n        print(os.listdir(INPUT_PATH))\n\n        #######################################################################\n        # 图片列表\n        #######################################################################\n\n        # 自动读取 processed_output 中的所有 *_pcd.txt。\n        imgs_list = [\n            f\n            for f in os.listdir(INPUT_PATH)\n            if f.lower().endswith("_pcd.txt")\n        ]\n\n        # 手动指定需要处理的图片。\n        # 需要批量处理全部图片时，将下面这一行注释掉即可。\n        # imgs_list = ["fibro_mito488_lyso640_nuc405_17689_5ADVMLE"]\n\n        if len(imgs_list) == 0:\n            print(f"No images selected in {INPUT_PATH}")\n            continue\n\n        print("========================================================================= ")\n        print(f"Dataset            : {file_name}")\n        print(f"BASE_DIR           : {BASE_DIR}")\n        print(f"INPUT_PATH         : {INPUT_PATH}")\n        print(f"OUTPUT_PATH        : {OUTPUT_PATH}")\n        print(f"Selected images    : {len(imgs_list)}")\n        print("========================================================================= ")\n\n        for i, img_name in enumerate(imgs_list, start=1):\n            # 同时兼容：\n            #     自动扫描得到的 xxx_pcd.txt\n            #     手动填写的 xxx\n            if img_name.lower().endswith("_pcd.txt"):\n                stem = img_name[:-len("_pcd.txt")]\n            else:\n                stem = img_name\n\n            pcd_path = INPUT_PATH / f"{stem}_pcd.txt"\n\n            print("-------------------------------------------------------------------------")\n            print(f"Processing {file_name} | {i}/{len(imgs_list)} | {stem}")\n            print("-------------------------------------------------------------------------")\n\n            ###################################################################\n            # 输入文件路径\n            ###################################################################\n\n            voxel_path = INPUT_PATH / f"{stem}_voxel.txt"\n            skeleton_path = find_skeleton_path(INPUT_PATH, stem)\n            refined_skeleton_path = find_refined_skeleton_path(INPUT_PATH, stem)\n\n            ###################################################################\n            # 将来要保存的文件路径\n            # 第一版只打印路径，不实际保存文件。\n            ###################################################################\n\n            save_pcd_instance_path = OUTPUT_PATH / f"{stem}_pcd_instance.txt"\n            save_skeleton_instance_path = OUTPUT_PATH / f"{stem}_skeleton_instance.txt"\n            save_voxel_instance_path = OUTPUT_PATH / f"{stem}_voxel_instance.txt"\n            save_skeleton_edges_path = OUTPUT_PATH / f"{stem}_skeleton_edges.txt"\n            save_skeleton_social_instance_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_instance_social.txt"\n            )\n            save_skeleton_social_version_path = (\n                OUTPUT_PATH\n                / f"{stem}_skeleton_instance_social_algorithm_version.txt"\n            )\n            save_original_skeleton_edges_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edges_original.txt"\n            )\n            save_skeleton_edges_version_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edges_algorithm_version.txt"\n            )\n            save_edge_features_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edge_features.csv"\n            )\n            save_edge_umap_csv_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edge_umap.csv"\n            )\n            save_edge_umap_png_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edge_umap.png"\n            )\n            save_clustered_skeleton_edges_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edges_clustered.txt"\n            )\n            save_cross_section_edges_path = (\n                OUTPUT_PATH / f"{stem}_skeleton_edges_cross_section.txt"\n            )\n            save_cross_section_points_path = (\n                OUTPUT_PATH\n                / f"{stem}_pcd_cross_sections_first_shell_local_drop_flat.txt"\n            )\n            save_deletion_points_path = (\n                OUTPUT_PATH\n                / f"{stem}_skeleton_edges_deletion_candidates_recovery.txt"\n            )\n\n            edge_algorithm_version = (\n                "cross_section_yellow_deletion_and_topology_rebuild_v10"\n            )\n            social_instance_algorithm_version = (\n                "connected_components_spatial_maximin_color_v1"\n            )\n            social_instance_is_current = False\n            if (\n                save_skeleton_social_instance_path.exists()\n                and save_skeleton_social_version_path.exists()\n            ):\n                try:\n                    saved_social_version = (\n                        save_skeleton_social_version_path.read_text(\n                            encoding="utf-8"\n                        ).strip()\n                    )\n                    social_instance_is_current = (\n                        saved_social_version\n                        == social_instance_algorithm_version\n                    )\n                except OSError:\n                    social_instance_is_current = False\n            edge_segmentation_is_current = False\n            if (\n                save_skeleton_edges_path.exists()\n                and save_skeleton_edges_version_path.exists()\n            ):\n                try:\n                    saved_version = (\n                        save_skeleton_edges_version_path.read_text(\n                            encoding="utf-8"\n                        ).strip()\n                    )\n                    edge_segmentation_is_current = (\n                        saved_version == edge_algorithm_version\n                    )\n                except OSError:\n                    edge_segmentation_is_current = False\n\n            edge_segmentation_rebuilt = False\n\n            # 正式 skeleton_edges 现在是删除黄色点并重建拓扑后的最终文件。\n            # 如果任一删除前检查文件缺失，不能直接在最终 edge 上重算截面；\n            # 必须从 refined skeleton 重新走完整流程，才能恢复原始候选连接。\n            cross_section_outputs_complete = (\n                save_original_skeleton_edges_path.exists()\n                and save_cross_section_edges_path.exists()\n                and save_cross_section_points_path.exists()\n                and save_deletion_points_path.exists()\n            )\n            if (\n                edge_segmentation_is_current\n                and not cross_section_outputs_complete\n            ):\n                print(\n                    "[REBUILD] A pre-deletion cross-section output is missing; "\n                    "the complete cleanup pipeline will be rerun."\n                )\n                edge_segmentation_is_current = False\n\n            ###################################################################\n            # 两个最终可视化文件都存在，表示当前流程已经完整完成。\n            ###################################################################\n\n            if (\n                edge_segmentation_is_current\n                and\n                save_clustered_skeleton_edges_path.exists()\n                and save_cross_section_edges_path.exists()\n                and save_cross_section_points_path.exists()\n                and save_deletion_points_path.exists()\n                and save_original_skeleton_edges_path.exists()\n                and social_instance_is_current\n            ):\n                print("[SKIPPED] Final visualization files already exist:")\n                print(f"    {save_clustered_skeleton_edges_path}")\n                print(f"    {save_cross_section_edges_path}")\n                print(f"    {save_cross_section_points_path}")\n                print(f"    {save_deletion_points_path}")\n                print(f"    {save_original_skeleton_edges_path}")\n                print(f"    {save_skeleton_social_instance_path}")\n                continue\n\n            ###################################################################\n            # 之前已经完成聚类，但还没有导出 xyzrgb：直接复用旧结果。\n            # 这里不会读取 PCD、voxel，也不会重新分 edge 或重新聚类。\n            ###################################################################\n\n            if (\n                edge_segmentation_is_current\n                and\n                not save_clustered_skeleton_edges_path.exists()\n                and\n                save_skeleton_edges_path.exists()\n                and save_edge_umap_csv_path.exists()\n            ):\n                print("[REUSED] Existing edges and clustering results found.")\n                print(f"    {save_skeleton_edges_path}")\n                print(f"    {save_edge_umap_csv_path}")\n\n                skeleton_edges = load_txt_array(save_skeleton_edges_path)\n                clustering_table = pd.read_csv(save_edge_umap_csv_path)\n\n                save_clustered_skeleton_edges_xyzrgb(\n                    skeleton_edges=skeleton_edges,\n                    clustering_table=clustering_table,\n                    save_path=save_clustered_skeleton_edges_path,\n                )\n\n            if (\n                edge_segmentation_is_current\n                and\n                save_clustered_skeleton_edges_path.exists()\n                and save_cross_section_edges_path.exists()\n                and save_cross_section_points_path.exists()\n                and save_deletion_points_path.exists()\n                and save_original_skeleton_edges_path.exists()\n                and social_instance_is_current\n            ):\n                total_done += 1\n                continue\n\n            ###################################################################\n            # 已有 edge 和聚类可视化、只缺截面结果时走快速分支。\n            # 只读取 PCD 与 skeleton_edges，不读取 voxel/raw/refined skeleton。\n            ###################################################################\n\n            if (\n                edge_segmentation_is_current\n                and\n                save_clustered_skeleton_edges_path.exists()\n                and save_skeleton_edges_path.exists()\n                and (\n                    not save_cross_section_edges_path.exists()\n                    or not save_cross_section_points_path.exists()\n                    or not save_deletion_points_path.exists()\n                )\n            ):\n                if not pcd_path.exists():\n                    print("[SKIPPED] Missing PCD file:")\n                    print(f"    {pcd_path}")\n                    continue\n\n                print("[FAST PATH] Only cross-section result is missing.")\n                print("Reading PCD and existing skeleton edges only.")\n                pcd = load_txt_array(pcd_path)\n                skeleton_edges = load_txt_array(save_skeleton_edges_path)\n\n                print_file_information(\n                    label="PCD",\n                    path=pcd_path,\n                    data=pcd,\n                    column_format="x,y,z,nx,ny,nz",\n                )\n                print_file_information(\n                    label="SKELETON EDGES",\n                    path=save_skeleton_edges_path,\n                    data=skeleton_edges,\n                    column_format="x,y,z,r,g,b,edge_id",\n                )\n\n                detect_edges_with_abrupt_cross_section_narrowing(\n                    skeleton_edges=skeleton_edges,\n                    pcd=pcd,\n                    save_path=save_cross_section_edges_path,\n                    section_points_save_path=save_cross_section_points_path,\n                    deletion_points_save_path=save_deletion_points_path,\n                    neighbor_radius=1.8,\n                    tangent_window=2,\n                    cross_section_half_thickness=1.0,\n                    cross_section_max_radius=8.0,\n                    min_section_points=1,\n                    section_radius_quantile=0.75,\n                    smoothing_window=1,\n                    endpoint_margin=4,\n                    side_window=4,\n                    guard_sections=0,\n                    min_side_sections=1,\n                    narrow_ratio_threshold=0.65,\n                    visualization_section_stride=4,\n                    angular_bins=36,\n                    points_per_angle_bin=2,\n                    radial_outlier_factor=1.75,\n                    min_owned_angular_coverage=0.20,\n                    short_junction_edge_max_arc_length=5.0,\n                    junction_attachment_radius=1.8,\n                    local_cut_padding_points=0,\n                    local_cut_recovery_ratio=0.85,\n                    local_cut_recovery_stable_sections=2,\n                    local_cut_recovery_reference_quantile=0.75,\n                    short_single_junction_max_points=2,\n                )\n\n                total_done += 1\n                continue\n\n            ###################################################################\n            # 检查四个输入文件是否存在\n            ###################################################################\n\n            required_input_paths = {\n                "PCD": pcd_path,\n                "VOXEL": voxel_path,\n                "SKELETON": skeleton_path,\n                "REFINED SKELETON": refined_skeleton_path,\n            }\n\n            missing_paths = [\n                path\n                for path in required_input_paths.values()\n                if not path.exists()\n            ]\n\n            if len(missing_paths) > 0:\n                print("[SKIPPED] Missing input files:")\n                for missing_path in missing_paths:\n                    print(f"    {missing_path}")\n                continue\n\n            ###################################################################\n            # 读取输入文件\n            ###################################################################\n\n            pcd = load_txt_array(pcd_path)\n            voxel = load_txt_array(voxel_path)\n            skeleton = load_txt_array(skeleton_path)\n            refined_skeleton = load_txt_array(refined_skeleton_path)\n\n            ###################################################################\n            # 打印路径、shape和列格式\n            ###################################################################\n\n            print_file_information(\n                label="PCD",\n                path=pcd_path,\n                data=pcd,\n                column_format="x,y,z,nx,ny,nz",\n            )\n            check_expected_columns("PCD", pcd, expected_columns=6)\n            print()\n\n            print_file_information(\n                label="VOXEL",\n                path=voxel_path,\n                data=voxel,\n                column_format="x,y,z,intensity",\n            )\n            check_expected_columns("VOXEL", voxel, expected_columns=4)\n            print()\n\n            print_file_information(\n                label="SKELETON",\n                path=skeleton_path,\n                data=skeleton,\n                column_format="x,y,z,radius,degree,type",\n            )\n            check_expected_columns("SKELETON", skeleton, expected_columns=6)\n            print()\n\n            print_file_information(\n                label="REFINED SKELETON",\n                path=refined_skeleton_path,\n                data=refined_skeleton,\n                column_format="x,y,z,score",\n            )\n            check_expected_columns(\n                "REFINED SKELETON",\n                refined_skeleton,\n                expected_columns=4,\n            )\n\n            ###################################################################\n            # 打印将来的输出文件路径\n            ###################################################################\n\n            print()\n            print("Planned instance output files — not saved yet:")\n            print(f"PCD instance      : {save_pcd_instance_path}")\n            print(f"Skeleton instance : {save_skeleton_instance_path}")\n            print(f"Voxel instance    : {save_voxel_instance_path}")\n            print(f"Edge features     : {save_edge_features_path}")\n            print(f"Edge UMAP CSV     : {save_edge_umap_csv_path}")\n            print(f"Edge UMAP PNG     : {save_edge_umap_png_path}")\n            print(f"Clustered skeleton: {save_clustered_skeleton_edges_path}")\n            print(f"Cross-section edge: {save_cross_section_edges_path}")\n            print(f"Original edges    : {save_original_skeleton_edges_path}")\n            print(f"Social instances  : {save_skeleton_social_instance_path}")\n\n            ###################################################################\n            # 如果 edge 文件已经存在，直接读取，不重复补点和分 edge。\n            ###################################################################\n\n            if (\n                save_skeleton_edges_path.exists()\n                and edge_segmentation_is_current\n            ):\n                print()\n                print("[REUSED] Existing skeleton edges found:")\n                print(f"    {save_skeleton_edges_path}")\n\n                skeleton_edges = load_txt_array(save_skeleton_edges_path)\n                check_expected_columns(\n                    "SKELETON EDGES",\n                    skeleton_edges,\n                    expected_columns=7,\n                )\n\n                print(f"Skeleton edges shape : {skeleton_edges.shape}")\n                print("Skeleton edges format: x,y,z,r,g,b,edge_id")\n                print("Skipped: adding missing skeleton points and edge segmentation.")\n\n            else:\n                print()\n                print("[REBUILD] Recomputing and overwriting skeleton edges:")\n                print(f"    {save_skeleton_edges_path}")\n                print(f"Algorithm version: {edge_algorithm_version}")\n\n                ###############################################################\n                # 第零步：为没有 skeleton 的小 PCD component 添加质心骨架点\n                ###############################################################\n\n                print()\n                print("Adding missing skeleton points from small PCD components...")\n\n                (\n                    refined_skeleton,\n                    added_skeleton_points,\n                    pcd_component_labels,\n                    pcd_component_information,\n                ) = add_missing_refined_skeleton_from_small_pcd_components(\n                    pcd=pcd,\n                    refined_skeleton=refined_skeleton,\n                    dbscan_eps=2.0,\n                    dbscan_min_samples=10,\n                    min_component_points=20,\n                    max_component_points=1000,\n                    solid_grid_spacing=1.0,\n                    local_padding=3,\n                    surface_dilation_iterations=1,\n                    surface_closing_iterations=1,\n                )\n\n                # 按照原来的路径覆盖保存，格式仍然是 x,y,z,score。\n                np.savetxt(\n                    refined_skeleton_path,\n                    refined_skeleton,\n                    delimiter=",",\n                    fmt="%.6f",\n                )\n\n                print("Refined skeleton updated:")\n                print(f"    {refined_skeleton_path}")\n                print("Refined skeleton format: x,y,z,score")\n\n                ###############################################################\n                # 第一步：将 refined skeleton 按照分叉点切分成拓扑边\n                ###############################################################\n\n                print()\n                print("Segmenting refined skeleton into topology edges...")\n\n                skeleton_edges = segment_refined_skeleton_edges(\n                    refined_skeleton=refined_skeleton,\n                    neighbor_radius=1.8,\n                    random_seed=42,\n                )\n\n                np.savetxt(\n                    save_original_skeleton_edges_path,\n                    skeleton_edges,\n                    delimiter=",",\n                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],\n                )\n\n                edge_segmentation_rebuilt = True\n\n                print("Original topology edges saved:")\n                print(f"    {save_original_skeleton_edges_path}")\n                print("The canonical skeleton_edges file will be overwritten after cleanup.")\n\n            ###################################################################\n            # 第二步：沿 edge 法平面分析 PCD 截面是否出现急剧缩小\n            ###################################################################\n\n            if (\n                not edge_segmentation_rebuilt\n                and\n                save_cross_section_edges_path.exists()\n                and save_cross_section_points_path.exists()\n                and save_deletion_points_path.exists()\n            ):\n                print()\n                print("[REUSED] Existing cross-section results found:")\n                print(f"    {save_cross_section_edges_path}")\n                print(f"    {save_cross_section_points_path}")\n                print(f"    {save_deletion_points_path}")\n            else:\n                print()\n                print("Detecting abrupt cross-section narrowing along edges...")\n\n                (\n                    _,\n                    cross_section_diagnostics,\n                    deletion_row_indices,\n                ) = detect_edges_with_abrupt_cross_section_narrowing(\n                    skeleton_edges=skeleton_edges,\n                    pcd=pcd,\n                    save_path=save_cross_section_edges_path,\n                    section_points_save_path=save_cross_section_points_path,\n                    deletion_points_save_path=save_deletion_points_path,\n                    neighbor_radius=1.8,\n                    tangent_window=2,\n                    cross_section_half_thickness=1.0,\n                    cross_section_max_radius=8.0,\n                    min_section_points=1,\n                    section_radius_quantile=0.75,\n                    smoothing_window=1,\n                    endpoint_margin=4,\n                    side_window=4,\n                    guard_sections=0,\n                    min_side_sections=1,\n                    narrow_ratio_threshold=0.65,\n                    visualization_section_stride=4,\n                    angular_bins=36,\n                    points_per_angle_bin=2,\n                    radial_outlier_factor=1.75,\n                    min_owned_angular_coverage=0.20,\n                    short_junction_edge_max_arc_length=15.0,\n                    junction_attachment_radius=1.8,\n                    local_cut_padding_points=0,\n                    local_cut_recovery_ratio=0.90,\n                    local_cut_recovery_stable_sections=2,\n                    local_cut_recovery_reference_quantile=0.75,\n                    short_single_junction_max_points=2,\n                )\n\n                ###############################################################\n                # 第三步：删除黄色点，重新计算 junction 和 edge，覆盖正式文件\n                ###############################################################\n\n                print()\n                print("Deleting yellow points and rebuilding final topology...")\n\n                skeleton_edges = rebuild_skeleton_edges_after_yellow_deletion(\n                    skeleton_edges=skeleton_edges,\n                    deletion_row_indices=deletion_row_indices,\n                    neighbor_radius=1.8,\n                    random_seed=42,\n                )\n\n                np.savetxt(\n                    save_skeleton_edges_path,\n                    skeleton_edges,\n                    delimiter=",",\n                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],\n                )\n                save_skeleton_edges_version_path.write_text(\n                    edge_algorithm_version + "\\n",\n                    encoding="utf-8",\n                )\n                edge_segmentation_is_current = True\n\n                print("Final skeleton edges saved:")\n                print(f"    {save_skeleton_edges_path}")\n                print("Final format: x,y,z,r,g,b,edge_id")\n                print("Junctions and edge IDs were recalculated after deletion.")\n\n            ###################################################################\n            # 第四步：跨 junction 合并 physical edges，生成 social instance\n            ###################################################################\n\n            if (\n                social_instance_is_current\n                and not edge_segmentation_rebuilt\n            ):\n                print()\n                print("[REUSED] Existing social skeleton instances found:")\n                print(f"    {save_skeleton_social_instance_path}")\n            else:\n                print()\n                print(\n                    "Merging physical edges through junctions into "\n                    "social instances..."\n                )\n                skeleton_social_instances = segment_skeleton_social_instances(\n                    skeleton_edges=skeleton_edges,\n                    neighbor_radius=1.8,\n                    random_seed=42,\n                    prune_shortcut_edges=True,\n                )\n                np.savetxt(\n                    save_skeleton_social_instance_path,\n                    skeleton_social_instances,\n                    delimiter=",",\n                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],\n                )\n                save_skeleton_social_version_path.write_text(\n                    social_instance_algorithm_version + "\\n",\n                    encoding="utf-8",\n                )\n                social_instance_is_current = True\n                print("Social skeleton instances saved:")\n                print(f"    {save_skeleton_social_instance_path}")\n                print("Format: x,y,z,r,g,b,instance_id")\n\n            ###################################################################\n            # 第五步：基于最终 edge 计算 feature，并进行无监督 UMAP 分析\n            ###################################################################\n\n            if (\n                save_clustered_skeleton_edges_path.exists()\n                and not edge_segmentation_rebuilt\n            ):\n                print()\n                print("[REUSED] Existing clustered skeleton found:")\n                print(f"    {save_clustered_skeleton_edges_path}")\n            else:\n                if (\n                    save_edge_umap_csv_path.exists()\n                    and not edge_segmentation_rebuilt\n                ):\n                    print()\n                    print("[REUSED] Existing edge clustering table found:")\n                    print(f"    {save_edge_umap_csv_path}")\n                    clustering_table = pd.read_csv(save_edge_umap_csv_path)\n                else:\n                    if (\n                        save_edge_features_path.exists()\n                        and not edge_segmentation_rebuilt\n                    ):\n                        print()\n                        print("[REUSED] Existing edge features found:")\n                        print(f"    {save_edge_features_path}")\n                        edge_feature_table = pd.read_csv(save_edge_features_path)\n                    else:\n                        print()\n                        print("Computing skeleton-edge features...")\n\n                        edge_feature_table = compute_skeleton_edge_features(\n                            skeleton_edges=skeleton_edges,\n                            refined_skeleton=refined_skeleton,\n                            voxel=voxel,\n                            pcd=pcd,\n                            neighbor_radius=1.8,\n                            voxel_assignment_max_distance=8.0,\n                            pcd_assignment_max_distance=8.0,\n                            junction_search_radius=2.5,\n                            profile_bins=10,\n                        )\n\n                        edge_feature_table.to_csv(\n                            save_edge_features_path,\n                            index=False,\n                        )\n                        print("Edge features saved:")\n                        print(f"    {save_edge_features_path}")\n\n                    clustering_table = run_edge_umap_and_clustering(\n                        feature_table=edge_feature_table,\n                        save_umap_csv_path=save_edge_umap_csv_path,\n                        save_umap_png_path=save_edge_umap_png_path,\n                        random_seed=42,\n                        umap_n_neighbors=15,\n                        umap_min_dist=0.10,\n                        hdbscan_min_cluster_size=10,\n                    )\n\n                print()\n                print("Mapping edge clusters back to skeleton points...")\n                save_clustered_skeleton_edges_xyzrgb(\n                    skeleton_edges=skeleton_edges,\n                    clustering_table=clustering_table,\n                    save_path=save_clustered_skeleton_edges_path,\n                )\n\n            ###################################################################\n            # TODO：后续版本从这里开始逐步加入分割代码\n            ###################################################################\n\n            # voxel_instance, pcd_instance, skeleton_instance = \\\n            #     segment_mito_instances(\n            #         voxel=voxel,\n            #         pcd=pcd,\n            #         skeleton=skeleton,\n            #         refined_skeleton=refined_skeleton,\n            #     )\n\n            # skeleton_instance = split_instance_skeleton_by_branch_points(\n            #     skeleton_instance\n            # )\n\n            # save_pcd2txt(voxel_instance, save_voxel_instance_path)\n            # save_pcd2txt(pcd_instance, save_pcd_instance_path)\n            # save_pcd2txt(skeleton_instance, save_skeleton_instance_path)\n\n            total_done += 1\n\n    print("=========================================================================")\n    print(f"Skeleton edge segmentation finished. Images processed: {total_done}")\n    print(\n        "Saved files: *_skeleton_edges.txt, "\n        "*_skeleton_instance_social.txt, "\n        "*_skeleton_edges_original.txt, "\n        "*_skeleton_edges_algorithm_version.txt, "\n        "*_skeleton_edge_features.csv, "\n        "*_skeleton_edge_umap.csv, *_skeleton_edge_umap.png, "\n        "*_skeleton_edges_clustered.txt, "\n        "*_skeleton_edges_cross_section.txt, "\n        "*_pcd_cross_sections_first_shell_local_drop_flat.txt, "\n        "*_skeleton_edges_deletion_candidates_recovery.txt"\n    )\n    print(\n        "Final *_skeleton_edges.txt contains only the remaining skeleton "\n        "points, with junctions and edge IDs recalculated after yellow deletion."\n    )\n    print(\n        "Final *_skeleton_instance_social.txt merges all physical edges that "\n        "remain connected through junctions."\n    )\n    print("=========================================================================")\n'

LINKNEX_INSTANCE_ALGO02_SOURCE = 'from pathlib import Path\nimport os\n\nimport numpy as np\nfrom scipy.sparse import csr_matrix\nfrom scipy.sparse.csgraph import connected_components\nfrom scipy.spatial import cKDTree\n\n\ndef load_txt_array(path):\n    """读取逗号分隔、无表头的 txt 数据。"""\n    data = np.loadtxt(path, delimiter=",", dtype=np.float64)\n\n    if data.ndim == 1:\n        data = data.reshape(1, -1)\n\n    return data\n\n\ndef print_file_information(label, path, data, column_format):\n    """清楚打印一个输入文件的路径、shape和列格式。"""\n    print(f"{label} path   : {path}")\n    print(f"{label} shape  : {data.shape}")\n    print(f"{label} format : {column_format}")\n\n\ndef check_minimum_columns(label, data, minimum_columns):\n    """检查输入数据至少具有要求的列数。"""\n    if data.ndim != 2 or data.shape[1] < minimum_columns:\n        raise ValueError(\n            f"{label} must contain at least {minimum_columns} columns, "\n            f"but its shape is {data.shape}."\n        )\n\n\ndef summarize_distances(label, distances):\n    """打印最近邻分配距离，便于检查坐标是否对齐。"""\n    distances = np.asarray(distances, dtype=np.float64)\n\n    print(f"{label}:")\n    print(f"    minimum : {np.min(distances):.4f}")\n    print(f"    median  : {np.median(distances):.4f}")\n    print(f"    q95     : {np.quantile(distances, 0.95):.4f}")\n    print(f"    maximum : {np.max(distances):.4f}")\n\n\ndef build_canonical_colors(labels, rgb, layer_name):\n    """\n    为每个内部 label 确定唯一RGB。\n\n    正常情况下，同一 edge/instance 的所有骨架点颜色完全相同。如果输入中\n    意外出现多种颜色，则采用该 label 中出现次数最多的颜色，并打印警告。\n    label 只在计算过程中使用，不会写入最终文件。\n    """\n    labels = np.asarray(labels, dtype=np.int64)\n    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int32)\n    unique_labels = np.unique(labels)\n    canonical_colors = np.empty((len(unique_labels), 3), dtype=np.int32)\n    inconsistent_label_count = 0\n\n    for position, one_label in enumerate(unique_labels):\n        one_rgb = rgb[labels == one_label]\n        colors, counts = np.unique(one_rgb, axis=0, return_counts=True)\n        canonical_colors[position] = colors[np.argmax(counts)]\n\n        if len(colors) > 1:\n            inconsistent_label_count += 1\n\n    if inconsistent_label_count > 0:\n        print(\n            f"[WARNING] {layer_name}: {inconsistent_label_count} labels "\n            "contained more than one RGB value. The most frequent RGB "\n            "within each label was used."\n        )\n\n    return unique_labels, canonical_colors\n\n\ndef _disabled_assign_pcd_by_neighborhood_normals(\n    pcd,\n    skeleton_instances,\n    layer_name,\n    normal_candidate_count=64,\n    normal_neighborhood_count=16,\n    normal_neighborhood_radius=2.5,\n    normal_neighbor_min_cosine=0.25,\n    normal_min_support_fraction=0.60,\n    normal_min_support_count=3,\n    normal_search_max_distance=12.0,\n    normal_ray_max_perpendicular_distance=2.5,\n    normal_ray_axial_weight=0.05,\n    query_chunk_size=5000,\n    workers=-1,\n):\n    """\n    已停用的旧实验函数；当前main流程不会调用。\n\n    让PCD局部表面邻域的多条反法向共同投票寻找内部骨架instance。\n\n    skeleton_instances列格式必须至少为：x,y,z,r,g,b,label。\n    label>0才是有效instance；physical层中的junction label=0不会成为\n    独立instance。\n\n    对每个PCD点，先取得附近且法向方向相容的PCD邻居。每个邻居都沿\n    自己的反法向对候选骨架投票。只有获得足够邻域支持的骨架候选才能\n    给当前PCD点赋值，因此不会因为单个法向量噪声或纯欧氏距离而轻易\n    串到相邻线粒体。\n\n    没有可靠投票结果的PCD点不会退回最近骨架，而是从附近已经可靠\n    分割、法向方向相容的PCD点继承结果。\n    """\n    check_minimum_columns("PCD", pcd, minimum_columns=6)\n    check_minimum_columns(\n        f"{layer_name} skeleton",\n        skeleton_instances,\n        minimum_columns=7,\n    )\n\n    all_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)\n    valid_rows = np.where(all_labels > 0)[0]\n\n    if len(valid_rows) == 0:\n        raise ValueError(\n            f"{layer_name} skeleton contains no valid label > 0."\n        )\n\n    seed_xyz = skeleton_instances[valid_rows, :3]\n    seed_labels = all_labels[valid_rows]\n    seed_rgb = skeleton_instances[valid_rows, 3:6]\n\n    unique_labels, canonical_colors = build_canonical_colors(\n        labels=seed_labels,\n        rgb=seed_rgb,\n        layer_name=layer_name,\n    )\n\n    skeleton_tree = cKDTree(seed_xyz)\n    candidate_count = min(int(normal_candidate_count), len(seed_xyz))\n    neighborhood_count = min(int(normal_neighborhood_count), len(pcd))\n    if candidate_count < 1:\n        raise ValueError("normal_candidate_count must be at least 1.")\n    if neighborhood_count < 1:\n        raise ValueError("normal_neighborhood_count must be at least 1.")\n    if normal_neighborhood_radius <= 0:\n        raise ValueError("normal_neighborhood_radius must be positive.")\n    if not -1.0 <= normal_neighbor_min_cosine <= 1.0:\n        raise ValueError(\n            "normal_neighbor_min_cosine must be between -1 and 1."\n        )\n    if not 0.0 < normal_min_support_fraction <= 1.0:\n        raise ValueError(\n            "normal_min_support_fraction must be in (0, 1]."\n        )\n    if normal_min_support_count < 1:\n        raise ValueError("normal_min_support_count must be at least 1.")\n    if normal_search_max_distance <= 0:\n        raise ValueError("normal_search_max_distance must be positive.")\n    if normal_ray_max_perpendicular_distance < 0:\n        raise ValueError(\n            "normal_ray_max_perpendicular_distance cannot be negative."\n        )\n    if query_chunk_size < 1:\n        raise ValueError("query_chunk_size must be at least 1.")\n\n    point_count = len(pcd)\n    pcd_xyz = pcd[:, :3]\n    unit_normals = pcd[:, 3:6].copy()\n    normal_length = np.linalg.norm(unit_normals, axis=1)\n    valid_normal = normal_length > 1e-8\n    unit_normals[valid_normal] /= normal_length[valid_normal, None]\n    unit_normals[~valid_normal] = 0.0\n\n    pcd_tree = cKDTree(pcd_xyz)\n    assigned_seed_rows = np.full(point_count, -1, dtype=np.int64)\n    selected_distances = np.full(point_count, np.nan, dtype=np.float64)\n    reliable_match_count = 0\n    insufficient_vote_count = 0\n\n    for chunk_start in range(0, point_count, query_chunk_size):\n        chunk_end = min(chunk_start + query_chunk_size, point_count)\n        point_xyz = pcd_xyz[chunk_start:chunk_end]\n        point_normal = unit_normals[chunk_start:chunk_end]\n        point_valid_normal = valid_normal[chunk_start:chunk_end]\n\n        neighbor_distances, neighbor_rows = pcd_tree.query(\n            point_xyz,\n            k=neighborhood_count,\n            workers=workers,\n        )\n        if neighborhood_count == 1:\n            neighbor_distances = neighbor_distances[:, None]\n            neighbor_rows = neighbor_rows[:, None]\n\n        neighbor_normals = unit_normals[neighbor_rows]\n        neighbor_normal_valid = valid_normal[neighbor_rows]\n        normal_cosine = np.einsum(\n            "nmd,nd->nm",\n            neighbor_normals,\n            point_normal,\n        )\n        active_neighbor = (\n            point_valid_normal[:, None]\n            & neighbor_normal_valid\n            & (neighbor_distances <= normal_neighborhood_radius)\n            & (normal_cosine >= normal_neighbor_min_cosine)\n        )\n        active_neighbor_count = np.sum(active_neighbor, axis=1)\n\n        candidate_distances, candidate_rows = skeleton_tree.query(\n            point_xyz,\n            k=candidate_count,\n            workers=workers,\n        )\n        if candidate_count == 1:\n            candidate_distances = candidate_distances[:, None]\n            candidate_rows = candidate_rows[:, None]\n\n        candidate_xyz = seed_xyz[candidate_rows]\n        vote_count = np.zeros(\n            (len(point_xyz), candidate_count),\n            dtype=np.int16,\n        )\n        vote_score_sum = np.zeros(\n            (len(point_xyz), candidate_count),\n            dtype=np.float64,\n        )\n\n        # 每个局部表面邻居分别用自己的反法向给候选骨架投票。\n        # 按邻居循环可以避免创建 n×neighbor×candidate×3 的巨大数组。\n        for neighbor_position in range(neighborhood_count):\n            neighbor_is_active = active_neighbor[:, neighbor_position]\n            if not np.any(neighbor_is_active):\n                continue\n\n            neighbor_xyz = pcd_xyz[\n                neighbor_rows[:, neighbor_position]\n            ]\n            inward_direction = -neighbor_normals[:, neighbor_position]\n            candidate_vector = candidate_xyz - neighbor_xyz[:, None, :]\n            axial_depth = np.einsum(\n                "nkd,nd->nk",\n                candidate_vector,\n                inward_direction,\n            )\n            squared_distance = np.sum(candidate_vector ** 2, axis=2)\n            perpendicular_distance = np.sqrt(\n                np.maximum(squared_distance - axial_depth ** 2, 0.0)\n            )\n            one_vote = (\n                neighbor_is_active[:, None]\n                & (axial_depth > 1e-8)\n                & (\n                    perpendicular_distance\n                    <= normal_ray_max_perpendicular_distance\n                )\n            )\n            one_score = (\n                perpendicular_distance\n                + normal_ray_axial_weight * axial_depth\n            )\n            vote_count += one_vote.astype(np.int16)\n            vote_score_sum += np.where(one_vote, one_score, 0.0)\n\n        required_vote_count = np.ceil(\n            normal_min_support_fraction\n            * active_neighbor_count\n        ).astype(np.int32)\n        required_vote_count = np.maximum(\n            required_vote_count,\n            int(normal_min_support_count),\n        )\n        # 当局部本来就少于minimum count时，不能要求超过现有邻居数。\n        required_vote_count = np.minimum(\n            required_vote_count,\n            active_neighbor_count,\n        )\n        required_vote_count = np.maximum(required_vote_count, 1)\n\n        candidate_is_reliable = (\n            (candidate_distances <= normal_search_max_distance)\n            & (vote_count >= required_vote_count[:, None])\n            & (active_neighbor_count[:, None] > 0)\n        )\n        maximum_vote = np.max(\n            np.where(candidate_is_reliable, vote_count, -1),\n            axis=1,\n        )\n        has_reliable = maximum_vote >= 0\n\n        mean_vote_score = vote_score_sum / np.maximum(vote_count, 1)\n        top_vote_candidate = (\n            candidate_is_reliable\n            & (vote_count == maximum_vote[:, None])\n        )\n        final_score = np.where(\n            top_vote_candidate,\n            mean_vote_score,\n            np.inf,\n        )\n        selected_candidate = np.argmin(final_score, axis=1)\n        local_rows = np.arange(len(point_xyz), dtype=np.int64)\n        chosen_seed_rows = candidate_rows[\n            local_rows,\n            selected_candidate,\n        ]\n        chosen_distances = candidate_distances[\n            local_rows,\n            selected_candidate,\n        ]\n\n        global_rows = np.arange(chunk_start, chunk_end, dtype=np.int64)\n        reliable_global_rows = global_rows[has_reliable]\n        assigned_seed_rows[reliable_global_rows] = chosen_seed_rows[\n            has_reliable\n        ]\n        selected_distances[reliable_global_rows] = chosen_distances[\n            has_reliable\n        ]\n        reliable_match_count += int(np.sum(has_reliable))\n        insufficient_vote_count += int(np.sum(~has_reliable))\n\n    # 不可靠点只向已经可靠分割的PCD表面点传播，绝不退回最近骨架。\n    reliable_rows = np.where(assigned_seed_rows >= 0)[0]\n    unresolved_rows = np.where(assigned_seed_rows < 0)[0]\n    propagated_compatible_count = 0\n    emergency_pcd_count = 0\n\n    if len(reliable_rows) == 0:\n        raise ValueError(\n            f"{layer_name}: no PCD point obtained a reliable local-normal "\n            "vote. Check whether PCD normals point outward, or relax the "\n            "normal-neighborhood parameters."\n        )\n\n    if len(unresolved_rows) > 0:\n        reliable_tree = cKDTree(pcd_xyz[reliable_rows])\n        propagation_k = min(16, len(reliable_rows))\n        propagation_distance, propagation_local_rows = reliable_tree.query(\n            pcd_xyz[unresolved_rows],\n            k=propagation_k,\n            workers=workers,\n        )\n        if propagation_k == 1:\n            propagation_distance = propagation_distance[:, None]\n            propagation_local_rows = propagation_local_rows[:, None]\n\n        propagation_global_rows = reliable_rows[propagation_local_rows]\n        unresolved_normals = unit_normals[unresolved_rows]\n        resolved_neighbor_normals = unit_normals[propagation_global_rows]\n        propagation_cosine = np.einsum(\n            "nkd,nd->nk",\n            resolved_neighbor_normals,\n            unresolved_normals,\n        )\n        compatible = (\n            valid_normal[unresolved_rows, None]\n            & valid_normal[propagation_global_rows]\n            & (propagation_cosine >= normal_neighbor_min_cosine)\n        )\n        compatible_distance = np.where(\n            compatible,\n            propagation_distance,\n            np.inf,\n        )\n        compatible_best = np.argmin(compatible_distance, axis=1)\n        has_compatible = np.any(compatible, axis=1)\n\n        selected_propagation = np.zeros(\n            len(unresolved_rows),\n            dtype=np.int64,\n        )\n        selected_propagation[has_compatible] = compatible_best[\n            has_compatible\n        ]\n        source_rows = propagation_global_rows[\n            np.arange(len(unresolved_rows)),\n            selected_propagation,\n        ]\n        assigned_seed_rows[unresolved_rows] = assigned_seed_rows[source_rows]\n        selected_distances[unresolved_rows] = np.linalg.norm(\n            pcd_xyz[unresolved_rows]\n            - seed_xyz[assigned_seed_rows[unresolved_rows]],\n            axis=1,\n        )\n        propagated_compatible_count = int(np.sum(has_compatible))\n        emergency_pcd_count = int(np.sum(~has_compatible))\n\n    assigned_labels = seed_labels[assigned_seed_rows]\n\n    # unique_labels已经排序，可用searchsorted快速把内部label映射到RGB。\n    color_positions = np.searchsorted(unique_labels, assigned_labels)\n    assigned_rgb = canonical_colors[color_positions]\n\n    print(f"{layer_name} PCD neighborhood-normal assignment:")\n    print(f"    Valid skeleton seeds : {len(valid_rows)}")\n    print(f"    Skeleton instances   : {len(unique_labels)}")\n    print(f"    PCD points assigned  : {point_count}")\n    print(f"    Reliable local votes : {reliable_match_count}")\n    print(f"    Insufficient votes   : {insufficient_vote_count}")\n    print(f"    Propagated compatible: {propagated_compatible_count}")\n    print(f"    Emergency PCD inherit: {emergency_pcd_count}")\n    print(f"    Invalid normals      : {int(np.sum(~valid_normal))}")\n    print("    Nearest skeleton fallback: 0")\n    print("    Unassigned PCD points: 0")\n    summarize_distances(\n        "    Selected PCD-to-skeleton distance",\n        selected_distances,\n    )\n\n    return assigned_rgb, assigned_labels, selected_distances\n\n\ndef assign_pcd_from_nearest_skeleton(\n    pcd,\n    skeleton_instances,\n    layer_name,\n    workers=-1,\n):\n    """\n    按照最近有效骨架点快速完成PCD初始分割。\n\n    label只在内部使用，不会写入最终文件。physical层的junction label=0\n    不作为instance种子。\n    """\n    check_minimum_columns("PCD", pcd, minimum_columns=6)\n    check_minimum_columns(\n        f"{layer_name} skeleton",\n        skeleton_instances,\n        minimum_columns=7,\n    )\n\n    all_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)\n    valid_rows = np.where(all_labels > 0)[0]\n    if len(valid_rows) == 0:\n        raise ValueError(\n            f"{layer_name} skeleton contains no valid label > 0."\n        )\n\n    seed_xyz = skeleton_instances[valid_rows, :3]\n    seed_labels = all_labels[valid_rows]\n    seed_rgb = skeleton_instances[valid_rows, 3:6]\n    unique_labels, canonical_colors = build_canonical_colors(\n        labels=seed_labels,\n        rgb=seed_rgb,\n        layer_name=layer_name,\n    )\n\n    skeleton_tree = cKDTree(seed_xyz)\n    distances, nearest_seed_rows = skeleton_tree.query(\n        pcd[:, :3],\n        k=1,\n        workers=workers,\n    )\n    assigned_labels = seed_labels[nearest_seed_rows]\n    color_positions = np.searchsorted(unique_labels, assigned_labels)\n    assigned_rgb = canonical_colors[color_positions]\n\n    print(f"{layer_name} initial nearest-skeleton PCD assignment:")\n    print(f"    Valid skeleton seeds : {len(valid_rows)}")\n    print(f"    Skeleton instances   : {len(unique_labels)}")\n    print(f"    PCD points assigned  : {len(pcd)}")\n    print("    Unassigned PCD points: 0")\n    summarize_distances(\n        "    PCD-to-nearest-skeleton distance",\n        distances,\n    )\n\n    return assigned_rgb, assigned_labels, distances\n\n\ndef estimate_pcd_connect_radius(\n    pcd_xyz,\n    connect_radius_factor=2.5,\n    workers=-1,\n):\n    """根据PCD典型最近邻距离自动估计表面连通半径。"""\n    if len(pcd_xyz) < 2:\n        raise ValueError("At least two PCD points are required.")\n    if connect_radius_factor <= 0:\n        raise ValueError("connect_radius_factor must be positive.")\n\n    tree = cKDTree(pcd_xyz)\n    distances, _ = tree.query(pcd_xyz, k=2, workers=workers)\n    nearest_distance = distances[:, 1]\n    positive_distance = nearest_distance[\n        np.isfinite(nearest_distance) & (nearest_distance > 1e-8)\n    ]\n    if len(positive_distance) == 0:\n        raise ValueError(\n            "Could not estimate PCD spacing because all nearest-neighbor "\n            "distances are zero or invalid."\n        )\n\n    median_nearest_distance = float(np.median(positive_distance))\n    connect_radius = median_nearest_distance * connect_radius_factor\n    print("Automatic PCD connectivity radius:")\n    print(\n        "    Median nearest-neighbor distance: "\n        f"{median_nearest_distance:.4f}"\n    )\n    print(f"    Radius factor                 : {connect_radius_factor:.3f}")\n    print(f"    PCD connectivity radius       : {connect_radius:.4f}")\n    return connect_radius\n\n\ndef compute_same_instance_pcd_components(\n    pcd_xyz,\n    assigned_labels,\n    connect_radius,\n):\n    """只连接距离足够近且当前内部label相同的PCD点。"""\n    if connect_radius <= 0:\n        raise ValueError("connect_radius must be positive.")\n\n    point_count = len(pcd_xyz)\n    tree = cKDTree(pcd_xyz)\n    neighbor_pairs = tree.query_pairs(\n        r=connect_radius,\n        output_type="ndarray",\n    )\n\n    if len(neighbor_pairs) == 0:\n        component_labels = np.arange(point_count, dtype=np.int64)\n        return component_labels, 0, 0\n\n    same_label = (\n        assigned_labels[neighbor_pairs[:, 0]]\n        == assigned_labels[neighbor_pairs[:, 1]]\n    )\n    same_pairs = neighbor_pairs[same_label]\n\n    if len(same_pairs) == 0:\n        component_labels = np.arange(point_count, dtype=np.int64)\n        return component_labels, len(neighbor_pairs), 0\n\n    graph_rows = np.concatenate([same_pairs[:, 0], same_pairs[:, 1]])\n    graph_columns = np.concatenate([same_pairs[:, 1], same_pairs[:, 0]])\n    graph = csr_matrix(\n        (\n            np.ones(len(graph_rows), dtype=np.uint8),\n            (graph_rows, graph_columns),\n        ),\n        shape=(point_count, point_count),\n    )\n    _, component_labels = connected_components(\n        graph,\n        directed=False,\n        return_labels=True,\n    )\n    return component_labels, len(neighbor_pairs), len(same_pairs)\n\n\ndef repair_disconnected_pcd_instances(\n    pcd,\n    assigned_labels,\n    skeleton_instances,\n    layer_name,\n    connect_radius=None,\n    connect_radius_factor=2.5,\n    repair_iterations=2,\n    reassignment_neighbor_count=64,\n    reassignment_max_distance_factor=2.0,\n    reassignment_min_vote_fraction=0.60,\n    reassignment_min_other_neighbors=3,\n    component_vote_sample_points=500,\n    workers=-1,\n):\n    """\n    修复同一个初始instance中明显断开的PCD component。\n\n    每个内部label中，与该label骨架结合最紧密、获得最多骨架点支持的\n    component作为主体。其他断开component只有在周围另一个label获得\n    足够强的距离加权投票时才整体改归属；没有明确证据时保持不变。\n    """\n    pcd_xyz = np.asarray(pcd[:, :3], dtype=np.float64)\n    labels = np.asarray(assigned_labels, dtype=np.int64).copy()\n    skeleton_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)\n    valid_skeleton_rows = np.where(skeleton_labels > 0)[0]\n    skeleton_xyz = skeleton_instances[valid_skeleton_rows, :3]\n    skeleton_labels = skeleton_labels[valid_skeleton_rows]\n\n    unique_labels, canonical_colors = build_canonical_colors(\n        labels=skeleton_labels,\n        rgb=skeleton_instances[valid_skeleton_rows, 3:6],\n        layer_name=layer_name,\n    )\n\n    if connect_radius is None:\n        connect_radius = estimate_pcd_connect_radius(\n            pcd_xyz=pcd_xyz,\n            connect_radius_factor=connect_radius_factor,\n            workers=workers,\n        )\n    else:\n        connect_radius = float(connect_radius)\n        if connect_radius <= 0:\n            raise ValueError("connect_radius must be positive or None.")\n        print(f"Fixed PCD connectivity radius: {connect_radius:.4f}")\n\n    if repair_iterations < 0:\n        raise ValueError("repair_iterations cannot be negative.")\n    if reassignment_neighbor_count < 1:\n        raise ValueError("reassignment_neighbor_count must be at least 1.")\n    if reassignment_max_distance_factor <= 0:\n        raise ValueError(\n            "reassignment_max_distance_factor must be positive."\n        )\n    if not 0.0 < reassignment_min_vote_fraction <= 1.0:\n        raise ValueError(\n            "reassignment_min_vote_fraction must be in (0, 1]."\n        )\n    if reassignment_min_other_neighbors < 1:\n        raise ValueError(\n            "reassignment_min_other_neighbors must be at least 1."\n        )\n    if component_vote_sample_points < 1:\n        raise ValueError("component_vote_sample_points must be at least 1.")\n\n    pcd_tree = cKDTree(pcd_xyz)\n    neighbor_count = min(int(reassignment_neighbor_count), len(pcd_xyz))\n    reassignment_max_distance = (\n        connect_radius * reassignment_max_distance_factor\n    )\n    total_reassigned_components = 0\n    total_reassigned_points = 0\n\n    print(f"{layer_name} disconnected-component repair:")\n    print(f"    Connectivity radius       : {connect_radius:.4f}")\n    print(f"    Reassignment max distance : {reassignment_max_distance:.4f}")\n    print(f"    Maximum repair iterations : {repair_iterations}")\n\n    for iteration in range(1, repair_iterations + 1):\n        component_ids, all_pair_count, same_pair_count = (\n            compute_same_instance_pcd_components(\n                pcd_xyz=pcd_xyz,\n                assigned_labels=labels,\n                connect_radius=connect_radius,\n            )\n        )\n\n        proposals = []\n        split_instance_count = 0\n        detached_component_count = 0\n        unresolved_component_count = 0\n\n        for one_label in np.unique(labels):\n            label_rows = np.where(labels == one_label)[0]\n            one_component_ids = np.unique(component_ids[label_rows])\n            if len(one_component_ids) <= 1:\n                continue\n\n            split_instance_count += 1\n            component_rows = {\n                int(component_id): label_rows[\n                    component_ids[label_rows] == component_id\n                ]\n                for component_id in one_component_ids\n            }\n\n            # 该label的骨架点分别落到哪个PCD component附近。\n            one_skeleton_xyz = skeleton_xyz[skeleton_labels == one_label]\n            label_tree = cKDTree(pcd_xyz[label_rows])\n            skeleton_distance, nearest_local_rows = label_tree.query(\n                one_skeleton_xyz,\n                k=1,\n                workers=workers,\n            )\n            nearest_global_rows = label_rows[nearest_local_rows]\n            skeleton_hit_components = component_ids[nearest_global_rows]\n\n            anchor_component = None\n            anchor_key = None\n            for component_id in one_component_ids:\n                component_id = int(component_id)\n                rows = component_rows[component_id]\n                hit_mask = skeleton_hit_components == component_id\n                hit_count = int(np.sum(hit_mask))\n                minimum_skeleton_distance = (\n                    float(np.min(skeleton_distance[hit_mask]))\n                    if np.any(hit_mask)\n                    else np.inf\n                )\n                anchor_candidate_key = (\n                    hit_count,\n                    -minimum_skeleton_distance,\n                    len(rows),\n                )\n                if anchor_key is None or anchor_candidate_key > anchor_key:\n                    anchor_key = anchor_candidate_key\n                    anchor_component = component_id\n\n            for component_id in one_component_ids:\n                component_id = int(component_id)\n                if component_id == anchor_component:\n                    continue\n\n                detached_component_count += 1\n                rows = component_rows[component_id]\n                if len(rows) > component_vote_sample_points:\n                    sample_positions = np.linspace(\n                        0,\n                        len(rows) - 1,\n                        component_vote_sample_points,\n                        dtype=np.int64,\n                    )\n                    sample_rows = rows[sample_positions]\n                else:\n                    sample_rows = rows\n\n                neighbor_distance, neighbor_rows = pcd_tree.query(\n                    pcd_xyz[sample_rows],\n                    k=neighbor_count,\n                    workers=workers,\n                )\n                if neighbor_count == 1:\n                    neighbor_distance = neighbor_distance[:, None]\n                    neighbor_rows = neighbor_rows[:, None]\n\n                neighbor_labels = labels[neighbor_rows]\n                valid_other = (\n                    (neighbor_labels != one_label)\n                    & (neighbor_distance <= reassignment_max_distance)\n                )\n                if int(np.sum(valid_other)) < reassignment_min_other_neighbors:\n                    unresolved_component_count += 1\n                    continue\n\n                other_labels = neighbor_labels[valid_other]\n                other_distances = neighbor_distance[valid_other]\n                vote_weights = 1.0 / np.maximum(other_distances, 1e-6)\n                candidate_labels, inverse = np.unique(\n                    other_labels,\n                    return_inverse=True,\n                )\n                candidate_scores = np.bincount(\n                    inverse,\n                    weights=vote_weights,\n                )\n                best_position = int(np.argmax(candidate_scores))\n                best_label = int(candidate_labels[best_position])\n                vote_fraction = float(\n                    candidate_scores[best_position]\n                    / np.sum(candidate_scores)\n                )\n\n                if vote_fraction < reassignment_min_vote_fraction:\n                    unresolved_component_count += 1\n                    continue\n\n                proposals.append(\n                    (rows, int(one_label), best_label, vote_fraction)\n                )\n\n        if len(proposals) > 0:\n            for rows, old_label, new_label, vote_fraction in proposals:\n                labels[rows] = new_label\n                print(\n                    f"    iteration={iteration} | points={len(rows)} | "\n                    f"{old_label} -> {new_label} | "\n                    f"vote={vote_fraction:.3f}"\n                )\n                total_reassigned_components += 1\n                total_reassigned_points += len(rows)\n\n        print(\n            f"    Iteration {iteration}: split_instances={split_instance_count}, "\n            f"detached_components={detached_component_count}, "\n            f"reassigned_components={len(proposals)}, "\n            f"unresolved_components={unresolved_component_count}, "\n            f"all_neighbor_pairs={all_pair_count}, "\n            f"same_label_pairs={same_pair_count}"\n        )\n\n        if len(proposals) == 0:\n            break\n\n    color_positions = np.searchsorted(unique_labels, labels)\n    if np.any(color_positions >= len(unique_labels)):\n        raise ValueError(\n            f"{layer_name}: repaired PCD contains a label that is absent "\n            "from the skeleton input."\n        )\n    if np.any(unique_labels[color_positions] != labels):\n        raise ValueError(\n            f"{layer_name}: repaired PCD contains a label that is absent "\n            "from the skeleton input."\n        )\n    repaired_rgb = canonical_colors[color_positions]\n\n    print(f"{layer_name} component repair summary:")\n    print(f"    Components reassigned: {total_reassigned_components}")\n    print(f"    PCD points reassigned : {total_reassigned_points}")\n    print(f"    Final instances       : {len(np.unique(labels))}")\n\n    return repaired_rgb, labels, connect_radius\n\n\ndef assign_voxel_from_segmented_pcd(\n    voxel,\n    pcd,\n    pcd_rgb,\n    pcd_labels,\n    layer_name,\n    workers=-1,\n):\n    """\n    使用已经完成连通块修复的PCD为每个voxel赋值。\n\n    voxel寻找最近PCD表面点，然后继承该PCD点的RGB。内部label仅用于打印\n    检查统计，不写入最终文件。\n    """\n    check_minimum_columns("VOXEL", voxel, minimum_columns=4)\n\n    pcd_tree = cKDTree(pcd[:, :3])\n    distances, nearest_pcd_rows = pcd_tree.query(\n        voxel[:, :3],\n        k=1,\n        workers=workers,\n    )\n\n    assigned_rgb = pcd_rgb[nearest_pcd_rows]\n    assigned_labels = pcd_labels[nearest_pcd_rows]\n\n    print(f"{layer_name} voxel assignment from segmented PCD:")\n    print(f"    Voxels assigned      : {len(voxel)}")\n    print(f"    Instances represented: {len(np.unique(assigned_labels))}")\n    print("    Unassigned voxels    : 0")\n    summarize_distances(\n        "    Voxel-to-segmented-PCD nearest distance",\n        distances,\n    )\n\n    return assigned_rgb, assigned_labels, distances\n\n\ndef save_voxel_xyzrgb_intensity(save_path, voxel, rgb):\n    """保存格式：x,y,z,r,g,b,intensity。"""\n    output = np.column_stack(\n        [\n            voxel[:, :3],\n            np.asarray(rgb, dtype=np.int32),\n            voxel[:, 3],\n        ]\n    )\n    np.savetxt(\n        save_path,\n        output,\n        delimiter=",",\n        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%.6f"],\n    )\n\n\ndef save_pcd_xyzrgbnormal_social_id(\n    save_path,\n    pcd,\n    rgb,\n    social_instance_ids,\n):\n    """保存格式：x,y,z,r,g,b,nx,ny,nz,social_instance_id。"""\n    social_instance_ids = np.asarray(\n        social_instance_ids,\n        dtype=np.int64,\n    )\n    if len(social_instance_ids) != len(pcd):\n        raise ValueError(\n            "social_instance_ids and PCD must contain the same number "\n            "of rows."\n        )\n    if np.any(social_instance_ids <= 0):\n        raise ValueError(\n            "Every PCD point must have a social_instance_id > 0."\n        )\n\n    output = np.column_stack(\n        [\n            pcd[:, :3],\n            np.asarray(rgb, dtype=np.int32),\n            pcd[:, 3:6],\n            social_instance_ids,\n        ]\n    )\n    np.savetxt(\n        save_path,\n        output,\n        delimiter=",",\n        fmt=[\n            "%.3f",\n            "%.3f",\n            "%.3f",\n            "%d",\n            "%d",\n            "%d",\n            "%.6f",\n            "%.6f",\n            "%.6f",\n            "%d",\n        ],\n    )\n\n\ndef main():\n\n\n    file_name_list = [\n\n                            # ====================================================\n        # MCF-7\n        # # ====================================================\n        "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",\n        # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",\n        #\n        # # ====================================================\n        # # AC16\n        # # ====================================================\n        "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",\n        # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",\n        #\n        # # ====================================================\n        # # MDA\n        # # ====================================================\n        "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",\n        # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",\n        #\n        # # ====================================================\n        # # Fibroblast\n        # # ====================================================\n        "20260207_fibro_mito488_lyso640_nuc405_tif",\n        # "20251216_fibro_mito488tracker_ld561bodipy_nuc405hurst",\n        #\n        # # ====================================================\n        # # H28\n        # # ====================================================\n        "20250917_h28_mitotracker488_lysotracker640_nuc405",\n        # "20251213_h28_mito488_ld561_nuc405",\n        #\n        # # ====================================================\n        # # HeLa\n        # # ====================================================\n        "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",\n        # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",\n\n        # ====================================================\n        # U2OS\n        # ====================================================\n        # "20251201_u2os_ld488_mito640_nuc405",\n        "20250909_u2os_mitotracker488_lysotracker640_nuc405",\n\n        # ====================================================\n        # MSTO\n        # ====================================================\n        # "20251127_msto_ld488_mito640_nuc405",\n        "20250905_msto_mitotracekr488_lysotracker640_nuc405",\n\n                    ]\n\n    file_name_list = [\n\n        # ====================================================\n        # GLUC\n        # ====================================================\n        # "gluc_mito-mChGFP_LTRDR50nM_0031",\n        # "gluc_mito-mChGFP_LTRDR50nM_0032",\n        # "gluc_mito-mChGFP_LTRDR50nM_0051",\n        # "gluc_mito-mChGFP_LTRDR50nM_0061",\n        # "gluc_mito-mChGFP_LTRDR50nM_0071",\n        # "gluc_mito-mChGFP_LTRDR50nM_0072",\n        # "gluc_mito-mChGFP_LTRDR50nM_0081",\n        # "gluc_mito-mChGFP_LTRDR50nM_0082",\n        # "gluc_mito-mChGFP_LTRDR50nM_0091",\n        #\n        # # ====================================================\n        # # Non-GLUC\n        # # ====================================================\n        #\n        # "mito-mChGFP_LTRDR50nM_0021",\n        # "mito-mChGFP_LTRDR50nM_0022",\n        # "mito-mChGFP_LTRDR50nM_0031",\n        # "mito-mChGFP_LTRDR50nM_0032",\n        # "mito-mChGFP_LTRDR50nM_0041",\n        # "mito-mChGFP_LTRDR50nM_0042",\n        # "mito-mChGFP_LTRDR50nM_0051",\n        # "mito-mChGFP_LTRDR50nM_0061",\n        # "mito-mChGFP_LTRDR50nM_0062",\n        # "mito-mChGFP_LTRDR50nM_0071",\n        # "mito-mChGFP_LTRDR50nM_0072",\n        "mito-mChGFP_LTRDR50nM_0081",\n\n    ]\n\n    ###########################################################################\n    # 运行参数：不使用argparse，直接在这里修改\n    ###########################################################################\n\n    # True：覆盖已有4个输出文件并重新计算。\n    # False：如果4个输出文件都已存在，则直接跳过该图片。\n    overwrite_existing = True\n\n    # cKDTree并行线程数。-1表示使用全部可用CPU线程。\n    workers = -1\n\n    # None表示根据PCD典型最近邻距离自动计算连通半径。\n    # 如需固定半径，可改成例如：pcd_connect_radius = 1.5\n    pcd_connect_radius = None\n\n    # 自动连通半径 = PCD最近邻距离中位数 × 该系数。\n    pcd_connect_radius_factor = 2.5\n\n    # 明显断开的PCD component最多修复多少轮。\n    component_repair_iterations = 2\n\n    # 对每个断开component采样点周围查询多少个PCD邻居进行归属投票。\n    reassignment_neighbor_count = 64\n\n    # 允许投票的最大距离 = PCD连通半径 × 该系数。\n    reassignment_max_distance_factor = 2.0\n\n    # 另一个instance的距离加权票数至少占多少，才整体重新归属。\n    reassignment_min_vote_fraction = 0.60\n\n    # 至少需要多少个其他instance邻居参与投票。\n    reassignment_min_other_neighbors = 3\n\n    # 每个断开component最多采样多少个点进行周围instance投票。\n    component_vote_sample_points = 500\n\n    total_done = 0\n    total_skipped = 0\n\n    for file_name in file_name_list:\n\n        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")\n        # BASE_DIR = Path(f"D:/public_data/S-BIAD1445/neuron/mito_test")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/{file_name}/mito")\n        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")\n        # BASE_DIR = Path(f"E:/data/20260811-YF_output/image_tif/single_cell_tifs")\n        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")\n\n        INPUT_PATH = BASE_DIR / "processed_output"\n        OUTPUT_PATH = BASE_DIR / "processed_output"\n\n        if not os.path.exists(INPUT_PATH):\n            raise FileNotFoundError(\n                f"processed_output folder not found: {INPUT_PATH}"\n            )\n\n        if not os.path.exists(OUTPUT_PATH):\n            os.makedirs(OUTPUT_PATH)\n\n        print(os.listdir(INPUT_PATH))\n\n        #######################################################################\n        # 图片列表\n        #######################################################################\n\n        # 自动读取 processed_output 中的所有原始 *_pcd.txt。\n        imgs_list = [\n            f\n            for f in os.listdir(INPUT_PATH)\n            if f.lower().endswith("_pcd.txt")\n        ]\n\n        # 手动指定需要处理的图片。\n        # 需要批量处理全部图片时，将下面这一行注释掉即可。\n        # imgs_list = ["fibro_mito488_lyso640_nuc405_17689_5ADVMLE"]\n\n        if len(imgs_list) == 0:\n            print(f"No images selected in {INPUT_PATH}")\n            continue\n\n        print("=========================================================================")\n        print(f"Dataset            : {file_name}")\n        print(f"BASE_DIR           : {BASE_DIR}")\n        print(f"INPUT_PATH         : {INPUT_PATH}")\n        print(f"OUTPUT_PATH        : {OUTPUT_PATH}")\n        print(f"Selected images    : {len(imgs_list)}")\n        print(f"Overwrite existing : {overwrite_existing}")\n        print(f"PCD connect radius : {pcd_connect_radius}")\n        print(f"PCD radius factor  : {pcd_connect_radius_factor}")\n        print(f"Repair iterations  : {component_repair_iterations}")\n        print(f"Reassign neighbors : {reassignment_neighbor_count}")\n        print(\n            "Reassign max factor: "\n            f"{reassignment_max_distance_factor}"\n        )\n        print(\n            "Reassign vote frac : "\n            f"{reassignment_min_vote_fraction}"\n        )\n        print("=========================================================================")\n\n        for i, img_name in enumerate(imgs_list, start=1):\n            # 同时兼容：自动扫描得到的 xxx_pcd.txt 和手动填写的 xxx。\n            if img_name.lower().endswith("_pcd.txt"):\n                stem = img_name[:-len("_pcd.txt")]\n            else:\n                stem = img_name\n\n            print("-------------------------------------------------------------------------")\n            print(f"Processing {file_name} | {i}/{len(imgs_list)} | {stem}")\n            print("-------------------------------------------------------------------------")\n\n            ###################################################################\n            # 输入文件路径\n            ###################################################################\n\n            pcd_path = INPUT_PATH / f"{stem}_pcd.txt"\n            voxel_path = INPUT_PATH / f"{stem}_voxel.txt"\n            skeleton_edges_path = INPUT_PATH / f"{stem}_skeleton_edges.txt"\n            skeleton_social_path = (\n                INPUT_PATH / f"{stem}_skeleton_instance_social.txt"\n            )\n\n            ###################################################################\n            # 输出文件路径\n            ###################################################################\n\n            save_voxel_phy_path = (\n                OUTPUT_PATH / f"{stem}_voxel_instance_phy.txt"\n            )\n            save_voxel_social_path = (\n                OUTPUT_PATH / f"{stem}_voxel_instance_social.txt"\n            )\n            save_pcd_phy_path = (\n                OUTPUT_PATH / f"{stem}_pcd_instance_physical.txt"\n            )\n            save_pcd_social_path = (\n                OUTPUT_PATH / f"{stem}_pcd_instance_social.txt"\n            )\n\n            input_paths = {\n                "PCD": pcd_path,\n                "VOXEL": voxel_path,\n                "PHYSICAL SKELETON EDGES": skeleton_edges_path,\n                "SOCIAL SKELETON INSTANCES": skeleton_social_path,\n            }\n            output_paths = {\n                "PHYSICAL VOXEL": save_voxel_phy_path,\n                "SOCIAL VOXEL": save_voxel_social_path,\n                "PHYSICAL PCD": save_pcd_phy_path,\n                "SOCIAL PCD": save_pcd_social_path,\n            }\n\n            print("Input files:")\n            for label, path in input_paths.items():\n                print(f"    {label:<28}: {path}")\n\n            print("Output files:")\n            for label, path in output_paths.items():\n                print(f"    {label:<28}: {path}")\n\n            missing_paths = [\n                path for path in input_paths.values() if not path.exists()\n            ]\n            if len(missing_paths) > 0:\n                print("[SKIPPED] Missing required input files:")\n                for missing_path in missing_paths:\n                    print(f"    {missing_path}")\n                total_skipped += 1\n                continue\n\n            if (\n                not overwrite_existing\n                and all(path.exists() for path in output_paths.values())\n            ):\n                print("[SKIPPED] All four output files already exist.")\n                total_skipped += 1\n                continue\n\n            ###################################################################\n            # 读取并检查输入\n            ###################################################################\n\n            pcd = load_txt_array(pcd_path)\n            voxel = load_txt_array(voxel_path)\n            skeleton_edges = load_txt_array(skeleton_edges_path)\n            skeleton_social = load_txt_array(skeleton_social_path)\n\n            print()\n            print_file_information(\n                "PCD",\n                pcd_path,\n                pcd,\n                "x,y,z,nx,ny,nz",\n            )\n            check_minimum_columns("PCD", pcd, minimum_columns=6)\n\n            print()\n            print_file_information(\n                "VOXEL",\n                voxel_path,\n                voxel,\n                "x,y,z,intensity",\n            )\n            check_minimum_columns("VOXEL", voxel, minimum_columns=4)\n\n            print()\n            print_file_information(\n                "PHYSICAL SKELETON EDGES",\n                skeleton_edges_path,\n                skeleton_edges,\n                "x,y,z,r,g,b,edge_id",\n            )\n            check_minimum_columns(\n                "PHYSICAL SKELETON EDGES",\n                skeleton_edges,\n                minimum_columns=7,\n            )\n\n            print()\n            print_file_information(\n                "SOCIAL SKELETON INSTANCES",\n                skeleton_social_path,\n                skeleton_social,\n                "x,y,z,r,g,b,instance_id",\n            )\n            check_minimum_columns(\n                "SOCIAL SKELETON INSTANCES",\n                skeleton_social,\n                minimum_columns=7,\n            )\n\n            ###################################################################\n            # 第一层：最近physical edge初分PCD，修复断开块，再分voxel\n            ###################################################################\n\n            print()\n            print("Initial PHYSICAL PCD assignment by nearest skeleton...")\n            _, pcd_phy_labels, _ = assign_pcd_from_nearest_skeleton(\n                pcd=pcd,\n                skeleton_instances=skeleton_edges,\n                layer_name="PHYSICAL",\n                workers=workers,\n            )\n            pcd_phy_rgb, pcd_phy_labels, _ = (\n                repair_disconnected_pcd_instances(\n                    pcd=pcd,\n                    assigned_labels=pcd_phy_labels,\n                    skeleton_instances=skeleton_edges,\n                    layer_name="PHYSICAL",\n                    connect_radius=pcd_connect_radius,\n                    connect_radius_factor=pcd_connect_radius_factor,\n                    repair_iterations=component_repair_iterations,\n                    reassignment_neighbor_count=(\n                        reassignment_neighbor_count\n                    ),\n                    reassignment_max_distance_factor=(\n                        reassignment_max_distance_factor\n                    ),\n                    reassignment_min_vote_fraction=(\n                        reassignment_min_vote_fraction\n                    ),\n                    reassignment_min_other_neighbors=(\n                        reassignment_min_other_neighbors\n                    ),\n                    component_vote_sample_points=(\n                        component_vote_sample_points\n                    ),\n                    workers=workers,\n                )\n            )\n            voxel_phy_rgb, _, _ = assign_voxel_from_segmented_pcd(\n                voxel=voxel,\n                pcd=pcd,\n                pcd_rgb=pcd_phy_rgb,\n                pcd_labels=pcd_phy_labels,\n                layer_name="PHYSICAL",\n                workers=workers,\n            )\n\n            save_voxel_xyzrgb_intensity(\n                save_path=save_voxel_phy_path,\n                voxel=voxel,\n                rgb=voxel_phy_rgb,\n            )\n\n            print("PHYSICAL voxel layer saved:")\n            print(f"    {save_voxel_phy_path}")\n            print("    Format: x,y,z,r,g,b,intensity")\n\n            ###################################################################\n            # 第二层：最近social骨架初分PCD，修复断开块，再分voxel\n            ###################################################################\n\n            print()\n            print("Initial SOCIAL PCD assignment by nearest skeleton...")\n            _, pcd_social_labels, _ = assign_pcd_from_nearest_skeleton(\n                pcd=pcd,\n                skeleton_instances=skeleton_social,\n                layer_name="SOCIAL",\n                workers=workers,\n            )\n            pcd_social_rgb, pcd_social_labels, _ = (\n                repair_disconnected_pcd_instances(\n                    pcd=pcd,\n                    assigned_labels=pcd_social_labels,\n                    skeleton_instances=skeleton_social,\n                    layer_name="SOCIAL",\n                    connect_radius=pcd_connect_radius,\n                    connect_radius_factor=pcd_connect_radius_factor,\n                    repair_iterations=component_repair_iterations,\n                    reassignment_neighbor_count=(\n                        reassignment_neighbor_count\n                    ),\n                    reassignment_max_distance_factor=(\n                        reassignment_max_distance_factor\n                    ),\n                    reassignment_min_vote_fraction=(\n                        reassignment_min_vote_fraction\n                    ),\n                    reassignment_min_other_neighbors=(\n                        reassignment_min_other_neighbors\n                    ),\n                    component_vote_sample_points=(\n                        component_vote_sample_points\n                    ),\n                    workers=workers,\n                )\n            )\n            voxel_social_rgb, _, _ = assign_voxel_from_segmented_pcd(\n                voxel=voxel,\n                pcd=pcd,\n                pcd_rgb=pcd_social_rgb,\n                pcd_labels=pcd_social_labels,\n                layer_name="SOCIAL",\n                workers=workers,\n            )\n\n            # 两个PCD文件最后一列都使用同一套最终social instance ID。\n            # physical文件的RGB仍表示physical edge，不额外保存physical ID。\n            save_pcd_xyzrgbnormal_social_id(\n                save_path=save_pcd_phy_path,\n                pcd=pcd,\n                rgb=pcd_phy_rgb,\n                social_instance_ids=pcd_social_labels,\n            )\n            save_pcd_xyzrgbnormal_social_id(\n                save_path=save_pcd_social_path,\n                pcd=pcd,\n                rgb=pcd_social_rgb,\n                social_instance_ids=pcd_social_labels,\n            )\n            save_voxel_xyzrgb_intensity(\n                save_path=save_voxel_social_path,\n                voxel=voxel,\n                rgb=voxel_social_rgb,\n            )\n\n            print("PHYSICAL and SOCIAL PCD layers saved:")\n            print(f"    {save_pcd_phy_path}")\n            print("    Format: x,y,z,r,g,b,nx,ny,nz,social_instance_id")\n            print(f"    {save_pcd_social_path}")\n            print("    Format: x,y,z,r,g,b,nx,ny,nz,social_instance_id")\n            print("SOCIAL voxel layer saved:")\n            print(f"    {save_voxel_social_path}")\n            print("    Format: x,y,z,r,g,b,intensity")\n\n            total_done += 1\n            print()\n            print(f"[DONE] Dual-layer mapping completed for: {stem}")\n\n    print("=========================================================================")\n    print("Dual-layer mitochondrial instance mapping finished.")\n    print(f"Images processed : {total_done}")\n    print(f"Images skipped   : {total_skipped}")\n    print("Saved files:")\n    print("    *_voxel_instance_phy.txt")\n    print("    *_voxel_instance_social.txt")\n    print("    *_pcd_instance_physical.txt")\n    print("    *_pcd_instance_social.txt")\n    print("Both PCD files end with the same social_instance_id column.")\n    print("No separate mapping CSV or physical ID column was saved.")\n    print("=========================================================================")\n'

LINKNEX_INSTANCE_ALGO03_SOURCE = 'from pathlib import Path\nimport os\n\nimport numpy as np\nimport pandas as pd\nfrom scipy.sparse import csr_matrix\nfrom scipy.sparse.csgraph import connected_components\nfrom scipy.spatial import cKDTree\n\n\nINSTANCE_COLUMNS = [\n    "image_name",\n    "instance_id",\n    "r",\n    "g",\n    "b",\n    "voxel_count",\n    "pcd_surface_point_count",\n    "volume",\n    "surface_area_voxel_faces",\n    "surface_volume_ratio",\n    "equivalent_sphere_diameter",\n    "equivalent_sphere_radius",\n    "centroid_x",\n    "centroid_y",\n    "centroid_z",\n    "major_axis_length_pca",\n    "middle_axis_length_pca",\n    "minor_axis_length_pca",\n    "mean_width_pca",\n    "aspect_ratio_major_minor",\n    "length_width_ratio",\n    "elongation_major_middle",\n    "flatness_middle_minor",\n    "pca_eig_major",\n    "pca_eig_middle",\n    "pca_eig_minor",\n    "pca_linearity",\n    "pca_planarity",\n    "pca_sphericity_like",\n    "pca_anisotropy",\n    "radius_of_gyration",\n    "sphericity",\n    "compactness_3d",\n    "intensity_mean",\n    "intensity_median",\n    "intensity_std",\n    "intensity_min",\n    "intensity_max",\n    "intensity_sum",\n]\n\n\nDATASET_SHORT_NAMES = {\n    "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif": "MCF-7",\n    "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif": "AC16",\n    "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif": "MDA",\n    "20260207_fibro_mito488_lyso640_nuc405_tif": "Fibroblast",\n    "20250917_h28_mitotracker488_lysotracker640_nuc405": "H28",\n    "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst": "HeLa",\n    "20250909_u2os_mitotracker488_lysotracker640_nuc405": "U2OS",\n    "20250905_msto_mitotracekr488_lysotracker640_nuc405": "MSTO",\n}\n\n\ndef load_txt_array(path):\n    """读取逗号分隔、无表头的数值txt。"""\n    data = np.loadtxt(path, delimiter=",", dtype=np.float64)\n    if data.ndim == 1:\n        data = data.reshape(1, -1)\n    return data\n\n\ndef check_minimum_columns(label, data, minimum_columns):\n    if data.ndim != 2 or data.shape[1] < minimum_columns:\n        raise ValueError(\n            f"{label} must contain at least {minimum_columns} columns, "\n            f"but its shape is {data.shape}."\n        )\n\n\ndef print_file_information(label, path, data, column_format):\n    print(f"{label} path   : {path}")\n    print(f"{label} shape  : {data.shape}")\n    print(f"{label} format : {column_format}")\n\n\ndef safe_ratio(numerator, denominator):\n    denominator = float(denominator)\n    if abs(denominator) < 1e-12:\n        return np.nan\n    return float(numerator) / denominator\n\n\ndef rgb_to_key(rgb):\n    """把0-255 RGB编码成一个整数，仅用于内部匹配。"""\n    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int64)\n    return rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]\n\n\ndef split_voxel_instances_by_rgb_and_connectivity(\n    voxel_xyz,\n    voxel_rgb,\n    connect_radius=1.8,\n):\n    """\n    使用RGB和空间连通性共同定义social instance。\n\n    这样即使两个远处线粒体意外复用了相同RGB，也不会被合并为同一个\n    morphology instance。\n    """\n    if connect_radius <= 0:\n        raise ValueError("connect_radius must be positive.")\n\n    point_count = len(voxel_xyz)\n    if point_count == 0:\n        raise ValueError("No voxel points were provided.")\n\n    color_keys = rgb_to_key(voxel_rgb)\n    tree = cKDTree(voxel_xyz)\n    neighbor_pairs = tree.query_pairs(\n        r=connect_radius,\n        output_type="ndarray",\n    )\n\n    if len(neighbor_pairs) > 0:\n        same_color = (\n            color_keys[neighbor_pairs[:, 0]]\n            == color_keys[neighbor_pairs[:, 1]]\n        )\n        same_color_pairs = neighbor_pairs[same_color]\n    else:\n        same_color_pairs = np.empty((0, 2), dtype=np.int64)\n\n    if len(same_color_pairs) == 0:\n        raw_components = np.arange(point_count, dtype=np.int64)\n    else:\n        graph_rows = np.concatenate(\n            [same_color_pairs[:, 0], same_color_pairs[:, 1]]\n        )\n        graph_columns = np.concatenate(\n            [same_color_pairs[:, 1], same_color_pairs[:, 0]]\n        )\n        graph = csr_matrix(\n            (\n                np.ones(len(graph_rows), dtype=np.uint8),\n                (graph_rows, graph_columns),\n            ),\n            shape=(point_count, point_count),\n        )\n        _, raw_components = connected_components(\n            graph,\n            directed=False,\n            return_labels=True,\n        )\n\n    # 按每个component首次出现的行号重新编号，保证结果稳定且从1开始。\n    component_values, first_rows = np.unique(\n        raw_components,\n        return_index=True,\n    )\n    component_order = component_values[np.argsort(first_rows)]\n    remap = np.empty(np.max(component_order) + 1, dtype=np.int64)\n    remap[component_order] = np.arange(1, len(component_order) + 1)\n    instance_ids = remap[raw_components]\n\n    unique_rgb_count = len(np.unique(color_keys))\n    repeated_color_component_count = len(component_order) - unique_rgb_count\n\n    print("Social voxel instance reconstruction:")\n    print(f"    Voxel points                 : {point_count}")\n    print(f"    Connectivity radius          : {connect_radius}")\n    print(f"    Neighbor pairs               : {len(neighbor_pairs)}")\n    print(f"    Same-RGB neighbor pairs      : {len(same_color_pairs)}")\n    print(f"    Unique RGB values            : {unique_rgb_count}")\n    print(f"    Spatially connected instances: {len(component_order)}")\n    print(\n        "    Extra components from reused/disconnected RGB: "\n        f"{repeated_color_component_count}"\n    )\n\n    return instance_ids\n\n\ndef map_pcd_to_voxel_instances_by_rgb(\n    pcd_xyz,\n    pcd_rgb,\n    voxel_xyz,\n    voxel_rgb,\n    voxel_instance_ids,\n    workers=-1,\n):\n    """\n    每个PCD点只在相同RGB的voxel中寻找最近点，取得内部instance_id。\n\n    instance_id只用于Morphology计算，不写回xyzrgb点云。\n    """\n    pcd_keys = rgb_to_key(pcd_rgb)\n    voxel_keys = rgb_to_key(voxel_rgb)\n    pcd_instance_ids = np.zeros(len(pcd_xyz), dtype=np.int64)\n    matched_distances = np.full(len(pcd_xyz), np.nan, dtype=np.float64)\n    missing_color_point_count = 0\n\n    voxel_rows_by_color = {\n        int(color_key): np.where(voxel_keys == color_key)[0]\n        for color_key in np.unique(voxel_keys)\n    }\n\n    for color_key in np.unique(pcd_keys):\n        pcd_rows = np.where(pcd_keys == color_key)[0]\n        voxel_rows = voxel_rows_by_color.get(int(color_key))\n        if voxel_rows is None or len(voxel_rows) == 0:\n            missing_color_point_count += len(pcd_rows)\n            continue\n\n        one_color_tree = cKDTree(voxel_xyz[voxel_rows])\n        distances, nearest_local_rows = one_color_tree.query(\n            pcd_xyz[pcd_rows],\n            k=1,\n            workers=workers,\n        )\n        nearest_global_rows = voxel_rows[nearest_local_rows]\n        pcd_instance_ids[pcd_rows] = voxel_instance_ids[nearest_global_rows]\n        matched_distances[pcd_rows] = distances\n\n    if missing_color_point_count > 0:\n        raise ValueError(\n            f"{missing_color_point_count} PCD points have RGB values that "\n            "do not exist in the social voxel instance file."\n        )\n\n    print("PCD-to-voxel social instance matching:")\n    print(f"    PCD points matched        : {len(pcd_xyz)}")\n    print(f"    PCD points missing color  : {missing_color_point_count}")\n    print(\n        "    Median PCD-to-voxel distance: "\n        f"{np.nanmedian(matched_distances):.4f}"\n    )\n    print(\n        "    Maximum PCD-to-voxel distance: "\n        f"{np.nanmax(matched_distances):.4f}"\n    )\n    return pcd_instance_ids\n\n\ndef voxel_surface_area_from_faces(\n    voxel_xyz,\n    voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),\n    voxel_size_xyz=(1.0, 1.0, 1.0),\n):\n    """通过统计暴露的6邻域voxel表面计算三维表面积。"""\n    if len(voxel_xyz) == 0:\n        return 0.0\n\n    grid_spacing = np.asarray(\n        voxel_grid_spacing_xyz,\n        dtype=np.float64,\n    )\n    if np.any(grid_spacing <= 0):\n        raise ValueError("voxel_grid_spacing_xyz values must be positive.")\n\n    grid = np.unique(\n        np.rint(voxel_xyz / grid_spacing[None, :]).astype(np.int64),\n        axis=0,\n    )\n    occupied = set(map(tuple, grid.tolist()))\n\n    sx, sy, sz = map(float, voxel_size_xyz)\n    directions = (\n        ((1, 0, 0), sy * sz),\n        ((-1, 0, 0), sy * sz),\n        ((0, 1, 0), sx * sz),\n        ((0, -1, 0), sx * sz),\n        ((0, 0, 1), sx * sy),\n        ((0, 0, -1), sx * sy),\n    )\n\n    surface_area = 0.0\n    for x, y, z in occupied:\n        for (dx, dy, dz), face_area in directions:\n            if (x + dx, y + dy, z + dz) not in occupied:\n                surface_area += face_area\n    return float(surface_area)\n\n\ndef calculate_social_instance_morphology(\n    image_name,\n    voxel,\n    pcd,\n    voxel_connect_radius=1.8,\n    voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),\n    voxel_size_xyz=(1.0, 1.0, 1.0),\n    workers=-1,\n):\n    """计算一个图片中每个social mitochondrial object的典型Morphology。"""\n    check_minimum_columns(\n        "SOCIAL VOXEL INSTANCE",\n        voxel,\n        minimum_columns=7,\n    )\n    check_minimum_columns(\n        "SOCIAL PCD INSTANCE",\n        pcd,\n        minimum_columns=9,\n    )\n\n    voxel_xyz = voxel[:, :3]\n    voxel_rgb = np.clip(np.rint(voxel[:, 3:6]), 0, 255).astype(np.int32)\n    voxel_intensity = voxel[:, 6]\n    pcd_xyz = pcd[:, :3]\n    pcd_rgb = np.clip(np.rint(pcd[:, 3:6]), 0, 255).astype(np.int32)\n\n    voxel_valid = (\n        np.isfinite(voxel_xyz).all(axis=1)\n        & np.isfinite(voxel[:, 3:7]).all(axis=1)\n        & np.any(voxel_rgb != 0, axis=1)\n    )\n    pcd_valid = (\n        np.isfinite(pcd_xyz).all(axis=1)\n        & np.isfinite(pcd[:, 3:9]).all(axis=1)\n        & np.any(pcd_rgb != 0, axis=1)\n    )\n    voxel_xyz = voxel_xyz[voxel_valid]\n    voxel_rgb = voxel_rgb[voxel_valid]\n    voxel_intensity = voxel_intensity[voxel_valid]\n    pcd_xyz = pcd_xyz[pcd_valid]\n    pcd_rgb = pcd_rgb[pcd_valid]\n\n    if len(voxel_xyz) == 0 or len(pcd_xyz) == 0:\n        raise ValueError("No valid colored social voxel/PCD points remain.")\n\n    voxel_instance_ids = split_voxel_instances_by_rgb_and_connectivity(\n        voxel_xyz=voxel_xyz,\n        voxel_rgb=voxel_rgb,\n        connect_radius=voxel_connect_radius,\n    )\n    pcd_instance_ids = map_pcd_to_voxel_instances_by_rgb(\n        pcd_xyz=pcd_xyz,\n        pcd_rgb=pcd_rgb,\n        voxel_xyz=voxel_xyz,\n        voxel_rgb=voxel_rgb,\n        voxel_instance_ids=voxel_instance_ids,\n        workers=workers,\n    )\n\n    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float64)\n    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):\n        raise ValueError(\n            "voxel_size_xyz must contain three positive values in x,y,z order."\n        )\n    voxel_volume = float(np.prod(voxel_size))\n    rows = []\n\n    for instance_id in np.unique(voxel_instance_ids):\n        voxel_mask = voxel_instance_ids == instance_id\n        pcd_mask = pcd_instance_ids == instance_id\n        points_voxel = voxel_xyz[voxel_mask]\n        points = points_voxel * voxel_size[None, :]\n        intensity = voxel_intensity[voxel_mask]\n        color = voxel_rgb[np.where(voxel_mask)[0][0]]\n        voxel_count = int(len(points_voxel))\n        volume = float(voxel_count * voxel_volume)\n        centroid = np.mean(points, axis=0)\n        centered = points - centroid[None, :]\n        radius_of_gyration = float(\n            np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))\n        )\n\n        if voxel_count >= 2:\n            covariance = np.cov(centered, rowvar=False, ddof=1)\n            covariance = np.atleast_2d(covariance).astype(np.float64)\n            if covariance.shape != (3, 3):\n                covariance = np.zeros((3, 3), dtype=np.float64)\n            try:\n                eigvals, eigvecs = np.linalg.eigh(covariance)\n                eig_order = np.argsort(eigvals)[::-1]\n                eigvals = np.maximum(eigvals[eig_order], 0.0)\n                eigvecs = eigvecs[:, eig_order]\n                projected = centered @ eigvecs\n                axis_lengths = np.ptp(projected, axis=0)\n            except np.linalg.LinAlgError:\n                eigvals = np.zeros(3, dtype=np.float64)\n                axis_lengths = np.zeros(3, dtype=np.float64)\n        else:\n            eigvals = np.zeros(3, dtype=np.float64)\n            axis_lengths = np.zeros(3, dtype=np.float64)\n\n        major_axis, middle_axis, minor_axis = map(float, axis_lengths)\n        eig_major, eig_middle, eig_minor = map(float, eigvals)\n        mean_width = (middle_axis + minor_axis) / 2.0\n\n        surface_area = voxel_surface_area_from_faces(\n            voxel_xyz=points_voxel,\n            voxel_grid_spacing_xyz=voxel_grid_spacing_xyz,\n            voxel_size_xyz=voxel_size_xyz,\n        )\n        equivalent_sphere_diameter = float(\n            (6.0 * volume / np.pi) ** (1.0 / 3.0)\n        )\n        equivalent_sphere_radius = equivalent_sphere_diameter / 2.0\n\n        if surface_area > 0 and volume > 0:\n            sphericity = float(\n                (np.pi ** (1.0 / 3.0))\n                * ((6.0 * volume) ** (2.0 / 3.0))\n                / surface_area\n            )\n            compactness_3d = float(\n                36.0 * np.pi * volume ** 2 / surface_area ** 3\n            )\n        else:\n            sphericity = np.nan\n            compactness_3d = np.nan\n\n        rows.append(\n            {\n                "image_name": image_name,\n                "instance_id": int(instance_id),\n                "r": int(color[0]),\n                "g": int(color[1]),\n                "b": int(color[2]),\n                "voxel_count": voxel_count,\n                "pcd_surface_point_count": int(np.sum(pcd_mask)),\n                "volume": volume,\n                "surface_area_voxel_faces": surface_area,\n                "surface_volume_ratio": safe_ratio(surface_area, volume),\n                "equivalent_sphere_diameter": equivalent_sphere_diameter,\n                "equivalent_sphere_radius": equivalent_sphere_radius,\n                "centroid_x": float(centroid[0]),\n                "centroid_y": float(centroid[1]),\n                "centroid_z": float(centroid[2]),\n                "major_axis_length_pca": major_axis,\n                "middle_axis_length_pca": middle_axis,\n                "minor_axis_length_pca": minor_axis,\n                "mean_width_pca": mean_width,\n                "aspect_ratio_major_minor": safe_ratio(\n                    major_axis,\n                    minor_axis,\n                ),\n                "length_width_ratio": safe_ratio(major_axis, mean_width),\n                "elongation_major_middle": safe_ratio(\n                    major_axis,\n                    middle_axis,\n                ),\n                "flatness_middle_minor": safe_ratio(\n                    middle_axis,\n                    minor_axis,\n                ),\n                "pca_eig_major": eig_major,\n                "pca_eig_middle": eig_middle,\n                "pca_eig_minor": eig_minor,\n                "pca_linearity": safe_ratio(\n                    eig_major - eig_middle,\n                    eig_major,\n                ),\n                "pca_planarity": safe_ratio(\n                    eig_middle - eig_minor,\n                    eig_major,\n                ),\n                "pca_sphericity_like": safe_ratio(eig_minor, eig_major),\n                "pca_anisotropy": safe_ratio(\n                    eig_major - eig_minor,\n                    eig_major,\n                ),\n                "radius_of_gyration": radius_of_gyration,\n                "sphericity": sphericity,\n                "compactness_3d": compactness_3d,\n                "intensity_mean": float(np.mean(intensity)),\n                "intensity_median": float(np.median(intensity)),\n                "intensity_std": float(np.std(intensity)),\n                "intensity_min": float(np.min(intensity)),\n                "intensity_max": float(np.max(intensity)),\n                "intensity_sum": float(np.sum(intensity)),\n            }\n        )\n\n    morphology_df = pd.DataFrame(rows, columns=INSTANCE_COLUMNS)\n    morphology_df = morphology_df.sort_values("instance_id").reset_index(\n        drop=True\n    )\n    print("Social instance morphology calculation finished:")\n    print(f"    Image     : {image_name}")\n    print(f"    Instances : {len(morphology_df)}")\n    print(f"    Columns   : {len(morphology_df.columns)}")\n    print("    BBox columns included: 0")\n    return morphology_df\n\n\ndef build_one_image_summary(image_name, morphology_df):\n    """把一个图片的instance级Morphology汇总成一行。"""\n    if morphology_df.empty:\n        raise ValueError(f"Empty morphology table for {image_name}.")\n\n    row = {\n        "image_name": image_name,\n        "n_instances": int(len(morphology_df)),\n        "total_voxel_count": float(morphology_df["voxel_count"].sum()),\n        "total_pcd_surface_point_count": float(\n            morphology_df["pcd_surface_point_count"].sum()\n        ),\n        "total_volume": float(morphology_df["volume"].sum()),\n        "total_surface_area_voxel_faces": float(\n            morphology_df["surface_area_voxel_faces"].sum()\n        ),\n        "total_intensity_sum": float(morphology_df["intensity_sum"].sum()),\n    }\n\n    excluded = {\n        "instance_id",\n        "r",\n        "g",\n        "b",\n        "centroid_x",\n        "centroid_y",\n        "centroid_z",\n    }\n    numeric_columns = morphology_df.select_dtypes(\n        include=[np.number]\n    ).columns\n    for column in numeric_columns:\n        if column in excluded:\n            continue\n        values = morphology_df[column]\n        row[f"mean_{column}"] = float(values.mean(skipna=True))\n        row[f"median_{column}"] = float(values.median(skipna=True))\n    return row\n\n\ndef build_dataset_summary(output_path):\n    """汇总目录中所有social morphology CSV；每张图片一行。"""\n    morphology_paths = sorted(\n        output_path.glob("*_instance_morphology_social.csv"),\n        key=lambda path: path.name.lower(),\n    )\n    summary_rows = []\n\n    print("-------------------------------------------------------------------------")\n    print("Building dataset-level social Morphology summary...")\n    print(f"Morphology CSV files found: {len(morphology_paths)}")\n\n    suffix = "_instance_morphology_social.csv"\n    for morphology_path in morphology_paths:\n        morphology_df = pd.read_csv(morphology_path)\n        if morphology_df.empty:\n            print(f"[SUMMARY SKIPPED] Empty CSV: {morphology_path}")\n            continue\n        image_name = morphology_path.name[:-len(suffix)]\n        summary_rows.append(\n            build_one_image_summary(\n                image_name=image_name,\n                morphology_df=morphology_df,\n            )\n        )\n\n    summary_path = output_path / "all_images_morphology_social_summary.csv"\n    summary_df = pd.DataFrame(summary_rows)\n    summary_df.to_csv(summary_path, index=False)\n    print("Dataset summary saved:")\n    print(f"    {summary_path}")\n    print(f"    Images summarized: {len(summary_df)}")\n    print("    One row = one image/cell")\n    return summary_df, summary_path\n\n\ndef main():\n\n    #\n    # file_name_list = [\n    #\n    #                         # ====================================================\n    #     # MCF-7\n    #     # # ====================================================\n    #     "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # AC16\n    #     # # ====================================================\n    #     "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # MDA\n    #     # # ====================================================\n    #     "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",\n    #     # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",\n    #     #\n    #     # # ====================================================\n    #     # # Fibroblast\n    #     # # ====================================================\n    #     "20260207_fibro_mito488_lyso640_nuc405_tif",\n    #     # "20251216_fibro_mito488tracker_ld561bodipy_nuc405hurst",\n    #     #\n    #     # # ====================================================\n    #     # # H28\n    #     # # ====================================================\n    #     "20250917_h28_mitotracker488_lysotracker640_nuc405",\n    #     # "20251213_h28_mito488_ld561_nuc405",\n    #     #\n    #     # # ====================================================\n    #     # # HeLa\n    #     # # ====================================================\n    #     "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",\n    #     # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",\n    #\n    #     # ====================================================\n    #     # U2OS\n    #     # ====================================================\n    #     # "20251201_u2os_ld488_mito640_nuc405",\n    #     "20250909_u2os_mitotracker488_lysotracker640_nuc405",\n    #\n    #     # ====================================================\n    #     # MSTO\n    #     # ====================================================\n    #     # "20251127_msto_ld488_mito640_nuc405",\n    #     "20250905_msto_mitotracekr488_lysotracker640_nuc405",\n    #\n    #                 ]\n\n    file_name_list = [\n\n        # ====================================================\n        # GLUC\n        # ====================================================\n        # "gluc_mito-mChGFP_LTRDR50nM_0031",\n        # "gluc_mito-mChGFP_LTRDR50nM_0032",\n        # "gluc_mito-mChGFP_LTRDR50nM_0051",\n        # "gluc_mito-mChGFP_LTRDR50nM_0061",\n        # "gluc_mito-mChGFP_LTRDR50nM_0071",\n        # "gluc_mito-mChGFP_LTRDR50nM_0072",\n        # "gluc_mito-mChGFP_LTRDR50nM_0081",\n        # "gluc_mito-mChGFP_LTRDR50nM_0082",\n        # "gluc_mito-mChGFP_LTRDR50nM_0091",\n        #\n        # # ====================================================\n        # # Non-GLUC\n        # # ====================================================\n        #\n        # "mito-mChGFP_LTRDR50nM_0021",\n        # "mito-mChGFP_LTRDR50nM_0022",\n        # "mito-mChGFP_LTRDR50nM_0031",\n        # "mito-mChGFP_LTRDR50nM_0032",\n        # "mito-mChGFP_LTRDR50nM_0041",\n        # "mito-mChGFP_LTRDR50nM_0042",\n        # "mito-mChGFP_LTRDR50nM_0051",\n        # "mito-mChGFP_LTRDR50nM_0061",\n        # "mito-mChGFP_LTRDR50nM_0062",\n        # "mito-mChGFP_LTRDR50nM_0071",\n        # "mito-mChGFP_LTRDR50nM_0072",\n        "mito-mChGFP_LTRDR50nM_0081",\n\n    ]\n\n    # 所有 8 个数据集共同根目录。\n    # DATA_ROOT = Path(\n    #     "D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis"\n    # )\n\n    DATA_ROOT = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")\n\n\n\n    # 收集每个数据集的一行-per-image summary，最后再汇总成 8-type 总表。\n    all_dataset_summary_tables = []\n\n    ###########################################################################\n    # 参数：不使用argparse，直接在这里修改\n    ###########################################################################\n\n    # xyz坐标中相邻voxel之间的坐标间隔。\n    voxel_grid_spacing_xyz = (1.0, 1.0, 1.0)\n\n    # 一个voxel在真实物理空间中的x,y,z尺寸。\n    # 保持(1,1,1)时，体积和长度使用voxel单位。\n    # 如果需要微米单位，可改为实际值，例如(0.07857, 0.07857, 0.16)。\n    voxel_size_xyz = (1.0, 1.0, 1.0)\n\n    # 同RGB voxel的连通半径。1.8覆盖规则网格的26邻域。\n    voxel_connect_radius = 1.8\n\n    # False时，如果单图Morphology CSV已存在则直接复用，不重复计算。\n    overwrite_existing = False\n\n    # cKDTree并行线程数；-1表示使用全部可用CPU线程。\n    workers = -1\n\n    total_done = 0\n    total_reused = 0\n    total_skipped = 0\n\n    for file_name in file_name_list:\n        # BASE_DIR = Path(f"D:/public_data/S-BIAD1445/neuron/mito_test")\n        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")\n        # BASE_DIR = DATA_ROOT / file_name / "mito"\n        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")\n        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")\n        # BASE_DIR = Path(f"E:/data/20260811-YF_output/image_tif/single_cell_tifs")\n        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")\n\n        INPUT_PATH = BASE_DIR / "processed_output"\n        OUTPUT_PATH = BASE_DIR / "processed_output"\n\n        if not INPUT_PATH.exists():\n            raise FileNotFoundError(\n                f"processed_output folder not found: {INPUT_PATH}"\n            )\n        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)\n\n        print(os.listdir(INPUT_PATH))\n\n        #######################################################################\n        # 图片列表\n        #######################################################################\n\n        # 自动寻找全部 *_voxel_instance_social.txt。\n        suffix = "_voxel_instance_social.txt"\n        imgs_list = [\n            path.name[:-len(suffix)]\n            for path in sorted(\n                INPUT_PATH.glob(f"*{suffix}"),\n                key=lambda path: path.name.lower(),\n            )\n        ]\n\n        # 手动指定需要处理的图片。\n        # 需要处理全部图片时，将下面这一行注释掉即可。\n        # imgs_list = ["fibro_mito488_lyso640_nuc405_17689_5ADVMLE"]\n\n        if len(imgs_list) == 0:\n            print(f"No social voxel instance files found in {INPUT_PATH}")\n            continue\n\n        print("=========================================================================")\n        print(f"Dataset               : {file_name}")\n        print(f"BASE_DIR              : {BASE_DIR}")\n        print(f"INPUT_PATH            : {INPUT_PATH}")\n        print(f"OUTPUT_PATH           : {OUTPUT_PATH}")\n        print(f"Selected images       : {len(imgs_list)}")\n        print(f"Voxel grid spacing xyz: {voxel_grid_spacing_xyz}")\n        print(f"Voxel physical size xyz: {voxel_size_xyz}")\n        print(f"Voxel connect radius  : {voxel_connect_radius}")\n        print(f"Overwrite existing    : {overwrite_existing}")\n        print("=========================================================================")\n\n        for index, image_name in enumerate(imgs_list, start=1):\n            voxel_path = (\n                INPUT_PATH / f"{image_name}_voxel_instance_social.txt"\n            )\n            pcd_path = INPUT_PATH / f"{image_name}_pcd_instance_social.txt"\n            morphology_path = (\n                OUTPUT_PATH\n                / f"{image_name}_instance_morphology_social.csv"\n            )\n\n            print("-------------------------------------------------------------------------")\n            print(\n                f"Processing {file_name} | {index}/{len(imgs_list)} | "\n                f"{image_name}"\n            )\n            print("-------------------------------------------------------------------------")\n            print("Input files:")\n            print(f"    SOCIAL VOXEL: {voxel_path}")\n            print(f"    SOCIAL PCD  : {pcd_path}")\n            print("Output file:")\n            print(f"    MORPHOLOGY  : {morphology_path}")\n\n            if morphology_path.exists() and not overwrite_existing:\n                print("[REUSED] Existing social Morphology CSV:")\n                print(f"    {morphology_path}")\n                total_reused += 1\n                continue\n\n            missing_paths = [\n                path for path in (voxel_path, pcd_path) if not path.exists()\n            ]\n            if len(missing_paths) > 0:\n                print("[SKIPPED] Missing required input files:")\n                for missing_path in missing_paths:\n                    print(f"    {missing_path}")\n                total_skipped += 1\n                continue\n\n            voxel = load_txt_array(voxel_path)\n            pcd = load_txt_array(pcd_path)\n\n            print_file_information(\n                "SOCIAL VOXEL",\n                voxel_path,\n                voxel,\n                "x,y,z,r,g,b,intensity",\n            )\n            print_file_information(\n                "SOCIAL PCD",\n                pcd_path,\n                pcd,\n                "x,y,z,r,g,b,nx,ny,nz",\n            )\n\n            morphology_df = calculate_social_instance_morphology(\n                image_name=image_name,\n                voxel=voxel,\n                pcd=pcd,\n                voxel_connect_radius=voxel_connect_radius,\n                voxel_grid_spacing_xyz=voxel_grid_spacing_xyz,\n                voxel_size_xyz=voxel_size_xyz,\n                workers=workers,\n            )\n            morphology_df.to_csv(morphology_path, index=False)\n            print("Social instance Morphology saved:")\n            print(f"    {morphology_path}")\n            print("    One row = one social mitochondrial object")\n            total_done += 1\n\n        dataset_summary_df, dataset_summary_path = build_dataset_summary(\n            OUTPUT_PATH\n        )\n\n        if not dataset_summary_df.empty:\n            dataset_summary_df = dataset_summary_df.copy()\n            dataset_summary_df.insert(0, "dataset_name", file_name)\n            dataset_summary_df.insert(\n                1,\n                "cell_type",\n                DATASET_SHORT_NAMES.get(file_name, file_name),\n            )\n            all_dataset_summary_tables.append(dataset_summary_df)\n\n            # 单数据集 summary 也写入 cell type metadata，方便独立查看。\n            dataset_summary_df.to_csv(\n                OUTPUT_PATH / "all_images_morphology_social_summary_with_dataset.csv",\n                index=False,\n            )\n\n    ###########################################################################\n    # 8-type 总体 Morphology summary\n    ###########################################################################\n    if len(all_dataset_summary_tables) == 0:\n        raise RuntimeError(\n            "No dataset-level morphology summary was successfully generated."\n        )\n\n    all_8types_morphology = pd.concat(\n        all_dataset_summary_tables,\n        axis=0,\n        ignore_index=True,\n        sort=False,\n    )\n\n    all_8types_morphology_path = (\n        DATA_ROOT / "all_8types_morphology_social_summary.csv"\n    )\n    all_8types_morphology.to_csv(\n        all_8types_morphology_path,\n        index=False,\n    )\n\n    # 同时尝试与 fixed hierarchical network metrics 自动合并。\n    # merge key = dataset_name + image_name，避免不同数据集重名时错误匹配。\n    network_candidates = [\n        DATA_ROOT / "all_8types_hierarchical_network_metrics.csv",\n        DATA_ROOT\n        / "all_8types_hierarchical_network_metrics_FIXED_UNFILTERED.csv",\n        DATA_ROOT\n        / "all_8types_hierarchical_network_metrics_unfiltered.csv",\n    ]\n    network_path = next(\n        (path for path in network_candidates if path.exists()),\n        None,\n    )\n\n    if network_path is not None:\n        network_df = pd.read_csv(network_path)\n\n        required_merge_columns = {"dataset_name", "image_name"}\n        if required_merge_columns.issubset(network_df.columns):\n            merged = all_8types_morphology.merge(\n                network_df,\n                on=["dataset_name", "image_name"],\n                how="inner",\n                suffixes=("_morphology", "_network"),\n                validate="one_to_one",\n            )\n\n            merged_path = (\n                DATA_ROOT\n                / "all_8types_morphology_plus_hierarchical_network.csv"\n            )\n            merged.to_csv(merged_path, index=False)\n\n            morphology_only = set(\n                zip(\n                    all_8types_morphology["dataset_name"],\n                    all_8types_morphology["image_name"],\n                )\n            ) - set(\n                zip(\n                    merged["dataset_name"],\n                    merged["image_name"],\n                )\n            )\n            network_only = set(\n                zip(\n                    network_df["dataset_name"],\n                    network_df["image_name"],\n                )\n            ) - set(\n                zip(\n                    merged["dataset_name"],\n                    merged["image_name"],\n                )\n            )\n\n            merge_qc = pd.DataFrame(\n                [\n                    {\n                        "morphology_rows": len(all_8types_morphology),\n                        "network_rows": len(network_df),\n                        "matched_rows": len(merged),\n                        "morphology_only_rows": len(morphology_only),\n                        "network_only_rows": len(network_only),\n                        "network_source": str(network_path),\n                    }\n                ]\n            )\n            merge_qc_path = (\n                DATA_ROOT / "all_8types_morphology_network_merge_QC.csv"\n            )\n            merge_qc.to_csv(merge_qc_path, index=False)\n\n            print("8-type Morphology + Network merged table saved:")\n            print(f"    {merged_path}")\n            print(\n                f"    Matched rows: {len(merged)} / "\n                f"Morphology {len(all_8types_morphology)} / "\n                f"Network {len(network_df)}"\n            )\n        else:\n            print(\n                "[WARNING] Network summary exists but lacks "\n                "dataset_name and/or image_name; merge skipped."\n            )\n    else:\n        print(\n            "[INFO] No 8-type network summary found yet. "\n            "Morphology summary was still saved normally."\n        )\n\n    print("=========================================================================")\n    print("Social mitochondrial Morphology calculation finished.")\n    print(f"Images calculated : {total_done}")\n    print(f"Images reused     : {total_reused}")\n    print(f"Images skipped    : {total_skipped}")\n    print("Per-image output  : *_instance_morphology_social.csv")\n    print("Dataset summary   : all_images_morphology_social_summary.csv")\n    print(f"8-type summary    : {all_8types_morphology_path}")\n    print("BBox metrics included: 0")\n    print("=========================================================================")\n'


# =============================================================================
# Embedded Instance Segmentation backend
# =============================================================================

# -*- coding: utf-8 -*-
"""
LinkODT Instance segmentation backend.

Pipeline
--------
*_mask.tif
 -> voxel / surface PCD / OBJ / skeleton / refined skeleton
 -> skeleton physical edges + social instances
 -> map skeleton instances to PCD + voxel
 -> social morphology CSV

The algorithmic segmentation stages call the user's existing LinkODT 01/02/03
implementations, which are bundled beside this file.
"""

from pathlib import Path
import importlib.util
import csv

import numpy as np
import pandas as pd
import tifffile


def _load_local_module(filename, module_name):
    """
    Load the user's original 01/02/03 instance algorithms from source strings
    embedded inside this single LinkODT main program.
    """
    import types

    source_map = {
        "linknex_instance_algo01.py": LINKNEX_INSTANCE_ALGO01_SOURCE,
        "linknex_instance_algo02.py": LINKNEX_INSTANCE_ALGO02_SOURCE,
        "linknex_instance_algo03.py": LINKNEX_INSTANCE_ALGO03_SOURCE,
    }

    source = source_map.get(str(filename))

    if source is None:
        raise FileNotFoundError(
            f"Unknown embedded LinkNex instance algorithm: {filename}"
        )

    module = types.ModuleType(module_name)
    module.__file__ = f"<embedded:{filename}>"

    exec(
        compile(
            source,
            module.__file__,
            "exec",
        ),
        module.__dict__,
    )

    return module


def _progress(callback, text, value=None):
    if callback is not None:
        callback(str(text), value)


def _check_stop(stop_callback):
    if stop_callback is not None:
        stop_callback()


def mask_stem(mask_path):
    p = Path(mask_path)
    name = p.name
    lower = name.lower()
    for suffix in ('_mask.tiff', '_mask.tif'):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError(
        f"Expected a semantic binary mask named *_mask.tif or *_mask.tiff, got: {name}"
    )


def find_source_image(mask_path):
    mask_path = Path(mask_path)
    stem = mask_stem(mask_path)
    candidates = [
        mask_path.with_name(stem + '.tif'),
        mask_path.with_name(stem + '.tiff'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def processed_output_dir(mask_path):
    p = Path(mask_path).parent / 'processed_output'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _squeeze_zyx(arr, label):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"{label} must be a 2D or 3D grayscale volume after squeeze; got {arr.shape}."
        )
    return arr


def _write_obj(path, vertices_xyz, faces, normals_xyz):
    path = Path(path)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('# LinkODT surface mesh generated from semantic mask\n')
        for x, y, z in vertices_xyz:
            handle.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for nx, ny, nz in normals_xyz:
            handle.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        # OBJ uses 1-based indices. Each marching-cubes vertex has a matching normal.
        for a, b, c in np.asarray(faces, dtype=np.int64):
            a += 1; b += 1; c += 1
            handle.write(f'f {a}//{a} {b}//{b} {c}//{c}\n')


def prepare_geometry_from_mask(
    mask_path,
    progress_callback=None,
    stop_callback=None,
):
    """Convert semantic mask to voxel/PCD/OBJ/skeleton inputs."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    from skimage.morphology import skeletonize

    mask_path = Path(mask_path)
    stem = mask_stem(mask_path)
    out_dir = processed_output_dir(mask_path)

    _check_stop(stop_callback)
    _progress(progress_callback, f'[1/5] Reading mask: {mask_path.name}', 2)

    mask_raw = tifffile.imread(mask_path)
    mask = _squeeze_zyx(mask_raw, 'Semantic mask') > 0

    if mask.shape[0] < 2:
        raise ValueError(
            'The LinkODT PCD/skeleton instance pipeline currently requires a 3D mask (Z >= 2). '
            f'This mask is {mask.shape}.'
        )
    if not np.any(mask):
        raise ValueError(f'{mask_path.name}: mask has no foreground voxels.')

    source_path = find_source_image(mask_path)
    if source_path is not None:
        source = _squeeze_zyx(tifffile.imread(source_path), 'Source image')
        if source.shape != mask.shape:
            raise ValueError(
                'Source image and mask dimensions do not match.\n\n'
                f'Image: {source.shape}\nMask: {mask.shape}\n{source_path}'
            )
    else:
        source = mask.astype(np.uint8)

    _check_stop(stop_callback)
    _progress(progress_callback, '[2/5] Creating voxel.txt...', 12)

    z, y, x = np.nonzero(mask)
    intensity = np.asarray(source[z, y, x], dtype=np.float64)
    voxel = np.column_stack([
        x.astype(np.float64),
        y.astype(np.float64),
        z.astype(np.float64),
        intensity,
    ])
    voxel_path = out_dir / f'{stem}_voxel.txt'
    np.savetxt(
        voxel_path,
        voxel,
        delimiter=',',
        fmt=['%.3f', '%.3f', '%.3f', '%.6f'],
    )

    _check_stop(stop_callback)
    _progress(progress_callback, '[3/5] Creating closed surface PCD + OBJ...', 28)

    # Pad with one zero layer so objects touching the TIFF border still get a closed surface.
    padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
    verts_zyx, faces, normals_zyx, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        allow_degenerate=False,
    )
    verts_zyx -= 1.0

    vertices_xyz = verts_zyx[:, [2, 1, 0]].astype(np.float64)
    normals_xyz = normals_zyx[:, [2, 1, 0]].astype(np.float64)
    normal_norm = np.linalg.norm(normals_xyz, axis=1, keepdims=True)
    normals_xyz = normals_xyz / np.maximum(normal_norm, 1e-12)

    pcd = np.column_stack([vertices_xyz, normals_xyz])
    pcd_path = out_dir / f'{stem}_pcd.txt'
    np.savetxt(
        pcd_path,
        pcd,
        delimiter=',',
        fmt=['%.6f'] * 6,
    )

    obj_path = out_dir / f'{stem}.obj'
    _write_obj(obj_path, vertices_xyz, faces, normals_xyz)

    _check_stop(stop_callback)
    _progress(progress_callback, '[4/5] Skeletonizing 3D mask...', 45)

    skel_mask = skeletonize(mask)
    sz, sy, sx = np.nonzero(skel_mask)
    if len(sx) == 0:
        raise ValueError('3D skeletonization produced zero skeleton points.')

    skeleton_xyz = np.column_stack([
        sx.astype(np.float64),
        sy.astype(np.float64),
        sz.astype(np.float64),
    ])

    distance_map = ndi.distance_transform_edt(mask)
    radius = distance_map[sz, sy, sx].astype(np.float64)

    tree = cKDTree(skeleton_xyz)
    neighbor_lists = tree.query_ball_point(skeleton_xyz, r=1.8)
    degree = np.asarray([max(0, len(n) - 1) for n in neighbor_lists], dtype=np.int32)
    point_type = np.where(degree > 2, 3, np.where(degree <= 1, 1, 2)).astype(np.int32)

    skeleton = np.column_stack([
        skeleton_xyz,
        radius,
        degree,
        point_type,
    ])
    skeleton_path = out_dir / f'{stem}_skeleton.txt'
    np.savetxt(
        skeleton_path,
        skeleton,
        delimiter=',',
        fmt=['%.6f', '%.6f', '%.6f', '%.6f', '%d', '%d'],
    )

    refined = np.column_stack([skeleton_xyz, radius])
    refined_path = out_dir / f'{stem}_refinedskelenton.txt'
    np.savetxt(refined_path, refined, delimiter=',', fmt='%.6f')

    _check_stop(stop_callback)
    _progress(
        progress_callback,
        (
            '[5/5] Geometry ready.\n'
            f'Voxel points: {len(voxel):,}\n'
            f'PCD points: {len(pcd):,}\n'
            f'Skeleton points: {len(skeleton):,}'
        ),
        55,
    )

    return {
        'stem': stem,
        'mask_path': mask_path,
        'source_path': source_path,
        'output_dir': out_dir,
        'voxel_path': voxel_path,
        'pcd_path': pcd_path,
        'obj_path': obj_path,
        'skeleton_path': skeleton_path,
        'refined_skeleton_path': refined_path,
    }


def run_skeleton_and_instance_mapping(
    geometry,
    progress_callback=None,
    stop_callback=None,
):
    """Run the existing LinkODT 01 + 02 instance algorithms."""
    algo1 = _load_local_module('linknex_instance_algo01.py', 'linknex_instance_algo01')
    algo2 = _load_local_module('linknex_instance_algo02.py', 'linknex_instance_algo02')

    stem = geometry['stem']
    out_dir = Path(geometry['output_dir'])

    pcd = algo1.load_txt_array(geometry['pcd_path'])
    voxel = algo1.load_txt_array(geometry['voxel_path'])
    refined = algo1.load_txt_array(geometry['refined_skeleton_path'])

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 1/6] Adding missing skeleton points...', 58)

    refined, _, _, _ = algo1.add_missing_refined_skeleton_from_small_pcd_components(
        pcd=pcd,
        refined_skeleton=refined,
        dbscan_eps=2.0,
        dbscan_min_samples=10,
        min_component_points=20,
        max_component_points=1000,
        solid_grid_spacing=1.0,
        local_padding=3,
        surface_dilation_iterations=1,
        surface_closing_iterations=1,
    )
    np.savetxt(geometry['refined_skeleton_path'], refined, delimiter=',', fmt='%.6f')

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 2/6] Segmenting refined skeleton edges...', 62)

    skeleton_edges = algo1.segment_refined_skeleton_edges(
        refined_skeleton=refined,
        neighbor_radius=1.8,
        random_seed=42,
        prune_shortcut_edges=True,
    )

    original_edges_path = out_dir / f'{stem}_skeleton_edges_original.txt'
    np.savetxt(
        original_edges_path,
        skeleton_edges,
        delimiter=',',
        fmt=['%.3f', '%.3f', '%.3f', '%d', '%d', '%d', '%d'],
    )

    cross_edges_path = out_dir / f'{stem}_skeleton_edges_cross_section.txt'
    cross_points_path = out_dir / f'{stem}_pcd_cross_sections_first_shell_local_drop_flat.txt'
    deletion_path = out_dir / f'{stem}_skeleton_edges_deletion_candidates_recovery.txt'

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 3/6] Detecting narrow false bridges...', 67)

    _, _, deletion_rows = algo1.detect_edges_with_abrupt_cross_section_narrowing(
        skeleton_edges=skeleton_edges,
        pcd=pcd,
        save_path=cross_edges_path,
        section_points_save_path=cross_points_path,
        deletion_points_save_path=deletion_path,
        neighbor_radius=1.8,
        tangent_window=2,
        cross_section_half_thickness=1.0,
        cross_section_max_radius=8.0,
        min_section_points=1,
        section_radius_quantile=0.75,
        smoothing_window=1,
        endpoint_margin=4,
        side_window=4,
        guard_sections=0,
        min_side_sections=1,
        narrow_ratio_threshold=0.65,
        visualization_section_stride=4,
        angular_bins=36,
        points_per_angle_bin=2,
        radial_outlier_factor=1.75,
        min_owned_angular_coverage=0.20,
        short_junction_edge_max_arc_length=15.0,
        junction_attachment_radius=1.8,
        local_cut_padding_points=0,
        local_cut_recovery_ratio=0.90,
        local_cut_recovery_stable_sections=2,
        local_cut_recovery_reference_quantile=0.75,
        short_single_junction_max_points=2,
    )

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 4/6] Rebuilding final skeleton topology...', 73)

    skeleton_edges = algo1.rebuild_skeleton_edges_after_yellow_deletion(
        skeleton_edges=skeleton_edges,
        deletion_row_indices=deletion_rows,
        neighbor_radius=1.8,
        random_seed=42,
    )

    skeleton_edges_path = out_dir / f'{stem}_skeleton_edges.txt'
    np.savetxt(
        skeleton_edges_path,
        skeleton_edges,
        delimiter=',',
        fmt=['%.3f', '%.3f', '%.3f', '%d', '%d', '%d', '%d'],
    )

    skeleton_social = algo1.segment_skeleton_social_instances(
        skeleton_edges=skeleton_edges,
        neighbor_radius=1.8,
        random_seed=42,
        prune_shortcut_edges=True,
    )
    skeleton_social_path = out_dir / f'{stem}_skeleton_instance_social.txt'
    np.savetxt(
        skeleton_social_path,
        skeleton_social,
        delimiter=',',
        fmt=['%.3f', '%.3f', '%.3f', '%d', '%d', '%d', '%d'],
    )

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 5/6] Mapping physical/social instances to PCD...', 80)

    pcd64 = algo2.load_txt_array(geometry['pcd_path'])
    voxel64 = algo2.load_txt_array(geometry['voxel_path'])
    edges64 = algo2.load_txt_array(skeleton_edges_path)
    social64 = algo2.load_txt_array(skeleton_social_path)

    _, pcd_phy_labels, _ = algo2.assign_pcd_from_nearest_skeleton(
        pcd=pcd64,
        skeleton_instances=edges64,
        layer_name='PHYSICAL',
        workers=-1,
    )
    pcd_phy_rgb, pcd_phy_labels, _ = algo2.repair_disconnected_pcd_instances(
        pcd=pcd64,
        assigned_labels=pcd_phy_labels,
        skeleton_instances=edges64,
        layer_name='PHYSICAL',
        connect_radius=None,
        connect_radius_factor=2.5,
        repair_iterations=2,
        reassignment_neighbor_count=64,
        reassignment_max_distance_factor=2.0,
        reassignment_min_vote_fraction=0.60,
        reassignment_min_other_neighbors=3,
        component_vote_sample_points=500,
        workers=-1,
    )
    voxel_phy_rgb, _, _ = algo2.assign_voxel_from_segmented_pcd(
        voxel=voxel64,
        pcd=pcd64,
        pcd_rgb=pcd_phy_rgb,
        pcd_labels=pcd_phy_labels,
        layer_name='PHYSICAL',
        workers=-1,
    )

    _, pcd_social_labels, _ = algo2.assign_pcd_from_nearest_skeleton(
        pcd=pcd64,
        skeleton_instances=social64,
        layer_name='SOCIAL',
        workers=-1,
    )
    pcd_social_rgb, pcd_social_labels, _ = algo2.repair_disconnected_pcd_instances(
        pcd=pcd64,
        assigned_labels=pcd_social_labels,
        skeleton_instances=social64,
        layer_name='SOCIAL',
        connect_radius=None,
        connect_radius_factor=2.5,
        repair_iterations=2,
        reassignment_neighbor_count=64,
        reassignment_max_distance_factor=2.0,
        reassignment_min_vote_fraction=0.60,
        reassignment_min_other_neighbors=3,
        component_vote_sample_points=500,
        workers=-1,
    )
    voxel_social_rgb, _, _ = algo2.assign_voxel_from_segmented_pcd(
        voxel=voxel64,
        pcd=pcd64,
        pcd_rgb=pcd_social_rgb,
        pcd_labels=pcd_social_labels,
        layer_name='SOCIAL',
        workers=-1,
    )

    voxel_phy_path = out_dir / f'{stem}_voxel_instance_phy.txt'
    voxel_social_path = out_dir / f'{stem}_voxel_instance_social.txt'
    pcd_phy_path = out_dir / f'{stem}_pcd_instance_physical.txt'
    pcd_social_path = out_dir / f'{stem}_pcd_instance_social.txt'

    algo2.save_voxel_xyzrgb_intensity(voxel_phy_path, voxel64, voxel_phy_rgb)
    algo2.save_voxel_xyzrgb_intensity(voxel_social_path, voxel64, voxel_social_rgb)
    algo2.save_pcd_xyzrgbnormal_social_id(
        pcd_phy_path, pcd64, pcd_phy_rgb, pcd_social_labels
    )
    algo2.save_pcd_xyzrgbnormal_social_id(
        pcd_social_path, pcd64, pcd_social_rgb, pcd_social_labels
    )

    _check_stop(stop_callback)
    _progress(progress_callback, '[Instance 6/6] PCD + voxel instance outputs ready.', 90)

    geometry.update({
        'skeleton_edges_path': skeleton_edges_path,
        'skeleton_social_path': skeleton_social_path,
        'pcd_instance_physical_path': pcd_phy_path,
        'pcd_instance_social_path': pcd_social_path,
        'voxel_instance_phy_path': voxel_phy_path,
        'voxel_instance_social_path': voxel_social_path,
    })
    return geometry


def calculate_morphology(geometry, progress_callback=None, stop_callback=None):
    """
    Calculate two independent morphology tables:
      1) Physical instance morphology
      2) Social instance morphology

    The morphology core reconstructs objects from RGB + voxel connectivity,
    so it is valid for both LinkODT physical and social instance outputs.
    """
    algo3 = _load_local_module(
        'linknex_instance_algo03.py',
        'linknex_instance_algo03',
    )

    output_dir = Path(
        geometry['output_dir']
    )

    # --------------------------------------------------------------
    # Physical morphology
    # --------------------------------------------------------------
    _check_stop(stop_callback)
    _progress(
        progress_callback,
        '[Morphology 1/2] Calculating physical instance morphology...',
        92,
    )

    voxel_phy = algo3.load_txt_array(
        geometry['voxel_instance_phy_path']
    )
    pcd_phy = algo3.load_txt_array(
        geometry['pcd_instance_physical_path']
    )

    physical_df = (
        algo3.calculate_social_instance_morphology(
            image_name=geometry['stem'],
            voxel=voxel_phy,
            pcd=pcd_phy,
            voxel_connect_radius=1.8,
            voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            workers=-1,
        )
    )

    physical_path = (
        output_dir
        / f"{geometry['stem']}_instance_morphology_physical.csv"
    )

    physical_df.to_csv(
        physical_path,
        index=False,
    )

    geometry[
        'morphology_physical_path'
    ] = physical_path

    # --------------------------------------------------------------
    # Social morphology
    # --------------------------------------------------------------
    _check_stop(stop_callback)
    _progress(
        progress_callback,
        '[Morphology 2/2] Calculating social instance morphology...',
        96,
    )

    voxel_social = algo3.load_txt_array(
        geometry['voxel_instance_social_path']
    )
    pcd_social = algo3.load_txt_array(
        geometry['pcd_instance_social_path']
    )

    social_df = (
        algo3.calculate_social_instance_morphology(
            image_name=geometry['stem'],
            voxel=voxel_social,
            pcd=pcd_social,
            voxel_connect_radius=1.8,
            voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),
            voxel_size_xyz=(1.0, 1.0, 1.0),
            workers=-1,
        )
    )

    social_path = (
        output_dir
        / f"{geometry['stem']}_instance_morphology_social.csv"
    )

    social_df.to_csv(
        social_path,
        index=False,
    )

    geometry[
        'morphology_social_path'
    ] = social_path

    # Backward compatibility with earlier LinkNex code.
    geometry[
        'morphology_path'
    ] = social_path

    _progress(
        progress_callback,
        (
            '[DONE] Morphology saved:\n'
            f'  Physical: {physical_path.name}\n'
            f'  Social: {social_path.name}'
        ),
        100,
    )

    return geometry


def run_complete_pipeline(mask_path, progress_callback=None, stop_callback=None):
    geometry = prepare_geometry_from_mask(
        mask_path,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )
    geometry = run_skeleton_and_instance_mapping(
        geometry,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )
    geometry = calculate_morphology(
        geometry,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )
    return geometry


def _load_editor_txt(path):
    path = Path(path)
    first = ''
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                first = line
                break
    delimiter = ',' if ',' in first else None
    tokens = [t for t in first.replace(',', ' ').split() if t]

    def is_num(t):
        try:
            float(t); return True
        except Exception:
            return False

    has_header = bool(tokens) and not all(is_num(t) for t in tokens)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1 if has_header else 0, ndmin=2)

    if has_header:
        names = [t.lower() for t in tokens]
        idx = {n:i for i,n in enumerate(names)}
        required = ['x','y','z','r','g','b']
        if not all(k in idx for k in required):
            raise ValueError('Edited TXT header must contain x,y,z,r,g,b.')
        xyz = data[:, [idx['x'],idx['y'],idx['z']]]
        rgb = data[:, [idx['r'],idx['g'],idx['b']]]
        if all(k in idx for k in ['nx','ny','nz']):
            normals = data[:, [idx['nx'],idx['ny'],idx['nz']]]
        else:
            normals = None
        id_name = next((k for k in ['instance_id','instance','label','id'] if k in idx), None)
        ids = np.rint(data[:, idx[id_name]]).astype(np.int64) if id_name else None
    else:
        n = data.shape[1]
        if n == 10:
            xyz = data[:,0:3]
            # Canonical LinkNex social PCD is RGB then normals then ID.
            a, b = data[:,3:6], data[:,6:9]
            a_rgb_like = np.all((a >= 0) & (a <= 255))
            b_rgb_like = np.all((b >= 0) & (b <= 255))
            if a_rgb_like and not b_rgb_like:
                rgb, normals = a, b
            elif b_rgb_like and not a_rgb_like:
                normals, rgb = a, b
            else:
                rgb, normals = a, b
            ids = np.rint(data[:,9]).astype(np.int64)
        elif n == 9:
            xyz = data[:,0:3]
            rgb, normals = data[:,3:6], data[:,6:9]
            ids = None
        elif n == 7:
            xyz, rgb, ids = data[:,0:3], data[:,3:6], np.rint(data[:,6]).astype(np.int64)
            normals = None
        else:
            raise ValueError(f'Unsupported edited point-cloud TXT with {n} columns.')

    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int32)
    if ids is None:
        keys = rgb[:,0].astype(np.int64)*65536 + rgb[:,1].astype(np.int64)*256 + rgb[:,2].astype(np.int64)
        _, inv = np.unique(keys, return_inverse=True)
        ids = inv.astype(np.int64) + 1
    if normals is None:
        normals = np.zeros_like(xyz, dtype=np.float64)
    return np.asarray(xyz,float), rgb, np.asarray(normals,float), np.asarray(ids,np.int64)


def apply_edited_pcd(mask_path, edited_pcd_path, progress_callback=None):
    """
    Make an editor-corrected PCD the canonical social instance result, then
    synchronize voxel colours, skeleton social IDs, physical PCD social IDs,
    and recalculate morphology.
    """
    from scipy.spatial import cKDTree

    mask_path = Path(mask_path)
    stem = mask_stem(mask_path)
    out_dir = processed_output_dir(mask_path)
    edited_pcd_path = Path(edited_pcd_path)

    _progress(progress_callback, '[Edit sync 1/4] Reading corrected PCD...', 15)
    xyz, rgb, normals, ids = _load_editor_txt(edited_pcd_path)

    canonical_social = out_dir / f'{stem}_pcd_instance_social.txt'
    canonical_data = np.column_stack([xyz, rgb, normals, ids])
    np.savetxt(
        canonical_social,
        canonical_data,
        delimiter=',',
        fmt=['%.6f','%.6f','%.6f','%d','%d','%d','%.6f','%.6f','%.6f','%d'],
    )

    _progress(progress_callback, '[Edit sync 2/4] Updating voxel + skeleton social labels...', 40)
    voxel_path = out_dir / f'{stem}_voxel.txt'
    voxel = np.loadtxt(voxel_path, delimiter=',', ndmin=2)
    tree = cKDTree(xyz)
    _, nearest = tree.query(voxel[:,0:3], k=1, workers=-1)
    voxel_rgb = rgb[nearest]
    voxel_social = np.column_stack([voxel[:,0:3], voxel_rgb, voxel[:,3]])
    voxel_social_path = out_dir / f'{stem}_voxel_instance_social.txt'
    np.savetxt(
        voxel_social_path,
        voxel_social,
        delimiter=',',
        fmt=['%.3f','%.3f','%.3f','%d','%d','%d','%.6f'],
    )

    skeleton_edges_path = out_dir / f'{stem}_skeleton_edges.txt'
    if skeleton_edges_path.exists():
        skeleton_edges = np.loadtxt(skeleton_edges_path, delimiter=',', ndmin=2)
        _, sk_nearest = tree.query(skeleton_edges[:,0:3], k=1, workers=-1)
        sk_ids = ids[sk_nearest].copy()
        sk_rgb = rgb[sk_nearest].copy()

        # Keep each physical edge assigned to a single social object by majority.
        edge_ids = np.rint(skeleton_edges[:,6]).astype(np.int64)
        for edge_id in np.unique(edge_ids[edge_ids > 0]):
            rows = np.where(edge_ids == edge_id)[0]
            values, counts = np.unique(sk_ids[rows], return_counts=True)
            winner = int(values[np.argmax(counts)])
            sk_ids[rows] = winner
            pcd_rows = np.where(ids == winner)[0]
            if len(pcd_rows):
                color = np.median(rgb[pcd_rows], axis=0)
                sk_rgb[rows] = np.rint(color).astype(np.int32)

        # Junctions inherit nearest corrected social label.
        skeleton_social = np.column_stack([
            skeleton_edges[:,0:3], sk_rgb, sk_ids
        ])
        skeleton_social_path = out_dir / f'{stem}_skeleton_instance_social.txt'
        np.savetxt(
            skeleton_social_path,
            skeleton_social,
            delimiter=',',
            fmt=['%.3f','%.3f','%.3f','%d','%d','%d','%d'],
        )

    physical_pcd_path = out_dir / f'{stem}_pcd_instance_physical.txt'
    if physical_pcd_path.exists():
        physical = np.loadtxt(physical_pcd_path, delimiter=',', ndmin=2)
        _, pn = tree.query(physical[:,0:3], k=1, workers=-1)
        physical[:,-1] = ids[pn]
        np.savetxt(
            physical_pcd_path,
            physical,
            delimiter=',',
            fmt=['%.3f','%.3f','%.3f','%d','%d','%d','%.6f','%.6f','%.6f','%d'],
        )

    _progress(progress_callback, '[Edit sync 3/4] Recalculating morphology...', 72)
    geometry = {
        'stem': stem,
        'output_dir': out_dir,
        'voxel_instance_social_path': voxel_social_path,
        'pcd_instance_social_path': canonical_social,
    }
    calculate_morphology(geometry, progress_callback=progress_callback)
    _progress(progress_callback, '[Edit sync 4/4] Manual correction synchronized.', 100)
    return geometry



# =============================================================================
# In-window 3D PCD viewer for Instance Segmentation
# =============================================================================

class InstancePCDViewerWidget(QWidget):
    """
    Lightweight in-window 3D point-cloud viewer.

    Main-window responsibilities only:
      - auto-display Physical instance after processing
      - switch Physical / Social
      - load arbitrary LinkODT point-cloud outputs
      - rotate / pan / zoom
      - change point size

    Editing remains in the separate full 3D Editor popup.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_mask_path = None
        self.current_mode = "physical"
        self.current_loaded_path = None
        self.current_point_size = 3
        self._current_actor = None
        self.highlighted_morphology_mode = None
        self.highlighted_morphology_instance_id = None
        self._morphology_highlight_cache = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.physical_btn = QPushButton(
            "Physical ✓"
        )
        self.social_btn = QPushButton(
            "Social"
        )
        self.open_cloud_btn = QPushButton(
            "Load Point Cloud"
        )
        self.reload_btn = QPushButton(
            "Reload"
        )

        self.physical_btn.clicked.connect(
            lambda: self.set_mode(
                "physical"
            )
        )
        self.social_btn.clicked.connect(
            lambda: self.set_mode(
                "social"
            )
        )
        self.open_cloud_btn.clicked.connect(
            self.open_point_cloud
        )
        self.reload_btn.clicked.connect(
            self.reload_current
        )

        toolbar.addWidget(
            self.physical_btn
        )
        toolbar.addWidget(
            self.social_btn
        )
        toolbar.addSpacing(
            5
        )

        toolbar.addWidget(
            QLabel(
                "Point size"
            )
        )

        self.point_size_slider = QSlider(
            Qt.Horizontal
        )
        self.point_size_slider.setRange(
            1,
            15,
        )
        self.point_size_slider.setValue(
            self.current_point_size
        )
        self.point_size_slider.setFixedWidth(
            140
        )
        self.point_size_slider.valueChanged.connect(
            self._point_size_changed
        )

        toolbar.addWidget(
            self.point_size_slider
        )

        self.point_size_value = QLabel(
            str(
                self.current_point_size
            )
        )
        self.point_size_value.setFixedWidth(
            22
        )
        toolbar.addWidget(
            self.point_size_value
        )

        toolbar.addStretch(
            1
        )
        toolbar.addWidget(
            self.open_cloud_btn
        )
        toolbar.addWidget(
            self.reload_btn
        )

        root.addLayout(
            toolbar
        )

        self.info_label = QLabel(
            "Default display: Physical instance. "
            "Mouse: rotate / pan / zoom."
        )
        self.info_label.setWordWrap(
            True
        )
        root.addWidget(
            self.info_label
        )

        self.viewer_frame = QFrame()
        self.viewer_frame.setObjectName(
            "InteractionFrame"
        )
        # Laptop / high-DPI safety: never allow the 3D canvas to collapse
        # to a tiny strip when the work window is smaller than 1560x930.
        self.viewer_frame.setMinimumHeight(360)

        self.viewer_frame_layout = QVBoxLayout(
            self.viewer_frame
        )
        self.viewer_frame_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        self.viewer_frame_layout.setSpacing(
            0
        )

        # IMPORTANT:
        # Do not create a VTK/OpenGL window while Module 3 is still hidden
        # inside QStackedWidget. Some laptop integrated/hybrid GPUs fail to
        # create a valid context for a hidden QtInteractor.
        self.plotter = None
        self.viewer_placeholder = QLabel(
            "3D viewer will initialize when Instance segmentation is opened."
        )
        self.viewer_placeholder.setAlignment(
            Qt.AlignCenter
        )
        self.viewer_placeholder.setWordWrap(
            True
        )
        self.viewer_frame_layout.addWidget(
            self.viewer_placeholder
        )

        root.addWidget(
            self.viewer_frame,
            1,
        )

        self._refresh_button_styles()

    def ensure_plotter(self):
        """Create the Module-3 VTK viewer only when the page is visible/used."""
        if self.plotter is not None:
            return True

        if pv is None or QtInteractor is None:
            self.viewer_placeholder.setText(
                "3D viewer unavailable.\n\n"
                "PyVista / PyVistaQt / VTK could not be imported."
            )
            return False

        try:
            if self.viewer_placeholder is not None:
                self.viewer_frame_layout.removeWidget(
                    self.viewer_placeholder
                )
                self.viewer_placeholder.hide()

            self.plotter = QtInteractor(
                self.viewer_frame
            )
            self.plotter.set_background(
                "white"
            )

            # Improve compatibility with Intel / hybrid laptop GPUs.
            try:
                render_window = getattr(
                    self.plotter,
                    "ren_win",
                    None,
                )
                if render_window is not None:
                    render_window.SetMultiSamples(0)
            except Exception:
                pass

            try:
                self.plotter.enable_trackball_style()
            except Exception:
                pass

            self.viewer_frame_layout.addWidget(
                self.plotter
            )
            self.plotter.show()

            # Let Qt finish exposing the native window before the first VTK
            # render; this is especially important on high-DPI laptops.
            QTimer.singleShot(
                80,
                self._finish_plotter_initialization,
            )
            return True

        except Exception as exc:
            self.plotter = None

            if self.viewer_placeholder is None:
                self.viewer_placeholder = QLabel()

            self.viewer_placeholder.setText(
                "3D viewer initialization failed.\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "The standalone 3D Editor can still be used."
            )
            self.viewer_placeholder.setAlignment(
                Qt.AlignCenter
            )
            self.viewer_placeholder.setWordWrap(
                True
            )
            self.viewer_placeholder.show()

            if self.viewer_frame_layout.indexOf(
                self.viewer_placeholder
            ) < 0:
                self.viewer_frame_layout.addWidget(
                    self.viewer_placeholder
                )
            return False

    def _finish_plotter_initialization(self):
        if self.plotter is None:
            return

        try:
            self.plotter.resize(
                max(320, self.viewer_frame.width()),
                max(300, self.viewer_frame.height()),
            )
        except Exception:
            pass

        try:
            self.plotter.render()
        except Exception:
            pass

        # If an already-processed image is selected, display it now.
        if self.current_mask_path is not None:
            QTimer.singleShot(
                20,
                lambda: self.reload_current(
                    reset_camera=True
                ),
            )

    def showEvent(self, event):
        super().showEvent(event)

        # QStackedWidget exposes this page only when the user actually enters
        # Module 3. Initialize OpenGL at that point rather than at app startup.
        QTimer.singleShot(
            0,
            self.ensure_plotter,
        )

    # ------------------------------------------------------------------
    # Display controls
    # ------------------------------------------------------------------
    def _refresh_button_styles(self):
        physical_on = (
            self.current_mode
            == "physical"
        )
        social_on = (
            self.current_mode
            == "social"
        )

        self.physical_btn.setText(
            "Physical ✓"
            if physical_on
            else "Physical"
        )
        self.social_btn.setText(
            "Social ✓"
            if social_on
            else "Social"
        )

        self.physical_btn.setProperty(
            "activeMode",
            physical_on,
        )
        self.social_btn.setProperty(
            "activeMode",
            social_on,
        )

        for button in (
            self.physical_btn,
            self.social_btn,
        ):
            button.style().unpolish(
                button
            )
            button.style().polish(
                button
            )
            button.update()

    def _point_size_changed(
        self,
        value,
    ):
        self.current_point_size = int(
            value
        )

        self.point_size_value.setText(
            str(
                self.current_point_size
            )
        )

        if (
            self.highlighted_morphology_instance_id
            is not None
            and self.highlighted_morphology_mode
            is not None
        ):
            self.highlight_morphology_instance(
                self.highlighted_morphology_mode,
                self.highlighted_morphology_instance_id,
                preserve_camera=True,
            )

        elif self.current_loaded_path is not None:
            self.load_point_cloud(
                self.current_loaded_path,
                preserve_camera=True,
            )

    def set_mode(
        self,
        mode,
    ):
        mode = str(
            mode
        ).lower().strip()

        if mode not in (
            "physical",
            "social",
        ):
            return

        self.current_mode = mode
        self.highlighted_morphology_mode = None
        self.highlighted_morphology_instance_id = None
        self._refresh_button_styles()
        self.reload_current()

    def set_mask_path(
        self,
        mask_path,
    ):
        self.current_mask_path = (
            Path(
                mask_path
            )
            if mask_path
            else None
        )

        self.highlighted_morphology_mode = None
        self.highlighted_morphology_instance_id = None

        self.reload_current(
            reset_camera=True
        )

    # ------------------------------------------------------------------
    # LinkNex path helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stem_from_mask(
        mask_path,
    ):
        mask_path = Path(
            mask_path
        )

        name = mask_path.name
        lower = name.lower()

        if lower.endswith(
            "_mask.tiff"
        ):
            return name[
                :-len(
                    "_mask.tiff"
                )
            ]

        if lower.endswith(
            "_mask.tif"
        ):
            return name[
                :-len(
                    "_mask.tif"
                )
            ]

        return mask_path.stem

    @classmethod
    def _pcd_path_from_mask(
        cls,
        mask_path,
        mode,
    ):
        mask_path = Path(
            mask_path
        )

        stem = cls._stem_from_mask(
            mask_path
        )

        out_dir = (
            mask_path.parent
            / "processed_output"
        )

        if str(
            mode
        ).lower() == "social":
            return (
                out_dir
                / f"{stem}_pcd_instance_social.txt"
            )

        return (
            out_dir
            / f"{stem}_pcd_instance_physical.txt"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def open_point_cloud(
        self,
    ):
        start_dir = ""

        if self.current_mask_path is not None:
            candidate = (
                self.current_mask_path.parent
                / "processed_output"
            )

            if candidate.exists():
                start_dir = str(
                    candidate
                )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load point cloud / skeleton / voxel",
            start_dir,
            (
                "Point cloud data "
                "(*.txt *.csv *.pcd *.ply);;"
                "Text data (*.txt *.csv);;"
                "PCD (*.pcd);;"
                "PLY (*.ply);;"
                "All files (*)"
            ),
        )

        if path:
            # Manual load is independent of Physical/Social auto mode.
            self.load_point_cloud(
                Path(
                    path
                ),
                preserve_camera=False,
            )

    def clear_view(
        self,
        message="No point cloud loaded.",
    ):
        self.info_label.setText(
            str(
                message
            )
        )

        if self.plotter is not None:
            try:
                self.plotter.clear()
                self.plotter.render()
            except Exception:
                pass

    def reload_current(
        self,
        reset_camera=False,
    ):
        if self.current_mask_path is None:
            return

        path = self._pcd_path_from_mask(
            self.current_mask_path,
            self.current_mode,
        )

        if not path.exists():
            self.current_loaded_path = None

            self.clear_view(
                (
                    f"{self.current_mode.capitalize()} "
                    f"instance file not found yet: {path.name}. "
                    "Run Instance segmentation first, or use Load Point Cloud."
                )
            )
            return

        self.load_point_cloud(
            path,
            preserve_camera=(
                not reset_camera
            ),
        )

    # ------------------------------------------------------------------
    # Generic point-cloud parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _load_ascii_pcd(
        path,
    ):
        path = Path(
            path
        )

        fields = None
        data_start = None
        data_mode = None

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            lines = handle.readlines()

        for index, line in enumerate(
            lines
        ):
            stripped = line.strip()

            if not stripped:
                continue

            upper = stripped.upper()

            if upper.startswith(
                "FIELDS "
            ):
                fields = stripped.split()[1:]

            elif upper.startswith(
                "DATA "
            ):
                parts = stripped.split()

                if len(
                    parts
                ) >= 2:
                    data_mode = parts[
                        1
                    ].lower()

                data_start = (
                    index
                    + 1
                )
                break

        if data_mode != "ascii":
            # Open3D handles many non-ASCII PCDs if installed.
            try:
                import open3d as o3d

                cloud = o3d.io.read_point_cloud(
                    str(
                        path
                    )
                )

                points = np.asarray(
                    cloud.points
                )

                colors = np.asarray(
                    cloud.colors
                )

                if (
                    len(
                        colors
                    )
                    == len(
                        points
                    )
                    and len(
                        colors
                    )
                    > 0
                ):
                    rgb = np.clip(
                        np.rint(
                            colors
                            * 255.0
                        ),
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )
                else:
                    rgb = None

                return points, rgb

            except Exception as exc:
                raise ValueError(
                    "Binary PCD requires Open3D."
                ) from exc

        if data_start is None:
            raise ValueError(
                "Invalid PCD header."
            )

        arr = np.loadtxt(
            lines[
                data_start:
            ],
            ndmin=2,
        )

        if fields is None:
            fields = [
                f"field_{i}"
                for i in range(
                    arr.shape[
                        1
                    ]
                )
            ]

        mapping = {
            name.lower(): idx
            for idx, name in enumerate(
                fields
            )
        }

        for required in (
            "x",
            "y",
            "z",
        ):
            if required not in mapping:
                raise ValueError(
                    "PCD requires x/y/z fields."
                )

        points = arr[
            :,
            [
                mapping[
                    "x"
                ],
                mapping[
                    "y"
                ],
                mapping[
                    "z"
                ],
            ],
        ]

        rgb = None

        if all(
            name in mapping
            for name in (
                "r",
                "g",
                "b",
            )
        ):
            rgb = arr[
                :,
                [
                    mapping[
                        "r"
                    ],
                    mapping[
                        "g"
                    ],
                    mapping[
                        "b"
                    ],
                ],
            ]

        return (
            points,
            None
            if rgb is None
            else np.clip(
                np.rint(
                    rgb
                ),
                0,
                255,
            ).astype(
                np.uint8
            ),
        )

    @staticmethod
    def _looks_like_rgb(
        values,
    ):
        values = np.asarray(
            values
        )

        if (
            values.ndim != 2
            or values.shape[
                1
            ]
            != 3
        ):
            return False

        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.size == 0:
            return False

        minimum = float(
            finite.min()
        )
        maximum = float(
            finite.max()
        )

        if (
            minimum
            < -1e-6
            or maximum
            > 255.0
            + 1e-6
        ):
            return False

        # Normals are usually -1..1.
        # RGB generally has values significantly above 1.
        return maximum > 2.0

    @classmethod
    def _load_linknex_text(
        cls,
        path,
    ):
        path = Path(
            path
        )

        # LinkNex outputs are comma-separated, but user point clouds can also
        # be whitespace-separated.
        try:
            arr = np.loadtxt(
                path,
                delimiter=",",
                dtype=np.float64,
                ndmin=2,
            )
        except Exception:
            arr = np.loadtxt(
                path,
                dtype=np.float64,
                ndmin=2,
            )

        if arr.shape[
            1
        ] < 3:
            raise ValueError(
                f"At least x,y,z are required; got {arr.shape[1]} columns."
            )

        points = arr[
            :,
            :3,
        ]

        rgb = None

        # Common LinkNex layouts:
        # raw voxel: x,y,z,intensity
        # refined skeleton: x,y,z,score
        # skeleton: x,y,z,radius,degree,type
        # skeleton edge/social: x,y,z,r,g,b,id
        # voxel instance: x,y,z,r,g,b,intensity
        # raw PCD: x,y,z,nx,ny,nz
        # PCD instance: x,y,z,r,g,b,nx,ny,nz,id
        if (
            arr.shape[
                1
            ]
            >= 6
            and cls._looks_like_rgb(
                arr[
                    :,
                    3:6,
                ]
            )
        ):
            rgb = arr[
                :,
                3:6,
            ]

        elif (
            arr.shape[
                1
            ]
            >= 9
            and cls._looks_like_rgb(
                arr[
                    :,
                    6:9,
                ]
            )
        ):
            rgb = arr[
                :,
                6:9,
            ]

        # No explicit RGB: color by scalar / ID if available.
        if rgb is None:
            scalar = None

            if arr.shape[
                1
            ] >= 7:
                scalar = arr[
                    :,
                    -1,
                ]

            elif arr.shape[
                1
            ] >= 4:
                scalar = arr[
                    :,
                    3,
                ]

            if scalar is not None:
                scalar = np.asarray(
                    scalar,
                    dtype=np.float64,
                )

                finite = np.isfinite(
                    scalar
                )

                if np.any(
                    finite
                ):
                    safe = scalar.copy()

                    minimum = float(
                        np.min(
                            safe[
                                finite
                            ]
                        )
                    )
                    maximum = float(
                        np.max(
                            safe[
                                finite
                            ]
                        )
                    )

                    if maximum > minimum:
                        norm = (
                            safe
                            - minimum
                        ) / (
                            maximum
                            - minimum
                        )
                    else:
                        norm = np.zeros_like(
                            safe
                        )

                    # Simple grayscale for intensity/score/raw IDs.
                    gray = np.clip(
                        np.rint(
                            norm
                            * 220
                            + 25
                        ),
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )

                    rgb = np.column_stack(
                        [
                            gray,
                            gray,
                            gray,
                        ]
                    )

        if rgb is None:
            rgb = np.tile(
                np.array(
                    [
                        70,
                        120,
                        190,
                    ],
                    dtype=np.uint8,
                ),
                (
                    len(
                        points
                    ),
                    1,
                ),
            )

        else:
            rgb = np.asarray(
                rgb,
                dtype=np.float64,
            )

            finite = rgb[
                np.isfinite(
                    rgb
                )
            ]

            if (
                finite.size
                and float(
                    finite.max()
                )
                <= 1.0
                + 1e-6
            ):
                rgb = (
                    rgb
                    * 255.0
                )

            rgb = np.clip(
                np.rint(
                    rgb
                ),
                0,
                255,
            ).astype(
                np.uint8
            )

        return (
            points,
            rgb,
        )

    @classmethod
    def _load_cloud(
        cls,
        path,
    ):
        path = Path(
            path
        )

        suffix = path.suffix.lower()

        if suffix == ".pcd":
            return cls._load_ascii_pcd(
                path
            )

        if suffix == ".ply":
            if pv is None:
                raise RuntimeError(
                    "PyVista is required for PLY."
                )

            mesh = pv.read(
                str(
                    path
                )
            )

            points = np.asarray(
                mesh.points
            )

            rgb = None

            for key in (
                "RGB",
                "rgb",
                "colors",
                "Colors",
            ):
                if key in mesh.point_data:
                    candidate = np.asarray(
                        mesh.point_data[
                            key
                        ]
                    )

                    if (
                        candidate.ndim
                        == 2
                        and candidate.shape[
                            1
                        ]
                        >= 3
                    ):
                        rgb = candidate[
                            :,
                            :3,
                        ]
                        break

            if rgb is None:
                rgb = np.tile(
                    np.array(
                        [
                            70,
                            120,
                            190,
                        ],
                        dtype=np.uint8,
                    ),
                    (
                        len(
                            points
                        ),
                        1,
                    ),
                )

            return (
                points,
                np.clip(
                    np.rint(
                        rgb
                    ),
                    0,
                    255,
                ).astype(
                    np.uint8
                ),
            )

        return cls._load_linknex_text(
            path
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def load_point_cloud(
        self,
        path,
        preserve_camera=True,
    ):
        path = Path(
            path
        )

        try:
            points, rgb = self._load_cloud(
                path
            )

        except Exception as exc:
            self.current_loaded_path = None

            self.clear_view(
                f"Failed to load {path.name}: {exc}"
            )
            return

        self.current_loaded_path = path

        self.info_label.setText(
            f"Loaded: {path.name} | "
            f"points = {len(points):,}"
        )

        if self.plotter is None:
            if not self.ensure_plotter():
                return

        camera_position = None

        if preserve_camera:
            try:
                camera_position = (
                    self.plotter.camera_position
                )
            except Exception:
                camera_position = None

        try:
            self.plotter.clear()

            poly = pv.PolyData(
                np.asarray(
                    points,
                    dtype=np.float32,
                )
            )

            poly[
                "rgb"
            ] = np.asarray(
                rgb,
                dtype=np.uint8,
            )

            try:
                self._current_actor = (
                    self.plotter.add_points(
                        poly,
                        scalars="rgb",
                        rgb=True,
                        point_size=float(
                            self.current_point_size
                        ),
                        render_points_as_spheres=True,
                    )
                )
            except Exception:
                # Compatibility fallback for older Intel / hybrid GPU drivers:
                # draw standard OpenGL points instead of sphere sprites.
                self.plotter.clear()
                self._current_actor = (
                    self.plotter.add_points(
                        poly,
                        scalars="rgb",
                        rgb=True,
                        point_size=float(
                            self.current_point_size
                        ),
                        render_points_as_spheres=False,
                    )
                )

            if (
                preserve_camera
                and camera_position
                is not None
            ):
                try:
                    self.plotter.camera_position = (
                        camera_position
                    )
                except Exception:
                    self.plotter.reset_camera()

            else:
                self.plotter.reset_camera()

            self.plotter.render()

        except Exception as exc:
            self.clear_view(
                f"Failed to render {path.name}: {exc}"
            )

    def _morphology_instance_assignment(
        self,
        mode,
    ):
        """
        Reconstruct exactly the same morphology instance IDs used by the
        physical/social morphology CSVs, then map those IDs onto PCD points.
        """
        if self.current_mask_path is None:
            raise ValueError(
                "No current semantic mask."
            )

        mode = str(
            mode
        ).lower()

        if mode not in (
            "physical",
            "social",
        ):
            raise ValueError(
                f"Unknown morphology mode: {mode}"
            )

        cache_key = (
            str(
                Path(
                    self.current_mask_path
                ).resolve()
            ).lower(),
            mode,
        )

        cached = self._morphology_highlight_cache.get(
            cache_key
        )

        if cached is not None:
            return cached

        mask_path = Path(
            self.current_mask_path
        )
        stem = self._stem_from_mask(
            mask_path
        )
        out_dir = (
            mask_path.parent
            / "processed_output"
        )

        if mode == "physical":
            voxel_path = (
                out_dir
                / f"{stem}_voxel_instance_phy.txt"
            )
            pcd_path = (
                out_dir
                / f"{stem}_pcd_instance_physical.txt"
            )
        else:
            voxel_path = (
                out_dir
                / f"{stem}_voxel_instance_social.txt"
            )
            pcd_path = (
                out_dir
                / f"{stem}_pcd_instance_social.txt"
            )

        if not voxel_path.exists():
            raise FileNotFoundError(
                voxel_path
            )

        if not pcd_path.exists():
            raise FileNotFoundError(
                pcd_path
            )

        algo3 = _load_local_module(
            'linknex_instance_algo03.py',
            (
                'linknex_instance_algo03_'
                + mode
                + '_viewer'
            ),
        )

        voxel = algo3.load_txt_array(
            voxel_path
        )
        pcd = algo3.load_txt_array(
            pcd_path
        )

        voxel_xyz = voxel[
            :,
            :3,
        ]
        voxel_rgb = np.clip(
            np.rint(
                voxel[
                    :,
                    3:6,
                ]
            ),
            0,
            255,
        ).astype(
            np.int32
        )

        pcd_xyz = pcd[
            :,
            :3,
        ]
        pcd_rgb = np.clip(
            np.rint(
                pcd[
                    :,
                    3:6,
                ]
            ),
            0,
            255,
        ).astype(
            np.int32
        )

        voxel_valid = (
            np.isfinite(
                voxel_xyz
            ).all(
                axis=1
            )
            & np.isfinite(
                voxel[
                    :,
                    3:7,
                ]
            ).all(
                axis=1
            )
            & np.any(
                voxel_rgb
                != 0,
                axis=1,
            )
        )

        pcd_valid = (
            np.isfinite(
                pcd_xyz
            ).all(
                axis=1
            )
            & np.isfinite(
                pcd[
                    :,
                    3:9,
                ]
            ).all(
                axis=1
            )
            & np.any(
                pcd_rgb
                != 0,
                axis=1,
            )
        )

        voxel_xyz = voxel_xyz[
            voxel_valid
        ]
        voxel_rgb = voxel_rgb[
            voxel_valid
        ]

        pcd_xyz = pcd_xyz[
            pcd_valid
        ]
        pcd_rgb = pcd_rgb[
            pcd_valid
        ]

        if (
            len(
                voxel_xyz
            )
            == 0
            or len(
                pcd_xyz
            )
            == 0
        ):
            raise ValueError(
                f"No valid colored {mode} voxel/PCD points remain."
            )

        voxel_instance_ids = (
            algo3.split_voxel_instances_by_rgb_and_connectivity(
                voxel_xyz=voxel_xyz,
                voxel_rgb=voxel_rgb,
                connect_radius=1.8,
            )
        )

        pcd_instance_ids = (
            algo3.map_pcd_to_voxel_instances_by_rgb(
                pcd_xyz=pcd_xyz,
                pcd_rgb=pcd_rgb,
                voxel_xyz=voxel_xyz,
                voxel_rgb=voxel_rgb,
                voxel_instance_ids=voxel_instance_ids,
                workers=-1,
            )
        )

        result = {
            "path": pcd_path,
            "points": pcd_xyz,
            "rgb": pcd_rgb.astype(
                np.uint8,
                copy=False,
            ),
            "instance_ids": pcd_instance_ids,
        }

        self._morphology_highlight_cache[
            cache_key
        ] = result

        return result

    def highlight_morphology_instance(
        self,
        mode,
        instance_id,
        preserve_camera=True,
    ):
        """
        Highlight one physical or social morphology instance in gray.

        The base colour display follows the selected morphology layer:
          Physical table -> Physical PCD base
          Social table   -> Social PCD base
        """
        mode = str(
            mode
        ).lower()

        try:
            assignment = (
                self._morphology_instance_assignment(
                    mode
                )
            )

            points = assignment[
                "points"
            ]
            rgb = assignment[
                "rgb"
            ]
            instance_ids = assignment[
                "instance_ids"
            ]
            pcd_path = assignment[
                "path"
            ]

            selected = (
                instance_ids
                == int(
                    instance_id
                )
            )

            if not np.any(
                selected
            ):
                raise ValueError(
                    f"{mode.capitalize()} instance {int(instance_id)} "
                    "was not found in the PCD."
                )

        except Exception as exc:
            self.info_label.setText(
                (
                    f"Failed to highlight "
                    f"{mode} instance {instance_id}: {exc}"
                )
            )
            return

        self.current_mode = mode
        self._refresh_button_styles()

        self.current_loaded_path = (
            pcd_path
        )

        self.highlighted_morphology_mode = (
            mode
        )
        self.highlighted_morphology_instance_id = int(
            instance_id
        )

        self.info_label.setText(
            f"{mode.capitalize()} morphology | "
            f"selected instance = {int(instance_id)} | "
            f"selected points = "
            f"{int(np.count_nonzero(selected)):,}"
        )

        if self.plotter is None:
            return

        camera_position = None

        if preserve_camera:
            try:
                camera_position = (
                    self.plotter.camera_position
                )
            except Exception:
                camera_position = None

        try:
            self.plotter.clear()

            base_poly = pv.PolyData(
                np.asarray(
                    points,
                    dtype=np.float32,
                )
            )
            base_poly[
                "rgb"
            ] = rgb

            self.plotter.add_points(
                base_poly,
                scalars="rgb",
                rgb=True,
                point_size=float(
                    self.current_point_size
                ),
                render_points_as_spheres=True,
            )

            selected_points = np.asarray(
                points[
                    selected
                ],
                dtype=np.float32,
            )

            selected_poly = pv.PolyData(
                selected_points
            )

            # Selected morphology instance: neutral gray so it stands out
            # clearly against the multicolour instance PCD.
            selected_rgb = np.tile(
                np.array(
                    [
                        150,
                        150,
                        150,
                    ],
                    dtype=np.uint8,
                ),
                (
                    len(
                        selected_points
                    ),
                    1,
                ),
            )

            selected_poly[
                "selected_rgb"
            ] = selected_rgb

            self.plotter.add_points(
                selected_poly,
                scalars="selected_rgb",
                rgb=True,
                # Selected instance uses exactly two larger point-size steps.
                point_size=float(
                    self.current_point_size
                    + 2
                ),
                render_points_as_spheres=True,
            )

            if (
                preserve_camera
                and camera_position
                is not None
            ):
                try:
                    self.plotter.camera_position = (
                        camera_position
                    )
                except Exception:
                    self.plotter.reset_camera()
            else:
                self.plotter.reset_camera()

            self.plotter.render()

        except Exception as exc:
            self.info_label.setText(
                (
                    f"Failed to render highlighted "
                    f"{mode} instance {instance_id}: {exc}"
                )
            )

    def clear_instance_highlight(
        self,
    ):
        mode = (
            self.highlighted_morphology_mode
            if self.highlighted_morphology_mode
            in (
                "physical",
                "social",
            )
            else self.current_mode
        )

        self.highlighted_morphology_mode = None
        self.highlighted_morphology_instance_id = None

        if self.current_mask_path is not None:
            self.current_mode = mode
            self._refresh_button_styles()
            self.reload_current()



# =============================================================================
# Instance Segmentation module
# =============================================================================

# =============================================================================

class InstancePipelineWorker(QObject):
    progress = Signal(str, int)
    fileFinished = Signal(object)
    finished = Signal(object)
    stopped = Signal(str)
    failed = Signal(str)

    def __init__(self, mask_paths):
        super().__init__()
        self.mask_paths = [Path(p) for p in mask_paths]

    def _check_stop(self):
        if QThread.currentThread().isInterruptionRequested():
            raise InterruptedError('Instance segmentation stopped by user.')

    def run(self):
        try:
            results = []
            total = len(self.mask_paths)

            for image_index, mask_path in enumerate(self.mask_paths, start=1):
                self._check_stop()

                def callback(text, value=None):
                    self._check_stop()
                    local = 0 if value is None else int(value)
                    overall = int(
                        round(
                            100 * ((image_index - 1) + local / 100.0)
                            / max(1, total)
                        )
                    )
                    self.progress.emit(
                        f'[{image_index}/{total}] {mask_path.name}\n{text}',
                        overall,
                    )

                result = run_complete_pipeline(
                    mask_path,
                    progress_callback=callback,
                    stop_callback=self._check_stop,
                )
                results.append(result)
                self.fileFinished.emit(result)

            self.finished.emit(results)

        except InterruptedError:
            self.stopped.emit('INSTANCE SEGMENTATION STOPPED BY USER')
        except Exception as exc:
            self.failed.emit(str(exc))


class InstanceSyncWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, mask_path, edited_pcd_path):
        super().__init__()
        self.mask_path = Path(mask_path)
        self.edited_pcd_path = Path(edited_pcd_path)

    def run(self):
        try:
            result = apply_edited_pcd(
                self.mask_path,
                self.edited_pcd_path,
                progress_callback=lambda text, value=None: self.progress.emit(
                    str(text),
                    int(0 if value is None else value),
                ),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))



MORPHOLOGY_DISPLAY_METRICS = [
    (
        "voxel_count",
        "Voxel count",
    ),
    (
        "volume",
        "Volume",
    ),
    (
        "surface_area_voxel_faces",
        "Surface area",
    ),
    (
        "surface_volume_ratio",
        "Surface / volume",
    ),
    (
        "equivalent_sphere_diameter",
        "Equivalent diameter",
    ),
    (
        "major_axis_length_pca",
        "Major axis length",
    ),
    (
        "mean_width_pca",
        "Mean width",
    ),
    (
        "length_width_ratio",
        "Length / width",
    ),
    (
        "aspect_ratio_major_minor",
        "Major / minor ratio",
    ),
    (
        "radius_of_gyration",
        "Radius of gyration",
    ),
    (
        "sphericity",
        "Sphericity",
    ),
    (
        "compactness_3d",
        "Compactness 3D",
    ),
    (
        "intensity_mean",
        "Mean intensity",
    ),
]


class MorphologyTableModel(QAbstractTableModel):
    """Fast read-only morphology table model."""

    def __init__(self, morphology_df, metrics, instance_ids, parent=None):
        super().__init__(parent)
        self.df = morphology_df
        self.metrics = list(metrics)
        self.instance_ids = list(instance_ids)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.metrics)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.instance_ids)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            key = self.metrics[index.row()][0]
            value = self.df.iloc[index.column()][key]
            return MorphologyTableDialog._format_value(
                value,
                integer=(key == "voxel_count"),
            )

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if 0 <= section < len(self.instance_ids):
                return f"Instance {self.instance_ids[section]}"
        else:
            if 0 <= section < len(self.metrics):
                return self.metrics[section][1]

        return None


class MorphologyTableDialog(QDialog):
    """
    Interactive morphology table with two real morphology layers.

    Default:
        Physical

    User can switch:
        Physical <-> Social

    Columns = instance IDs
    Rows    = selected representative morphology metrics
    """

    def __init__(
        self,
        morphology_paths,
        viewer,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self.viewer = viewer

        self.morphology_paths = {
            str(
                key
            ).lower(): Path(
                value
            )
            for key, value
            in dict(
                morphology_paths
            ).items()
            if value is not None
        }

        self.current_mode = (
            "physical"
            if (
                "physical"
                in self.morphology_paths
                and self.morphology_paths[
                    "physical"
                ].exists()
            )
            else "social"
        )

        self.morphology_df = None
        self.instance_ids = []
        self.metrics = []

        self.setWindowTitle(
            "Instance Morphology"
        )
        self.resize(
            1120,
            650,
        )

        # Morphology dialog uses its own fully light theme so it does not
        # inherit any dark-blue background from the main application.
        self.setStyleSheet(
            """
            QDialog {
                background: #F6F8FB;
                color: #071D49;
            }

            QLabel {
                background: transparent;
                color: #071D49;
            }

            QPushButton {
                background: #FFFFFF;
                color: #071D49;
                border: 1px solid #C9D4E2;
                border-radius: 5px;
                padding: 6px 12px;
                min-height: 24px;
            }

            QPushButton:hover {
                background: #EEF3F8;
            }

            QPushButton:pressed {
                background: #E2EAF2;
            }

            QPushButton:disabled {
                background: #F0F2F5;
                color: #8A98A8;
                border: 1px solid #D8E0E8;
            }

            QTableView {
                background: #F6F8FB;
                alternate-background-color: #F6F8FB;
                color: #071D49;
                gridline-color: #C9D4E2;
                border: 1px solid #C9D4E2;
                selection-background-color: #FFF3B8;
                selection-color: #071D49;
            }

            QTableView::item {
                background: #F6F8FB;
                color: #071D49;
                padding: 4px;
            }

            QTableView::item:selected {
                background: #FFF3B8;
                color: #071D49;
            }

            QHeaderView::section {
                background: #EEF3F8;
                color: #071D49;
                border: 1px solid #C9D4E2;
                padding: 5px;
                font-weight: 700;
            }

            QTableCornerButton::section {
                background: #EEF3F8;
                border: 1px solid #C9D4E2;
            }

            QScrollBar:vertical,
            QScrollBar:horizontal {
                background: #F1F4F7;
                border: none;
            }

            QScrollBar::handle:vertical,
            QScrollBar::handle:horizontal {
                background: #B8C6D4;
                border-radius: 4px;
                min-height: 24px;
                min-width: 24px;
            }
            """ + DISABLED_STYLE
        )

        root = QVBoxLayout(
            self
        )
        root.setContentsMargins(
            12,
            12,
            12,
            12,
        )
        root.setSpacing(
            8
        )

        top_row = QHBoxLayout()

        title = QLabel(
            "Instance Morphology"
        )
        title.setObjectName(
            "PageTitle"
        )
        top_row.addWidget(
            title,
            1,
        )

        self.physical_btn = QPushButton(
            "Physical"
        )
        self.social_btn = QPushButton(
            "Social"
        )

        self.physical_btn.clicked.connect(
            lambda: self.set_mode(
                "physical"
            )
        )
        self.social_btn.clicked.connect(
            lambda: self.set_mode(
                "social"
            )
        )

        top_row.addWidget(
            self.physical_btn
        )
        top_row.addWidget(
            self.social_btn
        )

        root.addLayout(
            top_row
        )

        self.subtitle = QLabel()
        self.subtitle.setWordWrap(
            True
        )
        self.subtitle.setObjectName(
            "Metric"
        )
        root.addWidget(
            self.subtitle
        )

        self.table = QTableView()
        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectColumns
        )
        self.table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.table.setWordWrap(
            False
        )
        # Keep the morphology table visually simple and highly readable:
        # one light background everywhere with dark navy text.
        self.table.setAlternatingRowColors(
            False
        )

        self.table.setStyleSheet(
            """
            QTableView {
                background: #F6F8FB;
                color: #071D49;
                gridline-color: #C9D4E2;
                border: 1px solid #C9D4E2;
                selection-background-color: #FFF3B8;
                selection-color: #071D49;
            }

            QTableView::item {
                background: #F6F8FB;
                color: #071D49;
                padding: 4px;
            }

            QTableView::item:selected {
                background: #FFF3B8;
                color: #071D49;
            }

            QHeaderView::section {
                background: #EEF3F8;
                color: #071D49;
                border: 1px solid #C9D4E2;
                padding: 5px;
                font-weight: 700;
            }

            QTableCornerButton::section {
                background: #EEF3F8;
                border: 1px solid #C9D4E2;
            }
            """
        )

        self.table.clicked.connect(
            self._cell_clicked
        )
        self.table.horizontalHeader().sectionClicked.connect(
            self._instance_column_clicked
        )

        root.addWidget(
            self.table,
            1,
        )

        bottom = QHBoxLayout()

        self.selection_label = QLabel(
            "Selected: none"
        )
        bottom.addWidget(
            self.selection_label,
            1,
        )

        clear_btn = QPushButton(
            "Clear highlight"
        )
        clear_btn.clicked.connect(
            self.clear_highlight
        )
        bottom.addWidget(
            clear_btn
        )

        close_btn = QPushButton(
            "Close"
        )
        close_btn.clicked.connect(
            self.close
        )
        bottom.addWidget(
            close_btn
        )

        root.addLayout(
            bottom
        )

        self._refresh_mode_buttons()
        self._load_current_mode()

    def _refresh_mode_buttons(
        self,
    ):
        physical_exists = (
            "physical"
            in self.morphology_paths
            and self.morphology_paths[
                "physical"
            ].exists()
        )
        social_exists = (
            "social"
            in self.morphology_paths
            and self.morphology_paths[
                "social"
            ].exists()
        )

        self.physical_btn.setEnabled(
            physical_exists
        )
        self.social_btn.setEnabled(
            social_exists
        )

        self.physical_btn.setText(
            "Physical ✓"
            if self.current_mode
            == "physical"
            else "Physical"
        )

        self.social_btn.setText(
            "Social ✓"
            if self.current_mode
            == "social"
            else "Social"
        )

    def set_mode(
        self,
        mode,
    ):
        mode = str(
            mode
        ).lower()

        if mode not in (
            "physical",
            "social",
        ):
            return

        path = self.morphology_paths.get(
            mode
        )

        if (
            path is None
            or not path.exists()
        ):
            return

        if self.current_mode == mode:
            return

        self.clear_highlight()

        self.current_mode = mode
        self._refresh_mode_buttons()
        self._load_current_mode()

    def _load_current_mode(
        self,
    ):
        path = self.morphology_paths.get(
            self.current_mode
        )

        if (
            path is None
            or not path.exists()
        ):
            raise FileNotFoundError(
                (
                    f"{self.current_mode.capitalize()} morphology "
                    "CSV is not available."
                )
            )

        self.morphology_df = pd.read_csv(
            path
        )

        if self.morphology_df.empty:
            raise ValueError(
                f"Morphology CSV is empty:\n{path}"
            )

        if (
            "instance_id"
            not in self.morphology_df.columns
        ):
            raise ValueError(
                "Morphology CSV does not contain instance_id."
            )

        self.morphology_df = (
            self.morphology_df
            .sort_values(
                "instance_id"
            )
            .reset_index(
                drop=True
            )
        )

        self.instance_ids = [
            int(
                value
            )
            for value
            in self.morphology_df[
                "instance_id"
            ].tolist()
        ]

        self.metrics = [
            (
                key,
                label,
            )
            for (
                key,
                label,
            )
            in MORPHOLOGY_DISPLAY_METRICS
            if key
            in self.morphology_df.columns
        ]

        self.subtitle.setText(
            (
                f"{self.current_mode.capitalize()} morphology · "
                "Columns = instance IDs · Rows = representative morphology metrics. "
                "Click one instance column to highlight that same "
                f"{self.current_mode} instance in gray in the main 3D viewer."
            )
        )

        # Fast model/view update: no per-cell QTableWidgetItem creation and
        # no expensive ResizeToContents scan across hundreds/thousands of
        # instance columns.
        self.table_model = MorphologyTableModel(
            morphology_df=self.morphology_df,
            metrics=self.metrics,
            instance_ids=self.instance_ids,
            parent=self.table,
        )
        self.table.setModel(
            self.table_model
        )

        horizontal = self.table.horizontalHeader()
        horizontal.setSectionResizeMode(
            QHeaderView.Fixed
        )
        horizontal.setDefaultSectionSize(
            105
        )
        horizontal.setMinimumSectionSize(
            85
        )

        vertical = self.table.verticalHeader()
        vertical.setSectionResizeMode(
            QHeaderView.Fixed
        )
        vertical.setDefaultSectionSize(
            30
        )

        self.table.clearSelection()
        self.selection_label.setText(
            "Selected: none"
        )

    @staticmethod
    def _format_value(
        value,
        integer=False,
    ):
        try:
            value = float(
                value
            )
        except Exception:
            return str(
                value
            )

        if not np.isfinite(
            value
        ):
            return "—"

        if integer:
            return f"{int(round(value)):,}"

        absolute = abs(
            value
        )

        if (
            absolute != 0
            and (
                absolute
                < 0.001
                or absolute
                >= 10000
            )
        ):
            return f"{value:.3e}"

        return f"{value:.4g}"

    def _cell_clicked(
        self,
        index,
    ):
        if not index.isValid():
            return

        self._instance_column_clicked(
            index.column()
        )

    def _instance_column_clicked(
        self,
        column,
    ):
        if not (
            0
            <= column
            < len(
                self.instance_ids
            )
        ):
            return

        instance_id = (
            self.instance_ids[
                column
            ]
        )

        self.table.selectColumn(
            column
        )

        self.selection_label.setText(
            (
                f"Selected: "
                f"{self.current_mode.capitalize()} "
                f"Instance {instance_id}"
            )
        )

        if self.viewer is not None:
            self.viewer.highlight_morphology_instance(
                self.current_mode,
                instance_id,
                preserve_camera=True,
            )

    def clear_highlight(
        self,
    ):
        self.table.clearSelection()

        self.selection_label.setText(
            "Selected: none"
        )

        if self.viewer is not None:
            self.viewer.clear_instance_highlight()

    def closeEvent(
        self,
        event,
    ):
        if self.viewer is not None:
            self.viewer.clear_instance_highlight()

        super().closeEvent(
            event
        )


class InstanceSegmentationWidget(QWidget):
    backRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mask_paths = []
        self.pipeline_thread = None
        self.pipeline_worker = None
        self.sync_thread = None
        self.sync_worker = None
        self.last_results = {}
        self.morphology_dialog = None

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # --------------------------------------------------------------
        # Left workflow panel
        # --------------------------------------------------------------
        left = QFrame()
        left.setObjectName('SidePanel')
        left.setFixedWidth(325)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 12, 14, 14)
        ll.setSpacing(8)

        # Reuse EXACTLY the same LinkNex full logo as Semantic Segmentation.
        # No Instance-specific logo image is used.
        self.instance_sidebar_logo_label = QLabel()
        self.instance_sidebar_logo_label.setObjectName("LogoImage")
        self.instance_sidebar_logo_label.setFixedSize(230, 62)
        self.instance_sidebar_logo_label.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )

        instance_logo_pix = QPixmap()
        for _candidate in LOGO_FULL_CANDIDATES:
            if _candidate.exists():
                instance_logo_pix = QPixmap(
                    str(_candidate)
                )
                if not instance_logo_pix.isNull():
                    break

        if not instance_logo_pix.isNull():
            self.instance_sidebar_logo_label.setPixmap(
                instance_logo_pix.scaled(
                    220,
                    55,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.instance_sidebar_logo_label.setText(
                "LinkODT"
            )

        ll.addWidget(
            self.instance_sidebar_logo_label
        )

        home_btn = QPushButton('← Home')
        home_btn.clicked.connect(self.backRequested.emit)
        ll.addWidget(home_btn)

        title = QLabel('Instance segmentation')
        title.setObjectName('SectionTitle')
        ll.addWidget(title)

        desc = QLabel(
            '3D mask → voxel / surface PCD / OBJ → skeleton → '
            'physical + social instances → morphology'
        )
        desc.setWordWrap(True)
        desc.setObjectName('Metric')
        ll.addWidget(desc)

        open_row = QHBoxLayout()
        self.open_masks_btn = QPushButton('Open Masks')
        self.open_folder_btn = QPushButton('Folder')
        self.open_masks_btn.clicked.connect(self.open_masks)
        self.open_folder_btn.clicked.connect(self.open_folder)
        open_row.addWidget(self.open_masks_btn, 1)
        open_row.addWidget(self.open_folder_btn)
        ll.addLayout(open_row)

        self.mask_list = QListWidget()
        self.mask_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mask_list.currentRowChanged.connect(self._selection_changed)
        ll.addWidget(self.mask_list, 1)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton('Run selected')
        self.run_all_btn = QPushButton('All')
        self.run_btn.clicked.connect(self.run_selected)
        self.run_all_btn.clicked.connect(self.run_all)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.run_all_btn)
        ll.addLayout(run_row)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        ll.addWidget(self.stop_btn)

        ll.addSpacing(4)
        editor_title = QLabel('3D view / edit')
        editor_title.setObjectName('SectionTitle')
        ll.addWidget(editor_title)

        self.editor_btn = QPushButton('3D Editor')
        self.editor_btn.clicked.connect(self.open_editor)
        ll.addWidget(self.editor_btn)

        self.morphology_btn = QPushButton('Morphology table')
        self.morphology_btn.clicked.connect(
            self.show_morphology_table
        )
        ll.addWidget(
            self.morphology_btn
        )

        self.apply_edit_btn = QPushButton('Apply Edited PCD')
        self.apply_edit_btn.setVisible(False)
        self.apply_edit_btn.clicked.connect(self.apply_edited_pcd)

        self.output_btn = QPushButton('Open processed_output')
        self.output_btn.clicked.connect(self.open_output_folder)
        ll.addWidget(self.output_btn)

        root.addWidget(left)

        # --------------------------------------------------------------
        # Right results panel
        # --------------------------------------------------------------
        right = QFrame()
        right.setObjectName('CanvasFrame')
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(10)

        header = QLabel('Instance segmentation Pipeline')
        header.setObjectName('PageTitle')
        rl.addWidget(header)

        pipeline = QLabel(
            'Semantic *_mask.tif  →  voxel.txt + pcd.txt + OBJ  →  '
            '3D skeleton  →  skeleton instances  →  PCD/voxel instances  →  Morphology'
        )
        pipeline.setWordWrap(True)
        pipeline.setObjectName('Metric')
        rl.addWidget(pipeline)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        rl.addWidget(self.progress)

        self.status_label = QLabel('Load one or more 3D *_mask.tif files.')
        self.status_label.setWordWrap(True)
        rl.addWidget(self.status_label)

        preview_title = QLabel('3D point-cloud viewer')
        preview_title.setObjectName('SectionTitle')
        rl.addWidget(preview_title)

        # Lightweight viewer only:
        # Physical/Social display, arbitrary point-cloud loading,
        # rotate / pan / zoom, and point-size adjustment.
        self.viewer = InstancePCDViewerWidget()
        rl.addWidget(
            self.viewer,
            1,
        )

        # Keep an internal log object for existing pipeline messages, but do
        # not place it in the layout. This removes the extra rectangle below
        # the main 3D viewer.
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVisible(False)

        root.addWidget(right, 1)

        self._update_controls()

    @staticmethod
    def _is_semantic_mask(path):
        name = Path(path).name.lower()
        return (
            (name.endswith('_mask.tif') or name.endswith('_mask.tiff'))
            and not name.endswith('_mask_instance.tif')
            and not name.endswith('_mask_instance.tiff')
        )

    def _add_mask_paths(self, paths):
        existing = {str(p.resolve()).lower() for p in self.mask_paths}
        added = 0
        for raw in paths:
            p = Path(raw)
            if not p.exists() or not self._is_semantic_mask(p):
                continue
            key = str(p.resolve()).lower()
            if key in existing:
                continue
            self.mask_paths.append(p)
            self.mask_list.addItem(p.name)
            existing.add(key)
            added += 1
        if added and self.mask_list.currentRow() < 0:
            self.mask_list.setCurrentRow(0)
        self.status_label.setText(
            f'{len(self.mask_paths)} semantic mask(s) loaded.'
        )
        self._update_controls()

        if hasattr(
            self,
            "viewer",
        ):
            self.viewer.set_mask_path(
                self._current_mask()
            )

    def open_masks(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Open semantic masks',
            '',
            'Semantic masks (*_mask.tif *_mask.tiff);;TIFF (*.tif *.tiff)',
        )
        if paths:
            self._add_mask_paths(paths)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            'Open folder containing semantic masks',
            '',
        )
        if not folder:
            return
        folder = Path(folder)
        paths = list(folder.glob('*_mask.tif')) + list(folder.glob('*_mask.tiff'))
        paths = sorted(paths, key=natural_sort_key)
        self._add_mask_paths(paths)

    def _selected_paths(self):
        rows = sorted({idx.row() for idx in self.mask_list.selectedIndexes()})
        return [self.mask_paths[r] for r in rows if 0 <= r < len(self.mask_paths)]

    def _current_mask(self):
        row = self.mask_list.currentRow()
        if 0 <= row < len(self.mask_paths):
            return self.mask_paths[row]
        selected = self._selected_paths()
        return selected[0] if selected else None

    def _selection_changed(self, _row):
        self._update_controls()

        if hasattr(
            self,
            "viewer",
        ):
            self.viewer.set_mask_path(
                self._current_mask()
            )

    def run_selected(self):
        paths = self._selected_paths()
        if not paths:
            current = self._current_mask()
            paths = [current] if current is not None else []
        self._start_pipeline(paths, auto_open_editor=False)

    def run_all(self):
        self._start_pipeline(list(self.mask_paths), auto_open_editor=False)

    def _start_pipeline(self, paths, auto_open_editor=False):
        if not paths:
            QMessageBox.warning(self, 'Instance segmentation', 'Load/select mask files first.')
            return
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            return

        self.auto_open_editor_after_run = False
        self.progress.setValue(0)
        self.log.appendPlainText('\n' + '=' * 72)
        self.log.appendPlainText(f'Starting instance segmentation for {len(paths)} mask(s)...')

        self.pipeline_thread = QThread(self)
        self.pipeline_worker = InstancePipelineWorker(paths)
        self.pipeline_worker.moveToThread(self.pipeline_thread)
        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.progress.connect(self._pipeline_progress)
        self.pipeline_worker.fileFinished.connect(self._pipeline_file_finished)
        self.pipeline_worker.finished.connect(self._pipeline_finished)
        self.pipeline_worker.stopped.connect(self._pipeline_stopped)
        self.pipeline_worker.failed.connect(self._pipeline_failed)
        self.pipeline_worker.finished.connect(self.pipeline_thread.quit)
        self.pipeline_worker.stopped.connect(self.pipeline_thread.quit)
        self.pipeline_worker.failed.connect(self.pipeline_thread.quit)
        self.pipeline_thread.finished.connect(self.pipeline_worker.deleteLater)
        self.pipeline_thread.finished.connect(self._pipeline_thread_closed)
        self.pipeline_thread.finished.connect(self.pipeline_thread.deleteLater)

        self.stop_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.pipeline_thread.start()

    def _pipeline_thread_closed(self):
        self.pipeline_thread = None
        self.pipeline_worker = None
        self._update_controls()

    def stop_pipeline(self):
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            self.stop_btn.setEnabled(False)
            self.status_label.setText('Stopping after the active stage...')
            self.pipeline_thread.requestInterruption()

    def _pipeline_progress(self, text, value):
        self.progress.setValue(max(0, min(100, int(value))))
        self.status_label.setText(str(text).split('\n')[0])
        self.log.appendPlainText(str(text))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _pipeline_file_finished(self, result):
        result = dict(result)
        stem = str(result.get('stem', ''))
        if stem:
            self.last_results[stem] = result
        self.log.appendPlainText(
            '\nOUTPUTS:\n'
            f"  PCD physical: {result.get('pcd_instance_physical_path')}\n"
            f"  Voxel physical: {result.get('voxel_instance_phy_path')}\n"
            f"  PCD social: {result.get('pcd_instance_social_path')}\n"
            f"  Voxel social: {result.get('voxel_instance_social_path')}\n"
            f"  Morphology physical: {result.get('morphology_physical_path')}\n"
            f"  Morphology social: {result.get('morphology_social_path')}\n"
        )

    def _pipeline_finished(self, results):
        self.progress.setValue(100)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f'Completed {len(results)} image(s).')
        self._update_controls()

        if hasattr(self, "viewer"):
            # Keep Physical instance as the default display.
            self.viewer.current_mode = "physical"
            self.viewer._refresh_button_styles()
            self.viewer.set_mask_path(
                self._current_mask()
            )

        # Morphology is calculated at the end of the instance pipeline.
        # Pop up the current image's interactive morphology table automatically.
        QTimer.singleShot(
            250,
            self.show_morphology_table,
        )

    def _pipeline_stopped(self, message):
        self.stop_btn.setEnabled(False)
        self.status_label.setText(str(message))
        self.log.appendPlainText('\n' + str(message))
        self._update_controls()

    def _pipeline_failed(self, message):
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Instance segmentation failed.')
        self.log.appendPlainText('\nFAILED:\n' + str(message))
        self._update_controls()
        QMessageBox.critical(self, 'Instance segmentation failed', str(message))

    def _result_path_for_current(self, mode='physical'):
        mask = self._current_mask()
        if mask is None:
            return None
        name = mask.name
        lower = name.lower()
        if lower.endswith('_mask.tiff'):
            stem = name[:-len('_mask.tiff')]
        elif lower.endswith('_mask.tif'):
            stem = name[:-len('_mask.tif')]
        else:
            return None
        if str(mode).lower() == 'social':
            return mask.parent / 'processed_output' / f'{stem}_pcd_instance_social.txt'
        return mask.parent / 'processed_output' / f'{stem}_pcd_instance_physical.txt'

    def _morphology_paths_for_current(
        self,
    ):
        mask = self._current_mask()

        if mask is None:
            return {
                "physical": None,
                "social": None,
            }

        name = mask.name
        lower = name.lower()

        if lower.endswith(
            "_mask.tiff"
        ):
            stem = name[
                :-len(
                    "_mask.tiff"
                )
            ]

        elif lower.endswith(
            "_mask.tif"
        ):
            stem = name[
                :-len(
                    "_mask.tif"
                )
            ]

        else:
            return {
                "physical": None,
                "social": None,
            }

        out_dir = (
            mask.parent
            / "processed_output"
        )

        return {
            "physical": (
                out_dir
                / f"{stem}_instance_morphology_physical.csv"
            ),
            "social": (
                out_dir
                / f"{stem}_instance_morphology_social.csv"
            ),
        }

    def show_morphology_table(
        self,
    ):
        paths = (
            self._morphology_paths_for_current()
        )

        available = {
            key: path
            for key, path
            in paths.items()
            if (
                path is not None
                and path.exists()
            )
        }

        if not available:
            QMessageBox.information(
                self,
                "Morphology",
                (
                    "Morphology result is not available yet.\n\n"
                    "Run Instance segmentation first."
                ),
            )
            return

        try:
            if (
                self.morphology_dialog
                is not None
            ):
                try:
                    self.morphology_dialog.close()
                except Exception:
                    pass

            self.morphology_dialog = (
                MorphologyTableDialog(
                    morphology_paths=paths,
                    viewer=self.viewer,
                    parent=self,
                )
            )

            self.morphology_dialog.show()
            self.morphology_dialog.raise_()
            self.morphology_dialog.activateWindow()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Morphology",
                str(
                    exc
                ),
            )

    def _launch_editor_path(self, pcd_path):
        pcd_path = Path(
            pcd_path
        )

        if not pcd_path.exists():
            QMessageBox.warning(
                self,
                "3D Editor",
                (
                    f"Instance PCD not found yet:\n{pcd_path}\n\n"
                    "Run Instance segmentation first."
                ),
            )
            return

        try:
            if getattr(
                sys,
                "frozen",
                False,
            ):
                # Public EXE: launch the SAME LinkNex.exe in editor mode.
                command = [
                    sys.executable,
                    "--pcd-editor",
                    str(
                        pcd_path
                    ),
                ]
            else:
                # Development .py mode: launch this same source in editor mode.
                command = [
                    sys.executable,
                    str(
                        Path(
                            __file__
                        ).resolve()
                    ),
                    "--pcd-editor",
                    str(
                        pcd_path
                    ),
                ]

            subprocess.Popen(
                command
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "3D Editor",
                str(
                    exc
                ),
            )

    def open_editor(self):
        path = self._result_path_for_current(mode='physical')
        if path is not None:
            self._launch_editor_path(path)

    def apply_edited_pcd(self):
        mask = self._current_mask()
        if mask is None:
            QMessageBox.warning(self, 'Apply Edited PCD', 'Select one mask first.')
            return
        edited, _ = QFileDialog.getOpenFileName(
            self,
            'Select edited PCD/TXT from the 3D editor',
            str(mask.parent / 'processed_output'),
            'Point cloud TXT (*.txt);;All files (*)',
        )
        if not edited:
            return

        self.sync_thread = QThread(self)
        self.sync_worker = InstanceSyncWorker(mask, edited)
        self.sync_worker.moveToThread(self.sync_thread)
        self.sync_thread.started.connect(self.sync_worker.run)
        self.sync_worker.progress.connect(self._pipeline_progress)
        self.sync_worker.finished.connect(self._sync_finished)
        self.sync_worker.failed.connect(self._sync_failed)
        self.sync_worker.finished.connect(self.sync_thread.quit)
        self.sync_worker.failed.connect(self.sync_thread.quit)
        self.sync_thread.finished.connect(self.sync_worker.deleteLater)
        self.sync_thread.finished.connect(self._sync_thread_closed)
        self.sync_thread.finished.connect(self.sync_thread.deleteLater)
        self.apply_edit_btn.setEnabled(False)
        self.sync_thread.start()

    def _sync_thread_closed(self):
        self.sync_thread = None
        self.sync_worker = None
        self._update_controls()

    def _sync_finished(self, result):
        self.apply_edit_btn.setEnabled(True)
        self.progress.setValue(100)
        self.status_label.setText('Edited PCD synchronized; morphology recalculated.')
        self.log.appendPlainText('\nManual correction synchronized successfully.')
        self._update_controls()

    def _sync_failed(self, message):
        self.apply_edit_btn.setEnabled(True)
        self.status_label.setText('Edited PCD synchronization failed.')
        self.log.appendPlainText('\nEDIT SYNC FAILED:\n' + str(message))
        QMessageBox.critical(self, 'Apply Edited PCD failed', str(message))
        self._update_controls()

    def open_output_folder(self):
        mask = self._current_mask()
        if mask is None:
            return
        folder = mask.parent / 'processed_output'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(str(folder))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(folder)])
            else:
                subprocess.Popen(['xdg-open', str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, 'Open folder', str(exc))

    def _update_controls(self):
        try:
            running = self.pipeline_thread is not None and self.pipeline_thread.isRunning()
        except RuntimeError:
            running = False
            self.pipeline_thread = None
            self.pipeline_worker = None
        has_masks = bool(self.mask_paths)
        has_current = self._current_mask() is not None
        self.run_btn.setEnabled(has_current and not running)
        self.run_all_btn.setEnabled(has_masks and not running)
        self.editor_btn.setEnabled(has_current and not running)

        morphology_paths = (
            self._morphology_paths_for_current()
            if has_current
            else {
                "physical": None,
                "social": None,
            }
        )

        morphology_available = any(
            (
                path is not None
                and path.exists()
            )
            for path
            in morphology_paths.values()
        )

        self.morphology_btn.setEnabled(
            morphology_available
            and not running
        )

        self.apply_edit_btn.setEnabled(has_current and not running)
        self.output_btn.setEnabled(has_current)



# =============================================================================
# Embedded standalone 3D PCD Editor for EXE distribution
# =============================================================================
#
# The public Windows build uses a single LinkNex.exe.
# Clicking "3D Editor" launches the same executable with --pcd-editor.
#

LINKNEX_PCD_EDITOR_B85 = 'c%1EhYj@i?lIVB;3aoNwhw7H;SCUSgcAs@@C+*XY?XjKio%Q<a&=MtcB8eK3vaQbMcir#zm)&{*#Dkzjxs#dQd#+|WmPnvbC=?1+g+k$(I7ud-PUA;$esCy``bm*y;{KIb7neme=|`h<5{q}U(MTLma(L2<WudiJtF?AKDdH^Z70G=pb`HffO(w;9Z<NmZB2mRh(QP~uF+(1}f4?P@e)^DyYil2(WFiK$2>?hZ`P#bmS=$Q5*)*P%kk$#(DDESw8(05+#q9}d55R4Q;xNglqi8OQyV!gwW_dChiloRjS{Lu%ie3tDGpM=)RkLU)B9Zs97~0I@9uymnVo{_bk4NxR#P@MF*8qxi*1H2_Zs#JJ%tbtbrgHdA?Y>zPLL3y?==(mxEz%*t%0<)%GC*IWQ3)px#lFbz(hPA;dY@!#xj@3JLgCu;P<)89VJu(AeG|YHba#+sc~N~UKE?4gr>-9#0-}A{>2U2uC@!a?q@aeX^&XPquB69$K8<=v($RdFPC{{xeI#Ng{TQi@-IZ^Vb}=$UBqs3#w7S79zG2Eh$7TgpL(FKT(kvM!NRzT`xb_mTi|z?F@s1*_9MC8i$N)h<9z?TI(FPdE6~I%Mv*{FwNnM%F3ShrEHs~;y=CYsTs2^uAI0SLy;PTg9?2LF6bMd>##8BM2ze^%@lfLn8@+g20Ce=#zi<;W}P8F?vVbxySzdSfT#*Q=2JY_SQAixmusGsvVK5oo6e(#8Ec#DL}HijFw9g$8YD^AjE9F6i0v4?s}0FA~rX!eETEsPXQ18gD5VXUJ*j8T@(hIceuK?*$FsY#2I6@~=aQ%^UnRzN)*A{a&-U`JLEfM_D(N8ryMaH!d5>K4L?Zu59jgyNL37d@z<kRQ#Fso=GEfBtb@Rqu<wln?N?m%mwqwqf#Sz}Q)w$Jss3W?&h7GD(5e?mz_a4Cna)us4tyL6|_dj=764e|p#j>WP$vQp9bX&kydRYz)HTKFQL_7%<|InkG|4A~Bhbr$Es8B}|H!VmiN1;JNzy_d?tkpP*wvS{U$)*+c@E7k4n%Xiju#PIOBu5f!^Eal`4HB=y=LOUI(y9n6YZ7I(WM8RIkoGDA-zmcLwP8Wne=<W?0phoAhSm`_QH^7F%_S9HWN%KzI@+<|2zhu>#%06O9_{`(AMBD}epO-J#XtcE<0<?~$qPRnn`!`mp2Up`+`Kbb(lE0gKk+B0!9zK#2R7#Fp2-DbeJh+35Aago1)!2{j_%#gXyPv65-AmIzcngLcxMqK&-Fd%@m_ccEN@8jjsg}6Mrx;j36fBDtnul;s>dU*C*_v-lS<Vfs^7WV(_@Jjrj|6QwBFn_n!-tHg#^8VuN<LP1d;Oyk=0;@lJcku4@yBDps!=rcmA5X5j=V!;KSKZ6w|3K(F;pW=q(aF)l)zM-1`1JB>|McLfixO%NSS*aPESd+`TQ4>{Lj7~o5q;qQcn{vs0GU_qwGYRq-OKZnV+s5cDsOGBz5jUJJ^y%d&fR$S;>GsitLH6xe6s)R(dpq4Be=i+YV++Dzc}2#_@(>);^^p<UTtmeyxiX5R|jVwFRqMtJ6k_&J>TSa?=SW*FO7FEpYOkY@dLj*-#<ArUhQum?Ei3}Kwj*hFp*xq{o$<yc7C*f@H0Q%Ie7c_6+iuL|KdaUuSX{*XTM3Z9=?11wk4@|e0JLX`S|_MC-C1@NqkKK<io4GQf(lijZ(eHX2|1~YB2%^Pj^K$0Ny4^+Ka{@TCmPE%}HsB`yBv?5>qJ>11jm!UG6{=Q>xrFOaI2QTa1#}Qd*vssNn<}r?>!szK1|jb<x__x0q3f`&awo^6KK_!PUo$qf4=WdMIFQj>OU7G0rk^a`yiC;OkGdHPD$vH_E%j$6e3~0{CwiIUkh=;Qh8(|H=5t3l8cIJUw!rAn>5gO2PnNLg{2I{$)>WlZpWk^1EmnUvJ&u!yN`ffT9Ej=htX7i;uD_&4L!ryzvaQ*E`hssG0~)J7T9Dw#p`;<s{E%w=@m0@sPfC@Wo)1Mz|=qNrR!bliy?A<RF=V2H71)kI)4w9ANjN2|S^Gw9BH<l2Q6cnA72=_+D(q>n}AtO%MF{JF)%Z1&9=S8pHBBL7Xz6F17+n{S@2i2#_RDu-yg<QXjM^kz)cA;S-Jrjm*w=(_z`!21F_Y>Nc~p-GRk-I^TN-%2M1mu?+~w^)8@zBYyl*Y`tuY|1zF#(bHG>uza?Cqp+S$T)gpj++T`17qgj)pE<&IXS?kX7qp1kc+!RG+4~ff!xG5q;-{ae5C4kv?g-=QtMZ|IM!y*P;&F-lU1FSWkrIXPk$>~0edJ<xANg6y*!+{ZIs62QBkGc^WY*IBR#HlqT|TlgeDl<=U|~?2x(B2V@00i;@RPoFuuEFu8z7NMvPOJdqUf6iEATSIDf^&0rO-1hiM)$iZDaWdO7wo7C&LL?sX0i6@E4`?rx_TK%9kMR@Mt=Xqbx+tdwK7u$ZV2;!rbBdfM-8`>_7|maR$<oMvhAa&Zu@|&$-IpRfa&yM^v32GLNsZIK(CxZK?4Lb5OJcqiqwZE-0a&xMVz@7V}^fPXd0`HW&yx^zwP7-nG%%jcN|<s$i=l3_vJk{lMhNnlNR@N*Odx`*D{?z&n^^f<UbTNy`7+G~qjA4kO3SQ0p+q!WjZCLFBejFR=Ll7lg+)xxgNgcRZ8nn%lbOF~QY|LAGyh+Si*mB^$HF4(llKF)Vtr^LgN!?Usll4}ke#&7h8142?S`mydEcl6Gb@ws0}uR0cH!V;k4_V127?s20@~Oi4)O5))z@H41!)g%&9mMkUe=P|<*T!{RO%{bVqRGh4xkCN}EgN3q$s0V(Q9k-0?&!eo6C4<qsiK)2!{a4a@Q;GUs3Zj&_P(I|ZY4rH%Mfsd5mV>D;S#H4s5-xz+AXd(fj9fT&hR(~@v*-48AZTl_(dvXF6b??(K177L_ZoOy*{Xh>+FP#(-)D!7sG?(zm<Mt?hXzX(X(=(*E%w?b&O?Q{^2#o0GFI34RXl(w#MQ^W_Q!nIr<kY-mEVJ}A3lg(Xsfn5IZn3XXmZ-&dFp3HolHcPj#dQHkKhTY|N&GO-BFfY(&$`#3v}sTbI-d0Pd;OB7nE{ha<8&vO9ju7+{&!FaxfObp8ysfx6ZX3>;9XOl8?CD9!MHZ+A>l1~sHwydcA?E1OFY@5p~05YCVuGllN>bcUZJ2;eSDFCqDP}1KT?t5Zs`=d_A<uW5{azfbV-!has@<Q5vT&?+Cz&x_h^`ofZbt{0DBLrZiX+nUTyB|ynglShv%<fy?#*+NmLZLmf?8tYP9|0r7;k9GhB27?jUIR7ToX4+TTG7n?G!|#h=hlC~M~t(3PBOgu@TpP-QdRdIjZ!K-K*d*f(NBY=@hANJ)lJo!2j&akdGe+-Tuley>#0J6!p7(;bEetZf=7E<iF<_<=_OF8Olc3=0`pV3~?d$0}O1;K^pn3dmEmgTbhba1US@;@TECOt3ENUZ18D=;ZEAM?_F!Z_75^lw~(gCP4`t2=LRM_yGp$iKsRz2%jGMFnDX^09{pwX>X=IzBH4%Xj*o=T8GQy(V(L>I`V5+D|Y!#I_Y;y7pt*g-{@k_0^jk8mZd;)=n|AfPq0xGJ;5klXbj4-#beRMm1W8*-Yy6`WUZFO7df+Km+r@-WDFLzE-Rc8cc2Q~(Z0+sNZ)h>Wk|1O9d_5EUMQVGwi!v!T;oP5l#*SH+P3q;d{N`XAW~P<fq(z&e<<J7Vt<CjfEEnhSlc(DDvew@dz+{*pjPg{#tjVenXo&<Y9|DxBWO42O{pN9MyE)cWWwN-C93q9vQgfP0i2La>=p<ERf5^5d~e&U2EW_mTJBWU53G#|j~@<RWl-(r{3K9&OL{cP;TVn8u!156WHL$b>8bp-W=_(2h1cK>e5Q0=bNrT#9O}8K@8Ssh^<CMBrR;hK&C&7e4d}K#D{Y;E9%;n_0WC3MeC?C@Qohv<J`?A1|E}Dl$DLoK0xziXW7Gp}Iuz)fE-iKR`GR;(vRLZA3@eYPV4wnvBL!(I5P^z^EP;g^l_lvRHa8{uQ4xh6v7~j=5d9T_F<YpRA!wl(3fRDS)Fvrq6v0_y&l`f`h+u%^JgF8hWw0z<{;n3)!Whb;G!E>i&cG{gJg+mjD~MVit{c}kc1cj2Q+DCeOQd}-9YaTx+hmj!b23Wah>uVVc@q|NR4V&6kSuNl<RW%}5R`wLX6ZD}ig~$0Pr5wCXu6T2ny3YwrkYZii@=?e?K+UsZvo`cKwamO0b4s4D)XqeiAgyvS3z7OrjSQia*Ld%brXDTX4i1Qrb_^yE@(<+-$BhsrchhM#2B^k<r;=He0p2yjN0^jjqIY1`IqF$q)Ok8<-+iIE=)R&F&d|<xLuQ~RF^=vxv^9g-lPI9b#=(fyurTfH(GAqn8cCeNR`^+N3rcC6G;oLT!8@|7i?Xsy4RZ;tcrNpQ$oBlhTNPB&}X7JxL=oTd-hW6*N$t65(<zKLXD&~%c5x$Ph+qLEUH-e8rB37Q`T-v+d|t*y1O`JU<o1td32b9Te~+NRY&(7P@D!_uR|lUKxQI?Xi}}YygX=e7}#f3=L<63=d#9^mS)i$!@Ek+AZMTow$g^XZk>m_;ytZlX#<jURv1s(70y@SEeER2FIKtg>@&?|i-lE{YikfH5YZ0Yv^BMV^_2Qi>Thx>@Ktbl)CF;aioBN?da4y^0q@GtHzpu$ZOwq2m8`0?<wCCLj6@rh2X1^qy?8JnI9RPe6{I#>*@IGc4}IFzg-yHPRGm_yc)d>ywW3#sY?TBCRWvN0KS{q`NhT3NNXif%b>c`D^9v&vH%A^Z*%wG7SHdzj3NL=KSkM@p)RCR3iX0p1L5rhCOK<aM1IU+M@mZt(MH$1m!c!=y_{@!bX&K#-ZpJdit~<?Q+`l6ggoT;%q<Fk5ru?cX7i)D}d2eUQs4s?DJSUStwpf}57&MuJ2{E1(6j&hgBF##7I2ab?&fa~J&!Fq9=PNuzQ;BLT)t{zVDFe&0b&D)(e0CexTEGZz1e58o($&66{K~uIlOu6_dj9e1Tg9k=9R}uMl=qUPJMHxY_D7wQWdy@o=V@cJq&rF`v9dsL;nEfeT5I&0w>nK<1ieD<x%rSnXtD7S!!UzZcH3&B<q!Hoe`JQ4fOeq4g0&hwP&242#FH7uGZk^bwHzCsitUv(!hYNXB!U)ozSR*i8us~KD;a{_9Jksb^eCAI_S$52Jj5O7Nk70J73T;wF@pkGfRJ!20{a-WC?=`ZuDV7{x=?M>Edr3{^kr|WeZ6%P;!alPAmt~x@xzq->T{dL(I*x}fUaA;;;5*4KyY(Q8m)I2IGS>}6``)T{B*DWWlu<2FMv8Aump%I0Un=A5MMfEt`LyYEygj}UM5t`g3MuYto8=WjVOcO)@sl1(Mv(!K#x~riZDO7-W?yE99{zR@VC|%dst<BI3V<>22PtK3=#C&8><eFi)<geD?7N-NHAXISd{Zl?gFGyzj!QYvg2md7$-Jx{xj%E{wV?5_aWbNI08uDA19N&ZEIvL-z~06jO?NfU1z03+$FY%`)32l7r`+|IYo>U`c%Yf_)BQ6bA+1Y|Id?X9E1K%6i)zubSYY5LbYR38{M|5BtydL>PgUg#KE59Kfkxy3pcK@XXMnB4qgE?oD(noF2__&lhVD8J;TVaYiju#8oTKT`F(DD|Gjl{Bb5mD@pTbFgb>J0fiB@Gzt4^DzsokY>r{`3)BNL&W>&MA8_PyUf<4;Qs?^YP%66}{OZl^m>=|}pi2vN$WM`$op<8CqYG{1FCAu*l&}a)cPRbTtpx&4?)skvxh#=cG$mkElj0~XpQXy<H8PwiD2)xN6-K?N`4~g?`4bEfI_i2PkpM5lRXC4fD3=I7o2SWYC?{pl39`;-0Qnkjit+$@^#+I6_mhW35p!Q8|;z)FXI>tsSo37+*m%VYhjA+LiTUDbece2Wz4+<r<y;kHaE(7pD)^=~{kPa$MU0m3^e0-#-0##(D8&00nAPA`Sq&sZz#E-prT8LwMPa}?R;i=0KMhRQMwv4kx^7KM34-thzovr~>1G}(m9hqTl4B;y1gbFlHkJu@{hp2{08WKs2lQIFU#6*j>BjU^rb}Se!-bYl{f~O+y&(~E{d!rtUl&de<=c;TiNl1F%OYPFrCe_}aFgLy&4uneT9Avp!=HDl)&Tx6+tC!Ben!*)|&Vw08D|uXsJ5ZJIcwTE|c9(G2ZRW5#M;4-RfvThqxE#(=CNSJ)fJsH2NBw#N`ZgGPd#zb9Sbw#iCqs|c-9+hcsPL2iJ|$Nx>vtY~9(R9KF%*zUVD-RK+n@PVcIBU!6*>l%YyQ#XW2B>4@A`taMhZ>K2%sH+rivkpxMe4YDUix84hJ?jiVHjJFb8u6ihJJGMIc(WOSdCf-N%RkM$2)G+e0d56O^lcv9r!9nV_97-roaO++ZYe&EanwXutCcQ=Vxy_XN4rXl{k@VT|XLgYS^$2^~;WVwe{VIz2g%dxRM7w@&UTausieA;#c{%UL$S<1T6jmu-y_;+plXFx)Dk(T?+d+{2U8ypJpBkTD)of$4xJhXE@N5#&I_JQ`0&WSPwHEOMT*K!<@X;vA>vn8tMg3lwuM`i-4))RNF^)j_pImG!DlRM{i%?hu&oY9UaUy0kwD^<Qkl$MA8}+os|8a#l$f6-MFX5f8HkuS=85YFAnj;}$l$GPgPwu#F81SY;<lhWGC7B8Z#oU9+&%nb8cAow5kbux|BRn|j91lvHl>9>}aZ!Vrbs12Q~1WyFZ=6a|ZTqOD|r3C7HF1ve~#yS%=&xoO0nl?NguDY&Xu+U4Xjj_Hi!XhQo0x4AiJTjAyl!_TY294e7yDtg=P!E8e8<ZVU)TgO(YwaswLYz=q6@hkvn%Vmc!=nRT?9FX$;k3fic{W(SBZEcz{f09ny;R|>ryZ^oXNl0SsApWQ{s)roq@&C;6PB5r)GK?!jp$PZW9#M{t^OVHg<A_FiFsp6)`_oCc47+w=x*(1IvBZX%_^DdWVm@FBrT$BTF+0ZVCWGX?2<G)|I+<bL@eG+6pD0hBmGysu5-(USGimosuTQ8_q31)cLSR<)?!*`dT&_42>z&(+OK9>sAD!T+Sd>_L3(lG#n{A9qtx7hNUF9;!-CB3~-fXK~;d0vm*do4GEMUF*UcNNP(<xxm_&aL^oq`HwbI*Zi0UdYAb(D^rL=o;sSz`3oTFpFFZYxzHuEKFwxFdmlWwy9(gDbraz-qb@Zo6%UJ1?3#ty%|$E<Q@4+Yx0uz)g%<lHV!aoIF>gn@jCTJ_|+52Dje_-BUU6IE_l2km~j4W$7vY?7~XRO7$5r{#<pL4NMyTWClp8J#kwRan%ZRS}0AmW!fKZK?jZ%!fKh<7yy-K-Clm*)TCz6sDWE2ywK}XW5r1BG2*mZofZ;<{!9#rjy+eTpW+Gob+yAwm&TpQ5lD#u;B+wbwyk{26m@cAMDXNYuwicFEFh2+txKK6ATT`$mbZZVN^*ee*!GMhdwCiSKhle!+1b0O&^bSKa^@sWKOrAd82@N`7v081(!<ylOhpuD6v;@o(rhqDdXxr49eSaSNeL&XR|ymmDhGzPd|Gi!1lU6S$SU<#xE-&9%E40A3W`&%S%+ZkPN%PQRFK9+$`R?v)*?Y}g#=qT3I=pu6xuaTa$~QcxYLfTl56(LZ~CR>Qr8QXa;>W`M&vpB)=keC9(7|GvS+*6d5r8}M0*~|DQ$#!>xQJC_?k$iOk!(u!l7m*Ac{S@=)x#bTpCZ!S$~v_L%WLzEsaTe>dzBC+t`)QNPTgYsi%7L8+D_`xo>vhiJtv(v?|VhcI&f4|MzM%`r`gtE_q)l`0CpTx~}cMRdZ>z?F8AA)wUGyQ;phPku>!?46Cz(_O_d>@YJ1(vI}L{KlY{?MUE>lAB<)@sY-OWk3I%i455Ai9C2i-12TDq1pTAk5)E8cdQqoqwkGgn|04XMXyxJ4_Ae@6-2DuzfU&uOAN%Bv!G-04#;>%iLEBBnBC#io>=`v%8huj~ZSL;eD4(x-wzK<EKC5ZthsA9)0KbGHFFi!4!R2)om%1LBv(UiU=GsM$SM{+8RU*Cn`o<aaD!N;nF1`+`|9B+phv9YoFf81Vw5^e9;y>e<&&!VZ0UAQ5@z!RYaWrVXqZ8GiOHx)h{Oh+mT>fyEEbB~P$0UhyV9-+goKqir#`l^Ncv?Tio-R0tx2L}<J&+9|)I_blwT1?fdE`Z#J90Us%^5lZng^9IhIh#*K1~ZEJ&&tZpX!q4uftk^!CT?9*KaMbT2zJ;U(lHmBH)@Ul|4_1Mym`4YVk1!+KCCEsBfzVJLbh^*>3K_Zc`Kxp5ej8BLr)e{yxrsH$#D}mK?)YvT#MgaC2;8M!HFkIRtkF(zzqBRrnVGk+MOJw!d8%;@QWmZx@G{gFrACVsS94!~R+o@-zIRxKUY+VlrfeMoo4d$pw8Uu2jbeRK;cKT2aMok8YN6VlC<L8Cnh-E%5M2IF)-Y<p1<LY@Ob{6a2qp{jSWb(%KU8fBGf4u*oPaF;(lSCJinMi1ZCDvgI$mq725UH(s~?6uu@6)AN^9tv~f!Dg-BAKmuids4sK%WhLq?rBM?~)ziyfbsggQUshHgn*%kOZE0taT)(Ly{*CxfycFN_Pfc@nswjs_>pv&0@Na;rUL@`Aw4R9&qekvG!ym*tPeeg8Q5>S}PP^Cx>`l?Kk!}5Yba8onc8X~koL2)GF)2Rtcz>z9r~HwhX(X#Jul{-tFKG~~FAvT>o?eO1Jb=~Lza1Z5{VYD~fskiU+>))Ik1*FF47l5?<D=i`OorH`|84Q#N<%VLWrLO|%vFXTJA=yN256flkF1_q`Pi5RxsV%c@nFSTsOaWN{D7{>Jsfv$kOlt^@eFPpNH+PN*u=2(ry)vebc=EEjzqD89JmMnbqvQCIzCPLj@ABQlHISpt|Y&yLKE>_J4{u9s_ZZ$$4u#xv%FF0Q#Y)2!(K&^w_xV)@XYVikLTJJDN(5$Wp?cKr}wKO<m!5APlLtXc3fT#A)3fSiy}KpFuLzNYtjn+uXcUWNH6SvwZjV-VTHB8<Op-m!(t5kkb1dVl#hAIkS<`Ct6ZPDJilW|jT);;3K6plBuvT&3%%{id@@WY{Aee9%eJRT7%9X-(Mz&rVhOAQZ`9j41<+H{7AdD>$x>Fy-|djB3N{`BHk6nzkZ!VuD(sX+<3fe$c(r7Wpi#^eww<ySJCCERrC)y&>*4(D<gf2R8z>VRi}y!oAC9gr{`!rKiE91~PE*`1Z=7I0lv#^G@p_zB6Z+gId2$O2WaUL|=07Gr<^PiWLDqxZwU>HZJn1iZOV?9$v#6iU05-h=#&{iWb~uQy%=J*l>Rs?yKHR0FxXi*&`7_PSGpcuPV0gH`5kqIuG-zT+l9_Mn;>aY;?27WCs)IY5-q`P^xeZHm+(1Z&)drPJh8?j@cUMrT1>;g?e9<l+qXr9$g;Dpa>C*8?fH@mRG%?a-<3Q+rEdu$t*%9+i3^GGj;YVBWXbT<zchG}TJOIwflHnbWKc|-q0yah;H`^)@<Z(;I<=m!4k&a;|r_)M>xvnrTD<pZB??M5RaXahODh5M54>ze70YY1su;mLyeN|tRiS~W%6X<422t|5La&|<OXIWKMf6SAS*#wpV+N3oZ;MxLl!x2P^ovur28FVn#<^1onC<8DmpBW<`!ywMaK`Z%NC;8j@PpCvw%kqzT{uyfH)_>YFPnlzOQA~m10Cy5$6FP30YYXHJMlW$eQ$DoCci+`yq%er3o*+`z0c#murHSS?3jbSE^E9DTiF&%o!7P9OtFYvQ%ya$I9<$VvVn6ONr!OqGMc^bWE-45Afpgo2A!WzbafjL!2c0CFNixpwM;l3;=O{7L$2bH<s6;hdhFCOo`Egff!S8PO<>JF{os~x=%ctk%F>B9qQGe|ESBhm$Ff<t+ulFJfOp>XTlF7PC$B)6vTI?%q9GR~k*9m+D%&DdNP*)A5x4h?wqZ*J4tOWLDRvDBX6fLTeX;+sLRMN}$%vSV#?mpkTv01u#3ra*w#IM&SsCb$uBSnHiG8v>ScH1Gmp%S!xx((I0>DVXN*M03l^EvOjLjymFIjv3sl*aF6vBxfk-(|VHX8}q)ZiAQup@u0hF@GZnHL*bz@jEE}5;Y8)d(;PjW#9kTUfD9tqDP>=-Gb4w>N<DAoFfcNF+9)2?n9c5`shQI2D!=bPO4;B@8S%UcKNOQK$rR~Kd_+nAGs9c6(-a!sQqBpCX=Y3dnk%j{Cf3Ec`4B}d5lM9k#YveyD;`joWl2Uu}@bG9ndT3EROOiHfNW+N=9*ipWa+xkW*l~Z@PH7(4K4nOCO|Tx^E*sFR#tG%A!e*&a@9yCE!*%7^>Q#0O^W5h^LzN-CgU(u#KZpOgmsnNBoV&*WYv=*5_^B(@(5NhPCBPS~D!^E*+=CcoL_x+-De<o5IsDoy6v}YI}r%T!Zoc1^{M){x{j#@QM{nnT^JbrN&xTBsKD}j3dT(v;#_O=T^MFCC@ES`aL;QXHd~bZ>P;H7ENjoh4p(n;|R4uzdB-f=LV*p`D90WYuBWYu8(%sT`V;^i`{X&E|cG5$u;+=Al}dMADnsjH1$kw`!$IlaBnD3(XY(gM)G*OU)w&?n^omH3?c>E#t1p=CqMh1!}GtB@$pb#3#1DaqbPgomJ4Z%A<a!lTOLSrf;4AHOc}?B;8BNB)x(Lwsg@Bd8}5`U3Cp6t)9G563`z)f)fG29XW;##G_X4A!Q6d7-Qx^S>Jj5<Ji{j3a`?<D=t;+{CD=iVnf43u7tn>%VS+j&iK(3Q?{(CmWICd@km8KFK^Bwetf2Zh#rVtk9=#rTI0+Qhd(iCGG0FrpRbZ5+-9!frNA%+(Q1q9SIm)vbIgGE4<JQ)hS!t7n*0H;JZX$~}{D?c~^!nlXu^JYR<h+DpsDs?{t&w=3Eb0|AsLMMxe1GB;R9m`+YohKg5-AX7gnKPB>aV;6QS@e6PWh$9*5>A)fV;L|{E4oR`jfn0kAkP=1##v4pjdX}?3Fpl`lt~qed?)H)yrY8l;X#-Sjh=?vGTD|%m+?CZ)=?fUr<VUY04qn;&jn%Z3qY3xm8VL36rgaF>fm%^4*4Tds8_~fVK>x`v`IY1d7IBU2GXKP^#Em7c1LW<-W}>_4|*<-^ihq%_iO9ED3PkNum+C%JwM7oV`ndS8?HW15><PfP<YAQbI5WTHcvBfZ0MfV9Cn}Lt$J)$Opl52@Gbmy7uw!#S{-|rs6MG-F=u|>IQ^7#y<l*q#}<D;8tQZj1?#V5=wCA^S=}?faB)!h@v{_4e0PO#oKr^jyLX8n9c)`9_m5_G@R(9BY=+?Sel3Oy2a}jW7I;S-<r-*9el}i%L@OXa}etOMS0%+FQ5r!aR>gspC$O`x1>Le3w;PsmQ&X?Mj#{ZZHO-E0LZbtFE`CxIKYdnu`fSWKc!%()`ujg;~1E~N{P_kd>1#^9(!%Kv^q{tXX#s&*H62T(u^P_Kh)(sFXHjhV^RcL#+iIx1FpI!NAIq>Z$DmLo#C}~OyUrj-~+A=Z)adWPr?&eQuwQ?bwSt5Eva^a)7Y+eaCUliady)EaCUh7?wIb~({+D|=eKE;^*^NjWRQUF0xMmSr6cuQ0lED77;tU|;_@yT6fS_D&n}Mt12CaG_jKhmx}KEgVfZtwpub~$?nqU-I{J5LntamxZ|N#QUGpk_Ecpm(zZ*qEdUi>9!!Wz;+c{B20sC$L7>M!l;_}SK|A1DiOSEyr-=d`8-(2ew-LJh_K^0&Igyj<+%;|kN9HqA)Z(w4N<IuQYWKXy^GOkHQ8q?z+>l>p{RSe8IVK_akZ26qHavH%}U+i(vCSU%@a?UweKjXAkWo=oH|Jd=floL<hOuZ*zRtxCFba<C@S31$lq?nHCq%!0$h0|k(gW`O;rBa^!P9V=GU3sf)ZOI(1L6jSZSvt)FzJJK3Wku6uqLMyVuK|`)Le@em-SgI>1Y;AlTzEUSDnVF+b#Ou28Y@3AMz(EHiV|Mkmubm9{KCh;33;~O&V6qQH$uAhE%(FsT6fs{iYwah+>yzFCe%&Rkh?@*b9B0VzJ!uk=;5ly*n`^mF3leB`sWCj=W@)XUe1U2Y<-zs4tUB~I~&x84vom6s}2EUVpT}WiM(*c^?Y~JU6e1v%eXkg2F_J;h|HqWQhI|(t}+KMaFn;4S$8Z2bRM8l?pY_C!v~9+C5iu}M(uT_X_Lp3`}9*xqzN>0I2=WMgc<SkBbS%jl{nQZ=B7|I4s=w?ktw9hhMXM`k9$S;$uKJ7mbVJi<Ca!x4$XCzc$*T{VA*HlLo{Wl{5lF!$=`p@LA~$yNAWttF3E;d02YBsJM?$eBWA4=idJF&Yz18mTW0k`Q@*9BUV+6#h)8C8@+Gnabcd>DVCt!As9RMst~|Z*qt~pJCRBBH4E(IaOJu8xZ8-qG#hCof6#=oRn(l}ER$o)1NDXy07`<Iz-|~`C#*b&D7tu%h=NA!3R^br1F6GgsjXA=zqx$<b5mp+PD|aLhRb|tM^geDNn!K~K+Vh3;(rWH2&&O{dIwk+tSMaibT~i+6`J`EIqB@)QLA0+h5mqOy#ZQ`7YT4-maEbCKb5yfcX+^#>)&mp0dJb%^@js2If0$pFtuX)Hcvdb9wHPc|+ZL=CeiGI!N?uy7Y6?2zguA#&e&?Nl@|sDWhzmshADoibe5;+6hw1Eg6d&Ny({yIm=r${^gdSq4O@B9u@>e2ArK>)o;6NXtO8OP4(IZKLSOXuWQ>Bk3;}}mt1nx<3x2xw+C(Yyje$Qjd*L0lgb3(+x)5TTR_Wy}JOs8M5cj!f>eogT-S2Bw8s*HAfyi8hti|(`Q?ud|<rhMrT5C8HZN+$etu>-HVL$Snj=W*8MaluVpB?U;l-;PfY&wlG(9bcUsc}jyWiteSY=Px!p;`QdHA7IsWRw<=Ju=Q=2m00Y0&)+!C?`mJ(MtI&-x=s8=VA1O}DS}?uCgsL>Ilz)KeDB=iGAcC!ux_yo9(9w01I3<oTdpw{l{IZiZ+tbL?4Om_8`StO<lQd3kl>y;UzJ2b&%PKqVO@Avfv~t=e4v`|*cFFI@Af~Q$UAqscm}On;a-MabhRU1UDwdc(~)@?V__YRVf6X%8&&5B-`2Ck!=qR4e(<!}y~T@Hy0gSLxx2VIIqyCsW4xjd*Xv?Fa=pV`QtnlCbGa*#*9Z$DLZFA#kmvZJD=@WGs!w1-h0T)k42AiuDqhuI%N0oPEJXk7D89T)i>j2z&t}z!jGs|7zU@b1x@Wev%RS`N_jH)U<D51+Yu_1&AKE*Xfjelx@C|lu#4f*AIx!A8bVNON%97U}mV~G%vOq`Awba*^j;V8f;QoR4VAN{0_OYyt{ZLuqrqL)a3d)O1H+<e@={TmVA3O5;&g#uQ>Z4fY65G7Ii(}954BSfiF&4g#_zN$G;`3+kAnG}?PN~6n#b<YheR<=psft6ty5}D+&QFeN3JyTsUmrx{WHjFut;=|r#^U2~>kYp^S+Xm(o=+dEwZD9+_U#0B+@BS=%F@@mjx|Pa8uj~hJN(P(qu7T3Uh&^%=)B9Kar}(=kG?M_^sXo{aFO@@{5ajpE2|M5l9CPj!FATvF_@Q4d+;7lHNH(Bn{&vsckkXEzB^cq`j(CQ$^NfLr-w&%%+v&V{EVY)m`v6|oI?wnb*1Exa<Nx+rHK6=WW?Fa&CS)wwYy6^SVy&db}P-?bq+(@mALzH!FaU>J=ztU!aGy&)0pE|7*QVu4$|>$`t=y5Ioe-?vtBHv@Cz0JtI=Y27p<jrkhqgtvPkqNu!j3lkP(sFjfKzoEWcx4-U{L!X?Ae-@#3oc{^IE9bg^`QxJ!x}Y-@(&aR1_$g^*VskRPCd*I#!MYr^f`VYuXq<a>Xye|cHk$t4Y9T6_!)Tdm23WL?r??=Hqn87sEfWU*;feld2>@Zjx}nD`%$;FFkTfqe2cd<ioNrqwP$g2tdl6@pdQtcsbgT{BJ1g9)*Ja<s}6-DxV16z(!h#&X=H#M@f3-tUTal**pT?ZLKoy|2<?6^6J3^I?|1jaD80rSfpGr1hcl0R%`ijA7XU^&Yd19fI|h4*y9iz6V3sC~>W-($@AEQaw<LGddA!c=Gs5jVXP=HDnoC-!3|e=BbPsEMfi(^gE&Fi|SAVoB})lK+Kp2&pSf?-?7@K(IjqyIh8}m?=Y_UHwy9Ee)YmrsO!W0SB*Ipa<GsF_%Q8Kbh2HSY$hKebp=H1>=w^hoq|0RwD1j8Yb~ut{h`RY?H4aPpwDhEt+GE#h7&q<r2|0AXB0z9zi^PzJYNFMu$5bN7MHDwxt^D2;v&XVAZW)Q{rmpG6<%w?$1gCOT!yz#W1E!9DaROSEO{A^qe+qU98h5Fj#V@*Ud=fh#i2M&WklyXo>!wZ-<ZyOd>i-suu2nfc~6cV8I<?31os<~OnG^_<ecwz+vrhFNB42i4!M_hrzqfgmT*k0itN5PJG(;sZ!^q^3CoSW^@wM^x<uU`yifXcHqcGOf*)O9Y~W?3Fbw&|D4Bdp`*f_a4?_G?IGqezo+w7$C8&#)Ch;S;O6?otwmO8TR=!QrBJSOFCoznA_KAQ6+ieew8=K73>NI&AM^hc2d``auht25l02L?q-o#68kE?H>Yk3k;Hbur;fe2D$LD_&m(8lSAff*$x(5;3^ev0P>Yw*L5ah>jy;MjdM>(IHSzQ3c}&wucHG+)2;KQ|<kKCb#9?Z@FSaXdAiEvsIhSS5)-5H80saqjFFK98yU#3TF`4X!~O4{+-dd(8jv%ujmhWc4{K-SvND+GE3Abg}+{s==0Tx;Mdnxs=H9Vl{56xvb#gQdW3J$SW+$YD<L0a-0^5jIx=hh>S|xe@Jkw0xg<Osif}ganBdU@1@YPX+)K*t>Jp$3Xvu-p{%Xa?dy|sWo|x~g(QKn?!v7k*+3JV7vk-~-PC8psNa{Gu_MQPx-3N_$lz(K=h{f;8$ap7)(y;>?Ne0^Fd;>r#&{j-WCFrF&;?MTs#nBa#0{7Z=D5Gd83S#cmH!w|zUeq{EbZW|0eCg*SDw;Kj0FwRLmjmfxs=eSvY-flZTrZ~Ti;8;@ToAJ^C`IWLUYg}a}3>Cc$};*D8j;^Q4O~hmpyo^bGLr?b}1GOA<8)0cJ1K04Cr9qxSoYwBYroz$10D;%rZ`wgN#KJSRdOp0DPSO`lYAz5>D3;Jp(h2v*B_!|A3_glL@S{;DeVvO;~=R!X*?O4+s@_u?(SenKg)F1x0Y~#m5Hb$B0i1SfPg=NKS;}GVI<_@iMhhKoNyr6up4`nEIk5JrH}yTcgIP9#C-u)x76L)C<;+s9m&{U)@P)a!=z*Ha?B3dCA_mwwD%-#D+N)?eY{#v`}swssuz7BY}|u&Nvw`nk<rBe3?s4!}Lxt3V<R<w+(`w-+^_qdI<o;`GPVDZDBr+2Bj&VVt{A(s=-&jMaZg8x!4ZwutAr9>0wt#ch%Fa%8t4PUw|Tp*wF^mZTe^vj8<hF=c@nsl;Zbr=N#RBeauGE%SG`1{A1hoEH7@cT>ayIZ3v}X{%C@#LO-C8_a+st(sXo{cub@gA2Jnj)Vq^R9Oal-6m1FW6s91}V*y+~q06zSafS&KlHP0-Wuh9|(<)N(H|9l}#eH>0Cf~#zii`N~vm}ev?cI5}BC(2eI2^gW!4G9fWtT>xId@p_eeK6n%-VYw;a2bJd~%4@g{&@?_tx856CSCpxRhCHYu0lOUTw;`eAe?R-kaOEx=F#>MV7*hRX4}4V-RcK$1fw?9E4gSXIbHiuG9ONk!ZNj*{6G0`EK0?+~e<@%SL{)Sn`zxV?l!pxy*}4qvgqx_ErTWdUW9;wR30`b=D^rhHsI2zFk{q$x7uy_u0-)*lo9m0V~e@k})j4T_JCEB;RonkHLte2;Png<y{@Xd)&3<^JlH0`ra5!Z<e8p@EGrdbJsOL(^cSbtp$8n84j2Eyr|Xl67H$avTl|>)NV)CBB1LUNK`iWT5mxZcumsVp^T$msv~>Hz^QFB^YfP-vHhxHH6{C62X`s1f4YwwD9HqOv=I`nK3?4;FfHvKwbylMnguNe>a@7pKzQ+M2;vgtJjkx#)gK-Et8`j7^hG*->bUEw%?I7C<BqtR?38<Z;cy#pzS3A1NTaVX&>FL*qg-Th(Yp(_8i<&xjMuOlt!&lRp+em(9)PxZXOoD6R&&>ZOyx08Z+%dKmw9v*GG92|q|%n^oMw1#roNd==}X}a{CXE;35C4bY1aQOi>9@Tnx_u&`OlnUXZv}xIv=9PWIP+on2r|<VmkaDxf0kC@43NB9=z}_izQRNsw!UM{gP;|s|@{uE>0A|L6MEVzg%v48O^#eBszx9J6=Xbzsrj6@K}ME8lpp>ss_j*G-jD+;yljsB*&<qGPCM{qWdGxC5AQ_D1PdwQRq~E%a;TdchHv!UgQa?H{~#VQ&!=Wpm~viB{)b&BTTx9P-S-8Dv|h%L6_06MwQMT_=A`#HQj+RPA>RH#;r;PDKlH(A`o)Iw$qe;*|1D7%e2}`yGud&Vzf3pLjJ!|b+!I(`lb`t0fgNgaWaEj1>-Ue9L2XeykVXd$!EBC7xT!YEmm?{$tE1ERLc*ck%Y;Oe{#Z{nn7qvNP@t}qbQe)UPF9#glr0CT{n7+-9vG{L2)K@1y^sdp0|=3lbl87-j+`Ssp!t29p-oGgAB2h+lmowET$RmESjP*Y`Kw*(+JH;V?E6uA{5iuNJp?h>PUTuQl~k)(pq{hI`!4EBLAya|DffG`Q%2wyzw{*s-gVPUcP(z{MBZCVJm*$Q%WCL@xTAFFhX>ZV&aJzm0SOJQBFtvnAj~>6>s@2xUHTuFEvx+Y@z=I>z-k=Hc)|;t4P)H9UraZpBih{>0xKt`PpSF`0DNJE-q*s3N0G#8lzl}W0$Ub_npg^=ZKd(2s{Z2mJBULwa{0*e!AAXu~>nw6t5GP7B%Y37N*7ns1$cT$pLeK-Vs}(g;QmANsUE}?T(avwciSWJp9H2$n`xz*EoDV9OB|La<4<jj||S?x3cjz^Bc@_%K@is$pxG)r3DuZll)gqnC}Y@@@FerM&(7crMzy`ET%_vSj123B@|u=@tg6ezffbz%<FQCE*LX4tjZmsi$`Jw@w|Ypys>8Xqf)lPIQSz!VaWFu@x!t}&m}BTMSTgD^_m$UfOp5)lg_|1$mL6KkZ92Qwbc>JBWS9$08_udX09S-eDyxGI%a%E8GYd^d#=j*;v#$%`0_wi%NwKMt9F$4z&=vdoQMHm1va}ek?Z&Wp{iwZKV49`?25FPYfGM~k~-$ijafB+v2tLIvt3|7GnsGR0{s?P26+!-Rm(~qJ~ncBt1RAul=Z*Fb9Iwfpa4se?}>$$l|!=4fSt7aK%r6-K&|_Pf-lAyBp*04p=3N3VfeLX6(-^HFUvE5^+s`&m3$*F?Qk;>8{qP~6r_&#$VoV}Fx?yD;HfLqWbD;>%cPcsf2uF*V)j6#tLEZu9KUDQR+HaLEDL;<Tw^xv#GWVbAbxa!BM&kj&F6f+SqfHS+%Z%9PVw5K2e8sc;`cNihgD0*73e0GwL`iTNA`hA3<;A+UsL)<M3Z@WlW7@YiBc;E^Cq5$o}FR>OR_=cF2>#299qcY5`?=Y?CYOz!P4C+qu3a8zD!dkEqWs;YHXuVY3tVcCCgWOnfC;XO&5=aCcJFina`!%^DSr?$qut80|OT9569V7p0`AW72N~)gq&&KIaU?H>{B2o@5pwwdX5{ke9z8xm0Seh;l;?}JG}B3{_W-W@bBaN_x77N`{-MwV`{;7?UtME-TWo>G^68<lrSQ^&h=B?G%sPa%|t&qK1pvJ7k6pj8j3Q}9M;SO%qZ9GgT<!r?lta;E?b^Wr*vN{jLv7i8X6Q!y7n3W0u|^l(4qCk*VD3<5OmXz*~oh`qIxl?`Lt4;x<AWCVUeETZJ@{)LA$#0;9?2IaCOvZ%(99;F40AZNp3)_6vBv*n#M!E5gP{S0o6NNkrnBj1>d$po&1J!*|Ww5Q*C>1Ql%W+yuQ|rt#3No2?v|CNN{x$)|%Gpc-^Cv$Szd#o#N3uydVy%(fxRJD{+k+@=Dx6>yS|vpEd9=26q!J-Vl8-_6PmMgCxrf0c-kFY3xi-f_}Yu)Ao%SrIxqGWaY%5|Lt;Ou9o%lza9F`Q+j%)&p#P^&2@}D=Y~Te<sedXkO{Y`KrhIWe$3bZ1onkNHiFizStS}W{Ux0YMQkY4i&jm&QhCHPQPilczWNIiXjKI7f=$UU;-O%z>ARX78MIEINt76f{}=Zx1z*#3M1SyPk$Xv(hweI`MA*xc0`DhOo*GhB1`*!W+M*ql0Ppa!n*<FFFKJuYggLZ$T{duxGrT_j@sQQ$elXk^WYDbQ)~W+kvs1;q@jpB3sZ9Dhy$?|tC+T3Zd?@cg_1BaE-?*$^x5}%$v*9gWW@iIZInm*5)v&dTO57KpPV^(e!Q#26eF?PM5i7N=JM7p%&4vfQ@*N3|Mp2gU+2p8}gQt7A%=#{ZQi1pEF4Vt0Q1}*HdF^(Tck?OUF_^Qh?tz`~*|(ae(QUlfB7ztKa-p#R8nfcHI+E9>(bwGSi=s9lKPd{kwm$Hi@viR9gkBb-+rkKS@A~e8Dl2$G<2_;Tpu8&wM%`-Ti-lg`hOQxm(aw|1^m<h4+(4!C7?-u4dE>WLOy13bv0yU8!GcoS8|C+jHSk)?Eg~L4-l*n<18oROgX}e1Ov$ZtLVrj1&2EO<FQm)V$bq~ecEZg{>vde@S-bEiV=6A*pMP9e@<SbX!8<Yen24HI5$3|-t5Df*P@eQ~QZBgyR_(2nTQ0sC+j#Q@GvSbKG?|9eV(MK#I)ZK(cfv;cNi@Xl*HD)))&&S7R3L_gNWhSN5v-nM-8e8+wNJIM!50PNW$@5Q{`ahK@SI2?+U5(kcb-?J<5s->(iMfvR%pCT9{#(Chj@rbI=n9FZLK!JmGS1b)7kho&dvsS$!Q*J`J!%|8k`y_pj+?NKW@;`VUqErS|{h@a5SIZNp9lvM2!67I*yr8hm1Hnsqetz(?)c$Zd9J4v=1%;fJbo~_fl9eE3X`I`f${r)xjkZ4N!qejjKqP(~WBKaQms?t97A;L+BM7wVWtnypJa!^om#x^O;w&)l~q-kIx1JH`ex<s9_(JFHnWU#rVvaK_iMU64At^?<1<-Nugy(mPK<~pyRkN?=jBeQG!b-4%Yer4Cji$^Gv9tL~?XMU@;$=IM9XWqY>VFw}H09`x8OQ-txKutmv$A_9)d<DtEP1wd>=M9#5t-jAENG!`EnX_K*~3vjS?Jr<r^DqG7uS=@{zA<d~HxFJJ|K<%4)MdP3a^2dehw%`08vCM6CfGhZX#gUN}0`uLF|RFMtF(I?4@L&?*wq);c;<gDfuE1l96NwJbLeI=UszHGkk-T}mXChepIMHZfwOr|l;pisOUB~uQcw6Ia9+#FzP9)SG%K0#(t_$Az%TiV~*NM{A!%MI3Vk1t4B=VA^tS*agopW<u-R&Z4(6qn{9&0Jj|KAXs$nPyWW#Lk&)vCN2+6X-pXZ0GO;J+5U8u2oJOZ49ENv%C_sMzhK0bn7XTPB(1aQoWDjEGVtOmFi*!uB)_M9G8)aYc7~)24yh+7=u4ek*xRtRq19`qm1i*%vq+qOX5+#E{x5tkJ>XuN=d?4M(Og_sj5jAMyFl^tEyCt4tOLWUFM5;RaxGWRrftMmQxrpldSBv>>LWnD9XyM4%~JwFPd<3al+&9ZPfdObM|01!a=HxFcrF#R|JE~L(FrPp5LXF5v8Iha;dBUdL~ZN2N8{LDZ-RiRDlVh<Lq8Vfm614kVbd27-P*~O^)z%fb{;m64p8gu0<)@2U*&m_2M$3oEqCeIaP+ZI;*nk<vyba8l|g#suG9EU;xq@O1!QqaXF1i-$Nj776zxtIc-gCKM&pDdrZYS;d2zRlD>lEE1#kV{Sw6TAmjTbdht4d!yQ!~atGAM$td}Ri8>24*oq-Kyqqvo@_`H>W1OfF!Tg^NFw~M{QVL>wJT*-UF)6d<j{B!Lo`PXj<O1`;!+_R_e9KeVXjbSNji&}xsmu=pQ2it+tA_88+KZ-&F4}Cq=<>Cb#saCC)6MQ-71JELv#6NCrkgtr$}z8Q`E3jH)|zm?-`_8-enS*Jp`IQh$JJ2hf9eRC@_m<fOlvlcT-QL;u*<qdzQ&<YDP4LX^it5Ms4+==<vkl*$)F^}0%w}#tGgYgy-$~57NgNo%?7ccrV*#)ahEl1vZksUGsAJcG>@NC0ohnD&ozCki}A2mVgn3q^-Cfz@Y<Nds#)z-d%U=}SZ^7s3w?@Sx9IY@)@rq`sHQkNJXxpPqkO47db60leRwdAjz8dXPjMk0$DX>+p}e)xoXOQY9ar09@ouHMG>XaJ-i_x#krdBtNzv`Q!nEmzq-GhvcYCFoxirzs(ka>iiDfWWiUh&Wz)5I>S->aJeUyyIH=_(+cC2jtdi4uCS9Zl`py(HWOkWenf1@#nnf&BI_d}mXxo2-~;WVrys*-b%GS>L^g{=z<+2hP*zpAjYG$#~2WVBr>{KC4#TIEt~IFOe&{0DuQUN~U!C4;20b+mj$IBGuk-TF);^SCSMipx3u^$n4yZW+2vsm>nTbVFt^_epKMsQSU%zFW6bf#!5&!4>ii`l80NzB{YvSxfrnAJE#St6i_#(~tp~O`2tz$XiXF%S<~W#%mXxr)6~?8$2Zz{c#fpW0Q{Sxi_ajZo#W9cm<WM7riu_#2IcM1p(Jb&Y<G@+ZKJ>w7zXRR0C~YZXlU%Yy9m@B6P6RUGX5mW$RTQ(9pu;Ne+_aG=5|;60q39!Q*duhkC^VnaceH+l77tPh1CuEJ#8N;UD*N7FufS2_IlIqQun=_d(%^M=~iBIb|=j3Rk;U-{NsUAW$8KXgAc0M<au@=pJX6mZz9OGb|UVCaYaXrK-Bk;Gq#|UrLncfgYdwqDbvLAcX8fu5(s3%Xa(rVc;undu3HSc1fSlpaRZy6Zo71HR~83xdbD6X9sz1>xY;4=Y{Lisi}?hkeN`z7arN|8oroR0U-o+ma<R4DH71%Rn>`FnZv@3Yq4ezO_0{vT<SMfBrGyq#N(9HOPs5?>w4g|L53^bSDmQqb{8Ks^b=5?ervW?*iZcHhOMo7){u@P%;_X6kC1+~s+!>sQ+mpZ)ZDq^C9YdYY6h<V^lx;Wu~YhX_tZnYZ)b!@OS^LKYnih~#%e5Q%$g!r2fCcRW>M@|E{|e%lRnD!a&rQHcJY>?E?dWoR2|*_>LSYYsKK@DwAresENM(tn`Yu^kj^McJ$nzsO<rT@fopkuFJ8iOy)cdh&3ehpl}32QEm5X390nQv#ueQl?$$AAupB~_Ze|@Aauig7(TLv^_M4my8juGvYa3V)#O4LaZtwD^XLG7y1RBrr1=L&J`X96|6vLMX_vl23*--mSIix|}ep7RZZ_rW|Yv=%Aq?I_DmmPWH$-ad>ZJnliFXRDlN|Qa~J$_NrxspeRFOcMLZJJ_U1FU0<|Aubv(y83wpl+RErcUjulZ%R1J*v9TW|e%l`0Vd$Of)D@Rw|r!h2{en-}V>Z5zv<B_+^pGHNKJ!c(LiM`jl5=9+wXc2B<GHmTglk$!F0Bceu+*jz~<im{<to&ld}d&H}lyU^P>B4y_<r9bKeM9C!t$%aYuN{watnX91P*bS)lv{2Q!0HL=4*2CT{E=;8(8dUt#CW{E(M{a+$$#7|;-vu1@-h<{I*hoMIXEu>bB_)|0#$A`HHmI}i*OcaHR$A{_*Vpx6-LNjDdZDk6x@ww3bM&+OEpHt~yI7>|4rQp3T=BN91bMpxVvqz}@;K;Y3;6FG(-P7C(V^Jeh>`bdHPEJ!Rljn#Ij(HB8`NmC!GdpvK9`luQcyy$ClE)}P5=*o`PKu<!6J%BcDjrP9u6dld8l4N0P$#J;34n(Po~07cM#HRUOuzi+#Fs&j-9A{##z#7Vb~`oo)Llous%0L9+S)RT4eFZYsVV#sj4ip!596Xtuw*Q1f1W`Rv~qi&{@fAQH*L`U{>!uSqx!iN7UrI@VJ%b%?9@AdjKTeZ2+Nk9AH{albC2d!^sk+yfAXzRxXkh*BccMZi1LE&2X3Awqoz;&m~WR%C~vRbHHVxPJ;bP6>t?E5306FUYP99u4YHCU{WC@$w0h6+{68}E-Z%E0T%HquR7kQ){22An;z9{lT1B_<028bb*>J~qY61*B5Sxc=xH^9|X_K~s3?ryoVnI5e#mTf+snFEU@g|Vbv+yp9<I(~$lQmD#(FewiFQF-&QxTDv#1G}jU{GK@q1eov)6$>{WST~e3?kRW^UMXq)##Qs-HaWICM-vDuNnbi&Wnn{=NE>n_#n+kPulwOBDT^-7()8TY<>BO=-CLfH^FMEt-2-F5y}7G3NwrUzqA#$(<pXy=~kHHvOm84<&#lr>x&PKJEDh$xPL{r(fSH|Ulza`J;kMkZ;uP{u)0x$(DG&T4{ey)c^?<)MrD?`uaHNuEL_c>xuQ$gzJ-)~gLi$o9$a5q%%?dVp5&!nwifPuZNtFP1V!sGiymCXL}_=W+xUrg!jm;%W*D0Vn{7I(V=1_ov`0Zi3%rMJiYYuW$WzI#B%gPI#8@x5Hd=8&e7iJ3e{egsB&)NfdM!J5<F^Vb!f!#b<$J97@pxzXFyda>uYVOXYY4w7XXkcJ&SQf#-kH<@2bvFE%>eS-HDgfqF)uQK6)d?0W-v>~ZpZ7WHX+xVoA`O6AfVpXnu_>oWG>=o(7L+Q(qcTzNsQdbTAqCVaD3XmJU=;>7r<VAkwaUvDM-zyn3)g5i!rr;NC&(LS7&MB#}nCGyPLbZkD;v9$j@M5@pZFY*^AmgDBYEp5`T0tw{1`m6kpx@vS@-*ga1^|ROu8o8vWwZ1s$3HP`Xfq|75yI2k#=f$gJVt!eLjtMd0cAY&e7|Lz!Lc_a5-DOn2hVx?2*%TM;0S#rLAs<(tkZW9*W3tS0oqz?0Gmwiu$BMvI_3I>=+8lU=l&3xzP5FBEXDTATLzMisn@$f*Ja+$v@U<5D*(fz;+R#yj9v&SzY6ahzvOL5W(@&6Aye(E6<D^X1K(&${bhJm7TMZ;;zKrR)T}k0nRgOl|BoVH9_B?qcf+Uer{kC&qNeU#eKxO_~si-5G36MsR{@g(j)d-ZL49_s*Jd{kPM&7gGyOUg}JZn!!S<mn2;%<Ze<Ka97e@5Gc@OC2&gwF3t)(nq_ts1(ut!)1mcO+L@r%YqUCvo0_tiFLw7oCnlo4$@Wb(FcKmEj69<tsq&>6g1ua9nq~yWb$!>NYo^^*=tNWJesQxSrNu)R=J*Q*`8*G!Y<SN|U?{DaC0)?)agTmXR|)pg$sifdGMGQQA{F<(3E(?JW0MmQJfdq2f1{s4GJ!D!^9vbf&-`IlqDdhsgDA$2pu7NJ_!9T<WZK#qZ5Bt#B%t(sTB|TYDhaH5dA)VhF5Mui=a`h61zhO>%x07BaF#5>b!`pcbV--!c1iklyNFJ=D<u%)w)TGkBN%sX'


def run_embedded_pcd_editor(initial_path=None):
    editor_source = zlib.decompress(
        base64.b85decode(
            LINKNEX_PCD_EDITOR_B85.encode(
                "ascii"
            )
        )
    ).decode(
        "utf-8"
    )

    namespace = {
        "__name__": "linknex_embedded_pcd_editor",
        "__package__": None,
        "__file__": str(
            SCRIPT_DIR
            / "_embedded_linknex_pcd_editor.py"
        ),
    }

    exec(
        compile(
            editor_source,
            namespace["__file__"],
            "exec",
        ),
        namespace,
    )

    run_gui = namespace.get(
        "run_gui"
    )

    if run_gui is None:
        raise RuntimeError(
            "Embedded LinkODT 3D Editor did not expose run_gui()."
        )

    run_gui(
        initial_path=(
            Path(initial_path)
            if initial_path
            else None
        )
    )


# =============================================================================
# Embedded hierarchical organelle-network backend (Module 4)
# =============================================================================
# Source: user's hierarchical physical + whole-physical + social + cross-scale
# network analysis implementation. It is compressed here so LinkNex remains a
# single development file / single EXE entry.

LINKNEX_HIERARCHICAL_NETWORK_B85 = 'c%1CL{g+$Ckudzb|BAcAIe8_}$TLpJW@nL~B!ukdEK7J7-p#W|XO7O5bZ6$q8cDfVvS&=ZXGkEniH#wFKy34s0Ab03&4wpyYy<yGGSZBH;$L{W`m?G(uCB&olIJ~-4tS({yQ{0KtE;Q4tE<+s!Dcj!M;pDg9i^MYK{kpWXFrd~?;A-x{Lzs$D3y+qY&013a#gM~*lef$c$5zMxhg-<--c=DL@zm_e)h+k!|f=}qyF$nYm~*q<n$ozMh*76Zhod~(MFudqfu7r4{OnaezQ=Gl3t#u(x-JRalgxevj2u%-BdC<HONkE@z%Bk@_8p6ZrAf+oG~B@|2%7L#-l9Vszu#2Ue9Y$vYoi)>z#a^wYj0{RgOgL(;4*pNoSOFTZ|6FK|kq_@*02BO@DYIXIP3~n`uAY9B;N5tbRZ3ud~i4N$p70(@u|3v$w9ANz(Ov!zwjO<DN$UpT6@XfMV_CqqyIpn2+L-9;httpXkP;_{fnXk39Yaqxo{OxO|NL>LzPZZxDA|qpeXZ&a!yB0=!VuNOuOk@n)Y9rWd!9UUh}{XklSt|Jtp|rC;6Id1Z3xzbEH@cIWa-yEk5%zVhb&TklQY{&?^5Gt-x^Pj+rIuz%k92`ew?!cY(c4sEh`vaxQ`OE-bL8Vj`r_H(p7Od71z8Y?#X`q!$wGUE)WU+;I*&8We;AaFr&`RgpnHyF!QmTS@CvZv<XG@@hCAQMk`(dP1M^pM?Q13ZgU#xegI_r}R1SvJTj`VD;67M|jiITK~ckH#r$J&H$BFNyO}^ppkjOr)u~U;$~5S#co78DRg2;}tz+a6D5tHAJVfWHiqDksMVR$2?w3THW+y+D$5KvL#tO8e~A+{$LYk=^&HC$LP@?j9?hje2t+vN-DSzP$zEZ^w~p=h;a_0ifEp}=*NAG5Fazsesq)|AW`iMlC`z8ld>t(8mzTW##zdxNaZAuIhS~fAA(_+OyJxpo`rZs*AQQtxQi0>sv<!z(aixnEfw``Gl}~xUYNm_Jsj5IsRE%n4Jc=d&ri4^o4H-n&F)~Wu~cPLL@*^0`-5yV?xm-h0Ah5^2E%PiJ~r#7&pmhNwb%DQenY&NeEN&s8#|K=znY$VVfyY*_MSgGdExo}58vAR@RjCLjZK3mHkgPS^dkumP2GDk7SUHC`87HoE%}G$Au`h?t(NLbo*`V;D}*BLFdMYvcG^ovDME!&6UaQ?RF!#I*=JU2QNE?M?-z1w-Fo~03~#mSj--NjRMi7)N!@By4dv;)Gsuz_rOrVo!C*WZGI7Fdtc`oU3L;ZYQYTbbU?A$+qorksppJOS1}Hk_jE_KpKZTX48)w~CH#sRLIi5m>*P8;~s^)p!o(F<GsYH7U1Z#jXh!|AH5Cz4kET@P>ixHjU>P6K^Vm7!gc9<0;A;c()`Yl+kvy@4ER<^|`xfy3>NQYUnmTs9DJh}D>o3gt%-oNvkXC}{{n_f6O`S3DZUemr<@+Tj?KHd4<{%60OoV~gC-XA7Uzr$AJ$&H^*Uwn1{-KX#M1THU3pR!j?(a|%l&4g)Q{e>09n4xm%kFwc?hRaJO3jLk0&>8;iN6SmK=tsvMWdA$}-~JZ9{oRrv+mFU^f0XtT!!?qC`ge1w4gi+w0AZ>AApici{QKW6t%`C)R2Z`ONFt^Q!=zgCl@lb9h5q(I#;^=?e^EggS#&u7PY^~H`a4zVjO1R>A=y$hiqjrbn)4I6W}mez?f^4`u90^Vre^im8{duD)H8g{|4uKQ2Pk)T&QGq~WPCim_R{WWKmX$Pg*#7wK6&<{qVJto&oR~U&eLZn=dMrQ{@tCkmq1!hU-=`exO?LRrdhH_yEiTZzkhg{@%RxAegE^9CRbk1lO9*fC%?EbdG?RHpZ<D!_5AeZb85b@sl#T+<oZj{1*nSgJJ1|SyRB?+O4d5Hkz`55GgGR~B{5kPexm&8Bpc*ZK#PXjY@ek#Wiw9u-6Shr!^}sCTVPHqRBEOTf+Evf$9-5xHsdYOr_(-{*_|ZSYl?2*e$53{DY*xk#yQT`bG8jpa@5pg(*6W%x3R#q#q|WrEEttt^AZx^5+?uKTv=IMrbJaEVOnU)qO?z|v|5wYhlSzEY@AT)W32rJ93XrO9GO9IF}cK)Z><e7KDRX2!H)wg0#KQjhbM$zm<ASF>20-Cps2<WAgxxbQXh!>0(O;XV4`d540H=A{Em_;G*q^m9x(>uyHPet|AE!|;`Wuvv+pgji9h|Xx7akET)M{Q`|c;_*?U!-{q2CJ#aG@hQ(tX_6jqpOu!W*3W{G_OMj1^PIT}oEs_wAw>wJTYn|&Z!4)u(Tq(-1}j49FgD>McC`y&PJXoO{qDW}AlQGym}Nrp$AZ&=Vnx$X&m7!8IkB`aENbDJg3LehX`)Mk+B&MhwXKVa*HFfL55?%etPtE$rE!`Injvj6d0dlx=r46=Lc(>uR;?JvU-qroWd5j7W4u83zu#o4SfhlG~HXG>?g3=QLKl%fqdWvxZ4y=v18K7Au_UiEKl84Xlv;1Awc#T5B4iP>(p&%k6^(!<7pR$P+<&}tjIRKM|cq$J;Xm+`{n`lacc|2@6>pNvgl^}lh0vB~teJJUDbVf^rfY#~A|INZ^Sef$9!6mdwa?XA@=R2sZ{tXAxo$gQ?!8#%<1qtb4*ii}tNLOX*{;Gm1j9;llYs~cO6H<QuEfGx=J$s~){lZ7g-+V)g!&#EeyJ}^$<!ByDQG;ln)`dbwnx5_ie*T|yzf55PkAuT(ite%nQWH+rxR4qJ6g|bSveI={&7_4AeV1akJ$!?;_Hbl#-YE0YXwAXD7vvHpde}00=h7ns5;_eS)wxoBqMc1~rPm6s>KS|d&+Dt*p;%++58_V@?)TEJz0bnZ$TQWQ2Q7h>(O|k*{S?!3m|JeQX^Szhe+r9Dp<ov}uul)u}!fyHef9+klF?rzv><unHH~H+DJ3qNKef}LLLj<>-Ocqt+8-qc2a{m44>#tecj@_FVU`6`*FQ?~U6xIK{^YremclNHn!4{^;YcEXSJiGtjukF2iYxl;@&LAJL#zZ-`oUw9~OV90o{@UKlzh=+(e)0+1_S>5iMhMD}K$!q54hKw3=*0nKIV&t)!QlgYCeIGc^G?ikZ8ou!)-X;pSoNSJ+w&&bws>S%Eo)eVVYZXP;L+k(Er4N!VY;9;zE(pD2sFmD=TW5^F$FkkF&P&L9S8M8F0D{NJY}<2d}%{@3w^`BvM_4`B&YmUwO+T$>$Z9ALAEJjtk&ahcb*z;TqBm$IFSrTn&Qn>^NbDJ`e*|^{gQjG&Xihl8*4J+gsKj<XW$)845lQQpY2toAD&2(AxBN^n)DkczI=1n^nlZIu(`>k0_GKJwVO3ePT{W<ju9zdCBMuFoZy7FFk5k2GBiA)Y3LM!Pg%mASgkts-6y))_CcF%AEa5b8imO45MiuEIEeg)w-Os7(veaXtL_-U2eY8rl-r+nmc%Crcj|f+teLa*csNY@-AcutnANIVL(XVb2X#lv0=X$L;BC82vN;^V0fjlYu3!;a!*ahGS{vXaS8xumCjPKhe7Ic^2L${<rNS2LV;BsB4M1GEg%&0A+^uL+!c$Ye3GQUIp$+!^Yy$V$|COd`w7F7?mcXh+o-D6Q2A1cqGeImnAeI0GhXEcBY)RB=3LOu#y(SF6nj7>JVYlN3Gfxkx!I~GVDU8ue^pzhCHwsuluDkHBpzQo;yHwI+gnzBxWAr7*+QhFp{lWt$^-Ne*TI<PxZBVl93jf7*CrgPj$|6$|xo-HgXYc&<y?=h<$vZE9KDqg>I4F&?j%<@DcpbKIfu(uG*01p(Pe=I3R5e7|+%#WSbyGF;UV#*$hF#lNVc}cRaAP}9J8_TW14|#A{d4mfTf@MbEa2t!g-`?D^VgdAXBqxc7&c+*p&v(l8Yk^eP}y{KOuueP>?vNUd0#Yr{prc|7pE^@-@S2h|MS=PZhZhqfRX6)*Csb#{^IuK>4o#tt2_IjgYLfn@%xj{e>#2s7n7TploBrb4mR7gzs}GFZB?7$FwYtpt8CTE@DHC;8)+{wfGkaC%yzeY)P>Dj8MpOhOtwZXFdA=x!6)szi>(G+b6%ZDwi~^8v)zp&4r0Y#{Jl$}9!p*almh)kV(Y3`mLQ-DA@7-50ZoT!A>EBS`!+b4cnfk!Y~Ls*m_`fDG+x44g$%-MCQIq%*OKIx+@cfv*-h8hDyaQh6mO+@W2q{?EF&&3-`HxOjU>9LGG-4(6i)%g2lOl(Se-7U!O_&j!K)b_OT19h(G&ftdAcJU;{1yUd|ZaI7Zcor8tIR{nE2s^U}`0Rtp6^=!urd&{*t%8&FJ)Jxy6mrbTc1`sqOj;Rye(@Ma(Ptl>w=Pfv#FTf&sV2Y|oP^n=iEV0Q_5jQWsidy&bibS_(|1{1_?iDKOe=^;8`H(syhErwndm$=7w!IGX^RMsiasY))~u4(4qOJv7JigpFbMOTe5W>Fb6>rKle^wpG7~(36uk>Jt{asMWe$f4P$4VHPYe6tG2ugT%|IHe}~5pdc<hO4>1fVk<L2;5e~H^5eo~>8wLEJp;N%0MrmhHG^$U(SzC+rS8KG3$0PyR?ZvR+Q=;qHTo<cbW&C)?aCLO0c#!{7?W1cfC=@Dzj4f9$o1~prQI9v^LEqj@g?D*vwLG_|E&vO+`cgR<d4%=KbzdXB#eHlMd_r#YS?-tLq<sX3>b={4<hR8t5vq}!BSVD8!nEq;uN>u%2!vT`@KDEKE`$|tbjVr>98Ww9Hs415;d@fz=>tjg~rGIlqqr*g?3Zpf1C-@Dz+?JZWE@X4F$Q1mq$q@e2^#G@(f**pTZM91PzhEsDd}}GS$M0-F<n-7pO}8l*IK+)J}SX{(8<hG};{ZM(L0#d!oVUd*8`dEU#Yy>nVa!eKhDX!BDB5p=GN!5%@_1X9ZlHQHv}Qa_Y>QGfIZ&J(5>_Xc$%!#HB*gw$FPM09pc0CAD(_6O2A4wCpI}RWn;}qp4v$6)a{K8`7_?e0^2xYkJDCir6^9SzT$V1c9fVPAyOwlXe$Y__!zrcJ+n@ceb-SdR%=KR8NosA)C<#1z$rNwM>>~qyiQxD!Pt}E*BSVqoT*WMU4<=MS&mPB3QPwLV{o2VupcPA;irdVhQ%vo@|=k13ee-w>{<k-52jqd&>K1z~`7QEoWQ+7kInvl=b5QZ%;dA9URVBGe$12!*2aMOiBE17H=kXxHi&~Y+sRoJznNaTS(eWK6YC2>Pvc(4MD}W$)Iz2<!}CIJnY4czx!~y^`HL<hh^6<?VbN*^4Zgqo!ejBz9Kf}cXlpMU-{$ijhjno`Rfn5iMEH>ZOI!POWG3$mG(I%qf<<MX-8@2E!fR??({evtXayUHbGb**@EXao$yE9WSyz?mJQVH9#iPmp0b|BCzD=9w$?wHWI2<RKgvefG0^lsY+zy&Cp`%!8HIz<$uQWaDha-9Oh=P$^XyGURx~A~jZqCVQ5BQzgrKzs95rezu$|)qb|d09uqqe>^Y)w0$fH-Gid7#|G@@fm*rcAWCw;6@NUfMXXHWBuitZCzn#CYhtM-_RDF|yA0@s8sBpYp6oHoR-b{P`sM$Ftj(Ku#)oZ3h`8;yl0{&}HhwcQeM88;5H9PeFxtt!VdSSypbhq3kDU<h|e`DAPkd)(h~%+_q>;HM7m9m6bts#RwOTsSzevB?|99+bPnEbe!iFv=4)xxRJ`E7X%u{xEs%Z1l*u(@VPv<1w{^0w;XAKt@c1Vc^n^ei;%ZHg<O*NNC~Xv}dLwqoceMC6pE3(K@4J%v8lz)?crnz9W545E+dg5$_smovoZ~lS*b<&ct?upaY(#z-kGWT!>wWiHS5Vy$-$d1#LI<0QmOcewKhwX!osl-Vznb+OTnvPo!x~Getv>Jd;%8I7>j<=(D!^5Wew)PE}tr{+KCynYxUcj$5)}u3yQb)f$_W%e0$jGoqd}IV1}$Q%pa?<&@_Tapuv!LRMR5%)S!Y?}Q}BB`_cx9Y;!zlrOJvLn?MA8>Iu%A+-e&u0Cv1TDZv<v^Ln^dg-tt9+)$QkPC9$mqoP@;`id1?BVK2!nBxNj&+d~Qgb5NiaR4Ig1)x2gvE{U6#{$8xWnN!mX8WxOo!3>%>MO{#IcZ&PU5b?-b?@M&a1Z$N%C-qEuVNnxF3$1<DR75g_=V!82Gw98n71?buO#P+>)1hwZb9$3)@3(*Q^;9N7X{dnKtBF!~wA_6C6$s>YYKa>KsYJ3;qo+r9$VS`Wk{BR@JapBvcgv3LI+@0RMVjS{*@Z9bU6Bz~lVd_Y(11SJk*F1o7Eo@tBgCNgT<=xEZiVqY4G9XYLQjec=S2CF}5qm+&at$}A-3rm0q7K3b|aX+KG=z<rn{3SXUY*cdeltW{Wy=o87C#j}kz<DY91FW6&?Lj%8YbI|@FtBTE=4iO7vp|R?S--?X#tz`@KSX3>Ms?|WLageu0Jma@W@zElf7#!gN*c*-U%V%*kxMC~gdT-DM6NtqvrZ6yr{zVLqy#Y6KaANux0fFXhfPY&+wE5j0gn57<OR%=C78LhMEZn8A$eMHEKrsD`%*O%vTy#M#h1`IJ5$g9(M@NrF$4o7<bSi@v>NDDmb+$V_rU1nK6Y@d?QIw&sgTf+)|A5_P%aY<fx8lGgZ+YfMlR)ql@L2;_fhCb~H%0paGstb$)y@Vw1J<}SvEgl%trRUDe5E23b}qCHKz<J+aF$4$ARFJgY@}X`C+%R7p)p|pM0=5f%J!hMiuR+T>K0E(yo))7LLf~m7S>}`Y`E2{gsSgfG#v==A=9#;1Qq!BQ`{|i%1F8<`G(4$1}yr8G}bd0EWr9lQ5C_pnlZHf7;|$e!giCsv?}BHDPD5<X-}meC27Fxpcu(=RG$Oy;Gi5jc4qA6!LMk<?-C&*z*LwNE;^jpDc+{30GzLF+bqUOQDiZ*KUj0{QwE*LVLiZKm|EP5^)>s0&2s2FTQg@#@j?>$PZSuT0*pNKECNqPkI#A$E<r;=5S<yO2j6G*kaEBt^F*-*?ez2@eFveX1*yLe)D#m`5Aa8Bz^t<14B#zphwv%g2ZVmBa}pMkP>`uZpXr=@<b<*e^^m%T>aG@((dZ3_LBYE{9BdE9JVHaOJ=kjX2B(s&#UBZ`w_IO{fSZnDN2Z+H<c&;lV>M;vu{_x^s)9_P4-}LT+YZFGA7a~qXcb(Khq?$iJ(=Bzd1gAYXdu@RHs*1Rs163sBM!m6K3NP6Z7b>Y26>W?TKPCzWBW$#Bkdt_ErPg*NtQPjmr<-faYOOO+q*a3-}}iYcdq;lj$!}nv%R-I6i(0LV&de|bK)l}16LtG+PU-7_ZX`oL#8$wiH7-=QS_jVw%wN3w=SUzXa~3&cLd_YV!Xgn=T0=1O~2${v=bSe(s$hWF~vG?fX}-QN~-H_qQxe!yo$FqXgD+sRjei&6Mx!A!o8HY?Cc)K-X6A#dWdlj_YTuQw8T+vuX>n__Nwjl3R{@uy3RMM^7`knqaM%ljgyu2HK)-Agk~J{<ROk>RAP0%_0HoDe`o)#_wKy*8_b&zHFWjKt($wl|G9Lll94_Uiy8zipE>gatSpPmA%{5=(ZmAv5SSIq#)VvXawCa)5CxL;M_z-8HpjWSdg1PgL83cx9*O@R?L%4LJz*Y`<Z!HJw3phN40{YQOSXlxnc3D;m2bt^)NI;^>}rlF&tS;Qh-Hay?`N9<O<l*c(C$T(Vr~!;QxT|RxwskQ65l4M`~7_V7M1AI!5a_wqoKho?TdzKaF2~GfSp=h=5sSsv_Wo8>QTVYN!QqRYxupLbdA$)4Zq)&?CosVH(7AD<C`p4==O#+D*ja{Y(6d)XCx5upk_F_1^08JM8;1|Du%l)$5}4$iy54)LpF=^S{9SzJJMu!iTmXSgQme4!u+hPTAs6N6`z4s8)1boYkM%Wu48*|7|)}FI+c%ymnl-#bFmoYc-kM+yTnTUQ{a?AC{|yfmtJMgfv?dIy=l8|EFGfgCKjWmjcgG$O7{O0xcH93x6a{m@z&74J6fzb2ptV0RE_SxKRV`aa0<F?0wO5dHs$De7}9VuD2j6MXgNH!y}}tju7HLQlO^Mfbw|$%TSSvh1F1P;N?<Z7sM!wH)EA2+-*t_F!{OTy*1Qoh%B-6B5-iNSsJPuM4ssm);@DSp+o|dm(7vjtovPl|0#i1GaA5uLRkaZgtSG*!Ho^hd>N6If8;yx;>%hCgO*~dAs%DWgm*dT;mCL6F6_*pyyCARCFl@|CHK$QC5aD;*1@BIKkoCfJMlbwM!CW<Tn@w;mDD<891|~Y$sF`7`p%#?Wr68I<KW_1s<QC0eQ!ZX~SENh&Pay3>L%K)Oioop6fhtN=4E3-^Re7WKpbd$=BfwYV?P55(fp&=|K8xmiFhGW0YamQhWWs2N;c()th2X*Ov<GWgygum1y$WeSxYTN;kf}YK5_<6JJnP1I2*b;-K$FP2VqAgi`D%qnr75ieb$bdU^-C~!xT-zFQB`ZD7Y(8LjP4<6R|kPfO*RW|<)W;os3=mCo%0lxgCpFHz?LW1BI*E1NbPAK9tR4$80f+(rk(MPiHO%60wZUwKn~m{dgGi8DA!lTiA%-4I6=5P;3jCm%;)@XyBB^{#)AU_8azS~C{3#pQFR=H5!GIM2%x}u+y>M2K+dU78$8i92o;@-?UdJhj9W*U@<`w}?&I}}ssGK$``z`^;|?wwvE$K_zM;>HD!*G<<#<CXl%}7`n-%-!Iyhp_R`?u^q=%oXeZT!cxWPeLBKQ$~Ew*VVqf<%J?<aXq#fLGSeH+f@l3dq`P#p-dz4H@@eRuPD!L^g?mnK&}3;C#$CMf42sPnNNiS}j;9KM*@bRq)iRUT>8^PDG+aEwdv%0!VvvRba)$dbfmr>`gyu%ozh!tCAQGt821(qS}^{b{b+q3aMNbb~cr-;C!Yo37S0f=FGM#zHGcD-~I&3fCqA0kyD%T41|>p7*3#;ojMzP<7Qr7Z6cRcrkB86SN!V1Je*mrRwnSa4@X!n$GUjuFOW-U@FG`0g=b3<ldAIqmDho0JzcJ6pz9%n$U2(F}IOnHekyp7k&<|4@54nnt(%Y&=ejDPl9_0_fypn5zl;mr9)^cG&DqS6*L6nd_#;uEvl$GGM2D}Rt+3V3dY23$|pcEZw)+gNX0;%q$hLgX!0!jdMon8gb8FL`oKXkrTk;&Py$bF&9mI>7So4wCq-UP7QweLQCx;aNCMANoc(MI8C6Wz#f^~wSLseQ5nxWdu!n<yP0AmgNo$Bu%{|8-z>NyQZ_5QvmA`37IpkuF7APeq8V=MW&RTXq2)zeAO44zB`wQ1N%E(~*wnIP2qhsI!f6Thjf}oiA*PT%Yt^|YbxEE*O%i_6E`&e=cLRj~b7%ZNP=<eU0onK5o{l(<mFCd`pXRl1oUDtYm0U28E14^dwyAp6(qf=>zWWOT1Rn@Ov9@9b1DVV|t=d~FV2nV+yS|Y)SYI=qS@`?!2=-f-CJogVcX=N4FMoqV-qnDFTA$1NSgO2+u=~T1S?Kycaz!ac-gRdzn&V~>V2qbcC&j#Zm-<ar^mYk3W_Pn5#stGrzq?#y4*Km6&&Btv`LWit8M*dby0ICTDtE)1$qzA;JQSapd>UW-CTIe!MEy5RH?YlA?28^jKXH84(b?LD>z3W<atDPk2-|!t98?YU91lO0#)Y2i0L%QCl_g@e};CR?Ui>D}%A^R5fB|C{Gkb$X=0xBrga%X5BK17bc4~jNonRPe@_Z}692EU0I<(u+ZaOS~>bjWV#@F<U`8?M&W#6uh~qwqt#FfYyXNY36=A@K(qfPxQp8#&>qDNXfI27YNQqd^o+HhHa8BuD_5B{$Kg0GibeDex<v5r&mF99XZU@SuZHV*KS|V!+GfrE0Am#_I`pQRbFY6K)=xaN^A&B(Ykmf8)WL_TvIeoJe-tcyI~jC)vGmard*=_W#$f9uwXu|8wv1GkW!XoVE5ZtTnXuT2@MRH3zrv-$9pHv*`*P%4;bC{4RwWD!r^Sm8GwRsbbq!FFldi#vgP&QA0OjCkz}Cbi$Ye&C{d0zJ6GXz};olhV<{A_!XKof>Xp@A>X#yE*~P*DCs4TVpnB0<XY^aVxX$qre!8@c?(U?@eqQPgEByyc|gHu>Za*%{YfSzp-Yxnmq9FH6Quwy@^U<Nh-cFz$s!ArS0YAsK|wyTSc|r%DB8pGlUF`!Dx_=XJLe4H$jUo}oFGAI7BiPejfXQC$eWv2)K8K-5mE>7xH6L~pNSjYKZx@@fkZHi-;KxlX!r9Eryu+l5=trHAthPCw+*qf+FjXUeKB@5WX(1~s}`;scQ)jC6t<`-<dHFrKC&YRXh7uhof=H_vPxE5rw)-moLcr$s;wj*mLUk%40P-vt%)RiVed?Bl#Y@X&B6iuHn*&MbJ;sSGRm}3zgi+CSr3-B_O`h~hQ+A4K_(U_#I|e)ckcI-tM7>rg8Is(sUiYXi@*b@*~m0I=T6U5oR5vA0wOyi4)Kg<VW5kghSqIdyD0Du223oxa#N#ka$6DnOlP<gQ4@mQQk$ul6`8K7ZhV_{M`|R(UynD40&vc?8?oZI$`KO^0F%tN@?NkrS+$|G;1neaMD`R#=tRZUJoX=roBVPVS1?nLl6h5^CzC`MlC$I_&zLc0^0%*VUU>{1ySLt9!`~KJ!@<#442_PvF@61+J3DX6J7$qkA7vm86?5!gAM>QK{_<2Wkm)E5;OxUSbYmlUghjP499Tl_qD^QCYSdt(K+QM&fe9yJ3^=0XY$Wa6+C{-CKy3%GRc}r5yGh@Qm=(vQyM;;7O;(CBiMp%42v<&yZnUVXs~7S{r@1Dl&TYeS?}XrnLU(?0Yx2>v``132T)RE_;030IFjYfb)r$;B#Gsz+68&&|xKiv|95W9G8<|J!mrzFViJzAAP7noijE7~sjundCJv2aSxu8WIBbUnFC>4aIITivJS+O2yM9i5EUKy0;5}_biC_0T^4Z^_^25W2DUxY#Vci6K*7(LQeH}kLTE<e;JAim<&28kYMYk)GIp0lyjmF<$|tXyoKwEkutk6YsrHV{a-B8Chd<olf`Mn8iOnd_oKm5vqyrllPj768KCrk%VM_q#@EOVwo;&ei&6ygrP#0;q`e+4vTVSc%m2z@y^Coozzc_Jc`){SR+WFPxped3JK?IcQ<Amz+#`5pLdA-}FruB?vTTUb~QsDXs14Nf|-Hr=V=O8j@BX;CynU%QmD1Cm@IlJz+~%6d1%_A6>DCmr2)};p;!HLA+~<Tz6bRa%HyG3?N?VWkgs5|6*Qf^x)bVVX$r@b$W)VqrNJBd55>qCAav3`G_Y<30E1+viUF;!VMwoj)YhrtT;ICUd104M++m;WO2+xyGe-PD5pMyA_7j>?ej~g;@)z@j!Pn@7}_E>O6^lMjv<*xv6OhA9w-|tXm<f~t8eaj^k{I(LJ8ml>!PEw{2#U^%IFP5Vw7AM4XNz2XLmuQ@jZQwN(Eo;FS2!*d}N1jADUUbTCI|*W%l{LX6*277+%%l3H1(%B4aud`WmiYBgi4GttbR2se<Bw;`Xs8mW>mR_6Z-EBbm!J6||t&>EI0-zhmgYMnlRNnW-;H=2T>(QdCa`N*2`wUpnQ(Oyb28!L0gNE!^Ke4OC!t%6c0@26L@4h(kOOh(a^i6jlV=yeO3j?VBkkjO~WXkGZXxsCUoK;^>B{_B_IlpBUMIlPSaCXs}aw9~A7kC*=yFtL-3BL#Kdxf%EyN6bW#o)zQf+ymoTsgH42koOm8}4(1!zA}u&17YIq@B>in=bE`F}GfJ=Sd=(kdq?7wBP^zW|rC=ja;!P_Czb81WafTH>Zr*ItnHoh%=gh_+<iXCUORHAp!3pF}mG3DmixqtAU|53A;6T^#%oB?b+QvlmQj8%3y%e_~Hx(+Rn-hy{ZQ&`g$vPv$SFw`fOR$_$`XT6&a^MM-HHtk%jnO}X<!x1-)v_NIv?Uc>-jw_vERA$oRv!Hd)<D`FtATz6YbgDU6<1%w*rO$yN7%7e^9Q2kqPp6*K-48d#?(85#gf>cNBD*jf~l`8kr;GnQ{<~8V@9}f`OE`<MHAA@H=&QQr38t+nW~xBu@TA4*|XG(B#%X;X9-=UrE$2RBAPa|s`h;gk%mi$b*4*0d`0%Bg~BhXhP_qE><dUrj)Eyzt$OG+2aWJb2{8lFPj7ig?SMYo&%8A~qN1pacJaIIs9CMyj;~qby{L9NY>O822!{n|dQ^Vm<ZzGJS}BmkA^<o;Vj;hB>1R}=U&1PaewG`0S<(BvXljs!Q?Fn2aZ&xk*oDROOPN#(LFi(BC}`~|hJy<FAz}MNAtY4L4@p}V3c;vSei)wfNCw7tN=X>k)qQRL6{{G&(Tg>snp)ZX5xw=gCf6z%dr7@U*L;N$PK81B$NeUW&`@Y^U%o?Q#O8k=SDEhQT?C!aaPRUnIt!k{RwYX|l?3)8O<p4s5y+fs`TpmxPoIBha_;)%g$uhkc2t<vNQ7BcP<4POO{6Xfx{Pl$J2~0z?mn~1<u>y+Gs=f(956A$_%?Geki9crG@`w@8$8e<Y`JzsLGr*WlIo$ZKnm-soy8I3DpAE|7%PhYaqQpY#LW+KWw`+Q1g5m@>E9(SuP$CxJq@Lusgm*45VYsGKKG;uTJF&@Ot2hY(#59JlZM-jo&H;8PZaR@_|ZigE;K5l%Zua)2luKBcL**TDK<yC!%^16P(63F`rUbfz)ryhM7mfPZiW7vMum-~{4OK(ZH1w>s&i^OkStxL-$}cm)u1r@GFqBb(;wBvNzJMfk*>OJ!v|gXCtddgUvmXHhwTjnYR3n$Ai-pI`s=wM0w3>XN?n;mm2k<DhN6>u<aV8@zfEu9Kn0(mN6BNu0d{>MFZ`uf>nUKh<T9%}q?Rmo=u0Y`DI5|M*us;e7Cy7{)N+&-^lJK)E5_I)JfkFNt_kc(Q!7B}1v`SH940*1))Kh(N^;yj<>|v27b2yeMok|uc#(C*T@krmnmNm36EBIY^|;%uAXGfkRan~;e<@2H!a{kiK`ZW%I5@gYcZ@#r1RjHMVPI4gkybE5RpB>CvGQEqbu@@NTTEmox(a}=MNJ5g83<`*jhJU_`J<2*g77lZwfOl8(UEGA^9U*L_VG48b|j`9C+MDv-?yz5ZGLLqK@#e8Y6T}Nt6rkujkVbTyb*W2An;)UaG)A-#C?3+(Rk!2IgUHh=_uLED^(|c489g;MT5~z+JR#V>S)}sS5%yFJ<WKNV64%FXoqo{7LGL=0e<xy!7qYkRjt_s2*UDlSNWivq764$v>8yhmO|C7rRf`vQ|ZkwjhG*es}G2vfRliPOC3ZVZ=g)8E*o7|bK&n7>OUN${mN!M1bZT7T063+6G)R#e|F5R!coIc3?dj8)jNX2MR5#{PtArlL)!>YdqyE3t0}TdJV<=DhNv-@_ZugkJdbH$X<RFqU5z4-5n*&`X!*GiLK!x8=x3d28E}522J0zwV&n?rGu4o_#BVOBvwiwH+Gh~j>8FE!!)?mE-WZ%}EcB8!JZRj<gKc3N3v6EsQkeZAT`pc+%s30JK_7F_Yc3^MC|$k|)2|nAw!5)XpH_%44roV6k8na2i+}HnrcYm=yz}Yg()-ir-`TyfvwP!x1$6K7GXQCN{^Io2YojECv+15py70yAE0Yi2-uwN}p~UWui<8?o_uslO`ScfiAKg;cpW<Goc}u~RDPQIyQW2$@@{PKv5+rymaUKCx^Z8F%mcMv?o$b*fdN%>e2SLb<il;T?YyRAG>iwb;FocDtADx}<+yd*`vwxhP-<f>$x{967pdm-o;z--*YO!w$@t5uGici#$>uFzR+A#0wpb3XCqb_48U?=ZsVIaCKe57}YxZ`c$$rHI%84kqNnP=t~37qK#8=4I<c-Uvd)CJ2xCFgm?a8c1RkUfWK#RK7O)PdqL&Xnn;Ime&XNma5;AL)tz0lJ3)%-yqiYd!?xt;1~}D1iJCiV3R=I&{6;v^G+_VD-p3!6_SSH5?i;Y7_@^rAin&?tzb;el_5(ZwE*%7aTF<t6w@UKxqeh%PH>AUAk!peI5J%i<9>uIN~3`xmx*s>cDr!p3#Tlhix}m#>Lv_>1>9MxP{u_OE_86+h_rI2fy3pXfPb~2J2##^^73pD(|Ra5F!6`Uo&oPH9Ceec%1Y@I$mMWqO7+79?xJenhtJMrx(NR1MjIBXsQfjkflX(i3eRfPR)Ft`3K^j132y4zPggZ^tvFv@K<iCW|e<#gknFGb+@JsF@n$sXd>wsea^|VML4f@fGbzLg#w3f+s|8I^v$T?r-vwiF}Lm`62J^!Y~%RcbnLsnR|UQjyun%V-}%HkNDO%R7|rg0!P4OqyXNW=vC=tfhHDmtN_S{oPqs#||5dK#v2iMIuUbko840^$+f<BKY1p~D@DI1BWqEvuGrP>=&ra%B2^SYPk67HgVp8ClJ{XUNW0^@c!pRSCWI|j>AJM4oG!>2BqGp#InIKo;;&}7RZ$6UP>fS2u*3~2B!qv&vjMc<pNW*Aimv0#(&T;xKtZux%X^1%$lnv_o{Ja<RHf$b`W>RY|cV;@|<$ak=y}wrXa1YDfQ&(%%UMW9o&D?)|{>M5s+<gAuy8A_nrqjj3(nZ+awU7Bdpx8{ub8E8A=iO{e+~Vqbt_NGUeX|2j*&aCxCbME@#s1KDGnekd0h_oYV=1lAB9M{lh&q?(_Xi_5mguluB-=*D+X;l)=#L;Jxx5QE%Pm=8ZkwuPl0WUtW-`Y59Fob4fCIDN?O7qKTp;6aI+isz#%(QY8cX%%CCBl3a{Xu1YnP_4y``L+xHs+I<!6Kwv&>NY{5$)<dw1{m*QT$&Il2DK<lJ?Uv(~s$@mlH};)~lC1nd_sOkerq<lNiSUwt4zOfJ2+_tG1atMBb!dq*`mX#bFH-y#v~4e~Mn@_p8@0WIalb~L$gbMnR;a8M?;PyGB#1{DzlrIe(_NLY~I!tbC%LJ+Y>*!YIRNG?X!aUCnl=qEw*TqRIUp|c2(xlf4Qw)R9j&4jlLoIK4p;<b@@M4<CWFbtCqFYkVGUW9bij1cYKcn(rqUwvcp;p=<<ar4eAA4349H{ST-_GJlrxf+R#!s_g^(O3d_8oNcoE(Yd94L>KDDZhAJ*Om{DRoNiFJ304j*_H+-1Wu!|3{JB=!CN`_ic0d~R~h_g|G4|<ui+i1l}t9;XR~=@6MXWZPyKO^4O*Nhy_8XH_XeE3POjW!w2PLHewDG~{>}w>bnO!nv1<P}zZd*Sc~J2!gs*w_kNaoe-@o~L(<PZZcmIV?8UJcR990+}HA4wFO|QMQ``OQTKfStp<L3108`G;-COfy)A@AhMo6`^8W^+Ks$GP(Q-Ul!JY3HiMMv^H4lwWw4dlPW~L-ww>EfJW$@xtCmx6CYIb7%L{i+|dAEp&|_;1eMC1wwN2u%yGp)=%TCBd;3~KZH~1Xd~4^Ox{Q02Ng)FhmWJ|)g%bmwr=nyyi17paHecKc;WFjA(j`|kIS`1X1f+n@{)3OvHPo_t+@6>BGp!{t(z`|PSoUd!I&|L#J3lGzpoK$d5*0M&#G(Oan7+CNSAwFv(60*<~|9%E5N%cyHYD8h;+pB#VS)xH8C{j3JX^=s&;jiqFNhh<+huY12k~LAaf{Q(@7YL8}Wr~;xAlvl9UhlGBb3H9be_)P#>NH21B7{ECc0KOLYwXcH>bjdoN*geP?6PZ7j$ugA48frG-nyKwSijGd~W%xQM3&jOH}(92X%+92VywP5u~ANH*jAL@jE`mrO$%#c7W}wLFOnk6is+V%^GCst?wKteY`O&QvMLAJBL*8+)(DkZw};f*UTXOO|ut>}iZ#+88aMbt(z8;ZY-ygC``+%8JKDE9x?}SRI&9y%C>=s@Q0{@$eNBe(;G-JFE4K@l#JN7e;$+i9oob0T~q6qvWbg;k|S{h48~FbGBUein43!J7a{#zD8i`(b|-0zY=-wosp0mPL?%dw{?0g*9QKFEzo!0t5Tgh3~ZHZ_Lm<B#h?xQ?!!GDEpk)suY(c!1<1qb+kMgEU4JAXtv1^`wp%y%e*bgNhDd4RN)^4&=;lnNWqR(N%a^8CZyMEF)nw|c&~7I$JOkP-Ri+KNL+YBsyw!<28;Q7GdZZkw@!^sx7o4I>DzFl+fN1IpDB?Drm~b1}KT&;E?aQ_S9>-M)o^qGd@L4aK3VT#Xqc##N44oI9fBMdo?0LlpjC9J4`@;`magnNX|BqXg9y8l%ylCrw&y5$oqU*)Wy&gOhuOkQqrCFNuvQ*F0MSN*59bFD^Qe)$BJd*)4o#*}=MlLPg!zwR&q+4}1NwRGhRl{R0J|`MhR2?q=Fy-i|dMT%i{B=^!iyrCLJ)qouQL+U`9gvr%(~I*wWt(X(asS>}t9X>MVU`YfMkBV!2?JIh4cH!{7Z2-ZAad<Vj*&M#nrZ4me(cWNto%EI?<|=f&I`5Xbv?;j!cNnO{=I_x@neh=nyz$>Kw{Hz2v{Ke4?(H$Z)<ZvMx2!;ze8owsgV_x5E0Dc;st$Qps2R)xLunQtFS6Bf$J&d@+2HwInMUV+}`CnPi}lek#hsKNogV~yOi01S!Nw^81G(5qK*#M`=%<uy_fS*Ym059>}T7aXQvm?eh>BDLQ2?%1J|j-NXY^ODeKv`=a~;FlmuF6nXrG9p1jhp#2ORsTmk|S8*O-fvmApnus-<~HXyck%I~@GZ@~jcg`ll7DP)=i<SLqm@ae>CpAPqov|8lfNY)uiT=8jcc=)8en<z{~O$iBh_()}dqE+=|1LDZ5-~!|g95W0uP&-u28K459WXKpspxh^qAdIb79AJ<_!i;Jyia3FaOEHgfL^})FnoSp6!V2R-5e@rEoB`*`gJ59DhCh}9ZMlhp^QP*o7X^V0`Q5<BHk~NN-3dD3BDWfVB=Dw&8*VPL+p<HQhz2l;1FE+N7s>?A!q}2sVE}<d4nd2rJQQ+95&pr#i-cU^_!sbDVj04kNNLQ!Zkw;C7u+Zn>QlygtUje~ZTxkUiH*7-5`5WKU%oT~(E=NBZ>?nxi-7Ustq|jpUg9Grb#=@gibo!QqNsVT>Eul3NTi=UPjKAn+2Ic06P__g?XI}EEcp;zwja8JTAlA|W65AG8f~zhj8tK*6;IeDf=Z~p9*OsscDd@`D8WZ@rlZ5lvz7juv$35INWtv!<0^MQz9}QZBl|<t1PDP@@)Cs5JajjN@Bx`k2n*LkB1Cv_Lfa;{Uz+`-%Rs~INfs=!pcY*!+*OMToZX^tAUxIiHf6zDGVvBA7zM^30&4+pTewb*gX{7Kp2dXMBx*F?CvvsAur+U}i%<Y4_l#)8(R~6nrBcUM3S**|qgZ@d5<ajw^yp;%6()dycEBPsUAIji&jNy76J}yxe)TQm2@zg~4@3^XFT|J8r(!3J!|9#lf$=4DEbR4hIDJZU=iGNCf_sIB+o{kQ^Wpc#jz2M1Z;B&X90;Z00_QmX;7_pDUE$y?d8Kf0Y8^~c(lrm<(CL%eL#v4acHctOodq_i5|l+B^TUCf_v;69G%rk@2Wnd?7bP3ELsnWAA!>{BhKfXnJa>1s@Dx!f&diG1w2CbkvdBHh$>vrUC94EBDl01^HU@k2G2|FXRW8@*+#j*3C=ODgiiC;<6d6y6fCFXm((ftgtZ1Gxo>j?T%9hcdQegpHP}r4}KAHuYBrLUNbfcgQn*YRd&UwyG@e+u^Ms^gY85Pfqu&;`d!2fu>Kea7fU}!>OErn`8CrG8FYx#;6IGIug<=4s<Y#&u_C2x7}u!-9f^xgz!X#sFWd0oz`=wtTzK-kk@zS=%&l&uR6fyP07{wBp#n+pUQO_?J?9odycovGPfbrWz_TXC8N2FSi&G^^uke)~>1%hvUsZ<ej=oNmsI@Lc2UpnRa(0_k8?8pvJ<FZ)nRUXk)gfTc^Zwf57O0=o4K%LwdF%fAw7#?SCL2ydua<rEGjrT4j}9B{R%Zd~0RFjb@?1lD39?8Slh0mQKzd~bGGLP`b{a1c`0_Tvij0Sv?-oJ0H<A{gkFa%l=SHfz}^M-@&CI;g6X{Lg*#mKR*dGK|klfUs5Gg~Wxk)HMjq9p(1Ynol(Uw=fEbqrI_@p})FGKklOcMd!YQcAjJ>*_6a??XHbkIca#NtgF5ksfJWHpK*SKnKaoNia=6cL=}E66{~vq2feO!(NDPzp(|iaoW-nGmpdu6<Iagwan|Mc_v0*Xr;zrSN{>Tk7>}MD;uQylxfrIP19NbVt!Uf>4)T)Yt~h9Zqyq%E<csAp5h(>KYHurdXW6fuh(aMLiy5eJn<)njZuUzK5uU|HcoX2(LRSK`SKjvJ?F<!~O-X&(X)L$H=;$IxgXC~+1YvOEqM0tdiU$ak&7^DkKTJ;!&{&JuH-&qDwf73z0L0CyMyZW<Rq0mB+l*hh<2crpRRV&}D8DcFHM$lEdbAzPC6CvDMp>5h`0=o%qH$2lXKlf{aireLwyn2p7Nncw%~rqhwVE~6=pmXHFT4x9TmO3r#JxA5B$U3l1u<9nz%Qy0iTWl%et4*8!RTy5W(XH1vx9aBo#sFYH!Ra!L42V0s7<3xdvv=qS7XRH5@?Bv>p4hCxYS)%EZ+z2sRdD|njXJc_XL68<d6Zk3YJ;AK9yXF?qS#5P3@P`wXzIy;C@*f?t9fK{^zdDRGxE3dsbk3Ez$Hi4tHhYD$}}mm6{)qkeRFm3XY+9{?`3wwi0byTlo1~c8qj<<?XWdu4~=x^X|4Rn}=_o<8==|faz~GVj<l)faL7rVT5NJ8vU_EMYJtT-7GU)*f8mYD5L66S>r~TZkD%kx*8fSf?Lzy@6-N&i{Ks)*Ks0_!6Ntl)e<zwH$|4y0GBvL%kXEBm30<39;6+|A{%IgmYb&qzv*`NBy-3Ym|X?s=ArXJML^lPe<18UrsvtH(V+_96)L1dukb%sR+5@yomLiUEBDAS&PF_r+AxW;O(u!9Y7q}5d@{~bR$W9Lm)oPS%Uds7G+x41&AIg_<kvPWw;d{HTB8*(7XF&x&#DX6``vWYiZ4Pyn;(#H7TjHh#Ph{W*{WJmuD9ibxYw)T=LTq%M!;vdxsg$#hYUTPQwJ4yB@j?W#st~I!A-VgoDx!E8<*Jjm9Qx;w%{$MM|MsaW^HO9h;kfSEm9YexL!)4+pJ@n)EWB@ljibj702K1CV9HvFKn8<fh(3gE{*DV)x!ogw@bLCjJ4DqGX_^w;{;q(7<?1mROscS{6`h>LPKS|ow3Q@Z2-4u$6^h2+d$o(1M0RHstMU^u&#n@Fa+#nRJLDktCw>#k|?EZ%{pqYX}5;@DC6~3z3{xr%fZ0hOST}|xRbDGEs0s{Dx{I+s^ZOe<8wQ)^UM^KQMp&VMXYou%Dn;pg-umA&bkm<F(N6E47*#ML9q`|W#vD8TB!0;ZlE+&h0&eln296~o+*P{GHE3Q^{x0$Xe{g?Pqtu1v9-mNw(6t%k2+wLK(R>@MC+F_3X>-+l_@MwW=$k&>$afjn8<o;AD76Agvm`E$*SBG8jl#N?eWM8{29pT!sNOtFZ->|kWDCjnmC!v!y|?eI&jg9oTCTJ6l&GfNLzL(+sG-ihxkLm2g1M~F!s7QxQ;bHl#jhGYPLf)3u3Peu}ZnF6<1XVbn8XCYG26pMQxWrY6`7V5!+>1RViMrVqUVsv_BMmt%#k1;oY6(!V2$RxMCEq4kco`tMveO`#+339VVB^oI(*-k3A?Ut+g-1x!A0x$H$wu&rENeYU(S&icX;RDw6m{{xAkA7=aol7j#uc!h=hF%X*pYB&m?Q65##1`%0+mZZIJH!au&A&1=>BJs@*Fs_1rVk+=KR*^+|Hc3yPOc5Uj-*CtP>mnX`CN-{iiM81O2(kFj`!asHk%R#~$FWhELKx1oPxP<PlP6QSMg}ZJ%?h2$)!bu%fNgh{O@;1rFMJq_n80mhdEL%%p26M4Mpw3!ld<lfmUWg#9pZgM2iSc5qRFXFc3i7lMHc;fb=tdxal-m(EkJKS1hKh<)P^siXC9+1@vJj<*;$a{?BXqy_sSM~Zc!~`&f5)TY4_T|qDNKx&G@n}|b%2=J{@~k#&15}RQe5hmTYN_<aK_`z5bO@vizMml_}F%aIfCm9u6B6W3KS!(D|({)Lgw4ffT*6LD>6)q&MwbgZ#_=?^Hmn-(kGL7t3ob3)NnlRl&r+t@fR2OD0rr4nTUy%5Mx<q3>c4lqgK{mub|<rdIs#cmov~N2lKdS1e9qjRH?$E-q{#1c|wxOfsaSS%YKYOhCRkpIL{fEJp3Cym-pZii(QMGxK76}u_B7fbc_x^i-<dHYu3p#d5VG@l)|e60c<Z09E`Xo;n%6%$UiexYpI>d05EgM&-AL_hU)lAwN(7>rX@#R_qU{Su=6PC{ur(BX(HrxV4zy$dAjX=dfIXb3CA$2+C(UYnjN!9AYn%>@;*HXYO;5NzkESs5UaRH)FyO{^Gd!h71_fVsRgwQ@p16z`Y)lu#}_Hb?X&-hef|v}N#HPb5PN^A)Ise6bufc{dDM|YNtvuFgqLcRcr~dPi5bObvT#PKDlb(n4^!KAz}H8RbXXfP!c6YOBsRXtQ5dwsmO3)3kB%>;9;|2aa06fM0};(6e2;C{K-}|i*EFWPAcN|^4)7(qD*}aaOs!DvDHGXdZ4qiF>=6asc-KK;B+zO`oz;D1gFF(MbVW_xgC<4FjmN30u|5A1>?a6559#-Aln5_;FufoEfl7u#P&ru1H+)n^1y@$yj#`ktE305>9|s~ajo*Z<pqM#nZK6a6YWD`~LJyP?#RlukOQa%hWY7mmDDnzMI;OsqHSn#Sle#kMyiQ&$l+Z!nL{z~YY`Tp|ei7O#uZDpvYAr|ZgNSn5r~a_|;ZdU-G5*uvMan^c0#iBD30bV@{y;~r&grgc$~G65K~|dIOK2KUty))xZPze4JxIG24`?V=o*hRvTBY5@kO;#Dz#>Y9S}4?Qolc-p@?oaN0&ia(FR#K4mPXgERk*!EXqBmGZITq$!K$I%mBW~R2;NN=eGsmQHd^YdK6@95vewr#hR`T!p|dBkq1O*YWN>8%rT*vmcuZ_Sa=bj@s>t*?8jQder&I_PBdp*GfvBNFkZG$1n~`AHkWH(_8+s<UL(0dvB2J!shd+L7%jGR8$XQ}pfE0>g08*49G4J^XcElxhe_I+}>H6jx&CpX#5}6&uP97o%kvC-9>S(=sBrx4(4Au%665j!(w8EksoN`EljzHM9DLM%nhay~!wCIi6scGyD-fhv%n`1~j`UtnO6<5bLOgD?mEm%mrdz;o*84jUo3)ZZrJKG%yxGJLa+(okp&a5U)U;&+YH0X777bK_K3qQ*ikH(?m=ZwpN@iU}8Tnxz&9vD06gxV{S;F2`qSzKrs$yv9|e8^-2I4viNg1tBpBnoO4H^es23F~><0VlgYoXbde(vr;u>K8Z4T}2p*yf_=VBsXD-O12vYQKo4aDZMBZ<!+rhxNi0>a(0GY77i@n5xT9;Nv{2A*W=b;ttC9M+)G;bf-yP$Mgz9T92DeUE+ikSeUHd@=15po<V(9uD6joV$$^kxiR7=qfG)Cxk)w4dnd$Vo=gfspp2;=Zz4iQ`cCJp&zPWqj{mHZMO+MO@mJuf5HsfsjPdl$o&;Q5dqnCGY+}yo!o|T%u`uyHoAMV}ycyjYy$fwC6FIS`CU^o_+@*>n3MT13#By7=Ta-2MxTzqqS?IrVC$hoA;_Cz2!(@wDD<n7;0u6?rq(OdB1;)j#BFHf$&h?3=qeCxdYv8stEJ79wddX$3e1ITl%nv@T6IgltN=r<Q<Xzw7J`)ZXB#iDLg?H){9GB*$ui*|23Z`x9AAiph0-T8!c$u$soCl+%i{88>#FmGjXF4KQb|M1@QnM-VN_CNpiowFCF|Mk}N>iIi6FNj`7CkI<ekNwNEa>zw1vTah#cs^Cic3$N*5*a-AZoRR4W9QF1Pfspgo4kE-@A>P1I|nt)23y;9(oN30q5T5MZy=|hoNUwrcp}-Ro`X~-2!>OjwW?ezRoqsvL$ws!mTpEccj@9v^`8?hUhS?b*;*wNM%^?nS=m*o=BjQh3*{>FDdFO+rbemS9>nLW?_fT&w(K$&;STP~A-_~@5AJi-cX0RAHm3z@o69DlI(p_ta`qcc;V?JltpqLx8j<DHHEUPQ3l^q#(B4MM4I$V9Em;CI-TV}Rwzf&LL7P!30*m$zb6~61r!x4etxj2tOThMQq-I06xvr=I|0bdkyhzb1S&+C9u}8v%2;OP)QOG(58$Kx6Gyqpk7#6A9G$OEnkNav-s70~VWwh3nC+j*kkFI68-Ervp5w(3$Rt`BkQmzb@GUJ6ySWUpW(K2~&O^%01N=Eh`akzO`#YyS8EBzdu4tK{!=D5|KB>!Bk1;~QBWw#<$fKqvTp<N?RwYu(V@W?aImx5P1aPmdxOK2E51tP<+Af>8UffkXSmC1ksT&5f86Lx%SgpWX#*lbA!=d8D!$rj)P9&E%=GfcZiz+KaJs0$1!A`oo9>G|;D^20%n=kuWUSdE_29+V**9fxt2<B!4RnCnL^w*K;Auc#wQ!<fi1g!9a-nufC+v$|ssGlqGC6x~4EUiJtFw6UL?u9ygm5dnGx7516p1Dk^to2NxNug#%&!?5D;RspEsF2cb_l?+-kJ$nlWh$x3IT$kQ(lsjl}R>~PC4)~--%-DmOqzbn^?ud{^%v`~3%xWaGO<Hkg7HQ0`Rtumq!#G@W(=`JbqTn3{CZYY?3_R>=rBDgtGHW}yVx()~(3I)-?9F4SHA4IHO@;TFp@!@lGiX?DTk1#y`63*mi<cs*NQmxk33LaYzMNqkF1txS%-$5tsl)0@c)xy_9Vwoa^Yul!N}#i=cHD!#hOw>l^;`SaiPhY)VqZPK_wof*VNr$vlz_gteYx`Z;uF=Ql^;L;;|G5H#E+|BdldDae`o*m7nk?ndT;OTo2vTc;+wm-U;X0t1@U%Kz7|0B%O&x2_r}|kkA5=w<PZCAy=9v;jn9GBa%i8b{5qah#M4bqbVv80$~5?aS!v<~h-&lzdP$&bPb&;+&WnInbKHFwHQ}h);-X93>%hrC(%MLqEY3O`+iKVYCmbX?vTT7o6TUh0(O|O;K~7ufrXf22*uC}a?&q)Vz5Hu7hH!{+_5Aee&B?hx?*Hz$cg{Zd=bfi_Z(N+ddTsLj8`BGCr*EF!z4hDa%h#tDe=~XE;@;(F_W%1mHuB;X8+kqGaBZwFPlQBp2?q<mK}y!^C!<q??1VT`Dq%u`cW{%GR~9$p1Ffn%dGh*v5kP07hWqq@-5$jO1w+G^%rP6-x6Dz?KA#_4753d2J9Nv-!NA<)vJh4;2y=SwccE>9VN?wZpCjX-PExT69yh}KN(fo^oRRn}&7}kzoVCvy^@|hEYeB~bGN+nB_c!Y*g11(IXQ7*b+4Y~(2B<8_2R(?9;};WnrXxvVKkf}qxv{E78w~iypx2crrBDz@Cu5J7m*zfQg{F@x^d}#^KHd4<<oTcP-Tsg1xfiC_E=_K~Il1!L-ubgkqU_$-*}L`3-g|$TJpB$6KSCN!uHBxz@S+wo^mY<QM4Zf-<MJ<VUy%!-dHeXo-(l<3o!5S|_mfYU5K}7^>)zzt&nD-7KfQ2X!xU3w@{0=!ioDg!Fr1vfh$%Ap@bc~_=M^xnj2`96s0J$fHu>4J+@uOR7VqcVd%wTt(vX5vG-S%ga0~|br_NYcv+NPXr-#MD)~-2Q!WuUcoUGN?Tp7m&)UoD*=ItPQN~X{#@wP+>R;MO4;<n%vCC&?e&up<JCuH}kI1cQYz4lB>6Zq28ciZaZM%paom!{nlH@N0QGekv5xg`cNY{{_6;U}}j=P<EmT5~-ddU#f4W8s47{pI3949zZKC?^K32|+HXd5Um?2fKd{S`=hOErzEFr<%i&db97(B;k^3G2AJOc|9}R0*0@Nb^TTBHj22vxc|~(=TJazbtwgAZ*&Wpwd6p;$GhpRmg*CZ3p*MNmzN;A9|UQQcvd%~T=K`B;q}Hq!2zrSE0Jebx&c|GJPPDBh1<@ymgOdyn}5Z%Vv2a1al^Kh04NTAJ-~*YZYv*WYq7d~svL>S=mnlb>msro!U49t3LaXA1B4H-qAq=J*5Cj#%|KVagoW5?bLM)C!ph7(TBUqhYMQ~P=wMfIQ(p$>`F9+v=PSOkSIyg@eM&B1fkCU;1vo`Qm>zOlvipTgCF*F|b<$6p#b8%LOQaW0czFzgFUyPNG(#l#4$;p*0xhKv9tsAI9w^ihkKmoHl7&s!gtEB5-b0qdX>7LkhX~jT!VNKZfKH2nK81MO52k>*z0hYVe$w(LC;;B^NA5e^&yoA0liyg0e(;aqfAo>XhaY+L(dhAi{I@5*_wctKU3}t^fB5G|9(yu+{9nHRKfm|S-+S`kqQ@S2@(17lmw!5R9JqN)M*QTl+NnFo!otEou!r1xk_m^!K3KuhlSw3M$lH~FFNu{$kHwEgr#6y)H10FNVE3&n^NH`0dhzx2(xEx6u$<n6iD-PKeoUfsrdDuH22vG!2t%`QV!SJh@e<sTz;?~E;fg|m(~YJ2l0z*XvIt+uSMlcXGZ1<-w9s;m6?`CC4w09fo@hLd)t93CBZ;~IdVi!Sdo-%_*-zdBGk+H0=>z<>KP#b^2f<qM0$7VD-Q7s@(O^A`Hyti}Odq>NPi#PF@O~s~!sUoK>ZW;zA)G*t$}|(^K?aP64`p(?1eaA0!YVC@BQ4Wivd1z5mttE{US@(sK{*s0_Zplis=^PV5i#`A7qpnx-A*z=IFPqFcU?g=dqfqDL{XrWzJ{A0-5hksJx+^cYgn<UTGbTeKq<laqY&Gu^87EElL9^Ew<_*ElL;yd%p&?Q8?;64ztpq@oMl|q<Y+1uX+(?iC9QkE3~OOMet;j<cxC}a17x&PG}41|ifQ)5KsY`LXAV3zJ>SRkdr3Sb^`JMFX7~qdI67V9YsV~`{0{)=VkhbKj2T_rZyM(e$<amvp+l1BQPJ8W<C$I(@llO4aCc@MOFJA?EQ+&w+5eL>`XwjR!8m7x>#wsW<Ut-=7vmmlC$AGU*w~1GC6eC;Ya&ufB`{FURT5p)Q=K2qIn^0wZnF6!$F%trG;&5REZqeB@+BP6xZ}-_>eslK6cRVv4mHu=s>i+c0VCDMW?onCw(Dy2P-!b593M>b2o0?HN{jG8CgBLx8+9nyNrt275&nnO4x!ykZTFZ^N8og%yWKJzLAu0y;e?wZ^g=b2&!ulK9cbRvR_u?OcjDm%d+$M_k22QzM&GUyn5OwK=BXo;@WpX{+f1Wve9fAXPlyV5<<YJjD#s*$mv-eO`X}gA42c6l$o2+2j^H+=K2_6|pAOczb446Mcy*E7v@?>g=w$2+23Z$Qfdk>^tsW|DS~zQBu+VmyLZ?ZHdFeR<;dfoss!a6fY!A#{8q%s{i9wGMtDiXytn9YpSfK0$iNZp^U~zJjVd&Q;8wOdVaEYR5w*2vZ(RUc9FjZs|ys04T;VCvvq9o55JsB(Ywj+r0Dnit=T~?$upl+AJ+hlSmhfw@LIHks|tL1n&8e(iDy3)P>M3M}}UU!`@wQNP3L5-CfucSMR-svP5ev$8&wz?}~*Ub<{oui2J*#`d@v7pr|U8$bMCzD>qQ$&1e%HK?_a2AK4sx?dcSYzsyBPR#_RzDeUG6hOQml97!2IY%Z;~Ps$UXAu&ncnr70H9?HsC9hNHmJ+h|IrxYin6V|R?$G;53JF8mh!tW!R-Vuah#%C$k=>JvJp%&wIq?%Sq2DIK}f}FtLACT3NvY~rvL&VHcT7lr3hlc(~cn9i#MjEauAY^Mp6Mp@vcd(oBqjG{-bP!A}?2hxh2E}@sj{mf;aO^0cxs3CO$J@z}{4(YmY7<H*JPRduoJ<b)+vsQ3tVt&!TBxB?w|lt{I2>&3jJ5Xa(n*VYE&9c!wlIL9*xUX}(eM;pnT1el_u%<I!@mxO|MrQ}jNdQ7R_2^?nRlYyTMBJ=p^&ct=&eesCY4B5rtM03NXzpNBR-k!&{>;L>@zDYP|x_tudkjt9>VR2E&)7P{Zh)P9Je7=wRI+8MDWNAs_4i??4tqR5O80PbbPP-lkn4q&S*V}ro`A9b`LR;VYLz>eaRpab!>ZBaU*2$#Hg#1;vr#3_$ywjge#Ob-%y-Z`FP+A8?Ngs~vgCZq3-A_h1g4~M-3P>8ebzvlvSn57Wy!BMJmId-s!itSHdry3TLRF<PER%v0%ePZVln&KXb48Lwj*w7qK)=w}>nqX>N%kJ^)g?USb#%DiAD!$IlJ&F{aiqq`<2Yhg(P#HK?VE`dQ4MOy~6I4@SGZ9824m5@WL7+DoA>DwTjH;q2hyVf0EhHw26$}+YAP4#e50NPaBX-=q3a37SF?v7=aRkO5Y9Sv20HkS*{UamkL&f>@4w0iu$QUpS7(rvEEFbq7zJ}%n_Ta_Nh@DxH`eoL6dAy}MX-+vE@$_xLozFruO;33Y9NHE!lJ{+oN{PeP3a0%@5%n1B5&xnSeSg27AZJ4rPX*9CDtBg92%NNAmB1e9u3Nyq)!&X78mwPyYe@!;YtZrz8;Gqzo@g7eYH?A7(^do#0wV`;;h|BKX6)-w+UVG3^3+DcD3k%2wxf2^8!&M=8fZ(87y{wz$Oe1R*%<V@8T=AgD`bDT00tGR4PomOkoT+^NZ2T>K%xtaoSsYs<^z$JRP&HjW7c4;=kC#?jhM?`*qCOpb?FKlek?}Niwi3+#V>EX&-1%)^E>9U2f&qT1lU~JMvD;)5HuKfyWD)n_eq^zoagCU$^gY${w!C)=bH&a6@-PYEiEg<>M^TKf6y1Ft+dBH1P*r~R|V;NWDj%4lHV3bvo}r|bNnl4A&+F)AgkEhX%B?WF<Wfg{5pSq!S<}<(dWxLjY|FwIB4{B{s5~5wzk7+58HLP@GP%IfNU3bm$Xmtp*|i;FxWBm9f`rMzA8k%a9&~J1=@9mPSXP%SJW^Yu4MAoMTT{pLqQ%|wzqlhXl8P=Vs9oz{(>B5UI69Oq>&5!_eI~PrYD#=<2+d#Gifzoiye%h(07t8$koDE!<?G3gbJjMJXz9V`z|Zux!pmVArSiuCh&{^jdx^^qPS`A$Z1{2NwfzbeX-RwDdc$<Y&V<gCI|hyTC|;_k&Io)+=s}=xSZRlLB>a5Z{Tp6Q3gIPZ>{(#<z8?run4Mr-s%yAo}L51whLf+E&!)p0LN?qXDpuAm30)ZZv8v)DE@B7*2g+O3X`O**t8Y^1TtpY`K2lwSf+ti+;Kxn5_|^zhC4Rq^~T^-W1*L<;Rs56Qm|#Yu>i5mnQUf%7Kr<zyiet^?y7tf;T9KE(uIN|c9IlNb?>{-7Lep6<ujURO6T^q7MMHJzFFo-VXd`_t%?5~+64>r*+?q{#X?Qqg5PN_^Z>QT#P0=hL6y*UY;26s3sM!QFeA$zwfP#=*<wZ+X*zG<;WUy!>i0G~;-sBQy#K#r(2gKvo~32-y<j9LY<A&{DS6HV(9l@3y?B@>64Hq4thh4MF2$L(bCAK`*M&04{@Wb0XKOKJtJT0_)@mXWG$dSuLscR7S^~0-aY|1vFC%&5@1MT&B>RSFjh1wQR)d$V{|I_69+|nA2^Z6ft3n~!A-r;0T(B=dc3HLp&{;+XoHt=FB=>x1<wTS^2VD%C2r)3_Ev=GlrlsmE%Q|`+qpIPC|E7Uq9`OtXhDt%Ldb)vrc}vp{pkGeuzm{RTFZzdZoUvU8+#}3EecGlw%;<aH;iq77r$h82w5A|+nK!|w2~torW>iUng+8uY$`;qvL63vCZQonRKoi%ZHk1Qx77bDjj{#D(Z0kDYIV3ktf;?h%aTb$!i-)<yDgM{OCNfUAluFcqPID<~!55n3@L3EOWAoaK4$hxM0+s2p=?7EP6CM#>Zm`0K*`wyv5Vp@UHRM4<U<)QzA%Sy^roog>RT>cPo<)bHnl*094(E6UD`mzSFMbJJ!vL1Th@swa^w-KAK^e-{*}CPlth-yYdB~n=E$nO7PW$|)L05s%4_H&eQrQUfBs3a2iZgH3*ae7>T!9(UN_~#<0v1%7ogt8qbL6D^3vU=XyF^abzTyf_=j(B|TY+(_fDEnjQ(f`Z<jwHym!UfENG(zmB<<?y0ST2M`FMZ4$?bG>e4Mn1e9H;_(7z4V8lo&>TCkx=+na)f%^pn+c-4ViR82RAsv#ZSNPG3WL7|#T9@U*3fWjwCNN3p4))CtcxW9_Q&q~Usw+F1{1-~`Wgu_pIaBSg0dj`;ey%Gbq?G6PnTzlri5iA0Tw9zhHju>hhenlf`@6iG-V8Sob+50d06A0REdkf)2b3R?)%`yj{YIOho(J}Jacge~*<MecDxlW_wKTwSKeE1@soYNK*G6Jj^N#QZVgzN$<W)^vBIi`XYI+?=_;VR^jr%KSUu;NhMQ+u|_!IR8JeVnxw&jOdZe9hZNj1_0j6xQPAjTMU_eM8}zJys%6$}8xN4?(X*L{;^?^T9w?lNBV2!nQj8HK=k^L!CVl3{;<Hjuop?y17hPzBz>JJ-)+JZX2gzTI>^3`+(GU;K>mNJ0aVkKx8(Q3%c>uB^8fFx||C4QuT}tW@h|QTTG5%M-M2YJxOq^4YwHeCwC5E9~M+7;|ngx5v|3WX%CL=gezKdN?>CSZk5SLpWR8vH}2*ryfc!ubStuq!1CblpTRZ590=#>mRj{x^tnjpF=Grj2ICXMk}dp@c0LvHj1;#xE;f*?(7L0E9D^kS-|FQ9CLI`XNW8tqTH>xH+<RM{jnp9e%uzG6wdlkmchqIX*4(iOYBWwB;&(1lnOpl?^?%?e^>V5p4H;PsW-7$OPlc(coDryqQ#$XALjw2IsbSP>`;~o+HOL#TTs1j$gJ%yO88f@q_JL}hwrY`MG@JJGY(@<2p_ZTap$$sSTROBm@kej4SIdT>MW~29eAYUJAlR>hHDR^#)`J%|t2s*;w<-xrIq&Sso}~&^pS_ToWvh{oGG5C(a=Bub8`!4jy}Z~W-izrE;h;2V`zVZ0b)4r9<7^j3VfIAf&4R-20;Lbzf_C#W+LYE0T){#stnzP`o_4(GecXfQ?miOG_j_Gf4a9{AwI;E7n6iz^;kKp2zu6R&Ss+%;b<M51d?@eEc(2R^z`4s9f=u^PTXnavds~OIbokQDd}t6S#WLN&D{w(aaIwScg46We$a^CD1$z+4-hKDo@FOvIUz~6?;jY+45346$6zFB1g1cF)?Ck54bkW5|UUHw(JV$!4bLFmDIPVR9ydTb0vYB<2kY1JMGI3|fTzo6hd?Ml}sDzoEo&_m0(aqc@w6>lN#>1-g`(a-}vq2aHJznOKXWf)7*i3?05gg#d7cwR*WeF`TNtrd^$TU}>Nt3S4jhq*KVM7ndr37JmMeER3t%ud+>gXulOB`-P<_@>Id&6j{328Dd=Y;eo0kh{uw<cdxva2X6m`MMWj3l8uM}+><57VwnNJW}xq3haiu4owp3!ia_cntBD1HuU&X%}!qMlK;d-NsRTU^|#p0MnyUp9XR&5v4L^A}k)M0LY3)DCLYZaFEmp$9gdZxPmHcWAWet)3m`p)QFC)5<#GLu;U>(EE#~*DCzij(eakbs`b>;&1b-94ViuoeJhFq?BO+czt|6M27PGwM;VfFcwfv5ESQ)CKH=i?^uT5i)MOzcy^M9Y<n(YlaEJSJ9x!ElG_#rc`$I&$uo-)Y5-yG`mK{)zt0$h3x2Ajk;af6GbcHhV9I3kDV@Z;E+BMNkT(RLcR2clkyP|l|ID<vpebJ*2KTJ(^_TrIShuLa0l&mF7Q4V*gx8b<mwBPG=w5=c?x5W(|0;Q_b#;Orqu`0t|P2=do9;G=y$AstH4TwGS#OFAUZ~PLq4nE;M;(9CJ7-ZnvCoh<FI-z|9e9(^7swXa{vFg@nm?bf&#~lnqi}OqOS!stFkoUw0Yqf{oDF=mv_Qly*mKZ;Gn|tW}3!{ZPN}2zz$N!+TK5!ondq?pdalQN~96_*u+i~Ya8)U(uHY4t%x8Tx?y89xem3X0U^UGqs2wG<QfM7L246PPK@R{cVe+#Z!L^vWFT)iMDRjXR%c9ua|Fdua>4EB?PU_0aPV6Cwf0(BQ^G3o|e+t9D#1yUFE=IVft9bMm|iC4Ms(nx(~EZGX9*stsublAH1CE>!x(a@CrC{?L*a+b;%0*KzZ16U4Yh@{AxCxI0yu}Xn4dRi!sUI=l~N+}(QUP{j)+F&fqui_w6LL_gP5~qaZ$hPc+E$%KHk9kQj7%Ao2mXP*bQ1VdiOCjv(%J6V+kS7r6yzbJrquwb|3}zku%p5@51T|#}Wix!5w2e7LH0K?!m@Y##r)yMLLUSY&yB8aaJ;Pn+UcCB-pE~9hQ%%=^n<!Ro60OjFDWgDz#@`g@B0iV_g-K`|6W4O(lqg|GDuEai*&)2Zb?q;N7MF#97m<brN2}4fV3#$a2O5!+_S+^u0W29j^ZB6P+vcpwcXEPJ;dmn+bvEEelniyiHpkt>GuP$H7jPJ&8Z}P3U?c;6ie!T$@I8@0;B%r8Xxn2I(S$>8^!Vcq%A`KwxQAvS75jV%{XTeS45}D~*&^5O#+N2^J@h*yu^S>lk>EvjSYa1}xs8IQuv<>xQ!I>et+L`6)d@;IT;?uhA3?8_Q2q-cS57pe8gs&*Q#`v3NdYY!@;qWz_;&UOO!D*?T?a#Li9;4QZ^3$&Yz|I}D@{4qs+|LMRaE-@I3RqE=PDd7Y03?ad)$d=!1{;PLij1F6}78t&XiK{Y3fGi6jAUAiz!L~&Kfde7WeLH&@;U(uwCSD(1~2gmL(@O>2-s&7&L3S!5$FNpAy1yDNK|zMSWZ<?2L;4RTKl*L1mWQHuzY$SUVmY8IP^QbyJA9T8IaQs|0GLSUuwv$>9yuCwLsXCG8O%woeCvJF5M4b;vjqb`x|r6%hMyaS4&d3P)_M3bTS2I}5uVs{%gU?Hv1{yD?>oFcgx;6Yg{duD$@N1TT-SMk5}EedOosK{z7gj4sRy<yg>@8KE8h9WHeE4@3eAbsDa*F}y@xHkf5rU*~X9({=7IbDn8gV&-VaNM$=&OPRya`4pZ!5y^Y!B@gx#Lv#rh_nj86&PyuXW|&5Ep>R;LZH(Johx3t40C6*{(HrMQ(6OD=qivK|UTNJ`ue4A?rNaoxgYyKBH|UHFu5aGMbPJWaJOIm8wXDJ=YdPx<y0c&Ih{j7l_o07>=awt8!&9}oY0e`=d)big^E?eXTPaVO8C$8t)>9r!PWdkstwg6F4kI|lLafLU1m-bUr=*X(1j4E%N|(ua=suCdU0;%pPJ#o0xHo*1zg?7Z8}%CXKMuENkT8jEYsBqpNJZw@_{}|>Ccb;f!_UvWm0dLv579+-bvyh(?ajQf?QK(E%$}=FQpjoFciG#EghghLU!kFY)k7yw1wzBBr=OlWga%cFR_GKJKAv089+gJ(fPFJYb=Px|btafXpFEqKN25)rOo<41goh~hl>7$*HM0lS5FfA;f-Cj#l3q}TI2f#%-LSasfZc$Vp;d4JJz!>l51zn52&-c4@dPTElRIZ2>c_DFecA=M9pY+d_+!L{i#h2<i8z7wCrBmu_@A=%1W(1|mW4(4FyL6g0nZ)+)dwCGKm-fe9QQ`)u*bG4Y#%4xIm&TKu=^!q7<lWJWiC64rq|2r&?WKGUCQ63dI<;xvh-lJ=S84M4j!=jyeP~P4Km>EmmzQN`AqZm#R)udU-YeRmm9%klyjpe>k~uB@nJ_s76%A2cBJy}C8?{leuoPM_F>j$gw}8Zv9P%u4=c!hPRHg#N!&}ct&gw|(mZw3DBUL7bswZv`<$|TUo8vtyuDR6bz(&$d3~k5&AIL@f{Mhw&KNEgIJQ|CAuCIvJrY^Occ~j^UXaP&t;cLX*M?-dB$b1mVS%@%ZPz5qKML3(_#GTcb!q_-m#Y_VCz%Y3@m)4ClDM;>-tjn$-&*{(S~T#?gejuai>i*{P_Vf=?2W#~EmhwZ2?gTJANNTGw4HgQy)LVBi6L)h!!ZQL4YF<+7~g_qKj!DRt0ZGW%0t<X(;U(}fMb2y@1zjFRwp%Z_Q+z39kbpJVi603x2|I7a)Bz(RrvR{xRcO;Adqx*#P6k`q2~eZZB9_qNA0G;T;Z?ngudJyaslcFB;c5IvH_U!P<vVyYkPHBQG5_!(zErDz;U<K!+|B8G!G0+rCGuY`n<ppZy%KI6hvuiixUV9hEOISCULfTFhB&K0zhU}mlD~}>@^z9wOXU>P9mZT-MYPPLKDtN^x%=Z^W<p=f<P*l1S0Q@(=|VE&Li|RB`AU`irH_3i?|WqjQ#0vieS5O-bBS!BBuCQJHauf@Iu|F`YeO|jV5=joM}~%T0&*3hTW%N4iqO4kAdQLumqH9VISJM4>D}-UuhroS5S~uywA9%^Q2U3Ca?f%ug)K~ZT~kjg0%=y!n8-7E|}EJ({?Y>@r&J^hJQNO*Jb7!)n3&&qS+AcFA1X^fMNNwP<2Q}eGqXNXi+>Pyb&Ma<L<iQW9YaN(QXzNZ`0!KVDZAm5MnW+pR-^f89(cmg^!3~OA+M`f4NW(4`$|r4<E^5D_<HkUxl!N4fZ`jrNETt;&R5~Ya_Skh}?*|+BBiUH&VkF5(nPrZRdG0V}?p&7PQKgv`_|y$T=RxMC%O(C&u7q_>>?l4t=eOIy<-=g4L{8@%TWlS&!P)Dy%H*$q};)5d;#cQsv5vh(ezGqM<ieO?W)9Zx*4fWpDi281ES*KdLwptRo$U+qb&eA*XR^cNMD=LctmV^G{{26y;@{ZtG#y;#t<22BtP&0N^(b8m$YnjUsr4TUTDG{HK=ayIao``UhIbR8G#>E10#1c2QiaFF6@DNylW8P1;$XRX%vtkCN1lTW!^G-8DVSu+zF4$a=9<DNiaPu3fOG)Xbg*M7{<Zl2<;_4WGc=bdGq?v_7E7U9}7@_W%XI9A8z_R=zs?hBeVuUN_$Y0NuBPg6p?b3M*|-hF?cSvxSlt(6kC+DRiCEmfcoE)@yYJ?<^6>--*5pW<j?X69j?23qc?T0)XHjJdyzo)d(lJ)UAjWOxzwYbCzO9ZvD|$)m7qINo;2awww9m=X0%%5+2<jsNGew(&iItm|q*eH%-S|;da^Ld`}vcW;@y-xxjK*&H4;{rAL=1LY=p1k2%nHJKDP1<O&V=(Az_*jI(BYFG(K+TW<sdJ9@#Fj&VsdC&gO5fD@=J>BTfir7lIpo5zWjidpx)6y5zs@HS1^ZiS4l_o7>t?7xd{ogpW3LHn=XSJ?-2Ia|3q!d0Bj&sMq_qvJUF*E&Tz`He^xk9cu;fR=YQlFeA3j*Uf7#9qR^sk3bzqWEN*oXX*7D>?xed6F$LyuIxeQFXZ<J^qjX_Qdxd{??;fcjX?M;?PSSFljf<2be}Z9E*{_&Y<ED=~zAb!9RZg(MJ{^e&o?dUBKql+e)QwMeC1Z&N}=sP-OImlS0wNSL@Le-+%af7^*l5lXQ`3ga_Gr+)q!(2Dh)(qlf?H`%gTv_yj}g5ncP#M%vksF!G~1n>tXeGhnoY2$r>o=Z;nn@@Nf`Uk*%)AI&zf<=9}mGIH2ydKa$?&wkbb+YAiwX=bjE{*vgs{~x2?Fv%AAh(`}U@voxkw4bvrWPcmwA;+Es%0-xl594QviiLy;<wTMUbCU@@=IP$`g(Get7?09^5_Jcs`uQkJ;!Q};Ko1Jn(k&j_(~}lf@O3}lB&k?s11-V2GyyE1KAsrisL<?y;#l;Ek!mq&!_9&$?xy1$7w8R6IdOvhpf<*Bvz%h2!5^NDNzePXGZ<vuv=3+TJRNgutMKi1;oH;ft$ux9)VPZe0;8s9^rxMdd~-g|IvZ?R>o8fD9Cqu$11Hka#|uOag_5T{UVwBJ@(oC!sE7zxTtf<lZYl7c?GBT$rHZPXLvx_QtAr~c+I>%X*vq4TC8`khH0gtpvz3xmE2@OOWmqfAOD&%Kz>j~YQ`|vxpNMqlQRuj{8Xb*b7B1@FjEC6z)?fFr2qB$H$zW@PGieTzF)5~yRZ&DPbs7W-mBloZw5}-|<08hXskNoTQQ>u+@kFg2zCiIT{am)7d^*m0kFSXwA~FSuEy7C*a+NGtcO1)4iIfK6!X$Y>xR`q*F<jOOe;+PNW?K+mI8Jc5lJ90;5mC+Mx^OAyE?q&#tn1+6k_8^Wg_XiiviD+Y36+=6CZQ0s$usC|4Eo)SX--Nm`jp>8pIK5_E96L7#?M|#+aqIs_$1@XzNA=cJO?GIxK1q9`~Pw1G!i>srG(1kA}rL|tUn>_CC+}?UrX%nwm|-uuS-_0q|dbP4VaTN>&eCrP?IWGMHR7q22+N{t0RnIkEre2=^xVoX@u@H2QBTTw0u=8x60RIVrdhSbxZZL)F>cFWVr|_sifKrkP~*dGN+!Vs@AH?(N$eq(E5sZTjZ~!`#MJ*rev=9m!esSmEvaywYh2u<VY1S<sDlAa*Xzx4YbjqUNoV~*!>T|f`8XW@dMhJ@5*!<oC833?sM+jMu>JC?Vqk?JU<ZonQaVQ97A<^Uu~&p)dd>_%;;V`^m6=0%4R#;uJbFD9QBqkm!4z`EMzW<#~oRIm6$Ni>5(wxtZ3U29M_9y?s2`y!2<8V@q!qFl$jK)ooi{2hqNA?VhYn5MAMRz2wG4TggG}KZ*IofHojt@tg<o-xd@%q__};4W0Q0TFgcM*+E}Xp?Gi1sndXp76c;(Rgl{WMuD&<<@WS-vkEiE9xpVf??v3Xr=ia>Y>$j&*|6=mt<v;H{J-K~z@7-I|mw!7w_p80zHz#j@wEy|p$=jFrE<ZCl_p|BiFHA3-oxXYY&pSU+#xch9BbEF{8OYE{-TyU^9ln7~zYUv~1aHE>-E1&qeUI~j7#ianW%!!#WL=IU37BchI^%7}?dp%x{@9q&bojGjSDqHLjbf)lnrsSCtF;I`n6kXV`kiE*WC2Y?e(=P=c*4b!AhBXX(RskC$wVOf2&)<$<-Z6XBqoIe<altNQ4(N^tDbosd)5}53Hy`-y8hL*GxQ3o9h}V~I60}a@1qppx6mJ`lUhO0g%Se`THd<xqal$KtBl~nX<R)Z^u_fbVgr;cE|*1=t#6=+0MCzRU@2uP4v!#^srSXJ3dM?)f5<PIOJTXTK*jk}81k7JSUbYr@2MnvA}A@6s8@6>M-gL{qt}o5k7_*yUz{a75|8Y<+vxE|c=<Un&$|oRQ!a2H*9*bLSK0(}2~ep87*v$oE6>b^ajKF$FH$8?uxbSTBV<WuugCo^;8u1qmjsL3JU?Qh)iJghf8#;A;Df$*D$QYAm?zmut~5Z#a_55K^uqc5x8B>m@xh(nJhT7dXKXdz``wF^58mCq{p#e|b9*;`xBK~PdoTZ5EY4rtz93fh{m)<Ay?uLf@#X#NKfQDI;^gg%ci#GhE!~2WMJKeZCBEXna_6m^lbe?&*FT!RdQCU>#qG=Ni)eEHcfY-J_PObmkN2*8JbnHhrb2Mw?ERhhFEE{A_mlJN<@Cn0(|16LP#9jByzu<qtq;V{{hz<Hd*k`ZfBzTm#gCtmbc>jd`Qzh_JFlLb?A&I36Y$UdTJ({1XaCwe(+}Rc^ZwhDOYiS~`uW~V?@gY4@6J!(n_T&9@BCSy)vaGnUw>xr%8kjhf0*tt^e^n*xF|}ByA-{+o%EQ-GsG=id2@e<7rcF$k$?K?XZtU{!fH=1yr^{`GDiPb5&G*^gup8R-Ncmc*MDjC$Q)(gt%hVI6&!A=%vJ7NHl19LUg$6<km(7g7wrW1udvZ|Bt69+0r$j|vpN;>O@R+v?9#62fXNr2h**ujLS8_tjvo^Wk;H*jt7xJ3isTgsVP_mUjX7n@dGctKn;ZuZO#t`cOXJ8>h<Fhp86~_id^Pb{cU-4;aQk3O8-I!Q182FTExQzS*cEX|L195-_ODZ>&1UflJSnXfyw?XXlPl+4fkRPn;yJV(ibUfn`uoh^M{anTwFRv*JViPl+zWguN|6<JL6~b)Xf|%hg?$T&xZsVQ051Zu7To0nEEo?u>d@rM>w6!(q|P~%1K6K-uI_*SIw%>hUYq>#{k`8^q-RI+K;%z5uW21<dhOEmwYNy9h5e7;+Pm=K?v1y1Z@eEdcHr?)ScS>8Pp0R0rdQ9?<CUB#LvlnGBbhSS9C;2#a6YZP*o7rtmo^6-tPazDn2s{Jb(7&AA>+0xg4*CYFj$R?FYn&iVG7sul{X`0@s&TpH6w`tl)zG2!s7mk2uwgPFui7Ha{Eo8Ax(bvENDwySKGaDGiol?qUBYk-Wba+wE5xdcYgis{%?Ly{JEmZi=PUOQ4=iEdx)bBe@9GV+xzI&<n^DcBPzh@+4puo{iPymB*C>~rlo>=mI7xhUFxEohttSyXnrx>=R;oV<jTDvKmQ3<Ss0;{tg`jPCt7781W&et0Lq<k(duVNG!0>X;l_#x61#w9#+DJbEkg<{KD0-){9uD@UBO={o{xd63^MeOl4EA32BDi$B|uU|wFs=G(#{60Pk&~DrX|Y|0^|y3>Ow;3QYApbsscrqmX#gPjIb|tg3cC*AsLp)$Gs6shyskHvbrhu^H=rhl{o7OeEKxNrqmv>&e7*7!l||(kdmq#&p7ZP(k(<bqDFMgs>DbFJ%_(cO^hWjN+E>9i;Y3Iu^?S=7OI6!Rs^_3FoRZkWFGn5o=l>mnN4b}<p{x}sx>97BIbo0U?I?W@WQAi%P~RCmw9-K?lRL%-dYE6)dEwfo8%p~=rGj-<rE?)mTTIeeK-h63UW#6#Gu7IMyIzK5W10}Gc{DoOgvn|YJwYE$$Lv+u=h*kJ(~6Aoswfbx=6|JEOw+`<>yKkW)v`i&8`)Gvs#acxYDh|VXzs?cB%>AgF*mBhD{Z(E=nC~r8o!*x)|<;dYXAZ9K%#w26HEzUam&Mp{SaD%UWF&)OFxmcd5AOilR*oiv~4@u3NM-XxZCCQcywGg`5E8sfssC17(p;%wEHl$|n?|$WA+Jq=}@hE-z=WQv-_uyN*_-X%#eY)li6CH4`4Q&}x$YuF4j5^)gt_05KbB7XaSj9aIPQU$0Zi26S<|ab3Z^Fg2I@T->}>>EK9&_3vOP70Mgb=0e%IRvwQcJ$^POvUUI$=j}{v3ibrEkr9;xxY#d??>83hE<7>b<IyNf+hCR=g<V69Ct(kK0(J`gnYFu^eb3HoSmEwDcV|tJj(>6Yhj9l?`LZxr&$|ZAt?kqDg?ggJzBC^WHk0*OW$ETml+lm7bVKx3oEt|OBt<{ai;xyY5>(2C*3;MGU4Eca)4M$bmAE}ld)=1jEzt>WctpKC8E2`<SR%}9EvEW(PK1_cR;)KzUk00;57iDEmobl2DZ}up&~)DYlNC{D2JB)ko!!K$t66hrvr#7fOE(5J{G!h<`wK5IWzQ@mZD6Qs#dRdJDl}+$H9+Q>Yd3e;RL1ExCt$>>6?%!@-AylTH@UKXB_V$gsk<bSjr8&3ef7P#n@-zN7t3{7MRh);-Z~D7xW}QFe~F<!<bK1kEn5saan=>`ycYTVEm=oV$-+K*Hp{|dZyGHK;Ng=a*t2vcC~3+->eKqJ*)lN4DFy|Cpd`FLh2rAjk$Z=#+!T(R`pl*3AOSr?zd_wZJjFWX@}jT2;|jdu<yfaK5nEx`Wt*-6F<8*4f`$NDulOwOFkJ%U0#R0g^^67q^Pm{K8+Y%&Ha>Rb@=R?V{tadAnD213pjVBVJ#~87n9*;~(Q1)Z)05EC?JhXrgE>|20=)gRw;>JxzC<JH?oS%QJ&Ug7?p+$e{mg-Bh&MtFcUPo7YIl^!e9tVBiWY8})CcW^8e~`G8y0EJ@v({(izV)MTX2Su^V?u26CuBoelC1ds0%bbX_x+VpZ;=k{-U@uFn#&@^x}tiuKaBJ=Gn>FpG_|OYVz4DlXKUn&;Mrc<KIkoeurV3gZ2-8JGp&n@76PSUinyDB$&MWlRH;;c5l7&FkI9g^mcFD%#yW)t=*jjJQ#1jG`afT-mN$GKYVNdnak6w{|VlYXD?2Fb)7xg`|uU}B$)KVafCp-L^Jv0Q`!-V=F1H3=HTPz5~7|?Zqx&Fo!E|J;PLGF-{n1Pa}>w5-}5Ub%ZErtyL#AwqGWHCBCSBzLL!oI?5n7@R?<jpvD)42?D8yCfo&dkz{tLFFc|y_*jy(V>^Mds2>&IeS&2{n3%5_dpR+TP_BvH}r~-DTAE!^BK7IQ1dAJ&j5+UK4)p->>i0FcBuPlvG5^vYoY4egpc<Uk3I!eHV*tIMPY`29Mj^{H7FJ}Uh5SzxOlL$sZqFGd6x6>p_<rE~+0;9rMAt85TtISYaIpzm)J<(_EE*Ud8qjr2IPPb}Zh6!6uA(=CF!1CTo6}@ze{z)rDYaxw+B%R@{?3G%93(h{8)txd)rA^8ei2|rp0cQ*Bu_ex?xEP^)VYb_)(uqR}XU`JTeD_^eG5p{PwVh$zh2MUH*k-tTIeh#>^YUu*!8LB0v$1w=W9@5zF->6H`uv{kd&L}Bvo^SFOfQ2v^=tKxo-WlQSsJIkLNBU7bdsPRf^hPzM9N-)ueCl&V3cnL#cNj8nk^_gXg<4WWl1HBS&)%H-AUu3KrL<}fP7Ec3MUvmspz;njWu|cY>8p+e5E`u-<HfU%(AP~S&bP=xz379WngLK_EX$x6JTp~av>`7&c$LaV5Xa~;$z8pY=s{iV#wE7y26(Wi20&bYy&{ECspmbRm3sFZh;4G*IlmEiTkHCP6Y(aWYp$nRIKbZ0i9)`@(L&XJo9TMF;E~hT)tYIKhD??E>kVkX@2|ITuFQ&Mv{hX%M0+V_nKU83oj$pZ;@0>5AK3<dCB||yF4>D=4f^*y>{g8a*csqW+B;E>m~jzh^8?&yU=|bj26oP-CU(cJuei<P0tN0{#?bZt{vo)?5bhA&k>u{#`D|Z{g1+rKMucrlN;4;ti9*qEjW#bg$_5>DxzVqEhV1B;k5@yEZ=(eEPU{I>)X$z-D%l}fHc9eVpVV^lZ?b#wXpv#>+-jyyV>JSp(UaO6=0j=I2p?8EaGB-^9pujYhMgP;8l^lN}J*HnqI6SfpVdA8ctIM5PLGBsbW3wON(Grj+kM^!%LMMz2+Kq2rfZeV^JtsN6Urdz6UCT8nBD1PxP%|R8NpxDwLNR5DBs7Hv-@D3qcVw!$D(&Da<txNWQ8}rQ*v8glt>Z-V|6ts|<#5gA@Ho_G#4UZKFzTQgUmVNehNPV6Y)vqFR7#6=;M3Dxv2)v#b(Gy$p&3i}gId@yCSPiCH)PFK}X8J4$u!ZY<?mgxs#cRlsP->6XP?a+T1VYVEw`%t+F*Y9uo3j$o`a>sMD3iX#|nL4K)Ml$j5odG|!i#igcJw1{~S%A$9Ggm$z<btwnM1sv0~j9s&=ZUKQUk{(`jCK+Syg|==u^pSz5AZa1*nGqA#Z+7#Hg!W7)nAfR|p~b0({6V|}#~SnsWR%QiPj3xtl~5bb4O{6~E$I*u1i>2HE8MUIkeRw(E1v3zkR@8oUEhEdI#1%^ES04mL{FM4vJ{}SSPvo)%TF_b=7ME;Z+OdT;wHf}QLsFn%G8^_K`BlRzcy0DS%d=*g6Y)Ih7km|#wyIlkA+neTXrgvgD@&<AsiaC)<Rko?UQLf#plm0N4iMr<pC{Rvf?Zx5fD=(C889;OK7_`ma-#4IWjctQDYE6CBM$P1;%(Y=8&t6nQEa5Z^;Xs61kHT44G19V1y-NW>K6>2MjUOxSt|*RD(9qOd50DYHFmkylE^wa&ISq7tsRE>I?BH{Gc9V9TUl~R0x9NVi7ffWK+i2yIDA{ZX98YrRKFXg%b%QkxO+<izWwr5(@AIx#MU=C&-FReqO84V^VTl;$W1F?bxJt8Y@h&e&T9o89$F1w&2a3rWvogX(}s~y}@BK8ZXXMsaO)O6DviSgfZiOENE2WUlUvMIh6QGF*|SNglgt;af!L&^r|*zV6WvZ89b`?sVf?Tu5&VA(Df~Ym}gMy=rl-=+VNut-bxj`!nh&?@u0a>CoGx8BH9~{72sebrE$aBlgvV6SxYP{pQBXRrFPU0;M8B4uavU1(L<vL59Ow&^G7BprYD9b#^eh+tkOXh=E3v>Z51T22G!!!{bj1aDMN;Hq0pfz&&7*>TYZAt8dSD%k<3BqSJXc1%<?svC24e^3YXr)b>FRLw)Y|AZ3&waNjJFGviJmF132Igvx>9sL=@+{p&V6PtAT_QpJmsyS;q3|$NUge4E9Q6<l}XFi>b3fX)AfPY^HvS?h5*%kJkuS!BAZ{Zw}k7#%Kx3SsTHvV6Pni4~=3CAqZDR+nvx3S;pOEew4tF?iW<}sP<S1-~k$5rSpt`3X!XlOOeb-G4Q<+#3;rq^^po<A~^_M(V^63ml6uZ2rGzZqut<CJ+JZN$f5B?K)v`6h)+0dO*S?4YHlnyJu&X(h7abZQp#<wikCPrSTPs_2okoL!J<GR@E-wyb&_m-_x89tM<O%t_h)osu_v*D-%iM`t-L1PN;6Gn`8D8iWO^N29GO^<;arOZ`<ao*Tu6#vT=Z)hNw3Ddi#e;uaZq@kYUNVC1hU{#2Eu!3nS&H)W(K;vftP0?P=Go-`a#{;<a=;2U!4qj24w#MVcPb^M${BLMq~dq&|bhgL%emV>@WO^F@Nv%H~xzv2aG>Jsj3QTh=J&k3}p5%t(d5^NHYYRH6<gZ1vUGreLHhk<X3I@HPuH)H^%YrZi^unB(m~podR(W*uye5hI1$U+G(M#pr?@rI(!@hb-1ZKhm2j3B*>)=hrgj)Fo8!GP!(omT5%~}u~tInUthBJJOd$JP(kLMfdAQ9Y;z;d`5tdD<4sQ<&52wO(-TJ~#wHHF(T?us&=6~<rTZn*k%xq>F_YR-pO#vqEaREiQ)wrbVVx#pW+M&|T2Z6mOKOZr+iYq%=d_~*9#<ZTc(wX`gJduWwEJn*HJNaKfh(M@PAo0iyUd8*V_BLrq9q{Co>|qk7VRpnEA`Ggf-h2zAj4!8JzSgsp2ZZFNW8OBi~^;>&K1EbI8}Irjw}ajTo*3;7;|RlX>W8m$moHH$X0Kv_Do+Z@y)gPv+7T`px4sS(?(|5Xdb51eDr~f4qOssqO_s0l<h6=<gG1Hn@<XriFLjS&<c?dXCrmA93wif${NheKEZh2*nhS;yX|83zp-ZaOHbtv9?p$Vd+5yn`sm@&={LOb-1P4zCSRr1Jeq$FG=G(vPg43d0>Ac)0C6t=M6uEctcotIj~EcGC>Ie_O_;@RVS0YWULmnq(F^nAFUD)89k>w`6GQLcLe(z_G{yx^Oj1p_;Igq6V`6*sID~HMU9vGXGX}d!J{8*KiJnhLXCq+138u%kc_O+r!|Y$0oEYFqGZN!|X#1#x$D!QOn~2l_;YlX>hkyYFaxPW3LP!&xVk?q_xFAhglpZO%qsXeZz9M}R+Qoy1M6OnE;y3@98=6j!<zCAP!Q1Qhh9(XlnV1@#9-SC(2M~IOI)^MHf0YFB`1@i4_ZZ4{T5_PM9X6)eWLbjIB7sW@oQ>8F%T;v^aW-0Y9G5t4i+b*~ko)=dk;<h4Q7Cfzk@Z{-lCi$eEc>-3-_o=VOflk0IqSBFh38o1WHwdu7v<Y$90v@IFbj9raBGMDNws$c53}UOOiUvA5zD^yrd$$m+5wJv>22O`^QBF{5Q9CNeljcmyQ6c4^QtE6{32}u<eng_+!q+aLBwr8{iJO7(tFp@qf;2!v)JWm-O;R*jB<-I;#$LC$cAmA$()iG8*L!t(}P3P-t^$)!QAxd_`&wd%#kusgt<lJ4ZUI2V9IYScYEG#4H4`>z)s{vxqV?e?#7kqn3-Ukw%&+lBkoDzk?$b0=7prLv?dDXz^_}TL*6=C2ilW2CAf4m8#Y#DqDCP&oDB8Pv#g)wcLQ*Jf{mTHYKh&H7}o$3<&sc#XU~E+ioejIK$2#Fbgpz6^(#;x^~I|)<fN71bgozw<hFQ3rnix6Q<tYN>1JLmmzBJ(t#@;)PcT{jA{vM8t!{}p<!CF~Jv|XaNG|=hk$1%K?{?TLw07^d*fO}IwP5IzIqbKPgz-$u1i-xcof`<>gwwn9(B{T-zZ<~GAVELoN(f0kHAFH<#D7>iNMZ<<X$N!$d$jIht<AvLYvwyEBVJOh@CB$V14y;GT5&<7<Z)_zlBrJ;t<FSiKZ3}j@X;<|ZzSW59v(cHql$yWx#`K#p{cfN3IuPpoGFcIwC)Yr7CT0wb9tu~JT@t=3xw!e91_bAahwAZC068zwZ;V5*4Vef6_xuykvi1235u_Cg%aM0-7`kNWc3%W(^b}W*iu$P*1{*Yu?q1&5d2&kraceUfMk$}ziHX13`a8XD#lq_N`9fd4d=1e&Qt1SyNX*I-=;fNd{-(ZjM9!=s9I&(vH}`i#|mW<?<`yHF7DE1d{S{?r1cVru$XN0%tqEe9bLOOy7rmq+I_6{in&e<#;7nvTS@!XiMzbY2HfSe4boN2U})~57Pie#9EgG{C5Y1-&8p)n1f-RlB%)O`d0PxmVy*{mO~BLQSk4ErgT01jRTL~ZD&1m-JZ0??hz@4FslnHB?Py%GUe7EaUntf(ss%6?q2@Urwx!Kij*~2;j;Iy!;<4bWd<)B}8`yT0SMkMLEo1~be9s3b1Y{B-WW#4(t6Hm6{Tk>;GpL?)rk7F;Xgbi}xl9IR2XdmUCTAk&jcZ(JEe*r~*V46=RbhKw3o`(6=8FY{^L!Q;2wMXjl0+~hM}Sl@z!~B=1j~(5y-0!y!~lXo5hJh&F9(oh9Z=a~2V|<pOEL_*wkCIMu@PZDm5uo3w1`M%&9I}>xy@2Tucw)BILSR5M^TY%pH8OhcU>f9onIm^J+hG(A^#Uw3JH{z+qn*Om>IUPKa^0t&6~tl2cT$!l6VKEc3hvp(?Vs-chMmtcF}x+Bmq#P4w(fPDksYjGP<zr70QJY8Wr6~s|%UoLcK5o(M&pKG=XlmGcHHB#$933d0Zi2<mlL#lqwjHVgSqvafS_IYltxKfweA@7Fh5eCVk?ht)&ue?8JfY6aAvKAmbekP@7=9Sg&|K;14<pic94p60BizyK9xXMo@=v*4~_t(8sRn7i+$M226wjf<<sq@v4nEv`gV2eG~w=d9vYDmU?#wp?N@5OstCdsq(}U|4oS0#azGk?A}9>w)K2<>+>t&wKtm|KK$|NO`eS0E7X>d;2c2wQ<6?SymfEu`HdHsu53L2runDGyADr#P8$8*#@g3gm%eV^dWT1gZBI|wV;6*uJS4Z}k>9W#?KXQ;8aO;Ol0LATO{V&K`u6v{+}E35o?lEK$b&J=EB*a-#MqAewLD@!mHn^uu%G3|{3|{Cjx`{VT)ntRiF(iOj&X3PcYhm(*!xPpFyFiX-(iG@hX>m+z@Gf_LgC+Gd?Up<#GgxrpuYRV?CUv@UxfDAK-mcC(AqbuLw)_bJ)C_#zwAAbKi1ce2SewGjfwsgLu=Ep{ISBWkE^$@w?BW(FPY@Huy23wTxFqHJsm-s?HTFOzKN+e%wW$i^Nqer@Donq;i>6~b_{G^emSUD_`v+*tq11KQa!f05pn@&2k?>V2gi;M?OtoCrBY)a4>G+xbnM{Bk^I>7<nZLap7P;*4`Qp5=+?4uUxLE>lC-=(N#O&D3MYZNKS|5|Nm~A;(y~--$Md_#8?Thp8qMsqRe4;lA@ym<a+d3hS57CzxK!vW_e#w6UWwVBPS9~_*;y`TzdE_cqI72|OyaoMEIW?u@tW_PlMfH!g>S>vE1NgpZhrQ7^W$&AXCJcc4|}}f!Rf*L<ix}@VtPl=QL#g2tiuD{N2jN{vC~rEFLlp@F1x!@TPl=Gh+(2vIG#NX*V7IE$A9JP5bZul(YE+G+~{mEl^Y<{u(Z_Q+nau4Bu`GR-9#sf1f*_qE+@`Y=yyPahPP}EZ3(D39KSVgn^qtZ9>}pA54U@Me!kj`TZPRfj~T^Mp;Vafo`1PTGZu8>hTx|JcwcXNaBRveR;PyYM_xTVIz5qJ^vg>rj!E#_M`z#WgKOxl$R`>d8pC(*HQ)Jb^V&5+1KvxnurY!J!Y|)!esXJL{cgDWUU=>6=39T?dh~wt(^a|UjkWtgKUXe=-`pgz^&!c~wFgj(9kRoVHxNj)(d9z90h&OJZ_qqohY*_mfSXk8#;I;VIkn>UNUuFTy{Rtoqc0^Dn`Ht3O-a-OT0l2LJzT<}amvY5!bjgBTob$qykeaRD!KK?yIT+LlCP-b=K3|5m=pcpkyppOy-b7B4eA`Oa@xzU$~<xD1t`CKgQzyP9;}Dg{>(aqgkC3WMF{T!cZHx)%3?qn(BbXX=4Y2S)~{}?J@HOdPWdG-y!zqBv#ZVPKWsd|-Mn(%>*(!5e>z!{j+ho5y_udakNw5zabxYh@SUqKZhxW1{`A7j7sGeHBRy=ao!h$fJ!{{kmPLK^X8L(lYT^X5v9|u=qs!sBr&!-}2T>-oW@KmzAfz@ngx>o69)hEJ>)h7!_qQJ1-u!YM{CoG&QExr}Wb2=gIS?-WE`nlTriWi5nhwnhNVFJPjV;2UZalk%=740Tz#cn-U^dp*n{QvE<D<o4)!jN5KD_1(eToN%rcoUksL5`G55H}G^q_hEi}21>)X=}q-D+O{Dtvkk;*LN4Y;*nM)?1$laH%rzU*~SKMq28sRW!*VD5-8*L=i^)Z~o7d7uUak@#cE?>8Ih<cf)_a{o~WiKslH=(H2EW^2ev|h*7Y1c&tb?rDl*vfog<}=KVWpS`bFidST>`$nf%C!)G4=`Lo*TGvGAU?XgDi?;oFj9A5Zg<Jk=;i9<)?*J4#zRPG21FAFTtxP>0KWk#fOpONc{Vy)hwM!0~It#e;O=8~sR!Y5zy#VqOJVa4@idV0G&^!Ll~H~#O->go&ce1+HxO*ZxM6U3vN?>=jO_6Q;1#rfyqg{$F(dz)`Q4j(*5w6yu@{jEn25c0#TpP}+w&o6~{ZiW{xg!eyU5V)KU5j5zer4-Sf3sI3*dekDCQ8!?#Cvs_tSjOb09`J<u?28+#;pHC$ar{Q4iS*DmMA5MJu8axqcpAF(3RQ9m%@b=`2;AYfOd+Pr>fyJW7yl7{aD}@OOW7&CtsgERgd%p2m$nD{Ct8`0{*GRX$v|85x$|UeMemyzzHor!_G(5D^;3wFI4IL<Y$TWd9zJ~k1!9z6o(B?m_`bwfX;g0{N?s+n-+FWp5*yHDiipZ<HNT`F(_)X0I^55?W1Kq845mCJPQbeV=H|v@zT%_Jl2Ch)9~GCBZ?y3HUV;$Q$cWV<_#z&E=v}%OzmUxRfq47c<vn)_@fqMU)c*wFsx?^1%{|gx4@B@<tX0nVWgL(-%J8X^+3}B8y#6$}8l@NfD&mz8Y-X`oB1y!cK_o(eSr7zy^%0JDJ-`KNZx`=Y=$49;=7o3O-dw%7>s{am3rHAeNx5o;lhkn)R6w)FcYNcf7hoTy`!;MdyrO(Xv|pDhEh74C)WCQUEu;nd0e_(h#vAh_0xA-JD}@D2nmzLO<3$LWEQ@<|T~~2<wNy6cz2x=w^mNKE4@q}_nj<m0^F(4Pcp7t|U@!G6kzP;^v6q-%r{o^c8HXQxv}eFO%fGGYl?ajoO6+K~q7uZQQq^;vdi4n={dyc7yt*kF>$RuIa0HK$I(N#4D@2^sn!?6rQU(RyBx<UqEM;$Uh{``i;P9xxL7=k=Bl_@$xExPhz#wp$#~UzZd;_^4jd)ZV(I!-uRMdJB{SkVv)c8`sH_BZ?=L&&J{c>X&(PrJ(=DN#6XsXq-y;3wUwC@j2<?_R$lZfgCy>&}1b`yn3DnT7y)K*YvgGa}Y9G%V|8Js?ZzVo7}KdJ;lYmzVm(f7pBX{nBAfETcFkQojbWa^We5TH;@|Gu0;YcaJ)S@7aJ|B0soug|I6w0BmSjg@ZeIzv;;Unx$V8VIhw7gHQVgkacDMl5ns3A0dC{InI8af%8{KUbKN1RffiR<z}aOUs&_Vd$B9b@a%Q-0-Y73ZZxr`IZ)ZM8Oq5zzyP7oIuKRiury9;@g)OOe5^G#v6E2tTJ^o;d9VsFAjZR_X$MwIO1I1AVybB%5RBS0ZsI{N|Fr;?SNuWiYl({PZ337D{KM5it<K9G%&$pnQ1!Uk%%F-IW;`H;|(b_M^!$TuWx;RuU)+q7h5Q=kn9KAE0x}96<nC9x7ato6hxPf>@TzA*99L@m!i^+6|SIxg#J^gkCZ#wwOn3;0sE6<uXKZauwgxbGc%CJk&AIJowG(k*rizLY>8(;l3c_bUOG!E>Px*d0(B~Yz*ReBEU5Sa*rW=bNII7TZsMVsJRJjW;auoQ;8gx%mSPK9xtLnSF)$GV6y_GO0Dl@zSFSLQ@&Adn@x2na$i-zUCd;&=Y=aET!kuPj*7!)r#Sa?tH!=`$$bbhH3+a&xbkMOX%Siy)1UYzEYGE4r-Xt7(r%z%R8i7~gPeM9%bm4M4QQD#|0g(+k?f+HT3=Wq`&|!Q_Y~`up$iit-X+yLVA8l+XmR!l`DwwshpqaB0UC|~K+9q^|YLn4Mj$`Nht=3%l=sU>sdFx*D`VaD5%#Tm6Y^+^<ap`J!hpNol2?yFPO#FjrY94rvVmRWyL}*+nww5&LGP=03xT1@SwPP=*u?l$VwbSNHcqBzgbs<yp3k%p9T86tbQ!G{HXL@j6=!N2Z-2xUo2(_#=EY0YC-FRP4;>>M_7#_zulAqKJ-_dhyjO6aH%agARyKFaxmgS|i4YR;D1-$WR+!gG|f`?&iM%~Wg;A$SP^+(&Am;k}rboZ+7TYIx!ud}d2YZh6DZ@;#Wls(Q&<|dDhOH;$J3F&7GWsjU23E^Z?=3x9&Kt!lI{DX7Z1|G@K5Yej>ViO!n|8OZ)TEP5a&i9CjGY~->Xdl6Gt1u{?nLJ_Mc~A)8S1b0a?Pk5r`+o`7*4ocptz1S3Pmc6d`U88D=FKNRBFRryNZR(Il42+;2>|xO+wh%Ipx(Y8mD0>ke5&Lpsee_Xx^^md;dL86Sn?AiK9>CKASA4<nQerJwKbDCa+lIb(kNc?lQ_abyoyOkAZFwP_9TpeAV2XgrsI>oqk$xYk9|2EAF3E(yORT!B)a!@n@I$oE!3?3M8=;}I&tkwiG(kQ(51vdk)H^;BQ-NTF`k<hw_e;LD{B`LkZi#~{)L%c11J0X#jHiZP@zg7IB7}@tIQWMKy)4*85|u00h}B+kr0$rI5IJ+T>v}vX7Jc`XYeG{pW!MI5v+`|cszL+EhL+_47z3AWh5w}{%}_{TH9V}#ey|mJF3y`_B5nyHpeSOkKmm|v?G*&rPI`t*~e;7n}R~R6*Kbq$iIFv)u^P>OsXZ-l`U;knri2_m?~6tG4E8Ly2smiemlJX(bj{z+|+sV=G);nSDGJw&n$BphyC&C<?!Ln=B=;8^EbC1-J=FK*syW^Q!s9QeyO>BN3vdoM=~uTstMu1trN4*tQNtxnI>%HUb$iv@Lp9NR%-M$PWq{mPM5OeyNob(>NN;XTgY0HfX`5Hcwtek=7YaCTsuaCtbrNTxC#jzTTulCEfzRbR7<~RRD|v|A)YmL<p7rMQV)+y0o(1Rz*LA3YP9~fQ6{EY{(|OfkB6=M+ruMh!)~s`c#HI9H!n7&mb6d`LU~jpArRO#pi~9tb;9Q)w9Dg22e7cvS<;s*JvA#MY7MtoST2@MOKnw4jR0`K#f@%qob1NO2PrB;7Y($)7ct><NK(kO0WXSy7|zzIT(JOl*zJhKTh)yV*ZoQK{tEdKiwO>l!(aj)tA5O?I$>zhl(W2(IGS0j!dq_KuB)|H`FS$^ttpt;wR28|Ap>z+GCHRzw#eCwXn{ClzFblj?@7z_s6*r0$V7dWurc?=A6**+0ngaiaBX#Oiy<pHBnf#Fw3pCm$1G)93|O~I7^-zkQ|pUeXm@bavU?R^WmlCUhyYom0{N#2v}NA+ItCrHOY4RVPc(Ooy)j$D-*CLywq<x!T6b6r>-#f&k~V0U&F|i<_0(f*JMQy^dA3M!u8__YY{y=(pk`Z#2-g9^HeOrpZJZ|(oh8IMqEk1&P??)JR}wvl7*{tLRQ9ZGOqTQ-`k1n3wIQK81XS^b{5BJA?53GKZjYUIa<5P4CdUWIyw?WDM$yPeC&rlv1NFMWZh*wkF}%3Gx&Gham>2EE;1L@+#dqz+r7N53x5EdIH`cC(7q2!ydyL&>ZjkUu!ngVS&5iZ-jb|Tvi#{UOn%~VtN9O&&Tis=p(9P2>kn5TWlFenZfp*|!n$IOQUSSjnH%beQd9>mS<<njxz){wSZh~p(QhII@exL+FMh<pu2S`viPFgRW(~C=GBqtMN3oRcs7}Qw<s<oO<;&*ueFX2;(2AGEn-d)+OEEZ4k(8<ykh%MzJdpWz(nK|yCM%{6!WX|Rx0FX>Xd~OidAni3kP=G=|KE2ZX(?2)=_BV8DN2JGK@lRC=ACtCXU1Hm7g_yfk{qZh!oWA}Q47uvp(qtZore5<lpFRm6zE2(Wu<sKBgas(Oz4_q{Z@e%Lug-xb2^v6n`7;T;Jr>&zJYgR3btJc=&9^Ru7yq%b_DA9XPI^PFym;%`hU&G<r$CV+bR-BGHbe)R7Bfb1omyu|2@rbA)l9ii)(9Hol_X3Rb#!QEQ9sqw9m4aGXliiXG$pX+xpj$|sO_5vEZG?V3ktyIqU8i}hU&C*!Gsh`$6c}*A;z$Jg?M2j3Lt*L;p)}RuO2o(dT;adZ$KUO&V%Nyzk&Dn(<fUG-)esMIZW*No1DJWz(Wj%rvV_e4iQM;Z(<`OKQj7yZa9B*d}MSCZ4J5MZDB_P&W<Rodw^qF8Xwd2h>jVeX($pM0|`}Gudq1J-pONrnK24~zPJG95hu}*j)4}UyX>E|rkCb1AZxydm{6kt`r$_4qy0Uj4bOUh-v)VU=X9PjOy6xJ`BFQuDKOc034N!K8dC)!eY<F}Wx%6v8)6g&a3(o0ENUbGHC6L!%OJ&(R0Jl#`d#KvN%kICNvrQ|uHW}Yic6WkC2$XKezF=~`#QXEe(RexBFNC77MHJwm!632D^NEs(jc35$7GWgqV5=CeigX+=v2WyMQR~QdBLwC$rUlsLak6<@;iumrMV+@YEpAb`Hba$vy!Mn#9tm%)EKOTlRdnMm>F!Ucdzq}C%kwg{P>e_^=&S*$s3P2ekcJ;rQ@vZ6_qK%aY9x`QGJyhLf@{jGPNI^^!1GgztA<<$~QxgV^id)dldy%(YHWbwmYmAu;$AZBI9J$_nC!SrP`sjgeNwuLu^@>)-J3kZZ)!O+|zY@T{r*l9`D5$x5GPsCX1NtjeJ)m6znzsd}(9-2ku-?!`OU)w$!gYwx`fS%}0-#k1pD7XA+HBd7nhP0Bj&37Iyp4vIohEcf;q`+tRqAju7daZK^_Cq|;JsM;mCZQD&CP`u=Hwjm@(OfQ`Gc3;P`bNE<OENc$e3*$Bwt!0D=1S$u>R)UExL@A$1>*bTn;NP{Pd-9bW)ePj6T5Cp6G%em&Kcj>ZavN}5CtbPf`^taA!u3u(a1|+|4tlh^k%)Uok2ppX5e9IQ@ALg|y7WDvQ+t_S}7;tflizBCIc*ViE*$W=UrHUnA>={#>*Lk$Ex6osf3IU#qAdBPxzcw#}C-^r-OI$S|84?l%NPiY}SsAy7N`TS}_AM1qUBaMs7K>%HNz+<LPALZDO$R|mX9g|M4r^WJylUjnGK#{}o9Xe=p0fkA7%d<hE!oG=C>Q_G@YSJMfgoMZSfH5Alu0X=tE^79suBK16B%Eg4D!Dna>$aDAowz+?b=qT>Pl{n9r0SfH!^W_GCel<MsCtOG@6?noE$oY<Mpr<^@HPsV{c52PI)7v<D*lDa>K5W0OI9hYD(9aQs*9U>vrhHjR+3Z1Av?L(-x#AG}Wpz00rcXp?f`^5$ajzQ1a=`Oafby6eIKy%A3)sQLjFNIU`d(Ne>T$VUhE-xC^iG=rXEO0JSkbF`iQk|Gs><z&Ivjy>$N^P)b^K8CpXla0t`w0$Mf3?|-$n$YD)2zoBP0wJGCVqPHaLg?>AvR;KGgt1{Ur+rc))1idr5xoaRwo1-g2#?R2M#4PHtb5TAIw`BP|bsZx46n^%<N|Obf'


def _load_hierarchical_network_module():
    import types

    source = zlib.decompress(
        base64.b85decode(
            LINKNEX_HIERARCHICAL_NETWORK_B85.encode("ascii")
        )
    ).decode("utf-8")

    module = types.ModuleType("linknex_hierarchical_network")
    module.__file__ = "<embedded:hierarchical_network>"
    exec(
        compile(source, module.__file__, "exec"),
        module.__dict__,
    )
    return module


NETWORK_DISPLAY_METRICS = [
    # layer, key, display label
    ("Physical", "physical_instance_count", "Instance count"),
    ("Physical", "physical_simple_object_fraction", "Simple-object fraction"),
    ("Physical", "physical_branched_object_fraction", "Branched-object fraction"),
    ("Physical", "physical_cyclic_object_fraction", "Cyclic-object fraction"),
    ("Physical", "physical_modularity", "Modularity"),

    ("Whole Physical", "whole_physical_density", "Density"),
    ("Whole Physical", "whole_physical_average_clustering", "Clustering"),
    ("Whole Physical", "whole_physical_transitivity", "Transitivity"),
    ("Whole Physical", "whole_physical_global_efficiency", "Global efficiency"),
    ("Whole Physical", "whole_physical_local_efficiency", "Local efficiency"),
    ("Whole Physical", "whole_physical_modularity", "Modularity"),
    ("Whole Physical", "whole_physical_largest_component_fraction", "LCC fraction"),

    ("Social", "social_density", "Density"),
    ("Social", "social_clustering", "Clustering"),
    ("Social", "social_transitivity", "Transitivity"),
    ("Social", "social_global_efficiency", "Global efficiency"),
    ("Social", "social_local_efficiency", "Local efficiency"),
    ("Social", "social_modularity", "Modularity"),
    ("Social", "social_degree_gini", "Degree Gini"),

    ("Cross-scale", "cross_contact_topology_junction_enrichment", "Junction enrichment"),
    ("Cross-scale", "cross_contact_topology_terminal_enrichment", "Terminal enrichment"),
    ("Cross-scale", "cross_contact_topology_shaft_enrichment", "Shaft enrichment"),
    ("Cross-scale", "cross_contact_physical_coverage_fraction_mean", "Physical coverage fraction"),

    ("Hierarchy", "hierarchy_external_connectivity_fraction_median", "External connectivity fraction"),
    ("Hierarchy", "hierarchy_coupling_relative_gain_auc", "Coupling relative gain AUC"),
]



class NetworkMetricsModel(QAbstractTableModel):
    """Read-only representative network metrics table.

    This table is deliberately display-only. Selecting/clicking a metric does
    not alter the 3D network viewer and does not highlight nodes or edges.
    """

    def __init__(self, rows=None, parent=None):
        super().__init__(parent)
        self.rows = list(rows or [])

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = list(rows or [])
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        layer, label, value = self.rows[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return layer
            if index.column() == 1:
                return label
            if index.column() == 2:
                return value

        if role == Qt.TextAlignmentRole:
            if index.column() in (0, 2):
                return int(Qt.AlignCenter)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            headers = ("Layer", "Metric", "Value")
            return headers[section] if section in (0, 1, 2) else None
        return None


class HierarchicalNetworkWorker(QObject):
    progress = Signal(str, int)
    fileFinished = Signal(str, object)
    finished = Signal(object)
    stopped = Signal(str)
    failed = Signal(str)

    def __init__(self, processed_dir, stems, parameters, overwrite_existing=False):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.stems = list(stems)
        self.parameters = dict(parameters)
        self.overwrite_existing = bool(overwrite_existing)

    def _check_stop(self):
        if QThread.currentThread().isInterruptionRequested():
            raise InterruptedError("Network analysis stopped by user.")

    @staticmethod
    def _existing_complete(folder, stem):
        required = [
            folder / f"{stem}_hierarchical_network.pkl",
            folder / f"{stem}_hierarchical_network_metrics.csv",
            folder / f"{stem}_hierarchical_network_metrics_FULL.csv",
            folder / f"{stem}_instance_properties.csv",
        ]
        return all(path.exists() for path in required)

    def _write_folder_summaries(self):
        import pandas as pd

        compact_paths = sorted(
            p for p in self.processed_dir.glob("*_hierarchical_network_metrics.csv")
            if not p.name.startswith("all_images_")
        )
        compact_rows = []
        for path in compact_paths:
            try:
                compact_rows.append(pd.read_csv(path).iloc[0].to_dict())
            except Exception:
                pass
        if compact_rows:
            pd.DataFrame(compact_rows).to_csv(
                self.processed_dir / "all_images_hierarchical_network_metrics.csv",
                index=False,
            )

        full_paths = sorted(
            p for p in self.processed_dir.glob("*_hierarchical_network_metrics_FULL.csv")
            if not p.name.startswith("all_images_")
        )
        full_rows = []
        for path in full_paths:
            try:
                full_rows.append(pd.read_csv(path).iloc[0].to_dict())
            except Exception:
                pass
        if full_rows:
            pd.DataFrame(full_rows).to_csv(
                self.processed_dir / "all_images_hierarchical_network_metrics_FULL.csv",
                index=False,
            )

    def run(self):
        try:
            import pandas as pd

            net = _load_hierarchical_network_module()
            results = []
            total = max(1, len(self.stems))

            for index, stem in enumerate(self.stems, start=1):
                self._check_stop()
                start_pct = int(round(100 * (index - 1) / total))
                self.progress.emit(
                    f"[{index}/{total}] {stem} — preparing hierarchical network",
                    start_pct,
                )

                if (
                    not self.overwrite_existing
                    and self._existing_complete(self.processed_dir, stem)
                ):
                    metrics_path = self.processed_dir / f"{stem}_hierarchical_network_metrics.csv"
                    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
                    results.append(metrics)
                    self.fileFinished.emit(stem, metrics)
                    self.progress.emit(
                        f"[{index}/{total}] {stem} — reused existing result",
                        int(round(100 * index / total)),
                    )
                    continue

                self.progress.emit(
                    f"[{index}/{total}] {stem} — Physical + Whole Physical + Social + Cross-scale",
                    start_pct,
                )

                metrics = net.process_one_image(
                    stem=stem,
                    input_path=self.processed_dir,
                    output_path=self.processed_dir,
                    coordinate_scale_xyz=self.parameters["coordinate_scale_xyz"],
                    skeleton_neighbor_radius=1.8,
                    social_distance_threshold=self.parameters["social_distance_threshold"],
                    whole_physical_distance_threshold=self.parameters["whole_physical_distance_threshold"],
                    contact_region_radius=self.parameters["contact_region_radius"],
                    omega_values=(0.01, 0.03, 0.1, 0.3, 1.0),
                    maximum_efficiency_nodes=250,
                    maximum_resistance_nodes=400,
                    maximum_transport_nodes=300,
                    minimum_cross_nodes=6,
                    low_local_efficiency_threshold=0.25,
                    hub_fraction=0.10,
                    size_tail_fraction=0.25,
                    whole_physical_centrality_sample_nodes=300,
                    whole_physical_spectral_entropy_max_nodes=800,
                    whole_physical_weighted_efficiency_max_nodes=800,
                    workers=-1,
                    random_seed=42,
                )
                results.append(metrics)
                self.fileFinished.emit(stem, metrics)
                self.progress.emit(
                    f"[{index}/{total}] {stem} — completed",
                    int(round(100 * index / total)),
                )

            self._check_stop()
            self._write_folder_summaries()
            self.finished.emit(results)

        except InterruptedError:
            self.stopped.emit("NETWORK ANALYSIS STOPPED BY USER")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class HierarchicalNetworkViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pickle = None
        self.current_layer = "whole_physical"
        self.point_size = 9
        self.edge_width = 3
        self.plotter = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        controls = QHBoxLayout()
        self.physical_btn = QPushButton("Physical")
        self.whole_btn = QPushButton("Whole Physical")
        self.social_btn = QPushButton("Social")
        self.physical_btn.clicked.connect(lambda: self.set_layer("physical"))
        self.whole_btn.clicked.connect(lambda: self.set_layer("whole_physical"))
        self.social_btn.clicked.connect(lambda: self.set_layer("social"))
        controls.addWidget(self.physical_btn)
        controls.addWidget(self.whole_btn)
        controls.addWidget(self.social_btn)
        controls.addSpacing(8)
        controls.addWidget(QLabel("Node size"))
        self.point_spin = QSpinBox()
        self.point_spin.setRange(2, 30)
        self.point_spin.setValue(self.point_size)
        self.point_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.point_spin.setAlignment(Qt.AlignCenter)
        self.point_spin.setFixedWidth(62)
        self.point_spin.setStyleSheet(
            f"""
            QSpinBox {{
                background: {WHITE};
                color: {NAVY};
                border: 1px solid {BORDER};
                border-radius: 5px;
                padding: 4px 6px;
            }}
            """
        )
        self.point_spin.valueChanged.connect(self._style_changed)
        controls.addWidget(self.point_spin)

        controls.addWidget(QLabel("Edge"))
        self.edge_spin = QSpinBox()
        self.edge_spin.setRange(1, 12)
        self.edge_spin.setValue(self.edge_width)
        self.edge_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.edge_spin.setAlignment(Qt.AlignCenter)
        self.edge_spin.setFixedWidth(62)
        self.edge_spin.setStyleSheet(
            f"""
            QSpinBox {{
                background: {WHITE};
                color: {NAVY};
                border: 1px solid {BORDER};
                border-radius: 5px;
                padding: 4px 6px;
            }}
            """
        )
        self.edge_spin.valueChanged.connect(self._style_changed)
        controls.addWidget(self.edge_spin)
        controls.addStretch()
        root.addLayout(controls)

        self.frame = QFrame()
        self.frame.setObjectName("InteractionFrame")
        self.frame_layout = QVBoxLayout(self.frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(0)
        root.addWidget(self.frame, 1)

        self.info = QLabel("Run or select an existing network result.")
        self.info.setObjectName("Muted")
        self.info.setWordWrap(True)
        root.addWidget(self.info)

        self._init_plotter()
        self._refresh_buttons()

    def _init_plotter(self):
        if pv is None or QtInteractor is None:
            label = QLabel("3D network viewer unavailable. PyVista / pyvistaqt is required.")
            label.setAlignment(Qt.AlignCenter)
            self.frame_layout.addWidget(label)
            return
        try:
            self.plotter = QtInteractor(self.frame)
            self.plotter.set_background("white")
            self.frame_layout.addWidget(self.plotter.interactor)
        except Exception as exc:
            self.plotter = None
            label = QLabel(f"3D network viewer could not start:\n{exc}")
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            self.frame_layout.addWidget(label)

    def _refresh_buttons(self):
        for button, layer in (
            (self.physical_btn, "physical"),
            (self.whole_btn, "whole_physical"),
            (self.social_btn, "social"),
        ):
            active = self.current_layer == layer
            button.setStyleSheet(
                f"background: {DEEP_BLUE}; color: {WHITE}; "
                f"border: 2px solid {DEEP_BLUE};"
                if active
                else f"background: {WHITE}; color: {DEEP_BLUE}; "
                     f"border: 2px solid {BORDER};"
            )

    def _style_changed(self):
        self.point_size = int(self.point_spin.value())
        self.edge_width = int(self.edge_spin.value())
        if self.current_pickle is not None:
            self.render_current(preserve_camera=True)

    def set_layer(self, layer):
        self.current_layer = str(layer)
        self._refresh_buttons()
        if self.current_pickle is not None:
            self.render_current(preserve_camera=True)

    def clear(self, message="Run or select an existing network result."):
        self.current_pickle = None
        self.info.setText(message)
        if self.plotter is not None:
            try:
                self.plotter.clear()
                self.plotter.render()
            except Exception:
                pass

    def load_pickle(self, path):
        path = Path(path)
        if not path.exists():
            self.clear("Hierarchical-network PKL is not available yet.")
            return
        self.current_pickle = path
        self.render_current(preserve_camera=False)

    @staticmethod
    def _build_lines(points, id_values, edges, source_col, target_col):
        id_to_pos = {int(value): idx for idx, value in enumerate(id_values)}
        flat = []
        kept = 0
        for row in edges.itertuples(index=False):
            source = int(getattr(row, source_col))
            target = int(getattr(row, target_col))
            if source not in id_to_pos or target not in id_to_pos:
                continue
            flat.extend([2, id_to_pos[source], id_to_pos[target]])
            kept += 1
        poly = pv.PolyData(np.asarray(points, dtype=np.float32))
        if flat:
            poly.lines = np.asarray(flat, dtype=np.int64)
        return poly, kept

    def render_current(self, preserve_camera=True):
        if self.plotter is None or self.current_pickle is None:
            return
        import pickle

        camera = None
        if preserve_camera:
            try:
                camera = self.plotter.camera_position
            except Exception:
                camera = None

        try:
            with open(self.current_pickle, "rb") as handle:
                bundle = pickle.load(handle)

            layer = self.current_layer
            if layer in ("physical", "whole_physical"):
                whole = bundle["whole_cell_physical_proximity_graph"]
                nodes = whole["nodes"]
                points = nodes[[
                    "whole_physical_centroid_x",
                    "whole_physical_centroid_y",
                    "whole_physical_centroid_z",
                ]].to_numpy(dtype=float)
                ids = nodes["physical_edge_id"].to_numpy(dtype=int)
                if layer == "physical":
                    edges = bundle["physical_graph"]["edges"]
                    source_col = "source_physical_edge_id"
                    target_col = "target_physical_edge_id"
                    line_color = "#003B70"
                    node_color = "#006A80"
                    label = "Physical true topology"
                else:
                    edges = whole["edges"]
                    source_col = "source_physical_edge_id"
                    target_col = "target_physical_edge_id"
                    line_color = "#006A80"
                    node_color = "#003B70"
                    label = "Whole-cell physical proximity"
            else:
                social = bundle["social_graph"]
                nodes = social["nodes"]
                points = nodes[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
                ids = nodes["social_instance_id"].to_numpy(dtype=int)
                edges = social["edges"]
                source_col = "source_social_id"
                target_col = "target_social_id"
                line_color = "#008A9B"
                node_color = "#003B70"
                label = "Social spatial network"

            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            ids = ids[finite]

            poly, kept_edges = self._build_lines(
                points, ids, edges, source_col, target_col
            )
            self.plotter.clear()
            if kept_edges > 0:
                self.plotter.add_mesh(
                    poly,
                    color=line_color,
                    line_width=float(self.edge_width),
                    render_lines_as_tubes=True,
                    show_scalar_bar=False,
                )
            point_poly = pv.PolyData(np.asarray(points, dtype=np.float32))
            self.plotter.add_points(
                point_poly,
                color=node_color,
                point_size=float(self.point_size),
                render_points_as_spheres=True,
            )
            if camera is not None:
                try:
                    self.plotter.camera_position = camera
                except Exception:
                    self.plotter.reset_camera()
            else:
                self.plotter.reset_camera()
            self.plotter.render()
            self.info.setText(
                f"{label} | nodes={len(points):,} | edges={kept_edges:,}"
            )
        except Exception as exc:
            self.info.setText(f"Failed to render network: {type(exc).__name__}: {exc}")


class OrganelleNetworkAnalysisWidget(QWidget):
    backRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processed_dir = None
        self.image_stems = []
        self.worker_thread = None
        self.worker = None
        self.last_metrics = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # ----------------------------------------------------------
        # Left workflow panel
        # ----------------------------------------------------------
        left = QFrame()
        left.setObjectName("SidePanel")
        left.setFixedWidth(340)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 12, 14, 14)
        ll.setSpacing(8)

        logo = QLabel()
        logo.setObjectName("LogoImage")
        logo.setFixedSize(230, 62)
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_pix = QPixmap()
        for candidate in LOGO_FULL_CANDIDATES:
            if candidate.exists():
                logo_pix = QPixmap(str(candidate))
                if not logo_pix.isNull():
                    break
        if not logo_pix.isNull():
            logo.setPixmap(
                logo_pix.scaled(220, 55, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            logo.setText("LinkODT")
        ll.addWidget(logo)

        home_btn = QPushButton("← Home")
        home_btn.clicked.connect(self.backRequested.emit)
        ll.addWidget(home_btn)

        title = QLabel("Organelle network analysis")
        title.setObjectName("SectionTitle")
        ll.addWidget(title)

        desc = QLabel(
            "Instance outputs → Physical topology → Whole-cell Physical proximity → "
            "Social spatial network → Cross-scale coupling"
        )
        desc.setObjectName("Metric")
        desc.setWordWrap(True)
        ll.addWidget(desc)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("processed_output folder")
        self.folder_btn = QPushButton("Folder")
        self.folder_btn.clicked.connect(self.browse_processed_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.folder_btn)
        ll.addLayout(folder_row)

        self.scan_btn = QPushButton("Scan Instance Results")
        self.scan_btn.clicked.connect(self.scan_folder)
        ll.addWidget(self.scan_btn)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.currentRowChanged.connect(self._selection_changed)
        ll.addWidget(self.image_list, 1)

        parameter_title = QLabel("Network parameters")
        parameter_title.setObjectName("SectionTitle")
        parameter_title.setStyleSheet(
            f"""
            QLabel {{
                background: {WHITE};
                color: {NAVY};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 6px 8px;
                font-weight: 600;
            }}
            """
        )
        ll.addWidget(parameter_title)

        self.social_threshold = self._double_spin(0.1, 1000.0, 5.0, 0.5, 2)
        self.whole_threshold = self._double_spin(0.1, 1000.0, 5.0, 0.5, 2)
        self.contact_radius = self._double_spin(0.1, 1000.0, 3.0, 0.5, 2)
        self.scale_x = self._double_spin(0.0001, 1000.0, 1.0, 0.1, 4)
        self.scale_y = self._double_spin(0.0001, 1000.0, 1.0, 0.1, 4)
        self.scale_z = self._double_spin(0.0001, 1000.0, 1.0, 0.1, 4)

        ll.addLayout(self._param_row("Social distance", self.social_threshold))
        ll.addLayout(self._param_row("Whole physical", self.whole_threshold))
        ll.addLayout(self._param_row("Contact radius", self.contact_radius))

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("XYZ scale"))
        scale_row.addWidget(self.scale_x)
        scale_row.addWidget(self.scale_y)
        scale_row.addWidget(self.scale_z)
        ll.addLayout(scale_row)

        self.overwrite_cb = QCheckBox("Overwrite existing network results")
        self.overwrite_cb.setChecked(False)
        ll.addWidget(self.overwrite_cb)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run selected")
        self.run_all_btn = QPushButton("Run all")
        self.run_btn.clicked.connect(self.run_selected)
        self.run_all_btn.clicked.connect(self.run_all)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.run_all_btn)
        ll.addLayout(run_row)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_analysis)
        ll.addWidget(self.stop_btn)

        self.open_output_btn = QPushButton("Open processed_output")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        ll.addWidget(self.open_output_btn)

        root.addWidget(left)

        # ----------------------------------------------------------
        # Right results panel
        # ----------------------------------------------------------
        right = QFrame()
        right.setObjectName("CanvasFrame")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(8)

        header = QLabel("Hierarchical Organelle Network")
        header.setObjectName("PageTitle")
        rl.addWidget(header)

        subtitle = QLabel(
            "Physical = true within-object branch topology · Whole Physical = branch-proximity network · "
            "Social = object-level spatial organization · Cross = physical↔social coupling"
        )
        subtitle.setObjectName("Metric")
        subtitle.setWordWrap(True)
        rl.addWidget(subtitle)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        rl.addWidget(self.progress)

        self.status_label = QLabel("Select a processed_output folder from Instance segmentation.")
        self.status_label.setWordWrap(True)
        rl.addWidget(self.status_label)

        self.viewer = HierarchicalNetworkViewerWidget()
        rl.addWidget(self.viewer, 1)

        metrics_title = QLabel("Network Metrics Table")
        metrics_title.setObjectName("SectionTitle")
        rl.addWidget(metrics_title)

        self.metrics_model = NetworkMetricsModel([], self)
        self.metrics_table = QTableView()
        self.metrics_table.setModel(self.metrics_model)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Network metrics are image/network-level summaries, not selectable
        # objects. Keep the table display-only and completely independent of
        # the 3D network viewer.
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setFocusPolicy(Qt.NoFocus)
        self.metrics_table.setMinimumHeight(235)
        self.metrics_table.setMaximumHeight(285)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.metrics_table.setStyleSheet(
            f"""
            QTableView {{
                background: {WHITE};
                color: {NAVY};
                gridline-color: {BORDER};
                border: 1px solid {BORDER};
            }}
            QTableView::item {{
                background: {WHITE};
                color: {NAVY};
                padding: 4px;
            }}
            QHeaderView::section {{
                background: #EEF3F8;
                color: {NAVY};
                border: 1px solid {BORDER};
                padding: 5px;
                font-weight: 700;
            }}
            """
        )
        rl.addWidget(self.metrics_table)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVisible(False)

        root.addWidget(right, 1)
        self._update_controls()

    @staticmethod
    def _double_spin(minimum, maximum, value, step, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spin.setAlignment(Qt.AlignCenter)
        spin.setFixedWidth(78)
        spin.setStyleSheet(
            f"""
            QDoubleSpinBox {{
                background: {WHITE};
                color: {NAVY};
                border: 1px solid {BORDER};
                border-radius: 5px;
                padding: 4px 6px;
                selection-background-color: #DCEAF7;
                selection-color: {NAVY};
            }}
            QDoubleSpinBox:focus {{
                border: 1.5px solid {US_BLUE};
            }}
            """
        )
        return spin

    @staticmethod
    def _param_row(label, widget):
        row = QHBoxLayout()
        row.addWidget(QLabel(label), 1)
        row.addWidget(widget)
        return row

    def set_processed_dir(self, path, auto_scan=True):
        if path is None:
            return
        path = Path(path)
        if path.name.lower() != "processed_output" and (path / "processed_output").is_dir():
            path = path / "processed_output"
        self.processed_dir = path
        self.folder_edit.setText(str(path))
        if auto_scan:
            self.scan_folder()

    def browse_processed_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select processed_output folder",
            str(self.processed_dir or ""),
        )
        if folder:
            self.set_processed_dir(folder, auto_scan=True)

    @staticmethod
    def _valid_stems(folder):
        folder = Path(folder)
        suffix = "_pcd_instance_social.txt"
        stems = []
        for pcd in folder.glob(f"*{suffix}"):
            stem = pcd.name[:-len(suffix)]
            required = [
                folder / f"{stem}_skeleton_edges.txt",
                folder / f"{stem}_skeleton_instance_social.txt",
                folder / f"{stem}_pcd_instance_social.txt",
            ]
            if all(path.exists() for path in required):
                stems.append(stem)
        return sorted(set(stems), key=natural_sort_key)

    def scan_folder(self):
        text = self.folder_edit.text().strip()
        if not text:
            self.status_label.setText("Select a processed_output folder first.")
            return
        folder = Path(text)
        if folder.name.lower() != "processed_output" and (folder / "processed_output").is_dir():
            folder = folder / "processed_output"
            self.folder_edit.setText(str(folder))
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Network analysis", "The selected folder does not exist.")
            return

        self.processed_dir = folder
        self.image_stems = self._valid_stems(folder)
        self.image_list.clear()
        for stem in self.image_stems:
            done = (folder / f"{stem}_hierarchical_network_metrics.csv").exists()
            item = QListWidgetItem(("✓  " if done else "   ") + stem)
            item.setData(Qt.UserRole, stem)
            self.image_list.addItem(item)
        if self.image_stems:
            self.image_list.setCurrentRow(0)
        else:
            self.viewer.clear("No complete Instance segmentation network inputs were found.")
            self.metrics_model.set_rows([])
        self.status_label.setText(
            f"Found {len(self.image_stems)} image(s) with skeleton_edges + skeleton_instance_social + pcd_instance_social."
        )
        self._update_controls()

    def _selected_stems(self):
        stems = []
        for item in self.image_list.selectedItems():
            stem = item.data(Qt.UserRole)
            if stem:
                stems.append(str(stem))
        return stems

    def _current_stem(self):
        item = self.image_list.currentItem()
        if item is not None:
            stem = item.data(Qt.UserRole)
            if stem:
                return str(stem)
        selected = self._selected_stems()
        return selected[0] if selected else None

    def _selection_changed(self, _row):
        self.preview_current_result()
        self._update_controls()

    def _parameters(self):
        return {
            "social_distance_threshold": float(self.social_threshold.value()),
            "whole_physical_distance_threshold": float(self.whole_threshold.value()),
            "contact_region_radius": float(self.contact_radius.value()),
            "coordinate_scale_xyz": (
                float(self.scale_x.value()),
                float(self.scale_y.value()),
                float(self.scale_z.value()),
            ),
        }

    def run_selected(self):
        stems = self._selected_stems()
        if not stems:
            current = self._current_stem()
            stems = [current] if current else []
        self._start_analysis(stems)

    def run_all(self):
        self._start_analysis(list(self.image_stems))

    def _start_analysis(self, stems):
        if not stems:
            QMessageBox.warning(self, "Network analysis", "Select one or more valid images first.")
            return
        if self.processed_dir is None:
            QMessageBox.warning(self, "Network analysis", "Select processed_output first.")
            return
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return

        self.progress.setValue(0)
        self.status_label.setText(f"Starting hierarchical network analysis for {len(stems)} image(s)...")
        self.log.appendPlainText("\n" + "=" * 70)
        self.log.appendPlainText(self.status_label.text())

        self.worker_thread = QThread(self)
        self.worker = HierarchicalNetworkWorker(
            processed_dir=self.processed_dir,
            stems=stems,
            parameters=self._parameters(),
            overwrite_existing=self.overwrite_cb.isChecked(),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._analysis_progress)
        self.worker.fileFinished.connect(self._analysis_file_finished)
        self.worker.finished.connect(self._analysis_finished)
        self.worker.stopped.connect(self._analysis_stopped)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.stopped.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._thread_closed)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.stop_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.worker_thread.start()

    def stop_analysis(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping after the current image...")
            self.worker_thread.requestInterruption()

    def _analysis_progress(self, text, value):
        self.progress.setValue(max(0, min(100, int(value))))
        self.status_label.setText(str(text))
        self.log.appendPlainText(str(text))

    def _analysis_file_finished(self, stem, metrics):
        self.last_metrics[str(stem)] = dict(metrics)
        if str(stem) == self._current_stem():
            self.preview_current_result()

    def _analysis_finished(self, results):
        self.progress.setValue(100)
        self.stop_btn.setEnabled(False)
        self.scan_folder()
        self.status_label.setText(
            f"Completed {len(results)} image(s). Network CSV/PKL results saved in processed_output."
        )
        self._update_controls()

    def _analysis_stopped(self, message):
        self.status_label.setText(str(message))
        self.stop_btn.setEnabled(False)
        self._update_controls()

    def _analysis_failed(self, message):
        self.status_label.setText("Network analysis failed.")
        self.stop_btn.setEnabled(False)
        self.log.appendPlainText(str(message))
        self._update_controls()
        QMessageBox.critical(self, "Network analysis failed", str(message))

    def _thread_closed(self):
        self.worker_thread = None
        self.worker = None
        self._update_controls()

    @staticmethod
    def _format_metric(value):
        try:
            value = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(value):
            return "—"
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.3e}"
        return f"{value:.4g}"

    def preview_current_result(self):
        if self.processed_dir is None:
            return
        stem = self._current_stem()
        if not stem:
            return

        metrics_path = self.processed_dir / f"{stem}_hierarchical_network_metrics.csv"
        pkl_path = self.processed_dir / f"{stem}_hierarchical_network.pkl"

        if metrics_path.exists():
            try:
                import pandas as pd
                row = pd.read_csv(metrics_path).iloc[0].to_dict()
                metric_rows = []
                for layer, key, label in NETWORK_DISPLAY_METRICS:
                    if key in row:
                        metric_rows.append(
                            (
                                layer,
                                label,
                                self._format_metric(row[key]),
                            )
                        )
                self.metrics_model.set_rows(metric_rows)
            except Exception as exc:
                self.metrics_model.set_rows([("Result", "Load error", str(exc))])
        else:
            self.metrics_model.set_rows([])

        if pkl_path.exists():
            self.viewer.load_pickle(pkl_path)
        else:
            self.viewer.clear("This image has not been analysed yet.")

    def open_output_folder(self):
        if self.processed_dir is None or not self.processed_dir.exists():
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self.processed_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.processed_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.processed_dir)])
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", str(exc))

    def _update_controls(self):
        running = self.worker_thread is not None and self.worker_thread.isRunning()
        has_images = bool(self.image_stems)
        self.scan_btn.setEnabled(not running)
        self.folder_btn.setEnabled(not running)
        self.run_btn.setEnabled(has_images and not running)
        self.run_all_btn.setEnabled(has_images and not running)
        self.stop_btn.setEnabled(running)
        self.open_output_btn.setEnabled(self.processed_dir is not None)


# =============================================================================
# Home page
# =============================================================================

class HomeWidget(QWidget):
    preprocessingRequested = Signal()
    semanticRequested = Signal()
    instanceRequested = Signal()
    networkRequested = Signal()
    dynamicsRequested = Signal()
    visualizationRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('HomePage')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('QWidget#HomePage { background: #F5F5F5; }')

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        logo = QLabel()
        logo.setObjectName("HomeLogoImage")
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo.setFixedHeight(88)

        home_logo_pix = QPixmap()
        for _candidate in LOGO_FULL_CANDIDATES:
            if _candidate.exists():
                home_logo_pix = QPixmap(
                    str(_candidate)
                )
                if not home_logo_pix.isNull():
                    break

        if not home_logo_pix.isNull():
            logo.setPixmap(
                home_logo_pix.scaled(
                    290,
                    76,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            logo.setText(
                "LinkODT"
            )

        root.addWidget(
            logo
        )

        subtitle = QLabel("Organelle imaging, analysis and visualization")
        subtitle.setObjectName("HomeSubtitle")
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle.setFixedHeight(26)
        root.addWidget(subtitle)


        root.addSpacing(16)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.preprocessing_btn = self._module_button(
            "1", "Image preprocessing", "Review images · Cell masks · Channels"
        )
        self.semantic_btn = self._module_button(
            "2", "Semantic segmentation", "Predict · Edit masks · Train models"
        )
        self.instance_btn = self._module_button(
            "3", "Instance segmentation", "Separate objects · Measure morphology"
        )
        self.network_btn = self._module_button(
            "4", "Network analysis", "Build graphs · Inspect metrics"
        )
        self.dynamics_btn = self._module_button(
            "5", "Punctate organelle dynamics", "Review masks · Track objects"
        )
        self.visualization_btn = self._module_button(
            "6", "Visualization", "3D scenes · Camera paths · Movies"
        )

        self.preprocessing_btn.clicked.connect(self.preprocessingRequested.emit)
        self.semantic_btn.clicked.connect(self.semanticRequested.emit)
        self.instance_btn.clicked.connect(self.instanceRequested.emit)
        self.network_btn.clicked.connect(self.networkRequested.emit)
        self.dynamics_btn.clicked.connect(self.dynamicsRequested.emit)
        self.visualization_btn.clicked.connect(self.visualizationRequested.emit)

        row1.addWidget(self.preprocessing_btn)
        row1.addWidget(self.semantic_btn)
        row2.addWidget(self.instance_btn)
        row2.addWidget(self.network_btn)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(self.dynamics_btn)
        row3.addWidget(self.visualization_btn)

        root.addLayout(row1)
        root.addLayout(row2)
        root.addLayout(row3)
        root.addStretch(1)

    def _module_button(self, number, title, subtitle):
        btn = QPushButton()
        btn.setObjectName("ModuleButton")
        btn.setMinimumHeight(76)
        btn.setAccessibleName(f"{title}. {subtitle}")
        btn.setToolTip(subtitle)
        layout = QVBoxLayout(btn)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        heading = QLabel(title)
        heading.setObjectName("ModuleTitle")
        heading.setStyleSheet("font-size: 16px; font-weight: 600; background: transparent; border: none;")
        heading.setMinimumHeight(24)
        description = QLabel(subtitle)
        description.setObjectName("ModuleSubtitle")
        description.setStyleSheet("font-size: 13px; font-weight: 400; color: #616161; background: transparent; border: none;")
        description.setMinimumHeight(22)
        for label in (heading, description):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            layout.addWidget(label)
        btn.setMinimumHeight(90)
        btn.setMaximumHeight(96)
        btn.setMinimumWidth(250)
        btn.setMaximumWidth(370)
        return btn


# =============================================================================
# Main window
# =============================================================================

class LinkNexMainWindow(QMainWindow):
    PAGE_HOME = 0
    PAGE_PREPROCESSING = 1
    PAGE_SEMANTIC = 2
    PAGE_INSTANCE = 3
    PAGE_NETWORK = 4
    PAGE_DYNAMICS = 5

    def __init__(self):
        super().__init__()

        self.setWindowTitle("LinkODT")

        # Window title-bar icon.
        for _candidate in LOGO_ICON_CANDIDATES:
            if _candidate.exists():
                _icon = QIcon(str(_candidate))
                if not _icon.isNull():
                    self.setWindowIcon(_icon)
                    break

        # Home is intentionally a small launcher window.
        # Work modules restore the normal large workspace size.
        self.HOME_WIDTH = 790
        self.HOME_HEIGHT = 540
        self.WORK_WIDTH = 1500
        self.WORK_HEIGHT = 920
        self.WORK_MIN_WIDTH = 1180
        self.WORK_MIN_HEIGHT = 760

        self.SEMANTIC_WIDTH = 1560
        self.SEMANTIC_HEIGHT = 930
        self.SEMANTIC_MIN_WIDTH = 1120
        self.SEMANTIC_MIN_HEIGHT = 700

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home = HomeWidget()
        self.preprocessing = ImagePreprocessingWidget()
        self.semantic = SemanticSegmentationWidget()
        self.instance = InstanceSegmentationWidget()
        self.network = OrganelleNetworkAnalysisWidget()
        self.dynamics = PunctateDynamicsWidget()

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.preprocessing)
        self.stack.addWidget(self.semantic)
        self.stack.addWidget(self.instance)
        self.stack.addWidget(self.network)
        self.stack.addWidget(self.dynamics)

        self.home.preprocessingRequested.connect(
            self._open_preprocessing
        )
        self.preprocessing.backRequested.connect(
            self._show_home
        )

        self.home.semanticRequested.connect(
            self._open_semantic
        )
        self.semantic.backRequested.connect(
            self._show_home
        )

        self.home.instanceRequested.connect(
            self._open_instance
        )
        self.instance.backRequested.connect(
            self._show_home
        )

        self.home.networkRequested.connect(
            self._open_network
        )
        self.network.backRequested.connect(
            self._show_home
        )

        self.home.dynamicsRequested.connect(
            self._open_dynamics
        )
        self.dynamics.backRequested.connect(
            self._show_home
        )

        # Module 6 is intentionally a launcher for a separate visualization
        # application, not an embedded QStackedWidget page.
        self.home.visualizationRequested.connect(
            self._open_visualization
        )

        self._apply_style()

        self._build_menu_bar()

        # Start as a genuinely small Home launcher, not a large empty frame.
        self._show_home()

    def _build_menu_bar(self):
        file = self.menuBar().addMenu("File")
        self.menu_open = file.addAction("Open data…")
        self.menu_open.triggered.connect(lambda: self._click_current_control(("open_btn", "open_masks_btn", "folder_btn", "open_files_btn")))
        self.menu_save = file.addAction("Save masks")
        self.menu_save.triggered.connect(lambda: self._click_current_control(("save_btn", "save_cell_mask_btn")))
        self.menu_output = file.addAction("Open output folder")
        self.menu_output.triggered.connect(lambda: self._click_current_control(("output_btn", "open_output_btn")))
        file.addSeparator()
        file.addAction("Close", self.close)
        file.aboutToShow.connect(self._update_menu_actions)
        view = self.menuBar().addMenu("View")
        view.addAction("Home", self._show_home)
        self.menu_fit = view.addAction("Fit image / scene")
        self.menu_fit.triggered.connect(lambda: self._click_current_control(("fit_btn",)))
        view.aboutToShow.connect(self._update_menu_actions)
        install_appearance_controls(self, view)
        analysis = self.menuBar().addMenu("Analysis")
        for label, callback in [("Image preparation", self._open_preprocessing),
                ("Semantic segmentation", self._open_semantic), ("Instances and morphology", self._open_instance),
                ("Network analysis", self._open_network), ("Organelle dynamics", self._open_dynamics),
                ("Visualization Studio", self._open_visualization)]:
            analysis.addAction(label, callback)
        analysis.addSeparator()
        self.menu_run = analysis.addAction("Run selected")
        self.menu_run.triggered.connect(lambda: self._click_current_control(("run_btn", "run_dynamic_btn", "semantic_predict_btn")))
        analysis.aboutToShow.connect(self._update_menu_actions)
        help_menu = self.menuBar().addMenu("Help")
        guide = help_menu.addAction("User guide", lambda: open_user_guide(self))
        guide.setShortcut(QKeySequence("F1"))
        help_menu.addAction("About LinkODT", lambda: show_about(self))
        self._update_menu_actions()

    def _current_control(self, names):
        page = self.stack.currentWidget()
        for name in names:
            control = getattr(page, name, None)
            if control is not None and control.isEnabled():
                return control
        return None

    def _click_current_control(self, names):
        control = self._current_control(names)
        if control is not None:
            control.click()

    def _update_menu_actions(self):
        for action, names in [(self.menu_open, ("open_btn", "open_masks_btn", "folder_btn", "open_files_btn")),
                (self.menu_save, ("save_btn", "save_cell_mask_btn")),
                (self.menu_output, ("output_btn", "open_output_btn")),
                (self.menu_fit, ("fit_btn",)), (self.menu_run, ("run_btn", "run_dynamic_btn", "semantic_predict_btn"))]:
            action.setEnabled(self._current_control(names) is not None)

    def closeEvent(self, event):
        running = []
        for thread in self.findChildren(QThread):
            try:
                if thread.isRunning():
                    running.append(thread)
            except RuntimeError:
                pass
        if running:
            QMessageBox.information(self, "Analysis in progress",
                "Finish the current operation, or use its Stop control, before closing LinkODT.")
            event.ignore()
            return
        if not self.semantic._confirm_discard():
            event.ignore()
            return
        for viewer in (getattr(self.instance, "viewer", None), getattr(self.network, "viewer", None)):
            try:
                viewer.close()
            except (AttributeError, RuntimeError):
                pass
        event.accept()

    def _fit_workspace(self):
        """Keep all controls inside the available desktop on smaller displays."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        bounds = screen.availableGeometry()
        width = min(self.width(), bounds.width() - 24)
        height = min(self.height(), bounds.height() - 60)
        self.setMinimumSize(min(1100, width), min(650, height))
        self.resize(width, height)
        self.move(bounds.x() + max(0, (bounds.width() - self.width()) // 2),
                  bounds.y() + max(0, (bounds.height() - self.height() - 36) // 2))

    def _show_home(self):
        self.stack.setCurrentIndex(
            self.PAGE_HOME
        )

        # If the work area was maximized, return to a normal window first.
        if self.isMaximized():
            self.showNormal()

        self.setMinimumSize(
            self.HOME_WIDTH,
            self.HOME_HEIGHT,
        )
        self.setMaximumSize(
            self.HOME_WIDTH,
            self.HOME_HEIGHT,
        )
        self.resize(
            self.HOME_WIDTH,
            self.HOME_HEIGHT,
        )

    def _open_preprocessing(self):
        # Release the Home fixed-size constraint before opening the work area.
        self.setMaximumSize(
            16777215,
            16777215,
        )
        self.setMinimumSize(
            self.WORK_MIN_WIDTH,
            self.WORK_MIN_HEIGHT,
        )
        self.resize(
            self.WORK_WIDTH,
            self.WORK_HEIGHT,
        )

        self.stack.setCurrentIndex(
            self.PAGE_PREPROCESSING
        )
        self._fit_workspace()

    def _open_semantic(self):
        # Release the compact Home fixed-size constraint.
        self.setMaximumSize(
            16777215,
            16777215,
        )
        self.setMinimumSize(
            self.SEMANTIC_MIN_WIDTH,
            self.SEMANTIC_MIN_HEIGHT,
        )
        self.resize(
            self.SEMANTIC_WIDTH,
            self.SEMANTIC_HEIGHT,
        )

        self.stack.setCurrentIndex(
            self.PAGE_SEMANTIC
        )
        self._fit_workspace()


    def _open_instance(self):
        self.setMaximumSize(
            16777215,
            16777215,
        )

        # Laptop/high-DPI compatibility: fit the work window to the actual
        # available logical screen size instead of always forcing 1560x930.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(
                self.SEMANTIC_WIDTH,
                max(900, int(available.width() * 0.96)),
            )
            target_height = min(
                self.SEMANTIC_HEIGHT,
                max(600, int(available.height() * 0.94)),
            )
        else:
            target_width = self.SEMANTIC_WIDTH
            target_height = self.SEMANTIC_HEIGHT

        self.setMinimumSize(
            min(self.SEMANTIC_MIN_WIDTH, target_width),
            min(self.SEMANTIC_MIN_HEIGHT, target_height),
        )
        self.resize(
            target_width,
            target_height,
        )
        self.stack.setCurrentIndex(
            self.PAGE_INSTANCE
        )

        # Trigger the lazy VTK viewer only after the page is visible.
        QTimer.singleShot(
            80,
            self.instance.viewer.ensure_plotter,
        )
        self._fit_workspace()


    def _open_network(self):
        self.setMaximumSize(
            16777215,
            16777215,
        )

        # Use the same screen-aware sizing for Module 4 so the right-side
        # network 3D canvas also remains visible on smaller laptops.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(
                self.SEMANTIC_WIDTH,
                max(900, int(available.width() * 0.96)),
            )
            target_height = min(
                self.SEMANTIC_HEIGHT,
                max(600, int(available.height() * 0.94)),
            )
        else:
            target_width = self.SEMANTIC_WIDTH
            target_height = self.SEMANTIC_HEIGHT

        self.setMinimumSize(
            min(self.SEMANTIC_MIN_WIDTH, target_width),
            min(self.SEMANTIC_MIN_HEIGHT, target_height),
        )
        self.resize(
            target_width,
            target_height,
        )

        # If Module 3 already has a current mask, hand its processed_output
        # directory directly to Module 4. The user can still choose another
        # folder manually from the Network page.
        try:
            current_mask = self.instance._current_mask()
            if current_mask is not None:
                self.network.set_processed_dir(
                    Path(current_mask).parent / "processed_output",
                    auto_scan=True,
                )
        except Exception:
            pass

        self.stack.setCurrentIndex(
            self.PAGE_NETWORK
        )
        self._fit_workspace()


    def _open_dynamics(self):
        self.setMaximumSize(
            16777215,
            16777215,
        )

        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(
                self.SEMANTIC_WIDTH,
                max(1050, int(available.width() * 0.96)),
            )
            target_height = min(
                self.SEMANTIC_HEIGHT,
                max(700, int(available.height() * 0.94)),
            )
        else:
            target_width = self.SEMANTIC_WIDTH
            target_height = self.SEMANTIC_HEIGHT

        self.setMinimumSize(
            min(self.SEMANTIC_MIN_WIDTH, target_width),
            min(self.SEMANTIC_MIN_HEIGHT, target_height),
        )
        self.resize(target_width, target_height)
        self.stack.setCurrentIndex(self.PAGE_DYNAMICS)
        self._fit_workspace()



    def _open_visualization(self):
        """Launch the standalone LinkODT 3D visualization application.

        Module 6 is deliberately kept outside the main LinkODT process so that
        mesh rendering / movie-generation dependencies do not make the core GUI
        heavier.  When the standalone application is packaged later, placing it
        in ``visualization/`` is enough; this launcher does not need to change.
        """
        if getattr(sys, "frozen", False):
            try:
                subprocess.Popen([sys.executable, "--visualization"], cwd=str(Path(sys.executable).parent))
            except Exception as exc:
                QMessageBox.critical(self, "Visualization", f"Visualization could not be started.\n\n{exc}")
            return
        base_dir = Path(__file__).resolve().parent
        viz_dir = base_dir / "visualization"

        # Preferred packaged executable first, followed by development entries.
        executable_candidates = [
            viz_dir / "LinkNex_Visualization.exe",
            base_dir / "LinkNex_Visualization.exe",
        ]
        python_candidates = [
            viz_dir / "main.py",
            viz_dir / "visualization_app.py",
            viz_dir / "LinkNex_Visualization.py",
        ]

        try:
            for candidate in executable_candidates:
                if candidate.is_file():
                    subprocess.Popen(
                        [str(candidate)],
                        cwd=str(candidate.parent),
                    )
                    return

            for candidate in python_candidates:
                if candidate.is_file():
                    subprocess.Popen(
                        [sys.executable, str(candidate)],
                        cwd=str(candidate.parent),
                    )
                    return

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Visualization",
                "The standalone Visualization application could not be started.\n\n"
                f"{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Visualization",
            "The standalone 3D Visualization application has not been added yet.\n\n"
            "When it is ready, place it here:\n"
            f"{viz_dir / 'LinkNex_Visualization.exe'}\n\n"
            "The Visualization button will then launch it directly.",
        )

    def _reserved(self, name):
        QMessageBox.information(
            self,
            name,
            f"{name} module entry is reserved.\n\nWe will connect this module next.",
        )

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {OFFWHITE};
                color: {TEXT};
            }}

            QWidget {{
                color: {TEXT};
                font-family: "Segoe UI";
                font-size: 14px;
            }}

            QFrame#SidePanel {{
                background: transparent;
                border: none;
                border-radius: 0px;
            }}

            QFrame#InteractionFrame {{
                background: #F5F7FB;
                border: 2px solid {BORDER};
                border-radius: 8px;
            }}

            QLabel#Section {{
                color: {DEEP_BLUE};
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0px;
                padding-top: 4px;
            }}

            QLabel#Metric {{
                color: {TEXT};
                font-size: 14px;
                font-weight: 600;
            }}

            QLabel#DimensionInfo {{
                color: {TEXT};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                font-weight: 600;
            }}

            QPlainTextEdit#BatchStatus {{
                color: {TEXT};
                background: {WHITE};
                border: 2px solid {BORDER};
                border-radius: 6px;
                padding: 6px;
                font-family: "Segoe UI";
                font-size: 13px;
            }}

            QLabel#Muted {{
                color: {MUTED};
                font-size: 14px;
            }}

            QLabel#CanvasTitle {{
                color: {DEEP_BLUE};
                font-size: 14px;
                font-weight: 600;
            }}

            QLabel#SmallLabel {{
                color: {MUTED};
                font-size: 13px;
                font-weight: 600;
            }}

            QFrame#ProjectionBar {{
                background: transparent;
                border: none;
            }}

            QPushButton#ProjectionButton {{
                min-height: 28px;
                padding: 2px 10px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {DEEP_BLUE};
                font-size: 13px;
                font-weight: 600;
            }}

            QPushButton#ProjectionButton:hover {{
                border-color: {US_BLUE};
                background: #EEF7FD;
            }}

            QPushButton#ProjectionButton:checked {{
                background: {CYAN};
                color: {WHITE};
                border-color: {US_BLUE};
            }}

            QPushButton#ProjectionButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QLabel#HomeLogo {{
                color: {NAVY};
                font-size: 26px;
                font-weight: 600;
            }}

            QLabel#HomeSubtitle {{
                color: {MUTED};
                font-size: 14px;
                font-weight: 600;
            }}

            QPushButton {{
                min-height: 34px;
                padding: 3px 9px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {NAVY};
                font-weight: 700;
            }}

            QPushButton:hover {{
                border-color: {US_BLUE};
                background: #EEF7FD;
            }}

            QPushButton:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
            }}

            QPushButton:disabled {{
                color: #A9B4C0;
                background: #F1F3F6;
                border-color: #C9D4E2;
            }}

            QPushButton:checked {{
                background: {CYAN};
                color: {WHITE};
                border-color: {US_BLUE};
            }}

            QPushButton#Primary {{
                background: {US_BLUE};
                color: {WHITE};
                border-color: {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#Primary:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#Good {{
                min-height: 40px;
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
                font-weight: 600;
            }}

            QPushButton#Good:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#Bad {{
                min-height: 40px;
                background: #D94B4B;
                color: {WHITE};
                border: 2.5px solid #AE3232;
                font-weight: 600;
            }}

            QPushButton#Bad:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#Save {{
                min-height: 42px;
                background: {NAVY};
                color: {WHITE};
                border: 2.5px solid {NAVY};
                font-weight: 600;
            }}

            QPushButton#Save:hover {{
                background: {DEEP_BLUE};
                border-color: {US_BLUE};
            }}

            QPushButton#Save:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#SaveMask {{
                min-height: 38px;
                background: {WHITE};
                color: {DEEP_BLUE};
                border: 2.5px solid {US_BLUE};
                font-weight: 600;
            }}

            QPushButton#SaveMask:hover {{
                background: #EEF7FD;
                border-color: {DEEP_BLUE};
            }}

            QPushButton#SaveMask:pressed {{
                background: {YELLOW};
                color: {NAVY};
                border-color: #C89C00;
            }}

            QPushButton#Save[state="working"],
            QPushButton#Primary[state="working"],
            QPushButton#SaveMask[state="working"],
            QPushButton#AutoQC[state="working"] {{
                background: {YELLOW};
                color: {NAVY};
                border: 2.5px solid #C89C00;
                font-weight: 600;
            }}

            QPushButton#Save[state="done"],
            QPushButton#Primary[state="done"],
            QPushButton#SaveMask[state="done"],
            QPushButton#AutoQC[state="done"] {{
                background: #1E9E5A;
                color: {WHITE};
                border: 2.5px solid #167B45;
                font-weight: 600;
            }}

            QPushButton#ModuleButton {{
                text-align: left;
                min-height: 68px;
                padding: 8px 12px;
                background: {WHITE};
                border: 1px solid {BORDER};
                border-radius: 6px;
                color: {NAVY};
                font-size: 14px;
                font-weight: 600;
            }}

            QPushButton#ModuleButton:hover {{
                background: #EEF7FD;
                border-color: {US_BLUE};
            }}

            QPushButton#ModuleButton:pressed {{
                background: #E1EAF8;
                color: {NAVY};
                border-color: {US_BLUE};
            }}

            QListWidget {{
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {TEXT};
                outline: none;
            }}

            QListWidget::item {{
                padding: 7px 6px;
                border-radius: 4px;
                color: {TEXT};
                background: {WHITE};
            }}

            QListWidget::item:hover {{
                background: #EAF5FC;
                color: {NAVY};
            }}

            QListWidget::item:selected {{
                background: {US_BLUE};
                color: {WHITE};
            }}

            QListWidget::item:selected:!active {{
                background: {US_BLUE};
                color: {WHITE};
            }}

            QSpinBox,
            QLineEdit {{
                min-height: 32px;
                padding: 2px 7px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {TEXT};
                font-weight: 700;
            }}

            QLineEdit#ChannelFolderEdit {{
                min-height: 30px;
                border: 2px solid {BORDER};
                border-radius: 6px;
                background: {WHITE};
                color: {TEXT};
                font-weight: 600;
            }}

            QLineEdit#ChannelFolderEdit:focus {{
                border-color: {US_BLUE};
                background: #F7FCFF;
            }}

            QTableWidget {{
                background: {WHITE};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E3E9F0;
                outline: none;
            }}

            QHeaderView::section {{
                background: #EEF3F8;
                color: {DEEP_BLUE};
                border: none;
                border-bottom: 2px solid {BORDER};
                padding: 5px;
                font-weight: 600;
            }}

            QMessageBox {{
                background: {WHITE};
                color: {TEXT};
            }}

            QMessageBox QLabel {{
                background: transparent;
                color: {TEXT};
                font-size: 14px;
                font-weight: 600;
                min-width: 260px;
            }}

            QMessageBox QPushButton {{
                min-width: 88px;
                background: {WHITE};
                color: {NAVY};
                border: 2px solid {BORDER};
            }}

            QMessageBox QPushButton:hover {{
                background: #EEF7FD;
                color: {NAVY};
                border-color: {US_BLUE};
            }}
        """ + DISABLED_STYLE)




# =============================================================================
# Module 3 / Module 4 direct-load enhancement
# =============================================================================
# Allows image/mask/TXT-PCD drag-and-drop and direct browsing of existing
# processed_output results without forcing the pipeline to rerun.
from linknex_module34_directload_patch import apply_linknex_module34_patch
apply_linknex_module34_patch(
    InstanceSegmentationWidget,
    OrganelleNetworkAnalysisWidget,
)

def main():
    # Same EXE serves two modes:
    #   LinkNex.exe                         -> main LinkNex GUI
    #   LinkNex.exe --pcd-editor FILE.txt   -> standalone full 3D Editor
    if (
        len(
            sys.argv
        )
        >= 2
        and sys.argv[
            1
        ]
        == "--pcd-editor"
    ):
        initial_path = (
            Path(
                sys.argv[
                    2
                ]
            )
            if len(
                sys.argv
            )
            >= 3
            else None
        )

        run_embedded_pcd_editor(
            initial_path
        )
        return

    app = QApplication(
        sys.argv
    )

    # Set an application-wide icon as well as the main-window icon.
    # This helps Windows taskbar, dialogs, and secondary windows.
    for _candidate in LOGO_ICON_CANDIDATES:
        if _candidate.exists():
            _app_icon = QIcon(str(_candidate))
            if not _app_icon.isNull():
                app.setWindowIcon(_app_icon)
                break

    configure_application(app)
    win = LinkNexMainWindow()
    win.show()
    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()

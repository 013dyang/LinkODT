from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("QT_API", "pyside6")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QThread, Signal, QSettings
from PySide6.QtGui import QColor, QIcon, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QToolButton,
    QMenu,
    QScrollArea,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from director import CameraKeyframe, interpolate_camera, sort_keyframes
from io_utils import LoadedGeometry, load_geometry
from ai_director import request_qwen_plan, summarize_plan


# Source and bundled launchers share the application theme.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui_theme import configure_application, DISABLED_STYLE, install_appearance_controls, open_user_guide, show_about

APP_NAME = "LinkODT Visualization Studio"
DEFAULT_BG = (0.055, 0.065, 0.08)
DEFAULT_OBJECT_COLOR = (0.75, 0.82, 0.92)
DEFAULT_INTENSITY_BASE_COLOR = (0.10, 0.85, 0.25)
# Intensity-only XYZ+I point clouds: lowest displayed brightness as a fraction
# of the brightest same-hue color. Change 0.35 here to tune the dark floor.
DEFAULT_INTENSITY_MIN_BRIGHTNESS = 0.35
# <1 brightens low-to-mid intensities while preserving ordering.
DEFAULT_INTENSITY_GAMMA = 0.72
DEFAULT_NORMAL_COLOR = (1.0, 0.72, 0.18)


@dataclass
class SceneObject:
    geometry: LoadedGeometry
    actor: vtk.vtkActor
    list_item: QListWidgetItem
    color: tuple[float, float, float] = DEFAULT_OBJECT_COLOR
    normal_color: tuple[float, float, float] = DEFAULT_NORMAL_COLOR
    line_color: tuple[float, float, float] = (0.18, 0.22, 0.26)
    use_file_colors: bool = True
    use_intensity: bool = False
    representation: str = "Surface"
    point_size: float = 3.0
    line_width: float = 1.0
    opacity: float = 1.0
    normal_actor: Optional[vtk.vtkActor] = None
    normals_visible: bool = False
    normal_length: float = 2.0
    normal_stride: int = 10

    @property
    def name(self) -> str:
        return self.geometry.path.name


@dataclass
class ROI3D:
    name: str
    center: tuple[float, float, float]
    radius: float
    actor: vtk.vtkActor
    source: vtk.vtkSphereSource
    list_item: QListWidgetItem

    def to_dict(self):
        return {
            "name": self.name,
            "center": [float(x) for x in self.center],
            "radius": float(self.radius),
        }


class QwenPlanWorker(QThread):
    finishedPlan = Signal(object)
    failed = Signal(str)

    def __init__(self, kwargs, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        try:
            self.finishedPlan.emit(request_qwen_plan(**self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))


class AIDirectorDialog(QDialog):
    def __init__(self, studio):
        super().__init__(studio)
        self.studio = studio
        self.plan = None
        self.worker = None
        self.settings = QSettings("LinkNex", "VisualizationStudio")
        self.setWindowTitle("AI Director — Local Qwen GGUF")
        self.resize(820, 760)
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Describe the shot naturally in Chinese or English. A local Qwen GGUF translates your words into LinkODT camera actions. "
            "Nothing is sent to a cloud API."
        )
        intro.setWordWrap(True)
        intro.setObjectName("infoLabel")
        lay.addWidget(intro)

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText(
            "例如：先以选中的 ROI 为中心，慢慢推近 1.6 倍；然后向左绕 35 度并稍微抬高视角，停 1 秒；"
            "再向右做半圈 turntable，最后拉远并切到 top view。"
        )
        self.prompt.setMinimumHeight(130)
        lay.addWidget(QLabel("Describe the shot"))
        lay.addWidget(self.prompt)

        cfg = QGroupBox("Local Qwen model (.gguf)")
        form = QFormLayout(cfg)
        model_row = QWidget()
        model_lay = QHBoxLayout(model_row)
        model_lay.setContentsMargins(0, 0, 0, 0)
        self.model_path_edit = QLineEdit(str(self.settings.value("local_qwen_model", "") or ""))
        self.model_path_edit.setPlaceholderText("Qwen3-8B-Q4_K_M.gguf")
        self.browse_model_btn = QPushButton("Choose GGUF…")
        model_lay.addWidget(self.model_path_edit, 1)
        model_lay.addWidget(self.browse_model_btn)
        form.addRow("Model file", model_row)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(2048, 65536)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(int(self.settings.value("local_qwen_ctx", 8192)))
        form.addRow("Context tokens", self.ctx_spin)

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 200)
        self.gpu_layers_spin.setValue(int(self.settings.value("local_qwen_gpu_layers", -1)))
        self.gpu_layers_spin.setSpecialValueText("All possible (-1)")
        self.gpu_layers_spin.setToolTip("-1 asks llama.cpp to offload all possible layers. Use 0 for CPU-only.")
        form.addRow("GPU layers", self.gpu_layers_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 128)
        self.threads_spin.setValue(int(self.settings.value("local_qwen_threads", 0)))
        self.threads_spin.setSpecialValueText("Auto")
        form.addRow("CPU threads", self.threads_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 8192)
        self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setValue(int(self.settings.value("local_qwen_max_tokens", 2048)))
        form.addRow("Max output tokens", self.max_tokens_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.5)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(float(self.settings.value("local_qwen_temperature", 0.2)))
        form.addRow("Temperature", self.temp_spin)

        model_note = QLabel(
            "Recommended for this translation task: Qwen3-4B/8B/14B Q4_K_M. First use loads the model into RAM/VRAM; later requests reuse it."
        )
        model_note.setWordWrap(True)
        model_note.setObjectName("infoLabel")
        form.addRow(model_note)
        lay.addWidget(cfg)

        timing = QGroupBox("Timeline")
        tform = QFormLayout(timing)
        self.start_time = QDoubleSpinBox()
        self.start_time.setRange(0.0, 3600.0)
        self.start_time.setDecimals(2)
        self.start_time.setValue(float(studio._duration()))
        self.replace_timeline = QCheckBox("Replace existing camera timeline")
        self.replace_timeline.setChecked(True)
        tform.addRow("Start time (s)", self.start_time)
        tform.addRow(self.replace_timeline)
        lay.addWidget(timing)

        roi_name = studio.selected_roi_name() or "None"
        self.context_label = QLabel(
            f"Selected ROI: {roi_name}   |   Defined ROIs: {len(studio.rois)}   |   Loaded objects: {len(studio.objects)}"
        )
        self.context_label.setObjectName("infoLabel")
        lay.addWidget(self.context_label)

        row = QHBoxLayout()
        self.translate_btn = QPushButton("Translate with Local Qwen")
        self.apply_btn = QPushButton("Apply Plan to Timeline")
        self.apply_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        row.addWidget(self.translate_btn, 1)
        row.addWidget(self.apply_btn, 1)
        row.addWidget(self.close_btn)
        lay.addLayout(row)

        self.status = QLabel("Ready — choose a local GGUF model and describe the shot.")
        self.status.setObjectName("infoLabel")
        lay.addWidget(self.status)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("The locally validated action plan will appear here before it is applied.")
        lay.addWidget(self.preview, 1)

        self.browse_model_btn.clicked.connect(self.choose_model)
        self.translate_btn.clicked.connect(self.translate)
        self.apply_btn.clicked.connect(self.apply_plan)
        self.close_btn.clicked.connect(self.close)

    def choose_model(self):
        start = self.model_path_edit.text().strip()
        folder = str(Path(start).parent) if start else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local Qwen GGUF", folder, "GGUF model (*.gguf)")
        if path:
            self.model_path_edit.setText(path)
            self.settings.setValue("local_qwen_model", path)

    def _save_settings(self):
        self.settings.setValue("local_qwen_model", self.model_path_edit.text().strip())
        self.settings.setValue("local_qwen_ctx", self.ctx_spin.value())
        self.settings.setValue("local_qwen_gpu_layers", self.gpu_layers_spin.value())
        self.settings.setValue("local_qwen_threads", self.threads_spin.value())
        self.settings.setValue("local_qwen_max_tokens", self.max_tokens_spin.value())
        self.settings.setValue("local_qwen_temperature", self.temp_spin.value())

    def translate(self):
        self._save_settings()
        selected = self.studio.selected_roi_name()
        kwargs = {
            "user_prompt": self.prompt.toPlainText(),
            "rois": [r.to_dict() for r in self.studio.rois],
            "selected_roi": selected,
            "camera": self.studio.current_camera_dict(),
            "objects": [{"name": o.name} for o in self.studio.objects],
            "model_path": self.model_path_edit.text().strip(),
            "n_ctx": int(self.ctx_spin.value()),
            "n_gpu_layers": int(self.gpu_layers_spin.value()),
            "n_threads": int(self.threads_spin.value()),
            "max_tokens": int(self.max_tokens_spin.value()),
            "temperature": float(self.temp_spin.value()),
        }
        self.translate_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.browse_model_btn.setEnabled(False)
        self.status.setText("Loading/running local Qwen… First use may take longer while the GGUF is loaded.")
        self.worker = QwenPlanWorker(kwargs, self)
        self.worker.finishedPlan.connect(self._plan_ready)
        self.worker.failed.connect(self._plan_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _worker_finished(self):
        self.translate_btn.setEnabled(True)
        self.browse_model_btn.setEnabled(True)

    def _plan_ready(self, plan):
        self.plan = plan
        self.preview.setPlainText(summarize_plan(plan) + "\n\nJSON\n" + json.dumps(plan, indent=2, ensure_ascii=False))
        self.status.setText("Local Qwen plan validated. Review it, then apply to the timeline.")
        self.apply_btn.setEnabled(True)

    def _plan_failed(self, message):
        self.plan = None
        self.preview.setPlainText(message)
        self.status.setText("Could not build a valid local camera plan.")
        self.apply_btn.setEnabled(False)

    def apply_plan(self):
        if not self.plan:
            return
        try:
            self.studio.apply_ai_plan(
                self.plan,
                start_time=float(self.start_time.value()),
                replace_timeline=bool(self.replace_timeline.isChecked()),
            )
            self.status.setText("Applied. Use Preview or Export MP4 in Director.")
        except Exception as exc:
            QMessageBox.critical(self, "AI Director", str(exc))



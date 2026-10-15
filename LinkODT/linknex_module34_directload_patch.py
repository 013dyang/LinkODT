# -*- coding: utf-8 -*-
"""
LinkODT Module 3 / Module 4 direct-load patch
==============================================

Purpose
-------
1. Module 3 (Instance segmentation)
   - Drag image / mask / TXT-PCD directly into the module.
   - Resolve the sample stem automatically.
   - Search ONLY:
       a) the input file's directory
       b) <input directory>/processed_output
     If the dropped file is already inside processed_output, the parent dataset
     directory is inferred automatically.
   - Existing physical/social instance PCD results are listed immediately and
     shown in the main 3D viewer without rerunning segmentation.
   - If only a semantic mask exists, Run selected / All still works normally.

2. Module 4 (Network analysis)
   - Drag image / mask / TXT-PCD / network result directly into the module.
   - Locate the corresponding processed_output automatically.
   - Existing *_hierarchical_network.pkl and metrics CSV are listed and
     previewed immediately.
   - Network results do not need to be rerun just to view them.

Integration
-----------
In the complete LinkODT project package supplied with this file, LinkNex.py
already imports and applies this patch automatically before the application is
started. No manual patching step is required.

This patch is intentionally limited to Modules 3 and 4.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import os
import subprocess
import sys

from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
)


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
POINT_SUFFIXES = {".txt", ".csv", ".pcd", ".ply"}
NETWORK_SUFFIXES = {".pkl"}

# Longest / most specific first.
KNOWN_STEM_SUFFIXES = (
    "_hierarchical_network_metrics",
    "_hierarchical_network",
    "_instance_morphology_physical",
    "_instance_morphology_social",
    "_pcd_instance_physical",
    "_pcd_instance_social",
    "_voxel_instance_physical",
    "_voxel_instance_phy",
    "_voxel_instance_social",
    "_skeleton_instance_physical",
    "_skeleton_instance_social",
    "_skeleton_edges",
    "_mask_instance",
    "_pcd",
    "_voxel",
    "_mask",
)


def _natural_key(value):
    """Small local natural-sort helper; avoids depending on LinkODT globals."""
    import re
    parts = re.split(r"(\d+)", str(value))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _stem_from_any(path: Path) -> str:
    """
    Convert LinkODT input/result filename to the original sample stem.

    Examples
    --------
    cell001.tif                            -> cell001
    cell001_mask.tif                       -> cell001
    cell001_pcd.txt                        -> cell001
    cell001_pcd_instance_physical.txt      -> cell001
    cell001_hierarchical_network.pkl       -> cell001
    """
    p = Path(path)
    name = p.name

    # Strip one extension first.
    base = p.stem

    # CSV such as "..._hierarchical_network_metrics.csv" is handled naturally.
    lower = base.lower()
    changed = True
    while changed:
        changed = False
        for suffix in KNOWN_STEM_SUFFIXES:
            if lower.endswith(suffix):
                base = base[: -len(suffix)]
                lower = base.lower()
                changed = True
                break

    return base


def _resolve_base_and_processed(path: Path):
    """
    Resolve:
        base_dir        = dataset/image folder
        processed_dir   = base_dir/processed_output, or the current folder if
                          the input is already inside processed_output.
    """
    p = Path(path)

    anchor = p if p.is_dir() else p.parent

    if anchor.name.lower() == "processed_output":
        base_dir = anchor.parent
        processed_dir = anchor
    else:
        base_dir = anchor
        candidate = base_dir / "processed_output"
        processed_dir = candidate if candidate.is_dir() else candidate

    return base_dir, processed_dir


def _first_existing(*paths):
    for path in paths:
        if path is not None:
            p = Path(path)
            if p.exists():
                return p
    return None


def _discover_record(path: Path, forced_stem: str | None = None):
    """
    Discover all known Module 3 / 4 files for one sample.

    Search scope is deliberately restricted to:
        - base_dir
        - base_dir/processed_output
    """
    p = Path(path)
    base_dir, processed_dir = _resolve_base_and_processed(p)

    stem = str(forced_stem or _stem_from_any(p)).strip()
    if not stem:
        return None

    # Source/mask files: prefer base directory.
    mask_path = _first_existing(
        base_dir / f"{stem}_mask.tif",
        base_dir / f"{stem}_mask.tiff",
        processed_dir / f"{stem}_mask.tif",
        processed_dir / f"{stem}_mask.tiff",
    )

    image_path = _first_existing(
        base_dir / f"{stem}.tif",
        base_dir / f"{stem}.tiff",
        base_dir / f"{stem}.png",
        base_dir / f"{stem}.jpg",
        base_dir / f"{stem}.jpeg",
        base_dir / f"{stem}.bmp",
    )

    # Module 3 results: prefer processed_output, then same directory.
    pcd_raw = _first_existing(
        processed_dir / f"{stem}_pcd.txt",
        base_dir / f"{stem}_pcd.txt",
    )
    pcd_physical = _first_existing(
        processed_dir / f"{stem}_pcd_instance_physical.txt",
        base_dir / f"{stem}_pcd_instance_physical.txt",
    )
    pcd_social = _first_existing(
        processed_dir / f"{stem}_pcd_instance_social.txt",
        base_dir / f"{stem}_pcd_instance_social.txt",
    )

    morphology_physical = _first_existing(
        processed_dir / f"{stem}_instance_morphology_physical.csv",
        base_dir / f"{stem}_instance_morphology_physical.csv",
    )
    morphology_social = _first_existing(
        processed_dir / f"{stem}_instance_morphology_social.csv",
        base_dir / f"{stem}_instance_morphology_social.csv",
    )

    skeleton_edges = _first_existing(
        processed_dir / f"{stem}_skeleton_edges.txt",
        base_dir / f"{stem}_skeleton_edges.txt",
    )
    skeleton_social = _first_existing(
        processed_dir / f"{stem}_skeleton_instance_social.txt",
        base_dir / f"{stem}_skeleton_instance_social.txt",
    )

    # Module 4 results.
    network_pkl = _first_existing(
        processed_dir / f"{stem}_hierarchical_network.pkl",
        base_dir / f"{stem}_hierarchical_network.pkl",
    )
    network_metrics = _first_existing(
        processed_dir / f"{stem}_hierarchical_network_metrics.csv",
        base_dir / f"{stem}_hierarchical_network_metrics.csv",
    )

    # If the user directly dropped one of these exact files, trust that exact
    # path even when it does not use the normal directory placement.
    if p.is_file():
        lname = p.name.lower()
        if lname.endswith("_mask.tif") or lname.endswith("_mask.tiff"):
            mask_path = p
        elif lname.endswith("_pcd_instance_physical.txt"):
            pcd_physical = p
        elif lname.endswith("_pcd_instance_social.txt"):
            pcd_social = p
        elif lname.endswith("_pcd.txt"):
            pcd_raw = p
        elif lname.endswith("_instance_morphology_physical.csv"):
            morphology_physical = p
        elif lname.endswith("_instance_morphology_social.csv"):
            morphology_social = p
        elif lname.endswith("_skeleton_edges.txt"):
            skeleton_edges = p
        elif lname.endswith("_skeleton_instance_social.txt"):
            skeleton_social = p
        elif lname.endswith("_hierarchical_network.pkl"):
            network_pkl = p
        elif lname.endswith("_hierarchical_network_metrics.csv"):
            network_metrics = p
        elif p.suffix.lower() in IMAGE_SUFFIXES and "_mask" not in lname:
            image_path = p

    return {
        "stem": stem,
        "base_dir": base_dir,
        "processed_dir": processed_dir,
        "source_path": p,
        "image_path": image_path,
        "mask_path": mask_path,
        "pcd_raw": pcd_raw,
        "pcd_physical": pcd_physical,
        "pcd_social": pcd_social,
        "morphology_physical": morphology_physical,
        "morphology_social": morphology_social,
        "skeleton_edges": skeleton_edges,
        "skeleton_social": skeleton_social,
        "network_pkl": network_pkl,
        "network_metrics": network_metrics,
    }


def _record_key(record):
    return (
        str(Path(record["base_dir"]).resolve()).lower(),
        str(record["stem"]).lower(),
    )


def _candidate_stems_from_folder(folder: Path):
    """
    Scan only folder and folder/processed_output (one known result subfolder).
    No general recursive walk.
    """
    folder = Path(folder)

    if folder.name.lower() == "processed_output":
        base_dir = folder.parent
        processed_dir = folder
    else:
        base_dir = folder
        processed_dir = folder / "processed_output"

    dirs = [base_dir]
    if processed_dir.is_dir() and processed_dir != base_dir:
        dirs.append(processed_dir)

    stems = set()

    accepted = IMAGE_SUFFIXES | POINT_SUFFIXES | NETWORK_SUFFIXES | {".csv"}

    for d in dirs:
        if not d.is_dir():
            continue
        try:
            entries = list(d.iterdir())
        except Exception:
            entries = []

        for p in entries:
            if not p.is_file() or p.suffix.lower() not in accepted:
                continue

            lname = p.name.lower()

            # Ignore broad unrelated CSV/TXT files. Only accept result-like
            # files or source image/mask files.
            result_like = any(
                token in lname
                for token in (
                    "_mask",
                    "_pcd",
                    "_voxel",
                    "_skeleton",
                    "_instance_morphology",
                    "_hierarchical_network",
                )
            )
            source_image = p.suffix.lower() in IMAGE_SUFFIXES

            if source_image or result_like:
                stem = _stem_from_any(p)
                if stem:
                    stems.add(stem)

    return sorted(stems, key=_natural_key)


class _DropFilter(QObject):
    """Reusable drag/drop event filter for a LinkODT module."""

    def __init__(self, owner, callback):
        super().__init__(owner)
        self.owner = owner
        self.callback = callback

    def eventFilter(self, watched, event):
        et = event.type()

        if et in (QEvent.DragEnter, QEvent.DragMove):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True

        if et == QEvent.Drop and event.mimeData().hasUrls():
            paths = [
                Path(url.toLocalFile())
                for url in event.mimeData().urls()
                if url.isLocalFile()
            ]
            if paths:
                self.callback(paths)
                event.acceptProposedAction()
                return True

        return False


def _install_drop_filter(owner, callback, *widgets):
    filt = _DropFilter(owner, callback)
    owner._linknex_directload_drop_filter = filt

    targets = [owner]
    targets.extend(w for w in widgets if w is not None)

    for w in targets:
        try:
            w.setAcceptDrops(True)
        except Exception:
            pass
        try:
            w.installEventFilter(filt)
        except Exception:
            pass
        try:
            viewport = w.viewport()
        except Exception:
            viewport = None
        if viewport is not None:
            try:
                viewport.setAcceptDrops(True)
                viewport.installEventFilter(filt)
            except Exception:
                pass


# =============================================================================
# Module 3 patch
# =============================================================================

def _patch_module3(InstanceSegmentationWidget):
    if getattr(InstanceSegmentationWidget, "_linknex_directload_patched", False):
        return

    original_init = InstanceSegmentationWidget.__init__
    original_pipeline_file_finished = getattr(
        InstanceSegmentationWidget, "_pipeline_file_finished", None
    )
    original_pipeline_finished = getattr(
        InstanceSegmentationWidget, "_pipeline_finished", None
    )

    def __init__(self, parent=None):
        self._lnx_records = OrderedDict()
        original_init(self, parent)

        try:
            self.open_masks_btn.setText("Open Image / Mask / PCD")
        except Exception:
            pass

        try:
            self.open_folder_btn.setText("Folder")
        except Exception:
            pass

        try:
            self.status_label.setText(
                "Drag an image, *_mask.tif, or TXT/PCD here. Existing "
                "processed_output results are loaded directly."
            )
        except Exception:
            pass

        _install_drop_filter(
            self,
            self._lnx_add_input_paths,
            getattr(self, "mask_list", None),
            getattr(self, "viewer", None),
        )

    def _lnx_rebuild_list(self, preferred_key=None):
        old_key = None
        current = self._lnx_current_record()
        if current is not None:
            old_key = _record_key(current)

        if preferred_key is None:
            preferred_key = old_key

        records = list(self._lnx_records.values())
        records.sort(key=lambda r: (_natural_key(r["stem"]), str(r["base_dir"]).lower()))

        self._lnx_records = OrderedDict((_record_key(r), r) for r in records)

        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        self.mask_paths = []

        select_row = -1

        for row, (key, rec) in enumerate(self._lnx_records.items()):
            physical = rec.get("pcd_physical")
            social = rec.get("pcd_social")
            raw = rec.get("pcd_raw")
            mask = rec.get("mask_path")
            image = rec.get("image_path")

            if physical and social:
                state = "✓ Physical + Social"
            elif physical:
                state = "✓ Physical"
            elif social:
                state = "✓ Social"
            elif raw:
                state = "PCD"
            elif mask:
                state = "Mask"
            elif image:
                state = "Image"
            else:
                state = "Detected"

            item = QListWidgetItem(f"{rec['stem']}    [{state}]")
            item.setData(Qt.UserRole, key)
            item.setToolTip(
                "Base: {}\nprocessed_output: {}".format(
                    rec["base_dir"], rec["processed_dir"]
                )
            )
            self.mask_list.addItem(item)

            # Keep this list populated for compatibility with legacy methods.
            compatibility_path = (
                rec.get("mask_path")
                or rec.get("image_path")
                or rec.get("pcd_physical")
                or rec.get("pcd_social")
                or rec.get("pcd_raw")
                or rec.get("source_path")
            )
            self.mask_paths.append(Path(compatibility_path))

            if preferred_key is not None and key == preferred_key:
                select_row = row

        self.mask_list.blockSignals(False)

        if self.mask_list.count():
            if select_row < 0:
                select_row = 0
            self.mask_list.setCurrentRow(select_row)
            self._selection_changed(select_row)

        self._update_controls()

    def _lnx_add_input_paths(self, paths):
        records_added = 0
        preferred_key = None

        for raw in paths:
            p = Path(raw)
            if not p.exists():
                continue

            if p.is_dir():
                stems = _candidate_stems_from_folder(p)
                for stem in stems:
                    rec = _discover_record(p, forced_stem=stem)
                    if rec is None:
                        continue
                    key = _record_key(rec)
                    self._lnx_records[key] = rec
                    preferred_key = preferred_key or key
                    records_added += 1
                continue

            if p.suffix.lower() not in (
                IMAGE_SUFFIXES | POINT_SUFFIXES | NETWORK_SUFFIXES | {".csv"}
            ):
                continue

            rec = _discover_record(p)
            if rec is None:
                continue
            key = _record_key(rec)
            self._lnx_records[key] = rec
            preferred_key = preferred_key or key
            records_added += 1

        if records_added:
            self._lnx_rebuild_list(preferred_key=preferred_key)
            self.status_label.setText(
                f"{len(self._lnx_records)} sample(s) loaded. Existing instance "
                "results are available immediately; Run is only needed for masks "
                "without results."
            )
        else:
            self.status_label.setText(
                "No supported LinkODT image / mask / PCD input was found."
            )

    def _add_mask_paths(self, paths):
        # Legacy name retained so existing buttons/workflows keep working.
        self._lnx_add_input_paths(paths)

    def open_masks(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open image / semantic mask / point cloud",
            "",
            (
                "LinkODT inputs (*.tif *.tiff *.png *.jpg *.jpeg *.bmp "
                "*.txt *.csv *.pcd *.ply);;"
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;"
                "Point clouds (*.txt *.csv *.pcd *.ply);;"
                "All files (*)"
            ),
        )
        if paths:
            self._lnx_add_input_paths([Path(p) for p in paths])

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open image folder / processed_output",
            "",
        )
        if folder:
            self._lnx_add_input_paths([Path(folder)])

    def _lnx_record_from_item(self, item):
        if item is None:
            return None
        key = item.data(Qt.UserRole)
        return self._lnx_records.get(key)

    def _lnx_current_record(self):
        return self._lnx_record_from_item(self.mask_list.currentItem())

    def _lnx_selected_records(self):
        out = []
        for item in self.mask_list.selectedItems():
            rec = self._lnx_record_from_item(item)
            if rec is not None:
                out.append(rec)
        return out

    def _selected_paths(self):
        """Return only runnable semantic masks."""
        return [
            Path(rec["mask_path"])
            for rec in self._lnx_selected_records()
            if rec.get("mask_path") is not None
        ]

    def _current_mask(self):
        rec = self._lnx_current_record()
        if rec is None or rec.get("mask_path") is None:
            return None
        return Path(rec["mask_path"])

    def _lnx_existing_display_path(self, rec, mode=None):
        if rec is None:
            return None

        mode = str(
            mode or getattr(getattr(self, "viewer", None), "current_mode", "physical")
        ).lower()

        if mode == "social":
            return (
                rec.get("pcd_social")
                or rec.get("pcd_physical")
                or rec.get("pcd_raw")
            )

        return (
            rec.get("pcd_physical")
            or rec.get("pcd_social")
            or rec.get("pcd_raw")
        )

    def _selection_changed(self, _row):
        rec = self._lnx_current_record()
        self._update_controls()

        if rec is None or not hasattr(self, "viewer"):
            return

        # Give the existing Physical/Social viewer a normal synthetic mask
        # anchor so its own Physical/Social toggle continues to work with
        # <base>/processed_output/<stem>_... files.
        anchor = rec.get("mask_path")
        if anchor is None:
            anchor = Path(rec["base_dir"]) / f"{rec['stem']}_mask.tif"

        try:
            self.viewer.set_mask_path(anchor)
        except Exception:
            pass

        display_path = self._lnx_existing_display_path(rec)

        if display_path is not None and Path(display_path).exists():
            try:
                self.viewer.load_point_cloud(
                    Path(display_path),
                    preserve_camera=False,
                )
            except Exception as exc:
                self.status_label.setText(f"Could not load {Path(display_path).name}: {exc}")
                return

            self.status_label.setText(
                f"{rec['stem']}: loaded existing {Path(display_path).name}"
            )
        else:
            if rec.get("mask_path") is not None:
                self.status_label.setText(
                    f"{rec['stem']}: semantic mask found; no instance PCD yet. "
                    "Run selected to generate it."
                )
            else:
                self.status_label.setText(
                    f"{rec['stem']}: no displayable instance PCD found."
                )

    def run_selected(self):
        records = self._lnx_selected_records()
        if not records:
            rec = self._lnx_current_record()
            records = [rec] if rec is not None else []

        paths = [
            Path(rec["mask_path"])
            for rec in records
            if rec is not None and rec.get("mask_path") is not None
        ]

        if not paths:
            QMessageBox.information(
                self,
                "Instance segmentation",
                "The selected item has no semantic *_mask.tif to run.\n\n"
                "Existing PCD results can already be viewed directly.",
            )
            return

        self._start_pipeline(paths, auto_open_editor=False)

    def run_all(self):
        paths = [
            Path(rec["mask_path"])
            for rec in self._lnx_records.values()
            if rec.get("mask_path") is not None
        ]
        if not paths:
            QMessageBox.information(
                self,
                "Instance segmentation",
                "No semantic *_mask.tif files are available to run.",
            )
            return
        self._start_pipeline(paths, auto_open_editor=False)

    def _result_path_for_current(self, mode="physical"):
        rec = self._lnx_current_record()
        if rec is None:
            return None

        if str(mode).lower() == "social":
            path = rec.get("pcd_social")
            if path is not None:
                return Path(path)
            return Path(rec["processed_dir"]) / f"{rec['stem']}_pcd_instance_social.txt"

        path = rec.get("pcd_physical")
        if path is not None:
            return Path(path)
        return Path(rec["processed_dir"]) / f"{rec['stem']}_pcd_instance_physical.txt"

    def _morphology_paths_for_current(self):
        rec = self._lnx_current_record()
        if rec is None:
            return {"physical": None, "social": None}

        return {
            "physical": (
                Path(rec["morphology_physical"])
                if rec.get("morphology_physical") is not None
                else Path(rec["processed_dir"])
                / f"{rec['stem']}_instance_morphology_physical.csv"
            ),
            "social": (
                Path(rec["morphology_social"])
                if rec.get("morphology_social") is not None
                else Path(rec["processed_dir"])
                / f"{rec['stem']}_instance_morphology_social.csv"
            ),
        }

    def open_output_folder(self):
        rec = self._lnx_current_record()
        if rec is None:
            return

        folder = Path(rec["processed_dir"])
        folder.mkdir(parents=True, exist_ok=True)

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", str(exc))

    def _update_controls(self):
        try:
            running = (
                self.pipeline_thread is not None
                and self.pipeline_thread.isRunning()
            )
        except RuntimeError:
            running = False
            self.pipeline_thread = None
            self.pipeline_worker = None

        rec = self._lnx_current_record() if hasattr(self, "_lnx_records") else None

        selected_records = (
            self._lnx_selected_records()
            if hasattr(self, "_lnx_records")
            else []
        )
        if not selected_records and rec is not None:
            selected_records = [rec]

        has_runnable_selected = any(
            r is not None and r.get("mask_path") is not None
            for r in selected_records
        )
        has_any_runnable = any(
            r.get("mask_path") is not None
            for r in getattr(self, "_lnx_records", {}).values()
        )

        display_path = self._lnx_existing_display_path(rec) if rec else None
        display_available = (
            display_path is not None and Path(display_path).exists()
        )

        self.run_btn.setEnabled(has_runnable_selected and not running)
        self.run_all_btn.setEnabled(has_any_runnable and not running)
        self.editor_btn.setEnabled(
            rec is not None
            and rec.get("pcd_physical") is not None
            and Path(rec["pcd_physical"]).exists()
            and not running
        )

        morphology_paths = self._morphology_paths_for_current() if rec else {}
        morphology_available = any(
            p is not None and Path(p).exists()
            for p in morphology_paths.values()
        )

        self.morphology_btn.setEnabled(morphology_available and not running)

        try:
            self.apply_edit_btn.setEnabled(
                rec is not None
                and rec.get("mask_path") is not None
                and not running
            )
        except Exception:
            pass

        self.output_btn.setEnabled(rec is not None)

    def _lnx_refresh_current_record(self):
        rec = self._lnx_current_record()
        if rec is None:
            return

        refreshed = _discover_record(
            rec.get("mask_path")
            or rec.get("image_path")
            or rec.get("source_path")
            or rec["base_dir"],
            forced_stem=rec["stem"],
        )
        if refreshed is not None:
            key = _record_key(refreshed)
            self._lnx_records[key] = refreshed
            self._lnx_rebuild_list(preferred_key=key)

    def _pipeline_file_finished(self, result):
        if original_pipeline_file_finished is not None:
            original_pipeline_file_finished(self, result)

        # Refresh the matching record as soon as outputs appear.
        try:
            stem = str(dict(result).get("stem", ""))
        except Exception:
            stem = ""

        if stem:
            for key, rec in list(self._lnx_records.items()):
                if rec["stem"] == stem and rec.get("mask_path") is not None:
                    refreshed = _discover_record(
                        rec["mask_path"], forced_stem=stem
                    )
                    if refreshed is not None:
                        self._lnx_records[key] = refreshed

    def _pipeline_finished(self, results):
        if original_pipeline_finished is not None:
            original_pipeline_finished(self, results)

        # Rebuild statuses after all outputs have been written.
        rec = self._lnx_current_record()
        preferred = _record_key(rec) if rec is not None else None

        refreshed_records = OrderedDict()
        for old in self._lnx_records.values():
            source = (
                old.get("mask_path")
                or old.get("image_path")
                or old.get("source_path")
                or old["base_dir"]
            )
            new = _discover_record(source, forced_stem=old["stem"])
            if new is not None:
                refreshed_records[_record_key(new)] = new

        self._lnx_records = refreshed_records
        self._lnx_rebuild_list(preferred_key=preferred)

    # Install methods.
    InstanceSegmentationWidget.__init__ = __init__
    InstanceSegmentationWidget._lnx_rebuild_list = _lnx_rebuild_list
    InstanceSegmentationWidget._lnx_add_input_paths = _lnx_add_input_paths
    InstanceSegmentationWidget._add_mask_paths = _add_mask_paths
    InstanceSegmentationWidget.open_masks = open_masks
    InstanceSegmentationWidget.open_folder = open_folder
    InstanceSegmentationWidget._lnx_record_from_item = _lnx_record_from_item
    InstanceSegmentationWidget._lnx_current_record = _lnx_current_record
    InstanceSegmentationWidget._lnx_selected_records = _lnx_selected_records
    InstanceSegmentationWidget._selected_paths = _selected_paths
    InstanceSegmentationWidget._current_mask = _current_mask
    InstanceSegmentationWidget._lnx_existing_display_path = _lnx_existing_display_path
    InstanceSegmentationWidget._selection_changed = _selection_changed
    InstanceSegmentationWidget.run_selected = run_selected
    InstanceSegmentationWidget.run_all = run_all
    InstanceSegmentationWidget._result_path_for_current = _result_path_for_current
    InstanceSegmentationWidget._morphology_paths_for_current = _morphology_paths_for_current
    InstanceSegmentationWidget.open_output_folder = open_output_folder
    InstanceSegmentationWidget._update_controls = _update_controls
    InstanceSegmentationWidget._lnx_refresh_current_record = _lnx_refresh_current_record
    InstanceSegmentationWidget._pipeline_file_finished = _pipeline_file_finished
    InstanceSegmentationWidget._pipeline_finished = _pipeline_finished

    InstanceSegmentationWidget._linknex_directload_patched = True


# =============================================================================
# Module 4 patch
# =============================================================================

def _patch_module4(OrganelleNetworkAnalysisWidget):
    if getattr(OrganelleNetworkAnalysisWidget, "_linknex_directload_patched", False):
        return

    original_init = OrganelleNetworkAnalysisWidget.__init__

    def __init__(self, parent=None):
        original_init(self, parent)

        try:
            self.folder_edit.setPlaceholderText(
                "Drop image / PCD here, or choose processed_output"
            )
        except Exception:
            pass

        try:
            self.status_label.setText(
                "Drag an image, TXT/PCD, or existing network result here. "
                "LinkODT will locate processed_output automatically."
            )
        except Exception:
            pass

        _install_drop_filter(
            self,
            self._lnx_add_input_paths,
            getattr(self, "image_list", None),
            getattr(self, "viewer", None),
        )

    def _valid_stems(folder):
        """
        Include BOTH:
          - samples ready for network analysis
          - samples that already have network result(s)

        This is the key change that makes existing result browsing independent
        from rerunning Module 4.
        """
        folder = Path(folder)
        stems = set()

        patterns = (
            "*_skeleton_edges.txt",
            "*_skeleton_instance_social.txt",
            "*_pcd_instance_social.txt",
            "*_pcd_instance_physical.txt",
            "*_hierarchical_network.pkl",
            "*_hierarchical_network_metrics.csv",
        )

        for pattern in patterns:
            for p in folder.glob(pattern):
                stem = _stem_from_any(p)
                if stem:
                    stems.add(stem)

        return sorted(stems, key=_natural_key)

    def _lnx_network_input_ready(folder, stem):
        folder = Path(folder)
        required = (
            folder / f"{stem}_skeleton_edges.txt",
            folder / f"{stem}_skeleton_instance_social.txt",
            folder / f"{stem}_pcd_instance_social.txt",
        )
        return all(p.exists() for p in required)

    def _lnx_add_input_paths(self, paths):
        """
        Drop handler:
        - image/mask in dataset dir -> dataset/processed_output
        - TXT/PCD/PKL/CSV already in processed_output -> use that folder directly
        - folder -> use folder or folder/processed_output
        """
        candidates = []

        for raw in paths:
            p = Path(raw)
            if not p.exists():
                continue

            if p.is_dir():
                if p.name.lower() == "processed_output":
                    candidates.append(p)
                elif (p / "processed_output").is_dir():
                    candidates.append(p / "processed_output")
                else:
                    candidates.append(p)
                continue

            base_dir, processed_dir = _resolve_base_and_processed(p)

            # If a result/PCD itself is in processed_output, this is exact.
            if p.parent.name.lower() == "processed_output":
                candidates.append(p.parent)
            elif processed_dir.is_dir():
                candidates.append(processed_dir)
            else:
                # Permit same-folder result layout as a fallback.
                candidates.append(p.parent)

        if not candidates:
            self.status_label.setText("No supported LinkODT input was found.")
            return

        # Module 4's backend is directory-based. If several files from the same
        # dataset are dropped, this collapses to one processed_output.
        unique = []
        seen = set()
        for folder in candidates:
            key = str(Path(folder).resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(Path(folder))

        if len(unique) > 1:
            QMessageBox.information(
                self,
                "Network analysis",
                "The dropped files belong to more than one result folder.\n\n"
                "Module 4 displays one processed_output at a time. "
                "The first folder will be opened.",
            )

        self.set_processed_dir(unique[0], auto_scan=True)

    def scan_folder(self):
        text = self.folder_edit.text().strip()
        if not text:
            self.status_label.setText(
                "Drop an image/PCD or select a processed_output folder first."
            )
            return

        folder = Path(text)

        if folder.name.lower() != "processed_output":
            if (folder / "processed_output").is_dir():
                folder = folder / "processed_output"

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self,
                "Network analysis",
                "The selected/result folder does not exist.",
            )
            return

        self.processed_dir = folder
        self.folder_edit.setText(str(folder))
        self.image_stems = self._valid_stems(folder)

        old_stem = self._current_stem()

        self.image_list.blockSignals(True)
        self.image_list.clear()

        select_row = -1

        for row, stem in enumerate(self.image_stems):
            pkl_path = folder / f"{stem}_hierarchical_network.pkl"
            metrics_path = folder / f"{stem}_hierarchical_network_metrics.csv"
            ready = self._lnx_network_input_ready(folder, stem)

            if pkl_path.exists() and metrics_path.exists():
                state = "✓ Network"
            elif pkl_path.exists() or metrics_path.exists():
                state = "◐ Partial result"
            elif ready:
                state = "Ready"
            else:
                state = "Missing inputs"

            item = QListWidgetItem(f"{stem}    [{state}]")
            item.setData(Qt.UserRole, stem)
            item.setToolTip(str(folder))
            self.image_list.addItem(item)

            if stem == old_stem:
                select_row = row

        self.image_list.blockSignals(False)

        if self.image_stems:
            if select_row < 0:
                select_row = 0
            self.image_list.setCurrentRow(select_row)
            self.preview_current_result()
        else:
            self.viewer.clear(
                "No Module 3 input or existing Module 4 result was found."
            )
            self.metrics_model.set_rows([])

        n_done = sum(
            (
                (folder / f"{stem}_hierarchical_network.pkl").exists()
                or (folder / f"{stem}_hierarchical_network_metrics.csv").exists()
            )
            for stem in self.image_stems
        )
        n_ready = sum(
            self._lnx_network_input_ready(folder, stem)
            for stem in self.image_stems
        )

        self.status_label.setText(
            f"Found {len(self.image_stems)} sample(s): "
            f"{n_done} existing network result(s), "
            f"{n_ready} sample(s) with runnable Module 4 inputs."
        )
        self._update_controls()

    def run_selected(self):
        stems = self._selected_stems()
        if not stems:
            current = self._current_stem()
            stems = [current] if current else []

        if self.processed_dir is None:
            return self._start_analysis([])

        ready = [
            stem
            for stem in stems
            if self._lnx_network_input_ready(self.processed_dir, stem)
        ]

        if not ready:
            QMessageBox.information(
                self,
                "Network analysis",
                "The selected item has no complete Module 4 input set.\n\n"
                "If an existing network result is present, it can already be "
                "viewed directly without rerunning.",
            )
            return

        self._start_analysis(ready)

    def run_all(self):
        if self.processed_dir is None:
            return

        ready = [
            stem
            for stem in self.image_stems
            if self._lnx_network_input_ready(self.processed_dir, stem)
        ]

        if not ready:
            QMessageBox.information(
                self,
                "Network analysis",
                "No complete Module 4 input set was found.",
            )
            return

        self._start_analysis(ready)

    def preview_current_result(self):
        if self.processed_dir is None:
            return

        stem = self._current_stem()
        if not stem:
            return

        folder = Path(self.processed_dir)
        metrics_path = folder / f"{stem}_hierarchical_network_metrics.csv"
        pkl_path = folder / f"{stem}_hierarchical_network.pkl"

        if metrics_path.exists():
            try:
                import numpy as np
                import pandas as pd

                row = pd.read_csv(metrics_path).iloc[0].to_dict()
                metric_rows = []

                # Use the host LinkNex constant if available through the
                # method's module globals. If not, show all scalar columns.
                display_metrics = self.__class__.preview_current_result.__globals__.get(
                    "NETWORK_DISPLAY_METRICS", None
                )

                if display_metrics:
                    for layer, key, label in display_metrics:
                        if key in row:
                            metric_rows.append(
                                (layer, label, self._format_metric(row[key]))
                            )
                else:
                    for key, value in row.items():
                        if isinstance(value, (int, float, np.number)):
                            metric_rows.append(
                                ("Network", str(key), self._format_metric(value))
                            )

                self.metrics_model.set_rows(metric_rows)
            except Exception as exc:
                self.metrics_model.set_rows(
                    [("Result", "Load error", str(exc))]
                )
        else:
            self.metrics_model.set_rows([])

        if pkl_path.exists():
            try:
                self.viewer.load_pickle(pkl_path)
            except Exception as exc:
                self.viewer.clear(f"Could not load network PKL: {exc}")
        else:
            if self._lnx_network_input_ready(folder, stem):
                self.viewer.clear(
                    "No network PKL yet. This sample is ready to run."
                )
            else:
                self.viewer.clear(
                    "No existing network PKL. Required Module 3 inputs are incomplete."
                )

    def _update_controls(self):
        try:
            running = (
                self.worker_thread is not None
                and self.worker_thread.isRunning()
            )
        except RuntimeError:
            running = False
            self.worker_thread = None
            self.worker = None

        stem = self._current_stem() if self.image_stems else None

        current_ready = (
            stem is not None
            and self.processed_dir is not None
            and self._lnx_network_input_ready(self.processed_dir, stem)
        )

        selected = self._selected_stems()
        selected_ready = any(
            self._lnx_network_input_ready(self.processed_dir, s)
            for s in selected
        ) if self.processed_dir is not None else False

        any_ready = (
            self.processed_dir is not None
            and any(
                self._lnx_network_input_ready(self.processed_dir, s)
                for s in self.image_stems
            )
        )

        self.scan_btn.setEnabled(not running)
        self.folder_btn.setEnabled(not running)
        self.run_btn.setEnabled(
            (selected_ready or current_ready) and not running
        )
        self.run_all_btn.setEnabled(any_ready and not running)
        self.stop_btn.setEnabled(running)
        self.open_output_btn.setEnabled(
            self.processed_dir is not None
            and Path(self.processed_dir).exists()
        )

    OrganelleNetworkAnalysisWidget.__init__ = __init__
    OrganelleNetworkAnalysisWidget._valid_stems = staticmethod(_valid_stems)
    OrganelleNetworkAnalysisWidget._lnx_network_input_ready = staticmethod(
        _lnx_network_input_ready
    )
    OrganelleNetworkAnalysisWidget._lnx_add_input_paths = _lnx_add_input_paths
    OrganelleNetworkAnalysisWidget.scan_folder = scan_folder
    OrganelleNetworkAnalysisWidget.run_selected = run_selected
    OrganelleNetworkAnalysisWidget.run_all = run_all
    OrganelleNetworkAnalysisWidget._update_controls = _update_controls

    OrganelleNetworkAnalysisWidget._linknex_directload_patched = True


# =============================================================================
# Public entry
# =============================================================================

def apply_linknex_module34_patch(
    InstanceSegmentationWidget,
    OrganelleNetworkAnalysisWidget,
):
    """
    Apply the patch once, before LinkODTMainWindow is instantiated.
    """
    _patch_module3(InstanceSegmentationWidget)
    _patch_module4(OrganelleNetworkAnalysisWidget)

    print(
        "[LinkODT] Module 3/4 direct-load patch enabled: "
        "image/mask/PCD drag-drop + existing-result browsing."
    )

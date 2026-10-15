# LinkNex PCD Editor v8 - Standalone Full Instance Editor
"""
Interactive 3D point-cloud instance label editor for Windows.

Main functions
--------------
1. Open instance-labelled PCD/TXT point clouds.
2. Display the point cloud using its instance RGB colours.
3. Drag a screen rectangle to select every instance touched by any enclosed point;
   Ctrl+drag toggles additional instances. A short click selects one instance.
4. Merge selected instances. The first selected instance keeps its ID and colour.
5. Split one selected instance with a screen-space polygon. Points inside the
   polygon receive a new ID/colour; points outside keep the original ID/colour.
6. Save the edited point cloud as TXT (default) or PCD.

Supported input layouts
-----------------------
Headerless TXT/CSV:
    x y z r g b
    x y z r g b instance_id
    x y z nx ny nz r g b
    x y z nx ny nz r g b instance_id
    x y z r g b nx ny nz
    x y z r g b nx ny nz instance_id

ASCII PCD:
    Standard PCD fields with x/y/z, rgb or r/g/b, optional normals, and optional
    instance_id/label. Binary PCD is read through Open3D when installed.

The RGB colour is treated as the instance label when an explicit instance_id
field is absent. Normals can optionally orient GPU-instanced display discs;
they are preserved but are not changed by selection/editing.

Install once in the active Windows/PyCharm environment:
    pip install numpy PySide6 pyvista pyvistaqt vtk open3d

Run:
    python point_cloud_instance_label_editor.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import colorsys
import os
import sys
import base64

import numpy as np

# Embedded LinkNex branding assets; no external PNG files are required.

# Embedded LinkNex branding assets; no external PNG files are required.

# Embedded LinkNex branding assets; no external PNG files are required.

# Embedded LinkNex branding assets; no external PNG files are required.


# =============================================================================
# USER SETTINGS
# =============================================================================

WINDOW_TITLE = "LinkODT · 3D point-cloud instance editor"
BACKGROUND_COLOR = "#FCF9F5"
DEFAULT_POINT_SIZE = 3.0
SELECTED_INSTANCE_RGB = np.array([150, 150, 150], dtype=np.uint8)
MIN_SPLIT_POINTS = 10
GUI_PURPLE = "#552D84"
GUI_LAVENDER = "#AA80B1"
GUI_DARK_GREEN = "#103623"
GUI_COURT_GREEN = "#317140"
GUI_GRASS_GREEN = "#64AB57"
GUI_PALE_GREEN = "#A2CA7C"
GUI_CORAL = "#F6B7B0"
GUI_PEACH = "#F3CBB8"
GUI_WARM_YELLOW = "#FCDF9B"
SELECTION_HIGHLIGHT_COLOR = GUI_PURPLE

# The polygon is an extruded screen-space lasso: after the camera is positioned,
# all points of the selected instance whose screen projection lies inside the
# polygon are moved to the new instance.


# =============================================================================
# DATA STRUCTURES AND CORE EDITING LOGIC
# =============================================================================


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("RGB must have shape (N, 3).")
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size:
        finite_max = float(np.nanmax(rgb))
        if finite_max <= 1.0 + 1e-6:
            rgb = rgb * 255.0
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def _pack_rgb_uint32(rgb: np.ndarray) -> np.ndarray:
    rgb32 = _as_uint8_rgb(rgb).astype(np.uint32, copy=False)
    return (rgb32[:, 0] << 16) | (rgb32[:, 1] << 8) | rgb32[:, 2]


def _unpack_rgb_uint32(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint32)
    return np.column_stack(
        ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
    ).astype(np.uint8)


def _packed_float_to_rgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        packed = values.astype(np.uint32, copy=False)
    else:
        packed = np.asarray(values, dtype=np.float32).view(np.uint32)
    return _unpack_rgb_uint32(packed)


def _rgb_to_packed_float(rgb: np.ndarray) -> np.ndarray:
    packed = _pack_rgb_uint32(rgb).astype(np.uint32, copy=False)
    return packed.view(np.float32)


def _instance_ids_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Assign IDs 1..K in the order colours first appear."""
    packed = _pack_rgb_uint32(rgb)
    unique, first_index, inverse = np.unique(
        packed, return_index=True, return_inverse=True
    )
    first_order = np.argsort(first_index)
    unique_rank = np.empty(len(unique), dtype=np.int64)
    unique_rank[first_order] = np.arange(1, len(unique) + 1, dtype=np.int64)
    return unique_rank[inverse]


def _mode_colour(rgb: np.ndarray) -> Tuple[int, int, int]:
    packed = _pack_rgb_uint32(rgb)
    values, counts = np.unique(packed, return_counts=True)
    return tuple(int(x) for x in _unpack_rgb_uint32([values[np.argmax(counts)]])[0])


def _normalise_ids_and_colours(
    instance_ids: np.ndarray, rgb: np.ndarray
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]]]:
    instance_ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    rgb = _as_uint8_rgb(rgb)
    if len(instance_ids) != len(rgb):
        raise ValueError("instance_id and RGB lengths differ.")
    if np.any(instance_ids < 0):
        raise ValueError("instance_id values must be non-negative integers.")

    # ID 0 is allowed when it is explicitly present; it is treated as an
    # editable instance rather than hidden background because PCD contains only
    # the exported point set.
    colour_map: Dict[int, Tuple[int, int, int]] = {}
    order = np.argsort(instance_ids, kind="stable")
    sorted_ids = instance_ids[order]
    sorted_rgb = rgb[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sorted_ids)]
    for start, end in zip(starts, ends):
        instance_id = int(sorted_ids[start])
        colour_map[instance_id] = _mode_colour(sorted_rgb[start:end])
    return instance_ids, colour_map


def _new_distinct_colour(
    new_id: int, existing: Iterable[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    existing_array = np.asarray(list(existing), dtype=float)
    golden_ratio = 0.618033988749895
    for attempt in range(256):
        hue = (new_id * golden_ratio + attempt * 0.071) % 1.0
        saturation = 0.72 + 0.18 * ((attempt % 3) / 2.0)
        value = 0.96
        colour = np.asarray(colorsys.hsv_to_rgb(hue, saturation, value)) * 255.0
        colour = np.rint(colour).astype(np.uint8)
        if existing_array.size == 0:
            return tuple(int(x) for x in colour)
        distances = np.linalg.norm(existing_array - colour[None, :], axis=1)
        if float(np.min(distances)) >= 70.0:
            return tuple(int(x) for x in colour)
    return tuple(int(x) for x in colour)



class PointCloudDocument:
    def __init__(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        rgb: np.ndarray,
        instance_ids: Optional[np.ndarray],
        source_path: Optional[Path] = None,
        txt_group_order: str = "normals_rgb",
        txt_delimiter: str = " ",
        txt_has_header: bool = True,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("XYZ points must have shape (N, 3).")
        if len(self.points) == 0:
            raise ValueError("The point cloud is empty.")
        if not np.isfinite(self.points).all():
            raise ValueError("XYZ contains NaN or infinite values.")

        if normals is None:
            self.normals = None
        else:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.shape != self.points.shape:
                raise ValueError("Normals must have the same shape as XYZ.")
            self.normals = normals

        rgb = _as_uint8_rgb(rgb)
        if len(rgb) != len(self.points):
            raise ValueError("RGB and XYZ lengths differ.")
        if instance_ids is None:
            instance_ids = _instance_ids_from_rgb(rgb)
        self.instance_ids, self.colour_map = _normalise_ids_and_colours(
            instance_ids, rgb
        )
        # Preserve every input RGB value exactly.  GUI colours are entirely
        # separate and are never written into the point-cloud data.
        self.point_rgb = rgb.copy()

        self.source_path = Path(source_path) if source_path else None
        self.txt_group_order = txt_group_order
        self.txt_delimiter = txt_delimiter
        self.txt_has_header = bool(txt_has_header)
        self.selected_ids: List[int] = []
        self.undo_stack = []  # compatibility only; Undo is disabled
        self.redo_stack = []  # compatibility only; Redo is disabled

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def unique_ids(self) -> np.ndarray:
        return np.unique(self.instance_ids)

    @property
    def n_instances(self) -> int:
        return int(len(self.unique_ids))

    def display_rgb(self) -> np.ndarray:
        return self.point_rgb

    def indices_for_ids(self, ids: Sequence[int]) -> np.ndarray:
        if not ids:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.isin(self.instance_ids, np.asarray(ids, dtype=np.int64)))

    def merge_selected(self) -> Optional[str]:
        selected = [int(x) for x in self.selected_ids if x in self.colour_map]
        if len(selected) < 2:
            return None

        target_id = selected[0]
        changed = self.indices_for_ids(selected)
        target_colour = np.asarray(self.colour_map[target_id], dtype=np.uint8)

        self.instance_ids[changed] = target_id
        self.point_rgb[changed] = np.repeat(
            target_colour[None, :], len(changed), axis=0
        )
        for old_id in selected[1:]:
            self.colour_map.pop(old_id, None)
        self.selected_ids = [target_id]

        return f"Merge {selected} -> {target_id}"

    def split_instance(self, original_id: int, selected_indices: np.ndarray) -> Optional[str]:
        original_id = int(original_id)
        selected_indices = np.unique(np.asarray(selected_indices, dtype=np.int64))
        valid = selected_indices[
            (selected_indices >= 0)
            & (selected_indices < self.n_points)
            & (self.instance_ids[selected_indices] == original_id)
        ]
        total_original = int(np.count_nonzero(self.instance_ids == original_id))
        if len(valid) < MIN_SPLIT_POINTS:
            return None
        if total_original - len(valid) < MIN_SPLIT_POINTS:
            return None

        new_id = int(np.max(self.instance_ids)) + 1
        new_colour = _new_distinct_colour(new_id, self.colour_map.values())
        new_colour_array = np.asarray(new_colour, dtype=np.uint8)

        self.instance_ids[valid] = new_id
        self.point_rgb[valid] = np.repeat(
            new_colour_array[None, :], len(valid), axis=0
        )
        self.colour_map[new_id] = new_colour
        self.selected_ids = [original_id, new_id]

        return (
            f"Split instance {original_id}: {len(valid)} points -> new ID {new_id}"
        )



def selection_preview_rgb(
    document: PointCloudDocument, selected_ids: Sequence[int]
) -> np.ndarray:
    """Build grey GUI preview colours without mutating stored point RGB."""
    visual_rgb = document.display_rgb().copy()
    indices = document.indices_for_ids(selected_ids)
    visual_rgb[indices] = SELECTED_INSTANCE_RGB
    return visual_rgb


# =============================================================================
# FILE INPUT
# =============================================================================


def _parse_ascii_pcd(path: Path) -> PointCloudDocument:
    header_lines: List[str] = []
    data_line_index = None
    data_mode = None
    with path.open("rb") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.upper().startswith("DATA "):
                data_mode = line.split(maxsplit=1)[1].lower()
                data_line_index = line_index + 1
                break

    if data_line_index is None:
        raise ValueError("PCD header has no DATA line.")
    if data_mode != "ascii":
        raise ValueError(f"PCD DATA mode is {data_mode}, not ascii.")

    fields: List[str] = []
    for line in header_lines:
        parts = line.split()
        if parts and parts[0].upper() in {"FIELDS", "FIELD"}:
            fields = [x.lower() for x in parts[1:]]
            break
    if not fields:
        raise ValueError("PCD header has no FIELDS line.")

    data = np.loadtxt(path, comments="#", skiprows=data_line_index, ndmin=2)
    if data.shape[1] != len(fields):
        raise ValueError(
            f"PCD FIELDS has {len(fields)} columns but data has {data.shape[1]}."
        )
    index = {name: i for i, name in enumerate(fields)}
    if not all(name in index for name in ("x", "y", "z")):
        raise ValueError("PCD must contain x, y and z fields.")
    points = data[:, [index["x"], index["y"], index["z"]]]

    normal_names = ("normal_x", "normal_y", "normal_z")
    normals = (
        data[:, [index[x] for x in normal_names]]
        if all(x in index for x in normal_names)
        else None
    )

    if all(name in index for name in ("r", "g", "b")):
        rgb = data[:, [index["r"], index["g"], index["b"]]]
    elif "rgb" in index:
        rgb = _packed_float_to_rgb(data[:, index["rgb"]])
    elif "rgba" in index:
        rgb = _packed_float_to_rgb(data[:, index["rgba"]])
    else:
        raise ValueError("PCD must contain rgb/rgba or separate r, g and b fields.")

    id_name = next(
        (name for name in ("instance_id", "instance", "label", "id") if name in index),
        None,
    )
    instance_ids = (
        np.rint(data[:, index[id_name]]).astype(np.int64) if id_name else None
    )
    return PointCloudDocument(points, normals, rgb, instance_ids, path)


def _read_binary_pcd_with_open3d(path: Path) -> PointCloudDocument:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Binary PCD requires Open3D. Install it with: pip install open3d"
        ) from exc
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    colours = np.asarray(cloud.colors, dtype=np.float64)
    normals_array = np.asarray(cloud.normals, dtype=np.float64)
    normals = normals_array if normals_array.shape == points.shape else None
    if colours.shape != points.shape:
        raise ValueError("Open3D could not read RGB colours from this PCD.")
    return PointCloudDocument(points, normals, colours, None, path)


def _first_content_line(path: Path) -> Tuple[str, int]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped, line_index
    raise ValueError("The file contains no data.")


def _infer_rgb_and_normals(
    first_triplet: np.ndarray, second_triplet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Infer whether two unnamed 3-column groups are RGB/normal or normal/RGB.

    Instance colours are normally non-negative, repeated within an instance,
    and often integer-valued in 0..255. Surface normals normally lie in -1..1
    and have vector length close to one. A representative sample is used so
    the test remains fast for large point clouds.
    """

    first_triplet = np.asarray(first_triplet, dtype=np.float64)
    second_triplet = np.asarray(second_triplet, dtype=np.float64)
    sample_limit = 50000

    def sample(values: np.ndarray) -> np.ndarray:
        if len(values) <= sample_limit:
            return values
        step = max(1, len(values) // sample_limit)
        return values[::step][:sample_limit]

    a = sample(first_triplet)
    b = sample(second_triplet)

    def normal_score(values: np.ndarray) -> float:
        finite = np.isfinite(values).all(axis=1)
        if not np.any(finite):
            return -100.0
        values = values[finite]
        within = float(np.mean(np.abs(values) <= 1.05))
        lengths = np.linalg.norm(values, axis=1)
        unit = float(np.mean(np.abs(lengths - 1.0) <= 0.15))
        has_signed_values = min(1.0, float(np.mean(values < -1e-4)) * 10.0)
        return 2.5 * within + 2.5 * unit + 0.5 * has_signed_values

    def rgb_score(values: np.ndarray) -> float:
        finite = np.isfinite(values).all(axis=1)
        if not np.any(finite):
            return -100.0
        values = values[finite]
        in_range = float(np.mean((values >= -1e-6) & (values <= 255.0 + 1e-6)))
        non_negative = float(np.mean(values >= -1e-6))
        integer_like = float(np.mean(np.abs(values - np.rint(values)) <= 1e-5))
        scale_bonus = 1.0 if float(np.nanmax(values)) > 1.5 else 0.0
        # Instance RGB rows repeat much more often than surface normals.
        unique_ratio = len(np.unique(values, axis=0)) / max(1, len(values))
        repeated = 1.0 - min(1.0, float(unique_ratio))
        return (
            1.5 * in_range
            + 1.5 * non_negative
            + scale_bonus * integer_like
            + repeated
        )

    first_is_rgb = rgb_score(a) + normal_score(b)
    second_is_rgb = rgb_score(b) + normal_score(a)
    difference = first_is_rgb - second_is_rgb
    if abs(difference) < 0.35:
        raise ValueError(
            "Could not reliably distinguish RGB columns from normal columns. "
            "Add a header using either 'x y z r g b nx ny nz instance_id' "
            "or 'x y z nx ny nz r g b instance_id'."
        )
    if difference > 0:
        return first_triplet, second_triplet, "rgb_normals"
    return second_triplet, first_triplet, "normals_rgb"


def _read_txt_or_csv(path: Path) -> PointCloudDocument:
    first_line, first_index = _first_content_line(path)
    delimiter = "," if "," in first_line else None
    tokens = [x for x in first_line.replace(",", " ").split() if x]

    def is_number(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    # Looking for alphabetic characters is not sufficient because valid
    # scientific notation contains "e" (for example 1.2e-05).
    has_header = not all(is_number(token) for token in tokens)
    skiprows = first_index + (1 if has_header else 0)
    data = np.loadtxt(
        path,
        delimiter=delimiter,
        comments="#",
        skiprows=skiprows,
        ndmin=2,
    )
    n_columns = data.shape[1]

    group_order = "normals_rgb"
    if has_header:
        names = [token.strip().lower() for token in tokens]
        if len(names) != n_columns:
            raise ValueError("TXT/CSV header and data column counts differ.")
        index = {name: i for i, name in enumerate(names)}
        if not all(name in index for name in ("x", "y", "z")):
            raise ValueError("TXT/CSV header must contain x, y and z.")
        points = data[:, [index["x"], index["y"], index["z"]]]
        normal_names = ("nx", "ny", "nz")
        if not all(name in index for name in normal_names):
            normal_names = ("normal_x", "normal_y", "normal_z")
        normals = (
            data[:, [index[x] for x in normal_names]]
            if all(x in index for x in normal_names)
            else None
        )
        if not all(name in index for name in ("r", "g", "b")):
            raise ValueError("TXT/CSV header must contain r, g and b.")
        rgb = data[:, [index["r"], index["g"], index["b"]]]
        if normals is not None:
            rgb_first = min(index[x] for x in ("r", "g", "b"))
            normal_first = min(index[x] for x in normal_names)
            group_order = "rgb_normals" if rgb_first < normal_first else "normals_rgb"
        id_name = next(
            (
                name
                for name in ("instance_id", "instance", "label", "id")
                if name in index
            ),
            None,
        )
        ids = np.rint(data[:, index[id_name]]).astype(np.int64) if id_name else None
    elif n_columns == 6:
        points, normals, rgb, ids = data[:, 0:3], None, data[:, 3:6], None
    elif n_columns == 7:
        points, normals, rgb = data[:, 0:3], None, data[:, 3:6]
        ids = np.rint(data[:, 6]).astype(np.int64)
    elif n_columns == 9:
        points, ids = data[:, 0:3], None
        rgb, normals, group_order = _infer_rgb_and_normals(
            data[:, 3:6], data[:, 6:9]
        )
    elif n_columns == 10:
        points = data[:, 0:3]
        rgb, normals, group_order = _infer_rgb_and_normals(
            data[:, 3:6], data[:, 6:9]
        )
        ids = np.rint(data[:, 9]).astype(np.int64)
    else:
        raise ValueError(
            "Headerless TXT/CSV must have 6, 7, 9 or 10 columns. "
            f"Found {n_columns}."
        )
    return PointCloudDocument(
        points,
        normals,
        rgb,
        ids,
        path,
        txt_group_order=group_order,
        txt_delimiter=delimiter or " ",
        txt_has_header=has_header,
    )


def load_point_cloud(path: Path) -> PointCloudDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pcd":
        try:
            return _parse_ascii_pcd(path)
        except ValueError as exc:
            if "not ascii" not in str(exc):
                raise
            return _read_binary_pcd_with_open3d(path)
    if path.suffix.lower() in {".txt", ".csv", ".xyz"}:
        return _read_txt_or_csv(path)
    raise ValueError("Supported input extensions: .pcd, .txt, .csv and .xyz")


# =============================================================================
# FILE OUTPUT
# =============================================================================


def save_ascii_pcd(document: PointCloudDocument, path: Path) -> None:
    path = Path(path)
    rgb = document.display_rgb()
    packed_rgb = _rgb_to_packed_float(rgb)
    ids = document.instance_ids.astype(np.int64)

    if document.normals is None:
        fields = "x y z rgb instance_id"
        sizes = "4 4 4 4 4"
        types = "F F F F I"
        counts = "1 1 1 1 1"
        arrays = [document.points, packed_rgb[:, None], ids[:, None]]
        fmt = ["%.9g", "%.9g", "%.9g", "%.9e", "%d"]
    else:
        fields = "x y z normal_x normal_y normal_z rgb instance_id"
        sizes = "4 4 4 4 4 4 4 4"
        types = "F F F F F F F I"
        counts = "1 1 1 1 1 1 1 1"
        arrays = [
            document.points,
            document.normals,
            packed_rgb[:, None],
            ids[:, None],
        ]
        fmt = ["%.9g"] * 6 + ["%.9e", "%d"]

    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f"FIELDS {fields}",
            f"SIZE {sizes}",
            f"TYPE {types}",
            f"COUNT {counts}",
            f"WIDTH {document.n_points}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {document.n_points}",
            "DATA ascii",
        ]
    )
    matrix = np.column_stack(arrays)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(header + "\n")
        np.savetxt(handle, matrix, fmt=fmt, delimiter=" ")


def save_txt(document: PointCloudDocument, path: Path) -> None:
    path = Path(path)
    rgb = document.display_rgb().astype(np.int64)
    ids = document.instance_ids.astype(np.int64)
    if document.normals is None:
        matrix = np.column_stack([document.points, rgb, ids])
        names = ["x", "y", "z", "r", "g", "b", "instance_id"]
        fmt = ["%.9g"] * 3 + ["%d"] * 4
    elif document.txt_group_order == "rgb_normals":
        matrix = np.column_stack([document.points, rgb, document.normals, ids])
        names = [
            "x", "y", "z", "r", "g", "b", "nx", "ny", "nz", "instance_id"
        ]
        fmt = ["%.9g"] * 3 + ["%d"] * 3 + ["%.9g"] * 3 + ["%d"]
    else:
        matrix = np.column_stack([document.points, document.normals, rgb, ids])
        names = [
            "x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "instance_id"
        ]
        fmt = ["%.9g"] * 6 + ["%d"] * 4
    delimiter = document.txt_delimiter
    header = delimiter.join(names) if document.txt_has_header else ""
    np.savetxt(
        path,
        matrix,
        fmt=fmt,
        delimiter=delimiter,
        header=header,
        comments="",
    )


def save_document_pair(document: PointCloudDocument, pcd_path: Path) -> Tuple[Path, Path]:
    pcd_path = Path(pcd_path).with_suffix(".pcd")
    txt_path = pcd_path.with_suffix(".txt")
    save_ascii_pcd(document, pcd_path)
    save_txt(document, txt_path)
    return pcd_path, txt_path


# =============================================================================
# POLYGON SELECTION GEOMETRY
# =============================================================================


def instance_ids_from_screen_rectangle(
    projected_xy: np.ndarray,
    visible: np.ndarray,
    instance_ids: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    click_radius: float = 9.0,
) -> List[int]:
    """Return whole-instance IDs touched by a screen rectangle or short click."""
    projected_xy = np.asarray(projected_xy, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if projected_xy.shape != (len(instance_ids), 2) or len(visible) != len(
        instance_ids
    ):
        raise ValueError("Projected points, visibility and instance IDs differ.")

    x0, y0 = start
    x1, y1 = end
    left, right = sorted((float(x0), float(x1)))
    bottom, top = sorted((float(y0), float(y1)))
    is_click = (right - left) < 4.0 and (top - bottom) < 4.0
    if is_click:
        valid_indices = np.flatnonzero(visible)
        if not len(valid_indices):
            return []
        delta = projected_xy[valid_indices] - np.asarray([float(x1), float(y1)])
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        nearest_local = int(np.argmin(squared_distance))
        if float(squared_distance[nearest_local]) > float(click_radius) ** 2:
            return []
        return [int(instance_ids[int(valid_indices[nearest_local])])]

    inside = (
        visible
        & (projected_xy[:, 0] >= left)
        & (projected_xy[:, 0] <= right)
        & (projected_xy[:, 1] >= bottom)
        & (projected_xy[:, 1] <= top)
    )
    if not np.any(inside):
        return []
    return [int(value) for value in np.unique(instance_ids[inside])]


def points_inside_polygon_2d(
    points_xy: np.ndarray, polygon_xy: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon test."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.shape[0] < 3:
        return np.zeros(len(points_xy), dtype=bool)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = np.zeros(len(points_xy), dtype=bool)
    x1, y1 = polygon[-1]
    epsilon = np.finfo(float).eps
    for x2, y2 in polygon:
        crosses = (y1 > y) != (y2 > y)
        x_intersection = (x2 - x1) * (y - y1) / (y2 - y1 + epsilon) + x1
        inside ^= crosses & (x < x_intersection)
        x1, y1 = x2, y2
    return inside


def project_world_to_display(
    points: np.ndarray, renderer, render_window
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world coordinates into VTK display pixel coordinates."""
    camera = renderer.GetActiveCamera()
    aspect = renderer.GetTiledAspectRatio()
    matrix_vtk = camera.GetCompositeProjectionTransformMatrix(aspect, 0.0, 1.0)
    matrix = np.array(
        [[matrix_vtk.GetElement(i, j) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    clip = homogeneous @ matrix.T
    valid = np.isfinite(clip).all(axis=1) & (np.abs(clip[:, 3]) > 1e-12)
    ndc = np.full((len(points), 3), np.nan, dtype=np.float64)
    ndc[valid] = clip[valid, :3] / clip[valid, 3, None]

    width, height = render_window.GetSize()
    vx0, vy0, vx1, vy1 = renderer.GetViewport()
    display = np.empty((len(points), 2), dtype=np.float64)
    display[:, 0] = (vx0 + (ndc[:, 0] + 1.0) * 0.5 * (vx1 - vx0)) * width
    display[:, 1] = (vy0 + (ndc[:, 1] + 1.0) * 0.5 * (vy1 - vy0)) * height
    visible = valid & (clip[:, 3] > 0)
    return display, visible


def qt_to_vtk_display_position(
    qt_x: float,
    qt_y: float,
    qt_width: int,
    qt_height: int,
    vtk_width: int,
    vtk_height: int,
) -> Tuple[int, int]:
    """Convert Qt logical pixels (top-left origin) to VTK render pixels.

    Qt mouse events use device-independent logical pixels, while VTK commonly
    reports the render-window size in physical pixels on a high-DPI display.
    Scaling by the two actual sizes keeps picking and screen overlays under the
    cursor at 100%, 125%, 150%, and other Windows display scales.
    """
    qt_width = max(1, int(qt_width))
    qt_height = max(1, int(qt_height))
    vtk_width = max(1, int(vtk_width))
    vtk_height = max(1, int(vtk_height))
    x = int(round(float(qt_x) * vtk_width / qt_width))
    y = int(round((qt_height - float(qt_y)) * vtk_height / qt_height))
    return (
        min(max(x, 0), vtk_width - 1),
        min(max(y, 0), vtk_height - 1),
    )


# =============================================================================
# GUI
# =============================================================================


def run_gui(initial_path=None) -> None:
    # This application is intentionally fixed to PySide6.  pyvistaqt imports
    # Qt through qtpy, so QT_API must be set before either package imports Qt.
    # PyQt5 may remain installed for labelme/voxelfuse; it is not used here.
    os.environ["QT_API"] = "pyside6"
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        import pyvista as pv
        from pyvistaqt import QtInteractor
        import vtk
    except ImportError as exc:
        print("Missing GUI dependency:", exc)
        print("Install with:")
        print("pip install numpy PySide6 pyvista pyvistaqt vtk open3d")
        raise SystemExit(1) from exc

    QT_LEFT_BUTTON = QtCore.Qt.MouseButton.LeftButton
    QT_RIGHT_BUTTON = QtCore.Qt.MouseButton.RightButton
    QT_CONTROL_MODIFIER = QtCore.Qt.KeyboardModifier.ControlModifier
    QT_SHIFT_MODIFIER = QtCore.Qt.KeyboardModifier.ShiftModifier
    QT_HORIZONTAL = QtCore.Qt.Orientation.Horizontal
    QT_TEXT_SELECTABLE = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    QT_WAIT_CURSOR = QtCore.Qt.CursorShape.WaitCursor
    QShortcut = QtGui.QShortcut

    pv.global_theme.background = BACKGROUND_COLOR

    class InteractiveView(QtInteractor):
        def __init__(self, parent=None, editor=None):
            super().__init__(parent=parent)
            self.editor = editor
            self.selection_drag_start: Optional[Tuple[int, int]] = None
            self.selection_drag_control = False
            self.right_pan_active = False
            self.setAcceptDrops(True)

        def _display_position(self, event) -> Tuple[int, int]:
            position = event.position()
            vtk_width, vtk_height = self.GetRenderWindow().GetSize()
            return qt_to_vtk_display_position(
                position.x(),
                position.y(),
                self.width(),
                self.height(),
                vtk_width,
                vtk_height,
            )

        def _send_vtk_mouse_event(self, event_name: str, event) -> None:
            """Forward a Qt position to VTK using render-window pixel units."""
            x, y = self._display_position(event)
            interactor = self.GetRenderWindow().GetInteractor()
            interactor.SetEventPosition(x, y)
            interactor.SetControlKey(
                int(bool(event.modifiers() & QT_CONTROL_MODIFIER))
            )
            interactor.SetShiftKey(
                int(bool(event.modifiers() & QT_SHIFT_MODIFIER))
            )
            interactor.InvokeEvent(event_name)

        def mousePressEvent(self, event):
            if (
                self.editor is not None
                and self.editor.mode == "navigate"
                and event.button() == QT_RIGHT_BUTTON
            ):
                # Map right-drag to VTK's middle-button interaction (pan).
                self.right_pan_active = True
                self._send_vtk_mouse_event("MiddleButtonPressEvent", event)
                event.accept()
                return
            if self.editor is not None and event.button() == QT_LEFT_BUTTON:
                x, y = self._display_position(event)
                control = bool(event.modifiers() & QT_CONTROL_MODIFIER)
                if self.editor.mode == "select":
                    self.selection_drag_start = (x, y)
                    self.selection_drag_control = control
                    self.editor.start_selection_rectangle(x, y)
                    event.accept()
                    return
                if self.editor.mode == "polygon":
                    self.editor.handle_view_click(x, y, control)
                    event.accept()
                    return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.right_pan_active:
                self._send_vtk_mouse_event("MouseMoveEvent", event)
                event.accept()
                return
            if (
                self.editor is not None
                and self.editor.mode == "select"
                and self.selection_drag_start is not None
            ):
                x, y = self._display_position(event)
                self.editor.update_selection_rectangle(x, y)
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self.right_pan_active and event.button() == QT_RIGHT_BUTTON:
                self._send_vtk_mouse_event("MiddleButtonReleaseEvent", event)
                self.right_pan_active = False
                event.accept()
                return
            if (
                self.editor is not None
                and event.button() == QT_LEFT_BUTTON
                and self.selection_drag_start is not None
            ):
                x, y = self._display_position(event)
                start = self.selection_drag_start
                control = self.selection_drag_control
                self.selection_drag_start = None
                self.selection_drag_control = False
                self.editor.finish_selection_rectangle(start, (x, y), control)
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def mouseDoubleClickEvent(self, event):
            if self.editor is not None and self.editor.mode == "polygon":
                self.editor.finish_polygon_split()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)

        def dragEnterEvent(self, event):
            if self.editor is not None and self.editor.can_accept_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dragMoveEvent(self, event):
            if self.editor is not None and self.editor.can_accept_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dropEvent(self, event):
            if self.editor is not None and self.editor.open_dropped_files(
                event.mimeData()
            ):
                event.acceptProposedAction()
                return
            event.ignore()

    class EditorWindow(QtWidgets.QMainWindow):
        def __init__(self, initial_path=None):
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            self.resize(1450, 900)
            self.setAcceptDrops(True)
            self.document: Optional[PointCloudDocument] = None
            self.mesh = None
            self.base_actor = None
            self.disc_actor = None
            self.disc_back_actor = None
            self.disc_mapper = None
            self.disc_source = None
            self.disc_normals_filter = None
            self.disc_base_radius = 1.0
            self.selection_rectangle_actor = None
            self.selection_rectangle_start: Optional[Tuple[int, int]] = None
            self.mode = "navigate"
            self.polygon_vertices: List[Tuple[float, float]] = []
            self.polygon_actor = None
            self.point_size = DEFAULT_POINT_SIZE
            self.viewport_background_mode = "solid"
            self.viewport_background_bottom = BACKGROUND_COLOR
            self.viewport_background_top = "#DDE8F7"
            self._build_ui()
            self._apply_wimbledon_style()
            self._install_shortcuts()
            self.set_mode("navigate")

            if initial_path is not None:
                initial_path = Path(initial_path)
                QtCore.QTimer.singleShot(
                    0,
                    lambda p=initial_path: self.load_path(p),
                )

        @property
        def renderer_vtk(self):
            renderer = self.viewer.renderer
            return getattr(renderer, "renderer", renderer)

        def _apply_wimbledon_style(self):
            """Apply the reference palette to GUI chrome only, never point RGB."""
            self.setStyleSheet(
                f"""
                QMainWindow, QWidget {{
                    background-color: {BACKGROUND_COLOR};
                    color: {GUI_PURPLE};
                    font-family: "Segoe UI"; font-size: 14px;
                }}
                QLabel#titleLabel {{
                    color: {GUI_PURPLE};
                    padding: 6px 2px 8px 2px;
                }}
                QFrame#viewerFrame {{
                    background: transparent;
                    border: 2px solid {GUI_PURPLE};
                    border-radius: 6px;
                }}
                QGroupBox {{
                    background-color: #FFFDFC;
                    border: 1px solid {GUI_LAVENDER};
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 8px;
                    font-weight: 600;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                    color: {GUI_PURPLE};
                    background-color: #FFFDFC;
                }}
                QComboBox {{
                    background-color: #FFFDFC;
                    color: {GUI_PURPLE};
                    border: 1px solid {GUI_LAVENDER};
                    border-radius: 5px;
                    padding: 5px 8px;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 22px;
                }}
                QPushButton {{
                    background-color: {GUI_COURT_GREEN};
                    color: white;
                    border: 1px solid {GUI_DARK_GREEN};
                    border-radius: 5px;
                    padding: 7px 9px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {GUI_GRASS_GREEN};
                }}
                QPushButton:pressed {{
                    background-color: {GUI_DARK_GREEN};
                }}
                QPushButton:checked {{
                    background-color: {GUI_PURPLE};
                    border-color: {GUI_PURPLE};
                }}
                QPushButton#saveButton {{
                    background-color: {GUI_PURPLE};
                    border-color: {GUI_PURPLE};
                }}
                QPushButton#saveButton:hover {{
                    background-color: {GUI_LAVENDER};
                }}
                QSlider::groove:horizontal {{
                    height: 6px;
                    background: {GUI_PALE_GREEN};
                    border-radius: 3px;
                }}
                QSlider::handle:horizontal {{
                    width: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                    background: {GUI_PURPLE};
                }}
                QStatusBar {{
                    background-color: {GUI_DARK_GREEN};
                    color: white;
                }}
                QMessageBox, QFileDialog {{
                    background-color: {BACKGROUND_COLOR};
                    color: {GUI_PURPLE};
                }}
                """
            )

        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)
            layout.setContentsMargins(4, 4, 4, 4)

            panel = QtWidgets.QWidget()
            panel.setFixedWidth(285)
            panel_layout = QtWidgets.QVBoxLayout(panel)

            self.logo_label = QtWidgets.QLabel()
            self.logo_label.setObjectName("logoLabel")
            self.logo_label.setFixedSize(255, 102)
            self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.logo_label.setStyleSheet("background: transparent; border: none;")

            # Reuse the EXACT same two external logo files as the Semantic
            # Segmentation module. No editor-specific or embedded logo is used.
            script_dir = Path(__file__).resolve().parent

            from ui_theme import ROOT as brand_root
            full_logo_candidates = [
                brand_root / "assets/linkodt_wordmark.png",
                script_dir / "linknex_logo_full_transparent.png",
                script_dir / "linknex_logo_transparent.png",
                script_dir / "linknex_biotech_network_logo(2).png",
            ]

            logo_pixmap = QtGui.QPixmap()

            for logo_path in full_logo_candidates:
                if logo_path.exists():
                    logo_pixmap = QtGui.QPixmap(
                        str(logo_path)
                    )

                    if not logo_pixmap.isNull():
                        break

            if not logo_pixmap.isNull():
                logo_pixmap = logo_pixmap.scaled(
                    247,
                    96,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.logo_label.setPixmap(
                    logo_pixmap
                )

            round_logo_candidates = [
                brand_root / "assets/linkodt_icon.png",
                script_dir / "linknex_logo_round_transparent.png",
                script_dir / "linknex_logo_icon_only.png",
                script_dir / "linknex_logo_icon_only(1).png",
                script_dir / "linknex_logo_round.png",
            ]

            small_icon_pixmap = QtGui.QPixmap()

            for icon_path in round_logo_candidates:
                if icon_path.exists():
                    small_icon_pixmap = QtGui.QPixmap(
                        str(icon_path)
                    )

                    if not small_icon_pixmap.isNull():
                        break

            if not small_icon_pixmap.isNull():
                app_icon = QtGui.QIcon(
                    small_icon_pixmap
                )

                self.setWindowIcon(
                    app_icon
                )

                app = QtWidgets.QApplication.instance()

                if app is not None:
                    app.setWindowIcon(
                        app_icon
                    )

            panel_layout.addWidget(self.logo_label)

            self.open_button = QtWidgets.QPushButton("Open PCD / TXT")
            self.open_button.clicked.connect(self.open_file)
            panel_layout.addWidget(self.open_button)

            mode_group = QtWidgets.QGroupBox("Interaction mode")
            mode_group.setFixedSize(255, 108)
            mode_layout = QtWidgets.QVBoxLayout(mode_group)
            self.navigate_button = QtWidgets.QPushButton("Navigate (V)")
            self.select_button = QtWidgets.QPushButton("Select instance (S)")
            self.navigate_button.setCheckable(True)
            self.select_button.setCheckable(True)
            self.navigate_button.clicked.connect(lambda: self.set_mode("navigate"))
            self.select_button.clicked.connect(lambda: self.set_mode("select"))
            mode_layout.addWidget(self.navigate_button)
            mode_layout.addWidget(self.select_button)
            panel_layout.addWidget(mode_group)

            edit_group = QtWidgets.QGroupBox("Instance editing")
            edit_group.setFixedSize(255, 196)
            edit_layout = QtWidgets.QVBoxLayout(edit_group)
            self.merge_button = QtWidgets.QPushButton("Merge selected (M)")
            self.merge_button.clicked.connect(self.merge_selected)
            self.polygon_button = QtWidgets.QPushButton("Polygon split (P)")
            self.polygon_button.clicked.connect(self.start_polygon_split)
            self.finish_polygon_button = QtWidgets.QPushButton("Finish polygon (Enter)")
            self.finish_polygon_button.clicked.connect(self.finish_polygon_split)
            self.cancel_polygon_button = QtWidgets.QPushButton("Cancel polygon (Esc)")
            self.cancel_polygon_button.clicked.connect(self.cancel_polygon)
            edit_layout.addWidget(self.merge_button)
            edit_layout.addWidget(self.polygon_button)
            edit_layout.addWidget(self.finish_polygon_button)
            edit_layout.addWidget(self.cancel_polygon_button)
            panel_layout.addWidget(edit_group)

            point_group = QtWidgets.QGroupBox("Display point size")
            point_group.setFixedSize(255, 108)
            point_layout = QtWidgets.QVBoxLayout(point_group)
            self.point_slider = QtWidgets.QSlider(QT_HORIZONTAL)
            self.point_slider.setRange(1, 12)
            self.point_slider.setValue(int(DEFAULT_POINT_SIZE))
            self.point_slider.valueChanged.connect(self.change_point_size)
            point_layout.addWidget(self.point_slider)
            self.disc_checkbox = QtWidgets.QCheckBox(
                "Normal-oriented discs (GPU)"
            )
            self.disc_checkbox.setChecked(True)
            self.disc_checkbox.setEnabled(False)
            self.disc_checkbox.setToolTip(
                "Render each point as a small disc whose plane is perpendicular "
                "to the stored normal vector. Requires normals."
            )
            self.disc_checkbox.toggled.connect(self.toggle_normal_discs)
            point_layout.addWidget(self.disc_checkbox)
            self.depth_shading_checkbox = QtWidgets.QCheckBox(
                "Depth shading "
            )
            self.depth_shading_checkbox.setChecked(True)
            self.depth_shading_checkbox.setEnabled(False)
            self.depth_shading_checkbox.setToolTip(
                "Add screen-space depth shading and stronger normal-based "
                "lighting. This changes display only, never saved RGB values."
            )
            self.depth_shading_checkbox.toggled.connect(
                self.toggle_depth_shading
            )
            point_layout.addWidget(self.depth_shading_checkbox)
            panel_layout.addWidget(point_group)

            background_group = QtWidgets.QGroupBox("3D background")
            background_group.setFixedSize(255, 142)
            background_layout = QtWidgets.QVBoxLayout(background_group)

            self.background_mode_combo = QtWidgets.QComboBox()
            self.background_mode_combo.addItem("Solid", "solid")
            self.background_mode_combo.addItem("Gradient", "gradient")
            self.background_mode_combo.currentIndexChanged.connect(
                self.background_mode_changed
            )
            background_layout.addWidget(self.background_mode_combo)

            background_colour_row = QtWidgets.QHBoxLayout()
            background_colour_row.addWidget(QtWidgets.QLabel("Base"))
            self.bg_bottom_button = QtWidgets.QPushButton("")
            self.bg_bottom_button.setFixedSize(46, 28)
            self.bg_bottom_button.setToolTip("Choose background/base colour")
            self.bg_bottom_button.clicked.connect(
                lambda: self.choose_background_color("bottom")
            )
            background_colour_row.addWidget(self.bg_bottom_button)
            background_colour_row.addSpacing(8)
            background_colour_row.addWidget(QtWidgets.QLabel("Top"))
            self.bg_top_button = QtWidgets.QPushButton("")
            self.bg_top_button.setFixedSize(46, 28)
            self.bg_top_button.setToolTip("Choose gradient top colour")
            self.bg_top_button.clicked.connect(
                lambda: self.choose_background_color("top")
            )
            background_colour_row.addWidget(self.bg_top_button)
            background_colour_row.addStretch(1)
            background_layout.addLayout(background_colour_row)
            panel_layout.addWidget(background_group)
            self.refresh_background_controls()

            self.info_label = QtWidgets.QLabel("")
            self.info_label.setVisible(False)

            self.selection_label = QtWidgets.QLabel("Selected: none")
            self.selection_label.setWordWrap(False)
            self.selection_label.setFixedHeight(24)
            self.selection_label.setMaximumWidth(255)
            panel_layout.addWidget(self.selection_label)

            panel_layout.addStretch(1)

            self.save_button = QtWidgets.QPushButton(
                "Save edited point cloud (Ctrl+S)"
            )
            self.save_button.setObjectName("saveButton")
            self.save_button.clicked.connect(self.save_as)
            panel_layout.addWidget(self.save_button)

            layout.addWidget(panel)
            # Persistent visible frame around the 3D viewport.
            # Border thickness is fixed; viewport size still follows the window.
            self.viewer_frame = QtWidgets.QFrame(central)
            self.viewer_frame.setObjectName("viewerFrame")
            self.viewer_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            viewer_frame_layout = QtWidgets.QVBoxLayout(self.viewer_frame)
            viewer_frame_layout.setContentsMargins(0, 0, 0, 0)
            viewer_frame_layout.setSpacing(0)

            self.viewer = InteractiveView(parent=self.viewer_frame, editor=self)
            # viewerFrame supplies the only visible viewport border.
            self.viewer.interactor.setStyleSheet("border: none;")
            self.viewer.interactor.setContentsMargins(0, 0, 0, 0)
            self.apply_view_background()
            self.viewer.add_axes()
            viewer_frame_layout.addWidget(self.viewer.interactor)
            layout.addWidget(self.viewer_frame, stretch=1)
            self.statusBar().showMessage("Open a PCD or TXT point cloud.")

        def _paint_background_swatch(self, button, colour: str):
            button.setStyleSheet(
                "QPushButton {"
                f"background-color: {colour};"
                "border: 1px solid #6F6480;"
                "border-radius: 5px;"
                "padding: 0px;"
                "}"
            )

        def refresh_background_controls(self):
            if not hasattr(self, "bg_bottom_button"):
                return
            gradient = self.viewport_background_mode == "gradient"
            self.bg_top_button.setEnabled(gradient)
            self._paint_background_swatch(
                self.bg_bottom_button, self.viewport_background_bottom
            )
            self._paint_background_swatch(
                self.bg_top_button, self.viewport_background_top
            )

        def background_mode_changed(self, _index=None):
            mode = self.background_mode_combo.currentData()
            self.viewport_background_mode = str(mode or "solid")
            self.refresh_background_controls()
            self.apply_view_background()

        def choose_background_color(self, which: str):
            if which == "top":
                current = self.viewport_background_top
            else:
                current = self.viewport_background_bottom
            colour = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(current), self, "3D background colour"
            )
            if not colour.isValid():
                return
            if which == "top":
                self.viewport_background_top = colour.name()
            else:
                self.viewport_background_bottom = colour.name()
            self.refresh_background_controls()
            self.apply_view_background()

        def apply_view_background(self):
            if not hasattr(self, "viewer") or self.viewer is None:
                return
            if self.viewport_background_mode == "gradient":
                self.viewer.set_background(
                    self.viewport_background_bottom,
                    top=self.viewport_background_top,
                )
            else:
                self.viewer.set_background(self.viewport_background_bottom)
            self.viewer.render()

        def _install_shortcuts(self):
            shortcuts = [
                ("V", lambda: self.set_mode("navigate")),
                ("S", lambda: self.set_mode("select")),
                ("M", self.merge_selected),
                ("P", self.start_polygon_split),
                ("Return", self.finish_polygon_split),
                ("Escape", self.cancel_polygon),
                ("Ctrl+Shift+Z", self.redo),
                ("Ctrl+S", self.save_as),
                ("Ctrl+O", self.open_file),
            ]
            self._shortcuts = []
            for sequence, callback in shortcuts:
                shortcut = QShortcut(QtGui.QKeySequence(sequence), self)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

        def set_mode(self, mode: str):
            if mode != "polygon":
                self.clear_polygon_overlay()
            if mode != "select":
                self.selection_rectangle_start = None
                self.clear_selection_rectangle(render=False)
            self.mode = mode
            self.navigate_button.setChecked(mode == "navigate")
            self.select_button.setChecked(mode == "select")
            if mode == "navigate":
                message = (
                    "Navigate mode: left-drag rotate, right-drag pan, wheel zoom."
                )
            elif mode == "select":
                message = (
                    "Select mode: drag a rectangle; any touched instance turns grey. "
                    "Ctrl+drag toggles more instances."
                )
            else:
                message = "Polygon mode: click vertices; Enter/double-click finishes; Esc cancels."
            self.statusBar().showMessage(message)

        def open_file(self):
            path_string, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open labelled point cloud",
                "",
                "Point clouds (*.pcd *.txt *.csv *.xyz);;All files (*)",
            )
            if not path_string:
                return
            self.load_path(Path(path_string))

        @staticmethod
        def _dropped_paths(mime_data) -> List[Path]:
            supported = {".pcd", ".txt", ".csv", ".xyz"}
            if not mime_data.hasUrls():
                return []
            return [
                Path(url.toLocalFile())
                for url in mime_data.urls()
                if url.isLocalFile()
                and Path(url.toLocalFile()).suffix.lower() in supported
            ]

        def can_accept_drop(self, mime_data) -> bool:
            return bool(self._dropped_paths(mime_data))

        def open_dropped_files(self, mime_data) -> bool:
            paths = self._dropped_paths(mime_data)
            if not paths:
                return False
            if len(paths) > 1:
                self.statusBar().showMessage(
                    f"Dropped {len(paths)} supported files; opening the first one."
                )
            self.load_path(paths[0])
            return True

        def dragEnterEvent(self, event):
            if self.can_accept_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dragMoveEvent(self, event):
            if self.can_accept_drop(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def dropEvent(self, event):
            if self.open_dropped_files(event.mimeData()):
                event.acceptProposedAction()
                return
            event.ignore()

        def load_path(self, path: Path):
            QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
            try:
                document = load_point_cloud(Path(path))
                self.document = document
                self.load_document_into_view()
                self.set_mode("navigate")
                self.statusBar().showMessage(f"Loaded {path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def load_document_into_view(self):
            self.viewer.clear()
            self.apply_view_background()
            self.viewer.add_axes()
            self.disc_actor = None
            self.disc_back_actor = None
            self.disc_mapper = None
            self.disc_source = None
            self.disc_normals_filter = None
            self.mesh = pv.PolyData(self.document.points)
            self.mesh.point_data["instance_rgb"] = self.document.display_rgb()
            if self.document.normals is not None:
                self.mesh.point_data["Normals"] = self.document.normals
            self.base_actor = self.viewer.add_mesh(
                self.mesh,
                scalars="instance_rgb",
                rgb=True,
                point_size=self.point_size,
                render_points_as_spheres=False,
                pickable=True,
                name="base_point_cloud",
            )
            has_normals = self.document.normals is not None
            self.disc_checkbox.setEnabled(has_normals)
            self.depth_shading_checkbox.setEnabled(has_normals)
            if has_normals:
                self._create_normal_disc_actor()
                self.toggle_normal_discs(self.disc_checkbox.isChecked())
                self.toggle_depth_shading(
                    self.depth_shading_checkbox.isChecked()
                )
            else:
                self.toggle_depth_shading(False)
            self.viewer.reset_camera()
            self.update_information()
            self.update_selection_overlay()
            self.viewer.render()

        def _current_disc_radius(self) -> float:
            return self.disc_base_radius * max(0.25, self.point_size / 3.0)

        def _create_normal_disc_actor(self):
            """Create GPU-instanced discs with disc normals aligned to point normals."""
            if self.document is None or self.document.normals is None:
                return
            extent = np.ptp(self.document.points, axis=0)
            diagonal = float(np.linalg.norm(extent))
            self.disc_base_radius = max(
                diagonal / max(1.0, np.sqrt(self.document.n_points)) * 0.34,
                1e-6,
            )

            self.disc_source = vtk.vtkRegularPolygonSource()
            self.disc_source.SetNumberOfSides(10)
            self.disc_source.SetCenter(0.0, 0.0, 0.0)
            # Direction-mode glyphs align the source x-axis to the supplied
            # vector, so a polygon perpendicular to x becomes perpendicular to
            # each point's normal after orientation.
            self.disc_source.SetNormal(1.0, 0.0, 0.0)
            self.disc_source.SetRadius(self._current_disc_radius())
            self.disc_source.GeneratePolygonOn()
            self.disc_source.GeneratePolylineOff()

            # vtkRegularPolygonSource defines the plane orientation, but an
            # explicit normals array is needed for reliable per-fragment
            # lighting after GPU glyph orientation on all VTK/OpenGL builds.
            self.disc_normals_filter = vtk.vtkPolyDataNormals()
            self.disc_normals_filter.SetInputConnection(
                self.disc_source.GetOutputPort()
            )
            self.disc_normals_filter.ComputePointNormalsOn()
            self.disc_normals_filter.ComputeCellNormalsOn()
            self.disc_normals_filter.ConsistencyOn()
            self.disc_normals_filter.SplittingOff()
            # vtkRegularPolygonSource's vertex winding makes the generated
            # shading normal point opposite to the direction used by the
            # glyph mapper in this pipeline. Flip only the generated display
            # normals so +input normal is the bright/outside face and -normal
            # is the darker/inside face. Stored normals remain untouched.
            self.disc_normals_filter.FlipNormalsOn()

            self.disc_mapper = vtk.vtkGlyph3DMapper()
            self.disc_mapper.SetInputData(self.mesh)
            self.disc_mapper.SetSourceConnection(
                self.disc_normals_filter.GetOutputPort()
            )
            self.disc_mapper.SetOrientationArray("Normals")
            self.disc_mapper.SetOrientationModeToDirection()
            self.disc_mapper.OrientOn()
            self.disc_mapper.SetScaleModeToNoDataScaling()
            self.disc_mapper.SetScalarModeToUsePointFieldData()
            self.disc_mapper.SelectColorArray("instance_rgb")
            self.disc_mapper.SetColorModeToDirectScalars()
            self.disc_mapper.ScalarVisibilityOn()

            self.disc_actor = vtk.vtkActor()
            self.disc_actor.SetMapper(self.disc_mapper)
            prop = self.disc_actor.GetProperty()
            prop.BackfaceCullingOn()
            prop.SetInterpolationToPhong()
            prop.LightingOn()
            # Low ambient light preserves the original hue while allowing the
            # stored normal direction to produce visible light/dark gradients.
            prop.SetAmbient(0.10)
            prop.SetDiffuse(0.90)
            prop.SetSpecular(0.18)
            prop.SetSpecularPower(24.0)

            # Render the -normal side as a second front-face-culled actor.
            # Unlike SetBackfaceProperty, this actor still receives the same
            # per-point scalar RGB, so each instance keeps its own hue.
            self.disc_back_actor = vtk.vtkActor()
            self.disc_back_actor.SetMapper(self.disc_mapper)
            back_prop = self.disc_back_actor.GetProperty()
            back_prop.FrontfaceCullingOn()
            back_prop.SetInterpolationToPhong()
            back_prop.LightingOn()
            back_prop.SetAmbient(0.08)
            back_prop.SetDiffuse(0.38)
            back_prop.SetSpecular(0.03)
            back_prop.SetSpecularPower(10.0)
            self.renderer_vtk.AddActor(self.disc_actor)
            self.renderer_vtk.AddActor(self.disc_back_actor)

        def toggle_normal_discs(self, checked: bool):
            if self.document is None:
                return
            if (
                self.document.normals is None
                or self.disc_actor is None
                or self.disc_back_actor is None
            ):
                if checked:
                    self.disc_checkbox.blockSignals(True)
                    self.disc_checkbox.setChecked(False)
                    self.disc_checkbox.blockSignals(False)
                if self.base_actor is not None:
                    self.base_actor.SetVisibility(True)
                return
            self.disc_actor.SetVisibility(bool(checked))
            self.disc_back_actor.SetVisibility(bool(checked))
            if self.base_actor is not None:
                self.base_actor.SetVisibility(not bool(checked))
            self.viewer.render()

        def toggle_depth_shading(self, checked: bool):
            """Toggle EDL-style screen-space depth cues without changing data."""
            if checked and self.document is not None and self.document.normals is not None:
                try:
                    self.viewer.enable_eye_dome_lighting()
                except Exception as exc:
                    self.depth_shading_checkbox.blockSignals(True)
                    self.depth_shading_checkbox.setChecked(False)
                    self.depth_shading_checkbox.blockSignals(False)
                    self.statusBar().showMessage(
                        f"Depth shading is unavailable in this OpenGL/VTK build: {exc}"
                    )
            else:
                try:
                    self.viewer.disable_eye_dome_lighting()
                except Exception:
                    pass
            self.viewer.render()

        def change_point_size(self, value):
            self.point_size = float(value)
            if self.base_actor is not None:
                self.base_actor.GetProperty().SetPointSize(self.point_size)
            if self.disc_source is not None:
                self.disc_source.SetRadius(self._current_disc_radius())
                self.disc_source.Modified()
            self.viewer.render()

        def start_selection_rectangle(self, x: int, y: int):
            self.selection_rectangle_start = (int(x), int(y))
            self._draw_selection_rectangle((int(x), int(y)), (int(x), int(y)))

        def update_selection_rectangle(self, x: int, y: int):
            if self.selection_rectangle_start is None:
                return
            self._draw_selection_rectangle(
                self.selection_rectangle_start, (int(x), int(y))
            )

        def _draw_selection_rectangle(
            self, start: Tuple[int, int], end: Tuple[int, int]
        ):
            self.clear_selection_rectangle(render=False)
            x0, y0 = start
            x1, y1 = end
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
            vtk_points = vtk.vtkPoints()
            for x, y in corners:
                vtk_points.InsertNextPoint(float(x), float(y), 0.0)
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(corners))
            for index in range(len(corners)):
                polyline.GetPointIds().SetId(index, index)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyline)
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.SetLines(cells)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(polydata)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.333, 0.176, 0.518)
            actor.GetProperty().SetLineWidth(2.5)
            self.renderer_vtk.AddViewProp(actor)
            self.selection_rectangle_actor = actor
            self.viewer.render()

        def clear_selection_rectangle(self, render: bool = True):
            if self.selection_rectangle_actor is not None:
                self.renderer_vtk.RemoveViewProp(self.selection_rectangle_actor)
                self.selection_rectangle_actor = None
            if render:
                self.viewer.render()

        def finish_selection_rectangle(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int],
            control: bool,
        ):
            self.selection_rectangle_start = None
            self.clear_selection_rectangle(render=False)
            if self.document is None:
                self.viewer.render()
                return

            projected, visible = project_world_to_display(
                self.document.points,
                self.renderer_vtk,
                self.viewer.render_window,
            )
            selected_ids = instance_ids_from_screen_rectangle(
                projected,
                visible,
                self.document.instance_ids,
                start,
                end,
                click_radius=max(9.0, self.point_size + 6.0),
            )

            if control:
                for instance_id in selected_ids:
                    if instance_id in self.document.selected_ids:
                        self.document.selected_ids.remove(instance_id)
                    else:
                        self.document.selected_ids.append(instance_id)
            else:
                self.document.selected_ids = selected_ids

            self.update_selection_overlay()
            if selected_ids:
                self.statusBar().showMessage(
                    f"Rectangle touched {len(selected_ids)} instance(s); "
                    f"{len(self.document.selected_ids)} currently selected."
                )
            else:
                self.statusBar().showMessage("No instance was touched by the selection.")

        def handle_view_click(self, x: int, y: int, control: bool):
            if self.document is None or self.mode != "polygon":
                return
            self.polygon_vertices.append((float(x), float(y)))
            self.update_polygon_overlay()
            self.statusBar().showMessage(
                f"Polygon vertices: {len(self.polygon_vertices)}. "
                "Enter/double-click to finish."
            )

        def update_selection_overlay(self):
            if self.document is None or self.mesh is None:
                return

            visual_rgb = selection_preview_rgb(
                self.document, self.document.selected_ids
            )
            if not self.document.selected_ids:
                self.selection_label.setText("Selected: none")
            else:
                ids_text = ", ".join(
                    str(x) for x in self.document.selected_ids[:20]
                )
                if len(self.document.selected_ids) > 20:
                    ids_text += ", ..."
                self.selection_label.setText(
                    f"Selected IDs ({len(self.document.selected_ids)}): {ids_text}"
                )

            self.mesh.point_data["instance_rgb"] = visual_rgb
            self.mesh.GetPointData().Modified()
            self.mesh.Modified()
            self.viewer.render()

        def refresh_colours_after_edit(self):
            if self.document is None or self.mesh is None:
                return
            self.mesh.point_data["instance_rgb"] = self.document.display_rgb()
            self.mesh.GetPointData().Modified()
            self.mesh.Modified()
            self.update_information()
            self.update_selection_overlay()
            self.viewer.render()

        def update_information(self):
            if self.document is None:
                self.info_label.setText("No point cloud loaded")
                return
            source = self.document.source_path.name if self.document.source_path else "Untitled"
            normals_text = "yes" if self.document.normals is not None else "no"
            self.info_label.setText(
                f"File: {source}\n"
                f"Points: {self.document.n_points:,}\n"
                f"Instances: {self.document.n_instances:,}\n"
                f"Normals: {normals_text}\n"
                f"Undo: {len(getattr(self.document, 'undo_stack', []))} | "
                f"Redo: {len(getattr(self.document, 'redo_stack', []))}"
            )

        def merge_selected(self):
            if self.document is None:
                return
            if len(self.document.selected_ids) < 2:
                QtWidgets.QMessageBox.information(
                    self, "Merge", "Select at least two instances."
                )
                return
            description = self.document.merge_selected()
            if description:
                self.refresh_colours_after_edit()
                self.statusBar().showMessage(description)

        def start_polygon_split(self):
            if self.document is None:
                return
            if len(self.document.selected_ids) != 1:
                QtWidgets.QMessageBox.information(
                    self,
                    "Polygon split",
                    "Select exactly one instance before starting polygon split.",
                )
                return
            self.polygon_vertices = []
            self.mode = "polygon"
            self.navigate_button.setChecked(False)
            self.select_button.setChecked(False)
            self.statusBar().showMessage(
                "Polygon mode: click at least three vertices around the part to "
                "become a new instance, then press Enter or double-click."
            )

        def update_polygon_overlay(self):
            self.clear_polygon_overlay(render=False, keep_vertices=True)
            if len(self.polygon_vertices) < 2:
                self.viewer.render()
                return
            vtk_points = vtk.vtkPoints()
            for x, y in self.polygon_vertices:
                vtk_points.InsertNextPoint(x, y, 0.0)
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(self.polygon_vertices))
            for i in range(len(self.polygon_vertices)):
                polyline.GetPointIds().SetId(i, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyline)
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.SetLines(cells)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(polydata)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.333, 0.176, 0.518)
            actor.GetProperty().SetLineWidth(3.0)
            self.renderer_vtk.AddViewProp(actor)
            self.polygon_actor = actor
            self.viewer.render()

        def clear_polygon_overlay(self, render=True, keep_vertices=False):
            if self.polygon_actor is not None:
                self.renderer_vtk.RemoveViewProp(self.polygon_actor)
                self.polygon_actor = None
            if not keep_vertices:
                self.polygon_vertices = []
            if render:
                self.viewer.render()

        def cancel_polygon(self):
            if self.mode == "polygon":
                self.clear_polygon_overlay()
                self.set_mode("select")
                self.statusBar().showMessage("Polygon split cancelled.")

        def finish_polygon_split(self):
            if self.document is None or self.mode != "polygon":
                return
            if len(self.polygon_vertices) < 3:
                QtWidgets.QMessageBox.information(
                    self, "Polygon split", "Draw at least three polygon vertices."
                )
                return
            if len(self.document.selected_ids) != 1:
                self.cancel_polygon()
                return

            original_id = int(self.document.selected_ids[0])
            indices = self.document.indices_for_ids([original_id])
            QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
            try:
                projected, visible = project_world_to_display(
                    self.document.points[indices],
                    self.renderer_vtk,
                    self.viewer.render_window,
                )
                inside = visible & points_inside_polygon_2d(
                    projected, self.polygon_vertices
                )
                split_indices = indices[inside]
                description = self.document.split_instance(original_id, split_indices)
                if description is None:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Polygon split",
                        f"The polygon must leave at least {MIN_SPLIT_POINTS} points "
                        "on both sides of the split.",
                    )
                else:
                    self.refresh_colours_after_edit()
                    self.statusBar().showMessage(description)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Polygon split failed", str(exc))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
                self.clear_polygon_overlay()
                self.set_mode("select")

        def undo(self):
            if self.document is None:
                return
            description = self.document.undo()
            if description:
                self.refresh_colours_after_edit()
                self.statusBar().showMessage(f"Undo: {description}")

        def redo(self):
            if self.document is None:
                return
            description = self.document.redo()
            if description:
                self.refresh_colours_after_edit()
                self.statusBar().showMessage(f"Redo: {description}")

        def save_as(self):
            if self.document is None:
                return
            if self.document.source_path:
                suggested = self.document.source_path.with_name(
                    self.document.source_path.stem + "_edited.txt"
                )
            else:
                suggested = Path("edited_point_cloud.txt")
            txt_filter = "TXT point cloud (*.txt)"
            pcd_filter = "PCD point cloud (*.pcd)"
            path_string, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save edited point cloud",
                str(suggested),
                f"{txt_filter};;{pcd_filter}",
                txt_filter,
            )
            if not path_string:
                return
            output_path = Path(path_string)
            if output_path.suffix.lower() not in {".txt", ".pcd"}:
                output_path = output_path.with_suffix(
                    ".pcd" if selected_filter == pcd_filter else ".txt"
                )
            QtWidgets.QApplication.setOverrideCursor(QT_WAIT_CURSOR)
            try:
                if output_path.suffix.lower() == ".pcd":
                    save_ascii_pcd(self.document, output_path)
                else:
                    save_txt(self.document, output_path)
                self.statusBar().showMessage(f"Saved: {output_path.name}")
                QtWidgets.QMessageBox.information(
                    self,
                    "Saved",
                    f"Saved edited file:\n{output_path}",
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def closeEvent(self, event):
            try:
                self.viewer.close()
            finally:
                event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from ui_theme import configure_application
    configure_application(app)
    window = EditorWindow(initial_path=initial_path)
    window.show()
    sys.exit(app.exec())


def main() -> None:
    initial_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_gui(initial_path=initial_path)


if __name__ == "__main__":
    main()

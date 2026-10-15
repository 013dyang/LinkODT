from pathlib import Path
import colorsys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def load_txt_array(path):
    """读取逗号分隔、无表头的 txt 数据。"""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


def split_stem_from_pcd(path):
    """从 xxx_pcd.txt 得到 xxx。"""
    suffix = "_pcd.txt"

    if not path.name.endswith(suffix):
        raise ValueError(f"Unsupported PCD file: {path.name}")

    return path.name[:-len(suffix)]


def find_pcd_files(processed_dir):
    """寻找 processed_output 下所有原始 *_pcd.txt。"""
    return sorted(
        processed_dir.glob("*_pcd.txt"),
        key=lambda p: p.name.lower(),
    )


def find_skeleton_path(processed_dir, stem):
    """
    兼容两种拼写：
        *_skelenton.txt
        *_skeleton.txt
    """
    candidate_paths = [
        processed_dir / f"{stem}_skelenton.txt",
        processed_dir / f"{stem}_skeleton.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    # 如果都不存在，返回当前旧代码使用的拼写，便于报错时看清预期路径。
    return candidate_paths[0]


def find_refined_skeleton_path(processed_dir, stem):
    """
    兼容两种拼写：
        *_refinedskelenton.txt
        *_refinedskeleton.txt
    """
    candidate_paths = [
        processed_dir / f"{stem}_refinedskelenton.txt",
        processed_dir / f"{stem}_refinedskeleton.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    return candidate_paths[0]


def print_file_information(label, path, data, column_format):
    """打印文件路径、shape和当前约定的列格式。"""
    print(f"{label} path   : {path}")
    print(f"{label} shape  : {data.shape}")
    print(f"{label} format : {column_format}")


def check_expected_columns(label, data, expected_columns):
    """只检查列数，不修改数据。"""
    actual_columns = data.shape[1]

    if actual_columns != expected_columns:
        print(
            f"[WARNING] {label} expected {expected_columns} columns, "
            f"but found {actual_columns} columns."
        )


def add_missing_refined_skeleton_from_small_pcd_components(
    pcd,
    refined_skeleton,
    dbscan_eps=2.0,
    dbscan_min_samples=10,
    min_component_points=20,
    max_component_points=1000,
    solid_grid_spacing=1.0,
    local_padding=3,
    surface_dilation_iterations=1,
    surface_closing_iterations=1,
):
    """
    逐个检查小 PCD instance 内部是否存在 refined skeleton。

    处理流程
    --------
    1. 对 PCD xyz 做 DBSCAN，得到一个个 PCD instance；
    2. 只检查指定点数范围内的小 PCD instance；
    3. 将每个小 instance 的空心PCD表面转换成局部三维binary mask；
    4. 封闭表面小缝并填充内部，得到实心占据范围；
    5. 检查该实心范围内是否存在 skeleton；
    6. 内部已有 skeleton：不处理；
    7. 内部没有 skeleton：添加该 PCD instance 的质心。

    输入
    ----
    pcd : ndarray, shape (N, >=3)
        PCD 数据，前三列为 x,y,z。

    refined_skeleton : ndarray, shape (M, >=4)
        refined skeleton 数据，前四列为 x,y,z,score。

    dbscan_eps : float
        DBSCAN 的邻域半径。

    dbscan_min_samples : int
        DBSCAN 的 min_samples。

    min_component_points, max_component_points : int
        只检查点数位于这个范围内的 PCD component：
            min_component_points <= component points < max_component_points

    solid_grid_spacing : float
        将局部PCD表面栅格化时的网格间距。当前坐标为voxel单位，默认1.0。

    local_padding : int
        局部三维网格四周增加的空白层数，避免封闭表面接触网格边界。

    surface_dilation_iterations : int
        PCD表面栅格化后轻微膨胀的次数，用于连接相邻表面采样点。

    surface_closing_iterations : int
        填充前进行binary closing的次数，用于封闭表面小缝。

    返回
    ----
    new_refined_skeleton : ndarray, shape (M + K, 4)
        补充后的 refined skeleton，格式为 x,y,z,score。

    added_points : ndarray, shape (K, 4)
        本次新添加的 x,y,z,score 骨架点。

    labels : ndarray, shape (N,)
        每个 PCD 点的 DBSCAN component label。

    component_information : list of dict
        被检查的小 component 信息。

    说明
    ----
    包围盒只用于建立每个PCD instance自己的局部三维网格。
    最终是否存在skeleton，由填充后的solid_mask决定，不由包围盒决定。

    新增骨架点的 xyz 为 component 的点云质心。
    新增点的 score 为质心到该 component 表面点云的最近距离，
    与当前 refined skeleton 第四列的距离型 score 保持一致。
    """
    pcd = np.asarray(pcd, dtype=np.float32)
    refined_skeleton = np.asarray(refined_skeleton, dtype=np.float32)

    if pcd.ndim != 2 or pcd.shape[1] < 3:
        raise ValueError("pcd must contain at least 3 columns: x,y,z")

    if solid_grid_spacing <= 0:
        raise ValueError("solid_grid_spacing must be larger than 0.")

    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 4:
        raise ValueError(
            "refined_skeleton must contain at least 4 columns: x,y,z,score"
        )

    if min_component_points > max_component_points:
        raise ValueError(
            "min_component_points cannot be larger than max_component_points."
        )

    pcd_xyz = pcd[:, :3]
    refined_skeleton_xyzi = refined_skeleton[:, :4].copy()
    skeleton_xyz = refined_skeleton_xyzi[:, :3]

    print("Running DBSCAN for small PCD components...")
    print(f"DBSCAN eps              : {dbscan_eps}")
    print(f"DBSCAN min_samples      : {dbscan_min_samples}")
    print(
        "Checked component size : "
        f"{min_component_points} to {max_component_points} points"
    )

    labels = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        n_jobs=-1,
    ).fit_predict(pcd_xyz)

    component_labels = [
        int(label)
        for label in np.unique(labels)
        if label != -1
    ]

    added_rows = []
    component_information = []
    checked_component_count = 0
    components_with_skeleton_count = 0
    components_without_skeleton_count = 0

    surface_structure = ndi.generate_binary_structure(
        rank=3,
        connectivity=2,
    )

    for component_label in component_labels:
        component_mask = labels == component_label
        component_xyz = pcd_xyz[component_mask]
        component_point_count = len(component_xyz)

        if component_point_count < min_component_points:
            continue

        if component_point_count >= max_component_points:
            continue

        checked_component_count += 1

        # 使用该PCD instance自己的坐标范围建立局部三维网格。
        grid_min_index = (
            np.floor(component_xyz.min(axis=0) / solid_grid_spacing)
            .astype(np.int64)
            - local_padding
        )
        grid_max_index = (
            np.ceil(component_xyz.max(axis=0) / solid_grid_spacing)
            .astype(np.int64)
            + local_padding
        )
        grid_shape = tuple(
            (grid_max_index - grid_min_index + 1).astype(np.int64)
        )

        surface_mask = np.zeros(grid_shape, dtype=bool)
        surface_indices = (
            np.rint(component_xyz / solid_grid_spacing).astype(np.int64)
            - grid_min_index[None, :]
        )
        surface_mask[
            surface_indices[:, 0],
            surface_indices[:, 1],
            surface_indices[:, 2],
        ] = True

        closed_surface_mask = surface_mask

        if surface_dilation_iterations > 0:
            closed_surface_mask = ndi.binary_dilation(
                closed_surface_mask,
                structure=surface_structure,
                iterations=surface_dilation_iterations,
            )

        if surface_closing_iterations > 0:
            closed_surface_mask = ndi.binary_closing(
                closed_surface_mask,
                structure=surface_structure,
                iterations=surface_closing_iterations,
            )

        solid_mask = ndi.binary_fill_holes(closed_surface_mask)
        solid_mask_point_count = int(np.sum(solid_mask))

        # 将所有refined skeleton转换到这个局部网格中。
        skeleton_indices = (
            np.rint(skeleton_xyz / solid_grid_spacing).astype(np.int64)
            - grid_min_index[None, :]
        )
        valid_skeleton_mask = np.all(
            (skeleton_indices >= 0)
            & (skeleton_indices < np.asarray(grid_shape)[None, :]),
            axis=1,
        )

        has_internal_skeleton = False

        if np.any(valid_skeleton_mask):
            valid_indices = skeleton_indices[valid_skeleton_mask]
            skeleton_inside_component = solid_mask[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ]
            has_internal_skeleton = bool(
                np.any(skeleton_inside_component)
            )

        if has_internal_skeleton:
            components_with_skeleton_count += 1
            component_information.append(
                {
                    "component_label": component_label,
                    "component_points": int(component_point_count),
                    "solid_grid_shape": grid_shape,
                    "solid_mask_points": solid_mask_point_count,
                    "status": "has_internal_skeleton",
                }
            )
            continue

        components_without_skeleton_count += 1

        centroid = component_xyz.mean(axis=0)
        component_tree = cKDTree(component_xyz)
        centroid_score, _ = component_tree.query(centroid, k=1)

        added_row = np.asarray(
            [
                centroid[0],
                centroid[1],
                centroid[2],
                centroid_score,
            ],
            dtype=np.float32,
        )
        added_rows.append(added_row)

        component_information.append(
            {
                "component_label": component_label,
                "component_points": int(component_point_count),
                "solid_grid_shape": grid_shape,
                "solid_mask_points": solid_mask_point_count,
                "status": "centroid_skeleton_added",
                "centroid": centroid.tolist(),
                "score": float(centroid_score),
            }
        )

    if len(added_rows) > 0:
        added_points = np.vstack(added_rows).astype(np.float32)
        new_refined_skeleton = np.vstack(
            [refined_skeleton_xyzi, added_points]
        ).astype(np.float32)
    else:
        added_points = np.empty((0, 4), dtype=np.float32)
        new_refined_skeleton = refined_skeleton_xyzi

    print(f"PCD components          : {len(component_labels)}")
    print(f"Small components checked: {checked_component_count}")
    print(f"Small with skeleton     : {components_with_skeleton_count}")
    print(f"Small without skeleton  : {components_without_skeleton_count}")
    print(f"Skeleton points before  : {len(refined_skeleton_xyzi)}")
    print(f"Skeleton points added   : {len(added_points)}")
    print(f"Skeleton points after   : {len(new_refined_skeleton)}")

    return (
        new_refined_skeleton,
        added_points,
        labels,
        component_information,
    )


def segment_refined_skeleton_edges(
    refined_skeleton,
    neighbor_radius=1.8,
    random_seed=42,
    prune_shortcut_edges=True,
):
    """
    将 refined skeleton 按照拓扑分叉点切分成边。

    输入
    ----
    refined_skeleton : ndarray, shape (N, >=3)
        refined skeleton 数据，前三列必须是 x,y,z。

    neighbor_radius : float
        两个 skeleton 点建立邻接关系的最大距离。
        与 refined skeleton 生成代码的 connect_radius 保持一致，第一版使用 1.8。

    random_seed : int
        控制边颜色起始位置，保证同一次设置下颜色可复现。

    prune_shortcut_edges : bool
        True 时使用相对邻域规则删除急弯两侧、同一骨架跨点之间的
        空间捷径，再计算 degree，减少弯折被误判为分叉。

    返回
    ----
    skeleton_edges : ndarray, shape (N, 7)
        列格式：x,y,z,r,g,b,edge_id。

        edge_id == 0：分叉点区域，即计算得到的 degree > 2 的点。
        edge_id >= 1：去掉分叉点后得到的各个连通骨架边。

    说明
    ----
    这一版只做拓扑边切分：
        1. 根据 refined skeleton 的 xyz 建立半径邻接图；
        2. 删除可由更近中间点替代的空间捷径；
        3. 计算每个点的 degree；
        4. 将 degree > 2 的点作为分叉点；
        5. 暂时移除分叉点；
        6. 剩余图的每个连通分量作为一条边。

    分叉点统一保存为红色、edge_id=0；不同边使用不同颜色。
    当前函数不进行 GNN 分类，也不判断某条边是否为错误粘连边。
    """
    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 3:
        raise ValueError(
            "refined_skeleton must be a 2D array with at least 3 columns."
        )

    xyz = refined_skeleton[:, :3].astype(np.float64, copy=False)
    point_count = len(xyz)

    if point_count == 0:
        return np.empty((0, 7), dtype=np.float64)

    # 使用 KD-tree 查找 neighbor_radius 内的邻居。
    # query_pairs 不包含点自身，因此 degree 就是邻接表长度。
    tree = cKDTree(xyz)
    neighbor_pairs = tree.query_pairs(
        r=neighbor_radius,
        output_type="ndarray",
    )

    candidate_adjacency = [set() for _ in range(point_count)]
    for point_a, point_b in neighbor_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        candidate_adjacency[point_a].add(point_b)
        candidate_adjacency[point_b].add(point_a)

    adjacency = [[] for _ in range(point_count)]
    kept_pair_count = 0

    for point_a, point_b in neighbor_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        keep_pair = True

        if prune_shortcut_edges:
            distance_ab = float(np.linalg.norm(xyz[point_a] - xyz[point_b]))
            common_neighbors = (
                candidate_adjacency[point_a]
                & candidate_adjacency[point_b]
            )

            # Relative-neighborhood pruning: 如果存在点 k，它到 a和b
            # 都比 a-b 更近，则 a-b 是跨过中间点的捷径。
            for point_k in common_neighbors:
                distance_ak = float(
                    np.linalg.norm(xyz[point_a] - xyz[point_k])
                )
                distance_bk = float(
                    np.linalg.norm(xyz[point_b] - xyz[point_k])
                )
                if (
                    distance_ak < distance_ab - 1e-8
                    and distance_bk < distance_ab - 1e-8
                ):
                    keep_pair = False
                    break

        if keep_pair:
            adjacency[point_a].append(point_b)
            adjacency[point_b].append(point_a)
            kept_pair_count += 1

    degree = np.asarray(
        [len(neighbors) for neighbors in adjacency],
        dtype=np.int32,
    )

    junction_mask = degree > 2
    edge_id = np.full(point_count, -1, dtype=np.int32)
    edge_id[junction_mask] = 0

    # 去掉分叉点后，对剩余骨架点进行连通分量搜索。
    current_edge_id = 0

    for start_index in range(point_count):
        if edge_id[start_index] != -1:
            continue

        current_edge_id += 1
        edge_id[start_index] = current_edge_id
        stack = [start_index]

        while stack:
            current_index = stack.pop()

            for neighbor_index in adjacency[current_index]:
                if junction_mask[neighbor_index]:
                    continue

                if edge_id[neighbor_index] == -1:
                    edge_id[neighbor_index] = current_edge_id
                    stack.append(neighbor_index)

    # 为不同 edge_id 生成稳定、容易区分的颜色。
    rgb = np.zeros((point_count, 3), dtype=np.int32)
    rgb[edge_id == 0] = [255, 0, 0]

    golden_ratio = 0.618033988749895
    hue_offset = (random_seed * golden_ratio) % 1.0

    for one_edge_id in range(1, current_edge_id + 1):
        hue = (hue_offset + one_edge_id * golden_ratio) % 1.0
        color = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        color = np.rint(np.asarray(color) * 255).astype(np.int32)
        rgb[edge_id == one_edge_id] = color

    skeleton_edges = np.column_stack(
        [
            xyz,
            rgb,
            edge_id,
        ]
    )

    print(f"Skeleton points       : {point_count}")
    print(f"Neighbor radius       : {neighbor_radius}")
    print(f"Candidate graph edges : {len(neighbor_pairs)}")
    print(f"Pruned graph edges    : {kept_pair_count}")
    print(f"Endpoint points       : {np.sum(degree == 1)}")
    print(f"Degree-2 points       : {np.sum(degree == 2)}")
    print(f"Junction points       : {np.sum(junction_mask)}")
    print(f"Segmented edges       : {current_edge_id}")

    return skeleton_edges


def _safe_numeric_features(values, prefix):
    """计算一组一维数值的稳定统计量。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    feature_names = [
        "mean",
        "median",
        "std",
        "cv",
        "min",
        "q10",
        "q90",
    ]

    if len(values) == 0:
        return {
            f"{prefix}_{name}": np.nan
            for name in feature_names
        }

    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    std_value = float(np.std(values))

    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_median": median_value,
        f"{prefix}_std": std_value,
        f"{prefix}_cv": std_value / (abs(mean_value) + 1e-8),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
    }


def _indices_grouped_by_positive_label(labels):
    """将正整数label对应的数组下标分组。"""
    labels = np.asarray(labels, dtype=np.int32)
    valid_indices = np.where(labels > 0)[0]

    if len(valid_indices) == 0:
        return {}

    order = valid_indices[np.argsort(labels[valid_indices], kind="stable")]
    ordered_labels = labels[order]
    unique_labels, starts, counts = np.unique(
        ordered_labels,
        return_index=True,
        return_counts=True,
    )

    return {
        int(label): order[start:start + count]
        for label, start, count in zip(unique_labels, starts, counts)
    }


def _compute_edge_geometry(edge_xyz, neighbor_radius=1.8):
    """
    使用局部邻接图的最小生成树估计边长，并提取主路径、曲率和主轴。
    """
    edge_xyz = np.asarray(edge_xyz, dtype=np.float64)
    point_count = len(edge_xyz)

    if point_count == 0:
        raise ValueError("edge_xyz cannot be empty.")

    if point_count == 1:
        return {
            "ordered_local_indices": np.asarray([0], dtype=np.int32),
            "arc_length": 0.0,
            "chord_length": 0.0,
            "tortuosity": 1.0,
            "straightness": 1.0,
            "turn_angle_mean": 0.0,
            "turn_angle_std": 0.0,
            "turn_angle_max": 0.0,
            "pca_linearity": 0.0,
            "principal_axis": np.asarray([1.0, 0.0, 0.0]),
            "normalized_position": np.asarray([0.5]),
        }

    tree = cKDTree(edge_xyz)
    pairs = tree.query_pairs(
        r=neighbor_radius,
        output_type="ndarray",
    )

    if len(pairs) == 0:
        # 极少数不连续短边：使用最近邻顺序作为保底。
        remaining = set(range(1, point_count))
        ordered = [0]

        while remaining:
            current = ordered[-1]
            next_index = min(
                remaining,
                key=lambda idx: np.linalg.norm(
                    edge_xyz[idx] - edge_xyz[current]
                ),
            )
            ordered.append(next_index)
            remaining.remove(next_index)

        ordered = np.asarray(ordered, dtype=np.int32)
        segment_lengths = np.linalg.norm(
            np.diff(edge_xyz[ordered], axis=0),
            axis=1,
        )
        arc_length = float(np.sum(segment_lengths))
    else:
        pair_distances = np.linalg.norm(
            edge_xyz[pairs[:, 0]] - edge_xyz[pairs[:, 1]],
            axis=1,
        )
        graph = csr_matrix(
            (
                np.concatenate([pair_distances, pair_distances]),
                (
                    np.concatenate([pairs[:, 0], pairs[:, 1]]),
                    np.concatenate([pairs[:, 1], pairs[:, 0]]),
                ),
            ),
            shape=(point_count, point_count),
        )
        mst = minimum_spanning_tree(graph).tocoo()
        arc_length = float(np.sum(mst.data))

        adjacency = [[] for _ in range(point_count)]
        for point_a, point_b, weight in zip(mst.row, mst.col, mst.data):
            point_a = int(point_a)
            point_b = int(point_b)
            weight = float(weight)
            adjacency[point_a].append((point_b, weight))
            adjacency[point_b].append((point_a, weight))

        def farthest_from(start_index):
            distances = np.full(point_count, -np.inf, dtype=np.float64)
            parents = np.full(point_count, -1, dtype=np.int32)
            distances[start_index] = 0.0
            stack = [start_index]

            while stack:
                current = stack.pop()
                for neighbor, weight in adjacency[current]:
                    if neighbor == parents[current]:
                        continue
                    parents[neighbor] = current
                    distances[neighbor] = distances[current] + weight
                    stack.append(neighbor)

            farthest = int(np.nanargmax(distances))
            return farthest, distances, parents

        first_endpoint, _, _ = farthest_from(0)
        second_endpoint, _, parents = farthest_from(first_endpoint)

        path = [second_endpoint]
        while path[-1] != first_endpoint and parents[path[-1]] != -1:
            path.append(int(parents[path[-1]]))
        ordered = np.asarray(path[::-1], dtype=np.int32)

    ordered_xyz = edge_xyz[ordered]
    chord_length = float(
        np.linalg.norm(ordered_xyz[-1] - ordered_xyz[0])
    )
    tortuosity = arc_length / (chord_length + 1e-8)
    straightness = chord_length / (arc_length + 1e-8)

    turn_angles = []
    if len(ordered_xyz) >= 3:
        vector_a = ordered_xyz[1:-1] - ordered_xyz[:-2]
        vector_b = ordered_xyz[2:] - ordered_xyz[1:-1]
        norm_a = np.linalg.norm(vector_a, axis=1)
        norm_b = np.linalg.norm(vector_b, axis=1)
        valid = (norm_a > 1e-8) & (norm_b > 1e-8)

        if np.any(valid):
            cosine = np.sum(vector_a[valid] * vector_b[valid], axis=1)
            cosine = cosine / (norm_a[valid] * norm_b[valid])
            cosine = np.clip(cosine, -1.0, 1.0)
            turn_angles = np.degrees(np.arccos(cosine))

    if len(turn_angles) == 0:
        turn_angle_mean = 0.0
        turn_angle_std = 0.0
        turn_angle_max = 0.0
    else:
        turn_angle_mean = float(np.mean(turn_angles))
        turn_angle_std = float(np.std(turn_angles))
        turn_angle_max = float(np.max(turn_angles))

    centered = edge_xyz - edge_xyz.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(point_count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order_eigen = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order_eigen], 0.0)
    principal_axis = eigenvectors[:, order_eigen[0]]
    principal_axis = principal_axis / (
        np.linalg.norm(principal_axis) + 1e-8
    )
    pca_linearity = float(
        (eigenvalues[0] - eigenvalues[1])
        / (eigenvalues[0] + 1e-8)
    )

    projected_position = centered @ principal_axis
    projected_min = float(np.min(projected_position))
    projected_range = float(
        np.max(projected_position) - projected_min
    )

    if projected_range <= 1e-8:
        normalized_position = np.full(point_count, 0.5)
    else:
        normalized_position = (
            projected_position - projected_min
        ) / projected_range

    return {
        "ordered_local_indices": ordered,
        "arc_length": arc_length,
        "chord_length": chord_length,
        "tortuosity": float(tortuosity),
        "straightness": float(straightness),
        "turn_angle_mean": turn_angle_mean,
        "turn_angle_std": turn_angle_std,
        "turn_angle_max": turn_angle_max,
        "pca_linearity": pca_linearity,
        "principal_axis": principal_axis,
        "normalized_position": normalized_position,
    }


def _extract_first_shell_cross_section(
    center_xyz,
    tangent,
    candidate_global_indices,
    pcd_xyz_all,
    cross_section_half_thickness,
    cross_section_max_radius,
    angular_bins=36,
    points_per_angle_bin=2,
    radial_outlier_factor=1.75,
):
    """
    从候选PCD中提取当前骨架中心在法平面内看到的第一层外壳。

    每个角度分箱只保留径向距离最小的若干个点，并删除明显
    远于主要外壳的角度分箱，以排除相邻线粒体表面。
    """
    candidate_global_indices = np.asarray(
        candidate_global_indices,
        dtype=np.int64,
    )
    if len(candidate_global_indices) == 0:
        return None

    tangent = np.asarray(tangent, dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm <= 1e-8:
        return None
    tangent = tangent / tangent_norm

    # 构造法平面内的两个正交基向量。
    reference_axis = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference_axis, tangent))) > 0.90:
        reference_axis = np.asarray([0.0, 1.0, 0.0])
    plane_axis_u = np.cross(tangent, reference_axis)
    plane_axis_u = plane_axis_u / (np.linalg.norm(plane_axis_u) + 1e-8)
    plane_axis_v = np.cross(tangent, plane_axis_u)
    plane_axis_v = plane_axis_v / (np.linalg.norm(plane_axis_v) + 1e-8)

    candidate_xyz = pcd_xyz_all[candidate_global_indices]
    relative_xyz = candidate_xyz - center_xyz
    axial_distance = relative_xyz @ tangent
    radial_vectors = (
        relative_xyz - axial_distance[:, None] * tangent[None, :]
    )
    radial_distance = np.linalg.norm(radial_vectors, axis=1)

    slab_mask = (
        (np.abs(axial_distance) <= cross_section_half_thickness)
        & (radial_distance <= cross_section_max_radius)
        & (radial_distance > 1e-8)
    )
    if not np.any(slab_mask):
        return None

    slab_global_indices = candidate_global_indices[slab_mask]
    slab_radial_vectors = radial_vectors[slab_mask]
    slab_radial_distance = radial_distance[slab_mask]
    slab_axial_distance = axial_distance[slab_mask]

    coordinate_u = slab_radial_vectors @ plane_axis_u
    coordinate_v = slab_radial_vectors @ plane_axis_v
    angle = np.mod(np.arctan2(coordinate_v, coordinate_u), 2.0 * np.pi)
    angle_bin = np.floor(
        angle / (2.0 * np.pi) * angular_bins
    ).astype(np.int32)
    angle_bin = np.clip(angle_bin, 0, angular_bins - 1)

    selected_local_indices = []
    nearest_radius_by_bin = []
    selected_by_bin = {}

    for one_bin in np.unique(angle_bin):
        one_bin_indices = np.where(angle_bin == one_bin)[0]
        order = one_bin_indices[
            np.argsort(slab_radial_distance[one_bin_indices])
        ]
        retained = order[:points_per_angle_bin]
        selected_by_bin[int(one_bin)] = retained
        nearest_radius_by_bin.append(float(slab_radial_distance[order[0]]))

    if len(nearest_radius_by_bin) == 0:
        return None

    typical_first_shell_radius = float(np.median(nearest_radius_by_bin))
    maximum_first_shell_radius = (
        typical_first_shell_radius * radial_outlier_factor
    )

    retained_bin_count = 0
    for retained in selected_by_bin.values():
        nearest_radius = float(slab_radial_distance[retained[0]])
        if nearest_radius <= maximum_first_shell_radius + 1e-8:
            selected_local_indices.extend(retained.tolist())
            retained_bin_count += 1

    if len(selected_local_indices) == 0:
        return None

    selected_local_indices = np.asarray(
        selected_local_indices,
        dtype=np.int64,
    )
    selected_global_indices = slab_global_indices[selected_local_indices]
    selected_radial_distance = slab_radial_distance[selected_local_indices]

    # 保存时投影回零厚度法平面。
    selected_candidate_xyz = pcd_xyz_all[selected_global_indices]
    selected_axial_distance = slab_axial_distance[selected_local_indices]
    projected_xyz = (
        selected_candidate_xyz
        - selected_axial_distance[:, None] * tangent[None, :]
    )

    return {
        "global_indices": selected_global_indices,
        "projected_xyz": projected_xyz,
        "radial_distance": selected_radial_distance,
        "occupied_bin_count": retained_bin_count,
        "angular_coverage": retained_bin_count / float(angular_bins),
        "typical_first_shell_radius": typical_first_shell_radius,
    }


def _signed_offsets_from_pcd_surface(query_xyz, pcd, pcd_tree=None):
    """
    利用最近 PCD 表面点及其法向，判断 query 点位于表面内侧还是外侧。

    返回的 signed_offset > 0 表示位于 PCD 外侧，< 0 表示位于内侧。
    PCD 法向如果整体朝内，会根据绝大多数 skeleton 点应在内部这一事实
    自动整体翻转。该函数只服务于后置黄色 candidate 判断，不参与前面
    的截面尺度计算或红色 edge 判定。
    """
    query_xyz = np.asarray(query_xyz, dtype=np.float64)
    pcd = np.asarray(pcd, dtype=np.float64)

    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError(
            "The outside-PCD edge check requires PCD columns "
            "x,y,z,nx,ny,nz."
        )

    pcd_xyz = pcd[:, :3]
    pcd_normals = pcd[:, 3:6].copy()
    normal_length = np.linalg.norm(pcd_normals, axis=1)
    valid_normal = np.isfinite(normal_length) & (normal_length > 1e-8)
    if not np.any(valid_normal):
        raise ValueError(
            "The outside-PCD edge check found no valid PCD surface normals."
        )

    pcd_normals[valid_normal] /= normal_length[valid_normal, None]
    pcd_normals[~valid_normal] = np.nan

    if pcd_tree is None:
        pcd_tree = cKDTree(pcd_xyz)
    _, nearest_pcd_rows = pcd_tree.query(
        query_xyz,
        k=1,
        workers=-1,
    )

    nearest_surface_xyz = pcd_xyz[nearest_pcd_rows]
    nearest_surface_normal = pcd_normals[nearest_pcd_rows]
    signed_offset = np.einsum(
        "ij,ij->i",
        query_xyz - nearest_surface_xyz,
        nearest_surface_normal,
    )

    finite_offset = np.isfinite(signed_offset)
    if not np.any(finite_offset):
        raise ValueError(
            "The outside-PCD edge check could not calculate signed offsets."
        )

    # outward normal: 大多数内部骨架点的 signed offset 应为负。
    # 如果中位数为正，说明输入法向整体朝内，统一翻转符号。
    normals_were_flipped = bool(
        np.median(signed_offset[finite_offset]) > 0.0
    )
    if normals_were_flipped:
        signed_offset = -signed_offset

    return signed_offset, normals_were_flipped


def detect_edges_with_abrupt_cross_section_narrowing(
    skeleton_edges,
    pcd,
    save_path,
    section_points_save_path=None,
    deletion_points_save_path=None,
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
    short_junction_edge_max_arc_length=8.0,
    junction_attachment_radius=None,
    local_cut_padding_points=0,
    local_cut_recovery_ratio=0.85,
    local_cut_recovery_stable_sections=2,
    local_cut_recovery_reference_quantile=0.75,
    short_single_junction_max_points=2,
):
    """
    沿每条骨架 edge 的法平面切 PCD 外壳，寻找相对左右两侧急剧缩小的截面。

    判定逻辑
    --------
    1. 使用 MST 主路径排列每条 edge 的 skeleton 点；
    2. 在每个骨架点用前后邻域估计局部切向量；
    3. 以切向量为法向建立薄截面，从属于该 edge 的 PCD 外壳点中
       提取该截面；
    4. 使用截面内 PCD 点到骨架中心的平面径向距离 q75 作为截面尺度；
    5. 排除 edge 两端 endpoint_margin 个截面；它们仍保存为
       蓝色可视化截面，但不参与窄颈判定。
    6. 对每个内部截面计算：
           local_ratio = 当前截面 / min(左侧中位数, 右侧中位数)
       guard_sections=0 时，紧邻当前位置的左右截面直接参与计算。
    7. 取整条 edge 中最小的 local_ratio。只有当它不大于
       narrow_ratio_threshold 时，才认为出现“两侧较粗、中间急降”的窄颈。

    输出格式
    --------
    x,y,z,r,g,b

    颜色
    ----
    普通 edge       : 蓝色 [0, 102, 255]
    疑似粘连假 edge : 红色 [255, 0, 0]
    分叉点 edge_id=0: 绿色 [0, 255, 0]

    截面点云
    ----------
    如果传入 section_points_save_path，另外保存 x,y,z,r,g,b：
    普通采样截面 : 蓝色 [0, 102, 255]
    疑似窄颈的局部急降截面 : 红色 [255, 0, 0]

    建议删除点可视化
    ----------------
    如果传入 deletion_points_save_path，另外保存 x,y,z,r,g,b：
    正常 edge                         : 蓝色 [0, 102, 255]
    检测到急降的 edge               : 红色 [255, 0, 0]
    建议删除的 skeleton 点            : 黄色 [255, 255, 0]
    junction edge_id=0                 : 绿色 [0, 255, 0]
    """
    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)
    pcd = np.asarray(pcd, dtype=np.float64)

    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:
        raise ValueError(
            "skeleton_edges must contain x,y,z,r,g,b,edge_id."
        )
    if pcd.ndim != 2 or pcd.shape[1] < 3:
        raise ValueError("pcd must contain at least x,y,z.")
    if not 0.0 < narrow_ratio_threshold < 1.0:
        raise ValueError("narrow_ratio_threshold must be between 0 and 1.")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer.")
    if side_window < 1:
        raise ValueError("side_window must be >= 1.")
    if guard_sections < 0:
        raise ValueError("guard_sections must be >= 0.")
    if min_side_sections < 1:
        raise ValueError("min_side_sections must be >= 1.")
    if angular_bins < 8:
        raise ValueError("angular_bins must be >= 8.")
    if points_per_angle_bin < 1:
        raise ValueError("points_per_angle_bin must be >= 1.")
    if radial_outlier_factor <= 1.0:
        raise ValueError("radial_outlier_factor must be > 1.")
    if not 0.0 <= min_owned_angular_coverage <= 1.0:
        raise ValueError(
            "min_owned_angular_coverage must be between 0 and 1."
        )
    if short_junction_edge_max_arc_length <= 0:
        raise ValueError(
            "short_junction_edge_max_arc_length must be > 0."
        )
    if local_cut_padding_points < 0:
        raise ValueError("local_cut_padding_points must be >= 0.")
    if not 0.0 < local_cut_recovery_ratio <= 1.0:
        raise ValueError(
            "local_cut_recovery_ratio must be between 0 and 1."
        )
    if local_cut_recovery_stable_sections < 1:
        raise ValueError(
            "local_cut_recovery_stable_sections must be >= 1."
        )
    if not 0.5 <= local_cut_recovery_reference_quantile <= 1.0:
        raise ValueError(
            "local_cut_recovery_reference_quantile must be between 0.5 and 1."
        )
    if (
        short_single_junction_max_points < 1
        or int(short_single_junction_max_points)
        != short_single_junction_max_points
    ):
        raise ValueError(
            "short_single_junction_max_points must be a positive integer."
        )
    short_single_junction_max_points = int(
        short_single_junction_max_points
    )
    if junction_attachment_radius is None:
        junction_attachment_radius = neighbor_radius
    if junction_attachment_radius <= 0:
        raise ValueError("junction_attachment_radius must be > 0.")
    if visualization_section_stride < 1:
        raise ValueError("visualization_section_stride must be >= 1.")

    edge_xyz_all = skeleton_edges[:, :3]
    pcd_xyz_all = pcd[:, :3]
    edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int32)
    positive_rows = np.where(edge_ids > 0)[0]

    if len(positive_rows) == 0:
        raise ValueError("No edge_id >= 1 was found.")

    edge_row_groups = _indices_grouped_by_positive_label(edge_ids)
    junction_xyz = edge_xyz_all[edge_ids == 0]
    junction_tree = cKDTree(junction_xyz) if len(junction_xyz) > 0 else None
    if len(junction_xyz) > 0:
        # 一个实际 junction 区域可能包含多个相邻的 edge_id=0 点。
        # 先把这些点聚成独立 junction 区域，避免单点 edge 因为靠近
        # 同一 junction 的多个点而被误认为连接了多个 junction。
        junction_region_labels = DBSCAN(
            eps=junction_attachment_radius,
            min_samples=1,
        ).fit_predict(junction_xyz)
    else:
        junction_region_labels = np.empty(0, dtype=np.int32)

    # 第一级过滤：每个 PCD 点归属于最近的非分叉 skeleton edge。
    # 截面优先使用当前 edge 自己的 PCD，减少相邻线粒体混入。
    positive_tree = cKDTree(edge_xyz_all[positive_rows])
    _, nearest_positive_local = positive_tree.query(
        pcd_xyz_all,
        k=1,
        workers=-1,
    )
    nearest_positive_rows = positive_rows[nearest_positive_local]
    pcd_owner_edge = edge_ids[nearest_positive_rows]
    owned_pcd_groups = _indices_grouped_by_positive_label(pcd_owner_edge)

    # 全局 PCD 只在当前 edge 所属 PCD 的角度覆盖不足时用作后备。
    pcd_tree = cKDTree(pcd_xyz_all)

    # 独立的后置 candidate 条件所需：判断每个 skeleton 点是否位于
    # PCD 表面的外侧。它不参与截面计算，也不会改变红色 edge 集合。
    skeleton_signed_surface_offset, pcd_normals_were_flipped = (
        _signed_offsets_from_pcd_surface(
            query_xyz=edge_xyz_all,
            pcd=pcd,
            pcd_tree=pcd_tree,
        )
    )
    skeleton_outside_pcd = (
        np.isfinite(skeleton_signed_surface_offset)
        & (skeleton_signed_surface_offset > 1e-6)
    )

    suspected_edge_ids = set()
    diagnostic_rows = []
    visualization_blue_sections = []
    visualization_red_sections = []
    deletion_row_indices = set()
    whole_edge_deletion_ids = set()
    local_cut_deletion_ids = set()
    outside_pcd_whole_edge_deletion_ids = set()
    short_one_junction_edge_deletion_ids = set()
    analyzed_edge_count = 0
    insufficient_edge_count = 0
    fallback_section_total = 0
    smoothing_half_window = smoothing_window // 2

    required_path_points = 1

    for edge_id in sorted(edge_row_groups):
        rows = edge_row_groups[edge_id]
        geometry = _compute_edge_geometry(
            edge_xyz_all[rows],
            neighbor_radius=neighbor_radius,
        )
        ordered_rows = rows[geometry["ordered_local_indices"]]
        ordered_xyz = edge_xyz_all[ordered_rows]
        if junction_tree is None:
            start_is_junction = False
            end_is_junction = False
        else:
            start_is_junction = bool(
                junction_tree.query(ordered_xyz[0], k=1)[0]
                <= junction_attachment_radius
            )
            end_is_junction = bool(
                junction_tree.query(ordered_xyz[-1], k=1)[0]
                <= junction_attachment_radius
            )
        if len(ordered_xyz) < required_path_points:
            insufficient_edge_count += 1
            diagnostic_rows.append(
                {
                    "edge_id": edge_id,
                    "status": "insufficient_data",
                    "path_point_count": len(ordered_xyz),
                    "assigned_pcd_count": 0,
                    "fallback_section_count": 0,
                    "start_is_junction": int(start_is_junction),
                    "end_is_junction": int(end_is_junction),
                    "edge_arc_length": float(geometry["arc_length"]),
                    "deletion_mode": "none",
                    "delete_point_count": 0,
                    "minimum_ratio": np.nan,
                    "minimum_scale": np.nan,
                    "maximum_scale": np.nan,
                    "left_reference": np.nan,
                    "right_reference": np.nan,
                    "neck_x": np.nan,
                    "neck_y": np.nan,
                    "neck_z": np.nan,
                }
            )
            continue

        analyzed_edge_count += 1
        section_scale = np.full(len(ordered_xyz), np.nan, dtype=np.float64)
        section_global_pcd_indices = [None] * len(ordered_xyz)
        section_projected_xyz = [None] * len(ordered_xyz)
        edge_used_pcd_indices = set()
        fallback_section_count = 0

        owned_pcd_indices = owned_pcd_groups.get(
            edge_id,
            np.empty(0, dtype=np.int64),
        )
        if len(owned_pcd_indices) > 0:
            owned_pcd_tree = cKDTree(pcd_xyz_all[owned_pcd_indices])
        else:
            owned_pcd_tree = None

        search_radius = float(
            np.hypot(
                cross_section_max_radius,
                cross_section_half_thickness,
            )
        )

        for path_index, center_xyz in enumerate(ordered_xyz):
            global_near_indices = pcd_tree.query_ball_point(
                center_xyz,
                r=search_radius,
            )

            if len(global_near_indices) < min_section_points:
                continue

            global_near_indices = np.asarray(
                global_near_indices,
                dtype=np.int64,
            )

            if len(ordered_xyz) >= 2:
                left_index = max(0, path_index - tangent_window)
                right_index = min(
                    len(ordered_xyz) - 1,
                    path_index + tangent_window,
                )
                tangent = (
                    ordered_xyz[right_index] - ordered_xyz[left_index]
                )
            else:
                # 单点 edge 没有骨架走向，用附近 PCD 点云的
                # 第一主轴作为截面法向，保证它也会被计算。
                local_surface_xyz = pcd_xyz_all[global_near_indices]
                centered_surface = (
                    local_surface_xyz
                    - np.mean(local_surface_xyz, axis=0, keepdims=True)
                )
                if len(centered_surface) >= 2:
                    _, _, vh = np.linalg.svd(
                        centered_surface,
                        full_matrices=False,
                    )
                    tangent = vh[0]
                else:
                    tangent = np.asarray([1.0, 0.0, 0.0])

            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm <= 1e-8:
                tangent = np.asarray([1.0, 0.0, 0.0])
            else:
                tangent = tangent / tangent_norm

            owned_result = None
            if owned_pcd_tree is not None:
                owned_near_local = owned_pcd_tree.query_ball_point(
                    center_xyz,
                    r=search_radius,
                )
                if len(owned_near_local) > 0:
                    owned_near_global = owned_pcd_indices[
                        np.asarray(owned_near_local, dtype=np.int64)
                    ]
                    owned_result = _extract_first_shell_cross_section(
                        center_xyz=center_xyz,
                        tangent=tangent,
                        candidate_global_indices=owned_near_global,
                        pcd_xyz_all=pcd_xyz_all,
                        cross_section_half_thickness=(
                            cross_section_half_thickness
                        ),
                        cross_section_max_radius=cross_section_max_radius,
                        angular_bins=angular_bins,
                        points_per_angle_bin=points_per_angle_bin,
                        radial_outlier_factor=radial_outlier_factor,
                    )

            use_fallback = (
                owned_result is None
                or len(owned_result["global_indices"]) < min_section_points
                or owned_result["angular_coverage"]
                < min_owned_angular_coverage
            )

            if use_fallback:
                fallback_result = _extract_first_shell_cross_section(
                    center_xyz=center_xyz,
                    tangent=tangent,
                    candidate_global_indices=global_near_indices,
                    pcd_xyz_all=pcd_xyz_all,
                    cross_section_half_thickness=cross_section_half_thickness,
                    cross_section_max_radius=cross_section_max_radius,
                    angular_bins=angular_bins,
                    points_per_angle_bin=points_per_angle_bin,
                    radial_outlier_factor=radial_outlier_factor,
                )
                if fallback_result is not None:
                    section_result = fallback_result
                    fallback_section_count += 1
                else:
                    section_result = owned_result
            else:
                section_result = owned_result

            if (
                section_result is None
                or len(section_result["global_indices"])
                < min_section_points
            ):
                continue

            section_global_pcd_indices[path_index] = section_result[
                "global_indices"
            ]
            section_projected_xyz[path_index] = section_result[
                "projected_xyz"
            ]
            edge_used_pcd_indices.update(
                section_global_pcd_indices[path_index].tolist()
            )
            section_scale[path_index] = float(
                np.quantile(
                    section_result["radial_distance"],
                    section_radius_quantile,
                )
            )

        # 小窗口滚动中位数，降低单个 PCD 点缺失造成的假急降。
        fallback_section_total += fallback_section_count
        smoothed_scale = np.full_like(section_scale, np.nan)
        for path_index in range(len(section_scale)):
            start = max(0, path_index - smoothing_half_window)
            stop = min(
                len(section_scale),
                path_index + smoothing_half_window + 1,
            )
            local_values = section_scale[start:stop]
            local_values = local_values[np.isfinite(local_values)]
            if len(local_values) > 0:
                smoothed_scale[path_index] = float(np.median(local_values))

        # 短的假连接 edge 不能因固定 endpoint_margin 而完全失去
        # 可检验位置。左右至少需要 min_side_sections，中间还需要
        # guard_sections，在此基础上自适应缩小端点保护范围。
        minimum_required_without_margin = (
            2 * (guard_sections + min_side_sections) + 1
        )
        maximum_feasible_margin = max(
            (
                len(smoothed_scale)
                - minimum_required_without_margin
            ) // 2,
            0,
        )
        effective_endpoint_margin = min(
            endpoint_margin,
            maximum_feasible_margin,
        )
        first_test_index = effective_endpoint_margin
        last_test_index = len(smoothed_scale) - effective_endpoint_margin
        candidate_necks = []

        for path_index in range(first_test_index, last_test_index):
            center_scale = smoothed_scale[path_index]
            if (
                not np.isfinite(center_scale)
                or not np.isfinite(section_scale[path_index])
                or section_projected_xyz[path_index] is None
            ):
                continue

            # endpoint_margin 只禁止把端点本身标红；端点区域的
            # 正常截面仍可作为左右参考，否则短 edge 没有足够上下文。
            left_start = max(
                0,
                path_index - guard_sections - side_window,
            )
            left_stop = path_index - guard_sections
            right_start = path_index + guard_sections + 1
            right_stop = min(
                len(smoothed_scale),
                path_index + guard_sections + side_window + 1,
            )

            left_values = smoothed_scale[left_start:left_stop]
            right_values = smoothed_scale[right_start:right_stop]
            left_values = left_values[np.isfinite(left_values)]
            right_values = right_values[np.isfinite(right_values)]

            if (
                len(left_values) < min_side_sections
                or len(right_values) < min_side_sections
            ):
                continue

            left_reference = float(np.median(left_values))
            right_reference = float(np.median(right_values))
            two_sided_reference = min(left_reference, right_reference)

            if two_sided_reference <= 1e-8:
                continue

            local_ratio = float(center_scale / two_sided_reference)
            candidate_necks.append(
                (
                    local_ratio,
                    path_index,
                    float(center_scale),
                    left_reference,
                    right_reference,
                    float(
                        np.quantile(
                            left_values,
                            local_cut_recovery_reference_quantile,
                        )
                    ),
                    float(
                        np.quantile(
                            right_values,
                            local_cut_recovery_reference_quantile,
                        )
                    ),
                )
            )

        if candidate_necks:
            candidate_necks.sort(key=lambda item: item[0])
            (
                minimum_ratio,
                neck_path_index,
                minimum_scale,
                left_reference,
                right_reference,
                left_recovery_reference,
                right_recovery_reference,
            ) = candidate_necks[0]
            maximum_scale = min(left_reference, right_reference)
            neck_xyz = ordered_xyz[neck_path_index]

            if minimum_ratio <= narrow_ratio_threshold:
                suspected_edge_ids.add(edge_id)
                status = "suspected_narrow_neck"
                red_section_xyz = section_projected_xyz[neck_path_index]
                visualization_red_sections.append(red_section_xyz)
            else:
                status = "normal"
        else:
            minimum_ratio = np.nan
            minimum_scale = np.nan
            maximum_scale = np.nan
            left_reference = np.nan
            right_reference = np.nan
            left_recovery_reference = np.nan
            right_recovery_reference = np.nan
            neck_xyz = np.full(3, np.nan)
            status = "insufficient_local_context"

        deletion_mode = "none"
        delete_path_indices = []
        delete_start_path_index = np.nan
        delete_end_path_index = np.nan
        outside_pcd_point_count = int(
            np.count_nonzero(skeleton_outside_pcd[ordered_rows])
        )
        finite_edge_offsets = skeleton_signed_surface_offset[ordered_rows]
        finite_edge_offsets = finite_edge_offsets[
            np.isfinite(finite_edge_offsets)
        ]
        maximum_signed_surface_offset = (
            float(np.max(finite_edge_offsets))
            if len(finite_edge_offsets) > 0
            else np.nan
        )
        attached_junction_region_count = 0
        if len(ordered_rows) == 1 and junction_tree is not None:
            nearby_junction_rows = junction_tree.query_ball_point(
                ordered_xyz[0],
                r=junction_attachment_radius,
            )
            if len(nearby_junction_rows) > 0:
                attached_junction_region_count = int(
                    len(
                        np.unique(
                            junction_region_labels[
                                np.asarray(
                                    nearby_junction_rows,
                                    dtype=np.int64,
                                )
                            ]
                        )
                    )
                )
        if len(ordered_rows) == 1:
            attached_junction_side_count = attached_junction_region_count
        else:
            attached_junction_side_count = int(start_is_junction) + int(
                end_is_junction
            )

        # 只处理当前已经被截面算法标红的 edge。
        if status == "suspected_narrow_neck":
            if len(ordered_rows) == 1:
                both_ends_are_junctions = (
                    attached_junction_region_count >= 2
                )
            else:
                both_ends_are_junctions = (
                    start_is_junction and end_is_junction
                )
            short_whole_edge = (
                geometry["arc_length"]
                <= short_junction_edge_max_arc_length
            )

            if both_ends_are_junctions and short_whole_edge:
                # 短的 junction-junction 假连接：删整条正 edge，
                # 但 edge_id=0 的共享 junction 点本来就不在 rows 中。
                delete_path_indices = list(range(len(ordered_rows)))
                delete_start_path_index = 0
                delete_end_path_index = len(ordered_rows) - 1
                deletion_mode = "whole_short_junction_junction_edge"
                whole_edge_deletion_ids.add(edge_id)
            else:
                # 其余红色 edge：从最强急降点向左右扩展，直到截面
                # 分别恢复至左、右正常参考尺寸的指定比例，并且连续
                # local_cut_recovery_stable_sections 个截面都维持恢复。
                stable_count = local_cut_recovery_stable_sections
                left_recovery_threshold = (
                    local_cut_recovery_ratio * left_recovery_reference
                )
                right_recovery_threshold = (
                    local_cut_recovery_ratio * right_recovery_reference
                )

                left_recovery_near_index = None
                for distance_from_neck in range(1, len(ordered_rows)):
                    block_end = neck_path_index - distance_from_neck
                    block_start = block_end - stable_count + 1
                    if block_start < 0:
                        break
                    recovery_block = smoothed_scale[
                        block_start:block_end + 1
                    ]
                    if (
                        len(recovery_block) == stable_count
                        and np.all(np.isfinite(recovery_block))
                        and np.all(
                            recovery_block >= left_recovery_threshold
                        )
                    ):
                        # 这一组恢复截面保留，删除从它右侧开始。
                        left_recovery_near_index = block_end
                        break

                right_recovery_near_index = None
                for distance_from_neck in range(1, len(ordered_rows)):
                    block_start = neck_path_index + distance_from_neck
                    block_end = block_start + stable_count - 1
                    if block_end >= len(ordered_rows):
                        break
                    recovery_block = smoothed_scale[
                        block_start:block_end + 1
                    ]
                    if (
                        len(recovery_block) == stable_count
                        and np.all(np.isfinite(recovery_block))
                        and np.all(
                            recovery_block >= right_recovery_threshold
                        )
                    ):
                        # 这一组恢复截面保留，删除在它左侧停止。
                        right_recovery_near_index = block_start
                        break

                if left_recovery_near_index is None:
                    delete_start = 0
                else:
                    delete_start = left_recovery_near_index + 1

                if right_recovery_near_index is None:
                    delete_end = len(ordered_rows) - 1
                else:
                    delete_end = right_recovery_near_index - 1

                delete_start = max(
                    0,
                    delete_start - local_cut_padding_points,
                )
                delete_end = min(
                    len(ordered_rows) - 1,
                    delete_end + local_cut_padding_points,
                )

                # 如果删除后两侧骨架点仍在建图半径内，重新建图
                # 时会再次连上。仅向能更快增大断口的一侧扩展。
                while delete_start > 0 and delete_end < len(ordered_rows) - 1:
                    left_remaining = delete_start - 1
                    right_remaining = delete_end + 1
                    current_gap = float(
                        np.linalg.norm(
                            ordered_xyz[left_remaining]
                            - ordered_xyz[right_remaining]
                        )
                    )
                    if current_gap > neighbor_radius:
                        break

                    gap_if_expand_left = -np.inf
                    if left_remaining - 1 >= 0:
                        gap_if_expand_left = float(
                            np.linalg.norm(
                                ordered_xyz[left_remaining - 1]
                                - ordered_xyz[right_remaining]
                            )
                        )

                    gap_if_expand_right = -np.inf
                    if right_remaining + 1 < len(ordered_rows):
                        gap_if_expand_right = float(
                            np.linalg.norm(
                                ordered_xyz[left_remaining]
                                - ordered_xyz[right_remaining + 1]
                            )
                        )

                    if gap_if_expand_left >= gap_if_expand_right:
                        delete_start -= 1
                    else:
                        delete_end += 1

                delete_path_indices = list(
                    range(delete_start, delete_end + 1)
                )
                delete_start_path_index = delete_start
                delete_end_path_index = delete_end
                deletion_mode = "local_cut_until_section_recovery"
                local_cut_deletion_ids.add(edge_id)

            deletion_row_indices.update(
                ordered_rows[
                    np.asarray(delete_path_indices, dtype=np.int64)
                ].tolist()
            )

        # 独立的后置黄色 candidate 条件：不改变前面的截面计算，也不把
        # edge 加入红色集合。只检查未标红的 edge；如果其中至少一个
        # skeleton 点位于 PCD 外侧、两端均为 junction，且弧长严格小于
        # 10，则在 deletion candidate 文件中将整条 edge 标黄。
        outside_pcd_whole_edge_candidate = (
            status != "suspected_narrow_neck"
            and outside_pcd_point_count > 0
            and (
                (
                    len(ordered_rows) == 1
                    and attached_junction_region_count >= 2
                )
                or (
                    len(ordered_rows) > 1
                    and start_is_junction
                    and end_is_junction
                )
            )
            and geometry["arc_length"] < 10.0
        )
        if outside_pcd_whole_edge_candidate:
            delete_path_indices = list(range(len(ordered_rows)))
            delete_start_path_index = 0
            delete_end_path_index = len(ordered_rows) - 1
            deletion_mode = (
                "whole_nonred_junction_junction_edge_with_outside_pcd_point"
            )
            outside_pcd_whole_edge_deletion_ids.add(edge_id)
            deletion_row_indices.update(ordered_rows.tolist())

        # 统一的短 edge 点数规则。参数为2时，包含1或2个 skeleton 点
        # 的 edge 都使用同一判断；改成3时则覆盖1、2、3点。
        # 只有恰好一侧连接 junction 才整条标黄。单点 edge 没有两个
        # 不同几何端点，因此使用它连接的独立 junction 区域数量；
        # 多点 edge 使用首尾端点是否连接 junction。
        short_one_junction_edge_candidate = (
            len(ordered_rows) <= short_single_junction_max_points
            and attached_junction_side_count == 1
        )
        if short_one_junction_edge_candidate:
            delete_path_indices = list(range(len(ordered_rows)))
            delete_start_path_index = 0
            delete_end_path_index = len(ordered_rows) - 1
            deletion_mode = "whole_short_one_junction_edge_by_point_count"
            short_one_junction_edge_deletion_ids.add(edge_id)
            deletion_row_indices.update(ordered_rows.tolist())

        # 每隔若干个骨架点保存一个蓝色截面，留出空隙，
        # 便于在三维点云中看出一片片法平面。
        visualization_indices = set(
            range(
                0,
                len(section_projected_xyz),
                visualization_section_stride,
            )
        )
        if len(section_projected_xyz) > 0:
            visualization_indices.add(len(section_projected_xyz) - 1)

        for path_index in sorted(visualization_indices):
            if (
                status == "suspected_narrow_neck"
                and path_index == neck_path_index
            ):
                continue
            one_section_xyz = section_projected_xyz[path_index]
            if one_section_xyz is not None:
                visualization_blue_sections.append(one_section_xyz)

        diagnostic_rows.append(
            {
                "edge_id": edge_id,
                "status": status,
                "path_point_count": len(ordered_xyz),
                "assigned_pcd_count": len(edge_used_pcd_indices),
                "fallback_section_count": fallback_section_count,
                "start_is_junction": int(start_is_junction),
                "end_is_junction": int(end_is_junction),
                "edge_arc_length": float(geometry["arc_length"]),
                "deletion_mode": deletion_mode,
                "delete_point_count": len(delete_path_indices),
                "delete_start_path_index": delete_start_path_index,
                "delete_end_path_index": delete_end_path_index,
                "outside_pcd_point_count": outside_pcd_point_count,
                "maximum_signed_surface_offset": (
                    maximum_signed_surface_offset
                ),
                "attached_junction_region_count": (
                    attached_junction_region_count
                ),
                "attached_junction_side_count": (
                    attached_junction_side_count
                ),
                "minimum_ratio": minimum_ratio,
                "minimum_scale": minimum_scale,
                "maximum_scale": maximum_scale,
                "left_reference": left_reference,
                "right_reference": right_reference,
                "left_recovery_reference": left_recovery_reference,
                "right_recovery_reference": right_recovery_reference,
                "neck_x": neck_xyz[0],
                "neck_y": neck_xyz[1],
                "neck_z": neck_xyz[2],
            }
        )

    output = np.zeros((len(skeleton_edges), 6), dtype=np.float64)
    output[:, :3] = edge_xyz_all
    output[:, 3:6] = [0, 102, 255]
    output[edge_ids == 0, 3:6] = [0, 255, 0]

    if suspected_edge_ids:
        suspected_mask = np.isin(
            edge_ids,
            np.asarray(sorted(suspected_edge_ids), dtype=np.int32),
        )
        output[suspected_mask, 3:6] = [255, 0, 0]

    np.savetxt(
        save_path,
        output,
        delimiter=",",
        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
    )

    if deletion_points_save_path is not None:
        deletion_output = output.copy()
        if deletion_row_indices:
            deletion_rows_array = np.asarray(
                sorted(deletion_row_indices),
                dtype=np.int64,
            )
            deletion_output[deletion_rows_array, 3:6] = [255, 255, 0]

        np.savetxt(
            deletion_points_save_path,
            deletion_output,
            delimiter=",",
            fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
        )

    if section_points_save_path is not None:
        if visualization_blue_sections:
            blue_xyz = np.vstack(visualization_blue_sections)
        else:
            blue_xyz = np.empty((0, 3), dtype=np.float64)
        if visualization_red_sections:
            red_xyz = np.vstack(visualization_red_sections)
        else:
            red_xyz = np.empty((0, 3), dtype=np.float64)

        section_output_parts = []
        if len(blue_xyz) > 0:
            blue_output = np.zeros((len(blue_xyz), 6), dtype=np.float64)
            blue_output[:, :3] = blue_xyz
            blue_output[:, 3:6] = [0, 102, 255]
            section_output_parts.append(blue_output)
        if len(red_xyz) > 0:
            red_output = np.zeros((len(red_xyz), 6), dtype=np.float64)
            red_output[:, :3] = red_xyz
            red_output[:, 3:6] = [255, 0, 0]
            section_output_parts.append(red_output)

        if section_output_parts:
            section_output = np.vstack(section_output_parts)
        else:
            section_output = np.empty((0, 6), dtype=np.float64)

        np.savetxt(
            section_points_save_path,
            section_output,
            delimiter=",",
            fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
        )

    diagnostic_table = pd.DataFrame(diagnostic_rows)

    print(f"Cross-section edges total      : {len(edge_row_groups)}")
    print(f"Cross-section edges analyzed   : {analyzed_edge_count}")
    print(f"Edges with insufficient data   : {insufficient_edge_count}")
    print(f"Sections using global fallback : {fallback_section_total}")
    print(f"Suspected narrow-neck edges    : {len(suspected_edge_ids)}")
    print(f"Whole-edge deletion candidates : {len(whole_edge_deletion_ids)}")
    print(f"Local-cut deletion candidates  : {len(local_cut_deletion_ids)}")
    print(
        "Outside-PCD whole-edge candidates: "
        f"{len(outside_pcd_whole_edge_deletion_ids)}"
    )
    print(
        "Short one-junction edge candidates: "
        f"{len(short_one_junction_edge_deletion_ids)}"
    )
    print(
        "Short-edge maximum skeleton points: "
        f"{short_single_junction_max_points}"
    )
    print(f"Skeleton points marked yellow  : {len(deletion_row_indices)}")
    print(f"Local-drop ratio threshold     : {narrow_ratio_threshold}")
    print(f"PCD normals auto-flipped       : {pcd_normals_were_flipped}")

    if outside_pcd_whole_edge_deletion_ids:
        outside_candidate_rows = diagnostic_table[
            diagnostic_table["deletion_mode"]
            == "whole_nonred_junction_junction_edge_with_outside_pcd_point"
        ].sort_values("edge_arc_length")
        for _, row in outside_candidate_rows.iterrows():
            print(
                f"    outside-PCD edge_id={int(row['edge_id'])} | "
                f"arc_length={row['edge_arc_length']:.3f} | "
                f"outside_points={int(row['outside_pcd_point_count'])} | "
                f"max_signed_offset="
                f"{row['maximum_signed_surface_offset']:.3f}"
            )

    if short_one_junction_edge_deletion_ids:
        short_one_junction_rows = diagnostic_table[
            diagnostic_table["deletion_mode"]
            == "whole_short_one_junction_edge_by_point_count"
        ].sort_values(["path_point_count", "edge_id"])
        for _, row in short_one_junction_rows.iterrows():
            print(
                f"    short edge_id={int(row['edge_id'])} | "
                f"skeleton_points={int(row['path_point_count'])} | "
                f"junction_sides="
                f"{int(row['attached_junction_side_count'])}"
            )

    if suspected_edge_ids:
        suspected_rows = diagnostic_table[
            diagnostic_table["status"] == "suspected_narrow_neck"
        ].sort_values("minimum_ratio")
        for _, row in suspected_rows.iterrows():
            print(
                f"    edge_id={int(row['edge_id'])} | "
                f"ratio={row['minimum_ratio']:.3f} | "
                f"neck=({row['neck_x']:.2f},"
                f"{row['neck_y']:.2f},{row['neck_z']:.2f})"
            )

    print("Cross-section edge result saved:")
    print(f"    {save_path}")
    print("Cross-section edge format: x,y,z,r,g,b")
    print("Blue=ordinary edge, red=suspected narrow-neck edge, green=junction")

    if deletion_points_save_path is not None:
        print("Deletion-candidate skeleton visualization saved:")
        print(f"    {deletion_points_save_path}")
        print("Deletion visualization format: x,y,z,r,g,b")
        print(
            "Blue=normal, red=suspected edge, yellow=delete candidate, "
            "green=junction"
        )

    if section_points_save_path is not None:
        print("Cross-section PCD visualization saved:")
        print(f"    {section_points_save_path}")
        print("Cross-section PCD format: x,y,z,r,g,b")
        print(
            f"Blue section points={sum(map(len, visualization_blue_sections))}, "
            f"red local-drop section points={sum(map(len, visualization_red_sections))}"
        )

    return (
        output,
        diagnostic_table,
        np.asarray(sorted(deletion_row_indices), dtype=np.int64),
    )


def rebuild_skeleton_edges_after_yellow_deletion(
    skeleton_edges,
    deletion_row_indices,
    neighbor_radius=1.8,
    random_seed=42,
):
    """
    删除截面算法标黄的骨架点，并基于剩余点重新判断 junction 和 edge。

    参数
    ----
    skeleton_edges : ndarray, shape (N, >=7)
        删除前的 x,y,z,r,g,b,edge_id 骨架边点云。

    deletion_row_indices : array-like
        ``detect_edges_with_abrupt_cross_section_narrowing`` 返回的黄色点行号。

    返回
    ----
    final_skeleton_edges : ndarray, shape (M, 7)
        黄色点删除后重新分割的最终 x,y,z,r,g,b,edge_id。

    说明
    ----
    不保留旧的 edge_id 或旧 junction 标签。黄色点删除后，直接使用所有
    剩余 xyz 重新建邻接图、重新计算 degree、重新识别 junction，并重新
    编号 edge。因此，原先由错误粘连造成的假分叉会随连接被切断而消失。
    """
    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)
    deletion_row_indices = np.asarray(
        deletion_row_indices,
        dtype=np.int64,
    ).reshape(-1)

    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 3:
        raise ValueError(
            "skeleton_edges must be a 2D array with at least 3 columns."
        )

    point_count = len(skeleton_edges)
    if len(deletion_row_indices) > 0:
        deletion_row_indices = np.unique(deletion_row_indices)
        if (
            deletion_row_indices[0] < 0
            or deletion_row_indices[-1] >= point_count
        ):
            raise IndexError(
                "deletion_row_indices contains an out-of-range row index."
            )

    keep_mask = np.ones(point_count, dtype=bool)
    keep_mask[deletion_row_indices] = False
    remaining_xyz = skeleton_edges[keep_mask, :3]

    final_skeleton_edges = segment_refined_skeleton_edges(
        refined_skeleton=remaining_xyz,
        neighbor_radius=neighbor_radius,
        random_seed=random_seed,
    )

    print("Final edge topology rebuilt after yellow-point deletion:")
    print(f"    Skeleton points before : {point_count}")
    print(f"    Yellow points deleted  : {len(deletion_row_indices)}")
    print(f"    Skeleton points after  : {len(final_skeleton_edges)}")

    return final_skeleton_edges


def _assign_spatially_separated_instance_colors(
    xyz,
    instance_id,
    random_seed=42,
):
    """让空间邻近的 instance 优先获得色相距离尽可能大的颜色。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    instance_id = np.asarray(instance_id, dtype=np.int32)
    unique_ids = np.unique(instance_id[instance_id > 0])
    instance_count = len(unique_ids)
    rgb = np.zeros((len(xyz), 3), dtype=np.int32)

    if instance_count == 0:
        return rgb, np.nan

    id_to_position = {
        int(one_id): position
        for position, one_id in enumerate(unique_ids)
    }
    spatial_neighbors = [set() for _ in range(instance_count)]

    # 质心近邻保证体积较小或点数很少的 instance 也有颜色约束。
    centroids = np.vstack([
        np.mean(xyz[instance_id == one_id], axis=0)
        for one_id in unique_ids
    ])
    if instance_count > 1:
        centroid_tree = cKDTree(centroids)
        centroid_k = min(9, instance_count)
        _, centroid_neighbor_rows = centroid_tree.query(
            centroids,
            k=centroid_k,
            workers=-1,
        )
        if centroid_neighbor_rows.ndim == 1:
            centroid_neighbor_rows = centroid_neighbor_rows[:, None]
        for position, neighbor_positions in enumerate(
            centroid_neighbor_rows
        ):
            for neighbor_position in neighbor_positions:
                neighbor_position = int(neighbor_position)
                if neighbor_position == position:
                    continue
                spatial_neighbors[position].add(neighbor_position)
                spatial_neighbors[neighbor_position].add(position)

    # 再从骨架点边界寻找近邻，避免两个细长 instance 的质心较远，
    # 但局部末端实际上紧挨在一起时被分到相近颜色。
    if len(xyz) > 1 and instance_count > 1:
        point_tree = cKDTree(xyz)
        point_k = min(32, len(xyz))
        _, point_neighbor_rows = point_tree.query(
            xyz,
            k=point_k,
            workers=-1,
        )
        if point_neighbor_rows.ndim == 1:
            point_neighbor_rows = point_neighbor_rows[:, None]

        for point_row, neighbor_rows in enumerate(point_neighbor_rows):
            source_id = int(instance_id[point_row])
            source_position = id_to_position[source_id]
            nearby_other_ids = []
            for neighbor_row in neighbor_rows:
                neighbor_id = int(instance_id[int(neighbor_row)])
                if (
                    neighbor_id != source_id
                    and neighbor_id not in nearby_other_ids
                ):
                    nearby_other_ids.append(neighbor_id)
                    if len(nearby_other_ids) >= 3:
                        break
            for neighbor_id in nearby_other_ids:
                neighbor_position = id_to_position[neighbor_id]
                spatial_neighbors[source_position].add(neighbor_position)
                spatial_neighbors[neighbor_position].add(source_position)

    # 高度饱和的候选色相；颜色只要求邻近 instance 互相远离，
    # 因而远处 instance 可以安全复用颜色。
    golden_ratio = 0.618033988749895
    hue_offset = (random_seed * golden_ratio) % 1.0
    candidate_hues = (
        hue_offset + np.arange(64, dtype=np.float64) / 64.0
    ) % 1.0
    candidate_use_count = np.zeros(len(candidate_hues), dtype=np.int32)
    assigned_candidate = np.full(instance_count, -1, dtype=np.int32)

    # 先处理空间邻居最多的 instance，让最拥挤区域优先获得最大色差。
    coloring_order = sorted(
        range(instance_count),
        key=lambda position: (
            -len(spatial_neighbors[position]),
            int(unique_ids[position]),
        ),
    )

    for position in coloring_order:
        colored_neighbors = [
            assigned_candidate[neighbor_position]
            for neighbor_position in spatial_neighbors[position]
            if assigned_candidate[neighbor_position] >= 0
        ]

        if len(colored_neighbors) == 0:
            best_candidate = int(np.argmin(candidate_use_count))
        else:
            neighbor_hues = candidate_hues[
                np.asarray(colored_neighbors, dtype=np.int32)
            ]
            hue_difference = np.abs(
                candidate_hues[:, None] - neighbor_hues[None, :]
            )
            circular_difference = np.minimum(
                hue_difference,
                1.0 - hue_difference,
            )
            minimum_neighbor_difference = np.min(
                circular_difference,
                axis=1,
            )
            best_score = np.max(minimum_neighbor_difference)
            best_candidates = np.where(
                np.isclose(minimum_neighbor_difference, best_score)
            )[0]
            best_candidate = int(
                best_candidates[
                    np.argmin(candidate_use_count[best_candidates])
                ]
            )

        assigned_candidate[position] = best_candidate
        candidate_use_count[best_candidate] += 1

    assigned_hues = candidate_hues[assigned_candidate]
    for position, one_instance_id in enumerate(unique_ids):
        color = colorsys.hsv_to_rgb(
            float(assigned_hues[position]),
            0.90,
            1.0,
        )
        color = np.rint(np.asarray(color) * 255).astype(np.int32)
        rgb[instance_id == one_instance_id] = color

    neighbor_hue_distances = []
    for position, neighbors in enumerate(spatial_neighbors):
        for neighbor_position in neighbors:
            if neighbor_position <= position:
                continue
            difference = abs(
                assigned_hues[position]
                - assigned_hues[neighbor_position]
            )
            neighbor_hue_distances.append(min(difference, 1.0 - difference))
    minimum_neighbor_hue_distance = (
        float(np.min(neighbor_hue_distances))
        if neighbor_hue_distances
        else np.nan
    )

    return rgb, minimum_neighbor_hue_distance


def segment_skeleton_social_instances(
    skeleton_edges,
    neighbor_radius=1.8,
    random_seed=42,
    prune_shortcut_edges=True,
):
    """
    将最终 physical edges 按完整骨架连通性合并为 social instances。

    与 physical edge 分割不同，这里不会移除 junction，也不会把
    junction 当作 edge 边界。所有能够沿骨架经过 junction 相互到达的
    点都属于同一个 social instance；不连通的骨架属于不同 instance。

    返回格式：x,y,z,r,g,b,instance_id，其中 instance_id 从1开始。
    """
    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)
    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 3:
        raise ValueError(
            "skeleton_edges must be a 2D array with at least 3 columns."
        )

    xyz = skeleton_edges[:, :3]
    point_count = len(xyz)
    if point_count == 0:
        return np.empty((0, 7), dtype=np.float64)

    tree = cKDTree(xyz)
    neighbor_pairs = tree.query_pairs(
        r=neighbor_radius,
        output_type="ndarray",
    )

    candidate_adjacency = [set() for _ in range(point_count)]
    for point_a, point_b in neighbor_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        candidate_adjacency[point_a].add(point_b)
        candidate_adjacency[point_b].add(point_a)

    adjacency = [[] for _ in range(point_count)]
    kept_pair_count = 0
    for point_a, point_b in neighbor_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        keep_pair = True

        if prune_shortcut_edges:
            distance_ab = float(np.linalg.norm(xyz[point_a] - xyz[point_b]))
            common_neighbors = (
                candidate_adjacency[point_a]
                & candidate_adjacency[point_b]
            )
            for point_k in common_neighbors:
                distance_ak = float(
                    np.linalg.norm(xyz[point_a] - xyz[point_k])
                )
                distance_bk = float(
                    np.linalg.norm(xyz[point_b] - xyz[point_k])
                )
                if (
                    distance_ak < distance_ab - 1e-8
                    and distance_bk < distance_ab - 1e-8
                ):
                    keep_pair = False
                    break

        if keep_pair:
            adjacency[point_a].append(point_b)
            adjacency[point_b].append(point_a)
            kept_pair_count += 1

    instance_id = np.zeros(point_count, dtype=np.int32)
    current_instance_id = 0
    for start_index in range(point_count):
        if instance_id[start_index] != 0:
            continue

        current_instance_id += 1
        instance_id[start_index] = current_instance_id
        stack = [start_index]
        while stack:
            current_index = stack.pop()
            for neighbor_index in adjacency[current_index]:
                if instance_id[neighbor_index] == 0:
                    instance_id[neighbor_index] = current_instance_id
                    stack.append(neighbor_index)

    rgb, minimum_neighbor_hue_distance = (
        _assign_spatially_separated_instance_colors(
            xyz=xyz,
            instance_id=instance_id,
            random_seed=random_seed,
        )
    )

    social_instances = np.column_stack([xyz, rgb, instance_id])

    print("Social skeleton instances generated:")
    print(f"    Skeleton points       : {point_count}")
    print(f"    Candidate graph edges : {len(neighbor_pairs)}")
    print(f"    Pruned graph edges    : {kept_pair_count}")
    print(f"    Social instances      : {current_instance_id}")
    if np.isfinite(minimum_neighbor_hue_distance):
        print(
            "    Minimum neighbor hue gap: "
            f"{minimum_neighbor_hue_distance:.3f}"
        )

    return social_instances


def _profile_features(values, positions, prefix, profile_bins=10):
    """计算沿骨架方向分箱后的profile变化。"""
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(positions)
    values = values[valid]
    positions = np.clip(positions[valid], 0.0, 1.0)

    output = {
        f"{prefix}_bin_std": np.nan,
        f"{prefix}_bin_min_median_ratio": np.nan,
        f"{prefix}_profile_max_drop": np.nan,
        f"{prefix}_endpoint_center_ratio": np.nan,
        f"{prefix}_low_bin_fraction": np.nan,
    }

    if len(values) == 0:
        return output

    bin_indices = np.minimum(
        (positions * profile_bins).astype(np.int32),
        profile_bins - 1,
    )
    bin_means = np.full(profile_bins, np.nan, dtype=np.float64)

    for bin_index in range(profile_bins):
        one_bin = values[bin_indices == bin_index]
        if len(one_bin) > 0:
            bin_means[bin_index] = np.mean(one_bin)

    valid_bin_means = bin_means[np.isfinite(bin_means)]
    if len(valid_bin_means) == 0:
        return output

    median_bin = float(np.median(valid_bin_means))
    output[f"{prefix}_bin_std"] = float(np.std(valid_bin_means))
    output[f"{prefix}_bin_min_median_ratio"] = float(
        np.min(valid_bin_means) / (abs(median_bin) + 1e-8)
    )
    output[f"{prefix}_low_bin_fraction"] = float(
        np.mean(valid_bin_means < 0.5 * median_bin)
    )

    adjacent_valid = np.isfinite(bin_means[:-1]) & np.isfinite(bin_means[1:])
    if np.any(adjacent_valid):
        output[f"{prefix}_profile_max_drop"] = float(
            np.max(
                np.abs(np.diff(bin_means)[adjacent_valid])
            ) / (abs(median_bin) + 1e-8)
        )

    endpoint_values = bin_means[[0, -1]]
    endpoint_values = endpoint_values[np.isfinite(endpoint_values)]
    center_values = bin_means[1:-1]
    center_values = center_values[np.isfinite(center_values)]

    if len(endpoint_values) > 0 and len(center_values) > 0:
        output[f"{prefix}_endpoint_center_ratio"] = float(
            np.mean(endpoint_values)
            / (abs(np.median(center_values)) + 1e-8)
        )

    return output


def compute_skeleton_edge_features(
    skeleton_edges,
    refined_skeleton,
    voxel,
    pcd,
    neighbor_radius=1.8,
    voxel_assignment_max_distance=8.0,
    pcd_assignment_max_distance=8.0,
    junction_search_radius=2.5,
    profile_bins=10,
):
    """
    为每条 edge_id>=1 的骨架边计算几何、score、intensity、PCD和局部拓扑feature。
    """
    skeleton_edges = np.asarray(skeleton_edges, dtype=np.float64)
    refined_skeleton = np.asarray(refined_skeleton, dtype=np.float64)
    voxel = np.asarray(voxel, dtype=np.float64)
    pcd = np.asarray(pcd, dtype=np.float64)

    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:
        raise ValueError(
            "skeleton_edges must contain x,y,z,r,g,b,edge_id."
        )
    if refined_skeleton.ndim != 2 or refined_skeleton.shape[1] < 4:
        raise ValueError(
            "refined_skeleton must contain x,y,z,score."
        )

    # 最终 skeleton_edges 已经删除黄色点，行数通常少于原始
    # refined_skeleton。不能再假设二者逐行对应，而应按照 xyz 将每个
    # 最终骨架点重新匹配回原始 refined skeleton，取回它的 score。
    edge_xyz_for_alignment = skeleton_edges[:, :3]
    refined_xyz_for_alignment = refined_skeleton[:, :3]
    rows_already_aligned = (
        len(skeleton_edges) == len(refined_skeleton)
        and np.allclose(
            edge_xyz_for_alignment,
            refined_xyz_for_alignment,
            atol=1e-3,
            rtol=0.0,
        )
    )

    if not rows_already_aligned:
        refined_tree = cKDTree(refined_xyz_for_alignment)
        alignment_distance, alignment_rows = refined_tree.query(
            edge_xyz_for_alignment,
            k=1,
            workers=-1,
        )
        maximum_alignment_distance = float(np.max(alignment_distance))
        if maximum_alignment_distance > 0.01:
            raise ValueError(
                "Final skeleton edge points could not be matched back to "
                "refined_skeleton within 0.01 coordinate units. "
                f"Maximum distance: {maximum_alignment_distance:.6f}."
            )
        refined_skeleton = refined_skeleton[alignment_rows]
        print(
            "Aligned refined-skeleton scores to final skeleton edges: "
            f"{len(skeleton_edges)} rows, "
            f"max distance={maximum_alignment_distance:.6f}"
        )

    edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int32)
    edge_xyz_all = skeleton_edges[:, :3]
    score_all = refined_skeleton[:, 3]
    positive_rows = np.where(edge_ids > 0)[0]

    if len(positive_rows) == 0:
        raise ValueError("No edge_id >= 1 was found.")

    unique_edge_ids = np.unique(edge_ids[positive_rows])
    edge_row_groups = _indices_grouped_by_positive_label(edge_ids)

    edge_cache = {}
    principal_axis_by_row = np.full((len(edge_ids), 3), np.nan)
    normalized_position_by_row = np.full(len(edge_ids), np.nan)

    for edge_id in unique_edge_ids:
        rows = edge_row_groups[int(edge_id)]
        edge_xyz = edge_xyz_all[rows]
        geometry = _compute_edge_geometry(
            edge_xyz,
            neighbor_radius=neighbor_radius,
        )
        ordered_rows = rows[geometry["ordered_local_indices"]]
        edge_cache[int(edge_id)] = {
            "rows": rows,
            "xyz": edge_xyz,
            "geometry": geometry,
            "ordered_rows": ordered_rows,
            "centroid": edge_xyz.mean(axis=0),
            "start_xyz": edge_xyz_all[ordered_rows[0]],
            "end_xyz": edge_xyz_all[ordered_rows[-1]],
        }
        principal_axis_by_row[rows] = geometry["principal_axis"]
        normalized_position_by_row[rows] = geometry["normalized_position"]

    # 将voxel和PCD分别分配给最近的非分叉骨架边。
    positive_tree = cKDTree(edge_xyz_all[positive_rows])

    voxel_distance, voxel_nearest_local = positive_tree.query(
        voxel[:, :3],
        k=1,
        workers=-1,
    )
    voxel_nearest_rows = positive_rows[voxel_nearest_local]
    voxel_assigned_edge = edge_ids[voxel_nearest_rows].copy()
    voxel_assigned_edge[
        voxel_distance > voxel_assignment_max_distance
    ] = -1
    voxel_groups = _indices_grouped_by_positive_label(voxel_assigned_edge)

    pcd_distance, pcd_nearest_local = positive_tree.query(
        pcd[:, :3],
        k=1,
        workers=-1,
    )
    pcd_nearest_rows = positive_rows[pcd_nearest_local]
    pcd_assigned_edge = edge_ids[pcd_nearest_rows].copy()
    pcd_assigned_edge[
        pcd_distance > pcd_assignment_max_distance
    ] = -1
    pcd_groups = _indices_grouped_by_positive_label(pcd_assigned_edge)

    junction_xyz = edge_xyz_all[edge_ids == 0]
    junction_tree = cKDTree(junction_xyz) if len(junction_xyz) > 0 else None

    def endpoint_context(edge_id, endpoint_xyz):
        if junction_tree is None:
            junction_distance = np.nan
        else:
            junction_distance = float(junction_tree.query(endpoint_xyz, k=1)[0])

        nearby_local_rows = positive_tree.query_ball_point(
            endpoint_xyz,
            r=junction_search_radius,
        )
        nearby_global_rows = positive_rows[
            np.asarray(nearby_local_rows, dtype=np.int64)
        ]
        nearby_edge_ids = set(
            int(value)
            for value in edge_ids[nearby_global_rows]
            if int(value) > 0 and int(value) != int(edge_id)
        )

        current_centroid = edge_cache[int(edge_id)]["centroid"]
        current_vector = current_centroid - endpoint_xyz
        current_norm = np.linalg.norm(current_vector)
        continuation_angles = []

        if current_norm > 1e-8:
            current_vector = current_vector / current_norm
            for neighbor_edge_id in nearby_edge_ids:
                neighbor_vector = (
                    edge_cache[neighbor_edge_id]["centroid"] - endpoint_xyz
                )
                neighbor_norm = np.linalg.norm(neighbor_vector)
                if neighbor_norm <= 1e-8:
                    continue
                neighbor_vector = neighbor_vector / neighbor_norm
                cosine = np.clip(
                    np.dot(-current_vector, neighbor_vector),
                    -1.0,
                    1.0,
                )
                continuation_angles.append(
                    float(np.degrees(np.arccos(cosine)))
                )

        best_angle = (
            float(np.min(continuation_angles))
            if continuation_angles
            else np.nan
        )

        return {
            "junction_distance": junction_distance,
            "is_junction": int(
                np.isfinite(junction_distance)
                and junction_distance <= junction_search_radius
            ),
            "neighbor_edge_count": len(nearby_edge_ids),
            "best_continuation_angle": best_angle,
        }

    feature_rows = []

    for edge_id in unique_edge_ids:
        edge_id = int(edge_id)
        cache = edge_cache[edge_id]
        rows = cache["rows"]
        geometry = cache["geometry"]
        edge_xyz = cache["xyz"]
        ordered_rows = cache["ordered_rows"]
        point_count = len(rows)

        rgb = np.rint(skeleton_edges[rows[0], 3:6]).astype(np.int32)
        one_feature = {
            "edge_id": edge_id,
            "color_r": int(rgb[0]),
            "color_g": int(rgb[1]),
            "color_b": int(rgb[2]),
            "skeleton_point_count": point_count,
            "is_single_point_edge": int(point_count == 1),
            "is_short_edge": int(point_count < 3),
            "edge_arc_length": geometry["arc_length"],
            "edge_chord_length": geometry["chord_length"],
            "tortuosity": geometry["tortuosity"],
            "straightness": geometry["straightness"],
            "turn_angle_mean": geometry["turn_angle_mean"],
            "turn_angle_std": geometry["turn_angle_std"],
            "turn_angle_max": geometry["turn_angle_max"],
            "pca_linearity": geometry["pca_linearity"],
        }

        edge_scores = score_all[rows]
        one_feature.update(_safe_numeric_features(edge_scores, "score"))
        score_median = np.nanmedian(edge_scores)
        one_feature["score_min_median_ratio"] = float(
            np.nanmin(edge_scores) / (abs(score_median) + 1e-8)
        )
        one_feature.update(
            _profile_features(
                score_all[rows],
                normalized_position_by_row[rows],
                prefix="score",
                profile_bins=profile_bins,
            )
        )

        voxel_indices = voxel_groups.get(edge_id, np.empty(0, dtype=np.int64))
        one_feature["assigned_voxel_count"] = len(voxel_indices)

        if len(voxel_indices) > 0 and voxel.shape[1] >= 4:
            intensity = voxel[voxel_indices, 3]
            one_feature.update(
                _safe_numeric_features(intensity, "intensity")
            )
            intensity_positions = normalized_position_by_row[
                voxel_nearest_rows[voxel_indices]
            ]
            one_feature.update(
                _profile_features(
                    intensity,
                    intensity_positions,
                    prefix="intensity",
                    profile_bins=profile_bins,
                )
            )
        else:
            one_feature.update(_safe_numeric_features([], "intensity"))
            one_feature.update(
                _profile_features([], [], "intensity", profile_bins)
            )

        pcd_indices = pcd_groups.get(edge_id, np.empty(0, dtype=np.int64))
        one_feature["assigned_pcd_count"] = len(pcd_indices)
        one_feature["surface_points_per_length"] = float(
            len(pcd_indices) / (geometry["arc_length"] + 1e-8)
        )

        if len(pcd_indices) > 0:
            one_feature.update(
                _safe_numeric_features(
                    pcd_distance[pcd_indices],
                    "surface_distance",
                )
            )
        else:
            one_feature.update(
                _safe_numeric_features([], "surface_distance")
            )

        one_feature["normal_tangent_abs_dot_mean"] = np.nan
        one_feature["normal_tangent_abs_dot_std"] = np.nan
        one_feature["normal_tangent_abs_dot_q90"] = np.nan
        one_feature["normal_resultant_length"] = np.nan
        one_feature["normal_dispersion"] = np.nan
        one_feature["cross_section_ellipticity"] = np.nan

        if len(pcd_indices) >= 3 and pcd.shape[1] >= 6:
            normals = pcd[pcd_indices, 3:6]
            normal_lengths = np.linalg.norm(normals, axis=1)
            valid_normals = normal_lengths > 1e-8

            if np.any(valid_normals):
                normals = normals[valid_normals] / normal_lengths[
                    valid_normals, None
                ]
                tangent = principal_axis_by_row[
                    pcd_nearest_rows[pcd_indices][valid_normals]
                ]
                abs_dot = np.abs(np.sum(normals * tangent, axis=1))
                one_feature["normal_tangent_abs_dot_mean"] = float(
                    np.mean(abs_dot)
                )
                one_feature["normal_tangent_abs_dot_std"] = float(
                    np.std(abs_dot)
                )
                one_feature["normal_tangent_abs_dot_q90"] = float(
                    np.quantile(abs_dot, 0.90)
                )
                resultant = float(np.linalg.norm(np.mean(normals, axis=0)))
                one_feature["normal_resultant_length"] = resultant
                one_feature["normal_dispersion"] = 1.0 - resultant

            axis = geometry["principal_axis"]
            reference = np.asarray([1.0, 0.0, 0.0])
            if abs(np.dot(axis, reference)) > 0.90:
                reference = np.asarray([0.0, 1.0, 0.0])
            basis_u = np.cross(axis, reference)
            basis_u = basis_u / (np.linalg.norm(basis_u) + 1e-8)
            basis_v = np.cross(axis, basis_u)
            centered_surface = pcd[pcd_indices, :3] - cache["centroid"]
            projected_surface = np.column_stack(
                [
                    centered_surface @ basis_u,
                    centered_surface @ basis_v,
                ]
            )
            cross_covariance = np.cov(projected_surface, rowvar=False)
            cross_eigenvalues = np.linalg.eigvalsh(cross_covariance)
            cross_eigenvalues = np.maximum(cross_eigenvalues, 0.0)
            one_feature["cross_section_ellipticity"] = float(
                (cross_eigenvalues[-1] - cross_eigenvalues[0])
                / (cross_eigenvalues[-1] + cross_eigenvalues[0] + 1e-8)
            )

        start_context = endpoint_context(edge_id, cache["start_xyz"])
        end_context = endpoint_context(edge_id, cache["end_xyz"])
        one_feature["start_junction_distance"] = start_context[
            "junction_distance"
        ]
        one_feature["end_junction_distance"] = end_context[
            "junction_distance"
        ]
        one_feature["start_is_junction"] = start_context["is_junction"]
        one_feature["end_is_junction"] = end_context["is_junction"]
        one_feature["junction_junction_edge"] = int(
            start_context["is_junction"] and end_context["is_junction"]
        )
        one_feature["endpoint_junction_edge"] = int(
            start_context["is_junction"] != end_context["is_junction"]
        )
        one_feature["endpoint_endpoint_edge"] = int(
            not start_context["is_junction"]
            and not end_context["is_junction"]
        )
        one_feature["start_neighbor_edge_count"] = start_context[
            "neighbor_edge_count"
        ]
        one_feature["end_neighbor_edge_count"] = end_context[
            "neighbor_edge_count"
        ]
        one_feature["best_continuation_angle_start"] = start_context[
            "best_continuation_angle"
        ]
        one_feature["best_continuation_angle_end"] = end_context[
            "best_continuation_angle"
        ]

        one_feature["eligible_for_umap"] = int(point_count >= 3)
        feature_rows.append(one_feature)

    feature_table = pd.DataFrame(feature_rows).sort_values(
        "edge_id"
    ).reset_index(drop=True)

    print(f"Feature rows           : {len(feature_table)}")
    print(f"Feature columns        : {len(feature_table.columns)}")
    print(
        "Eligible for UMAP      : "
        f"{int(feature_table['eligible_for_umap'].sum())}"
    )

    return feature_table


def run_edge_umap_and_clustering(
    feature_table,
    save_umap_csv_path,
    save_umap_png_path,
    random_seed=42,
    umap_n_neighbors=15,
    umap_min_dist=0.10,
    hdbscan_min_cluster_size=10,
):
    """对边feature做稳健标准化、无监督聚类和UMAP可视化。"""
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP is not installed. Run: pip install umap-learn"
        ) from error

    metadata_columns = {
        "edge_id",
        "color_r",
        "color_g",
        "color_b",
        "eligible_for_umap",
    }
    feature_columns = [
        column
        for column in feature_table.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(feature_table[column])
    ]

    eligible_mask = feature_table["eligible_for_umap"].astype(bool).to_numpy()
    eligible_table = feature_table.loc[eligible_mask].copy()

    if len(eligible_table) < 3:
        raise ValueError("At least 3 eligible edges are required for UMAP.")

    feature_matrix = eligible_table[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # 删除全缺失或没有变化的feature，避免影响UMAP。
    usable_columns = []
    for column in feature_columns:
        values = feature_matrix[column]
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        usable_columns.append(column)

    feature_matrix = feature_matrix[usable_columns].copy()

    log1p_columns = {
        "skeleton_point_count",
        "edge_arc_length",
        "edge_chord_length",
        "assigned_voxel_count",
        "assigned_pcd_count",
        "surface_points_per_length",
    }

    for column in usable_columns:
        if column in log1p_columns:
            feature_matrix[column] = np.log1p(
                np.maximum(feature_matrix[column], 0.0)
            )

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    feature_matrix_imputed = imputer.fit_transform(feature_matrix)
    feature_matrix_scaled = scaler.fit_transform(feature_matrix_imputed)

    effective_neighbors = min(
        umap_n_neighbors,
        len(eligible_table) - 1,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_seed,
    )
    embedding = reducer.fit_transform(feature_matrix_scaled)

    cluster_method = "HDBSCAN"
    try:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=min(
                hdbscan_min_cluster_size,
                max(3, len(eligible_table) // 3),
            ),
            min_samples=5,
        )
        cluster_labels = clusterer.fit_predict(feature_matrix_scaled)
        cluster_probability = getattr(
            clusterer,
            "probabilities_",
            np.full(len(eligible_table), np.nan),
        )
    except ImportError:
        cluster_method = "DBSCAN fallback"
        fallback_clusterer = DBSCAN(eps=1.5, min_samples=5)
        cluster_labels = fallback_clusterer.fit_predict(embedding)
        cluster_probability = np.full(len(eligible_table), np.nan)

    result_table = feature_table.copy()
    result_table["umap_x"] = np.nan
    result_table["umap_y"] = np.nan
    result_table["cluster"] = -99
    result_table["cluster_probability"] = np.nan
    result_table.loc[eligible_mask, "umap_x"] = embedding[:, 0]
    result_table.loc[eligible_mask, "umap_y"] = embedding[:, 1]
    result_table.loc[eligible_mask, "cluster"] = cluster_labels
    result_table.loc[eligible_mask, "cluster_probability"] = cluster_probability
    result_table.to_csv(save_umap_csv_path, index=False)

    plt.figure(figsize=(8, 7))
    unique_clusters = np.unique(cluster_labels)
    color_map = plt.get_cmap("tab20")

    for cluster_index, cluster_label in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_label

        if cluster_label == -1:
            color = "#9e9e9e"
            label = "Noise / outlier"
        else:
            color = color_map(cluster_index % 20)
            label = f"Cluster {cluster_label}"

        plt.scatter(
            embedding[cluster_mask, 0],
            embedding[cluster_mask, 1],
            s=24,
            alpha=0.80,
            c=[color],
            edgecolors="none",
            label=label,
        )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(
        "Mitochondrial skeleton-edge features\n"
        f"{cluster_method}, n={len(eligible_table)} edges"
    )
    plt.legend(frameon=False, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_umap_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"UMAP feature columns   : {len(usable_columns)}")
    print(f"UMAP edges             : {len(eligible_table)}")
    print(f"Clustering method      : {cluster_method}")
    print(f"UMAP CSV saved         : {save_umap_csv_path}")
    print(f"UMAP PNG saved         : {save_umap_png_path}")

    return result_table


def save_clustered_skeleton_edges_xyzrgb(
    skeleton_edges,
    clustering_table,
    save_path,
):
    """
    将 edge 的无监督聚类标签映射回 skeleton 点，并保存为 x,y,z,r,g,b。

    颜色规则：
        cluster >= 0 : 每个 cluster 使用一种固定的高区分度颜色；
        cluster == -1: HDBSCAN/DBSCAN noise，灰色；
        cluster == -99: 不满足 UMAP 条件的短边，深灰色；
        edge_id == 0 : 分叉点，红色。
    """
    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] < 7:
        raise ValueError(
            "skeleton_edges must contain x,y,z,r,g,b,edge_id."
        )

    required_columns = {"edge_id", "cluster"}
    missing_columns = required_columns.difference(clustering_table.columns)
    if missing_columns:
        raise ValueError(
            "Clustering table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    edge_to_cluster = {}
    for _, row in clustering_table[["edge_id", "cluster"]].dropna().iterrows():
        edge_id = int(round(float(row["edge_id"])))
        cluster_label = int(round(float(row["cluster"])))
        edge_to_cluster[edge_id] = cluster_label

    valid_cluster_labels = sorted(
        {
            cluster_label
            for cluster_label in edge_to_cluster.values()
            if cluster_label >= 0
        }
    )

    # 使用 HSV 色环生成确定性的高对比颜色，不受原 edge 随机颜色影响。
    cluster_colors = {}
    number_of_clusters = len(valid_cluster_labels)
    for color_index, cluster_label in enumerate(valid_cluster_labels):
        if number_of_clusters == 1:
            color_position = 0.58
        else:
            # 整体错开纯红色，避免与红色分叉点混淆。
            color_position = (
                0.08 + color_index / number_of_clusters
            ) % 1.0

        rgba = plt.get_cmap("hsv")(color_position)
        cluster_colors[cluster_label] = np.rint(
            np.asarray(rgba[:3]) * 255.0
        ).astype(np.int32)

    output = np.zeros((len(skeleton_edges), 6), dtype=np.float64)
    output[:, :3] = skeleton_edges[:, :3]

    skeleton_edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int64)
    output_rgb = np.zeros((len(skeleton_edges), 3), dtype=np.int32)

    for row_index, edge_id in enumerate(skeleton_edge_ids):
        if edge_id == 0:
            output_rgb[row_index] = [255, 0, 0]
            continue

        cluster_label = edge_to_cluster.get(int(edge_id), -99)

        if cluster_label >= 0:
            output_rgb[row_index] = cluster_colors[cluster_label]
        elif cluster_label == -1:
            output_rgb[row_index] = [160, 160, 160]
        else:
            output_rgb[row_index] = [80, 80, 80]

    output[:, 3:6] = output_rgb

    np.savetxt(
        save_path,
        output,
        delimiter=",",
        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
    )

    cluster_edge_counts = {
        cluster_label: sum(
            1
            for value in edge_to_cluster.values()
            if value == cluster_label
        )
        for cluster_label in valid_cluster_labels
    }
    noise_edge_count = sum(
        1 for value in edge_to_cluster.values() if value == -1
    )
    ineligible_edge_count = sum(
        1 for value in edge_to_cluster.values() if value == -99
    )

    print(f"Valid clusters          : {number_of_clusters}")
    for cluster_label in valid_cluster_labels:
        rgb = cluster_colors[cluster_label]
        print(
            f"    Cluster {cluster_label}: "
            f"{cluster_edge_counts[cluster_label]} edges | "
            f"RGB=({rgb[0]},{rgb[1]},{rgb[2]})"
        )
    print(f"Noise / outlier edges   : {noise_edge_count}")
    print(f"Ineligible short edges  : {ineligible_edge_count}")
    print("Clustered skeleton saved:")
    print(f"    {save_path}")
    print("Clustered skeleton format: x,y,z,r,g,b")

    return output


###############################################################################
# 后续准备逐步添加的函数
###############################################################################


def segment_mito_instances(voxel, pcd, skeleton, refined_skeleton):
    """
    TODO：精细线粒体实例分割。

    以后再逐步加入：
        1. 分离由于分辨率原因黏在一起的线粒体；
        2. 分离平行、同向、相邻的线粒体；
        3. 使用 voxel intensity、PCD normal 和 skeleton 信息；
        4. 返回 voxel、PCD 和 skeleton 的实例分割结果。

    第一版暂时不调用这个函数。
    """
    raise NotImplementedError("Mitochondrial instance segmentation is not written yet.")


def split_instance_skeleton_by_branch_points(skeleton_instance):
    """
    TODO：在线粒体实例内部，按照端点和分叉点切分骨架。

    第一版暂时不调用这个函数。
    """
    raise NotImplementedError("Skeleton branch segmentation is not written yet.")


def main():

    # file_name_list = [
    #
    #                         # ====================================================
    #     # MCF-7
    #     # # ====================================================
    #     "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #     # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",
    #     #
    #     # # ====================================================
    #     # # AC16
    #     # # ====================================================
    #     "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #     # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",
    #     #
    #     # # ====================================================
    #     # # MDA
    #     # # ====================================================
    #     "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #     # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",
    #     #
    #     # # ====================================================
    #     # # Fibroblast
    #     # # ====================================================
    #     "20260207_fibro_mito488_lyso640_nuc405_tif",
    #     # "20251216_fibro_mito488tracker_ld561bodipy_nuc405hurst",
    #     #
    #     # # ====================================================
    #     # # H28
    #     # # ====================================================
    #     "20250917_h28_mitotracker488_lysotracker640_nuc405",
    #     # "20251213_h28_mito488_ld561_nuc405",
    #     #
    #     # # ====================================================
    #     # # HeLa
    #     # # ====================================================
    #     "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",
    #     # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",
    #
    #     # ====================================================
    #     # U2OS
    #     # ====================================================
    #     # "20251201_u2os_ld488_mito640_nuc405",
    #     "20250909_u2os_mitotracker488_lysotracker640_nuc405",
    #
    #     # ====================================================
    #     # MSTO
    #     # ====================================================
    #     # "20251127_msto_ld488_mito640_nuc405",
    #     "20250905_msto_mitotracekr488_lysotracker640_nuc405",
    #
    #                 ]

    file_name_list = [

        # ====================================================
        # GLUC
        # ====================================================
        # "gluc_mito-mChGFP_LTRDR50nM_0031",
        # "gluc_mito-mChGFP_LTRDR50nM_0032",
        # "gluc_mito-mChGFP_LTRDR50nM_0051",
        # "gluc_mito-mChGFP_LTRDR50nM_0061",
        # "gluc_mito-mChGFP_LTRDR50nM_0071",
        # "gluc_mito-mChGFP_LTRDR50nM_0072",
        # "gluc_mito-mChGFP_LTRDR50nM_0081",
        # "gluc_mito-mChGFP_LTRDR50nM_0082",
        # "gluc_mito-mChGFP_LTRDR50nM_0091",

        # ====================================================
        # Non-GLUC
        # ====================================================

        # "mito-mChGFP_LTRDR50nM_0021",
        # "mito-mChGFP_LTRDR50nM_0022",
        # "mito-mChGFP_LTRDR50nM_0031",
        # "mito-mChGFP_LTRDR50nM_0032",
        # "mito-mChGFP_LTRDR50nM_0041",
        # "mito-mChGFP_LTRDR50nM_0042",
        # "mito-mChGFP_LTRDR50nM_0051",
        # "mito-mChGFP_LTRDR50nM_0061",
        # "mito-mChGFP_LTRDR50nM_0062",
        # "mito-mChGFP_LTRDR50nM_0071",
        # "mito-mChGFP_LTRDR50nM_0072",
        "mito-mChGFP_LTRDR50nM_0081",

    ]

    total_done = 0

    for file_name in file_name_list:
        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/{file_name}/mito")
        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")
        # BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")
        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/masked")

        INPUT_PATH = BASE_DIR / "processed_output"
        OUTPUT_PATH = BASE_DIR / "processed_output"

        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(
                f"processed_output folder not found: {INPUT_PATH}"
            )

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        print(os.listdir(INPUT_PATH))

        #######################################################################
        # 图片列表
        #######################################################################

        # 自动读取 processed_output 中的所有 *_pcd.txt。
        imgs_list = [
            f
            for f in os.listdir(INPUT_PATH)
            if f.lower().endswith("_pcd.txt")
        ]

        # 手动指定需要处理的图片。
        # 需要批量处理全部图片时，将下面这一行注释掉即可。
        # imgs_list = ["fibro_mito488_lyso640_nuc405_17689_5ADVMLE"]

        if len(imgs_list) == 0:
            print(f"No images selected in {INPUT_PATH}")
            continue

        print("========================================================================= ")
        print(f"Dataset            : {file_name}")
        print(f"BASE_DIR           : {BASE_DIR}")
        print(f"INPUT_PATH         : {INPUT_PATH}")
        print(f"OUTPUT_PATH        : {OUTPUT_PATH}")
        print(f"Selected images    : {len(imgs_list)}")
        print("========================================================================= ")

        for i, img_name in enumerate(imgs_list, start=1):
            # 同时兼容：
            #     自动扫描得到的 xxx_pcd.txt
            #     手动填写的 xxx
            if img_name.lower().endswith("_pcd.txt"):
                stem = img_name[:-len("_pcd.txt")]
            else:
                stem = img_name

            pcd_path = INPUT_PATH / f"{stem}_pcd.txt"

            print("-------------------------------------------------------------------------")
            print(f"Processing {file_name} | {i}/{len(imgs_list)} | {stem}")
            print("-------------------------------------------------------------------------")

            ###################################################################
            # 输入文件路径
            ###################################################################

            voxel_path = INPUT_PATH / f"{stem}_voxel.txt"
            skeleton_path = find_skeleton_path(INPUT_PATH, stem)
            refined_skeleton_path = find_refined_skeleton_path(INPUT_PATH, stem)

            ###################################################################
            # 将来要保存的文件路径
            # 第一版只打印路径，不实际保存文件。
            ###################################################################

            save_pcd_instance_path = OUTPUT_PATH / f"{stem}_pcd_instance.txt"
            save_skeleton_instance_path = OUTPUT_PATH / f"{stem}_skeleton_instance.txt"
            save_voxel_instance_path = OUTPUT_PATH / f"{stem}_voxel_instance.txt"
            save_skeleton_edges_path = OUTPUT_PATH / f"{stem}_skeleton_edges.txt"
            save_skeleton_social_instance_path = (
                OUTPUT_PATH / f"{stem}_skeleton_instance_social.txt"
            )
            save_skeleton_social_version_path = (
                OUTPUT_PATH
                / f"{stem}_skeleton_instance_social_algorithm_version.txt"
            )
            save_original_skeleton_edges_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edges_original.txt"
            )
            save_skeleton_edges_version_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edges_algorithm_version.txt"
            )
            save_edge_features_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edge_features.csv"
            )
            save_edge_umap_csv_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edge_umap.csv"
            )
            save_edge_umap_png_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edge_umap.png"
            )
            save_clustered_skeleton_edges_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edges_clustered.txt"
            )
            save_cross_section_edges_path = (
                OUTPUT_PATH / f"{stem}_skeleton_edges_cross_section.txt"
            )
            save_cross_section_points_path = (
                OUTPUT_PATH
                / f"{stem}_pcd_cross_sections_first_shell_local_drop_flat.txt"
            )
            save_deletion_points_path = (
                OUTPUT_PATH
                / f"{stem}_skeleton_edges_deletion_candidates_recovery.txt"
            )

            edge_algorithm_version = (
                "cross_section_yellow_deletion_and_topology_rebuild_v10"
            )
            social_instance_algorithm_version = (
                "connected_components_spatial_maximin_color_v1"
            )
            social_instance_is_current = False
            if (
                save_skeleton_social_instance_path.exists()
                and save_skeleton_social_version_path.exists()
            ):
                try:
                    saved_social_version = (
                        save_skeleton_social_version_path.read_text(
                            encoding="utf-8"
                        ).strip()
                    )
                    social_instance_is_current = (
                        saved_social_version
                        == social_instance_algorithm_version
                    )
                except OSError:
                    social_instance_is_current = False
            edge_segmentation_is_current = False
            if (
                save_skeleton_edges_path.exists()
                and save_skeleton_edges_version_path.exists()
            ):
                try:
                    saved_version = (
                        save_skeleton_edges_version_path.read_text(
                            encoding="utf-8"
                        ).strip()
                    )
                    edge_segmentation_is_current = (
                        saved_version == edge_algorithm_version
                    )
                except OSError:
                    edge_segmentation_is_current = False

            edge_segmentation_rebuilt = False

            # 正式 skeleton_edges 现在是删除黄色点并重建拓扑后的最终文件。
            # 如果任一删除前检查文件缺失，不能直接在最终 edge 上重算截面；
            # 必须从 refined skeleton 重新走完整流程，才能恢复原始候选连接。
            cross_section_outputs_complete = (
                save_original_skeleton_edges_path.exists()
                and save_cross_section_edges_path.exists()
                and save_cross_section_points_path.exists()
                and save_deletion_points_path.exists()
            )
            if (
                edge_segmentation_is_current
                and not cross_section_outputs_complete
            ):
                print(
                    "[REBUILD] A pre-deletion cross-section output is missing; "
                    "the complete cleanup pipeline will be rerun."
                )
                edge_segmentation_is_current = False

            ###################################################################
            # 两个最终可视化文件都存在，表示当前流程已经完整完成。
            ###################################################################

            if (
                edge_segmentation_is_current
                and
                save_clustered_skeleton_edges_path.exists()
                and save_cross_section_edges_path.exists()
                and save_cross_section_points_path.exists()
                and save_deletion_points_path.exists()
                and save_original_skeleton_edges_path.exists()
                and social_instance_is_current
            ):
                print("[SKIPPED] Final visualization files already exist:")
                print(f"    {save_clustered_skeleton_edges_path}")
                print(f"    {save_cross_section_edges_path}")
                print(f"    {save_cross_section_points_path}")
                print(f"    {save_deletion_points_path}")
                print(f"    {save_original_skeleton_edges_path}")
                print(f"    {save_skeleton_social_instance_path}")
                continue

            ###################################################################
            # 之前已经完成聚类，但还没有导出 xyzrgb：直接复用旧结果。
            # 这里不会读取 PCD、voxel，也不会重新分 edge 或重新聚类。
            ###################################################################

            if (
                edge_segmentation_is_current
                and
                not save_clustered_skeleton_edges_path.exists()
                and
                save_skeleton_edges_path.exists()
                and save_edge_umap_csv_path.exists()
            ):
                print("[REUSED] Existing edges and clustering results found.")
                print(f"    {save_skeleton_edges_path}")
                print(f"    {save_edge_umap_csv_path}")

                skeleton_edges = load_txt_array(save_skeleton_edges_path)
                clustering_table = pd.read_csv(save_edge_umap_csv_path)

                save_clustered_skeleton_edges_xyzrgb(
                    skeleton_edges=skeleton_edges,
                    clustering_table=clustering_table,
                    save_path=save_clustered_skeleton_edges_path,
                )

            if (
                edge_segmentation_is_current
                and
                save_clustered_skeleton_edges_path.exists()
                and save_cross_section_edges_path.exists()
                and save_cross_section_points_path.exists()
                and save_deletion_points_path.exists()
                and save_original_skeleton_edges_path.exists()
                and social_instance_is_current
            ):
                total_done += 1
                continue

            ###################################################################
            # 已有 edge 和聚类可视化、只缺截面结果时走快速分支。
            # 只读取 PCD 与 skeleton_edges，不读取 voxel/raw/refined skeleton。
            ###################################################################

            if (
                edge_segmentation_is_current
                and
                save_clustered_skeleton_edges_path.exists()
                and save_skeleton_edges_path.exists()
                and (
                    not save_cross_section_edges_path.exists()
                    or not save_cross_section_points_path.exists()
                    or not save_deletion_points_path.exists()
                )
            ):
                if not pcd_path.exists():
                    print("[SKIPPED] Missing PCD file:")
                    print(f"    {pcd_path}")
                    continue

                print("[FAST PATH] Only cross-section result is missing.")
                print("Reading PCD and existing skeleton edges only.")
                pcd = load_txt_array(pcd_path)
                skeleton_edges = load_txt_array(save_skeleton_edges_path)

                print_file_information(
                    label="PCD",
                    path=pcd_path,
                    data=pcd,
                    column_format="x,y,z,nx,ny,nz",
                )
                print_file_information(
                    label="SKELETON EDGES",
                    path=save_skeleton_edges_path,
                    data=skeleton_edges,
                    column_format="x,y,z,r,g,b,edge_id",
                )

                detect_edges_with_abrupt_cross_section_narrowing(
                    skeleton_edges=skeleton_edges,
                    pcd=pcd,
                    save_path=save_cross_section_edges_path,
                    section_points_save_path=save_cross_section_points_path,
                    deletion_points_save_path=save_deletion_points_path,
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
                    short_junction_edge_max_arc_length=5.0,
                    junction_attachment_radius=1.8,
                    local_cut_padding_points=0,
                    local_cut_recovery_ratio=0.85,
                    local_cut_recovery_stable_sections=2,
                    local_cut_recovery_reference_quantile=0.75,
                    short_single_junction_max_points=2,
                )

                total_done += 1
                continue

            ###################################################################
            # 检查四个输入文件是否存在
            ###################################################################

            required_input_paths = {
                "PCD": pcd_path,
                "VOXEL": voxel_path,
                "SKELETON": skeleton_path,
                "REFINED SKELETON": refined_skeleton_path,
            }

            missing_paths = [
                path
                for path in required_input_paths.values()
                if not path.exists()
            ]

            if len(missing_paths) > 0:
                print("[SKIPPED] Missing input files:")
                for missing_path in missing_paths:
                    print(f"    {missing_path}")
                continue

            ###################################################################
            # 读取输入文件
            ###################################################################

            pcd = load_txt_array(pcd_path)
            voxel = load_txt_array(voxel_path)
            skeleton = load_txt_array(skeleton_path)
            refined_skeleton = load_txt_array(refined_skeleton_path)

            ###################################################################
            # 打印路径、shape和列格式
            ###################################################################

            print_file_information(
                label="PCD",
                path=pcd_path,
                data=pcd,
                column_format="x,y,z,nx,ny,nz",
            )
            check_expected_columns("PCD", pcd, expected_columns=6)
            print()

            print_file_information(
                label="VOXEL",
                path=voxel_path,
                data=voxel,
                column_format="x,y,z,intensity",
            )
            check_expected_columns("VOXEL", voxel, expected_columns=4)
            print()

            print_file_information(
                label="SKELETON",
                path=skeleton_path,
                data=skeleton,
                column_format="x,y,z,radius,degree,type",
            )
            check_expected_columns("SKELETON", skeleton, expected_columns=6)
            print()

            print_file_information(
                label="REFINED SKELETON",
                path=refined_skeleton_path,
                data=refined_skeleton,
                column_format="x,y,z,score",
            )
            check_expected_columns(
                "REFINED SKELETON",
                refined_skeleton,
                expected_columns=4,
            )

            ###################################################################
            # 打印将来的输出文件路径
            ###################################################################

            print()
            print("Planned instance output files — not saved yet:")
            print(f"PCD instance      : {save_pcd_instance_path}")
            print(f"Skeleton instance : {save_skeleton_instance_path}")
            print(f"Voxel instance    : {save_voxel_instance_path}")
            print(f"Edge features     : {save_edge_features_path}")
            print(f"Edge UMAP CSV     : {save_edge_umap_csv_path}")
            print(f"Edge UMAP PNG     : {save_edge_umap_png_path}")
            print(f"Clustered skeleton: {save_clustered_skeleton_edges_path}")
            print(f"Cross-section edge: {save_cross_section_edges_path}")
            print(f"Original edges    : {save_original_skeleton_edges_path}")
            print(f"Social instances  : {save_skeleton_social_instance_path}")

            ###################################################################
            # 如果 edge 文件已经存在，直接读取，不重复补点和分 edge。
            ###################################################################

            if (
                save_skeleton_edges_path.exists()
                and edge_segmentation_is_current
            ):
                print()
                print("[REUSED] Existing skeleton edges found:")
                print(f"    {save_skeleton_edges_path}")

                skeleton_edges = load_txt_array(save_skeleton_edges_path)
                check_expected_columns(
                    "SKELETON EDGES",
                    skeleton_edges,
                    expected_columns=7,
                )

                print(f"Skeleton edges shape : {skeleton_edges.shape}")
                print("Skeleton edges format: x,y,z,r,g,b,edge_id")
                print("Skipped: adding missing skeleton points and edge segmentation.")

            else:
                print()
                print("[REBUILD] Recomputing and overwriting skeleton edges:")
                print(f"    {save_skeleton_edges_path}")
                print(f"Algorithm version: {edge_algorithm_version}")

                ###############################################################
                # 第零步：为没有 skeleton 的小 PCD component 添加质心骨架点
                ###############################################################

                print()
                print("Adding missing skeleton points from small PCD components...")

                (
                    refined_skeleton,
                    added_skeleton_points,
                    pcd_component_labels,
                    pcd_component_information,
                ) = add_missing_refined_skeleton_from_small_pcd_components(
                    pcd=pcd,
                    refined_skeleton=refined_skeleton,
                    dbscan_eps=2.0,
                    dbscan_min_samples=10,
                    min_component_points=20,
                    max_component_points=1000,
                    solid_grid_spacing=1.0,
                    local_padding=3,
                    surface_dilation_iterations=1,
                    surface_closing_iterations=1,
                )

                # 按照原来的路径覆盖保存，格式仍然是 x,y,z,score。
                np.savetxt(
                    refined_skeleton_path,
                    refined_skeleton,
                    delimiter=",",
                    fmt="%.6f",
                )

                print("Refined skeleton updated:")
                print(f"    {refined_skeleton_path}")
                print("Refined skeleton format: x,y,z,score")

                ###############################################################
                # 第一步：将 refined skeleton 按照分叉点切分成拓扑边
                ###############################################################

                print()
                print("Segmenting refined skeleton into topology edges...")

                skeleton_edges = segment_refined_skeleton_edges(
                    refined_skeleton=refined_skeleton,
                    neighbor_radius=1.8,
                    random_seed=42,
                )

                np.savetxt(
                    save_original_skeleton_edges_path,
                    skeleton_edges,
                    delimiter=",",
                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],
                )

                edge_segmentation_rebuilt = True

                print("Original topology edges saved:")
                print(f"    {save_original_skeleton_edges_path}")
                print("The canonical skeleton_edges file will be overwritten after cleanup.")

            ###################################################################
            # 第二步：沿 edge 法平面分析 PCD 截面是否出现急剧缩小
            ###################################################################

            if (
                not edge_segmentation_rebuilt
                and
                save_cross_section_edges_path.exists()
                and save_cross_section_points_path.exists()
                and save_deletion_points_path.exists()
            ):
                print()
                print("[REUSED] Existing cross-section results found:")
                print(f"    {save_cross_section_edges_path}")
                print(f"    {save_cross_section_points_path}")
                print(f"    {save_deletion_points_path}")
            else:
                print()
                print("Detecting abrupt cross-section narrowing along edges...")

                (
                    _,
                    cross_section_diagnostics,
                    deletion_row_indices,
                ) = detect_edges_with_abrupt_cross_section_narrowing(
                    skeleton_edges=skeleton_edges,
                    pcd=pcd,
                    save_path=save_cross_section_edges_path,
                    section_points_save_path=save_cross_section_points_path,
                    deletion_points_save_path=save_deletion_points_path,
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

                ###############################################################
                # 第三步：删除黄色点，重新计算 junction 和 edge，覆盖正式文件
                ###############################################################

                print()
                print("Deleting yellow points and rebuilding final topology...")

                skeleton_edges = rebuild_skeleton_edges_after_yellow_deletion(
                    skeleton_edges=skeleton_edges,
                    deletion_row_indices=deletion_row_indices,
                    neighbor_radius=1.8,
                    random_seed=42,
                )

                np.savetxt(
                    save_skeleton_edges_path,
                    skeleton_edges,
                    delimiter=",",
                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],
                )
                save_skeleton_edges_version_path.write_text(
                    edge_algorithm_version + "\n",
                    encoding="utf-8",
                )
                edge_segmentation_is_current = True

                print("Final skeleton edges saved:")
                print(f"    {save_skeleton_edges_path}")
                print("Final format: x,y,z,r,g,b,edge_id")
                print("Junctions and edge IDs were recalculated after deletion.")

            ###################################################################
            # 第四步：跨 junction 合并 physical edges，生成 social instance
            ###################################################################

            if (
                social_instance_is_current
                and not edge_segmentation_rebuilt
            ):
                print()
                print("[REUSED] Existing social skeleton instances found:")
                print(f"    {save_skeleton_social_instance_path}")
            else:
                print()
                print(
                    "Merging physical edges through junctions into "
                    "social instances..."
                )
                skeleton_social_instances = segment_skeleton_social_instances(
                    skeleton_edges=skeleton_edges,
                    neighbor_radius=1.8,
                    random_seed=42,
                    prune_shortcut_edges=True,
                )
                np.savetxt(
                    save_skeleton_social_instance_path,
                    skeleton_social_instances,
                    delimiter=",",
                    fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%d"],
                )
                save_skeleton_social_version_path.write_text(
                    social_instance_algorithm_version + "\n",
                    encoding="utf-8",
                )
                social_instance_is_current = True
                print("Social skeleton instances saved:")
                print(f"    {save_skeleton_social_instance_path}")
                print("Format: x,y,z,r,g,b,instance_id")

            ###################################################################
            # 第五步：基于最终 edge 计算 feature，并进行无监督 UMAP 分析
            ###################################################################

            if (
                save_clustered_skeleton_edges_path.exists()
                and not edge_segmentation_rebuilt
            ):
                print()
                print("[REUSED] Existing clustered skeleton found:")
                print(f"    {save_clustered_skeleton_edges_path}")
            else:
                if (
                    save_edge_umap_csv_path.exists()
                    and not edge_segmentation_rebuilt
                ):
                    print()
                    print("[REUSED] Existing edge clustering table found:")
                    print(f"    {save_edge_umap_csv_path}")
                    clustering_table = pd.read_csv(save_edge_umap_csv_path)
                else:
                    if (
                        save_edge_features_path.exists()
                        and not edge_segmentation_rebuilt
                    ):
                        print()
                        print("[REUSED] Existing edge features found:")
                        print(f"    {save_edge_features_path}")
                        edge_feature_table = pd.read_csv(save_edge_features_path)
                    else:
                        print()
                        print("Computing skeleton-edge features...")

                        edge_feature_table = compute_skeleton_edge_features(
                            skeleton_edges=skeleton_edges,
                            refined_skeleton=refined_skeleton,
                            voxel=voxel,
                            pcd=pcd,
                            neighbor_radius=1.8,
                            voxel_assignment_max_distance=8.0,
                            pcd_assignment_max_distance=8.0,
                            junction_search_radius=2.5,
                            profile_bins=10,
                        )

                        edge_feature_table.to_csv(
                            save_edge_features_path,
                            index=False,
                        )
                        print("Edge features saved:")
                        print(f"    {save_edge_features_path}")

                    clustering_table = run_edge_umap_and_clustering(
                        feature_table=edge_feature_table,
                        save_umap_csv_path=save_edge_umap_csv_path,
                        save_umap_png_path=save_edge_umap_png_path,
                        random_seed=42,
                        umap_n_neighbors=15,
                        umap_min_dist=0.10,
                        hdbscan_min_cluster_size=10,
                    )

                print()
                print("Mapping edge clusters back to skeleton points...")
                save_clustered_skeleton_edges_xyzrgb(
                    skeleton_edges=skeleton_edges,
                    clustering_table=clustering_table,
                    save_path=save_clustered_skeleton_edges_path,
                )

            ###################################################################
            # TODO：后续版本从这里开始逐步加入分割代码
            ###################################################################

            # voxel_instance, pcd_instance, skeleton_instance = \
            #     segment_mito_instances(
            #         voxel=voxel,
            #         pcd=pcd,
            #         skeleton=skeleton,
            #         refined_skeleton=refined_skeleton,
            #     )

            # skeleton_instance = split_instance_skeleton_by_branch_points(
            #     skeleton_instance
            # )

            # save_pcd2txt(voxel_instance, save_voxel_instance_path)
            # save_pcd2txt(pcd_instance, save_pcd_instance_path)
            # save_pcd2txt(skeleton_instance, save_skeleton_instance_path)

            total_done += 1

    print("=========================================================================")
    print(f"Skeleton edge segmentation finished. Images processed: {total_done}")
    print(
        "Saved files: *_skeleton_edges.txt, "
        "*_skeleton_instance_social.txt, "
        "*_skeleton_edges_original.txt, "
        "*_skeleton_edges_algorithm_version.txt, "
        "*_skeleton_edge_features.csv, "
        "*_skeleton_edge_umap.csv, *_skeleton_edge_umap.png, "
        "*_skeleton_edges_clustered.txt, "
        "*_skeleton_edges_cross_section.txt, "
        "*_pcd_cross_sections_first_shell_local_drop_flat.txt, "
        "*_skeleton_edges_deletion_candidates_recovery.txt"
    )
    print(
        "Final *_skeleton_edges.txt contains only the remaining skeleton "
        "points, with junctions and edge IDs recalculated after yellow deletion."
    )
    print(
        "Final *_skeleton_instance_social.txt merges all physical edges that "
        "remain connected through junctions."
    )
    print("=========================================================================")


if __name__ == "__main__":
    main()

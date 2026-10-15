from pathlib import Path
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def load_txt_array(path):
    """读取逗号分隔、无表头的 txt 数据。"""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


def print_file_information(label, path, data, column_format):
    """清楚打印一个输入文件的路径、shape和列格式。"""
    print(f"{label} path   : {path}")
    print(f"{label} shape  : {data.shape}")
    print(f"{label} format : {column_format}")


def check_minimum_columns(label, data, minimum_columns):
    """检查输入数据至少具有要求的列数。"""
    if data.ndim != 2 or data.shape[1] < minimum_columns:
        raise ValueError(
            f"{label} must contain at least {minimum_columns} columns, "
            f"but its shape is {data.shape}."
        )


def summarize_distances(label, distances):
    """打印最近邻分配距离，便于检查坐标是否对齐。"""
    distances = np.asarray(distances, dtype=np.float64)

    print(f"{label}:")
    print(f"    minimum : {np.min(distances):.4f}")
    print(f"    median  : {np.median(distances):.4f}")
    print(f"    q95     : {np.quantile(distances, 0.95):.4f}")
    print(f"    maximum : {np.max(distances):.4f}")


def build_canonical_colors(labels, rgb, layer_name):
    """
    为每个内部 label 确定唯一RGB。

    正常情况下，同一 edge/instance 的所有骨架点颜色完全相同。如果输入中
    意外出现多种颜色，则采用该 label 中出现次数最多的颜色，并打印警告。
    label 只在计算过程中使用，不会写入最终文件。
    """
    labels = np.asarray(labels, dtype=np.int64)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int32)
    unique_labels = np.unique(labels)
    canonical_colors = np.empty((len(unique_labels), 3), dtype=np.int32)
    inconsistent_label_count = 0

    for position, one_label in enumerate(unique_labels):
        one_rgb = rgb[labels == one_label]
        colors, counts = np.unique(one_rgb, axis=0, return_counts=True)
        canonical_colors[position] = colors[np.argmax(counts)]

        if len(colors) > 1:
            inconsistent_label_count += 1

    if inconsistent_label_count > 0:
        print(
            f"[WARNING] {layer_name}: {inconsistent_label_count} labels "
            "contained more than one RGB value. The most frequent RGB "
            "within each label was used."
        )

    return unique_labels, canonical_colors


def _disabled_assign_pcd_by_neighborhood_normals(
    pcd,
    skeleton_instances,
    layer_name,
    normal_candidate_count=64,
    normal_neighborhood_count=16,
    normal_neighborhood_radius=2.5,
    normal_neighbor_min_cosine=0.25,
    normal_min_support_fraction=0.60,
    normal_min_support_count=3,
    normal_search_max_distance=12.0,
    normal_ray_max_perpendicular_distance=2.5,
    normal_ray_axial_weight=0.05,
    query_chunk_size=5000,
    workers=-1,
):
    """
    已停用的旧实验函数；当前main流程不会调用。

    让PCD局部表面邻域的多条反法向共同投票寻找内部骨架instance。

    skeleton_instances列格式必须至少为：x,y,z,r,g,b,label。
    label>0才是有效instance；physical层中的junction label=0不会成为
    独立instance。

    对每个PCD点，先取得附近且法向方向相容的PCD邻居。每个邻居都沿
    自己的反法向对候选骨架投票。只有获得足够邻域支持的骨架候选才能
    给当前PCD点赋值，因此不会因为单个法向量噪声或纯欧氏距离而轻易
    串到相邻线粒体。

    没有可靠投票结果的PCD点不会退回最近骨架，而是从附近已经可靠
    分割、法向方向相容的PCD点继承结果。
    """
    check_minimum_columns("PCD", pcd, minimum_columns=6)
    check_minimum_columns(
        f"{layer_name} skeleton",
        skeleton_instances,
        minimum_columns=7,
    )

    all_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)
    valid_rows = np.where(all_labels > 0)[0]

    if len(valid_rows) == 0:
        raise ValueError(
            f"{layer_name} skeleton contains no valid label > 0."
        )

    seed_xyz = skeleton_instances[valid_rows, :3]
    seed_labels = all_labels[valid_rows]
    seed_rgb = skeleton_instances[valid_rows, 3:6]

    unique_labels, canonical_colors = build_canonical_colors(
        labels=seed_labels,
        rgb=seed_rgb,
        layer_name=layer_name,
    )

    skeleton_tree = cKDTree(seed_xyz)
    candidate_count = min(int(normal_candidate_count), len(seed_xyz))
    neighborhood_count = min(int(normal_neighborhood_count), len(pcd))
    if candidate_count < 1:
        raise ValueError("normal_candidate_count must be at least 1.")
    if neighborhood_count < 1:
        raise ValueError("normal_neighborhood_count must be at least 1.")
    if normal_neighborhood_radius <= 0:
        raise ValueError("normal_neighborhood_radius must be positive.")
    if not -1.0 <= normal_neighbor_min_cosine <= 1.0:
        raise ValueError(
            "normal_neighbor_min_cosine must be between -1 and 1."
        )
    if not 0.0 < normal_min_support_fraction <= 1.0:
        raise ValueError(
            "normal_min_support_fraction must be in (0, 1]."
        )
    if normal_min_support_count < 1:
        raise ValueError("normal_min_support_count must be at least 1.")
    if normal_search_max_distance <= 0:
        raise ValueError("normal_search_max_distance must be positive.")
    if normal_ray_max_perpendicular_distance < 0:
        raise ValueError(
            "normal_ray_max_perpendicular_distance cannot be negative."
        )
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be at least 1.")

    point_count = len(pcd)
    pcd_xyz = pcd[:, :3]
    unit_normals = pcd[:, 3:6].copy()
    normal_length = np.linalg.norm(unit_normals, axis=1)
    valid_normal = normal_length > 1e-8
    unit_normals[valid_normal] /= normal_length[valid_normal, None]
    unit_normals[~valid_normal] = 0.0

    pcd_tree = cKDTree(pcd_xyz)
    assigned_seed_rows = np.full(point_count, -1, dtype=np.int64)
    selected_distances = np.full(point_count, np.nan, dtype=np.float64)
    reliable_match_count = 0
    insufficient_vote_count = 0

    for chunk_start in range(0, point_count, query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, point_count)
        point_xyz = pcd_xyz[chunk_start:chunk_end]
        point_normal = unit_normals[chunk_start:chunk_end]
        point_valid_normal = valid_normal[chunk_start:chunk_end]

        neighbor_distances, neighbor_rows = pcd_tree.query(
            point_xyz,
            k=neighborhood_count,
            workers=workers,
        )
        if neighborhood_count == 1:
            neighbor_distances = neighbor_distances[:, None]
            neighbor_rows = neighbor_rows[:, None]

        neighbor_normals = unit_normals[neighbor_rows]
        neighbor_normal_valid = valid_normal[neighbor_rows]
        normal_cosine = np.einsum(
            "nmd,nd->nm",
            neighbor_normals,
            point_normal,
        )
        active_neighbor = (
            point_valid_normal[:, None]
            & neighbor_normal_valid
            & (neighbor_distances <= normal_neighborhood_radius)
            & (normal_cosine >= normal_neighbor_min_cosine)
        )
        active_neighbor_count = np.sum(active_neighbor, axis=1)

        candidate_distances, candidate_rows = skeleton_tree.query(
            point_xyz,
            k=candidate_count,
            workers=workers,
        )
        if candidate_count == 1:
            candidate_distances = candidate_distances[:, None]
            candidate_rows = candidate_rows[:, None]

        candidate_xyz = seed_xyz[candidate_rows]
        vote_count = np.zeros(
            (len(point_xyz), candidate_count),
            dtype=np.int16,
        )
        vote_score_sum = np.zeros(
            (len(point_xyz), candidate_count),
            dtype=np.float64,
        )

        # 每个局部表面邻居分别用自己的反法向给候选骨架投票。
        # 按邻居循环可以避免创建 n×neighbor×candidate×3 的巨大数组。
        for neighbor_position in range(neighborhood_count):
            neighbor_is_active = active_neighbor[:, neighbor_position]
            if not np.any(neighbor_is_active):
                continue

            neighbor_xyz = pcd_xyz[
                neighbor_rows[:, neighbor_position]
            ]
            inward_direction = -neighbor_normals[:, neighbor_position]
            candidate_vector = candidate_xyz - neighbor_xyz[:, None, :]
            axial_depth = np.einsum(
                "nkd,nd->nk",
                candidate_vector,
                inward_direction,
            )
            squared_distance = np.sum(candidate_vector ** 2, axis=2)
            perpendicular_distance = np.sqrt(
                np.maximum(squared_distance - axial_depth ** 2, 0.0)
            )
            one_vote = (
                neighbor_is_active[:, None]
                & (axial_depth > 1e-8)
                & (
                    perpendicular_distance
                    <= normal_ray_max_perpendicular_distance
                )
            )
            one_score = (
                perpendicular_distance
                + normal_ray_axial_weight * axial_depth
            )
            vote_count += one_vote.astype(np.int16)
            vote_score_sum += np.where(one_vote, one_score, 0.0)

        required_vote_count = np.ceil(
            normal_min_support_fraction
            * active_neighbor_count
        ).astype(np.int32)
        required_vote_count = np.maximum(
            required_vote_count,
            int(normal_min_support_count),
        )
        # 当局部本来就少于minimum count时，不能要求超过现有邻居数。
        required_vote_count = np.minimum(
            required_vote_count,
            active_neighbor_count,
        )
        required_vote_count = np.maximum(required_vote_count, 1)

        candidate_is_reliable = (
            (candidate_distances <= normal_search_max_distance)
            & (vote_count >= required_vote_count[:, None])
            & (active_neighbor_count[:, None] > 0)
        )
        maximum_vote = np.max(
            np.where(candidate_is_reliable, vote_count, -1),
            axis=1,
        )
        has_reliable = maximum_vote >= 0

        mean_vote_score = vote_score_sum / np.maximum(vote_count, 1)
        top_vote_candidate = (
            candidate_is_reliable
            & (vote_count == maximum_vote[:, None])
        )
        final_score = np.where(
            top_vote_candidate,
            mean_vote_score,
            np.inf,
        )
        selected_candidate = np.argmin(final_score, axis=1)
        local_rows = np.arange(len(point_xyz), dtype=np.int64)
        chosen_seed_rows = candidate_rows[
            local_rows,
            selected_candidate,
        ]
        chosen_distances = candidate_distances[
            local_rows,
            selected_candidate,
        ]

        global_rows = np.arange(chunk_start, chunk_end, dtype=np.int64)
        reliable_global_rows = global_rows[has_reliable]
        assigned_seed_rows[reliable_global_rows] = chosen_seed_rows[
            has_reliable
        ]
        selected_distances[reliable_global_rows] = chosen_distances[
            has_reliable
        ]
        reliable_match_count += int(np.sum(has_reliable))
        insufficient_vote_count += int(np.sum(~has_reliable))

    # 不可靠点只向已经可靠分割的PCD表面点传播，绝不退回最近骨架。
    reliable_rows = np.where(assigned_seed_rows >= 0)[0]
    unresolved_rows = np.where(assigned_seed_rows < 0)[0]
    propagated_compatible_count = 0
    emergency_pcd_count = 0

    if len(reliable_rows) == 0:
        raise ValueError(
            f"{layer_name}: no PCD point obtained a reliable local-normal "
            "vote. Check whether PCD normals point outward, or relax the "
            "normal-neighborhood parameters."
        )

    if len(unresolved_rows) > 0:
        reliable_tree = cKDTree(pcd_xyz[reliable_rows])
        propagation_k = min(16, len(reliable_rows))
        propagation_distance, propagation_local_rows = reliable_tree.query(
            pcd_xyz[unresolved_rows],
            k=propagation_k,
            workers=workers,
        )
        if propagation_k == 1:
            propagation_distance = propagation_distance[:, None]
            propagation_local_rows = propagation_local_rows[:, None]

        propagation_global_rows = reliable_rows[propagation_local_rows]
        unresolved_normals = unit_normals[unresolved_rows]
        resolved_neighbor_normals = unit_normals[propagation_global_rows]
        propagation_cosine = np.einsum(
            "nkd,nd->nk",
            resolved_neighbor_normals,
            unresolved_normals,
        )
        compatible = (
            valid_normal[unresolved_rows, None]
            & valid_normal[propagation_global_rows]
            & (propagation_cosine >= normal_neighbor_min_cosine)
        )
        compatible_distance = np.where(
            compatible,
            propagation_distance,
            np.inf,
        )
        compatible_best = np.argmin(compatible_distance, axis=1)
        has_compatible = np.any(compatible, axis=1)

        selected_propagation = np.zeros(
            len(unresolved_rows),
            dtype=np.int64,
        )
        selected_propagation[has_compatible] = compatible_best[
            has_compatible
        ]
        source_rows = propagation_global_rows[
            np.arange(len(unresolved_rows)),
            selected_propagation,
        ]
        assigned_seed_rows[unresolved_rows] = assigned_seed_rows[source_rows]
        selected_distances[unresolved_rows] = np.linalg.norm(
            pcd_xyz[unresolved_rows]
            - seed_xyz[assigned_seed_rows[unresolved_rows]],
            axis=1,
        )
        propagated_compatible_count = int(np.sum(has_compatible))
        emergency_pcd_count = int(np.sum(~has_compatible))

    assigned_labels = seed_labels[assigned_seed_rows]

    # unique_labels已经排序，可用searchsorted快速把内部label映射到RGB。
    color_positions = np.searchsorted(unique_labels, assigned_labels)
    assigned_rgb = canonical_colors[color_positions]

    print(f"{layer_name} PCD neighborhood-normal assignment:")
    print(f"    Valid skeleton seeds : {len(valid_rows)}")
    print(f"    Skeleton instances   : {len(unique_labels)}")
    print(f"    PCD points assigned  : {point_count}")
    print(f"    Reliable local votes : {reliable_match_count}")
    print(f"    Insufficient votes   : {insufficient_vote_count}")
    print(f"    Propagated compatible: {propagated_compatible_count}")
    print(f"    Emergency PCD inherit: {emergency_pcd_count}")
    print(f"    Invalid normals      : {int(np.sum(~valid_normal))}")
    print("    Nearest skeleton fallback: 0")
    print("    Unassigned PCD points: 0")
    summarize_distances(
        "    Selected PCD-to-skeleton distance",
        selected_distances,
    )

    return assigned_rgb, assigned_labels, selected_distances


def assign_pcd_from_nearest_skeleton(
    pcd,
    skeleton_instances,
    layer_name,
    workers=-1,
):
    """
    按照最近有效骨架点快速完成PCD初始分割。

    label只在内部使用，不会写入最终文件。physical层的junction label=0
    不作为instance种子。
    """
    check_minimum_columns("PCD", pcd, minimum_columns=6)
    check_minimum_columns(
        f"{layer_name} skeleton",
        skeleton_instances,
        minimum_columns=7,
    )

    all_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)
    valid_rows = np.where(all_labels > 0)[0]
    if len(valid_rows) == 0:
        raise ValueError(
            f"{layer_name} skeleton contains no valid label > 0."
        )

    seed_xyz = skeleton_instances[valid_rows, :3]
    seed_labels = all_labels[valid_rows]
    seed_rgb = skeleton_instances[valid_rows, 3:6]
    unique_labels, canonical_colors = build_canonical_colors(
        labels=seed_labels,
        rgb=seed_rgb,
        layer_name=layer_name,
    )

    skeleton_tree = cKDTree(seed_xyz)
    distances, nearest_seed_rows = skeleton_tree.query(
        pcd[:, :3],
        k=1,
        workers=workers,
    )
    assigned_labels = seed_labels[nearest_seed_rows]
    color_positions = np.searchsorted(unique_labels, assigned_labels)
    assigned_rgb = canonical_colors[color_positions]

    print(f"{layer_name} initial nearest-skeleton PCD assignment:")
    print(f"    Valid skeleton seeds : {len(valid_rows)}")
    print(f"    Skeleton instances   : {len(unique_labels)}")
    print(f"    PCD points assigned  : {len(pcd)}")
    print("    Unassigned PCD points: 0")
    summarize_distances(
        "    PCD-to-nearest-skeleton distance",
        distances,
    )

    return assigned_rgb, assigned_labels, distances


def estimate_pcd_connect_radius(
    pcd_xyz,
    connect_radius_factor=2.5,
    workers=-1,
):
    """根据PCD典型最近邻距离自动估计表面连通半径。"""
    if len(pcd_xyz) < 2:
        raise ValueError("At least two PCD points are required.")
    if connect_radius_factor <= 0:
        raise ValueError("connect_radius_factor must be positive.")

    tree = cKDTree(pcd_xyz)
    distances, _ = tree.query(pcd_xyz, k=2, workers=workers)
    nearest_distance = distances[:, 1]
    positive_distance = nearest_distance[
        np.isfinite(nearest_distance) & (nearest_distance > 1e-8)
    ]
    if len(positive_distance) == 0:
        raise ValueError(
            "Could not estimate PCD spacing because all nearest-neighbor "
            "distances are zero or invalid."
        )

    median_nearest_distance = float(np.median(positive_distance))
    connect_radius = median_nearest_distance * connect_radius_factor
    print("Automatic PCD connectivity radius:")
    print(
        "    Median nearest-neighbor distance: "
        f"{median_nearest_distance:.4f}"
    )
    print(f"    Radius factor                 : {connect_radius_factor:.3f}")
    print(f"    PCD connectivity radius       : {connect_radius:.4f}")
    return connect_radius


def compute_same_instance_pcd_components(
    pcd_xyz,
    assigned_labels,
    connect_radius,
):
    """只连接距离足够近且当前内部label相同的PCD点。"""
    if connect_radius <= 0:
        raise ValueError("connect_radius must be positive.")

    point_count = len(pcd_xyz)
    tree = cKDTree(pcd_xyz)
    neighbor_pairs = tree.query_pairs(
        r=connect_radius,
        output_type="ndarray",
    )

    if len(neighbor_pairs) == 0:
        component_labels = np.arange(point_count, dtype=np.int64)
        return component_labels, 0, 0

    same_label = (
        assigned_labels[neighbor_pairs[:, 0]]
        == assigned_labels[neighbor_pairs[:, 1]]
    )
    same_pairs = neighbor_pairs[same_label]

    if len(same_pairs) == 0:
        component_labels = np.arange(point_count, dtype=np.int64)
        return component_labels, len(neighbor_pairs), 0

    graph_rows = np.concatenate([same_pairs[:, 0], same_pairs[:, 1]])
    graph_columns = np.concatenate([same_pairs[:, 1], same_pairs[:, 0]])
    graph = csr_matrix(
        (
            np.ones(len(graph_rows), dtype=np.uint8),
            (graph_rows, graph_columns),
        ),
        shape=(point_count, point_count),
    )
    _, component_labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return component_labels, len(neighbor_pairs), len(same_pairs)


def repair_disconnected_pcd_instances(
    pcd,
    assigned_labels,
    skeleton_instances,
    layer_name,
    connect_radius=None,
    connect_radius_factor=2.5,
    repair_iterations=2,
    reassignment_neighbor_count=64,
    reassignment_max_distance_factor=2.0,
    reassignment_min_vote_fraction=0.60,
    reassignment_min_other_neighbors=3,
    component_vote_sample_points=500,
    workers=-1,
):
    """
    修复同一个初始instance中明显断开的PCD component。

    每个内部label中，与该label骨架结合最紧密、获得最多骨架点支持的
    component作为主体。其他断开component只有在周围另一个label获得
    足够强的距离加权投票时才整体改归属；没有明确证据时保持不变。
    """
    pcd_xyz = np.asarray(pcd[:, :3], dtype=np.float64)
    labels = np.asarray(assigned_labels, dtype=np.int64).copy()
    skeleton_labels = np.rint(skeleton_instances[:, 6]).astype(np.int64)
    valid_skeleton_rows = np.where(skeleton_labels > 0)[0]
    skeleton_xyz = skeleton_instances[valid_skeleton_rows, :3]
    skeleton_labels = skeleton_labels[valid_skeleton_rows]

    unique_labels, canonical_colors = build_canonical_colors(
        labels=skeleton_labels,
        rgb=skeleton_instances[valid_skeleton_rows, 3:6],
        layer_name=layer_name,
    )

    if connect_radius is None:
        connect_radius = estimate_pcd_connect_radius(
            pcd_xyz=pcd_xyz,
            connect_radius_factor=connect_radius_factor,
            workers=workers,
        )
    else:
        connect_radius = float(connect_radius)
        if connect_radius <= 0:
            raise ValueError("connect_radius must be positive or None.")
        print(f"Fixed PCD connectivity radius: {connect_radius:.4f}")

    if repair_iterations < 0:
        raise ValueError("repair_iterations cannot be negative.")
    if reassignment_neighbor_count < 1:
        raise ValueError("reassignment_neighbor_count must be at least 1.")
    if reassignment_max_distance_factor <= 0:
        raise ValueError(
            "reassignment_max_distance_factor must be positive."
        )
    if not 0.0 < reassignment_min_vote_fraction <= 1.0:
        raise ValueError(
            "reassignment_min_vote_fraction must be in (0, 1]."
        )
    if reassignment_min_other_neighbors < 1:
        raise ValueError(
            "reassignment_min_other_neighbors must be at least 1."
        )
    if component_vote_sample_points < 1:
        raise ValueError("component_vote_sample_points must be at least 1.")

    pcd_tree = cKDTree(pcd_xyz)
    neighbor_count = min(int(reassignment_neighbor_count), len(pcd_xyz))
    reassignment_max_distance = (
        connect_radius * reassignment_max_distance_factor
    )
    total_reassigned_components = 0
    total_reassigned_points = 0

    print(f"{layer_name} disconnected-component repair:")
    print(f"    Connectivity radius       : {connect_radius:.4f}")
    print(f"    Reassignment max distance : {reassignment_max_distance:.4f}")
    print(f"    Maximum repair iterations : {repair_iterations}")

    for iteration in range(1, repair_iterations + 1):
        component_ids, all_pair_count, same_pair_count = (
            compute_same_instance_pcd_components(
                pcd_xyz=pcd_xyz,
                assigned_labels=labels,
                connect_radius=connect_radius,
            )
        )

        proposals = []
        split_instance_count = 0
        detached_component_count = 0
        unresolved_component_count = 0

        for one_label in np.unique(labels):
            label_rows = np.where(labels == one_label)[0]
            one_component_ids = np.unique(component_ids[label_rows])
            if len(one_component_ids) <= 1:
                continue

            split_instance_count += 1
            component_rows = {
                int(component_id): label_rows[
                    component_ids[label_rows] == component_id
                ]
                for component_id in one_component_ids
            }

            # 该label的骨架点分别落到哪个PCD component附近。
            one_skeleton_xyz = skeleton_xyz[skeleton_labels == one_label]
            label_tree = cKDTree(pcd_xyz[label_rows])
            skeleton_distance, nearest_local_rows = label_tree.query(
                one_skeleton_xyz,
                k=1,
                workers=workers,
            )
            nearest_global_rows = label_rows[nearest_local_rows]
            skeleton_hit_components = component_ids[nearest_global_rows]

            anchor_component = None
            anchor_key = None
            for component_id in one_component_ids:
                component_id = int(component_id)
                rows = component_rows[component_id]
                hit_mask = skeleton_hit_components == component_id
                hit_count = int(np.sum(hit_mask))
                minimum_skeleton_distance = (
                    float(np.min(skeleton_distance[hit_mask]))
                    if np.any(hit_mask)
                    else np.inf
                )
                anchor_candidate_key = (
                    hit_count,
                    -minimum_skeleton_distance,
                    len(rows),
                )
                if anchor_key is None or anchor_candidate_key > anchor_key:
                    anchor_key = anchor_candidate_key
                    anchor_component = component_id

            for component_id in one_component_ids:
                component_id = int(component_id)
                if component_id == anchor_component:
                    continue

                detached_component_count += 1
                rows = component_rows[component_id]
                if len(rows) > component_vote_sample_points:
                    sample_positions = np.linspace(
                        0,
                        len(rows) - 1,
                        component_vote_sample_points,
                        dtype=np.int64,
                    )
                    sample_rows = rows[sample_positions]
                else:
                    sample_rows = rows

                neighbor_distance, neighbor_rows = pcd_tree.query(
                    pcd_xyz[sample_rows],
                    k=neighbor_count,
                    workers=workers,
                )
                if neighbor_count == 1:
                    neighbor_distance = neighbor_distance[:, None]
                    neighbor_rows = neighbor_rows[:, None]

                neighbor_labels = labels[neighbor_rows]
                valid_other = (
                    (neighbor_labels != one_label)
                    & (neighbor_distance <= reassignment_max_distance)
                )
                if int(np.sum(valid_other)) < reassignment_min_other_neighbors:
                    unresolved_component_count += 1
                    continue

                other_labels = neighbor_labels[valid_other]
                other_distances = neighbor_distance[valid_other]
                vote_weights = 1.0 / np.maximum(other_distances, 1e-6)
                candidate_labels, inverse = np.unique(
                    other_labels,
                    return_inverse=True,
                )
                candidate_scores = np.bincount(
                    inverse,
                    weights=vote_weights,
                )
                best_position = int(np.argmax(candidate_scores))
                best_label = int(candidate_labels[best_position])
                vote_fraction = float(
                    candidate_scores[best_position]
                    / np.sum(candidate_scores)
                )

                if vote_fraction < reassignment_min_vote_fraction:
                    unresolved_component_count += 1
                    continue

                proposals.append(
                    (rows, int(one_label), best_label, vote_fraction)
                )

        if len(proposals) > 0:
            for rows, old_label, new_label, vote_fraction in proposals:
                labels[rows] = new_label
                print(
                    f"    iteration={iteration} | points={len(rows)} | "
                    f"{old_label} -> {new_label} | "
                    f"vote={vote_fraction:.3f}"
                )
                total_reassigned_components += 1
                total_reassigned_points += len(rows)

        print(
            f"    Iteration {iteration}: split_instances={split_instance_count}, "
            f"detached_components={detached_component_count}, "
            f"reassigned_components={len(proposals)}, "
            f"unresolved_components={unresolved_component_count}, "
            f"all_neighbor_pairs={all_pair_count}, "
            f"same_label_pairs={same_pair_count}"
        )

        if len(proposals) == 0:
            break

    color_positions = np.searchsorted(unique_labels, labels)
    if np.any(color_positions >= len(unique_labels)):
        raise ValueError(
            f"{layer_name}: repaired PCD contains a label that is absent "
            "from the skeleton input."
        )
    if np.any(unique_labels[color_positions] != labels):
        raise ValueError(
            f"{layer_name}: repaired PCD contains a label that is absent "
            "from the skeleton input."
        )
    repaired_rgb = canonical_colors[color_positions]

    print(f"{layer_name} component repair summary:")
    print(f"    Components reassigned: {total_reassigned_components}")
    print(f"    PCD points reassigned : {total_reassigned_points}")
    print(f"    Final instances       : {len(np.unique(labels))}")

    return repaired_rgb, labels, connect_radius


def assign_voxel_from_segmented_pcd(
    voxel,
    pcd,
    pcd_rgb,
    pcd_labels,
    layer_name,
    workers=-1,
):
    """
    使用已经完成连通块修复的PCD为每个voxel赋值。

    voxel寻找最近PCD表面点，然后继承该PCD点的RGB。内部label仅用于打印
    检查统计，不写入最终文件。
    """
    check_minimum_columns("VOXEL", voxel, minimum_columns=4)

    pcd_tree = cKDTree(pcd[:, :3])
    distances, nearest_pcd_rows = pcd_tree.query(
        voxel[:, :3],
        k=1,
        workers=workers,
    )

    assigned_rgb = pcd_rgb[nearest_pcd_rows]
    assigned_labels = pcd_labels[nearest_pcd_rows]

    print(f"{layer_name} voxel assignment from segmented PCD:")
    print(f"    Voxels assigned      : {len(voxel)}")
    print(f"    Instances represented: {len(np.unique(assigned_labels))}")
    print("    Unassigned voxels    : 0")
    summarize_distances(
        "    Voxel-to-segmented-PCD nearest distance",
        distances,
    )

    return assigned_rgb, assigned_labels, distances


def save_voxel_xyzrgb_intensity(save_path, voxel, rgb):
    """保存格式：x,y,z,r,g,b,intensity。"""
    output = np.column_stack(
        [
            voxel[:, :3],
            np.asarray(rgb, dtype=np.int32),
            voxel[:, 3],
        ]
    )
    np.savetxt(
        save_path,
        output,
        delimiter=",",
        fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d", "%.6f"],
    )

def save_pcd_xyzrgbnormal_social_id(
    save_path,
    pcd,
    rgb,
    social_instance_ids,
):
    """保存格式：x,y,z,r,g,b,nx,ny,nz,social_instance_id。"""
    social_instance_ids = np.asarray(
        social_instance_ids,
        dtype=np.int64,
    )
    if len(social_instance_ids) != len(pcd):
        raise ValueError(
            "social_instance_ids and PCD must contain the same number "
            "of rows."
        )
    if np.any(social_instance_ids <= 0):
        raise ValueError(
            "Every PCD point must have a social_instance_id > 0."
        )

    output = np.column_stack(
        [
            pcd[:, :3],
            np.asarray(rgb, dtype=np.int32),
            pcd[:, 3:6],
            social_instance_ids,
        ]
    )
    np.savetxt(
        save_path,
        output,
        delimiter=",",
        fmt=[
            "%.3f",
            "%.3f",
            "%.3f",
            "%d",
            "%d",
            "%d",
            "%.6f",
            "%.6f",
            "%.6f",
            "%d",
        ],
    )


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
        #
        # # ====================================================
        # # Non-GLUC
        # # ====================================================
        #
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

    ###########################################################################
    # 运行参数：不使用argparse，直接在这里修改
    ###########################################################################

    # True：覆盖已有4个输出文件并重新计算。
    # False：如果4个输出文件都已存在，则直接跳过该图片。
    overwrite_existing = True

    # cKDTree并行线程数。-1表示使用全部可用CPU线程。
    workers = -1

    # None表示根据PCD典型最近邻距离自动计算连通半径。
    # 如需固定半径，可改成例如：pcd_connect_radius = 1.5
    pcd_connect_radius = None

    # 自动连通半径 = PCD最近邻距离中位数 × 该系数。
    pcd_connect_radius_factor = 2.5

    # 明显断开的PCD component最多修复多少轮。
    component_repair_iterations = 2

    # 对每个断开component采样点周围查询多少个PCD邻居进行归属投票。
    reassignment_neighbor_count = 64

    # 允许投票的最大距离 = PCD连通半径 × 该系数。
    reassignment_max_distance_factor = 2.0

    # 另一个instance的距离加权票数至少占多少，才整体重新归属。
    reassignment_min_vote_fraction = 0.60

    # 至少需要多少个其他instance邻居参与投票。
    reassignment_min_other_neighbors = 3

    # 每个断开component最多采样多少个点进行周围instance投票。
    component_vote_sample_points = 500

    total_done = 0
    total_skipped = 0

    for file_name in file_name_list:

        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")
        # BASE_DIR = Path(f"D:/public_data/S-BIAD1445/neuron/mito_test")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/{file_name}/mito")
        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")
        # BASE_DIR = Path(f"E:/data/20260811-YF_output/image_tif/single_cell_tifs")
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

        # 自动读取 processed_output 中的所有原始 *_pcd.txt。
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

        print("=========================================================================")
        print(f"Dataset            : {file_name}")
        print(f"BASE_DIR           : {BASE_DIR}")
        print(f"INPUT_PATH         : {INPUT_PATH}")
        print(f"OUTPUT_PATH        : {OUTPUT_PATH}")
        print(f"Selected images    : {len(imgs_list)}")
        print(f"Overwrite existing : {overwrite_existing}")
        print(f"PCD connect radius : {pcd_connect_radius}")
        print(f"PCD radius factor  : {pcd_connect_radius_factor}")
        print(f"Repair iterations  : {component_repair_iterations}")
        print(f"Reassign neighbors : {reassignment_neighbor_count}")
        print(
            "Reassign max factor: "
            f"{reassignment_max_distance_factor}"
        )
        print(
            "Reassign vote frac : "
            f"{reassignment_min_vote_fraction}"
        )
        print("=========================================================================")

        for i, img_name in enumerate(imgs_list, start=1):
            # 同时兼容：自动扫描得到的 xxx_pcd.txt 和手动填写的 xxx。
            if img_name.lower().endswith("_pcd.txt"):
                stem = img_name[:-len("_pcd.txt")]
            else:
                stem = img_name

            print("-------------------------------------------------------------------------")
            print(f"Processing {file_name} | {i}/{len(imgs_list)} | {stem}")
            print("-------------------------------------------------------------------------")

            ###################################################################
            # 输入文件路径
            ###################################################################

            pcd_path = INPUT_PATH / f"{stem}_pcd.txt"
            voxel_path = INPUT_PATH / f"{stem}_voxel.txt"
            skeleton_edges_path = INPUT_PATH / f"{stem}_skeleton_edges.txt"
            skeleton_social_path = (
                INPUT_PATH / f"{stem}_skeleton_instance_social.txt"
            )

            ###################################################################
            # 输出文件路径
            ###################################################################

            save_voxel_phy_path = (
                OUTPUT_PATH / f"{stem}_voxel_instance_phy.txt"
            )
            save_voxel_social_path = (
                OUTPUT_PATH / f"{stem}_voxel_instance_social.txt"
            )
            save_pcd_phy_path = (
                OUTPUT_PATH / f"{stem}_pcd_instance_physical.txt"
            )
            save_pcd_social_path = (
                OUTPUT_PATH / f"{stem}_pcd_instance_social.txt"
            )

            input_paths = {
                "PCD": pcd_path,
                "VOXEL": voxel_path,
                "PHYSICAL SKELETON EDGES": skeleton_edges_path,
                "SOCIAL SKELETON INSTANCES": skeleton_social_path,
            }
            output_paths = {
                "PHYSICAL VOXEL": save_voxel_phy_path,
                "SOCIAL VOXEL": save_voxel_social_path,
                "PHYSICAL PCD": save_pcd_phy_path,
                "SOCIAL PCD": save_pcd_social_path,
            }

            print("Input files:")
            for label, path in input_paths.items():
                print(f"    {label:<28}: {path}")

            print("Output files:")
            for label, path in output_paths.items():
                print(f"    {label:<28}: {path}")

            missing_paths = [
                path for path in input_paths.values() if not path.exists()
            ]
            if len(missing_paths) > 0:
                print("[SKIPPED] Missing required input files:")
                for missing_path in missing_paths:
                    print(f"    {missing_path}")
                total_skipped += 1
                continue

            if (
                not overwrite_existing
                and all(path.exists() for path in output_paths.values())
            ):
                print("[SKIPPED] All four output files already exist.")
                total_skipped += 1
                continue

            ###################################################################
            # 读取并检查输入
            ###################################################################

            pcd = load_txt_array(pcd_path)
            voxel = load_txt_array(voxel_path)
            skeleton_edges = load_txt_array(skeleton_edges_path)
            skeleton_social = load_txt_array(skeleton_social_path)

            print()
            print_file_information(
                "PCD",
                pcd_path,
                pcd,
                "x,y,z,nx,ny,nz",
            )
            check_minimum_columns("PCD", pcd, minimum_columns=6)

            print()
            print_file_information(
                "VOXEL",
                voxel_path,
                voxel,
                "x,y,z,intensity",
            )
            check_minimum_columns("VOXEL", voxel, minimum_columns=4)

            print()
            print_file_information(
                "PHYSICAL SKELETON EDGES",
                skeleton_edges_path,
                skeleton_edges,
                "x,y,z,r,g,b,edge_id",
            )
            check_minimum_columns(
                "PHYSICAL SKELETON EDGES",
                skeleton_edges,
                minimum_columns=7,
            )

            print()
            print_file_information(
                "SOCIAL SKELETON INSTANCES",
                skeleton_social_path,
                skeleton_social,
                "x,y,z,r,g,b,instance_id",
            )
            check_minimum_columns(
                "SOCIAL SKELETON INSTANCES",
                skeleton_social,
                minimum_columns=7,
            )

            ###################################################################
            # 第一层：最近physical edge初分PCD，修复断开块，再分voxel
            ###################################################################

            print()
            print("Initial PHYSICAL PCD assignment by nearest skeleton...")
            _, pcd_phy_labels, _ = assign_pcd_from_nearest_skeleton(
                pcd=pcd,
                skeleton_instances=skeleton_edges,
                layer_name="PHYSICAL",
                workers=workers,
            )
            pcd_phy_rgb, pcd_phy_labels, _ = (
                repair_disconnected_pcd_instances(
                    pcd=pcd,
                    assigned_labels=pcd_phy_labels,
                    skeleton_instances=skeleton_edges,
                    layer_name="PHYSICAL",
                    connect_radius=pcd_connect_radius,
                    connect_radius_factor=pcd_connect_radius_factor,
                    repair_iterations=component_repair_iterations,
                    reassignment_neighbor_count=(
                        reassignment_neighbor_count
                    ),
                    reassignment_max_distance_factor=(
                        reassignment_max_distance_factor
                    ),
                    reassignment_min_vote_fraction=(
                        reassignment_min_vote_fraction
                    ),
                    reassignment_min_other_neighbors=(
                        reassignment_min_other_neighbors
                    ),
                    component_vote_sample_points=(
                        component_vote_sample_points
                    ),
                    workers=workers,
                )
            )
            voxel_phy_rgb, _, _ = assign_voxel_from_segmented_pcd(
                voxel=voxel,
                pcd=pcd,
                pcd_rgb=pcd_phy_rgb,
                pcd_labels=pcd_phy_labels,
                layer_name="PHYSICAL",
                workers=workers,
            )

            save_voxel_xyzrgb_intensity(
                save_path=save_voxel_phy_path,
                voxel=voxel,
                rgb=voxel_phy_rgb,
            )

            print("PHYSICAL voxel layer saved:")
            print(f"    {save_voxel_phy_path}")
            print("    Format: x,y,z,r,g,b,intensity")

            ###################################################################
            # 第二层：最近social骨架初分PCD，修复断开块，再分voxel
            ###################################################################

            print()
            print("Initial SOCIAL PCD assignment by nearest skeleton...")
            _, pcd_social_labels, _ = assign_pcd_from_nearest_skeleton(
                pcd=pcd,
                skeleton_instances=skeleton_social,
                layer_name="SOCIAL",
                workers=workers,
            )
            pcd_social_rgb, pcd_social_labels, _ = (
                repair_disconnected_pcd_instances(
                    pcd=pcd,
                    assigned_labels=pcd_social_labels,
                    skeleton_instances=skeleton_social,
                    layer_name="SOCIAL",
                    connect_radius=pcd_connect_radius,
                    connect_radius_factor=pcd_connect_radius_factor,
                    repair_iterations=component_repair_iterations,
                    reassignment_neighbor_count=(
                        reassignment_neighbor_count
                    ),
                    reassignment_max_distance_factor=(
                        reassignment_max_distance_factor
                    ),
                    reassignment_min_vote_fraction=(
                        reassignment_min_vote_fraction
                    ),
                    reassignment_min_other_neighbors=(
                        reassignment_min_other_neighbors
                    ),
                    component_vote_sample_points=(
                        component_vote_sample_points
                    ),
                    workers=workers,
                )
            )
            voxel_social_rgb, _, _ = assign_voxel_from_segmented_pcd(
                voxel=voxel,
                pcd=pcd,
                pcd_rgb=pcd_social_rgb,
                pcd_labels=pcd_social_labels,
                layer_name="SOCIAL",
                workers=workers,
            )

            # 两个PCD文件最后一列都使用同一套最终social instance ID。
            # physical文件的RGB仍表示physical edge，不额外保存physical ID。
            save_pcd_xyzrgbnormal_social_id(
                save_path=save_pcd_phy_path,
                pcd=pcd,
                rgb=pcd_phy_rgb,
                social_instance_ids=pcd_social_labels,
            )
            save_pcd_xyzrgbnormal_social_id(
                save_path=save_pcd_social_path,
                pcd=pcd,
                rgb=pcd_social_rgb,
                social_instance_ids=pcd_social_labels,
            )
            save_voxel_xyzrgb_intensity(
                save_path=save_voxel_social_path,
                voxel=voxel,
                rgb=voxel_social_rgb,
            )

            print("PHYSICAL and SOCIAL PCD layers saved:")
            print(f"    {save_pcd_phy_path}")
            print("    Format: x,y,z,r,g,b,nx,ny,nz,social_instance_id")
            print(f"    {save_pcd_social_path}")
            print("    Format: x,y,z,r,g,b,nx,ny,nz,social_instance_id")
            print("SOCIAL voxel layer saved:")
            print(f"    {save_voxel_social_path}")
            print("    Format: x,y,z,r,g,b,intensity")

            total_done += 1
            print()
            print(f"[DONE] Dual-layer mapping completed for: {stem}")

    print("=========================================================================")
    print("Dual-layer mitochondrial instance mapping finished.")
    print(f"Images processed : {total_done}")
    print(f"Images skipped   : {total_skipped}")
    print("Saved files:")
    print("    *_voxel_instance_phy.txt")
    print("    *_voxel_instance_social.txt")
    print("    *_pcd_instance_physical.txt")
    print("    *_pcd_instance_social.txt")
    print("Both PCD files end with the same social_instance_id column.")
    print("No separate mapping CSV or physical ID column was saved.")
    print("=========================================================================")


if __name__ == "__main__":
    main()

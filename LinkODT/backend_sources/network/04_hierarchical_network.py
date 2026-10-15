from pathlib import Path
import heapq
from itertools import combinations
import os
import pickle

import numpy as np
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree, distance
from scipy.stats import rankdata


EPS = 1e-12


def load_txt_array(path, minimum_columns, label):
    """读取逗号分隔且无表头的数值txt。"""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < minimum_columns:
        raise ValueError(
            f"{label} requires at least {minimum_columns} columns, "
            f"but {path} has shape {data.shape}."
        )
    return data


def safe_divide(numerator, denominator):
    if not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    mean_value = float(np.mean(values))
    if abs(mean_value) <= EPS:
        return np.nan
    return float(np.std(values, ddof=0) / mean_value)


def normalized_entropy(values):
    """把非负values归一化成概率后计算[0,1] Shannon entropy。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else np.nan
    probabilities = values / np.sum(values)
    entropy = -np.sum(probabilities * np.log(probabilities + EPS))
    return float(entropy / np.log(len(probabilities)))


def zscore_finite(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if np.sum(finite) < 2:
        return output
    standard_deviation = float(np.std(values[finite], ddof=0))
    if standard_deviation <= EPS:
        output[finite] = 0.0
    else:
        output[finite] = (
            values[finite] - np.mean(values[finite])
        ) / standard_deviation
    return output


def finite_distribution_summary(values, prefix):
    """对一个预先指定的instance属性输出稳健分布摘要。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_median": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
        }
    q10, q25, q50, q75, q90 = np.quantile(
        values,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )
    return {
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p90": float(q90),
    }


def ranked_tail_masks(values, fraction, descending=False):
    """
    按rank选固定比例，避免quantile阈值遇到大量ties时选中数量不稳定。

    返回selected和其余有效instance的布尔mask。
    """
    values = np.asarray(values, dtype=np.float64)
    valid_rows = np.where(np.isfinite(values))[0]
    selected = np.zeros(len(values), dtype=bool)
    remainder = np.zeros(len(values), dtype=bool)
    if len(valid_rows) == 0:
        return selected, remainder
    selected_count = max(1, int(np.ceil(len(valid_rows) * fraction)))
    order = np.argsort(values[valid_rows], kind="mergesort")
    if descending:
        order = order[::-1]
    selected_rows = valid_rows[order[:selected_count]]
    selected[selected_rows] = True
    remainder[valid_rows] = True
    remainder[selected_rows] = False
    return selected, remainder


def median_for_mask(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    return float(np.median(values[mask])) if np.any(mask) else np.nan


def gini_coefficient(values):
    """非负contact load的Gini；全0时没有可定义的contact concentration。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) == 0 or np.sum(values) <= EPS:
        return np.nan
    values = np.sort(values)
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(ranks * values) / (len(values) * np.sum(values)))
        - (len(values) + 1.0) / len(values)
    )


def top_fraction_share(values, fraction=0.10):
    """非负总量中，数值最高fraction实例贡献的份额。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    total = float(np.sum(values))
    if len(values) == 0 or total <= EPS:
        return np.nan
    selected_count = max(1, int(np.ceil(len(values) * fraction)))
    selected = np.partition(values, len(values) - selected_count)[
        -selected_count:
    ]
    return float(np.sum(selected) / total)


def spearman_correlation(values_a, values_b, minimum_n=6):
    """带最小样本量和常数检查的Spearman correlation。"""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if int(np.sum(valid)) < minimum_n:
        return np.nan
    ranked_a = rankdata(values_a[valid], method="average")
    ranked_b = rankdata(values_b[valid], method="average")
    return safe_correlation(ranked_a, ranked_b, minimum_n=minimum_n)


def safe_correlation(values_a, values_b, minimum_n=4):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if np.sum(valid) < minimum_n:
        return np.nan
    if (
        np.std(values_a[valid]) <= EPS
        or np.std(values_b[valid]) <= EPS
    ):
        return np.nan
    return float(np.corrcoef(values_a[valid], values_b[valid])[0, 1])


def build_pruned_skeleton_adjacency(
    xyz,
    neighbor_radius=1.8,
    prune_shortcut_edges=True,
):
    """
    使用与前面edge分割相同的半径邻接和relative-neighborhood剪枝。

    返回二值拓扑邻接、以真实坐标距离为cost的邻接，以及保留的点对。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    point_count = len(xyz)
    tree = cKDTree(xyz)
    candidate_pairs = tree.query_pairs(
        r=neighbor_radius,
        output_type="ndarray",
    )

    candidate_neighbors = [set() for _ in range(point_count)]
    for point_a, point_b in candidate_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        candidate_neighbors[point_a].add(point_b)
        candidate_neighbors[point_b].add(point_a)

    kept_pairs = []
    kept_lengths = []
    for point_a, point_b in candidate_pairs:
        point_a = int(point_a)
        point_b = int(point_b)
        distance_ab = float(np.linalg.norm(xyz[point_a] - xyz[point_b]))
        keep_pair = True

        if prune_shortcut_edges:
            common_neighbors = (
                candidate_neighbors[point_a]
                & candidate_neighbors[point_b]
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
            kept_pairs.append((point_a, point_b))
            kept_lengths.append(distance_ab)

    if len(kept_pairs) == 0:
        empty = csr_matrix((point_count, point_count), dtype=np.float64)
        return empty, empty.copy(), np.empty((0, 2), dtype=np.int64)

    kept_pairs = np.asarray(kept_pairs, dtype=np.int64)
    kept_lengths = np.asarray(kept_lengths, dtype=np.float64)
    rows = np.concatenate([kept_pairs[:, 0], kept_pairs[:, 1]])
    columns = np.concatenate([kept_pairs[:, 1], kept_pairs[:, 0]])

    binary_adjacency = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (rows, columns),
        ),
        shape=(point_count, point_count),
    )
    length_adjacency = csr_matrix(
        (
            np.concatenate([kept_lengths, kept_lengths]),
            (rows, columns),
        ),
        shape=(point_count, point_count),
    )
    return binary_adjacency, length_adjacency, kept_pairs


def edge_geometry(edge_xyz, neighbor_radius):
    """用edge内部MST长度计算arc length、chord和tortuosity。"""
    edge_xyz = np.asarray(edge_xyz, dtype=np.float64)
    point_count = len(edge_xyz)
    if point_count == 0:
        raise ValueError("A physical edge cannot contain zero points.")
    if point_count == 1:
        return 0.0, 0.0, 1.0, edge_xyz[0], edge_xyz[0]

    tree = cKDTree(edge_xyz)
    pairs = tree.query_pairs(r=neighbor_radius, output_type="ndarray")
    if len(pairs) == 0:
        # 极少数不连续edge的保底：按最近邻贪心排序。
        remaining = set(range(1, point_count))
        order = [0]
        while remaining:
            current = order[-1]
            next_index = min(
                remaining,
                key=lambda index: np.linalg.norm(
                    edge_xyz[index] - edge_xyz[current]
                ),
            )
            order.append(next_index)
            remaining.remove(next_index)
        ordered_xyz = edge_xyz[np.asarray(order, dtype=np.int64)]
        arc_length = float(
            np.sum(np.linalg.norm(np.diff(ordered_xyz, axis=0), axis=1))
        )
        start_xyz = ordered_xyz[0]
        end_xyz = ordered_xyz[-1]
    else:
        pair_lengths = np.linalg.norm(
            edge_xyz[pairs[:, 0]] - edge_xyz[pairs[:, 1]],
            axis=1,
        )
        graph = csr_matrix(
            (
                np.concatenate([pair_lengths, pair_lengths]),
                (
                    np.concatenate([pairs[:, 0], pairs[:, 1]]),
                    np.concatenate([pairs[:, 1], pairs[:, 0]]),
                ),
            ),
            shape=(point_count, point_count),
        )
        mst = minimum_spanning_tree(graph)
        arc_length = float(np.sum(mst.data))

        mst_undirected = mst + mst.T
        first_distances = dijkstra(
            mst_undirected,
            directed=False,
            indices=0,
        )
        first_endpoint = int(np.nanargmax(first_distances))
        second_distances = dijkstra(
            mst_undirected,
            directed=False,
            indices=first_endpoint,
        )
        second_endpoint = int(np.nanargmax(second_distances))
        start_xyz = edge_xyz[first_endpoint]
        end_xyz = edge_xyz[second_endpoint]

    chord_length = float(np.linalg.norm(end_xyz - start_xyz))
    tortuosity = (
        float(arc_length / chord_length)
        if chord_length > EPS
        else 1.0
    )
    return arc_length, chord_length, tortuosity, start_xyz, end_xyz


def build_physical_edge_table(
    skeleton_edges,
    skeleton_social_ids,
    coordinate_scale_xyz,
    neighbor_radius,
):
    """每个edge_id>0输出一行，并显式保留social_instance_id。"""
    edge_ids = np.rint(skeleton_edges[:, 6]).astype(np.int64)
    scaled_xyz = skeleton_edges[:, :3] * coordinate_scale_xyz[None, :]
    records = []

    for edge_id in sorted(np.unique(edge_ids[edge_ids > 0])):
        rows = np.where(edge_ids == edge_id)[0]
        social_values = np.unique(skeleton_social_ids[rows])
        if len(social_values) != 1:
            raise ValueError(
                f"physical edge {edge_id} belongs to multiple social IDs: "
                f"{social_values.tolist()}"
            )
        social_id = int(social_values[0])
        arc, chord, tortuosity, start_xyz, end_xyz = edge_geometry(
            edge_xyz=scaled_xyz[rows],
            neighbor_radius=(
                neighbor_radius * float(np.max(coordinate_scale_xyz))
            ),
        )
        rgb = np.rint(np.median(skeleton_edges[rows, 3:6], axis=0)).astype(int)
        records.append(
            {
                "physical_edge_id": int(edge_id),
                "social_instance_id": social_id,
                "skeleton_point_count": int(len(rows)),
                "physical_r": int(rgb[0]),
                "physical_g": int(rgb[1]),
                "physical_b": int(rgb[2]),
                "arc_length": arc,
                "chord_length": chord,
                "tortuosity": tortuosity,
                "start_x": float(start_xyz[0]),
                "start_y": float(start_xyz[1]),
                "start_z": float(start_xyz[2]),
                "end_x": float(end_xyz[0]),
                "end_y": float(end_xyz[1]),
                "end_z": float(end_xyz[2]),
            }
        )
    return pd.DataFrame.from_records(records)


def normalized_algebraic_connectivity(binary_adjacency):
    """归一化Laplacian的第二小特征值；不连通时为0。"""
    node_count = binary_adjacency.shape[0]
    if node_count < 2:
        return np.nan
    component_count, _ = connected_components(binary_adjacency, directed=False)
    if component_count > 1:
        return 0.0
    degrees = np.asarray(binary_adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros(node_count, dtype=np.float64)
    positive = degrees > EPS
    inverse_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    laplacian = (
        eye(node_count, format="csr")
        - diags(inverse_sqrt)
        @ binary_adjacency
        @ diags(inverse_sqrt)
    )
    if node_count <= 200:
        eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
        return float(max(eigenvalues[1], 0.0))
    eigenvalues = eigsh(
        laplacian,
        k=2,
        which="SM",
        return_eigenvectors=False,
    )
    eigenvalues = np.sort(np.real(eigenvalues))
    return float(max(eigenvalues[1], 0.0))


def sampled_route_efficiency(
    length_adjacency,
    xyz,
    maximum_nodes=250,
    random_seed=42,
):
    """平均 Euclidean distance / skeleton geodesic distance。"""
    node_count = length_adjacency.shape[0]
    if node_count < 2:
        return np.nan
    if node_count > maximum_nodes:
        generator = np.random.default_rng(random_seed)
        sampled = np.sort(
            generator.choice(node_count, size=maximum_nodes, replace=False)
        )
    else:
        sampled = np.arange(node_count, dtype=np.int64)
    geodesic = dijkstra(
        length_adjacency,
        directed=False,
        indices=sampled,
    )[:, sampled]
    euclidean = distance.squareform(distance.pdist(xyz[sampled]))
    upper = np.triu_indices(len(sampled), k=1)
    geodesic_values = geodesic[upper]
    euclidean_values = euclidean[upper]
    valid = (
        np.isfinite(geodesic_values)
        & (geodesic_values > EPS)
        & np.isfinite(euclidean_values)
    )
    if not np.any(valid):
        return np.nan
    ratios = euclidean_values[valid] / geodesic_values[valid]
    ratios = np.clip(ratios, 0.0, 1.0)
    return float(np.mean(ratios))


def mean_effective_resistance(
    length_adjacency,
    maximum_exact_nodes=400,
):
    """使用conductance=1/length的Laplacian谱计算平均有效电阻。"""
    node_count = length_adjacency.shape[0]
    if node_count < 2 or node_count > maximum_exact_nodes:
        return np.nan
    component_count, _ = connected_components(length_adjacency, directed=False)
    if component_count > 1:
        return np.nan
    coo = length_adjacency.tocoo()
    conductance_data = 1.0 / np.maximum(coo.data, EPS)
    conductance = csr_matrix(
        (conductance_data, (coo.row, coo.col)),
        shape=coo.shape,
    )
    degree = np.asarray(conductance.sum(axis=1)).ravel()
    laplacian = diags(degree) - conductance
    eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
    positive = eigenvalues[eigenvalues > 1e-10]
    if len(positive) != node_count - 1:
        return np.nan
    return float(2.0 * np.sum(1.0 / positive) / (node_count - 1))


def junction_region_count(binary_adjacency, edge_ids):
    junction_rows = np.where(edge_ids == 0)[0]
    if len(junction_rows) == 0:
        return 0
    junction_graph = binary_adjacency[junction_rows][:, junction_rows]
    count, _ = connected_components(junction_graph, directed=False)
    return int(count)


def build_physical_object_table(
    binary_adjacency,
    length_adjacency,
    scaled_xyz,
    edge_ids,
    social_ids,
    physical_edge_table,
    maximum_efficiency_nodes,
    maximum_resistance_nodes,
    random_seed,
):
    records = []
    for social_id in sorted(np.unique(social_ids)):
        global_rows = np.where(social_ids == social_id)[0]
        local_binary = binary_adjacency[global_rows][:, global_rows]
        local_length = length_adjacency[global_rows][:, global_rows]
        local_edge_ids = edge_ids[global_rows]
        local_degree = np.asarray(local_binary.sum(axis=1)).ravel()
        component_count, _ = connected_components(local_binary, directed=False)
        graph_edge_count = int(local_binary.nnz // 2)
        node_count = int(len(global_rows))
        cycle_rank = max(
            int(graph_edge_count - node_count + component_count),
            0,
        )
        edge_rows = physical_edge_table[
            physical_edge_table["social_instance_id"] == int(social_id)
        ]
        branch_lengths = edge_rows["arc_length"].to_numpy(dtype=float)
        total_length = float(np.sum(branch_lengths))
        records.append(
            {
                "social_instance_id": int(social_id),
                "physical_skeleton_point_count": node_count,
                "physical_graph_edge_count": graph_edge_count,
                "physical_component_count": int(component_count),
                "physical_edge_count": int(len(edge_rows)),
                "physical_junction_count": junction_region_count(
                    local_binary,
                    local_edge_ids,
                ),
                "physical_endpoint_count": int(np.sum(local_degree == 1)),
                "physical_cycle_rank": cycle_rank,
                "physical_total_skeleton_length": total_length,
                "physical_junction_density": safe_divide(
                    junction_region_count(local_binary, local_edge_ids),
                    total_length,
                ),
                "physical_endpoint_density": safe_divide(
                    np.sum(local_degree == 1),
                    total_length,
                ),
                "physical_cycle_rank_density": safe_divide(
                    cycle_rank,
                    total_length,
                ),
                "physical_branch_length_entropy": normalized_entropy(
                    branch_lengths
                ),
                "physical_route_efficiency": sampled_route_efficiency(
                    local_length,
                    scaled_xyz[global_rows],
                    maximum_nodes=maximum_efficiency_nodes,
                    random_seed=random_seed + int(social_id),
                ),
                "physical_algebraic_connectivity": (
                    normalized_algebraic_connectivity(local_binary)
                ),
                "physical_mean_effective_resistance": (
                    mean_effective_resistance(
                        local_length,
                        maximum_exact_nodes=maximum_resistance_nodes,
                    )
                ),
                "physical_mean_edge_length": (
                    float(np.mean(branch_lengths))
                    if len(branch_lengths) > 0
                    else np.nan
                ),
                "physical_edge_length_cv": coefficient_of_variation(
                    branch_lengths
                ),
                "physical_mean_tortuosity": (
                    float(np.mean(edge_rows["tortuosity"]))
                    if len(edge_rows) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def bounding_box_lower_bound(points_a, points_b):
    minimum_a = np.min(points_a, axis=0)
    maximum_a = np.max(points_a, axis=0)
    minimum_b = np.min(points_b, axis=0)
    maximum_b = np.max(points_b, axis=0)
    gap = np.maximum(
        np.maximum(minimum_a - maximum_b, minimum_b - maximum_a),
        0.0,
    )
    return float(np.linalg.norm(gap))


def closest_surface_pair(points_a, points_b, workers=-1):
    """返回两个点集的精确最近距离及最近点的局部row。"""
    if len(points_a) <= len(points_b):
        tree = cKDTree(points_b)
        distances, rows_b = tree.query(points_a, k=1, workers=workers)
        row_a = int(np.argmin(distances))
        row_b = int(rows_b[row_a])
        return float(distances[row_a]), row_a, row_b
    tree = cKDTree(points_a)
    distances, rows_a = tree.query(points_b, k=1, workers=workers)
    row_b = int(np.argmin(distances))
    row_a = int(rows_a[row_b])
    return float(distances[row_b]), row_a, row_b


def build_social_tables(
    pcd_social,
    coordinate_scale_xyz,
    social_distance_threshold,
    workers=-1,
):
    """以PCD表面最近距离<=threshold建立social network。"""
    pcd_social_ids = np.rint(pcd_social[:, -1]).astype(np.int64)
    if np.any(pcd_social_ids <= 0):
        raise ValueError(
            "The last PCD column must contain social_instance_id > 0."
        )
    scaled_xyz = pcd_social[:, :3] * coordinate_scale_xyz[None, :]
    unique_ids = sorted(np.unique(pcd_social_ids))
    point_rows_by_id = {
        int(social_id): np.where(pcd_social_ids == social_id)[0]
        for social_id in unique_ids
    }
    node_records = []
    for social_id in unique_ids:
        rows = point_rows_by_id[int(social_id)]
        points = scaled_xyz[rows]
        centroid = np.mean(points, axis=0)
        rgb = np.rint(np.median(pcd_social[rows, 3:6], axis=0)).astype(int)
        node_records.append(
            {
                "social_instance_id": int(social_id),
                "pcd_surface_point_count": int(len(rows)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "social_r": int(rgb[0]),
                "social_g": int(rgb[1]),
                "social_b": int(rgb[2]),
            }
        )

    edge_records = []
    for position_a, social_id_a in enumerate(unique_ids):
        rows_a = point_rows_by_id[int(social_id_a)]
        points_a = scaled_xyz[rows_a]
        for social_id_b in unique_ids[position_a + 1:]:
            rows_b = point_rows_by_id[int(social_id_b)]
            points_b = scaled_xyz[rows_b]
            if (
                bounding_box_lower_bound(points_a, points_b)
                > social_distance_threshold
            ):
                continue
            minimum_distance, local_a, local_b = closest_surface_pair(
                points_a,
                points_b,
                workers=workers,
            )
            if minimum_distance > social_distance_threshold:
                continue
            weight = float(
                np.exp(
                    -(minimum_distance / social_distance_threshold) ** 2
                )
            )
            point_a = points_a[local_a]
            point_b = points_b[local_b]
            edge_records.append(
                {
                    "source_social_id": int(social_id_a),
                    "target_social_id": int(social_id_b),
                    "minimum_surface_distance": minimum_distance,
                    "social_weight": weight,
                    "source_closest_x": float(point_a[0]),
                    "source_closest_y": float(point_a[1]),
                    "source_closest_z": float(point_a[2]),
                    "target_closest_x": float(point_b[0]),
                    "target_closest_y": float(point_b[1]),
                    "target_closest_z": float(point_b[2]),
                }
            )

    node_table = pd.DataFrame.from_records(node_records)
    edge_columns = [
        "source_social_id",
        "target_social_id",
        "minimum_surface_distance",
        "social_weight",
        "source_closest_x",
        "source_closest_y",
        "source_closest_z",
        "target_closest_x",
        "target_closest_y",
        "target_closest_z",
    ]
    edge_table = pd.DataFrame.from_records(edge_records, columns=edge_columns)
    return node_table, edge_table


def social_adjacency_from_tables(node_table, edge_table):
    social_ids = node_table["social_instance_id"].to_numpy(dtype=np.int64)
    position_by_id = {
        int(social_id): position
        for position, social_id in enumerate(social_ids)
    }
    node_count = len(social_ids)
    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    distance_cost = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(distance_cost, 0.0)
    for row in edge_table.itertuples(index=False):
        source = position_by_id[int(row.source_social_id)]
        target = position_by_id[int(row.target_social_id)]
        weight = float(row.social_weight)
        adjacency[source, target] = weight
        adjacency[target, source] = weight
        cost = 1.0 / max(weight, EPS)
        distance_cost[source, target] = cost
        distance_cost[target, source] = cost
    return social_ids, adjacency, distance_cost


def weighted_clustering_by_node(adjacency):
    node_count = adjacency.shape[0]
    clustering = np.zeros(node_count, dtype=np.float64)
    if node_count < 3 or np.max(adjacency) <= EPS:
        return clustering
    normalized = adjacency / np.max(adjacency)
    cube_root = np.cbrt(normalized)
    triangle_strength = np.diag(cube_root @ cube_root @ cube_root)
    degree = np.sum(adjacency > 0, axis=1)
    valid = degree >= 2
    clustering[valid] = (
        triangle_strength[valid]
        / (degree[valid] * (degree[valid] - 1))
    )
    return clustering


def weighted_betweenness(distance_cost):
    """Brandes weighted betweenness，适用于social小图。"""
    node_count = distance_cost.shape[0]
    betweenness = np.zeros(node_count, dtype=np.float64)
    neighbors = [
        np.where(np.isfinite(distance_cost[node]) & (distance_cost[node] > 0))[0]
        for node in range(node_count)
    ]

    for source in range(node_count):
        stack = []
        predecessors = [[] for _ in range(node_count)]
        sigma = np.zeros(node_count, dtype=np.float64)
        sigma[source] = 1.0
        distances = np.full(node_count, np.inf, dtype=np.float64)
        distances[source] = 0.0
        queue = [(0.0, source)]

        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance > distances[node] + 1e-12:
                continue
            stack.append(node)
            for neighbor in neighbors[node]:
                proposed = current_distance + distance_cost[node, neighbor]
                if proposed < distances[neighbor] - 1e-12:
                    distances[neighbor] = proposed
                    heapq.heappush(queue, (proposed, int(neighbor)))
                    sigma[neighbor] = sigma[node]
                    predecessors[neighbor] = [node]
                elif abs(proposed - distances[neighbor]) <= 1e-12:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)

        dependency = np.zeros(node_count, dtype=np.float64)
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if sigma[node] > EPS:
                    dependency[predecessor] += (
                        sigma[predecessor] / sigma[node]
                    ) * (1.0 + dependency[node])
            if node != source:
                betweenness[node] += dependency[node]

    betweenness *= 0.5
    if node_count > 2:
        betweenness /= ((node_count - 1) * (node_count - 2) / 2.0)
    return betweenness


def spectral_modularity(adjacency):
    """Newman leading-eigenvector递归划分得到weighted modularity。"""
    node_count = adjacency.shape[0]
    total_weight_twice = float(np.sum(adjacency))
    if node_count < 2 or total_weight_twice <= EPS:
        return np.nan, np.zeros(node_count, dtype=np.int64)
    strength = np.sum(adjacency, axis=1)
    modularity_matrix = (
        adjacency
        - np.outer(strength, strength) / total_weight_twice
    )
    communities = [np.arange(node_count, dtype=np.int64)]
    finished = []

    while communities:
        group = communities.pop()
        if len(group) < 2:
            finished.append(group)
            continue
        submatrix = modularity_matrix[np.ix_(group, group)].copy()
        submatrix -= np.diag(np.sum(submatrix, axis=1))
        eigenvalues, eigenvectors = np.linalg.eigh(submatrix)
        leading_position = int(np.argmax(eigenvalues))
        leading_value = float(eigenvalues[leading_position])
        if leading_value <= 1e-10:
            finished.append(group)
            continue
        signs = np.where(eigenvectors[:, leading_position] >= 0, 1.0, -1.0)
        if np.all(signs > 0) or np.all(signs < 0):
            finished.append(group)
            continue
        modularity_gain = float(
            signs @ submatrix @ signs / total_weight_twice
        )
        if modularity_gain <= 1e-10:
            finished.append(group)
            continue
        communities.append(group[signs > 0])
        communities.append(group[signs < 0])

    labels = np.zeros(node_count, dtype=np.int64)
    for community_id, group in enumerate(finished, start=1):
        labels[group] = community_id
    modularity = 0.0
    for group in finished:
        modularity += float(
            np.sum(modularity_matrix[np.ix_(group, group)])
            / total_weight_twice
        )
    return modularity, labels


def pagerank_from_adjacency(adjacency, damping=0.85, maximum_iterations=500):
    """不依赖NetworkX的weighted PageRank。"""
    adjacency = np.asarray(adjacency, dtype=np.float64)
    node_count = adjacency.shape[0]
    if node_count == 0:
        return np.empty(0, dtype=np.float64)
    strength = np.sum(adjacency, axis=1)
    transition = np.zeros_like(adjacency)
    positive = strength > EPS
    transition[positive] = adjacency[positive] / strength[positive, None]
    transition[~positive] = 1.0 / node_count
    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    teleportation = (1.0 - damping) / node_count
    for _ in range(maximum_iterations):
        updated = teleportation + damping * transition.T @ rank
        if np.sum(np.abs(updated - rank)) < 1e-12:
            rank = updated
            break
        rank = updated
    rank_sum = float(np.sum(rank))
    return rank / rank_sum if rank_sum > EPS else rank


def closeness_from_shortest_paths(shortest):
    """对非连通图使用Wasserman-Faust修正的closeness。"""
    node_count = shortest.shape[0]
    closeness = np.zeros(node_count, dtype=np.float64)
    if node_count <= 1:
        return closeness
    for node in range(node_count):
        reachable = np.isfinite(shortest[node]) & (shortest[node] > EPS)
        reachable_count = int(np.sum(reachable))
        if reachable_count == 0:
            continue
        distance_sum = float(np.sum(shortest[node, reachable]))
        closeness[node] = (
            reachable_count / distance_sum
        ) * (reachable_count / (node_count - 1))
    return closeness


def eigenvector_centrality_by_component(adjacency, component_labels):
    """分别在每个connected component内部计算eigenvector centrality。"""
    node_count = adjacency.shape[0]
    centrality = np.zeros(node_count, dtype=np.float64)
    for component_id in np.unique(component_labels):
        rows = np.where(component_labels == component_id)[0]
        if len(rows) == 1:
            centrality[rows[0]] = 1.0
            continue
        subgraph = adjacency[np.ix_(rows, rows)]
        eigenvalues, eigenvectors = np.linalg.eigh(subgraph)
        vector = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > EPS:
            vector = vector / vector_norm
        centrality[rows] = vector
    return centrality


def von_neumann_entropy(adjacency):
    """以combinatorial graph Laplacian构造density matrix计算归一化VN entropy。"""
    adjacency = np.asarray(adjacency, dtype=np.float64)
    node_count = adjacency.shape[0]
    if node_count <= 1 or np.sum(adjacency) <= EPS:
        return np.nan
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency
    trace = float(np.trace(laplacian))
    if trace <= EPS:
        return np.nan
    probabilities = np.linalg.eigvalsh(laplacian / trace)
    probabilities = probabilities[probabilities > 1e-12]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / np.log(node_count))


def local_efficiency_by_node(binary_adjacency):
    """每个node邻居诱导子图的global efficiency。"""
    binary_adjacency = np.asarray(binary_adjacency, dtype=np.float64)
    node_count = binary_adjacency.shape[0]
    values = np.zeros(node_count, dtype=np.float64)
    for node in range(node_count):
        neighbors = np.where(binary_adjacency[node] > 0)[0]
        neighbor_count = len(neighbors)
        if neighbor_count < 2:
            values[node] = 0.0
            continue
        subgraph = binary_adjacency[np.ix_(neighbors, neighbors)]
        shortest = dijkstra(csr_matrix(subgraph), directed=False)
        off_diagonal = ~np.eye(neighbor_count, dtype=bool)
        valid = off_diagonal & np.isfinite(shortest) & (shortest > EPS)
        inverse = np.zeros_like(shortest)
        inverse[valid] = 1.0 / shortest[valid]
        values[node] = float(
            np.sum(inverse) / (neighbor_count * (neighbor_count - 1))
        )
    return values


def generic_graph_metrics_and_centralities(
    node_ids,
    weighted_adjacency,
    prefix,
):
    """计算physical/social通用图论指标及node-level centralities。"""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    weighted_adjacency = np.asarray(weighted_adjacency, dtype=np.float64)
    node_count = len(node_ids)
    binary_adjacency = (weighted_adjacency > 0).astype(np.float64)
    edge_count = int(np.sum(binary_adjacency) // 2)
    degree = np.sum(binary_adjacency, axis=1)
    strength = np.sum(weighted_adjacency, axis=1)
    clustering = weighted_clustering_by_node(weighted_adjacency)
    local_efficiency = local_efficiency_by_node(binary_adjacency)

    distance_cost = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(distance_cost, 0.0)
    distance_cost[binary_adjacency > 0] = 1.0
    shortest = dijkstra(csr_matrix(distance_cost), directed=False)
    closeness = closeness_from_shortest_paths(shortest)
    betweenness = weighted_betweenness(distance_cost)
    binary_sparse = csr_matrix(binary_adjacency)
    component_count, component_labels = connected_components(
        binary_sparse,
        directed=False,
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)
    eigenvector = eigenvector_centrality_by_component(
        weighted_adjacency,
        component_labels,
    )
    pagerank = pagerank_from_adjacency(weighted_adjacency)

    if node_count >= 2:
        off_diagonal = ~np.eye(node_count, dtype=bool)
        valid = off_diagonal & np.isfinite(shortest) & (shortest > EPS)
        inverse = np.zeros_like(shortest)
        inverse[valid] = 1.0 / shortest[valid]
        global_efficiency = float(
            np.sum(inverse) / (node_count * (node_count - 1))
        )
    else:
        global_efficiency = np.nan

    largest_component_id = int(np.argmax(component_sizes))
    largest_rows = np.where(component_labels == largest_component_id)[0]
    if len(largest_rows) >= 2:
        largest_shortest = shortest[np.ix_(largest_rows, largest_rows)]
        upper = np.triu_indices(len(largest_rows), k=1)
        largest_distances = largest_shortest[upper]
        average_path_length_lcc = float(np.mean(largest_distances))
        diameter_lcc = float(np.max(largest_distances))
        inverse_lcc = 1.0 / largest_distances
        global_efficiency_lcc = float(np.mean(inverse_lcc))
    else:
        average_path_length_lcc = np.nan
        diameter_lcc = np.nan
        global_efficiency_lcc = np.nan

    transitivity_denominator = float(np.sum(degree * (degree - 1)))
    triangle_six = float(
        np.trace(binary_adjacency @ binary_adjacency @ binary_adjacency)
    )
    transitivity = (
        triangle_six / transitivity_denominator
        if transitivity_denominator > EPS
        else np.nan
    )
    modularity, community_labels = spectral_modularity(weighted_adjacency)

    if edge_count >= 2:
        sources, targets = np.where(np.triu(binary_adjacency, k=1) > 0)
        source_values = np.concatenate([degree[sources], degree[targets]])
        target_values = np.concatenate([degree[targets], degree[sources]])
        degree_assortativity = (
            float(np.corrcoef(source_values, target_values)[0, 1])
            if np.std(source_values) > EPS and np.std(target_values) > EPS
            else np.nan
        )
    else:
        degree_assortativity = np.nan

    node_centralities = pd.DataFrame(
        {
            "node_id": node_ids,
            f"{prefix}_degree": degree,
            f"{prefix}_strength": strength,
            f"{prefix}_clustering": clustering,
            f"{prefix}_local_efficiency": local_efficiency,
            f"{prefix}_closeness": closeness,
            f"{prefix}_betweenness": betweenness,
            f"{prefix}_eigenvector": eigenvector,
            f"{prefix}_pagerank": pagerank,
            f"{prefix}_component_id": component_labels + 1,
            f"{prefix}_community_id": community_labels,
        }
    )
    metrics = {
        f"{prefix}_instance_count": int(node_count),
        f"{prefix}_network_edge_count": int(edge_count),
        f"{prefix}_density": (
            float(2 * edge_count / (node_count * (node_count - 1)))
            if node_count >= 2
            else np.nan
        ),
        f"{prefix}_mean_degree": float(np.mean(degree)),
        f"{prefix}_mean_strength": float(np.mean(strength)),
        f"{prefix}_degree_gini": gini_coefficient(degree),
        f"{prefix}_clustering": (
            float(np.mean(clustering)) if node_count >= 2 else np.nan
        ),
        f"{prefix}_transitivity": transitivity,
        f"{prefix}_global_efficiency": global_efficiency,
        f"{prefix}_local_efficiency": (
            float(np.mean(local_efficiency)) if node_count >= 2 else np.nan
        ),
        f"{prefix}_modularity": modularity,
        f"{prefix}_lcc_fraction": (
            float(np.max(component_sizes) / node_count)
            if node_count >= 2
            else np.nan
        ),
        f"{prefix}_average_shortest_path_lcc": average_path_length_lcc,
        f"{prefix}_diameter_lcc": diameter_lcc,
        f"{prefix}_global_efficiency_lcc": global_efficiency_lcc,
        f"{prefix}_degree_assortativity": degree_assortativity,
        f"{prefix}_degree_entropy": normalized_entropy(degree),
        f"{prefix}_closeness_entropy": normalized_entropy(closeness),
        f"{prefix}_betweenness_entropy": normalized_entropy(betweenness),
        f"{prefix}_eigenvector_entropy": normalized_entropy(eigenvector),
        f"{prefix}_pagerank_entropy": normalized_entropy(pagerank),
        f"{prefix}_von_neumann_entropy": von_neumann_entropy(
            weighted_adjacency
        ),
    }
    return node_centralities, metrics


def build_physical_instance_network(
    physical_node_table,
    skeleton_binary_adjacency,
    skeleton_edge_ids,
    skeleton_social_ids,
):
    """
    Physical node = edge_id>0的physical instance。
    Physical network edge = 两个physical instances连接到同一junction region。
    """
    physical_ids = physical_node_table["physical_edge_id"].to_numpy(dtype=np.int64)
    position_by_id = {
        int(edge_id): position
        for position, edge_id in enumerate(physical_ids)
    }
    junction_rows = np.where(skeleton_edge_ids == 0)[0]
    edge_to_junctions = {}
    junctions_by_edge = {
        int(edge_id): set() for edge_id in physical_ids
    }

    if len(junction_rows) > 0:
        junction_graph = skeleton_binary_adjacency[junction_rows][:, junction_rows]
        junction_count, junction_labels = connected_components(
            junction_graph,
            directed=False,
        )
        for junction_position in range(junction_count):
            region_rows = junction_rows[junction_labels == junction_position]
            neighbor_rows = set()
            for skeleton_row in region_rows:
                start = skeleton_binary_adjacency.indptr[skeleton_row]
                end = skeleton_binary_adjacency.indptr[skeleton_row + 1]
                neighbor_rows.update(
                    skeleton_binary_adjacency.indices[start:end].tolist()
                )
            incident_edge_ids = sorted(
                {
                    int(skeleton_edge_ids[row])
                    for row in neighbor_rows
                    if skeleton_edge_ids[row] > 0
                }
            )
            if len(incident_edge_ids) < 2:
                continue
            incident_social_ids = {
                int(
                    physical_node_table.loc[
                        physical_node_table["physical_edge_id"] == edge_id,
                        "social_instance_id",
                    ].iloc[0]
                )
                for edge_id in incident_edge_ids
            }
            if len(incident_social_ids) != 1:
                raise ValueError(
                    f"junction {junction_position + 1} connects physical "
                    f"instances from different social IDs: {incident_social_ids}"
                )
            junction_id = int(junction_position + 1)
            for edge_id in incident_edge_ids:
                junctions_by_edge[int(edge_id)].add(junction_id)
            for source_edge_id, target_edge_id in combinations(
                incident_edge_ids,
                2,
            ):
                pair = (int(source_edge_id), int(target_edge_id))
                edge_to_junctions.setdefault(pair, []).append(
                    junction_id
                )

    weighted_adjacency = np.zeros(
        (len(physical_ids), len(physical_ids)),
        dtype=np.float64,
    )
    network_edge_records = []
    social_by_physical = dict(
        zip(
            physical_node_table["physical_edge_id"].astype(int),
            physical_node_table["social_instance_id"].astype(int),
        )
    )
    for (source_id, target_id), junction_ids in sorted(edge_to_junctions.items()):
        source_position = position_by_id[source_id]
        target_position = position_by_id[target_id]
        shared_count = len(junction_ids)
        weighted_adjacency[source_position, target_position] = shared_count
        weighted_adjacency[target_position, source_position] = shared_count
        network_edge_records.append(
            {
                "source_physical_edge_id": source_id,
                "target_physical_edge_id": target_id,
                "social_instance_id": social_by_physical[source_id],
                "shared_junction_count": int(shared_count),
                "shared_junction_ids": ";".join(map(str, junction_ids)),
            }
        )

    network_edge_table = pd.DataFrame.from_records(
        network_edge_records,
        columns=[
            "source_physical_edge_id",
            "target_physical_edge_id",
            "social_instance_id",
            "shared_junction_count",
            "shared_junction_ids",
        ],
    )
    centralities, metrics = generic_graph_metrics_and_centralities(
        node_ids=physical_ids,
        weighted_adjacency=weighted_adjacency,
        prefix="physical",
    )
    centralities = centralities.rename(columns={"node_id": "physical_edge_id"})
    physical_node_table = physical_node_table.merge(
        centralities,
        on="physical_edge_id",
        how="left",
        validate="one_to_one",
    )
    physical_node_table["physical_junction_incidence_count"] = (
        physical_node_table["physical_edge_id"]
        .map(lambda edge_id: len(junctions_by_edge[int(edge_id)]))
        .astype(int)
    )
    # 恰好只接一个junction的edge才是terminal branch；孤立edge不强行归类。
    physical_node_table["physical_is_terminal_branch"] = (
        physical_node_table["physical_junction_incidence_count"] == 1
    )
    return physical_node_table, network_edge_table, weighted_adjacency, metrics


def aggregate_physical_nodes_by_social(physical_node_table):
    """把physical-instance centralities汇总到其所属social node。"""
    records = []
    for social_id, group in physical_node_table.groupby("social_instance_id"):
        records.append(
            {
                "social_instance_id": int(social_id),
                "physical_instance_count": int(len(group)),
                "physical_internal_network_edge_count": float(
                    np.sum(group["physical_degree"]) / 2.0
                ),
                "physical_mean_degree": float(np.mean(group["physical_degree"])),
                "physical_mean_strength": float(np.mean(group["physical_strength"])),
                "physical_mean_clustering": float(np.mean(group["physical_clustering"])),
                "physical_mean_local_efficiency": float(
                    np.mean(group["physical_local_efficiency"])
                ),
                "physical_mean_closeness": float(np.mean(group["physical_closeness"])),
                "physical_mean_betweenness": float(
                    np.mean(group["physical_betweenness"])
                ),
                "physical_max_betweenness": float(
                    np.max(group["physical_betweenness"])
                ),
                "physical_mean_pagerank": float(np.mean(group["physical_pagerank"])),
            }
        )
    return pd.DataFrame.from_records(records)


def social_metrics_and_node_centralities(node_table, edge_table):
    social_ids, adjacency, distance_cost = social_adjacency_from_tables(
        node_table,
        edge_table,
    )
    node_count = len(social_ids)
    degree = np.sum(adjacency > 0, axis=1).astype(np.float64)
    strength = np.sum(adjacency, axis=1)
    clustering = weighted_clustering_by_node(adjacency)
    betweenness = weighted_betweenness(distance_cost)
    binary_sparse = csr_matrix((adjacency > 0).astype(np.float64))
    component_count, component_labels = connected_components(
        binary_sparse,
        directed=False,
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)
    shortest = dijkstra(csr_matrix(distance_cost), directed=False)
    off_diagonal = ~np.eye(node_count, dtype=bool)
    inverse_distance = np.zeros_like(shortest)
    valid = off_diagonal & np.isfinite(shortest) & (shortest > EPS)
    inverse_distance[valid] = 1.0 / shortest[valid]
    global_efficiency = (
        float(np.sum(inverse_distance) / (node_count * (node_count - 1)))
        if node_count >= 2
        else np.nan
    )
    topological_shortest = dijkstra(binary_sparse, directed=False)
    topological_inverse = np.zeros_like(topological_shortest)
    topological_valid = (
        off_diagonal
        & np.isfinite(topological_shortest)
        & (topological_shortest > EPS)
    )
    topological_inverse[topological_valid] = (
        1.0 / topological_shortest[topological_valid]
    )
    topological_global_efficiency = (
        float(
            np.sum(topological_inverse)
            / (node_count * (node_count - 1))
        )
        if node_count >= 2
        else np.nan
    )
    modularity, community_labels = spectral_modularity(adjacency)

    binary_degree = degree
    edge_count = int(len(edge_table))
    transitivity_denominator = float(np.sum(binary_degree * (binary_degree - 1)))
    binary_dense = (adjacency > 0).astype(np.float64)
    local_efficiency = local_efficiency_by_node(binary_dense)
    triangle_six = float(np.trace(binary_dense @ binary_dense @ binary_dense))
    transitivity = (
        triangle_six / transitivity_denominator
        if transitivity_denominator > EPS
        else np.nan
    )

    if edge_count >= 2:
        source_values = []
        target_values = []
        position_by_id = {
            int(social_id): position
            for position, social_id in enumerate(social_ids)
        }
        for row in edge_table.itertuples(index=False):
            a = position_by_id[int(row.source_social_id)]
            b = position_by_id[int(row.target_social_id)]
            source_values.extend([degree[a], degree[b]])
            target_values.extend([degree[b], degree[a]])
        if np.std(source_values) > EPS and np.std(target_values) > EPS:
            assortativity = float(
                np.corrcoef(source_values, target_values)[0, 1]
            )
        else:
            assortativity = np.nan
    else:
        assortativity = np.nan

    output_nodes = node_table.copy()
    output_nodes["social_degree"] = degree
    output_nodes["social_strength"] = strength
    output_nodes["social_clustering"] = clustering
    output_nodes["social_local_efficiency"] = local_efficiency
    output_nodes["social_betweenness"] = betweenness
    output_nodes["social_component_id"] = component_labels + 1
    output_nodes["social_community_id"] = community_labels

    metrics = {
        "social_mean_degree": float(np.mean(degree)),
        "social_mean_strength": float(np.mean(strength)),
        "social_degree_gini": gini_coefficient(degree),
        "social_density": (
            float(2 * edge_count / (node_count * (node_count - 1)))
            if node_count >= 2
            else np.nan
        ),
        "social_clustering": (
            float(np.mean(clustering)) if node_count >= 2 else np.nan
        ),
        "social_transitivity": transitivity,
        "social_lcc_fraction": (
            float(np.max(component_sizes) / node_count)
            if node_count >= 2
            else np.nan
        ),
        "social_global_efficiency": global_efficiency,
        "social_topological_global_efficiency": (
            topological_global_efficiency
        ),
        "social_local_efficiency": (
            float(np.mean(local_efficiency))
            if node_count >= 2
            else np.nan
        ),
        "social_modularity": modularity,
        "social_degree_entropy": normalized_entropy(degree),
        "social_strength_entropy": normalized_entropy(strength),
        "social_degree_assortativity": assortativity,
        "social_betweenness_max": (
            float(np.max(betweenness)) if node_count >= 2 else np.nan
        ),
        "social_betweenness_entropy": normalized_entropy(betweenness),
    }
    return output_nodes, metrics, adjacency



def annotate_cross_layer_contact_targets(
    physical_node_table,
    social_edge_table,
    scaled_skeleton_xyz,
    skeleton_social_ids,
    skeleton_edge_ids,
    skeleton_binary_adjacency,
    contact_region_radius,
    hub_fraction=0.10,
):
    """
    将每条 social edge 的两个表面接触端映射到最近 physical edge instance，
    同时判断接触发生在该 social object 内部的哪个 physical topology 区域。

    branch-level target:
      - length / local efficiency / betweenness rank
      - terminal branch / hub targeting

    topology-location target:
      - junction zone
      - terminal zone
      - shaft zone

    topology zone 的定义使用 skeleton 上的真实空间邻域：
      1) 距离 edge_id==0 的 junction-region skeleton <= contact_region_radius
         -> junction
      2) 否则距离 skeleton degree==1 endpoint <= contact_region_radius
         -> terminal
      3) 其余 -> shaft

    enrichment 的 null baseline 不是固定 1/3，而是对每个被接触 social object，
    用其自身 skeleton points 落在 junction/terminal/shaft zone 的比例作为期望值。
    因此该指标回答“social contact 是否 preferentially 落在某类 physical 部位”。
    """
    physical = physical_node_table.copy()

    rank_specs = {
        "arc_length": "physical_within_social_length_rank",
        "physical_local_efficiency": (
            "physical_within_social_local_efficiency_rank"
        ),
        "physical_betweenness": (
            "physical_within_social_betweenness_rank"
        ),
    }
    for output_column in rank_specs.values():
        physical[output_column] = np.nan

    physical["physical_contact_betweenness_hub"] = False
    for _, group in physical.groupby("social_instance_id", sort=False):
        rows = group.index.to_numpy()
        for value_column, output_column in rank_specs.items():
            values = group[value_column].to_numpy(dtype=float)
            valid = np.isfinite(values)
            ranks = np.full(len(group), np.nan, dtype=float)
            valid_count = int(np.sum(valid))
            if valid_count == 1:
                ranks[valid] = 0.5
            elif valid_count >= 2:
                raw_ranks = rankdata(values[valid], method="average")
                ranks[valid] = (raw_ranks - 1.0) / (valid_count - 1.0)
            physical.loc[rows, output_column] = ranks

        hub_mask, _ = ranked_tail_masks(
            group["physical_betweenness"].to_numpy(dtype=float),
            fraction=hub_fraction,
            descending=True,
        )
        physical.loc[
            rows[hub_mask],
            "physical_contact_betweenness_hub",
        ] = True

    physical_by_id = physical.set_index("physical_edge_id", drop=False)

    skeleton_degree = np.asarray(
        skeleton_binary_adjacency.sum(axis=1)
    ).ravel()

    social_ids_unique = np.unique(skeleton_social_ids)

    eligible_rows_by_social = {
        int(social_id): np.where(
            (skeleton_social_ids == int(social_id))
            & (skeleton_edge_ids > 0)
        )[0]
        for social_id in social_ids_unique
    }

    all_rows_by_social = {
        int(social_id): np.where(
            skeleton_social_ids == int(social_id)
        )[0]
        for social_id in social_ids_unique
    }

    endpoint_rows_by_social = {
        int(social_id): np.where(
            (skeleton_social_ids == int(social_id))
            & (skeleton_degree == 1)
            & (skeleton_edge_ids > 0)
        )[0]
        for social_id in social_ids_unique
    }

    junction_rows_by_social = {
        int(social_id): np.where(
            (skeleton_social_ids == int(social_id))
            & (skeleton_edge_ids == 0)
        )[0]
        for social_id in social_ids_unique
    }

    # ------------------------------------------------------------------
    # 对每个 social object 建立 topology-location baseline。
    # baseline 是“随机 skeleton location”落在各类 zone 的比例。
    # ------------------------------------------------------------------
    topology_cache = {}

    for social_id in social_ids_unique:
        social_id = int(social_id)
        all_rows = all_rows_by_social[social_id]
        endpoint_rows = endpoint_rows_by_social[social_id]
        junction_rows = junction_rows_by_social[social_id]

        all_points = scaled_skeleton_xyz[all_rows]

        endpoint_tree = (
            cKDTree(scaled_skeleton_xyz[endpoint_rows])
            if len(endpoint_rows) > 0
            else None
        )
        junction_tree = (
            cKDTree(scaled_skeleton_xyz[junction_rows])
            if len(junction_rows) > 0
            else None
        )

        if endpoint_tree is None:
            endpoint_distances = np.full(
                len(all_rows),
                np.inf,
                dtype=np.float64,
            )
        else:
            endpoint_distances = endpoint_tree.query(
                all_points,
                k=1,
            )[0]

        if junction_tree is None:
            junction_distances = np.full(
                len(all_rows),
                np.inf,
                dtype=np.float64,
            )
        else:
            junction_distances = junction_tree.query(
                all_points,
                k=1,
            )[0]

        # exclusive classification; junction gets priority when zones overlap.
        is_junction = junction_distances <= contact_region_radius
        is_terminal = (
            (~is_junction)
            & (endpoint_distances <= contact_region_radius)
        )
        is_shaft = ~(is_junction | is_terminal)

        topology_cache[social_id] = {
            "endpoint_tree": endpoint_tree,
            "junction_tree": junction_tree,
            "junction_fraction": float(np.mean(is_junction)),
            "terminal_fraction": float(np.mean(is_terminal)),
            "shaft_fraction": float(np.mean(is_shaft)),
        }

    event_records = []
    annotated_edges = social_edge_table.copy()

    endpoint_specs = [
        (
            "source",
            "source_social_id",
            ["source_closest_x", "source_closest_y", "source_closest_z"],
        ),
        (
            "target",
            "target_social_id",
            ["target_closest_x", "target_closest_y", "target_closest_z"],
        ),
    ]

    for edge_index, edge_row in annotated_edges.iterrows():
        for endpoint_role, social_column, xyz_columns in endpoint_specs:
            social_id = int(edge_row[social_column])

            candidate_rows = eligible_rows_by_social.get(
                social_id,
                np.empty(0, dtype=np.int64),
            )
            if len(candidate_rows) == 0:
                continue

            anchor_xyz = edge_row[xyz_columns].to_numpy(dtype=float)

            # target physical branch
            local_distances = np.linalg.norm(
                scaled_skeleton_xyz[candidate_rows] - anchor_xyz,
                axis=1,
            )
            nearest_global_row = int(
                candidate_rows[int(np.argmin(local_distances))]
            )
            physical_edge_id = int(skeleton_edge_ids[nearest_global_row])
            physical_row = physical_by_id.loc[physical_edge_id]
            nearest_skeleton_xyz = scaled_skeleton_xyz[nearest_global_row]

            endpoint_distance_on_edge = min(
                float(
                    np.linalg.norm(
                        nearest_skeleton_xyz
                        - physical_row[
                            ["start_x", "start_y", "start_z"]
                        ].to_numpy(dtype=float)
                    )
                ),
                float(
                    np.linalg.norm(
                        nearest_skeleton_xyz
                        - physical_row[
                            ["end_x", "end_y", "end_z"]
                        ].to_numpy(dtype=float)
                    )
                ),
            )
            half_arc_length = max(
                float(physical_row["arc_length"]) / 2.0,
                EPS,
            )
            terminal_score = 1.0 - np.clip(
                endpoint_distance_on_edge / half_arc_length,
                0.0,
                1.0,
            )

            # topology location of this contact
            cache = topology_cache[social_id]
            if cache["junction_tree"] is None:
                nearest_junction_distance = np.inf
            else:
                nearest_junction_distance = float(
                    cache["junction_tree"].query(nearest_skeleton_xyz, k=1)[0]
                )

            if cache["endpoint_tree"] is None:
                nearest_terminal_distance = np.inf
            else:
                nearest_terminal_distance = float(
                    cache["endpoint_tree"].query(nearest_skeleton_xyz, k=1)[0]
                )

            if nearest_junction_distance <= contact_region_radius:
                topology_region = "junction"
            elif nearest_terminal_distance <= contact_region_radius:
                topology_region = "terminal"
            else:
                topology_region = "shaft"

            event_records.append(
                {
                    "social_edge_index": int(edge_index),
                    "endpoint_role": endpoint_role,
                    "social_instance_id": social_id,
                    "physical_edge_id": physical_edge_id,
                    "anchor_to_skeleton_distance": float(
                        np.min(local_distances)
                    ),
                    "contact_terminal_score": float(terminal_score),
                    "contact_length_rank": float(
                        physical_row[
                            "physical_within_social_length_rank"
                        ]
                    ),
                    "contact_local_efficiency_rank": float(
                        physical_row[
                            "physical_within_social_local_efficiency_rank"
                        ]
                    ),
                    "contact_betweenness_rank": float(
                        physical_row[
                            "physical_within_social_betweenness_rank"
                        ]
                    ),
                    "contact_is_betweenness_hub": bool(
                        physical_row[
                            "physical_contact_betweenness_hub"
                        ]
                    ),
                    "contact_is_terminal_branch": bool(
                        physical_row["physical_is_terminal_branch"]
                    ),
                    "contact_nearest_junction_distance": (
                        nearest_junction_distance
                    ),
                    "contact_nearest_terminal_distance": (
                        nearest_terminal_distance
                    ),
                    "contact_topology_region": topology_region,
                    "contact_is_junction_zone": (
                        topology_region == "junction"
                    ),
                    "contact_is_terminal_zone": (
                        topology_region == "terminal"
                    ),
                    "contact_is_shaft_zone": (
                        topology_region == "shaft"
                    ),
                }
            )

            annotated_edges.loc[
                edge_index,
                f"{endpoint_role}_physical_edge_id",
            ] = physical_edge_id

            annotated_edges.loc[
                edge_index,
                f"{endpoint_role}_topology_region",
            ] = topology_region

    event_columns = [
        "social_edge_index",
        "endpoint_role",
        "social_instance_id",
        "physical_edge_id",
        "anchor_to_skeleton_distance",
        "contact_terminal_score",
        "contact_length_rank",
        "contact_local_efficiency_rank",
        "contact_betweenness_rank",
        "contact_is_betweenness_hub",
        "contact_is_terminal_branch",
        "contact_nearest_junction_distance",
        "contact_nearest_terminal_distance",
        "contact_topology_region",
        "contact_is_junction_zone",
        "contact_is_terminal_zone",
        "contact_is_shaft_zone",
    ]

    events = pd.DataFrame.from_records(
        event_records,
        columns=event_columns,
    )

    physical["physical_social_contact_count"] = 0
    physical["physical_is_social_contacted"] = False
    physical["physical_contact_terminal_score_mean"] = np.nan

    empty_metrics = {
        "cross_contact_target_length_rank_mean": np.nan,
        "cross_contact_target_local_efficiency_rank_mean": np.nan,
        "cross_contact_target_betweenness_rank_mean": np.nan,
        "cross_contact_target_terminal_score_mean": np.nan,
        "cross_contact_target_hub_enrichment": np.nan,
        "cross_contact_target_terminal_branch_enrichment": np.nan,
        "cross_contact_physical_coverage_fraction_mean": np.nan,
        "cross_contact_load_gini": np.nan,
        "cross_contact_pair_local_efficiency_rank_assortativity": np.nan,
        "cross_contact_topology_junction_fraction": np.nan,
        "cross_contact_topology_terminal_fraction": np.nan,
        "cross_contact_topology_shaft_fraction": np.nan,
        "cross_contact_topology_junction_enrichment": np.nan,
        "cross_contact_topology_terminal_enrichment": np.nan,
        "cross_contact_topology_shaft_enrichment": np.nan,
    }

    if len(events) == 0:
        return (
            physical,
            annotated_edges,
            events,
            empty_metrics,
        )

    contact_counts = events["physical_edge_id"].value_counts()
    terminal_means = events.groupby("physical_edge_id")[
        "contact_terminal_score"
    ].mean()

    physical["physical_social_contact_count"] = (
        physical["physical_edge_id"]
        .map(contact_counts)
        .fillna(0)
        .astype(int)
    )
    physical["physical_is_social_contacted"] = (
        physical["physical_social_contact_count"] > 0
    )
    physical["physical_contact_terminal_score_mean"] = (
        physical["physical_edge_id"].map(terminal_means)
    )

    # branch-level hub enrichment
    hub_baseline_by_social = physical.groupby("social_instance_id")[
        "physical_contact_betweenness_hub"
    ].mean()
    event_hub_baseline = events["social_instance_id"].map(
        hub_baseline_by_social
    ).to_numpy(dtype=float)

    observed_hub_fraction = float(
        np.mean(
            events[
                "contact_is_betweenness_hub"
            ].to_numpy(dtype=float)
        )
    )
    expected_hub_fraction = float(np.mean(event_hub_baseline))

    # old branch-level terminal enrichment retained for backward comparability
    terminal_baseline_by_social = physical.groupby("social_instance_id")[
        "physical_is_terminal_branch"
    ].mean()
    event_terminal_baseline = events["social_instance_id"].map(
        terminal_baseline_by_social
    ).to_numpy(dtype=float)

    observed_terminal_branch_fraction = float(
        np.mean(
            events[
                "contact_is_terminal_branch"
            ].to_numpy(dtype=float)
        )
    )
    expected_terminal_branch_fraction = float(
        np.mean(event_terminal_baseline)
    )

    # physical branch coverage by social contacts
    physical_count_by_social = physical.groupby(
        "social_instance_id"
    ).size()
    unique_contacted_by_social = events.groupby(
        "social_instance_id"
    )["physical_edge_id"].nunique()
    contact_coverage = (
        unique_contacted_by_social
        / physical_count_by_social.loc[
            unique_contacted_by_social.index
        ]
    )

    paired = events.pivot(
        index="social_edge_index",
        columns="endpoint_role",
        values="contact_local_efficiency_rank",
    )
    if "source" in paired.columns and "target" in paired.columns:
        pair_assortativity = safe_correlation(
            paired["source"],
            paired["target"],
            minimum_n=4,
        )
    else:
        pair_assortativity = np.nan

    # --------------------------------------------------------------
    # topology-location observed fractions + event-weighted baselines
    # --------------------------------------------------------------
    observed_junction_fraction = float(
        np.mean(events["contact_is_junction_zone"].astype(float))
    )
    observed_terminal_fraction = float(
        np.mean(events["contact_is_terminal_zone"].astype(float))
    )
    observed_shaft_fraction = float(
        np.mean(events["contact_is_shaft_zone"].astype(float))
    )

    expected_junction_fraction = float(
        np.mean(
            [
                topology_cache[int(sid)]["junction_fraction"]
                for sid in events["social_instance_id"]
            ]
        )
    )
    expected_terminal_fraction = float(
        np.mean(
            [
                topology_cache[int(sid)]["terminal_fraction"]
                for sid in events["social_instance_id"]
            ]
        )
    )
    expected_shaft_fraction = float(
        np.mean(
            [
                topology_cache[int(sid)]["shaft_fraction"]
                for sid in events["social_instance_id"]
            ]
        )
    )

    metrics = {
        "cross_contact_target_length_rank_mean": float(
            events["contact_length_rank"].mean()
        ),
        "cross_contact_target_local_efficiency_rank_mean": float(
            events["contact_local_efficiency_rank"].mean()
        ),
        "cross_contact_target_betweenness_rank_mean": float(
            events["contact_betweenness_rank"].mean()
        ),
        "cross_contact_target_terminal_score_mean": float(
            events["contact_terminal_score"].mean()
        ),
        "cross_contact_target_hub_enrichment": safe_divide(
            observed_hub_fraction,
            expected_hub_fraction,
        ),
        "cross_contact_target_terminal_branch_enrichment": safe_divide(
            observed_terminal_branch_fraction,
            expected_terminal_branch_fraction,
        ),
        "cross_contact_physical_coverage_fraction_mean": float(
            np.mean(contact_coverage)
        ),
        "cross_contact_load_gini": gini_coefficient(
            physical["physical_social_contact_count"]
        ),
        "cross_contact_pair_local_efficiency_rank_assortativity": (
            pair_assortativity
        ),

        # location-specific cross-scale topology
        "cross_contact_topology_junction_fraction": (
            observed_junction_fraction
        ),
        "cross_contact_topology_terminal_fraction": (
            observed_terminal_fraction
        ),
        "cross_contact_topology_shaft_fraction": (
            observed_shaft_fraction
        ),
        "cross_contact_topology_junction_enrichment": safe_divide(
            observed_junction_fraction,
            expected_junction_fraction,
        ),
        "cross_contact_topology_terminal_enrichment": safe_divide(
            observed_terminal_fraction,
            expected_terminal_fraction,
        ),
        "cross_contact_topology_shaft_enrichment": safe_divide(
            observed_shaft_fraction,
            expected_shaft_fraction,
        ),
    }

    return physical, annotated_edges, events, metrics


def partial_spearman(x, y, covariates, minimum_n=6):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    valid = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    if int(np.sum(valid)) < minimum_n:
        return np.nan, int(np.sum(valid))
    ranked_x = rankdata(x[valid])
    ranked_y = rankdata(y[valid])
    ranked_covariates = np.column_stack(
        [rankdata(covariates[valid, column]) for column in range(covariates.shape[1])]
    )
    design = np.column_stack(
        [np.ones(len(ranked_x), dtype=np.float64), ranked_covariates]
    )
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.std(residual_x) <= EPS or np.std(residual_y) <= EPS:
        return np.nan, int(np.sum(valid))
    return float(np.corrcoef(residual_x, residual_y)[0, 1]), int(np.sum(valid))


def build_physical_complexity(node_table):
    feature_columns = [
        "physical_mean_degree",
        "physical_mean_clustering",
        "physical_mean_closeness",
        "physical_max_betweenness",
    ]
    standardized = []
    for column in feature_columns:
        standardized.append(zscore_finite(node_table[column].to_numpy(dtype=float)))
    matrix = np.column_stack(standardized)
    complexity = np.full(len(node_table), np.nan, dtype=np.float64)
    for row in range(len(node_table)):
        values = matrix[row]
        values = values[np.isfinite(values)]
        if len(values) > 0:
            complexity[row] = float(np.mean(values))
    return complexity


def edge_attribute_assortativity(node_table, edge_table, attribute):
    if len(edge_table) < 2:
        return np.nan
    values_by_id = dict(
        zip(
            node_table["social_instance_id"].astype(int),
            node_table[attribute].astype(float),
        )
    )
    source_values = []
    target_values = []
    for row in edge_table.itertuples(index=False):
        value_a = values_by_id[int(row.source_social_id)]
        value_b = values_by_id[int(row.target_social_id)]
        if np.isfinite(value_a) and np.isfinite(value_b):
            source_values.extend([value_a, value_b])
            target_values.extend([value_b, value_a])
    if len(source_values) < 4:
        return np.nan
    if np.std(source_values) <= EPS or np.std(target_values) <= EPS:
        return np.nan
    return float(np.corrcoef(source_values, target_values)[0, 1])


def hub_jaccard(node_table, fraction=0.25):
    valid = node_table[
        np.isfinite(node_table["physical_complexity"])
        & np.isfinite(node_table["social_strength"])
    ]
    if len(valid) < 4:
        return np.nan, np.nan
    hub_count = max(1, int(np.ceil(len(valid) * fraction)))
    physical_hubs = set(
        valid.nlargest(hub_count, "physical_complexity")["social_instance_id"].astype(int)
    )
    social_hubs = set(
        valid.nlargest(hub_count, "social_strength")["social_instance_id"].astype(int)
    )
    intersection = len(physical_hubs & social_hubs)
    union = len(physical_hubs | social_hubs)
    return safe_divide(intersection, union), safe_divide(intersection, hub_count)


def sampled_global_efficiency(cost_adjacency, sampled_nodes):
    node_count = cost_adjacency.shape[0]
    if node_count < 2 or len(sampled_nodes) < 2:
        return np.nan
    shortest = dijkstra(
        cost_adjacency,
        directed=False,
        indices=sampled_nodes,
    )[:, sampled_nodes]
    off_diagonal = ~np.eye(len(sampled_nodes), dtype=bool)
    valid = off_diagonal & np.isfinite(shortest) & (shortest > EPS)
    inverse = np.zeros_like(shortest)
    inverse[valid] = 1.0 / shortest[valid]
    return float(np.sum(inverse) / (len(sampled_nodes) * (len(sampled_nodes) - 1)))


def coupled_transport_metrics(
    length_adjacency,
    scaled_skeleton_xyz,
    skeleton_social_ids,
    social_edge_table,
    omega_values,
    maximum_sample_nodes,
    random_seed,
):
    node_count = length_adjacency.shape[0]
    if node_count < 2:
        return {
            "cross_physical_global_efficiency": np.nan,
            "cross_coupled_efficiency_omega_1": np.nan,
            "cross_coupled_efficiency_gain_omega_1": np.nan,
            "cross_coupled_relative_gain_omega_1": np.nan,
            "cross_coupling_efficiency_auc": np.nan,
            "cross_coupling_gain_auc": np.nan,
        }
    generator = np.random.default_rng(random_seed)
    if node_count > maximum_sample_nodes:
        sampled_nodes = np.sort(
            generator.choice(node_count, size=maximum_sample_nodes, replace=False)
        )
    else:
        sampled_nodes = np.arange(node_count, dtype=np.int64)

    physical_cost = length_adjacency.copy().tocsr()
    baseline = sampled_global_efficiency(physical_cost, sampled_nodes)
    anchors = []
    for row in social_edge_table.itertuples(index=False):
        source_rows = np.where(
            skeleton_social_ids == int(row.source_social_id)
        )[0]
        target_rows = np.where(
            skeleton_social_ids == int(row.target_social_id)
        )[0]
        source_point = np.asarray(
            [row.source_closest_x, row.source_closest_y, row.source_closest_z],
            dtype=np.float64,
        )
        target_point = np.asarray(
            [row.target_closest_x, row.target_closest_y, row.target_closest_z],
            dtype=np.float64,
        )
        source_anchor = int(
            source_rows[
                np.argmin(
                    np.linalg.norm(
                        scaled_skeleton_xyz[source_rows] - source_point,
                        axis=1,
                    )
                )
            ]
        )
        target_anchor = int(
            target_rows[
                np.argmin(
                    np.linalg.norm(
                        scaled_skeleton_xyz[target_rows] - target_point,
                        axis=1,
                    )
                )
            ]
        )
        anchors.append((source_anchor, target_anchor, float(row.social_weight)))

    efficiencies = []
    for omega in omega_values:
        if len(anchors) == 0:
            efficiencies.append(baseline)
            continue
        graph = physical_cost.tolil(copy=True)
        for source, target, social_weight in anchors:
            virtual_cost = 1.0 / max(float(omega) * social_weight, EPS)
            existing = float(graph[source, target])
            if existing <= 0 or virtual_cost < existing:
                graph[source, target] = virtual_cost
                graph[target, source] = virtual_cost
        efficiencies.append(
            sampled_global_efficiency(graph.tocsr(), sampled_nodes)
        )

    efficiencies = np.asarray(efficiencies, dtype=np.float64)
    omega_values = np.asarray(omega_values, dtype=np.float64)
    log_omega = np.log10(omega_values)
    omega_one_position = int(np.argmin(np.abs(omega_values - 1.0)))
    coupled_one = float(efficiencies[omega_one_position])
    gain_one = coupled_one - baseline
    relative_gain_one = (
        gain_one / baseline if baseline > EPS else np.nan
    )
    if len(omega_values) >= 2:
        log_range = float(log_omega[-1] - log_omega[0])
        efficiency_auc = float(_trapezoid(efficiencies, log_omega) / log_range)
        gain_auc = float(
            _trapezoid(efficiencies - baseline, log_omega) / log_range
        )
    else:
        efficiency_auc = coupled_one
        gain_auc = gain_one
    return {
        "cross_physical_global_efficiency": baseline,
        "cross_coupled_efficiency_omega_1": coupled_one,
        "cross_coupled_efficiency_gain_omega_1": gain_one,
        "cross_coupled_relative_gain_omega_1": relative_gain_one,
        "cross_coupling_efficiency_auc": efficiency_auc,
        "cross_coupling_gain_auc": gain_auc,
    }


def aggregate_physical_metrics(physical_object_table, physical_edge_table):
    lengths = physical_object_table["physical_total_skeleton_length"].to_numpy(dtype=float)
    total_length = float(np.nansum(lengths))

    def length_weighted(column):
        values = physical_object_table[column].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(lengths) & (lengths > 0)
        if not np.any(valid):
            return np.nan
        return float(np.average(values[valid], weights=lengths[valid]))

    return {
        "physical_total_skeleton_length": total_length,
        "physical_largest_component_length_fraction": (
            safe_divide(np.nanmax(lengths), total_length)
            if len(lengths) > 0
            else np.nan
        ),
        "physical_junction_count": int(
            np.nansum(physical_object_table["physical_junction_count"])
        ),
        "physical_endpoint_count": int(
            np.nansum(physical_object_table["physical_endpoint_count"])
        ),
        "physical_cycle_rank": int(
            np.nansum(physical_object_table["physical_cycle_rank"])
        ),
        "physical_junction_density": safe_divide(
            np.nansum(physical_object_table["physical_junction_count"]),
            total_length,
        ),
        "physical_endpoint_density": safe_divide(
            np.nansum(physical_object_table["physical_endpoint_count"]),
            total_length,
        ),
        "physical_cycle_rank_density": safe_divide(
            np.nansum(physical_object_table["physical_cycle_rank"]),
            total_length,
        ),
        "physical_branch_length_entropy": normalized_entropy(
            physical_edge_table["arc_length"].to_numpy(dtype=float)
        ),
        "physical_route_efficiency": length_weighted(
            "physical_route_efficiency"
        ),
        "physical_algebraic_connectivity": length_weighted(
            "physical_algebraic_connectivity"
        ),
        "physical_mean_effective_resistance": length_weighted(
            "physical_mean_effective_resistance"
        ),
        "physical_mean_edge_length": (
            float(np.mean(physical_edge_table["arc_length"]))
            if len(physical_edge_table) > 0
            else np.nan
        ),
        "physical_edge_length_cv": coefficient_of_variation(
            physical_edge_table["arc_length"].to_numpy(dtype=float)
        ),
        "physical_mean_tortuosity": (
            float(np.mean(physical_edge_table["tortuosity"]))
            if len(physical_edge_table) > 0
            else np.nan
        ),
    }



def aggregate_physical_composition_topology_metrics(
    physical_object_table,
    size_tail_fraction=0.25,
):
    """
    把 physical layer 从“几个全局 graph summary”扩展为三个明确维度：

    1) population composition
       - simple objects 占比
       - branched / cyclic objects 占比
       - 大对象占据多少 skeleton length

    2) topology of large objects
       - 大对象内部 junction density / cycle density / route efficiency

    3) size-topology coupling
       - object size 与 junction / cycle complexity 的 Spearman
       - large-small topology contrast

    注意：这里没有逐对象 voxel volume，因此 object size 使用
    physical_total_skeleton_length 作为统一、可复现的 size proxy。
    """
    table = physical_object_table.copy()

    if len(table) == 0:
        keys = [
            "physical_simple_object_fraction",
            "physical_simple_object_length_fraction",
            "physical_branched_object_fraction",
            "physical_cyclic_object_fraction",
            "physical_largest_object_length_fraction",
            "physical_large_object_length_fraction",
            "physical_large_object_junction_density_median",
            "physical_large_object_cycle_rank_density_median",
            "physical_large_object_route_efficiency_median",
            "physical_large_small_junction_density_contrast",
            "physical_large_small_cycle_rank_density_contrast",
            "physical_large_small_route_efficiency_contrast",
            "physical_object_size_junction_density_spearman",
            "physical_object_size_cycle_rank_density_spearman",
            "physical_object_size_route_efficiency_spearman",
        ]
        return {key: np.nan for key in keys}

    size = table[
        "physical_total_skeleton_length"
    ].to_numpy(dtype=float)

    instance_count = table[
        "physical_instance_count"
    ].to_numpy(dtype=float)

    junction_count = table[
        "physical_junction_count"
    ].to_numpy(dtype=float)

    cycle_rank = table[
        "physical_cycle_rank"
    ].to_numpy(dtype=float)

    junction_density = table[
        "physical_junction_density"
    ].to_numpy(dtype=float)

    cycle_density = table[
        "physical_cycle_rank_density"
    ].to_numpy(dtype=float)

    route_efficiency = table[
        "physical_route_efficiency"
    ].to_numpy(dtype=float)

    valid_size = np.isfinite(size) & (size > 0)
    total_length = float(np.sum(size[valid_size]))

    # simple = one unbranched, acyclic physical segment
    simple = (
        valid_size
        & np.isfinite(instance_count)
        & np.isfinite(junction_count)
        & np.isfinite(cycle_rank)
        & (instance_count <= 1)
        & (junction_count <= 0)
        & (cycle_rank <= 0)
    )

    branched = (
        valid_size
        & np.isfinite(junction_count)
        & (junction_count > 0)
    )

    cyclic = (
        valid_size
        & np.isfinite(cycle_rank)
        & (cycle_rank > 0)
    )

    large_mask, _ = ranked_tail_masks(
        size,
        fraction=size_tail_fraction,
        descending=True,
    )
    small_mask, _ = ranked_tail_masks(
        size,
        fraction=size_tail_fraction,
        descending=False,
    )

    def valid_fraction(mask):
        denominator = int(np.sum(valid_size))
        if denominator == 0:
            return np.nan
        return float(np.sum(mask & valid_size) / denominator)

    def length_fraction(mask):
        if total_length <= EPS:
            return np.nan
        return float(np.sum(size[mask & valid_size]) / total_length)

    def contrast(values):
        large_value = median_for_mask(values, large_mask)
        small_value = median_for_mask(values, small_mask)
        if not (
            np.isfinite(large_value)
            and np.isfinite(small_value)
        ):
            return np.nan
        return float(large_value - small_value)

    return {
        "physical_simple_object_fraction": valid_fraction(simple),
        "physical_simple_object_length_fraction": length_fraction(simple),
        "physical_branched_object_fraction": valid_fraction(branched),
        "physical_cyclic_object_fraction": valid_fraction(cyclic),

        "physical_largest_object_length_fraction": (
            safe_divide(np.nanmax(size[valid_size]), total_length)
            if np.any(valid_size)
            else np.nan
        ),
        "physical_large_object_length_fraction": length_fraction(
            large_mask
        ),

        "physical_large_object_junction_density_median": median_for_mask(
            junction_density,
            large_mask,
        ),
        "physical_large_object_cycle_rank_density_median": median_for_mask(
            cycle_density,
            large_mask,
        ),
        "physical_large_object_route_efficiency_median": median_for_mask(
            route_efficiency,
            large_mask,
        ),

        "physical_large_small_junction_density_contrast": contrast(
            junction_density
        ),
        "physical_large_small_cycle_rank_density_contrast": contrast(
            cycle_density
        ),
        "physical_large_small_route_efficiency_contrast": contrast(
            route_efficiency
        ),

        "physical_object_size_junction_density_spearman": (
            spearman_correlation(
                size,
                junction_density,
                minimum_n=6,
            )
        ),
        "physical_object_size_cycle_rank_density_spearman": (
            spearman_correlation(
                size,
                cycle_density,
                minimum_n=6,
            )
        ),
        "physical_object_size_route_efficiency_spearman": (
            spearman_correlation(
                size,
                route_efficiency,
                minimum_n=6,
            )
        ),
    }


def normalized_balance(physical_value, social_value):
    """
    有界physical-social balance：(P-S)/(|P|+|S|)。

    接近1表示physical占优，接近-1表示social占优，0表示两层平衡。
    """
    if not np.isfinite(physical_value) or not np.isfinite(social_value):
        return np.nan
    denominator = abs(float(physical_value)) + abs(float(social_value))
    if denominator <= EPS:
        return np.nan
    return float((physical_value - social_value) / denominator)


def calculate_hierarchy_balance_metrics(
    physical_metrics,
    social_metrics,
    cross_metrics,
    combined_node_table,
):
    """仅保留两个有机制解释、不是原指标代数换名的跨层指标。"""
    internal_edges = combined_node_table[
        "physical_internal_network_edge_count"
    ].to_numpy(dtype=float)
    external_degree = combined_node_table["social_degree"].to_numpy(dtype=float)
    denominator = internal_edges + external_degree
    external_fraction = np.full(len(denominator), np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator > EPS)
    external_fraction[valid] = external_degree[valid] / denominator[valid]
    return {
        "hierarchy_external_connectivity_fraction_median": (
            float(np.median(external_fraction[valid]))
            if np.any(valid)
            else np.nan
        ),
        "hierarchy_coupling_relative_gain_auc": safe_divide(
            cross_metrics["cross_coupling_gain_auc"],
            cross_metrics["cross_physical_global_efficiency"],
        ),
    }


def aggregate_instance_resolved_metrics(
    physical_node_table,
    social_node_table,
    low_local_efficiency_threshold=0.25,
    hub_fraction=0.10,
    size_tail_fraction=0.25,
):
    """
    把node/instance属性压缩成每张图片的一组稳健、机制导向指标。

    physical size使用arc_length；social size使用PCD表面点数。
    hub分别按physical degree和social strength的前hub_fraction定义。
    large/small分别按图片内size rank的两端size_tail_fraction定义。
    """
    output = {}

    physical_local_efficiency = physical_node_table[
        "physical_local_efficiency"
    ].to_numpy(dtype=float)
    physical_arc_length = physical_node_table["arc_length"].to_numpy(dtype=float)

    output["instance_physical_local_efficiency_median"] = (
        float(np.median(physical_local_efficiency))
        if len(physical_local_efficiency) > 0
        else np.nan
    )
    valid_efficiency = np.isfinite(physical_local_efficiency)
    output["instance_physical_low_local_efficiency_fraction"] = (
        float(
            np.mean(
                physical_local_efficiency[valid_efficiency]
                < low_local_efficiency_threshold
            )
        )
        if np.any(valid_efficiency)
        else np.nan
    )

    physical_large, _ = ranked_tail_masks(
        physical_arc_length,
        fraction=size_tail_fraction,
        descending=True,
    )
    physical_small, _ = ranked_tail_masks(
        physical_arc_length,
        fraction=size_tail_fraction,
        descending=False,
    )
    physical_large_efficiency = median_for_mask(
        physical_local_efficiency,
        physical_large,
    )
    physical_small_efficiency = median_for_mask(
        physical_local_efficiency,
        physical_small,
    )
    output["instance_physical_size_local_efficiency_contrast"] = (
        physical_large_efficiency - physical_small_efficiency
        if np.isfinite(physical_large_efficiency)
        and np.isfinite(physical_small_efficiency)
        else np.nan
    )
    output["instance_physical_length_local_efficiency_spearman"] = (
        spearman_correlation(
            physical_arc_length,
            physical_local_efficiency,
            minimum_n=6,
        )
    )
    output["instance_physical_betweenness_top10_share"] = top_fraction_share(
        physical_node_table["physical_betweenness"].to_numpy(dtype=float),
        fraction=hub_fraction,
    )

    social_local_efficiency = social_node_table[
        "social_local_efficiency"
    ].to_numpy(dtype=float)
    social_strength = social_node_table["social_strength"].to_numpy(dtype=float)
    social_size = social_node_table["pcd_surface_point_count"].to_numpy(dtype=float)
    physical_route_efficiency = social_node_table[
        "physical_route_efficiency"
    ].to_numpy(dtype=float)

    valid_social_efficiency = np.isfinite(social_local_efficiency)
    output["instance_social_low_local_efficiency_fraction"] = (
        float(
            np.mean(
                social_local_efficiency[valid_social_efficiency]
                < low_local_efficiency_threshold
            )
        )
        if np.any(valid_social_efficiency)
        else np.nan
    )

    social_hubs, _ = ranked_tail_masks(
        social_strength,
        fraction=hub_fraction,
        descending=True,
    )
    social_nonhubs = ~social_hubs & np.isfinite(social_strength)
    social_hub_efficiency = median_for_mask(
        social_local_efficiency,
        social_hubs,
    )
    social_nonhub_efficiency = median_for_mask(
        social_local_efficiency,
        social_nonhubs,
    )
    output["instance_social_hub_local_efficiency_contrast"] = (
        social_hub_efficiency - social_nonhub_efficiency
        if np.isfinite(social_hub_efficiency)
        and np.isfinite(social_nonhub_efficiency)
        else np.nan
    )

    social_large, _ = ranked_tail_masks(
        social_size,
        fraction=size_tail_fraction,
        descending=True,
    )
    social_small, _ = ranked_tail_masks(
        social_size,
        fraction=size_tail_fraction,
        descending=False,
    )
    large_route_efficiency = median_for_mask(
        physical_route_efficiency,
        social_large,
    )
    small_route_efficiency = median_for_mask(
        physical_route_efficiency,
        social_small,
    )
    output["instance_social_size_physical_route_efficiency_contrast"] = (
        large_route_efficiency - small_route_efficiency
        if np.isfinite(large_route_efficiency)
        and np.isfinite(small_route_efficiency)
        else np.nan
    )
    return output




###############################################################################
# v8: WHOLE-CELL PHYSICAL-SEGMENT PROXIMITY NETWORK
###############################################################################

def _finite_gini(values):
    """Gini for non-negative finite values; returns NaN when undefined."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    minimum = float(np.min(values))
    if minimum < 0:
        values = values - minimum
    total = float(np.sum(values))
    if total <= EPS:
        return 0.0
    values = np.sort(values)
    n = len(values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(
        2.0 * np.sum(ranks * values) / (n * total)
        - (n + 1.0) / n
    )


def _entropy_from_histogram(values):
    """Normalized Shannon entropy of a discrete empirical distribution."""
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    return normalized_entropy(counts.astype(np.float64))


def _effective_number_from_sizes(sizes):
    """Effective number of components/modules = exp(Shannon entropy)."""
    sizes = np.asarray(sizes, dtype=np.float64)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if len(sizes) == 0:
        return np.nan
    probabilities = sizes / np.sum(sizes)
    entropy = -np.sum(probabilities * np.log(probabilities + EPS))
    return float(np.exp(entropy))


def _whole_physical_louvain_partition(graph, resolution=0.5, random_seed=42):
    """
    Whole-cell physical graph uses the same Louvain-style modularity logic
    as the user's previous single-network analysis.
    """
    if graph.number_of_nodes() == 0:
        return [], np.nan
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes()]
        return communities, 0.0

    try:
        communities = nx.algorithms.community.louvain_communities(
            graph,
            resolution=resolution,
            seed=random_seed,
        )
    except Exception:
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(graph)
        )

    try:
        modularity_value = float(
            nx.algorithms.community.modularity(graph, communities)
        )
    except Exception:
        modularity_value = np.nan

    return communities, modularity_value


def _whole_physical_spatial_entropy(node_table, bins=4):
    """
    Spatial entropy of physical-branch centroids.
    Kept as an exploratory old-style metric, not a core compact metric.
    """
    coordinate_columns = [
        "whole_physical_centroid_x",
        "whole_physical_centroid_y",
        "whole_physical_centroid_z",
    ]
    if not set(coordinate_columns).issubset(node_table.columns):
        return np.nan

    coordinates = node_table[coordinate_columns].to_numpy(dtype=np.float64)
    coordinates = coordinates[np.all(np.isfinite(coordinates), axis=1)]
    if len(coordinates) < 2:
        return np.nan

    # Drop dimensions with essentially zero range so histogramdd remains stable.
    ranges = np.ptp(coordinates, axis=0)
    keep = ranges > EPS
    coordinates = coordinates[:, keep]
    if coordinates.shape[1] == 0:
        return 0.0

    try:
        histogram, _ = np.histogramdd(coordinates, bins=bins)
        counts = histogram.ravel()
        counts = counts[counts > 0]
        return normalized_entropy(counts)
    except Exception:
        return np.nan


def _whole_physical_von_neumann_entropy(graph, maximum_nodes=800):
    """Normalized Von Neumann entropy from the normalized graph Laplacian."""
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    if n > maximum_nodes:
        return np.nan

    node_order = list(graph.nodes())
    adjacency = nx.to_numpy_array(
        graph,
        nodelist=node_order,
        weight=None,
        dtype=np.float64,
    )
    degree = np.sum(adjacency, axis=1)
    inv_sqrt = np.zeros(n, dtype=np.float64)
    positive = degree > EPS
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_laplacian = (
        np.eye(n, dtype=np.float64)
        - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    )
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    probabilities = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(probabilities * np.log(probabilities + EPS))
    return float(entropy / np.log(n))


def _whole_physical_weighted_global_efficiency(
    graph,
    cost_key="proximity_cost",
    maximum_nodes=800,
):
    """
    Weighted global efficiency using a strictly positive proximity cost.

    proximity_cost = 1 + minimum_segment_distance / threshold.
    This avoids zero-length edges at true junctions while retaining a
    distance penalty. It is supplementary; the primary old-style metrics
    remain the unweighted graph metrics.
    """
    n = graph.number_of_nodes()
    if n <= 1 or n > maximum_nodes:
        return np.nan

    total_inverse_distance = 0.0
    try:
        all_lengths = nx.all_pairs_dijkstra_path_length(
            graph,
            weight=cost_key,
        )
        for source, source_lengths in all_lengths:
            for target, path_length in source_lengths.items():
                if source == target:
                    continue
                if np.isfinite(path_length) and path_length > EPS:
                    total_inverse_distance += 1.0 / path_length
        return float(total_inverse_distance / (n * (n - 1)))
    except Exception:
        return np.nan


def build_whole_cell_physical_proximity_graph(
    physical_node_table,
    physical_network_edge_table,
    scaled_skeleton_xyz,
    skeleton_edge_ids,
    distance_threshold,
):
    """
    Build the old-style whole-cell physical network at branch/segment level.

    Node
    ----
    One physical skeleton edge/branch (physical_edge_id).

    Edge
    ----
    Any two different physical branches whose minimum skeleton-point distance
    is <= distance_threshold, irrespective of whether they belong to the same
    social mitochondrial object.

    This deliberately preserves the branch-level mesoscale proximity
    information that is compressed when a whole mitochondrial object is
    represented by one social node.

    Every proximity edge is additionally classified as:
      1) intra_object_true_topology
      2) intra_object_nonlocal_proximity
      3) inter_object_proximity
    """
    if distance_threshold <= 0:
        raise ValueError("whole_physical_distance_threshold must be > 0.")

    physical_ids = physical_node_table[
        "physical_edge_id"
    ].to_numpy(dtype=np.int64)
    physical_id_set = set(int(value) for value in physical_ids)

    valid_skeleton_rows = np.where(skeleton_edge_ids > 0)[0]
    point_xyz = scaled_skeleton_xyz[valid_skeleton_rows]
    point_edge_ids = skeleton_edge_ids[valid_skeleton_rows].astype(np.int64)

    social_by_edge = dict(
        zip(
            physical_node_table["physical_edge_id"].astype(int),
            physical_node_table["social_instance_id"].astype(int),
        )
    )

    # Branch centroid is useful both for graph export and spatial entropy.
    centroid_records = []
    for physical_edge_id in physical_ids:
        rows = valid_skeleton_rows[
            point_edge_ids == int(physical_edge_id)
        ]
        if len(rows) == 0:
            centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        else:
            centroid = np.mean(scaled_skeleton_xyz[rows], axis=0)
        centroid_records.append(
            {
                "physical_edge_id": int(physical_edge_id),
                "whole_physical_centroid_x": float(centroid[0]),
                "whole_physical_centroid_y": float(centroid[1]),
                "whole_physical_centroid_z": float(centroid[2]),
            }
        )
    centroid_table = pd.DataFrame.from_records(centroid_records)
    whole_node_table = physical_node_table.merge(
        centroid_table,
        on="physical_edge_id",
        how="left",
        validate="one_to_one",
    )

    edge_columns = [
        "source_physical_edge_id",
        "target_physical_edge_id",
        "source_social_id",
        "target_social_id",
        "minimum_segment_distance",
        "point_pair_count_within_threshold",
        "proximity_weight",
        "proximity_cost",
        "same_social_object",
        "shares_true_junction",
        "edge_class",
    ]

    if len(point_xyz) < 2 or len(physical_ids) < 2:
        edge_table = pd.DataFrame(columns=edge_columns)
        return whole_node_table, edge_table

    # Query all skeleton-point pairs within threshold once, then collapse
    # those point pairs to unique physical-branch pairs. This is much faster
    # than testing every branch pair independently.
    tree = cKDTree(point_xyz)
    point_pairs = tree.query_pairs(
        r=float(distance_threshold),
        output_type="ndarray",
    )

    if len(point_pairs) == 0:
        edge_table = pd.DataFrame(columns=edge_columns)
        return whole_node_table, edge_table

    source_edge = point_edge_ids[point_pairs[:, 0]]
    target_edge = point_edge_ids[point_pairs[:, 1]]
    different_edge = source_edge != target_edge

    point_pairs = point_pairs[different_edge]
    source_edge = source_edge[different_edge]
    target_edge = target_edge[different_edge]

    if len(point_pairs) == 0:
        edge_table = pd.DataFrame(columns=edge_columns)
        return whole_node_table, edge_table

    # Guard against any skeleton edge ID not present in physical_node_table.
    valid_edge_pair = np.array(
        [
            int(a) in physical_id_set and int(b) in physical_id_set
            for a, b in zip(source_edge, target_edge)
        ],
        dtype=bool,
    )
    point_pairs = point_pairs[valid_edge_pair]
    source_edge = source_edge[valid_edge_pair]
    target_edge = target_edge[valid_edge_pair]

    if len(point_pairs) == 0:
        edge_table = pd.DataFrame(columns=edge_columns)
        return whole_node_table, edge_table

    point_distances = np.linalg.norm(
        point_xyz[point_pairs[:, 0]]
        - point_xyz[point_pairs[:, 1]],
        axis=1,
    )

    branch_pairs = np.column_stack(
        [
            np.minimum(source_edge, target_edge),
            np.maximum(source_edge, target_edge),
        ]
    ).astype(np.int64)

    unique_pairs, inverse = np.unique(
        branch_pairs,
        axis=0,
        return_inverse=True,
    )
    minimum_distances = np.full(
        len(unique_pairs),
        np.inf,
        dtype=np.float64,
    )
    np.minimum.at(minimum_distances, inverse, point_distances)
    point_pair_counts = np.bincount(
        inverse,
        minlength=len(unique_pairs),
    ).astype(np.int64)

    topology_pair_set = set()
    if len(physical_network_edge_table) > 0:
        for row in physical_network_edge_table.itertuples(index=False):
            a = int(row.source_physical_edge_id)
            b = int(row.target_physical_edge_id)
            topology_pair_set.add((min(a, b), max(a, b)))

    records = []
    for pair_index, (source_id, target_id) in enumerate(unique_pairs):
        source_id = int(source_id)
        target_id = int(target_id)
        source_social = int(social_by_edge[source_id])
        target_social = int(social_by_edge[target_id])
        same_social = source_social == target_social
        shares_true_junction = (
            (source_id, target_id) in topology_pair_set
        )

        if not same_social:
            edge_class = "inter_object_proximity"
        elif shares_true_junction:
            edge_class = "intra_object_true_topology"
        else:
            edge_class = "intra_object_nonlocal_proximity"

        minimum_distance = float(minimum_distances[pair_index])
        proximity_weight = float(
            np.exp(
                -(
                    minimum_distance
                    / max(float(distance_threshold), EPS)
                ) ** 2
            )
        )
        proximity_cost = float(
            1.0
            + minimum_distance
            / max(float(distance_threshold), EPS)
        )

        records.append(
            {
                "source_physical_edge_id": source_id,
                "target_physical_edge_id": target_id,
                "source_social_id": source_social,
                "target_social_id": target_social,
                "minimum_segment_distance": minimum_distance,
                "point_pair_count_within_threshold": int(
                    point_pair_counts[pair_index]
                ),
                "proximity_weight": proximity_weight,
                "proximity_cost": proximity_cost,
                "same_social_object": bool(same_social),
                "shares_true_junction": bool(shares_true_junction),
                "edge_class": edge_class,
            }
        )

    edge_table = pd.DataFrame.from_records(
        records,
        columns=edge_columns,
    )
    return whole_node_table, edge_table


def compute_whole_cell_physical_graph_metrics(
    node_table,
    edge_table,
    distance_threshold,
    centrality_sample_nodes=300,
    spatial_bins=4,
    spectral_entropy_max_nodes=800,
    weighted_efficiency_max_nodes=800,
    random_seed=42,
):
    """
    Compute the user's previous single-network metric family on the new
    whole-cell branch proximity graph, with the prefix whole_physical_.

    Main old-style six:
      density, average_clustering, transitivity,
      global_efficiency, local_efficiency, modularity.

    Also includes fragmentation, entropy, hub/centrality, k-core and
    edge-type composition metrics.
    """
    graph = nx.Graph()

    for row in node_table.itertuples(index=False):
        graph.add_node(
            int(row.physical_edge_id),
            social_instance_id=int(row.social_instance_id),
            pos=(
                float(row.whole_physical_centroid_x),
                float(row.whole_physical_centroid_y),
                float(row.whole_physical_centroid_z),
            ),
        )

    for row in edge_table.itertuples(index=False):
        graph.add_edge(
            int(row.source_physical_edge_id),
            int(row.target_physical_edge_id),
            minimum_segment_distance=float(row.minimum_segment_distance),
            proximity_weight=float(row.proximity_weight),
            proximity_cost=float(row.proximity_cost),
            same_social_object=bool(row.same_social_object),
            shares_true_junction=bool(row.shares_true_junction),
            edge_class=str(row.edge_class),
        )

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    metrics = {
        "whole_physical_node_count": int(n),
        "whole_physical_edge_count": int(m),
        "whole_physical_distance_threshold": float(distance_threshold),
    }

    if n == 0:
        return node_table, graph, metrics

    # ------------------------------------------------------------------
    # Old six core network indices
    # ------------------------------------------------------------------
    metrics["whole_physical_density"] = float(nx.density(graph))

    try:
        metrics["whole_physical_average_clustering"] = float(
            nx.average_clustering(graph)
        )
    except Exception:
        metrics["whole_physical_average_clustering"] = np.nan

    try:
        metrics["whole_physical_transitivity"] = float(
            nx.transitivity(graph)
        )
    except Exception:
        metrics["whole_physical_transitivity"] = np.nan

    try:
        metrics["whole_physical_global_efficiency"] = float(
            nx.global_efficiency(graph)
        )
    except Exception:
        metrics["whole_physical_global_efficiency"] = np.nan

    try:
        metrics["whole_physical_local_efficiency"] = float(
            nx.local_efficiency(graph)
        )
    except Exception:
        metrics["whole_physical_local_efficiency"] = np.nan

    communities, modularity_value = _whole_physical_louvain_partition(
        graph,
        resolution=0.5,
        random_seed=random_seed,
    )
    metrics["whole_physical_modularity"] = modularity_value
    metrics["whole_physical_module_count"] = int(len(communities))
    metrics["whole_physical_module_size_entropy"] = normalized_entropy(
        np.asarray([len(group) for group in communities], dtype=np.float64)
    )

    try:
        metrics["whole_physical_diameter"] = (
            int(nx.diameter(graph))
            if nx.is_connected(graph)
            else np.nan
        )
    except Exception:
        metrics["whole_physical_diameter"] = np.nan

    # ------------------------------------------------------------------
    # Component / fragmentation
    # ------------------------------------------------------------------
    components = list(nx.connected_components(graph))
    component_sizes = np.asarray(
        [len(component) for component in components],
        dtype=np.float64,
    )
    component_count = len(component_sizes)
    largest_component_size = (
        int(np.max(component_sizes))
        if len(component_sizes) > 0
        else 0
    )
    largest_component_fraction = (
        float(largest_component_size / n)
        if n > 0
        else np.nan
    )
    isolated_node_count = int(np.sum(component_sizes == 1))
    small_component_nodes = int(
        np.sum(component_sizes[component_sizes <= 2])
    )

    metrics.update(
        {
            "whole_physical_component_count": int(component_count),
            "whole_physical_component_count_per_node": safe_divide(
                component_count,
                n,
            ),
            "whole_physical_largest_component_size": largest_component_size,
            "whole_physical_largest_component_fraction": (
                largest_component_fraction
            ),
            "whole_physical_small_component_fraction": safe_divide(
                small_component_nodes,
                n,
            ),
            "whole_physical_isolated_node_count": isolated_node_count,
            "whole_physical_isolated_node_fraction": safe_divide(
                isolated_node_count,
                n,
            ),
            "whole_physical_fragmentation_index": (
                1.0 - largest_component_fraction
                if np.isfinite(largest_component_fraction)
                else np.nan
            ),
            "whole_physical_component_size_entropy": normalized_entropy(
                component_sizes
            ),
            "whole_physical_effective_component_count": (
                _effective_number_from_sizes(component_sizes)
            ),
        }
    )

    # LCC metrics.
    if len(components) > 0:
        largest_nodes = max(components, key=len)
        lcc = graph.subgraph(largest_nodes).copy()
    else:
        lcc = None

    if lcc is not None and lcc.number_of_nodes() > 1:
        try:
            metrics["whole_physical_avg_shortest_path_lcc"] = float(
                nx.average_shortest_path_length(lcc)
            )
        except Exception:
            metrics["whole_physical_avg_shortest_path_lcc"] = np.nan
        try:
            metrics["whole_physical_diameter_lcc"] = int(
                nx.diameter(lcc)
            )
        except Exception:
            metrics["whole_physical_diameter_lcc"] = np.nan
        try:
            metrics["whole_physical_global_efficiency_lcc"] = float(
                nx.global_efficiency(lcc)
            )
        except Exception:
            metrics["whole_physical_global_efficiency_lcc"] = np.nan
    else:
        metrics["whole_physical_avg_shortest_path_lcc"] = np.nan
        metrics["whole_physical_diameter_lcc"] = np.nan
        metrics["whole_physical_global_efficiency_lcc"] = np.nan

    # ------------------------------------------------------------------
    # Degree / hub / backbone
    # ------------------------------------------------------------------
    degree_dict = dict(graph.degree())
    degree_values = np.asarray(
        [degree_dict[node] for node in graph.nodes()],
        dtype=np.float64,
    )
    degree_mean = float(np.mean(degree_values)) if n > 0 else np.nan
    degree_std = float(np.std(degree_values, ddof=0)) if n > 0 else np.nan

    metrics.update(
        {
            "whole_physical_degree_mean": degree_mean,
            "whole_physical_degree_median": (
                float(np.median(degree_values))
                if n > 0 else np.nan
            ),
            "whole_physical_degree_max": (
                float(np.max(degree_values))
                if n > 0 else np.nan
            ),
            "whole_physical_degree_std": degree_std,
            "whole_physical_degree_cv": safe_divide(
                degree_std,
                degree_mean,
            ),
            "whole_physical_degree_gini": _finite_gini(degree_values),
            "whole_physical_degree_entropy": _entropy_from_histogram(
                degree_values
            ),
        }
    )

    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
        metrics["whole_physical_degree_assortativity"] = (
            float(assortativity)
            if np.isfinite(assortativity)
            else np.nan
        )
    except Exception:
        metrics["whole_physical_degree_assortativity"] = np.nan

    # Closeness.
    try:
        closeness_dict = nx.closeness_centrality(graph)
        closeness_values = np.asarray(
            [closeness_dict[node] for node in graph.nodes()],
            dtype=np.float64,
        )
    except Exception:
        closeness_dict = {
            node: np.nan for node in graph.nodes()
        }
        closeness_values = np.full(n, np.nan)

    finite_closeness = closeness_values[np.isfinite(closeness_values)]
    metrics["whole_physical_closeness_mean"] = (
        float(np.mean(finite_closeness))
        if len(finite_closeness) > 0
        else np.nan
    )
    metrics["whole_physical_closeness_max"] = (
        float(np.max(finite_closeness))
        if len(finite_closeness) > 0
        else np.nan
    )
    metrics["whole_physical_closeness_gini"] = _finite_gini(
        finite_closeness
    )
    metrics["whole_physical_closeness_entropy"] = normalized_entropy(
        finite_closeness
    )

    # Betweenness: sample only for large graphs, matching the old code.
    try:
        if n > centrality_sample_nodes:
            betweenness_dict = nx.betweenness_centrality(
                graph,
                k=centrality_sample_nodes,
                normalized=True,
                weight=None,
                seed=random_seed,
            )
        else:
            betweenness_dict = nx.betweenness_centrality(
                graph,
                normalized=True,
                weight=None,
            )
        betweenness_values = np.asarray(
            [betweenness_dict[node] for node in graph.nodes()],
            dtype=np.float64,
        )
    except Exception:
        betweenness_dict = {
            node: np.nan for node in graph.nodes()
        }
        betweenness_values = np.full(n, np.nan)

    finite_betweenness = betweenness_values[
        np.isfinite(betweenness_values)
    ]
    metrics["whole_physical_betweenness_mean"] = (
        float(np.mean(finite_betweenness))
        if len(finite_betweenness) > 0
        else np.nan
    )
    metrics["whole_physical_betweenness_max"] = (
        float(np.max(finite_betweenness))
        if len(finite_betweenness) > 0
        else np.nan
    )
    metrics["whole_physical_betweenness_gini"] = _finite_gini(
        finite_betweenness
    )
    metrics["whole_physical_betweenness_entropy"] = normalized_entropy(
        finite_betweenness
    )

    # k-core.
    try:
        graph_no_selfloop = graph.copy()
        graph_no_selfloop.remove_edges_from(
            nx.selfloop_edges(graph_no_selfloop)
        )
        if graph_no_selfloop.number_of_edges() > 0:
            core_dict = nx.core_number(graph_no_selfloop)
            core_values = np.asarray(
                [core_dict[node] for node in graph.nodes()],
                dtype=np.float64,
            )
            maximum_core = int(np.max(core_values))
            core_fraction = float(
                np.mean(core_values == maximum_core)
            )
        else:
            core_dict = {node: 0 for node in graph.nodes()}
            maximum_core = 0
            core_fraction = 0.0
    except Exception:
        core_dict = {node: np.nan for node in graph.nodes()}
        maximum_core = np.nan
        core_fraction = np.nan

    metrics["whole_physical_max_core_number"] = maximum_core
    metrics["whole_physical_core_node_fraction"] = core_fraction

    # ------------------------------------------------------------------
    # Entropy / weighted / composite metrics from the old analysis
    # ------------------------------------------------------------------
    metrics["whole_physical_spatial_entropy"] = (
        _whole_physical_spatial_entropy(
            node_table,
            bins=spatial_bins,
        )
    )
    metrics["whole_physical_graph_von_neumann_entropy"] = (
        _whole_physical_von_neumann_entropy(
            graph,
            maximum_nodes=spectral_entropy_max_nodes,
        )
    )
    metrics["whole_physical_weighted_global_efficiency"] = (
        _whole_physical_weighted_global_efficiency(
            graph,
            cost_key="proximity_cost",
            maximum_nodes=weighted_efficiency_max_nodes,
        )
    )

    metrics["whole_physical_integration_score"] = (
        metrics["whole_physical_global_efficiency"]
        * largest_component_fraction
        if np.isfinite(metrics["whole_physical_global_efficiency"])
        and np.isfinite(largest_component_fraction)
        else np.nan
    )
    metrics["whole_physical_segregation_score"] = (
        metrics["whole_physical_modularity"]
        * metrics["whole_physical_fragmentation_index"]
        if np.isfinite(metrics["whole_physical_modularity"])
        and np.isfinite(metrics["whole_physical_fragmentation_index"])
        else np.nan
    )
    metrics["whole_physical_hubness_score"] = (
        np.nanmean(
            [
                metrics["whole_physical_degree_gini"],
                metrics["whole_physical_betweenness_gini"],
            ]
        )
    )
    entropy_candidates = np.asarray(
        [
            metrics["whole_physical_component_size_entropy"],
            metrics["whole_physical_module_size_entropy"],
            metrics["whole_physical_degree_entropy"],
            metrics["whole_physical_spatial_entropy"],
        ],
        dtype=np.float64,
    )
    entropy_candidates = entropy_candidates[
        np.isfinite(entropy_candidates)
    ]
    metrics["whole_physical_entropy_disorder_score"] = (
        float(np.mean(entropy_candidates))
        if len(entropy_candidates) > 0
        else np.nan
    )

    # ------------------------------------------------------------------
    # New edge-type composition: what part of this old-style network
    # comes from true internal topology vs nonlocal/inter-object proximity?
    # ------------------------------------------------------------------
    if m > 0:
        inter_mask = (
            edge_table["edge_class"] == "inter_object_proximity"
        ).to_numpy()
        topology_mask = (
            edge_table["edge_class"] == "intra_object_true_topology"
        ).to_numpy()
        nonlocal_mask = (
            edge_table["edge_class"]
            == "intra_object_nonlocal_proximity"
        ).to_numpy()

        inter_count = int(np.sum(inter_mask))
        topology_count = int(np.sum(topology_mask))
        nonlocal_count = int(np.sum(nonlocal_mask))

        metrics.update(
            {
                "whole_physical_inter_object_edge_count": inter_count,
                "whole_physical_inter_object_edge_fraction": (
                    inter_count / m
                ),
                "whole_physical_true_topology_edge_count": topology_count,
                "whole_physical_true_topology_edge_fraction": (
                    topology_count / m
                ),
                "whole_physical_intra_nonlocal_edge_count": nonlocal_count,
                "whole_physical_intra_nonlocal_edge_fraction": (
                    nonlocal_count / m
                ),
                "whole_physical_mean_minimum_segment_distance": float(
                    edge_table["minimum_segment_distance"].mean()
                ),
                "whole_physical_inter_object_mean_minimum_segment_distance": (
                    float(
                        edge_table.loc[
                            inter_mask,
                            "minimum_segment_distance",
                        ].mean()
                    )
                    if inter_count > 0
                    else np.nan
                ),
                "whole_physical_segment_pair_point_multiplicity_mean": float(
                    edge_table[
                        "point_pair_count_within_threshold"
                    ].mean()
                ),
            }
        )
    else:
        metrics.update(
            {
                "whole_physical_inter_object_edge_count": 0,
                "whole_physical_inter_object_edge_fraction": 0.0,
                "whole_physical_true_topology_edge_count": 0,
                "whole_physical_true_topology_edge_fraction": 0.0,
                "whole_physical_intra_nonlocal_edge_count": 0,
                "whole_physical_intra_nonlocal_edge_fraction": 0.0,
                "whole_physical_mean_minimum_segment_distance": np.nan,
                "whole_physical_inter_object_mean_minimum_segment_distance": (
                    np.nan
                ),
                "whole_physical_segment_pair_point_multiplicity_mean": np.nan,
            }
        )

    # Add node-level old-style centralities for graph export.
    output_nodes = node_table.copy()
    output_nodes["whole_physical_degree"] = (
        output_nodes["physical_edge_id"].map(degree_dict).astype(float)
    )
    output_nodes["whole_physical_closeness"] = (
        output_nodes["physical_edge_id"].map(closeness_dict)
    )
    output_nodes["whole_physical_betweenness"] = (
        output_nodes["physical_edge_id"].map(betweenness_dict)
    )
    output_nodes["whole_physical_core_number"] = (
        output_nodes["physical_edge_id"].map(core_dict)
    )

    return output_nodes, graph, metrics


def calculate_whole_physical_social_bridge_metrics(
    whole_physical_edge_table,
    social_edge_table,
    combined_node_table,
):
    """
    Bridge the old branch-level whole-cell graph back to the social layer.

    For each social edge A-B:
      multiplicity = number of physical branch pairs between A and B that are
                     within whole_physical_distance_threshold.
      pair_density = multiplicity / (n_physical_A * n_physical_B).

    This distinguishes one incidental object-object proximity from a broad
    multi-branch interaction interface.
    """
    annotated_social_edges = social_edge_table.copy()

    if len(annotated_social_edges) == 0:
        empty_metrics = {
            "cross_social_edge_segment_proximity_multiplicity_mean": np.nan,
            "cross_social_edge_segment_proximity_multiplicity_median": np.nan,
            "cross_social_edge_segment_proximity_multiplicity_max": np.nan,
            "cross_social_edge_segment_proximity_multiplicity_gini": np.nan,
            "cross_social_edge_segment_proximity_positive_fraction": np.nan,
            "cross_social_edge_segment_pair_density_mean": np.nan,
            "cross_social_edge_segment_pair_density_median": np.nan,
            "cross_social_weight_segment_multiplicity_spearman": np.nan,
            "cross_social_weight_segment_pair_density_spearman": np.nan,
        }
        return annotated_social_edges, pd.DataFrame(), empty_metrics

    branch_count_by_social = dict(
        zip(
            combined_node_table["social_instance_id"].astype(int),
            combined_node_table["physical_instance_count"].astype(float),
        )
    )

    inter_edges = whole_physical_edge_table[
        whole_physical_edge_table["source_social_id"]
        != whole_physical_edge_table["target_social_id"]
    ].copy()

    pair_records = []
    if len(inter_edges) > 0:
        inter_edges["social_a"] = np.minimum(
            inter_edges["source_social_id"].to_numpy(dtype=np.int64),
            inter_edges["target_social_id"].to_numpy(dtype=np.int64),
        )
        inter_edges["social_b"] = np.maximum(
            inter_edges["source_social_id"].to_numpy(dtype=np.int64),
            inter_edges["target_social_id"].to_numpy(dtype=np.int64),
        )

        for (social_a, social_b), group in inter_edges.groupby(
            ["social_a", "social_b"]
        ):
            possible_pairs = (
                branch_count_by_social.get(int(social_a), np.nan)
                * branch_count_by_social.get(int(social_b), np.nan)
            )
            multiplicity = int(len(group))
            pair_records.append(
                {
                    "social_a": int(social_a),
                    "social_b": int(social_b),
                    "segment_proximity_multiplicity": multiplicity,
                    "segment_pair_density": safe_divide(
                        multiplicity,
                        possible_pairs,
                    ),
                    "minimum_segment_distance": float(
                        group["minimum_segment_distance"].min()
                    ),
                    "mean_segment_distance": float(
                        group["minimum_segment_distance"].mean()
                    ),
                }
            )

    pair_table = pd.DataFrame.from_records(
        pair_records,
        columns=[
            "social_a",
            "social_b",
            "segment_proximity_multiplicity",
            "segment_pair_density",
            "minimum_segment_distance",
            "mean_segment_distance",
        ],
    )

    pair_lookup = {}
    for row in pair_table.itertuples(index=False):
        pair_lookup[(int(row.social_a), int(row.social_b))] = row

    multiplicities = []
    pair_densities = []
    minimum_distances = []

    for edge_index, row in annotated_social_edges.iterrows():
        source = int(row["source_social_id"])
        target = int(row["target_social_id"])
        pair = (min(source, target), max(source, target))
        pair_row = pair_lookup.get(pair)

        if pair_row is None:
            multiplicity = 0
            possible_pairs = (
                branch_count_by_social.get(source, np.nan)
                * branch_count_by_social.get(target, np.nan)
            )
            pair_density = (
                0.0 if np.isfinite(possible_pairs) and possible_pairs > 0
                else np.nan
            )
            minimum_distance = np.nan
        else:
            multiplicity = int(
                pair_row.segment_proximity_multiplicity
            )
            pair_density = float(pair_row.segment_pair_density)
            minimum_distance = float(
                pair_row.minimum_segment_distance
            )

        annotated_social_edges.loc[
            edge_index,
            "segment_proximity_multiplicity",
        ] = multiplicity
        annotated_social_edges.loc[
            edge_index,
            "segment_pair_density",
        ] = pair_density
        annotated_social_edges.loc[
            edge_index,
            "segment_minimum_skeleton_distance",
        ] = minimum_distance

        multiplicities.append(float(multiplicity))
        pair_densities.append(float(pair_density))
        minimum_distances.append(float(minimum_distance))

    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    pair_densities = np.asarray(pair_densities, dtype=np.float64)
    social_weights = annotated_social_edges[
        "social_weight"
    ].to_numpy(dtype=np.float64)

    finite_density = pair_densities[np.isfinite(pair_densities)]

    metrics = {
        "cross_social_edge_segment_proximity_multiplicity_mean": float(
            np.mean(multiplicities)
        ),
        "cross_social_edge_segment_proximity_multiplicity_median": float(
            np.median(multiplicities)
        ),
        "cross_social_edge_segment_proximity_multiplicity_max": float(
            np.max(multiplicities)
        ),
        "cross_social_edge_segment_proximity_multiplicity_gini": (
            _finite_gini(multiplicities)
        ),
        "cross_social_edge_segment_proximity_positive_fraction": float(
            np.mean(multiplicities > 0)
        ),
        "cross_social_edge_segment_pair_density_mean": (
            float(np.mean(finite_density))
            if len(finite_density) > 0
            else np.nan
        ),
        "cross_social_edge_segment_pair_density_median": (
            float(np.median(finite_density))
            if len(finite_density) > 0
            else np.nan
        ),
        "cross_social_weight_segment_multiplicity_spearman": (
            spearman_correlation(
                social_weights,
                multiplicities,
                minimum_n=4,
            )
        ),
        "cross_social_weight_segment_pair_density_spearman": (
            spearman_correlation(
                social_weights,
                pair_densities,
                minimum_n=4,
            )
        ),
    }

    return annotated_social_edges, pair_table, metrics


def select_compact_image_metrics(metrics):
    """
    v8 compact image-level schema.

    Four complementary network views are kept explicitly:

    1. PHYSICAL:
       true within-object topology + population composition.
    2. WHOLE-CELL PHYSICAL:
       old-style branch/segment proximity graph across the whole cell.
    3. SOCIAL:
       object-level spatial organization.
    4. CROSS-SCALE:
       which objects/branches are contacted, where contacts fall on topology,
       and how branch-level proximity aggregates into social edges.

    The FULL per-image CSV contains many additional old-style candidate metrics.
    This compact table keeps representative metrics for routine downstream use.
    """
    fixed_columns = [
        "image_name",
        "metrics_schema",
        "social_distance_threshold",
        "whole_physical_distance_threshold",
        "contact_region_radius",
        "low_local_efficiency_threshold",
        "hub_fraction",
        "size_tail_fraction",
        "coordinate_scale_x",
        "coordinate_scale_y",
        "coordinate_scale_z",

        # ============================================================
        # PHYSICAL — true within-object architecture
        # ============================================================
        "physical_instance_count",

        # Population composition.
        "physical_simple_object_fraction",
        "physical_simple_object_length_fraction",
        "physical_branched_object_fraction",
        "physical_cyclic_object_fraction",
        "physical_largest_object_length_fraction",
        "physical_large_object_length_fraction",

        # Internal branch topology.
        "instance_physical_local_efficiency_median",
        "instance_physical_betweenness_top10_share",
        "physical_lcc_fraction",
        "physical_modularity",

        # Large-object topology / size-topology coupling.
        "physical_large_object_junction_density_median",
        "physical_large_object_route_efficiency_median",
        "physical_large_small_junction_density_contrast",
        "physical_object_size_junction_density_spearman",
        "instance_physical_size_local_efficiency_contrast",

        # ============================================================
        # WHOLE-CELL PHYSICAL — old-style segment proximity network
        # ============================================================
        "whole_physical_node_count",
        "whole_physical_edge_count",
        "whole_physical_density",
        "whole_physical_average_clustering",
        "whole_physical_transitivity",
        "whole_physical_global_efficiency",
        "whole_physical_local_efficiency",
        "whole_physical_modularity",
        "whole_physical_largest_component_fraction",
        "whole_physical_fragmentation_index",
        "whole_physical_degree_gini",
        "whole_physical_betweenness_gini",
        "whole_physical_max_core_number",
        "whole_physical_inter_object_edge_fraction",
        "whole_physical_intra_nonlocal_edge_fraction",

        # ============================================================
        # SOCIAL — whole mitochondrial-object spatial organization
        # ============================================================
        "social_global_efficiency",
        "social_modularity",
        "social_degree_gini",

        # ============================================================
        # CROSS — which social object is contacted?
        # ============================================================
        "cross_contact_target_object_size_rank_mean",
        "cross_contact_target_object_complexity_rank_mean",
        "cross_contact_target_large_object_enrichment",
        "cross_contact_target_complex_object_enrichment",
        "cross_contact_target_simple_object_enrichment",
        "cross_social_edge_object_size_assortativity",

        # CROSS — which physical branch is contacted?
        "cross_contact_target_length_rank_mean",
        "cross_contact_target_local_efficiency_rank_mean",
        "cross_contact_target_betweenness_rank_mean",
        "cross_contact_target_terminal_branch_enrichment",
        "cross_contact_physical_coverage_fraction_mean",

        # CROSS — where on the physical topology is contacted?
        "cross_contact_topology_junction_fraction",
        "cross_contact_topology_terminal_fraction",
        "cross_contact_topology_shaft_fraction",
        "cross_contact_topology_junction_enrichment",
        "cross_contact_topology_terminal_enrichment",
        "cross_contact_topology_shaft_enrichment",

        # CROSS — how branch-level whole-cell proximity aggregates to social.
        "cross_social_edge_segment_proximity_multiplicity_mean",
        "cross_social_edge_segment_proximity_multiplicity_median",
        "cross_social_edge_segment_proximity_multiplicity_gini",
        "cross_social_edge_segment_pair_density_mean",
        "cross_social_weight_segment_multiplicity_spearman",

        # CROSS — whole-layer coupling.
        "cross_route_efficiency_social_degree_partial_rho",
        "cross_route_efficiency_social_edge_assortativity",
        "cross_hub_overlap_fraction",
        "hierarchy_coupling_relative_gain_auc",
    ]

    return {
        column: metrics.get(column, np.nan)
        for column in fixed_columns
    }

def filter_low_information_metric_columns(
    summary_table,
    minimum_unique_values=6,
    maximum_dominant_fraction=0.70,
    maximum_missing_fraction=0.20,
):
    """在完整批量上删除恒定、强离散或缺失过多的分析指标。"""
    protected_columns = {
        "image_name",
        "metrics_schema",
        "social_distance_threshold",
        "whole_physical_distance_threshold",
        "contact_region_radius",
        "low_local_efficiency_threshold",
        "hub_fraction",
        "size_tail_fraction",
        "coordinate_scale_x",
        "coordinate_scale_y",
        "coordinate_scale_z",
        "physical_instance_count",
    }
    drop_reasons = {}
    for column in summary_table.columns:
        if column in protected_columns:
            continue
        values = pd.to_numeric(summary_table[column], errors="coerce")
        finite = values[np.isfinite(values)]
        missing_fraction = 1.0 - len(finite) / len(summary_table)
        if missing_fraction > maximum_missing_fraction:
            drop_reasons[column] = (
                f"missing_fraction={missing_fraction:.3f}"
            )
            continue
        if len(finite) == 0:
            drop_reasons[column] = "no_finite_values"
            continue
        value_range = float(np.max(finite) - np.min(finite))
        if value_range <= 1e-10:
            drop_reasons[column] = "near_constant"
            continue
        unique_count = int(finite.nunique())
        if unique_count < minimum_unique_values:
            drop_reasons[column] = f"unique_values={unique_count}"
            continue
        dominant_fraction = float(
            finite.value_counts(normalize=True, dropna=True).iloc[0]
        )
        if dominant_fraction >= maximum_dominant_fraction:
            drop_reasons[column] = (
                f"dominant_fraction={dominant_fraction:.3f}"
            )

    filtered = summary_table.drop(columns=list(drop_reasons)).copy()
    return filtered, drop_reasons


def filter_redundant_metric_columns(
    summary_table,
    maximum_absolute_spearman=0.85,
    minimum_pairwise_observations=30,
):
    """
    按表中预设的科学优先级保留指标，删除近似单调重复项。

    select_compact_image_metrics中的列顺序就是优先级：先保留解释更直接、
    先验定义更清楚的指标；后续指标若与已保留指标|Spearman rho|>=阈值，
    则删除。该步骤只使用全部图片的结构相关性，不使用group label或p值，
    因而不会按显著性挑指标。
    """
    protected_columns = {
        "image_name",
        "metrics_schema",
        "social_distance_threshold",
        "whole_physical_distance_threshold",
        "contact_region_radius",
        "low_local_efficiency_threshold",
        "hub_fraction",
        "size_tail_fraction",
        "coordinate_scale_x",
        "coordinate_scale_y",
        "coordinate_scale_z",
        "physical_instance_count",
    }
    kept_metrics = []
    drop_reasons = {}
    for column in summary_table.columns:
        if column in protected_columns:
            continue
        candidate = pd.to_numeric(summary_table[column], errors="coerce")
        redundant_with = None
        redundant_rho = np.nan
        for kept_column in kept_metrics:
            kept = pd.to_numeric(
                summary_table[kept_column],
                errors="coerce",
            )
            valid = np.isfinite(candidate) & np.isfinite(kept)
            if int(np.sum(valid)) < minimum_pairwise_observations:
                continue
            rho = spearman_correlation(
                candidate[valid],
                kept[valid],
                minimum_n=minimum_pairwise_observations,
            )
            if (
                np.isfinite(rho)
                and abs(rho) >= maximum_absolute_spearman
            ):
                redundant_with = kept_column
                redundant_rho = rho
                break
        if redundant_with is None:
            kept_metrics.append(column)
        else:
            drop_reasons[column] = (
                f"redundant_with={redundant_with}; "
                f"spearman={redundant_rho:.3f}"
            )
    filtered = summary_table.drop(columns=list(drop_reasons)).copy()
    return filtered, drop_reasons



def calculate_cross_object_targeting_metrics(
    combined_node_table,
    social_edge_table,
    size_tail_fraction=0.25,
):
    """
    回答 social network “连的是哪种 mitochondrial object”。

    每条 social edge 贡献两个 endpoint。对所有 social objects：
      - size = physical_total_skeleton_length
      - complexity = 已构建的 physical_complexity
      - simple = 单一、无 junction、无 cycle 的 object

    rank 在同一张图片内部定义为 [0, 1]。
    enrichment = 实际被 social edges 命中的 endpoint fraction /
                 该类型 object 在全体 objects 中的 baseline fraction。
    """
    table = combined_node_table.copy()

    empty = {
        "cross_contact_target_object_size_rank_mean": np.nan,
        "cross_contact_target_object_complexity_rank_mean": np.nan,
        "cross_contact_target_large_object_enrichment": np.nan,
        "cross_contact_target_complex_object_enrichment": np.nan,
        "cross_contact_target_simple_object_enrichment": np.nan,
        "cross_social_edge_object_size_assortativity": np.nan,
    }

    if len(table) == 0 or len(social_edge_table) == 0:
        return empty

    size = table[
        "physical_total_skeleton_length"
    ].to_numpy(dtype=float)

    complexity = table[
        "physical_complexity"
    ].to_numpy(dtype=float)

    instance_count = table[
        "physical_instance_count"
    ].to_numpy(dtype=float)

    junction_count = table[
        "physical_junction_count"
    ].to_numpy(dtype=float)

    cycle_rank = table[
        "physical_cycle_rank"
    ].to_numpy(dtype=float)

    def rank01(values):
        values = np.asarray(values, dtype=float)
        result = np.full(len(values), np.nan, dtype=float)
        valid = np.isfinite(values)
        n = int(np.sum(valid))
        if n == 1:
            result[valid] = 0.5
        elif n >= 2:
            ranks = rankdata(values[valid], method="average")
            result[valid] = (ranks - 1.0) / (n - 1.0)
        return result

    size_rank = rank01(size)
    complexity_rank = rank01(complexity)

    large_mask, _ = ranked_tail_masks(
        size,
        fraction=size_tail_fraction,
        descending=True,
    )
    complex_mask, _ = ranked_tail_masks(
        complexity,
        fraction=size_tail_fraction,
        descending=True,
    )

    simple_mask = (
        np.isfinite(instance_count)
        & np.isfinite(junction_count)
        & np.isfinite(cycle_rank)
        & (instance_count <= 1)
        & (junction_count <= 0)
        & (cycle_rank <= 0)
    )

    ids = table["social_instance_id"].astype(int).to_numpy()
    position = {
        int(social_id): row
        for row, social_id in enumerate(ids)
    }

    endpoint_ids = []
    for row in social_edge_table.itertuples(index=False):
        endpoint_ids.append(int(row.source_social_id))
        endpoint_ids.append(int(row.target_social_id))

    endpoint_rows = np.asarray(
        [
            position[social_id]
            for social_id in endpoint_ids
            if social_id in position
        ],
        dtype=int,
    )

    if len(endpoint_rows) == 0:
        return empty

    baseline_large = float(np.mean(large_mask))
    baseline_complex = float(np.mean(complex_mask))
    baseline_simple = float(np.mean(simple_mask))

    observed_large = float(np.mean(large_mask[endpoint_rows]))
    observed_complex = float(np.mean(complex_mask[endpoint_rows]))
    observed_simple = float(np.mean(simple_mask[endpoint_rows]))

    return {
        "cross_contact_target_object_size_rank_mean": float(
            np.nanmean(size_rank[endpoint_rows])
        ),
        "cross_contact_target_object_complexity_rank_mean": float(
            np.nanmean(complexity_rank[endpoint_rows])
        ),
        "cross_contact_target_large_object_enrichment": safe_divide(
            observed_large,
            baseline_large,
        ),
        "cross_contact_target_complex_object_enrichment": safe_divide(
            observed_complex,
            baseline_complex,
        ),
        "cross_contact_target_simple_object_enrichment": safe_divide(
            observed_simple,
            baseline_simple,
        ),
        "cross_social_edge_object_size_assortativity": (
            edge_attribute_assortativity(
                table,
                social_edge_table,
                "physical_total_skeleton_length",
            )
        ),
    }


def calculate_cross_layer_metrics(
    combined_node_table,
    social_edge_table,
    length_adjacency,
    scaled_skeleton_xyz,
    skeleton_social_ids,
    omega_values,
    maximum_transport_nodes,
    minimum_cross_nodes,
    size_tail_fraction,
    random_seed,
):
    combined_node_table = combined_node_table.copy()
    combined_node_table["physical_complexity"] = build_physical_complexity(
        combined_node_table
    )
    covariates = np.column_stack(
        [
            np.log1p(combined_node_table["pcd_surface_point_count"].to_numpy(dtype=float)),
            np.log1p(combined_node_table["physical_total_skeleton_length"].to_numpy(dtype=float)),
        ]
    )
    complexity_strength_rho, complexity_n = partial_spearman(
        combined_node_table["physical_complexity"],
        combined_node_table["social_strength"],
        covariates,
        minimum_n=minimum_cross_nodes,
    )
    efficiency_betweenness_rho, efficiency_n = partial_spearman(
        combined_node_table["physical_route_efficiency"],
        combined_node_table["social_betweenness"],
        covariates,
        minimum_n=minimum_cross_nodes,
    )
    branch_count_degree_rho, branch_count_degree_n = partial_spearman(
        combined_node_table["physical_instance_count"],
        combined_node_table["social_degree"],
        covariates,
        minimum_n=minimum_cross_nodes,
    )
    route_efficiency_degree_rho, route_efficiency_degree_n = partial_spearman(
        combined_node_table["physical_route_efficiency"],
        combined_node_table["social_degree"],
        covariates,
        minimum_n=minimum_cross_nodes,
    )
    hub_jaccard_value, hub_overlap_fraction = hub_jaccard(
        combined_node_table,
        fraction=0.25,
    )
    metrics = {
        "cross_complexity_strength_partial_rho": complexity_strength_rho,
        "cross_complexity_strength_n": complexity_n,
        "cross_efficiency_betweenness_partial_rho": efficiency_betweenness_rho,
        "cross_efficiency_betweenness_n": efficiency_n,
        "cross_physical_complexity_assortativity": (
            edge_attribute_assortativity(
                combined_node_table,
                social_edge_table,
                "physical_complexity",
            )
        ),
        "cross_hub_jaccard": hub_jaccard_value,
        "cross_hub_overlap_fraction": hub_overlap_fraction,
        "cross_branch_count_social_degree_partial_rho": (
            branch_count_degree_rho
        ),
        "cross_branch_count_social_degree_n": branch_count_degree_n,
        "cross_route_efficiency_social_degree_partial_rho": (
            route_efficiency_degree_rho
        ),
        "cross_route_efficiency_social_degree_n": route_efficiency_degree_n,
        "cross_route_efficiency_social_edge_assortativity": (
            edge_attribute_assortativity(
                combined_node_table,
                social_edge_table,
                "physical_route_efficiency",
            )
        ),
    }
    metrics.update(
        calculate_cross_object_targeting_metrics(
            combined_node_table=combined_node_table,
            social_edge_table=social_edge_table,
            size_tail_fraction=size_tail_fraction,
        )
    )
    metrics.update(
        coupled_transport_metrics(
            length_adjacency=length_adjacency,
            scaled_skeleton_xyz=scaled_skeleton_xyz,
            skeleton_social_ids=skeleton_social_ids,
            social_edge_table=social_edge_table,
            omega_values=omega_values,
            maximum_sample_nodes=maximum_transport_nodes,
            random_seed=random_seed,
        )
    )
    return combined_node_table, metrics


def add_dataset_level_compensation(summary_table):
    """
    使用当前批量数据集标准化得到探索性compensation。

    正式组间分析如需以Control为reference，应在统计脚本中重新计算。
    """
    summary_table = summary_table.copy()
    physical_raw = np.nanmean(
        np.column_stack(
            [
                summary_table["physical_largest_component_length_fraction"],
                summary_table["physical_route_efficiency"],
            ]
        ),
        axis=1,
    )
    social_raw = np.nanmean(
        np.column_stack(
            [
                summary_table["social_clustering"],
                summary_table["social_lcc_fraction"],
                summary_table["social_global_efficiency"],
            ]
        ),
        axis=1,
    )
    physical_z = zscore_finite(physical_raw)
    social_z = zscore_finite(social_raw)
    summary_table["physical_integration_raw"] = physical_raw
    summary_table["social_cohesion_raw"] = social_raw
    summary_table["physical_integration_dataset_z"] = physical_z
    summary_table["social_cohesion_dataset_z"] = social_z
    summary_table["cross_compensation_index"] = social_z - physical_z
    summary_table["cross_total_integration"] = (social_z + physical_z) / 2.0
    return summary_table


def build_long_format_instance_table(
    image_name,
    physical_node_table,
    social_node_table,
):
    """把physical和social instance属性合并成每张图片一个long-format表。"""
    physical = physical_node_table.copy()
    physical.insert(0, "image_name", image_name)
    physical.insert(1, "layer_type", "physical")
    physical.insert(
        2,
        "instance_id",
        physical["physical_edge_id"].to_numpy(dtype=np.int64),
    )

    social = social_node_table.copy()
    social.insert(0, "image_name", image_name)
    social.insert(1, "layer_type", "social")
    social.insert(
        2,
        "instance_id",
        social["social_instance_id"].to_numpy(dtype=np.int64),
    )

    instance_table = pd.concat(
        [physical, social],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    first_columns = [
        "image_name",
        "layer_type",
        "instance_id",
        "social_instance_id",
    ]
    remaining_columns = [
        column
        for column in instance_table.columns
        if column not in first_columns
    ]
    return instance_table[first_columns + remaining_columns]


def save_hierarchical_graph_pickle(
    save_path,
    image_name,
    physical_node_table,
    physical_network_edge_table,
    physical_weighted_adjacency,
    whole_physical_node_table,
    whole_physical_edge_table,
    whole_physical_graph,
    social_pair_segment_table,
    social_node_table,
    social_edge_table,
    social_weighted_adjacency,
    contact_event_table,
    skeleton_xyz,
    skeleton_social_ids,
    skeleton_length_adjacency,
    coordinate_scale_xyz,
    skeleton_neighbor_radius,
    social_distance_threshold,
    whole_physical_distance_threshold,
    contact_region_radius,
    hub_fraction,
    omega_values,
):
    """保存四层 network 与 cross-scale mapping 的完整可重载对象。"""
    membership_table = physical_node_table[
        ["physical_edge_id", "social_instance_id"]
    ].copy()

    # Save whole-cell physical NetworkX graph as node/edge tables plus
    # adjacency-friendly metadata. NetworkX itself is also picklable, but the
    # explicit tables make later inspection and cross-language reuse easier.
    graph_bundle = {
        "format_version": (
            "mito_hierarchical_network_v8_four_layer_whole_physical"
        ),
        "image_name": image_name,
        "parameters": {
            "coordinate_scale_xyz": tuple(
                float(value) for value in coordinate_scale_xyz
            ),
            "skeleton_neighbor_radius": float(skeleton_neighbor_radius),
            "social_distance_threshold": float(social_distance_threshold),
            "whole_physical_distance_threshold": float(
                whole_physical_distance_threshold
            ),
            "contact_region_radius": float(contact_region_radius),
            "contact_hub_fraction": float(hub_fraction),
            "omega_values": tuple(float(value) for value in omega_values),
        },

        # Layer 1: true physical topology inside connected mitochondrial objects.
        "physical_graph": {
            "node_definition": "one physical skeleton edge/branch instance",
            "edge_definition": "two physical branches share a true junction",
            "nodes": physical_node_table,
            "edges": physical_network_edge_table,
            "weighted_adjacency": csr_matrix(physical_weighted_adjacency),
        },

        # Layer 2: old-style whole-cell branch proximity network.
        "whole_cell_physical_proximity_graph": {
            "node_definition": "one physical skeleton edge/branch instance",
            "edge_definition": (
                "minimum skeleton-point distance between two physical "
                "branches <= whole_physical_distance_threshold, regardless "
                "of social-object membership"
            ),
            "nodes": whole_physical_node_table,
            "edges": whole_physical_edge_table,
            "networkx_graph": whole_physical_graph,
        },

        # Layer 3: object-level social network.
        "social_graph": {
            "node_definition": "one connected social mitochondrial object",
            "edge_definition": (
                "minimum PCD surface distance <= social_distance_threshold"
            ),
            "nodes": social_node_table,
            "edges": social_edge_table,
            "weighted_adjacency": csr_matrix(social_weighted_adjacency),
        },

        # Cross-scale mappings.
        "cross_layer_membership": membership_table,
        "cross_layer_contact_targeting": {
            "definition": (
                "each social-edge endpoint mapped to the nearest physical "
                "branch and classified as junction/terminal/shaft within "
                "the same social object"
            ),
            "events": contact_event_table,
        },
        "cross_layer_segment_to_social": {
            "definition": (
                "inter-object whole-cell physical branch-proximity edges "
                "aggregated by source/target social-object pair"
            ),
            "social_pair_segment_table": social_pair_segment_table,
        },

        "coupled_transport_support": {
            "skeleton_xyz": np.asarray(skeleton_xyz, dtype=np.float64),
            "skeleton_social_ids": np.asarray(
                skeleton_social_ids,
                dtype=np.int64,
            ),
            "skeleton_length_adjacency": skeleton_length_adjacency.tocsr(),
        },
    }

    with open(save_path, "wb") as file_handle:
        pickle.dump(
            graph_bundle,
            file_handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

def process_one_image(
    stem,
    input_path,
    output_path,
    coordinate_scale_xyz,
    skeleton_neighbor_radius,
    social_distance_threshold,
    whole_physical_distance_threshold,
    contact_region_radius,
    omega_values,
    maximum_efficiency_nodes,
    maximum_resistance_nodes,
    maximum_transport_nodes,
    minimum_cross_nodes,
    low_local_efficiency_threshold,
    hub_fraction,
    size_tail_fraction,
    whole_physical_centrality_sample_nodes,
    whole_physical_spectral_entropy_max_nodes,
    whole_physical_weighted_efficiency_max_nodes,
    workers,
    random_seed,
):
    skeleton_edges_path = input_path / f"{stem}_skeleton_edges.txt"
    skeleton_social_path = input_path / f"{stem}_skeleton_instance_social.txt"
    pcd_social_path = input_path / f"{stem}_pcd_instance_social.txt"

    for required_path in [
        skeleton_edges_path,
        skeleton_social_path,
        pcd_social_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    skeleton_edges = load_txt_array(
        skeleton_edges_path,
        minimum_columns=7,
        label="SKELETON EDGES",
    )
    skeleton_social = load_txt_array(
        skeleton_social_path,
        minimum_columns=7,
        label="SOCIAL SKELETON",
    )
    pcd_social = load_txt_array(
        pcd_social_path,
        minimum_columns=10,
        label="SOCIAL PCD",
    )

    if len(skeleton_edges) != len(skeleton_social):
        raise ValueError(
            "skeleton_edges and skeleton_instance_social row counts differ."
        )

    coordinate_error = np.max(
        np.linalg.norm(
            skeleton_edges[:, :3] - skeleton_social[:, :3],
            axis=1,
        )
    )
    if coordinate_error > 1e-5:
        raise ValueError(
            "Skeleton rows are not aligned; maximum XYZ difference is "
            f"{coordinate_error:.6g}."
        )

    skeleton_social_ids = np.rint(
        skeleton_social[:, 6]
    ).astype(np.int64)
    edge_ids = np.rint(
        skeleton_edges[:, 6]
    ).astype(np.int64)

    if np.any(skeleton_social_ids <= 0):
        raise ValueError(
            "Every skeleton point must have social_instance_id > 0."
        )

    coordinate_scale_xyz = np.asarray(
        coordinate_scale_xyz,
        dtype=np.float64,
    )
    scaled_skeleton_xyz = (
        skeleton_edges[:, :3]
        * coordinate_scale_xyz[None, :]
    )

    # ========================================================================
    # A. TRUE PHYSICAL TOPOLOGY
    # ========================================================================
    binary_adjacency, length_adjacency, kept_pairs = (
        build_pruned_skeleton_adjacency(
            xyz=scaled_skeleton_xyz,
            neighbor_radius=(
                skeleton_neighbor_radius
                * float(np.max(coordinate_scale_xyz))
            ),
            prune_shortcut_edges=True,
        )
    )

    if len(kept_pairs) > 0:
        cross_social = (
            skeleton_social_ids[kept_pairs[:, 0]]
            != skeleton_social_ids[kept_pairs[:, 1]]
        )
        if np.any(cross_social):
            raise ValueError(
                f"Found {int(np.sum(cross_social))} skeleton adjacency pairs "
                "crossing different social IDs."
            )

    physical_edge_table = build_physical_edge_table(
        skeleton_edges=skeleton_edges,
        skeleton_social_ids=skeleton_social_ids,
        coordinate_scale_xyz=coordinate_scale_xyz,
        neighbor_radius=skeleton_neighbor_radius,
    )

    (
        physical_node_table,
        physical_network_edge_table,
        physical_weighted_adjacency,
        physical_network_metrics,
    ) = build_physical_instance_network(
        physical_node_table=physical_edge_table,
        skeleton_binary_adjacency=binary_adjacency,
        skeleton_edge_ids=edge_ids,
        skeleton_social_ids=skeleton_social_ids,
    )

    # ========================================================================
    # B. WHOLE-CELL PHYSICAL-SEGMENT PROXIMITY NETWORK
    #    This restores the user's previous whole-cell physical graph:
    #    one physical branch = one node; branch min distance <= threshold=edge.
    # ========================================================================
    (
        whole_physical_node_table,
        whole_physical_edge_table,
    ) = build_whole_cell_physical_proximity_graph(
        physical_node_table=physical_node_table,
        physical_network_edge_table=physical_network_edge_table,
        scaled_skeleton_xyz=scaled_skeleton_xyz,
        skeleton_edge_ids=edge_ids,
        distance_threshold=whole_physical_distance_threshold,
    )

    (
        whole_physical_node_table,
        whole_physical_graph,
        whole_physical_metrics,
    ) = compute_whole_cell_physical_graph_metrics(
        node_table=whole_physical_node_table,
        edge_table=whole_physical_edge_table,
        distance_threshold=whole_physical_distance_threshold,
        centrality_sample_nodes=whole_physical_centrality_sample_nodes,
        spatial_bins=4,
        spectral_entropy_max_nodes=(
            whole_physical_spectral_entropy_max_nodes
        ),
        weighted_efficiency_max_nodes=(
            whole_physical_weighted_efficiency_max_nodes
        ),
        random_seed=random_seed,
    )

    # ========================================================================
    # C. PHYSICAL OBJECT-LEVEL TOPOLOGY / COMPOSITION
    # ========================================================================
    physical_object_table = build_physical_object_table(
        binary_adjacency=binary_adjacency,
        length_adjacency=length_adjacency,
        scaled_xyz=scaled_skeleton_xyz,
        edge_ids=edge_ids,
        social_ids=skeleton_social_ids,
        physical_edge_table=physical_edge_table,
        maximum_efficiency_nodes=maximum_efficiency_nodes,
        maximum_resistance_nodes=maximum_resistance_nodes,
        random_seed=random_seed,
    )

    physical_centrality_by_social = aggregate_physical_nodes_by_social(
        physical_node_table
    )
    physical_object_table = physical_object_table.merge(
        physical_centrality_by_social,
        on="social_instance_id",
        how="left",
        validate="one_to_one",
    )

    # ========================================================================
    # D. SOCIAL NETWORK — one mitochondrial object = one node
    # ========================================================================
    social_node_table, social_edge_table = build_social_tables(
        pcd_social=pcd_social,
        coordinate_scale_xyz=coordinate_scale_xyz,
        social_distance_threshold=social_distance_threshold,
        workers=workers,
    )

    (
        social_node_table,
        social_metrics,
        social_weighted_adjacency,
    ) = social_metrics_and_node_centralities(
        social_node_table,
        social_edge_table,
    )

    # ========================================================================
    # E. CROSS-SCALE CONTACT TARGETING
    # ========================================================================
    (
        physical_node_table,
        social_edge_table,
        contact_event_table,
        contact_targeting_metrics,
    ) = annotate_cross_layer_contact_targets(
        physical_node_table=physical_node_table,
        social_edge_table=social_edge_table,
        scaled_skeleton_xyz=scaled_skeleton_xyz,
        skeleton_social_ids=skeleton_social_ids,
        skeleton_edge_ids=edge_ids,
        skeleton_binary_adjacency=binary_adjacency,
        contact_region_radius=contact_region_radius,
        hub_fraction=hub_fraction,
    )

    skeleton_ids = set(
        physical_object_table["social_instance_id"].astype(int)
    )
    pcd_ids = set(
        social_node_table["social_instance_id"].astype(int)
    )
    if skeleton_ids != pcd_ids:
        raise ValueError(
            "Skeleton and PCD social IDs differ. "
            f"Only skeleton={sorted(skeleton_ids - pcd_ids)}, "
            f"only PCD={sorted(pcd_ids - skeleton_ids)}"
        )

    combined_node_table = social_node_table.merge(
        physical_object_table,
        on="social_instance_id",
        how="inner",
        validate="one_to_one",
    )

    combined_node_table, cross_metrics = calculate_cross_layer_metrics(
        combined_node_table=combined_node_table,
        social_edge_table=social_edge_table,
        length_adjacency=length_adjacency,
        scaled_skeleton_xyz=scaled_skeleton_xyz,
        skeleton_social_ids=skeleton_social_ids,
        omega_values=omega_values,
        maximum_transport_nodes=maximum_transport_nodes,
        minimum_cross_nodes=minimum_cross_nodes,
        size_tail_fraction=size_tail_fraction,
        random_seed=random_seed,
    )
    cross_metrics.update(contact_targeting_metrics)

    # ------------------------------------------------------------------------
    # NEW: how the branch-level whole-cell graph aggregates into social edges.
    # ------------------------------------------------------------------------
    (
        social_edge_table,
        social_pair_segment_table,
        segment_social_cross_metrics,
    ) = calculate_whole_physical_social_bridge_metrics(
        whole_physical_edge_table=whole_physical_edge_table,
        social_edge_table=social_edge_table,
        combined_node_table=combined_node_table,
    )
    cross_metrics.update(segment_social_cross_metrics)

    # ========================================================================
    # F. IMAGE-LEVEL METRICS
    # ========================================================================
    physical_metrics = aggregate_physical_metrics(
        physical_object_table=physical_object_table,
        physical_edge_table=physical_edge_table,
    )
    physical_metrics.update(physical_network_metrics)
    physical_metrics.update(
        aggregate_physical_composition_topology_metrics(
            physical_object_table=physical_object_table,
            size_tail_fraction=size_tail_fraction,
        )
    )

    hierarchy_metrics = calculate_hierarchy_balance_metrics(
        physical_metrics=physical_metrics,
        social_metrics=social_metrics,
        cross_metrics=cross_metrics,
        combined_node_table=combined_node_table,
    )

    full_metrics = {
        "image_name": stem,
        "metrics_schema": (
            "concept_clean_v8_four_layer_whole_physical"
        ),
        "social_distance_threshold": float(
            social_distance_threshold
        ),
        "whole_physical_distance_threshold": float(
            whole_physical_distance_threshold
        ),
        "contact_region_radius": float(contact_region_radius),
        "low_local_efficiency_threshold": float(
            low_local_efficiency_threshold
        ),
        "hub_fraction": float(hub_fraction),
        "size_tail_fraction": float(size_tail_fraction),
        "coordinate_scale_x": float(coordinate_scale_xyz[0]),
        "coordinate_scale_y": float(coordinate_scale_xyz[1]),
        "coordinate_scale_z": float(coordinate_scale_xyz[2]),
    }

    full_metrics.update(physical_metrics)
    full_metrics.update(whole_physical_metrics)
    full_metrics.update(social_metrics)
    full_metrics.update(cross_metrics)
    full_metrics.update(hierarchy_metrics)
    full_metrics.update(
        aggregate_instance_resolved_metrics(
            physical_node_table=physical_node_table,
            social_node_table=combined_node_table,
            low_local_efficiency_threshold=(
                low_local_efficiency_threshold
            ),
            hub_fraction=hub_fraction,
            size_tail_fraction=size_tail_fraction,
        )
    )

    compact_metrics = select_compact_image_metrics(full_metrics)

    # ========================================================================
    # G. SAVE
    # ========================================================================
    output_path.mkdir(parents=True, exist_ok=True)

    instance_table = build_long_format_instance_table(
        image_name=stem,
        physical_node_table=physical_node_table,
        social_node_table=combined_node_table,
    )
    instance_table.to_csv(
        output_path / f"{stem}_instance_properties.csv",
        index=False,
    )

    # Whole-cell physical graph inspection tables.
    whole_physical_node_table.to_csv(
        output_path / f"{stem}_whole_physical_nodes.csv",
        index=False,
    )
    whole_physical_edge_table.to_csv(
        output_path / f"{stem}_whole_physical_proximity_edges.csv",
        index=False,
    )
    social_pair_segment_table.to_csv(
        output_path / f"{stem}_social_pair_segment_proximity.csv",
        index=False,
    )
    social_edge_table.to_csv(
        output_path / f"{stem}_social_edges_with_segment_multiplicity.csv",
        index=False,
    )

    save_hierarchical_graph_pickle(
        save_path=output_path / f"{stem}_hierarchical_network.pkl",
        image_name=stem,
        physical_node_table=physical_node_table,
        physical_network_edge_table=physical_network_edge_table,
        physical_weighted_adjacency=physical_weighted_adjacency,
        whole_physical_node_table=whole_physical_node_table,
        whole_physical_edge_table=whole_physical_edge_table,
        whole_physical_graph=whole_physical_graph,
        social_pair_segment_table=social_pair_segment_table,
        social_node_table=combined_node_table,
        social_edge_table=social_edge_table,
        social_weighted_adjacency=social_weighted_adjacency,
        contact_event_table=contact_event_table,
        skeleton_xyz=scaled_skeleton_xyz,
        skeleton_social_ids=skeleton_social_ids,
        skeleton_length_adjacency=length_adjacency,
        coordinate_scale_xyz=coordinate_scale_xyz,
        skeleton_neighbor_radius=skeleton_neighbor_radius,
        social_distance_threshold=social_distance_threshold,
        whole_physical_distance_threshold=(
            whole_physical_distance_threshold
        ),
        contact_region_radius=contact_region_radius,
        hub_fraction=hub_fraction,
        omega_values=omega_values,
    )

    # Compact table for routine downstream analysis.
    pd.DataFrame([compact_metrics]).to_csv(
        output_path / f"{stem}_hierarchical_network_metrics.csv",
        index=False,
    )

    # FULL candidate table: preserves the old single-network metrics and all
    # v7/v8 candidates. Use this to evaluate significance/robustness before
    # freezing the final publication metric set.
    pd.DataFrame([full_metrics]).to_csv(
        output_path / f"{stem}_hierarchical_network_metrics_FULL.csv",
        index=False,
    )

    return compact_metrics

def main():
    ###########################################################################
    # 运行参数：不使用 argparse，直接在这里修改
    ###########################################################################

    # ========================================================================
    # 8 个自有数据集
    # ========================================================================
    # file_name_list = [
    #     # ====================================================
    #     # MCF-7
    #     # ====================================================
    #     "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #
    #     # ====================================================
    #     # AC16
    #     # ====================================================
    #     "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #
    #     # ====================================================
    #     # MDA
    #     # ====================================================
    #     "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",
    #
    #     # ====================================================
    #     # Fibroblast
    #     # ====================================================
    #     "20260207_fibro_mito488_lyso640_nuc405_tif",
    #
    #     # ====================================================
    #     # H28
    #     # ====================================================
    #     "20250917_h28_mitotracker488_lysotracker640_nuc405",
    #
    #     # ====================================================
    #     # HeLa
    #     # ====================================================
    #     "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",
    #
    #     # ====================================================
    #     # U2OS
    #     # ====================================================
    #     "20250909_u2os_mitotracker488_lysotracker640_nuc405",
    #
    #     # ====================================================
    #     # MSTO
    #     # ====================================================
    #     "20250905_msto_mitotracekr488_lysotracker640_nuc405",
    # ]

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

    # # 所有数据集共同的根目录。
    # DATA_ROOT = Path(
    #     "D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis"
    # )

    # DATA_ROOT = Path(f"E:/data/20260811-YF_output/image_tif/single_cell_tifs")
    DATA_ROOT = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")
    # DATA_ROOT = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3")
    # DATA_ROOT = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")

    # None：每个数据集自动扫描所有 *_pcd_instance_social.txt。
    # 如果以后只想跑某个数据集中的指定图片，可以在这里写：
    # manual_image_stems_by_dataset = {
    #     "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif": [
    #         "image_001", "image_002"
    #     ],
    # }
    manual_image_stems_by_dataset = None

    # True：重新计算已有图片。
    # False：若该图片已经有 v8 PKL + compact/full metrics + instance properties，则直接读取。
    overwrite_existing = True

    # 坐标仍为 voxel 单位时保持 (1,1,1)。
    coordinate_scale_xyz = (1.0, 1.0, 1.0)

    # 与前面 skeleton edge 分割保持一致。
    skeleton_neighbor_radius = 1.8

    # SOCIAL 主阈值：两个 social instances 的 PCD 表面最近距离 <= 5 voxel 连边。
    social_distance_threshold = 5.0

    # WHOLE-CELL PHYSICAL 主阈值：
    # 每个 physical skeleton edge/branch 作为一个 node。
    # 任意两个 branch 的最小 skeleton-point distance <= 5 时连边。
    # 这就是恢复的“旧式整张细胞 physical network”。
    whole_physical_distance_threshold = 5.0

    # Cross-scale contact topology 的空间邻域半径（与坐标同单位）。
    # contact endpoint 距 junction/terminal skeleton <= 此值时，
    # 分别归为 junction-zone / terminal-zone；其余为 shaft-zone。
    contact_region_radius = 3.0

    # coupled transport 中 social virtual edge 的耦合强度。
    omega_values = (0.01, 0.03, 0.1, 0.3, 1.0)

    # 大图使用固定抽样，避免全点对最短路径占用过多内存。
    maximum_efficiency_nodes = 250
    maximum_resistance_nodes = 400
    maximum_transport_nodes = 300

    # WHOLE-CELL PHYSICAL old-style graph 的计算限制。
    # Betweenness 在大图上使用固定 k-node approximation。
    whole_physical_centrality_sample_nodes = 300
    # 谱熵和 weighted efficiency 超过该 node 数时跳过，避免动态数据过慢。
    whole_physical_spectral_entropy_max_nodes = 800
    whole_physical_weighted_efficiency_max_nodes = 800

    # 至少需要多少个 social objects 才计算跨层 partial correlation。
    minimum_cross_nodes = 6

    # instance-resolved 指标定义。
    low_local_efficiency_threshold = 0.25
    hub_fraction = 0.10
    size_tail_fraction = 0.25

    workers = -1
    random_seed = 42

    # ========================================================================
    # External-validation rule:
    # metric set is frozen by select_compact_image_metrics().
    # No 8-type-dependent filtering is performed below.
    # ========================================================================

    ###########################################################################
    # 批量处理
    ###########################################################################

    all_dataset_raw_tables = []
    all_dataset_full_tables = []
    dataset_output_paths = {}
    dataset_stats = []
    failure_records = []

    total_success_or_loaded = 0
    total_skipped_existing = 0

    print("=" * 100)
    print("Hierarchical mitochondrial network v8 — physical + whole-physical + social + cross")
    print(f"DATA_ROOT                : {DATA_ROOT}")
    print(f"Datasets selected        : {len(file_name_list)}")
    print(f"Overwrite existing       : {overwrite_existing}")
    print(f"Social threshold         : {social_distance_threshold} voxel")
    print(f"Whole physical threshold : {whole_physical_distance_threshold} voxel")
    print(f"Contact region radius    : {contact_region_radius}")
    print(f"Coordinate scale xyz     : {coordinate_scale_xyz}")
    print(f"Omega values             : {omega_values}")
    print("=" * 100)

    for dataset_index, file_name in enumerate(file_name_list, start=1):
        # BASE_DIR = DATA_ROOT / file_name / "mito"
        # BASE_DIR = DATA_ROOT / file_name
        BASE_DIR = DATA_ROOT

        INPUT_PATH = BASE_DIR / "processed_output"
        OUTPUT_PATH = INPUT_PATH
        dataset_output_paths[file_name] = OUTPUT_PATH

        print("\n" + "#" * 100)
        print(
            f"DATASET {dataset_index}/{len(file_name_list)}: {file_name}"
        )
        print("#" * 100)
        print(f"BASE_DIR    : {BASE_DIR}")
        print(f"INPUT_PATH  : {INPUT_PATH}")
        print(f"OUTPUT_PATH : {OUTPUT_PATH}")

        if not INPUT_PATH.exists():
            print(f"[DATASET SKIPPED] Input folder does not exist: {INPUT_PATH}")
            dataset_stats.append(
                {
                    "dataset_name": file_name,
                    "status": "missing_processed_output",
                    "selected_images": 0,
                    "successful_or_loaded": 0,
                    "skipped_existing": 0,
                    "failed_images": 0,
                }
            )
            continue

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------------
        # 图片列表
        # --------------------------------------------------------------------
        if manual_image_stems_by_dataset is None:
            manual_stems = None
        else:
            manual_stems = manual_image_stems_by_dataset.get(file_name)

        if manual_stems is None:
            suffix = "_pcd_instance_social.txt"
            image_stems = sorted(
                path.name[:-len(suffix)]
                for path in INPUT_PATH.glob(f"*{suffix}")
            )
        else:
            image_stems = list(manual_stems)

        if len(image_stems) == 0:
            print(
                f"[DATASET SKIPPED] No *_pcd_instance_social.txt files "
                f"found in {INPUT_PATH}"
            )
            dataset_stats.append(
                {
                    "dataset_name": file_name,
                    "status": "no_images",
                    "selected_images": 0,
                    "successful_or_loaded": 0,
                    "skipped_existing": 0,
                    "failed_images": 0,
                }
            )
            continue

        print(f"Images selected: {len(image_stems)}")

        dataset_metrics = []
        dataset_skipped = 0
        dataset_failed = 0

        for image_index, stem in enumerate(image_stems, start=1):
            summary_path = (
                OUTPUT_PATH / f"{stem}_hierarchical_network_metrics.csv"
            )
            instance_path = OUTPUT_PATH / f"{stem}_instance_properties.csv"
            graph_path = OUTPUT_PATH / f"{stem}_hierarchical_network.pkl"
            full_summary_path = (
                OUTPUT_PATH
                / f"{stem}_hierarchical_network_metrics_FULL.csv"
            )

            print("-" * 100)
            print(
                f"[{dataset_index}/{len(file_name_list)} datasets] "
                f"[{image_index}/{len(image_stems)} images] {stem}"
            )

            # ----------------------------------------------------------------
            # 已存在时直接读取，不重复计算
            # ----------------------------------------------------------------
            if (
                not overwrite_existing
                and summary_path.exists()
                and full_summary_path.exists()
                and instance_path.exists()
                and graph_path.exists()
            ):
                try:
                    existing = pd.read_csv(summary_path).iloc[0].to_dict()
                    if existing.get("metrics_schema") == "concept_clean_v8_four_layer_whole_physical":
                        print(
                            "[SKIPPED] Existing v8 PKL, full metrics, instance properties "
                            "and metrics found."
                        )
                        existing["dataset_name"] = file_name
                        dataset_metrics.append(existing)
                        dataset_skipped += 1
                        total_skipped_existing += 1
                        continue
                    print("[RERUN] Existing metrics use an older schema.")
                except Exception as error:
                    print(
                        "[RERUN] Existing summary could not be loaded: "
                        f"{type(error).__name__}: {error}"
                    )

            # ----------------------------------------------------------------
            # 正常计算
            # ----------------------------------------------------------------
            try:
                metrics = process_one_image(
                    stem=stem,
                    input_path=INPUT_PATH,
                    output_path=OUTPUT_PATH,
                    coordinate_scale_xyz=coordinate_scale_xyz,
                    skeleton_neighbor_radius=skeleton_neighbor_radius,
                    social_distance_threshold=social_distance_threshold,
                    whole_physical_distance_threshold=(
                        whole_physical_distance_threshold
                    ),
                    contact_region_radius=contact_region_radius,
                    omega_values=omega_values,
                    maximum_efficiency_nodes=maximum_efficiency_nodes,
                    maximum_resistance_nodes=maximum_resistance_nodes,
                    maximum_transport_nodes=maximum_transport_nodes,
                    minimum_cross_nodes=minimum_cross_nodes,
                    low_local_efficiency_threshold=(
                        low_local_efficiency_threshold
                    ),
                    hub_fraction=hub_fraction,
                    size_tail_fraction=size_tail_fraction,
                    whole_physical_centrality_sample_nodes=(
                        whole_physical_centrality_sample_nodes
                    ),
                    whole_physical_spectral_entropy_max_nodes=(
                        whole_physical_spectral_entropy_max_nodes
                    ),
                    whole_physical_weighted_efficiency_max_nodes=(
                        whole_physical_weighted_efficiency_max_nodes
                    ),
                    workers=workers,
                    random_seed=random_seed,
                )
                metrics["dataset_name"] = file_name
                dataset_metrics.append(metrics)
                print(
                    "[DONE] physical instances="
                    f"{metrics['physical_instance_count']}"
                )
            except Exception as error:
                dataset_failed += 1
                print(
                    f"[FAILED] {stem}: {type(error).__name__}: {error}"
                )
                failure_records.append(
                    {
                        "dataset_name": file_name,
                        "image_name": stem,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

        if len(dataset_metrics) == 0:
            print(
                f"[DATASET FAILED] No image was successfully analysed in "
                f"{file_name}."
            )
            dataset_stats.append(
                {
                    "dataset_name": file_name,
                    "status": "no_successful_images",
                    "selected_images": len(image_stems),
                    "successful_or_loaded": 0,
                    "skipped_existing": dataset_skipped,
                    "failed_images": dataset_failed,
                }
            )
            continue

        # 保存该数据集的完整 compact metrics（尚未做跨 8-type 的统一过滤）。
        dataset_raw = pd.DataFrame(dataset_metrics)
        first_columns = [
            column
            for column in ["dataset_name", "image_name", "metrics_schema"]
            if column in dataset_raw.columns
        ]
        remaining_columns = [
            column for column in dataset_raw.columns
            if column not in first_columns
        ]
        dataset_raw = dataset_raw[first_columns + remaining_columns]

        dataset_raw.to_csv(
            OUTPUT_PATH
            / "all_images_hierarchical_network_metrics_unfiltered.csv",
            index=False,
        )
        all_dataset_raw_tables.append(dataset_raw)

        # ---------------------------------------------------------------
        # FULL candidate metrics for this dataset.
        # This contains the complete old-style whole-physical metric family
        # plus all physical/social/cross candidates, before compact selection.
        # ---------------------------------------------------------------
        dataset_full_rows = []
        for stem in image_stems:
            full_path = (
                OUTPUT_PATH
                / f"{stem}_hierarchical_network_metrics_FULL.csv"
            )
            if not full_path.exists():
                continue
            try:
                full_row = pd.read_csv(full_path).iloc[0].to_dict()
                if (
                    full_row.get("metrics_schema")
                    != "concept_clean_v8_four_layer_whole_physical"
                ):
                    continue
                full_row["dataset_name"] = file_name
                dataset_full_rows.append(full_row)
            except Exception:
                continue

        if len(dataset_full_rows) > 0:
            dataset_full = pd.DataFrame(dataset_full_rows)
            first_full_columns = [
                column
                for column in [
                    "dataset_name",
                    "image_name",
                    "metrics_schema",
                ]
                if column in dataset_full.columns
            ]
            remaining_full_columns = [
                column
                for column in dataset_full.columns
                if column not in first_full_columns
            ]
            dataset_full = dataset_full[
                first_full_columns + remaining_full_columns
            ]
            dataset_full.to_csv(
                OUTPUT_PATH
                / "all_images_hierarchical_network_metrics_FULL.csv",
                index=False,
            )
            all_dataset_full_tables.append(dataset_full)

        total_success_or_loaded += len(dataset_metrics)
        dataset_stats.append(
            {
                "dataset_name": file_name,
                "status": "completed",
                "selected_images": len(image_stems),
                "successful_or_loaded": len(dataset_metrics),
                "skipped_existing": dataset_skipped,
                "failed_images": dataset_failed,
            }
        )

        print(
            f"[DATASET DONE] successful/loaded={len(dataset_metrics)}, "
            f"skipped_existing={dataset_skipped}, failed={dataset_failed}"
        )

    ###########################################################################
    # 8-type EXTERNAL VALIDATION:
    # 使用前面已经冻结的 select_compact_image_metrics() 列表。
    # 这里绝对不再根据这 8 个数据集做任何 feature/metric filtering。
    ###########################################################################
    if len(all_dataset_raw_tables) == 0:
        raise RuntimeError("No dataset produced any usable image metrics.")

    combined_raw = pd.concat(
        all_dataset_raw_tables,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    # select_compact_image_metrics() 就是正式冻结的 metric schema。
    fixed_metric_columns = list(select_compact_image_metrics({}).keys())

    # dataset_name 是 external cohort metadata，放第一列。
    expected_columns = ["dataset_name"] + fixed_metric_columns

    # 若某个旧的 per-image CSV 缺少固定指标，这里保留 NaN，而不是删列。
    # overwrite_existing=True 时，正常情况下所有固定列都会重新计算。
    for column in expected_columns:
        if column not in combined_raw.columns:
            combined_raw[column] = np.nan

    combined_fixed = combined_raw[expected_columns].copy()

    # 完整 fixed-metric 总表。
    combined_fixed_path = (
        DATA_ROOT / "all_types_hierarchical_network_metrics.csv"
    )
    combined_fixed.to_csv(combined_fixed_path, index=False)

    # 同时保留一个名称明确的副本，强调没有做 external-set filtering。
    combined_fixed_unfiltered_path = (
        DATA_ROOT
        / "all_types_hierarchical_network_metrics_FIXED_UNFILTERED.csv"
    )
    combined_fixed.to_csv(combined_fixed_unfiltered_path, index=False)

    # FULL four-layer candidate table. This is the table to inspect when
    # deciding which old/new whole-cell physical metrics are actually useful.
    combined_full_path = (
        DATA_ROOT / "all_types_hierarchical_network_metrics_FULL.csv"
    )
    if len(all_dataset_full_tables) > 0:
        combined_full = pd.concat(
            all_dataset_full_tables,
            axis=0,
            ignore_index=True,
            sort=False,
        )
        combined_full.to_csv(combined_full_path, index=False)
    else:
        combined_full = pd.DataFrame()

    # 保存 frozen schema，后续 Fig.2g 分析只允许使用这里列出的 metrics。
    fixed_schema_path = DATA_ROOT / "all_types_FIXED_metric_columns.csv"
    pd.DataFrame(
        {
            "column_order": np.arange(len(fixed_metric_columns)),
            "metric_column": fixed_metric_columns,
        }
    ).to_csv(fixed_schema_path, index=False)

    # 将同一套固定列写回各数据集。
    for file_name, output_path in dataset_output_paths.items():
        if not output_path.exists():
            continue

        dataset_fixed = combined_fixed[
            combined_fixed["dataset_name"] == file_name
        ].copy()

        if len(dataset_fixed) == 0:
            continue

        dataset_fixed_no_name = dataset_fixed.drop(
            columns=["dataset_name"]
        )
        dataset_fixed_no_name.to_csv(
            output_path / "all_images_hierarchical_network_metrics.csv",
            index=False,
        )

        # 额外保存一个明确的 fixed 文件，避免以后和旧 filtered 文件混淆。
        dataset_fixed_no_name.to_csv(
            output_path
            / "all_images_hierarchical_network_metrics_FIXED.csv",
            index=False,
        )

    ###########################################################################
    # 日志
    ###########################################################################
    dataset_stats_table = pd.DataFrame(dataset_stats)
    dataset_stats_path = (
        DATA_ROOT / "all_types_hierarchical_network_run_summary.csv"
    )
    dataset_stats_table.to_csv(dataset_stats_path, index=False)

    failure_path = (
        DATA_ROOT / "all_types_hierarchical_network_failures.csv"
    )
    pd.DataFrame(
        failure_records,
        columns=[
            "dataset_name",
            "image_name",
            "error_type",
            "error_message",
        ],
    ).to_csv(failure_path, index=False)

    # QC：检查 fixed metric 是否在最终 8-type 表中有整列缺失。
    qc_rows = []
    for column in fixed_metric_columns:
        if column in {
            "image_name",
            "metrics_schema",
        }:
            continue
        values = pd.to_numeric(
            combined_fixed[column],
            errors="coerce",
        )
        qc_rows.append(
            {
                "metric": column,
                "n_rows": int(len(combined_fixed)),
                "n_finite": int(np.isfinite(values).sum()),
                "missing_fraction": float(
                    1.0 - np.isfinite(values).sum() / len(combined_fixed)
                ),
                "n_unique_finite": int(
                    values[np.isfinite(values)].nunique()
                ),
            }
        )
    qc_path = DATA_ROOT / "all_types_FIXED_metrics_QC.csv"
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

    print("\n" + "=" * 100)
    print("V8 FOUR-LAYER HIERARCHICAL NETWORK ANALYSIS FINISHED")
    print("=" * 100)
    print(f"Successful/loaded images : {total_success_or_loaded}")
    print(f"Skipped existing images  : {total_skipped_existing}")
    print(f"Datasets with output     : {len(all_dataset_raw_tables)}")
    print(f"Rows in combined table   : {len(combined_fixed)}")
    print(f"Fixed columns            : {len(fixed_metric_columns)}")
    print("Validation-set filtering : NONE")
    print("\nMain fixed 8-type summary:")
    print(combined_fixed_path)
    print("\nFull candidate metric summary:")
    print(combined_full_path)
    print("\nFrozen compact metric schema:")
    print(fixed_schema_path)
    print("\nFixed-metric QC:")
    print(qc_path)
    print("\nRun summary:")
    print(dataset_stats_path)
    print("\nFailures:")
    print(failure_path)
    print("=" * 100)


if __name__ == "__main__":
    main()

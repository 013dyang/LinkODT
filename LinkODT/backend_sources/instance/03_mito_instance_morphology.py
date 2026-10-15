from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


INSTANCE_COLUMNS = [
    "image_name",
    "instance_id",
    "r",
    "g",
    "b",
    "voxel_count",
    "pcd_surface_point_count",
    "volume",
    "surface_area_voxel_faces",
    "surface_volume_ratio",
    "equivalent_sphere_diameter",
    "equivalent_sphere_radius",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "major_axis_length_pca",
    "middle_axis_length_pca",
    "minor_axis_length_pca",
    "mean_width_pca",
    "aspect_ratio_major_minor",
    "length_width_ratio",
    "elongation_major_middle",
    "flatness_middle_minor",
    "pca_eig_major",
    "pca_eig_middle",
    "pca_eig_minor",
    "pca_linearity",
    "pca_planarity",
    "pca_sphericity_like",
    "pca_anisotropy",
    "radius_of_gyration",
    "sphericity",
    "compactness_3d",
    "intensity_mean",
    "intensity_median",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "intensity_sum",
]


DATASET_SHORT_NAMES = {
    "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif": "MCF-7",
    "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif": "AC16",
    "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif": "MDA",
    "20260207_fibro_mito488_lyso640_nuc405_tif": "Fibroblast",
    "20250917_h28_mitotracker488_lysotracker640_nuc405": "H28",
    "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst": "HeLa",
    "20250909_u2os_mitotracker488_lysotracker640_nuc405": "U2OS",
    "20250905_msto_mitotracekr488_lysotracker640_nuc405": "MSTO",
}


def load_txt_array(path):
    """读取逗号分隔、无表头的数值txt。"""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def check_minimum_columns(label, data, minimum_columns):
    if data.ndim != 2 or data.shape[1] < minimum_columns:
        raise ValueError(
            f"{label} must contain at least {minimum_columns} columns, "
            f"but its shape is {data.shape}."
        )


def print_file_information(label, path, data, column_format):
    print(f"{label} path   : {path}")
    print(f"{label} shape  : {data.shape}")
    print(f"{label} format : {column_format}")


def safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return np.nan
    return float(numerator) / denominator


def rgb_to_key(rgb):
    """把0-255 RGB编码成一个整数，仅用于内部匹配。"""
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.int64)
    return rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]


def split_voxel_instances_by_rgb_and_connectivity(
    voxel_xyz,
    voxel_rgb,
    connect_radius=1.8,
):
    """
    使用RGB和空间连通性共同定义social instance。

    这样即使两个远处线粒体意外复用了相同RGB，也不会被合并为同一个
    morphology instance。
    """
    if connect_radius <= 0:
        raise ValueError("connect_radius must be positive.")

    point_count = len(voxel_xyz)
    if point_count == 0:
        raise ValueError("No voxel points were provided.")

    color_keys = rgb_to_key(voxel_rgb)
    tree = cKDTree(voxel_xyz)
    neighbor_pairs = tree.query_pairs(
        r=connect_radius,
        output_type="ndarray",
    )

    if len(neighbor_pairs) > 0:
        same_color = (
            color_keys[neighbor_pairs[:, 0]]
            == color_keys[neighbor_pairs[:, 1]]
        )
        same_color_pairs = neighbor_pairs[same_color]
    else:
        same_color_pairs = np.empty((0, 2), dtype=np.int64)

    if len(same_color_pairs) == 0:
        raw_components = np.arange(point_count, dtype=np.int64)
    else:
        graph_rows = np.concatenate(
            [same_color_pairs[:, 0], same_color_pairs[:, 1]]
        )
        graph_columns = np.concatenate(
            [same_color_pairs[:, 1], same_color_pairs[:, 0]]
        )
        graph = csr_matrix(
            (
                np.ones(len(graph_rows), dtype=np.uint8),
                (graph_rows, graph_columns),
            ),
            shape=(point_count, point_count),
        )
        _, raw_components = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

    # 按每个component首次出现的行号重新编号，保证结果稳定且从1开始。
    component_values, first_rows = np.unique(
        raw_components,
        return_index=True,
    )
    component_order = component_values[np.argsort(first_rows)]
    remap = np.empty(np.max(component_order) + 1, dtype=np.int64)
    remap[component_order] = np.arange(1, len(component_order) + 1)
    instance_ids = remap[raw_components]

    unique_rgb_count = len(np.unique(color_keys))
    repeated_color_component_count = len(component_order) - unique_rgb_count

    print("Social voxel instance reconstruction:")
    print(f"    Voxel points                 : {point_count}")
    print(f"    Connectivity radius          : {connect_radius}")
    print(f"    Neighbor pairs               : {len(neighbor_pairs)}")
    print(f"    Same-RGB neighbor pairs      : {len(same_color_pairs)}")
    print(f"    Unique RGB values            : {unique_rgb_count}")
    print(f"    Spatially connected instances: {len(component_order)}")
    print(
        "    Extra components from reused/disconnected RGB: "
        f"{repeated_color_component_count}"
    )

    return instance_ids


def map_pcd_to_voxel_instances_by_rgb(
    pcd_xyz,
    pcd_rgb,
    voxel_xyz,
    voxel_rgb,
    voxel_instance_ids,
    workers=-1,
):
    """
    每个PCD点只在相同RGB的voxel中寻找最近点，取得内部instance_id。

    instance_id只用于Morphology计算，不写回xyzrgb点云。
    """
    pcd_keys = rgb_to_key(pcd_rgb)
    voxel_keys = rgb_to_key(voxel_rgb)
    pcd_instance_ids = np.zeros(len(pcd_xyz), dtype=np.int64)
    matched_distances = np.full(len(pcd_xyz), np.nan, dtype=np.float64)
    missing_color_point_count = 0

    voxel_rows_by_color = {
        int(color_key): np.where(voxel_keys == color_key)[0]
        for color_key in np.unique(voxel_keys)
    }

    for color_key in np.unique(pcd_keys):
        pcd_rows = np.where(pcd_keys == color_key)[0]
        voxel_rows = voxel_rows_by_color.get(int(color_key))
        if voxel_rows is None or len(voxel_rows) == 0:
            missing_color_point_count += len(pcd_rows)
            continue

        one_color_tree = cKDTree(voxel_xyz[voxel_rows])
        distances, nearest_local_rows = one_color_tree.query(
            pcd_xyz[pcd_rows],
            k=1,
            workers=workers,
        )
        nearest_global_rows = voxel_rows[nearest_local_rows]
        pcd_instance_ids[pcd_rows] = voxel_instance_ids[nearest_global_rows]
        matched_distances[pcd_rows] = distances

    if missing_color_point_count > 0:
        raise ValueError(
            f"{missing_color_point_count} PCD points have RGB values that "
            "do not exist in the social voxel instance file."
        )

    print("PCD-to-voxel social instance matching:")
    print(f"    PCD points matched        : {len(pcd_xyz)}")
    print(f"    PCD points missing color  : {missing_color_point_count}")
    print(
        "    Median PCD-to-voxel distance: "
        f"{np.nanmedian(matched_distances):.4f}"
    )
    print(
        "    Maximum PCD-to-voxel distance: "
        f"{np.nanmax(matched_distances):.4f}"
    )
    return pcd_instance_ids


def voxel_surface_area_from_faces(
    voxel_xyz,
    voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),
    voxel_size_xyz=(1.0, 1.0, 1.0),
):
    """通过统计暴露的6邻域voxel表面计算三维表面积。"""
    if len(voxel_xyz) == 0:
        return 0.0

    grid_spacing = np.asarray(
        voxel_grid_spacing_xyz,
        dtype=np.float64,
    )
    if np.any(grid_spacing <= 0):
        raise ValueError("voxel_grid_spacing_xyz values must be positive.")

    grid = np.unique(
        np.rint(voxel_xyz / grid_spacing[None, :]).astype(np.int64),
        axis=0,
    )
    occupied = set(map(tuple, grid.tolist()))

    sx, sy, sz = map(float, voxel_size_xyz)
    directions = (
        ((1, 0, 0), sy * sz),
        ((-1, 0, 0), sy * sz),
        ((0, 1, 0), sx * sz),
        ((0, -1, 0), sx * sz),
        ((0, 0, 1), sx * sy),
        ((0, 0, -1), sx * sy),
    )

    surface_area = 0.0
    for x, y, z in occupied:
        for (dx, dy, dz), face_area in directions:
            if (x + dx, y + dy, z + dz) not in occupied:
                surface_area += face_area
    return float(surface_area)


def calculate_social_instance_morphology(
    image_name,
    voxel,
    pcd,
    voxel_connect_radius=1.8,
    voxel_grid_spacing_xyz=(1.0, 1.0, 1.0),
    voxel_size_xyz=(1.0, 1.0, 1.0),
    workers=-1,
):
    """计算一个图片中每个social mitochondrial object的典型Morphology。"""
    check_minimum_columns(
        "SOCIAL VOXEL INSTANCE",
        voxel,
        minimum_columns=7,
    )
    check_minimum_columns(
        "SOCIAL PCD INSTANCE",
        pcd,
        minimum_columns=9,
    )

    voxel_xyz = voxel[:, :3]
    voxel_rgb = np.clip(np.rint(voxel[:, 3:6]), 0, 255).astype(np.int32)
    voxel_intensity = voxel[:, 6]
    pcd_xyz = pcd[:, :3]
    pcd_rgb = np.clip(np.rint(pcd[:, 3:6]), 0, 255).astype(np.int32)

    voxel_valid = (
        np.isfinite(voxel_xyz).all(axis=1)
        & np.isfinite(voxel[:, 3:7]).all(axis=1)
        & np.any(voxel_rgb != 0, axis=1)
    )
    pcd_valid = (
        np.isfinite(pcd_xyz).all(axis=1)
        & np.isfinite(pcd[:, 3:9]).all(axis=1)
        & np.any(pcd_rgb != 0, axis=1)
    )
    voxel_xyz = voxel_xyz[voxel_valid]
    voxel_rgb = voxel_rgb[voxel_valid]
    voxel_intensity = voxel_intensity[voxel_valid]
    pcd_xyz = pcd_xyz[pcd_valid]
    pcd_rgb = pcd_rgb[pcd_valid]

    if len(voxel_xyz) == 0 or len(pcd_xyz) == 0:
        raise ValueError("No valid colored social voxel/PCD points remain.")

    voxel_instance_ids = split_voxel_instances_by_rgb_and_connectivity(
        voxel_xyz=voxel_xyz,
        voxel_rgb=voxel_rgb,
        connect_radius=voxel_connect_radius,
    )
    pcd_instance_ids = map_pcd_to_voxel_instances_by_rgb(
        pcd_xyz=pcd_xyz,
        pcd_rgb=pcd_rgb,
        voxel_xyz=voxel_xyz,
        voxel_rgb=voxel_rgb,
        voxel_instance_ids=voxel_instance_ids,
        workers=workers,
    )

    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float64)
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(
            "voxel_size_xyz must contain three positive values in x,y,z order."
        )
    voxel_volume = float(np.prod(voxel_size))
    rows = []

    for instance_id in np.unique(voxel_instance_ids):
        voxel_mask = voxel_instance_ids == instance_id
        pcd_mask = pcd_instance_ids == instance_id
        points_voxel = voxel_xyz[voxel_mask]
        points = points_voxel * voxel_size[None, :]
        intensity = voxel_intensity[voxel_mask]
        color = voxel_rgb[np.where(voxel_mask)[0][0]]
        voxel_count = int(len(points_voxel))
        volume = float(voxel_count * voxel_volume)
        centroid = np.mean(points, axis=0)
        centered = points - centroid[None, :]
        radius_of_gyration = float(
            np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        )

        if voxel_count >= 2:
            covariance = np.cov(centered, rowvar=False, ddof=1)
            covariance = np.atleast_2d(covariance).astype(np.float64)
            if covariance.shape != (3, 3):
                covariance = np.zeros((3, 3), dtype=np.float64)
            try:
                eigvals, eigvecs = np.linalg.eigh(covariance)
                eig_order = np.argsort(eigvals)[::-1]
                eigvals = np.maximum(eigvals[eig_order], 0.0)
                eigvecs = eigvecs[:, eig_order]
                projected = centered @ eigvecs
                axis_lengths = np.ptp(projected, axis=0)
            except np.linalg.LinAlgError:
                eigvals = np.zeros(3, dtype=np.float64)
                axis_lengths = np.zeros(3, dtype=np.float64)
        else:
            eigvals = np.zeros(3, dtype=np.float64)
            axis_lengths = np.zeros(3, dtype=np.float64)

        major_axis, middle_axis, minor_axis = map(float, axis_lengths)
        eig_major, eig_middle, eig_minor = map(float, eigvals)
        mean_width = (middle_axis + minor_axis) / 2.0

        surface_area = voxel_surface_area_from_faces(
            voxel_xyz=points_voxel,
            voxel_grid_spacing_xyz=voxel_grid_spacing_xyz,
            voxel_size_xyz=voxel_size_xyz,
        )
        equivalent_sphere_diameter = float(
            (6.0 * volume / np.pi) ** (1.0 / 3.0)
        )
        equivalent_sphere_radius = equivalent_sphere_diameter / 2.0

        if surface_area > 0 and volume > 0:
            sphericity = float(
                (np.pi ** (1.0 / 3.0))
                * ((6.0 * volume) ** (2.0 / 3.0))
                / surface_area
            )
            compactness_3d = float(
                36.0 * np.pi * volume ** 2 / surface_area ** 3
            )
        else:
            sphericity = np.nan
            compactness_3d = np.nan

        rows.append(
            {
                "image_name": image_name,
                "instance_id": int(instance_id),
                "r": int(color[0]),
                "g": int(color[1]),
                "b": int(color[2]),
                "voxel_count": voxel_count,
                "pcd_surface_point_count": int(np.sum(pcd_mask)),
                "volume": volume,
                "surface_area_voxel_faces": surface_area,
                "surface_volume_ratio": safe_ratio(surface_area, volume),
                "equivalent_sphere_diameter": equivalent_sphere_diameter,
                "equivalent_sphere_radius": equivalent_sphere_radius,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "major_axis_length_pca": major_axis,
                "middle_axis_length_pca": middle_axis,
                "minor_axis_length_pca": minor_axis,
                "mean_width_pca": mean_width,
                "aspect_ratio_major_minor": safe_ratio(
                    major_axis,
                    minor_axis,
                ),
                "length_width_ratio": safe_ratio(major_axis, mean_width),
                "elongation_major_middle": safe_ratio(
                    major_axis,
                    middle_axis,
                ),
                "flatness_middle_minor": safe_ratio(
                    middle_axis,
                    minor_axis,
                ),
                "pca_eig_major": eig_major,
                "pca_eig_middle": eig_middle,
                "pca_eig_minor": eig_minor,
                "pca_linearity": safe_ratio(
                    eig_major - eig_middle,
                    eig_major,
                ),
                "pca_planarity": safe_ratio(
                    eig_middle - eig_minor,
                    eig_major,
                ),
                "pca_sphericity_like": safe_ratio(eig_minor, eig_major),
                "pca_anisotropy": safe_ratio(
                    eig_major - eig_minor,
                    eig_major,
                ),
                "radius_of_gyration": radius_of_gyration,
                "sphericity": sphericity,
                "compactness_3d": compactness_3d,
                "intensity_mean": float(np.mean(intensity)),
                "intensity_median": float(np.median(intensity)),
                "intensity_std": float(np.std(intensity)),
                "intensity_min": float(np.min(intensity)),
                "intensity_max": float(np.max(intensity)),
                "intensity_sum": float(np.sum(intensity)),
            }
        )

    morphology_df = pd.DataFrame(rows, columns=INSTANCE_COLUMNS)
    morphology_df = morphology_df.sort_values("instance_id").reset_index(
        drop=True
    )
    print("Social instance morphology calculation finished:")
    print(f"    Image     : {image_name}")
    print(f"    Instances : {len(morphology_df)}")
    print(f"    Columns   : {len(morphology_df.columns)}")
    print("    BBox columns included: 0")
    return morphology_df


def build_one_image_summary(image_name, morphology_df):
    """把一个图片的instance级Morphology汇总成一行。"""
    if morphology_df.empty:
        raise ValueError(f"Empty morphology table for {image_name}.")

    row = {
        "image_name": image_name,
        "n_instances": int(len(morphology_df)),
        "total_voxel_count": float(morphology_df["voxel_count"].sum()),
        "total_pcd_surface_point_count": float(
            morphology_df["pcd_surface_point_count"].sum()
        ),
        "total_volume": float(morphology_df["volume"].sum()),
        "total_surface_area_voxel_faces": float(
            morphology_df["surface_area_voxel_faces"].sum()
        ),
        "total_intensity_sum": float(morphology_df["intensity_sum"].sum()),
    }

    excluded = {
        "instance_id",
        "r",
        "g",
        "b",
        "centroid_x",
        "centroid_y",
        "centroid_z",
    }
    numeric_columns = morphology_df.select_dtypes(
        include=[np.number]
    ).columns
    for column in numeric_columns:
        if column in excluded:
            continue
        values = morphology_df[column]
        row[f"mean_{column}"] = float(values.mean(skipna=True))
        row[f"median_{column}"] = float(values.median(skipna=True))
    return row


def build_dataset_summary(output_path):
    """汇总目录中所有social morphology CSV；每张图片一行。"""
    morphology_paths = sorted(
        output_path.glob("*_instance_morphology_social.csv"),
        key=lambda path: path.name.lower(),
    )
    summary_rows = []

    print("-------------------------------------------------------------------------")
    print("Building dataset-level social Morphology summary...")
    print(f"Morphology CSV files found: {len(morphology_paths)}")

    suffix = "_instance_morphology_social.csv"
    for morphology_path in morphology_paths:
        morphology_df = pd.read_csv(morphology_path)
        if morphology_df.empty:
            print(f"[SUMMARY SKIPPED] Empty CSV: {morphology_path}")
            continue
        image_name = morphology_path.name[:-len(suffix)]
        summary_rows.append(
            build_one_image_summary(
                image_name=image_name,
                morphology_df=morphology_df,
            )
        )

    summary_path = output_path / "all_images_morphology_social_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print("Dataset summary saved:")
    print(f"    {summary_path}")
    print(f"    Images summarized: {len(summary_df)}")
    print("    One row = one image/cell")
    return summary_df, summary_path


def main():

    #
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

    # 所有 8 个数据集共同根目录。
    # DATA_ROOT = Path(
    #     "D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis"
    # )

    DATA_ROOT = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")



    # 收集每个数据集的一行-per-image summary，最后再汇总成 8-type 总表。
    all_dataset_summary_tables = []

    ###########################################################################
    # 参数：不使用argparse，直接在这里修改
    ###########################################################################

    # xyz坐标中相邻voxel之间的坐标间隔。
    voxel_grid_spacing_xyz = (1.0, 1.0, 1.0)

    # 一个voxel在真实物理空间中的x,y,z尺寸。
    # 保持(1,1,1)时，体积和长度使用voxel单位。
    # 如果需要微米单位，可改为实际值，例如(0.07857, 0.07857, 0.16)。
    voxel_size_xyz = (1.0, 1.0, 1.0)

    # 同RGB voxel的连通半径。1.8覆盖规则网格的26邻域。
    voxel_connect_radius = 1.8

    # False时，如果单图Morphology CSV已存在则直接复用，不重复计算。
    overwrite_existing = False

    # cKDTree并行线程数；-1表示使用全部可用CPU线程。
    workers = -1

    total_done = 0
    total_reused = 0
    total_skipped = 0

    for file_name in file_name_list:
        # BASE_DIR = Path(f"D:/public_data/S-BIAD1445/neuron/mito_test")
        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")
        # BASE_DIR = DATA_ROOT / file_name / "mito"
        # BASE_DIR = Path(f"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")
        # BASE_DIR = Path(f"E:/data/20260811-YF_output/image_tif/single_cell_tifs")
        # BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/single_cell_tifs")
        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/masked")

        INPUT_PATH = BASE_DIR / "processed_output"
        OUTPUT_PATH = BASE_DIR / "processed_output"

        if not INPUT_PATH.exists():
            raise FileNotFoundError(
                f"processed_output folder not found: {INPUT_PATH}"
            )
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        print(os.listdir(INPUT_PATH))

        #######################################################################
        # 图片列表
        #######################################################################

        # 自动寻找全部 *_voxel_instance_social.txt。
        suffix = "_voxel_instance_social.txt"
        imgs_list = [
            path.name[:-len(suffix)]
            for path in sorted(
                INPUT_PATH.glob(f"*{suffix}"),
                key=lambda path: path.name.lower(),
            )
        ]

        # 手动指定需要处理的图片。
        # 需要处理全部图片时，将下面这一行注释掉即可。
        # imgs_list = ["fibro_mito488_lyso640_nuc405_17689_5ADVMLE"]

        if len(imgs_list) == 0:
            print(f"No social voxel instance files found in {INPUT_PATH}")
            continue

        print("=========================================================================")
        print(f"Dataset               : {file_name}")
        print(f"BASE_DIR              : {BASE_DIR}")
        print(f"INPUT_PATH            : {INPUT_PATH}")
        print(f"OUTPUT_PATH           : {OUTPUT_PATH}")
        print(f"Selected images       : {len(imgs_list)}")
        print(f"Voxel grid spacing xyz: {voxel_grid_spacing_xyz}")
        print(f"Voxel physical size xyz: {voxel_size_xyz}")
        print(f"Voxel connect radius  : {voxel_connect_radius}")
        print(f"Overwrite existing    : {overwrite_existing}")
        print("=========================================================================")

        for index, image_name in enumerate(imgs_list, start=1):
            voxel_path = (
                INPUT_PATH / f"{image_name}_voxel_instance_social.txt"
            )
            pcd_path = INPUT_PATH / f"{image_name}_pcd_instance_social.txt"
            morphology_path = (
                OUTPUT_PATH
                / f"{image_name}_instance_morphology_social.csv"
            )

            print("-------------------------------------------------------------------------")
            print(
                f"Processing {file_name} | {index}/{len(imgs_list)} | "
                f"{image_name}"
            )
            print("-------------------------------------------------------------------------")
            print("Input files:")
            print(f"    SOCIAL VOXEL: {voxel_path}")
            print(f"    SOCIAL PCD  : {pcd_path}")
            print("Output file:")
            print(f"    MORPHOLOGY  : {morphology_path}")

            if morphology_path.exists() and not overwrite_existing:
                print("[REUSED] Existing social Morphology CSV:")
                print(f"    {morphology_path}")
                total_reused += 1
                continue

            missing_paths = [
                path for path in (voxel_path, pcd_path) if not path.exists()
            ]
            if len(missing_paths) > 0:
                print("[SKIPPED] Missing required input files:")
                for missing_path in missing_paths:
                    print(f"    {missing_path}")
                total_skipped += 1
                continue

            voxel = load_txt_array(voxel_path)
            pcd = load_txt_array(pcd_path)

            print_file_information(
                "SOCIAL VOXEL",
                voxel_path,
                voxel,
                "x,y,z,r,g,b,intensity",
            )
            print_file_information(
                "SOCIAL PCD",
                pcd_path,
                pcd,
                "x,y,z,r,g,b,nx,ny,nz",
            )

            morphology_df = calculate_social_instance_morphology(
                image_name=image_name,
                voxel=voxel,
                pcd=pcd,
                voxel_connect_radius=voxel_connect_radius,
                voxel_grid_spacing_xyz=voxel_grid_spacing_xyz,
                voxel_size_xyz=voxel_size_xyz,
                workers=workers,
            )
            morphology_df.to_csv(morphology_path, index=False)
            print("Social instance Morphology saved:")
            print(f"    {morphology_path}")
            print("    One row = one social mitochondrial object")
            total_done += 1

        dataset_summary_df, dataset_summary_path = build_dataset_summary(
            OUTPUT_PATH
        )

        if not dataset_summary_df.empty:
            dataset_summary_df = dataset_summary_df.copy()
            dataset_summary_df.insert(0, "dataset_name", file_name)
            dataset_summary_df.insert(
                1,
                "cell_type",
                DATASET_SHORT_NAMES.get(file_name, file_name),
            )
            all_dataset_summary_tables.append(dataset_summary_df)

            # 单数据集 summary 也写入 cell type metadata，方便独立查看。
            dataset_summary_df.to_csv(
                OUTPUT_PATH / "all_images_morphology_social_summary_with_dataset.csv",
                index=False,
            )

    ###########################################################################
    # 8-type 总体 Morphology summary
    ###########################################################################
    if len(all_dataset_summary_tables) == 0:
        raise RuntimeError(
            "No dataset-level morphology summary was successfully generated."
        )

    all_8types_morphology = pd.concat(
        all_dataset_summary_tables,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    all_8types_morphology_path = (
        DATA_ROOT / "all_8types_morphology_social_summary.csv"
    )
    all_8types_morphology.to_csv(
        all_8types_morphology_path,
        index=False,
    )

    # 同时尝试与 fixed hierarchical network metrics 自动合并。
    # merge key = dataset_name + image_name，避免不同数据集重名时错误匹配。
    network_candidates = [
        DATA_ROOT / "all_8types_hierarchical_network_metrics.csv",
        DATA_ROOT
        / "all_8types_hierarchical_network_metrics_FIXED_UNFILTERED.csv",
        DATA_ROOT
        / "all_8types_hierarchical_network_metrics_unfiltered.csv",
    ]
    network_path = next(
        (path for path in network_candidates if path.exists()),
        None,
    )

    if network_path is not None:
        network_df = pd.read_csv(network_path)

        required_merge_columns = {"dataset_name", "image_name"}
        if required_merge_columns.issubset(network_df.columns):
            merged = all_8types_morphology.merge(
                network_df,
                on=["dataset_name", "image_name"],
                how="inner",
                suffixes=("_morphology", "_network"),
                validate="one_to_one",
            )

            merged_path = (
                DATA_ROOT
                / "all_8types_morphology_plus_hierarchical_network.csv"
            )
            merged.to_csv(merged_path, index=False)

            morphology_only = set(
                zip(
                    all_8types_morphology["dataset_name"],
                    all_8types_morphology["image_name"],
                )
            ) - set(
                zip(
                    merged["dataset_name"],
                    merged["image_name"],
                )
            )
            network_only = set(
                zip(
                    network_df["dataset_name"],
                    network_df["image_name"],
                )
            ) - set(
                zip(
                    merged["dataset_name"],
                    merged["image_name"],
                )
            )

            merge_qc = pd.DataFrame(
                [
                    {
                        "morphology_rows": len(all_8types_morphology),
                        "network_rows": len(network_df),
                        "matched_rows": len(merged),
                        "morphology_only_rows": len(morphology_only),
                        "network_only_rows": len(network_only),
                        "network_source": str(network_path),
                    }
                ]
            )
            merge_qc_path = (
                DATA_ROOT / "all_8types_morphology_network_merge_QC.csv"
            )
            merge_qc.to_csv(merge_qc_path, index=False)

            print("8-type Morphology + Network merged table saved:")
            print(f"    {merged_path}")
            print(
                f"    Matched rows: {len(merged)} / "
                f"Morphology {len(all_8types_morphology)} / "
                f"Network {len(network_df)}"
            )
        else:
            print(
                "[WARNING] Network summary exists but lacks "
                "dataset_name and/or image_name; merge skipped."
            )
    else:
        print(
            "[INFO] No 8-type network summary found yet. "
            "Morphology summary was still saved normally."
        )

    print("=========================================================================")
    print("Social mitochondrial Morphology calculation finished.")
    print(f"Images calculated : {total_done}")
    print(f"Images reused     : {total_reused}")
    print(f"Images skipped    : {total_skipped}")
    print("Per-image output  : *_instance_morphology_social.csv")
    print("Dataset summary   : all_images_morphology_social_summary.csv")
    print(f"8-type summary    : {all_8types_morphology_path}")
    print("BBox metrics included: 0")
    print("=========================================================================")


if __name__ == "__main__":
    main()
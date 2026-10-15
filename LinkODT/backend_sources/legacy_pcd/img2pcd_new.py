import os
import tifffile
import numpy as np
import open3d as o3d

if not hasattr(np, "Infinity"):
    np.Infinity = np.inf
from skimage import measure
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes, label
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

def keep_largest_component_3d(mask_stack, connectivity=3):
    """
    只保留最大的 3D connected component。

    参数
    ----
    mask_stack : ndarray, shape = z,y,x
        3D binary/int mask。
    connectivity : int
        1 = 6-connectivity
        2 = 18-connectivity
        3 = 26-connectivity

    返回
    ----
    largest_mask : bool ndarray
        最大 3D component。
    """
    binary = mask_stack > 0

    if binary.sum() == 0:
        return binary

    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)

    labeled, num = ndi.label(binary, structure=structure)

    if num == 0:
        return binary

    counts = np.bincount(labeled.ravel())

    # label 0 是背景，不要
    counts[0] = 0

    largest_label = np.argmax(counts)

    largest_mask = labeled == largest_label

    return largest_mask


def fill_component_3d_and_2d(mask_bool):
    """
    对 3D component 填洞。
    先做 3D binary_fill_holes，再逐 z-slice 做 2D fill holes。

    这样比只做 3D 或只做 2D 更稳。
    """
    if mask_bool.sum() == 0:
        return mask_bool

    # 3D 填洞
    filled = ndi.binary_fill_holes(mask_bool)

    # 每一层再填 2D 洞
    filled_2d = np.zeros_like(filled, dtype=bool)

    for z in range(filled.shape[0]):
        filled_2d[z] = ndi.binary_fill_holes(filled[z])

    return filled_2d


def convex_hull_2d_binary(mask_2d):
    """
    对单张 2D binary mask 计算凸包并填充。
    """
    binary = mask_2d > 0

    if binary.sum() < 3:
        return binary

    binary_uint8 = binary.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return binary

    # 合并所有 contour 点，计算一个整体 convex hull
    all_points = np.vstack(contours)

    if len(all_points) < 3:
        return binary

    hull = cv2.convexHull(all_points)

    hull_mask = np.zeros_like(binary_uint8, dtype=np.uint8)

    cv2.fillConvexPoly(
        hull_mask,
        hull,
        color=1
    )

    return hull_mask.astype(bool)


def convex_hull_stack_by_slice(mask_stack_bool):
    """
    对 3D mask 逐 z-slice 做 2D convex hull。

    这样不会把整个 3D 直接包成一个很粗的球，
    而是每一层保持该 z 层的 nuclear contour。
    """
    if mask_stack_bool.ndim != 3:
        raise ValueError(
            f"mask_stack_bool 必须是 3D, 当前 shape = {mask_stack_bool.shape}"
        )

    hull_stack = np.zeros_like(mask_stack_bool, dtype=bool)

    for z in range(mask_stack_bool.shape[0]):
        hull_stack[z] = convex_hull_2d_binary(mask_stack_bool[z])

    return hull_stack


def make_largest_filled_convex_nucleus_mask(mask_stack, max_val=None):
    """
    从 threshold 后的 nucleus mask 得到最终核区域 mask。

    流程：
    1. 只保留最大的 3D component
    2. 对最大 component 填洞
    3. 逐 z-slice 计算 convex hull

    返回：
    lcc_filled_stack : 最大 component + 填洞
    hull_stack       : 最大 component + 填洞 + 逐层 convex hull
    """
    if max_val is None:
        max_val = np.max(mask_stack)

        if max_val == 0:
            max_val = 255

    # 1. largest 3D component
    largest_bool = keep_largest_component_3d(
        mask_stack,
        connectivity=3
    )

    # 2. fill holes
    largest_filled_bool = fill_component_3d_and_2d(largest_bool)

    # 3. slice-wise convex hull
    hull_bool = convex_hull_stack_by_slice(largest_filled_bool)

    lcc_filled_stack = np.zeros_like(mask_stack)
    hull_stack = np.zeros_like(mask_stack)

    lcc_filled_stack[largest_filled_bool] = max_val
    hull_stack[hull_bool] = max_val

    return lcc_filled_stack, hull_stack


def save_pcd2txt(points, save_path):
    np.savetxt(save_path, points, fmt="%.6f", delimiter=",")


def save_pcd2txt_with_header(points, save_path):
    header = "z,y,x,nz,ny,nx"
    np.savetxt(
        save_path,
        points,
        fmt="%.6f",
        delimiter=",",
        header=header,
        comments=""
    )


def mask_to_mesh_marching_cubes_zyx(
        mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        level=0.5,
        pad=True,
        pre_smooth_sigma_um=0.0,
):
    """
    Convert 3D binary mask to mesh using marching cubes.

    Input:
        mask: shape = (Z, Y, X)
        voxel_size_zyx: physical voxel size, e.g. (0.16, 0.04, 0.04)

    Output:
        verts_zyx: vertex coordinates, order = (z, y, x), physical unit
        faces: triangle indices
    """
    mask = (mask != 0)

    z_um, y_um, x_um = voxel_size_zyx

    if pad:
        mask_work = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    else:
        mask_work = mask.copy()

    volume = mask_work.astype(np.float32)

    # Optional: very light volume smoothing before marching cubes.
    # Do not use too large sigma, otherwise thin mitochondria may shrink or merge.
    if pre_smooth_sigma_um is not None and pre_smooth_sigma_um > 0:
        sigma_zyx = (
            pre_smooth_sigma_um / z_um,
            pre_smooth_sigma_um / y_um,
            pre_smooth_sigma_um / x_um,
        )
        volume = ndi.gaussian_filter(volume, sigma=sigma_zyx)

    verts_zyx, faces, normals_zyx, values = measure.marching_cubes(
        volume,
        level=level,
        spacing=voxel_size_zyx,
        allow_degenerate=False,
    )

    # If padded by 1 voxel, subtract the physical padding offset.
    if pad:
        offset = np.asarray(voxel_size_zyx, dtype=np.float32)
        verts_zyx = verts_zyx - offset

    return verts_zyx.astype(np.float32), faces.astype(np.int32)


def build_vertex_adjacency(n_vertices, faces):
    """
    Build vertex adjacency list from triangle faces.
    """
    adjacency = [set() for _ in range(n_vertices)]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        adjacency[a].add(b)
        adjacency[a].add(c)

        adjacency[b].add(a)
        adjacency[b].add(c)

        adjacency[c].add(a)
        adjacency[c].add(b)

    adjacency = [np.array(list(nei), dtype=np.int32) for nei in adjacency]
    return adjacency


def taubin_smooth_mesh(
        verts,
        faces,
        iterations=10,
        lamb=0.5,
        mu=-0.53,
):
    """
    Taubin smoothing.
    Better than simple Laplacian smoothing because it reduces shrinkage.

    verts are in ZYX physical coordinates.
    """
    verts = verts.astype(np.float32).copy()
    adjacency = build_vertex_adjacency(len(verts), faces)

    def laplacian_step(v, step):
        new_v = v.copy()

        for i, nei in enumerate(adjacency):
            if len(nei) == 0:
                continue

            mean_neighbor = v[nei].mean(axis=0)
            new_v[i] = v[i] + step * (mean_neighbor - v[i])

        return new_v

    for _ in range(iterations):
        verts = laplacian_step(verts, lamb)
        verts = laplacian_step(verts, mu)

    return verts.astype(np.float32)


def compute_vertex_normals(verts, faces):
    """
    Compute vertex normals from mesh.
    Coordinate order remains ZYX.
    """
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int32)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / (face_norm + 1e-8)

    vertex_normals = np.zeros_like(verts, dtype=np.float32)

    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norm + 1e-8)

    # Try to orient normals outward using global center.
    # For very concave shapes this is not perfect, but usually works for surface sampling.
    center = verts.mean(axis=0, keepdims=True)
    direction = verts - center
    sign = np.sum(vertex_normals * direction, axis=1)

    vertex_normals[sign < 0] *= -1.0

    return vertex_normals.astype(np.float32)


def voxel_mask_to_smoothed_mesh(
        mask,
        voxel_size_zyx=(0.16, 0.04, 0.04),
        mesh_smooth_iterations=6,
        pre_smooth_sigma_um=0.0,

):
    """
    Full pipeline:
        voxel mask -> marching cubes mesh -> Taubin smoothing -> surface point cloud + normals

    Output coordinate order:
        z, y, x, nz, ny, nx
    """
    mask = (mask != 0)

    verts_zyx, faces = mask_to_mesh_marching_cubes_zyx(
        mask,
        voxel_size_zyx=voxel_size_zyx,
        level=0.5,
        pad=True,
        pre_smooth_sigma_um=pre_smooth_sigma_um,
    )

    print("Raw mesh vertices:", verts_zyx.shape)
    print("Raw mesh faces:", faces.shape)

    if mesh_smooth_iterations is not None and mesh_smooth_iterations > 0:
        verts_smooth_zyx = taubin_smooth_mesh(
            verts_zyx,
            faces,
            iterations=mesh_smooth_iterations,
            lamb=0.5,
            mu=-0.53,
        )
    else:
        verts_smooth_zyx = verts_zyx

    return {
        "verts_zyx": verts_zyx,
        "verts_smooth_zyx": verts_smooth_zyx,
        "faces": faces,
    }


def read_tif(img_path):
    try:
        with tifffile.TiffFile(img_path) as tif:
            data = tif.asarray()
    except AttributeError as exc:
        if "newbyteorder" not in str(exc):
            raise

        from PIL import Image, ImageSequence

        with Image.open(img_path) as img:
            frames = [np.asarray(frame) for frame in ImageSequence.Iterator(img)]

        if len(frames) == 0:
            raise ValueError(f"No frames found in TIFF: {img_path}") from exc

        data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]
    print("************** read tif *******************")
    print(f"Original Shape: {data.shape}")
    print(f"Original Data Type: {data.dtype}")

        # 如果是2D图像，添加一个前置维度
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        print("Converted to 3D by adding leading dimension.")

    print(f"Processed Shape: {data.shape}")
    print("************** read tif *******************")
    return data


def save_tif(save_path, tif_img):
    tifffile.imwrite(save_path, tif_img)
    print(f"TIFF image saved at: {save_path}")


def enhance_adaptive(img_stack, block_size=25, C=0.1):
    """
    使用 OpenCV 的自适应阈值增强 3D TIF 图像中的线粒体对比度（输入输出都是 NumPy）。

    :param img_stack: 3D NumPy 数组，形状 (Z, H, W)
    :param block_size: 自适应阈值计算窗口大小，必须是奇数（一般 11~25）
    :param C: 调整整体亮度（较大值会减少线粒体区域）
    :return: 增强后的 3D NumPy 数组
    """
    enhanced_stack = np.zeros_like(img_stack)

    for i in range(img_stack.shape[0]):
        img = img_stack[i]

        # **确保图像是 8-bit**
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # **Step 2: 自适应阈值处理**
        binary_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )

        enhanced_stack[i] = binary_img / 255.0

    return enhanced_stack * img_stack  # 返回 NumPy 格式的 3D 图像数据


def compute_otsu_threshold_uint8(image_uint8, ignore_zero=False):
    """
    对 uint8 图像计算全局 Otsu 阈值。
    支持 2D / 3D。
    """

    values = image_uint8.ravel()

    if ignore_zero:
        values = values[values > 0]

    if len(values) == 0:
        raise ValueError("用于 Otsu 的像素为空，请检查图像或 ignore_zero 设置。")

    values_2d = values.reshape(-1, 1).astype(np.uint8)

    otsu_threshold, _ = cv2.threshold(
        values_2d,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return int(otsu_threshold)


def plot_otsu_histogram_check(
        image_uint8,
        otsu_threshold,
        save_fig_path=None,
        show=True,
        use_log_y=True,
        zoom_nonzero=True
):
    """
    画 Otsu 检查图。

    第一张：全部像素 histogram，包括 0 背景。
    第二张：去掉 0 背景后的 histogram，更容易看信号分布。
    """

    values_all = image_uint8.ravel()
    values_nonzero = values_all[values_all > 0]

    hist_all, _ = np.histogram(values_all, bins=256, range=(0, 255))
    peak_all = int(np.argmax(hist_all))

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(256), hist_all)

    plt.axvline(
        peak_all,
        linestyle="--",
        linewidth=2,
        label=f"Peak = {peak_all}"
    )

    plt.axvline(
        otsu_threshold,
        linestyle="-",
        linewidth=2,
        label=f"Otsu threshold = {otsu_threshold}"
    )

    if use_log_y:
        plt.yscale("log")

    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("Histogram with Otsu threshold")
    plt.legend()
    plt.tight_layout()

    # 不使用 1e7 这种科学计数法
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")

    if save_fig_path is not None:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)

        full_path = save_fig_path.with_name(save_fig_path.stem + "_all_pixels.png")
        plt.savefig(full_path, dpi=300)
        print(f"Saved: {full_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # 再画一个去掉 0 背景的 histogram
    if zoom_nonzero and len(values_nonzero) > 0:
        hist_nonzero, _ = np.histogram(values_nonzero, bins=256, range=(0, 255))
        peak_nonzero = int(np.argmax(hist_nonzero))

        plt.figure(figsize=(7, 5))
        plt.plot(np.arange(256), hist_nonzero)

        plt.axvline(
            peak_nonzero,
            linestyle="--",
            linewidth=2,
            label=f"Non-zero peak = {peak_nonzero}"
        )

        plt.axvline(
            otsu_threshold,
            linestyle="-",
            linewidth=2,
            label=f"Otsu threshold = {otsu_threshold}"
        )

        if use_log_y:
            plt.yscale("log")

        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("Histogram excluding zero background")
        plt.legend()
        plt.tight_layout()

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="y")

        if save_fig_path is not None:
            nonzero_path = save_fig_path.with_name(save_fig_path.stem + "_nonzero_pixels.png")
            plt.savefig(nonzero_path, dpi=300)
            print(f"Saved: {nonzero_path}")

        if show:
            plt.show()
        else:
            plt.close()


def convert_tiff_to_point_cloud_otsu(
        image: np.ndarray,
        scale_factor: float = 1.0,
        show_otsu_plot: bool = True,
        save_otsu_fig_path: str = None,
        ignore_zero_for_otsu: bool = False,
        return_otsu: bool = False
):
    """
    将 TIFF 图像转换为 3D 点云，并使用 Otsu 自动阈值。

    :param image: TIFF 图像，支持 2D 或 3D
    :param scale_factor: z 方向缩放因子
    :param show_otsu_plot: 是否显示 Otsu 检查图
    :param save_otsu_fig_path: 保存 Otsu 检查图路径
    :param ignore_zero_for_otsu: 计算 Otsu 时是否忽略 0 背景
    :param return_otsu: 是否返回 Otsu threshold
    :return: points, 或 points, otsu_threshold
    """

    print("********************* convert tif to point cloud ********************")

    image = np.asarray(image)

    print(f"Input Image Shape: {image.shape}")
    print(f"Original dtype: {image.dtype}")
    print(f"Original max: {np.max(image)}, min: {np.min(image)}")

    # 2D 图像转成 3D，方便后面统一处理
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
        print(f"2D image detected, converted to 3D shape: {image.shape}")

    elif image.ndim == 3:
        pass

    else:
        raise ValueError(
            f"当前只支持 2D 或 3D 图像，但输入 image.ndim = {image.ndim}, shape = {image.shape}"
        )

    # 归一化到 uint8
    if image.dtype != np.uint8:
        image_uint8 = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    print(f"Normalized dtype: {image_uint8.dtype}")
    print(f"Normalized max: {np.max(image_uint8)}, min: {np.min(image_uint8)}")

    # 计算 Otsu 阈值
    otsu_threshold = compute_otsu_threshold_uint8(
        image_uint8,
        ignore_zero=ignore_zero_for_otsu
    )

    print(f"Otsu threshold = {otsu_threshold}")

    # 画图检查
    if show_otsu_plot or save_otsu_fig_path is not None:
        plot_otsu_histogram_check(
            image_uint8=image_uint8,
            otsu_threshold=otsu_threshold,
            save_fig_path=save_otsu_fig_path,
            show=show_otsu_plot,
            use_log_y=True,
            zoom_nonzero=True
        )

    # Otsu 二值化
    img_mask = image_uint8 > otsu_threshold

    z_indices, y_indices, x_indices = np.where(img_mask)

    intensity_values = image_uint8[z_indices, y_indices, x_indices].astype(np.uint8)

    points = np.vstack(
        (
            x_indices,
            y_indices,
            z_indices * scale_factor,
            intensity_values
        )
    ).T.astype(np.float32)

    print(f"Point Cloud Shape: {points.shape}")
    print("********************* convert tif to point cloud ********************")

    if return_otsu:
        return points, otsu_threshold
    else:
        return points


#
#
# def convert_tiff_to_point_cloud(image: np.ndarray, scale_factor: float = 1.0, threshold: int = 100):
#     """
#     将 TIFF 图像转换为 3D 点云，并将灰度值（intensity）映射到 0-255。
#     :param image: 读取的 TIFF 图像 (numpy 数组)，支持 2D 或 3D
#     :param save_path: (可选) 若提供，则保存点云到 TXT
#     :param scale_factor: 深度缩放因子
#     :param threshold: 过滤低于此值的像素
#     :return: numpy 数组 (N, 4) -> (x, y, z, intensity)
#     """
#     # 确保数据是 NumPy 数组
#
#     print("********************* convert tif to point cloud ********************")
#
#     # image = fill_closed_regions(image)
#
#     image = np.asarray(image)
#
#     print(f"Input Image Shape: {image.shape}")  # Debug
#
#     # 归一化到 8-bit（防止过大）
#     if image.dtype != np.uint8:
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
#     print(image.dtype, np.max(image), np.min(image))
#
#     img = image > threshold
#     # img = fill_holes_3D(img)
#
#     # 过滤非零像素（Z-stack 时支持 3D）
#     z_indices, y_indices, x_indices = np.where(img)
#
#     # **灰度值归一化到 [0, 255]**
#     intensity_values = image[z_indices, y_indices, x_indices].astype(np.uint8)
#
#     # **构造点云 (N, 4) -> (x, y, z, intensity)**
#     points = np.vstack((x_indices, y_indices, z_indices * scale_factor, intensity_values)).T.astype(np.float32)
#
#     print(f"Point Cloud Shape: {points.shape}")  # Debug
#     print("********************* convert tif to point cloud ********************")
#
#     return points

from pathlib import Path
import numpy as np
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import filters, morphology


def percentile_normalize_3d(image, p_low=1, p_high=99.8):
    """
    用 percentile 做归一化，比 cv2.normalize 更稳。
    避免极端亮点把整体动态范围拉坏。
    """
    image = image.astype(np.float32)

    low = np.percentile(image, p_low)
    high = np.percentile(image, p_high)

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - low) / (high - low)
    image = np.clip(image, 0, 1)

    return image.astype(np.float32)


def auto_threshold_3d(image, method="li"):
    """
    自动阈值。
    method 可选：
        "otsu"      适合前景/背景比较双峰
        "li"        对荧光细胞器通常比较稳
        "triangle"  适合前景较少、背景占多数的图
        "yen"       有时对亮点结构敏感
    """
    vals = image[np.isfinite(image)]

    if vals.size == 0:
        return 0.0

    if method == "otsu":
        thr = filters.threshold_otsu(vals)
    elif method == "li":
        thr = filters.threshold_li(vals)
    elif method == "triangle":
        thr = filters.threshold_triangle(vals)
    elif method == "yen":
        thr = filters.threshold_yen(vals)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return float(thr)


def local_threshold_3d_slice_by_slice(image, block_size=51, offset=0.0):
    """
    对每个 z-slice 做 local adaptive threshold。
    适合 z-stack 内部照明不均匀、背景不均匀的情况。

    block_size 必须是奇数，比如 31, 51, 75。
    """
    if block_size % 2 == 0:
        block_size += 1

    mask = np.zeros_like(image, dtype=bool)

    for z in range(image.shape[0]):
        thr = filters.threshold_local(
            image[z],
            block_size=block_size,
            method="gaussian",
            offset=offset
        )
        mask[z] = image[z] > thr

    return mask


def semantic_segment_organelle_3d(
        image,
        threshold_method="otsu",
        use_local_threshold=False,
        local_block_size=51,
        local_offset=0.0,
        denoise_sigma=0.7,
        background_sigma_xy=8.0,
        background_sigma_z=1.0,
        min_voxels=20,
        closing_radius=1,
        fill_holes=False,
        threshold_scale = 5.0
):
    """
    从 3D 荧光 z-stack 得到 organelle semantic mask。

    参数建议：
    - 线粒体：threshold_method="li" 或 "triangle"
    - 溶酶体/脂滴：threshold_method="otsu" 或 "li"
    - 背景不均匀：use_local_threshold=True
    """

    print("********************* semantic segment organelle 3D *********************")

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image with shape (z, y, x), but got shape {image.shape}"
        )

    print(f"Input image shape: {image.shape}")
    print(f"Input dtype: {image.dtype}, min={image.min()}, max={image.max()}")

    # 1. percentile normalize 到 0-1
    img_norm = percentile_normalize_3d(image, p_low=1, p_high=99.8)

    # 2. 轻微去噪
    img_smooth = ndi.gaussian_filter(
        img_norm,
        sigma=(denoise_sigma, denoise_sigma, denoise_sigma)
    )

    # 3. 背景扣除
    # 对 xy 用大 sigma 去估计慢变化背景，对 z 用小 sigma，避免跨 z 过度模糊
    background = ndi.gaussian_filter(
        img_smooth,
        sigma=(background_sigma_z, background_sigma_xy, background_sigma_xy)
    )

    corrected = img_smooth - background
    corrected[corrected < 0] = 0

    # 再归一化一次，方便阈值
    corrected = percentile_normalize_3d(corrected, p_low=1, p_high=99.8)

    # 4. 阈值
    if use_local_threshold:
        mask = local_threshold_3d_slice_by_slice(
            corrected,
            block_size=local_block_size,
            offset=local_offset
        )
        print(f"Using local threshold, block_size={local_block_size}, offset={local_offset}")
    else:
        thr = auto_threshold_3d(corrected, method=threshold_method)
        mask = corrected > (thr * threshold_scale)
        print(f"Using global auto threshold: method={threshold_method}, threshold={thr:.4f}")

    # 5. 去掉小噪点
    if min_voxels is not None and min_voxels > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_voxels)

    # 6. 3D closing，让细小断裂略微连接
    if closing_radius is not None and closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.ball(closing_radius))

    # 7. 是否填洞
    # 线粒体一般不建议强行 fill_holes，脂滴/溶酶体可以考虑
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    mask = mask.astype(bool)

    print(f"Mask voxel count: {mask.sum()}")
    print("********************* semantic segment organelle 3D *********************")

    return mask, corrected


def convert_mask_to_point_cloud(
        mask,
        intensity_image,
        scale_factor=1.0
):
    """
    把 semantic mask 转成点云。
    输出格式：
        x, y, z, intensity

    intensity_image 建议用 corrected 或原始 image。
    如果后续只是做几何实例分割，用 corrected 更稳。
    如果想保留真实荧光强度，用原始 image。
    """

    z_indices, y_indices, x_indices = np.where(mask)

    intensity_image = np.asarray(intensity_image).astype(np.float32)

    # intensity 映射到 0-255，方便保存
    inten = intensity_image[z_indices, y_indices, x_indices]
    if inten.size > 0:
        inten = percentile_normalize_3d(inten, p_low=1, p_high=99.8)
        inten = (inten * 255).astype(np.uint8)
    else:
        inten = np.array([], dtype=np.uint8)

    points = np.vstack((
        x_indices,
        y_indices,
        z_indices * scale_factor,
        inten
    )).T.astype(np.float32)

    print(f"Point cloud shape: {points.shape}")

    return points


# def convert_tiff_to_point_cloud(
#         image,
#         scale_factor=1.0,
#         threshold_method="otsu",
#         use_local_threshold=False,
#         min_voxels=20,
#         closing_radius=1,
#         threshold_scale=5.0
# ):
#     """
#     你的原函数的替代版：
#     不再需要手动输入 threshold。
#     """

#     mask, corrected = semantic_segment_organelle_3d(
#         image=image,
#         threshold_method=threshold_method,
#         use_local_threshold=use_local_threshold,
#         min_voxels=min_voxels,
#         closing_radius=closing_radius,
#         threshold_scale=threshold_scale,
#     )

#     points = convert_mask_to_point_cloud(
#         mask=mask,
#         intensity_image=corrected,
#         scale_factor=scale_factor
#     )

#     return points, mask, corrected




def convert_tiff_to_point_cloud(image: np.ndarray, scale_factor: float = 1.0, threshold: int = 100):
    """
    将 TIFF 图像转换为 3D 点云，并将灰度值（intensity）映射到 0-255。
    :param image: 读取的 TIFF 图像 (numpy 数组)，支持 2D 或 3D
    :param save_path: (可选) 若提供，则保存点云到 TXT
    :param scale_factor: 深度缩放因子
    :param threshold: 过滤低于此值的像素
    :return: numpy 数组 (N, 4) -> (x, y, z, intensity)
    """
    # 确保数据是 NumPy 数组

    print("********************* convert tif to point cloud ********************")

    # image = fill_closed_regions(image)

    image = np.asarray(image)

    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image, got shape {image.shape}."
        )

    print(
        f"Original image: dtype={image.dtype}, "
        f"max={np.nanmax(image)}, min={np.nanmin(image)}"
    )

    # Use P99.9 instead of the true maximum for non-uint8 images.
    # Values above P99.9 are clipped to 255.
    if image.dtype != np.uint8:
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            raise ValueError("Cannot normalize image: no finite pixel values.")

        lower_value = float(np.min(finite_values))
        upper_value = float(np.percentile(finite_values, 99.9))

        if upper_value <= lower_value:
            raise ValueError(
                f"P99.9 ({upper_value}) must be greater than "
                f"the minimum value ({lower_value})."
            )

        image = np.asarray(image, dtype=np.float64)
        image = np.nan_to_num(
            image,
            nan=lower_value,
            posinf=upper_value,
            neginf=lower_value,
        )
        image = np.clip(image, lower_value, upper_value)
        image = (image - lower_value) / (upper_value - lower_value)
        image = np.rint(image * 255.0).astype(np.uint8)

        print(
            f"Percentile normalization: min={lower_value:.6g}, "
            f"P99.9={upper_value:.6g} -> uint8 [0, 255]"
        )

    print(f"Input Image Shape: {image.shape}")  # Debug

    print(
        f"Processed image: dtype={image.dtype}, "
        f"max={np.max(image)}, min={np.min(image)}, threshold={threshold}"
    )

    print(np.sum(image))

    # print(image)

    img = image > threshold
    # img = fill_holes_3D(img)
    print(np.sum(img))

    # 过滤非零像素（Z-stack 时支持 3D）
    z_indices, y_indices, x_indices = np.where(img)

    # **灰度值归一化到 [0, 255]**
    intensity_values = image[z_indices, y_indices, x_indices].astype(np.uint8)

    # **构造点云 (N, 4) -> (x, y, z, intensity)**
    points = np.vstack((x_indices, y_indices, z_indices * scale_factor, intensity_values)).T.astype(np.float32)

    print(f"Point Cloud Shape: {points.shape}")  # Debug
    print("********************* convert tif to point cloud ********************")

    return points


def save_pcd2txt(points, save_path):
    np.savetxt(save_path, points, fmt="%.3f", delimiter=",")


def convert_point_cloud_to_tiff_fixed_size(points: np.ndarray, save_img_path=None, fixed_size=(1200, 1200, 60),
                                           scale_factor=1.0):
    """
    将点云数据转换回 3D TIFF 图像，并限制 X, Y 轴大小为 1200x1200。
    :param points: (N, 4) -> (x, y, z, intensity) 的点云数据
    :param save_path: 保存的 .tif 文件路径
    :param fixed_size: (width, height) 固定的 X, Y 尺寸
    """
    # **确保 points 是 (N, 4)**
    if points.shape[1] != 4:
        raise ValueError("Points should have shape (N, 4) -> (x, y, z, intensity)")

    # **提取 x, y, z, intensity**
    x, y, z, intensity = points.T
    intensity = np.full_like(x, 255, dtype=np.uint8)
    z = z / scale_factor

    # **限制 X, Y 到 1200x1200**
    max_x, max_y, max_z = fixed_size

    print(f"Reconstructing 3D Image Shape: ({max_z}, {max_y}, {max_x})")  # Debug

    # **创建 3D NumPy 体数据**
    voxel_grid = np.zeros((max_z, max_y, max_x), dtype=np.uint8)

    # **裁剪超出 1200x1200 范围的点**
    valid_indices = (x < max_x) & (y < max_y)
    x, y, z, intensity = x[valid_indices], y[valid_indices], z[valid_indices], intensity[valid_indices]

    # **填充体数据**
    voxel_grid[z.astype(int), y.astype(int), x.astype(int)] = intensity.astype(np.uint8)

    # **确保目录存在**
    if save_img_path is not None:
        save_path = Path(save_img_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    return voxel_grid


def save_mesh(mesh, save_path):
    mesh.export(save_path)


def read_mesh(mesh_path, iterations=3):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    return mesh


def save_obj_from_zyx_mesh(save_path, verts_zyx, faces):
    """
    Save mesh to OBJ for Blender/MeshLab.

    Internal verts are ZYX.
    OBJ writes XYZ.
    """
    verts_xyz = verts_zyx[:, [2, 1, 0]]

    with open(save_path, "w", encoding="utf-8") as f:
        for v in verts_xyz:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")

    print(f"OBJ saved at: {save_path}")


def mesh_2_pcd(mesh):
    vertices = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # mesh compute normals
    mat_normals = np.hstack((mesh.vertices, mesh.vertex_normals))  # mesh to point cloud with normals

    return mat_normals


def fill_closed_regions(img_stack, hole_size_threshold=100):
    # 复制输入，避免修改原数据
    filled_stack = np.copy(img_stack)

    for i in range(img_stack.shape[0]):

        img = img_stack[i]

        # 定义前景区域（非零区域）
        foreground = (img != 0)

        # 填充孔洞（binary_fill_holes 返回填充后的布尔图像）
        filled = binary_fill_holes(foreground)

        # 取出填充的区域（原来为背景但填充为前景的位置，即孔洞）
        holes = filled & (~foreground)

        # 对孔洞进行连通区域标记
        labeled_holes, num_features = label(holes)

        # 获取当前层的最大值（填充时使用）
        max_val = img.max()

        # 遍历每个连通的孔洞区域
        for j in range(1, num_features + 1):
            hole_mask = (labeled_holes == j)
            # 仅当该孔洞区域内的像素数小于等于阈值时，进行填充
            if np.sum(hole_mask) <= hole_size_threshold:
                filled_stack[i][hole_mask] = max_val

    return filled_stack




def convex_hull_2d_binary(mask_2d):
    """
    对单张 2D binary mask 计算最小凸包，并填充。

    输入:
        mask_2d: 2D array, 非零为前景

    输出:
        hull_mask: 2D bool array
    """
    binary = mask_2d > 0

    if binary.sum() < 3:
        return binary

    # OpenCV 需要 uint8
    binary_uint8 = binary.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return binary

    # 把所有 contour 点合并，算一个整体 convex hull
    all_points = np.vstack(contours)

    if len(all_points) < 3:
        return binary

    hull = cv2.convexHull(all_points)

    hull_mask = np.zeros_like(binary_uint8, dtype=np.uint8)

    cv2.fillConvexPoly(
        hull_mask,
        hull,
        color=1
    )

    return hull_mask.astype(bool)

def remove_small_noise_3d(
        tif_img,
        min_size=8,
        connectivity=3,
        smooth_method="median",   # None / "median" / "gaussian" / "morph"
        median_size=3,
        gaussian_sigma=1.0,
        morph_radius=1,
):
    """
    去掉二值 3D 图像里的孤立噪点（小连通域），并可选把毛躁的边缘圆滑一下。

    参数
    ----
    tif_img : ndarray, shape = (z, y, x)
        二值化后的图像，非零为前景（比如 0 / 255）。
    min_size : int
        连通域体素数小于该值的都会被删掉。=1 只删单个孤立点。
    connectivity : int
        1 = 6 邻域, 2 = 18 邻域, 3 = 26 邻域（判断“连在一起”用）。
    smooth_method : str or None
        圆滑方式：
          None       不圆滑，只去噪
          "median"   中值滤波，最稳，磨锯齿+去点，形状基本不变（推荐）
          "gaussian" 高斯模糊后重新按 0.5 二值化，边缘最圆润，半径大细结构会缩
          "morph"    闭运算+开运算，顺一顺边缘
    median_size : int
        中值滤波核大小，奇数，越大越圆滑（3 / 5）。
    gaussian_sigma : float
        高斯平滑 sigma，越大越圆滑（0.8 ~ 1.5 常用）。
    morph_radius : int
        闭/开运算球半径。线粒体这种细结构建议 1，别太大。

    返回
    ----
    cleaned : ndarray
        去噪+圆滑后的图像，dtype 和前景值与输入保持一致。
    """
    print("********************* remove small noise 3D *********************")

    arr = np.asarray(tif_img)
    binary = arr > 0

    print(f"Before: foreground voxels = {binary.sum()}")

    # 记录原始前景值（一般是 255），最后填回去
    fg_val = arr.max()
    if fg_val == 0:
        fg_val = 255

    # ---- 1. 先按体素数删小连通块（去孤立噪点）----
    if min_size is not None and min_size > 0:
        structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
        labeled, num = ndi.label(binary, structure=structure)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # 背景不算
            keep = counts >= min_size
            binary = keep[labeled]

    print(f"After remove small objects: foreground voxels = {int(binary.sum())}")

    # ---- 2. 圆滑边缘 ----
    if smooth_method == "median":
        # 中值滤波：磨锯齿、去孤立点，形状基本保留
        binary = ndi.median_filter(binary.astype(np.uint8), size=median_size) > 0

    elif smooth_method == "gaussian":
        # 高斯平滑后重新二值化：边缘最圆润
        blurred = ndi.gaussian_filter(binary.astype(np.float32), sigma=gaussian_sigma)
        binary = blurred > 0.5

    elif smooth_method == "morph":
        # 闭运算填小坑 + 开运算去小凸起
        ball = morphology.ball(morph_radius)
        binary = morphology.binary_closing(binary, ball)
        binary = morphology.binary_opening(binary, ball)

    elif smooth_method is None:
        pass
    else:
        raise ValueError(f"未知的 smooth_method: {smooth_method}")

    print(f"After smoothing ({smooth_method}): foreground voxels = {int(binary.sum())}")

    cleaned = np.zeros_like(arr)
    cleaned[binary] = fg_val

    print("********************* remove small noise 3D *********************")

    return cleaned


def convex_hull_stack_by_slice(mask_stack, max_val=None):
    """
    对 3D mask 逐 z-slice 计算 2D convex hull。

    输入:
        mask_stack: 3D array, shape 通常为 z,y,x
        max_val: 输出前景强度。None 时自动用原图最大值。

    输出:
        hull_stack: 3D array, 与输入 shape 相同
    """
    if mask_stack.ndim != 3:
        raise ValueError(f"mask_stack 必须是 3D，但当前 shape 是 {mask_stack.shape}")

    if max_val is None:
        max_val = np.max(mask_stack)

        if max_val == 0:
            max_val = 255

    hull_stack = np.zeros_like(mask_stack)

    for i in range(mask_stack.shape[0]):
        slice_mask = mask_stack[i]

        hull_mask = convex_hull_2d_binary(slice_mask)

        hull_stack[i][hull_mask] = max_val

    return hull_stack

def convert_image_to_point_cloud(
    image: np.ndarray,
    scale_factor: float = 1.0
) -> np.ndarray:
    """
    将 2D / 3D intensity 图像转换为点云。

    Parameters
    ----------
    image : np.ndarray
        输入 intensity 图像。

        支持：
        - 2D: (Y, X)
        - 3D: (Z, Y, X)

        所有非零像素 / voxel 都会转换为点。
        原始像素值作为 intensity 保存。

    scale_factor : float
        Z 方向缩放因子。

        例如：
        XY pixel size = 1
        Z spacing = 3

        则：
        scale_factor = 3

    Returns
    -------
    points : np.ndarray
        shape = (N, 4)

        每一行为：

        [x, y, z, intensity]

    """

    print("******** intensity image -> point cloud ********")

    image = np.asarray(image)

    # --------------------------------------------------
    # 2D -> 增加一个 Z 维度
    # --------------------------------------------------
    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    elif image.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D image, got shape {image.shape}"
        )

    # --------------------------------------------------
    # 所有非零 voxel 都作为点云
    # --------------------------------------------------
    valid_mask = image != 0

    # --------------------------------------------------
    # 提取坐标
    # np.where 返回顺序：
    # Z, Y, X
    # --------------------------------------------------
    z_indices, y_indices, x_indices = np.where(valid_mask)

    # --------------------------------------------------
    # 提取对应位置的 intensity
    # --------------------------------------------------
    intensities = image[
        z_indices,
        y_indices,
        x_indices
    ]

    # --------------------------------------------------
    # 构建：
    # x, y, z, intensity
    # --------------------------------------------------
    points = np.column_stack(
        (
            x_indices.astype(np.float32),
            y_indices.astype(np.float32),
            z_indices.astype(np.float32) * scale_factor,
            intensities.astype(np.float32)
        )
    )

    # --------------------------------------------------
    # 信息
    # --------------------------------------------------
    print(f"Image shape      : {image.shape}")
    print(f"Non-zero voxels  : {len(points)}")
    print(f"Point cloud shape: {points.shape}")

    if len(points) > 0:
        print(f"Intensity min    : {intensities.min()}")
        print(f"Intensity max    : {intensities.max()}")

    print("***********************************************")

    return points

def main():
    ##################################################
    #################### paths #######################

    file_name_list = [

# "        20251127_msto_ld488_mito640_nuc405",
        # "20250905_msto_mitotracekr488_lysotracker640_nuc405",

        # "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",
        # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",
        
        # "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",
        # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",
        
        # "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",
        # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",
        
        # "20260207_fibro_mito488_lyso640_nuc405_tif",
        # "20251216_fibro_mito488tracker_ld561bodipy_nuc405hurst",
        
        # "20251213_h28_mito488_ld561_nuc405",
        # "20250917_h28_mitotracker488_lysotracker640_nuc405",
        
        # "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",
        # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",

        # "20251201_u2os_ld488_mito640_nuc405",


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
        # "mito-mChGFP_LTRDR50nM_0081",

        # "dTag_CCCP10uM_1518_conTL_002",
        "H9c2-dTag_normal_1518_conTL_001",

    ]

    for file_name in file_name_list:

        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/MGO_cell/organized_tif/ld")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/{file_name}/ld")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/xiao_tubulin_488/recon")
        # BASE_DIR = Path(f"./data/pub_cccp/cccp_cell1/C2-20210611_H9c2-dTag_CCCP10uM_1518_conTL_002_ALX")
        # BASE_DIR = Path(f"./data/pub_cccp/gluc/20210330_H9c2_mito-mChGFP_LTRDR50nM_1522_dvt-5sTL_003_ALX")
        # BASE_DIR = Path(f"./data/organ/mito")
        # BASE_DIR = Path(fr"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path(fr"D:/public_data/cccp/cccp_mito/c2/{file_name}")
        # BASE_DIR = Path(f"./data/pub_cccp/galac/20210330_H9c2-gluc_mito-mChGFP_LTRDR50nM_1522_dvt-5sTL_007_ALX")

        # BASE_DIR = Path(
        #     f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/20251002_remodeling_tif/24h/{file_name}")
    
        # BASE_DIR = Path(f"./data/{file_name}")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/mito_seg/data/20260821-ALS ipSC_PKMITO_feng")
        # BASE_DIR = Path(fr"D:/UTS/cell_seg/codes/mito_seg/data/example/mito")
        # BASE_DIR = Path(f"E:/data/label_SIM/er")
        BASE_DIR = Path(f"E:/data/20260805_YF_mito_output/image_tif/masked")
        # BASE_DIR = Path(f"E:/xuyu/U2OS-ER-Mito-Cu-20260902-3D/er")

        # BASE_DIR = Path(f"./data/raw")

        threshold =100
        threshold_scale = 2.6

        print(f"Processing {file_name}")

        INPUT_PATH = BASE_DIR
        OUTPUT_PATH = BASE_DIR / "processed_output"
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        print(os.listdir(INPUT_PATH))

        imgs_list = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('.tif')]
        # imgs_list = ["U2OS_6h_001_20260902_164335_denoise"]

        print("processing {}".format(imgs_list))

        name_lists = []

        i = 0
        num_imgs = len(imgs_list)

        for img_name in imgs_list:
            i = i + 1
            print(f"-------------------------------------------------------------------------")
            print(f"------- processing -------- {i}/{num_imgs} ------------------------------")
            print(f"-------------------------------------------------------------------------")

            img_name = img_name.split('.')[0]
            name_lists.append(img_name)

            ##################################################
            #################### paths #######################

            img_path = INPUT_PATH / (img_name + ".tif")
            save_mask_path = OUTPUT_PATH / (img_name + "_mask.tif")
            save_adpting_path = OUTPUT_PATH / (img_name + "_adpt.tif")
            save_txt_path = OUTPUT_PATH / (img_name + "_voxel.txt")
            save_obj_path = OUTPUT_PATH / (img_name + "_mesh.obj")
            save_pcd_path = OUTPUT_PATH / (img_name + "_pcd.txt")
            save_mask_cov_path = OUTPUT_PATH / (img_name + "_mask_cov.tif")


            print("Reading image", img_path)

            ################### paths #######################
            #################################################

            #################################################
            ############ read tif and adoptive ################

            img = read_tif(img_path)  # read tif img
            # img_adpt = enhance_adaptive(img, block_size=37, C=12)  # adaptive process tif img
            img_adpt = enhance_adaptive(img)  # adaptive process tif img
            # img_adpt = enhance_adaptive(img_adpt)  # adaptive process tif img
            # img_adpt = img
            # save_tif(save_adpting_path, img_adpt)
            #
            # # # ############ read tif and adoptive ################
            # # # #################################################
            #
            # # # #################################################
            # # # # ######### tif to mesh, tif, point cloud #########
            #
            z, x, y = img_adpt.shape
            #
            # point_cloud_img = convert_tiff_to_point_cloud(img_adpt, threshold=threshold) # tif img to point cloud with threshold; 50 for ld,mito 25, er 20; neclear 35; lysosome
            # # # point_cloud_img, mask, corrected = convert_tiff_to_point_cloud(
            # # #     img_adpt, threshold=threshold)  # tif img to point cloud with threshold; 50 for ld,mito 25, er 20; neclear 35; lysosome
            # #
            # #
            # #
            # # # point_cloud_img = convert_tiff_to_point_cloud(img_adpt)  # tif img to point cloud with threshold; 50 for ld,mito 25, er 20; neclear 35; lysosome
            # scale = 1
            # save_pcd2txt(point_cloud_img, save_txt_path)
            # tif_img = convert_point_cloud_to_tiff_fixed_size(point_cloud_img, save_mask_path, fixed_size=(x, y, z),
            #                                                  scale_factor=scale)  # point cloud to mat and save tif img
            # tif_img = fill_closed_regions(tif_img, hole_size_threshold=1000)
            #
            # tif_img = remove_small_noise_3d(
            #     tif_img,
            #     min_size=8,
            #     connectivity=3,
            #     smooth_method="median",  # 想更圆滑改 "gaussian"，不圆滑改 None
            #     # gaussian_sigma=1.0,
            #     # morph_radius=1.0,
            #     median_size=3,
            # )
            #
            # save_tif(save_mask_path, tif_img)
            

            # # 只保留最大 3D component，并填充内部
            # tif_img_lcc, tif_img = make_largest_filled_convex_nucleus_mask(
            #     tif_img,
            #     max_val=np.max(tif_img)
            # )

            # save_tif(save_mask_path, tif_img)

            # tif_img = read_tif(save_mask_path)


            # tif_img_cov = convex_hull_stack_by_slice(
            #     tif_img,
            #     max_val=np.max(tif_img)
            # )

            # save_tif(save_mask_cov_path, tif_img_cov)

            # print(f"Saved convex hull mask: {save_mask_cov_path}")

            ######### tif to mesh, tif, point cloud #########
            #################################################

            tif_img = read_tif(save_mask_path)
            mask = (tif_img != 0)

            mask_bool = mask > 0
            masked_image = img_adpt * mask_bool

            points = convert_image_to_point_cloud(
                masked_image,
                scale_factor=1.0
            )

            print(points.shape)
            save_pcd2txt(points, save_txt_path)
            #
            #
            #
            #
            # if not os.path.exists(save_obj_path):
            #     tif_img = read_tif(save_mask_path)
            #     mask = (tif_img != 0)
            #
            #     mesh = voxel_mask_to_smoothed_mesh(
            #         mask,
            #         voxel_size_zyx=(1, 1, 1),
            #         mesh_smooth_iterations=6,
            #     )
            #
            #     save_obj_from_zyx_mesh(
            #         save_obj_path,
            #         mesh["verts_smooth_zyx"],
            #         mesh["faces"],
            #     )
            #
            #
            #
            # else:
            #     print(f"Skipping mesh generation; using existing OBJ: {save_obj_path}")
            #
            # mesh = read_mesh(save_obj_path, iterations=1)  # load mesh with smooth
            #
            # pcd = mesh_2_pcd(mesh)
            # save_pcd2txt(pcd, save_pcd_path)

            




if __name__ == '__main__':
    main()

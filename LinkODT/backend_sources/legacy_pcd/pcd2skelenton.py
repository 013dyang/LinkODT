import os
from pathlib import Path
import tifffile
import cv2
import numpy as np
import open3d as o3d
if not hasattr(np, "Infinity"):
    np.Infinity = np.inf
from skimage import measure
from scipy import ndimage as ndi
from skimage import morphology
from img2pcd_new import read_tif, save_pcd2txt
from scipy import ndimage as ndi
from skimage import morphology
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


RUN_MASK_SKELETON_PIPELINE_AFTER_CONE_SCORE = True

# DBSCAN components with fewer than this many PCD points are treated as noise.
# Set to 1 to keep every non-noise component.
PCD_MIN_INSTANCE_POINTS = 20
PCD_INSTANCE_DBSCAN_EPS = 1.8
PCD_INSTANCE_DBSCAN_MIN_SAMPLES = 1


def remove_skeleton_too_close_to_pointcloud(
    pcd,
    skeleton,
    min_surface_distance=1.0,
    k=1,
    return_debug=False,
):
    """
    Delete all skeleton points that are too close to the point-cloud surface.

    Parameters
    ----------
    pcd : np.ndarray
        Shape = (N, 3) or (N, 6)
        Columns:
            x,y,z
        or:
            x,y,z,nx,ny,nz

    skeleton : np.ndarray
        Shape = (M, 3) or (M, 6)
        Columns:
            x,y,z
        or:
            x,y,z,radius,degree,type

    min_surface_distance : float
        If distance from skeleton point to nearest surface point is smaller than this,
        delete this skeleton point.

    k : int
        Number of nearest surface points.
        k=1 uses nearest distance.
        k>1 uses mean distance of k nearest points.

    return_debug : bool
        If True, return refined_skeleton, keep_mask, debug.

    Returns
    -------
    refined_skeleton : np.ndarray
    """

    pcd = np.asarray(pcd, dtype=np.float32)
    skeleton = np.asarray(skeleton, dtype=np.float32)

    if pcd.shape[1] < 3:
        raise ValueError("pcd must have at least 3 columns: x,y,z")

    if skeleton.shape[1] < 3:
        raise ValueError("skeleton must have at least 3 columns: x,y,z")

    surface_points = pcd[:, :3]
    skel_points = skeleton[:, :3]

    if len(surface_points) == 0:
        raise ValueError("pcd has no points.")

    if len(skel_points) == 0:
        raise ValueError("skeleton has no points.")

    k = min(k, len(surface_points))

    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm="auto",
        n_jobs=-1
    )
    nn.fit(surface_points)

    dists, idx = nn.kneighbors(skel_points)

    if k == 1:
        surface_dist = dists[:, 0]
    else:
        surface_dist = dists.mean(axis=1)

    remove_mask = surface_dist < min_surface_distance
    keep_mask = ~remove_mask

    refined_skeleton = skeleton[keep_mask]

    print("Skeleton before:", skeleton.shape[0])
    print("Removed too-close skeleton points:", int(remove_mask.sum()))
    print("Skeleton after:", refined_skeleton.shape[0])
    print("min surface distance:", float(surface_dist.min()))
    print("median surface distance:", float(np.median(surface_dist)))
    print("max surface distance:", float(surface_dist.max()))

    if return_debug:
        debug = {
            "surface_dist": surface_dist,
            "nearest_indices": idx,
            "remove_mask": remove_mask,
            "keep_mask": keep_mask,
        }
        return refined_skeleton, keep_mask, debug

    return refined_skeleton


def remove_skeleton_outside_pointcloud(
    pcd,
    skeleton,
    k=16,
    outside_margin=0.0,
    outside_vote_threshold=0.7,
    normals_are_outward=True,
    return_debug=False,
):
    """
    Remove skeleton points that are outside the point-cloud surface.

    Input
    -----
    pcd:
        N x 6
        x,y,z,nx,ny,nz

    skeleton:
        M x 3 or M x 6
        x,y,z
        or
        x,y,z,radius,degree,type

    Output
    ------
    refined_skeleton:
        skeleton after deleting outside points
    """

    pcd = np.asarray(pcd, dtype=np.float32)
    skeleton = np.asarray(skeleton, dtype=np.float32)

    if pcd.shape[1] < 6:
        raise ValueError("pcd must have 6 columns: x,y,z,nx,ny,nz")

    if skeleton.shape[1] < 3:
        raise ValueError("skeleton must have at least 3 columns: x,y,z")

    surface_points = pcd[:, :3]
    normals = pcd[:, 3:6]
    skel_points = skeleton[:, :3]

    # normalize normals
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (normal_norm + 1e-8)

    if not normals_are_outward:
        normals = -normals

    # k 不能超过点云数量
    k = min(k, len(surface_points))

    # nearest neighbors
    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm="auto",
        n_jobs=-1
    )

    nn.fit(surface_points)

    dists, idx = nn.kneighbors(skel_points)

    neighbor_points = surface_points[idx]  # M,k,3
    neighbor_normals = normals[idx]        # M,k,3

    # 对于外法向量：
    # 如果 skeleton 点在表面外侧，dot(P - Q, n) > 0
    # 如果 skeleton 点在表面内侧，dot(P - Q, n) < 0
    vec = skel_points[:, None, :] - neighbor_points
    signed = np.sum(vec * neighbor_normals, axis=2)

    outside_votes = signed > outside_margin
    outside_ratio = outside_votes.mean(axis=1)

    remove_mask = outside_ratio >= outside_vote_threshold
    keep_mask = ~remove_mask

    refined_skeleton = skeleton[keep_mask]

    print("Skeleton before:", skeleton.shape[0])
    print("Removed outside skeleton points:", int(remove_mask.sum()))
    print("Skeleton after:", refined_skeleton.shape[0])

    if return_debug:
        debug = {
            "outside_ratio": outside_ratio,
            "signed": signed,
            "nearest_distance": dists[:, 0],
            "remove_mask": remove_mask,
            "keep_mask": keep_mask,
        }
        return refined_skeleton, keep_mask, debug

    return refined_skeleton

def add_skeleton_points_for_small_pcd_components(
    pcd,
    skeleton,
    dbscan_eps=2.0,
    dbscan_min_samples=10,
    max_component_points=1000,
    min_component_points=20,
    bbox_margin=1.0,
    add_radius=True,
):
    """
    DBSCAN point cloud components.
    For components smaller than max_component_points, check whether skeleton exists inside.
    If no skeleton point is inside the component bbox, add one centroid point to skeleton.

    Parameters
    ----------
    pcd : np.ndarray
        Point cloud.
        Shape can be:
            (N, 3): x,y,z
            (N, 6): x,y,z,nx,ny,nz

    skeleton : np.ndarray
        Original skeleton.
        Shape can be:
            (M, 3): x,y,z
            (M, 6): x,y,z,radius,degree,type

    dbscan_eps : float
        DBSCAN eps. If coordinates are voxel-index units, try 1.5~3.0.

    dbscan_min_samples : int
        DBSCAN min_samples.

    max_component_points : int
        Only components with point number < max_component_points will be checked.

    min_component_points : int
        Ignore very tiny components smaller than this number.

    bbox_margin : float
        Expand component bounding box by this margin when checking skeleton inside.

    add_radius : bool
        If skeleton has 6 columns, estimate radius from component point cloud.

    Returns
    -------
    new_skeleton : np.ndarray
        Skeleton after adding missing center points.

    added_points : np.ndarray
        Added skeleton points.
        Shape = (K, 3), x,y,z

    labels : np.ndarray
        DBSCAN labels for pcd.

    info : list of dict
        Debug information for each checked component.
    """

    pcd = np.asarray(pcd, dtype=np.float32)
    skeleton = np.asarray(skeleton, dtype=np.float32)

    if pcd.shape[1] < 3:
        raise ValueError("pcd must have at least 3 columns: x,y,z")

    if skeleton.shape[1] < 3:
        raise ValueError("skeleton must have at least 3 columns: x,y,z")

    pcd_points = pcd[:, :3]
    skel_points = skeleton[:, :3]

    print("Running DBSCAN...")
    labels = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        n_jobs=-1
    ).fit_predict(pcd_points)

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]

    added_rows = []
    added_points = []
    info = []

    for lab in unique_labels:
        comp_mask = labels == lab
        comp_points = pcd_points[comp_mask]
        n_points = comp_points.shape[0]

        if n_points < min_component_points:
            continue

        if n_points >= max_component_points:
            continue

        bbox_min = comp_points.min(axis=0) - bbox_margin
        bbox_max = comp_points.max(axis=0) + bbox_margin

        inside_bbox = np.all(
            (skel_points >= bbox_min) & (skel_points <= bbox_max),
            axis=1
        )

        has_skeleton = np.any(inside_bbox)

        if has_skeleton:
            info.append({
                "label": int(lab),
                "n_points": int(n_points),
                "status": "has_skeleton",
                "added": False,
            })
            continue

        # Add centroid
        centroid = comp_points.mean(axis=0)

        if skeleton.shape[1] >= 6:
            row = np.zeros((skeleton.shape[1],), dtype=np.float32)

            row[0:3] = centroid

            if add_radius:
                radius = np.median(
                    np.linalg.norm(comp_points - centroid, axis=1)
                )
            else:
                radius = 0.0

            row[3] = radius   # radius
            row[4] = 0        # degree = 0
            row[5] = 1        # type = 1, isolated center / endpoint
        else:
            row = centroid.astype(np.float32)

        added_rows.append(row)
        added_points.append(centroid)

        info.append({
            "label": int(lab),
            "n_points": int(n_points),
            "status": "added_center",
            "added": True,
            "centroid": centroid.tolist(),
        })

    if len(added_rows) > 0:
        added_rows = np.vstack(added_rows).astype(np.float32)
        new_skeleton = np.vstack([skeleton, added_rows])
        added_points = np.vstack(added_points).astype(np.float32)
    else:
        new_skeleton = skeleton.copy()
        added_points = np.zeros((0, 3), dtype=np.float32)

    print("Point cloud components:", len(unique_labels))
    print("Added skeleton points:", len(added_points))
    print("Skeleton before:", skeleton.shape)
    print("Skeleton after:", new_skeleton.shape)

    return new_skeleton, added_points, labels, info

def inward_shift_point_cloud(
    pcd,
    inward_distance=1.0,
    normals_are_outward=True,
):
    """
    Push surface point cloud inward along the opposite direction of normals.

    Input pcd columns:
        x,y,z,nx,ny,nz

    Return columns:
        inner_x,inner_y,inner_z,nx,ny,nz
    """

    pcd = np.asarray(pcd, dtype=np.float32)

    points = pcd[:, :3]
    normals = pcd[:, 3:6]

    # normalize normals
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norm + 1e-8)

    if normals_are_outward:
        inner_points = points - inward_distance * normals
    else:
        inner_points = points + inward_distance * normals

    inner_pcd = np.concatenate(
        [inner_points, normals],
        axis=1
    )

    return inner_pcd.astype(np.float32)

def skeleton_zyx_to_xyz_for_view(skeleton_table_zyx):
    """
    Input skeleton_table:
        z,y,x,radius,degree,type

    Output:
        x,y,z,radius,degree,type
    """
    table_xyz = skeleton_table_zyx.copy()

    table_xyz[:, 0] = skeleton_table_zyx[:, 2]  # x
    table_xyz[:, 1] = skeleton_table_zyx[:, 1]  # y
    table_xyz[:, 2] = skeleton_table_zyx[:, 0]  # z

    return table_xyz


def skeleton_to_table_zyx(
    mask,
    voxel_size_zyx=(1, 1, 1),
    min_object_size=20,
    do_small_closing=False,
    physical=True,
):
    """
    Extract raw 3D skeleton and return a matrix.

    Returns
    -------
    skeleton_table : np.ndarray
        Columns:
        z, y, x, radius, degree, type

        type:
        1 = endpoint
        2 = regular skeleton point
        3 = junction

    skel : np.ndarray
        Binary skeleton volume.

    dist_map : np.ndarray
        Distance transform map.

    degree : np.ndarray
        Skeleton degree map.
    """

    mask = (mask != 0)

    mask_clean = morphology.remove_small_objects(
        mask,
        min_size=min_object_size
    )

    if do_small_closing:
        mask_clean = morphology.binary_closing(
            mask_clean,
            morphology.ball(1)
        )

    dist_map = ndi.distance_transform_edt(
        mask_clean,
        sampling=voxel_size_zyx
    )

    try:
        skel = morphology.skeletonize(mask_clean, method="lee")
    except TypeError:
        skel = morphology.skeletonize(mask_clean)

    skel = skel.astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    neighbor_count = ndi.convolve(
        skel.astype(np.int32),
        kernel,
        mode="constant",
        cval=0
    )

    degree = np.zeros_like(neighbor_count, dtype=np.int32)
    degree[skel] = neighbor_count[skel]

    skel_idx_zyx = np.argwhere(skel).astype(np.float32)

    if skel_idx_zyx.shape[0] == 0:
        raise ValueError("No skeleton points found. Check your mask.")

    z_idx = skel_idx_zyx[:, 0].astype(np.int64)
    y_idx = skel_idx_zyx[:, 1].astype(np.int64)
    x_idx = skel_idx_zyx[:, 2].astype(np.int64)

    radius = dist_map[z_idx, y_idx, x_idx]
    deg = degree[z_idx, y_idx, x_idx]

    point_type = np.zeros_like(deg, dtype=np.int32)
    point_type[deg == 1] = 1
    point_type[deg == 2] = 2
    point_type[deg >= 3] = 3
    point_type[deg == 0] = 1

    if physical:
        scale = np.asarray(voxel_size_zyx, dtype=np.float32)
        coords_zyx = skel_idx_zyx * scale
    else:
        coords_zyx = skel_idx_zyx

    skeleton_table = np.column_stack([
        coords_zyx[:, 2],   # z
        coords_zyx[:, 1],   # y
        coords_zyx[:, 0],   # x
        radius,             # radius
        deg,                # degree
        point_type,         # type
    ])

    return skeleton_table, skel, dist_map, degree


def smooth_skeleton_graph_laplacian(
    skeleton,
    connect_radius=1.8,
    iterations=10,
    alpha=0.35,
    protect_endpoints=True,
    protect_junctions=True,
    max_step_per_iter=1.0,
    recompute_degree=True,
    return_debug=False,
):
    """
    Smooth skeleton coordinates by graph Laplacian smoothing.

    Input
    -----
    skeleton : np.ndarray
        Shape = (N, 3) or (N, >=6)

        If 3 columns:
            x,y,z

        If >=6 columns:
            x,y,z,radius,degree,type

            degree column = column 4
            type column   = column 5
            type:
                1 = endpoint / isolated center
                2 = regular
                3 = junction

    Parameters
    ----------
    connect_radius : float
        Radius used to build skeleton neighbor graph.
        For voxel-index skeleton, try:
            1.8  : 26-neighborhood-like
            2.2  : allow small gaps
            2.5  : stronger connection, but may connect nearby wrong branches

    iterations : int
        Number of smoothing iterations.

    alpha : float
        Smoothing strength per iteration.
        0.2 = mild
        0.35 = recommended
        0.5 = stronger

    protect_endpoints : bool
        If True, endpoint / degree<=1 points do not move.

    protect_junctions : bool
        If True, junction / degree>=3 points do not move.
        This helps preserve branch topology.

    max_step_per_iter : float or None
        Maximum movement allowed for one point per iteration.
        Prevents large jumps.

    recompute_degree : bool
        If True, degree is recomputed from neighbor graph.
        If skeleton has >=6 columns, degree/type columns will be updated.

    return_debug : bool
        If True, return smoothed_skeleton, debug.

    Returns
    -------
    smoothed_skeleton : np.ndarray
        Same shape as input skeleton.
        First three columns are smoothed x,y,z.

    debug : dict, optional
        Contains neighbor list, degree, movable mask.
    """

    skeleton = np.asarray(skeleton, dtype=np.float32)

    if skeleton.ndim != 2 or skeleton.shape[1] < 3:
        raise ValueError("skeleton must be 2D array with at least 3 columns: x,y,z")

    points = skeleton[:, :3].copy()
    n_points = points.shape[0]

    if n_points == 0:
        raise ValueError("Empty skeleton.")

    if n_points == 1:
        if return_debug:
            return skeleton.copy(), {
                "degree": np.array([0], dtype=np.int32),
                "movable": np.array([False]),
                "neighbors": [[]],
            }
        return skeleton.copy()

    # --------------------------------------------------
    # 1. Build radius-neighbor graph
    # --------------------------------------------------
    nn = NearestNeighbors(
        radius=connect_radius,
        algorithm="auto",
        n_jobs=-1
    )
    nn.fit(points)

    neighbor_indices = nn.radius_neighbors(
        points,
        radius=connect_radius,
        return_distance=False
    )

    neighbors = []
    degree = np.zeros(n_points, dtype=np.int32)

    for i, inds in enumerate(neighbor_indices):
        inds = np.asarray(inds, dtype=np.int32)

        # remove self
        inds = inds[inds != i]

        neighbors.append(inds)
        degree[i] = len(inds)

    # --------------------------------------------------
    # 2. Decide which points can move
    # --------------------------------------------------
    movable = np.ones(n_points, dtype=bool)

    if protect_endpoints:
        movable[degree <= 1] = False

    if protect_junctions:
        movable[degree >= 3] = False

    # If skeleton already has type column, use it too
    if skeleton.shape[1] >= 6:
        old_degree = skeleton[:, 4].astype(np.int32)
        old_type = skeleton[:, 5].astype(np.int32)

        if protect_endpoints:
            movable[(old_type == 1) | (old_degree <= 1)] = False

        if protect_junctions:
            movable[(old_type == 3) | (old_degree >= 3)] = False

    # isolated points never move
    movable[degree == 0] = False

    # --------------------------------------------------
    # 3. Laplacian smoothing
    # --------------------------------------------------
    smoothed = points.copy()

    for it in range(iterations):
        new_points = smoothed.copy()

        for i in range(n_points):
            if not movable[i]:
                continue

            nei = neighbors[i]

            if len(nei) == 0:
                continue

            nei_mean = smoothed[nei].mean(axis=0)

            target = smoothed[i] + alpha * (nei_mean - smoothed[i])
            step = target - smoothed[i]

            if max_step_per_iter is not None:
                step_norm = np.linalg.norm(step)
                if step_norm > max_step_per_iter:
                    step = step / (step_norm + 1e-8) * max_step_per_iter

            new_points[i] = smoothed[i] + step

        smoothed = new_points

    # --------------------------------------------------
    # 4. Build output table
    # --------------------------------------------------
    smoothed_skeleton = skeleton.copy()
    smoothed_skeleton[:, :3] = smoothed

    # Update degree/type if table has those columns
    if recompute_degree and smoothed_skeleton.shape[1] >= 6:
        smoothed_skeleton[:, 4] = degree

        point_type = np.zeros_like(degree, dtype=np.int32)
        point_type[degree <= 1] = 1
        point_type[degree == 2] = 2
        point_type[degree >= 3] = 3

        smoothed_skeleton[:, 5] = point_type

    print("Skeleton smoothing finished.")
    print("Skeleton points:", n_points)
    print("Movable points:", int(movable.sum()))
    print("Protected points:", int((~movable).sum()))
    print("Degree 0:", int(np.sum(degree == 0)))
    print("Degree 1:", int(np.sum(degree == 1)))
    print("Degree 2:", int(np.sum(degree == 2)))
    print("Degree >=3:", int(np.sum(degree >= 3)))

    if return_debug:
        debug = {
            "neighbors": neighbors,
            "degree": degree,
            "movable": movable,
            "connect_radius": connect_radius,
            "iterations": iterations,
            "alpha": alpha,
        }
        return smoothed_skeleton, debug

    return smoothed_skeleton


def skeleton_to_instance(
    smoothed_skeleton,
    connect_radius=1.8,
    degree_col=4,
    color_seed=42,
    background_color=(255, 255, 255),
    red_branch_rounds=20,
    min_direction_cos=0.2,
):
    """
    Split a smoothed skeleton into endpoint-to-endpoint instances.

    Valid endpoint paths:
        degree 1 -> degree 1
        degree 1 -> degree 2 -> ... -> degree 2 -> degree 1

    Invalid endpoint paths stay id=0 with the background color. Degree-0 points
    are assigned directly as local connected components.

    Input
    -----
    smoothed_skeleton : np.ndarray
        Columns are expected to include:
            x,y,z,radius,degree,type

    Output
    ------
    instance_skeleton : np.ndarray
        Columns:
            x,y,z,instance_id,r,g,b
    """

    skeleton = np.asarray(smoothed_skeleton, dtype=np.float32)

    if skeleton.ndim != 2 or skeleton.shape[1] <= degree_col:
        raise ValueError("smoothed_skeleton must contain x,y,z and degree columns.")

    n_points = skeleton.shape[0]
    if n_points == 0:
        return np.zeros((0, 7), dtype=np.float32)

    points = skeleton[:, :3]
    degree = np.rint(skeleton[:, degree_col]).astype(np.int32)

    nn = NearestNeighbors(
        radius=connect_radius,
        algorithm="auto",
        n_jobs=-1
    )
    nn.fit(points)
    radius_neighbors = nn.radius_neighbors(
        points,
        radius=connect_radius,
        return_distance=False
    )

    neighbors = []
    for i, inds in enumerate(radius_neighbors):
        inds = np.asarray(inds, dtype=np.int32)
        inds = inds[inds != i]

        if len(inds) > 1:
            dists = np.linalg.norm(points[inds] - points[i], axis=1)
            inds = inds[np.argsort(dists)]

        neighbors.append(inds)

    processed = np.zeros(n_points, dtype=bool)
    instance_ids = np.zeros(n_points, dtype=np.int32)
    endpoint_indices = np.where(degree == 1)[0]
    rng = np.random.default_rng(color_seed)
    color_map = {
        0: np.asarray(background_color, dtype=np.uint8)
    }
    instance_count = 0

    def add_instance(indices):
        nonlocal instance_count
        indices = np.asarray(indices, dtype=np.int32)
        indices = indices[instance_ids[indices] == 0]
        if len(indices) == 0:
            return False

        instance_count += 1
        instance_ids[indices] = instance_count
        processed[indices] = True
        color_map[instance_count] = rng.integers(32, 256, size=3, dtype=np.uint8)
        return True

    # Degree-0 points are isolated from the skeleton graph, but nearby degree-0
    # points should share one instance/color instead of being colored one by one.
    isolated_indices = np.where(degree == 0)[0]
    isolated_set = set(int(idx) for idx in isolated_indices)
    degree0_component_count = 0
    for seed_idx in isolated_indices:
        seed_idx = int(seed_idx)
        if processed[seed_idx]:
            continue

        component = []
        stack = [seed_idx]
        processed[seed_idx] = True

        while len(stack) > 0:
            curr_idx = stack.pop()
            component.append(curr_idx)

            for next_idx in neighbors[curr_idx]:
                next_idx = int(next_idx)
                if next_idx not in isolated_set or processed[next_idx]:
                    continue

                processed[next_idx] = True
                stack.append(next_idx)

        add_instance(component)
        degree0_component_count += 1

    def choose_next_by_direction(prev_idx, curr_idx, candidates):
        candidates = np.asarray(candidates, dtype=np.int32)
        direction = points[curr_idx] - points[prev_idx]
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-8:
            dists = np.linalg.norm(points[candidates] - points[curr_idx], axis=1)
            return int(candidates[np.argmin(dists)])

        direction = direction / direction_norm
        candidate_vectors = points[candidates] - points[curr_idx]
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        valid = candidate_norms > 1e-8

        if not np.any(valid):
            return int(candidates[0])

        scores = np.full(len(candidates), -np.inf, dtype=np.float32)
        scores[valid] = (candidate_vectors[valid] / candidate_norms[valid, None]) @ direction
        dists = np.linalg.norm(points[candidates] - points[curr_idx], axis=1)
        order = np.lexsort((dists, -scores))
        return int(candidates[order[0]])

    def filter_candidates_by_direction(prev_idx, curr_idx, candidates, min_cos=None):
        candidates = np.asarray(candidates, dtype=np.int32)
        if len(candidates) == 0:
            return []

        if min_cos is None:
            min_cos = min_direction_cos

        direction = points[curr_idx] - points[prev_idx]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            return candidates.tolist()

        direction = direction / direction_norm
        candidate_vectors = points[candidates] - points[curr_idx]
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        valid = candidate_norms > 1e-8

        if not np.any(valid):
            return candidates.tolist()

        scores = np.full(len(candidates), -np.inf, dtype=np.float32)
        scores[valid] = (candidate_vectors[valid] / candidate_norms[valid, None]) @ direction

        filtered = candidates[scores >= min_cos]
        return filtered.astype(np.int32).tolist()

    def find_first_step(start_idx):
        # Only use graph neighbors within connect_radius. Do not jump to a
        # globally nearest point, because that can incorrectly bridge branches.
        local_degree2_candidates = [
            int(idx)
            for idx in neighbors[start_idx]
            if degree[int(idx)] == 2 and not processed[int(idx)]
        ]
        if len(local_degree2_candidates) > 0:
            return local_degree2_candidates[0]

        local_endpoint_candidates = [
            int(idx)
            for idx in neighbors[start_idx]
            if degree[int(idx)] == 1 and not processed[int(idx)]
        ]
        if len(local_endpoint_candidates) > 0:
            return local_endpoint_candidates[0]

        return None

    for start_idx in endpoint_indices:
        start_idx = int(start_idx)
        if processed[start_idx]:
            continue

        first_idx = find_first_step(start_idx)
        if first_idx is None:
            processed[start_idx] = True
            continue

        path = [start_idx, first_idx]
        path_set = {start_idx, first_idx}
        prev_idx = start_idx
        curr_idx = first_idx
        end_idx = first_idx if degree[first_idx] == 1 else None
        invalid_path = False

        for _ in range(n_points):
            if end_idx is not None:
                break

            bad_degree_neighbors = []
            candidates = []
            for idx in neighbors[curr_idx]:
                idx = int(idx)
                if idx == prev_idx or idx in path_set:
                    continue

                if degree[idx] not in (1, 2):
                    bad_degree_neighbors.append(idx)
                    continue

                if processed[idx]:
                    continue

                candidates.append(idx)

            if len(bad_degree_neighbors) > 0:
                invalid_path = True
                break

            if len(candidates) == 0:
                invalid_path = True
                break

            degree2_candidates = [idx for idx in candidates if degree[idx] == 2]
            if len(degree2_candidates) > 0:
                next_idx = choose_next_by_direction(prev_idx, curr_idx, degree2_candidates)
                path.append(next_idx)
                path_set.add(next_idx)
                prev_idx, curr_idx = curr_idx, next_idx
                continue

            endpoint_candidates = [idx for idx in candidates if degree[idx] == 1]
            if len(endpoint_candidates) > 0:
                end_idx = choose_next_by_direction(prev_idx, curr_idx, endpoint_candidates)
                path.append(end_idx)
                break

            break

        path_indices = np.asarray(path, dtype=np.int32)

        if invalid_path or end_idx is None:
            processed[path_indices] = True
            continue

        add_instance(path_indices)

    junction_indices = np.where(degree > 2)[0]
    junction_branch_count = 0
    junction_nearest_assign_count = 0
    junction_branch_fail_count = 0

    def trace_junction_branch(junction_idx, first_idx):
        path = [junction_idx, first_idx]
        path_set = {junction_idx, first_idx}

        if degree[first_idx] == 1:
            return path

        if degree[first_idx] != 2:
            return None

        prev_idx = junction_idx
        curr_idx = first_idx

        for _ in range(n_points):
            candidates = []

            for idx in neighbors[curr_idx]:
                idx = int(idx)
                if idx == prev_idx or idx in path_set:
                    continue

                if instance_ids[idx] != 0:
                    continue

                if degree[idx] in (1, 2):
                    candidates.append(idx)

            if len(candidates) == 0:
                return None

            degree2_candidates = [idx for idx in candidates if degree[idx] == 2]
            degree2_candidates = filter_candidates_by_direction(prev_idx, curr_idx, degree2_candidates)
            if len(degree2_candidates) > 0:
                next_idx = choose_next_by_direction(prev_idx, curr_idx, degree2_candidates)
                path.append(next_idx)
                path_set.add(next_idx)
                prev_idx, curr_idx = curr_idx, next_idx
                continue

            endpoint_candidates = [idx for idx in candidates if degree[idx] == 1]
            endpoint_candidates = filter_candidates_by_direction(prev_idx, curr_idx, endpoint_candidates)
            if len(endpoint_candidates) > 0:
                end_idx = choose_next_by_direction(prev_idx, curr_idx, endpoint_candidates)
                path.append(end_idx)
                return path

            return None

        return None

    def assign_point_to_direction_matched_instance(point_idx):
        if instance_ids[point_idx] != 0:
            return False

        local_labeled = [
            int(idx)
            for idx in neighbors[point_idx]
            if instance_ids[int(idx)] > 0
        ]

        if len(local_labeled) == 0:
            labeled_indices = np.where(instance_ids > 0)[0]
            if len(labeled_indices) == 0:
                return False

            dists = np.linalg.norm(points[labeled_indices] - points[point_idx], axis=1)
            nearest_idx = int(labeled_indices[np.argmin(dists)])
            instance_ids[point_idx] = instance_ids[nearest_idx]
            processed[point_idx] = True
            return True

        best_labeled_idx = None
        best_score = -np.inf
        best_dist = np.inf

        for labeled_idx in local_labeled:
            labeled_id = instance_ids[labeled_idx]
            incoming = points[point_idx] - points[labeled_idx]
            incoming_norm = np.linalg.norm(incoming)
            if incoming_norm < 1e-8:
                continue

            incoming = incoming / incoming_norm
            continuation_indices = [
                int(idx)
                for idx in neighbors[point_idx]
                if (
                    int(idx) != labeled_idx
                    and (
                        instance_ids[int(idx)] == 0
                        or instance_ids[int(idx)] != labeled_id
                    )
                )
            ]

            score = -1.0
            for cont_idx in continuation_indices:
                continuation = points[cont_idx] - points[point_idx]
                continuation_norm = np.linalg.norm(continuation)
                if continuation_norm < 1e-8:
                    continue

                continuation = continuation / continuation_norm
                score = max(score, float(np.dot(incoming, continuation)))

            dist = float(np.linalg.norm(points[labeled_idx] - points[point_idx]))
            if score > best_score or (np.isclose(score, best_score) and dist < best_dist):
                best_score = score
                best_dist = dist
                best_labeled_idx = labeled_idx

        if best_labeled_idx is None:
            dists = np.linalg.norm(points[local_labeled] - points[point_idx], axis=1)
            best_labeled_idx = int(np.asarray(local_labeled, dtype=np.int32)[np.argmin(dists)])

        instance_ids[point_idx] = instance_ids[best_labeled_idx]
        processed[point_idx] = True
        return True

    def assign_junction_to_direction_matched_instance(junction_idx):
        if instance_ids[junction_idx] != 0:
            return False

        return assign_point_to_direction_matched_instance(junction_idx)

    for junction_idx in junction_indices:
        junction_idx = int(junction_idx)

        attempted_starts = set()

        while True:
            degree2_starts = [
                int(idx)
                for idx in neighbors[junction_idx]
                if (
                    int(idx) not in attempted_starts
                    and instance_ids[int(idx)] == 0
                    and degree[int(idx)] == 2
                )
            ]
            endpoint_starts = [
                int(idx)
                for idx in neighbors[junction_idx]
                if (
                    int(idx) not in attempted_starts
                    and instance_ids[int(idx)] == 0
                    and degree[int(idx)] == 1
                )
            ]

            branch_starts = degree2_starts + endpoint_starts
            if len(branch_starts) == 0:
                break

            first_idx = branch_starts[0]
            attempted_starts.add(first_idx)

            path = trace_junction_branch(junction_idx, first_idx)
            if path is None:
                junction_branch_fail_count += 1
                continue

            branch_path = np.asarray(path[1:], dtype=np.int32)
            if add_instance(branch_path):
                junction_branch_count += 1

    red_connection_assign_count = 0
    white_branch_count = 0

    def compute_unlabeled_connection_mask():
        unlabeled_mask = instance_ids == 0
        unlabeled_connected_to_labeled = np.zeros(n_points, dtype=bool)

        for i in np.where(unlabeled_mask)[0]:
            i = int(i)
            if np.any(instance_ids[neighbors[i]] > 0):
                unlabeled_connected_to_labeled[i] = True

        return unlabeled_mask, unlabeled_connected_to_labeled

    def assign_unlabeled_connections_to_nearest_instance():
        _, red_unlabeled = compute_unlabeled_connection_mask()
        red_indices = np.where(red_unlabeled)[0]

        if len(red_indices) == 0:
            return np.zeros((0,), dtype=np.int32)

        assigned = []

        for idx in red_indices:
            idx = int(idx)
            if assign_point_to_direction_matched_instance(idx):
                assigned.append(idx)

        return np.asarray(assigned, dtype=np.int32)

    def trace_white_branch(boundary_idx, first_white_idx):
        path = [boundary_idx, first_white_idx]
        path_set = {boundary_idx, first_white_idx}
        prev_idx = boundary_idx
        curr_idx = first_white_idx

        for _ in range(n_points):
            loop_neighbors = [
                int(idx)
                for idx in neighbors[curr_idx]
                if int(idx) != prev_idx and int(idx) in path_set
            ]
            if len(loop_neighbors) > 0:
                return path

            labeled_neighbors = [
                int(idx)
                for idx in neighbors[curr_idx]
                if int(idx) != prev_idx and instance_ids[int(idx)] > 0
            ]
            labeled_neighbors = filter_candidates_by_direction(prev_idx, curr_idx, labeled_neighbors)
            if len(labeled_neighbors) > 0:
                return path

            red_neighbors = [
                int(idx)
                for idx in neighbors[curr_idx]
                if (
                    int(idx) != prev_idx
                    and int(idx) not in path_set
                    and instance_ids[int(idx)] == 0
                    and red_unlabeled[int(idx)]
                )
            ]
            red_neighbors = filter_candidates_by_direction(prev_idx, curr_idx, red_neighbors)
            if len(red_neighbors) > 0:
                return path

            white_candidates = [
                int(idx)
                for idx in neighbors[curr_idx]
                if (
                    int(idx) != prev_idx
                    and int(idx) not in path_set
                    and instance_ids[int(idx)] == 0
                    and not red_unlabeled[int(idx)]
                )
            ]
            white_candidates = filter_candidates_by_direction(prev_idx, curr_idx, white_candidates)
            if len(white_candidates) == 0:
                return None

            next_idx = choose_next_by_direction(prev_idx, curr_idx, white_candidates)
            path.append(next_idx)
            path_set.add(next_idx)
            prev_idx, curr_idx = curr_idx, next_idx

        return None

    def add_white_branch_instance(path):
        nonlocal instance_count
        path = np.asarray(path, dtype=np.int32)
        branch_indices = path[1:]
        branch_indices = branch_indices[instance_ids[branch_indices] == 0]

        if len(branch_indices) == 0:
            return False

        instance_count += 1
        instance_ids[branch_indices] = instance_count
        processed[branch_indices] = True
        color_map[instance_count] = rng.integers(32, 256, size=3, dtype=np.uint8)
        return True

    for _ in range(red_branch_rounds):
        round_branch_count = 0
        unlabeled_mask, red_unlabeled = compute_unlabeled_connection_mask()

        boundary_indices = np.where(red_unlabeled)[0].astype(np.int32).tolist()

        for boundary_idx in boundary_indices:
            attempted_starts = set()

            while True:
                white_starts = [
                    int(idx)
                    for idx in neighbors[boundary_idx]
                    if (
                        int(idx) not in attempted_starts
                        and instance_ids[int(idx)] == 0
                        and not red_unlabeled[int(idx)]
                    )
                ]
                if len(white_starts) == 0:
                    break

                first_idx = white_starts[0]
                attempted_starts.add(first_idx)

                path = trace_white_branch(boundary_idx, first_idx)
                if path is None:
                    continue

                if add_white_branch_instance(path):
                    white_branch_count += 1
                    round_branch_count += 1

        if round_branch_count == 0:
            break

    assigned_red_indices = assign_unlabeled_connections_to_nearest_instance()
    red_connection_assign_count += int(len(assigned_red_indices))

    rgb = np.zeros((n_points, 3), dtype=np.uint8)
    for instance_id, color in color_map.items():
        rgb[instance_ids == instance_id] = color

    unlabeled_mask, _ = compute_unlabeled_connection_mask()
    rgb[unlabeled_mask] = np.asarray([255, 255, 255], dtype=np.uint8)

    instance_skeleton = np.column_stack([
        points,
        instance_ids.astype(np.float32),
        rgb.astype(np.float32)
    ])

    print("Skeleton instance segmentation finished.")
    print("Degree-0 points:", int(len(isolated_indices)))
    print("Degree-0 components:", int(degree0_component_count))
    print("Endpoint points:", int(len(endpoint_indices)))
    print("Junction-to-endpoint branches:", int(junction_branch_count))
    print("Junction-to-endpoint failed starts:", int(junction_branch_fail_count))
    print("Red connection direction-id assignments:", int(red_connection_assign_count))
    print("White branch paths:", int(white_branch_count))
    print("Instances:", int(instance_count))
    print("Assigned points:", int(np.sum(instance_ids > 0)))
    print("Unassigned points:", int(np.sum(instance_ids == 0)))

    return instance_skeleton


def load_point_cloud(txt_path, return_columns=9, scale_yz=1, scale_x=1):
    """
    加载点云数据，并根据数据列数提供说明，同时允许用户指定返回的列数。

    功能：
      - 如果数据为 (N, 3)：仅包含点坐标 (xyz)。
      - 如果数据为 (N, 6)：包含点坐标和法向量 (xyz, nx, ny, nz)。
      - 如果数据列数大于 6：包含点坐标、法向量及其他属性。
      - 可选地对 y, z 列数据乘以一个数值（例如 scale_yz=2.0）。

    :param txt_path: 点云数据的 txt 文件路径，文件以逗号分隔。
    :param return_columns: 指定返回的列数。如果为 None，则返回所有列。
    :param scale_yz: 可选参数，对 y、z 列同时乘以该数值。
    :return: NumPy 数组，包含点云数据。
    """
    # 读取数据
    mat = np.loadtxt(txt_path, delimiter=",", dtype=np.float32)
    num_points, n_cols = mat.shape
    print(mat.shape)


    if n_cols == 3:
        info = "仅包含点坐标 (x, y, z)"
    elif n_cols == 6:
        info = "包含点坐标和法向量 (x, y, z, nx, ny, nz)"
    elif n_cols > 6:
        info = "包含点坐标、法向量及其他属性"
    else:
        pass
        # raise ValueError(f"数据维度错误，期望至少 (N, 3)，但读取到 {mat.shape}")

    # print(f"加载了 {num_points} 个点，形状为 {mat.shape}（{info}）。")

    # 缩放 y 和 z
    if scale_yz is not None:
        if n_cols >= 3:  # 确保至少有 x,y,z
            mat[:, 1] *= scale_yz  # y
            mat[:, 2] *= scale_yz  # z
            print(f"已将 y 和 z 列乘以 {scale_yz}。")
        else:
            print("数据列数不足 3，无法缩放 y 和 z。")

    if scale_x is not None:
        if n_cols >= 3:  # 确保至少有 x,y,z
            mat[:, 0] *= scale_x  # y
            print(f"已将 x 列乘以 {scale_x}。")

    # 根据用户指定返回的列数
    if return_columns is not None:
        if return_columns > n_cols:
            print(f"请求返回 {return_columns} 列，但数据只有 {n_cols} 列，返回全部数据。")
            result = mat
        else:
            result = mat[:, :return_columns]
            print(f"返回数据形状: {result.shape}")
            if return_columns == 3:
                print("返回数据仅包含点坐标 (x, y, z)。")
            elif return_columns == 6:
                print("返回数据包含点坐标和法向量 (x, y, z, nx, ny, nz)。")
            elif return_columns > 6:
                print("返回数据包含点坐标、法向量及其他属性。")
    else:
        result = mat

    return result



def get_valid_skeleton_xyzrgb(skeleton_instance, background_color=(255, 255, 255)):
    """
    Required input:
        x,y,z,instance_id,r,g,b
    """
    skeleton_instance = np.asarray(skeleton_instance, dtype=np.float32)
    if skeleton_instance.ndim != 2 or skeleton_instance.shape[1] < 7:

        xyz = skeleton_instance[:, :3].astype(np.float32)
        instance_id = np.rint(skeleton_instance[:, 3]).astype(np.int32)
        rgb = np.rint(skeleton_instance[:, 3:6]).astype(np.uint8)
    

    
    else:
        xyz = skeleton_instance[:, :3].astype(np.float32)
        instance_id = np.rint(skeleton_instance[:, 3]).astype(np.int32)
        rgb = np.rint(skeleton_instance[:, 4:7]).astype(np.uint8)




    bg = np.asarray(background_color, dtype=np.uint8)
    valid = instance_id > 0
    valid &= ~np.all(rgb == bg[None, :], axis=1)
    valid &= np.any(rgb != 0, axis=1)

    if not np.any(valid):
        raise ValueError("No valid skeleton instance points found.")

    return xyz[valid], rgb[valid]

def dbscan_coarse_skeleton_xyzrgb(
    smoothed_skeleton,
    eps=1.8,
    min_samples=1,
    color_seed=123,
):
    """
    Coarse segmentation of smoothed skeleton by DBSCAN connected components.

    Input
    -----
    smoothed_skeleton : ndarray, shape (N, >=3)
        x,y,z,radius,degree,type or x,y,z,...

    Output
    ------
    coarse_skeleton_xyzrgb : ndarray, shape (N, 6)
        x,y,z,r,g,b

    Notes
    -----
    This does not save component_id.
    Each DBSCAN coarse component is represented only by one RGB color.
    """
    smoothed_skeleton = np.asarray(smoothed_skeleton, dtype=np.float32)

    if smoothed_skeleton.ndim != 2 or smoothed_skeleton.shape[1] < 3:
        raise ValueError("smoothed_skeleton must be N x >=3: x,y,z,...")

    xyz = smoothed_skeleton[:, :3].astype(np.float32)

    if len(xyz) == 0:
        raise ValueError("Empty smoothed_skeleton.")

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1,
    ).fit_predict(xyz)

    valid_labels = [lab for lab in np.unique(labels) if lab != -1]

    # 按 component 大小排序：最大的 component 先分配颜色
    valid_labels = sorted(
        valid_labels,
        key=lambda lab: int(np.sum(labels == lab)),
        reverse=True,
    )

    rng = np.random.default_rng(color_seed)

    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

    used_colors = set()

    for lab in valid_labels:
        while True:
            color = tuple(int(v) for v in rng.integers(32, 256, size=3))
            if color not in used_colors:
                used_colors.add(color)
                break

        rgb[labels == lab] = np.asarray(color, dtype=np.uint8)

    # 如果 min_samples=1，理论上不会有 noise；
    # 如果以后 min_samples>1，noise 会保持黑色。
    coarse_skeleton_xyzrgb = np.column_stack([
        xyz,
        rgb.astype(np.float32),
    ]).astype(np.float32)

    print("Coarse DBSCAN skeleton segmentation finished.")
    print("Skeleton points:", len(xyz))
    print("Coarse components:", len(valid_labels))
    print("Noise points:", int(np.sum(labels == -1)))

    return coarse_skeleton_xyzrgb


def dbscan_coarse_pcd_xyzrgb(
    pointcloud,
    eps=1.8,
    min_samples=1,
    color_seed=123,
):
    """
    Coarse segmentation of original *_pcd.txt by DBSCAN connected components.

    Input
    -----
    pointcloud : ndarray, shape (N, >=3)
        Usually:
            x,y,z,nx,ny,nz
        At least x,y,z are required.

    Output
    ------
    pcd_instance_dbscan : ndarray
        If input has normals, output columns are:
            x,y,z,nx,ny,nz,r,g,b
        If input only has xyz, output columns are:
            x,y,z,r,g,b

    Notes
    -----
    The DBSCAN component id is not saved. Each component is represented by
    one RGB color. Noise points, if min_samples > 1, are black.
    """
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError("pointcloud must be N x >=3: x,y,z,...")

    xyz = pointcloud[:, :3].astype(np.float32)

    if len(xyz) == 0:
        raise ValueError("Empty pointcloud.")

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1,
    ).fit_predict(xyz)

    valid_labels = [lab for lab in np.unique(labels) if lab != -1]
    valid_labels = sorted(
        valid_labels,
        key=lambda lab: int(np.sum(labels == lab)),
        reverse=True,
    )

    rng = np.random.default_rng(color_seed)
    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    used_colors = set()

    for lab in valid_labels:
        while True:
            color = tuple(int(v) for v in rng.integers(32, 256, size=3))
            if color not in used_colors:
                used_colors.add(color)
                break

        rgb[labels == lab] = np.asarray(color, dtype=np.uint8)

    if pointcloud.shape[1] >= 6:
        pcd_instance_dbscan = np.column_stack([
            pointcloud[:, :6],
            rgb.astype(np.float32),
        ]).astype(np.float32)
    else:
        pcd_instance_dbscan = np.column_stack([
            xyz,
            rgb.astype(np.float32),
        ]).astype(np.float32)

    print("Coarse DBSCAN PCD segmentation finished.")
    print("PCD points:", len(xyz))
    print("PCD coarse components:", len(valid_labels))
    print("PCD noise points:", int(np.sum(labels == -1)))
    print("PCD DBSCAN instance shape:", pcd_instance_dbscan.shape)

    return pcd_instance_dbscan


def filter_small_pcd_instances(
    pointcloud,
    min_instance_points,
    eps=1.8,
    min_samples=1,
    return_labels=False,
):
    """Remove small DBSCAN instances before downstream segmentation.

    A DBSCAN component is kept when its point count is greater than or equal
    to ``min_instance_points``. DBSCAN noise (label ``-1``) is always removed.
    All input columns (for example normals) are preserved.
    """
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError("pointcloud must be N x >=3: x,y,z,...")
    if len(pointcloud) == 0:
        raise ValueError("Empty pointcloud.")

    min_instance_points = int(min_instance_points)
    if min_instance_points < 1:
        raise ValueError("min_instance_points must be >= 1")

    labels = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        n_jobs=-1,
    ).fit_predict(pointcloud[:, :3])

    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    kept_labels = valid_labels[counts >= min_instance_points]
    keep_mask = np.isin(labels, kept_labels)
    filtered_pointcloud = pointcloud[keep_mask].copy()

    print("PCD instance noise filtering finished.")
    print("PCD points before:", int(len(pointcloud)))
    print("PCD instances before:", int(len(valid_labels)))
    print("Minimum instance points:", min_instance_points)
    print("Removed small instances:", int(np.sum(counts < min_instance_points)))
    print("Removed points:", int(np.sum(~keep_mask)))
    print("PCD instances after:", int(len(kept_labels)))
    print("PCD points after:", int(len(filtered_pointcloud)))

    if len(filtered_pointcloud) == 0:
        raise ValueError(
            "PCD filtering removed every point. Lower PCD_MIN_INSTANCE_POINTS "
            "or adjust the DBSCAN parameters."
        )

    if return_labels:
        return filtered_pointcloud, labels, keep_mask
    return filtered_pointcloud


def fibonacci_sphere_directions(n_directions=32):
    """
    Generate approximately uniform direction vectors on a sphere.

    These directions are used as direction bins. If many normals fall into the
    same bin, they count as one direction for a voxel.
    """
    n_directions = int(n_directions)
    if n_directions < 1:
        raise ValueError("n_directions must be >= 1")

    if n_directions == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    dirs = np.zeros((n_directions, 3), dtype=np.float32)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_directions):
        z = 1.0 - 2.0 * i / (n_directions - 1)
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        dirs[i] = [
            radius * np.cos(theta),
            radius * np.sin(theta),
            z,
        ]

    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return dirs.astype(np.float32)


def score_voxels_by_inward_normal_cones(
    pointcloud,
    voxel,
    cone_angle_degrees=20.0,
    max_cone_distance=20.0,
    n_direction_bins=32,
    normals_are_outward=True,
    batch_size=2000,
):
    """
    Score voxels by inward normal cones from surface point cloud.

    Inputs
    ------
    pointcloud:
        N x >=6, columns x,y,z,nx,ny,nz.

    voxel:
        M x >=3, columns x,y,z,(intensity...)

    Algorithm
    ---------
    For every pcd point, use the opposite normal direction as the inward cone
    axis. A voxel is hit if:

        distance(voxel, pcd_point) <= max_cone_distance
        angle(voxel - pcd_point, inward_normal) <= cone_angle_degrees

    Each inward normal is first assigned to one of n_direction_bins direction
    bins. For one voxel, repeated hits from the same direction bin count as 1.
    Therefore:

        100 hits from the same direction -> score 1
        hits from 4 different directions -> score 4

    Output
    ------
    voxel_score:
        M x 4, columns x,y,z,score
    """
    pointcloud = np.asarray(pointcloud, dtype=np.float32)
    voxel = np.asarray(voxel, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] < 6:
        raise ValueError("pointcloud must be N x >=6: x,y,z,nx,ny,nz")
    if voxel.ndim != 2 or voxel.shape[1] < 3:
        raise ValueError("voxel must be M x >=3: x,y,z,...")
    if n_direction_bins > 64:
        raise ValueError("n_direction_bins must be <= 64 because uint64 bitmask is used")
    if max_cone_distance is None or max_cone_distance <= 0:
        raise ValueError("max_cone_distance must be a positive number")

    surface_xyz = pointcloud[:, :3].astype(np.float32)
    normals = pointcloud[:, 3:6].astype(np.float32)
    voxel_xyz = voxel[:, :3].astype(np.float32)

    normal_norm = np.linalg.norm(normals, axis=1)
    valid_normals = np.isfinite(surface_xyz).all(axis=1)
    valid_normals &= np.isfinite(normals).all(axis=1)
    valid_normals &= normal_norm > 1e-8

    surface_xyz = surface_xyz[valid_normals]
    normals = normals[valid_normals]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    inward_normals = -normals if normals_are_outward else normals

    direction_centers = fibonacci_sphere_directions(n_direction_bins)
    direction_bin_ids = np.argmax(inward_normals @ direction_centers.T, axis=1).astype(np.int32)

    cos_threshold = float(np.cos(np.deg2rad(cone_angle_degrees)))
    direction_masks = np.zeros(len(voxel_xyz), dtype=np.uint64)

    voxel_nn = NearestNeighbors(radius=max_cone_distance, algorithm="auto", n_jobs=-1)
    voxel_nn.fit(voxel_xyz)

    print("Scoring voxels by inward normal cones...")
    print("Surface points with valid normals:", len(surface_xyz))
    print("Voxel points:", len(voxel_xyz))
    print("Cone angle degrees:", cone_angle_degrees)
    print("Max cone distance:", max_cone_distance)
    print("Direction bins:", n_direction_bins)

    for start in range(0, len(surface_xyz), batch_size):
        end = min(start + batch_size, len(surface_xyz))
        candidate_indices_list = voxel_nn.radius_neighbors(
            surface_xyz[start:end],
            radius=max_cone_distance,
            return_distance=False,
        )

        for local_i, voxel_indices in enumerate(candidate_indices_list):
            if len(voxel_indices) == 0:
                continue

            p_idx = start + local_i
            axis = inward_normals[p_idx]
            p0 = surface_xyz[p_idx]

            vec = voxel_xyz[voxel_indices] - p0
            dist = np.linalg.norm(vec, axis=1)
            valid_dist = dist > 1e-8
            if not np.any(valid_dist):
                continue

            voxel_indices = voxel_indices[valid_dist]
            vec = vec[valid_dist]
            dist = dist[valid_dist]

            cos_values = (vec @ axis) / (dist + 1e-8)
            hit = cos_values >= cos_threshold

            if not np.any(hit):
                continue

            hit_voxel_indices = voxel_indices[hit]
            bit = np.uint64(1) << np.uint64(direction_bin_ids[p_idx])
            direction_masks[hit_voxel_indices] |= bit

        if (start // batch_size) % 10 == 0:
            print(f"  processed pcd normals: {end}/{len(surface_xyz)}", flush=True)

    scores = np.unpackbits(direction_masks.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
    voxel_score = np.column_stack([
        voxel_xyz,
        scores.astype(np.float32),
    ]).astype(np.float32)

    print("Voxel cone scoring finished.")
    print("Scored voxels:", int(np.sum(scores > 0)))
    print("Max score:", float(scores.max()) if len(scores) > 0 else 0.0)
    print("Mean score:", float(scores.mean()) if len(scores) > 0 else 0.0)

    return voxel_score


def refine_skeleton_by_voxel_cone_score(
    skeleton,
    voxel_cone_score,
    pointcloud=None,
    neighborhood_radius=2.0,
    min_local_score=2.0,
    score_mode="max",
    min_neighbor_voxels=1,
    max_nearest_voxel_distance=1.0,
    use_rounded_voxel_membership=True,
    voxel_round_decimals=0,
    skeleton_neighbor_k=3,
    min_low_score_neighbors=1,
    use_cross_section_for_low_score=True,
    tangent_neighbor_k=6,
    section_half_thickness=1.0,
    section_search_radius=8.0,
    section_radius_percentile=75.0,
    min_section_points=6,
    section_shrink_ratio=0.55,
    fallback_delete_low_score=False,
    keep_if_all_removed=False,
    return_debug=False,
):
    """
    Refine smoothed skeleton by local voxel cone score.

    A skeleton point is directly removed in two stages:
      1. outside-voxel deletion:
         If use_rounded_voxel_membership=True, round the skeleton point to the
         nearest voxel grid coordinate and check whether that voxel exists.
         If it does not exist, delete it unconditionally.
      2. score deletion:
         For skeleton points that are inside voxel, remove points whose local
         voxel cone score is lower than min_local_score.
         -> likely not supported by inward normal voting

    Inputs
    ------
    skeleton:
        N x >=3, x,y,z,...

    voxel_cone_score:
        M x >=4, x,y,z,score

    score_mode:
        "max"        : use max score in the local neighborhood, recommended
        "mean"       : use mean score
        "median"     : use median score
        "percentile" : use 75th percentile score

    Output
    ------
    refined_skeleton:
        Same columns as skeleton, but low-confidence rows removed.
    """
    skeleton = np.asarray(skeleton, dtype=np.float32)
    voxel_cone_score = np.asarray(voxel_cone_score, dtype=np.float32)

    if skeleton.ndim != 2 or skeleton.shape[1] < 3:
        raise ValueError("skeleton must be N x >=3: x,y,z,...")
    if voxel_cone_score.ndim != 2 or voxel_cone_score.shape[1] < 4:
        raise ValueError("voxel_cone_score must be M x >=4: x,y,z,score")
    if neighborhood_radius <= 0:
        raise ValueError("neighborhood_radius must be positive")

    skel_xyz = skeleton[:, :3].astype(np.float32)
    voxel_xyz = voxel_cone_score[:, :3].astype(np.float32)
    voxel_scores = voxel_cone_score[:, 3].astype(np.float32)

    finite_voxels = np.isfinite(voxel_xyz).all(axis=1) & np.isfinite(voxel_scores)
    voxel_xyz = voxel_xyz[finite_voxels]
    voxel_scores = voxel_scores[finite_voxels]

    if len(voxel_xyz) == 0:
        raise ValueError("No valid voxel_cone_score points.")

    nn = NearestNeighbors(radius=neighborhood_radius, algorithm="auto", n_jobs=-1)
    nn.fit(voxel_xyz)

    nearest_nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nearest_nn.fit(voxel_xyz)
    nearest_dist, nearest_idx = nearest_nn.kneighbors(skel_xyz)
    nearest_dist = nearest_dist[:, 0]
    nearest_score = voxel_scores[nearest_idx[:, 0]]

    neighbor_indices = nn.radius_neighbors(
        skel_xyz,
        radius=neighborhood_radius,
        return_distance=False,
    )

    local_scores = np.zeros(len(skel_xyz), dtype=np.float32)
    neighbor_counts = np.zeros(len(skel_xyz), dtype=np.int32)

    for i, inds in enumerate(neighbor_indices):
        neighbor_counts[i] = len(inds)

        if len(inds) == 0:
            local_scores[i] = 0.0
            continue

        scores = voxel_scores[inds]

        if score_mode == "max":
            local_scores[i] = float(np.max(scores))
        elif score_mode == "mean":
            local_scores[i] = float(np.mean(scores))
        elif score_mode == "median":
            local_scores[i] = float(np.median(scores))
        elif score_mode == "percentile":
            local_scores[i] = float(np.percentile(scores, 75))
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    self_scores = nearest_score.copy()

    if use_rounded_voxel_membership:
        if voxel_round_decimals == 0:
            voxel_key_array = np.rint(voxel_xyz).astype(np.int64)
            voxel_keys = set(map(tuple, voxel_key_array))
            skel_keys = np.rint(skel_xyz).astype(np.int64)
        else:
            voxel_key_array = np.round(voxel_xyz, decimals=int(voxel_round_decimals))
            voxel_keys = set(map(tuple, voxel_key_array))
            skel_keys = np.round(skel_xyz, decimals=int(voxel_round_decimals))

        inside_voxel_support = np.array(
            [tuple(key) in voxel_keys for key in skel_keys],
            dtype=bool,
        )

        voxel_score_by_key = {}
        for key, score in zip(voxel_key_array, voxel_scores):
            key = tuple(key)
            if key not in voxel_score_by_key or score > voxel_score_by_key[key]:
                voxel_score_by_key[key] = float(score)

        self_scores = np.array(
            [voxel_score_by_key.get(tuple(key), 0.0) for key in skel_keys],
            dtype=np.float32,
        )
    else:
        inside_voxel_support = nearest_dist <= float(max_nearest_voxel_distance)

    has_voxel_support = neighbor_counts >= int(min_neighbor_voxels)
    self_low_score = self_scores < float(min_local_score)

    if len(skel_xyz) <= 1:
        low_score_neighbor_counts = np.zeros(len(skel_xyz), dtype=np.int32)
    else:
        n_neighbors = min(int(skeleton_neighbor_k) + 1, len(skel_xyz))
        skel_nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
        skel_nn.fit(skel_xyz)
        _, skel_neighbor_idx = skel_nn.kneighbors(skel_xyz)

        low_score_neighbor_counts = np.zeros(len(skel_xyz), dtype=np.int32)
        for i, inds in enumerate(skel_neighbor_idx):
            inds = inds[inds != i]
            if len(inds) > int(skeleton_neighbor_k):
                inds = inds[:int(skeleton_neighbor_k)]
            low_score_neighbor_counts[i] = int(np.sum(self_low_score[inds]))

    has_low_score_neighbors = low_score_neighbor_counts >= int(min_low_score_neighbors)
    low_score_candidate = inside_voxel_support & self_low_score & has_low_score_neighbors

    cross_section_radii = np.full(len(skel_xyz), np.nan, dtype=np.float32)
    cross_section_counts = np.zeros(len(skel_xyz), dtype=np.int32)
    neighbor_median_radii = np.full(len(skel_xyz), np.nan, dtype=np.float32)
    shrink_ratio_values = np.full(len(skel_xyz), np.nan, dtype=np.float32)
    delete_by_cross_section = np.zeros(len(skel_xyz), dtype=bool)

    if use_cross_section_for_low_score and pointcloud is not None and np.any(low_score_candidate):
        pointcloud = np.asarray(pointcloud, dtype=np.float32)
        if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
            raise ValueError("pointcloud must be N x >=3 when use_cross_section_for_low_score=True")

        pcd_xyz = pointcloud[:, :3].astype(np.float32)
        pcd_xyz = pcd_xyz[np.isfinite(pcd_xyz).all(axis=1)]

        if len(pcd_xyz) > 0:
            pcd_nn = NearestNeighbors(radius=section_search_radius, algorithm="auto", n_jobs=-1)
            pcd_nn.fit(pcd_xyz)

            n_tangent_neighbors = min(int(tangent_neighbor_k) + 1, len(skel_xyz))
            tangent_nn = NearestNeighbors(n_neighbors=n_tangent_neighbors, algorithm="auto", n_jobs=-1)
            tangent_nn.fit(skel_xyz)
            _, tangent_neighbor_idx = tangent_nn.kneighbors(skel_xyz)

            tangent_vectors = np.zeros_like(skel_xyz, dtype=np.float32)
            for i, inds in enumerate(tangent_neighbor_idx):
                inds = inds[inds != i]
                neighbor_pts = skel_xyz[inds]

                if len(neighbor_pts) >= 2:
                    centered = neighbor_pts - neighbor_pts.mean(axis=0, keepdims=True)
                    try:
                        _, _, vh = np.linalg.svd(centered, full_matrices=False)
                        tangent = vh[0].astype(np.float32)
                    except np.linalg.LinAlgError:
                        tangent = neighbor_pts[-1] - neighbor_pts[0]
                elif len(neighbor_pts) == 1:
                    tangent = neighbor_pts[0] - skel_xyz[i]
                else:
                    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-8:
                    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    tangent = tangent / tangent_norm
                tangent_vectors[i] = tangent

            for i, center in enumerate(skel_xyz):
                candidate_pcd_idx = pcd_nn.radius_neighbors(
                    center.reshape(1, -1),
                    radius=section_search_radius,
                    return_distance=False,
                )[0]

                if len(candidate_pcd_idx) == 0:
                    continue

                vec = pcd_xyz[candidate_pcd_idx] - center
                tangent = tangent_vectors[i]
                axial = vec @ tangent
                in_slice = np.abs(axial) <= float(section_half_thickness)

                if not np.any(in_slice):
                    continue

                vec_slice = vec[in_slice]
                axial_slice = axial[in_slice]
                radial_vec = vec_slice - axial_slice[:, None] * tangent[None, :]
                radial_dist = np.linalg.norm(radial_vec, axis=1)

                if len(radial_dist) < int(min_section_points):
                    continue

                cross_section_counts[i] = len(radial_dist)
                cross_section_radii[i] = float(np.percentile(radial_dist, section_radius_percentile))

            for i, inds in enumerate(skel_neighbor_idx if len(skel_xyz) > 1 else [[] for _ in range(len(skel_xyz))]):
                inds = inds[inds != i]
                if len(inds) > int(skeleton_neighbor_k):
                    inds = inds[:int(skeleton_neighbor_k)]

                neighbor_radii = cross_section_radii[inds]
                neighbor_radii = neighbor_radii[np.isfinite(neighbor_radii) & (neighbor_radii > 0)]

                if len(neighbor_radii) == 0:
                    continue

                neighbor_median = float(np.median(neighbor_radii))
                neighbor_median_radii[i] = neighbor_median

                if np.isfinite(cross_section_radii[i]) and cross_section_radii[i] > 0 and neighbor_median > 0:
                    ratio = float(cross_section_radii[i] / (neighbor_median + 1e-8))
                    shrink_ratio_values[i] = ratio
                    delete_by_cross_section[i] = ratio < float(section_shrink_ratio)

        delete_by_score = low_score_candidate & delete_by_cross_section
    else:
        delete_by_score = low_score_candidate if fallback_delete_low_score else np.zeros(len(skel_xyz), dtype=bool)

    # Stage 1: outside voxel points are removed unconditionally.
    after_voxel_mask = inside_voxel_support

    # Stage 2: first check this skeleton point's own voxel score.
    # Only if this point itself is low-score, check nearby skeleton points.
    # If nearby skeleton points are not low-score, keep this point.
    # If cross-section check is enabled, delete only when its cross-section is
    # much smaller than nearby skeleton points.
    keep_mask = after_voxel_mask & has_voxel_support & ~delete_by_score

    if not np.any(keep_mask) and keep_if_all_removed:
        print("[WARNING] refine_skeleton_by_voxel_cone_score would remove all skeleton points.")
        print("[WARNING] Keeping original skeleton. Try lowering min_local_score or increasing neighborhood_radius.")
        keep_mask = np.ones(len(skeleton), dtype=bool)

    refined_skeleton = skeleton[keep_mask].copy()

    print("Skeleton cone-score refinement finished.")
    print("Skeleton before:", int(len(skeleton)))
    print("Skeleton after:", int(len(refined_skeleton)))
    print("Removed skeleton points:", int(np.sum(~keep_mask)))
    print("After outside-voxel deletion:", int(np.sum(after_voxel_mask)))
    print("Neighborhood radius:", float(neighborhood_radius))
    print("Max nearest voxel distance:", float(max_nearest_voxel_distance))
    print("Use rounded voxel membership:", bool(use_rounded_voxel_membership))
    print("Voxel round decimals:", int(voxel_round_decimals))
    print("Min local score:", float(min_local_score))
    print("Self-score threshold:", float(min_local_score))
    print("Skeleton neighbor k:", int(skeleton_neighbor_k))
    print("Min low-score neighbors:", int(min_low_score_neighbors))
    print("Outside-voxel skeleton points:", int(np.sum(~inside_voxel_support)))
    print("No-neighbor skeleton points:", int(np.sum(neighbor_counts < int(min_neighbor_voxels))))
    print("Self low-score skeleton points:", int(np.sum(after_voxel_mask & self_low_score)))
    print("Low-score candidates after neighbor check:", int(np.sum(low_score_candidate)))
    print("Use cross-section for low-score:", bool(use_cross_section_for_low_score))
    print("Section half thickness:", float(section_half_thickness))
    print("Section search radius:", float(section_search_radius))
    print("Section shrink ratio threshold:", float(section_shrink_ratio))
    print("Valid cross-section radii:", int(np.sum(np.isfinite(cross_section_radii))))
    print("Cross-section-shrunken candidates:", int(np.sum(low_score_candidate & delete_by_cross_section)))
    print("Low-score points deleted after cross-section check:", int(np.sum(delete_by_score)))
    print("Local score min/median/max:", float(np.min(local_scores)), float(np.median(local_scores)), float(np.max(local_scores)))
    print("Self voxel score min/median/max:", float(np.min(self_scores)), float(np.median(self_scores)), float(np.max(self_scores)))
    print("Nearest voxel score min/median/max:", float(np.min(nearest_score)), float(np.median(nearest_score)), float(np.max(nearest_score)))

    if return_debug:
        debug = {
            "local_scores": local_scores,
            "self_scores": self_scores,
            "self_low_score": self_low_score,
            "nearest_scores": nearest_score,
            "nearest_dist": nearest_dist,
            "neighbor_counts": neighbor_counts,
            "low_score_neighbor_counts": low_score_neighbor_counts,
            "low_score_candidate": low_score_candidate,
            "cross_section_radii": cross_section_radii,
            "cross_section_counts": cross_section_counts,
            "neighbor_median_radii": neighbor_median_radii,
            "shrink_ratio_values": shrink_ratio_values,
            "delete_by_cross_section": delete_by_cross_section,
            "delete_by_score": delete_by_score,
            "keep_mask": keep_mask,
            "after_voxel_mask": after_voxel_mask,
            "inside_voxel_support": inside_voxel_support,
            "has_voxel_support": has_voxel_support,
        }
        return refined_skeleton, keep_mask, debug

    return refined_skeleton


def compute_voxel_distance_to_pcd(voxel_xyz, pointcloud, batch_size=300000):
    """
    Compute each voxel point's nearest distance to pcd surface points.

    Input
    -----
    voxel_xyz:
        N x 3, x,y,z voxel coordinates.
    pointcloud:
        M x >=3, pcd coordinates. Usually x,y,z,nx,ny,nz.

    Output
    ------
    distance_values:
        N values. Smaller value means closer to pcd boundary/surface.
    """
    voxel_xyz = np.asarray(voxel_xyz, dtype=np.float32)
    pointcloud = np.asarray(pointcloud, dtype=np.float32)

    if voxel_xyz.ndim != 2 or voxel_xyz.shape[1] < 3:
        raise ValueError("voxel_xyz must be N x >=3")
    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError("pointcloud must be M x >=3")

    voxel_points = voxel_xyz[:, :3].astype(np.float32)
    pcd_points = pointcloud[:, :3].astype(np.float32)

    finite_voxels = np.isfinite(voxel_points).all(axis=1)
    finite_pcd = np.isfinite(pcd_points).all(axis=1)

    if not np.any(finite_pcd):
        raise ValueError("pointcloud has no finite xyz points.")

    pcd_points = pcd_points[finite_pcd]
    distance_values = np.full(len(voxel_points), np.inf, dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(pcd_points)

    valid_indices = np.where(finite_voxels)[0]
    valid_voxel_points = voxel_points[finite_voxels]

    for start in range(0, len(valid_voxel_points), batch_size):
        end = min(start + batch_size, len(valid_voxel_points))
        dist, _ = nn.kneighbors(valid_voxel_points[start:end])
        distance_values[valid_indices[start:end]] = dist[:, 0].astype(np.float32)

    return distance_values


def build_value_by_voxel_key(xyz, values, voxel_round_decimals=0, keep="max"):
    xyz = np.asarray(xyz, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    if voxel_round_decimals == 0:
        key_array = np.rint(xyz[:, :3]).astype(np.int64)
    else:
        key_array = np.round(xyz[:, :3], decimals=int(voxel_round_decimals))

    value_by_key = {}
    for key, value in zip(key_array, values):
        key = tuple(key)
        if key not in value_by_key:
            value_by_key[key] = float(value)
        elif keep == "max":
            value_by_key[key] = max(value_by_key[key], float(value))
        elif keep == "min":
            value_by_key[key] = min(value_by_key[key], float(value))
        else:
            value_by_key[key] = float(value)

    return key_array, value_by_key


def refine_skeleton_by_voxel_score_fast(
    skeleton,
    voxel_cone_score,
    voxel_distance=None,
    pointcloud=None,
    min_score_threshold=3.0,
    boundary_distance_threshold=2.0,
    skeleton_neighbor_k=6,
    neighbor_score_drop_threshold=2.0,
    use_rounded_voxel_membership=True,
    voxel_round_decimals=0,
    keep_if_all_removed=False,
    return_debug=False,
):
    """
    Fast refinement of smoothed skeleton.

    Logic
    -----
    1. Force-delete skeleton points outside voxel.
    2. Compute each skeleton point's own voxel cone score.
    3. Only focus on low-score points.
    4. Low-score points deep inside the voxel object are kept.
    5. For low-score points near the voxel boundary, compare with nearby skeleton
       scores. Delete only if this point is much lower than its neighbors.

    This avoids the expensive cross-section calculation and also avoids deleting
    genuinely thin/small mitochondria where neighboring skeleton points are also
    low or similar.
    """
    skeleton = np.asarray(skeleton, dtype=np.float32)
    voxel_cone_score = np.asarray(voxel_cone_score, dtype=np.float32)

    if skeleton.ndim != 2 or skeleton.shape[1] < 3:
        raise ValueError("skeleton must be N x >=3: x,y,z,...")
    if voxel_cone_score.ndim != 2 or voxel_cone_score.shape[1] < 4:
        raise ValueError("voxel_cone_score must be M x >=4: x,y,z,score")

    skel_xyz = skeleton[:, :3].astype(np.float32)
    voxel_xyz = voxel_cone_score[:, :3].astype(np.float32)
    voxel_scores = voxel_cone_score[:, 3].astype(np.float32)

    finite_voxels = np.isfinite(voxel_xyz).all(axis=1) & np.isfinite(voxel_scores)
    voxel_xyz = voxel_xyz[finite_voxels]
    voxel_scores = voxel_scores[finite_voxels]

    if len(voxel_xyz) == 0:
        raise ValueError("No valid voxel_cone_score points.")

    nearest_nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nearest_nn.fit(voxel_xyz)
    nearest_dist, nearest_idx = nearest_nn.kneighbors(skel_xyz)
    nearest_dist = nearest_dist[:, 0]
    nearest_score = voxel_scores[nearest_idx[:, 0]]

    self_scores = nearest_score.copy()

    if use_rounded_voxel_membership:
        if voxel_round_decimals == 0:
            voxel_key_array = np.rint(voxel_xyz).astype(np.int64)
            skel_key_array = np.rint(skel_xyz).astype(np.int64)
        else:
            voxel_key_array = np.round(voxel_xyz, decimals=int(voxel_round_decimals))
            skel_key_array = np.round(skel_xyz, decimals=int(voxel_round_decimals))

        voxel_keys = set(map(tuple, voxel_key_array))
        inside_voxel_mask = np.array(
            [tuple(key) in voxel_keys for key in skel_key_array],
            dtype=bool,
        )

        voxel_score_by_key = {}
        for key, score in zip(voxel_key_array, voxel_scores):
            key = tuple(key)
            if key not in voxel_score_by_key or score > voxel_score_by_key[key]:
                voxel_score_by_key[key] = float(score)

        self_scores = np.array(
            [voxel_score_by_key.get(tuple(key), 0.0) for key in skel_key_array],
            dtype=np.float32,
        )
    else:
        inside_voxel_mask = nearest_dist <= 1.0

    # Step 1: outside voxel points are removed unconditionally.
    keep_after_voxel = inside_voxel_mask.copy()

    # Step 2: low-score candidates.
    low_score_mask = self_scores < float(min_score_threshold)

    # Step 3: only low-score points near voxel boundary are suspicious.
    # Use voxel distance transform, not distance to pcd, because pcd may contain
    # points close to internal/deep skeleton locations.
    if voxel_distance is None:
        if pointcloud is None:
            raise ValueError("Either voxel_distance or pointcloud must be provided.")
        distance_values = compute_voxel_distance_to_pcd(voxel_xyz, pointcloud)
        distance_xyz = voxel_xyz
    else:
        voxel_distance = np.asarray(voxel_distance, dtype=np.float32)
        if voxel_distance.ndim != 2 or voxel_distance.shape[1] < 4:
            raise ValueError("voxel_distance must be M x >=4: x,y,z,distance")
        distance_xyz = voxel_distance[:, :3].astype(np.float32)
        distance_values = voxel_distance[:, 3].astype(np.float32)

    finite_distance = np.isfinite(distance_xyz).all(axis=1) & np.isfinite(distance_values)
    distance_xyz = distance_xyz[finite_distance]
    distance_values = distance_values[finite_distance]
    _, distance_by_key = build_value_by_voxel_key(
        distance_xyz,
        distance_values,
        voxel_round_decimals=voxel_round_decimals,
        keep="min",
    )

    if use_rounded_voxel_membership:
        self_distance = np.array(
            [distance_by_key.get(tuple(key), np.inf) for key in skel_key_array],
            dtype=np.float32,
        )
    else:
        nearest_voxel_keys = np.rint(voxel_xyz[nearest_idx[:, 0], :3]).astype(np.int64)
        self_distance = np.array(
            [distance_by_key.get(tuple(key), np.inf) for key in nearest_voxel_keys],
            dtype=np.float32,
        )

    # distance small -> close to pcd boundary/surface; distance large -> deep inside.
    near_boundary_mask = self_distance <= float(boundary_distance_threshold)

    # Step 4: compare candidate score with nearby skeleton scores.
    if len(skel_xyz) <= 1:
        neighbor_median_scores = np.full(len(skel_xyz), np.nan, dtype=np.float32)
        neighbor_score_drop = np.zeros(len(skel_xyz), dtype=np.float32)
    else:
        n_neighbors = min(int(skeleton_neighbor_k) + 1, len(skel_xyz))
        skel_nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
        skel_nn.fit(skel_xyz)
        _, neighbor_idx = skel_nn.kneighbors(skel_xyz)

        neighbor_median_scores = np.full(len(skel_xyz), np.nan, dtype=np.float32)
        neighbor_score_drop = np.zeros(len(skel_xyz), dtype=np.float32)

        for i, inds in enumerate(neighbor_idx):
            inds = inds[inds != i]
            if len(inds) > int(skeleton_neighbor_k):
                inds = inds[:int(skeleton_neighbor_k)]

            if len(inds) == 0:
                continue

            median_score = float(np.median(self_scores[inds]))
            neighbor_median_scores[i] = median_score
            neighbor_score_drop[i] = median_score - float(self_scores[i])

    # Delete only when:
    #   inside voxel but low-score,
    #   close to boundary,
    #   and much lower than surrounding skeleton points.
    # If surrounding points are also low/similar, keep it.
    delete_by_score = (
        keep_after_voxel
        & low_score_mask
        & near_boundary_mask
        & np.isfinite(neighbor_median_scores)
        & (neighbor_median_scores >= float(min_score_threshold))
        & (neighbor_score_drop >= float(neighbor_score_drop_threshold))
    )

    keep_mask = keep_after_voxel & ~delete_by_score

    if not np.any(keep_mask) and keep_if_all_removed:
        print("[WARNING] refine_skeleton_by_voxel_score_fast would remove all skeleton points.")
        print("[WARNING] Keeping original skeleton.")
        keep_mask = np.ones(len(skeleton), dtype=bool)

    refined_skeleton = skeleton[keep_mask].copy()

    print("Fast skeleton refinement finished.")
    print("Skeleton before:", int(len(skeleton)))
    print("Skeleton after:", int(len(refined_skeleton)))
    print("Removed skeleton points:", int(np.sum(~keep_mask)))
    print("Outside-voxel skeleton points:", int(np.sum(~inside_voxel_mask)))
    print("Self low-score skeleton points:", int(np.sum(keep_after_voxel & low_score_mask)))
    print("Low-score near-boundary points:", int(np.sum(keep_after_voxel & low_score_mask & near_boundary_mask)))
    print("Deleted low-score boundary outliers:", int(np.sum(delete_by_score)))
    print("Min score threshold:", float(min_score_threshold))
    print("Voxel-to-PCD distance boundary threshold:", float(boundary_distance_threshold))
    print("Neighbor score drop threshold:", float(neighbor_score_drop_threshold))
    print("Self score min/median/max:", float(np.min(self_scores)), float(np.median(self_scores)), float(np.max(self_scores)))
    print("Self voxel-to-PCD distance min/median/max:", float(np.min(self_distance)), float(np.median(self_distance)), float(np.max(self_distance)))
    print("All voxel-to-PCD distance min/median/max:", float(np.min(distance_values)), float(np.median(distance_values)), float(np.max(distance_values)))

    if return_debug:
        debug = {
            "keep_mask": keep_mask,
            "inside_voxel_mask": inside_voxel_mask,
            "self_scores": self_scores,
            "low_score_mask": low_score_mask,
            "self_distance": self_distance,
            "voxel_distance_values": distance_values,
            "near_boundary_mask": near_boundary_mask,
            "neighbor_median_scores": neighbor_median_scores,
            "neighbor_score_drop": neighbor_score_drop,
            "delete_by_score": delete_by_score,
        }
        return refined_skeleton, keep_mask, debug

    return refined_skeleton


def assign_rgb_by_nearest(query_xyz, source_xyz, source_rgb, max_assign_distance=None, batch_size=300000):
    query_xyz = np.asarray(query_xyz, dtype=np.float32)
    source_xyz = np.asarray(source_xyz, dtype=np.float32)
    source_rgb = np.asarray(source_rgb, dtype=np.uint8)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(source_xyz)

    out_rgb = np.zeros((len(query_xyz), 3), dtype=np.uint8)

    for start in range(0, len(query_xyz), batch_size):
        end = min(start + batch_size, len(query_xyz))
        dists, idx = nn.kneighbors(query_xyz[start:end])
        rgb = source_rgb[idx[:, 0]].copy()

        if max_assign_distance is not None:
            rgb[dists[:, 0] > max_assign_distance] = np.asarray([0, 0, 0], dtype=np.uint8)

        out_rgb[start:end] = rgb

    return out_rgb


def segment_voxel_by_skeleton_instance(
    skeleton_instance,
    voxel,
    max_assign_distance=None,
    batch_size=300000,
):
    """
    Inputs:
        skeleton_instance: x,y,z,instance_id,r,g,b
        voxel: x,y,z,instance

    Output:
        x,y,z,r,g,b
    """
    voxel = np.asarray(voxel, dtype=np.float32)
    if voxel.ndim != 2 or voxel.shape[1] < 4:
        raise ValueError("voxel must be N x >=4: x,y,z,instance")

    skel_xyz, skel_rgb = get_valid_skeleton_xyzrgb(skeleton_instance)
    voxel_xyz = voxel[:, :3].astype(np.float32)
    voxel_rgb = assign_rgb_by_nearest(
        query_xyz=voxel_xyz,
        source_xyz=skel_xyz,
        source_rgb=skel_rgb,
        max_assign_distance=max_assign_distance,
        batch_size=batch_size,
    )

    voxel_xyzrgb = np.column_stack([voxel_xyz, voxel_rgb.astype(np.float32)]).astype(np.float32)
    print("Voxel points:", len(voxel_xyzrgb))
    print("Valid skeleton seeds:", len(skel_xyz))
    print("Voxel instance shape:", voxel_xyzrgb.shape)
    return voxel_xyzrgb


def segment_pcd_by_voxel_instance(
    pointcloud,
    voxel_xyzrgb,
    max_assign_distance=None,
    batch_size=300000,
):
    """
    Inputs:
        pointcloud: x,y,z,nx,ny,nz
        voxel_xyzrgb: x,y,z,r,g,b

    Output:
        x,y,z,nx,ny,nz,r,g,b
    """
    pointcloud = np.asarray(pointcloud, dtype=np.float32)
    voxel_xyzrgb = np.asarray(voxel_xyzrgb, dtype=np.float32)

    if pointcloud.ndim != 2 or pointcloud.shape[1] < 6:
        raise ValueError("pointcloud must be N x >=6: x,y,z,nx,ny,nz")
    if voxel_xyzrgb.ndim != 2 or voxel_xyzrgb.shape[1] < 6:
        raise ValueError("voxel_xyzrgb must be N x >=6: x,y,z,r,g,b")

    pcd_xyz = pointcloud[:, :3].astype(np.float32)
    pcd_normals = pointcloud[:, 3:6].astype(np.float32)
    voxel_xyz = voxel_xyzrgb[:, :3].astype(np.float32)
    voxel_rgb = np.rint(voxel_xyzrgb[:, 3:6]).astype(np.uint8)

    valid = np.any(voxel_rgb != 0, axis=1)
    if not np.any(valid):
        raise ValueError("No valid colored voxel instance points found.")

    pcd_rgb = assign_rgb_by_nearest(
        query_xyz=pcd_xyz,
        source_xyz=voxel_xyz[valid],
        source_rgb=voxel_rgb[valid],
        max_assign_distance=max_assign_distance,
        batch_size=batch_size,
    )

    pcd_instance = np.column_stack([pcd_xyz, pcd_normals, pcd_rgb.astype(np.float32)]).astype(np.float32)
    print("PCD points:", len(pcd_instance))
    print("PCD instance shape:", pcd_instance.shape)
    return pcd_instance

def main():
    ##################################################
    #################### paths #######################

    
    file_name_list = [
                    # "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260321_mcf-7_mito488tracker_ld561bodipy_nuc405hurst_tif",
                    # "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260314_ac16_mito488tracker_ld561bodipy_nuc405hurst_tif",
                    # "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260309_mda_mito488tracker_ld561bodipy_nuc405hurst_tif",
                    # "20260207_fibro_mito488_lyso640_nuc405_tif",
                    # "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",
                    # "20251217_hela_mito488tracker_ld561bodipy_nuc405hurst",
                    # "20251213_h28_mito488_ld561_nuc405",
                    # "20251201_u2os_ld488_mito640_nuc405",
                    # "20251127_msto_ld488_mito640_nuc405",

                    # "20251122_fibro_cccp_10um_mito488_lyso640_nuc405",
                    # "20251113_fibro_cccp_5um_mito488_lyso640_nuc405",
                    # "20251113_fibro_cccp_2um_mito488_lyso640_nuc405",
                    # "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",

                    #  "grp1_decon",
                    #  "grp2_decon",
                    #  "grp3_decon",
                    #  "grp4_decon",
                    # "peroxi"，
                    # "20260207_fibro_mito488_lyso640_nuc405_tif"，

                    #  "20250905_msto_mitotracekr488_lysotracker640_nuc405",
                    # "20260321_mcf-7_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260314_ac16_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260310_mda_mito488tracker_lyso640tracker_nuc405hurst_tif",
                    # "20260207_fibro_mito488_lyso640_nuc405_tif",
                    # "20250917_h28_mitotracker488_lysotracker640_nuc405",
                    # "20251218_hela_mito488tracker_ld561bodipy_lyso640tracker_nuc405hurst",
                    # "20250909_u2os_mitotracker488_lysotracker640_nuc405",

                    # "20260327_oaloading_before_mito488tracker_ld561bodipy_nuc405hurst",
                    # "20260328_oaloading_loading24h_mito488tracker_ld561bodipy_nuc405hurst",
                    # "20260329_oaloading_unloading_high_mito488tracker_ld561bodipy_nuc405hurst",
                    # "20260329_oaloading_unloading_low_mito488tracker_ld561bodipi_nuc405hurst",
                    
                    #  "grp1_decon",
                    #  "grp2_decon",
                    #  "grp3_decon",
                    #  "grp4_decon",

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

                    "dTag_CCCP10uM_1518_conTL_002",
                    # "H9c2-dTag_normal_1518_conTL_001",

                    ]

    for file_name in file_name_list:

        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/20251002_remodeling_tif/24h/{file_name}")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/20260708_cccp_fibro/{file_name}/mito/")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/20260332_oa_loading/{file_name}/mito/")
        # BASE_DIR = Path(f"D:/UTS/cell_seg/codes/organlle_seg_dataprocess/data/data_thesis/20251002_remodeling_tif/24h/{file_name}/")
        # BASE_DIR = Path(f"./data/pub_cccp/gluc/20210330_H9c2_mito-mChGFP_LTRDR50nM_1522_dvt-5sTL_003_ALX")
        # BASE_DIR = Path(f"./data/pub_cccp/cccp_cell1/C2-20210611_H9c2-dTag_CCCP10uM_1518_conTL_002_ALX")

        # BASE_DIR = Path(f"./data/organ/mito")
        # BASE_DIR = Path(fr"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path(fr"D:/public_data/cccp/cccp_mito/c2/{file_name}")
        # BASE_DIR = Path(fr"D:/public_data/cccp/mito_lyso_gluc_galac/c3/{file_name}")
        # BASE_DIR = Path("./data/oa_loading/C2")
        # BASE_DIR = Path("./data/raw_cell/mito")
        BASE_DIR = Path(f"D:/public_data/S-BIAD1445/neuron/mito_test")


        INPUT_PATH = BASE_DIR / "processed_output"
        OUTPUT_PATH = BASE_DIR / "processed_output"

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        print(os.listdir(INPUT_PATH))

        imgs_list = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith('_pcd.txt')]
        # imgs_list = ["hela_mito488_ld561_lyso640_nuc405_17054"]


        print("processing {}".format(imgs_list))

        name_lists = []

        i = 0
        num_imgs = len(imgs_list)

        for img_name in imgs_list:
            img_name = img_name.split("_pcd")[0]
            i = i + 1
            print(f"-------------------------------------------------------------------------")
            print(f"------------ processing ----- {file_name} ----- {i}/{num_imgs} ----- {img_name} ----------------------")
            print(f"-------------------------------------------------------------------------")

            img_name = img_name.split('.')[0]
            name_lists.append(img_name)

            ##################################################
            #################### paths #######################

            pcd_path = INPUT_PATH / (img_name + "_pcd.txt")
            voxel_path = INPUT_PATH / (img_name + "_voxel.txt")
            
            mask_path = INPUT_PATH / (img_name + "_mask.tif")
            save_skelenton_path = OUTPUT_PATH / (img_name + "_skelenton.txt")
            save_skelenton_add_path = OUTPUT_PATH / (img_name + "_add_skelenton.txt")
            save_innerpcd_path = OUTPUT_PATH / (img_name + "_innerpcd.txt")
            save_refinedskelenton_path = OUTPUT_PATH / (img_name + "_refinedskelenton.txt")
            save_skeleton_instance_path = OUTPUT_PATH / (img_name + "_skelenton_instance.txt")

            save_skeleton_instance_coarse_path = OUTPUT_PATH / (img_name + "_skelenton_instance_coarse.txt")

            save_pcd_instance_coarse = OUTPUT_PATH / (img_name + "_pcd_instance_coarse.txt")
            save_voxel_instance_coarse = OUTPUT_PATH / (img_name + "_voxel_instance_coarse.txt")
            save_pcd_instance_dbscan_path = OUTPUT_PATH / (img_name + "_pcd_instance_dbscan.txt")
            save_voxel_cone_score_path = OUTPUT_PATH / (img_name + "_voxel_cone_score.txt")
            save_voxel_cone_score_depth_path = OUTPUT_PATH / (img_name + "_voxel_cone_score_distance.txt")
            save_voxel_distance_path = OUTPUT_PATH / (img_name + "_voxel_distance.txt")

            pcd = load_point_cloud(pcd_path)
            print(pcd.shape)

            # Build PCD instances first and discard components with < n points.
            # Every downstream step uses this filtered point cloud.
            pcd = filter_small_pcd_instances(
                pointcloud=pcd,
                min_instance_points=PCD_MIN_INSTANCE_POINTS,
                eps=PCD_INSTANCE_DBSCAN_EPS,
                min_samples=PCD_INSTANCE_DBSCAN_MIN_SAMPLES,
            )

            pcd_instance_dbscan = dbscan_coarse_pcd_xyzrgb(
                pointcloud=pcd,
                eps=PCD_INSTANCE_DBSCAN_EPS,
                min_samples=PCD_INSTANCE_DBSCAN_MIN_SAMPLES,
                color_seed=123,
            )
            save_pcd2txt(pcd_instance_dbscan, save_pcd_instance_dbscan_path)
            print(f"[SAVED PCD DBSCAN INSTANCE] {save_pcd_instance_dbscan_path}")

            voxel = load_point_cloud(voxel_path)

            # voxel_cone_score = score_voxels_by_inward_normal_cones(
            #     pointcloud=pcd,
            #     voxel=voxel,
            #     cone_angle_degrees=20.0,
            #     max_cone_distance=10.0,
            #     n_direction_bins=32,
            #     normals_are_outward=True,
            #     batch_size=2000,
            # )
            # save_pcd2txt(voxel_cone_score, save_voxel_cone_score_path)
            # print(f"[SAVED VOXEL CONE SCORE] {save_voxel_cone_score_path}")

            voxel_distance_values = compute_voxel_distance_to_pcd(
                voxel[:, :3],
                pcd,
            )
            voxel_distance = np.column_stack([
                voxel[:, :3],
                voxel_distance_values.astype(np.float32),
            ]).astype(np.float32)
            save_pcd2txt(voxel_distance, save_voxel_distance_path)
            print(f"[SAVED VOXEL DISTANCE] {save_voxel_distance_path}")

            # voxel_cone_score_depth = np.column_stack([
            #     voxele[:, :4],
            #     voxel_distance_values.astype(np.float32),
            # ]).astype(np.float32)
            # save_pcd2txt(voxel_cone_score_depth, save_voxel_cone_score_depth_path)
            # print(f"[SAVED VOXEL CONE SCORE DISTANCE] {save_voxel_cone_score_depth_path}")

            if not RUN_MASK_SKELETON_PIPELINE_AFTER_CONE_SCORE:
                print("[SKIP MASK SKELETON PIPELINE] Cone score has been saved.")
                continue

            mask = read_tif(mask_path)

            mask = (mask != 0)
            print(mask.shape)

            skeleton_table, skel, dist_map, degree = skeleton_to_table_zyx(
                mask,
                voxel_size_zyx=(1, 1, 1),
                min_object_size=10,
                do_small_closing=True,
                physical=True,
            )
            # z, y, x, radius, degree, type

            save_pcd2txt(skeleton_table, save_skelenton_path)


            smoothed_skeleton = smooth_skeleton_graph_laplacian(
                skeleton=skeleton_table,
                connect_radius=1.8,
                iterations=10,
                alpha=0.35,
                protect_endpoints=True,
                protect_junctions=True,
                max_step_per_iter=1.0,
                recompute_degree=True,
                return_debug=False,
            )

            # save_pcd2txt(smoothed_skeleton, save_refinedskelenton_path)

            smoothed_skeleton, refined_keep_mask, refined_debug = refine_skeleton_by_voxel_score_fast(
                skeleton=smoothed_skeleton,
                # voxel_cone_score=voxel_cone_score,
                voxel_cone_score=voxel_distance,
                voxel_distance=voxel_distance,
                pointcloud=pcd,
                min_score_threshold=4.0,
                boundary_distance_threshold=2.0,
                skeleton_neighbor_k=3,
                neighbor_score_drop_threshold=3.0,
                use_rounded_voxel_membership=True,
                voxel_round_decimals=0,
                keep_if_all_removed=False,
                return_debug=True,
            )

            refined_skeleton_score = np.column_stack([
                smoothed_skeleton[:, :3],
                refined_debug["self_scores"][refined_keep_mask].astype(np.float32),
            ]).astype(np.float32)
            save_pcd2txt(refined_skeleton_score, save_refinedskelenton_path)
            print("----------------------------------------------------------------------")
            print(f"[SAVED REFINED SKELETON XYZ+SCORE] {save_refinedskelenton_path}")

            instance_skeleton = skeleton_to_instance(
                smoothed_skeleton=smoothed_skeleton,
                connect_radius=1.8,
                degree_col=4,
                color_seed=42,
                background_color=(255, 255, 255),
                red_branch_rounds=20,
                min_direction_cos=0.2,
            )

            save_pcd2txt(instance_skeleton, save_skeleton_instance_path)


            coarse_skeleton_xyzrgb = dbscan_coarse_skeleton_xyzrgb(
                smoothed_skeleton=smoothed_skeleton,
                eps=2.0,
                min_samples=1,
                color_seed=123,
            )

            save_pcd2txt(coarse_skeleton_xyzrgb, save_skeleton_instance_coarse_path)
            print(f"[SAVED COARSE SKELETON INSTANCE] {save_skeleton_instance_coarse_path}")

            
            voxel_xyzrgb_coarse = segment_voxel_by_skeleton_instance(
                skeleton_instance=coarse_skeleton_xyzrgb,
                voxel=voxel,
                max_assign_distance=None,
            )

            save_pcd2txt(voxel_xyzrgb_coarse, save_voxel_instance_coarse)


            pcd_instance_coarse = segment_pcd_by_voxel_instance(
                pointcloud=pcd,
                voxel_xyzrgb=voxel_xyzrgb_coarse,
                max_assign_distance=None,
            )
            save_pcd2txt(pcd_instance_coarse, save_pcd_instance_coarse)




            # skeleton_table, added_points, labels, info = add_skeleton_points_for_small_pcd_components(
            #     pcd=pcd,
            #     skeleton=smoothed_skeleton,
            #     dbscan_eps=2.0,
            #     dbscan_min_samples=10,
            #     max_component_points=1000,
            #     min_component_points=100,
            #     bbox_margin=1.0,
            #     add_radius=True,
            # )
            # # save_pcd2txt(skeleton_table, save_skelenton_add_path)

            # inner_pcd = inward_shift_point_cloud(pcd, inward_distance=1.0)
            # # save_pcd2txt(inner_pcd, save_innerpcd_path)

            # refined_skeleton, keep_mask, debug = remove_skeleton_outside_pointcloud(
            #     pcd=inner_pcd,  # x,y,z,nx,ny,nz
            #     skeleton=skeleton_table,  # x,y,z,radius,degree,type
            #     k=16,
            #     outside_margin=0.0,
            #     outside_vote_threshold=0.7,
            #     normals_are_outward=True,
            #     return_debug=True,
            # )

            # refined_skeleton, keep_mask_close, debug_close = remove_skeleton_too_close_to_pointcloud(
            #     pcd=pcd,
            #     skeleton=refined_skeleton,
            #     min_surface_distance=0.8,
            #     k=6,
            #     return_debug=True,
            # )

            # save_pcd2txt(refined_skeleton, save_refinedskelenton_path)



if __name__ == '__main__':
    main()

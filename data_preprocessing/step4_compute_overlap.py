"""
step4_compute_overlap.py -- HM3D G2G data generation pipeline Step 4

Computes the per-frame overlap matrix between different view sequences
(traj_id, cam_id) based on monocular depth projection, with several
optimizations for speed:
  - No normal-consistency filter (saves ~40-50% of the computation: normal
    estimation + normal-consistency check)
  - Precomputed camera inverse matrices (reduces np.linalg.inv calls from
    2*T_a*T_b down to T_a+T_b)
  - Low-resolution projection: depth maps are downsampled to --proj_size
    (default 56) before projection, since the result is only used for coarse
    overlap screening and does not need full-resolution precision
  - Camera subsampling: front view (cam_0) + 2 of 5 excluding hard-left
    (cam_6) and hard-right (cam_2), i.e. 3 representative cameras per
    trajectory and 3x3=9 combinations per trajectory pair

Uses uint16 PNG depth maps (1mm precision, range 0~65.535m).
Depth maps are loaded at --image_size resolution, then downsampled to
--proj_size for projection.

A view sequence = one camera along one trajectory (single-camera time series).
During G2G training, Group A / Group B are sampled from two different view
sequences.

Output per pair:
  - Three overlap matrices (A->B, B->A, Symmetric), saved as .npz files
  - Matrices quantized to uint8 (0-255), corresponding to overlap rate
    0.0-1.0, with precision 1/255 ~= 0.004
  - Summary statistics recorded in JSON (best frame pair, etc.)
  - At training time the Dataset loads the matrices and slices 5x5 windows
    on the fly

Camera selection strategy:
  - 3 representative cameras per trajectory: cam_0 (front) + 2 randomly
    chosen from {1,3,4,5,7}
  - Exclude cam_2 (hard-right, 90 deg) and cam_6 (hard-left, 270 deg), since
    hard-left/hard-right views have larger overlap bias
  - 3x3=9 camera combinations per trajectory pair, C(25,2)=300 trajectory
    pairs -> 2700 pairs/scene

Algorithm:
  1. For each scene, enumerate all trajectory pairs C(N_traj, 2)
  2. For each trajectory pair (i, j), process bidirectionally:
     - Select representative cameras (front + 2 of 5)
     - For each (cam_a, cam_b) combination:
         downsample depth maps to proj_size -> project to compute the full
         T_a x T_b frame-pair overlap matrix
     - Save quantized matrices as .npz + metadata to JSON
  3. Single-frame overlap: back-project depth -> coordinate transform ->
     project -> depth consistency check

Usage:
# With 224x224 uint16 depth maps, downsampled to 56x56 for projection (default)
python step4_compute_overlap.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --step3_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step4_overlap_224_uint16_val \
    --image_size 224 --proj_size 56 \
    --depth_tolerance 0.20 \
    --body_distance_cutoff 20.0 \
    --skip_existing \
    --num_workers 1 --worker_id 0

# With 512x512 uint16 depth maps, downsampled to 56x56 for projection
python step4_compute_overlap.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --step3_root /path/to/data/HM3D/DATA_GEN/step3_render_512_512_uint16_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step4_overlap_512_uint16_val \
    --image_size 512 --proj_size 56 \
    --depth_tolerance 0.20 \
    --body_distance_cutoff 20.0 \
    --skip_existing \
    --num_workers 1 --worker_id 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from itertools import combinations, product

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEPTH_QUANTIZATION_M = 0.001  # uint16 PNG: 1 unit = 0.001m (1mm)
RENDER_INTRINSICS_SIZE = 518  # step2 default intrinsics resolution (step2.5 overrides via rig_config["width"])
NUM_CAMERAS = 8


# ---------------------------------------------------------------------------
# Overlap matrix quantization (uint8 for storage efficiency)
# ---------------------------------------------------------------------------
def quantize_overlap(overlap: np.ndarray) -> np.ndarray:
    """
    Quantize float32 overlap rate [0.0, 1.0] to uint8 [0, 255].

    Quantization precision: 1/255 ~= 0.004 (0.4%), precise enough for overlap
    rate computation.
    Storage optimization: each matrix shrinks from 4 bytes/pixel to 1 byte/pixel,
    saving 75% of the space.

    Args:
        overlap: (H, W) float32 array in [0.0, 1.0]

    Returns:
        (H, W) uint8 array in [0, 255]
    """
    return np.clip(np.round(overlap * 255.0), 0, 255).astype(np.uint8)


def dequantize_overlap(quantized: np.ndarray) -> np.ndarray:
    """
    Dequantize uint8 [0, 255] to float32 [0.0, 1.0].

    Args:
        quantized: (H, W) uint8 array in [0, 255]

    Returns:
        (H, W) float32 array in [0.0, 1.0]
    """
    return quantized.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HM3D G2G Overlap Computation (Step 4): "
        "compute full frame-pair overlap matrices via depth projection"
    )
    parser.add_argument(
        "--step1_root", type=str, required=True,
        help="Step 1 output dir (trajectory.tum, traj_meta.json)",
    )
    parser.add_argument(
        "--step2_root", type=str, required=True,
        help="Step 2 output dir (rig_config.json)",
    )
    parser.add_argument(
        "--step3_root", type=str, required=True,
        help="Step 3 output dir (depth/ images/)",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Step 4 output dir (overlap matrices + JSON per scene)",
    )
    parser.add_argument(
        "--depth_tolerance", type=float, default=0.30,
        help="Depth consistency tolerance in meters (default: 0.30)",
    )
    parser.add_argument(
        "--body_distance_cutoff", type=float, default=20.0,
        help="Skip pairs with body min distance > cutoff (default: 20.0m)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip scenes that already have output files",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Total number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--worker_id", type=int, default=0,
        help="Current worker ID, 0-indexed (default: 0)",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=-1,
        help="Max scenes to process, -1 for all (default: -1)",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Depth/RGB image resolution (must match step3 --image_size, default: 224)",
    )
    parser.add_argument(
        "--proj_size", type=int, default=56,
        help="Projection resolution for overlap computation (default: 56). "
        "Depth maps are downsampled from image_size to proj_size before projection. "
        "Lower resolution is sufficient for coarse overlap screening.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# TUM trajectory loading (reuse from step3)
# ---------------------------------------------------------------------------
def load_trajectory_tum(
    tum_path: str,
) -> list[tuple[float, list[float], list[float]]]:
    """
    Load TUM format trajectory.

    Format: timestamp tx ty tz qx qy qz qw

    Returns:
        [(timestamp, [tx, ty, tz], [qx, qy, qz, qw]), ...]
    """
    frames = []
    with open(tum_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            position = [float(parts[1]), float(parts[2]), float(parts[3])]
            rotation_xyzw = [
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            ]
            frames.append((timestamp, position, rotation_xyzw))
    return frames


# ---------------------------------------------------------------------------
# Depth loading with LRU cache
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def load_depth_png(path: str) -> np.ndarray:
    """
    Load uint16 depth PNG (same resolution as the RGB image) and convert to float32 meters.

    Quantization: 1 unit = 0.001m (1mm), range 0~65.535m.
    Invalid pixels (value=0) are returned as 0.0.
    Note: not thread-safe; designed for single-thread-per-worker usage.

    Returns:
        (H, W) float32 array in meters
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load depth: {path}")
    return img.astype(np.float32) * DEPTH_QUANTIZATION_M


# ---------------------------------------------------------------------------
# Pose conversions
# ---------------------------------------------------------------------------
def tum_pose_to_matrix(
    position: list[float], quat_xyzw: list[float]
) -> np.ndarray:
    """Convert TUM pose (position + quaternion xyzw) to 4x4 matrix."""
    qx, qy, qz, qw = quat_xyzw
    rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses xyzw
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = position
    return mat


def euler_to_body_T_cam(
    pitch_deg: float, yaw_deg: float, roll_deg: float, camera_height: float
) -> np.ndarray:
    """
    Generate body_T_cam matrix matching habitat_sim's sensor orientation convention.

    This function correctly handles the coordinate system transformation from:
    - Body frame (Habitat): Y-up, X-right, Z-back
    - Camera frame (OpenCV): Y-down, X-right, Z-forward

    The transformation requires:
    1. A 180° rotation around X-axis to convert from body coordinates to camera coordinates
    2. Then apply the sensor orientation (pitch, yaw, roll)

    Args:
        pitch_deg: Pitch angle in degrees
        yaw_deg: Yaw angle in degrees
        roll_deg: Roll angle in degrees
        camera_height: Camera height in meters (Y-axis)

    Returns:
        4x4 body_T_cam transformation matrix
    """
    # Base rotation: convert from body (Y-up, Z-back) to camera (Y-down, Z-forward)
    R_base = Rotation.from_euler('x', 180, degrees=True)

    # Sensor orientation
    R_sensor = Rotation.from_euler('xyz', [pitch_deg, yaw_deg, roll_deg], degrees=True)

    # Total: sensor orientation applied after base rotation
    R_total = R_sensor * R_base

    body_T_cam = np.eye(4)
    body_T_cam[:3, :3] = R_total.as_matrix()
    body_T_cam[1, 3] = camera_height  # Y is the height axis
    return body_T_cam


def scale_intrinsics(
    intrinsics: np.ndarray, target_size: int, source_size: int = RENDER_INTRINSICS_SIZE
) -> np.ndarray:
    """
    Scale intrinsics from source_size to target_size resolution.

    Args:
        intrinsics: 3x3 intrinsics matrix (for source_size x source_size resolution)
        target_size: target resolution (e.g. 56)
        source_size: original resolution the intrinsics correspond to (step2: 518, step2.5: 224, etc.)

    Returns:
        3x3 intrinsics matrix (for target_size x target_size resolution)
    """
    scale = target_size / float(source_size)
    K = intrinsics.copy()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy
    return K


# ---------------------------------------------------------------------------
# Core single-frame overlap computation
# ---------------------------------------------------------------------------
def compute_mono_frame_overlap(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    world_T_cam_a: np.ndarray,
    world_T_cam_b: np.ndarray,
    K_a: np.ndarray,
    tolerance_m: float = 0.30,
    cam_a_T_world: np.ndarray | None = None,
    cam_b_T_world: np.ndarray | None = None,
    K_b: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Compute bidirectional single-frame mono overlap between cameras A and B.

    Args:
        depth_a, depth_b: depth maps
        world_T_cam_a, world_T_cam_b: camera extrinsics
        K_a: intrinsics matrix of camera A (matching the depth map resolution)
        tolerance_m: depth consistency tolerance
        cam_a_T_world: precomputed inverse of world_T_cam_a (optional, avoids redundant inversion)
        cam_b_T_world: precomputed inverse of world_T_cam_b (optional, avoids redundant inversion)
        K_b: intrinsics matrix of camera B. If None, same as K_a (backward compatible)

    Returns: (overlap_a2b, overlap_b2a, symmetric_overlap)
        - overlap_a2b: overlap rate in the A->B direction [0, 1]
        - overlap_b2a: overlap rate in the B->A direction [0, 1]
        - symmetric_overlap: min(A->B, B->A), ensuring sufficient overlap in both directions
    """
    if K_b is None:
        K_b = K_a
    # A->B: back-project with K_a, project with K_b
    overlap_a2b = _project_overlap(
        depth_a, depth_b, world_T_cam_a, K_a, K_b, tolerance_m,
        cam_tgt_T_world=cam_b_T_world,
    )
    # B->A: back-project with K_b, project with K_a
    overlap_b2a = _project_overlap(
        depth_b, depth_a, world_T_cam_b, K_b, K_a, tolerance_m,
        cam_tgt_T_world=cam_a_T_world,
    )
    symmetric_overlap = min(overlap_a2b, overlap_b2a)
    return overlap_a2b, overlap_b2a, symmetric_overlap


def _project_overlap(
    depth_src: np.ndarray,
    depth_tgt: np.ndarray,
    world_T_cam_src: np.ndarray,
    K_src: np.ndarray,
    K_tgt: np.ndarray,
    tolerance_m: float,
    cam_tgt_T_world: np.ndarray | None = None,
) -> float:
    """
    Directional overlap: fraction of src valid pixels visible in tgt.

    1. Find valid pixels (depth > 0)
    2. Back-project to 3D in src camera frame (using K_src)
    3. Transform: src_cam -> world -> tgt_cam
    4. Project to tgt pixel coords (using K_tgt)
    5. Check z>0, bounds [0, H/W), depth consistency

    Args:
        depth_src: (H_s, W_s) src depth map
        depth_tgt: (H_t, W_t) tgt depth map
        world_T_cam_src: (4, 4) src camera extrinsics
        K_src: (3, 3) src intrinsics matrix (used for back-projection)
        K_tgt: (3, 3) tgt intrinsics matrix (used for projection)
        tolerance_m: depth consistency tolerance (meters)
        cam_tgt_T_world: (4, 4) precomputed inverse of the tgt camera matrix

    Returns:
        Overlap score in [0, 1]
    """
    H_s, W_s = depth_src.shape
    H_t, W_t = depth_tgt.shape

    valid = depth_src > 0
    num_valid = int(valid.sum())
    if num_valid == 0:
        return 0.0

    vs, us = np.where(valid)
    depths = depth_src[vs, us]

    # Back-projection uses K_src
    fx_s, fy_s = K_src[0, 0], K_src[1, 1]
    cx_s, cy_s = K_src[0, 2], K_src[1, 2]

    x_cam = (us.astype(np.float32) - cx_s) / fx_s * depths
    y_cam = (vs.astype(np.float32) - cy_s) / fy_s * depths
    z_cam = depths
    pts_src = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # src_cam -> world
    R_src = world_T_cam_src[:3, :3]
    t_src = world_T_cam_src[:3, 3]
    pts_world = (R_src @ pts_src.T).T + t_src

    # world -> tgt_cam
    if cam_tgt_T_world is None:
        raise ValueError("cam_tgt_T_world must be provided (pre-computed inverse)")
    R_tgt_inv = cam_tgt_T_world[:3, :3]
    t_tgt_inv = cam_tgt_T_world[:3, 3]
    pts_tgt = (R_tgt_inv @ pts_world.T).T + t_tgt_inv

    z_tgt = pts_tgt[:, 2]
    front_mask = z_tgt > 0
    if not front_mask.any():
        return 0.0

    # Projection uses K_tgt
    fx_t, fy_t = K_tgt[0, 0], K_tgt[1, 1]
    cx_t, cy_t = K_tgt[0, 2], K_tgt[1, 2]

    u_tgt = fx_t * pts_tgt[:, 0] / z_tgt + cx_t
    v_tgt = fy_t * pts_tgt[:, 1] / z_tgt + cy_t

    bounds_mask = (
        (u_tgt >= 0) & (u_tgt < W_t) &
        (v_tgt >= 0) & (v_tgt < H_t)
    )
    proj_mask = front_mask & bounds_mask
    if not proj_mask.any():
        return 0.0

    u_int = np.clip(u_tgt[proj_mask].astype(np.int32), 0, W_t - 1)
    v_int = np.clip(v_tgt[proj_mask].astype(np.int32), 0, H_t - 1)
    tgt_depths = depth_tgt[v_int, u_int]

    tgt_valid = tgt_depths > 0
    depth_diff = np.abs(z_tgt[proj_mask] - tgt_depths)
    depth_consistent = tgt_valid & (depth_diff < tolerance_m)

    return int(depth_consistent.sum()) / num_valid


# ---------------------------------------------------------------------------
# Full overlap matrix computation
# ---------------------------------------------------------------------------
def compute_full_overlap_matrix(
    depths_a: list[np.ndarray],
    depths_b: list[np.ndarray],
    poses_a: list[np.ndarray],
    poses_b: list[np.ndarray],
    K_a: np.ndarray,
    tolerance_m: float = 0.30,
    K_b: np.ndarray | None = None,
) -> dict:
    """
    Compute full T_a x T_b frame-pair overlap matrix (no subsampling).

    Args:
        depths_a, depths_b: depth-map lists of the two sequences
        poses_a, poses_b: extrinsics lists of the two sequences
        K_a: intrinsics matrix of sequence A (matching the depth map resolution)
        tolerance_m: depth consistency tolerance
        K_b: intrinsics matrix of sequence B. If None, same as K_a (backward compatible)

    Returns:
        dict with:
            "matrix_a2b":         (T_a, T_b) float32 ndarray, A->B direction overlap
            "matrix_b2a":         (T_a, T_b) float32 ndarray, B->A direction overlap
            "matrix_symmetric":   (T_a, T_b) float32 ndarray, min(A->B, B->A)
            "T_a":                int
            "T_b":                int
            "max_overlap":        float (best symmetric overlap)
            "max_overlap_pair":   [int, int] (frame indices)
    """
    if K_b is None:
        K_b = K_a

    T_a = len(depths_a)
    T_b = len(depths_b)

    if T_a == 0 or T_b == 0:
        zeros = np.zeros((T_a, T_b), dtype=np.float32)
        return {
            "matrix_a2b": zeros.copy(),
            "matrix_b2a": zeros.copy(),
            "matrix_symmetric": zeros.copy(),
            "T_a": T_a,
            "T_b": T_b,
            "max_overlap": 0.0,
            "max_overlap_pair": [0, 0],
        }

    inv_poses_a = [np.linalg.inv(p) for p in poses_a]
    inv_poses_b = [np.linalg.inv(p) for p in poses_b]

    matrix_a2b = np.zeros((T_a, T_b), dtype=np.float32)
    matrix_b2a = np.zeros((T_a, T_b), dtype=np.float32)
    matrix_symmetric = np.zeros((T_a, T_b), dtype=np.float32)

    for i in range(T_a):
        for j in range(T_b):
            a2b, b2a, sym = compute_mono_frame_overlap(
                depths_a[i], depths_b[j],
                poses_a[i], poses_b[j],
                K_a, tolerance_m,
                cam_a_T_world=inv_poses_a[i],
                cam_b_T_world=inv_poses_b[j],
                K_b=K_b,
            )
            matrix_a2b[i, j] = a2b
            matrix_b2a[i, j] = b2a
            matrix_symmetric[i, j] = sym

    best_idx = np.unravel_index(np.argmax(matrix_symmetric), matrix_symmetric.shape)
    max_overlap = float(matrix_symmetric[best_idx])
    max_overlap_pair = [int(best_idx[0]), int(best_idx[1])]

    return {
        "matrix_a2b": matrix_a2b,
        "matrix_b2a": matrix_b2a,
        "matrix_symmetric": matrix_symmetric,
        "T_a": T_a,
        "T_b": T_b,
        "max_overlap": round(max_overlap, 4),
        "max_overlap_pair": max_overlap_pair,
    }


def compute_window_score(
    matrix: np.ndarray,
    start_a: int,
    start_b: int,
    window_size: int = 5,
    score_type: str = "symmetric",
) -> float:
    """
    Compute Max-Mean overlap score for a window_size x window_size sub-block.

    This is the score relevant to G2G training, where each group sends
    `window_size` frames into MapAnything.

    Args:
        matrix: (T_a, T_b) overlap matrix (can be a2b, b2a, or symmetric)
        start_a: start frame index in sequence A
        start_b: start frame index in sequence B
        window_size: number of frames per group (default: 5)
        score_type: "symmetric" (default), "mean", or "min"
            - "symmetric": Max-Mean bidirectional average
            - "mean": simple average over all frame pairs
            - "min": strictest criterion, takes min(max_a2b, max_b2a)

    Returns:
        Overlap score of the sub-block, in [0, 1]
    """
    T_a, T_b = matrix.shape
    end_a = min(start_a + window_size, T_a)
    end_b = min(start_b + window_size, T_b)

    sub = matrix[start_a:end_a, start_b:end_b]
    if sub.size == 0:
        return 0.0

    if score_type == "mean":
        return float(np.mean(sub))
    elif score_type == "min":
        # Strictest: each A frame finds its best B frame, each B frame finds its best A frame, take the min of the two
        a2b = float(np.mean(np.max(sub, axis=1)))
        b2a = float(np.mean(np.max(sub, axis=0)))
        return min(a2b, b2a)
    else:  # "symmetric" (default)
        # Max-Mean: bidirectional average
        a2b = float(np.mean(np.max(sub, axis=1)))
        b2a = float(np.mean(np.max(sub, axis=0)))
        return (a2b + b2a) / 2.0


# ---------------------------------------------------------------------------
# Representative camera selection
# ---------------------------------------------------------------------------
def _deterministic_seed(scene_id: str, traj_id: str) -> int:
    """Generate a deterministic seed from scene+traj for reproducibility."""
    key = f"{scene_id}|{traj_id}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def select_representative_cams(
    traj_id: str, scene_id: str
) -> list[int]:
    """
    Select representative cameras: cam_0 (front) + 2 random from {1,3,4,5,7}.

    Exclude cam_2 (hard-right, 90 deg) and cam_6 (hard-left, 270 deg), since
    hard-left/hard-right views have larger overlap bias.
    Randomly pick 2 from the remaining 5 non-front cameras {1,3,4,5,7}.
    Each trajectory always selects 3 representative cameras for overlap computation.
    The selection is determined deterministically by (scene_id, traj_id), so it is reproducible.

    Returns:
        sorted list of 3 camera indices, always including cam_0
    """
    rng = np.random.default_rng(_deterministic_seed(scene_id, traj_id))
    # Exclude cam_2 (hard-right, 90 deg) and cam_6 (hard-left, 270 deg)
    candidates = [1, 3, 4, 5, 7]
    others = rng.choice(candidates, size=2, replace=False)
    return sorted([0] + others.tolist())


# ---------------------------------------------------------------------------
# Body distance computation
# ---------------------------------------------------------------------------
def compute_body_min_distance(
    positions_a: np.ndarray, positions_b: np.ndarray
) -> float:
    """Min Euclidean distance between any pair of body positions."""
    N_a = len(positions_a)
    N_b = len(positions_b)

    if N_a * N_b <= 500_000:
        diff = positions_a[:, None, :] - positions_b[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return float(np.min(dists))

    min_dist = float("inf")
    chunk = 500
    for i in range(0, N_a, chunk):
        a_chunk = positions_a[i : i + chunk]
        diff = a_chunk[:, None, :] - positions_b[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_dist = min(min_dist, float(np.min(dists)))
    return min_dist


# ---------------------------------------------------------------------------
# Load view sequence depths and poses
# ---------------------------------------------------------------------------
def load_view_sequence(
    step3_traj_dir: str,
    trajectory: list[tuple[float, list[float], list[float]]],
    cam_idx: int,
    body_T_cam: np.ndarray,
    proj_size: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load depth maps and compute world_T_cam poses for a view sequence.

    Args:
        step3_traj_dir: Step3 trajectory directory
        trajectory: TUM-format trajectory list
        cam_idx: camera index
        body_T_cam: body-to-camera transformation matrix
        proj_size: projection resolution. If specified and different from the
                   original depth-map size, downsample to proj_size x proj_size
                   (using INTER_NEAREST to preserve invalid-pixel semantics)

    Returns:
        (depths, world_T_cams)
    """
    depths = []
    world_T_cams = []

    for timestamp, position, quat_xyzw in trajectory:
        ts_str = f"{int(timestamp * 1000):010d}"
        depth_path = os.path.join(
            step3_traj_dir, "depth", ts_str, f"cam_{cam_idx}.png"
        )
        depth = load_depth_png(depth_path)

        # Downsample the depth map to the projection resolution (only used for coarse screening, no need for full-resolution precision)
        if proj_size is not None and depth.shape[0] != proj_size:
            depth = cv2.resize(
                depth, (proj_size, proj_size),
                interpolation=cv2.INTER_NEAREST,
            )

        depths.append(depth)

        world_T_body = tum_pose_to_matrix(position, quat_xyzw)
        world_T_cam = world_T_body @ body_T_cam
        world_T_cams.append(world_T_cam)

    return depths, world_T_cams


# ---------------------------------------------------------------------------
# Process single scene
# ---------------------------------------------------------------------------
def process_scene(
    scene_id: str,
    step1_scene_dir: str,
    step2_scene_dir: str,
    step3_scene_dir: str,
    output_dir: str,
    depth_tolerance: float,
    body_distance_cutoff: float,
    image_size: int = 224,
    proj_size: int = 56,
) -> dict | None:
    """
    Process one scene: compute full overlap matrices for all view-sequence pairs.

    For each pair, saves:
      - {output_dir}/matrices/{pair_id}.npy  (T_a x T_b float32 matrix)
      - entry in view_sequence_pairs.json with summary stats

    Returns:
        scene result dict, or None on error
    """
    step1_traj_parent = os.path.join(step1_scene_dir, "trajectories")
    step2_traj_parent = os.path.join(step2_scene_dir, "trajectories")
    step3_traj_parent = os.path.join(step3_scene_dir, "trajectories")

    if not os.path.isdir(step1_traj_parent):
        print(f"  [SKIP] No trajectories dir: {step1_traj_parent}")
        return None

    if not os.path.isdir(step3_traj_parent):
        print(f"  [SKIP] No Step 3 trajectories dir: {step3_traj_parent}")
        return None

    # Only process trajectories actually rendered in Step 3 (intersection of Step 1 and Step 3)
    step1_trajs = set(
        d for d in os.listdir(step1_traj_parent)
        if os.path.isdir(os.path.join(step1_traj_parent, d))
    )
    step3_trajs = set(
        d for d in os.listdir(step3_traj_parent)
        if os.path.isdir(os.path.join(step3_traj_parent, d))
    )
    traj_names = sorted(step1_trajs & step3_trajs)

    if len(step1_trajs) != len(step3_trajs):
        print(f"  [INFO] Step 1 has {len(step1_trajs)} trajs, Step 3 has {len(step3_trajs)} trajs, using intersection: {len(traj_names)}")
    if len(traj_names) < 2:
        print(f"  [SKIP] Need >=2 trajectories, found {len(traj_names)}")
        return None

    # Pre-load all trajectories and rig configs
    traj_data = {}
    for traj_name in traj_names:
        tum_path = os.path.join(step1_traj_parent, traj_name, "trajectory.tum")
        rig_path = os.path.join(step2_traj_parent, traj_name, "rig_config.json")

        if not os.path.isfile(tum_path) or not os.path.isfile(rig_path):
            continue

        trajectory = load_trajectory_tum(tum_path)
        if not trajectory:
            continue

        with open(rig_path, "r", encoding="utf-8") as f:
            rig_config = json.load(f)

        positions = np.array([frame[1] for frame in trajectory])

        body_T_cams = {}
        camera_height = rig_config["camera_height_m"]
        for cam in rig_config["cameras"]:
            # Use euler angles with correct xyz intrinsic order (matching habitat_sim)
            # instead of the incorrectly saved body_T_cam matrix (which used ZYX order)
            euler = cam["actual_euler_deg"]
            body_T_cams[cam["index"]] = euler_to_body_T_cam(
                euler["pitch"], euler["yaw"], euler["roll"], camera_height
            )

        # per-camera intrinsics: support step2.5 random intrinsics (different per camera)
        intrinsics_base_size = rig_config.get("width", RENDER_INTRINSICS_SIZE)
        per_cam_intrinsics = {}
        for cam in rig_config["cameras"]:
            per_cam_intrinsics[cam["index"]] = np.array(cam["intrinsics"])

        traj_data[traj_name] = {
            "trajectory": trajectory,
            "positions": positions,
            "rig_config": rig_config,
            "body_T_cams": body_T_cams,
            "per_cam_intrinsics": per_cam_intrinsics,
            "intrinsics_base_size": intrinsics_base_size,
        }

    valid_traj_names = sorted(traj_data.keys())
    num_traj = len(valid_traj_names)
    if num_traj < 2:
        print(f"  [SKIP] Need >=2 valid trajectories, found {num_traj}")
        return None

    print(f"  {num_traj} valid trajectories")

    # Create matrices output directory
    matrices_dir = os.path.join(output_dir, "matrices")
    os.makedirs(matrices_dir, exist_ok=True)

    pairs = []
    per_traj_cams = {}
    num_computed = 0
    num_skipped_distance = 0
    pair_counter = 0

    for traj_i, traj_j in combinations(valid_traj_names, 2):
        data_i = traj_data[traj_i]
        data_j = traj_data[traj_j]

        body_dist = compute_body_min_distance(
            data_i["positions"], data_j["positions"]
        )

        # Select 3 representative cameras per trajectory (cam_0 + 2 from {1,3,4,5,7})
        cams_i = select_representative_cams(traj_i, scene_id)
        cams_j = select_representative_cams(traj_j, scene_id)

        if traj_i not in per_traj_cams:
            per_traj_cams[traj_i] = {}
        per_traj_cams[traj_i]["representative_cams"] = cams_i

        if traj_j not in per_traj_cams:
            per_traj_cams[traj_j] = {}
        per_traj_cams[traj_j]["representative_cams"] = cams_j

        step3_i_dir = os.path.join(step3_traj_parent, traj_i)
        step3_j_dir = os.path.join(step3_traj_parent, traj_j)

        # Combinations (not permutations): only process one direction, the overlap matrix already contains a2b and b2a
        for cam_i, cam_j in product(cams_i, cams_j):
            pair_id = f"pair_{pair_counter:06d}"
            pair_counter += 1

            if body_dist > body_distance_cutoff:
                T_i = len(data_i["trajectory"])
                T_j = len(data_j["trajectory"])
                pair_entry = {
                    "pair_id": pair_id,
                    "seq_a": {"traj": traj_i, "cam": cam_i, "num_frames": T_i},
                    "seq_b": {"traj": traj_j, "cam": cam_j, "num_frames": T_j},
                    "max_overlap": 0.0,
                    "max_overlap_pair": [0, 0],
                    "matrix_file": None,
                    "body_distance_min_m": round(body_dist, 2),
                }
                pairs.append(pair_entry)
                num_skipped_distance += 1
                continue

            # Load view sequences (depth maps downsampled to proj_size for projection)
            depths_i, poses_i = load_view_sequence(
                step3_i_dir,
                data_i["trajectory"],
                cam_i,
                data_i["body_T_cams"][cam_i],
                proj_size=proj_size,
            )
            depths_j, poses_j = load_view_sequence(
                step3_j_dir,
                data_j["trajectory"],
                cam_j,
                data_j["body_T_cams"][cam_j],
                proj_size=proj_size,
            )

            # Compute projection-resolution intrinsics for each of the two cameras in this pair
            K_i = scale_intrinsics(
                data_i["per_cam_intrinsics"][cam_i],
                proj_size,
                source_size=data_i["intrinsics_base_size"],
            )
            K_j = scale_intrinsics(
                data_j["per_cam_intrinsics"][cam_j],
                proj_size,
                source_size=data_j["intrinsics_base_size"],
            )

            # Compute full overlap matrix
            result = compute_full_overlap_matrix(
                depths_i, depths_j,
                poses_i, poses_j,
                K_i,
                tolerance_m=depth_tolerance,
                K_b=K_j,
            )

            # Save matrices as .npz (quantized to uint8, saving 75% of storage space)
            matrix_filename = f"{pair_id}.npz"
            matrix_path = os.path.join(matrices_dir, matrix_filename)
            np.savez_compressed(
                matrix_path,
                a2b=quantize_overlap(result["matrix_a2b"]),
                b2a=quantize_overlap(result["matrix_b2a"]),
                symmetric=quantize_overlap(result["matrix_symmetric"]),
            )

            pair_entry = {
                "pair_id": pair_id,
                "seq_a": {
                    "traj": traj_i,
                    "cam": cam_i,
                    "num_frames": result["T_a"],
                },
                "seq_b": {
                    "traj": traj_j,
                    "cam": cam_j,
                    "num_frames": result["T_b"],
                },
                "max_overlap": result["max_overlap"],
                "max_overlap_pair": result["max_overlap_pair"],
                "matrix_file": f"matrices/{matrix_filename}",
                "body_distance_min_m": round(body_dist, 2),
            }
            pairs.append(pair_entry)
            num_computed += 1

    # Clear depth cache for this scene
    load_depth_png.cache_clear()

    print(
        f"  Computed: {num_computed}, skipped (distance): {num_skipped_distance}"
    )

    scene_result = {
        "scene_id": scene_id,
        "num_trajectories": num_traj,
        "num_pairs_computed": len(pairs),
        "overlap_method": "depth_projection_mono_full_matrix",
        "depth_tolerance_m": depth_tolerance,
        "body_distance_cutoff_m": body_distance_cutoff,
        "pairs": pairs,
        "per_traj_representative_cams": per_traj_cams,
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "view_sequence_pairs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_result, f, indent=2, ensure_ascii=False)

    return scene_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    step1_scenes_dir = os.path.join(args.step1_root, "scenes")
    step2_scenes_dir = os.path.join(args.step2_root, "scenes")
    step3_scenes_dir = os.path.join(args.step3_root, "scenes")

    if not os.path.isdir(step1_scenes_dir):
        print(f"[ERROR] Step 1 scenes dir not found: {step1_scenes_dir}")
        return
    if not os.path.isdir(step3_scenes_dir):
        print(f"[ERROR] Step 3 scenes dir not found: {step3_scenes_dir}")
        return

    output_scenes_dir = os.path.join(args.output_root, "scenes")
    os.makedirs(output_scenes_dir, exist_ok=True)

    print("=== HM3D G2G Overlap Computation (Step 4 - uint16 low-resolution fast version) ===")
    print(f"  Step 1 root: {os.path.abspath(args.step1_root)}")
    print(f"  Step 2 root: {os.path.abspath(args.step2_root)}")
    print(f"  Step 3 root: {os.path.abspath(args.step3_root)}")
    print(f"  Output root: {os.path.abspath(args.output_root)}")
    print(f"  Image size: {args.image_size}x{args.image_size} (load)")
    print(f"  Proj size: {args.proj_size}x{args.proj_size} (projection)")
    print(f"  Depth format: uint16 (1mm precision, range 0~65.535m)")
    print(f"  Depth tolerance: {args.depth_tolerance}m")
    print(f"  Body distance cutoff: {args.body_distance_cutoff}m")
    print(f"  Camera selection: cam_0 + 2 from {{1,3,4,5,7}} (3 per traj)")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    scene_names = sorted(
        d for d in os.listdir(step1_scenes_dir)
        if os.path.isdir(os.path.join(step1_scenes_dir, d))
    )
    if args.max_scenes > 0:
        scene_names = scene_names[: args.max_scenes]

    total_scenes = len(scene_names)
    if args.num_workers > 1:
        scene_names = scene_names[args.worker_id :: args.num_workers]
        print(
            f"Total {total_scenes} scenes, "
            f"Worker {args.worker_id}/{args.num_workers} handles {len(scene_names)}"
        )
    else:
        print(f"Total {total_scenes} scenes")
    print()

    t_start = time.time()
    total_pairs = 0
    scenes_processed = 0

    for scene_idx, scene_name in enumerate(scene_names):
        print(f"[{scene_idx + 1}/{len(scene_names)}] Scene: {scene_name}")

        output_scene_dir = os.path.join(output_scenes_dir, scene_name)

        if args.skip_existing:
            output_file = os.path.join(
                output_scene_dir, "view_sequence_pairs.json"
            )
            if os.path.isfile(output_file):
                print("  [SKIP] Already exists")
                continue

        step1_scene = os.path.join(step1_scenes_dir, scene_name)
        step2_scene = os.path.join(step2_scenes_dir, scene_name)
        step3_scene = os.path.join(step3_scenes_dir, scene_name)

        if not os.path.isdir(step3_scene):
            print(f"  [SKIP] Step 3 dir not found: {step3_scene}")
            continue

        result = process_scene(
            scene_name,
            step1_scene,
            step2_scene,
            step3_scene,
            output_scene_dir,
            args.depth_tolerance,
            args.body_distance_cutoff,
            image_size=args.image_size,
            proj_size=args.proj_size,
        )

        if result is not None:
            total_pairs += result["num_pairs_computed"]
            scenes_processed += 1

        print()

    elapsed = time.time() - t_start
    print("=== Done ===")
    print(f"  Scenes processed: {scenes_processed}")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()

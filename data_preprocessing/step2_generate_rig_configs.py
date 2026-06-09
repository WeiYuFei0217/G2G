"""
step2_generate_rig_configs.py — HM3D G2G data collection Step 2

Generate rig_config.json for each trajectory, containing the randomly perturbed
poses and intrinsics of 8 cameras.
- Read rig_perturbation_seed and camera_height_m from traj_meta.json
- 8-camera nominal yaw: [0, 45, 90, 135, 180, 225, 270, 315]°
- Perturbation distribution: truncated normal N(0, 10²), truncated at ±30°
- Output body_T_cam transformation matrix (ZYX Euler angles → 4x4)

Usage:
python step2_generate_rig_configs.py \
    --data_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --hfov 90.0 \
    --width 518 \
    --height 518

python step2_generate_rig_configs.py \
    --data_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --hfov 90.0 \
    --width 518 \
    --height 518
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_NAMES = [
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
]
NOMINAL_YAWS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HM3D G2G rig config generation (Step 2): generate perturbed poses and intrinsics for 8 cameras per trajectory"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Step 1 output directory (contains scenes/ subdirectory, reads traj_meta.json)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Step 2 output directory (writes rig_config.json)",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90.0,
        help="Horizontal field of view (degrees, default: 90.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=518,
        help="Image width (pixels, default: 518)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=518,
        help="Image height (pixels, default: 518)",
    )
    parser.add_argument(
        "--perturbation_sigma",
        type=float,
        default=10.0,
        help="Perturbation standard deviation (degrees, default: 10.0)",
    )
    parser.add_argument(
        "--perturbation_truncate",
        type=float,
        default=30.0,
        help="Perturbation truncation range (degrees, default: 30.0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Truncated normal sampling
# ---------------------------------------------------------------------------
def truncated_normal(
    rng: np.random.Generator,
    mean: float,
    sigma: float = 10.0,
    truncate: float = 30.0,
) -> float:
    """
    Sample from a truncated normal distribution.

    Args:
        rng: numpy random number generator
        mean: mean value
        sigma: standard deviation
        truncate: truncation range (|sample - mean| <= truncate)

    Returns:
        sampled value
    """
    while True:
        sample = float(rng.normal(mean, sigma))
        if abs(sample - mean) <= truncate:
            return sample


# ---------------------------------------------------------------------------
# Euler angles → body_T_cam transformation matrix
# ---------------------------------------------------------------------------
def euler_to_body_T_cam(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    camera_height: float,
) -> np.ndarray:
    """
    Convert Euler angles (ZYX order) + camera height into a 4x4 body_T_cam transformation matrix.

    Conventions:
    - ZYX Euler angles: first rotate yaw about the Z axis, then pitch about the Y axis, finally roll about the X axis
    - Translation: [0, camera_height, 0] (Y-up coordinate system, Y is the height axis)

    Args:
        roll_deg: roll angle (degrees)
        pitch_deg: pitch angle (degrees)
        yaw_deg: yaw angle (degrees)
        camera_height: camera mounting height (meters)

    Returns:
        4x4 body_T_cam transformation matrix
    """
    rot = Rotation.from_euler(
        "ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True
    )
    body_T_cam = np.eye(4)
    body_T_cam[:3, :3] = rot.as_matrix()
    body_T_cam[1, 3] = camera_height  # Y is the height axis
    return body_T_cam


# ---------------------------------------------------------------------------
# Intrinsics computation
# ---------------------------------------------------------------------------
def compute_intrinsics(
    hfov_deg: float, width: int, height: int
) -> list[list[float]]:
    """
    Compute the 3x3 intrinsics matrix from horizontal field of view and resolution.

    Args:
        hfov_deg: horizontal field of view (degrees)
        width: image width (pixels)
        height: image height (pixels)

    Returns:
        3x3 intrinsics matrix (nested list)
    """
    fx = width / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return [
        [round(fx, 4), 0.0, round(cx, 4)],
        [0.0, round(fy, 4), round(cy, 4)],
        [0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Generate the rig config for a single trajectory
# ---------------------------------------------------------------------------
def generate_rig_config(
    traj_meta: dict,
    hfov_deg: float,
    width: int,
    height: int,
    sigma: float,
    truncate: float,
) -> dict:
    """
    Generate the 8-camera rig config based on the seed and camera height in traj_meta.json.

    Args:
        traj_meta: trajectory metadata (loaded from traj_meta.json)
        hfov_deg: horizontal field of view
        width: image width
        height: image height
        sigma: perturbation standard deviation
        truncate: perturbation truncation range

    Returns:
        rig_config dictionary
    """
    seed = traj_meta["rig_perturbation_seed"]
    camera_height = traj_meta["camera_height_m"]
    rng = np.random.default_rng(seed)

    intrinsics = compute_intrinsics(hfov_deg, width, height)

    cameras = []
    for idx, (name, nominal_yaw) in enumerate(
        zip(CAMERA_NAMES, NOMINAL_YAWS_DEG)
    ):
        # Independently sample roll/pitch/yaw perturbations
        roll_deg = truncated_normal(rng, 0.0, sigma, truncate)
        pitch_deg = truncated_normal(rng, 0.0, sigma, truncate)
        # yaw is added on top of the nominal value
        yaw_deg = truncated_normal(rng, nominal_yaw, sigma, truncate)

        body_T_cam = euler_to_body_T_cam(
            roll_deg, pitch_deg, yaw_deg, camera_height
        )

        cam_entry = {
            "name": name,
            "index": idx,
            "nominal_yaw_deg": nominal_yaw,
            "actual_euler_deg": {
                "roll": round(roll_deg, 4),
                "pitch": round(pitch_deg, 4),
                "yaw": round(yaw_deg, 4),
            },
            "body_T_cam": [
                [round(v, 8) for v in row] for row in body_T_cam.tolist()
            ],
            "intrinsics": intrinsics,
        }
        cameras.append(cam_entry)

    rig_config = {
        "num_cameras": len(cameras),
        "coordinate_convention": "opencv",
        "hfov_deg": hfov_deg,
        "width": width,
        "height": height,
        "camera_height_m": camera_height,
        "perturbation_seed": seed,
        "perturbation_sigma_deg": sigma,
        "perturbation_truncate_deg": truncate,
        "cameras": cameras,
    }
    return rig_config


# ---------------------------------------------------------------------------
# Process a single trajectory
# ---------------------------------------------------------------------------
def process_trajectory(
    input_traj_dir: str,
    output_traj_dir: str,
    hfov_deg: float,
    width: int,
    height: int,
    sigma: float,
    truncate: float,
) -> bool:
    """
    Process a single trajectory: read traj_meta.json from input_traj_dir → generate rig_config.json in output_traj_dir.

    Args:
        input_traj_dir: input trajectory directory path (Step 1 output)
        output_traj_dir: output trajectory directory path (Step 2 output)
        hfov_deg, width, height: camera intrinsics parameters
        sigma, truncate: perturbation parameters

    Returns:
        True for success, False for skipped/failed
    """
    meta_path = os.path.join(input_traj_dir, "traj_meta.json")
    if not os.path.isfile(meta_path):
        print(f"  [SKIP] Missing traj_meta.json: {input_traj_dir}")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        traj_meta = json.load(f)

    rig_config = generate_rig_config(
        traj_meta, hfov_deg, width, height, sigma, truncate
    )

    os.makedirs(output_traj_dir, exist_ok=True)
    output_path = os.path.join(output_traj_dir, "rig_config.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rig_config, f, indent=2, ensure_ascii=False)

    return True


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    input_root = args.data_root
    output_root = args.output_root

    input_scenes_dir = os.path.join(input_root, "scenes")
    if not os.path.isdir(input_scenes_dir):
        print(f"[ERROR] Input scenes directory does not exist: {input_scenes_dir}")
        return

    output_scenes_dir = os.path.join(output_root, "scenes")
    os.makedirs(output_scenes_dir, exist_ok=True)

    print("=== HM3D G2G rig config generation (Step 2) ===")
    print(f"  Input directory (Step 1): {os.path.abspath(input_root)}")
    print(f"  Output directory (Step 2): {os.path.abspath(output_root)}")
    print(f"  HFOV: {args.hfov}°, resolution: {args.width}x{args.height}")
    print(
        f"  Perturbation parameters: sigma={args.perturbation_sigma}°, "
        f"truncate=±{args.perturbation_truncate}°"
    )
    print()

    t_start = time.time()
    total_success = 0
    total_skip = 0

    scene_dirs = sorted(
        d
        for d in os.listdir(input_scenes_dir)
        if os.path.isdir(os.path.join(input_scenes_dir, d))
    )

    for scene_idx, scene_name in enumerate(scene_dirs):
        input_traj_parent = os.path.join(input_scenes_dir, scene_name, "trajectories")
        if not os.path.isdir(input_traj_parent):
            print(f"[{scene_idx + 1}/{len(scene_dirs)}] {scene_name}: no trajectories directory, skipping")
            continue

        output_traj_parent = os.path.join(output_scenes_dir, scene_name, "trajectories")
        os.makedirs(output_traj_parent, exist_ok=True)

        traj_names = sorted(
            d
            for d in os.listdir(input_traj_parent)
            if os.path.isdir(os.path.join(input_traj_parent, d))
        )

        scene_success = 0
        for traj_name in traj_names:
            input_traj_dir = os.path.join(input_traj_parent, traj_name)
            output_traj_dir = os.path.join(output_traj_parent, traj_name)
            ok = process_trajectory(
                input_traj_dir,
                output_traj_dir,
                args.hfov,
                args.width,
                args.height,
                args.perturbation_sigma,
                args.perturbation_truncate,
            )
            if ok:
                scene_success += 1
                total_success += 1
            else:
                total_skip += 1

        print(
            f"[{scene_idx + 1}/{len(scene_dirs)}] {scene_name}: "
            f"rig_config generated for {scene_success}/{len(traj_names)} trajectories"
        )

    elapsed = time.time() - t_start
    print()
    print("=== Done ===")
    print(f"  Success: {total_success}, Skipped: {total_skip}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.abspath(output_root)}")


if __name__ == "__main__":
    main()

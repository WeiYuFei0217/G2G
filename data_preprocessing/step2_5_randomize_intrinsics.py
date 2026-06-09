"""
step2_5_randomize_intrinsics.py — HM3D G2G data generation, Step 2.5 (optional)

Reads the Step 2 rig_config.json and assigns each camera an independent random
intrinsic matrix (per-camera HFOV), then writes a new rig_config.json.

Why randomize intrinsics:
  Following the training recipe of MapAnything and VGGT, exposing the model to a
  wide range of camera intrinsics improves generalization. We sample HFOV in
  [45 deg, 120 deg], which covers the common range from DSLR tele (~45 deg) and
  smartphones (~70 deg) to action cameras (~120 deg).

Design:
  - Image resolution is fixed (default 224x224).
  - Within a trajectory, a given camera keeps the same intrinsics across all frames.
  - Different cameras may have different intrinsics (different FOV / focal length).
  - fx = fy is derived from HFOV and the image width (square-pixel assumption).
  - The principal point is fixed at the image center (cx = W/2, cy = H/2).
    habitat_sim's CameraSensorSpec only accepts an HFOV and always renders with a
    centered principal point, so the stored intrinsics must also use a centered
    principal point to stay consistent with the rendered images; otherwise the
    depth projection in Step 4 / covisibility would carry a systematic error.
  - The RNG is seeded with traj_meta["rig_perturbation_seed"] + INTRINSIC_SEED_OFFSET,
    so the sampling is reproducible and independent of the extrinsics perturbation.

Usage:
# Basic (read from step2, write to step2_5)
python step2_5_randomize_intrinsics.py \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --image_size 224

# Custom FOV range
python step2_5_randomize_intrinsics.py \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --image_size 224 \
    --hfov_min 60 --hfov_max 120

python step2_5_randomize_intrinsics.py \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_val \
    --image_size 224 \
    --hfov_min 60 --hfov_max 120
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


INTRINSIC_SEED_OFFSET = 314159


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HM3D G2G Step 2.5: assign each camera a random intrinsic "
        "matrix (per-camera HFOV)"
    )
    parser.add_argument(
        "--step2_root",
        type=str,
        required=True,
        help="Step 2 output directory (reads rig_config.json)",
    )
    parser.add_argument(
        "--step1_root",
        type=str,
        required=True,
        help="Step 1 output directory (reads traj_meta.json for the seed)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Step 2.5 output directory (writes rig_config.json with random intrinsics)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target image resolution, used for both width and height (default: 224)",
    )
    parser.add_argument(
        "--hfov_min",
        type=float,
        default=45.0,
        help="Minimum horizontal field of view (degrees, default: 45.0)",
    )
    parser.add_argument(
        "--hfov_max",
        type=float,
        default=120.0,
        help="Maximum horizontal field of view (degrees, default: 120.0)",
    )
    return parser.parse_args()


def compute_intrinsics_from_hfov(
    hfov_deg: float,
    width: int,
    height: int,
) -> list[list[float]]:
    """
    Compute a 3x3 intrinsic matrix from HFOV and resolution.

    Pinhole camera: fx = width / (2 * tan(hfov / 2)), fy = fx (square pixels).
    The principal point is fixed at the image center (width / 2, height / 2),
    matching how habitat_sim renders (it always uses a centered principal point).
    """
    fx = width / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return [
        [round(fx, 4), 0.0, round(cx, 4)],
        [0.0, round(fy, 4), round(cy, 4)],
        [0.0, 0.0, 1.0],
    ]


def randomize_rig_intrinsics(
    rig_config: dict,
    seed: int,
    image_size: int,
    hfov_min: float,
    hfov_max: float,
) -> dict:
    """
    Assign each camera in the rig an independent random intrinsic matrix.

    Per-camera sampling:
      1. HFOV ~ Uniform(hfov_min, hfov_max)
      2. fx = fy = image_size / (2 * tan(hfov / 2))
      3. cx = cy = image_size / 2 (centered principal point)
    """
    rng = np.random.default_rng(seed)
    new_config = dict(rig_config)
    new_config["width"] = image_size
    new_config["height"] = image_size
    new_config["intrinsics_randomized"] = True
    new_config["intrinsics_hfov_range_deg"] = [hfov_min, hfov_max]
    new_config["intrinsics_seed"] = seed

    new_cameras = []
    for cam in rig_config["cameras"]:
        cam_hfov = float(rng.uniform(hfov_min, hfov_max))
        intrinsics = compute_intrinsics_from_hfov(cam_hfov, image_size, image_size)

        new_cam = dict(cam)
        new_cam["hfov_deg"] = round(cam_hfov, 4)
        new_cam["intrinsics"] = intrinsics
        new_cameras.append(new_cam)

    new_config["cameras"] = new_cameras
    return new_config


def process_trajectory(
    step2_traj_dir: str,
    step1_traj_dir: str,
    output_traj_dir: str,
    image_size: int,
    hfov_min: float,
    hfov_max: float,
) -> bool:
    rig_path = os.path.join(step2_traj_dir, "rig_config.json")
    meta_path = os.path.join(step1_traj_dir, "traj_meta.json")

    if not os.path.isfile(rig_path):
        print(f"  [SKIP] missing rig_config.json: {step2_traj_dir}")
        return False
    if not os.path.isfile(meta_path):
        print(f"  [SKIP] missing traj_meta.json: {step1_traj_dir}")
        return False

    with open(rig_path, "r", encoding="utf-8") as f:
        rig_config = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        traj_meta = json.load(f)

    base_seed = traj_meta["rig_perturbation_seed"]
    intrinsic_seed = base_seed + INTRINSIC_SEED_OFFSET

    new_config = randomize_rig_intrinsics(
        rig_config,
        seed=intrinsic_seed,
        image_size=image_size,
        hfov_min=hfov_min,
        hfov_max=hfov_max,
    )

    os.makedirs(output_traj_dir, exist_ok=True)
    output_path = os.path.join(output_traj_dir, "rig_config.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    return True


def main():
    args = parse_args()

    step2_scenes_dir = os.path.join(args.step2_root, "scenes")
    step1_scenes_dir = os.path.join(args.step1_root, "scenes")
    output_scenes_dir = os.path.join(args.output_root, "scenes")

    if not os.path.isdir(step2_scenes_dir):
        print(f"[ERROR] Step 2 scenes directory not found: {step2_scenes_dir}")
        return
    if not os.path.isdir(step1_scenes_dir):
        print(f"[ERROR] Step 1 scenes directory not found: {step1_scenes_dir}")
        return

    os.makedirs(output_scenes_dir, exist_ok=True)

    print("=== HM3D G2G random intrinsics generation (Step 2.5) ===")
    print(f"  Input dir (Step 2):   {os.path.abspath(args.step2_root)}")
    print(f"  Input dir (Step 1):   {os.path.abspath(args.step1_root)}")
    print(f"  Output dir (Step 2.5): {os.path.abspath(args.output_root)}")
    print(f"  Image resolution: {args.image_size}x{args.image_size}")
    print(f"  HFOV range: [{args.hfov_min} deg, {args.hfov_max} deg]")
    print("  Principal point: fixed at image center (habitat_sim constraint)")
    print()

    # Preview the focal-length range implied by the HFOV range.
    fx_at_min_hfov = args.image_size / (
        2.0 * np.tan(np.radians(args.hfov_min / 2.0))
    )
    fx_at_max_hfov = args.image_size / (
        2.0 * np.tan(np.radians(args.hfov_max / 2.0))
    )
    print(
        f"  Focal length range ({args.image_size}px): "
        f"fx in [{fx_at_max_hfov:.1f}, {fx_at_min_hfov:.1f}] pixels"
    )
    print(
        f"  35mm-equivalent: "
        f"[{18*fx_at_max_hfov/args.image_size*2:.0f}mm, "
        f"{18*fx_at_min_hfov/args.image_size*2:.0f}mm]"
    )
    print()

    t_start = time.time()
    total_success = 0
    total_skip = 0

    scene_dirs = sorted(
        d
        for d in os.listdir(step2_scenes_dir)
        if os.path.isdir(os.path.join(step2_scenes_dir, d))
    )

    for scene_idx, scene_name in enumerate(scene_dirs):
        step2_traj_parent = os.path.join(
            step2_scenes_dir, scene_name, "trajectories"
        )
        step1_traj_parent = os.path.join(
            step1_scenes_dir, scene_name, "trajectories"
        )
        output_traj_parent = os.path.join(
            output_scenes_dir, scene_name, "trajectories"
        )

        if not os.path.isdir(step2_traj_parent):
            print(
                f"[{scene_idx + 1}/{len(scene_dirs)}] "
                f"{scene_name}: no Step 2 trajectories directory, skip"
            )
            continue
        if not os.path.isdir(step1_traj_parent):
            print(
                f"[{scene_idx + 1}/{len(scene_dirs)}] "
                f"{scene_name}: no Step 1 trajectories directory, skip"
            )
            continue

        traj_names = sorted(
            d
            for d in os.listdir(step2_traj_parent)
            if os.path.isdir(os.path.join(step2_traj_parent, d))
        )

        scene_success = 0
        for traj_name in traj_names:
            step2_traj_dir = os.path.join(step2_traj_parent, traj_name)
            step1_traj_dir = os.path.join(step1_traj_parent, traj_name)
            output_traj_dir = os.path.join(output_traj_parent, traj_name)

            ok = process_trajectory(
                step2_traj_dir,
                step1_traj_dir,
                output_traj_dir,
                args.image_size,
                args.hfov_min,
                args.hfov_max,
            )
            if ok:
                scene_success += 1
                total_success += 1
            else:
                total_skip += 1

        print(
            f"[{scene_idx + 1}/{len(scene_dirs)}] {scene_name}: "
            f"{scene_success}/{len(traj_names)} trajectories assigned random intrinsics"
        )

    elapsed = time.time() - t_start
    print()
    print("=== Done ===")
    print(f"  Success: {total_success}, skipped: {total_skip}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output dir: {os.path.abspath(args.output_root)}")
    print()
    print("Next: point step3 --step2_root at this output directory:")
    print(
        f"  python step3_render_rgb_depth.py \\\n"
        f"      --step1_root <step1_root> \\\n"
        f"      --step2_root {os.path.abspath(args.output_root)} \\\n"
        f"      --output_root <output_root> \\\n"
        f"      --image_size {args.image_size}"
    )


if __name__ == "__main__":
    main()

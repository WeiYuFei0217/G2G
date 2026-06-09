"""
step3_render_rgb_depth.py — HM3D G2G data collection Step 3 (RGB + same-resolution uint16 depth)

This script renders RGB images together with same-resolution uint16 depth maps:
  - Depth maps keep the same resolution as RGB (no downsampling)
  - Depth values are stored as uint16 PNG (1mm precision, range 0~65.535m)
  - There is no --depth_size argument

It reads the Step 1 trajectory (trajectory.tum, scene_info.json) + the Step 2 rig config (rig_config.json),
and uses habitat_sim.Simulator (low-level API) to render RGB and depth maps.

Output:
    {output_root}/scenes/{scene_id}/trajectories/{traj_id}/images/{timestamp}/cam_0.jpg ... cam_7.jpg
    {output_root}/scenes/{scene_id}/trajectories/{traj_id}/depth/{timestamp}/cam_0.png ... cam_7.png
        (depth maps at the same resolution as RGB, uint16 PNG, 1mm quantization, range 0~65.535m)

Usage:
# RGB 224x224, Depth 224x224 uint16
python step3_render_rgb_depth.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_val \
    --image_size 224 \
    --gpu_id 0 \
    --max_scenes -1

# RGB 224x224, Depth 224x224 uint16, step2.5 intrinsics version
python step3_render_rgb_depth.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_val \
    --image_size 224 \
    --gpu_id 0 \
    --max_scenes -1

# RGB 512x512, Depth 512x512 uint16
python step3_render_rgb_depth.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_val \
    --output_root /path/to/data/HM3D/DATA_GEN/step3_render_512_512_uint16_val \
    --image_size 512 \
    --gpu_id 0 \
    --max_scenes -1
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import of Habitat (GPU environment variables must be set first)
# ---------------------------------------------------------------------------
habitat_sim = None
quaternion = None


def _lazy_import_habitat(gpu_id: int = 0):
    """Import habitat_sim only after setting the GPU environment variables, to avoid EGL initialization issues."""
    global habitat_sim, quaternion

    os.environ["HABITAT_SIM_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import habitat_sim as _hsim
    import quaternion as _quat

    habitat_sim = _hsim
    quaternion = _quat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HM3D_ROOT = "/path/to/data/HM3D"

# JPEG compression quality
JPEG_QUALITY = 95

# Depth truncation (meters); depth values beyond this distance are set to inf
DEPTH_MAX = 100.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HM3D G2G image rendering (Step 3): render 8-camera RGB + depth maps along trajectories"
    )
    parser.add_argument(
        "--step1_root",
        type=str,
        required=True,
        help="Step 1 output directory (reads trajectory.tum, scene_info.json, traj_meta.json)",
    )
    parser.add_argument(
        "--step2_root",
        type=str,
        required=True,
        help="Step 2 output directory (reads rig_config.json)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Step 3 output directory (writes images/ and depth/)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID (default: 0)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=-1,
        help="Maximum number of scenes to process, -1 means all (default: -1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of parallel workers (used for multi-GPU sharding, default: 1)",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="Current worker index (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--max_traj_per_scene",
        type=int,
        default=-1,
        help="Maximum number of trajectories to render per scene, -1 means all (default: -1)",
    )
    parser.add_argument(
        "--start_traj_idx",
        type=int,
        default=0,
        help="Starting trajectory index (0-indexed), e.g. --start_traj_idx 1 starts from traj_001 (default: 0)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip trajectories that already have rendering results",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="RGB and Depth image resolution (both width and height equal this value, depth maps are not downsampled), default: 224",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# TUM trajectory parsing
# ---------------------------------------------------------------------------
def load_trajectory_tum(
    tum_path: str,
) -> list[tuple[float, list[float], list[float]]]:
    """
    Load a trajectory in TUM format.

    Format: timestamp tx ty tz qx qy qz qw

    Args:
        tum_path: path to the TUM file

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
# Build sensors from rig_config.json
# ---------------------------------------------------------------------------
def build_sensors_from_rig_config(
    rig_config: dict,
    image_size: int = 518,
) -> list:
    """
    Build the habitat_sim sensor list from rig_config.json.

    Each camera produces one RGB sensor and one Depth sensor.
    The sensor position and orientation are set directly in habitat_sim
    as an offset relative to the agent.

    Note: the habitat_sim sensor orientation format is [pitch, yaw, roll] (radians)

    Args:
        rig_config: contents of rig_config.json
        image_size: render resolution (both width and height equal this value), default: 518

    Returns:
        list of habitat_sim sensor specs (16 of them: 8 RGB + 8 Depth)
    """
    sensors = []
    camera_height = rig_config["camera_height_m"]

    global_hfov = rig_config.get("hfov_deg", rig_config["cameras"][0].get("hfov_deg", 90.0))

    for cam in rig_config["cameras"]:
        euler = cam["actual_euler_deg"]
        # habitat_sim orientation: [pitch, yaw, roll] in radians
        orientation = np.array([
            np.radians(euler["pitch"]),
            np.radians(euler["yaw"]),
            np.radians(euler["roll"]),
        ])
        position = np.array([0.0, camera_height, 0.0])

        # per-camera hfov (step2.5); fall back to the global hfov (step2)
        cam_hfov = cam.get("hfov_deg", global_hfov)

        # RGB sensor (the passed-in image_size overrides the resolution in rig_config)
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = f"rgb_{cam['name']}"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [image_size, image_size]
        rgb_spec.hfov = cam_hfov
        rgb_spec.position = position
        rgb_spec.orientation = orientation
        sensors.append(rgb_spec)

        # Depth sensor (same resolution as RGB, no downsampling, saved directly as uint16)
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = f"depth_{cam['name']}"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [image_size, image_size]
        depth_spec.hfov = cam_hfov
        depth_spec.position = position
        depth_spec.orientation = orientation
        sensors.append(depth_spec)

    return sensors


# ---------------------------------------------------------------------------
# Create Simulator
# ---------------------------------------------------------------------------
def create_simulator(
    scene_glb_path: str,
    sensors: list,
) -> "habitat_sim.Simulator":
    """
    Create a habitat_sim.Simulator instance.

    Args:
        scene_glb_path: path to the scene GLB file
        sensors: list of sensor specs

    Returns:
        a habitat_sim.Simulator instance
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb_path
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------
def save_rgb_jpeg(
    rgb: np.ndarray, path: str, quality: int = JPEG_QUALITY
) -> None:
    """
    Save an RGB image as JPEG.

    Args:
        rgb: (H, W, 3) uint8 array
        path: output path
        quality: JPEG compression quality (0-100)
    """
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(rgb)
    img.save(path, "JPEG", quality=quality)


def save_depth_png_uint16(depth: np.ndarray, path: str) -> None:
    """
    Save a depth map as a uint16 PNG (1mm precision), without downsampling, keeping the same resolution as RGB.

    Quantization: float32 meters → uint16, where each unit = 0.001m (1mm), range 0~65.535m.
    Invalid regions (depth <= 0) are saved as 0.

    Dequantization: depth_m = depth_uint16.astype(np.float32) / 1000.0

    Args:
        depth: (H, W) or (H, W, 1) float32 array (meters)
        path: output path (should end with .png)
    """
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Drop a possible single-channel dimension
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    # float32 meters → uint16 (1mm quantization, range 0~65.535m)
    depth_quantized = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)

    # Save as a 16-bit PNG
    cv2.imwrite(path, depth_quantized)


# ---------------------------------------------------------------------------
# Scene GLB path resolution
# ---------------------------------------------------------------------------
def resolve_scene_glb_path(scene_info: dict) -> str:
    """
    Resolve the full path of the scene GLB file from scene_info.json.

    Args:
        scene_info: contents of scene_info.json

    Returns:
        full path to the GLB file
    """
    # scene_glb_path format: "hm3d_val/00859-3t8DB4Uzvkt/3t8DB4Uzvkt.basis.glb"
    rel_path = scene_info["scene_glb_path"]
    return os.path.join(HM3D_ROOT, rel_path)


# ---------------------------------------------------------------------------
# Render a single trajectory
# ---------------------------------------------------------------------------
def render_trajectory(
    sim: "habitat_sim.Simulator",
    trajectory: list[tuple[float, list[float], list[float]]],
    rig_config: dict,
    output_dir: str,
    image_size: int = 224,
) -> int:
    """
    Render RGB and depth maps for all frames along the trajectory (depth maps at the same resolution as RGB, uint16).

    Args:
        sim: Simulator instance
        trajectory: list of TUM trajectory frames
        rig_config: rig configuration
        output_dir: trajectory output directory
        image_size: RGB and Depth render resolution (default: 224)

    Returns:
        number of successfully rendered frames
    """
    rendered = 0

    for timestamp, position, rotation_xyzw in trajectory:
        # Set the agent body pose (world_T_body)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position, dtype=np.float32)
        # TUM: qx qy qz qw → quaternion(w, x, y, z)
        qx, qy, qz, qw = rotation_xyzw
        agent_state.rotation = np.quaternion(qw, qx, qy, qz)
        sim.get_agent(0).set_state(agent_state)

        # Render all sensors
        observations = sim.get_sensor_observations()

        # Timestamp string (millisecond precision, zero-padded to 10 digits)
        ts_str = f"{int(timestamp * 1000):010d}"

        for cam in rig_config["cameras"]:
            cam_name = cam["name"]
            cam_idx = cam["index"]

            # RGB: drop the alpha channel (RGBA → RGB)
            rgb_key = f"rgb_{cam_name}"
            rgb = observations[rgb_key][:, :, :3]
            rgb_path = os.path.join(
                output_dir, "images", ts_str, f"cam_{cam_idx}.jpg"
            )
            save_rgb_jpeg(rgb, rgb_path)

            # Depth: float32 → uint16 PNG (no downsampling, 1mm quantization)
            depth_key = f"depth_{cam_name}"
            depth = observations[depth_key]
            depth_path = os.path.join(
                output_dir, "depth", ts_str, f"cam_{cam_idx}.png"
            )
            save_depth_png_uint16(depth, depth_path)

        rendered += 1

    return rendered


# ---------------------------------------------------------------------------
# Check whether a trajectory has already been rendered
# ---------------------------------------------------------------------------
def is_trajectory_rendered(traj_dir: str, num_frames: int) -> bool:
    """Check whether a trajectory already has complete rendering output."""
    images_dir = os.path.join(traj_dir, "images")
    if not os.path.isdir(images_dir):
        return False
    # Simple check: number of subdirectories under images >= num_frames
    subdirs = [
        d
        for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ]
    return len(subdirs) >= num_frames


# ---------------------------------------------------------------------------
# Process a single scene
# ---------------------------------------------------------------------------
def process_scene(
    step1_scene_dir: str,
    step2_scene_dir: str,
    output_scene_dir: str,
    skip_existing: bool,
    max_traj: int,
    start_traj_idx: int = 0,
    image_size: int = 224,
) -> tuple[int, int]:
    """
    Render all trajectories of one scene (depth maps at the same resolution as RGB, uint16).

    Args:
        step1_scene_dir: Step 1 scene directory (contains scene_info.json and trajectories/traj_*/trajectory.tum)
        step2_scene_dir: Step 2 scene directory (contains trajectories/traj_*/rig_config.json)
        output_scene_dir: Step 3 output scene directory (writes images/ and depth/)
        skip_existing: whether to skip already-rendered trajectories
        max_traj: maximum number of trajectories (-1 means all)
        start_traj_idx: starting trajectory index (0-indexed, default: 0)
        image_size: RGB and Depth render resolution (default: 224)

    Returns:
        (number of rendered trajectories, total number of frames)
    """
    # Load scene_info (from step1)
    scene_info_path = os.path.join(step1_scene_dir, "scene_info.json")
    if not os.path.isfile(scene_info_path):
        print(f"  [SKIP] Missing scene_info.json: {step1_scene_dir}")
        return 0, 0

    with open(scene_info_path, "r", encoding="utf-8") as f:
        scene_info = json.load(f)

    scene_glb = resolve_scene_glb_path(scene_info)
    if not os.path.isfile(scene_glb):
        print(f"  [SKIP] GLB file does not exist: {scene_glb}")
        return 0, 0

    # Enumerate trajectory directories (from step1)
    step1_traj_parent = os.path.join(step1_scene_dir, "trajectories")
    if not os.path.isdir(step1_traj_parent):
        print(f"  [SKIP] No trajectory directory: {step1_traj_parent}")
        return 0, 0

    step2_traj_parent = os.path.join(step2_scene_dir, "trajectories")
    output_traj_parent = os.path.join(output_scene_dir, "trajectories")

    traj_names = sorted(
        d
        for d in os.listdir(step1_traj_parent)
        if os.path.isdir(os.path.join(step1_traj_parent, d))
    )

    # Apply the starting index and the maximum-trajectory-count limit
    # e.g. start_traj_idx=1, max_traj=3 → take 3 trajectories starting from traj_001
    if start_traj_idx > 0:
        traj_names = traj_names[start_traj_idx:]
    if max_traj > 0:
        traj_names = traj_names[:max_traj]

    if not traj_names:
        return 0, 0

    # Pre-read all rig_config and trajectories (check completeness)
    traj_data = []
    for traj_name in traj_names:
        step1_traj_dir = os.path.join(step1_traj_parent, traj_name)
        step2_traj_dir = os.path.join(step2_traj_parent, traj_name)
        output_traj_dir = os.path.join(output_traj_parent, traj_name)

        rig_path = os.path.join(step2_traj_dir, "rig_config.json")
        tum_path = os.path.join(step1_traj_dir, "trajectory.tum")

        if not os.path.isfile(rig_path):
            print(f"  [SKIP] {traj_name}: missing rig_config.json (run Step 2 first)")
            continue
        if not os.path.isfile(tum_path):
            print(f"  [SKIP] {traj_name}: missing trajectory.tum")
            continue

        with open(rig_path, "r", encoding="utf-8") as f:
            rig_config = json.load(f)

        trajectory = load_trajectory_tum(tum_path)
        if not trajectory:
            print(f"  [SKIP] {traj_name}: empty trajectory")
            continue

        if skip_existing and is_trajectory_rendered(output_traj_dir, len(trajectory)):
            print(f"  [SKIP] {traj_name}: already rendered ({len(trajectory)} frames)")
            continue

        traj_data.append((traj_name, output_traj_dir, rig_config, trajectory))

    if not traj_data:
        print("  No trajectories to render")
        return 0, 0

    # Create the Simulator (using the rig_config of the first trajectory)
    # Note: all trajectories share the same scene, but rig_config may differ
    # The sensors must be rebuilt for each trajectory with a different rig_config

    total_traj = 0
    total_frames = 0
    sim = None
    prev_rig_key = None  # Used to detect whether the rig configuration changed

    try:
        for traj_name, output_traj_dir, rig_config, trajectory in traj_data:
            # Check whether the Simulator needs to be rebuilt (when the rig configuration changes)
            rig_key = json.dumps(rig_config, sort_keys=True)
            if rig_key != prev_rig_key:
                if sim is not None:
                    sim.close()
                sensors = build_sensors_from_rig_config(rig_config, image_size=image_size)
                sim = create_simulator(scene_glb, sensors)
                prev_rig_key = rig_key

            num_frames = render_trajectory(
                sim, trajectory, rig_config, output_traj_dir,
                image_size=image_size
            )
            total_traj += 1
            total_frames += num_frames
            print(
                f"  {traj_name}: rendered {num_frames} frames "
                f"({rig_config['num_cameras']} cameras)"
            )
    finally:
        if sim is not None:
            sim.close()

    return total_traj, total_frames


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Lazy import
    _lazy_import_habitat(args.gpu_id)

    step1_root = args.step1_root
    step2_root = args.step2_root
    output_root = args.output_root

    step1_scenes_dir = os.path.join(step1_root, "scenes")
    if not os.path.isdir(step1_scenes_dir):
        print(f"[ERROR] Step 1 scenes directory does not exist: {step1_scenes_dir}")
        return

    step2_scenes_dir = os.path.join(step2_root, "scenes")
    if not os.path.isdir(step2_scenes_dir):
        print(f"[ERROR] Step 2 scenes directory does not exist: {step2_scenes_dir}")
        return

    output_scenes_dir = os.path.join(output_root, "scenes")
    os.makedirs(output_scenes_dir, exist_ok=True)

    print("=== HM3D G2G image rendering (Step 3 - same-resolution uint16 depth maps) ===")
    print(f"  Input directory (Step 1): {os.path.abspath(step1_root)}")
    print(f"  Input directory (Step 2): {os.path.abspath(step2_root)}")
    print(f"  Output directory (Step 3): {os.path.abspath(output_root)}")
    print(f"  GPU: {args.gpu_id}")
    print(f"  RGB resolution: {args.image_size}x{args.image_size}")
    print(f"  Depth resolution: {args.image_size}x{args.image_size} (uint16, 1mm precision)")
    print(f"  Skip already rendered: {args.skip_existing}")
    if args.start_traj_idx > 0 or args.max_traj_per_scene > 0:
        start_str = f"starting from traj_{args.start_traj_idx:03d}" if args.start_traj_idx > 0 else "starting from the beginning"
        max_str = f"at most {args.max_traj_per_scene}" if args.max_traj_per_scene > 0 else "all"
        print(f"  Trajectory range: {start_str}, {max_str}")
    print()

    # Use the scene list from step1 as the reference
    scene_names = sorted(
        d
        for d in os.listdir(step1_scenes_dir)
        if os.path.isdir(os.path.join(step1_scenes_dir, d))
    )
    if args.max_scenes > 0:
        scene_names = scene_names[: args.max_scenes]

    # Multi-GPU sharding: evenly split scenes by worker_id
    total_scenes = len(scene_names)
    if args.num_workers > 1:
        scene_names = scene_names[args.worker_id :: args.num_workers]
        print(
            f"{total_scenes} scenes in total, "
            f"Worker {args.worker_id}/{args.num_workers} handles {len(scene_names)} scenes"
        )
    else:
        print(f"{total_scenes} scenes in total")
    print()

    t_start = time.time()
    grand_total_traj = 0
    grand_total_frames = 0

    for scene_idx, scene_name in enumerate(scene_names):
        print(
            f"[{scene_idx + 1}/{len(scene_names)}] scene: {scene_name}"
        )
        step1_scene_dir = os.path.join(step1_scenes_dir, scene_name)
        step2_scene_dir = os.path.join(step2_scenes_dir, scene_name)
        output_scene_dir = os.path.join(output_scenes_dir, scene_name)

        if not os.path.isdir(step2_scene_dir):
            print(f"  [SKIP] Step 2 directory does not exist: {step2_scene_dir}")
            continue

        n_traj, n_frames = process_scene(
            step1_scene_dir,
            step2_scene_dir,
            output_scene_dir,
            args.skip_existing,
            args.max_traj_per_scene,
            args.start_traj_idx,
            image_size=args.image_size,
        )
        grand_total_traj += n_traj
        grand_total_frames += n_frames

        if n_traj > 0:
            print(
                f"  Subtotal: {n_traj} trajectories, {n_frames} frames"
            )
        print()

    elapsed = time.time() - t_start
    print("=== Done ===")
    print(f"  Total trajectories: {grand_total_traj}")
    print(f"  Total frames: {grand_total_frames}")
    print(f"  Total images: {grand_total_frames * 8} RGB + {grand_total_frames * 8} Depth")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.abspath(output_root)}")


if __name__ == "__main__":
    main()

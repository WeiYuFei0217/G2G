"""
step1_generate_trajectories.py — HM3D G2G data collection Step 1

Navigate along the shortest path in HM3D scenes using ShortestPathFollower,
with action-magnitude randomization (to simulate real execution deviations) and
height randomization (to simulate different robot body heights [read and rendered by step3]).
Outputs TUM-format body trajectories + metadata files.

Usage:
python step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --split train \
    --max_scenes -1 \
    --max_traj_per_scene 80 \
    --min_frames 10 \
    --max_frames 100 \
    --gpu_id 0

python step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --split val \
    --max_scenes -1 \
    --max_traj_per_scene 10 \
    --min_frames 10 \
    --max_frames 100 \
    --gpu_id 1
"""

from __future__ import annotations

import os
import argparse
import gzip
import json
import hashlib
import time
import signal
import threading
import ctypes

import numpy as np


# ---------------------------------------------------------------------------
# Timeout mechanism (prevents ShortestPathFollower from hanging)
# ---------------------------------------------------------------------------
EPISODE_TIMEOUT_SEC = 5


class _EpisodeTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _EpisodeTimeout()


def _call_with_timeout(func, args=(), timeout=EPISODE_TIMEOUT_SEC):
    """
    Call func(*args) in a child thread; the main thread waits timeout seconds.
    On timeout, inject an exception into the child thread to force interruption.

    Returns:
        (True, result) — normal return
        (False, None)  — timeout
    """
    result_box = [None]
    error_box = [None]

    def _worker():
        try:
            result_box[0] = func(*args)
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Inject the _EpisodeTimeout exception into the child thread
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(t.ident),
                ctypes.py_object(_EpisodeTimeout),
            )
        except Exception:
            pass
        # Wait a short while for the exception to take effect
        t.join(timeout=1.0)
        return False, None

    if error_box[0] is not None:
        raise error_box[0]

    return True, result_box[0]

# ---------------------------------------------------------------------------
# Lazy import of Habitat (GPU environment variables must be set first)
# ---------------------------------------------------------------------------
habitat = None
ShortestPathFollower = None
HabitatSimActions = None
OmegaConf = None


def _lazy_import_habitat(gpu_id: int = 0):
    """Import Habitat only after setting GPU environment variables, to avoid EGL initialization issues."""
    global habitat, ShortestPathFollower, HabitatSimActions, OmegaConf

    os.environ["HABITAT_SIM_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import habitat as _habitat
    from habitat.tasks.nav.shortest_path_follower import (
        ShortestPathFollower as _SPF,
    )
    from habitat.sims.habitat_simulator.actions import (
        HabitatSimActions as _HSA,
    )
    from omegaconf import OmegaConf as _OC

    habitat = _habitat
    ShortestPathFollower = _SPF
    HabitatSimActions = _HSA
    OmegaConf = _OC


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HM3D_ROOT = "/path/to/data/HM3D"
POINTNAV_DIR = os.path.join(HM3D_ROOT, "pointnav_hm3d")

# Habitat relies on relative paths; the script must be run from the data_gen/ directory
# data/ -> /path/to/data/HM3D/habitat_data/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Safety cap: maximum number of steps executed per episode (prevents infinite loops)
MAX_STEPS_SAFETY = 500

# Minimum distance of start/goal points from the navmesh edge (meters)
EDGE_MARGIN = 0.0  # Problematic, must not be enabled; navmesh boundary distance check is buggy

# TUM timestamp interval (seconds)
TIMESTAMP_INTERVAL = 0.5

# Action-magnitude randomization parameters
FORWARD_MEAN = 0.50       # Forward step-size mean (meters)
FORWARD_STD = 0.10        # Forward step-size standard deviation (meters)
FORWARD_CLIP_MIN = 0.20   # Forward step-size lower bound (meters)
FORWARD_CLIP_MAX = 0.80   # Forward step-size upper bound (meters)
TURN_MEAN = 22.5          # Turn-angle mean (degrees)
TURN_STD = 5.0            # Turn-angle standard deviation (degrees)
TURN_CLIP_MIN = 0.0       # Turn-angle lower bound (degrees)
TURN_CLIP_MAX = 45.0      # Turn-angle upper bound (degrees)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HM3D G2G trajectory generation (Step 1): ShortestPathFollower navigation + action-magnitude randomization"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="hm3d_g2g_data",
        help="Output root directory (default: hm3d_g2g_data)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=-1,
        help="Maximum number of scenes to process, -1 means all (default: -1)",
    )
    parser.add_argument(
        "--max_traj_per_scene",
        type=int,
        default=20,
        help="Maximum number of trajectories per scene (default: 20)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=30,
        help="Minimum number of frames per trajectory (default: 30)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames per trajectory (default: 100)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID (default: 0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------
def generate_trajectory_seed(scene_id: str, traj_id: str) -> int:
    """Generate a deterministic seed from the scene and trajectory IDs (first 4 bytes of the MD5 hash)."""
    combined = f"{scene_id}_{traj_id}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "little")


# ---------------------------------------------------------------------------
# Habitat configuration
# ---------------------------------------------------------------------------
def setup_config(split: str, scene_id_short: str):
    """
    Load the Habitat PointNav configuration with a minimal 32x32 RGB sensor, restricted to a single scene.

    Args:
        split: "train" or "val"
        scene_id_short: short scene ID, e.g. "Dd4bFSTQ8gi" (used for content_scenes filtering)

    Returns:
        Habitat DictConfig
    """
    cfg = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_hm3d.yaml"
    )

    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.data_path = (
            f"data/datasets/pointnav/hm3d/v1/{split}/{split}.json.gz"
        )
        cfg.habitat.dataset.scenes_dir = "data/scene_datasets/"
        cfg.habitat.dataset.split = split

        # Restrict to a single scene (use the short ID to match the content filename)
        cfg.habitat.dataset.content_scenes = [scene_id_short]

        sim_sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors

        # Extract the original rgb template and delete the default sensors
        base_rgb = OmegaConf.to_container(
            sim_sensors.rgb_sensor, resolve=True
        )
        del sim_sensors.rgb_sensor
        if hasattr(sim_sensors, "depth_sensor"):
            del sim_sensors.depth_sensor

        # Minimal 32x32 sensor (only used for ShortestPathFollower navigation, does not render images)
        base_rgb["width"] = 32
        base_rgb["height"] = 32
        base_rgb["hfov"] = 90
        base_rgb["uuid"] = "rgb_minimal"
        base_rgb["position"] = [0, 0.88, 0]
        base_rgb["orientation"] = [0.0, 0.0, 0.0]

        sim_sensors["rgb_minimal"] = OmegaConf.create(base_rgb)

    return cfg


# ---------------------------------------------------------------------------
# Scene list
# ---------------------------------------------------------------------------
def get_scene_list(split: str) -> list[tuple[str, str]]:
    """
    Get the scene list from the PointNav dataset.
    Each scene corresponds to one .json.gz file.

    Returns:
        Scene list, each entry being (scene_id_full, scene_id_short)
        - scene_id_full: e.g. "00824-Dd4bFSTQ8gi" (used for the GLB path and output directory)
        - scene_id_short: e.g. "Dd4bFSTQ8gi" (used for content_scenes filtering)
    """
    content_dir = os.path.join(POINTNAV_DIR, split, "content")
    if not os.path.isdir(content_dir):
        raise FileNotFoundError(
            f"PointNav content directory does not exist: {content_dir}"
        )

    scene_files = sorted(
        f for f in os.listdir(content_dir) if f.endswith(".json.gz")
    )

    scene_list = []
    for sf in scene_files:
        # Read the scene_id of the first episode from the episode file to obtain the full scene ID
        gz_path = os.path.join(content_dir, sf)
        try:
            with gzip.open(gz_path, "rt") as f:
                data = json.load(f)
            if data.get("episodes"):
                ep_scene_id = data["episodes"][0]["scene_id"]
                # ep_scene_id: "hm3d/val/00859-3t8DB4Uzvkt/3t8DB4Uzvkt.basis.glb"
                parts = ep_scene_id.split("/")
                if len(parts) >= 3:
                    scene_folder = parts[-2]  # "00859-3t8DB4Uzvkt"
                    scene_short = sf.replace(".json.gz", "")  # "3t8DB4Uzvkt"
                    scene_list.append((scene_folder, scene_short))
        except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as e:
            print(f"  [WARN] Failed to read {sf}: {e}")
            continue

    return scene_list


# ---------------------------------------------------------------------------
# Edge-distance check
# ---------------------------------------------------------------------------
def is_point_far_from_edge(
    sim,
    point,
    margin: float = EDGE_MARGIN,
    num_checks: int = 8,
) -> bool:
    """
    Check whether a point is at least margin meters from the navmesh edge.

    Prefer pathfinder.distance_to_closest_obstacle() (accurate and efficient);
    if the API is unavailable, fall back to multi-direction ray checks on the XZ plane.

    Reference: custom_pointnav/sample_navigable_points.py
    """
    # Option A: directly query the distance to the closest obstacle
    try:
        dist = sim.pathfinder.distance_to_closest_obstacle(point, max_search_radius=margin + 0.1)
        return dist >= margin
    except (AttributeError, TypeError):
        pass

    # Option B: multi-direction ray fallback
    # First snap the point to the navmesh surface to ensure the Y value is correct
    snapped = sim.pathfinder.snap_point(point)
    angles = np.linspace(0, 2 * np.pi, num_checks, endpoint=False)
    for angle in angles:
        dx = margin * np.cos(angle)
        dz = margin * np.sin(angle)
        test_point = np.array([
            snapped[0] + dx,
            snapped[1],
            snapped[2] + dz,
        ])
        if not sim.pathfinder.is_navigable(test_point):
            return False
    return True


# ---------------------------------------------------------------------------
# Action-magnitude randomization
# ---------------------------------------------------------------------------
def randomize_action_amounts(sim, rng: np.random.Generator) -> None:
    """
    Called every step: randomize the current step's forward_step_size and turn_angle.

    ShortestPathFollower already copies the step-size parameters by value into the C++ backend
    at initialization, so runtime modifications do not affect the follower's decision logic;
    only env.step()'s actual execution uses the modified step sizes.
    """
    agent_cfg = sim.get_agent(0).agent_config

    fwd = float(np.clip(rng.normal(FORWARD_MEAN, FORWARD_STD), FORWARD_CLIP_MIN, FORWARD_CLIP_MAX))
    agent_cfg.action_space[1].actuation.amount = fwd  # move_forward

    turn = float(np.clip(rng.normal(TURN_MEAN, TURN_STD), TURN_CLIP_MIN, TURN_CLIP_MAX))
    agent_cfg.action_space[2].actuation.amount = turn  # turn_left
    agent_cfg.action_space[3].actuation.amount = turn  # turn_right


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------
def extract_agent_pose(env) -> tuple:
    """
    Extract the agent's current body pose.

    Returns:
        (position, rotation_xyzw):
            position: list[float] — [x, y, z]
            rotation_xyzw: list[float] — [qx, qy, qz, qw]
    """
    agent_state = env.sim.get_agent_state()
    position = agent_state.position.tolist()
    rotation_xyzw = [
        float(agent_state.rotation.x),
        float(agent_state.rotation.y),
        float(agent_state.rotation.z),
        float(agent_state.rotation.w),
    ]
    return position, rotation_xyzw


# ---------------------------------------------------------------------------
# Path length
# ---------------------------------------------------------------------------
def compute_path_length(positions: list) -> float:
    """Compute the total trajectory length (sum of Euclidean distances between consecutive points)."""
    if len(positions) < 2:
        return 0.0
    pts = np.array(positions)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ---------------------------------------------------------------------------
# Core navigation loop
# ---------------------------------------------------------------------------
def navigate_episode(
    env,
    rng: np.random.Generator,
    min_frames: int,
    max_frames: int,
) -> tuple[str, int] | tuple[str, tuple]:
    """
    Navigate one episode along the shortest path using ShortestPathFollower.
    Randomize the action magnitude at every step.

    Returns:
        ("ok", (positions, rotations_xyzw, timestamps, path_length)) — success
        ("too_short", num_frames) — too few frames
        ("too_long", num_frames) — too many frames
        ("timeout", num_frames) — single-step timeout
    """
    episode = env.current_episode
    goal_position = episode.goals[0].position

    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )

    positions = []
    rotations_xyzw = []
    timestamps = []

    # Record the initial frame (state after reset)
    pos, rot = extract_agent_pose(env)
    positions.append(pos)
    rotations_xyzw.append(rot)
    timestamps.append(0.0)

    wall_start = time.monotonic()

    for step in range(1, MAX_STEPS_SAFETY + 1):
        # Wall-time timeout check (prevents the overall runtime from being too long)
        if time.monotonic() - wall_start > EPISODE_TIMEOUT_SEC:
            return "timeout", len(positions)

        # Compute the remaining timeout, and wrap get_next_action with a thread timeout
        remaining = EPISODE_TIMEOUT_SEC - (time.monotonic() - wall_start)
        if remaining <= 0:
            return "timeout", len(positions)

        ok, best_action = _call_with_timeout(
            follower.get_next_action, args=(goal_position,), timeout=remaining
        )
        if not ok:
            return "timeout", len(positions)

        if best_action is None or best_action == HabitatSimActions.stop:
            break

        # Randomize the action magnitude each step (does not affect the follower's decisions)
        randomize_action_amounts(env.sim, rng)

        # Execute the action
        env.step(best_action)

        # Record the current frame
        pos, rot = extract_agent_pose(env)
        positions.append(pos)
        rotations_xyzw.append(rot)
        timestamps.append(step * TIMESTAMP_INTERVAL)

    num_frames = len(positions)
    if num_frames < min_frames:
        return "too_short", num_frames
    if num_frames > max_frames:
        return "too_long", num_frames

    path_length = compute_path_length(positions)
    return "ok", (positions, rotations_xyzw, timestamps, path_length)


# ---------------------------------------------------------------------------
# TUM trajectory saving
# ---------------------------------------------------------------------------
def save_tum_trajectory(
    filepath: str,
    positions: list,
    rotations_xyzw: list,
    timestamps: list,
) -> None:
    """
    Save the trajectory as a TUM-format file.

    Format: timestamp tx ty tz qx qy qz qw
    - Quaternion order: xyzw (TUM standard)
    - Pose meaning: world_T_body
    - 9 decimal places of precision
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# HM3D G2G trajectory - body frame in world coordinates\n")
        f.write("# format: timestamp tx ty tz qx qy qz qw\n")
        for ts, pos, rot in zip(timestamps, positions, rotations_xyzw):
            line = (
                f"{ts:.9f} "
                f"{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                f"{rot[0]:.9f} {rot[1]:.9f} {rot[2]:.9f} {rot[3]:.9f}"
            )
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Trajectory metadata
# ---------------------------------------------------------------------------
def save_traj_meta(
    filepath: str,
    scene_id: str,
    traj_id: str,
    positions: list,
    rotations_xyzw: list,
    timestamps: list,
    path_length: float,
    camera_height: float,
    seed: int,
) -> None:
    """Save the trajectory metadata JSON."""
    num_frames = len(positions)
    avg_interval = path_length / max(num_frames - 1, 1)

    meta = {
        "scene_id": scene_id,
        "traj_id": traj_id,
        "num_frames": num_frames,
        "num_cameras": 8,
        "timestamps": timestamps,
        "motion_type": "nav_planned",
        "start_position": positions[0],
        "end_position": positions[-1],
        "start_rotation_quat_xyzw": rotations_xyzw[0],
        "total_path_length_m": round(path_length, 4),
        "avg_frame_interval_m": round(avg_interval, 4),
        "camera_height_m": round(camera_height, 4),
        "rig_perturbation_seed": seed,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------
def save_scene_info(
    filepath: str,
    scene_id: str,
    scene_glb_path: str,
    num_trajectories: int,
    sim,
) -> None:
    """Save the scene metadata JSON, including navigable area and scene bounds."""
    # Compute the navigable area (estimated by sampling)
    navigable_area = 0.0
    try:
        navigable_area = float(sim.pathfinder.navigable_area)
    except (AttributeError, RuntimeError):
        # Fallback: estimate by sampling
        pass

    # Compute the scene bounds
    bounds_min = [0.0, 0.0, 0.0]
    bounds_max = [0.0, 0.0, 0.0]
    try:
        scene_bb = sim.pathfinder.get_bounds()
        bounds_min = [float(x) for x in scene_bb[0]]
        bounds_max = [float(x) for x in scene_bb[1]]
    except (AttributeError, RuntimeError):
        pass

    # Relative path (for recording)
    rel_glb = scene_glb_path
    if "hm3d" in scene_glb_path:
        idx = scene_glb_path.find("hm3d")
        rel_glb = scene_glb_path[idx:]

    info = {
        "scene_id": scene_id,
        "scene_glb_path": rel_glb,
        "num_trajectories": num_trajectories,
        "navigable_area_m2": round(navigable_area, 2),
        "scene_bounds": {
            "min": [round(v, 4) for v in bounds_min],
            "max": [round(v, 4) for v in bounds_max],
        },
        "has_bev_map": False,
        "has_semantic": False,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Scene GLB path resolution
# ---------------------------------------------------------------------------
def get_scene_glb_path(scene_id: str, split: str) -> str:
    """
    Get the GLB file path from the scene ID.

    Args:
        scene_id: e.g. "00824-Dd4bFSTQ8gi"
        split: "train" or "val"

    Returns:
        Full GLB path
    """
    # scene_id: "00824-Dd4bFSTQ8gi"
    # scene filename: Dd4bFSTQ8gi.basis.glb
    parts = scene_id.split("-", 1)
    if len(parts) == 2:
        scene_file = f"{parts[1]}.basis.glb"
    else:
        scene_file = f"{scene_id}.basis.glb"

    return os.path.join(
        HM3D_ROOT, f"hm3d_{split}", scene_id, scene_file
    )


# ---------------------------------------------------------------------------
# Single-scene processing
# ---------------------------------------------------------------------------
def process_scene(
    scene_id: str,
    scene_id_short: str,
    split: str,
    output_root: str,
    max_traj: int,
    min_frames: int,
    max_frames: int,
) -> int:
    """
    Process all trajectories for a single scene.

    Create a dedicated habitat.Env for the scene, iterate over PointNav episodes,
    run ShortestPathFollower navigation + action-magnitude randomization for each episode,
    and save trajectories that meet the frame-count requirements.

    Args:
        scene_id: full scene ID, e.g. "00824-Dd4bFSTQ8gi"
        scene_id_short: short ID, e.g. "Dd4bFSTQ8gi" (used for Habitat content_scenes)

    Returns:
        Number of trajectories successfully generated
    """
    scene_dir = os.path.join(output_root, "scenes", scene_id, "trajectories")
    os.makedirs(scene_dir, exist_ok=True)

    # Check whether the GLB file exists
    scene_glb = get_scene_glb_path(scene_id, split)
    if not os.path.exists(scene_glb):
        print(f"  [SKIP] GLB file does not exist: {scene_glb}")
        return 0

    # Create the Habitat environment (restricted to a single scene, using the short ID)
    try:
        config = setup_config(split, scene_id_short)
        env = habitat.Env(config=config)
    except Exception as e:
        print(f"  [SKIP] Failed to create environment: {e}")
        return 0

    num_episodes = len(env.episodes)
    print(f"  Scene {scene_id}: {num_episodes} episodes")

    # Use scene_id as the seed to deterministically shuffle the episode order
    shuffle_seed = int(hashlib.md5(scene_id.encode()).hexdigest()[:8], 16)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    episode_order = list(range(num_episodes))
    shuffle_rng.shuffle(episode_order)
    print(f"    Shuffled episode order (seed={shuffle_seed})")

    traj_count = 0
    skipped_short = 0
    skipped_long = 0
    skipped_edge = 0
    skipped_timeout = 0
    episodes_attempted = 0

    # Register the timeout signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        for ep_idx in episode_order:
            if traj_count >= max_traj:
                break

            episodes_attempted += 1

            try:
                # Jump to the specified episode
                # Habitat env.reset() internally does _current_episode_index += 1 before loading
                # Set it to ep_idx - 1 so that reset loads exactly ep_idx
                env._current_episode_index = ep_idx - 1
                env.reset()
            except Exception as e:
                print(f"  [WARN] episode {ep_idx} reset failed: {e}")
                continue

            # Check whether the start/goal points are far enough from the navmesh edge
            episode = env.current_episode
            start_pos = env.sim.get_agent_state().position
            goal_pos = np.array(episode.goals[0].position)

            if not is_point_far_from_edge(env.sim, start_pos):
                skipped_edge += 1
                continue
            if not is_point_far_from_edge(env.sim, goal_pos):
                skipped_edge += 1
                continue

            traj_id = f"traj_{traj_count:03d}"
            seed = generate_trajectory_seed(scene_id, traj_id)
            rng = np.random.default_rng(seed)

            # Random camera height per trajectory (recorded in meta, used by Step 3 rendering)
            camera_height = float(rng.uniform(0.6, 1.5))

            # Navigation with timeout protection (prevents ShortestPathFollower from hanging)
            # signal.alarm: interrupts Python-level blocking
            # _call_with_timeout: interrupts C++-level blocking (get_next_action)
            # wall time check: interrupts cumulative multi-step timeouts
            try:
                signal.alarm(EPISODE_TIMEOUT_SEC * 2)  # Loose backstop
                result = navigate_episode(env, rng, min_frames, max_frames)
                signal.alarm(0)
            except _EpisodeTimeout:
                signal.alarm(0)
                skipped_timeout += 1
                continue

            status, payload = result
            if status == "too_short":
                skipped_short += 1
                continue
            elif status == "too_long":
                skipped_long += 1
                continue
            elif status == "timeout":
                skipped_timeout += 1
                continue

            positions, rotations_xyzw, timestamps, path_length = payload

            # Save the TUM trajectory
            traj_dir = os.path.join(scene_dir, traj_id)
            tum_path = os.path.join(traj_dir, "trajectory.tum")
            save_tum_trajectory(tum_path, positions, rotations_xyzw, timestamps)

            # Save the trajectory metadata
            meta_path = os.path.join(traj_dir, "traj_meta.json")
            save_traj_meta(
                meta_path,
                scene_id,
                traj_id,
                positions,
                rotations_xyzw,
                timestamps,
                path_length,
                camera_height,
                seed,
            )

            traj_count += 1

            if traj_count % 5 == 0:
                print(f"    Generated {traj_count}/{max_traj} trajectories")

        # Save the scene metadata
        scene_info_path = os.path.join(
            output_root, "scenes", scene_id, "scene_info.json"
        )
        save_scene_info(
            scene_info_path,
            scene_id,
            scene_glb,
            traj_count,
            env.sim,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        env.close()

    print(
        f"  Done: {traj_count} trajectories "
        f"(attempted {episodes_attempted} episodes, "
        f"skipped: {skipped_edge} near-edge + {skipped_short} too-short + "
        f"{skipped_long} too-long + {skipped_timeout} timeout)"
    )
    return traj_count


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Switch to the script directory (Habitat relies on the data/ relative path)
    os.chdir(SCRIPT_DIR)
    print(f"Working directory: {os.getcwd()}")

    # Lazily import Habitat
    _lazy_import_habitat(args.gpu_id)

    print("=== HM3D G2G trajectory generation ===")
    print(f"  Split: {args.split}")
    print(f"  Output directory: {os.path.abspath(args.output_root)}")
    print(f"  Max trajectories per scene: {args.max_traj_per_scene}")
    print(f"  Frame range: [{args.min_frames}, {args.max_frames}]")
    print(f"  GPU: {args.gpu_id}")
    print()

    # Get the scene list
    scene_list = get_scene_list(args.split)
    if args.max_scenes > 0:
        scene_list = scene_list[: args.max_scenes]

    print(f"{len(scene_list)} scenes in total")
    print()

    total_traj = 0
    t_start = time.time()

    for scene_idx, (scene_id, scene_id_short) in enumerate(scene_list):
        print(
            f"[{scene_idx + 1}/{len(scene_list)}] Processing scene: {scene_id}"
        )

        n = process_scene(
            scene_id=scene_id,
            scene_id_short=scene_id_short,
            split=args.split,
            output_root=args.output_root,
            max_traj=args.max_traj_per_scene,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        total_traj += n

    elapsed = time.time() - t_start
    print()
    print("=== Done ===")
    print(f"  Total trajectories: {total_traj}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()

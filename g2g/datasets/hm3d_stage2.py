"""
hm3d_stage2.py -- HM3D G2G pose estimation PyTorch Dataset

Loads fixed-size frame groups from pre-selected windows, used for G2G
pose estimation training. Each sample returns W frames/group (W is configurable,
default 5). Top-K windows provide natural data augmentation.

Core features:
  - Fixed window size (no padding/mask needed)
  - Top-K windows expanded into independent samples (data augmentation)
  - Coordinate frame: window's first frame is the origin (not trajectory frame 0)
  - Optional loading of per-pixel covisibility maps
  - collate is a trivial torch.stack

Usage:
    from g2g.datasets.hm3d_stage2 import HM3DStage2Dataset, hm3d_stage2_collate_fn

    ds = HM3DStage2Dataset(
        step1_root="/.../step1_generate_trajectories_train",
        step2_root="/.../step2_rig_configs_train",
        step3_root="/.../step3_render_518_518_train",
        stage2_index_root="/.../step5_stage2_index_train",
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, collate_fn=hm3d_stage2_collate_fn, num_workers=4,
    )
    for batch in loader:
        # batch["images_a"]: [B, W, 3, 518, 518]
        # batch["T_rel_gt"]: [B, 4, 4]
        ...
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

try:
    import orjson
    def _load_json(path: str):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
except ImportError:
    import json
    def _load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compact pair data type (numpy structured array, replacing Python dict)
# ---------------------------------------------------------------------------
_PAIR_DTYPE = np.dtype([
    ('scene_idx', np.uint16),
    ('traj_a_idx', np.uint32),
    ('cam_a', np.int32),
    ('traj_b_idx', np.uint32),
    ('cam_b', np.int32),
    ('pair_id', np.int32),
])


def _make_sample_dtype(window_size: int) -> np.dtype:
    """Generate the per-sample structured array type based on window_size."""
    return np.dtype([
        ('pair_idx', np.uint32),
        ('indices_a', np.int32, (window_size,)),
        ('indices_b', np.int32, (window_size,)),
        ('score', np.float32),
        ('rank', np.int32),
        ('has_covis_maps', np.bool_),
    ])


# ---------------------------------------------------------------------------
# Pose utilities (kept consistent with rig_dataset.py)
# ---------------------------------------------------------------------------
def _tum_pose_to_matrix(
    position: list[float], quat_xyzw: list[float],
) -> np.ndarray:
    """TUM pose (position + quaternion xyzw) -> 4x4 matrix."""
    qx, qy, qz, qw = quat_xyzw
    rot = Rotation.from_quat([qx, qy, qz, qw])
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = position
    return mat


def _load_trajectory_tum(
    tum_path: str,
) -> list[tuple[float, list[float], list[float]]]:
    """Load a TUM-format trajectory."""
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
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            ]
            frames.append((timestamp, position, rotation_xyzw))
    return frames


def _euler_to_body_T_cam_opencv(
    pitch_deg: float, yaw_deg: float, roll_deg: float, camera_height: float,
) -> np.ndarray:
    """
    Compute body_T_cam from euler angles (OpenCV cam2world convention).

    Exactly matches euler_to_body_T_cam() in step4_compute_overlap.py.
    Fixes two issues with the body_T_cam stored in step2 rig_config.json:
      1. euler convention (ZYX extrinsic → xyz intrinsic)
      2. missing R_x_180 (Habitat Y-up,Z-back → OpenCV Y-down,Z-forward)

    Args:
        pitch_deg, yaw_deg, roll_deg: from rig_config["cameras"][i]["actual_euler_deg"]
        camera_height: from rig_config["camera_height_m"]
    """
    R_base = Rotation.from_euler("x", 180, degrees=True)
    R_sensor = Rotation.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True)
    R_total = R_sensor * R_base
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R_total.as_matrix()
    mat[1, 3] = camera_height
    return mat


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
def hm3d_stage2_collate_fn(batch: list[dict]) -> dict:
    """
    Fixed-window-size collate (no padding needed).

    torch.stack the tensor fields; collect string/scalar fields into lists.

    Input:  list of dicts from __getitem__
    Output: batched dict
    """
    if not batch:
        return {}

    # Identify tensor fields and non-tensor fields
    tensor_keys = []
    list_keys = []

    for key, val in batch[0].items():
        if isinstance(val, torch.Tensor):
            tensor_keys.append(key)
        else:
            list_keys.append(key)

    result = {}

    for key in tensor_keys:
        result[key] = torch.stack([s[key] for s in batch])

    for key in list_keys:
        vals = [s[key] for s in batch]
        # Try to convert numeric types to tensor
        if isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HM3DStage2Dataset(torch.utils.data.Dataset):
    """
    HM3D G2G pose estimation dataset.

    Loads pre-selected windows from the index generated by step5; each sample
    contains a fixed W frames/group. Top-K windows are expanded into independent
    samples, providing natural data augmentation.

    Args:
        step1_root: Step 1 output (trajectory.tum)
        step2_root: Step 2 output (rig_config.json, body_T_cam, intrinsics)
        step3_root: Step 3 output (518x518 images, G2G resolution)
        stage2_index_root: Step 5 output index directory
        img_size: image loading size
        window_size: window size (used for validation; mismatched windows are skipped)
        top_k: number of windows taken per pair (used for expansion)
        load_covis_maps: whether to load per-pixel covisibility maps
        covis_map_size: covisibility map size
        min_overlap_score: minimum overlap score threshold
        max_scenes: maximum number of scenes (-1 for all)
        seed: random seed
        step6_root: Step 6 precomputed DINOv2 feature directory (empty string or omitted disables the cache)
    """

    def __init__(
        self,
        step1_root: str,
        step2_root: str,
        step3_root: str,
        stage2_index_root: str,
        img_size: tuple[int, int] = (518, 518),
        window_size: int = 5,
        top_k: int = 3,
        load_covis_maps: bool = False,
        covis_map_size: tuple[int, int] = (56, 56),
        min_overlap_score: float = 0.1,
        max_scenes: int = -1,
        seed: int = 42,
        step6_root: str = "",
        extrinsics_noise_cfg: dict | None = None,
        anchor_frame_idx: int = 0,
        shuffle_within_group: bool = False,
        effective_window_size: int | None = None,
        max_samples_per_scene: int = -1,
    ):
        super().__init__()

        self.step1_root = step1_root
        self.step2_root = step2_root
        self.step3_root = step3_root
        self.stage2_index_root = stage2_index_root
        self.step6_root = step6_root
        self.use_cached_features = bool(step6_root)
        self.img_size = img_size
        self.window_size = window_size
        self.top_k = top_k
        self.load_covis_maps = load_covis_maps
        self.covis_map_size = covis_map_size
        self.min_overlap_score = min_overlap_score
        self.seed = seed
        self.max_samples_per_scene = max_samples_per_scene
        self.anchor_frame_idx = anchor_frame_idx
        self.shuffle_within_group = shuffle_within_group
        # effective_window_size: take the center N frames out of window_size frames (None means use all)
        self.effective_window_size = effective_window_size
        if effective_window_size is not None:
            assert effective_window_size <= window_size, (
                f"effective_window_size({effective_window_size}) > window_size({window_size})"
            )
            logger.info(
                "effective_window_size=%d: will select center %d frames from %d-frame window",
                effective_window_size, effective_window_size, window_size,
            )

        # Extrinsics noise augmentation config
        self._ext_noise_cfg = extrinsics_noise_cfg or {}
        self._ext_noise_enabled = self._ext_noise_cfg.get("enabled", False)
        self._ext_noise_group_a = self._ext_noise_cfg.get("noise_group_a", True)
        self._ext_noise_group_b = self._ext_noise_cfg.get("noise_group_b", True)
        self._ext_noise_rot_std = self._ext_noise_cfg.get("rotation_noise_std_deg", 1.0)
        self._ext_noise_trans_std = self._ext_noise_cfg.get("translation_noise_std_m", 0.05)
        if self._ext_noise_enabled:
            logger.info(
                "Extrinsics noise augmentation ENABLED: "
                "group_a=%s, group_b=%s, rot_std=%.2f°, trans_std=%.4fm",
                self._ext_noise_group_a, self._ext_noise_group_b,
                self._ext_noise_rot_std, self._ext_noise_trans_std,
            )

        # Caches
        self._trajectory_cache: dict[str, list] = {}  # key -> TUM frames
        self._rig_cache: dict[str, dict] = {}
        self._world_T_cam_cache: dict[str, list] = {}  # key -> list of 4x4
        # Intrinsics cache: key=scene/traj/cam (supports per-camera intrinsics)
        self._intrinsics_cache: dict[str, np.ndarray] = {}
        # rig_config base render resolution cache (for intrinsics scaling), key matches _intrinsics_cache
        self._rig_resolution_cache: dict[str, tuple[int, int]] = {}  # key -> (width, height)

        # scene_id -> scene directory name mapping (for fast covis_maps directory lookup)
        self._scene_dir_map: dict[str, str] = {}

        # ------ Compact storage: replaces Python list[tuple[str,dict,dict]] ------
        # String lookup tables (shared, tiny)
        self._scene_table: list[str] = []       # scene_idx -> scene_id
        self._scene_to_idx: dict[str, int] = {}
        self._traj_table: list[str] = []        # traj_idx -> traj_name
        self._traj_to_idx: dict[str, int] = {}
        # numpy structured arrays
        self._sample_dtype = _make_sample_dtype(window_size)
        self._pair_data: np.ndarray = np.zeros(0, dtype=_PAIR_DTYPE)
        self._sample_data: np.ndarray = np.zeros(0, dtype=self._sample_dtype)
        # Index arrays (int64): point to row numbers in _sample_data
        self._active_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self._curriculum_indices: np.ndarray = np.zeros(0, dtype=np.int64)

        # Curriculum learning state
        self._curriculum_threshold: float = min_overlap_score

        # Balanced sampling state
        self._balanced_enabled: bool = False
        self._balanced_bin_width: float = 0.05
        self._balanced_ref_bin: tuple[float, float] = (0.35, 0.40)
        self._balanced_hard_boost: float = 1.15
        self._balanced_epoch: int = 0
        self._balanced_seed: int = seed
        self._balanced_scene_balanced: bool = False

        self._build_index(max_scenes)

        # Ensure trajectories/rigs of all active scenes are preloaded
        self._ensure_scenes_preloaded()

        # Initial state: full sample set
        self._curriculum_indices = np.arange(len(self._sample_data), dtype=np.int64)
        self._active_indices = self._curriculum_indices.copy()

        n_scenes = len(np.unique(
            self._pair_data['scene_idx'][self._sample_data['pair_idx'][self._active_indices]]
        )) if len(self._active_indices) > 0 else 0
        logger.info(
            "HM3DStage2Dataset: %d samples from %d scenes",
            len(self._active_indices), n_scenes,
        )

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"HM3DStage2Dataset("
            f"samples={stats['total_samples']}, "
            f"scenes={stats['num_scenes']}, "
            f"window_size={self.window_size}, "
            f"top_k={self.top_k}, "
            f"score_range=[{stats['score_min']:.3f}, {stats['score_max']:.3f}]"
            f")"
        )

    # ------------------------------------------------------------------
    # Per-scene file cache: one independent .npz per scene, supports resume/skip-existing
    # ------------------------------------------------------------------
    _CACHE_VERSION = 3

    def _cache_dir(self) -> str:
        tag = (
            f"ws{self.window_size}_topk{self.top_k}"
            f"_min{self.min_overlap_score:.2f}"
        )
        return os.path.join(self.stage2_index_root, f"_cache_{tag}")

    def _scene_cache_path(self, scene_name: str) -> str:
        return os.path.join(self._cache_dir(), f"{scene_name}.npz")

    def _try_load_scene_cache(self, scene_name: str) -> dict | None:
        """Try to load single-scene data from cache; return dict on success, None on failure."""
        path = self._scene_cache_path(scene_name)
        if not os.path.isfile(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            if int(data.get("cache_version", 0)) != self._CACHE_VERSION:
                return None
            return {
                "scene_id": str(data["scene_id"]),
                "scene_name": scene_name,
                "pair_data": data["pair_data"],
                "sample_data": data["sample_data"],
                "traj_names": data["traj_names"].tolist(),
            }
        except Exception:
            return None

    def _save_scene_cache(
        self, scene_name: str, scene_id: str,
        pair_data: np.ndarray, sample_data: np.ndarray,
        traj_names: list[str],
    ) -> None:
        """Save the numpy cache for a single scene."""
        cache_dir = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        path = self._scene_cache_path(scene_name)
        try:
            np.savez(
                path,
                cache_version=np.array(self._CACHE_VERSION),
                scene_id=np.array(scene_id),
                pair_data=pair_data,
                sample_data=sample_data,
                traj_names=np.array(traj_names, dtype=object),
            )
        except Exception as e:
            logger.warning("Scene cache save failed for %s: %s", scene_name, e)

    def _ensure_scenes_preloaded(self) -> None:
        """Ensure trajectories/rig/world_T_cam are loaded for all scenes that have samples."""
        for scene_id in self._scene_table:
            self._preload_scene(scene_id)

    def _get_or_add_scene(self, scene_id: str) -> int:
        """Get or add scene_id to the string table; return its index."""
        idx = self._scene_to_idx.get(scene_id)
        if idx is None:
            idx = len(self._scene_table)
            self._scene_table.append(scene_id)
            self._scene_to_idx[scene_id] = idx
        return idx

    def _get_or_add_traj(self, traj_name: str) -> int:
        """Get or add traj_name to the string table; return its index."""
        idx = self._traj_to_idx.get(traj_name)
        if idx is None:
            idx = len(self._traj_table)
            self._traj_table.append(traj_name)
            self._traj_to_idx[traj_name] = idx
        return idx

    def _parse_scene_json(self, scene_name: str, scenes_dir: str) -> dict | None:
        """Parse a single-scene JSON and return scene-level numpy data (without global indices)."""
        index_path = os.path.join(scenes_dir, scene_name, "stage2_index.json")
        if not os.path.isfile(index_path):
            return None

        scene_index = _load_json(index_path)
        scene_id = scene_index["scene_id"]

        traj_set: dict[str, int] = {}
        pair_rows: list[tuple] = []
        _CHUNK = 500_000
        sample_chunks: list[np.ndarray] = []
        cur_chunk = np.zeros(_CHUNK, dtype=self._sample_dtype)
        cur_pos = 0
        skipped = 0

        for pair in scene_index.get("pairs", []):
            traj_a = pair["traj_a"]
            traj_b = pair["traj_b"]
            cam_a = pair["cam_a"]
            cam_b = pair["cam_b"]
            raw_pid = pair.get("pair_id", -1)
            if isinstance(raw_pid, str):
                raw_pid = int(raw_pid.replace("pair_", ""))

            for t in (traj_a, traj_b):
                if t not in traj_set:
                    traj_set[t] = len(traj_set)

            local_pair_idx = len(pair_rows)
            pair_rows.append((0, traj_set[traj_a], cam_a, traj_set[traj_b], cam_b, raw_pid))

            for win in pair.get("windows", [])[:self.top_k]:
                if win["score"] < self.min_overlap_score:
                    continue
                if (len(win["indices_a"]) != self.window_size
                        or len(win["indices_b"]) != self.window_size):
                    skipped += 1
                    continue
                row = cur_chunk[cur_pos]
                row['pair_idx'] = local_pair_idx
                row['indices_a'] = win["indices_a"]
                row['indices_b'] = win["indices_b"]
                row['score'] = win["score"]
                row['rank'] = win.get("rank", 0)
                row['has_covis_maps'] = win.get("has_covis_maps", False)
                cur_pos += 1
                if cur_pos >= _CHUNK:
                    sample_chunks.append(cur_chunk[:cur_pos].copy())
                    cur_chunk = np.zeros(_CHUNK, dtype=self._sample_dtype)
                    cur_pos = 0

        if cur_pos > 0:
            sample_chunks.append(cur_chunk[:cur_pos].copy())

        if not sample_chunks:
            return None

        sample_data = np.concatenate(sample_chunks)
        pair_data = np.zeros(len(pair_rows), dtype=_PAIR_DTYPE)
        for i, row in enumerate(pair_rows):
            pair_data[i] = row

        traj_names = sorted(traj_set.keys(), key=lambda t: traj_set[t])
        if skipped > 0:
            logger.debug("Scene %s: skipped %d windows", scene_name, skipped)
        return {
            "scene_id": scene_id,
            "scene_name": scene_name,
            "pair_data": pair_data,
            "sample_data": sample_data,
            "traj_names": traj_names,
        }

    def _build_index(self, max_scenes: int) -> None:
        """Build a compact numpy sample index from stage2_index_root (cached per scene in separate files)."""
        scenes_dir = os.path.join(self.stage2_index_root, "scenes")
        if not os.path.isdir(scenes_dir):
            logger.warning("G2G index scenes dir not found: %s", scenes_dir)
            return

        scene_names = sorted(
            d for d in os.listdir(scenes_dir)
            if os.path.isdir(os.path.join(scenes_dir, d))
        )
        if max_scenes > 0:
            scene_names = scene_names[:max_scenes]

        all_scene_data: list[dict] = []
        t_start = time.time()

        for si, scene_name in enumerate(scene_names):
            t0 = time.time()
            cached = self._try_load_scene_cache(scene_name)
            if cached is not None:
                all_scene_data.append(cached)
                logger.info(
                    "[%d/%d] %s: %d samples (cached, %.1fs)",
                    si + 1, len(scene_names), scene_name,
                    len(cached["sample_data"]), time.time() - t0,
                )
                continue

            parsed = self._parse_scene_json(scene_name, scenes_dir)
            if parsed is None:
                continue
            self._save_scene_cache(
                scene_name, parsed["scene_id"],
                parsed["pair_data"], parsed["sample_data"],
                parsed["traj_names"],
            )
            all_scene_data.append(parsed)
            logger.info(
                "[%d/%d] %s: %d samples (parsed JSON, %.1fs)",
                si + 1, len(scene_names), scene_name,
                len(parsed["sample_data"]), time.time() - t0,
            )

        # Merge all scenes: remap local pair_idx/traj_idx to global
        pair_offset = 0
        all_pair_chunks: list[np.ndarray] = []
        all_sample_chunks: list[np.ndarray] = []

        rng = np.random.RandomState(self.seed)
        for sd in all_scene_data:
            scene_id = sd["scene_id"]
            scene_name = sd["scene_name"]
            scene_idx = self._get_or_add_scene(scene_id)
            self._scene_dir_map[scene_id] = scene_name

            traj_names = sd["traj_names"]
            local_to_global_traj = np.array(
                [self._get_or_add_traj(t) for t in traj_names], dtype=np.uint32,
            )

            p = sd["pair_data"].copy()
            p['scene_idx'] = scene_idx
            p['traj_a_idx'] = local_to_global_traj[p['traj_a_idx']]
            p['traj_b_idx'] = local_to_global_traj[p['traj_b_idx']]
            all_pair_chunks.append(p)

            s = sd["sample_data"].copy()
            if 0 < self.max_samples_per_scene < len(s):
                chosen = rng.choice(len(s), self.max_samples_per_scene, replace=False)
                chosen.sort()
                s = s[chosen]
                logger.info(
                    "Scene %s: subsampled %d -> %d samples",
                    scene_name, len(sd["sample_data"]), self.max_samples_per_scene,
                )
            s['pair_idx'] += pair_offset
            all_sample_chunks.append(s)
            pair_offset += len(p)

        del all_scene_data

        if all_pair_chunks:
            self._pair_data = np.concatenate(all_pair_chunks)
        else:
            self._pair_data = np.zeros(0, dtype=_PAIR_DTYPE)
        del all_pair_chunks

        if all_sample_chunks:
            self._sample_data = np.concatenate(all_sample_chunks)
        else:
            self._sample_data = np.zeros(0, dtype=self._sample_dtype)
        del all_sample_chunks

        logger.info(
            "Built index: %d samples, %d pairs, %d scenes in %.0fs",
            len(self._sample_data), len(self._pair_data),
            len(self._scene_table), time.time() - t_start,
        )

    def _preload_scene(self, scene_id: str) -> None:
        """Preload a scene's trajectories and rig configs."""
        step1_traj_parent = os.path.join(
            self.step1_root, "scenes", scene_id, "trajectories",
        )
        step2_traj_parent = os.path.join(
            self.step2_root, "scenes", scene_id, "trajectories",
        )

        if not os.path.isdir(step1_traj_parent):
            return

        for traj_name in sorted(os.listdir(step1_traj_parent)):
            traj_dir = os.path.join(step1_traj_parent, traj_name)
            if not os.path.isdir(traj_dir):
                continue

            tum_path = os.path.join(traj_dir, "trajectory.tum")
            rig_path = os.path.join(
                step2_traj_parent, traj_name, "rig_config.json",
            )

            if not os.path.isfile(tum_path) or not os.path.isfile(rig_path):
                continue

            traj_key = f"{scene_id}/{traj_name}"

            # Cache trajectory
            if traj_key not in self._trajectory_cache:
                self._trajectory_cache[traj_key] = _load_trajectory_tum(tum_path)

            # Cache rig config
            if traj_key not in self._rig_cache:
                self._rig_cache[traj_key] = _load_json(rig_path)

            # Precompute world_T_cam
            trajectory = self._trajectory_cache[traj_key]
            rig_config = self._rig_cache[traj_key]

            for cam in rig_config["cameras"]:
                cam_idx = cam["index"]
                cam_key = f"{scene_id}/{traj_name}/cam_{cam_idx}"

                if cam_key in self._world_T_cam_cache:
                    continue

                if rig_config.get("body_T_cam_precomputed", False):
                    body_T_cam = np.array(
                        cam["body_T_cam"], dtype=np.float64,
                    ).reshape(4, 4)
                else:
                    euler = cam["actual_euler_deg"]
                    camera_height = rig_config["camera_height_m"]
                    body_T_cam = _euler_to_body_T_cam_opencv(
                        euler["pitch"], euler["yaw"], euler["roll"],
                        camera_height,
                    )
                world_T_cams = []
                for _, position, quat_xyzw in trajectory:
                    world_T_body = _tum_pose_to_matrix(position, quat_xyzw)
                    world_T_cams.append(world_T_body @ body_T_cam)

                self._world_T_cam_cache[cam_key] = world_T_cams

                # Cache per-camera intrinsics and rig base resolution
                if cam_key not in self._intrinsics_cache:
                    self._intrinsics_cache[cam_key] = self._get_cam_intrinsics(
                        rig_config, cam,
                    )
                    self._rig_resolution_cache[cam_key] = (
                        int(rig_config.get("width", self.img_size[1])),
                        int(rig_config.get("height", self.img_size[0])),
                    )

    @staticmethod
    def _get_cam_intrinsics(rig_config: dict, cam: dict) -> np.ndarray:
        """
        Parse single-camera intrinsics, compatible with both old and new step2 rig_config formats.

        Priority:
          1) camera-level intrinsics (new format, supports per-cam differences)
          2) rig-level intrinsics/camera_intrinsics/K (old format shared intrinsics)
          3) cameras[0].intrinsics (final fallback)
        """
        if "intrinsics" in cam:
            return np.array(cam["intrinsics"], dtype=np.float64)

        for key in ("intrinsics", "camera_intrinsics", "K"):
            if key in rig_config:
                return np.array(rig_config[key], dtype=np.float64)

        cameras = rig_config.get("cameras", [])
        if cameras and "intrinsics" in cameras[0]:
            return np.array(cameras[0]["intrinsics"], dtype=np.float64)

        raise KeyError("No intrinsics found in rig_config (camera-level or rig-level)")

    # ------------------------------------------------------------------
    # Compact storage access: reconstruct sample info from numpy structured arrays
    # ------------------------------------------------------------------
    def _get_sample_tuple(self, idx: int) -> tuple:
        """
        Fetch a sample by active index, returning (scene_id, traj_a, cam_a, traj_b, cam_b,
        indices_a, indices_b, score, rank, has_covis_maps, pair_id).

        This is the sole entry point for __getitem__ and subclasses to access sample data.
        """
        real_idx = int(self._active_indices[idx])
        s = self._sample_data[real_idx]
        p = self._pair_data[int(s['pair_idx'])]
        scene_id = self._scene_table[int(p['scene_idx'])]
        traj_a = self._traj_table[int(p['traj_a_idx'])]
        cam_a = int(p['cam_a'])
        traj_b = self._traj_table[int(p['traj_b_idx'])]
        cam_b = int(p['cam_b'])
        indices_a = s['indices_a'].tolist()
        indices_b = s['indices_b'].tolist()
        score = float(s['score'])
        rank = int(s['rank'])
        has_covis = bool(s['has_covis_maps'])
        pair_id = int(p['pair_id'])
        return (scene_id, traj_a, cam_a, traj_b, cam_b,
                indices_a, indices_b, score, rank, has_covis, pair_id)

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get one training sample.

        Returns:
            dict:
                images_a: [W, 3, H, W] float32 [0, 1]
                images_b: [W, 3, H, W] float32 [0, 1]
                extrinsics_a: [W, 4, 4] float32 (window's first frame = identity)
                extrinsics_b: [W, 4, 4] float32 (window's first frame = identity)
                intrinsics_a: [3, 3] float32
                intrinsics_b: [3, 3] float32
                T_rel_gt: [4, 4] float32 (A window first frame <- B window first frame)
                window_overlap_score: float
                covis_maps_ab: [W, W, H_c, W_c] float32 (optional)
                covis_maps_ba: [W, W, H_c, W_c] float32 (optional)
                scene_id: str
                traj_a_id: str
                cam_a_id: int
                traj_b_id: str
                cam_b_id: int
                window_rank: int
        """
        (scene_id, traj_a, cam_a, traj_b, cam_b,
         indices_a, indices_b, score, rank, has_covis, pair_id) = \
            self._get_sample_tuple(idx)

        # Reorder frames: move the anchor frame to index 0; all downstream code stays the same
        # e.g. anchor_frame_idx=2, window_size=5: [0,1,2,3,4] → [2,0,1,3,4]
        if self.anchor_frame_idx != 0:
            a = self.anchor_frame_idx
            reorder = [a] + [i for i in range(len(indices_a)) if i != a]
            indices_a = [indices_a[i] for i in reorder]
            indices_b = [indices_b[i] for i in reorder]

        # Select the center effective_window_size frames out of window_size frames
        # e.g. window_size=5, effective_window_size=3: [0,1,2,3,4] → [1,2,3]
        if self.effective_window_size is not None and self.effective_window_size < len(indices_a):
            W_full = len(indices_a)
            W_eff = self.effective_window_size
            start = (W_full - W_eff) // 2
            indices_a = indices_a[start:start + W_eff]
            indices_b = indices_b[start:start + W_eff]

        # Load window frame images (or cached features)
        if self.use_cached_features:
            enc_feats_a = self._load_window_features(scene_id, traj_a, cam_a, indices_a)
            enc_feats_b = self._load_window_features(scene_id, traj_b, cam_b, indices_b)
            # dummy images: _build_views only reads the shape, not the content
            H, W_img = self.img_size
            images_a = np.zeros((len(indices_a), 3, H, W_img), dtype=np.float32)
            images_b = np.zeros((len(indices_b), 3, H, W_img), dtype=np.float32)
        else:
            images_a = self._load_window_images(scene_id, traj_a, cam_a, indices_a)
            images_b = self._load_window_images(scene_id, traj_b, cam_b, indices_b)

        # Get world_T_cam poses
        cam_key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        cam_key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"
        world_T_cams_a = self._world_T_cam_cache[cam_key_a]
        world_T_cams_b = self._world_T_cam_cache[cam_key_b]

        # world_T_cam of the window's first frame (after reordering, index 0 is the anchor frame)
        world_T_cam_a0 = world_T_cams_a[indices_a[0]]
        world_T_cam_b0 = world_T_cams_b[indices_b[0]]

        # Intra-group relative extrinsics: window's first frame is the origin
        inv_a0 = np.linalg.inv(world_T_cam_a0)
        extrinsics_a = np.stack([
            inv_a0 @ world_T_cams_a[i] for i in indices_a
        ])

        inv_b0 = np.linalg.inv(world_T_cam_b0)
        extrinsics_b = np.stack([
            inv_b0 @ world_T_cams_b[j] for j in indices_b
        ])

        # Extrinsics noise augmentation: apply optional SE(3) perturbation to input extrinsics (T_rel_gt unchanged)
        if self._ext_noise_enabled:
            rng = np.random.default_rng()
            extrinsics_a = self._apply_extrinsics_noise(
                extrinsics_a, self._ext_noise_group_a, rng,
            )
            extrinsics_b = self._apply_extrinsics_noise(
                extrinsics_b, self._ext_noise_group_b, rng,
            )

        # Inter-group GT: T_rel = inv(world_T_cam_a[window_a[0]]) @ world_T_cam_b[window_b[0]]
        T_rel_gt = inv_a0 @ world_T_cam_b0

        # Intrinsics (cached at the rig_config base resolution, scaled to img_size)
        key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"
        intrinsics_a = self._intrinsics_cache[key_a].copy()
        intrinsics_b = self._intrinsics_cache[key_b].copy()
        # If img_size differs from the rig_config base resolution, scale fx, fy, cx, cy proportionally
        rig_w_a, rig_h_a = self._rig_resolution_cache[key_a]
        rig_w_b, rig_h_b = self._rig_resolution_cache[key_b]
        for K, rig_w, rig_h in (
            (intrinsics_a, rig_w_a, rig_h_a),
            (intrinsics_b, rig_w_b, rig_h_b),
        ):
            scale_x = self.img_size[1] / float(rig_w)
            scale_y = self.img_size[0] / float(rig_h)
            if scale_x != 1.0 or scale_y != 1.0:
                K[0, 0] *= scale_x   # fx
                K[0, 2] *= scale_x   # cx
                K[1, 1] *= scale_y   # fy
                K[1, 2] *= scale_y   # cy

        sample = {
            "images_a": torch.from_numpy(images_a).float(),
            "images_b": torch.from_numpy(images_b).float(),
            "extrinsics_a": torch.from_numpy(extrinsics_a).float(),
            "extrinsics_b": torch.from_numpy(extrinsics_b).float(),
            "intrinsics_a": torch.from_numpy(intrinsics_a).float(),
            "intrinsics_b": torch.from_numpy(intrinsics_b).float(),
            "T_rel_gt": torch.from_numpy(T_rel_gt).float(),
            "window_overlap_score": score,
            "scene_id": scene_id,
            "traj_a_id": traj_a,
            "cam_a_id": cam_a,
            "traj_b_id": traj_b,
            "cam_b_id": cam_b,
            "window_rank": rank,
        }

        # Optional: cached DINOv2 encoder features
        if self.use_cached_features:
            sample["enc_feats_a"] = enc_feats_a  # [W, C, Hp, Wp] float32
            sample["enc_feats_b"] = enc_feats_b  # [W, C, Hp, Wp] float32

        # Optional: load per-pixel covisibility maps
        if self.load_covis_maps and has_covis:
            covis_ab, covis_ba = self._load_covis_maps(
                scene_id, pair_id, indices_a, indices_b,
            )
            sample["covis_maps_ab"] = torch.from_numpy(covis_ab).float()
            sample["covis_maps_ba"] = torch.from_numpy(covis_ba).float()

        return sample

    def _load_window_images(
        self,
        scene_id: str,
        traj_name: str,
        cam_idx: int,
        frame_indices: list[int],
    ) -> np.ndarray:
        """
        Load window frame images.

        Note: output is normalized to [0, 1]. If the G2G model needs ImageNet
        normalization, it should be handled in the DataLoader transform or the model forward.

        Returns:
            [W, 3, H, W_img] float32 array, value range [0, 1]
        """
        traj_key = f"{scene_id}/{traj_name}"
        trajectory = self._trajectory_cache[traj_key]

        images = []
        for frame_idx in frame_indices:
            ts, _, _ = trajectory[frame_idx]
            ts_str = f"{int(round(ts * 1000)):010d}"
            img_path = os.path.join(
                self.step3_root, "scenes", scene_id, "trajectories",
                traj_name, "images", ts_str, f"cam_{cam_idx}.jpg",
            )

            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                # Try png
                img_path_png = img_path.replace(".jpg", ".png")
                try:
                    img = Image.open(img_path_png).convert("RGB")
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Image not found: {img_path}\n"
                        f"Scene: {scene_id}, Traj: {traj_name}, "
                        f"Cam: {cam_idx}, Frame: {frame_idx}"
                    ) from e

            H, W_img = self.img_size
            if img.size != (W_img, H):
                img = img.resize((W_img, H), Image.Resampling.BILINEAR)

            img_arr = np.array(img, dtype=np.float32) / 255.0
            images.append(img_arr.transpose(2, 0, 1))  # HWC -> CHW

        return np.stack(images, axis=0)

    def _load_window_features(
        self,
        scene_id: str,
        traj_name: str,
        cam_idx: int,
        frame_indices: list[int],
    ) -> torch.Tensor:
        """
        Load precomputed DINOv2 features from the step6 cache.

        Path construction is identical to _load_window_images
        (step6_root replaces step3_root, features/ replaces images/, .npy replaces .jpg).

        Returns:
            [W, C, Hp, Wp] float32 tensor
        """
        traj_key = f"{scene_id}/{traj_name}"
        trajectory = self._trajectory_cache[traj_key]

        feats = []
        for frame_idx in frame_indices:
            ts, _, _ = trajectory[frame_idx]
            ts_str = f"{int(round(ts * 1000)):010d}"
            npy_path = os.path.join(
                self.step6_root, "scenes", scene_id, "trajectories",
                traj_name, "features", ts_str, f"cam_{cam_idx}.npy",
            )
            # Load uint16 raw bits -> bf16 -> float32
            try:
                raw = np.load(npy_path)  # [C, Hp, Wp] uint16
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Cached DINOv2 feature not found: {npy_path}\n"
                    f"Scene: {scene_id}, Traj: {traj_name}, "
                    f"Cam: {cam_idx}, Frame: {frame_idx}"
                ) from e
            feat = torch.from_numpy(raw).view(torch.bfloat16).float()
            feats.append(feat)

        return torch.stack(feats)  # [W, C, Hp, Wp]

    def _load_covis_maps(
        self,
        scene_id: str,
        pair_id: str | int,
        indices_a: list[int],
        indices_b: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load per-pixel covisibility maps.

        Use _scene_dir_map for fast scene directory lookup (built in _build_index).

        Returns:
            covis_ab: [W_a, W_b, H_c, W_c] float32
            covis_ba: [W_a, W_b, H_c, W_c] float32
        """
        W_a = len(indices_a)
        W_b = len(indices_b)
        H_c, W_c = self.covis_map_size

        covis_ab = np.zeros((W_a, W_b, H_c, W_c), dtype=np.float32)
        covis_ba = np.zeros((W_a, W_b, H_c, W_c), dtype=np.float32)

        # Look up the scene directory from the prebuilt mapping
        scene_dir_name = self._scene_dir_map.get(scene_id)
        if scene_dir_name is None:
            return covis_ab, covis_ba

        covis_dir = os.path.join(
            self.stage2_index_root, "scenes", scene_dir_name, "covis_maps",
        )
        if not os.path.isdir(covis_dir):
            return covis_ab, covis_ba

        for wi, fa in enumerate(indices_a):
            for wj, fb in enumerate(indices_b):
                # pair_id may be a string (e.g. "pair_000000") or an integer
                if isinstance(pair_id, int):
                    pair_prefix = f"pair_{pair_id:06d}"
                else:
                    pair_prefix = str(pair_id)
                covis_path = os.path.join(
                    covis_dir,
                    f"{pair_prefix}_fa{fa:04d}_fb{fb:04d}.npz",
                )
                if not os.path.isfile(covis_path):
                    continue

                data = np.load(covis_path)
                ca = data["covis_a"].astype(np.float32)
                cb = data["covis_b"].astype(np.float32)

                # Resize (if mismatched): use torch float32 interpolation to avoid precision loss
                if ca.shape != (H_c, W_c):
                    ca_t = torch.from_numpy(ca).unsqueeze(0).unsqueeze(0)
                    ca = F.interpolate(
                        ca_t, size=(H_c, W_c), mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0).numpy()
                    cb_t = torch.from_numpy(cb).unsqueeze(0).unsqueeze(0)
                    cb = F.interpolate(
                        cb_t, size=(H_c, W_c), mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0).numpy()

                covis_ab[wi, wj] = ca
                covis_ba[wi, wj] = cb

        return covis_ab, covis_ba

    # ------------------------------------------------------------------
    # Extrinsics Noise Augmentation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_se3_noise(
        num_frames: int,
        rot_std_deg: float,
        trans_std_m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate SE(3) noise matrices.

        Sample independently for each frame:
          - Rotation: axis-angle perturbation about a random axis, magnitude ~ N(0, rot_std_deg)
          - Translation: each dimension independently ~ N(0, trans_std_m)

        The first frame (index 0) is always Identity (the window's first frame is the coordinate frame origin).

        Returns:
            [num_frames, 4, 4] noise matrices
        """
        noise = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
        if rot_std_deg <= 0 and trans_std_m <= 0:
            return noise

        for i in range(1, num_frames):
            if rot_std_deg > 0:
                axis = rng.standard_normal(3)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis /= axis_norm
                angle_deg = rng.normal(0.0, rot_std_deg)
                noise[i, :3, :3] = Rotation.from_rotvec(
                    axis * np.deg2rad(angle_deg),
                ).as_matrix()

            if trans_std_m > 0:
                noise[i, :3, 3] = rng.normal(0.0, trans_std_m, size=3)

        return noise

    def _apply_extrinsics_noise(
        self,
        extrinsics: np.ndarray,
        apply: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Optionally apply SE(3) noise to the extrinsics matrices.

        Noise is applied by right multiplication: extrinsics_noisy[i] = extrinsics[i] @ noise[i]
        This is equivalent to applying a small perturbation in the camera coordinate frame.

        Args:
            extrinsics: [W, 4, 4] original extrinsics (first frame = identity)
            apply: whether to actually apply the noise
            rng: numpy random number generator

        Returns:
            [W, 4, 4] (possibly noisy) extrinsics
        """
        if not apply or not self._ext_noise_enabled:
            return extrinsics

        noise = self._make_se3_noise(
            len(extrinsics),
            self._ext_noise_rot_std,
            self._ext_noise_trans_std,
            rng,
        )
        return extrinsics @ noise

    # ------------------------------------------------------------------
    # Curriculum Learning
    # ------------------------------------------------------------------
    def set_curriculum_overlap(self, threshold: float) -> int:
        """
        Update the curriculum learning overlap threshold and filter active samples.

        Keeps only samples with score >= threshold. When threshold <= min_overlap_score
        this is equivalent to using the full sample set. When balanced sampling is enabled,
        bin balancing is applied automatically after filtering.

        Args:
            threshold: overlap score threshold

        Returns:
            number of active samples
        """
        self._curriculum_threshold = threshold
        mask = self._sample_data['score'] >= threshold
        self._curriculum_indices = np.where(mask)[0].astype(np.int64)
        if self._balanced_enabled:
            self._apply_balanced_sampling(self._balanced_epoch)
        else:
            self._active_indices = self._curriculum_indices.copy()
        return len(self._active_indices)

    def get_full_size(self) -> int:
        """Return the full sample count (unaffected by curriculum learning filtering)."""
        return len(self._sample_data)

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        if len(self._active_indices) == 0:
            return {
                "total_samples": 0, "full_samples": len(self._sample_data),
                "num_scenes": 0, "curriculum_threshold": self._curriculum_threshold,
                "score_mean": 0.0, "score_median": 0.0,
                "score_min": 0.0, "score_max": 0.0,
            }
        active_scores = self._sample_data['score'][self._active_indices]
        active_pair_idx = self._sample_data['pair_idx'][self._active_indices]
        n_scenes = len(np.unique(self._pair_data['scene_idx'][active_pair_idx]))
        return {
            "total_samples": len(self._active_indices),
            "full_samples": len(self._sample_data),
            "num_scenes": int(n_scenes),
            "curriculum_threshold": self._curriculum_threshold,
            "score_mean": float(active_scores.mean()),
            "score_median": float(np.median(active_scores)),
            "score_min": float(active_scores.min()),
            "score_max": float(active_scores.max()),
        }

    # ------------------------------------------------------------------
    # Balanced Sampling
    # ------------------------------------------------------------------
    def enable_balanced_sampling(
        self,
        bin_width: float = 0.05,
        ref_bin: tuple[float, float] = (0.35, 0.40),
        hard_boost: float = 1.15,
        seed: int = 42,
        scene_balanced: bool = False,
    ) -> None:
        """
        Enable balanced sampling. After every set_curriculum_overlap or set_balanced_epoch,
        samples are binned by overlap score and over-full bins are downsampled, ensuring a
        balanced training amount across difficulty ranges.

        Args:
            bin_width: bin width (e.g. 0.05 → [0.10,0.15), [0.15,0.20), ...)
            ref_bin: [lo, hi) of the reference bin; this bin's sample count is the balancing baseline
            hard_boost: amplification factor for the sample count of hard bins (center < ref_center)
            seed: random seed for deterministic rotation
            scene_balanced: when downsampling, allocate quotas evenly across scenes to ensure scene diversity
        """
        self._balanced_enabled = True
        self._balanced_bin_width = bin_width
        self._balanced_ref_bin = ref_bin
        self._balanced_hard_boost = hard_boost
        self._balanced_seed = seed
        self._balanced_scene_balanced = scene_balanced
        logger.info(
            "Balanced sampling ENABLED: bin_width=%.3f, ref_bin=[%.2f,%.2f), "
            "hard_boost=%.2f, seed=%d, scene_balanced=%s",
            bin_width, ref_bin[0], ref_bin[1], hard_boost, seed, scene_balanced,
        )
        # If curriculum-filtered data already exists, apply immediately
        if len(self._curriculum_indices) > 0:
            self._apply_balanced_sampling(self._balanced_epoch)

    def set_balanced_epoch(self, epoch: int) -> int:
        """
        Set the current epoch, triggering deterministic rotation of over-full bins.
        Called at the start of each epoch.

        Args:
            epoch: current epoch number

        Returns:
            number of active samples after balancing
        """
        self._balanced_epoch = epoch
        if self._balanced_enabled and len(self._curriculum_indices) > 0:
            self._apply_balanced_sampling(epoch)
        return len(self._active_indices)

    def _apply_balanced_sampling(self, epoch: int) -> None:
        """Balanced sampling core: numpy-vectorized bin grouping + downsampling + cross-epoch deterministic rotation."""
        scores = self._sample_data['score'][self._curriculum_indices]
        bin_indices = (scores / self._balanced_bin_width).astype(np.int32)

        ref_lo, ref_hi = self._balanced_ref_bin
        ref_bin_idx = round(ref_lo / self._balanced_bin_width)
        ref_count = int(np.sum(bin_indices == ref_bin_idx))
        if ref_count == 0:
            self._active_indices = self._curriculum_indices.copy()
            logger.warning(
                "Balanced sampling: ref bin [%.2f,%.2f) has 0 samples, skipping",
                ref_lo, ref_hi,
            )
            return

        ref_bin_center = (ref_lo + ref_hi) / 2

        # Scene-aware mode: precompute the scene_idx of each sample
        if self._balanced_scene_balanced:
            sample_pair_idx = self._sample_data['pair_idx'][self._curriculum_indices]
            sample_scene_idx = self._pair_data['scene_idx'][sample_pair_idx]
            total_scenes = len(np.unique(self._pair_data['scene_idx']))

        balanced_parts: list[np.ndarray] = []
        for bi in np.unique(bin_indices):
            bi = int(bi)
            bin_mask = bin_indices == bi
            bin_sample_indices = self._curriculum_indices[bin_mask]

            bin_center = (bi + 0.5) * self._balanced_bin_width
            if bin_center < ref_bin_center:
                cap = int(ref_count * self._balanced_hard_boost)
            else:
                cap = ref_count

            if len(bin_sample_indices) <= cap:
                balanced_parts.append(bin_sample_indices)
            elif not self._balanced_scene_balanced:
                # Global random downsampling + cross-epoch deterministic rotation
                rng = np.random.RandomState(self._balanced_seed + bi)
                perm = np.arange(len(bin_sample_indices))
                rng.shuffle(perm)
                offset = (epoch * cap) % len(bin_sample_indices)
                sel = np.array(
                    [perm[(offset + i) % len(perm)] for i in range(cap)],
                    dtype=np.int64,
                )
                balanced_parts.append(bin_sample_indices[sel])
            else:
                # Scene-aware downsampling: iterative allocation (water-filling)
                bin_scene_idx = sample_scene_idx[bin_mask]
                unique_scenes = np.unique(bin_scene_idx)

                scene_groups: dict[int, np.ndarray] = {}
                for sid in unique_scenes:
                    scene_mask = bin_scene_idx == sid
                    scene_groups[int(sid)] = bin_sample_indices[scene_mask]

                # Iterative allocation: split the budget evenly each round; small scenes exit once saturated, the remainder is reallocated
                scene_quotas: dict[int, int] = {}
                remaining = cap
                pending = set(scene_groups.keys())
                while pending and remaining > 0:
                    per_scene = remaining / len(pending)
                    settled = []
                    for sid in list(pending):
                        if len(scene_groups[sid]) <= per_scene:
                            scene_quotas[sid] = len(scene_groups[sid])
                            remaining -= len(scene_groups[sid])
                            settled.append(sid)
                    if not settled:
                        # All pending scenes are > per_scene, split evenly
                        quota_each = remaining // len(pending)
                        leftover = remaining - quota_each * len(pending)
                        for i, sid in enumerate(sorted(pending)):
                            scene_quotas[sid] = quota_each + (1 if i < leftover else 0)
                        pending.clear()
                    else:
                        for sid in settled:
                            pending.discard(sid)

                for sid, indices in scene_groups.items():
                    q = scene_quotas.get(sid, 0)
                    if q >= len(indices):
                        balanced_parts.append(indices)
                    else:
                        rng = np.random.RandomState(
                            self._balanced_seed + bi * 10000 + sid
                        )
                        perm = np.arange(len(indices))
                        rng.shuffle(perm)
                        offset = (epoch * q) % len(indices)
                        sel = np.array(
                            [perm[(offset + i) % len(perm)] for i in range(q)],
                            dtype=np.int64,
                        )
                        balanced_parts.append(indices[sel])

        self._active_indices = np.concatenate(balanced_parts) if balanced_parts else \
            np.zeros(0, dtype=np.int64)
        logger.info(
            "Balanced sampling (epoch %d): %d → %d samples (ref_count=%d%s)",
            epoch, len(self._curriculum_indices), len(self._active_indices), ref_count,
            ", scene_balanced" if self._balanced_scene_balanced else "",
        )

    def get_bin_distribution(self) -> dict[str, int]:
        """
        Return the bin distribution of the current active samples.
        key format: "[lo, hi)", value: number of samples.
        """
        if len(self._active_indices) == 0:
            return {}
        active_scores = self._sample_data['score'][self._active_indices]
        bin_indices = (active_scores / self._balanced_bin_width).astype(np.int32)
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        result = {}
        for bi, cnt in zip(unique_bins, counts):
            lo = int(bi) * self._balanced_bin_width
            hi = lo + self._balanced_bin_width
            result[f"[{lo:.2f},{hi:.2f})"] = int(cnt)
        return result

    def get_balanced_state(self) -> dict:
        """Serialize the balanced sampling state for checkpointing."""
        return {
            "enabled": self._balanced_enabled,
            "bin_width": self._balanced_bin_width,
            "ref_bin": self._balanced_ref_bin,
            "hard_boost": self._balanced_hard_boost,
            "epoch": self._balanced_epoch,
            "seed": self._balanced_seed,
            "scene_balanced": self._balanced_scene_balanced,
        }

    def restore_balanced_state(self, state: dict) -> None:
        """Restore the balanced sampling state from a checkpoint."""
        self._balanced_enabled = state.get("enabled", False)
        self._balanced_bin_width = state.get("bin_width", 0.05)
        self._balanced_ref_bin = tuple(state.get("ref_bin", (0.35, 0.40)))
        self._balanced_hard_boost = state.get("hard_boost", 1.15)
        self._balanced_epoch = state.get("epoch", 0)
        self._balanced_seed = state.get("seed", 42)
        self._balanced_scene_balanced = state.get("scene_balanced", False)
        if self._balanced_enabled and len(self._curriculum_indices) > 0:
            self._apply_balanced_sampling(self._balanced_epoch)

"""
rig_dataset.py — Rig multi-camera dataset (G2G MultiFrame compatible format)

Loads rig multi-camera data from the HM3D 7-step preprocessing pipeline and outputs
a dict format compatible with the Stage2ModelMultiFrame model.

Each sample = 2 rigs (Rig_A @ t_a, Rig_B @ t_b), each randomly selecting K cameras:
  - Uses DINOv2 features precomputed by step6 (enc_feats)
  - Reads precomputed body_T_cam and intrinsics from rig_config.json
  - Reads body poses from trajectory.tum

Output fields:
  images_a/b:          [K, 3, 224, 224]  dummy (uses cached features)
  enc_feats_a/b:       [K, 1024, 16, 16] DINOv2 features
  extrinsics_a/b:      [K, 4, 4]         intra-rig relative extrinsics (first frame=I, may contain noise)
  extrinsics_a/b_clean: [K, 4, 4]        clean extrinsics (noise-free, used for GT computation)
  intrinsics_a/b:      [K, 3, 3]         per-camera intrinsics
  T_rel_gt:            [4, 4]            cross-rig GT relative pose
  scene_id:            str
  window_overlap_score: float            body_dist

Usage:
    ds = RigDatasetG2G(
        step1_root="/.../step1_..._train",
        step2_root="/.../step2_rig8_configs_train",
        step3_root="/.../step3_render_rig8_224_train",
        step6_root="/.../step6_dinov2_features_rig8_train",
        index_root="/.../step7_rig_index_train",
        num_cameras=3,
    )
    sample = ds[0]
    # sample["enc_feats_a"].shape == (3, 1024, 16, 16)
    # sample["T_rel_gt"].shape == (4, 4)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _load_json(path: str):
    """Load a JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tum_pose_to_matrix(position: list, quat_xyzw: list) -> np.ndarray:
    """TUM format (position, quaternion xyzw) → 4x4 homogeneous transform matrix"""
    rot = Rotation.from_quat(quat_xyzw)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = position
    return mat


def _load_tum(path: str) -> dict[str, np.ndarray]:
    """Load a TUM trajectory file → {ts_key: T_world_body (4,4)}

    TUM line format: timestamp tx ty tz qx qy qz qw
    Timestamps are in seconds, converted to a 10-digit millisecond string as the key
    """
    poses = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = parts[0]
            # seconds → milliseconds, zero-padded to 10 digits
            ts_key = f"{int(round(float(ts) * 1000)):010d}"
            pos = [float(x) for x in parts[1:4]]
            quat = [float(x) for x in parts[4:8]]
            poses[ts_key] = _tum_pose_to_matrix(pos, quat)
    return poses


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RigDatasetG2G(torch.utils.data.Dataset):
    """Rig multi-camera dataset, outputs G2G MultiFrame compatible format

    Loads from the HM3D preprocessing pipeline:
      - step1: body poses (TUM format)
      - step2: rig camera configuration (body_T_cam, intrinsics)
      - step6: DINOv2 features (uint16-quantized bfloat16)
      - step7: rig pair index (traj_a, t_a, traj_b, t_b, body_dist)

    Each __getitem__ randomly selects K cameras (both rigs use the same selection),
    providing a built-in data augmentation effect.
    """

    def __init__(
        self,
        step1_root: str,
        step2_root: str,
        step3_root: str,          # kept for interface consistency, not actually used
        step6_root: str,
        index_root: str,
        num_cameras: int = 3,     # K: number of cameras selected per rig
        total_cameras: int = 8,   # total number of cameras
        image_size: int = 224,
        patch_size: int = 14,
        max_scenes: int = -1,     # -1 means no limit
        max_pairs_per_scene: int = -1,  # -1 means no limit, >0 randomly subsamples per scene
        extrinsics_noise_cfg: dict | None = None,
        camera_select_seed: int | None = None,
    ):
        super().__init__()
        self.step1_root = step1_root
        self.step2_root = step2_root
        self.step3_root = step3_root
        self.step6_root = step6_root
        # When step6_root is empty, ship RGB instead of cached DINOv2 features:
        # __getitem__ loads images from step3_root and returns no enc_feats, so
        # Stage2ModelMultiFrame recomputes features with the frozen backbone on the
        # fly (mirrors the reloc RGB bundles). Keeps the shipped sample bundle small.
        self.use_rgb = not bool(step6_root)
        self.num_cameras = num_cameras
        self.total_cameras = total_cameras
        # When set, the per-rig camera subset is chosen deterministically from a hash
        # of the pair identity, so eval is reproducible and the same pair always gets
        # the same cameras regardless of its position in the index. Default None keeps
        # the random per-sample selection used as training-time augmentation.
        self.camera_select_seed = camera_select_seed
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # DINOv2 feature spatial resolution

        # extrinsics noise configuration
        self._ext_noise_cfg = extrinsics_noise_cfg or {}
        self._ext_noise_enabled = self._ext_noise_cfg.get("enabled", False)
        self._ext_noise_rot_std = self._ext_noise_cfg.get("rotation_noise_std_deg", 1.0)
        self._ext_noise_trans_std = self._ext_noise_cfg.get("translation_noise_std_m", 0.05)

        t0 = time.time()

        # feature-miss statistics (used for runtime warnings)
        self._feat_miss_count = 0
        self._feat_load_count = 0
        self._feat_miss_warned = False

        # cache: loaded at init time, looked up directly in __getitem__
        self.pairs: list[dict] = []
        self.rig_configs: dict[str, dict] = {}          # scene_id → rig_config
        self.body_poses: dict[tuple, dict] = {}          # (scene_id, traj_id) → {ts_key: T_world_body}

        scenes_dir = os.path.join(index_root, "scenes")
        if not os.path.isdir(scenes_dir):
            logger.warning(f"Index directory does not exist: {scenes_dir}")
            return

        scene_ids = sorted(os.listdir(scenes_dir))
        if max_scenes > 0:
            scene_ids = scene_ids[:max_scenes]

        rng = np.random.default_rng(42)

        for scene_id in scene_ids:
            # load the pair index
            index_path = os.path.join(scenes_dir, scene_id, "rig_pairs.json")
            if not os.path.isfile(index_path):
                continue
            pairs_data = _load_json(index_path)

            # load and cache rig_config
            rig_path = os.path.join(step2_root, "scenes", scene_id, "rig_config.json")
            if not os.path.isfile(rig_path):
                logger.warning(f"rig_config missing: {rig_path}")
                continue
            self.rig_configs[scene_id] = _load_json(rig_path)

            # random subsampling per scene
            scene_pairs = pairs_data["pairs"]
            if max_pairs_per_scene > 0 and len(scene_pairs) > max_pairs_per_scene:
                idx = rng.choice(len(scene_pairs), max_pairs_per_scene, replace=False)
                scene_pairs = [scene_pairs[i] for i in sorted(idx)]

            # expand all pairs
            for p in scene_pairs:
                self.pairs.append({
                    "scene_id": scene_id,
                    **p,
                })
                # preload the required body poses
                for traj_key in [p["traj_a"], p["traj_b"]]:
                    cache_key = (scene_id, traj_key)
                    if cache_key not in self.body_poses:
                        tum_path = os.path.join(
                            step1_root, "scenes", scene_id,
                            "trajectories", traj_key, "trajectory.tum",
                        )
                        if os.path.isfile(tum_path):
                            self.body_poses[cache_key] = _load_tum(tum_path)

        elapsed = time.time() - t0
        noise_str = ""
        if self._ext_noise_enabled:
            noise_str = (f", ext_noise: rot={self._ext_noise_rot_std}deg "
                         f"trans={self._ext_noise_trans_std}m")
        logger.info(
            f"RigDatasetG2G: {len(self.pairs)} pairs, "
            f"{len(self.rig_configs)} scenes, {elapsed:.1f}s{noise_str}"
        )
        print(
            f"[RigDatasetG2G] {len(self.pairs)} pairs, "
            f"{len(self.rig_configs)} scenes, loaded in {elapsed:.1f}s{noise_str}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    # Internal method: read precomputed matrices from rig_config
    # ------------------------------------------------------------------

    @staticmethod
    def _euler_to_body_T_cam_opencv(
        pitch_deg: float, yaw_deg: float, roll_deg: float, camera_height: float,
    ) -> np.ndarray:
        """
        Compute body_T_cam from euler angles (OpenCV cam2world convention).
        Exactly matches _euler_to_body_T_cam_opencv in hm3d_stage2.py.
        Fixes coordinate-frame issues in the raw body_T_cam stored in step2 rig_config.json:
          1. euler convention (ZYX extrinsic → xyz intrinsic)
          2. missing R_x_180 (Habitat Y-up,Z-back → OpenCV Y-down,Z-forward)
        """
        R_base = Rotation.from_euler("x", 180, degrees=True)
        R_sensor = Rotation.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True)
        R_total = R_sensor * R_base
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R_total.as_matrix()
        mat[1, 3] = camera_height
        return mat

    def _get_body_T_cam(self, rig_config: dict, cam_idx: int) -> np.ndarray:
        """Get body_T_cam from rig_config (OpenCV convention)

        Two modes:
          1. body_T_cam_precomputed=True (TartanGround/NCLT): read the 4x4 matrix directly
          2. default (HM3D): recompute from euler angles (including R_x_180 correction)
        """
        cam = rig_config["cameras"][cam_idx]
        if rig_config.get("body_T_cam_precomputed", False):
            return np.array(cam["body_T_cam"], dtype=np.float64).reshape(4, 4)
        euler = cam["actual_euler_deg"]
        camera_height = rig_config["camera_height_m"]
        return self._euler_to_body_T_cam_opencv(
            euler["pitch"], euler["yaw"], euler["roll"], camera_height,
        )

    def _get_intrinsics(self, rig_config: dict, cam_idx: int) -> np.ndarray:
        """Get the 3x3 intrinsics matrix from rig_config"""
        return np.array(rig_config["cameras"][cam_idx]["intrinsics"],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Extrinsics noise injection
    # ------------------------------------------------------------------

    @staticmethod
    def _make_se3_noise(
        num_frames: int,
        rot_std_deg: float,
        trans_std_m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate an SE(3) noise matrix [num_frames, 4, 4], keeping the first frame as Identity"""
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
        self, extrinsics: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray:
        """Right-multiply by SE(3) noise: extrinsics_noisy[i] = extrinsics[i] @ noise[i]"""
        if not self._ext_noise_enabled:
            return extrinsics
        noise = self._make_se3_noise(
            len(extrinsics), self._ext_noise_rot_std,
            self._ext_noise_trans_std, rng,
        )
        return (extrinsics @ noise).astype(np.float32)

    # ------------------------------------------------------------------
    # Timestamp format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _ms_key_to_raw_ts(ms_key: str) -> str:
        """rig_pairs.json millisecond format → raw TUM format used by step3/step6 directories

        rig_pairs timestamp = int(round(tum_ts * 1000)), zero-padded to 10 digits
        step3/step6 directory name = raw TUM timestamp, zero-padded to 10 digits
        """
        return f"{int(ms_key) // 1000:010d}"

    # ------------------------------------------------------------------
    # Internal method: load DINOv2 features
    # ------------------------------------------------------------------

    def _load_feature(
        self, scene_id: str, traj_id: str, ts_key: str, cam_idx: int,
    ) -> torch.Tensor:
        """Load DINOv2 features (1024, G, G)

        step6 encoding: uint16 bit-cast from bfloat16
        Returns: float32 Tensor (1024, G, G)
        """
        # the ts_key in rig_pairs.json is already in step6 directory-name format (10-digit zero-padded integer)
        path = os.path.join(
            self.step6_root, "scenes", scene_id,
            "trajectories", traj_id, "features", ts_key,
            f"cam_{cam_idx}.npy",
        )
        feat_u16 = np.load(path)  # (1024, G, G) uint16
        # uint16 → bfloat16 → float32 (consistent with the step6 encoding)
        feat_bf16 = torch.from_numpy(
            feat_u16.astype(np.uint16),
        ).view(torch.bfloat16)
        return feat_bf16.float()  # (1024, G, G) float32

    # ------------------------------------------------------------------
    # Internal method: load RGB image (RGB / on-the-fly mode)
    # ------------------------------------------------------------------

    def _load_image(
        self, scene_id: str, traj_id: str, ts_key: str, cam_idx: int,
    ) -> np.ndarray:
        """Load an RGB image as (3, S, S) float32 in [0, 1].

        Used when step6_root is empty: the frozen backbone recomputes DINOv2
        features from these images at forward time. Path layout mirrors the
        feature cache (images/ replaces features/, .jpg replaces .npy), so ts_key
        is the rig_pairs millisecond key used directly as the directory name.
        """
        path = os.path.join(
            self.step3_root, "scenes", scene_id,
            "trajectories", traj_id, "images", ts_key,
            f"cam_{cam_idx}.jpg",
        )
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            img = Image.open(path[:-4] + ".png").convert("RGB")
        S = self.image_size
        if img.size != (S, S):
            img = img.resize((S, S), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        scene_id = pair["scene_id"]
        traj_a = pair["traj_a"]
        traj_b = pair["traj_b"]
        t_a = pair["t_a"]
        t_b = pair["t_b"]
        body_dist = pair["body_dist"]
        rig_config = self.rig_configs[scene_id]

        K = self.num_cameras
        G = self.grid_size

        # Select K cameras (sorted to keep ordering consistent). With
        # camera_select_seed set, the subset is a deterministic function of the pair
        # identity (reproducible eval, position-independent); otherwise it is drawn
        # randomly per sample as a training-time augmentation. When total_cameras == K
        # both paths return [0..K-1], so seeding only changes behavior for K < total.
        if self.camera_select_seed is not None:
            _key = f"{scene_id}|{traj_a}|{t_a}|{traj_b}|{t_b}"
            _seed = (self.camera_select_seed + zlib.crc32(_key.encode())) & 0xFFFFFFFF
            _cam_rng = np.random.default_rng(_seed)
            selected_cams = np.sort(_cam_rng.choice(self.total_cameras, K, replace=False))
        else:
            selected_cams = np.sort(
                np.random.choice(self.total_cameras, K, replace=False)
            )

        # ----------------------------------------------------------
        # Load body poses
        # ----------------------------------------------------------
        body_poses_a = self.body_poses.get((scene_id, traj_a), {})
        body_poses_b = self.body_poses.get((scene_id, traj_b), {})
        T_world_body_a = body_poses_a.get(t_a, np.eye(4, dtype=np.float64))
        T_world_body_b = body_poses_b.get(t_b, np.eye(4, dtype=np.float64))

        # ----------------------------------------------------------
        # Get body_T_cam for all selected cameras
        # ----------------------------------------------------------
        body_T_cams = [
            self._get_body_T_cam(rig_config, int(c))
            for c in selected_cams
        ]

        # ----------------------------------------------------------
        # Compute intra-rig relative extrinsics: extrinsics[i] = inv(body_T_cam[0]) @ body_T_cam[i]
        # Note: the body pose is identical across all cameras within a rig, so it cancels out
        # ----------------------------------------------------------
        T_ref_a_inv = np.linalg.inv(body_T_cams[0])
        extrinsics_a = np.stack(
            [T_ref_a_inv @ body_T_cams[i] for i in range(K)],
            axis=0,
        ).astype(np.float32)  # (K, 4, 4)

        # Rig B uses the same camera selection, so body_T_cam is identical and the intra-rig relative extrinsics are too
        extrinsics_b = extrinsics_a.copy()

        # keep clean extrinsics (used for GT computation)
        extrinsics_a_clean = extrinsics_a.copy()
        extrinsics_b_clean = extrinsics_b.copy()

        # inject extrinsics noise (A/B sampled independently)
        if self._ext_noise_enabled:
            rng = np.random.default_rng()
            extrinsics_a = self._apply_extrinsics_noise(extrinsics_a, rng)
            extrinsics_b = self._apply_extrinsics_noise(extrinsics_b, rng)

        # ----------------------------------------------------------
        # Compute cross-rig GT relative pose:
        #   T_rel_gt = inv(T_world_cam_a0) @ T_world_cam_b0
        # ----------------------------------------------------------
        T_world_cam_a0 = T_world_body_a @ body_T_cams[0]
        T_world_cam_b0 = T_world_body_b @ body_T_cams[0]
        T_rel_gt = (
            np.linalg.inv(T_world_cam_a0) @ T_world_cam_b0
        ).astype(np.float32)  # (4, 4)

        # ----------------------------------------------------------
        # Per-camera intrinsics
        # ----------------------------------------------------------
        intrinsics_list = [
            self._get_intrinsics(rig_config, int(c))
            for c in selected_cams
        ]
        intrinsics_a = np.stack(intrinsics_list, axis=0)  # (K, 3, 3)
        intrinsics_b = intrinsics_a.copy()  # same rig configuration

        H = W = self.image_size

        if self.use_rgb:
            # ------------------------------------------------------
            # RGB mode: load images, return NO enc_feats so the model recomputes
            # features with the frozen backbone (use_cached=False branch).
            # ------------------------------------------------------
            imgs_a = np.zeros((K, 3, H, W), dtype=np.float32)
            imgs_b = np.zeros((K, 3, H, W), dtype=np.float32)
            for local_idx, cam_idx in enumerate(selected_cams):
                cam_idx_int = int(cam_idx)
                try:
                    imgs_a[local_idx] = self._load_image(
                        scene_id, traj_a, t_a, cam_idx_int,
                    )
                except (FileNotFoundError, OSError) as e:
                    logger.debug(
                        f"Image missing (A): {scene_id}/{traj_a}/{t_a}/cam_{cam_idx_int}: {e}"
                    )
                try:
                    imgs_b[local_idx] = self._load_image(
                        scene_id, traj_b, t_b, cam_idx_int,
                    )
                except (FileNotFoundError, OSError) as e:
                    logger.debug(
                        f"Image missing (B): {scene_id}/{traj_b}/{t_b}/cam_{cam_idx_int}: {e}"
                    )
            images_a = torch.from_numpy(imgs_a)
            images_b = torch.from_numpy(imgs_b)
            enc_feats_a = enc_feats_b = None
        else:
            # ------------------------------------------------------
            # Cached-feature mode: load DINOv2 features (fill with zeros when missing),
            # images are dummy (the model uses the cached features directly).
            # ------------------------------------------------------
            enc_feats_a = torch.zeros(K, 1024, G, G, dtype=torch.float32)
            enc_feats_b = torch.zeros(K, 1024, G, G, dtype=torch.float32)

            for local_idx, cam_idx in enumerate(selected_cams):
                cam_idx_int = int(cam_idx)
                # Rig A features
                self._feat_load_count += 1
                try:
                    enc_feats_a[local_idx] = self._load_feature(
                        scene_id, traj_a, t_a, cam_idx_int,
                    )
                except (FileNotFoundError, OSError) as e:
                    self._feat_miss_count += 1
                    logger.debug(
                        f"Feature missing (A): {scene_id}/{traj_a}/{t_a}/cam_{cam_idx_int}: {e}"
                    )
                # Rig B features
                self._feat_load_count += 1
                try:
                    enc_feats_b[local_idx] = self._load_feature(
                        scene_id, traj_b, t_b, cam_idx_int,
                    )
                except (FileNotFoundError, OSError) as e:
                    self._feat_miss_count += 1
                    logger.debug(
                        f"Feature missing (B): {scene_id}/{traj_b}/{t_b}/cam_{cam_idx_int}: {e}"
                    )

            # check the miss rate after the first 100 loads
            if (not self._feat_miss_warned
                    and self._feat_load_count >= 100
                    and self._feat_miss_count > 0):
                miss_rate = self._feat_miss_count / self._feat_load_count * 100
                if miss_rate > 0.1:
                    logger.warning(
                        f"[RigDatasetG2G] Feature miss rate abnormally high: "
                        f"{self._feat_miss_count}/{self._feat_load_count} "
                        f"({miss_rate:.1f}%). Check the step6_root path and timestamp format!"
                    )
                    print(
                        f"[WARNING] Feature miss rate {miss_rate:.1f}% "
                        f"({self._feat_miss_count}/{self._feat_load_count}). "
                        f"Check the step6 path!",
                        flush=True,
                    )
                self._feat_miss_warned = True

            # dummy images (uses cached features, does not load the original images)
            images_a = torch.zeros(K, 3, H, W, dtype=torch.float32)
            images_b = torch.zeros(K, 3, H, W, dtype=torch.float32)

        # ----------------------------------------------------------
        # Assemble the output dict
        # ----------------------------------------------------------
        out = {
            "images_a": images_a,
            "images_b": images_b,
            "extrinsics_a": torch.from_numpy(extrinsics_a),
            "extrinsics_b": torch.from_numpy(extrinsics_b),
            "extrinsics_a_clean": torch.from_numpy(extrinsics_a_clean),
            "extrinsics_b_clean": torch.from_numpy(extrinsics_b_clean),
            "intrinsics_a": torch.from_numpy(intrinsics_a),
            "intrinsics_b": torch.from_numpy(intrinsics_b),
            "T_rel_gt": torch.from_numpy(T_rel_gt),
            "scene_id": scene_id,
            "window_overlap_score": float(body_dist),
            # Pair identity (passed through collate as a list of strings) so eval
            # outputs can be traced back to individual rig pairs.
            "traj_a": traj_a,
            "t_a": t_a,
            "traj_b": traj_b,
            "t_b": t_b,
        }
        # Only attach cached features in feature mode; their absence makes the
        # model take its non-cached (image) branch.
        if not self.use_rgb:
            out["enc_feats_a"] = enc_feats_a
            out["enc_feats_b"] = enc_feats_b
        return out

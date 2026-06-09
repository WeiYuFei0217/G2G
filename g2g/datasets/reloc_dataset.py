"""
reloc_dataset.py -- Cross-sequence relocalization (Task1) multi-frame Dataset

Inherits HM3DStage2Dataset and additionally returns noise-free clean extrinsics,
used for multi-frame pose supervision (GT of non-anchor A frames + all B frames
relative to A0).

When the base class has anchor_frame_idx != 0, the base class has already reordered
the anchor frame to index 0, so this class needs no extra handling and always uses
index 0 as the origin when computing clean extrinsics.

New fields:
  extrinsics_a_clean: [W, 4, 4] - noise-free group A extrinsics (index 0 = identity)
  extrinsics_b_clean: [W, 4, 4] - noise-free group B extrinsics (index 0 = identity)

shuffle_within_group mode:
  When enabled, the order of the 5 frames within a group is randomly shuffled in
  __getitem__, so any frame may become A0/B0; used for ablation experiments.
  When enabled, anchor_frame_idx is ignored (since the order is fully random).

Usage:
    from g2g.datasets.reloc_dataset import (
        HM3DStage2MultiFrameDataset,
        hm3d_stage2_collate_fn,
    )
"""

from __future__ import annotations

import numpy as np
import torch

from .hm3d_stage2 import HM3DStage2Dataset, hm3d_stage2_collate_fn  # noqa: F401


class HM3DStage2MultiFrameDataset(HM3DStage2Dataset):
    """
    Inherits HM3DStage2Dataset and additionally returns noise-free clean extrinsics.

    When extrinsics noise is not enabled, clean extrinsics = normal extrinsics.
    The base class already handles anchor_frame_idx reordering, so this class
    always uses index 0 as the origin.

    shuffle_within_group mode: frames within a group are randomly shuffled,
    see _getitem_shuffled() for details.
    """

    def __getitem__(self, idx: int) -> dict:
        # If shuffle_within_group is enabled, dispatch to the dedicated function
        if self.shuffle_within_group:
            return self._getitem_shuffled(idx)

        sample = super().__getitem__(idx)

        # Recompute noise-free clean extrinsics (from the cached world_T_cam)
        (scene_id, traj_a, cam_a, traj_b, cam_b,
         indices_a, indices_b, _score, _rank, _hc, _pid) = \
            self._get_sample_tuple(idx)

        # Same reordering logic as the base class
        if self.anchor_frame_idx != 0:
            a = self.anchor_frame_idx
            reorder = [a] + [i for i in range(len(indices_a)) if i != a]
            indices_a = [indices_a[i] for i in reorder]
            indices_b = [indices_b[i] for i in reorder]

        # Same window cropping logic as the base class
        if self.effective_window_size is not None and self.effective_window_size < len(indices_a):
            W_full = len(indices_a)
            W_eff = self.effective_window_size
            start = (W_full - W_eff) // 2
            indices_a = indices_a[start:start + W_eff]
            indices_b = indices_b[start:start + W_eff]

        cam_key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        cam_key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"
        world_T_cams_a = self._world_T_cam_cache[cam_key_a]
        world_T_cams_b = self._world_T_cam_cache[cam_key_b]

        inv_a0 = np.linalg.inv(world_T_cams_a[indices_a[0]])
        inv_b0 = np.linalg.inv(world_T_cams_b[indices_b[0]])

        extrinsics_a_clean = np.stack([
            inv_a0 @ world_T_cams_a[i] for i in indices_a
        ])
        extrinsics_b_clean = np.stack([
            inv_b0 @ world_T_cams_b[j] for j in indices_b
        ])

        sample["extrinsics_a_clean"] = torch.from_numpy(
            extrinsics_a_clean,
        ).float()
        sample["extrinsics_b_clean"] = torch.from_numpy(
            extrinsics_b_clean,
        ).float()

        return sample

    # ------------------------------------------------------------------
    # Shuffle-within-group: dedicated handler function
    # ------------------------------------------------------------------
    def _getitem_shuffled(self, idx: int) -> dict:
        """
        __getitem__ implementation for the intra-group frame shuffle mode.

        Same logic as the base class __getitem__, but indices_a / indices_b are
        independently and randomly shuffled before loading. anchor_frame_idx is
        ignored in this mode.

        This is a complete __getitem__ implementation (it does not call
        super().__getitem__()), to ensure the shuffled indices stay fully
        consistent across images/extrinsics/T_rel_gt/clean_extrinsics.
        """
        (scene_id, traj_a, cam_a, traj_b, cam_b,
         indices_a, indices_b, score, rank, _hc, _pid) = \
            self._get_sample_tuple(idx)

        # Core: independent random shuffle within each group
        rng = np.random.default_rng()
        rng.shuffle(indices_a)
        rng.shuffle(indices_b)

        # Window cropping (take the first N frames after shuffling, equivalent to randomly sampling a subset)
        if self.effective_window_size is not None and self.effective_window_size < len(indices_a):
            W_eff = self.effective_window_size
            indices_a = indices_a[:W_eff]
            indices_b = indices_b[:W_eff]

        # --- The following matches the base class __getitem__ logic ---

        # Load window frame images (or cached features)
        if self.use_cached_features:
            enc_feats_a = self._load_window_features(scene_id, traj_a, cam_a, indices_a)
            enc_feats_b = self._load_window_features(scene_id, traj_b, cam_b, indices_b)
            H, W_img = self.img_size
            images_a = np.zeros((len(indices_a), 3, H, W_img), dtype=np.float32)
            images_b = np.zeros((len(indices_b), 3, H, W_img), dtype=np.float32)
        else:
            images_a = self._load_window_images(scene_id, traj_a, cam_a, indices_a)
            images_b = self._load_window_images(scene_id, traj_b, cam_b, indices_b)

        # Fetch world_T_cam poses
        cam_key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        cam_key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"
        world_T_cams_a = self._world_T_cam_cache[cam_key_a]
        world_T_cams_b = self._world_T_cam_cache[cam_key_b]

        # Window anchor frame (index 0 after shuffling)
        world_T_cam_a0 = world_T_cams_a[indices_a[0]]
        world_T_cam_b0 = world_T_cams_b[indices_b[0]]

        # Intra-group relative extrinsics
        inv_a0 = np.linalg.inv(world_T_cam_a0)
        extrinsics_a = np.stack([
            inv_a0 @ world_T_cams_a[i] for i in indices_a
        ])

        inv_b0 = np.linalg.inv(world_T_cam_b0)
        extrinsics_b = np.stack([
            inv_b0 @ world_T_cams_b[j] for j in indices_b
        ])

        # Clean extrinsics (after shuffling, matching the above, saved before noise)
        extrinsics_a_clean = extrinsics_a.copy()
        extrinsics_b_clean = extrinsics_b.copy()

        # Extrinsics noise augmentation (matching the base class)
        if self._ext_noise_enabled:
            noise_rng = np.random.default_rng()
            extrinsics_a = self._apply_extrinsics_noise(
                extrinsics_a, self._ext_noise_group_a, noise_rng,
            )
            extrinsics_b = self._apply_extrinsics_noise(
                extrinsics_b, self._ext_noise_group_b, noise_rng,
            )

        # Inter-group GT: T_rel = inv(world_T_cam_a0) @ world_T_cam_b0
        T_rel_gt = inv_a0 @ world_T_cam_b0

        # Intrinsics
        key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"
        intrinsics_a = self._intrinsics_cache[key_a].copy()
        intrinsics_b = self._intrinsics_cache[key_b].copy()

        rig_w_a, rig_h_a = self._rig_resolution_cache[key_a]
        rig_w_b, rig_h_b = self._rig_resolution_cache[key_b]
        for K, rig_w, rig_h in (
            (intrinsics_a, rig_w_a, rig_h_a),
            (intrinsics_b, rig_w_b, rig_h_b),
        ):
            scale_x = self.img_size[1] / float(rig_w)
            scale_y = self.img_size[0] / float(rig_h)
            if scale_x != 1.0 or scale_y != 1.0:
                K[0, 0] *= scale_x
                K[0, 2] *= scale_x
                K[1, 1] *= scale_y
                K[1, 2] *= scale_y

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

        # Cached features
        if self.use_cached_features:
            sample["enc_feats_a"] = enc_feats_a
            sample["enc_feats_b"] = enc_feats_b

        # Clean extrinsics (for multi-frame pose supervision)
        sample["extrinsics_a_clean"] = torch.from_numpy(
            extrinsics_a_clean,
        ).float()
        sample["extrinsics_b_clean"] = torch.from_numpy(
            extrinsics_b_clean,
        ).float()

        return sample

"""
stage2_model_multiframe.py -- G2G multi-frame pose estimation model

Key differences from stage2_model.py:
  - Uses G2GBridgeMultiFrame (merged self-attention over both groups; the optional alternating inter/intra layers are disabled in all released configs via bridge_alternating_pairs=0)
  - Uses MultiFramePoseHead (predicts A1-A4 + B0-B4, 9 frame poses in total)
  - Output: poses of all frames relative to A0 (including backward-compatible T_rel)

Data flow:
  images [B, W, 3, 224, 224]
    → ImageNet normalization + extrinsics/intrinsics processing
    → MapAnything(views) → features [B, W, N_patches, 768]
    → PerceiverResampler(keep_frame_dim=True) → [B, W, L, 768]  (L = num_latents)
    → G2GBridgeMultiFrame → all_tokens_a [B, W, L, 768], all_tokens_b [B, W, L, 768]
    → MultiFramePoseHead → rotations/translations for A1-A4, B0-B4
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# MapAnything provides the frozen backbone (DINOv2-large). Install it first:
#   git clone https://github.com/facebookresearch/map-anything && pip install -e ./map-anything
from mapanything.models import MapAnything

# These two geometry utilities are official APIs of MapAnything (Meta, Apache-2.0);
# import from the official mapanything first, and fall back to the vendored copy bundled
# with g2g if its module layout changes.
try:
    from mapanything.utils.geometry import (
        get_rays_in_camera_frame,
        rotation_matrix_to_quaternion,
    )
except ImportError:  # pragma: no cover
    from g2g.utils.geometry import (
        get_rays_in_camera_frame,
        rotation_matrix_to_quaternion,
    )

from .g2g_modules import PerceiverResampler, RotationUtils
from .g2g_modules_multiframe import G2GBridgeMultiFrame, MultiFramePoseHead


class Stage2ModelMultiFrame(nn.Module):
    """
    G2G multi-frame pose estimation model.

    MapAnything (frozen) + G2G Multi-Frame (trainable).
    Predicts all poses of A1-A4 and B0-B4 relative to A0.
    """

    def __init__(
        self,
        model_path: str,
        embed_dim: int = 768,
        num_latents: int = 32,
        num_frames_per_group: int = 5,
        resampler_layers: int = 2,
        bridge_alternating_pairs: int = 2,  # released checkpoints are merge_only (0); a nonzero value builds a larger architecture that will not match published weights
        bridge_merged_layers: int = 2,
        reinject_anchor_after_merge: bool = True,
        use_anchor_embed: bool = True,
        pose_head_hidden_dim: int = 512,
        pose_head_layers: int = 3,
        num_heads: int = 8,
        rotation_repr: str = "6d",
        freeze_backbone: bool = True,
        disable_extrinsics_group_a: bool = False,
        disable_extrinsics_group_b: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_frames_per_group = num_frames_per_group
        self.rotation_repr = rotation_repr
        self.freeze_backbone = freeze_backbone
        self.disable_extrinsics_group_a = disable_extrinsics_group_a
        self.disable_extrinsics_group_b = disable_extrinsics_group_b

        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
        )

        print(f"[Stage2ModelMultiFrame] Loading MapAnything from: {model_path}")
        model_path_obj = Path(model_path).expanduser()
        if model_path_obj.exists():
            self.backbone = MapAnything.from_pretrained(str(model_path_obj))
        else:
            self.backbone = MapAnything.from_pretrained(model_path)

        self.backbone_embed_dim = self.backbone.info_sharing.dim
        print(f"[Stage2ModelMultiFrame] Backbone info_sharing dim: {self.backbone_embed_dim}")

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[Stage2ModelMultiFrame] Backbone frozen")

        if self.backbone_embed_dim != embed_dim:
            self.embed_proj = nn.Linear(self.backbone_embed_dim, embed_dim)
        else:
            self.embed_proj = nn.Identity()

        # ===== G2G Multi-Frame modules =====
        self.resampler = PerceiverResampler(
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_layers=resampler_layers,
        )

        self.bridge = G2GBridgeMultiFrame(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_alternating_pairs=bridge_alternating_pairs,
            num_merged_layers=bridge_merged_layers,
            num_latents_per_frame=num_latents,
            num_frames=num_frames_per_group,
            reinject_anchor_after_merge=reinject_anchor_after_merge,
            use_anchor_embed=use_anchor_embed,
        )

        self.pose_head = MultiFramePoseHead(
            embed_dim=embed_dim,
            hidden_dim=pose_head_hidden_dim,
            num_heads=num_heads,
            num_layers=pose_head_layers,
            num_frames=num_frames_per_group,
            rotation_repr=rotation_repr,
        )

        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage2ModelMultiFrame] Trainable: {trainable:,} / Total: {total:,}")
        if not use_anchor_embed:
            print("[Stage2ModelMultiFrame] ⚠ Anchor embed disabled (ablation mode)")
        if disable_extrinsics_group_a:
            print("[Stage2ModelMultiFrame] ⚠ Group A extrinsics injection disabled (ablation mode)")
        if disable_extrinsics_group_b:
            print("[Stage2ModelMultiFrame] ⚠ Group B extrinsics injection disabled (ablation mode)")

        self._profile_enabled = False
        self._profile_timings: Dict[str, list] = defaultdict(list)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def enable_profiling(self, enabled: bool = True) -> None:
        self._profile_enabled = enabled
        self._profile_timings = defaultdict(list)
        if enabled:
            print("[Stage2ModelMultiFrame] Profiling ENABLED")

    def _prof_record(self, name: str, start_event: torch.cuda.Event) -> torch.cuda.Event:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._profile_timings[name].append((start_event, end))
        new_start = torch.cuda.Event(enable_timing=True)
        new_start.record()
        return new_start

    def get_profiling_summary(self) -> Dict[str, Dict[str, float]]:
        torch.cuda.synchronize()
        summary = {}
        for name, events in self._profile_timings.items():
            durations = [s.elapsed_time(e) for s, e in events]
            if durations:
                import numpy as _np
                summary[name] = {
                    "mean": float(_np.mean(durations)),
                    "std": float(_np.std(durations)),
                    "count": len(durations),
                }
        return summary

    def reset_profiling(self) -> None:
        self._profile_timings = defaultdict(list)

    # ------------------------------------------------------------------
    # Internal helper methods (identical to stage2_model.py)
    # ------------------------------------------------------------------

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self._img_mean) / self._img_std

    def _extrinsics_to_quat_trans(
        self, extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        R = extrinsics[:, :, :3, :3]
        t = extrinsics[:, :, :3, 3]
        quats = rotation_matrix_to_quaternion(R)
        return quats, t

    def _build_views(
        self,
        images_norm: torch.Tensor,
        quats: torch.Tensor,
        trans: torch.Tensor,
        ray_dirs: Union[torch.Tensor, List[torch.Tensor]],
        disable_extrinsics: bool = False,
    ) -> List[Dict[str, Any]]:
        B, W = images_norm.shape[:2]
        device = images_norm.device

        # Ablation: replace extrinsics with identity (quat=[0,0,0,1], trans=[0,0,0])
        if disable_extrinsics:
            quats = torch.zeros_like(quats)
            quats[..., 3] = 1.0  # identity quaternion (xyzw format)
            trans = torch.zeros_like(trans)

        is_metric = torch.ones(B, dtype=torch.bool, device=device)
        true_shape = torch.tensor(
            [[images_norm.shape[3], images_norm.shape[4]]],
            device=device,
        ).expand(B, -1).contiguous()
        norm_type = ["dinov2"] * B

        views = []
        for i in range(W):
            rd = ray_dirs[i] if isinstance(ray_dirs, list) else ray_dirs
            view = {
                "img": images_norm[:, i],
                "ray_directions_cam": rd,
                "camera_pose_quats": quats[:, i],
                "camera_pose_trans": trans[:, i],
                "is_metric_scale": is_metric,
                "data_norm_type": norm_type,
                "true_shape": true_shape,
            }
            views.append(view)
        return views

    def _extract_features(
        self, views: List[Dict[str, Any]],
        cached_enc_feats: Optional[torch.Tensor] = None,
        prof_prefix: Optional[str] = None,
    ) -> torch.Tensor:
        _prof = self._profile_enabled and prof_prefix is not None
        use_cached = cached_enc_feats is not None

        if not _prof and not use_cached:
            features_list = self.backbone.forward(
                views, return_info_sharing_features=True,
            )
        else:
            from uniception.models.info_sharing.cross_attention_transformer import (
                MultiViewTransformerInput,
            )

            if _prof:
                _t = torch.cuda.Event(enable_timing=True)
                _t.record()

            if use_cached:
                enc_feats = list(cached_enc_feats.unbind(dim=1))
                if _prof:
                    _t = self._prof_record(f"{prof_prefix}_dinov2_CACHED", _t)
            else:
                enc_feats = self.backbone._encode_n_views(views)
                if _prof:
                    _t = self._prof_record(f"{prof_prefix}_dinov2", _t)

            batch_size_per_view = views[0]["img"].shape[0]
            with torch.autocast("cuda", enabled=False):
                enc_feats = self.backbone._encode_and_fuse_optional_geometric_inputs(
                    views, enc_feats,
                )
            if _prof:
                _t = self._prof_record(f"{prof_prefix}_geom_fuse", _t)

            input_scale_token = (
                self.backbone.scale_token.unsqueeze(0)
                .unsqueeze(-1)
                .repeat(batch_size_per_view, 1, 1)
            )
            info_input = MultiViewTransformerInput(
                features=enc_feats,
                additional_input_tokens=input_scale_token,
            )
            info_result = self.backbone.info_sharing(info_input)
            if isinstance(info_result, tuple):
                features_list = info_result[0].features
            else:
                features_list = info_result.features
            if _prof:
                self._prof_record(f"{prof_prefix}_info_share", _t)

        features_per_view = []
        for view_feat in features_list:
            B, C, H_p, W_p = view_feat.shape
            feat = view_feat.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C)
            features_per_view.append(feat)

        return torch.stack(features_per_view, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict:
                rotations_a: [B, W-1, 6]       A1-A4
                translations_a: [B, W-1, 3]
                rotation_matrices_a: [B, W-1, 3, 3]
                rotations_b: [B, W, 6]         B0-B4
                translations_b: [B, W, 3]
                rotation_matrices_b: [B, W, 3, 3]
                # backward compatibility (B0→A0)
                rotation: [B, 6]
                translation: [B, 3]
                rotation_matrix: [B, 3, 3]
                T_rel: [B, 4, 4]
        """
        images_a = batch["images_a"]
        images_b = batch["images_b"]
        extrinsics_a = batch["extrinsics_a"]
        extrinsics_b = batch["extrinsics_b"]
        intrinsics_a = batch["intrinsics_a"]
        intrinsics_b = batch["intrinsics_b"]

        cached_enc_feats_a = batch.get("enc_feats_a")
        cached_enc_feats_b = batch.get("enc_feats_b")
        use_cached = cached_enc_feats_a is not None
        assert (cached_enc_feats_a is None) == (cached_enc_feats_b is None)

        B = images_a.shape[0]
        H, W_img = images_a.shape[3], images_a.shape[4]
        device = images_a.device

        _prof = self._profile_enabled and device.type == "cuda"
        if _prof:
            _t = torch.cuda.Event(enable_timing=True)
            _t.record()

        # 1. Normalization
        if use_cached:
            images_a_norm = images_a
            images_b_norm = images_b
        else:
            images_a_norm = self._normalize_images(images_a)
            images_b_norm = self._normalize_images(images_b)
        if _prof:
            _t = self._prof_record("1_normalize", _t)

        # 2. Extrinsics → quat + trans
        quats_a, trans_a = self._extrinsics_to_quat_trans(extrinsics_a)
        quats_b, trans_b = self._extrinsics_to_quat_trans(extrinsics_b)
        if _prof:
            _t = self._prof_record("2_ext_to_quat", _t)

        # 3. Intrinsics → ray_dirs
        # rig mode: intrinsics [B, W, 3, 3] → per-frame ray_dirs
        # original mode: intrinsics [B, 3, 3] → shared ray_dirs
        if intrinsics_a.dim() == 4:
            W = intrinsics_a.shape[1]
            ray_dirs_a = [
                get_rays_in_camera_frame(
                    intrinsics=intrinsics_a[:, i], height=H, width=W_img,
                    normalize_to_unit_sphere=True,
                )[1] for i in range(W)
            ]
            ray_dirs_b = [
                get_rays_in_camera_frame(
                    intrinsics=intrinsics_b[:, i], height=H, width=W_img,
                    normalize_to_unit_sphere=True,
                )[1] for i in range(W)
            ]
        else:
            _, ray_dirs_a = get_rays_in_camera_frame(
                intrinsics=intrinsics_a, height=H, width=W_img,
                normalize_to_unit_sphere=True,
            )
            _, ray_dirs_b = get_rays_in_camera_frame(
                intrinsics=intrinsics_b, height=H, width=W_img,
                normalize_to_unit_sphere=True,
            )
        if _prof:
            _t = self._prof_record("3_ray_dirs", _t)

        # 4. Assemble views
        views_a = self._build_views(images_a_norm, quats_a, trans_a, ray_dirs_a,
                                    disable_extrinsics=self.disable_extrinsics_group_a)
        views_b = self._build_views(images_b_norm, quats_b, trans_b, ray_dirs_b,
                                    disable_extrinsics=self.disable_extrinsics_group_b)
        if _prof:
            _t = self._prof_record("4_build_views", _t)

        # 5. MapAnything feature extraction
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features_a = self._extract_features(
                    views_a,
                    cached_enc_feats=cached_enc_feats_a,
                    prof_prefix="5a" if _prof else None,
                )
                if _prof:
                    _t = self._prof_record("5a_backbone_A_total", _t)
                features_b = self._extract_features(
                    views_b,
                    cached_enc_feats=cached_enc_feats_b,
                    prof_prefix="5b" if _prof else None,
                )
                if _prof:
                    _t = self._prof_record("5b_backbone_B_total", _t)

        features_a = features_a.detach().float()
        features_b = features_b.detach().float()
        if _prof:
            _t = self._prof_record("5c_detach_float", _t)

        # 6. Projection
        features_a = self.embed_proj(features_a)
        features_b = self.embed_proj(features_b)
        if _prof:
            _t = self._prof_record("6_embed_proj", _t)

        # 7. PerceiverResampler
        latents_a = self.resampler(features_a, keep_frame_dim=True)
        latents_b = self.resampler(features_b, keep_frame_dim=True)
        if _prof:
            _t = self._prof_record("7_resampler", _t)

        # 8. G2GBridgeMultiFrame → tokens for all frames
        all_tokens_a, all_tokens_b = self.bridge(latents_a, latents_b)
        if _prof:
            _t = self._prof_record("8_bridge", _t)

        # 9. MultiFramePoseHead → 9 poses
        rotations_a, translations_a, rotations_b, translations_b = self.pose_head(
            all_tokens_a, all_tokens_b,
        )
        if _prof:
            _t = self._prof_record("9_pose_head", _t)

        # 10. Build output
        W = self.num_frames_per_group

        # Rotation matrices (batched)
        rot_a_flat = rotations_a.reshape(-1, self.pose_head.rot_dim)
        rot_b_flat = rotations_b.reshape(-1, self.pose_head.rot_dim)
        rm_a = RotationUtils.rotation_6d_to_matrix(rot_a_flat).reshape(B, W - 1, 3, 3)
        rm_b = RotationUtils.rotation_6d_to_matrix(rot_b_flat).reshape(B, W, 3, 3)

        # backward compatibility: B0→A0 pose
        rotation_b0 = rotations_b[:, 0]          # [B, 6]
        translation_b0 = translations_b[:, 0]    # [B, 3]
        rotation_matrix_b0 = rm_b[:, 0]           # [B, 3, 3]

        T_rel = torch.eye(4, device=device, dtype=rotation_b0.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        T_rel[:, :3, :3] = rotation_matrix_b0
        T_rel[:, :3, 3] = translation_b0

        if _prof:
            self._prof_record("10_output", _t)

        return {
            "rotations_a": rotations_a,
            "translations_a": translations_a,
            "rotation_matrices_a": rm_a,
            "rotations_b": rotations_b,
            "translations_b": translations_b,
            "rotation_matrices_b": rm_b,
            # backward compatibility
            "rotation": rotation_b0,
            "translation": translation_b0,
            "rotation_matrix": rotation_matrix_b0,
            "T_rel": T_rel,
        }

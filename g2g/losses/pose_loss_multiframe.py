"""
pose_loss_multiframe.py -- G2G multi-frame pose loss functions and evaluation metrics

Differences from pose_loss.py:
  - Supervises the pose of all frames (A1-A4 + B0-B4, 9 frames total)
  - Separately weights the intra-group (intra, A1-A4) and inter-group (inter, B0-B4) losses
  - GT construction: A frames use clean extrinsics directly, B frames use T_rel_gt @ extrinsics_b_clean

Usage:
    from g2g.losses.pose_loss_multiframe import MultiFrameG2GLoss, compute_multiframe_metrics

    criterion = MultiFrameG2GLoss(intra_weight=0.3, inter_weight=1.0)
    losses = criterion(pred, batch)
    metrics = compute_multiframe_metrics(pred, batch)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFrameG2GLoss(nn.Module):
    """
    Multi-frame G2G pose loss.

    A1-A4 (intra-group): loss = intra_weight * (rot_weight * chordal + trans_weight * L1)
    B0-B4 (inter-group): loss = inter_weight * (rot_weight * chordal + trans_weight * L1)

    GT construction:
      - A frame GT: extrinsics_a_clean[:, 1:]  (T_{A0←Ai}, i.e. the noise-free input extrinsics)
      - B frame GT: T_rel_gt @ extrinsics_b_clean  (T_{A0←Bj} = T_{A0←B0} @ T_{B0←Bj})
    """

    def __init__(
        self,
        rotation_weight: float = 5.0,
        translation_weight: float = 1.0,
        intra_weight: float = 0.3,
        inter_weight: float = 1.0,
        loss_type: str = "chordal",
        loss_clamp: float = 10.0,
    ):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.loss_type = loss_type
        self.loss_clamp = loss_clamp
        assert loss_type in ("chordal", "chordal_frob", "geodesic"), (
            f"Unknown loss_type: {loss_type}"
        )

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute the multi-frame loss.

        Args:
            pred: model output, containing:
                rotation_matrices_a: [B, W-1, 3, 3]
                translations_a: [B, W-1, 3]
                rotation_matrices_b: [B, W, 3, 3]
                translations_b: [B, W, 3]
            batch: collated batch, containing:
                extrinsics_a_clean: [B, W, 4, 4]
                extrinsics_b_clean: [B, W, 4, 4]
                T_rel_gt: [B, 4, 4]

        Returns:
            dict: loss, rot_loss_intra, trans_loss_intra,
                  rot_loss_inter, trans_loss_inter,
                  rot_loss, trans_loss (backward compatible: B0→A0 only)
        """
        R_pred_a = pred["rotation_matrices_a"]   # [B, W-1, 3, 3]
        t_pred_a = pred["translations_a"]        # [B, W-1, 3]
        R_pred_b = pred["rotation_matrices_b"]   # [B, W, 3, 3]
        t_pred_b = pred["translations_b"]        # [B, W, 3]

        ext_a_clean = batch["extrinsics_a_clean"]  # [B, W, 4, 4]
        ext_b_clean = batch["extrinsics_b_clean"]  # [B, W, 4, 4]
        T_rel_gt = batch["T_rel_gt"]               # [B, 4, 4]

        # === Construct GT ===
        # A frame GT: extrinsics_a_clean[:, 1:]
        R_gt_a = ext_a_clean[:, 1:, :3, :3]  # [B, W-1, 3, 3]
        t_gt_a = ext_a_clean[:, 1:, :3, 3]   # [B, W-1, 3]

        # B frame GT: T_{A0←Bj} = T_rel_gt @ extrinsics_b_clean[:, j]
        T_gt_b = T_rel_gt.unsqueeze(1) @ ext_b_clean  # [B, W, 4, 4]
        R_gt_b = T_gt_b[:, :, :3, :3]  # [B, W, 3, 3]
        t_gt_b = T_gt_b[:, :, :3, 3]   # [B, W, 3]

        # === Intra loss (A1-A4) ===
        rot_loss_intra = self._batch_rotation_loss(R_pred_a, R_gt_a)
        trans_loss_intra = F.l1_loss(t_pred_a, t_gt_a)

        # === Inter loss (B0-B4) ===
        rot_loss_inter = self._batch_rotation_loss(R_pred_b, R_gt_b)
        trans_loss_inter = F.l1_loss(t_pred_b, t_gt_b)

        # === Weighted sum ===
        intra_loss = self.intra_weight * (
            self.rotation_weight * rot_loss_intra
            + self.translation_weight * trans_loss_intra
        )
        inter_loss = self.inter_weight * (
            self.rotation_weight * rot_loss_inter
            + self.translation_weight * trans_loss_inter
        )

        total = intra_loss + inter_loss

        # Clamp total loss to prevent gradient explosion from outlier
        # predictions (e.g. degenerate Gram-Schmidt output under noise).
        if self.loss_clamp > 0:
            total = torch.clamp(total, max=self.loss_clamp)

        # Backward compatible: B0-only loss (used for curriculum learning convergence detection)
        R_pred_b0 = R_pred_b[:, 0:1]
        R_gt_b0 = R_gt_b[:, 0:1]
        rot_loss_b0 = self._batch_rotation_loss(R_pred_b0, R_gt_b0)
        trans_loss_b0 = F.l1_loss(t_pred_b[:, 0], t_gt_b[:, 0])

        return {
            "loss": total,
            "rot_loss_intra": rot_loss_intra,
            "trans_loss_intra": trans_loss_intra,
            "rot_loss_inter": rot_loss_inter,
            "trans_loss_inter": trans_loss_inter,
            # Backward compatible (B0→A0 only, used for curriculum learning convergence detection)
            "rot_loss": rot_loss_b0,
            "trans_loss": trans_loss_b0,
        }

    def _batch_rotation_loss(
        self,
        R_pred: torch.Tensor,
        R_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched rotation loss.

        Args:
            R_pred, R_gt: [B, N, 3, 3]

        Returns:
            scalar loss
        """
        if self.loss_type == "chordal":
            return self._batch_chordal_loss(R_pred, R_gt)
        elif self.loss_type == "chordal_frob":
            return self._batch_chordal_frob_loss(R_pred, R_gt)
        else:
            loss = self._batch_geodesic_loss(R_pred, R_gt)
            if torch.isnan(loss) or torch.isinf(loss):
                loss = self._batch_chordal_loss(R_pred, R_gt)
            return loss

    @staticmethod
    def _batch_chordal_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """L1 chordal distance: L1(R_pred^T @ R_gt, I), supports [..., 3, 3].

        Note: L1 chordal is non-monotonic over the 90°-180° range (90° and 180° have the same loss),
        which may cause mode collapse for large-angle rig rotations. Consider using chordal_frob.
        """
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype)
        for _ in range(R_pred.dim() - 2):
            I = I.unsqueeze(0)
        I = I.expand_as(R_pred)
        R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        return F.l1_loss(R_diff, I)

    @staticmethod
    def _batch_chordal_frob_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Frobenius chordal distance: MSE(R_pred^T @ R_gt, I), supports [..., 3, 3].

        Unlike L1 chordal, Frobenius (MSE) is strictly monotonically increasing over [0°, 180°],
        and can correctly distinguish rotation errors of 90° and 180°.
        """
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype)
        for _ in range(R_pred.dim() - 2):
            I = I.unsqueeze(0)
        I = I.expand_as(R_pred)
        R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        return F.mse_loss(R_diff, I)

    @staticmethod
    def _batch_geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Geodesic distance loss, supports [..., 3, 3]."""
        R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.99999, 0.99999)
        return torch.acos(cos_angle).mean()


def compute_multiframe_metrics(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute multi-frame evaluation metrics.

    Returns:
        dict:
            rot_error_deg_mean: mean B0→A0 rotation error (backward compatible)
            trans_error_m_mean: mean B0→A0 translation error
            rot_error_a_mean: mean A1-A4 rotation error
            trans_error_a_mean: mean A1-A4 translation error
            rot_error_b_mean: mean B0-B4 rotation error
            trans_error_b_mean: mean B0-B4 translation error
    """
    with torch.no_grad():
        R_pred_a = pred["rotation_matrices_a"]   # [B, W-1, 3, 3]
        t_pred_a = pred["translations_a"]        # [B, W-1, 3]
        R_pred_b = pred["rotation_matrices_b"]   # [B, W, 3, 3]
        t_pred_b = pred["translations_b"]        # [B, W, 3]

        ext_a_clean = batch["extrinsics_a_clean"]  # [B, W, 4, 4]
        ext_b_clean = batch["extrinsics_b_clean"]  # [B, W, 4, 4]
        T_rel_gt = batch["T_rel_gt"]               # [B, 4, 4]

        R_gt_a = ext_a_clean[:, 1:, :3, :3]
        t_gt_a = ext_a_clean[:, 1:, :3, 3]

        T_gt_b = T_rel_gt.unsqueeze(1) @ ext_b_clean
        R_gt_b = T_gt_b[:, :, :3, :3]
        t_gt_b = T_gt_b[:, :, :3, 3]

        # A frame error
        rot_err_a = _batch_rotation_error_deg(R_pred_a, R_gt_a)
        trans_err_a = torch.norm(t_pred_a - t_gt_a, dim=-1)

        # B frame error
        rot_err_b = _batch_rotation_error_deg(R_pred_b, R_gt_b)
        trans_err_b = torch.norm(t_pred_b - t_gt_b, dim=-1)

        # Backward compatible: B0 only
        rot_err_b0 = rot_err_b[:, 0]
        trans_err_b0 = trans_err_b[:, 0]

        result = {
            "rot_error_deg_mean": rot_err_b0.mean().item(),
            "rot_error_deg_median": rot_err_b0.median().item(),
            "trans_error_m_mean": trans_err_b0.mean().item(),
            "trans_error_m_median": trans_err_b0.median().item(),
            "rot_error_a_mean": rot_err_a.mean().item(),
            "trans_error_a_mean": trans_err_a.mean().item(),
            "rot_error_b_mean": rot_err_b.mean().item(),
            "trans_error_b_mean": trans_err_b.mean().item(),
        }

        # per-frame A error (diagnose mode collapse)
        n_a = rot_err_a.shape[1]
        for i in range(n_a):
            result[f"rot_error_a{i+1}_mean"] = rot_err_a[:, i].mean().item()

        return result


def _batch_rotation_error_deg(
    R_pred: torch.Tensor, R_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Batched rotation error (degrees), supports [..., 3, 3].

    Returns:
        [...] rotation error (degrees)
    """
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle) * (180.0 / math.pi)

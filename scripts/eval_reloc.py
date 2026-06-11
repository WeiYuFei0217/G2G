#!/usr/bin/env python3
"""
eval_reloc.py -- G2G relocalization evaluation script

Runs inference on the validation set, bins rotation/translation errors by overlap,
generates visualization plots, and saves the raw results for later analysis.
Automatically detects a noresampler checkpoint and replaces it with IdentityResampler.

Supports single-GPU and multi-GPU inference:
    # Single GPU
    python scripts/eval_reloc.py \
        --config configs/reloc/hm3d.yaml \
        --checkpoint release_weights/HM3D-Reloc.pth \
        --output-dir outputs/eval_HM3D-Reloc \
        --batch-size 16 --min-overlap 0.1

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 --master-port=29590 \
        scripts/eval_reloc.py \
        --config configs/reloc/tartanground.yaml \
        --checkpoint release_weights/TartanGround-Reloc.pth \
        --batch-size 16 --min-overlap 0.1

    # Re-analysis only (skip inference, reload from existing CSV, no multi-GPU needed)
    python scripts/eval_reloc.py \
        --config configs/reloc/hm3d.yaml \
        --checkpoint release_weights/HM3D-Reloc.pth \
        --skip-inference

Note: --min-overlap selects the matching cached index; keep it consistent across runs
so the evaluation reuses the same downsampled cache instead of rebuilding the full index.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# ============================================================================
# Dependencies: the g2g package installed via `pip install -e .` in this repo,
# and an already-installed mapanything.
# ============================================================================
from g2g.datasets.hm3d_stage2 import HM3DStage2Dataset, hm3d_stage2_collate_fn
from g2g.datasets.reloc_dataset import (
    HM3DStage2MultiFrameDataset,
    hm3d_stage2_collate_fn as hm3d_stage2_multiframe_collate_fn,  # noqa: F811
)

logger = logging.getLogger(__name__)


# ============================================================================
# IdentityResampler: an identity replacement that skips the PerceiverResampler
# (active when num_latents equals the number of backbone patch tokens)
# ============================================================================
class IdentityResampler(torch.nn.Module):
    """
    Identity resampler: when num_latents == the number of backbone patch tokens
    (e.g. 256), directly skip the PerceiverResampler and apply no transformation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # No trainable parameters at all

    def forward(
        self,
        x: torch.Tensor,
        pos_embed=None,
        keep_frame_dim: bool = False,
    ) -> torch.Tensor:
        return x

# ============================================================================
# Overlap bin definitions (10 intervals, following reloc3r)
# ============================================================================
OVERLAP_BINS = [
    (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01),  # 1.01 sentinel value ensures 1.0 is included
]

# Display-label fix for the last bin (1.01 is a sentinel value; the actual range is [0.9, 1.0])
def _bin_label(low: float, high: float) -> str:
    if high > 1.0:
        return f"[{low:.1f}, 1.0]"
    return f"[{low:.1f}, {high:.1f})"

# Success threshold definitions
SUCCESS_ROT_THRESH_DEG = 10.0
SUCCESS_TRANS_THRESH_M = 0.5


# ============================================================================
# YAML config loading
# ============================================================================
def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# DDP utility functions
# ============================================================================
def setup_ddp(timeout_minutes: int = 30) -> tuple[int, int]:
    """
    Initialize the distributed environment.

    When launched via torchrun, the RANK/LOCAL_RANK/WORLD_SIZE environment
    variables are present automatically; when launched directly with python it
    falls back to single-GPU mode (local_rank=-1, world_size=1).

    Returns:
        (local_rank, world_size)
    """
    if "RANK" in os.environ:
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_minutes),
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, int(os.environ["WORLD_SIZE"])
    return -1, 1


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    return local_rank <= 0


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2G validation-set evaluation")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model weights (best_model.pth or a full checkpoint)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Output directory for evaluation results (default: <checkpoint dir>/eval_results)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Inference device (default: cuda; assigned automatically by local_rank in multi-GPU mode)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Per-GPU inference batch size (default: 16)",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip inference and re-analyze only from an existing raw_results.csv",
    )
    parser.add_argument(
        "--min-overlap", type=float, default=0.0,
        help="Minimum overlap score threshold (default: 0.0, evaluate all samples)",
    )
    parser.add_argument(
        "--max-overlap", type=float, default=1.01,
        help="Maximum overlap score threshold (default: 1.01, no filtering; set to 0.1 to keep only samples with score < 0.1)",
    )
    parser.add_argument(
        "--top-k", type=int, default=-1,
        help="Number of windows taken per pair (default: read from config; set to 1 to use only the best window)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=-1,
        help="Number of DataLoader workers (default: read from config)",
    )
    parser.add_argument(
        "--model-type", type=str, default="auto",
        choices=["auto", "singleframe", "multiframe"],
        help="Model type: auto (inferred from config) / singleframe / multiframe",
    )
    parser.add_argument(
        "--noise", action="store_true",
        help="Enable extrinsics noise injection (using the parameters in data.extrinsics_noise from the config)",
    )
    parser.add_argument(
        "--anchor-frame-idx", type=int, default=-1,
        help="Intra-group anchor frame index (default: read from config; 0=first frame, 2=middle frame)",
    )
    parser.add_argument(
        "--shuffle-within-group", action="store_true", default=None,
        help="Enable random shuffling of frames within a group (default: read from config)",
    )
    parser.add_argument(
        "--no-shuffle-within-group", action="store_true",
        help="Force-disable random shuffling of frames within a group (overrides the config setting)",
    )
    return parser.parse_args()


# ============================================================================
# Model construction and loading
# ============================================================================
def infer_model_type(cfg: dict, cli_model_type: str) -> str:
    """
    Infer the evaluation model type.

    Rules:
      1) If the CLI explicitly specifies singleframe/multiframe, use it first;
      2) In auto mode, if config.model contains multiframe-only fields, it is
         determined to be multiframe.
    """
    if cli_model_type in ("singleframe", "multiframe"):
        return cli_model_type

    model_cfg = cfg.get("model", {})
    multiframe_markers = {
        "bridge_alternating_pairs",
        "bridge_merged_layers",
        "reinject_anchor_after_merge",
    }
    if any(k in model_cfg for k in multiframe_markers):
        return "multiframe"
    return "singleframe"


def build_and_load_model(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    model_type: str,
    is_main: bool = True,
) -> torch.nn.Module:
    """
    Build a Stage2Model from the config and load a checkpoint.

    best_model.pth holds G2G-only weights (without backbone.*), loaded with strict=False.
    """
    from g2g.models.stage2_model_multiframe import Stage2ModelMultiFrame

    backbone_cfg = cfg.get("backbone", {})
    model_cfg = cfg.get("model", {})

    if model_type == "multiframe":
        model = Stage2ModelMultiFrame(
            model_path=backbone_cfg["model_path"],
            embed_dim=model_cfg.get("embed_dim", 768),
            num_latents=model_cfg.get("num_latents", 32),
            num_frames_per_group=model_cfg.get("num_frames_per_group", 5),
            resampler_layers=model_cfg.get("resampler_layers", 2),
            bridge_alternating_pairs=model_cfg.get("bridge_alternating_pairs", 2),
            bridge_merged_layers=model_cfg.get("bridge_merged_layers", 2),
            reinject_anchor_after_merge=model_cfg.get("reinject_anchor_after_merge", True),
            pose_head_hidden_dim=model_cfg.get("pose_head_hidden_dim", 512),
            pose_head_layers=model_cfg.get("pose_head_layers", 3),
            num_heads=model_cfg.get("num_heads", 8),
            rotation_repr=model_cfg.get("rotation_repr", "6d"),
            freeze_backbone=backbone_cfg.get("freeze", True),
            disable_extrinsics_group_a=model_cfg.get("disable_extrinsics_group_a", False),
            disable_extrinsics_group_b=model_cfg.get("disable_extrinsics_group_b", False),
            use_anchor_embed=model_cfg.get("use_anchor_embed", True),
        ).to(device)
    else:
        # Only the multi-frame model is shipped in this release. With --model-type
        # auto (the default), the released configs always resolve to multiframe, so
        # this branch is never taken in normal use.
        raise NotImplementedError(
            "Single-frame Stage2Model is not included in this release; "
            "only the multi-frame model is supported (use --model-type auto/multiframe)."
        )

    # Load the checkpoint
    if is_main:
        print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        # Full checkpoint (with backbone)
        # First check whether it is a noresampler checkpoint
        ckpt_keys = set(ckpt["model_state_dict"].keys())
        has_resampler_in_ckpt = any(k.startswith("resampler.") for k in ckpt_keys)
        if not has_resampler_in_ckpt and hasattr(model, "resampler"):
            _embed_dim = model_cfg.get("embed_dim", 768)
            model.resampler = IdentityResampler(_embed_dim).to(device)
            if is_main:
                print("[Eval] Detected noresampler checkpoint → replaced resampler with IdentityResampler")
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint does not match the current model type (model_type={model_type}). "
                f"Please check that --config / --model-type / --checkpoint correspond.\n"
                f"Original error: {exc}"
            ) from exc
        if is_main:
            print("[Eval] Loaded full checkpoint (strict=True)")
    else:
        # G2G-only weights (without backbone.*)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # Verify that all missing keys start with backbone.
        non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]

        # Check whether it is noresampler: non_backbone_missing are all resampler.* and there is no unexpected
        if non_backbone_missing and not unexpected:
            resampler_missing = [k for k in non_backbone_missing if k.startswith("resampler.")]
            other_missing = [k for k in non_backbone_missing if not k.startswith("resampler.")]
            if resampler_missing and not other_missing:
                # noresampler checkpoint: replace the resampler and reload
                _embed_dim = model_cfg.get("embed_dim", 768)
                model.resampler = IdentityResampler(_embed_dim).to(device)
                if is_main:
                    print(f"[Eval] Detected noresampler checkpoint "
                          f"(missing {len(resampler_missing)} resampler keys) "
                          f"→ replaced resampler with IdentityResampler")
                # Reload
                missing, unexpected = model.load_state_dict(ckpt, strict=False)
                non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]

        if non_backbone_missing or unexpected:
            raise RuntimeError(
                "[Eval] Failed to load G2G-only weights: detected non-backbone missing/unexpected parameters.\n"
                f"  model_type={model_type}\n"
                f"  non_backbone_missing({len(non_backbone_missing)}): {non_backbone_missing[:20]}\n"
                f"  unexpected({len(unexpected)}): {unexpected[:20]}\n"
                "This usually means a multiframe/singleframe checkpoint was mixed up."
            )
        if is_main:
            print(
                f"[Eval] Loaded G2G weights (strict=False), "
                f"{len(missing)} backbone keys skipped"
            )

    model.eval()
    return model


# ============================================================================
# Validation-set DataLoader construction
# ============================================================================
def build_val_loader(
    cfg: dict,
    min_overlap: float,
    batch_size: int,
    num_workers: int,
    top_k: int = -1,
    model_type: str = "singleframe",
    local_rank: int = -1,
    world_size: int = 1,
    extrinsics_noise_cfg: dict | None = None,
    anchor_frame_idx: int = -1,
    shuffle_within_group: bool | None = None,
) -> tuple[DataLoader, HM3DStage2Dataset]:
    """
    Build the validation-set DataLoader from the config.

    In multi-GPU mode, a DistributedSampler (shuffle=False) shards the data.

    Args:
        top_k: number of windows taken per pair. -1 means read from config.
        extrinsics_noise_cfg: extrinsics noise config (None or an empty dict means no noise injection).
        anchor_frame_idx: intra-group anchor frame index. -1 means read from config (multiframe only).
        shuffle_within_group: random shuffling of frames within a group. None means read from config (multiframe only).

    Returns:
        (DataLoader, dataset)
    """
    data_cfg = cfg.get("data", {})
    val_cfg = data_cfg.get("val", {})

    effective_top_k = top_k if top_k > 0 else data_cfg.get("top_k", 3)

    # multiframe-only parameters: anchor_frame_idx, shuffle_within_group
    effective_anchor = anchor_frame_idx if anchor_frame_idx >= 0 else data_cfg.get("anchor_frame_idx", 0)
    effective_shuffle = shuffle_within_group if shuffle_within_group is not None else data_cfg.get("shuffle_within_group", False)

    effective_window_size = data_cfg.get("effective_window_size", None)

    # Per-scene sampling cap (aligned with training val). Enabled when val.max_samples_per_scene > 0.
    max_samples_per_scene = val_cfg.get("max_samples_per_scene", -1)

    if model_type == "multiframe":
        dataset = HM3DStage2MultiFrameDataset(
            step1_root=val_cfg["step1_root"],
            step2_root=val_cfg["step2_root"],
            step3_root=val_cfg["step3_root"],
            stage2_index_root=val_cfg["stage2_index_root"],
            img_size=tuple(data_cfg.get("img_size", [518, 518])),
            window_size=data_cfg.get("window_size", 5),
            top_k=effective_top_k,
            min_overlap_score=min_overlap,
            step6_root=val_cfg.get("step6_root", ""),
            extrinsics_noise_cfg=extrinsics_noise_cfg,
            anchor_frame_idx=effective_anchor,
            shuffle_within_group=effective_shuffle,
            effective_window_size=effective_window_size,
            max_samples_per_scene=max_samples_per_scene,
        )
    else:
        dataset = HM3DStage2Dataset(
            step1_root=val_cfg["step1_root"],
            step2_root=val_cfg["step2_root"],
            step3_root=val_cfg["step3_root"],
            stage2_index_root=val_cfg["stage2_index_root"],
            img_size=tuple(data_cfg.get("img_size", [518, 518])),
            window_size=data_cfg.get("window_size", 5),
            top_k=effective_top_k,
            min_overlap_score=min_overlap,
            step6_root=val_cfg.get("step6_root", ""),
            extrinsics_noise_cfg=extrinsics_noise_cfg,
            effective_window_size=effective_window_size,
            max_samples_per_scene=max_samples_per_scene,
        )

    # Multi-GPU: DistributedSampler shards automatically
    sampler = None
    if local_rank != -1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=hm3d_stage2_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    if is_main_process(local_rank):
        noise_info = ""
        if extrinsics_noise_cfg and extrinsics_noise_cfg.get("enabled", False):
            noise_info = (
                f", NOISE: rot_std={extrinsics_noise_cfg.get('rotation_noise_std_deg', 1.0)}deg "
                f"trans_std={extrinsics_noise_cfg.get('translation_noise_std_m', 0.05)}m"
            )
        multiframe_info = ""
        if model_type == "multiframe":
            multiframe_info = (
                f", anchor_frame_idx={effective_anchor}"
                f", shuffle_within_group={effective_shuffle}"
            )
        print(
            f"[Eval] Val dataset: {len(dataset)} samples, "
            f"model_type={model_type}, "
            f"top_k={effective_top_k}, "
            f"{batch_size=} (per GPU), {num_workers=}, "
            f"world_size={world_size}{noise_info}{multiframe_info}"
        )

    return loader, dataset


# ============================================================================
# Per-sample error computation
# ============================================================================
def compute_per_sample_errors(
    pred: dict[str, torch.Tensor],
    T_rel_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotation error (degrees) and translation error (meters) per sample.

    Args:
        pred: model output (rotation_matrix: [B, 3, 3], translation: [B, 3])
        T_rel_gt: [B, 4, 4]

    Returns:
        rot_errors: [B] rotation error (degrees)
        trans_errors: [B] translation error (meters)
    """
    R_pred = pred["rotation_matrix"]  # [B, 3, 3]
    R_gt = T_rel_gt[:, :3, :3]
    t_pred = pred["translation"]      # [B, 3]
    t_gt = T_rel_gt[:, :3, 3]

    # Rotation error: arccos(clamp((trace(R_pred^T @ R_gt) - 1) / 2)) * 180/pi
    R_diff = torch.bmm(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    rot_errors = torch.acos(cos_angle) * (180.0 / math.pi)  # [B]

    # Translation error: ||t_pred - t_gt||_2
    trans_errors = torch.norm(t_pred - t_gt, dim=-1)  # [B]

    return rot_errors, trans_errors


def compute_multiframe_per_sample_errors(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute additional multi-frame errors:
      - A1-A4 average rotation/translation error
      - B0-B4 average rotation/translation error
    """
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

    def _rot_err_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        R_diff = torch.matmul(R1.transpose(-1, -2), R2)
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_angle) * (180.0 / math.pi)

    rot_a = _rot_err_deg(R_pred_a, R_gt_a).mean(dim=1)  # [B]
    trans_a = torch.norm(t_pred_a - t_gt_a, dim=-1).mean(dim=1)  # [B]
    rot_b = _rot_err_deg(R_pred_b, R_gt_b).mean(dim=1)  # [B]
    trans_b = torch.norm(t_pred_b - t_gt_b, dim=-1).mean(dim=1)  # [B]
    return rot_a, trans_a, rot_b, trans_b


# ============================================================================
# Inference
# ============================================================================
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str = "singleframe",
    is_main: bool = True,
) -> list[dict]:
    """
    Iterate over the validation set to run inference, collecting per-sample errors and metadata.

    Args:
        model: evaluation model
        loader: data loader (possibly sharded by a DistributedSampler)
        device: inference device
        is_main: whether this is the main process (controls tqdm display)

    Returns:
        list[dict]: sample results processed by this process
    """
    results = []

    # Only the main process shows the progress bar
    iterator = tqdm(loader, desc="Inference", disable=not is_main)

    for batch in iterator:
        # Move tensors to the device
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device, non_blocking=True)
            else:
                batch_device[k] = v

        # Forward inference
        pred = model(batch_device)

        T_rel_gt = batch_device["T_rel_gt"]  # [B, 4, 4]
        rot_errors, trans_errors = compute_per_sample_errors(pred, T_rel_gt)
        rot_a_mean = trans_a_mean = rot_b_mean = trans_b_mean = None
        if model_type == "multiframe":
            rot_a_mean, trans_a_mean, rot_b_mean, trans_b_mean = compute_multiframe_per_sample_errors(
                pred, batch_device,
            )

        B = rot_errors.shape[0]
        for i in range(B):
            record = {
                "scene_id": batch["scene_id"][i],
                "traj_a_id": batch["traj_a_id"][i],
                "cam_a_id": int(batch["cam_a_id"][i]) if isinstance(batch["cam_a_id"], torch.Tensor) else batch["cam_a_id"][i],
                "traj_b_id": batch["traj_b_id"][i],
                "cam_b_id": int(batch["cam_b_id"][i]) if isinstance(batch["cam_b_id"], torch.Tensor) else batch["cam_b_id"][i],
                "window_rank": int(batch["window_rank"][i]) if isinstance(batch["window_rank"], torch.Tensor) else batch["window_rank"][i],
                "window_overlap_score": float(batch["window_overlap_score"][i]),
                "rot_error_deg": float(rot_errors[i].item()),
                "trans_error_m": float(trans_errors[i].item()),
                "rot_error_a_mean_deg": (
                    float(rot_a_mean[i].item()) if rot_a_mean is not None else float("nan")
                ),
                "trans_error_a_mean_m": (
                    float(trans_a_mean[i].item()) if trans_a_mean is not None else float("nan")
                ),
                "rot_error_b_mean_deg": (
                    float(rot_b_mean[i].item()) if rot_b_mean is not None else float("nan")
                ),
                "trans_error_b_mean_m": (
                    float(trans_b_mean[i].item()) if trans_b_mean is not None else float("nan")
                ),
                "T_rel_pred": pred["T_rel"][i].cpu().numpy().tolist(),
                "T_rel_gt": T_rel_gt[i].cpu().numpy().tolist(),
            }
            results.append(record)

    return results


def gather_results(
    local_results: list[dict],
    local_rank: int,
    world_size: int,
) -> list[dict]:
    """
    Multi-GPU mode: gather each process's results onto rank 0.
    Single-GPU mode: return directly.

    Note: with drop_last=False, the DistributedSampler pads the last incomplete
    batch (by duplicating the first few samples), making the total count larger
    than the actual dataset size. After gathering, it must be truncated to the
    actual dataset size.
    """
    if local_rank == -1 or world_size == 1:
        return local_results

    # Use all_gather_object to collect each process's list[dict]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)

    if local_rank == 0:
        # Merge the results from all processes
        all_results = []
        for rank_results in gathered:
            all_results.extend(rank_results)
        return all_results
    else:
        return []  # Non-rank-0 processes do not need further processing


def deduplicate_results(
    results: list[dict],
    dataset_size: int,
) -> list[dict]:
    """
    Remove duplicate samples caused by DistributedSampler padding.

    DistributedSampler(drop_last=False) pads up to an integer multiple of
    world_size, and the padded samples are duplicated from the beginning. Use
    (scene_id, traj_a_id, cam_a_id, traj_b_id, cam_b_id, window_rank) as the
    unique key for deduplication.

    Args:
        results: the full results after gathering
        dataset_size: the actual dataset size

    Returns:
        the deduplicated results (at most dataset_size entries)
    """
    if len(results) <= dataset_size:
        return results

    seen = set()
    deduped = []
    for r in results:
        key = (
            r["scene_id"], r["traj_a_id"], r["cam_a_id"],
            r["traj_b_id"], r["cam_b_id"], r["window_rank"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


# ============================================================================
# Raw result saving / loading
# ============================================================================
CSV_COLUMNS = [
    "scene_id", "traj_a_id", "cam_a_id", "traj_b_id", "cam_b_id",
    "window_rank", "window_overlap_score", "rot_error_deg", "trans_error_m",
    "rot_error_a_mean_deg", "trans_error_a_mean_m",
    "rot_error_b_mean_deg", "trans_error_b_mean_m",
]


def save_raw_results(results: list[dict], output_dir: str) -> None:
    """Save raw results to CSV (scalars) and JSON (including 4x4 matrices)."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV: scalar fields only
    csv_path = os.path.join(output_dir, "raw_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in CSV_COLUMNS}
            writer.writerow(row)
    print(f"[Eval] Saved {csv_path} ({len(results)} rows)")

    # JSON: includes 4x4 matrices
    json_path = os.path.join(output_dir, "raw_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Saved {json_path}")


def load_results_from_csv(output_dir: str) -> list[dict]:
    """Load raw results from CSV (--skip-inference mode)."""
    csv_path = os.path.join(output_dir, "raw_results.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"raw_results.csv not found: {csv_path}")

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "scene_id": row["scene_id"],
                "traj_a_id": row["traj_a_id"],
                "cam_a_id": int(row["cam_a_id"]),
                "traj_b_id": row["traj_b_id"],
                "cam_b_id": int(row["cam_b_id"]),
                "window_rank": int(row["window_rank"]),
                "window_overlap_score": float(row["window_overlap_score"]),
                "rot_error_deg": float(row["rot_error_deg"]),
                "trans_error_m": float(row["trans_error_m"]),
                "rot_error_a_mean_deg": float(row.get("rot_error_a_mean_deg", "nan")),
                "trans_error_a_mean_m": float(row.get("trans_error_a_mean_m", "nan")),
                "rot_error_b_mean_deg": float(row.get("rot_error_b_mean_deg", "nan")),
                "trans_error_b_mean_m": float(row.get("trans_error_b_mean_m", "nan")),
            })

    print(f"[Eval] Loaded {len(results)} records from {csv_path}")
    return results


# ============================================================================
# Binned statistics
# ============================================================================
def compute_binned_statistics(
    results: list[dict],
) -> list[dict]:
    """
    Bin rotation/translation errors by overlap interval and compute statistics.

    Returns:
        list[dict]: one record per bin, including count, rot/trans mean/median/std, success rate
    """
    binned = []

    for low, high in OVERLAP_BINS:
        # Select the samples that fall into the interval
        samples = [
            r for r in results
            if low <= r["window_overlap_score"] < high
        ]
        count = len(samples)

        if count == 0:
            binned.append({
                "bin_low": low, "bin_high": high,
                "bin_label": _bin_label(low, high),
                "count": 0,
                "rot_mean": float("nan"), "rot_median": float("nan"), "rot_std": float("nan"),
                "trans_mean": float("nan"), "trans_median": float("nan"), "trans_std": float("nan"),
                "success_rate_pct": float("nan"),
            })
            continue

        rot_errs = np.array([s["rot_error_deg"] for s in samples])
        trans_errs = np.array([s["trans_error_m"] for s in samples])

        # Success: rotation < 10° and translation < 0.5m
        success_mask = (rot_errs < SUCCESS_ROT_THRESH_DEG) & (trans_errs < SUCCESS_TRANS_THRESH_M)
        success_rate = float(success_mask.sum()) / count * 100.0

        binned.append({
            "bin_low": low, "bin_high": high,
            "bin_label": _bin_label(low, high),
            "count": count,
            "rot_mean": float(np.mean(rot_errs)),
            "rot_median": float(np.median(rot_errs)),
            "rot_std": float(np.std(rot_errs)),
            "trans_mean": float(np.mean(trans_errs)),
            "trans_median": float(np.median(trans_errs)),
            "trans_std": float(np.std(trans_errs)),
            "success_rate_pct": success_rate,
        })

    return binned


def save_binned_summary(binned: list[dict], output_dir: str) -> None:
    """Save binned statistics to CSV."""
    csv_path = os.path.join(output_dir, "binned_summary.csv")
    fieldnames = [
        "bin_label", "count",
        "rot_mean", "rot_median", "rot_std",
        "trans_mean", "trans_median", "trans_std",
        "success_rate_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in binned:
            writer.writerow(row)
    print(f"[Eval] Saved {csv_path}")


# ============================================================================
# Visualization
# ============================================================================
_PLT = None

def _setup_matplotlib():
    """Configure the matplotlib backend and fonts (initialized only on the first call)."""
    global _PLT
    if _PLT is not None:
        return _PLT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })
    _PLT = plt
    return plt


def plot_error_bars(binned: list[dict], output_dir: str) -> None:
    """
    2x2 bar chart: sample count, median rotation error, median translation error, success rate.
    All text is in English.
    """
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = [b["bin_label"] for b in binned]
    counts = [b["count"] for b in binned]
    x = np.arange(len(labels))

    # (0,0) sample count
    ax = axes[0, 0]
    bars = ax.bar(x, counts, color="#4C72B0", edgecolor="white")
    ax.set_title("Sample Count per Overlap Bin")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    # Annotate the number on top of each bar
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(c), ha="center", va="bottom", fontsize=8)

    # (0,1) median rotation error
    ax = axes[0, 1]
    rot_medians = [b["rot_median"] if not np.isnan(b["rot_median"]) else 0 for b in binned]
    rot_stds = [b["rot_std"] if not np.isnan(b["rot_std"]) else 0 for b in binned]
    ax.bar(x, rot_medians, color="#DD8452", edgecolor="white",
           yerr=rot_stds, capsize=3, error_kw={"linewidth": 0.8})
    ax.set_title("Median Rotation Error (deg)")
    ax.set_ylabel("Rotation Error (deg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=SUCCESS_ROT_THRESH_DEG, color="red", linestyle="--",
               linewidth=0.8, label=f"Threshold = {SUCCESS_ROT_THRESH_DEG}deg")
    ax.legend(fontsize=8)

    # (1,0) median translation error
    ax = axes[1, 0]
    trans_medians = [b["trans_median"] if not np.isnan(b["trans_median"]) else 0 for b in binned]
    trans_stds = [b["trans_std"] if not np.isnan(b["trans_std"]) else 0 for b in binned]
    ax.bar(x, trans_medians, color="#55A868", edgecolor="white",
           yerr=trans_stds, capsize=3, error_kw={"linewidth": 0.8})
    ax.set_title("Median Translation Error (m)")
    ax.set_ylabel("Translation Error (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=SUCCESS_TRANS_THRESH_M, color="red", linestyle="--",
               linewidth=0.8, label=f"Threshold = {SUCCESS_TRANS_THRESH_M}m")
    ax.legend(fontsize=8)

    # (1,1) success rate
    ax = axes[1, 1]
    success_rates = [b["success_rate_pct"] if not np.isnan(b["success_rate_pct"]) else 0 for b in binned]
    bars = ax.bar(x, success_rates, color="#C44E52", edgecolor="white")
    ax.set_title(f"Success Rate (Rot<{SUCCESS_ROT_THRESH_DEG}deg & Trans<{SUCCESS_TRANS_THRESH_M}m)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for bar, sr in zip(bars, success_rates):
        if sr > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{sr:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("G2G Pose Estimation: Error by Overlap Bin", fontsize=14, y=1.01)
    fig.tight_layout()

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "error_by_overlap_bars.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {save_path}")


def plot_error_trend(binned: list[dict], output_dir: str) -> None:
    """
    2-panel trend plot: rot/trans error vs overlap midpoint.
    All text is in English.
    """
    plt = _setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Only plot bins that have data
    valid = [(b, (b["bin_low"] + b["bin_high"]) / 2) for b in binned if b["count"] > 0]
    if not valid:
        plt.close(fig)
        return

    bins_v, midpoints = zip(*valid)
    midpoints = list(midpoints)

    # Rotation error trend
    rot_medians = [b["rot_median"] for b in bins_v]
    rot_means = [b["rot_mean"] for b in bins_v]
    ax1.plot(midpoints, rot_medians, "o-", color="#DD8452", label="Median", markersize=6)
    ax1.plot(midpoints, rot_means, "s--", color="#DD8452", alpha=0.5, label="Mean", markersize=4)
    ax1.fill_between(
        midpoints,
        [b["rot_mean"] - b["rot_std"] for b in bins_v],
        [b["rot_mean"] + b["rot_std"] for b in bins_v],
        alpha=0.15, color="#DD8452", label="Mean +/- Std",
    )
    ax1.axhline(y=SUCCESS_ROT_THRESH_DEG, color="red", linestyle="--",
                linewidth=0.8, label=f"Threshold = {SUCCESS_ROT_THRESH_DEG}deg")
    ax1.set_xlabel("Overlap Score")
    ax1.set_ylabel("Rotation Error (deg)")
    ax1.set_title("Rotation Error vs Overlap")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Translation error trend
    trans_medians = [b["trans_median"] for b in bins_v]
    trans_means = [b["trans_mean"] for b in bins_v]
    ax2.plot(midpoints, trans_medians, "o-", color="#55A868", label="Median", markersize=6)
    ax2.plot(midpoints, trans_means, "s--", color="#55A868", alpha=0.5, label="Mean", markersize=4)
    ax2.fill_between(
        midpoints,
        [b["trans_mean"] - b["trans_std"] for b in bins_v],
        [b["trans_mean"] + b["trans_std"] for b in bins_v],
        alpha=0.15, color="#55A868", label="Mean +/- Std",
    )
    ax2.axhline(y=SUCCESS_TRANS_THRESH_M, color="red", linestyle="--",
                linewidth=0.8, label=f"Threshold = {SUCCESS_TRANS_THRESH_M}m")
    ax2.set_xlabel("Overlap Score")
    ax2.set_ylabel("Translation Error (m)")
    ax2.set_title("Translation Error vs Overlap")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("G2G: Error Trend by Overlap", fontsize=14, y=1.02)
    fig.tight_layout()

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "error_by_overlap_trend.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved {save_path}")


# ============================================================================
# Summary output
# ============================================================================
def print_and_save_summary(
    results: list[dict],
    binned: list[dict],
    output_dir: str,
    elapsed_sec: float,
) -> None:
    """Print the binned summary to the terminal and save overall_summary.txt."""
    if not results:
        print("[Eval] No results to summarize.")
        return

    hdr = (f"{'Overlap':<14} | {'Total':>5} | {'Done':>5} | {'NoWin':>5} | "
           f"{'t_mean(m)':>9} | {'t_med(m)':>8} | "
           f"{'r_mean(°)':>9} | {'r_med(°)':>8}")
    sep = "-" * len(hdr)

    lines = [sep, hdr, sep]

    for b in binned:
        n_total = b["count"]
        n_done = n_total
        n_nowin = 0

        if n_total > 0:
            t_avg = f"{b['trans_mean']:.3f}"
            t_med = f"{b['trans_median']:.3f}"
            r_avg = f"{b['rot_mean']:.1f}"
            r_med = f"{b['rot_median']:.1f}"
        else:
            t_avg = t_med = r_avg = r_med = "-"

        lines.append(
            f"{b['bin_label']:<14} | {n_total:>5} | {n_done:>5} | "
            f"{n_nowin:>5} | "
            f"{t_avg:>9} | {t_med:>8} | "
            f"{r_avg:>9} | {r_med:>8}"
        )

    lines.append(sep)

    rot_errs = np.array([r["rot_error_deg"] for r in results])
    trans_errs = np.array([r["trans_error_m"] for r in results])

    t_avg_all = f"{np.mean(trans_errs):.3f}"
    t_med_all = f"{np.median(trans_errs):.3f}"
    r_avg_all = f"{np.mean(rot_errs):.1f}"
    r_med_all = f"{np.median(rot_errs):.1f}"

    n_all = len(results)
    lines.append(
        f"{'OVERALL':<14} | {n_all:>5} | {n_all:>5} | "
        f"{0:>5} | "
        f"{t_avg_all:>9} | {t_med_all:>8} | "
        f"{r_avg_all:>9} | {r_med_all:>8}"
    )
    lines.append(sep)

    # Additional multi-frame metrics (if present)
    rot_a = np.array([r.get("rot_error_a_mean_deg", np.nan) for r in results], dtype=np.float64)
    trans_a = np.array([r.get("trans_error_a_mean_m", np.nan) for r in results], dtype=np.float64)
    rot_b = np.array([r.get("rot_error_b_mean_deg", np.nan) for r in results], dtype=np.float64)
    trans_b = np.array([r.get("trans_error_b_mean_m", np.nan) for r in results], dtype=np.float64)
    has_multiframe = (
        np.isfinite(rot_a).any()
        and np.isfinite(trans_a).any()
        and np.isfinite(rot_b).any()
        and np.isfinite(trans_b).any()
    )
    if has_multiframe:
        lines.extend([
            "",
            "Multi-frame overall metrics:",
            f"  A-intra mean:   rot={np.nanmean(rot_a):.2f} deg, trans={np.nanmean(trans_a):.3f} m",
            f"  A-intra median: rot={np.nanmedian(rot_a):.2f} deg, trans={np.nanmedian(trans_a):.3f} m",
            f"  B-all mean:     rot={np.nanmean(rot_b):.2f} deg, trans={np.nanmean(trans_b):.3f} m",
            f"  B-all median:   rot={np.nanmedian(rot_b):.2f} deg, trans={np.nanmedian(trans_b):.3f} m",
        ])

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n[Eval] Saved {summary_path}")
    print(f"[Eval] Elapsed time: {elapsed_sec:.1f}s")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # DDP initialization (multi-GPU automatically when launched via torchrun; falls back to single-GPU when launched directly with python)
    local_rank, world_size = setup_ddp()
    is_main = is_main_process(local_rank)

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    cfg = load_yaml(args.config)
    model_type = infer_model_type(cfg, args.model_type)

    # Determine the output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(args.checkpoint), "eval_results",
        )
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Eval] Output dir: {output_dir}")

    # Device: assigned by local_rank in multi-GPU mode, uses --device in single-GPU mode
    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Extrinsics noise config: read from config when --noise is enabled
    ext_noise_cfg = None
    if args.noise:
        ext_noise_cfg = cfg.get("data", {}).get("extrinsics_noise", {})
        if not ext_noise_cfg.get("enabled", False):
            if is_main:
                print("[Eval] WARNING: --noise was specified, but data.extrinsics_noise.enabled=false or missing in the config, so noise will not take effect")

    if is_main:
        print(f"[Eval] Device: {device}, world_size: {world_size}")
        print(f"[Eval] Model type: {model_type}")
        if ext_noise_cfg and ext_noise_cfg.get("enabled", False):
            print(
                f"[Eval] Extrinsics noise ENABLED: "
                f"rot_std={ext_noise_cfg.get('rotation_noise_std_deg', 1.0)}deg, "
                f"trans_std={ext_noise_cfg.get('translation_noise_std_m', 0.05)}m"
            )

    t_start = time.time()

    if args.skip_inference:
        # --skip-inference: reload from an existing CSV (rank 0 only)
        if is_main:
            results = load_results_from_csv(output_dir)
            before = len(results)
            results = [
                r for r in results
                if r["window_overlap_score"] >= args.min_overlap
                and r["window_overlap_score"] < args.max_overlap
            ]
            if len(results) < before:
                print(
                    f"[Eval] Filtered to {len(results)} samples "
                    f"(overlap in [{args.min_overlap}, {args.max_overlap}))"
                )
        else:
            results = []
    else:
        # ===== Build the model (multi-GPU: rank 0 loads first to avoid I/O congestion) =====
        if local_rank <= 0:
            if is_main:
                print(f"\n[Rank {local_rank}] Loading model...")
            model = build_and_load_model(
                cfg, args.checkpoint, device, model_type=model_type, is_main=is_main,
            )
            if is_main:
                print(f"[Rank {local_rank}] Model loaded.")

        if local_rank != -1:
            dist.barrier()

        if local_rank > 0:
            print(f"[Rank {local_rank}] Loading model...")
            model = build_and_load_model(
                cfg, args.checkpoint, device, model_type=model_type, is_main=False,
            )
            print(f"[Rank {local_rank}] Model loaded.")

        if local_rank != -1:
            dist.barrier()

        # ===== Build the data loader =====
        data_cfg = cfg.get("data", {})
        num_workers = args.num_workers if args.num_workers >= 0 else data_cfg.get("num_workers", 4)

        # Handle the shuffle_within_group CLI override logic
        shuffle_override = None
        if args.no_shuffle_within_group:
            shuffle_override = False
        elif args.shuffle_within_group:
            shuffle_override = True

        loader, dataset = build_val_loader(
            cfg,
            min_overlap=args.min_overlap,
            batch_size=args.batch_size,
            num_workers=num_workers,
            top_k=args.top_k,
            model_type=model_type,
            local_rank=local_rank,
            world_size=world_size,
            extrinsics_noise_cfg=ext_noise_cfg,
            anchor_frame_idx=args.anchor_frame_idx,
            shuffle_within_group=shuffle_override,
        )
        dataset_size = len(dataset)

        # ===== Inference =====
        try:
            local_results = run_inference(
                model, loader, device, model_type=model_type, is_main=is_main,
            )

            # Multi-GPU gather
            results = gather_results(local_results, local_rank, world_size)

            # rank 0: deduplicate (DistributedSampler padding) + max_overlap filtering + save
            if is_main:
                results = deduplicate_results(results, dataset_size)
                print(f"[Eval] Gathered {len(results)} unique results (dataset: {dataset_size})")

                # Post max_overlap filtering
                if args.max_overlap < 1.01:
                    before = len(results)
                    results = [
                        r for r in results
                        if r["window_overlap_score"] < args.max_overlap
                    ]
                    print(
                        f"[Eval] max_overlap filter: {before} -> {len(results)} "
                        f"(score < {args.max_overlap})"
                    )

                save_raw_results(results, output_dir)
        finally:
            del model
            torch.cuda.empty_cache()

    t_inference_done = time.time()

    # ===== Post-processing: rank 0 only =====
    if is_main:
        # Binned statistics
        binned = compute_binned_statistics(results)
        save_binned_summary(binned, output_dir)

        # Visualization
        plot_error_bars(binned, output_dir)
        plot_error_trend(binned, output_dir)

        # Summary
        elapsed = t_inference_done - t_start
        print_and_save_summary(results, binned, output_dir, elapsed)

        print(f"\n[Eval] Total time: {time.time() - t_start:.1f}s")
        print(f"[Eval] All results saved to: {output_dir}")

    # Clean up DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()

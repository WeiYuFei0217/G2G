#!/usr/bin/env python3
"""
train_reloc.py -- G2G multi-frame relocalization training script

Key features:
  - Uses Stage2ModelMultiFrame (merged self-attention bridge + multi-frame pose head; the alternating inter/intra Phase 1 is disabled in the released configs via bridge_alternating_pairs: 0)
  - Uses HM3DStage2MultiFrameDataset (additionally returns clean extrinsics)
  - Uses MultiFrameG2GLoss (supervises A1-A4 + B0-B4, 9 poses in total)
  - Logging adds intra-group/inter-group error statistics

Usage:
    # Overfit test
    torchrun --nproc_per_node=4 --master-port 29589 \
        scripts/train_reloc.py \
        --config configs/reloc/hm3d.yaml --overfit

    # Adaptive curriculum learning training
    torchrun --nproc_per_node=4 --master-port 29589 \
        scripts/train_reloc.py \
        --config configs/reloc/hm3d.yaml --curriculum
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# Dependencies: the g2g package installed via `pip install -e .` in this repo,
# plus the already-installed mapanything.
# ============================================================================
from g2g.datasets.reloc_dataset import (
    HM3DStage2MultiFrameDataset,
    hm3d_stage2_collate_fn,
)
from g2g.losses.pose_loss_multiframe import (
    MultiFrameG2GLoss,
    compute_multiframe_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================
# YAML config loading
# ============================================================================
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ============================================================================
# DDP utility functions
# ============================================================================
def setup_ddp(timeout_minutes: int = 60) -> tuple[int, int]:
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


def is_main_process(local_rank: int) -> bool:
    return local_rank <= 0


def sync_all(local_rank: int):
    if local_rank != -1:
        dist.barrier()


# ============================================================================
# Adaptive Curriculum Learning
# ============================================================================
class ConvergenceDetector:
    def __init__(self, window_size: int = 1000, rot_thresh: float = 0.1, trans_thresh: float = 0.3):
        self.window_size = window_size
        self.rot_thresh = rot_thresh
        self.trans_thresh = trans_thresh
        self.rot_losses: deque[float] = deque(maxlen=window_size)
        self.trans_losses: deque[float] = deque(maxlen=window_size)

    def update(self, rot_loss: float, trans_loss: float) -> bool:
        self.rot_losses.append(rot_loss)
        self.trans_losses.append(trans_loss)
        if len(self.rot_losses) < self.window_size:
            return False
        mean_rot = sum(self.rot_losses) / len(self.rot_losses)
        mean_trans = sum(self.trans_losses) / len(self.trans_losses)
        return mean_rot < self.rot_thresh and mean_trans < self.trans_thresh

    def reset(self):
        self.rot_losses.clear()
        self.trans_losses.clear()


# ============================================================================
# Sanity Check (multi-frame version)
# ============================================================================
def run_sanity_check(
    model: nn.Module,
    criterion: MultiFrameG2GLoss,
    batch: dict,
    device: torch.device,
    verbose: bool = True,
) -> None:
    def _print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    _print("\n" + "=" * 60)
    _print("SANITY CHECK (MultiFrame)")
    _print("=" * 60)

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.to(device)
        else:
            batch_dev[k] = v

    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    pred = model(batch_dev)

    losses = criterion(pred, batch_dev)

    B = batch_dev["images_a"].shape[0]
    W = batch_dev["images_a"].shape[1]

    # Check multi-frame output shapes
    assert pred["rotations_a"].shape == (B, W - 1, 6), (
        f"rotations_a shape {pred['rotations_a'].shape}, expected ({B}, {W-1}, 6)")
    assert pred["translations_a"].shape == (B, W - 1, 3), (
        f"translations_a shape {pred['translations_a'].shape}")
    assert pred["rotations_b"].shape == (B, W, 6), (
        f"rotations_b shape {pred['rotations_b'].shape}")
    assert pred["translations_b"].shape == (B, W, 3), (
        f"translations_b shape {pred['translations_b'].shape}")
    # Backward compatibility
    assert pred["T_rel"].shape == (B, 4, 4)
    assert pred["rotation"].shape == (B, 6)
    assert pred["translation"].shape == (B, 3)

    _print(f"  [OK] MultiFrame outputs: rot_a={pred['rotations_a'].shape}, "
           f"rot_b={pred['rotations_b'].shape}")
    _print(f"  [OK] Backward compat: T_rel={pred['T_rel'].shape}")

    # Check that loss is finite
    assert torch.isfinite(losses["loss"]), f"loss is not finite: {losses['loss']}"
    _print(f"  [OK] Loss: {losses['loss'].item():.4f}")
    _print(f"    Intra: rot={losses['rot_loss_intra'].item():.4f}, "
           f"trans={losses['trans_loss_intra'].item():.4f}")
    _print(f"    Inter: rot={losses['rot_loss_inter'].item():.4f}, "
           f"trans={losses['trans_loss_inter'].item():.4f}")

    # Backward
    losses["loss"].backward()

    # Check backbone gradients
    backbone_grad_norm = 0.0
    for p in model.backbone.parameters():
        if p.grad is not None:
            backbone_grad_norm += p.grad.norm().item()
    assert backbone_grad_norm == 0.0
    _print(f"  [OK] Backbone gradient norm: {backbone_grad_norm}")

    trainable_params = model.get_trainable_parameters()
    has_grad = sum(1 for p in trainable_params if p.grad is not None)
    total_trainable = len(trainable_params)
    assert has_grad > 0
    _print(f"  [OK] Trainable params with grad: {has_grad}/{total_trainable}")

    # Check that clean extrinsics are present
    assert "extrinsics_a_clean" in batch_dev, "Missing extrinsics_a_clean"
    assert "extrinsics_b_clean" in batch_dev, "Missing extrinsics_b_clean"
    _print(f"  [OK] Clean extrinsics present")

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    _print(f"  [OK] GPU peak memory: {peak_mb:.0f} MB")

    model.zero_grad(set_to_none=True)

    metrics = compute_multiframe_metrics(pred, batch_dev)
    _print(f"  [INFO] B0→A0: rot={metrics['rot_error_deg_mean']:.1f}°, "
           f"trans={metrics['trans_error_m_mean']:.3f}m")
    _print(f"  [INFO] A-intra: rot={metrics['rot_error_a_mean']:.1f}°, "
           f"trans={metrics['trans_error_a_mean']:.3f}m")
    _print(f"  [INFO] B-inter: rot={metrics['rot_error_b_mean']:.1f}°, "
           f"trans={metrics['trans_error_b_mean']:.3f}m")

    _print("SANITY CHECK PASSED")
    _print("=" * 60 + "\n")


# ============================================================================
# Validation function (multi-frame version)
# ============================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: MultiFrameG2GLoss,
    device: torch.device,
    local_rank: int,
) -> dict[str, float]:
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    total_loss = 0.0
    total_rot_err = 0.0
    total_trans_err = 0.0
    total_rot_err_a = 0.0
    total_trans_err_a = 0.0
    total_rot_err_b = 0.0
    total_trans_err_b = 0.0
    num_batches = 0

    with torch.no_grad():
      for batch in val_loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        pred = eval_model(batch_dev)
        losses = criterion(pred, batch_dev)
        metrics = compute_multiframe_metrics(pred, batch_dev)

        total_loss += losses["loss"].item()
        total_rot_err += metrics["rot_error_deg_mean"]
        total_trans_err += metrics["trans_error_m_mean"]
        total_rot_err_a += metrics["rot_error_a_mean"]
        total_trans_err_a += metrics["trans_error_a_mean"]
        total_rot_err_b += metrics["rot_error_b_mean"]
        total_trans_err_b += metrics["trans_error_b_mean"]
        num_batches += 1

    if local_rank != -1:
        stats = torch.tensor(
            [total_loss, total_rot_err, total_trans_err,
             total_rot_err_a, total_trans_err_a,
             total_rot_err_b, total_trans_err_b, num_batches],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        (total_loss, total_rot_err, total_trans_err,
         total_rot_err_a, total_trans_err_a,
         total_rot_err_b, total_trans_err_b, num_batches) = stats.tolist()

    n = max(num_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_rot_error_deg": total_rot_err / n,
        "val_trans_error_m": total_trans_err / n,
        "val_rot_error_a_deg": total_rot_err_a / n,
        "val_trans_error_a_m": total_trans_err_a / n,
        "val_rot_error_b_deg": total_rot_err_b / n,
        "val_trans_error_b_m": total_trans_err_b / n,
    }


# ============================================================================
# Main training function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="G2G MultiFrame Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--nccl-timeout", type=int, default=180)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run a single validation-set evaluation and exit, for testing zero-shot generalization")
    args = parser.parse_args()

    if args.resume and args.finetune:
        parser.error("--resume and --finetune cannot be used together")

    # ========== Load config ==========
    cfg = load_yaml(args.config)

    if args.overfit and "overfit" in cfg:
        cfg.setdefault("training", {})
        deep_update(cfg["training"], cfg["overfit"])

    train_cfg = cfg.get("training", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir

    # ========== Initialize DDP ==========
    print("[main] Initializing DDP...", flush=True)
    local_rank, world_size = setup_ddp(timeout_minutes=args.nccl_timeout)
    is_main = is_main_process(local_rank)
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")
    print(f"[Rank {local_rank}] DDP initialized. Device: {device}", flush=True)

    # Deferred import
    from g2g.models.stage2_model_multiframe import Stage2ModelMultiFrame

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = train_cfg.get("output_dir", "./outputs/stage2_multiframe")
    if args.overfit:
        output_dir = output_dir.rstrip("/") + "_overfit"
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.resume or args.finetune:
        ckpt_path = args.resume or args.finetune
        output_dir = str(Path(ckpt_path).resolve().parent)
    else:
        from datetime import datetime
        output_dir = output_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    if is_main:
        print("=" * 60)
        print("G2G MultiFrame Training")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Overfit mode: {args.overfit}")
        print(f"Device: {device}, World size: {world_size}")
        print(f"Output: {output_dir}")
        print(f"Batch size: {train_cfg.get('batch_size', 2)} x {world_size} GPUs "
              f"x {train_cfg.get('gradient_accumulation', 1)} accum "
              f"= {train_cfg.get('batch_size', 2) * world_size * train_cfg.get('gradient_accumulation', 1)} effective")
        print(f"LR: {train_cfg.get('lr', 3e-4)}")
        print(f"Epochs: {train_cfg.get('epochs', 50)}")
        print(f"Loss: {train_cfg.get('loss_type', 'chordal')}")
        print(f"Intra weight: {train_cfg.get('intra_pose_weight', 0.3)}")
        print(f"Inter weight: {train_cfg.get('inter_pose_weight', 1.0)}")
        print("=" * 60)

    # ========== Create datasets ==========
    data_cfg = cfg.get("data", {})
    max_scenes = args.max_scenes
    if args.overfit and max_scenes is None:
        max_scenes = cfg.get("overfit", {}).get("max_scenes", 3)

    # anchor_frame_idx: controls which frame serves as the intra-group anchor frame (origin)
    # Default 0 = first frame of the window; set to 2 = middle frame of the window (reorders frame order at the dataset level)
    anchor_frame_idx = data_cfg.get("anchor_frame_idx", 0)

    # shuffle_within_group: randomly shuffle frames within a group
    shuffle_within_group = data_cfg.get("shuffle_within_group", False)

    # effective_window_size: select the center N frames out of window_size frames (ablation: 3x3 window)
    effective_window_size = data_cfg.get("effective_window_size", None)

    shared_kwargs = dict(
        img_size=tuple(data_cfg.get("img_size", [224, 224])),
        window_size=data_cfg.get("window_size", 5),
        top_k=data_cfg.get("top_k", 10),
        min_overlap_score=data_cfg.get("min_overlap_score", 0.1),
        anchor_frame_idx=anchor_frame_idx,
        shuffle_within_group=shuffle_within_group,
        effective_window_size=effective_window_size,
    )

    ext_noise_cfg = data_cfg.get("extrinsics_noise", {})

    if is_main:
        print("\n[Step 1] Loading training dataset (MultiFrame)...")
    train_data_cfg = data_cfg.get("train", {})
    train_dataset = HM3DStage2MultiFrameDataset(
        step1_root=train_data_cfg["step1_root"],
        step2_root=train_data_cfg["step2_root"],
        step3_root=train_data_cfg["step3_root"],
        stage2_index_root=train_data_cfg["stage2_index_root"],
        max_scenes=max_scenes if max_scenes else -1,
        seed=seed,
        step6_root=train_data_cfg.get("step6_root", ""),
        extrinsics_noise_cfg=ext_noise_cfg,
        **shared_kwargs,
    )
    if is_main:
        print(f"  Training: {len(train_dataset)} samples (full: {train_dataset.get_full_size()})")
        print(f"  {train_dataset}")
        if shuffle_within_group:
            print(f"  [SHUFFLE] shuffle_within_group=True "
                  f"(frame order randomized within each group, anchor_frame_idx ignored)")
        elif anchor_frame_idx != 0:
            print(f"  [ANCHOR] anchor_frame_idx={anchor_frame_idx} "
                  f"(window mid-frame as anchor, reordered to index 0)")
        if effective_window_size is not None:
            print(f"  [WINDOW] effective_window_size={effective_window_size} "
                  f"(center {effective_window_size} frames from {data_cfg.get('window_size', 5)}-frame window)")
        if ext_noise_cfg.get("enabled", False):
            print(f"  [NOISE] Extrinsics noise: rot_std={ext_noise_cfg.get('rotation_noise_std_deg', 1.0)}°, "
                  f"trans_std={ext_noise_cfg.get('translation_noise_std_m', 0.05)}m")

    # ========== Adaptive curriculum learning initialization ==========
    curriculum_enabled = args.curriculum

    if curriculum_enabled:
        initial_threshold = 0.5
        n_init = train_dataset.set_curriculum_overlap(initial_threshold)
        if is_main:
            print(f"\n[Adaptive Curriculum] Enabled")
            print(f"  Initial threshold: {initial_threshold:.2f}, active: {n_init}/{train_dataset.get_full_size()}")

    # Validation set
    val_loader = None
    val_data_cfg = data_cfg.get("val", {})
    if val_data_cfg:
        if is_main:
            print("[Step 1.1] Loading validation dataset (MultiFrame)...")
        val_max = max_scenes if (args.overfit and max_scenes) else -1
        val_dataset = HM3DStage2MultiFrameDataset(
            step1_root=val_data_cfg["step1_root"],
            step2_root=val_data_cfg["step2_root"],
            step3_root=val_data_cfg["step3_root"],
            stage2_index_root=val_data_cfg["stage2_index_root"],
            max_scenes=val_max,
            seed=seed,
            step6_root=val_data_cfg.get("step6_root", ""),
            max_samples_per_scene=data_cfg.get("val_max_samples_per_scene", -1),
            **shared_kwargs,
        )
        if is_main:
            print(f"  Validation: {len(val_dataset)} samples")

        val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 2),
            sampler=val_sampler,
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=hm3d_stage2_collate_fn,
            pin_memory=True,
        )

    sync_all(local_rank)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if local_rank != -1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 2),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=hm3d_stage2_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ========== Create model ==========
    if is_main:
        print("\n[Step 2] Building model (MultiFrame)...")

    backbone_cfg = cfg.get("backbone", {})
    model_cfg = cfg.get("model", {})

    model_kwargs = dict(
        model_path=backbone_cfg["model_path"],
        embed_dim=model_cfg.get("embed_dim", 768),
        num_latents=model_cfg.get("num_latents", 32),
        num_frames_per_group=model_cfg.get("num_frames_per_group", 5),
        resampler_layers=model_cfg.get("resampler_layers", 2),
        bridge_alternating_pairs=model_cfg.get("bridge_alternating_pairs", 2),
        bridge_merged_layers=model_cfg.get("bridge_merged_layers", 2),
        reinject_anchor_after_merge=model_cfg.get("reinject_anchor_after_merge", True),
        use_anchor_embed=model_cfg.get("use_anchor_embed", True),
        pose_head_hidden_dim=model_cfg.get("pose_head_hidden_dim", 512),
        pose_head_layers=model_cfg.get("pose_head_layers", 3),
        num_heads=model_cfg.get("num_heads", 8),
        rotation_repr=model_cfg.get("rotation_repr", "6d"),
        freeze_backbone=backbone_cfg.get("freeze", True),
        disable_extrinsics_group_a=model_cfg.get("disable_extrinsics_group_a", False),
        disable_extrinsics_group_b=model_cfg.get("disable_extrinsics_group_b", False),
    )

    if local_rank <= 0:
        print(f"[Rank {local_rank}] Loading model...", flush=True)
        model = Stage2ModelMultiFrame(**model_kwargs).to(device)
        print(f"[Rank {local_rank}] Model loaded.", flush=True)
    sync_all(local_rank)
    if local_rank > 0:
        print(f"[Rank {local_rank}] Loading model...", flush=True)
        model = Stage2ModelMultiFrame(**model_kwargs).to(device)
        print(f"[Rank {local_rank}] Model loaded.", flush=True)
    sync_all(local_rank)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if hasattr(model, "module") else model

    # Profiling
    profiling_enabled = args.overfit and is_main
    profiling_warmup_steps = 3
    profiling_warmup_done = False
    if profiling_enabled:
        raw_model.enable_profiling(True)
        _train_timings: dict[str, list] = defaultdict(list)

    # ========== Optimizer & scheduler ==========
    trainable_params = raw_model.get_trainable_parameters()
    optimizer = AdamW(
        trainable_params,
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    accum_steps = train_cfg.get("gradient_accumulation", 1)
    epochs = train_cfg.get("epochs", 50)
    batch_size_cfg = train_cfg.get("batch_size", 2)
    num_workers_cfg = data_cfg.get("num_workers", 4)

    warmup_steps = train_cfg.get("warmup_steps", 1000)
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    lr_step_interval = train_cfg.get("lr_step_interval", 50)

    curriculum_cfg = train_cfg.get("curriculum", {})
    curriculum_thresholds = curriculum_cfg.get(
        "thresholds", [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1],
    )
    curriculum_phase2_end = curriculum_cfg.get("phase2_end", 0.15)

    convergence_cfg = curriculum_cfg.get("convergence", {})
    convergence_detector = ConvergenceDetector(
        window_size=convergence_cfg.get("window_size", 1000),
        rot_thresh=convergence_cfg.get("rot_thresh", 0.1),
        trans_thresh=convergence_cfg.get("trans_thresh", 0.3),
    ) if curriculum_enabled else None

    # Balanced sampling configuration
    balanced_cfg = curriculum_cfg.get("balanced_sampling", {})
    balanced_enabled = balanced_cfg.get("enabled", False) and curriculum_enabled
    if balanced_enabled:
        train_dataset.enable_balanced_sampling(
            bin_width=balanced_cfg.get("bin_width", 0.05),
            ref_bin=(balanced_cfg.get("ref_bin_lo", 0.35),
                     balanced_cfg.get("ref_bin_hi", 0.40)),
            hard_boost=balanced_cfg.get("hard_boost", 1.15),
            seed=seed,
            scene_balanced=balanced_cfg.get("scene_balanced", False),
        )

    current_threshold_idx = 0

    steps_per_epoch_phase3 = max(len(train_loader) // accum_steps, 1)
    total_steps_phase3 = steps_per_epoch_phase3 * epochs

    steps_per_epoch_nc = max(len(train_loader) // accum_steps, 1)
    total_steps_noncurriculum = steps_per_epoch_nc * epochs

    # LR schedule
    def lr_lambda(sched_step: int) -> float:
        current_step = sched_step * lr_step_interval
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        if not curriculum_enabled:
            decay_steps = current_step - warmup_steps
            total_decay = total_steps_noncurriculum - warmup_steps
            if total_decay <= 0:
                return 1.0
            progress = min(decay_steps / total_decay, 1.0)
            return max(0.5 * (1 + math.cos(math.pi * progress)), 0.0)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    phase3_scheduler = None

    # Loss function (multi-frame version)
    criterion = MultiFrameG2GLoss(
        rotation_weight=train_cfg.get("rotation_weight", 5.0),
        translation_weight=train_cfg.get("translation_weight", 1.0),
        intra_weight=train_cfg.get("intra_pose_weight", 0.3),
        inter_weight=train_cfg.get("inter_pose_weight", 1.0),
        loss_type=train_cfg.get("loss_type", "chordal"),
    )

    grad_clip = train_cfg.get("grad_clip", 5.0)

    if is_main:
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Warmup steps: {warmup_steps}, LR step interval: {lr_step_interval}")

    # ========== Resume training ==========
    start_epoch = 0
    best_val_loss = float("inf")
    _resume_cs = None

    if args.resume:
        if is_main:
            print(f"\n[Resume] Loading from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        if "model_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            _resume_cs = ckpt.get("curriculum_state")
            if is_main:
                print(f"  Resumed from epoch {start_epoch}")
        else:
            raw_model.load_state_dict(ckpt, strict=False)

    if args.finetune:
        if is_main:
            print(f"\n[Finetune] Loading weights from {args.finetune}...")
        ckpt = torch.load(args.finetune, map_location=device)
        if "model_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        else:
            raw_model.load_state_dict(ckpt, strict=False)
        _ft_cs = ckpt.get("curriculum_state") if "model_state_dict" in ckpt else None
        _finetune_start_step = 0
        if _ft_cs and "global_opt_step" in _ft_cs:
            _finetune_start_step = _ft_cs["global_opt_step"]
        if is_main:
            print(f"  [Finetune] Continuing from epoch {start_epoch}")

    # ========== Sanity Check ==========
    if is_main:
        print("\n[Step 3] Running sanity check (MultiFrame)...")
    sanity_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=0,
        collate_fn=hm3d_stage2_collate_fn,
    )
    sanity_batch = next(iter(sanity_loader))
    run_sanity_check(raw_model, criterion, sanity_batch, device, verbose=is_main)
    del sanity_loader, sanity_batch

    sync_all(local_rank)

    # ========== TensorBoard ==========
    writer = None
    if is_main:
        writer = SummaryWriter(os.path.join(output_dir, "logs"))

    # ========== Training loop ==========
    training_start_time = time.time()

    if is_main:
        print(f"\n[Step 4] Starting training from epoch {start_epoch}...\n")

    log_every = train_cfg.get("log_every", 10)
    val_every = train_cfg.get("val_every", 1)
    save_every = train_cfg.get("save_every", 5)
    # intra-epoch validation: run a validation + save a checkpoint every val_steps_interval optimizer steps
    # Suitable for very large datasets where a single epoch takes days to train
    val_steps_interval = train_cfg.get("val_steps_interval", 0)
    _last_intra_val_step = 0

    global_opt_step = 0
    if args.finetune and _finetune_start_step > 0:
        global_opt_step = _finetune_start_step
        _last_intra_val_step = global_opt_step

    def _get_curriculum_state() -> dict | None:
        if not curriculum_enabled:
            return None
        state = {
            "in_phase3": in_phase3,
            "phase3_start_epoch": phase3_start_epoch,
            "phase3_start_step": phase3_start_step,
            "current_threshold_idx": current_threshold_idx,
            "global_opt_step": global_opt_step,
            "total_steps_phase3": total_steps_phase3,
        }
        if phase3_scheduler is not None:
            state["phase3_scheduler_state_dict"] = phase3_scheduler.state_dict()
        if balanced_enabled:
            state["balanced_state"] = train_dataset.get_balanced_state()
        return state

    in_phase3 = False
    phase3_start_step = 0
    phase3_start_epoch = -1

    # Restore curriculum learning state
    if _resume_cs is not None and curriculum_enabled:
        in_phase3 = _resume_cs["in_phase3"]
        phase3_start_epoch = _resume_cs["phase3_start_epoch"]
        phase3_start_step = _resume_cs["phase3_start_step"]
        current_threshold_idx = _resume_cs["current_threshold_idx"]
        global_opt_step = _resume_cs["global_opt_step"]
        _last_intra_val_step = global_opt_step
        total_steps_phase3 = _resume_cs["total_steps_phase3"]

        if current_threshold_idx < len(curriculum_thresholds):
            resumed_threshold = curriculum_thresholds[current_threshold_idx]
            n_active = train_dataset.set_curriculum_overlap(resumed_threshold)
            if local_rank != -1:
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
            else:
                train_sampler = None
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size_cfg,
                sampler=train_sampler, shuffle=(train_sampler is None),
                num_workers=num_workers_cfg, collate_fn=hm3d_stage2_collate_fn,
                pin_memory=True, drop_last=True,
            )

        # If the threshold has already dropped below phase2_end but in_phase3=False (the phase2_end checkpoint case),
        # automatically enter Phase 3
        if not in_phase3 and curriculum_enabled:
            last_thr = curriculum_thresholds[current_threshold_idx] if current_threshold_idx < len(curriculum_thresholds) else 0.0
            if last_thr < curriculum_phase2_end - 0.001:
                in_phase3 = True
                phase3_start_step = global_opt_step
                phase3_start_epoch = start_epoch

                steps_per_epoch_phase3 = max(len(train_loader) // accum_steps, 1)
                total_steps_phase3 = steps_per_epoch_phase3 * epochs

                if is_main:
                    print(f"[Resume→Phase 3] threshold={last_thr:.2f} < phase2_end={curriculum_phase2_end:.2f}, "
                          f"auto-entering Phase 3: {epochs} epochs, {total_steps_phase3} steps")

                def phase3_lr_lambda_fresh(sched_step: int) -> float:
                    current_step = sched_step * lr_step_interval
                    progress = current_step / max(total_steps_phase3, 1)
                    return max(0.5 * (1 + math.cos(math.pi * progress)), 0.0)

                phase3_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, phase3_lr_lambda_fresh,
                )

        if in_phase3 and "phase3_scheduler_state_dict" in _resume_cs:
            _resumed_total_steps = total_steps_phase3

            def phase3_lr_lambda_resumed(sched_step: int) -> float:
                current_step = sched_step * lr_step_interval
                progress = current_step / max(_resumed_total_steps, 1)
                return max(0.5 * (1 + math.cos(math.pi * progress)), 0.0)

            phase3_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, phase3_lr_lambda_resumed,
            )
            phase3_scheduler.load_state_dict(_resume_cs["phase3_scheduler_state_dict"])

            _last_epoch = phase3_scheduler.last_epoch
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = phase3_scheduler.base_lrs[i] * phase3_lr_lambda_resumed(_last_epoch)

        # Restore balanced sampling state
        if balanced_enabled and "balanced_state" in _resume_cs:
            train_dataset.restore_balanced_state(_resume_cs["balanced_state"])

        del _resume_cs

    if args.finetune:
        end_epoch = start_epoch + epochs
    else:
        end_epoch = epochs

    # --eval-only: run validation once and then exit
    if args.eval_only:
        if local_rank <= 0:
            print("\n" + "=" * 60)
            print("[eval-only] Running validation-set evaluation...")
            print("=" * 60)
        val_results = validate(model, val_loader, criterion, device, local_rank)
        if local_rank <= 0:
            print(
                f"--- Eval-only result: Loss={val_results['val_loss']:.4f} | "
                f"B0: R={val_results['val_rot_error_deg']:.1f}° T={val_results['val_trans_error_m']:.3f}m | "
                f"A: R={val_results['val_rot_error_a_deg']:.1f}° T={val_results['val_trans_error_a_m']:.3f}m | "
                f"B: R={val_results['val_rot_error_b_deg']:.1f}° T={val_results['val_trans_error_b_m']:.3f}m ---"
            )
        if local_rank != -1:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, 999999):
        if not curriculum_enabled and epoch >= end_epoch:
            break
        if in_phase3 and (epoch - phase3_start_epoch) >= epochs:
            break

        epoch_start = time.time()

        # Balanced sampling: rotate the subset of over-populated bins every epoch
        if balanced_enabled:
            train_dataset.set_balanced_epoch(epoch)
            if is_main:
                dist_info = train_dataset.get_bin_distribution()
                print(f"  [Balanced] Epoch {epoch} bin distribution: "
                      + ", ".join(f"{k}={v}" for k, v in dist_info.items()))
                if writer:
                    for bin_range, count in dist_info.items():
                        writer.add_scalar(f"Balanced/{bin_range}", count, global_opt_step)
            # The sample count may change, so rebuild the DataLoader
            if local_rank != -1:
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler = None
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size_cfg,
                sampler=train_sampler, shuffle=(train_sampler is None),
                num_workers=num_workers_cfg, collate_fn=hm3d_stage2_collate_fn,
                pin_memory=True, drop_last=True,
            )
        elif train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()

        accum_loss = 0.0
        accum_rot_loss = 0.0
        accum_trans_loss = 0.0
        accum_rot_err = 0.0
        accum_trans_err = 0.0
        accum_rot_err_a = 0.0
        accum_trans_err_a = 0.0
        accum_rot_err_b = 0.0
        accum_trans_err_b = 0.0
        opt_step_in_epoch = 0
        curriculum_switched_this_epoch = False
        epoch_training_threshold_idx = current_threshold_idx

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if profiling_enabled and not profiling_warmup_done and step == profiling_warmup_steps:
                raw_model.reset_profiling()
                _train_timings.clear()
                profiling_warmup_done = True

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            pred = model(batch_dev)

            if profiling_enabled:
                _pt = torch.cuda.Event(enable_timing=True)
                _pt.record()

            losses = criterion(pred, batch_dev)
            loss_scaled = losses["loss"] / accum_steps

            if profiling_enabled:
                _pt_end = torch.cuda.Event(enable_timing=True)
                _pt_end.record()
                _train_timings["11_loss_compute"].append((_pt, _pt_end))

            if not torch.isfinite(loss_scaled):
                if is_main:
                    print(f"  [Warning] NaN/Inf loss at epoch {epoch}, step {step}")
                # torch.zeros_like() would create a leaf tensor without grad_fn,
                # causing backward() to fail. Instead, build a zero-gradient loss
                # through all model outputs so DDP backward hooks still fire.
                loss_scaled = (
                    torch.nan_to_num(pred["rotations_a"]).sum() * 0.0
                    + torch.nan_to_num(pred["translations_a"]).sum() * 0.0
                    + torch.nan_to_num(pred["rotations_b"]).sum() * 0.0
                    + torch.nan_to_num(pred["translations_b"]).sum() * 0.0
                )

            if profiling_enabled:
                _pt = torch.cuda.Event(enable_timing=True)
                _pt.record()

            is_accum_final = (step + 1) % accum_steps == 0
            sync_context = nullcontext() if is_accum_final or local_rank == -1 else model.no_sync()
            with sync_context:
                loss_scaled.backward()

            if profiling_enabled:
                _pt_end = torch.cuda.Event(enable_timing=True)
                _pt_end.record()
                _train_timings["12_backward"].append((_pt, _pt_end))

            if (step + 1) % accum_steps == 0:
                if profiling_enabled:
                    _pt = torch.cuda.Event(enable_timing=True)
                    _pt.record()

                if grad_clip > 0:
                    clip_grad_norm_(trainable_params, grad_clip)

                if profiling_enabled:
                    _pt_end = torch.cuda.Event(enable_timing=True)
                    _pt_end.record()
                    _train_timings["13_grad_clip"].append((_pt, _pt_end))

                if profiling_enabled:
                    _pt = torch.cuda.Event(enable_timing=True)
                    _pt.record()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if profiling_enabled:
                    _pt_end = torch.cuda.Event(enable_timing=True)
                    _pt_end.record()
                    _train_timings["14_optim_step"].append((_pt, _pt_end))

                global_opt_step += 1
                opt_step_in_epoch += 1

                if lr_step_interval > 0 and global_opt_step % lr_step_interval == 0:
                    if in_phase3 and phase3_scheduler is not None:
                        phase3_scheduler.step()
                    else:
                        scheduler.step()

                accum_loss += losses["loss"].item()
                accum_rot_loss += losses["rot_loss"].item()
                accum_trans_loss += losses["trans_loss"].item()

                with torch.no_grad():
                    metrics = compute_multiframe_metrics(pred, batch_dev)
                accum_rot_err += metrics["rot_error_deg_mean"]
                accum_trans_err += metrics["trans_error_m_mean"]
                accum_rot_err_a += metrics["rot_error_a_mean"]
                accum_trans_err_a += metrics["trans_error_a_mean"]
                accum_rot_err_b += metrics["rot_error_b_mean"]
                accum_trans_err_b += metrics["trans_error_b_mean"]

                # Curriculum learning convergence detection (uses the B0→A0 rot/trans loss)
                if curriculum_enabled and not in_phase3 and convergence_detector is not None:
                    converged = convergence_detector.update(
                        losses["rot_loss"].item(), losses["trans_loss"].item(),
                    )
                    if local_rank != -1:
                        converged_tensor = torch.tensor([1.0 if converged else 0.0], device=device)
                        dist.all_reduce(converged_tensor, op=dist.ReduceOp.MIN)
                        converged = converged_tensor.item() > 0.5

                    if converged:
                        next_idx = current_threshold_idx + 1
                        if next_idx < len(curriculum_thresholds):
                            old_threshold = curriculum_thresholds[current_threshold_idx]
                            new_threshold = curriculum_thresholds[next_idx]

                            if is_main:
                                print(f"\n{'='*60}")
                                print(f"[Curriculum] Converged at {old_threshold:.2f} → {new_threshold:.2f}")
                                print(f"{'='*60}\n")

                            current_threshold_idx = next_idx
                            n_active = train_dataset.set_curriculum_overlap(new_threshold)
                            if is_main and writer:
                                writer.add_scalar("Curriculum/threshold", new_threshold, global_opt_step)
                                writer.add_scalar("Curriculum/active_samples", n_active, global_opt_step)
                            if balanced_enabled and is_main:
                                dist_info = train_dataset.get_bin_distribution()
                                print(f"  [Balanced] After threshold {new_threshold:.2f}: "
                                      + ", ".join(f"{k}={v}" for k, v in dist_info.items()))
                                if writer:
                                    for bin_range, count in dist_info.items():
                                        writer.add_scalar(f"Balanced/{bin_range}", count, global_opt_step)

                            sync_all(local_rank)

                            if local_rank != -1:
                                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                                train_sampler.set_epoch(epoch)
                            else:
                                train_sampler = None
                            train_loader = DataLoader(
                                train_dataset, batch_size=batch_size_cfg,
                                sampler=train_sampler, shuffle=(train_sampler is None),
                                num_workers=num_workers_cfg, collate_fn=hm3d_stage2_collate_fn,
                                pin_memory=True, drop_last=True,
                            )

                            convergence_detector.reset()

                            if new_threshold < curriculum_phase2_end - 0.001:
                                # Phase 2 ends; save a checkpoint as the starting point for Phase 3
                                if is_main:
                                    print(f"[Phase 2→3] Saving phase2_end checkpoint "
                                          f"(step={global_opt_step}, threshold={new_threshold:.2f})")
                                    _save_checkpoint(
                                        raw_model, optimizer, scheduler, epoch + 1,
                                        best_val_loss, cfg, output_dir, "phase2_end",
                                        curriculum_state=_get_curriculum_state(),
                                    )

                                in_phase3 = True
                                phase3_start_step = global_opt_step
                                phase3_start_epoch = epoch + 1

                                steps_per_epoch_phase3 = max(len(train_loader) // accum_steps, 1)
                                total_steps_phase3 = steps_per_epoch_phase3 * epochs

                                if is_main:
                                    print(f"[Phase 3] Cosine decay: {epochs} epochs, "
                                          f"{total_steps_phase3} steps")

                                def phase3_lr_lambda(sched_step: int) -> float:
                                    current_step = sched_step * lr_step_interval
                                    progress = current_step / max(total_steps_phase3, 1)
                                    return max(0.5 * (1 + math.cos(math.pi * progress)), 0.0)

                                phase3_scheduler = torch.optim.lr_scheduler.LambdaLR(
                                    optimizer, phase3_lr_lambda,
                                )

                            optimizer.zero_grad(set_to_none=True)
                            curriculum_switched_this_epoch = True
                            break

                # Logging (multi-frame version: adds intra-group/inter-group error + timing)
                if is_main and opt_step_in_epoch % log_every == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    # Compute elapsed time and estimated total time
                    # Total steps = actual steps taken in phase1/2 + total phase3 steps (without curriculum, simply epochs*steps_per_epoch)
                    # Steps done so far = global_opt_step (the true cumulative step count from the start of training until now)
                    elapsed_sec = time.time() - training_start_time
                    elapsed_h, elapsed_m = int(elapsed_sec // 3600), int((elapsed_sec % 3600) // 60)
                    if in_phase3:
                        total_steps_est = phase3_start_step + total_steps_phase3
                    elif curriculum_enabled:
                        total_steps_est = total_steps_noncurriculum
                    else:
                        total_steps_est = total_steps_noncurriculum
                    done_steps = global_opt_step
                    if done_steps > 0:
                        est_total_sec = elapsed_sec * total_steps_est / done_steps
                    else:
                        est_total_sec = 0
                    est_h, est_m = int(est_total_sec // 3600), int((est_total_sec % 3600) // 60)
                    time_str = f"⏱ {elapsed_h}h{elapsed_m:02d}m/{est_h}h{est_m:02d}m"
                    print(
                        f"Epoch {epoch} | Step {opt_step_in_epoch}/{max(len(train_loader) // accum_steps, 1)} | "
                        f"B0: R={metrics['rot_error_deg_mean']:.1f}° T={metrics['trans_error_m_mean']:.3f}m | "
                        f"A-intra: R={metrics['rot_error_a_mean']:.1f}° T={metrics['trans_error_a_mean']:.3f}m | "
                        f"B-inter: R={metrics['rot_error_b_mean']:.1f}° T={metrics['trans_error_b_mean']:.3f}m | "
                        f"LR: {lr_now:.2e} | {time_str}"
                    )
                    if writer:
                        writer.add_scalar("Loss/train", losses["loss"].item(), global_opt_step)
                        writer.add_scalar("Loss/rot_intra", losses["rot_loss_intra"].item(), global_opt_step)
                        writer.add_scalar("Loss/trans_intra", losses["trans_loss_intra"].item(), global_opt_step)
                        writer.add_scalar("Loss/rot_inter", losses["rot_loss_inter"].item(), global_opt_step)
                        writer.add_scalar("Loss/trans_inter", losses["trans_loss_inter"].item(), global_opt_step)
                        writer.add_scalar("Loss/rot_b0", losses["rot_loss"].item(), global_opt_step)
                        writer.add_scalar("Loss/trans_b0", losses["trans_loss"].item(), global_opt_step)
                        writer.add_scalar("Metrics/rot_error_deg_b0", metrics["rot_error_deg_mean"], global_opt_step)
                        writer.add_scalar("Metrics/trans_error_m_b0", metrics["trans_error_m_mean"], global_opt_step)
                        writer.add_scalar("Metrics/rot_error_a", metrics["rot_error_a_mean"], global_opt_step)
                        writer.add_scalar("Metrics/rot_error_b", metrics["rot_error_b_mean"], global_opt_step)
                        writer.add_scalar("LR", lr_now, global_opt_step)

                # Intra-epoch validation (very large dataset scenario: periodic validation + saving within an epoch)
                # Note: not gated by the curriculum phase; always runs at the val_steps_interval interval
                if (val_steps_interval > 0 and val_loader
                        and (global_opt_step - _last_intra_val_step) >= val_steps_interval):
                    _last_intra_val_step = global_opt_step
                    if is_main:
                        print(f"\n{'='*60}")
                        print(f"[Intra-epoch Val] step={global_opt_step}, "
                              f"epoch={epoch}, opt_step_in_epoch={opt_step_in_epoch}")
                        print(f"{'='*60}")
                    val_results = validate(model, val_loader, criterion, device, local_rank)
                    model.train()
                    if is_main:
                        print(
                            f"--- Val @step{global_opt_step}: Loss={val_results['val_loss']:.4f} | "
                            f"B0: R={val_results['val_rot_error_deg']:.1f}° T={val_results['val_trans_error_m']:.3f}m | "
                            f"A: R={val_results['val_rot_error_a_deg']:.1f}° T={val_results['val_trans_error_a_m']:.3f}m | "
                            f"B: R={val_results['val_rot_error_b_deg']:.1f}° T={val_results['val_trans_error_b_m']:.3f}m ---\n"
                        )
                        if writer:
                            writer.add_scalar("Loss/val", val_results["val_loss"], global_opt_step)
                            writer.add_scalar("Metrics/val_rot_error_b0", val_results["val_rot_error_deg"], global_opt_step)
                            writer.add_scalar("Metrics/val_trans_error_b0", val_results["val_trans_error_m"], global_opt_step)
                            writer.add_scalar("Metrics/val_rot_error_a", val_results["val_rot_error_a_deg"], global_opt_step)
                            writer.add_scalar("Metrics/val_rot_error_b", val_results["val_rot_error_b_deg"], global_opt_step)

                        if val_results["val_loss"] < best_val_loss:
                            best_val_loss = val_results["val_loss"]
                            best_tag = "finetune_best" if args.finetune else "best"
                            _save_checkpoint(
                                raw_model, optimizer, scheduler, epoch + 1,
                                best_val_loss, cfg, output_dir, best_tag,
                                curriculum_state=_get_curriculum_state(),
                            )

                        _save_checkpoint(
                            raw_model, optimizer, scheduler, epoch + 1,
                            best_val_loss, cfg, output_dir, f"step_{global_opt_step}",
                            curriculum_state=_get_curriculum_state(),
                        )

        # Trailing residual gradients
        if accum_steps > 1 and (step + 1) % accum_steps != 0:
            if grad_clip > 0:
                clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_opt_step += 1
            opt_step_in_epoch += 1
            if lr_step_interval > 0 and global_opt_step % lr_step_interval == 0:
                if in_phase3 and phase3_scheduler is not None:
                    phase3_scheduler.step()
                else:
                    scheduler.step()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        n_steps = max(opt_step_in_epoch, 1)
        if is_main:
            threshold_info = ""
            if curriculum_enabled and epoch_training_threshold_idx < len(curriculum_thresholds):
                thr_val = curriculum_thresholds[epoch_training_threshold_idx]
                threshold_info = f" | Thr: {thr_val:.2f}"
            phase_info = ""
            if curriculum_enabled:
                if in_phase3 and epoch >= phase3_start_epoch:
                    p3_epoch = epoch - phase3_start_epoch
                    phase_info = f" | Phase 3 ({p3_epoch}/{epochs})"
                elif in_phase3:
                    phase_info = " | Phase 2→3"
                else:
                    phase_info = " | Phase 1/2"
            # Epoch-level time statistics (total steps = actual phase1/2 steps + total phase3 steps)
            epoch_elapsed_sec = time.time() - training_start_time
            ep_el_h, ep_el_m = int(epoch_elapsed_sec // 3600), int((epoch_elapsed_sec % 3600) // 60)
            if in_phase3:
                ep_total_est = phase3_start_step + total_steps_phase3
            elif curriculum_enabled:
                ep_total_est = total_steps_noncurriculum
            else:
                ep_total_est = total_steps_noncurriculum
            ep_done = global_opt_step
            if ep_done > 0:
                ep_est_total_sec = epoch_elapsed_sec * ep_total_est / ep_done
            else:
                ep_est_total_sec = 0
            ep_est_h, ep_est_m = int(ep_est_total_sec // 3600), int((ep_est_total_sec % 3600) // 60)
            print(
                f"\n>>> Epoch {epoch} ({epoch_time:.1f}s): "
                f"B0: R={accum_rot_err / n_steps:.1f}° T={accum_trans_err / n_steps:.3f}m | "
                f"A: R={accum_rot_err_a / n_steps:.1f}° T={accum_trans_err_a / n_steps:.3f}m | "
                f"B: R={accum_rot_err_b / n_steps:.1f}° T={accum_trans_err_b / n_steps:.3f}m"
                f"{threshold_info}{phase_info} | "
                f"⏱ {ep_el_h}h{ep_el_m:02d}m/{ep_est_h}h{ep_est_m:02d}m"
            )

        # Profiling summary
        if profiling_enabled and is_main:
            import numpy as _np
            torch.cuda.synchronize()
            fwd_summary = raw_model.get_profiling_summary()
            train_summary: dict[str, dict[str, float]] = {}
            for name, events in _train_timings.items():
                durations = [s.elapsed_time(e) for s, e in events]
                if durations:
                    train_summary[name] = {"mean": float(_np.mean(durations)), "std": float(_np.std(durations)), "count": len(durations)}
            all_summary = {**fwd_summary, **train_summary}
            if all_summary:
                print(f"\n{'='*72}\nPROFILING SUMMARY (Epoch {epoch})\n{'='*72}")
                total_mean = sum(v["mean"] for v in all_summary.values())
                for name in sorted(all_summary.keys()):
                    s = all_summary[name]
                    pct = s["mean"] / total_mean * 100 if total_mean > 0 else 0
                    print(f"  {name:<25s} {s['mean']:>10.2f}ms {pct:>6.1f}%")
                print(f"  {'TOTAL':<25s} {total_mean:>10.2f}ms\n{'='*72}\n")

        # Validation
        should_validate = val_loader and (epoch + 1) % val_every == 0
        if curriculum_enabled:
            should_validate = should_validate and in_phase3

        if should_validate:
            val_results = validate(model, val_loader, criterion, device, local_rank)
            if is_main:
                print(
                    f"--- Val: Loss={val_results['val_loss']:.4f} | "
                    f"B0: R={val_results['val_rot_error_deg']:.1f}° T={val_results['val_trans_error_m']:.3f}m | "
                    f"A: R={val_results['val_rot_error_a_deg']:.1f}° T={val_results['val_trans_error_a_m']:.3f}m | "
                    f"B: R={val_results['val_rot_error_b_deg']:.1f}° T={val_results['val_trans_error_b_m']:.3f}m ---\n"
                )
                if writer:
                    writer.add_scalar("Loss/val", val_results["val_loss"], epoch)
                    writer.add_scalar("Metrics/val_rot_error_b0", val_results["val_rot_error_deg"], epoch)
                    writer.add_scalar("Metrics/val_trans_error_b0", val_results["val_trans_error_m"], epoch)
                    writer.add_scalar("Metrics/val_rot_error_a", val_results["val_rot_error_a_deg"], epoch)
                    writer.add_scalar("Metrics/val_rot_error_b", val_results["val_rot_error_b_deg"], epoch)

                if val_results["val_loss"] < best_val_loss:
                    best_val_loss = val_results["val_loss"]
                    best_tag = "finetune_best" if args.finetune else "best"
                    _save_checkpoint(
                        raw_model, optimizer, scheduler, epoch + 1,
                        best_val_loss, cfg, output_dir, best_tag,
                        curriculum_state=_get_curriculum_state(),
                    )

        sync_all(local_rank)

        if is_main and (epoch + 1) % save_every == 0 and not curriculum_switched_this_epoch and (not curriculum_enabled or in_phase3):
            _save_checkpoint(
                raw_model, optimizer, scheduler, epoch + 1,
                best_val_loss, cfg, output_dir, f"epoch_{epoch + 1}",
                curriculum_state=_get_curriculum_state(),
            )

    # Training finished
    if is_main:
        final_tag = "finetune_final" if args.finetune else "final"
        _save_checkpoint(
            raw_model, optimizer, scheduler, epoch,
            best_val_loss, cfg, output_dir, final_tag,
            curriculum_state=_get_curriculum_state(),
        )
        print(f"\n[Done] MultiFrame training completed at epoch {epoch}! Output: {output_dir}")
        if writer:
            writer.close()

    if local_rank != -1:
        dist.destroy_process_group()


# ============================================================================
# Checkpoint saving
# ============================================================================
def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    config: dict,
    output_dir: str,
    tag: str,
    curriculum_state: dict | None = None,
) -> None:
    full_state = model.state_dict()
    g2g_state = {
        k: v for k, v in full_state.items()
        if not k.startswith("backbone.")
    }

    ckpt_path = os.path.join(output_dir, f"{tag}_checkpoint.pth")
    ckpt_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    if curriculum_state is not None:
        ckpt_data["curriculum_state"] = curriculum_state
    torch.save(ckpt_data, ckpt_path)

    model_path = os.path.join(output_dir, f"{tag}_model.pth")
    torch.save(g2g_state, model_path)

    print(f"  [Saved] {tag}_checkpoint.pth + {tag}_model.pth")


if __name__ == "__main__":
    main()

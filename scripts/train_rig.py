#!/usr/bin/env python3
"""
train_rig.py — G2G MultiFrame training (Rig multi-camera mode)

Uses RigDatasetG2G to load rig multi-camera data and trains the
Stage2ModelMultiFrame model. Differences from train_reloc.py:
  - Uses RigDatasetG2G (randomly selects K cameras per rig)
  - No curriculum learning
  - Supports extrinsics noise augmentation (training set only, see data.extrinsics_noise)
  - per-frame intrinsics (already patched into the model)

Usage:
  torchrun --nproc_per_node=4 --master-port=29595 \
    scripts/train_rig.py \
    --config configs/rig/hm3d_8cam.yaml

  Overfit test:
  torchrun --nproc_per_node=4 --master-port=29595 \
    scripts/train_rig.py \
    --config configs/rig/hm3d_8cam.yaml --overfit
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
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
# plus an already-installed mapanything.
# ============================================================================
from g2g.datasets.rig_dataset import RigDatasetG2G
from g2g.losses.pose_loss_multiframe import (
    MultiFrameG2GLoss,
    compute_multiframe_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions
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


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, keep strings as lists"""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], str):
            out[key] = vals
        else:
            out[key] = vals
    return out


# ============================================================================
# Sanity Check
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
    _print("SANITY CHECK (Rig MultiFrame)")
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

    assert pred["rotations_a"].shape == (B, W - 1, 6), (
        f"rotations_a shape {pred['rotations_a'].shape}, expected ({B}, {W-1}, 6)")
    assert pred["rotations_b"].shape == (B, W, 6), (
        f"rotations_b shape {pred['rotations_b'].shape}")
    assert pred["T_rel"].shape == (B, 4, 4)

    _print(f"  [OK] MultiFrame outputs: rot_a={pred['rotations_a'].shape}, "
           f"rot_b={pred['rotations_b'].shape}")
    _print(f"  [OK] Backward compat: T_rel={pred['T_rel'].shape}")

    assert torch.isfinite(losses["loss"]), f"loss is not finite: {losses['loss']}"
    _print(f"  [OK] Loss: {losses['loss'].item():.4f}")
    _print(f"    Intra: rot={losses['rot_loss_intra'].item():.4f}, "
           f"trans={losses['trans_loss_intra'].item():.4f}")
    _print(f"    Inter: rot={losses['rot_loss_inter'].item():.4f}, "
           f"trans={losses['trans_loss_inter'].item():.4f}")

    # Check intrinsics dimensions (rig mode should be [B, K, 3, 3])
    intr_a = batch_dev["intrinsics_a"]
    _print(f"  [OK] Intrinsics shape: {intr_a.shape} "
           f"({'per-frame' if intr_a.dim() == 4 else 'shared'})")

    losses["loss"].backward()

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

    assert "extrinsics_a_clean" in batch_dev, "Missing extrinsics_a_clean"
    assert "extrinsics_b_clean" in batch_dev, "Missing extrinsics_b_clean"
    _print(f"  [OK] Clean extrinsics present")

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    _print(f"  [OK] GPU peak memory: {peak_mb:.0f} MB")

    model.zero_grad(set_to_none=True)

    metrics = compute_multiframe_metrics(pred, batch_dev)
    _print(f"  [INFO] B0->A0: rot={metrics['rot_error_deg_mean']:.1f} deg, "
           f"trans={metrics['trans_error_m_mean']:.3f}m")
    _print(f"  [INFO] A-intra: rot={metrics['rot_error_a_mean']:.1f} deg, "
           f"trans={metrics['trans_error_a_mean']:.3f}m")
    _print(f"  [INFO] B-inter: rot={metrics['rot_error_b_mean']:.1f} deg, "
           f"trans={metrics['trans_error_b_mean']:.3f}m")

    _print("SANITY CHECK PASSED")
    _print("=" * 60 + "\n")


# ============================================================================
# Validation
# ============================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: MultiFrameG2GLoss,
    device: torch.device,
    local_rank: int,
    max_batches: int = -1,
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

    n_total = min(len(val_loader), max_batches) if max_batches > 0 else len(val_loader)
    t0 = time.time()

    for step, batch in enumerate(val_loader):
        if max_batches > 0 and step >= max_batches:
            break

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

        if local_rank <= 0 and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (n_total - step - 1)
            print(f"  [Val {step+1}/{n_total}] "
                  f"loss={total_loss/num_batches:.4f} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

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
    parser = argparse.ArgumentParser(description="G2G MultiFrame Rig Training")
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

    from g2g.models.stage2_model_multiframe import Stage2ModelMultiFrame

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = train_cfg.get("output_dir", "./outputs/g2g_rig")
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
        print("G2G MultiFrame Rig Training")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Overfit mode: {args.overfit}")
        print(f"Device: {device}, World size: {world_size}")
        print(f"Output: {output_dir}")
        bs = train_cfg.get("batch_size", 2)
        accum = train_cfg.get("gradient_accumulation", 1)
        print(f"Batch size: {bs} x {world_size} GPUs x {accum} accum = {bs * world_size * accum} effective")
        print(f"LR: {train_cfg.get('lr', 3e-4)}")
        print(f"Epochs: {train_cfg.get('epochs', 3)}")
        print(f"Loss: {train_cfg.get('loss_type', 'chordal')}")
        print(f"Intra weight: {train_cfg.get('intra_pose_weight', 0.3)}")
        print(f"Inter weight: {train_cfg.get('inter_pose_weight', 1.0)}")
        print("=" * 60)

    # ========== Create datasets ==========
    data_cfg = cfg.get("data", {})
    max_scenes = args.max_scenes
    if args.overfit and max_scenes is None:
        max_scenes = cfg.get("overfit", {}).get("max_scenes", 3)

    ds_kwargs = dict(
        num_cameras=data_cfg.get("num_cameras", 3),
        total_cameras=data_cfg.get("total_cameras", 8),
        image_size=data_cfg.get("image_size", 224),
        patch_size=data_cfg.get("patch_size", 14),
        max_scenes=max_scenes if max_scenes else -1,
    )
    # Extrinsics noise is applied to the training set only
    ext_noise_cfg = data_cfg.get("extrinsics_noise", None)
    if ext_noise_cfg and ext_noise_cfg.get("enabled", False) and is_main:
        print(f"  Extrinsics noise: rot={ext_noise_cfg.get('rotation_noise_std_deg', 1.0)}deg, "
              f"trans={ext_noise_cfg.get('translation_noise_std_m', 0.05)}m")

    if is_main:
        print("\n[Step 1] Loading training dataset (Rig)...")
    train_data_cfg = data_cfg.get("train", {})
    train_dataset = RigDatasetG2G(
        step1_root=train_data_cfg["step1_root"],
        step2_root=train_data_cfg["step2_root"],
        step3_root=train_data_cfg["step3_root"],
        step6_root=train_data_cfg["step6_root"],
        index_root=train_data_cfg["index_root"],
        extrinsics_noise_cfg=ext_noise_cfg,
        **ds_kwargs,
    )
    if is_main:
        print(f"  Training: {len(train_dataset)} pairs")

    val_loader = None
    val_data_cfg = data_cfg.get("val", {})
    if val_data_cfg:
        if is_main:
            print("[Step 1.1] Loading validation dataset (Rig)...")
        val_max = max_scenes if (args.overfit and max_scenes) else -1
        val_mpp = val_data_cfg.get("max_pairs_per_scene", -1)
        val_dataset = RigDatasetG2G(
            step1_root=val_data_cfg["step1_root"],
            step2_root=val_data_cfg["step2_root"],
            step3_root=val_data_cfg["step3_root"],
            step6_root=val_data_cfg["step6_root"],
            index_root=val_data_cfg["index_root"],
            num_cameras=data_cfg.get("num_cameras", 3),
            total_cameras=data_cfg.get("total_cameras", 8),
            image_size=data_cfg.get("image_size", 224),
            patch_size=data_cfg.get("patch_size", 14),
            max_scenes=val_max,
            max_pairs_per_scene=val_mpp,
        )
        if is_main:
            print(f"  Validation: {len(val_dataset)} pairs")

        val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 2),
            sampler=val_sampler,
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # REAL validation set (optional)
    val_real_loader = None
    val_real_cfg = data_cfg.get("val_real", {})
    if val_real_cfg:
        if is_main:
            print("[Step 1.2] Loading REAL validation dataset (Rig)...")
        val_real_mpp = val_real_cfg.get("max_pairs_per_scene", -1)
        val_real_dataset = RigDatasetG2G(
            step1_root=val_real_cfg["step1_root"],
            step2_root=val_real_cfg["step2_root"],
            step3_root=val_real_cfg.get("step3_root", val_real_cfg["step1_root"]),
            step6_root=val_real_cfg["step6_root"],
            index_root=val_real_cfg["index_root"],
            num_cameras=data_cfg.get("num_cameras", 3),
            total_cameras=data_cfg.get("total_cameras", 8),
            image_size=data_cfg.get("image_size", 224),
            patch_size=data_cfg.get("patch_size", 14),
            max_scenes=-1,
            max_pairs_per_scene=val_real_mpp,
        )
        if is_main:
            print(f"  REAL Validation: {len(val_real_dataset)} pairs")

        val_real_sampler = DistributedSampler(val_real_dataset, shuffle=False) if local_rank != -1 else None
        val_real_loader = DataLoader(
            val_real_dataset,
            batch_size=train_cfg.get("batch_size", 2),
            sampler=val_real_sampler,
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ========== Create model ==========
    if is_main:
        print("\n[Step 2] Building model (MultiFrame Rig)...")

    backbone_cfg = cfg.get("backbone", {})
    model_cfg = cfg.get("model", {})

    model_kwargs = dict(
        model_path=backbone_cfg["model_path"],
        embed_dim=model_cfg.get("embed_dim", 768),
        num_latents=model_cfg.get("num_latents", 64),
        num_frames_per_group=model_cfg.get("num_frames_per_group", 3),
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

    # rank 0 loads first, other ranks wait (avoid concurrent reads of the large model file)
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

    # ========== Optimizer & scheduler ==========
    trainable_params = raw_model.get_trainable_parameters()
    optimizer = AdamW(
        trainable_params,
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    accum_steps = train_cfg.get("gradient_accumulation", 1)
    epochs = train_cfg.get("epochs", 3)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    lr_step_interval = train_cfg.get("lr_step_interval", 10)

    steps_per_epoch = max(len(train_loader) // accum_steps, 1)
    total_steps = steps_per_epoch * epochs

    def lr_lambda(sched_step: int) -> float:
        current_step = sched_step * lr_step_interval
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        decay_steps = current_step - warmup_steps
        total_decay = total_steps - warmup_steps
        if total_decay <= 0:
            return 1.0
        progress = min(decay_steps / total_decay, 1.0)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = MultiFrameG2GLoss(
        rotation_weight=train_cfg.get("rotation_weight", 5.0),
        translation_weight=train_cfg.get("translation_weight", 1.0),
        intra_weight=train_cfg.get("intra_pose_weight", 0.5),
        inter_weight=train_cfg.get("inter_pose_weight", 1.0),
        loss_type=train_cfg.get("loss_type", "chordal"),
    )

    grad_clip = train_cfg.get("grad_clip", 5.0)

    if is_main:
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}, LR step interval: {lr_step_interval}")

    # ========== Resume training ==========
    start_epoch = 0
    best_val_loss = float("inf")
    global_opt_step = 0

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
            global_opt_step = ckpt.get("global_opt_step", 0)
            if is_main:
                print(f"  Resumed from epoch {start_epoch}")
        else:
            raw_model.load_state_dict(ckpt, strict=False)

    if args.finetune:
        if is_main:
            print(f"\n[Finetune] Loading weights from {args.finetune}...")
        ckpt = torch.load(args.finetune, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Handle num_frames_per_group mismatch: truncate frame-related embeddings
        model_state = raw_model.state_dict()
        adapted_keys = []
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                ckpt_shape = state_dict[key].shape
                model_shape = model_state[key].shape
                if key == "bridge.frame_embed":
                    # [W_ckpt, 1, C] → keep the first W_model frames
                    state_dict[key] = state_dict[key][:model_shape[0]]
                    adapted_keys.append(f"{key}: {ckpt_shape} → {model_shape}")
                elif key == "pose_head.frame_identity_embed":
                    # Layout: A1..A(W-1), B0..B(W-1)
                    # ckpt W=5: [A1,A2,A3,A4, B0,B1,B2,B3,B4] (9 entries)
                    # model W=3: [A1,A2, B0,B1,B2] (5 entries)
                    W_ckpt = (ckpt_shape[0] + 1) // 2
                    W_model = (model_shape[0] + 1) // 2
                    idx_a = list(range(W_model - 1))
                    idx_b = list(range(W_ckpt - 1, W_ckpt - 1 + W_model))
                    indices = idx_a + idx_b
                    state_dict[key] = state_dict[key][indices]
                    adapted_keys.append(f"{key}: {ckpt_shape} → {model_shape} "
                                        f"(indices {indices})")
                else:
                    del state_dict[key]
                    adapted_keys.append(f"{key}: {ckpt_shape} vs {model_shape} (skipped)")

        if adapted_keys and is_main:
            print(f"  [Finetune] Adapted {len(adapted_keys)} keys for shape mismatch:")
            for info in adapted_keys:
                print(f"    {info}")

        raw_model.load_state_dict(state_dict, strict=False)
        if "model_state_dict" in ckpt:
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main:
            print(f"  [Finetune] Starting from epoch {start_epoch}")

    # ========== Sanity Check ==========
    if is_main:
        print("\n[Step 3] Running sanity check (Rig MultiFrame)...")
    sanity_loader = DataLoader(
        train_dataset,
        batch_size=min(train_cfg.get("batch_size", 2), 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
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
    save_every = train_cfg.get("save_every", 1)
    max_val_batches = train_cfg.get("max_val_batches", -1)
    val_steps_interval = train_cfg.get("val_steps_interval", 0)
    _last_intra_val_step = global_opt_step

    end_epoch = start_epoch + epochs if args.finetune else epochs

    if is_main:
        est_hours = total_steps * 0.5 / 3600
        print(f"  Total opt steps: {total_steps}, estimated ~{est_hours:.1f}h (rough)")
        if val_steps_interval > 0:
            print(f"  Intra-epoch validation every {val_steps_interval} opt steps")

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()

        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()

        opt_step_in_epoch = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            pred = model(batch_dev)
            losses = criterion(pred, batch_dev)
            loss_scaled = losses["loss"] / accum_steps

            if not torch.isfinite(loss_scaled):
                if is_main:
                    print(f"  [Warning] NaN/Inf loss at epoch {epoch}, step {step}")
                loss_scaled = (
                    torch.nan_to_num(pred["rotations_a"]).sum() * 0.0
                    + torch.nan_to_num(pred["translations_a"]).sum() * 0.0
                    + torch.nan_to_num(pred["rotations_b"]).sum() * 0.0
                    + torch.nan_to_num(pred["translations_b"]).sum() * 0.0
                )

            from contextlib import nullcontext
            is_accum_final = (step + 1) % accum_steps == 0
            sync_context = nullcontext() if is_accum_final or local_rank == -1 else model.no_sync()
            with sync_context:
                loss_scaled.backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip > 0:
                    clip_grad_norm_(trainable_params, grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_opt_step += 1
                opt_step_in_epoch += 1

                if lr_step_interval > 0 and global_opt_step % lr_step_interval == 0:
                    scheduler.step()

                cur_loss = losses["loss"].item()
                cur_rot_loss = losses["rot_loss"].item()
                cur_trans_loss = losses["trans_loss"].item()

                with torch.no_grad():
                    metrics = compute_multiframe_metrics(pred, batch_dev)
                cur_rot_err = metrics["rot_error_deg_mean"]
                cur_trans_err = metrics["trans_error_m_mean"]
                cur_rot_err_a = metrics["rot_error_a_mean"]
                cur_trans_err_a = metrics["trans_error_a_mean"]
                cur_rot_err_b = metrics["rot_error_b_mean"]
                cur_trans_err_b = metrics["trans_error_b_mean"]

                if is_main and opt_step_in_epoch % log_every == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    elapsed_total = time.time() - training_start_time
                    eta_total = elapsed_total / max(global_opt_step, 1) * (total_steps - global_opt_step)

                    print(
                        f"  [E{epoch} S{opt_step_in_epoch}/{steps_per_epoch} G{global_opt_step}/{total_steps}] "
                        f"loss={cur_loss:.4f} "
                        f"rot={cur_rot_loss:.4f} trans={cur_trans_loss:.4f} | "
                        f"B0: {cur_rot_err:.1f}° {cur_trans_err:.3f}m | "
                        f"A-intra: {cur_rot_err_a:.1f}° {cur_trans_err_a:.3f}m | "
                        f"B-inter: {cur_rot_err_b:.1f}° {cur_trans_err_b:.3f}m | "
                        f"lr={cur_lr:.2e} {elapsed_total/3600:.1f}h/{eta_total/3600:.1f}h",
                        flush=True,
                    )

                    if writer:
                        writer.add_scalar("train/loss", cur_loss, global_opt_step)
                        writer.add_scalar("train/rot_loss", cur_rot_loss, global_opt_step)
                        writer.add_scalar("train/trans_loss", cur_trans_loss, global_opt_step)
                        writer.add_scalar("train/rot_err_deg", cur_rot_err, global_opt_step)
                        writer.add_scalar("train/trans_err_m", cur_trans_err, global_opt_step)
                        writer.add_scalar("train/rot_err_a", cur_rot_err_a, global_opt_step)
                        writer.add_scalar("train/trans_err_a", cur_trans_err_a, global_opt_step)
                        writer.add_scalar("train/rot_err_b", cur_rot_err_b, global_opt_step)
                        writer.add_scalar("train/trans_err_b", cur_trans_err_b, global_opt_step)
                        writer.add_scalar("train/lr", cur_lr, global_opt_step)

                # Convergence diagnostics: print detailed info every 200 steps
                if is_main and global_opt_step % 200 == 0 and global_opt_step > 0:
                    with torch.no_grad():
                        rot_b = pred["rotations_b"]
                        trans_b = pred["translations_b"]
                        rot_a = pred["rotations_a"]
                        trans_a = pred["translations_a"]
                        feats_a = batch_dev.get("enc_feats_a")
                        feats_b = batch_dev.get("enc_feats_b")
                        print(f"  [DIAG G{global_opt_step}] "
                              f"rot_b: mean={rot_b.mean():.4f} std={rot_b.std():.4f} "
                              f"min={rot_b.min():.3f} max={rot_b.max():.3f} | "
                              f"trans_b: mean={trans_b.mean():.4f} std={trans_b.std():.4f} "
                              f"min={trans_b.min():.3f} max={trans_b.max():.3f}",
                              flush=True)
                        print(f"  [DIAG G{global_opt_step}] "
                              f"rot_a: mean={rot_a.mean():.4f} std={rot_a.std():.4f} | "
                              f"trans_a: mean={trans_a.mean():.4f} std={trans_a.std():.4f}",
                              flush=True)
                        if feats_a is not None:
                            z_a = 100*(feats_a==0).float().mean()
                            z_b = 100*(feats_b==0).float().mean()
                            print(f"  [DIAG G{global_opt_step}] "
                                  f"enc_feats_a: mean={feats_a.mean():.4f} std={feats_a.std():.4f} "
                                  f"zeros%={z_a:.1f}% | "
                                  f"enc_feats_b: mean={feats_b.mean():.4f} std={feats_b.std():.4f} "
                                  f"zeros%={z_b:.1f}%",
                                  flush=True)
                            if z_a > 0.01 or z_b > 0.01:
                                print(f"  [WARNING G{global_opt_step}] "
                                      f"enc_feats zeros% abnormally high! "
                                      f"A={z_a:.1f}% B={z_b:.1f}% "
                                      f"(should be close to 0%). Check whether step6 features loaded successfully.",
                                      flush=True)
                        # Check whether B0 rot collapsed within the batch (all samples predict the same)
                        rot_b0 = rot_b[:, 0]  # [B, 6]
                        cross_sample_std = rot_b0.std(dim=0).mean()
                        print(f"  [DIAG G{global_opt_step}] "
                              f"B0 cross-sample rot std={cross_sample_std:.6f} "
                              f"(collapsed if ~0)",
                              flush=True)

                        # A-intra per-frame rotation error breakdown (diagnose mode collapse)
                        from g2g.losses.pose_loss_multiframe import _batch_rotation_error_deg
                        R_pred_a = pred["rotation_matrices_a"]  # [B, W-1, 3, 3]
                        ext_a_clean = batch_dev["extrinsics_a_clean"]
                        R_gt_a = ext_a_clean[:, 1:, :3, :3]
                        per_frame_err = _batch_rotation_error_deg(R_pred_a, R_gt_a)  # [B, W-1]
                        n_a = per_frame_err.shape[1]
                        per_frame_str = " ".join(
                            f"A{i+1}={per_frame_err[:, i].mean():.1f}°"
                            for i in range(n_a)
                        )
                        # A inter-frame prediction diversity: cross-frame std of each frame's 6D rotation
                        rot_a_per_frame_std = rot_a.std(dim=1).mean()
                        print(f"  [DIAG G{global_opt_step}] "
                              f"A-intra per-frame: {per_frame_str} "
                              f"| A cross-frame rot std={rot_a_per_frame_std:.4f} "
                              f"(collapse if ~0)",
                              flush=True)

                # Intra-epoch validation (every val_steps_interval opt steps)
                if (val_steps_interval > 0 and val_loader
                        and (global_opt_step - _last_intra_val_step) >= val_steps_interval):
                    _last_intra_val_step = global_opt_step
                    if is_main:
                        print(f"\n{'='*60}")
                        print(f"[Intra-epoch Val] step={global_opt_step}, "
                              f"epoch={epoch}, opt_step_in_epoch={opt_step_in_epoch}")
                        print(f"{'='*60}")
                    val_results = validate(
                        model, val_loader, criterion, device, local_rank,
                        max_batches=max_val_batches,
                    )
                    model.train()
                    if is_main:
                        print(
                            f"--- Val @step{global_opt_step}: "
                            f"Loss={val_results['val_loss']:.4f} | "
                            f"B0: R={val_results['val_rot_error_deg']:.1f} deg "
                            f"T={val_results['val_trans_error_m']:.3f}m | "
                            f"A: R={val_results['val_rot_error_a_deg']:.1f} deg "
                            f"T={val_results['val_trans_error_a_m']:.3f}m | "
                            f"B: R={val_results['val_rot_error_b_deg']:.1f} deg "
                            f"T={val_results['val_trans_error_b_m']:.3f}m ---\n"
                        )
                        if writer:
                            writer.add_scalar("val/loss", val_results["val_loss"], global_opt_step)
                            writer.add_scalar("val/rot_error_deg", val_results["val_rot_error_deg"], global_opt_step)
                            writer.add_scalar("val/trans_error_m", val_results["val_trans_error_m"], global_opt_step)

                        if val_results["val_loss"] < best_val_loss:
                            best_val_loss = val_results["val_loss"]
                            best_path = os.path.join(output_dir, "best.pt")
                            torch.save({
                                "epoch": epoch + 1,
                                "model_state_dict": raw_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "best_val_loss": best_val_loss,
                                "global_opt_step": global_opt_step,
                            }, best_path)
                            print(f"  New best! Saved to {best_path}")

                        ckpt_path = os.path.join(output_dir, f"step_{global_opt_step}.pt")
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                            "global_opt_step": global_opt_step,
                        }, ckpt_path)
                        print(f"  Saved checkpoint: {ckpt_path}")

                    # REAL validation (intra-epoch)
                    if val_real_loader:
                        if is_main:
                            print(f"  [REAL Val @step{global_opt_step}]")
                        val_real_results = validate(
                            model, val_real_loader, criterion, device, local_rank,
                            max_batches=max_val_batches,
                        )
                        model.train()
                        if is_main:
                            print(
                                f"--- REAL Val @step{global_opt_step}: "
                                f"Loss={val_real_results['val_loss']:.4f} | "
                                f"B0: R={val_real_results['val_rot_error_deg']:.1f} deg "
                                f"T={val_real_results['val_trans_error_m']:.3f}m | "
                                f"A: R={val_real_results['val_rot_error_a_deg']:.1f} deg "
                                f"T={val_real_results['val_trans_error_a_m']:.3f}m | "
                                f"B: R={val_real_results['val_rot_error_b_deg']:.1f} deg "
                                f"T={val_real_results['val_trans_error_b_m']:.3f}m ---\n"
                            )
                            if writer:
                                writer.add_scalar("val_real/loss", val_real_results["val_loss"], global_opt_step)
                                writer.add_scalar("val_real/rot_error_deg", val_real_results["val_rot_error_deg"], global_opt_step)
                                writer.add_scalar("val_real/trans_error_m", val_real_results["val_trans_error_m"], global_opt_step)
                                writer.add_scalar("val_real/rot_error_a_deg", val_real_results["val_rot_error_a_deg"], global_opt_step)
                                writer.add_scalar("val_real/trans_error_a_m", val_real_results["val_trans_error_a_m"], global_opt_step)
                                writer.add_scalar("val_real/rot_error_b_deg", val_real_results["val_rot_error_b_deg"], global_opt_step)
                                writer.add_scalar("val_real/trans_error_b_m", val_real_results["val_trans_error_b_m"], global_opt_step)

        epoch_elapsed = time.time() - epoch_start

        if is_main:
            print(f"\n  Epoch {epoch} done in {epoch_elapsed:.0f}s "
                  f"({opt_step_in_epoch} opt steps)")

        # ========== Validation ==========
        if val_loader and (epoch + 1) % val_every == 0:
            if is_main:
                print(f"\n  Validating (max {max_val_batches} batches)...")
            val_metrics = validate(
                model, val_loader, criterion, device, local_rank,
                max_batches=max_val_batches,
            )
            if is_main:
                print(f"  Val: loss={val_metrics['val_loss']:.4f} "
                      f"rot={val_metrics['val_rot_error_deg']:.1f} deg "
                      f"trans={val_metrics['val_trans_error_m']:.3f}m")
                print(f"  Val A-intra: rot={val_metrics['val_rot_error_a_deg']:.1f} deg "
                      f"trans={val_metrics['val_trans_error_a_m']:.3f}m")
                print(f"  Val B-inter: rot={val_metrics['val_rot_error_b_deg']:.1f} deg "
                      f"trans={val_metrics['val_trans_error_b_m']:.3f}m")

                if writer:
                    for k, v in val_metrics.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"val/{k}", v, epoch)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_path = os.path.join(output_dir, "best.pt")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "global_opt_step": global_opt_step,
                    }, best_path)
                    print(f"  New best! Saved to {best_path}")

        # REAL epoch-end validation
        if val_real_loader and (epoch + 1) % val_every == 0:
            if is_main:
                print(f"\n  REAL Validating (max {max_val_batches} batches)...")
            val_real_metrics = validate(
                model, val_real_loader, criterion, device, local_rank,
                max_batches=max_val_batches,
            )
            if is_main:
                print(f"  REAL Val: loss={val_real_metrics['val_loss']:.4f} "
                      f"rot={val_real_metrics['val_rot_error_deg']:.1f} deg "
                      f"trans={val_real_metrics['val_trans_error_m']:.3f}m")
                print(f"  REAL Val A-intra: rot={val_real_metrics['val_rot_error_a_deg']:.1f} deg "
                      f"trans={val_real_metrics['val_trans_error_a_m']:.3f}m")
                print(f"  REAL Val B-inter: rot={val_real_metrics['val_rot_error_b_deg']:.1f} deg "
                      f"trans={val_real_metrics['val_trans_error_b_m']:.3f}m")

                if writer:
                    for k, v in val_real_metrics.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(f"val_real/{k}", v, epoch)

        # ========== Save checkpoint ==========
        if is_main and (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "global_opt_step": global_opt_step,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        print("", flush=True)

    # ========== Training complete ==========
    total_time = time.time() - training_start_time
    if is_main:
        print("=" * 60)
        print(f"Training complete. Total time: {total_time/3600:.1f}h")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    if writer:
        writer.close()
    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

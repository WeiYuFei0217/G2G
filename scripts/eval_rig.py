#!/usr/bin/env python3
"""
eval_rig.py — G2G multi-frame (rig mode) evaluation script

Usage:
    # Single GPU
    python scripts/eval_rig.py \
        --config configs/rig/hm3d_8cam.yaml \
        --checkpoint release_weights/HM3D-Rig-8.pth \
        --batch-size 8 --num-workers 4 \
        --output-dir outputs/eval_HM3D-Rig-8

    # 4-GPU parallel
    torchrun --nproc_per_node=4 --master-port=29596 \
        scripts/eval_rig.py \
        --config configs/rig/hm3d_8cam.yaml \
        --checkpoint release_weights/HM3D-Rig-8.pth \
        --batch-size 8 --num-workers 4 \
        --output-dir outputs/eval_HM3D-Rig-8

    # Skip inference, only recompute statistics
    python scripts/eval_rig.py \
        --config configs/rig/hm3d_8cam.yaml --checkpoint release_weights/HM3D-Rig-8.pth \
        --skip-inference --output-dir outputs/eval_HM3D-Rig-8
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Dependencies: the g2g package installed via `pip install -e .` in this repo, plus the installed mapanything.
from g2g.datasets.rig_dataset import RigDatasetG2G
from g2g.models.stage2_model_multiframe import Stage2ModelMultiFrame


# ============================================================================
# Utility functions
# ============================================================================
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return -1, 1
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    return local_rank <= 0


# ============================================================================
# Model construction
# ============================================================================
def build_and_load_model(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    is_main: bool = True,
) -> Stage2ModelMultiFrame:
    model_cfg = cfg["model"]
    backbone_cfg = cfg["backbone"]

    model = Stage2ModelMultiFrame(
        model_path=backbone_cfg["model_path"],
        embed_dim=model_cfg.get("embed_dim", 768),
        num_latents=model_cfg.get("num_latents", 64),
        num_frames_per_group=model_cfg.get("num_frames_per_group", 8),
        resampler_layers=model_cfg.get("resampler_layers", 2),
        bridge_alternating_pairs=model_cfg.get("bridge_alternating_pairs", 2),
        bridge_merged_layers=model_cfg.get("bridge_merged_layers", 2),
        reinject_anchor_after_merge=model_cfg.get("reinject_anchor_after_merge", True),
        pose_head_hidden_dim=model_cfg.get("pose_head_hidden_dim", 512),
        pose_head_layers=model_cfg.get("pose_head_layers", 3),
        num_heads=model_cfg.get("num_heads", 8),
        rotation_repr=model_cfg.get("rotation_repr", "6d"),
        freeze_backbone=True,
    )

    if is_main:
        print(f"[Eval] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Compatible with three sources: a full training checkpoint (containing a model_state_dict / model key),
    # or an extracted lightweight G2G-only state_dict (without the frozen backbone.* keys).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt  # Lightweight weights: the object itself is the state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]
    if non_backbone_missing or unexpected:
        raise RuntimeError(
            "[eval_rig] checkpoint loading mismatch.\n"
            f"  missing(non-backbone)={non_backbone_missing[:20]}\n"
            f"  unexpected={list(unexpected)[:20]}"
        )
    if is_main:
        n_bb = len(missing) - len(non_backbone_missing)
        print(f"[Eval] Loaded weights (strict=False); skipped {n_bb} backbone keys")

    if is_main:
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("global_opt_step", ckpt.get("global_step", "?"))
        best_val = ckpt.get("best_val_loss", "?")
        print(f"[Eval] Model loaded (epoch={epoch}, step={step}, best_val_loss={best_val})")

    model.eval()
    model.to(device)
    return model


# ============================================================================
# DataLoader
# ============================================================================
def collate_fn(batch: list[dict]) -> dict:
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


def build_val_loader(
    cfg: dict,
    batch_size: int,
    num_workers: int,
    local_rank: int = -1,
    world_size: int = 1,
    max_scenes: int = -1,
    data_override: dict | None = None,
    sample_stride: int = 1,
    max_body_dist: float = -1.0,
    max_pairs_per_scene: int = -1,
    val_key: str = "val",
) -> tuple[DataLoader, RigDatasetG2G]:
    d = data_override or cfg["data"].get(val_key, cfg["data"]["val"])
    dataset = RigDatasetG2G(
        step1_root=d["step1_root"],
        step2_root=d["step2_root"],
        step3_root=d["step3_root"],
        step6_root=d["step6_root"],
        index_root=d["index_root"],
        num_cameras=cfg["data"].get("num_cameras", 8),
        total_cameras=cfg["data"].get("total_cameras", 8),
        image_size=cfg["data"].get("image_size", 224),
        patch_size=cfg["data"].get("patch_size", 14),
        camera_select_seed=cfg["data"].get("camera_select_seed", None),
        max_scenes=max_scenes,
        max_pairs_per_scene=max_pairs_per_scene,
    )

    if (sample_stride and sample_stride > 1) or (max_body_dist and max_body_dist > 0):
        from torch.utils.data import Subset
        _keep = list(range(len(dataset)))
        if max_body_dist and max_body_dist > 0:
            _keep = [i for i in _keep if float(dataset.pairs[i].get("body_dist", 1e9)) <= max_body_dist]
        if sample_stride and sample_stride > 1:
            _keep = _keep[::sample_stride]
        dataset = Subset(dataset, _keep)
        if is_main_process(local_rank):
            print(f"[Eval] max_body_dist={max_body_dist}, sample_stride={sample_stride} -> {len(dataset)} pairs")

    sampler = None
    if local_rank != -1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    if is_main_process(local_rank):
        print(f"[Eval] Val dataset: {len(dataset)} pairs, "
              f"num_cameras={cfg['data'].get('num_cameras', 8)}, "
              f"{batch_size=} (per GPU), world_size={world_size}")

    return loader, dataset


# ============================================================================
# Inference
# ============================================================================
@torch.no_grad()
def run_inference(
    model: Stage2ModelMultiFrame,
    loader: DataLoader,
    device: torch.device,
    is_main: bool = True,
) -> list[dict]:
    results = []
    iterator = tqdm(loader, desc="Inference", disable=not is_main)

    for batch in iterator:
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(device)
            else:
                batch_dev[k] = v

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch_dev)

        # G2G directly outputs the relative pose
        R_pred = pred["rotation_matrix"].float()   # (B, 3, 3) — B0→A0
        t_pred = pred["translation"].float()        # (B, 3)

        # GT relative pose
        T_rel_gt = batch_dev["T_rel_gt"].float()    # (B, 4, 4)
        R_gt = T_rel_gt[:, :3, :3]
        t_gt = T_rel_gt[:, :3, 3]

        # Rotation error
        R_diff = torch.bmm(R_pred.transpose(-1, -2), R_gt)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        rot_errors = torch.acos(cos_angle) * (180.0 / math.pi)

        # Translation error
        trans_errors = torch.norm(t_pred - t_gt, dim=-1)

        # Translation direction angle error (RTA, scale-invariant)
        _eps = 1e-8
        _tp = t_pred / (t_pred.norm(dim=-1, keepdim=True) + _eps)
        _tg = t_gt / (t_gt.norm(dim=-1, keepdim=True) + _eps)
        _cos = torch.clamp((_tp * _tg).sum(-1), -1.0, 1.0)
        t_dir_errors = torch.acos(_cos) * (180.0 / math.pi)
        # When the GT translation ≈ 0 the direction is meaningless → NaN (dropped during aggregation)
        t_dir_errors = torch.where(
            t_gt.norm(dim=-1) < 1e-6,
            torch.full_like(t_dir_errors, float("nan")),
            t_dir_errors,
        )

        # Per-view error (all B-group views relative to A0)
        rm_b = pred.get("rotation_matrices_b")  # (B, W, 3, 3)
        t_b = pred.get("translations_b")        # (B, W, 3)
        if rm_b is not None and t_b is not None:
            rm_b = rm_b.float()
            t_b = t_b.float()
            pv_rot, pv_trans = _compute_per_view_b_error(rm_b, t_b, batch_dev)
        else:
            pv_rot = rot_errors
            pv_trans = trans_errors

        B = rot_errors.shape[0]
        scene_ids = batch_dev["scene_id"]
        # Pair identity (optional; present when the dataset emits it)
        traj_a = batch_dev.get("traj_a"); t_a = batch_dev.get("t_a")
        traj_b = batch_dev.get("traj_b"); t_b = batch_dev.get("t_b")

        for i in range(B):
            record = {
                "scene_id": scene_ids[i],
                "traj_a": traj_a[i] if traj_a is not None else "",
                "t_a": t_a[i] if t_a is not None else "",
                "traj_b": traj_b[i] if traj_b is not None else "",
                "t_b": t_b[i] if t_b is not None else "",
                "rot_error_deg": float(rot_errors[i].item()),
                "trans_error_m": float(trans_errors[i].item()),
                "t_dir_error_deg": float(t_dir_errors[i].item()),
                "per_view_rot_mean_deg": float(pv_rot[i].item()) if pv_rot.dim() == 1 else float(pv_rot[i].mean().item()),
                "per_view_trans_mean_m": float(pv_trans[i].item()) if pv_trans.dim() == 1 else float(pv_trans[i].mean().item()),
            }
            results.append(record)

    return results


def _compute_per_view_b_error(
    rm_b: torch.Tensor,
    t_b: torch.Tensor,
    batch_dev: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the pose error of all B-group views relative to A0

    The model outputs rm_b[:, i], the rotation of B_i relative to A0.
    The GT must be composed from T_rel_gt (B0→A0) + extrinsics_b (relative extrinsics within the rig).
    """
    B, W = rm_b.shape[:2]
    T_rel_gt = batch_dev["T_rel_gt"].float()                     # (B, 4, 4)
    extrinsics_b = batch_dev.get("extrinsics_b_clean", batch_dev["extrinsics_b"]).float()  # (B, W, 4, 4)

    # GT: T_A0_Bi = T_rel_gt @ extrinsics_b[i]
    # where extrinsics_b[i] = inv(body_T_cam[0]) @ body_T_cam[i], i.e. B_ref→B_i
    # T_rel_gt = inv(T_world_cam_a0) @ T_world_cam_b0
    # therefore T_A0_Bi = T_rel_gt @ ext_b[i]
    T_A0_B0 = T_rel_gt  # (B, 4, 4)

    rot_errors_list = []
    trans_errors_list = []
    for i in range(W):
        T_B0_Bi = extrinsics_b[:, i]  # (B, 4, 4)
        T_A0_Bi_gt = torch.bmm(T_A0_B0, T_B0_Bi)  # (B, 4, 4)

        R_gt_i = T_A0_Bi_gt[:, :3, :3]
        t_gt_i = T_A0_Bi_gt[:, :3, 3]

        R_pred_i = rm_b[:, i]   # (B, 3, 3)
        t_pred_i = t_b[:, i]    # (B, 3)

        R_diff = torch.bmm(R_pred_i.transpose(-1, -2), R_gt_i)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        rot_err = torch.acos(cos_angle) * (180.0 / math.pi)

        trans_err = torch.norm(t_pred_i - t_gt_i, dim=-1)

        rot_errors_list.append(rot_err)
        trans_errors_list.append(trans_err)

    # (B, W) → mean over views
    all_rot = torch.stack(rot_errors_list, dim=1)    # (B, W)
    all_trans = torch.stack(trans_errors_list, dim=1)  # (B, W)

    return all_rot.mean(dim=1), all_trans.mean(dim=1)


# ============================================================================
# Multi-GPU gather + deduplication
# ============================================================================
def gather_results(
    local_results: list[dict], local_rank: int, world_size: int,
) -> list[dict]:
    if local_rank == -1 or world_size == 1:
        return local_results
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if local_rank == 0:
        all_results = []
        for rank_results in gathered:
            all_results.extend(rank_results)
        return all_results
    return []


def deduplicate_results(results: list[dict], dataset_size: int) -> list[dict]:
    if len(results) <= dataset_size:
        return results
    return results[:dataset_size]


# ============================================================================
# Result saving and summary
# ============================================================================
CSV_COLUMNS = [
    "scene_id", "traj_a", "t_a", "traj_b", "t_b",
    "rot_error_deg", "trans_error_m", "t_dir_error_deg",
    "per_view_rot_mean_deg", "per_view_trans_mean_m",
]


def save_raw_results(results: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "raw_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})
    print(f"[Eval] Saved {csv_path} ({len(results)} rows)")


def load_results_from_csv(output_dir: str) -> list[dict]:
    csv_path = os.path.join(output_dir, "raw_results.csv")
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "scene_id": row["scene_id"],
                "rot_error_deg": float(row["rot_error_deg"]),
                "trans_error_m": float(row["trans_error_m"]),
                "t_dir_error_deg": float(row.get("t_dir_error_deg") or "nan"),
                "per_view_rot_mean_deg": float(row["per_view_rot_mean_deg"]),
                "per_view_trans_mean_m": float(row["per_view_trans_mean_m"]),
            })
    print(f"[Eval] Loaded {len(results)} records from {csv_path}")
    return results


SUCCESS_ROT_THRESH_DEG = 10.0
SUCCESS_TRANS_THRESH_M = 0.5


def print_and_save_summary(results: list[dict], output_dir: str, elapsed_sec: float) -> None:
    if not results:
        print("[Eval] No results to summarize.")
        return

    rot_errs = np.array([r["rot_error_deg"] for r in results])
    trans_errs = np.array([r["trans_error_m"] for r in results])
    pv_rot = np.array([r["per_view_rot_mean_deg"] for r in results])
    pv_trans = np.array([r["per_view_trans_mean_m"] for r in results])

    success_mask = (rot_errs < SUCCESS_ROT_THRESH_DEG) & (trans_errs < SUCCESS_TRANS_THRESH_M)
    success_rate = success_mask.mean() * 100.0

    lines = [
        "=" * 70,
        "G2G Rig Evaluation Summary (anchor-to-anchor relative pose)",
        "=" * 70,
        f"  Total pairs:      {len(results)}",
        f"  t_mean:           {np.mean(trans_errs):.4f} m",
        f"  t_median:         {np.median(trans_errs):.4f} m",
        f"  r_mean:           {np.mean(rot_errs):.2f} deg",
        f"  r_median:         {np.median(rot_errs):.2f} deg",
        f"  Success rate:     {success_rate:.1f}% (r<{SUCCESS_ROT_THRESH_DEG}deg & t<{SUCCESS_TRANS_THRESH_M}m)",
        "",
        "Per-view B-group pose error (all B views relative to A0):",
        f"  pv_rot_mean:      {np.mean(pv_rot):.2f} deg",
        f"  pv_rot_median:    {np.median(pv_rot):.2f} deg",
        f"  pv_trans_mean:    {np.mean(pv_trans):.4f} m",
        f"  pv_trans_median:  {np.median(pv_trans):.4f} m",
        "",
        "Percentiles (anchor-to-anchor):",
        f"  t_25th:           {np.percentile(trans_errs, 25):.4f} m",
        f"  t_75th:           {np.percentile(trans_errs, 75):.4f} m",
        f"  r_25th:           {np.percentile(rot_errs, 25):.2f} deg",
        f"  r_75th:           {np.percentile(rot_errs, 75):.2f} deg",
        "=" * 70,
        f"  Elapsed:          {elapsed_sec:.1f}s",
    ]

    # Per-scene statistics
    scene_stats = {}
    for r in results:
        sid = r["scene_id"]
        if sid not in scene_stats:
            scene_stats[sid] = {"rot": [], "trans": []}
        scene_stats[sid]["rot"].append(r["rot_error_deg"])
        scene_stats[sid]["trans"].append(r["trans_error_m"])

    if len(scene_stats) <= 30:
        lines.append("")
        lines.append("Per-scene breakdown:")
        lines.append(f"  {'Scene':<30} | {'N':>5} | {'r_med(deg)':>10} | {'t_med(m)':>8}")
        lines.append("  " + "-" * 60)
        for sid in sorted(scene_stats):
            s = scene_stats[sid]
            lines.append(
                f"  {sid:<30} | {len(s['rot']):>5} | "
                f"{np.median(s['rot']):>10.2f} | {np.median(s['trans']):>8.4f}"
            )

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n[Eval] Saved {summary_path}")


def plot_error_distributions(results: list[dict], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Eval] matplotlib not available, skipping plots")
        return

    rot_errs = np.array([r["rot_error_deg"] for r in results])
    trans_errs = np.array([r["trans_error_m"] for r in results])

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sorted_rot = np.sort(rot_errs)
    cdf_rot = np.arange(1, len(sorted_rot) + 1) / len(sorted_rot)
    ax1.plot(sorted_rot, cdf_rot, color="#4C72B0", linewidth=2)
    ax1.axvline(x=SUCCESS_ROT_THRESH_DEG, color="red", linestyle="--", linewidth=0.8,
                label=f"Threshold = {SUCCESS_ROT_THRESH_DEG} deg")
    ax1.axvline(x=np.median(rot_errs), color="blue", linestyle="--", linewidth=0.8,
                label=f"Median = {np.median(rot_errs):.2f} deg")
    ax1.set_xlabel("Rotation Error (deg)")
    ax1.set_ylabel("CDF")
    ax1.set_title("Rotation Error CDF")
    ax1.set_xlim(0, min(180, np.percentile(rot_errs, 99) * 1.2))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    sorted_trans = np.sort(trans_errs)
    cdf_trans = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans)
    ax2.plot(sorted_trans, cdf_trans, color="#55A868", linewidth=2)
    ax2.axvline(x=SUCCESS_TRANS_THRESH_M, color="red", linestyle="--", linewidth=0.8,
                label=f"Threshold = {SUCCESS_TRANS_THRESH_M} m")
    ax2.axvline(x=np.median(trans_errs), color="blue", linestyle="--", linewidth=0.8,
                label=f"Median = {np.median(trans_errs):.4f} m")
    ax2.set_xlabel("Translation Error (m)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Translation Error CDF")
    ax2.set_xlim(0, min(10, np.percentile(trans_errs, 99) * 1.2))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("G2G Rig - Pose Estimation Error Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "error_cdf.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(rot_errs, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax1.axvline(x=np.median(rot_errs), color="blue", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(rot_errs):.2f} deg")
    ax1.set_xlabel("Rotation Error (deg)")
    ax1.set_ylabel("Count")
    ax1.set_title("Rotation Error Distribution")
    ax1.set_xlim(0, min(180, np.percentile(rot_errs, 99) * 1.2))
    ax1.legend()

    ax2.hist(trans_errs, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    ax2.axvline(x=np.median(trans_errs), color="blue", linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(trans_errs):.4f} m")
    ax2.set_xlabel("Translation Error (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Translation Error Distribution")
    ax2.set_xlim(0, min(10, np.percentile(trans_errs, 99) * 1.2))
    ax2.legend()

    fig.suptitle("G2G Rig - Pose Estimation Error Histograms", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "error_histogram.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"[Eval] Saved plots to {plots_dir}")


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2G Stage2 Rig validation set evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="Deterministic stride sampling, 10 ≈ 10%%; 1 = full set")
    parser.add_argument("--max-body-dist", type=float, default=-1.0,
                        help="Only evaluate pairs with body_dist <= this value (m); <=0 means no limit. Use 1.5 for 4cam")
    parser.add_argument("--max-pairs-per-scene", type=int, default=-1,
                        help="Take at most N pairs per scene (random, seed42); <=0 means no limit. Use 1000 for ZJH")
    parser.add_argument("--val-key", type=str, default="val",
                        help="Which key under cfg[data] to use as the evaluation set; use val_real for ZJH-real")
    parser.add_argument("--max-scenes", type=int, default=-1,
                        help="Limit the number of evaluation scenes (for debugging)")
    parser.add_argument("--val-data-override", type=str, default="",
                        help="Override the val data path prefix (when a different server's path is needed)")
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    local_rank, world_size = setup_ddp()
    is_main = is_main_process(local_rank)

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_yaml(args.config)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_results")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Eval] Output dir: {output_dir}")

    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data path override (the training config may point at the original training-machine paths, which can be swapped at eval time via --val-data-override)
    data_override = None
    if args.val_data_override:
        prefix = args.val_data_override
        data_override = {
            "step1_root": os.path.join(prefix, "step1_generate_trajectories_val"),
            "step2_root": os.path.join(prefix, "step2_rig8_configs_val"),
            "step3_root": os.path.join(prefix, "step3_render_rig8_224_val"),
            "step6_root": os.path.join(prefix, "step6_dinov2_features_rig8_val"),
            "index_root": os.path.join(prefix, "step7_rig_index_val_ms"),
        }

    t_start = time.time()

    if args.skip_inference:
        if is_main:
            results = load_results_from_csv(output_dir)
        else:
            results = []
    else:
        # Load the model
        if local_rank <= 0:
            model = build_and_load_model(cfg, args.checkpoint, device, is_main=is_main)
        if local_rank != -1:
            dist.barrier()
        if local_rank > 0:
            model = build_and_load_model(cfg, args.checkpoint, device, is_main=False)
        if local_rank != -1:
            dist.barrier()

        # Build the DataLoader
        loader, dataset = build_val_loader(
            cfg, args.batch_size, args.num_workers,
            local_rank, world_size, args.max_scenes,
            data_override=data_override,
            sample_stride=args.sample_stride,
            max_body_dist=args.max_body_dist,
            max_pairs_per_scene=args.max_pairs_per_scene,
            val_key=args.val_key,
        )
        dataset_size = len(dataset)

        try:
            local_results = run_inference(model, loader, device, is_main=is_main)
            results = gather_results(local_results, local_rank, world_size)
            if is_main:
                results = deduplicate_results(results, dataset_size)
                print(f"[Eval] Gathered {len(results)} unique results (dataset: {dataset_size})")
                save_raw_results(results, output_dir)
        finally:
            del model
            torch.cuda.empty_cache()

    t_done = time.time()

    if is_main:
        print_and_save_summary(results, output_dir, t_done - t_start)
        plot_error_distributions(results, output_dir)
        print(f"\n[Eval] Total time: {time.time() - t_start:.1f}s")
        print(f"[Eval] All results saved to: {output_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()

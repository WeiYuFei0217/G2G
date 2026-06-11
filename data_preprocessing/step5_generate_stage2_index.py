#!/usr/bin/env python3
"""
step5_generate_stage2_index.py -- offline generation of the G2G training data index

Based on the overlap matrix predicted by the covisibility model (or the GT
overlap matrix), performs sliding-window Top-K selection for each sequence pair and
generates the data index required for G2G training.

Processing flow:
  For each scene:
    Load view_sequence_pairs.json (from step4)
    For each sequence pair (traj_a/cam_a, traj_b/cam_b):
      1. Load all 224x224 images of sequences A/B
      2. engine.encode_sequence(images_a, key_a)
      3. engine.encode_sequence(images_b, key_b)
      4. pred_overlap = engine.predict_overlap_matrix(key_a, key_b)
      5. windows = select_windows_with_union(pred_overlap, ...)
      6. [optional] covis_maps = engine.predict_covisibility_maps(...)
      7. Save covisibility maps as .npz
      8. Record window indices into stage2_index
      9. engine.clear_cache()
    Save stage2_index.json

Usage:
  # Use the covisibility predicted overlap (single GPU)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split val --gpu 0

  # Use the covisibility predicted overlap (with multi-threaded image loading)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split val --gpu 0 --num-data-workers 4

  # Use GT overlap (upper-bound experiment)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split val --mode gt --max-scenes 5

  # Hybrid mode: GT prefilter + covisibility refinement (~25x speedup)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split val --mode hybrid --gpu 0

  # Skip precomputing covisibility maps (faster)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split train --no-precompute-covis

  # Multi-GPU sharding (4 GPUs in parallel, each worker processes 1/4 of the scenes)
  python data_preprocessing/step5_generate_stage2_index.py \
      --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
      --split train --gpu 0 --num-workers 4 --worker-id 0
  # Or use the launch script:
  bash data_preprocessing/run_step5_gt_multi_worker.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import yaml
from PIL import Image

# ImageNet normalization parameters (consistent with the DINOv2 training dataset)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Depends on the g2g package installed via `pip install -e .` in this repository.
from g2g.retrieval.stage1_inference import (
    Stage1InferenceEngine,
    select_windows_with_union,
)
from g2g.retrieval.utils import dequantize_overlap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def _load_single_image(
    img_path: str,
    img_size: tuple[int, int],
) -> np.ndarray:
    """
    Load and preprocess a single image (thread-safe, called by ThreadPoolExecutor).

    Args:
        img_path: image file path (already confirmed to exist)
        img_size: target size (H, W)

    Returns:
        [3, H, W] float32 array, ImageNet normalized
    """
    img = Image.open(img_path).convert("RGB")
    if img.size != (img_size[1], img_size[0]):
        img = img.resize((img_size[1], img_size[0]), Image.Resampling.BILINEAR)

    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = img_arr.transpose(2, 0, 1)  # HWC -> CHW
    img_arr = (img_arr - IMAGENET_MEAN) / IMAGENET_STD
    return img_arr


def load_sequence_images(
    step3_root: str,
    scene_id: str,
    traj_name: str,
    cam_idx: int,
    num_frames: int,
    step1_root: str,
    img_size: tuple[int, int] = (224, 224),
    num_data_workers: int = 1,
) -> torch.Tensor:
    """
    Load all frames of a view sequence (used for covisibility inference).

    Args:
        step3_root: step3 render output root directory (224x224 images)
        scene_id: scene ID
        traj_name: trajectory name
        cam_idx: camera index
        num_frames: number of frames
        step1_root: step1 trajectory output root directory (used to read timestamps)
        img_size: target image size
        num_data_workers: number of image-loading threads (>1 enables multi-threaded parallelism)

    Returns:
        [T, 3, H, W] float32 tensor, ImageNet normalized
    """
    # Read trajectory timestamps
    tum_path = os.path.join(
        step1_root, "scenes", scene_id, "trajectories", traj_name,
        "trajectory.tum",
    )
    timestamps = _load_timestamps(tum_path)

    # Collect the image path for each frame (or None to indicate missing)
    frame_paths: list[str | None] = []
    actual_frames = min(num_frames, len(timestamps))
    if num_frames > len(timestamps):
        logger.warning(
            "num_frames (%d) exceeds trajectory length (%d) for %s/%s, "
            "will load %d frames",
            num_frames, len(timestamps), scene_id, traj_name, actual_frames,
        )

    for frame_idx in range(actual_frames):
        ts_str = f"{int(timestamps[frame_idx] * 1000):010d}"
        img_path = os.path.join(
            step3_root, "scenes", scene_id, "trajectories", traj_name,
            "images", ts_str, f"cam_{cam_idx}.jpg",
        )

        if not os.path.isfile(img_path):
            # Try png format
            img_path_png = img_path.replace(".jpg", ".png")
            if os.path.isfile(img_path_png):
                img_path = img_path_png
            else:
                logger.warning("Image not found: %s", img_path)
                frame_paths.append(None)
                continue

        frame_paths.append(img_path)

    zero_img = np.zeros((3, img_size[0], img_size[1]), dtype=np.float32)

    # Multi-threaded parallel loading
    if num_data_workers > 1 and actual_frames > 1:
        images: list[np.ndarray] = [zero_img] * actual_frames
        # Only submit frames with valid paths
        with ThreadPoolExecutor(max_workers=num_data_workers) as executor:
            futures = {}
            for idx, path in enumerate(frame_paths):
                if path is not None:
                    fut = executor.submit(_load_single_image, path, img_size)
                    futures[fut] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                images[idx] = fut.result()
    else:
        # Single-threaded sequential loading
        images = []
        for path in frame_paths:
            if path is None:
                images.append(zero_img)
            else:
                images.append(_load_single_image(path, img_size))

    return torch.from_numpy(np.stack(images, axis=0))


def _load_timestamps(tum_path: str) -> list[float]:
    """Load the list of timestamps from a TUM file."""
    timestamps = []
    with open(tum_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
    return timestamps


# ---------------------------------------------------------------------------
# GT overlap loading
# ---------------------------------------------------------------------------
def load_gt_overlap_matrix(
    step4_root: str,
    scene_id: str,
    matrix_file: str | None,
) -> np.ndarray | None:
    """
    Load the GT overlap matrix (from step4).

    Args:
        step4_root: step4 output root directory
        scene_id: scene ID
        matrix_file: relative path (e.g. "matrices/pair_000000.npz") or None

    Returns:
        [T_a, T_b] float32 matrix, or None (if it cannot be loaded)
    """
    if matrix_file is None:
        return None

    # matrix_file is a path relative to the scene directory
    npz_path = os.path.join(
        step4_root, "scenes", scene_id, matrix_file,
    )
    if not os.path.isfile(npz_path):
        logger.warning("GT overlap file not found: %s", npz_path)
        return None

    data = np.load(npz_path)
    if "symmetric" in data:
        return dequantize_overlap(data["symmetric"])
    elif "a2b" in data and "b2a" in data:
        a2b = dequantize_overlap(data["a2b"])
        b2a = dequantize_overlap(data["b2a"])
        return np.minimum(a2b, b2a)
    else:
        logger.warning("Unexpected npz keys in %s: %s", npz_path, list(data.keys()))
        return None


# ---------------------------------------------------------------------------
# Hybrid mode: GT prefilter + covisibility refinement
# ---------------------------------------------------------------------------
def process_pair_hybrid(
    engine: Stage1InferenceEngine,
    key_a: str,
    key_b: str,
    gt_overlap: np.ndarray,
    window_cfg: dict,
    gt_prefilter_cfg: dict,
) -> tuple[list[dict], dict]:
    """
    Hybrid-mode processing of a single pair: GT prefilter for Top-K candidate windows
    -> covisibility decoder refinement.

    Algorithm:
      1. Use GT overlap to find top-K candidate windows (loose threshold)
      2. Collect and deduplicate all window frame pairs (~80-100 pairs vs ~2500 pairs in full mode)
      3. Run the covisibility decoder only on the deduplicated frame pairs
      4. Re-rank and filter by covisibility score

    Args:
        engine: Stage1InferenceEngine (encoder features for key_a, key_b already cached)
        key_a: cache key for sequence A
        key_b: cache key for sequence B
        gt_overlap: [T_a, T_b] float32 GT overlap matrix
        window_cfg: window config (size, top_k, stride, score_type, min_distance, min_overlap_threshold)
        gt_prefilter_cfg: GT prefilter config (candidate_k, min_overlap_threshold)

    Returns:
        (windows_list, submatrices_dict)
        - windows_list: list of windows sorted by covisibility score
        - submatrices_dict: {window_idx_gt: ..., window_idx_stage1: ..., ...} used to save npz
    """
    from g2g.retrieval.utils import find_topk_windows

    window_size = window_cfg.get("size", 5)
    top_k = window_cfg.get("top_k", 10)
    stride = window_cfg.get("stride", 1)
    score_type = window_cfg.get("score_type", "max_mean")
    min_distance = window_cfg.get("min_distance", 2)
    min_overlap_threshold = window_cfg.get("min_overlap_threshold", 0.1)
    max_overlap_threshold = window_cfg.get("max_overlap_threshold", 1.0)

    gt_candidate_k = gt_prefilter_cfg.get("candidate_k", 5)
    gt_min_overlap = gt_prefilter_cfg.get("min_overlap_threshold", 0.05)

    T_a, T_b = gt_overlap.shape

    # Step 1: GT prefilter. Find candidate windows with a loose threshold
    gt_candidates = find_topk_windows(
        gt_overlap,
        window_size=window_size,
        stride=stride,
        k=gt_candidate_k,
        score_type=score_type,
        min_distance=min_distance,
    )

    # Filter out candidates whose GT score is too low
    gt_candidates = [
        (sa, sb, sc) for sa, sb, sc in gt_candidates
        if sc >= gt_min_overlap
    ]

    if not gt_candidates:
        return [], {}

    # Step 2: Collect and deduplicate the frame pairs of all candidate windows
    unique_pairs: set[tuple[int, int]] = set()
    candidate_windows = []

    for start_a, start_b, gt_score in gt_candidates:
        end_a = min(start_a + window_size, T_a)
        end_b = min(start_b + window_size, T_b)
        indices_a = list(range(start_a, end_a))
        indices_b = list(range(start_b, end_b))

        for fa in indices_a:
            for fb in indices_b:
                unique_pairs.add((fa, fb))

        candidate_windows.append({
            "start_a": start_a,
            "start_b": start_b,
            "indices_a": indices_a,
            "indices_b": indices_b,
            "gt_score": gt_score,
        })

    # Step 3: covisibility decoder refinement. Run inference only on the deduplicated frame pairs
    pair_overlaps = engine.predict_overlap_for_pairs(
        key_a, key_b, sorted(unique_pairs),
    )

    # Step 4: Rebuild the submatrix for each window and compute its covisibility score
    submatrices_dict = {}
    for w_idx, win in enumerate(candidate_windows):
        ia = win["indices_a"]
        ib = win["indices_b"]
        h, w = len(ia), len(ib)

        # GT submatrix
        gt_sub = gt_overlap[
            win["start_a"]:win["start_a"] + h,
            win["start_b"]:win["start_b"] + w,
        ]

        # covisibility submatrix
        s1_sub = np.zeros((h, w), dtype=np.float32)
        for ri, fa in enumerate(ia):
            for ci, fb in enumerate(ib):
                s1_sub[ri, ci] = pair_overlaps.get((fa, fb), 0.0)

        # Compute the covisibility window score (max_mean)
        if s1_sub.size > 0:
            a2b = float(np.mean(np.max(s1_sub, axis=1)))
            b2a = float(np.mean(np.max(s1_sub, axis=0)))
            stage1_score = (a2b + b2a) / 2.0
        else:
            stage1_score = 0.0

        win["stage1_score"] = stage1_score

        # Save submatrix data
        submatrices_dict[f"window_{w_idx}_gt"] = gt_sub.copy()
        submatrices_dict[f"window_{w_idx}_stage1"] = s1_sub
        submatrices_dict[f"window_{w_idx}_indices_a"] = np.array(ia, dtype=np.int32)
        submatrices_dict[f"window_{w_idx}_indices_b"] = np.array(ib, dtype=np.int32)

    # Step 5: Re-rank by covisibility score
    candidate_windows.sort(key=lambda x: x["stage1_score"], reverse=True)

    # Step 6: Filter and build the final window list
    windows_list = []
    for rank, win in enumerate(candidate_windows):
        s1_score = win["stage1_score"]

        # Filter: score < min_overlap_threshold and no high value within the window
        if s1_score < min_overlap_threshold:
            # Check whether the covisibility submatrix contains any high-value frame pair
            w_idx_key = f"window_{candidate_windows.index(win)}_stage1"
            # Check directly via pair_overlaps
            has_high = False
            for fa in win["indices_a"]:
                for fb in win["indices_b"]:
                    if pair_overlaps.get((fa, fb), 0.0) >= 3.0 * min_overlap_threshold:
                        has_high = True
                        break
                if has_high:
                    break
            if not has_high:
                continue

        # Upper-bound filter
        if max_overlap_threshold < 1.0 and s1_score >= max_overlap_threshold:
            continue

        if rank >= top_k:
            break

        windows_list.append({
            "rank": rank,
            "indices_a": win["indices_a"],
            "indices_b": win["indices_b"],
            "score": round(s1_score, 4),
            "gt_score": round(win["gt_score"], 4),
            "stage1_score": round(s1_score, 4),
            "has_covis_maps": False,
        })

    return windows_list, submatrices_dict


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------
def process_scene(
    scene_id: str,
    scene_dir: str,
    engine: Stage1InferenceEngine | None,
    cfg: dict,
    args: argparse.Namespace,
    output_scene_dir: str,
) -> dict | None:
    """
    Process one scene: perform window selection and (optionally) covisibility
    precomputation for all sequence pairs.

    Args:
        scene_id: scene ID
        scene_dir: step4 scene directory (contains view_sequence_pairs.json)
        engine: Stage1InferenceEngine (may be None when use_gt_overlap)
        cfg: config dictionary
        args: command-line arguments
        output_scene_dir: output directory

    Returns:
        scene result dictionary, or None (on failure)
    """
    pairs_json_path = os.path.join(scene_dir, "view_sequence_pairs.json")
    if not os.path.isfile(pairs_json_path):
        logger.warning("view_sequence_pairs.json not found for %s", scene_id)
        return None

    with open(pairs_json_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    # Window parameters
    window_cfg = cfg.get("window", {})
    window_size = window_cfg.get("size", 5)
    top_k = window_cfg.get("top_k", 3)
    stride = window_cfg.get("stride", 1)
    score_type = window_cfg.get("score_type", "max_mean")
    min_distance = window_cfg.get("min_distance", 2)
    min_overlap_threshold = window_cfg.get("min_overlap_threshold", 0.1)
    max_overlap_threshold = window_cfg.get("max_overlap_threshold", 1.0)

    # Path configuration
    paths = cfg.get("paths", {})
    split = args.split
    step1_root = paths.get(f"step1_root_{split}", "")
    step3_root_224 = paths.get(f"step3_root_224_{split}", "")
    step4_root = paths.get(f"step4_root_{split}", "")

    precompute_covis = cfg.get("precompute_covisibility", True)
    if args.no_precompute_covis:
        precompute_covis = False

    # Determine the run mode
    mode = getattr(args, "_resolved_mode", "stage1")

    # Hybrid mode configuration
    gt_prefilter_cfg = cfg.get("gt_prefilter", {})
    save_overlap_submatrices = cfg.get("save_overlap_submatrices", False)

    # Output directory
    os.makedirs(output_scene_dir, exist_ok=True)
    overlap_mat_dir = os.path.join(output_scene_dir, "overlap_matrices")
    os.makedirs(overlap_mat_dir, exist_ok=True)
    if precompute_covis and mode == "stage1":
        covis_dir = os.path.join(output_scene_dir, "covis_maps")
        os.makedirs(covis_dir, exist_ok=True)
    if save_overlap_submatrices and mode == "hybrid":
        submat_dir = os.path.join(output_scene_dir, "overlap_submatrices")
        os.makedirs(submat_dir, exist_ok=True)

    pairs_result = []
    pairs = scene_data.get("pairs", [])

    for pair_idx, pair_info in enumerate(pairs):
        # pair_id is a string such as "pair_000000" in the JSON, but may also be an integer
        raw_pair_id = pair_info.get("pair_id", pair_idx)
        # Normalize to a string identifier (used for file names)
        if isinstance(raw_pair_id, int):
            pair_id_str = f"pair_{raw_pair_id:06d}"
        else:
            pair_id_str = str(raw_pair_id)

        traj_a = pair_info["seq_a"]["traj"]
        cam_a = pair_info["seq_a"]["cam"]
        num_frames_a = pair_info["seq_a"]["num_frames"]
        traj_b = pair_info["seq_b"]["traj"]
        cam_b = pair_info["seq_b"]["cam"]
        num_frames_b = pair_info["seq_b"]["num_frames"]
        matrix_file = pair_info.get("matrix_file")

        key_a = f"{scene_id}/{traj_a}/cam_{cam_a}"
        key_b = f"{scene_id}/{traj_b}/cam_{cam_b}"

        # ============================
        # Hybrid mode
        # ============================
        if mode == "hybrid":
            if engine is None:
                raise RuntimeError("engine is None in hybrid mode")

            # Load GT overlap (for prefiltering)
            gt_overlap = load_gt_overlap_matrix(
                step4_root, scene_id, matrix_file,
            )
            if gt_overlap is None:
                logger.info(
                    "Skipping pair %s (no GT overlap for hybrid): %s/%s -> %s/%s",
                    pair_id_str, traj_a, cam_a, traj_b, cam_b,
                )
                continue

            # Encode the sequence (if not cached; the encoder is fast ~150ms/sequence and reused across pairs)
            num_data_workers = getattr(args, "num_data_workers", 1)
            if key_a not in engine.cached_keys:
                images_a = load_sequence_images(
                    step3_root_224, scene_id, traj_a, cam_a,
                    num_frames_a, step1_root,
                    img_size=(224, 224),
                    num_data_workers=num_data_workers,
                )
                engine.encode_sequence(images_a, key_a)

            if key_b not in engine.cached_keys:
                images_b = load_sequence_images(
                    step3_root_224, scene_id, traj_b, cam_b,
                    num_frames_b, step1_root,
                    img_size=(224, 224),
                    num_data_workers=num_data_workers,
                )
                engine.encode_sequence(images_b, key_b)

            # Hybrid processing: GT prefilter + covisibility refinement
            windows_entry, submatrices = process_pair_hybrid(
                engine=engine,
                key_a=key_a,
                key_b=key_b,
                gt_overlap=gt_overlap,
                window_cfg=window_cfg,
                gt_prefilter_cfg=gt_prefilter_cfg,
            )

            # Save the GT overlap matrix (for analysis)
            overlap_save_path = os.path.join(
                overlap_mat_dir, f"{pair_id_str}.npy",
            )
            np.save(overlap_save_path, gt_overlap)

            # Save the overlap submatrices (GT vs covisibility)
            if save_overlap_submatrices and submatrices:
                submat_path = os.path.join(
                    submat_dir, f"{pair_id_str}.npz",
                )
                np.savez_compressed(submat_path, **submatrices)

        # ============================
        # GT mode
        # ============================
        elif mode == "gt":
            overlap_matrix = load_gt_overlap_matrix(
                step4_root, scene_id, matrix_file,
            )
            if overlap_matrix is None:
                logger.info(
                    "Skipping pair %s (no GT overlap): %s/%s -> %s/%s",
                    pair_id_str, traj_a, cam_a, traj_b, cam_b,
                )
                continue

            # Save the overlap matrix
            overlap_save_path = os.path.join(
                overlap_mat_dir, f"{pair_id_str}.npy",
            )
            np.save(overlap_save_path, overlap_matrix)

            # Sliding-window selection
            window_result = select_windows_with_union(
                overlap_matrix,
                window_size=window_size,
                top_k=top_k,
                stride=stride,
                score_type=score_type,
                min_distance=min_distance,
                min_overlap_threshold=min_overlap_threshold,
                max_overlap_threshold=max_overlap_threshold,
            )

            windows_entry = []
            for win in window_result["windows"]:
                windows_entry.append({
                    "rank": win["rank"],
                    "indices_a": win["indices_a"],
                    "indices_b": win["indices_b"],
                    "score": round(win["score"], 4),
                    "has_covis_maps": False,
                })

        # ============================
        # covisibility mode (full decoder)
        # ============================
        else:
            if engine is None:
                raise RuntimeError("engine is None but mode is stage1")

            # Load and encode the sequence (if not cached)
            num_data_workers = getattr(args, "num_data_workers", 1)
            if key_a not in engine.cached_keys:
                images_a = load_sequence_images(
                    step3_root_224, scene_id, traj_a, cam_a,
                    num_frames_a, step1_root,
                    img_size=(224, 224),
                    num_data_workers=num_data_workers,
                )
                engine.encode_sequence(images_a, key_a)

            if key_b not in engine.cached_keys:
                images_b = load_sequence_images(
                    step3_root_224, scene_id, traj_b, cam_b,
                    num_frames_b, step1_root,
                    img_size=(224, 224),
                    num_data_workers=num_data_workers,
                )
                engine.encode_sequence(images_b, key_b)

            # Predict the overlap matrix
            overlap_matrix = engine.predict_overlap_matrix(key_a, key_b)

            # Save the predicted overlap matrix
            overlap_save_path = os.path.join(
                overlap_mat_dir, f"{pair_id_str}.npy",
            )
            np.save(overlap_save_path, overlap_matrix)

            # Sliding-window selection
            window_result = select_windows_with_union(
                overlap_matrix,
                window_size=window_size,
                top_k=top_k,
                stride=stride,
                score_type=score_type,
                min_distance=min_distance,
                min_overlap_threshold=min_overlap_threshold,
                max_overlap_threshold=max_overlap_threshold,
            )

            # Optional: precompute per-pixel covisibility maps
            has_covis = False
            if precompute_covis and engine is not None:
                union_pairs = window_result["union_pairs"]
                if union_pairs:
                    covis_maps = engine.predict_covisibility_maps(
                        key_a, key_b, union_pairs,
                    )

                    for (fa, fb), maps in covis_maps.items():
                        covis_save_path = os.path.join(
                            covis_dir,
                            f"{pair_id_str}_fa{fa:04d}_fb{fb:04d}.npz",
                        )
                        np.savez_compressed(
                            covis_save_path,
                            covis_a=maps["covis_a"].astype(np.float16),
                            covis_b=maps["covis_b"].astype(np.float16),
                            overlap=np.float32(maps["overlap"]),
                        )

                    has_covis = True

            windows_entry = []
            for win in window_result["windows"]:
                windows_entry.append({
                    "rank": win["rank"],
                    "indices_a": win["indices_a"],
                    "indices_b": win["indices_b"],
                    "score": round(win["score"], 4),
                    "has_covis_maps": has_covis,
                })

        # GT overlap file: store a path relative to the scene directory (portable)
        gt_overlap_rel = None
        if matrix_file is not None:
            gt_path = os.path.join(
                step4_root, "scenes", scene_id, matrix_file,
            )
            if os.path.isfile(gt_path):
                gt_overlap_rel = matrix_file

        pair_entry = {
            "pair_id": pair_id_str,
            "traj_a": traj_a,
            "cam_a": cam_a,
            "traj_b": traj_b,
            "cam_b": cam_b,
            "num_frames_a": num_frames_a,
            "num_frames_b": num_frames_b,
            "windows": windows_entry,
            "predicted_overlap_file": f"overlap_matrices/{pair_id_str}.npy",
            "gt_overlap_file": gt_overlap_rel,
        }
        pairs_result.append(pair_entry)

    # Clear the encoder cache for this scene
    if engine is not None:
        engine.clear_cache()

    # Save the scene index
    scene_index = {
        "scene_id": scene_id,
        "config": {
            "stage1_checkpoint": args.stage1_checkpoint or cfg.get("stage1", {}).get("checkpoint", ""),
            "window_size": window_size,
            "top_k": top_k,
            "mode": mode,
        },
        "pairs": pairs_result,
    }

    index_path = os.path.join(output_scene_dir, "stage2_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(scene_index, f, indent=2, ensure_ascii=False)

    return scene_index


# ---------------------------------------------------------------------------
# covisibility model loading
# ---------------------------------------------------------------------------
def load_stage1_model(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load the covisibility model.

    Args:
        cfg: config dictionary (must contain stage1.config pointing to the covisibility training config)
        checkpoint_path: decoder checkpoint path
        device: inference device
    """
    # The covisibility model is bring-your-own: its code and
    # weights are NOT shipped in this open-source release. The default --mode gt
    # path (ground-truth overlap) needs no covisibility model. To use --mode stage1/hybrid,
    # construct your own covisibility model here and load your checkpoint.
    raise NotImplementedError(
        "covisibility model is not shipped in this open-source "
        "release. Use --mode gt (ground-truth overlap), or implement load_stage1_model() "
        "to construct and return your own covisibility model."
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def compute_summary(
    scene_results: list[dict],
    output_root: str,
) -> dict:
    """Compute and save the global summary statistics."""
    total_pairs = 0
    total_windows = 0
    all_scores = []

    for scene_res in scene_results:
        for pair in scene_res.get("pairs", []):
            total_pairs += 1
            windows = pair.get("windows", [])
            total_windows += len(windows)
            for w in windows:
                all_scores.append(w["score"])

    summary = {
        "num_scenes": len(scene_results),
        "num_pairs": total_pairs,
        "num_windows": total_windows,
        "avg_windows_per_pair": total_windows / max(total_pairs, 1),
        "avg_overlap_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "median_overlap_score": float(np.median(all_scores)) if all_scores else 0.0,
        "min_overlap_score": float(np.min(all_scores)) if all_scores else 0.0,
        "max_overlap_score": float(np.max(all_scores)) if all_scores else 0.0,
    }

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate G2G training data index from overlap predictions",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="G2G index generation config YAML",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Data split to process",
    )
    parser.add_argument(
        "--stage1-checkpoint", type=str, default=None,
        help="Override covisibility decoder checkpoint path",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["gt", "stage1", "hybrid"],
        help="Index generation mode: gt (fast, GT overlap), stage1 (accurate, full decoder), "
             "hybrid (GT prefilter + covisibility refinement, ~25x faster than stage1)",
    )
    parser.add_argument(
        "--use-gt-overlap", action="store_true",
        help="[Deprecated: use --mode gt] Use GT overlap matrices (backward compat)",
    )
    parser.add_argument(
        "--no-precompute-covis", action="store_true",
        help="Skip precomputing per-pixel covisibility maps",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=-1,
        help="Max scenes to process (-1 for all)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Override output root directory",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip scenes that already have stage2_index.json",
    )
    parser.add_argument(
        "--scene-offset", type=int, default=0,
        help="Skip the first N scenes after sorting and start from the N-th scene (default: 0)",
    )
    # Multi-GPU scene sharding parameters
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Total number of parallel workers for multi-GPU sharding (default: 1)",
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="Current worker ID (0-indexed) for multi-GPU sharding (default: 0)",
    )
    # Image-loading parallelization parameters
    parser.add_argument(
        "--num-data-workers", type=int, default=4,
        help="Number of threads for parallel image loading (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve the run mode: --mode takes priority, --use-gt-overlap is a backward-compatible alias
    if args.mode is not None:
        resolved_mode = args.mode
    elif args.use_gt_overlap:
        resolved_mode = "gt"
    else:
        resolved_mode = "stage1"

    # Backward compatibility: set use_gt_overlap so legacy code paths can use it
    args.use_gt_overlap = (resolved_mode == "gt")
    # Store the resolved mode for process_scene to use
    args._resolved_mode = resolved_mode

    logger.info("Mode: %s", resolved_mode)

    # Load the config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Determine paths
    paths = cfg.get("paths", {})
    split = args.split
    step4_root = paths.get(f"step4_root_{split}", "")
    output_root = args.output_root or paths.get(f"output_root_{split}", "")

    if not output_root:
        raise ValueError("output_root not specified (via --output-root or config)")

    # Enumerate scenes
    step4_scenes_dir = os.path.join(step4_root, "scenes")
    if not os.path.isdir(step4_scenes_dir):
        raise FileNotFoundError(f"step4 scenes directory not found: {step4_scenes_dir}")

    scene_names = sorted(
        d for d in os.listdir(step4_scenes_dir)
        if os.path.isdir(os.path.join(step4_scenes_dir, d))
    )

    # Skip the first N scenes (used for multi-machine division of work, avoiding duplicate processing)
    if args.scene_offset > 0:
        total_before_offset = len(scene_names)
        scene_names = scene_names[args.scene_offset:]
        logger.info(
            "Scene offset %d: skipped first %d scenes, remaining %d",
            args.scene_offset, total_before_offset - len(scene_names), len(scene_names),
        )

    if args.max_scenes > 0:
        scene_names = scene_names[:args.max_scenes]

    # Multi-GPU scene sharding: shard the scene list by worker_id via modulo
    if args.num_workers > 1:
        total_before = len(scene_names)
        scene_names = scene_names[args.worker_id :: args.num_workers]
        logger.info(
            "Worker %d/%d: assigned %d/%d scenes",
            args.worker_id, args.num_workers, len(scene_names), total_before,
        )

    logger.info("Found %d scenes to process", len(scene_names))

    # Initialize the covisibility inference engine (required for stage1/hybrid modes)
    engine = None
    if resolved_mode in ("stage1", "hybrid"):
        stage1_ckpt = args.stage1_checkpoint or cfg.get("stage1", {}).get("checkpoint", "")
        model = load_stage1_model(cfg, stage1_ckpt, device)

        inf_cfg = cfg.get("inference", {})
        engine = Stage1InferenceEngine(
            model=model,
            device=device,
            encoder_batch_size=inf_cfg.get("encoder_batch_size", 32),
            decoder_batch_size=inf_cfg.get("decoder_batch_size", 64),
            use_amp=inf_cfg.get("use_amp", True),
        )
        logger.info("covisibility inference engine initialized (mode=%s)", resolved_mode)
    else:
        logger.info("Using GT overlap matrices (mode=gt)")

    # Process each scene
    output_scenes_dir = os.path.join(output_root, "scenes")
    os.makedirs(output_scenes_dir, exist_ok=True)

    scene_results = []
    total_start = time.time()

    for idx, scene_name in enumerate(scene_names):
        scene_start = time.time()
        scene_dir = os.path.join(step4_scenes_dir, scene_name)
        output_scene_dir = os.path.join(output_scenes_dir, scene_name)

        # Check whether it is already done
        if args.skip_existing:
            existing_index = os.path.join(output_scene_dir, "stage2_index.json")
            if os.path.isfile(existing_index):
                logger.info("[%d/%d] SKIP %s (already exists)", idx + 1, len(scene_names), scene_name)
                # Load the existing result for the summary
                with open(existing_index, "r", encoding="utf-8") as f:
                    scene_results.append(json.load(f))
                continue

        # Get scene_id from view_sequence_pairs.json
        pairs_json = os.path.join(scene_dir, "view_sequence_pairs.json")
        if not os.path.isfile(pairs_json):
            logger.info("[%d/%d] SKIP %s (no pairs json)", idx + 1, len(scene_names), scene_name)
            continue

        with open(pairs_json, "r", encoding="utf-8") as f:
            scene_data = json.load(f)

        scene_id = scene_data.get("scene_id", scene_name)
        num_pairs = len(scene_data.get("pairs", []))

        logger.info("[%d/%d] Processing %s (%d pairs)...",
                    idx + 1, len(scene_names), scene_name, num_pairs)

        result = process_scene(
            scene_id=scene_id,
            scene_dir=scene_dir,
            engine=engine,
            cfg=cfg,
            args=args,
            output_scene_dir=output_scene_dir,
        )

        if result is not None:
            scene_results.append(result)
            num_windows = sum(
                len(p.get("windows", [])) for p in result.get("pairs", [])
            )
            elapsed = time.time() - scene_start
            logger.info("  -> %d pairs, %d windows, %.1fs",
                        len(result['pairs']), num_windows, elapsed)
        else:
            logger.warning("  -> FAILED")

    # Summary statistics
    total_elapsed = time.time() - total_start
    summary = compute_summary(scene_results, output_root)

    logger.info("")
    logger.info("=" * 60)
    worker_tag = ""
    if args.num_workers > 1:
        worker_tag = f" (Worker {args.worker_id}/{args.num_workers})"
    logger.info("G2G Index Generation Complete%s [mode=%s]", worker_tag, resolved_mode)
    logger.info("=" * 60)
    logger.info("Mode: %s", resolved_mode)
    logger.info("Scenes: %d", summary['num_scenes'])
    logger.info("Pairs: %d", summary['num_pairs'])
    logger.info("Windows: %d", summary['num_windows'])
    logger.info("Avg windows/pair: %.2f", summary['avg_windows_per_pair'])
    logger.info("Avg overlap score: %.4f", summary['avg_overlap_score'])
    logger.info("Median overlap score: %.4f", summary['median_overlap_score'])
    logger.info("Total time: %.1fs", total_elapsed)
    logger.info("Output: %s", output_root)


if __name__ == "__main__":
    main()

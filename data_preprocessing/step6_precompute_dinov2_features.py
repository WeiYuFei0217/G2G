#!/usr/bin/env python3
"""
step6_precompute_dinov2_features.py -- Precompute and cache DINOv2 encoder features

G2G training profiling shows that DINOv2 encoder inference accounts for ~50% of the forward pass.
The DINOv2 output only depends on the input image (not on extrinsics/intrinsics/ray_dirs), so it can be precomputed and cached.

This script uses MapAnything's internal DINOv2 encoder (_encode_n_views) to extract features,
ensuring the weights and inference path are exactly identical to those used during training.

Output directory structure (mirrors step3):
    step6_dinov2_features_224_{train,val}/
        scenes/{scene_id}/trajectories/{traj_id}/
            features/{timestamp_ms}/
                cam_0.npy   # [1024, 16, 16] uint16 (bf16 raw bits), 512 KB
                cam_1.npy
                ...
                cam_7.npy

bfloat16 storage:
    NumPy does not support the bf16 dtype, so the bf16 raw bits are stored as uint16:
    - Save: tensor.view(torch.uint16).cpu().numpy() -> np.save(path, arr)
    - Load: torch.from_numpy(np.load(path)).view(torch.bfloat16)
    This guarantees bit-identical results, with no bf16<->fp16 precision loss.

Usage:
    # Single GPU (val, small dataset)
    python data_preprocessing/step6_precompute_dinov2_features.py \
        --step3-root /path/to/data/HM3D/DATA_GEN/step3_render_rgb_depth_images_224_56_val \
        --output-root /path/to/data/HM3D/DATA_GEN/step6_dinov2_features_224_val \
        --model-path /path/to/map-anything-model/ \
        --gpu 0

    # Multi-GPU sharding (modulo by scene)
    python data_preprocessing/step6_precompute_dinov2_features.py \
        --step3-root ... --output-root ... --model-path ... \
        --gpu 0 --num-workers 4 --worker-id 0

    # Resume / skip-existing
    python data_preprocessing/step6_precompute_dinov2_features.py \
        ... --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Depends on the installed mapanything package (pip install -e ./map-anything).


# ImageNet normalization parameters (consistent with Stage2ModelMultiFrame._normalize_images)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_and_preprocess_images(
    img_paths: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Load RGB images and apply ImageNet normalization.

    Args:
        img_paths: list of image file paths
        device: target device

    Returns:
        [N, 3, H, W] float32 tensor after ImageNet normalization
    """
    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        images.append(arr.transpose(2, 0, 1))  # CHW

    batch = torch.from_numpy(np.stack(images))  # [N, 3, H, W]
    # ImageNet normalization
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    batch = batch.to(device)
    batch = (batch - mean) / std
    return batch


def save_bf16_as_uint16(tensor: torch.Tensor, path: str) -> None:
    """
    Save the tensor as a uint16 npy file holding the bf16 raw bits.

    Under autocast some layers (e.g. LayerNorm) may output float32,
    so we explicitly cast to bf16 before saving to keep the storage format consistent.
    """
    tensor_bf16 = tensor.to(torch.bfloat16)
    arr = tensor_bf16.view(torch.uint16).cpu().numpy()
    np.save(path, arr)


def verify_bf16_roundtrip(tensor_bf16: torch.Tensor, path: str) -> bool:
    """Verify that the save/load roundtrip is bit-identical."""
    loaded_raw = np.load(path)
    loaded = torch.from_numpy(loaded_raw).view(torch.bfloat16)
    original = tensor_bf16.cpu().to(torch.bfloat16)
    if original.shape != loaded.shape:
        print(f"    Shape mismatch: original {original.shape} vs loaded {loaded.shape}")
        return False
    return torch.equal(original, loaded)


def collect_scene_data(
    step3_root: str,
    scene_id: str,
    max_trajs: int = 25,
    num_cameras: int = 8,
) -> list[tuple[str, str, list[str]]]:
    """
    Collect the timestamps and camera image paths of all trajectories in a scene.

    Returns:
        list of (traj_name, timestamp_str, [cam_0_path, ..., cam_{N-1}_path])
    """
    traj_parent = os.path.join(step3_root, "scenes", scene_id, "trajectories")
    if not os.path.isdir(traj_parent):
        return []

    results = []
    traj_names = sorted(os.listdir(traj_parent))[:max_trajs]

    for traj_name in traj_names:
        images_dir = os.path.join(traj_parent, traj_name, "images")
        if not os.path.isdir(images_dir):
            continue

        for ts_str in sorted(os.listdir(images_dir)):
            ts_dir = os.path.join(images_dir, ts_str)
            if not os.path.isdir(ts_dir):
                continue

            cam_paths = []
            valid = True
            for cam_idx in range(num_cameras):
                cam_path = os.path.join(ts_dir, f"cam_{cam_idx}.jpg")
                if not os.path.isfile(cam_path):
                    cam_path = os.path.join(ts_dir, f"cam_{cam_idx}.png")
                    if not os.path.isfile(cam_path):
                        valid = False
                        break
                cam_paths.append(cam_path)

            if valid:
                results.append((traj_name, ts_str, cam_paths))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Precompute DINOv2 encoder features and cache them as .npy files",
    )
    parser.add_argument(
        "--step3-root", type=str, required=True,
        help="Step 3 output directory (contains scenes/)",
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Output directory (step6_dinov2_features_224_xxx)",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="MapAnything pretrained model path",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of images fed to DINOv2 at a time (default 32)",
    )
    parser.add_argument(
        "--max-trajs", type=int, default=25,
        help="Maximum number of trajectories per scene (default 25, 80 recommended for TartanGround)",
    )
    parser.add_argument(
        "--num-cameras", type=int, default=8,
        help="Number of cameras per timestamp (HM3D=8, TartanGround=1)",
    )
    # Multi-GPU sharding
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers")
    parser.add_argument("--worker-id", type=int, default=0, help="Current worker ID")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip timestamp directories whose .npy files already exist",
    )
    parser.add_argument(
        "--verify-samples", type=int, default=5,
        help="Number of samples to verify for bf16 roundtrip at the end of the script (0=skip verification)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"[Step6] DINOv2 feature precomputation")
    print(f"  Step3 root: {args.step3_root}")
    print(f"  Output root: {args.output_root}")
    print(f"  Model path: {args.model_path}")
    print(f"  GPU: {args.gpu}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max trajs/scene: {args.max_trajs}")
    print(f"  Num cameras: {args.num_cameras}")
    print(f"  Worker: {args.worker_id}/{args.num_workers}")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    # ===== Load MapAnything model =====
    print("[Step6] Loading MapAnything model...")
    from mapanything.models import MapAnything
    model_path_obj = Path(args.model_path).expanduser()
    if model_path_obj.exists():
        backbone = MapAnything.from_pretrained(str(model_path_obj))
    else:
        backbone = MapAnything.from_pretrained(args.model_path)

    backbone.eval()
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"[Step6] Model loaded. info_sharing dim: {backbone.info_sharing.dim}")

    # ===== Enumerate scenes and shard =====
    scenes_dir = os.path.join(args.step3_root, "scenes")
    if not os.path.isdir(scenes_dir):
        print(f"[ERROR] scenes dir not found: {scenes_dir}")
        return

    all_scenes = sorted(
        d for d in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, d))
    )
    # Shard by scene modulo
    my_scenes = [s for i, s in enumerate(all_scenes) if i % args.num_workers == args.worker_id]
    print(f"[Step6] Total scenes: {len(all_scenes)}, this worker: {len(my_scenes)}")
    print()

    # Collect samples for the final verification
    verify_samples: list[tuple[torch.Tensor, str]] = []

    total_files_saved = 0
    total_files_skipped = 0
    t_start = time.time()

    for scene_idx, scene_id in enumerate(my_scenes):
        scene_start = time.time()

        # Collect all frames to process for this scene
        scene_data = collect_scene_data(args.step3_root, scene_id, args.max_trajs, args.num_cameras)

        if not scene_data:
            print(f"[{scene_idx+1}/{len(my_scenes)}] {scene_id}: no data, skipping")
            continue

        # Check skip-existing: filter out already-processed timestamps
        if args.skip_existing:
            filtered = []
            for traj_name, ts_str, cam_paths in scene_data:
                feat_dir = os.path.join(
                    args.output_root, "scenes", scene_id, "trajectories",
                    traj_name, "features", ts_str,
                )
                all_exist = all(
                    os.path.isfile(os.path.join(feat_dir, f"cam_{ci}.npy"))
                    for ci in range(args.num_cameras)
                )
                if all_exist:
                    total_files_skipped += args.num_cameras
                else:
                    filtered.append((traj_name, ts_str, cam_paths))
            scene_data = filtered

        if not scene_data:
            print(f"[{scene_idx+1}/{len(my_scenes)}] {scene_id}: all exist, skipping")
            continue

        # Prepare output directories
        for traj_name, ts_str, _ in scene_data:
            feat_dir = os.path.join(
                args.output_root, "scenes", scene_id, "trajectories",
                traj_name, "features", ts_str,
            )
            os.makedirs(feat_dir, exist_ok=True)

        # Batch processing: flatten all camera images of all frames
        # Each timestamp has 8 images; process batch_size images at a time
        all_img_paths: list[str] = []
        all_meta: list[tuple[str, str, int]] = []  # (traj_name, ts_str, cam_idx)
        for traj_name, ts_str, cam_paths in scene_data:
            for cam_idx, cam_path in enumerate(cam_paths):
                all_img_paths.append(cam_path)
                all_meta.append((traj_name, ts_str, cam_idx))

        num_images = len(all_img_paths)
        num_batches = (num_images + args.batch_size - 1) // args.batch_size
        scene_files = 0

        for batch_idx in range(num_batches):
            start_i = batch_idx * args.batch_size
            end_i = min(start_i + args.batch_size, num_images)
            batch_paths = all_img_paths[start_i:end_i]
            batch_meta = all_meta[start_i:end_i]

            # Load and preprocess images
            images = load_and_preprocess_images(batch_paths, device)
            # images: [N, 3, H, W] float32, ImageNet normalized

            # Build MapAnything views (only img and data_norm_type are needed)
            N = images.shape[0]
            views = []
            for i in range(N):
                views.append({
                    "img": images[i:i+1],  # [1, 3, H, W]
                    "data_norm_type": ["dinov2"],
                })

            # DINOv2 inference
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc_feats = backbone._encode_n_views(views)
                    # enc_feats: List[Tensor], each [1, C, Hp, Wp] bf16

            # Save the features of each frame
            for i, (traj_name, ts_str, cam_idx) in enumerate(batch_meta):
                feat = enc_feats[i].squeeze(0)  # [C, Hp, Wp]

                save_path = os.path.join(
                    args.output_root, "scenes", scene_id, "trajectories",
                    traj_name, "features", ts_str, f"cam_{cam_idx}.npy",
                )
                save_bf16_as_uint16(feat, save_path)
                scene_files += 1

                # Collect a sample for the final verification (store the bf16-cast version, matching the file content)
                if (args.verify_samples > 0
                        and len(verify_samples) < args.verify_samples):
                    verify_samples.append(
                        (feat.to(torch.bfloat16).cpu(), save_path),
                    )

        total_files_saved += scene_files
        scene_time = time.time() - scene_start
        print(
            f"[{scene_idx+1}/{len(my_scenes)}] {scene_id}: "
            f"{scene_files} files saved ({scene_time:.1f}s)"
        )

    # ===== Statistics =====
    elapsed = time.time() - t_start
    print()
    print(f"[Step6] Done!")
    print(f"  Files saved: {total_files_saved}")
    print(f"  Files skipped (existing): {total_files_skipped}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if total_files_saved > 0:
        print(f"  Speed: {total_files_saved / elapsed:.1f} files/s")

    # ===== bf16 roundtrip verification =====
    if args.verify_samples > 0 and verify_samples:
        print()
        print(f"[Step6] Verifying bf16 roundtrip ({len(verify_samples)} samples)...")
        all_ok = True
        for tensor_bf16, path in verify_samples:
            ok = verify_bf16_roundtrip(tensor_bf16, path)
            if not ok:
                print(f"  [FAIL] {path}")
                all_ok = False
        if all_ok:
            print(f"  [OK] All {len(verify_samples)} samples are bit-identical")
        else:
            print(f"  [WARNING] Some samples failed roundtrip verification!")


if __name__ == "__main__":
    main()

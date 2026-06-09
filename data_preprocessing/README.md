# HM3D Data Preprocessing Pipeline

[Chinese version](README_CN.md)

This directory contains the full HM3D preprocessing pipeline for G2G, from rendering
multi-camera trajectories in Habitat all the way to the window indices (and optional
DINOv2 feature cache) consumed by G2G training and evaluation.

> The other datasets (TartanGround / NCLT / ZJH) are not produced by this pipeline.
> Their on-disk formats follow the same conventions described below; prepare them in
> the same layout and point the configs at your paths.

## Environments

- **Rendering (step1–step4)**: Habitat-Sim + HM3D scenes, conda env `habitat`.
- **Indexing / features (step5–step6)**: conda env `g2g` (this repo installed with
  `pip install -e .`, plus MapAnything for step6).
- Replace every `/path/to/...` placeholder in the scripts, configs, and wrappers with
  your real paths before running.

## Pipeline overview

```
Step 1  generate trajectories            -> trajectory.tum, traj_meta.json
   |
Step 2  generate rig configs (8-camera)  -> rig_config.json (fixed HFOV=90 deg)
   |
Step 2.5 (optional) randomize intrinsics -> rig_config.json (per-camera HFOV, centered principal point)
   |
Step 3  render RGB + uint16 depth        -> images/*.jpg, depth/*.png (224x224, 1mm)
   |
Step 4  compute overlap matrices         -> view_sequence_pairs.json, matrices/*.npz
   |
Step 5  generate G2G window index         -> stage2_index.json   (GT mode, no external covis model)
   |
Step 6  (optional) precompute DINOv2     -> features/*.npy ([1024,16,16] bf16)
```

| Step | Script | Purpose | Output |
|---|---|---|---|
| 1 | `step1_generate_trajectories.py` | Sample navigable trajectories in HM3D scenes (random action magnitude + random body height). | TUM trajectory + metadata |
| 2 | `step2_generate_rig_configs.py` | Build an 8-camera rig with perturbed extrinsics (truncated-normal roll/pitch/yaw). | `rig_config.json` |
| 2.5 | `step2_5_randomize_intrinsics.py` | (Optional) Assign each camera a random HFOV in [45 deg, 120 deg]; principal point stays centered. | `rig_config.json` with per-camera intrinsics |
| 3 | `step3_render_rgb_depth.py` | Render RGB + same-resolution uint16 depth (224x224, 1mm precision). | `images/`, `depth/` |
| 4 | `step4_compute_overlap.py` | Depth-projection overlap matrices between view sequences. | `view_sequence_pairs.json`, `matrices/*.npz` |
| 5 | `step5_generate_stage2_index.py` | Sliding-window Top-K selection into the G2G training index (**GT mode, no external covisibility model needed**). | `stage2_index.json` |
| 6 | `step6_precompute_dinov2_features.py` | (Optional) Precompute MapAnything DINOv2 patch features to speed up training. | `features/*.npy` |

A **view sequence** is one camera along one trajectory (a single-camera time series).
At training time, G2G samples Group A and Group B from two different view sequences.

---

## Steps in detail

### Step 1 — Generate trajectories (`habitat`)

```bash
conda activate habitat
python data_preprocessing/step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --split train --max_traj_per_scene 80 --min_frames 10 --max_frames 100 --gpu_id 0
python data_preprocessing/step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --split val   --max_traj_per_scene 10 --min_frames 10 --max_frames 100 --gpu_id 0
```
Output per trajectory: `trajectory.tum` (body poses, TUM format) and `traj_meta.json`.

### Step 2 — Generate rig configs (`habitat`)

```bash
python data_preprocessing/step2_generate_rig_configs.py \
    --data_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --hfov 90.0 --width 518 --height 518
```
Each trajectory gets a `rig_config.json` with 8 cameras at nominal yaw
`[0, 45, 90, 135, 180, 225, 270, 315]` deg, plus a truncated-normal extrinsics
perturbation (deterministic per `rig_perturbation_seed`).

### Step 2.5 — Randomize intrinsics (optional, `habitat` or `g2g`)

```bash
python data_preprocessing/step2_5_randomize_intrinsics.py \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --image_size 224 --hfov_min 60 --hfov_max 120
```
Assigns each camera an independent HFOV (square pixels, fx = fy). The **principal
point is fixed at the image center** because habitat_sim's `CameraSensorSpec` only
accepts an HFOV and always renders with a centered principal point; storing a centered
principal point keeps the depth projection in Step 4 consistent with the rendered images.
When using this step, point Step 3's `--step2_root` at the `step2_5_rig_configs_*` directory.

### Step 3 — Render RGB + depth (`habitat`)

```bash
python data_preprocessing/step3_render_rgb_depth.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_train \
    --image_size 224 --gpu_id 0
```
Renders RGB (JPEG) and **same-resolution uint16 depth** (1mm precision, no downsampling).

### Step 4 — Compute overlap matrices (`habitat`)

```bash
python data_preprocessing/step4_compute_overlap.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --step3_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step4_overlap_train
```
Per-frame overlap is computed by monocular depth projection (low-resolution projection
for fast coarse screening) and stored as `uint8`-quantized matrices.

### Step 5 — Generate the G2G index (`g2g`, GT mode recommended)

```bash
conda activate g2g
python data_preprocessing/step5_generate_stage2_index.py \
    --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
    --split train --mode gt --no-precompute-covis
python data_preprocessing/step5_generate_stage2_index.py \
    --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
    --split val   --mode gt --no-precompute-covis
# Multi-worker parallelism: see run_step5_gt_multi_worker.sh
```
> `--mode gt` selects windows directly from the Step 4 depth-projection overlap and
> needs **no covisibility weights** — this is the open-source path. `--mode stage1/hybrid`
> requires your own covisibility model (out of scope for this repo).

### Step 6 — Precompute DINOv2 features (optional, `g2g`)

```bash
bash data_preprocessing/run_step6_4gpu_train.sh
bash data_preprocessing/run_step6_4gpu_val.sh
```
Caches the MapAnything DINOv2 ViT-L/14 patch features so the encoder does not run during
G2G training (the encoder accounts for ~50% of the forward pass).

---

## On-disk formats (consumed directly by training / evaluation)

- **Trajectory** — `.../step1_*/scenes/{scene}/trajectories/{traj}/trajectory.tum`
  ```
  # timestamp tx ty tz qx qy qz qw   (world_T_body, Y up, right-handed, quaternion xyzw)
  ```
- **Rig config** — `.../step2_*/scenes/{scene}/trajectories/{traj}/rig_config.json`
  - per-camera `intrinsics` (3x3, centered principal point), `hfov_deg`, and `body_T_cam` (4x4).
- **RGB** — `.../step3_*/scenes/{scene}/trajectories/{traj}/images/{ts}/cam_{0-7}.jpg` (224x224)
- **Depth** — `.../step3_*/.../depth/{ts}/cam_{0-7}.png` (uint16 PNG, `depth_m = pixel * 0.001`)
- **Overlap matrices** — `.../step4_*/scenes/{scene}/matrices/pair_{XXXXXX}.npz`
  - keys `a2b`, `b2a`, `symmetric`, each `uint8 [T_a, T_b]`; dequantize with `/ 255.0`.
- **G2G window index** — `.../step5_*/scenes/{scene}/stage2_index.json`
  - `{scene_id, pairs: [{pair_id, traj_a, cam_a, traj_b, cam_b, windows: [{rank, indices_a:[5], indices_b:[5], score}]}]}`
- **DINOv2 features (optional)** — `.../step6_*/.../features/{ts}/cam_{0-7}.npy`
  - `[1024, 16, 16]` stored as `uint16` (bf16 raw bits); load with
    `torch.from_numpy(np.load(p)).view(torch.bfloat16)`.

---

## Data access snippets

```python
import cv2, numpy as np, json, torch

# Depth (uint16, 1mm)
def load_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)      # uint16 [0, 65535]
    return img.astype(np.float32) / 1000.0            # meters, [H, W]

# Overlap matrices
def load_overlap(npz_path):
    d = np.load(npz_path)
    return {k: d[k].astype(np.float32) / 255.0 for k in ("a2b", "b2a", "symmetric")}

# G2G window index
def load_stage2_index(json_path):
    with open(json_path) as f:
        return json.load(f)

# Precomputed DINOv2 feature (bit-identical bf16 restore)
def load_dinov2_feature(npy_path):
    raw = np.load(npy_path)                            # uint16, [1024, 16, 16]
    return torch.from_numpy(raw).view(torch.bfloat16)
```

---

## Notes

- **GT mode is self-contained.** Step 5 in `--mode gt` reproduces the released data
  without any covisibility model; covisibility is bring-your-own.
- **Step 2.5 is optional.** Skip it to keep the fixed HFOV=90 deg rig from Step 2; run it
  to train with diverse intrinsics (recommended, following MapAnything / VGGT).
- **Principal point is always centered** to match habitat_sim rendering.
- **Resuming.** Most steps support `--skip-existing`; Step 5 can be checked by counting
  finished pairs in each scene's index file.

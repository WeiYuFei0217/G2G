# HM3D 数据预处理流程

[English version / 英文版](README.md)

本目录包含 G2G 的完整 HM3D 预处理流程：从在 Habitat 中渲染多相机轨迹，一直到 G2G
训练/评估所需的窗口索引（以及可选的 DINOv2 特征缓存）。

> 其余数据集（TartanGround / NCLT / ZJH）不由本流程生成。它们的磁盘格式遵循下文相同的
> 约定，按相同布局自备数据并将配置指向你的路径即可。

## 运行环境

- **渲染步骤（step1–step4）**：Habitat-Sim + HM3D 场景，conda 环境 `habitat`。
- **索引/特征（step5–step6）**：conda 环境 `g2g`（本仓库 `pip install -e .` 安装，step6 还需 MapAnything）。
- 运行前请把脚本、配置、wrapper 里的 `/path/to/...` 占位符全部替换为你的真实路径。

## 流程总览

```
Step 1  生成轨迹                          -> trajectory.tum, traj_meta.json
   |
Step 2  生成 rig 配置（8 相机）           -> rig_config.json（固定 HFOV=90 度）
   |
Step 2.5（可选）随机化内参               -> rig_config.json（每相机 HFOV，主点居中）
   |
Step 3  渲染 RGB + uint16 深度            -> images/*.jpg, depth/*.png（224x224, 1mm）
   |
Step 4  计算重叠度矩阵                     -> view_sequence_pairs.json, matrices/*.npz
   |
Step 5  生成 G2G 窗口索引                 -> stage2_index.json（GT 模式，不依赖外部共视性模型）
   |
Step 6（可选）预计算 DINOv2 特征          -> features/*.npy（[1024,16,16] bf16）
```

| Step | 脚本 | 作用 | 产物 |
|---|---|---|---|
| 1 | `step1_generate_trajectories.py` | 在 HM3D 场景中采样可导航轨迹（随机动作幅度 + 随机本体高度）。 | TUM 轨迹 + 元数据 |
| 2 | `step2_generate_rig_configs.py` | 生成 8 相机 rig，外参带截断正态扰动（roll/pitch/yaw）。 | `rig_config.json` |
| 2.5 | `step2_5_randomize_intrinsics.py` | （可选）为每个相机随机化 HFOV ∈ [45 度, 120 度]；主点保持居中。 | 带每相机内参的 `rig_config.json` |
| 3 | `step3_render_rgb_depth.py` | 渲染 RGB + 同分辨率 uint16 深度（224x224，1mm 精度）。 | `images/`, `depth/` |
| 4 | `step4_compute_overlap.py` | 视图序列之间的深度投影重叠度矩阵。 | `view_sequence_pairs.json`, `matrices/*.npz` |
| 5 | `step5_generate_stage2_index.py` | 滑动窗口 Top-K 选择，生成 G2G 训练索引（**GT 模式，不依赖外部共视性模型**）。 | `stage2_index.json` |
| 6 | `step6_precompute_dinov2_features.py` | （可选）预计算 MapAnything DINOv2 patch 特征以加速训练。 | `features/*.npy` |

**视图序列**指一条轨迹上的一个相机（即单相机时间序列）。训练时，G2G 从两个不同的视图序列
分别采样 Group A 和 Group B。

---

## 各步骤详解

### Step 1 — 生成轨迹（`habitat`）

```bash
conda activate habitat
python data_preprocessing/step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --split train --max_traj_per_scene 80 --min_frames 10 --max_frames 100 --gpu_id 0
python data_preprocessing/step1_generate_trajectories.py \
    --output_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_val \
    --split val   --max_traj_per_scene 10 --min_frames 10 --max_frames 100 --gpu_id 0
```
每条轨迹的产物：`trajectory.tum`（body 位姿，TUM 格式）和 `traj_meta.json`。

### Step 2 — 生成 rig 配置（`habitat`）

```bash
python data_preprocessing/step2_generate_rig_configs.py \
    --data_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --hfov 90.0 --width 518 --height 518
```
每条轨迹生成一个 `rig_config.json`，包含 8 个相机，标称 yaw 为
`[0, 45, 90, 135, 180, 225, 270, 315]` 度，并叠加截断正态外参扰动（按
`rig_perturbation_seed` 确定性生成）。

### Step 2.5 — 随机化内参（可选，`habitat` 或 `g2g`）

```bash
python data_preprocessing/step2_5_randomize_intrinsics.py \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_rig_configs_train \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --image_size 224 --hfov_min 60 --hfov_max 120
```
为每个相机独立采样 HFOV（方形像素，fx = fy）。**主点固定在图像中心**，因为 habitat_sim
的 `CameraSensorSpec` 只接受 HFOV、渲染时主点恒在中心；存储居中主点可使 Step 4 的深度投影
与渲染图像保持一致。使用本步骤时，把 Step 3 的 `--step2_root` 指向 `step2_5_rig_configs_*` 目录。

### Step 3 — 渲染 RGB + 深度（`habitat`）

```bash
python data_preprocessing/step3_render_rgb_depth.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_train \
    --image_size 224 --gpu_id 0
```
渲染 RGB（JPEG）和**同分辨率 uint16 深度**（1mm 精度，不下采样）。

### Step 4 — 计算重叠度矩阵（`habitat`）

```bash
python data_preprocessing/step4_compute_overlap.py \
    --step1_root /path/to/data/HM3D/DATA_GEN/step1_generate_trajectories_train \
    --step2_root /path/to/data/HM3D/DATA_GEN/step2_5_rig_configs_train \
    --step3_root /path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_train \
    --output_root /path/to/data/HM3D/DATA_GEN/step4_overlap_train
```
逐帧重叠度由单目深度投影计算（低分辨率投影用于快速粗筛），以 `uint8` 量化矩阵存储。

### Step 5 — 生成 G2G 索引（`g2g`，推荐 GT 模式）

```bash
conda activate g2g
python data_preprocessing/step5_generate_stage2_index.py \
    --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
    --split train --mode gt --no-precompute-covis
python data_preprocessing/step5_generate_stage2_index.py \
    --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
    --split val   --mode gt --no-precompute-covis
# 多 worker 并行见 run_step5_gt_multi_worker.sh
```
> `--mode gt` 直接用 Step 4 的深度投影重叠度选窗口，**无需任何 共视性模型权重** —— 这是开源版
> 的标准路径。`--mode stage1/hybrid` 需要你自备 外部共视性模型（不在本仓库范围内）。

### Step 6 — 预计算 DINOv2 特征（可选，`g2g`）

```bash
bash data_preprocessing/run_step6_4gpu_train.sh
bash data_preprocessing/run_step6_4gpu_val.sh
```
缓存 MapAnything DINOv2 ViT-L/14 的 patch 特征，使 G2G 训练时无需再跑 encoder
（encoder 约占前向传播的 50%）。

---

## 磁盘格式（训练/评估直接消费）

- **轨迹** — `.../step1_*/scenes/{scene}/trajectories/{traj}/trajectory.tum`
  ```
  # timestamp tx ty tz qx qy qz qw   (world_T_body, Y 轴向上, 右手系, 四元数 xyzw)
  ```
- **rig 配置** — `.../step2_*/scenes/{scene}/trajectories/{traj}/rig_config.json`
  - 每相机的 `intrinsics`（3x3，主点居中）、`hfov_deg`、`body_T_cam`（4x4）。
- **RGB** — `.../step3_*/scenes/{scene}/trajectories/{traj}/images/{ts}/cam_{0-7}.jpg`（224x224）
- **深度** — `.../step3_*/.../depth/{ts}/cam_{0-7}.png`（uint16 PNG，`depth_m = pixel * 0.001`）
- **重叠度矩阵** — `.../step4_*/scenes/{scene}/matrices/pair_{XXXXXX}.npz`
  - 键 `a2b`、`b2a`、`symmetric`，均为 `uint8 [T_a, T_b]`；反量化 `/ 255.0`。
- **G2G 窗口索引** — `.../step5_*/scenes/{scene}/stage2_index.json`
  - `{scene_id, pairs: [{pair_id, traj_a, cam_a, traj_b, cam_b, windows: [{rank, indices_a:[5], indices_b:[5], score}]}]}`
- **DINOv2 特征（可选）** — `.../step6_*/.../features/{ts}/cam_{0-7}.npy`
  - `[1024, 16, 16]`，以 `uint16` 存储（bf16 原始位）；加载：
    `torch.from_numpy(np.load(p)).view(torch.bfloat16)`。

---

## 数据访问示例

```python
import cv2, numpy as np, json, torch

# 深度 (uint16, 1mm)
def load_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)      # uint16 [0, 65535]
    return img.astype(np.float32) / 1000.0            # 米, [H, W]

# 重叠度矩阵
def load_overlap(npz_path):
    d = np.load(npz_path)
    return {k: d[k].astype(np.float32) / 255.0 for k in ("a2b", "b2a", "symmetric")}

# G2G 窗口索引
def load_stage2_index(json_path):
    with open(json_path) as f:
        return json.load(f)

# 预计算的 DINOv2 特征 (逐位恢复 bf16)
def load_dinov2_feature(npy_path):
    raw = np.load(npy_path)                            # uint16, [1024, 16, 16]
    return torch.from_numpy(raw).view(torch.bfloat16)
```

---

## 说明

- **GT 模式自洽。** Step 5 在 `--mode gt` 下无需任何 共视性模型即可复现发布数据；共视性模型为自备。
- **Step 2.5 可选。** 跳过则沿用 Step 2 固定 HFOV=90 度的 rig；运行则用多样内参训练（推荐，参考 MapAnything / VGGT）。
- **主点始终居中**，以与 habitat_sim 渲染保持一致。
- **断点续算。** 多数步骤支持 `--skip-existing`；Step 5 可通过统计每个场景索引文件里已完成的 pair 数来检查进度。

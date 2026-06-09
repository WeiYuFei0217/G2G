# G2G: 利用组内几何进行组间位姿估计

<p align="center">
  <img src="assets/figs/G2G-method-fig-v1-final.png" width="100%"/>
</p>

<p align="center">
  <a href="https://weiyufei0217.github.io/G2G/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-2EA44F?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.08284"><img src="https://img.shields.io/badge/arXiv-2606.08284-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

**[CoRL 2026 (在审)]** G2G 官方实现。

[[English README]](README.md)

> 恢复两组图像之间的 6-DoF 相对位姿是跨序列重定位、多相机 rig 里程计等多视角任务的核心问题。每组图像都携带来自预先建立地图、里程计或 rig 标定的**已知组内几何**，预训练多视角 backbone 也已将这些几何信息融入视觉特征。然而，现有方法将所有视角视为无结构的集合，缺失了**跨组推理**这一关键环节。
>
> G2G 冻结多视角基础模型，仅添加三个轻量可训练模块（约 32M 参数，不到完整模型的 6%）来桥接两组图像：感知器重采样器、跨组桥接（融合自注意力）和多帧位姿头。仅以相对位姿作为监督信号，G2G 在四个数据集上的两个任务中均达到最优精度。

## 亮点

- **统一框架充分利用组内几何**：不同于将所有视角展平为无结构集合的方法，G2G 显式利用每组内部的已知外参（例如来自预先建立的地图、里程计或 rig 标定）进行跨组位姿推理；同一架构同时处理跨序列重定位和多相机 rig 里程计两个任务
- **轻量高效**：仅约 32M 可训练参数，基于冻结的 MapAnything backbone（约 539M），仅需相对位姿监督
- **四个数据集，10 个预训练权重**：HM3D（室内仿真）、TartanGround（室外仿真）、NCLT（真实跨季节）、ZJH（仿真到真实迁移）

## 安装

### 1. 创建 conda 环境

```bash
conda create -n g2g python=3.12 -y
conda activate g2g
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 MapAnything（backbone）

> **重要：backbone 代码已为你内置（可复现性）。** G2G 是基于 **DINOv2-large / 1024 维** 的
> MapAnything backbone 训练的（2025 年 10–11 月发布的版本）。官方此后已把默认 backbone 换成
> **DINOv2-giant / 1536 维**（`v1.1`+ 及当前 Hugging Face 权重），其张量 shape 与发布的 G2G 模块
> **不兼容**。为使发布包自洽，我们已把所需 backbone 代码内置在
> [`third_party/mapanything/`](third_party/mapanything/)：这是 MapAnything **v1.0.1**
> （large/1024，commit `fde8425`）的副本，仅含**一处**有文档记录的修改，用于暴露 G2G 所需的
> information-sharing 特征，即 `forward(..., return_info_sharing_features=True)`
> （详见 [`third_party/mapanything/NOTICE`](third_party/mapanything/NOTICE)）。直接本地安装即可，
> 无需 clone 官方仓库或切换 commit。

```bash
pip install -e ./third_party/mapanything
```

从我们的 [百度网盘](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888)（提取码：`8888`）、[Google Drive](https://drive.google.com/drive/folders/1z6RfJT5i8n5C9YaZSwGyzv9LdbEQWcq5?usp=sharing) 或 [Hugging Face](https://huggingface.co/feixue22/G2G) 下载
配套的 backbone checkpoint（large/1024，约 2.1 GB），放置到 `map-anything-model/` 目录。这是
`facebook/map-anything` 在 2025 年 12 月升级到 giant 之前的 **large** 变体；**请勿**下载当前的
Hugging Face 权重（已是不兼容的 giant 模型）。

### 3. 安装 G2G

```bash
cd G2G
pip install -e .
```

### 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

### HM3D

G2G 使用 HM3D 数据集，配套自定义预处理管线。完整流程详见 [`data_preprocessing/README_CN.md`](data_preprocessing/README_CN.md)（英文版 [`README.md`](data_preprocessing/README.md)）。

管线生成：
- **step1**：轨迹采样
- **step2**：rig 配置（8 相机）
- **step2.5**（可选）：每相机随机内参（HFOV ∈ [45 度, 120 度]，主点居中）
- **step3**：RGB + uint16 深度渲染（224x224，1mm 精度）
- **step4**：深度投影重叠度矩阵
- **step5**：G2G 窗口索引（GT 重叠度，不依赖外部共视性模型）
- **step6**（可选）：DINOv2 特征提取

### 其他数据集

对于 TartanGround、NCLT、ZJH 数据集，按类似的预处理步骤执行，并相应调整配置文件中的路径。

### 配置路径设置

所有配置文件使用占位符路径（`/path/to/...`），训练或评估前请替换为实际路径：

```bash
# 示例：HM3D-Reloc
sed -i 's|/path/to/data/HM3D|/your/actual/path/HM3D|g' configs/reloc/hm3d.yaml
sed -i 's|/path/to/map-anything-model/|/your/actual/path/map-anything-model/|g' configs/reloc/hm3d.yaml
```

## 预训练模型

从 [百度网盘](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888)（提取码：`8888`）、[Google Drive](https://drive.google.com/drive/folders/1z6RfJT5i8n5C9YaZSwGyzv9LdbEQWcq5?usp=sharing) 或 [Hugging Face](https://huggingface.co/feixue22/G2G) 下载所有权重，放到 `release_weights/` 目录。

| 权重文件 | 任务 | 数据集 |
|---------|------|--------|
| `HM3D-Reloc.pth` | 重定位 | HM3D |
| `TartanGround-Reloc.pth` | 重定位 | TartanGround |
| `NCLT-Reloc.pth` | 重定位 | NCLT |
| `ZJH-Reloc.pth` | 重定位 | ZJH |
| `HM3D-Rig-8.pth` | Rig | HM3D (8相机) |
| `HM3D-Rig-4.pth` | Rig | HM3D (4相机) |
| `TartanGround-Rig-4.pth` | Rig | TartanGround (4相机) |
| `NCLT-Rig-Intra.pth` | Rig | NCLT 同期 (5相机) |
| `NCLT-Rig-Cross.pth` | Rig | NCLT 跨期 (5相机) |
| `ZJH-Rig-4.pth` | Rig | ZJH (4相机) |

这些是 G2G-only 权重（不含冻结的 backbone）。评估脚本会自动处理部分加载。

同一个 [百度网盘](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888)（提取码：`8888`，也镜像在 [Google Drive](https://drive.google.com/drive/folders/1z6RfJT5i8n5C9YaZSwGyzv9LdbEQWcq5?usp=sharing) 和 [Hugging Face](https://huggingface.co/feixue22/G2G)）还提供：
- **示例样例包** —— 解压到 `examples/`（布局见 [`examples/README_CN.md`](examples/README_CN.md)）；
- **原始评估结果**（论文子集逐对 CSV）—— 解压到 `eval_results/`；
- **MapAnything backbone checkpoint** —— 解压到 `map-anything-model/`（见安装步骤 2）。

这些大文件不纳入 git 仓库，以保持仓库轻量。

## 训练

### 重定位（Task 1）

```bash
torchrun --nproc_per_node=4 scripts/train_reloc.py \
    --config configs/reloc/hm3d.yaml \
    --curriculum
```

### Rig 里程计（Task 2）

```bash
torchrun --nproc_per_node=4 scripts/train_rig.py \
    --config configs/rig/hm3d_8cam.yaml
```

添加 `--overfit` 可在小数据子集上快速验证流程。

## 评估

### 重定位

```bash
python scripts/eval_reloc.py \
    --config configs/reloc/hm3d.yaml \
    --checkpoint release_weights/HM3D-Reloc.pth \
    --output-dir outputs/eval_HM3D-Reloc \
    --batch-size 16 --min-overlap 0.1
```

### Rig 里程计

```bash
python scripts/eval_rig.py \
    --config configs/rig/hm3d_8cam.yaml \
    --checkpoint release_weights/HM3D-Rig-8.pth \
    --output-dir outputs/eval_HM3D-Rig-8 \
    --batch-size 8
```

多卡评估：
```bash
torchrun --nproc_per_node=4 --master-port=29590 \
    scripts/eval_reloc.py \
    --config configs/reloc/hm3d.yaml \
    --checkpoint release_weights/HM3D-Reloc.pth \
    --batch-size 16 --min-overlap 0.1
```

## 权重提取

从完整训练 checkpoint（含冻结 backbone）中提取 G2G-only 权重：

```bash
python scripts/extract_g2g_weights.py \
    --input /path/to/full_checkpoint.pt \
    --output release_weights/MyModel.pth
```

## 致谢

- [MapAnything](https://github.com/facebookresearch/map-anything): 多视角基础模型 backbone
- [DINOv2](https://github.com/facebookresearch/dinov2): 视觉编码器

## 许可证

本项目采用 [CC BY-NC 4.0](LICENSE) 许可证。

## 引用

如果本工作对你有帮助，请引用：

```bibtex
@misc{wei2026g2gexploitingintragroupgeometry,
      title={G2G: Exploiting Intra-Group Geometry for Inter-Group Pose Estimation},
      author={Yufei Wei and Shuhao Ye and Chenxiao Hu and Yiyuan Pan and Dongyu Feng and Rong Xiong and Yue Wang and Yanmei Jiao},
      year={2026},
      eprint={2606.08284},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.08284},
}
```

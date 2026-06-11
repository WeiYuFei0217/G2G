# 示例样例集

用于在不下载完整数据集的前提下，快速校验发布权重与代码的小型自包含输入包。每个
（任务, 数据集）组合一个 bundle。

> 样例数据（`examples/reloc/`、`examples/rig/`）通过
> [百度网盘](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888)（提取码：`8888`）分发，
> 不纳入 git。请下载并解压到 `examples/`，以还原下方的目录结构。

每个 bundle 是验证集的**分 bin 平衡随机**子集：候选按 overlap bin（reloc）/
body-distance 分位 bin（rig）、依完整验证集的分布做分层，然后随机抽取至目标数量。
每组的身份与 overlap 记录在各自的 `groups.csv` 中。

所有 bundle 均取自与发布完整表相同的验证采集（含 ZJH-real，使用 REAL_new 采集序列）。

## Bundle 清单

| combo | 任务 | 权重 | 样例形式 | 组数 | 体积 |
|---|---|---|---|---|---|
| reloc_hm3d | reloc | HM3D-Reloc.pth | RGB | 100 | 24 MB |
| reloc_nclt | reloc | NCLT-Reloc.pth | RGB | 100 | 21 MB |
| reloc_tartanground | reloc | TartanGround-Reloc.pth | RGB | 100 | 31 MB |
| reloc_zjh_sim | reloc | ZJH-Reloc.pth | RGB | 60（每场景 20 × 3） | 11 MB |
| reloc_zjh_real | reloc | ZJH-Reloc.pth | RGB | 60（每场景 20 × 3） | 11 MB |
| rig_hm3d_8cam | rig | HM3D-Rig-8.pth | RGB | 100 | 35 MB |
| rig_hm3d_4cam | rig | HM3D-Rig-4.pth | RGB | 100 | 34 MB |
| rig_tartanground_4cam | rig | TartanGround-Rig-4.pth | RGB | 100 | 29 MB |
| rig_nclt_intra | rig | NCLT-Rig-Intra.pth | RGB | 100 | 24 MB |
| rig_nclt_cross | rig | NCLT-Rig-Cross.pth | RGB | 100 | 25 MB |
| rig_zjh_sim | rig | ZJH-Rig-4.pth | RGB | 60（每场景 20 × 3） | 7 MB |
| rig_zjh_real | rig | ZJH-Rig-4.pth | RGB | 60（每场景 20 × 3） | 8 MB |

**样例形式**：所有 bundle 都提供 RGB JPEG（224x224），由冻结的 MapAnything backbone 在
评估时现算 DINOv2 特征（`step6_root: ""`）。

## 目录结构

```
examples/<task>/<combo>/
├── step1/scenes/<scene>/trajectories/<traj>/trajectory.tum            # 位姿真值 (TUM)
├── step2/scenes/<scene>/[trajectories/<traj>/]rig_config.json         # 内参/外参
├── step3/scenes/<scene>/trajectories/<traj>/images/<ts>/cam_<c>.jpg   # RGB（224x224）
├── index/scenes/<scene>/{stage2_index.json|rig_pairs.json}           # 最小索引
├── groups.csv          # 每组的身份与 overlap / body distance
└── selection_report.json
```

`<ts>` = `f"{int(round(timestamp_seconds * 1000)):010d}"`。`step2` 的 rig_config：reloc 为
按轨迹，rig 为场景级（`step2/scenes/<scene>/rig_config.json`）。

## 用法

将某个 config 的验证块指向对应 bundle，再运行匹配的评估脚本（`PYTHONPATH` 需包含仓库
根目录）。冻结的 MapAnything backbone 从你本地的模型路径加载。

### Relocalization（5 个组合全部为 RGB）

将 `step6_root` 置空，让 backbone 从 JPEG 现算特征。各数据集的 config 基底与权重不同
（hm3d/nclt/tartanground 用各自的 `configs/reloc/<ds>.yaml`；两个 ZJH 组合都用
`configs/reloc/zjh.yaml` + `ZJH-Reloc.pth`）：

```yaml
# my_eval.yaml（基于 configs/reloc/<dataset>.yaml 修改）
backbone: { model_path: /path/to/map-anything-model/ }
data:
  val:
    step1_root: examples/reloc/reloc_zjh_real/step1
    step2_root: examples/reloc/reloc_zjh_real/step2
    step3_root: examples/reloc/reloc_zjh_real/step3
    step6_root: ""                                  # 从 RGB 现算特征
    stage2_index_root: examples/reloc/reloc_zjh_real/index
```

```bash
PYTHONPATH=. python scripts/eval_reloc.py --config my_eval.yaml \
    --checkpoint release_weights/ZJH-Reloc.pth --output-dir /tmp/ex_zjh_real
```

### Rig 里程计（所有组合均为 RGB）

每个 rig 组合使用各自的 `configs/rig/<combo>.yaml` + 对应权重（如 `rig_hm3d_8cam` →
`configs/rig/hm3d_8cam.yaml` + `HM3D-Rig-8.pth`；两个 ZJH 组合都用 `configs/rig/zjh_4cam.yaml`
+ `ZJH-Rig-4.pth`）。将 `step6_root` 置空、验证块路径指向 bundle（`num_cameras`/`total_cameras`
保持各自 config 基底的取值）：

```yaml
data:
  num_cameras: 4          # rig_hm3d_8cam 为 8，NCLT 为 5，其余为 4（用 config 基底的值）
  total_cameras: 4
  val:
    step1_root: examples/rig/rig_zjh_real/step1
    step2_root: examples/rig/rig_zjh_real/step2
    step3_root: examples/rig/rig_zjh_real/step3
    step6_root: ""                                # 从 RGB 现算
    index_root: examples/rig/rig_zjh_real/index
```

```bash
PYTHONPATH=. python scripts/eval_rig.py --config my_rig_eval.yaml \
    --checkpoint release_weights/ZJH-Rig-4.pth --output-dir /tmp/ex_rig
```

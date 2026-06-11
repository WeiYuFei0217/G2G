# Example sample sets

Small, self-contained input bundles for sanity-checking the released weights and
code without downloading the full datasets. One bundle per (task, dataset) combo.

> The bundle data (`examples/reloc/`, `examples/rig/`) is distributed via
> [Baidu Cloud](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888) (code: `8888`), not
> tracked in git. Download and extract it into `examples/` so the layout below is restored.

Each bundle is a **bin-balanced random** subset of the validation set: candidates are
stratified across overlap bins (reloc) / body-distance quantile bins (rig) in
proportion to the full-validation distribution, then randomly drawn to reach the
final size. Per-group identity and overlap are recorded in each `groups.csv`.

All bundles are drawn from the same validation captures as the released full tables
(including ZJH-real, which uses the REAL_new capture).

## Bundle inventory

| combo | task | weight | sample form | groups | size |
|---|---|---|---|---|---|
| reloc_hm3d | reloc | HM3D-Reloc.pth | RGB | 100 | 24 MB |
| reloc_nclt | reloc | NCLT-Reloc.pth | RGB | 100 | 21 MB |
| reloc_tartanground | reloc | TartanGround-Reloc.pth | RGB | 100 | 31 MB |
| reloc_zjh_sim | reloc | ZJH-Reloc.pth | RGB | 60 (20/scene × 3) | 11 MB |
| reloc_zjh_real | reloc | ZJH-Reloc.pth | RGB | 60 (20/scene × 3) | 11 MB |
| rig_hm3d_8cam | rig | HM3D-Rig-8.pth | RGB | 100 | 35 MB |
| rig_hm3d_4cam | rig | HM3D-Rig-4.pth | RGB | 100 | 34 MB |
| rig_tartanground_4cam | rig | TartanGround-Rig-4.pth | RGB | 100 | 29 MB |
| rig_nclt_intra | rig | NCLT-Rig-Intra.pth | RGB | 100 | 24 MB |
| rig_nclt_cross | rig | NCLT-Rig-Cross.pth | RGB | 100 | 25 MB |
| rig_zjh_sim | rig | ZJH-Rig-4.pth | RGB | 60 (20/scene × 3) | 7 MB |
| rig_zjh_real | rig | ZJH-Rig-4.pth | RGB | 60 (20/scene × 3) | 8 MB |

**Sample form.** Every bundle ships RGB JPEGs (224x224) and lets the frozen MapAnything
backbone recompute DINOv2 features at eval time (set `step6_root: ""`).

## Layout

```
examples/<task>/<combo>/
├── step1/scenes/<scene>/trajectories/<traj>/trajectory.tum            # pose GT (TUM)
├── step2/scenes/<scene>/[trajectories/<traj>/]rig_config.json         # intrinsics/extrinsics
├── step3/scenes/<scene>/trajectories/<traj>/images/<ts>/cam_<c>.jpg   # RGB (224x224)
├── index/scenes/<scene>/{stage2_index.json|rig_pairs.json}           # minimal index
├── groups.csv          # per-group identity and overlap / body distance
└── selection_report.json
```

`<ts>` = `f"{int(round(timestamp_seconds * 1000)):010d}"`. `step2` rig_config is
per-trajectory for reloc and scene-level (`step2/scenes/<scene>/rig_config.json`) for rig.

## Usage

Point a config's validation block at the bundle and run the matching eval script
(`PYTHONPATH` must include the repo root). The frozen MapAnything backbone is loaded
from your local model path.

### Relocalization (all 5 combos ship RGB)

Leave `step6_root` empty so the backbone recomputes features from the shipped JPEGs.
The config base and checkpoint differ per dataset (hm3d/nclt/tartanground use their own
`configs/reloc/<ds>.yaml`; both ZJH combos use `configs/reloc/zjh.yaml` + `ZJH-Reloc.pth`):

```yaml
# my_eval.yaml (start from configs/reloc/<dataset>.yaml)
backbone: { model_path: /path/to/map-anything-model/ }
data:
  val:
    step1_root: examples/reloc/reloc_zjh_real/step1
    step2_root: examples/reloc/reloc_zjh_real/step2
    step3_root: examples/reloc/reloc_zjh_real/step3
    step6_root: ""                                  # recompute features from RGB
    stage2_index_root: examples/reloc/reloc_zjh_real/index
```

```bash
PYTHONPATH=. python scripts/eval_reloc.py --config my_eval.yaml \
    --checkpoint release_weights/ZJH-Reloc.pth --output-dir /tmp/ex_zjh_real
```

### Rig odometry (all combos ship RGB)

Each rig combo uses its own `configs/rig/<combo>.yaml` + matching weight (e.g.
`rig_hm3d_8cam` → `configs/rig/hm3d_8cam.yaml` + `HM3D-Rig-8.pth`; both ZJH combos use
`configs/rig/zjh_4cam.yaml` + `ZJH-Rig-4.pth`). Leave `step6_root` empty and point the
validation roots at the bundle (keep `num_cameras`/`total_cameras` as set in the base config):

```yaml
data:
  num_cameras: 4          # 8 for rig_hm3d_8cam, 5 for NCLT, 4 otherwise (use the base config value)
  total_cameras: 4
  val:
    step1_root: examples/rig/rig_zjh_real/step1
    step2_root: examples/rig/rig_zjh_real/step2
    step3_root: examples/rig/rig_zjh_real/step3
    step6_root: ""                                # recompute features from RGB
    index_root: examples/rig/rig_zjh_real/index
```

```bash
PYTHONPATH=. python scripts/eval_rig.py --config my_rig_eval.yaml \
    --checkpoint release_weights/ZJH-Rig-4.pth --output-dir /tmp/ex_rig
```

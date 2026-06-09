# Point-cloud viewer data

Data consumed by the interactive 3D viewer (`static/js/viewer.js`). Two modes:

- `reloc/` — inter-group relocalization: two point clouds (Group A / Group B) plus
  GT-vs-Pred cameras for the predicted `T_rel` that aligns B into A's frame.
- `rig/` — multi-camera rig odometry: per-frame fused reconstructions stitched along
  a trajectory with chained pred / GT poses.

`*.ply` are binary, RGB, randomly down-sampled for the web (reloc ~130k pts, rig
~22k pts/frame). All poses are `cam->world`, row-major 4×4, OpenCV convention
(Y-down; the viewer rotates the root 180° about X for Y-up display).

## Camera colors (matplotlib `tab10`, consistent with the paper case-study figures)

| Mode  | Color  | Meaning            |
|-------|--------|--------------------|
| reloc | orange | Group A · GT (anchor) |
| reloc | red    | Group A · Pred (rich schema only) |
| reloc | green  | Group B · GT       |
| reloc | blue   | Group B · Pred     |
| rig   | green  | latest frame · GT  |
| rig   | blue   | latest frame · Pred|
| rig   | amber  | previous frame · Pred |

## Adding / replacing a reloc example (no GPU needed)

Consumes a case-study-style folder produced by
`scripts/visualization/casestudy_*_reloc_g2g.py` (already-rendered PLYs + poses
JSON), so it only down-samples and re-packs — no model, no raw dataset.

```bash
conda activate mapanything
python scripts/import_reloc_example.py \
    --key zjh_real --display "ZJH (real)" \
    --src ../outputs/casestudy_zjh_real_reloc
```

This writes `reloc/<key>_A.ply`, `reloc/<key>_B.ply`, `reloc/<key>.json` and
updates `reloc/manifest.json`. See the header of `scripts/import_reloc_example.py`
for the full source-folder layout, CLI flags, and the rich-schema field reference.

### reloc `<key>.json` schema

| Field | Type | Notes |
|-------|------|-------|
| `posesA` | `[4×4]×N` | Group A GT poses (A0 frame) — **required** |
| `posesB` | `[4×4]×N` | Group B GT poses (B0 frame) — **required** |
| `T_rel_gt`, `T_rel_pred` | `4×4` | B0→A0 relative pose (GT / predicted) — **required** |
| `posesB_in_a0` | `[4×4]×N` | Group B GT in A0 frame → draws green directly (optional) |
| `pred_posesA` | `[4×4]×N` | Group A predicted → enables red A·Pred (optional, "rich") |
| `pred_posesB_in_a0` | `[4×4]×N` | Group B predicted in A0 → draws blue directly (optional) |
| `errors` | obj | `{rot_deg, trans_m, gt_rot, gt_trans, overlap}` |
| `ply_a`, `ply_b`, `n_cams`, `display`, `scene`, `traj_a`, `traj_b` | — | metadata |

When the optional `*_in_a0` / `pred_*` fields are absent the viewer falls back to
`T_rel_gt @ posesB` (green) and `T_rel_pred @ posesB` (blue), drawing the A anchor
plus B GT/Pred but no red A·Pred.

## Regenerating a rig example (needs GPU + model + raw dataset)

Rig frames are reconstructed per-frame with MapAnything, so there is no lightweight
import path — re-run the generator and pick a new scene/traj/window inside it:

```bash
conda activate mapanything
cd scripts/video                    # IROS-G2G/scripts/video/export_viewer_rig.py
CUDA_VISIBLE_DEVICES=0 python export_viewer_rig.py --examples nclt zjh_real
```

### rig `<key>.json` schema

| Field | Type | Notes |
|-------|------|-------|
| `extrinsics` | `[4×4]×K` | rig cam→cam0 extrinsics (K cameras) |
| `frames` | list | each `{ply, pred(4×4), gt(4×4), n_points}`, chained from frame 0 |
| `metrics` | obj | `{ate_rmse_m, drift_pct, per_step_rot_med, per_step_trans_med, path_length_m}` |
| `n_cams`, `mono_cam`, `display`, `scene`, `traj` | — | metadata |

## Manifests

`reloc/manifest.json` and `rig/manifest.json` hold the per-key summary the viewer
loads up front (display name + errors/metrics + counts); the importers/generators
keep them in sync automatically.

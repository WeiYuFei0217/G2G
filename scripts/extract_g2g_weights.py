#!/usr/bin/env python3
"""
extract_g2g_weights.py -- Extract a lightweight G2G-only inference weight file from a full training checkpoint.

A full training checkpoint (2.5-3.1GB) contains the frozen MapAnything backbone and the optimizer state;
at inference time the backbone is reloaded via MapAnything.from_pretrained, so it does not need to be distributed with the weights.
This script strips backbone.* and the optimizer state, keeping only the ~32M trainable G2G module (~123MB).

Example:
  python scripts/extract_g2g_weights.py \
      --input  /path/to/best.pt \
      --output release_weights/HM3D-Rig-8.pth
"""
import argparse

import torch


def extract(input_path: str, output_path: str) -> None:
    ckpt = torch.load(input_path, map_location="cpu")

    # Parse out the state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        sd = ckpt  # already a plain state_dict
    else:
        raise ValueError(
            f"Could not parse a state_dict from {input_path}, top-level keys={list(ckpt)[:8]}"
        )

    # Strip the frozen backbone (and any optimizer/EMA leftovers), keeping only the trainable G2G module
    g2g_sd = {k: v for k, v in sd.items() if not k.startswith("backbone.")}

    n_total = sum(v.numel() for v in g2g_sd.values() if isinstance(v, torch.Tensor))
    torch.save(g2g_sd, output_path)
    print(f"[extract] {input_path}")
    print(f"          -> {output_path}")
    print(
        f"          kept {len(g2g_sd)} tensors, {n_total / 1e6:.1f}M params "
        f"(dropped {len(sd) - len(g2g_sd)} backbone/other keys)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the full training checkpoint (.pt/.pth)")
    ap.add_argument("--output", required=True, help="Output path for the lightweight G2G weights (.pth)")
    args = ap.parse_args()
    extract(args.input, args.output)

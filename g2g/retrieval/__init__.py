"""
Retrieval Module: Overlap Prediction Utilities and covisibility Inference Engine.

Utility functions (utils.py):
  - sliding_window_scores: sliding window scoring
  - find_best_window / find_topk_windows: window selection
  - compute_window_score: single-window scoring

Covisibility inference engine (stage1_inference.py):
  - Stage1InferenceEngine: batched encoder->decoder inference
  - select_windows_with_union: sliding window Top-K + frame union

Note: covisibility models are NOT shipped in this release. For the optional
      --mode stage1/hybrid path, provide your own covisibility model and pass
      it into Stage1InferenceEngine (the default --mode gt path needs none).
"""

from .utils import (
    sliding_window_scores,
    find_best_window,
    find_topk_windows,
    compute_window_score,
    dequantize_overlap,
)
from .stage1_inference import (
    Stage1InferenceEngine,
    select_windows_with_union,
)

__all__ = [
    "sliding_window_scores",
    "find_best_window",
    "find_topk_windows",
    "compute_window_score",
    "dequantize_overlap",
    "Stage1InferenceEngine",
    "select_windows_with_union",
]

"""
Utility functions for overlap prediction and window selection.

Core functionality:
- Sliding window score computation
- Best window selection (greedy and multi-candidate)
- Overlap matrix de-quantization
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def dequantize_overlap(quantized: np.ndarray) -> np.ndarray:
    """
    De-quantize uint8 [0, 255] to float32 [0.0, 1.0].

    Args:
        quantized: (T_a, T_b) uint8 array in [0, 255]

    Returns:
        (T_a, T_b) float32 array in [0.0, 1.0]
    """
    return quantized.astype(np.float32) / 255.0


def compute_window_score(
    matrix: np.ndarray | torch.Tensor,
    start_a: int,
    start_b: int,
    window_size: int = 5,
    score_type: str = "max_mean",
) -> float:
    """
    Compute overlap score for a window_size x window_size sub-block.

    Args:
        matrix: (T_a, T_b) overlap matrix
        start_a: A sequence start frame index
        start_b: B sequence start frame index
        window_size: Number of frames per group (default 5)
        score_type: Scoring strategy
            - "max_mean": (mean(max(sub, axis=1)) + mean(max(sub, axis=0))) / 2
            - "mean": Simple average of all entries
            - "min_max": min of row-max and col-max means (most conservative)

    Returns:
        Window overlap score [0.0, 1.0]
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    T_a, T_b = matrix.shape
    end_a = min(start_a + window_size, T_a)
    end_b = min(start_b + window_size, T_b)

    sub = matrix[start_a:end_a, start_b:end_b]
    if sub.size == 0:
        return 0.0

    if score_type == "max_mean":
        # Each A frame finds best B frame, each B frame finds best A frame
        a2b = float(np.mean(np.max(sub, axis=1)))
        b2a = float(np.mean(np.max(sub, axis=0)))
        return (a2b + b2a) / 2.0
    elif score_type == "mean":
        return float(np.mean(sub))
    elif score_type == "min_max":
        a2b = float(np.mean(np.max(sub, axis=1)))
        b2a = float(np.mean(np.max(sub, axis=0)))
        return min(a2b, b2a)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


def sliding_window_scores(
    matrix: np.ndarray | torch.Tensor,
    window_size: int = 5,
    stride: int = 1,
    score_type: str = "max_mean",
) -> np.ndarray:
    """
    Compute overlap scores for all sliding windows.

    Vectorized implementation: uses numpy cumsum + sliding max instead of a
    per-window Python loop, giving 100x+ speedup on long sequences (1000+ frames).

    Args:
        matrix: (T_a, T_b) overlap matrix
        window_size: Window size
        stride: Sliding stride
        score_type: Scoring strategy

    Returns:
        (num_windows_a, num_windows_b) score matrix
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    T_a, T_b = matrix.shape
    num_a = max(1, (T_a - window_size) // stride + 1)
    num_b = max(1, (T_b - window_size) // stride + 1)

    if score_type == "mean":
        return _sliding_window_scores_mean(matrix, window_size, stride, num_a, num_b)
    elif score_type in ("max_mean", "min_max"):
        return _sliding_window_scores_max_mean(
            matrix, window_size, stride, num_a, num_b, score_type,
        )
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


def _sliding_window_scores_mean(
    matrix: np.ndarray,
    window_size: int,
    stride: int,
    num_a: int,
    num_b: int,
) -> np.ndarray:
    """2D mean sliding window: uses cumsum for O(T_a * T_b) complexity."""
    # 2D prefix sum
    cum = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1), dtype=np.float64)
    cum[1:, 1:] = np.cumsum(np.cumsum(matrix.astype(np.float64), axis=0), axis=1)

    w = window_size
    area = w * w
    scores = np.empty((num_a, num_b), dtype=np.float32)
    starts_a = np.arange(num_a) * stride
    starts_b = np.arange(num_b) * stride
    for i, sa in enumerate(starts_a):
        ea = sa + w
        row_sums = cum[ea, starts_b + w] - cum[ea, starts_b] - cum[sa, starts_b + w] + cum[sa, starts_b]
        scores[i, :] = row_sums / area
    return scores


def _sliding_window_scores_max_mean(
    matrix: np.ndarray,
    window_size: int,
    stride: int,
    num_a: int,
    num_b: int,
    score_type: str,
) -> np.ndarray:
    """
    max_mean / min_max sliding window: vectorized implementation.

    For each window [sa:sa+w, sb:sb+w]:
      a2b = mean(max(sub, axis=1))  -- mean of per-row maxima
      b2a = mean(max(sub, axis=0))  -- mean of per-column maxima

    Strategy: first apply a sliding max along the B axis to get the per-row
    maxima of shape [T_a, num_b], then compute the window mean along the A axis
    using cumsum. The column direction is handled symmetrically.
    """
    T_a, T_b = matrix.shape
    w = window_size
    mat = matrix.astype(np.float32)

    # --- a2b: per-row maximum within the B window, then averaged within the A window ---
    # row_max_over_b[r, j] = max(mat[r, sb:sb+w]) for window j along B
    row_max_over_b = _sliding_max_1d(mat, w, stride, axis=1, num_out=num_b)
    # row_max_over_b: [T_a, num_b]

    # Compute the window mean of row_max_over_b along the A axis
    cum_a = np.zeros((T_a + 1, num_b), dtype=np.float64)
    cum_a[1:, :] = np.cumsum(row_max_over_b.astype(np.float64), axis=0)
    starts_a = np.arange(num_a) * stride
    a2b = np.empty((num_a, num_b), dtype=np.float32)
    for i, sa in enumerate(starts_a):
        a2b[i, :] = (cum_a[sa + w, :] - cum_a[sa, :]) / w

    # --- b2a: per-column maximum within the A window, then averaged within the B window ---
    col_max_over_a = _sliding_max_1d(mat, w, stride, axis=0, num_out=num_a)
    # col_max_over_a: [num_a, T_b]

    cum_b = np.zeros((num_a, T_b + 1), dtype=np.float64)
    cum_b[:, 1:] = np.cumsum(col_max_over_a.astype(np.float64), axis=1)
    starts_b = np.arange(num_b) * stride
    b2a = np.empty((num_a, num_b), dtype=np.float32)
    for j, sb in enumerate(starts_b):
        b2a[:, j] = (cum_b[:, sb + w] - cum_b[:, sb]) / w

    if score_type == "max_mean":
        return (a2b + b2a) / 2.0
    else:  # min_max
        return np.minimum(a2b, b2a)


def _sliding_max_1d(
    mat: np.ndarray,
    window_size: int,
    stride: int,
    axis: int,
    num_out: int,
) -> np.ndarray:
    """
    Apply a 1D sliding max to the matrix along the given axis.

    axis=1: per row, returns [T_a, num_out], where out[r,j] = max(mat[r, j*stride : j*stride+w])
    axis=0: per column, returns [num_out, T_b], where out[i,c] = max(mat[i*stride : i*stride+w, c])
    """
    w = window_size
    if axis == 1:
        T_a, T_b = mat.shape
        out = np.empty((T_a, num_out), dtype=mat.dtype)
        for j in range(num_out):
            sb = j * stride
            out[:, j] = np.max(mat[:, sb:sb + w], axis=1)
        return out
    else:
        T_a, T_b = mat.shape
        out = np.empty((num_out, T_b), dtype=mat.dtype)
        for i in range(num_out):
            sa = i * stride
            out[i, :] = np.max(mat[sa:sa + w, :], axis=0)
        return out


def find_best_window(
    matrix: np.ndarray | torch.Tensor,
    window_size: int = 5,
    stride: int = 1,
    score_type: str = "max_mean",
) -> tuple[int, int, float]:
    """
    Find the best window position in the overlap matrix.

    Args:
        matrix: (T_a, T_b) overlap matrix
        window_size: Window size
        stride: Search stride
        score_type: Scoring strategy

    Returns:
        (best_start_a, best_start_b, best_score)
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    T_a, T_b = matrix.shape
    best_score = -1.0
    best_a, best_b = 0, 0

    for start_a in range(0, max(1, T_a - window_size + 1), stride):
        for start_b in range(0, max(1, T_b - window_size + 1), stride):
            score = compute_window_score(
                matrix, start_a, start_b, window_size, score_type
            )
            if score > best_score:
                best_score = score
                best_a = start_a
                best_b = start_b

    return best_a, best_b, max(0.0, best_score)


def find_topk_windows(
    matrix: np.ndarray | torch.Tensor,
    window_size: int = 5,
    stride: int = 1,
    k: int = 10,
    score_type: str = "max_mean",
    min_distance: int = 2,
) -> list[tuple[int, int, float]]:
    """
    Find top-K window positions with non-maximum suppression.

    Args:
        matrix: (T_a, T_b) overlap matrix
        window_size: Window size
        stride: Search stride
        k: Number of candidates to return
        score_type: Scoring strategy
        min_distance: Minimum frame distance between candidates (for NMS)

    Returns:
        List of (start_a, start_b, score) tuples, sorted by score descending
    """
    scores = sliding_window_scores(matrix, window_size, stride, score_type)

    # Flatten and sort
    flat_indices = np.argsort(scores.ravel())[::-1]

    candidates = []
    for idx in flat_indices:
        if len(candidates) >= k:
            break

        i = idx // scores.shape[1]
        j = idx % scores.shape[1]
        start_a = i * stride
        start_b = j * stride
        score = scores[i, j]

        # Non-maximum suppression: skip if too close to existing candidates
        too_close = False
        for ca, cb, _ in candidates:
            if abs(start_a - ca) < min_distance and abs(start_b - cb) < min_distance:
                too_close = True
                break

        if not too_close:
            candidates.append((start_a, start_b, float(score)))

    return candidates


def compute_window_overlap_batch(
    matrix_batch: torch.Tensor,
    starts_a: torch.Tensor,
    starts_b: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Compute window overlap scores for a batch (differentiable).

    Args:
        matrix_batch: (B, T_a, T_b) overlap matrix batch
        starts_a: (B,) start indices for A
        starts_b: (B,) start indices for B
        window_size: Window size

    Returns:
        (B,) window scores
    """
    B, T_a, T_b = matrix_batch.shape
    device = matrix_batch.device

    scores = []
    for b in range(B):
        sa = int(starts_a[b].item())
        sb = int(starts_b[b].item())
        ea = min(sa + window_size, T_a)
        eb = min(sb + window_size, T_b)

        sub = matrix_batch[b, sa:ea, sb:eb]
        if sub.numel() == 0:
            scores.append(torch.tensor(0.0, device=device))
            continue

        # Max-Mean score
        a2b = sub.max(dim=1)[0].mean()
        b2a = sub.max(dim=0)[0].mean()
        scores.append((a2b + b2a) / 2.0)

    return torch.stack(scores)


def create_position_encoding(
    max_len: int,
    embed_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Create sinusoidal position encoding.

    Args:
        max_len: Maximum sequence length
        embed_dim: Embedding dimension
        device: Target device

    Returns:
        (max_len, embed_dim) position encoding
    """
    position = torch.arange(max_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device) * (-np.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(max_len, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

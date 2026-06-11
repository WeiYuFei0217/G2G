"""
stage1_inference.py -- Covisibility batched inference engine + sliding window selection

Provides Stage1InferenceEngine for efficient batched inference of a covisibility
model:
  1. encode_sequence: encode an entire sequence, caching the encoder patch tokens
  2. predict_overlap_matrix: predict the N*M overlap matrix from cached features
  3. predict_covisibility_maps: predict per-pixel covisibility maps for given frame pairs

Also provides the select_windows_with_union function for sliding window Top-K
selection + frame union computation.

Usage:
    from g2g.retrieval.stage1_inference import (
        Stage1InferenceEngine,
        select_windows_with_union,
    )

    # Initialize the engine
    engine = Stage1InferenceEngine(model, device)

    # Encode and predict
    engine.encode_sequence(images_a, "seq_a")
    engine.encode_sequence(images_b, "seq_b")
    overlap = engine.predict_overlap_matrix("seq_a", "seq_b")
    windows = select_windows_with_union(overlap, window_size=5, top_k=3)

    # Optional: predict per-pixel covisibility maps
    covis = engine.predict_covisibility_maps("seq_a", "seq_b", windows["union_pairs"])

    engine.clear_cache()
"""

from __future__ import annotations

import logging
from itertools import product

import numpy as np
import torch
import torch.nn as nn

from .utils import find_topk_windows

logger = logging.getLogger(__name__)


class Stage1InferenceEngine:
    """
    Covisibility batched inference engine.

    Core idea: the DINOv3 encoder runs independently per frame -> cache the patch
    tokens -> the decoder needs the features of two frames for cross-attention ->
    run inference pair by pair (in batches).

    GPU memory analysis (cache_on_gpu=True, default):
      - Caching a single 50-frame sequence: 50 * 196 * 768 * 4B ~= 29 MB (on GPU)
      - Two sequences ~= 58 MB, completely negligible for a 24GB+ GPU
      - Decoder batch of 64 pairs ~= 64 * 196 * 768 * 4B * 2 ~= 75 MB per batch
      - Compared to CPU caching, this avoids the per-batch CPU->GPU transfer
        overhead during the decoder stage
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        encoder_batch_size: int = 32,
        decoder_batch_size: int = 64,
        use_amp: bool = True,
        cache_on_gpu: bool = True,
    ):
        """
        Args:
            model: CovisibilityModel instance (decoder weights already loaded)
            device: inference device
            encoder_batch_size: encoder batched inference size
            decoder_batch_size: decoder frame-pair batched inference size
            use_amp: whether to use AMP mixed precision
            cache_on_gpu: whether to cache encoder features on the GPU (avoids the
                          repeated CPU->GPU transfers during decoder inference; two
                          sequences of 50 frames each need only ~57MB of GPU memory)
        """
        self.model = model
        self.device = device
        self.encoder_batch_size = encoder_batch_size
        self.decoder_batch_size = decoder_batch_size
        self.use_amp = use_amp
        self.cache_on_gpu = cache_on_gpu

        # Feature cache: {cache_key: torch.Tensor [T, num_patches, d_model]}
        self._feature_cache: dict[str, torch.Tensor] = {}

        # Ensure the model is in eval mode
        self.model.eval()

    def encode_sequence(
        self,
        images: torch.Tensor,
        cache_key: str,
    ) -> None:
        """
        Encode an entire sequence, caching the encoder patch tokens.

        Args:
            images: [T, 3, H, W] float32 tensor (normalized to [0, 1] or the range
                the model expects)
            cache_key: cache key name (e.g. "scene/traj_a/cam_3")
        """
        T = images.shape[0]
        all_tokens = []

        with torch.no_grad():
            for start in range(0, T, self.encoder_batch_size):
                end = min(start + self.encoder_batch_size, T)
                batch = images[start:end].to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    # encoder.forward_tokens: [B, num_patches, d_model]
                    tokens = self.model.encoder.forward_tokens(batch)

                # Convert to float32; storage location depends on cache_on_gpu
                if self.cache_on_gpu:
                    all_tokens.append(tokens.float())
                else:
                    all_tokens.append(tokens.float().cpu())

        # Concatenate and cache: [T, num_patches, d_model]
        self._feature_cache[cache_key] = torch.cat(all_tokens, dim=0)
        logger.debug(
            "Encoded %d frames for '%s', shape=%s",
            T, cache_key, self._feature_cache[cache_key].shape,
        )

    def predict_overlap_matrix(
        self,
        key_a: str,
        key_b: str,
    ) -> np.ndarray:
        """
        Predict the N*M overlap matrix (symmetric = min(a2b, b2a)).

        Takes feat_a[i], feat_b[j] from the cache, feeds them into the decoder in
        batches, and extracts the scalar overlap.

        Args:
            key_a: cache key of sequence A
            key_b: cache key of sequence B

        Returns:
            [N, M] float32 numpy array, overlap values in [0, 1]
        """
        feat_a = self._feature_cache[key_a]  # [N, P, D]
        feat_b = self._feature_cache[key_b]  # [M, P, D]
        N, M = feat_a.shape[0], feat_b.shape[0]

        # Generate all frame-pair indices
        pairs = list(product(range(N), range(M)))
        total_pairs = len(pairs)

        # overlap result matrices
        overlap_a2b = np.zeros((N, M), dtype=np.float32)
        overlap_b2a = np.zeros((N, M), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, total_pairs, self.decoder_batch_size):
                end = min(start + self.decoder_batch_size, total_pairs)
                batch_pairs = pairs[start:end]

                # Build the batch: gather the encoder tokens of the corresponding frames
                idx_a = [p[0] for p in batch_pairs]
                idx_b = [p[1] for p in batch_pairs]

                # Gather the corresponding frame features (no transfer when cached on
                # GPU; moved to GPU when cached on CPU)
                batch_feat_a = feat_a[idx_a]  # [B, P, D]
                batch_feat_b = feat_b[idx_b]  # [B, P, D]
                if not self.cache_on_gpu:
                    batch_feat_a = batch_feat_a.to(self.device)
                    batch_feat_b = batch_feat_b.to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    outputs = self.model.decoder(batch_feat_a, batch_feat_b)

                # Parse the outputs
                if self.model.use_overlap_head:
                    _logits_a, _logits_b, ov_a, ov_b = outputs
                    ov_a = ov_a.float().cpu().numpy()
                    ov_b = ov_b.float().cpu().numpy()
                else:
                    logits_a, logits_b = outputs
                    # Compute overlap from the pixel path
                    covis_a = torch.sigmoid(logits_a.float())
                    covis_b = torch.sigmoid(logits_b.float())
                    ov_a = covis_a.mean(dim=(1, 2)).cpu().numpy()
                    ov_b = covis_b.mean(dim=(1, 2)).cpu().numpy()

                # Write into the matrices
                for k, (i, j) in enumerate(batch_pairs):
                    overlap_a2b[i, j] = ov_a[k]
                    overlap_b2a[i, j] = ov_b[k]

        # symmetric overlap = min(a2b, b2a)
        overlap = np.minimum(overlap_a2b, overlap_b2a)
        return overlap

    def predict_overlap_for_pairs(
        self,
        key_a: str,
        key_b: str,
        frame_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """
        Run the decoder only on the given list of frame pairs and return the
        symmetric overlap values.

        Same logic as predict_overlap_matrix, but iterates only over the passed-in
        frame_pairs instead of the full Cartesian product; used in hybrid mode to
        refine the candidate windows pre-filtered by GT.

        Args:
            key_a: cache key of sequence A
            key_b: cache key of sequence B
            frame_pairs: list of frame pairs [(frame_idx_a, frame_idx_b), ...]

        Returns:
            {(fa, fb): symmetric_overlap} dict
        """
        if not frame_pairs:
            return {}

        feat_a = self._feature_cache[key_a]  # [N, P, D]
        feat_b = self._feature_cache[key_b]  # [M, P, D]

        results: dict[tuple[int, int], float] = {}

        with torch.no_grad():
            for start in range(0, len(frame_pairs), self.decoder_batch_size):
                end = min(start + self.decoder_batch_size, len(frame_pairs))
                batch_pairs = frame_pairs[start:end]

                idx_a = [p[0] for p in batch_pairs]
                idx_b = [p[1] for p in batch_pairs]

                batch_feat_a = feat_a[idx_a]  # [B, P, D]
                batch_feat_b = feat_b[idx_b]  # [B, P, D]
                if not self.cache_on_gpu:
                    batch_feat_a = batch_feat_a.to(self.device)
                    batch_feat_b = batch_feat_b.to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    outputs = self.model.decoder(batch_feat_a, batch_feat_b)

                # Parse the outputs, compute symmetric overlap = min(a2b, b2a)
                if self.model.use_overlap_head:
                    _logits_a, _logits_b, ov_a, ov_b = outputs
                    ov_a = ov_a.float().cpu().numpy()
                    ov_b = ov_b.float().cpu().numpy()
                else:
                    logits_a, logits_b = outputs
                    covis_a = torch.sigmoid(logits_a.float())
                    covis_b = torch.sigmoid(logits_b.float())
                    ov_a = covis_a.mean(dim=(1, 2)).cpu().numpy()
                    ov_b = covis_b.mean(dim=(1, 2)).cpu().numpy()

                for k, (fa, fb) in enumerate(batch_pairs):
                    results[(fa, fb)] = float(min(ov_a[k], ov_b[k]))

        return results

    def predict_covisibility_maps(
        self,
        key_a: str,
        key_b: str,
        frame_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], dict]:
        """
        Predict per-pixel covisibility maps for the given frame pairs.

        Args:
            key_a: cache key of sequence A
            key_b: cache key of sequence B
            frame_pairs: list of frame pairs [(frame_idx_a, frame_idx_b), ...]

        Returns:
            {(fa, fb): {"covis_a": np.ndarray [H, W],
                        "covis_b": np.ndarray [H, W],
                        "overlap": float}} dict
        """
        if not frame_pairs:
            return {}

        feat_a = self._feature_cache[key_a]
        feat_b = self._feature_cache[key_b]

        results: dict[tuple[int, int], dict] = {}

        with torch.no_grad():
            for start in range(0, len(frame_pairs), self.decoder_batch_size):
                end = min(start + self.decoder_batch_size, len(frame_pairs))
                batch_pairs = frame_pairs[start:end]

                idx_a = [p[0] for p in batch_pairs]
                idx_b = [p[1] for p in batch_pairs]

                # Gather the corresponding frame features
                batch_feat_a = feat_a[idx_a]
                batch_feat_b = feat_b[idx_b]
                if not self.cache_on_gpu:
                    batch_feat_a = batch_feat_a.to(self.device)
                    batch_feat_b = batch_feat_b.to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    outputs = self.model.decoder(batch_feat_a, batch_feat_b)

                if self.model.use_overlap_head:
                    logits_a, logits_b, ov_a, ov_b = outputs
                else:
                    logits_a, logits_b = outputs

                # per-pixel covisibility (sigmoid)
                covis_a = torch.sigmoid(logits_a.float()).cpu().numpy()
                covis_b = torch.sigmoid(logits_b.float()).cpu().numpy()

                # scalar overlap
                if self.model.use_overlap_head:
                    overlap_vals = torch.min(ov_a, ov_b).float().cpu().numpy()
                else:
                    mean_a = covis_a.mean(axis=(1, 2))
                    mean_b = covis_b.mean(axis=(1, 2))
                    overlap_vals = np.minimum(mean_a, mean_b)

                for k, (fa, fb) in enumerate(batch_pairs):
                    results[(fa, fb)] = {
                        "covis_a": covis_a[k],
                        "covis_b": covis_b[k],
                        "overlap": float(overlap_vals[k]),
                    }

        return results

    def clear_cache(self, key: str | None = None) -> None:
        """
        Clear the feature cache to free GPU memory.

        Args:
            key: the cache key to clear. None clears all of them.
        """
        if key is None:
            self._feature_cache.clear()
            logger.debug("Cleared all feature caches")
        elif key in self._feature_cache:
            del self._feature_cache[key]
            logger.debug("Cleared cache for '%s'", key)

    @property
    def cached_keys(self) -> list[str]:
        """Return all key names currently cached."""
        return list(self._feature_cache.keys())

    def get_cache_info(self) -> dict[str, tuple[int, ...]]:
        """Return the tensor shape of each key in the cache."""
        return {k: tuple(v.shape) for k, v in self._feature_cache.items()}


def select_windows_with_union(
    overlap_matrix: np.ndarray,
    window_size: int = 5,
    top_k: int = 3,
    stride: int = 1,
    score_type: str = "max_mean",
    min_distance: int = 2,
    min_overlap_threshold: float = 0.1,
    max_overlap_threshold: float = 1.0,
) -> dict:
    """
    Sliding window Top-K selection + frame union computation.

    Algorithm:
      1. Call find_topk_windows() to obtain the Top-K windows (with NMS deduplication)
      2. Filter windows: keep a window if it satisfies either condition:
         a) score (max_mean) >= min_overlap_threshold
         b) any element in the window submatrix is >= 3 * min_overlap_threshold
      3. Extra filter: windows with score >= max_overlap_threshold are discarded
      4. Generate the frame index list for each window
      5. Compute the frame union over all K windows
      6. Compute the set of frame pairs whose covisibility needs to be precomputed

    Args:
        overlap_matrix: [N, M] float32 overlap matrix
        window_size: window size
        top_k: number of windows to return
        stride: sliding stride
        score_type: scoring strategy ("max_mean", "mean", "min_max")
        min_distance: NMS minimum distance
        min_overlap_threshold: minimum overlap threshold (windows below it are discarded)
        max_overlap_threshold: maximum overlap threshold (windows >= it are discarded,
            default 1.0 means no filtering)

    Returns:
        dict containing:
          "windows": list[dict] - information for each window
          "union_frames_a": list[int] - union of A frames over all windows
          "union_frames_b": list[int] - union of B frames over all windows
          "union_pairs": list[tuple[int, int]] - union of frame pairs over all windows
          "num_windows": int - actual number of valid windows
    """
    N, M = overlap_matrix.shape

    # Get the Top-K window candidates
    candidates = find_topk_windows(
        overlap_matrix,
        window_size=window_size,
        stride=stride,
        k=top_k,
        score_type=score_type,
        min_distance=min_distance,
    )

    # Build the window list, filtering out low-scoring windows
    windows = []
    all_frames_a: set[int] = set()
    all_frames_b: set[int] = set()
    all_pairs: set[tuple[int, int]] = set()

    for rank, (start_a, start_b, score) in enumerate(candidates):
        # Generate the window frame indices (clamped to the sequence boundaries)
        end_a = min(start_a + window_size, N)
        end_b = min(start_b + window_size, M)

        # Filter: condition a) score >= threshold, or b) window max >= 3 * threshold
        if score < min_overlap_threshold:
            sub = overlap_matrix[start_a:end_a, start_b:end_b]
            if sub.size == 0 or float(np.max(sub)) < 3.0 * min_overlap_threshold:
                continue

        # Upper-bound filter: discard windows with score >= max_overlap_threshold
        if max_overlap_threshold < 1.0 and score >= max_overlap_threshold:
            continue

        indices_a = list(range(start_a, end_a))
        indices_b = list(range(start_b, end_b))

        windows.append({
            "rank": rank,
            "start_a": start_a,
            "start_b": start_b,
            "indices_a": indices_a,
            "indices_b": indices_b,
            "score": score,
        })

        # Accumulate the frame union
        all_frames_a.update(indices_a)
        all_frames_b.update(indices_b)

        # All frame pairs within the window
        for fa in indices_a:
            for fb in indices_b:
                all_pairs.add((fa, fb))

    return {
        "windows": windows,
        "union_frames_a": sorted(all_frames_a),
        "union_frames_b": sorted(all_frames_b),
        "union_pairs": sorted(all_pairs),
        "num_windows": len(windows),
    }

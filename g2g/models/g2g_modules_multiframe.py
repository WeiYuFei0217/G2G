"""
G2G Multi-Frame Pose Estimation Modules.

Key differences from g2g_modules.py:
1. G2GBridgeMultiFrame: alternating inter-group/intra-group attention + merged self-attention, outputs tokens for all frames
2. MultiFramePoseHead: predicts the pose of all frames relative to A0 (A1-A4 + B0-B4)

Architecture:
  Phase 0: frame positional encoding + anchor (A0 only)
  Phase 1: alternating attention (inter -> intra) x N
  Phase 2: merge A+B -> group_embed + anchor re-injection -> merged self-attn x M
  Phase 3: output tokens for all frames -> MultiFramePoseHead -> 9 poses

Note: in all released configs/checkpoints, num_alternating_pairs=0
(bridge_alternating_pairs: 0), so Phase 1 is skipped and only Phase 2
(merged self-attention) is active; the alternating layers are an optional capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .g2g_modules import (
    G2GCrossAttentionBlock,
    IntraGroupSelfAttentionBlock,
    RotationUtils,
)


class G2GBridgeMultiFrame(nn.Module):
    """
    Multi-Frame G2G Bridge: alternating inter-group/intra-group attention + merged self-attention.

    Data flow:
      1. Add frame_embed + anchor_embed (A0 only; B0 does not get the anchor)
      2. (Inter-group cross-attn -> Intra-group self-attn) x num_alternating_pairs
      3. Merge A+B -> add group_embed + re-inject anchor on A0 -> merged self-attn x num_merged_layers
      4. Output: all_tokens_a [B, W, L, C], all_tokens_b [B, W, L, C]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_alternating_pairs: int = 2,
        num_merged_layers: int = 2,
        num_latents_per_frame: int = 32,
        num_frames: int = 5,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        reinject_anchor_after_merge: bool = True,
        use_anchor_embed: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents_per_frame = num_latents_per_frame
        self.num_frames = num_frames
        self.reinject_anchor_after_merge = reinject_anchor_after_merge
        self.use_anchor_embed = use_anchor_embed

        # Frame positional encoding (shared between A and B)
        self.frame_embed = nn.Parameter(torch.randn(num_frames, 1, embed_dim) * 0.02)

        # Anchor marker: used by A0 only (can be disabled for ablation)
        if use_anchor_embed:
            self.anchor_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Phase 1: alternating attention layers
        self.inter_layers_a = nn.ModuleList([
            G2GCrossAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_alternating_pairs)
        ])
        self.inter_layers_b = nn.ModuleList([
            G2GCrossAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_alternating_pairs)
        ])
        self.intra_layers_a = nn.ModuleList([
            IntraGroupSelfAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_alternating_pairs)
        ])
        self.intra_layers_b = nn.ModuleList([
            IntraGroupSelfAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_alternating_pairs)
        ])

        # Phase 2: layers after merging
        # Group embedding: distinguishes the A/B groups after merging
        self.group_embed = nn.Parameter(torch.randn(2, 1, embed_dim) * 0.02)

        # Anchor re-injection (reinforces A0's reference-frame identity after merging; can be disabled for ablation)
        if reinject_anchor_after_merge and use_anchor_embed:
            self.anchor_embed_merge = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Merged self-attention
        self.merged_layers = nn.ModuleList([
            IntraGroupSelfAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_merged_layers)
        ])

        self.out_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        if self.use_anchor_embed:
            nn.init.trunc_normal_(self.anchor_embed, std=0.02)
        nn.init.trunc_normal_(self.group_embed, std=0.02)
        if self.reinject_anchor_after_merge and self.use_anchor_embed:
            nn.init.trunc_normal_(self.anchor_embed_merge, std=0.02)

    def _add_frame_position_a(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Group A: add frame positional encoding + anchor (A0 only).

        Args:
            latents: [B, N_frames, num_latents, C]
        Returns:
            tokens: [B, N_frames * num_latents, C]
        """
        B, N_frames, N_latents, C = latents.shape

        frame_pe = self.frame_embed[:N_frames]
        latents = latents + frame_pe.unsqueeze(0)

        if self.use_anchor_embed:
            anchor_pe = self.anchor_embed.expand(B, 1, N_latents, C)
            anchor_mask = torch.zeros_like(latents)
            anchor_mask[:, 0:1] = anchor_pe
            latents = latents + anchor_mask

        return latents.reshape(B, N_frames * N_latents, C)

    def _add_frame_position_b(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Group B: add frame positional encoding only (no anchor).

        Args:
            latents: [B, N_frames, num_latents, C]
        Returns:
            tokens: [B, N_frames * num_latents, C]
        """
        B, N_frames, N_latents, C = latents.shape

        frame_pe = self.frame_embed[:N_frames]
        latents = latents + frame_pe.unsqueeze(0)

        return latents.reshape(B, N_frames * N_latents, C)

    def forward(
        self,
        latents_a: torch.Tensor,
        latents_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents_a: [B, W, L, C]
            latents_b: [B, W, L, C]

        Returns:
            all_tokens_a: [B, W, L, C] - tokens for all A frames
            all_tokens_b: [B, W, L, C] - tokens for all B frames
        """
        B, W, L, C = latents_a.shape

        # === Phase 0: frame positional encoding + anchor ===
        tokens_a = self._add_frame_position_a(latents_a)  # [B, W*L, C]
        tokens_b = self._add_frame_position_b(latents_b)  # [B, W*L, C]

        # === Phase 1: alternating inter/intra ===
        for inter_a, inter_b, intra_a, intra_b in zip(
            self.inter_layers_a, self.inter_layers_b,
            self.intra_layers_a, self.intra_layers_b,
        ):
            tokens_a_new = inter_a(tokens_a, tokens_b)
            tokens_b_new = inter_b(tokens_b, tokens_a)
            tokens_a, tokens_b = tokens_a_new, tokens_b_new

            tokens_a = intra_a(tokens_a)
            tokens_b = intra_b(tokens_b)

        # === Phase 2: merge + merged self-attn ===
        # Add group embedding to distinguish A/B
        tokens_a_framed = tokens_a.reshape(B, W, L, C) + self.group_embed[0:1]
        tokens_b_framed = tokens_b.reshape(B, W, L, C) + self.group_embed[1:2]

        merged = torch.cat([
            tokens_a_framed.reshape(B, W * L, C),
            tokens_b_framed.reshape(B, W * L, C),
        ], dim=1)  # [B, 2*W*L, C]

        # Re-inject the anchor at the A0 position (after merging, A0 tokens are at [0:L])
        if self.reinject_anchor_after_merge and self.use_anchor_embed:
            anchor_val = self.anchor_embed_merge.expand(B, L, -1)
            merged = torch.cat([
                merged[:, :L] + anchor_val,
                merged[:, L:],
            ], dim=1)

        for layer in self.merged_layers:
            merged = layer(merged)

        merged = self.out_norm(merged)

        # === Phase 3: split back into per-frame tokens ===
        all_tokens_a = merged[:, :W * L].reshape(B, W, L, C)
        all_tokens_b = merged[:, W * L:].reshape(B, W, L, C)

        return all_tokens_a, all_tokens_b


class MultiFramePoseHead(nn.Module):
    """
    Multi-frame pose prediction head.

    For the 9 frames A1-A4 and B0-B4, uses shared weights to predict the pose of each frame relative to A0.
    The frame_identity_embed distinguishes the prediction tasks of different frames.

    Implemented in a batched manner: the predictions for all 9 frames are computed in a single cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        num_frames: int = 5,
        rotation_repr: str = "6d",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.rotation_repr = rotation_repr

        if rotation_repr == "6d":
            rot_dim = 6
        elif rotation_repr == "quaternion":
            rot_dim = 4
        elif rotation_repr == "axis_angle":
            rot_dim = 3
        else:
            raise ValueError(f"Unknown rotation representation: {rotation_repr}")
        self.rot_dim = rot_dim

        # Number of poses to predict: (W-1) for A + W for B = 2W - 1
        self.num_poses = 2 * num_frames - 1

        # Shared pose queries (rotation + translation)
        self.pose_queries = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        # Shared cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_after_attn = nn.LayerNorm(embed_dim)

        # Shared MLP + prediction heads
        def _mlp(in_d, hid_d, n):
            layers = []
            for i in range(n):
                layers += [
                    nn.Linear(in_d if i == 0 else hid_d, hid_d),
                    nn.LayerNorm(hid_d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            return nn.Sequential(*layers)

        self.rot_mlp = _mlp(embed_dim, hidden_dim, num_layers - 1)
        self.trans_mlp = _mlp(embed_dim, hidden_dim, num_layers - 1)
        self.rotation_head = nn.Linear(hidden_dim, rot_dim)
        self.translation_head = nn.Linear(hidden_dim, 3)

        # Per-frame identity encoding (9: A1,A2,A3,A4,B0,B1,B2,B3,B4)
        self.frame_identity_embed = nn.Parameter(
            torch.randn(self.num_poses, 1, embed_dim) * 0.02,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pose_queries, std=0.02)
        nn.init.trunc_normal_(self.frame_identity_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.rotation_head.weight)
        nn.init.xavier_uniform_(self.translation_head.weight)

        if self.rotation_repr == "6d":
            self.rotation_head.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        elif self.rotation_repr == "quaternion":
            self.rotation_head.bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0])
        else:
            nn.init.zeros_(self.rotation_head.bias)

        nn.init.zeros_(self.translation_head.bias)

    def forward(
        self,
        all_tokens_a: torch.Tensor,
        all_tokens_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched prediction of the pose of all frames (relative to A0).

        Args:
            all_tokens_a: [B, W, L, C]
            all_tokens_b: [B, W, L, C]

        Returns:
            rotations_a: [B, W-1, rot_dim]   (A1-A4)
            translations_a: [B, W-1, 3]
            rotations_b: [B, W, rot_dim]     (B0-B4)
            translations_b: [B, W, 3]
        """
        B, W, L, C = all_tokens_a.shape
        ref = all_tokens_a[:, 0]  # A0 tokens [B, L, C]

        # Collect all target frames: A1..A(W-1), B0..B(W-1)
        target_list = []
        for i in range(1, W):
            target_list.append(all_tokens_a[:, i])
        for j in range(W):
            target_list.append(all_tokens_b[:, j])

        targets = torch.stack(target_list, dim=1)  # [B, num_poses, L, C]
        num_poses = targets.shape[1]

        # Add frame identity encoding
        targets = targets + self.frame_identity_embed[:num_poses].unsqueeze(0)

        # Batched: reshape to [B*num_poses, ...]
        ref_expanded = ref.unsqueeze(1).expand(-1, num_poses, -1, -1)
        ref_flat = ref_expanded.reshape(B * num_poses, L, C)
        targets_flat = targets.reshape(B * num_poses, L, C)

        kv = torch.cat([ref_flat, targets_flat], dim=1)  # [B*P, 2L, C]
        kv_normed = self.norm_kv(kv)

        queries = self.pose_queries.unsqueeze(0).expand(B * num_poses, -1, -1)
        queries_normed = self.norm_query(queries)

        attended, _ = self.cross_attn(
            query=queries_normed,
            key=kv_normed,
            value=kv_normed,
        )  # [B*P, 2, C]
        attended = self.norm_after_attn(queries + attended)

        rot_feat = self.rot_mlp(attended[:, 0])    # [B*P, hidden]
        trans_feat = self.trans_mlp(attended[:, 1])

        rotations = self.rotation_head(rot_feat)       # [B*P, rot_dim]
        translations = self.translation_head(trans_feat)  # [B*P, 3]

        if self.rotation_repr == "quaternion":
            rotations = F.normalize(rotations, dim=-1)

        # Reshape: [B, num_poses, ...]
        rotations = rotations.reshape(B, num_poses, -1)
        translations = translations.reshape(B, num_poses, 3)

        # Split A and B
        n_a = W - 1
        rotations_a = rotations[:, :n_a]
        translations_a = translations[:, :n_a]
        rotations_b = rotations[:, n_a:]
        translations_b = translations[:, n_a:]

        return rotations_a, translations_a, rotations_b, translations_b

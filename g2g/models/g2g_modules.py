"""
G2G (Group-to-Group) Pose Estimation Modules.

Building blocks used by the multi-frame G2G pipeline (see g2g_modules_multiframe.py
and stage2_model_multiframe.py):
1. PerceiverResampler: compresses patch tokens into a fixed number of latent tokens
2. G2GCrossAttentionBlock / IntraGroupSelfAttentionBlock: inter-/intra-group attention blocks
3. RotationUtils: conversions between rotation representations (6D / quaternion / matrix)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: uses cross-attention to compress variable-length patch
    tokens into a fixed number of latent tokens.

    Mechanism: Query = latent tokens, Key/Value = image tokens
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: token embedding dimension
            num_latents: number of output latent tokens
            num_heads: number of multi-head attention heads
            num_layers: number of cross-attention layers
            ff_mult: hidden dimension multiplier for the FFN
            dropout: dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_heads = num_heads

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverCrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output LayerNorm
        self.out_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
        keep_frame_dim: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: input patch tokens [B, N, C] or [B, N_frames, N_patches, C]
            pos_embed: positional encoding [N, C] or [N_frames, N_patches, C]
            keep_frame_dim: whether to keep the frame dimension (returns [B, N_frames, num_latents, C])

        Returns:
            latents: compressed latent tokens
                    - keep_frame_dim=False: [B, num_latents, C]
                    - keep_frame_dim=True:  [B, N_frames, num_latents, C]
        """
        # Handle multi-frame input
        N_frames = None
        if x.dim() == 4:
            B, N_frames, N_patches, C = x.shape
            if keep_frame_dim:
                # Resample each frame separately: [B*N_frames, N_patches, C]
                x = x.reshape(B * N_frames, N_patches, C)
                B_orig = B
                B = B * N_frames
            else:
                # Original logic: flatten all frames
                x = x.reshape(B, N_frames * N_patches, C)
            if pos_embed is not None and pos_embed.dim() == 3:
                if keep_frame_dim:
                    pos_embed = pos_embed.reshape(N_frames * N_patches, C)
                else:
                    pos_embed = pos_embed.reshape(N_frames * N_patches, C)
        else:
            B = x.shape[0]

        # Inject positional encoding
        if pos_embed is not None:
            x = x + pos_embed.unsqueeze(0).expand(B, -1, -1)

        # Expand latents to the batch dimension
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Pass through the cross-attention layers
        for layer in self.layers:
            latents = layer(latents, x)

        latents = self.out_norm(latents)

        # Restore the frame dimension
        if keep_frame_dim and N_frames is not None:
            latents = latents.reshape(B_orig, N_frames, self.num_latents, -1)

        return latents


class PerceiverCrossAttentionBlock(nn.Module):
    """
    Perceiver Cross-Attention Block:
    contains cross-attention + FFN + LayerNorm
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: query tokens [B, num_latents, C]
            context: key/value tokens (image patches) [B, N, C]

        Returns:
            latents: updated latent tokens [B, num_latents, C]
        """
        # Cross-Attention
        latents_normed = self.norm1(latents)
        context_normed = self.norm_context(context)

        attn_out, _ = self.cross_attn(
            query=latents_normed,
            key=context_normed,
            value=context_normed,
        )
        latents = latents + attn_out

        # FFN
        latents = latents + self.ffn(self.norm2(latents))

        return latents


class IntraGroupSelfAttentionBlock(nn.Module):
    """
    Intra-group self-attention block: lets all frames interact with each other.

    Contains: self-attention + FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tokens [B, N, C]

        Returns:
            x: updated tokens [B, N, C]
        """
        # Self-Attention
        x_normed = self.norm1(x)
        attn_out, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class G2GCrossAttentionBlock(nn.Module):
    """Single-layer cross-attention block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-Attention: query comes from self, key/value come from other
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: query tokens [B, N, C]
            context: key/value tokens [B, M, C]

        Returns:
            x: updated tokens [B, N, C]
        """
        # Cross-Attention
        x_normed = self.norm1(x)
        context_normed = self.norm_context(context)
        attn_out, _ = self.cross_attn(
            query=x_normed,
            key=context_normed,
            value=context_normed,
        )
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class RotationUtils:
    """
    Utility class for converting between rotation representations.
    """

    @staticmethod
    def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert a 6D rotation representation into a rotation matrix.

        6D representation: the first two columns of the rotation matrix; the third
        column is obtained via Gram-Schmidt orthogonalization.

        Args:
            rot_6d: [B, 6] or [6]

        Returns:
            R: [B, 3, 3] or [3, 3]
        """
        squeeze = False
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)
            squeeze = True

        a1 = rot_6d[:, :3]  # [B, 3]
        a2 = rot_6d[:, 3:]  # [B, 3]

        # Gram-Schmidt orthogonalization using F.normalize.
        # F.normalize caps the backward gradient at 1/eps, which bounds the
        # max gradient magnitude to 1e6 instead of the previous 1e8,
        # preventing gradient explosions that cause NaN.
        b1 = F.normalize(a1, dim=-1, eps=1e-6)

        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1, eps=1e-6)

        b3 = torch.cross(b1, b2, dim=-1)

        R = torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]

        if squeeze:
            R = R.squeeze(0)

        return R

    @staticmethod
    def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
        """
        Convert a rotation matrix into a 6D representation.

        Args:
            R: [B, 3, 3] or [3, 3]

        Returns:
            rot_6d: [B, 6] or [6]
        """
        squeeze = False
        if R.dim() == 2:
            R = R.unsqueeze(0)
            squeeze = True

        # Take the first two columns
        rot_6d = R[:, :, :2].transpose(-1, -2).reshape(-1, 6)

        if squeeze:
            rot_6d = rot_6d.squeeze(0)

        return rot_6d

    @staticmethod
    def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Convert a quaternion into a rotation matrix.

        Args:
            q: [B, 4] quaternion (x, y, z, w)

        Returns:
            R: [B, 3, 3] rotation matrix
        """
        squeeze = False
        if q.dim() == 1:
            q = q.unsqueeze(0)
            squeeze = True

        # Normalize
        q = F.normalize(q, dim=-1)

        x, y, z, w = q.unbind(-1)

        # Compute the rotation matrix elements
        R = torch.stack([
            1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
            2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
            2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)

        if squeeze:
            R = R.squeeze(0)

        return R

    @staticmethod
    def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Convert a rotation matrix into a quaternion.

        Args:
            R: [B, 3, 3] rotation matrix

        Returns:
            q: [B, 4] quaternion (x, y, z, w)
        """
        squeeze = False
        if R.dim() == 2:
            R = R.unsqueeze(0)
            squeeze = True

        batch = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        q = torch.zeros(batch, 4, device=R.device, dtype=R.dtype)

        # Choose the most numerically stable computation path
        mask1 = trace > 0
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        mask4 = (~mask1) & (~mask2) & (~mask3)

        # Case 1: trace > 0
        if mask1.any():
            s = 0.5 / torch.sqrt(trace[mask1] + 1.0)
            q[mask1, 3] = 0.25 / s
            q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) * s
            q[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) * s
            q[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) * s

        # Case 2: R[0,0] is largest
        if mask2.any():
            s = 2.0 * torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2])
            q[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
            q[mask2, 0] = 0.25 * s
            q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
            q[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        # Case 3: R[1,1] is largest
        if mask3.any():
            s = 2.0 * torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2])
            q[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
            q[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
            q[mask3, 1] = 0.25 * s
            q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        # Case 4: R[2,2] is largest
        if mask4.any():
            s = 2.0 * torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1])
            q[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
            q[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
            q[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
            q[mask4, 2] = 0.25 * s

        if squeeze:
            q = q.squeeze(0)

        return q

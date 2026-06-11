"""
geometry.py -- vendored copy of a few geometry utilities.

G2G only uses two geometry utilities from MapAnything:
  - get_rays_in_camera_frame: generate a raymap in the camera frame from intrinsics (MapAnything's geometric input)
  - rotation_matrix_to_quaternion: rotation matrix -> quaternion (scalar-last, used for pose injection)

These two functions are official utilities released by MapAnything (Meta) under Apache-2.0.
Normally g2g imports them directly from `mapanything.utils.geometry`; this file serves as a
**fallback copy** to keep g2g working in case the official mapanything module layout changes.

----------------------------------------------------------------------
Portions of this file are adapted from MapAnything:
  Copyright (c) Meta Platforms, Inc. and affiliates.
  Licensed under the Apache License, Version 2.0.
References: DUSt3R, MoGe, PyTorch3D.
----------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F


def get_rays_in_camera_frame(intrinsics, height, width, normalize_to_unit_sphere):
    """
    Convert camera intrinsics to a raymap (ray origins + directions) in camera frame.
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Add batch dimension if not present
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device

    # Compute rays in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    ray_origins = torch.zeros((batch_size, height, width, 3), device=device)
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    ray_directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)

    # Normalize ray directions to unit sphere if required (else rays will lie on unit plane)
    if normalize_to_unit_sphere:
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=-1, keepdim=True
        )

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        ray_origins = ray_origins.squeeze(0)
        ray_directions = ray_directions.squeeze(0)

    return ray_origins, ray_directions


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out

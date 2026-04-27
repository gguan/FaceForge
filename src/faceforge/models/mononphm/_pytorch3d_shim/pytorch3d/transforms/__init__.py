"""SO(3) exp/log maps — Rodrigues formula in pure torch."""

import torch


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices via Rodrigues.

    Mirrors ``pytorch3d.transforms.so3_exp_map``.

    Args:
        log_rot: ``[N, 3]`` axis-angle vectors (axis * angle).
        eps:     small-angle threshold for the Taylor-series fallback.

    Returns:
        ``[N, 3, 3]`` rotation matrices.
    """
    if log_rot.dim() != 2 or log_rot.shape[-1] != 3:
        raise ValueError(f"log_rot must be [N, 3]; got {tuple(log_rot.shape)}")

    nrms = log_rot.norm(dim=1)               # [N]
    rot_angles = nrms.clamp(min=eps)          # avoid 0/0
    rot_angles_inv = 1.0 / rot_angles
    fac1 = torch.sin(rot_angles) * rot_angles_inv          # sin(θ)/θ
    fac2 = (1.0 - torch.cos(rot_angles)) * (rot_angles_inv ** 2)  # (1-cos(θ))/θ²

    # Skew-symmetric [v]× per batch entry.
    N = log_rot.shape[0]
    zero = torch.zeros(N, dtype=log_rot.dtype, device=log_rot.device)
    x, y, z = log_rot[:, 0], log_rot[:, 1], log_rot[:, 2]
    skew = torch.stack([
        zero, -z,  y,
           z, zero, -x,
          -y,  x,  zero,
    ], dim=1).reshape(N, 3, 3)

    skew_sq = skew @ skew
    eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device).expand(N, -1, -1)
    R = eye + fac1[:, None, None] * skew + fac2[:, None, None] * skew_sq

    # For tiny rotations, fall back to identity + skew.
    small = nrms < eps
    if small.any():
        R[small] = eye[small] + skew[small]
    return R


def so3_log_map(R: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors.

    Mirrors ``pytorch3d.transforms.so3_log_map``.

    Args:
        R: ``[N, 3, 3]`` rotation matrices.
        eps: small-angle threshold for the Taylor-series fallback.

    Returns:
        ``[N, 3]`` axis-angle vectors (axis * angle).
    """
    if R.dim() != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must be [N, 3, 3]; got {tuple(R.shape)}")

    # angle from trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]                # [N]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)                                # [N], in [0, π]

    # axis * sin(θ) extracted from skew-symmetric part of R
    rxy = R[:, 1, 0] - R[:, 0, 1]
    rxz = R[:, 0, 2] - R[:, 2, 0]
    ryz = R[:, 2, 1] - R[:, 1, 2]
    axis_sin = torch.stack([ryz, rxz, rxy], dim=1) * 0.5         # [N, 3] = sin(θ) * axis

    sin_theta = torch.sin(theta)
    factor = torch.where(
        sin_theta.abs() < eps,
        # Small angle: sin(θ)/θ ≈ 1 + small Taylor terms; use θ/sin(θ) ≈ 1 + θ²/6
        torch.ones_like(theta) + theta * theta / 6.0,
        theta / sin_theta,
    )
    return axis_sin * factor[:, None]

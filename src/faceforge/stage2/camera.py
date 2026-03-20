"""Camera utilities for Stage 2 optimization.

Provides intrinsics/extrinsics construction, point projection, and
clip-space projection for nvdiffrast.

Reference: VHAP render_nvdiffrast.py L117-160 (clip projection),
           pixel3dmm tracker.py (screen-space projection)
"""

import torch
import torch.nn.functional as F

from .flame_model import batch_rodrigues


def build_intrinsics(focal_length: torch.Tensor, principal_point: torch.Tensor,
                     image_size: int) -> torch.Tensor:
    """Build intrinsics matrix K from normalized parameters.

    Args:
        focal_length: [B, 1] normalized focal length
        principal_point: [B, 2] normalized principal point
        image_size: int, render resolution

    Returns:
        K: [B, 3, 3] intrinsics matrix
    """
    B = focal_length.shape[0]
    device, dtype = focal_length.device, focal_length.dtype

    fx = fy = focal_length * image_size  # [B, 1]
    cx = principal_point[:, 0:1] * image_size + image_size / 2  # [B, 1]
    cy = principal_point[:, 1:2] * image_size + image_size / 2  # [B, 1]

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    ones = torch.ones(B, 1, device=device, dtype=dtype)

    K = torch.stack([
        torch.cat([fx, zeros, cx], dim=1),
        torch.cat([zeros, fy, cy], dim=1),
        torch.cat([zeros, zeros, ones], dim=1),
    ], dim=1)  # [B, 3, 3]
    return K


def build_extrinsics(head_pose: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Build world-to-camera extrinsics matrix.

    Args:
        head_pose: [B, 3] axis-angle rotation
        translation: [B, 3] translation

    Returns:
        RT: [B, 4, 4] extrinsics matrix
    """
    B = head_pose.shape[0]
    device, dtype = head_pose.device, head_pose.dtype

    R = batch_rodrigues(head_pose)  # [B, 3, 3]
    t = translation.unsqueeze(-1)   # [B, 3, 1]

    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype) \
        .view(1, 1, 4).expand(B, -1, -1)

    RT = torch.cat([
        torch.cat([R, t], dim=2),  # [B, 3, 4]
        bottom,                     # [B, 1, 4]
    ], dim=1)  # [B, 4, 4]
    return RT


def project_points(points_3d: torch.Tensor, K: torch.Tensor,
                   RT: torch.Tensor) -> torch.Tensor:
    """Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: [B, N, 3] points in world space
        K: [B, 3, 3] intrinsics
        RT: [B, 4, 4] world-to-camera extrinsics

    Returns:
        points_2d: [B, N, 2] pixel coordinates
        depths: [B, N] z-depth in camera space
    """
    R = RT[:, :3, :3]  # [B, 3, 3]
    t = RT[:, :3, 3:]  # [B, 3, 1]

    # World → Camera
    points_cam = torch.bmm(R, points_3d.transpose(1, 2)) + t  # [B, 3, N]

    # Camera → Image
    points_proj = torch.bmm(K, points_cam)  # [B, 3, N]

    # Perspective divide
    depths = points_proj[:, 2, :]  # [B, N]
    points_2d = points_proj[:, :2, :] / (depths.unsqueeze(1) + 1e-8)  # [B, 2, N]

    # FLAME uses y-up; image coordinates are y-down. Flip y: v_image = 2*cy - v
    # This matches the flip(1) applied to nvdiffrast output in renderer.py.
    cy = K[:, 1, 2]  # [B]
    points_2d = points_2d.clone()
    points_2d[:, 1, :] = 2.0 * cy.unsqueeze(1) - points_2d[:, 1, :]

    return points_2d.transpose(1, 2), depths  # [B, N, 2], [B, N]


def project_to_clip(vertices_cam: torch.Tensor, K: torch.Tensor,
                    image_size: int, znear: float = 0.1, zfar: float = 10.0) -> torch.Tensor:
    """Project camera-space vertices to clip space for nvdiffrast.

    Ref: VHAP render_nvdiffrast.py L117-160 projection_from_intrinsics

    Args:
        vertices_cam: [B, V, 3] vertices in camera space
        K: [B, 3, 3] intrinsics
        image_size: int
        znear, zfar: near/far clip planes

    Returns:
        verts_clip: [B, V, 4] clip-space coordinates (x, y, z, w)
    """
    B, V = vertices_cam.shape[:2]
    device, dtype = vertices_cam.device, vertices_cam.dtype

    fx = K[:, 0, 0]  # [B]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    x = vertices_cam[:, :, 0]  # [B, V]
    y = vertices_cam[:, :, 1]
    z = vertices_cam[:, :, 2]

    w = image_size
    h = image_size

    # NDC x, y
    ndc_x = 2.0 * fx.unsqueeze(1) * x / (w * z + 1e-8) - 2.0 * cx.unsqueeze(1) / w + 1.0
    ndc_y = 2.0 * fy.unsqueeze(1) * y / (h * z + 1e-8) - 2.0 * cy.unsqueeze(1) / h + 1.0

    # NDC z (linear depth mapping)
    ndc_z = (zfar + znear) / (zfar - znear) - 2.0 * zfar * znear / ((zfar - znear) * z + 1e-8)

    clip = torch.stack([ndc_x, ndc_y, ndc_z, torch.ones_like(z)], dim=-1)  # [B, V, 4]
    return clip


def world_to_camera(vertices: torch.Tensor, RT: torch.Tensor) -> torch.Tensor:
    """Transform vertices from world to camera space.

    Args:
        vertices: [B, V, 3]
        RT: [B, 4, 4]

    Returns:
        vertices_cam: [B, V, 3]
    """
    R = RT[:, :3, :3]
    t = RT[:, :3, 3:]
    return torch.bmm(vertices, R.transpose(1, 2)) + t.transpose(1, 2)

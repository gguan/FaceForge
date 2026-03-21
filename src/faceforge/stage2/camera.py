"""Camera utilities for Stage 2 optimization.

Provides intrinsics/extrinsics construction, point projection, and
clip-space projection for nvdiffrast.

Reference: VHAP render_nvdiffrast.py L117-160 (clip projection),
           pixel3dmm tracker.py (screen-space projection)
"""

import torch
import torch.nn.functional as F

from .flame_wrapper import batch_rodrigues


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
                   RT: torch.Tensor, image_size: int = 512) -> torch.Tensor:
    """Project 3D points to 2D pixel coordinates (image y-down convention).

    Ref: pixel3dmm tracker.py project_points_screen_space

    Args:
        points_3d: [B, N, 3] points in world space
        K: [B, 3, 3] intrinsics
        RT: [B, 4, 4] world-to-camera extrinsics
        image_size: int, needed for y-flip

    Returns:
        points_2d: [B, N, 2] pixel coordinates (x right, y down)
        depths: [B, N] z-depth in camera space
    """
    R = RT[:, :3, :3]  # [B, 3, 3]
    t = RT[:, :3, 3:]  # [B, 3, 1]

    # World → Camera
    points_cam = torch.bmm(R, points_3d.transpose(1, 2)) + t  # [B, 3, N]

    # Camera → Image (standard pinhole: u = fx*x/z + cx, v = fy*y/z + cy)
    points_proj = torch.bmm(K, points_cam)  # [B, 3, N]

    # Perspective divide
    depths = points_proj[:, 2, :]  # [B, N]
    points_2d = points_proj[:, :2, :] / (depths.unsqueeze(1) + 1e-8)  # [B, 2, N]

    # FLAME uses y-up; image coordinates are y-down.
    # Flip y: v_image = (image_size - 1) - v_pinhole
    # This matches the flip(1) applied to nvdiffrast output in renderer.py.
    points_2d = points_2d.clone()
    points_2d[:, 1, :] = (image_size - 1) - points_2d[:, 1, :]

    return points_2d.transpose(1, 2), depths  # [B, N, 2], [B, N]


def _build_projection_matrix(K: torch.Tensor, image_size: int,
                              znear: float = 0.1, zfar: float = 10.0) -> torch.Tensor:
    """Build OpenGL projection matrix from intrinsics.

    Exact port of VHAP projection_from_intrinsics.
    Camera convention: x right, y up, z out-of-screen.
    Clip convention: x right, y up (before y-flip in rasterizer), z into screen, w = -z.

    Args:
        K: [B, 3, 3] intrinsics
        image_size: int
        znear, zfar: clip planes

    Returns:
        proj: [B, 4, 4] projection matrix
    """
    B = K.shape[0]
    w = h = float(image_size)

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    proj = torch.zeros(B, 4, 4, device=K.device, dtype=K.dtype)
    proj[:, 0, 0] = fx * 2 / w
    proj[:, 1, 1] = fy * 2 / h
    proj[:, 0, 2] = (w - 2 * cx) / w
    proj[:, 1, 2] = (h - 2 * cy) / h
    proj[:, 2, 2] = -(zfar + znear) / (zfar - znear)
    proj[:, 2, 3] = -2 * zfar * znear / (zfar - znear)
    proj[:, 3, 2] = -1.0
    return proj


def project_to_clip(vertices_cam: torch.Tensor, K: torch.Tensor,
                    image_size: int, znear: float = 0.1, zfar: float = 10.0) -> torch.Tensor:
    """Project camera-space vertices to clip space for nvdiffrast.

    Ref: VHAP render_nvdiffrast.py projection_from_intrinsics + camera_to_clip.
    Uses matrix multiplication to ensure exact same convention.

    Args:
        vertices_cam: [B, V, 3] vertices in camera space
        K: [B, 3, 3] intrinsics
        image_size: int
        znear, zfar: near/far clip planes

    Returns:
        verts_clip: [B, V, 4] clip-space coordinates (x, y, z, w)
    """
    proj = _build_projection_matrix(K, image_size, znear, zfar)  # [B, 4, 4]

    # Homogeneous coordinates: [B, V, 3] → [B, V, 4]
    ones = torch.ones(*vertices_cam.shape[:2], 1, device=vertices_cam.device, dtype=vertices_cam.dtype)
    verts_h = torch.cat([vertices_cam, ones], dim=-1)  # [B, V, 4]

    # clip = verts_h @ proj^T  (each row of verts multiplied by proj columns)
    verts_clip = torch.bmm(verts_h, proj.transpose(1, 2))  # [B, V, 4]
    return verts_clip


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

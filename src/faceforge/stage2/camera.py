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
    """Build intrinsics matrix K matching pixel3dmm get_intrinsics(use_hack=True).

    pixel3dmm formula (tracker.py L76-85):
        fx = fy = focal_length * size
        cx = size/2 + 0.5 + pp_x * (size/2 + 0.5)
        cy = size/2 + 0.5 + pp_y * (size/2 + 0.5)
        if use_hack: cx = size - cx   (flips cx for projection compatibility)

    Args:
        focal_length: [B, 1] normalized focal length
        principal_point: [B, 2] normalized principal point
        image_size: int, render resolution

    Returns:
        K: [B, 3, 3] intrinsics matrix
    """
    B = focal_length.shape[0]
    device, dtype = focal_length.device, focal_length.dtype
    s = float(image_size)

    fx = fy = focal_length * s  # [B, 1]
    # pixel3dmm: cx_raw = size/2 + 0.5 + pp * (size/2 + 0.5)
    half_plus = s / 2 + 0.5
    cx_raw = half_plus + principal_point[:, 0:1] * half_plus  # [B, 1]
    cy = half_plus + principal_point[:, 1:2] * half_plus      # [B, 1]
    # pixel3dmm use_hack: cx = size - cx_raw
    cx = s - cx_raw  # [B, 1]

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
    """Project 3D points to 2D pixel coords matching pixel3dmm exactly.

    pixel3dmm project_points_screen_space (tracker.py L97-114):
        v_cam = v_world @ w2c^T
        v' = v_cam[:3] / -v_cam[2]
        proj = (-1) * (K @ v')[:2]
        x_screen = size - 1 - proj[0]
        y_screen = proj[1]

    With RT=identity (our convention), v_cam = v_world.

    Args:
        points_3d: [B, N, 3] points in world space
        K: [B, 3, 3] intrinsics (with pixel3dmm hack applied)
        RT: [B, 4, 4] world-to-camera extrinsics
        image_size: int

    Returns:
        points_2d: [B, N, 2] pixel coordinates
        depths: [B, N] z-depth in camera space (negative for visible)
    """
    R = RT[:, :3, :3]  # [B, 3, 3]
    t = RT[:, :3, 3:]  # [B, 3, 1]

    # World → Camera
    points_cam = torch.bmm(R, points_3d.transpose(1, 2)) + t  # [B, 3, N]
    z_cam = points_cam[:, 2, :]  # [B, N], negative for visible points

    # pixel3dmm: v' = v[:3] / -z  (divide by -z, making denominator positive)
    neg_z = -z_cam  # [B, N], positive for visible
    v_prime = points_cam / (neg_z.unsqueeze(1) + 1e-8)  # [B, 3, N]

    # pixel3dmm: proj = (-1) * (K @ v')[:2]
    kv = torch.bmm(K, v_prime)  # [B, 3, N]
    proj = (-1.0) * kv[:, :2, :]  # [B, 2, N]

    # pixel3dmm: x_screen = size - 1 - proj[0],  y_screen = proj[1]
    points_2d = proj.clone()
    points_2d[:, 0, :] = (image_size - 1) - proj[:, 0, :]  # x flip
    # y stays as proj[1]

    return points_2d.transpose(1, 2), z_cam  # [B, N, 2], [B, N]


def build_intrinsics_standard(focal_length: torch.Tensor, image_size: int) -> torch.Tensor:
    """Build standard intrinsics (no hack) for renderer projection matrix.

    pixel3dmm uses standard K (cx=cy=size//2) for nvdiffrast projection,
    NOT the hacked K used for project_points_screen_space.
    Ref: pixel3dmm tracker.py L448-450
    """
    B = focal_length.shape[0]
    device, dtype = focal_length.device, focal_length.dtype
    s = float(image_size)
    fx = fy = focal_length * s
    cx = torch.full((B, 1), s / 2, device=device, dtype=dtype)
    cy = torch.full((B, 1), s / 2, device=device, dtype=dtype)
    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    ones = torch.ones(B, 1, device=device, dtype=dtype)
    K = torch.stack([
        torch.cat([fx, zeros, cx], dim=1),
        torch.cat([zeros, fy, cy], dim=1),
        torch.cat([zeros, zeros, ones], dim=1),
    ], dim=1)
    return K


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
    proj[:, 1, 1] = fy * 2 / h   # Positive (VHAP convention, renderer does y-flip)
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

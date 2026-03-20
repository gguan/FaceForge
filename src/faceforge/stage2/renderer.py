"""nvdiffrast differentiable renderer for Stage 2 optimization.

Reference: VHAP render_nvdiffrast.py (rasterization pipeline + SH lighting)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import world_to_camera, project_to_clip, project_points


# ---------------------------------------------------------------------------
# Spherical Harmonics (ref: VHAP render_nvdiffrast.py L19-53)
# ---------------------------------------------------------------------------

SH_C0 = 1.0 / math.sqrt(4.0 * math.pi)
SH_C1 = math.sqrt(3.0 / (4.0 * math.pi))
SH_C2_0 = 0.5 * math.sqrt(15.0 / math.pi)
SH_C2_1 = 0.5 * math.sqrt(15.0 / math.pi)
SH_C2_2 = 0.25 * math.sqrt(5.0 / math.pi)
SH_C2_3 = 0.5 * math.sqrt(15.0 / math.pi)
SH_C2_4 = 0.25 * math.sqrt(15.0 / math.pi)


def compute_sh_basis(normals: torch.Tensor) -> torch.Tensor:
    """Compute 9-term SH basis from normals.

    Args:
        normals: [..., 3] unit normals

    Returns:
        basis: [..., 9] SH basis values
    """
    nx, ny, nz = normals[..., 0:1], normals[..., 1:2], normals[..., 2:3]
    return torch.cat([
        torch.ones_like(nx) * SH_C0,         # Y_00
        ny * SH_C1,                           # Y_1-1
        nz * SH_C1,                           # Y_10
        nx * SH_C1,                           # Y_11
        nx * ny * SH_C2_0,                    # Y_2-2
        ny * nz * SH_C2_1,                    # Y_2-1
        (3.0 * nz * nz - 1.0) * SH_C2_2,     # Y_20
        nx * nz * SH_C2_3,                    # Y_21
        (nx * nx - ny * ny) * SH_C2_4,        # Y_22
    ], dim=-1)


def compute_sh_shading(normals: torch.Tensor, sh_coeffs: torch.Tensor) -> torch.Tensor:
    """Compute SH shading from normals and coefficients.

    Args:
        normals: [B, H, W, 3] or [B, V, 3]
        sh_coeffs: [B, 9, 3] SH coefficients per channel

    Returns:
        shading: same shape as normals[..., :3], per-element RGB shading
    """
    orig_shape = normals.shape   # [B, ..., 3]
    B = orig_shape[0]
    basis = compute_sh_basis(normals)        # [B, ..., 9]
    basis_flat = basis.reshape(B, -1, 9)    # [B, N, 9]
    shading_flat = torch.bmm(basis_flat, sh_coeffs)  # [B, N, 3]
    shading = shading_flat.reshape(*orig_shape[:-1], 3)  # [B, ..., 3]
    return torch.clamp(shading, min=0.0)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class NvdiffrastRenderer(nn.Module):
    """Differentiable renderer using nvdiffrast."""

    def __init__(self, image_size: int = 512, use_opengl: bool = False):
        super().__init__()
        self.image_size = image_size

        try:
            import nvdiffrast.torch as dr
            self.dr = dr
            if use_opengl:
                self.glctx = dr.RasterizeGLContext()
            else:
                self.glctx = dr.RasterizeCudaContext()
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def render(self, vertices: torch.Tensor, faces: torch.Tensor,
               K: torch.Tensor, RT: torch.Tensor,
               vertex_normals: torch.Tensor,
               vertex_colors: torch.Tensor | None = None,
               sh_coefficients: torch.Tensor | None = None) -> dict:
        """Full differentiable render pipeline.

        Args:
            vertices: [B, V, 3] world-space vertices
            faces: [F, 3] int64 face indices
            K: [B, 3, 3] intrinsics
            RT: [B, 4, 4] world-to-camera extrinsics
            vertex_normals: [B, V, 3] per-vertex normals
            vertex_colors: [B, V, 3] optional vertex colors [0,1]
            sh_coefficients: [B, 9, 3] optional SH lighting

        Returns:
            dict with 'image' [B,H,W,3], 'mask' [B,H,W,1],
            'normal' [B,H,W,3], 'depth' [B,H,W,1]
        """
        assert self._available, "nvdiffrast not installed"
        dr = self.dr
        B = vertices.shape[0]
        H = W = self.image_size

        # 1. World → Camera
        verts_cam = world_to_camera(vertices, RT)

        # 2. Camera → Clip
        verts_clip = project_to_clip(verts_cam, K, self.image_size)

        # 3. Rasterize
        faces_int = faces.int()
        rast_out, rast_db = dr.rasterize(self.glctx, verts_clip, faces_int, (H, W))

        # 4. Interpolate normals
        normals_cam = torch.bmm(vertex_normals, RT[:, :3, :3].transpose(1, 2))
        normals_interp, _ = dr.interpolate(normals_cam.contiguous(), rast_out, faces_int)
        normals_interp = F.normalize(normals_interp, dim=-1)

        # 5. Interpolate positions (for depth)
        pos_interp, _ = dr.interpolate(verts_cam.contiguous(), rast_out, faces_int)
        depth = pos_interp[..., 2:3]

        # 6. Compute image color
        if vertex_colors is not None and sh_coefficients is not None:
            colors_interp, _ = dr.interpolate(vertex_colors.contiguous(), rast_out, faces_int)
            shading = compute_sh_shading(normals_interp, sh_coefficients)
            image = colors_interp * shading
        elif sh_coefficients is not None:
            # Gray mesh with SH lighting
            gray = torch.full_like(vertices, 0.7)
            colors_interp, _ = dr.interpolate(gray.contiguous(), rast_out, faces_int)
            shading = compute_sh_shading(normals_interp, sh_coefficients)
            image = colors_interp * shading
        else:
            image = normals_interp * 0.5 + 0.5  # normal-as-color fallback

        # 7. Mask
        mask = (rast_out[..., 3:] > 0).float()

        # 8. Antialias
        image = dr.antialias(image * mask, rast_out, verts_clip, faces_int)
        mask = dr.antialias(mask, rast_out, verts_clip, faces_int)

        # 9. Y-flip (OpenGL y-up → image y-down)
        image = image.flip(1)
        mask = mask.flip(1)
        normals_interp = normals_interp.flip(1)
        depth = depth.flip(1)

        return {
            'image': image,           # [B, H, W, 3]
            'mask': mask,             # [B, H, W, 1]
            'normal': normals_interp, # [B, H, W, 3]
            'depth': depth,           # [B, H, W, 1]
        }

    def compute_visibility_mask(self, depth_buffer: torch.Tensor,
                                projected_verts_2d: torch.Tensor,
                                proj_depths: torch.Tensor,
                                image_size: int,
                                eps: float = 0.01) -> torch.Tensor:
        """Occlusion filtering using depth buffer.

        Ref: pixel3dmm tracker.py occlusion filtering logic

        Args:
            depth_buffer: [B, H, W, 1] from render output
            projected_verts_2d: [B, V, 2] pixel coordinates
            proj_depths: [B, V] z-depths of projected vertices
            image_size: int
            eps: depth tolerance

        Returns:
            visibility: [B, V] bool, True = visible
        """
        B, V = projected_verts_2d.shape[:2]

        # Normalize coords to [-1, 1] for grid_sample
        grid = projected_verts_2d.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / image_size - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / image_size - 1.0
        grid = grid.unsqueeze(2)  # [B, V, 1, 2]

        # Sample depth at projected positions
        depth_map = depth_buffer.permute(0, 3, 1, 2)  # [B, 1, H, W]
        sampled_depth = F.grid_sample(
            depth_map, grid, mode='bilinear', align_corners=True,
        ).squeeze(1).squeeze(-1)  # [B, V]

        # Visible if sampled depth ≈ vertex depth
        visibility = sampled_depth < (proj_depths + eps)

        # Also filter vertices outside image bounds
        in_bounds = (
            (projected_verts_2d[..., 0] >= 0) &
            (projected_verts_2d[..., 0] < image_size) &
            (projected_verts_2d[..., 1] >= 0) &
            (projected_verts_2d[..., 1] < image_size)
        )
        return visibility & in_bounds

"""
nvdiffrast wrapper + coarse-to-fine texture baker (vendored from
modelscope, Apache 2.0). Trimmed to the methods HRN-head uses:
``MeshRenderer.__call__`` (silhouette + depth + features + occupancy),
``render_uv_texture`` (forward render with a baked texture), and
``pred_shape_and_texture`` (the 64→2048 coarse-to-fine texture solve).
"""

import warnings
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .losses import TVLoss

warnings.filterwarnings('ignore')


def _import_nvdiffrast():
    """Lazy-import nvdiffrast so the rest of the package stays importable
    on machines without it (nvdiffrast is GPU-only and ships from source)."""
    try:
        import nvdiffrast as _nvd
        import nvdiffrast.torch as _dr
    except ImportError as e:
        raise ImportError(
            "nvdiffrast is required for HRN-head reconstruction. "
            "Install with: pip install git+https://github.com/NVlabs/nvdiffrast.git"
        ) from e
    return _nvd, _dr


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0], [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


def to_image(face_shape):
    focal = 1015.
    center = 112.
    persc_proj = np.array([focal, 0, center, 0, focal, center, 0, 0,
                           1]).reshape([3, 3]).astype(np.float32).transpose()
    persc_proj = torch.tensor(persc_proj).to(face_shape.device)
    face_proj = face_shape @ persc_proj
    face_proj = face_proj[..., :2] / face_proj[..., 2:]
    return face_proj


class MeshRenderer(nn.Module):
    def __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224):
        super().__init__()
        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)) \
            .matmul(torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None

    def _ensure_ctx(self, device):
        if self.glctx is None:
            nvd, dr = _import_nvdiffrast()
            if nvd.__version__ == '0.2.7':
                self.glctx = dr.RasterizeGLContext(device=device)
            else:
                self.glctx = dr.RasterizeCudaContext(device=device)

    def forward(self, vertex, tri, feat=None):
        _, dr = _import_nvdiffrast()
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        verts_proj = to_image(vertex)

        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        self._ensure_ctx(device)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri,
                                   resolution=[rsize, rsize], ranges=ranges)
        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        verts_x = verts_proj[0, :, 0]
        verts_y = 224 - verts_proj[0, :, 1]
        verts_int = torch.ceil(verts_proj[0]).long()
        verts_xr_int = verts_int[:, 0].clamp(1, 224 - 1)
        verts_yt_int = 224 - verts_int[:, 1].clamp(2, 224)
        verts_right_float = verts_xr_int - verts_x
        verts_left_float = 1 - verts_right_float
        verts_top_float = verts_y - verts_yt_int
        verts_bottom_float = 1 - verts_top_float

        rast_lt = rast_out[0, verts_yt_int, verts_xr_int - 1, 3]
        rast_lb = rast_out[0, verts_yt_int + 1, verts_xr_int - 1, 3]
        rast_rt = rast_out[0, verts_yt_int, verts_xr_int, 3]
        rast_rb = rast_out[0, verts_yt_int + 1, verts_xr_int, 3]

        occ_feat = (rast_lt > 0) * 1.0 * (verts_left_float + verts_top_float) + \
                   (rast_lb > 0) * 1.0 * (verts_left_float + verts_bottom_float) + \
                   (rast_rt > 0) * 1.0 * (verts_right_float + verts_top_float) + \
                   (rast_rb > 0) * 1.0 * (verts_right_float + verts_bottom_float)
        occ_feat = occ_feat[None, :, None] / 4.0

        occ, _ = dr.interpolate(occ_feat, rast_out, tri)
        occ = occ.permute(0, 3, 1, 2)

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image, occ

    def render_uv_texture(self, vertex, tri, uv, uv_texture):
        _, dr = _import_nvdiffrast()
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)

        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        self._ensure_ctx(device)

        tri = tri.type(torch.int32).contiguous()
        rast_out, rast_db = dr.rasterize(
            self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize])
        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        uv = uv.clone()
        uv[..., -1] = 1.0 - uv[..., -1]
        interp_out, _ = dr.interpolate(uv, rast_out, tri, rast_db, diff_attrs='all')
        uv_texture = uv_texture.permute(0, 2, 3, 1).contiguous()
        img = dr.texture(uv_texture, interp_out, filter_mode='linear')
        img = img * torch.clamp(rast_out[..., -1:], 0, 1)
        image = img.permute(0, 3, 1, 2)
        return mask, depth, image

    def pred_shape_and_texture(self, vertex, tri, uv, target_img, base_tex=None):
        """Coarse-to-fine differentiable texture solve.

        Optimises a texture map at progressively higher resolutions
        (64 → 2048) so that rendering it through ``vertex/tri/uv`` matches
        ``target_img``. Returns the BGR-ordered texture in [0, 255].
        """
        _, dr = _import_nvdiffrast()
        uv = uv.clone()
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)

        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        self._ensure_ctx(device)

        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize])
        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        uv[..., -1] = 1.0 - uv[..., -1]

        rast_out, rast_db = dr.rasterize(
            self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize])
        interp_out, _ = dr.interpolate(uv, rast_out, tri, rast_db, diff_attrs='all')

        mask_3c = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        maskout_img = mask_3c * target_img
        mean_color = torch.sum(maskout_img, dim=(1, 2)) / torch.max(
            torch.sum(mask), torch.tensor(1.0).to(mask.device))

        tex = torch.zeros((1, 128, 128, 3), dtype=torch.float32, device=device)
        tex[:, :, :, 0] = mean_color[0, 0]
        tex[:, :, :, 1] = mean_color[0, 1]
        tex[:, :, :, 2] = mean_color[0, 2]

        tex_mask = torch.zeros((1, 2048, 2048, 3), dtype=torch.float32, device=device)
        tex_mask[:, :, :, 1] = 1.0
        tex_mask.requires_grad_(True)

        criterion_tv = TVLoss()

        if base_tex is not None:
            base_tex = base_tex.to(device)

        for tex_resolution in [64, 128, 256, 512, 1024, 2048]:
            tex = tex.detach().permute(0, 3, 1, 2)
            tex = F.interpolate(tex, (tex_resolution, tex_resolution))
            tex = tex.permute(0, 2, 3, 1).contiguous()

            if base_tex is not None:
                _bt = base_tex.permute(0, 3, 1, 2)
                _bt = F.interpolate(_bt, (tex_resolution, tex_resolution))
                tex += _bt.permute(0, 2, 3, 1).contiguous()

            tex.requires_grad_(True)
            optim = torch.optim.Adam([tex], lr=1e-2)
            n_iters = 200

            if tex_resolution == 2048:
                optim_mask = torch.optim.Adam([tex_mask], lr=1e-2)

            for _ in range(n_iters):
                if tex_resolution == 2048:
                    optim_mask.zero_grad()
                    rendered = dr.texture(tex_mask, interp_out, filter_mode='linear')
                    rendered = rendered * torch.clamp(rast_out[..., -1:], 0, 1)
                    tex_loss = torch.mean((target_img - rendered) ** 2)
                    tex_loss.backward()
                    optim_mask.step()

                optim.zero_grad()
                img = dr.texture(tex, interp_out, filter_mode='linear')
                img = img * torch.clamp(rast_out[..., -1:], 0, 1)
                recon_loss = torch.mean((target_img - img) ** 2)
                if tex_resolution < 2048:
                    tv_loss = criterion_tv(tex.permute(0, 3, 1, 2))
                    total = recon_loss + tv_loss * 0.01
                else:
                    total = recon_loss
                total.backward()
                optim.step()

        tex_map = tex[0].detach().cpu().numpy()[..., ::-1] * 255.0  # BGR
        image = img.permute(0, 3, 1, 2)

        tex_mask_np = tex_mask[0].detach().cpu().numpy() * 255.0
        tex_mask_out = (np.where(tex_mask_np[..., 1] > 250, 1.0, 0.0)
                        * np.where(tex_mask_np[..., 0] < 10, 1.0, 0)
                        * np.where(tex_mask_np[..., 2] < 10, 1.0, 0))
        tex_mask_out = 1.0 - tex_mask_out

        return mask, depth, image, tex_map, tex_mask_out

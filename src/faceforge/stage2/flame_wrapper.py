"""Thin wrapper around pixel3dmm's FLAME model for Stage 2 optimization.

Provides the same ``forward(shape, expression, head_pose, jaw_pose)``
interface as the old ``FLAMEModel`` while delegating all heavy lifting
(LBS, landmarks, kinematic chain) to ``pixel3dmm.tracking.flame.FLAME``.

Extra data that pixel3dmm's FLAME does not carry (UV coords, region
masks, vertex normals) is loaded here.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceforge._paths import PROJECT_ROOT
from .config import Stage2Config


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _aa_to_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle [B, 3] → 6D rotation [B, 6]."""
    from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
    R = _batch_rodrigues(axis_angle)             # [B, 3, 3]
    return matrix_to_rotation_6d(R)              # [B, 6]


def _batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Axis-angle [N, 3] → rotation matrices [N, 3, 3]."""
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


# Re-export so camera.py / pipeline.py can still do
#   ``from .flame_wrapper import batch_rodrigues``
batch_rodrigues = _batch_rodrigues


# ---------------------------------------------------------------------------
# Eyeball vertex indices (from flame-head-tracker)
# ---------------------------------------------------------------------------
R_EYE_INDICES = [4597, 4543, 4511, 4479, 4575]
L_EYE_INDICES = [4051, 3997, 3965, 3933, 4020]


# ---------------------------------------------------------------------------
# pixel3dmm FLAME config adapter
# ---------------------------------------------------------------------------

class _P3DMMConfig:
    """Minimal config object expected by ``pixel3dmm.tracking.flame.FLAME``."""

    def __init__(self):
        self.use_flame2023 = False
        self.num_shape_params = 300
        self.num_exp_params = 100


# ---------------------------------------------------------------------------
# FLAMEWrapper
# ---------------------------------------------------------------------------

class FLAMEWrapper(nn.Module):
    """Drop-in replacement for the old ``FLAMEModel``.

    Delegates LBS / landmark computation to pixel3dmm's FLAME and adds
    UV data, region masks, and vertex-normal computation on top.
    """

    def __init__(self, config: Stage2Config):
        super().__init__()

        # --- pixel3dmm FLAME (LBS + landmarks) ---
        from pixel3dmm.tracking.flame.FLAME import FLAME as P3DFLAME
        self._flame = P3DFLAME(_P3DMMConfig())

        # Expose faces from the inner model so the rest of the code
        # can access ``self.flame.faces`` as before.
        # (faces is a registered buffer on _flame, so it moves with .to())

        # --- Extra data not carried by pixel3dmm FLAME ---
        self._load_region_masks(config.flame_masks_path)
        self._load_uv_data(config.flame_uv_coords_path,
                           config.flame_uv_valid_verts_path)

    # ------------------------------------------------------------------
    # Properties that mirror the old FLAMEModel interface
    # ------------------------------------------------------------------

    @property
    def faces(self) -> torch.Tensor:
        return self._flame.faces

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, shape_params, expression_params, R_6d, jaw_6d,
                eyes_6d=None, neck_6d=None, eyelids=None):
        """Differentiable FLAME forward pass — all params in 6D rotation format.

        Matches pixel3dmm tracker.py exactly:
          1. FLAME outputs canonical vertices (NO rot_params)
          2. Contour landmark selection uses rot_params_lmk_shift=inv(R_head)
          3. Caller applies external rotation: R @ v_can + t

        Args:
            shape_params:      [B, 300] shape coefficients
            expression_params: [B, 100] expression coefficients
            R_6d:              [B, 6]   head rotation (6D format)
            jaw_6d:            [B, 6]   jaw rotation (6D format)
            eyes_6d:           [B, 12]  optional 6D eye pose (left 6 + right 6)
            neck_6d:           [B, 6]   optional 6D neck pose
            eyelids:           [B, 2]   optional eyelid params

        Returns:
            vertices_can:   [B, 5023, 3]  canonical (unrotated) vertices
            landmarks_68:   [B, 68, 3]    canonical landmarks
            landmarks_eyes: [B, 10, 3]    canonical eye landmarks
            R_head:         [B, 3, 3]     head rotation matrix
        """
        from pixel3dmm.utils.utils_3d import (
            rotation_6d_to_matrix, matrix_to_rotation_6d)

        B = R_6d.shape[0]
        device, dtype = R_6d.device, R_6d.dtype

        # 6D → rotation matrix for external application
        R_head = rotation_6d_to_matrix(R_6d)                    # [B, 3, 3]
        # Contour landmark selection: inv(R_head)
        R_head_inv = R_head.transpose(-1, -2)                   # [B, 3, 3]
        rot_lmk_shift_6d = matrix_to_rotation_6d(R_head_inv)   # [B, 6]
        cameras_I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

        # Default identity for optional params
        I6d = matrix_to_rotation_6d(torch.eye(3, device=device, dtype=dtype).unsqueeze(0))
        if eyes_6d is None:
            eyes_6d = I6d.repeat(1, 2)  # [B, 12]
        if neck_6d is None:
            neck_6d = I6d.expand(B, -1)  # [B, 6]
        if eyelids is None:
            eyelids = torch.zeros(B, 2, device=device, dtype=dtype)

        # pixel3dmm FLAME: no rot_params → canonical output
        # Ref: pixel3dmm tracker.py L624-631, L846-856
        vertices_can, lmk68, _jt, _vcan_can, _vnoneck = self._flame(
            shape_params=shape_params,
            cameras=cameras_I,
            jaw_pose_params=jaw_6d,
            expression_params=expression_params,
            eye_pose_params=eyes_6d,
            neck_pose_params=neck_6d,
            eyelid_params=eyelids,
            rot_params_lmk_shift=rot_lmk_shift_6d,
        )

        landmarks_eyes = vertices_can[:, R_EYE_INDICES + L_EYE_INDICES, :]
        return vertices_can, lmk68, landmarks_eyes, R_head

    # ------------------------------------------------------------------
    # Vertex normals (not in pixel3dmm FLAME)
    # ------------------------------------------------------------------

    def get_vertex_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals.

        Args:
            vertices: [B, 5023, 3]

        Returns:
            normals: [B, 5023, 3] unit normals
        """
        batch_size = vertices.shape[0]
        faces = self.faces.long()

        v0 = vertices[:, faces[:, 0]]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, i].unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3),
                face_normals,
            )
        return F.normalize(vertex_normals, dim=-1)

    # ------------------------------------------------------------------
    # Region masks
    # ------------------------------------------------------------------

    def _load_region_masks(self, path: str):
        resolved = _resolve(path)
        with open(resolved, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        self.region_masks = {}
        for key, val in masks.items():
            indices = np.array(val, dtype=np.int64).flatten()
            self.register_buffer(f'mask_{key}', torch.from_numpy(indices))
            self.region_masks[key] = f'mask_{key}'

    def get_region_indices(self, region_name: str) -> torch.Tensor:
        buf_name = self.region_masks.get(region_name)
        if buf_name is None:
            raise KeyError(
                f"Unknown region: {region_name}. "
                f"Available: {list(self.region_masks.keys())}"
            )
        return getattr(self, buf_name)

    # ------------------------------------------------------------------
    # UV data (for UV loss)
    # ------------------------------------------------------------------

    def _load_uv_data(self, uv_path: str, valid_verts_path: str):
        uv_resolved = _resolve(uv_path)
        uv_coords = np.load(uv_resolved).astype(np.float32)  # [5023, 2]
        # Flip V only (ref: pixel3dmm losses.py L61)
        uv_coords[:, 1] = (-uv_coords[:, 1]) + 1.0
        self.register_buffer('flame_uv_coords', torch.from_numpy(uv_coords))

        valid_resolved = _resolve(valid_verts_path)
        valid_verts = np.load(valid_resolved)
        if valid_verts.dtype == bool:
            valid_verts = np.where(valid_verts)[0]
        self.register_buffer('uv_valid_verts',
                             torch.from_numpy(valid_verts.astype(np.int64)))

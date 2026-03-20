"""Standalone differentiable FLAME model for Stage 2 optimization.

Loads FLAME directly from generic_model.pkl without MICA dependency.
All operations are differentiable for gradient-based optimization.

Reference: pixel3dmm tracking/flame/FLAME.py + lbs.py (algorithm reimplemented here)
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceforge._paths import PROJECT_ROOT
from .config import Stage2Config


def _resolve_path(path: str) -> Path:
    """Resolve a relative path against PROJECT_ROOT."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# LBS helpers (from pixel3dmm tracking/flame/lbs.py, reimplemented)
# ---------------------------------------------------------------------------

def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Axis-angle [N, 3] → rotation matrices [N, 3, 3]. Differentiable."""
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
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def _blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """Bx(NB) @ VxDx(NB) → BxVxD"""
    return torch.einsum('bl,mkl->bmk', betas, shape_disps)


def _vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """JxV @ BxVx3 → BxJx3"""
    return torch.einsum('bik,ji->bjk', vertices, J_regressor)


def _transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Bx3x3, Bx3x1 → Bx4x4"""
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def _batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """Apply rigid transforms along kinematic chain.

    Returns posed_joints [B,J,3] and rel_transforms [B,J,4,4].
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = _transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1),
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def lbs(betas, pose_rot_mats, v_template, shapedirs, posedirs,
        J_regressor, parents, lbs_weights, dtype=torch.float32):
    """Linear Blend Skinning. Expects rotation matrices (not axis-angle).

    Args:
        betas: [B, N_shape + N_exp]
        pose_rot_mats: [B, J, 3, 3] rotation matrices
        v_template: [B, V, 3]
        shapedirs: [V, 3, N_shape + N_exp]
        posedirs: [P, V*3]
        J_regressor: [J, V]
        parents: [J]
        lbs_weights: [V, J]

    Returns:
        vertices [B, V, 3], joint_transforms [B, J, 4, 4], v_shaped [B, V, 3]
    """
    batch_size = betas.shape[0]
    device = betas.device

    v_shaped = v_template + _blend_shapes(betas, shapedirs)
    J = _vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (pose_rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    J_transformed, A = _batch_rigid_transform(pose_rot_mats, J, parents, dtype=dtype)

    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts, A, v_shaped


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_tensor(array, dtype=torch.float32):
    if isinstance(array, torch.Tensor):
        return array.to(dtype)
    arr_np = np.array(array) if 'scipy.sparse' not in str(type(array)) else np.array(array.todense())
    return torch.tensor(arr_np.astype(np.float64 if dtype == torch.float64 else np.float32), dtype=dtype)


def _rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] ** 2 + rot_mats[:, 1, 0] ** 2)
    return torch.atan2(-rot_mats[:, 2, 0], sy)


# ---------------------------------------------------------------------------
# FLAME Model
# ---------------------------------------------------------------------------

class FLAMEModel(nn.Module):
    """Standalone differentiable FLAME model.

    Uses axis-angle [B, 3] for head_pose and jaw_pose (matching Stage 1 output).
    Internally converts to rotation matrices for LBS.
    """

    # Eyeball vertex indices (from flame-head-tracker)
    R_EYE_INDICES = [4597, 4543, 4511, 4479, 4575]
    L_EYE_INDICES = [4051, 3997, 3965, 3933, 4020]

    def __init__(self, config: Stage2Config):
        super().__init__()
        self.dtype = torch.float32
        self.n_shape = 300
        self.n_exp = 100

        self._load_flame_model(config.flame_model_path)
        self._load_landmark_embedding(config.flame_lmk_embedding_path)
        self._load_region_masks(config.flame_masks_path)
        self._load_uv_data(config.flame_uv_coords_path, config.flame_uv_valid_verts_path)

    def _load_flame_model(self, path: str):
        """Load FLAME from generic_model.pkl. Ref: pixel3dmm FLAME.py L75-97"""
        resolved = _resolve_path(path)
        with open(resolved, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        self.register_buffer('faces', _to_tensor(model['f'], dtype=torch.long))
        self.register_buffer('v_template', _to_tensor(model['v_template']))

        # shapedirs: [V, 3, 400] — first 300 for shape, last 100 for expression
        shapedirs = _to_tensor(model['shapedirs'])
        shapedirs = torch.cat([
            shapedirs[:, :, :self.n_shape],
            shapedirs[:, :, 300:300 + self.n_exp],
        ], dim=2)
        self.register_buffer('shapedirs', shapedirs)

        # posedirs: reshape [V*3, P] → [P, V*3] (transposed for matmul)
        num_pose_basis = model['posedirs'].shape[-1]
        posedirs = np.reshape(np.array(model['posedirs']), [-1, num_pose_basis]).T
        self.register_buffer('posedirs', _to_tensor(posedirs))

        self.register_buffer('J_regressor', _to_tensor(model['J_regressor']))

        parents = _to_tensor(model['kintree_table'][0], dtype=torch.long)
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights', _to_tensor(model['weights']))

    def _load_landmark_embedding(self, path: str):
        """Load landmark embedding. Ref: pixel3dmm FLAME.py L110-117"""
        resolved = _resolve_path(path)
        lmk = np.load(resolved, allow_pickle=True, encoding='latin1')[()]

        # Static landmarks: 51 points (eyebrows, eyes, nose, mouth)
        self.register_buffer('lmk_faces_idx',
                             torch.from_numpy(lmk['static_lmk_faces_idx'].astype(np.int64)))
        self.register_buffer('lmk_bary_coords',
                             torch.from_numpy(lmk['static_lmk_bary_coords']).float())

        # Dynamic landmarks: 17 contour points, pose-dependent (79 yaw bins)
        self.register_buffer('dynamic_lmk_faces_idx',
                             torch.from_numpy(np.array(lmk['dynamic_lmk_faces_idx']).astype(np.int64)))
        self.register_buffer('dynamic_lmk_bary_coords',
                             torch.from_numpy(np.array(lmk['dynamic_lmk_bary_coords'])).float())

        # Neck kinematic chain for dynamic landmark yaw computation
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _load_region_masks(self, path: str):
        """Load FLAME_masks.pkl. Keys: face, nose, left_eye_region, etc."""
        resolved = _resolve_path(path)
        with open(resolved, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        self.region_masks = {}
        for key, val in masks.items():
            indices = np.array(val, dtype=np.int64).flatten()
            self.register_buffer(f'mask_{key}', torch.from_numpy(indices))
            self.region_masks[key] = f'mask_{key}'

    def _load_uv_data(self, uv_path: str, valid_verts_path: str):
        """Load FLAME UV coords + valid vertex mask for UV loss."""
        uv_resolved = _resolve_path(uv_path)
        uv_coords = np.load(uv_resolved).astype(np.float32)  # [5023, 2]
        # Only flip V axis (ref: pixel3dmm losses.py L61: can_uv[...,1] = (-can_uv[...,1]) + 1)
        # U is NOT flipped here. The renderer flips U internally via uv_images=[1-texc_u, texc_v]
        # which nets out to U unchanged, so can_uv and the network output both use raw-U convention.
        uv_coords[:, 1] = (-uv_coords[:, 1]) + 1.0  # flip V only
        self.register_buffer('flame_uv_coords', torch.from_numpy(uv_coords))

        valid_resolved = _resolve_path(valid_verts_path)
        valid_verts = np.load(valid_resolved)  # boolean or index array
        if valid_verts.dtype == bool:
            valid_verts = np.where(valid_verts)[0]
        self.register_buffer('uv_valid_verts', torch.from_numpy(valid_verts.astype(np.int64)))

    def get_region_indices(self, region_name: str) -> torch.Tensor:
        """Get vertex indices for a named region."""
        buf_name = self.region_masks.get(region_name)
        if buf_name is None:
            raise KeyError(f"Unknown region: {region_name}. Available: {list(self.region_masks.keys())}")
        return getattr(self, buf_name)

    def forward(self, shape_params, expression_params, head_pose, jaw_pose):
        """Differentiable FLAME forward pass.

        Args:
            shape_params: [B, 300] shape coefficients
            expression_params: [B, 100] expression coefficients
            head_pose: [B, 3] axis-angle global rotation
            jaw_pose: [B, 3] axis-angle jaw rotation

        Returns:
            vertices: [B, 5023, 3]
            landmarks_68: [B, 68, 3]
            landmarks_eyes: [B, 10, 3]
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device

        betas = torch.cat([shape_params, expression_params], dim=1)

        # Build rotation matrices from axis-angle: [head, neck(zero), jaw, eye_L(zero), eye_R(zero)]
        zeros_3 = torch.zeros(batch_size, 3, device=device)
        full_pose_aa = torch.cat([head_pose, zeros_3, jaw_pose, zeros_3, zeros_3], dim=1)  # [B, 15]
        rot_mats = batch_rodrigues(full_pose_aa.reshape(-1, 3)).reshape(batch_size, 5, 3, 3)

        v_template = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, joint_transforms, v_shaped = lbs(
            betas, rot_mats, v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype,
        )

        landmarks_68 = self._get_landmarks_68(vertices, rot_mats, batch_size)
        landmarks_eyes = self._get_landmarks_eyes(vertices)

        return vertices, landmarks_68, landmarks_eyes

    def _get_landmarks_68(self, vertices, rot_mats, batch_size):
        """Extract 68 landmarks: 17 dynamic contour + 51 static.

        Ref: pixel3dmm FLAME.py L280-288
        """
        # Dynamic contour landmarks (pose-dependent)
        dyn_faces_idx, dyn_bary_coords = self._find_dynamic_lmk(vertices, rot_mats, batch_size)

        # Static landmarks
        static_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)
        static_bary_coords = self.lmk_bary_coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine: [dynamic_17, static_51] = 68 total
        lmk_faces_idx = torch.cat([dyn_faces_idx, static_faces_idx], dim=1)
        lmk_bary_coords = torch.cat([dyn_bary_coords, static_bary_coords], dim=1)

        return self._vertices2landmarks(vertices, self.faces, lmk_faces_idx, lmk_bary_coords)

    def _get_landmarks_eyes(self, vertices):
        """Extract 10 eyeball landmarks from FLAME vertices."""
        eye_indices = self.R_EYE_INDICES + self.L_EYE_INDICES
        return vertices[:, eye_indices, :]

    def _find_dynamic_lmk(self, vertices, rot_mats, batch_size):
        """Select contour landmarks based on head yaw.

        Ref: pixel3dmm FLAME.py L127-165 (simplified for axis-angle input)
        """
        device = vertices.device

        # Extract neck rotation from kinematic chain
        neck_rot_mats = rot_mats[:, self.neck_kin_chain]
        rel_rot_mat = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        for idx in range(len(self.neck_kin_chain)):
            rel_rot_mat = torch.bmm(neck_rot_mats[:, idx], rel_rot_mat)

        # Compute yaw angle and quantize to bin index [0, 78]
        y_rot_angle = torch.round(
            torch.clamp(-_rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).long()
        mask = y_rot_angle.lt(-39).long()
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_faces_idx = torch.index_select(self.dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_bary_coords = torch.index_select(self.dynamic_lmk_bary_coords, 0, y_rot_angle)
        return dyn_faces_idx, dyn_bary_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """Barycentric interpolation of landmarks on mesh faces.

        Ref: pixel3dmm FLAME.py L167-195
        """
        batch_size, num_verts = vertices.shape[:2]
        device = vertices.device

        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1).long()) \
            .view(batch_size, -1, 3)
        lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device) \
            .view(-1, 1, 1) * num_verts
        lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
        landmarks = torch.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)
        return landmarks

    def get_vertex_normals(self, vertices):
        """Compute per-vertex normals from vertices and faces.

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

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [B, F, 3]

        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, i].unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3),
                face_normals,
            )
        return F.normalize(vertex_normals, dim=-1)

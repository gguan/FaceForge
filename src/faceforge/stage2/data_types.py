from dataclasses import dataclass, field

import torch


@dataclass
class PreprocessedData:
    """每图一份, Pixel3DMM 一次性预处理结果"""
    pixel3dmm_uv: torch.Tensor        # [1, 2, 512, 512] UV prediction [0,1]
    pixel3dmm_normals: torch.Tensor    # [1, 3, 512, 512] normals [-1,1] with coord swap
    face_mask: torch.Tensor            # [1, 512, 512] binary
    face_segmentation: torch.Tensor    # [1, 512, 512] int, 19-class BiSeNet
    target_image: torch.Tensor         # [1, 3, 512, 512] aligned image
    target_lmks_68: torch.Tensor       # [1, 68, 2] pixel coords
    target_lmks_eyes: torch.Tensor     # [1, 10, 2] pixel coords (MediaPipe iris)
    lmk_mask: torch.Tensor = None      # [1, 68, 1] landmark validity mask
    target_iris_left: torch.Tensor = None   # [1, 1, 2] left iris center
    target_iris_right: torch.Tensor = None  # [1, 1, 2] right iris center
    iris_mask_left: torch.Tensor = None     # [1, 1, 1] iris detection mask
    iris_mask_right: torch.Tensor = None    # [1, 1, 1] iris detection mask
    arcface_feat: torch.Tensor = None       # [1, 512] L2-normalized


@dataclass
class SharedParams:
    """所有图共享的优化参数 (身份)"""
    shape: torch.Tensor          # [1, 300]
    texture: torch.Tensor        # [1, 50]
    focal_length: torch.Tensor   # [1, 1]


@dataclass
class PerImageParams:
    """每图独立的优化参数 (状态) — all rotations in 6D format."""
    expression: torch.Tensor      # [1, 100]
    R_6d: torch.Tensor            # [1, 6] head rotation (6D, pixel3dmm convention)
    jaw_6d: torch.Tensor          # [1, 6] jaw rotation (6D)
    translation: torch.Tensor     # [1, 3]
    lighting: torch.Tensor        # [1, 9, 3] SH coefficients
    principal_point: torch.Tensor = None  # [1, 2]
    eyes_6d: torch.Tensor = None  # [1, 12] eye pose (6D left + 6D right)
    neck_6d: torch.Tensor = None  # [1, 6] neck pose (6D)
    eyelids: torch.Tensor = None  # [1, 2] eyelid params


@dataclass
class Stage2Output:
    """Stage 2 优化结果, 传给 Stage 3"""
    shape: torch.Tensor           # [1, 300] 联合优化后
    texture: torch.Tensor         # [1, 50]
    focal_length: torch.Tensor    # [1, 1]
    vertices: torch.Tensor        # [1, 5023, 3] 中性表情 mesh
    landmarks_3d: torch.Tensor    # [1, 68, 3]
    # Per-image optimized params (needed by Stage 3 for texture extraction)
    per_image_params: list[PerImageParams] = field(default_factory=list)
    loss_history: dict = field(default_factory=dict)

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DetectionResult:
    """Face detection outputs used by Stage 1."""

    lmks_dense: np.ndarray
    lmks_68: np.ndarray
    lmks_eyes: np.ndarray
    blend_scores: np.ndarray
    retinaface_kps: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None


@dataclass
class Stage1Output:
    """Stage 1 outputs consumed by Stage 2 and StageP3M."""

    shape: torch.Tensor
    expression: torch.Tensor
    head_pose: torch.Tensor
    jaw_pose: torch.Tensor
    texture: torch.Tensor
    lighting: torch.Tensor

    arcface_feat: torch.Tensor

    aligned_image: torch.Tensor
    face_mask: torch.Tensor

    lmks_68: torch.Tensor
    lmks_dense: torch.Tensor
    lmks_eyes: torch.Tensor

    focal_length: torch.Tensor
    principal_point: torch.Tensor

    parsing_map: Optional[torch.Tensor] = None
    lmks_98: Optional[torch.Tensor] = None

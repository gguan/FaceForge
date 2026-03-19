from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DetectionResult:
    """人脸检测结果"""
    lmks_dense: np.ndarray          # [478, 2] MediaPipe 稠密 landmark (像素坐标)
    lmks_68: np.ndarray             # [68, 2] 从 478 转换的 dlib 68 点
    lmks_eyes: np.ndarray           # [10, 2] 眼部 landmark (右眼5+左眼5)
    blend_scores: np.ndarray        # [52] blendshape 分数
    retinaface_kps: Optional[np.ndarray] = None  # [5, 2] RetinaFace 原生 5 点
    bbox: Optional[np.ndarray] = None            # [4] 人脸 bounding box


@dataclass
class Stage1Output:
    """Stage 1 输出, 传给 Stage 2"""
    # FLAME 参数
    shape: torch.Tensor             # [1, 300] — MICA
    expression: torch.Tensor        # [1, 100] — DECA(前50)+零(后50)
    head_pose: torch.Tensor         # [1, 3]   — DECA
    jaw_pose: torch.Tensor          # [1, 3]   — DECA
    texture: torch.Tensor           # [1, 50]  — DECA
    lighting: torch.Tensor          # [1, 9, 3] — DECA SH 系数

    # 预计算特征
    arcface_feat: torch.Tensor      # [1, 512] — L2 归一化, 用于 L_identity

    # 图像数据
    aligned_image: torch.Tensor     # [1, 3, 512, 512] — 对齐裁剪后的输入图
    face_mask: torch.Tensor         # [1, 512, 512] — 语义分割 mask

    # Landmark
    lmks_68: torch.Tensor           # [1, 68, 2] — 68 点 landmark (图像坐标)
    lmks_dense: torch.Tensor        # [1, 478, 2] — MediaPipe 稠密 landmark
    lmks_eyes: torch.Tensor         # [1, 10, 2] — 眼部 landmark

    # 相机
    focal_length: torch.Tensor      # [1, 1] — 归一化焦距
    principal_point: torch.Tensor   # [1, 2] — 主点

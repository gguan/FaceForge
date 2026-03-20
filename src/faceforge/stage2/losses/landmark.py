"""Landmark loss with region-weighted L1/L2.

Reference: flame-head-tracker tracker_base.py L626-633 (region weights),
           VHAP tracker.py (nose 10x weight → we use 3x compromise)
"""

import torch


def build_landmark_weights(device: torch.device, nose_weight: float = 3.0) -> torch.Tensor:
    """Build per-landmark weights [68] (multi-source blend)."""
    w = torch.ones(68, device=device)
    w[17:] = 1.5       # face interior (51pt)
    w[0:17] = 0.75     # jawline
    w[0:3] = 0.5       # jaw corners
    w[14:17] = 0.5     # jaw corners
    w[27:31] = nose_weight  # nose bridge (VHAP: 10x, we compromise at 3x)
    w[31:36] = 0.75        # nose bottom line — noisy, down-weight (ref: flame-head-tracker L631)
    w[36:48] = 3.0     # eye contours
    w[49:] = 2.0       # mouth contours
    return w


def build_landmark_weights_pixel3dmm(device: torch.device) -> torch.Tensor:
    """Build pixel3dmm-style landmark weights [68].

    pixel3dmm online stage only uses eye contour landmarks (36-48) with weight 5000.
    All other landmarks are zeroed out. The 5000 factor is applied via w_landmark in config.

    Ref: pixel3dmm tracker.py L1108-1111 (lmk_eye2 loss)
    """
    w = torch.zeros(68, device=device)
    w[36:48] = 1.0  # eye contours only
    return w


def landmark_loss(pred_68: torch.Tensor, pred_eyes: torch.Tensor,
                  target_68: torch.Tensor, target_eyes: torch.Tensor,
                  weights: torch.Tensor, image_size: int,
                  use_l2: bool = False,
                  visibility_mask_68: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted landmark loss.

    Args:
        pred_68: [B, 68, 2] projected landmark positions
        pred_eyes: [B, 10, 2] projected eye landmarks
        target_68: [B, 68, 2] target landmark positions
        target_eyes: [B, 10, 2] target eye positions
        weights: [68] per-landmark weights
        image_size: int, for normalization
        use_l2: True for Coarse (L2), False for Medium+ (L1)
        visibility_mask_68: [B, 68] optional occlusion mask

    Returns:
        scalar loss
    """
    # Normalize to [-1, 1]
    half = image_size / 2.0
    pred_n = pred_68 / half - 1.0
    target_n = target_68 / half - 1.0
    pred_eyes_n = pred_eyes / half - 1.0
    target_eyes_n = target_eyes / half - 1.0

    diff_68 = pred_n - target_n  # [B, 68, 2]

    w = weights.unsqueeze(0).unsqueeze(-1)  # [1, 68, 1]
    if visibility_mask_68 is not None:
        w = w * visibility_mask_68.unsqueeze(-1).float()

    if use_l2:
        loss_68 = (w * diff_68.pow(2)).mean()
    else:
        loss_68 = (w * diff_68.abs()).mean()

    diff_eyes = pred_eyes_n - target_eyes_n
    loss_eyes = 3.0 * diff_eyes.abs().mean()

    return loss_68 + loss_eyes

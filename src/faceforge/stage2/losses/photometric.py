"""Photometric (pixel-level) loss.

Reference: flame-head-tracker loss_utils.py L37-63 compute_batch_pixelwise_l1_loss
"""

import torch


def photometric_loss(rendered_image: torch.Tensor, target_image: torch.Tensor,
                     mask: torch.Tensor,
                     region_weight_map: torch.Tensor | None = None) -> torch.Tensor:
    """Masked L1 photometric loss, area-normalized.

    Args:
        rendered_image: [B, H, W, 3] rendered (BHWC)
        target_image: [B, 3, H, W] target (BCHW) → permuted internally
        mask: [B, H, W] binary face mask
        region_weight_map: [B, H, W] optional per-pixel weights

    Returns:
        scalar loss
    """
    target = target_image.permute(0, 2, 3, 1)  # → [B, H, W, 3]
    diff = (rendered_image - target).abs()       # [B, H, W, 3]

    mask_3d = mask.unsqueeze(-1).float()         # [B, H, W, 1]
    diff = diff * mask_3d

    if region_weight_map is not None:
        diff = diff * region_weight_map.unsqueeze(-1)

    # Normalize by (mask_area × channels) — ref: flame-head-tracker
    C = 3
    mask_area = mask.float().sum(dim=(1, 2))  # [B]
    loss_per_sample = diff.sum(dim=(1, 2, 3)) / (mask_area * C + 1e-8)  # [B]
    return loss_per_sample.mean()

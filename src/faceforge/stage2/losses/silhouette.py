"""Silhouette loss.

Reference: pixel3dmm tracker.py L942-950
Only penalizes in background regions (seg <= 1) to avoid noisy gradients
from ambiguous face boundaries (hair, ears, neck).
"""

import torch


def silhouette_loss(rendered_mask: torch.Tensor,
                    target_fg_mask: torch.Tensor,
                    valid_bg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Foreground mask L1 loss, restricted to background pixels.

    Ref: pixel3dmm tracker.py L942-950
        loss = |valid_bg * (fg_mask - rendered_fg)|.mean()

    Args:
        rendered_mask: [B, H, W, 1] from renderer (soft foreground mask)
        target_fg_mask: [B, H, W] binary foreground mask
            pixel3dmm uses: (seg==2)|(seg==6-10)|(seg==12)|(seg==13)
        valid_bg_mask: [B, H, W] binary, True where pixel is background (seg<=1).
            If None, falls back to all-pixels (non-pixel3dmm behavior).

    Returns:
        scalar loss
    """
    rendered = rendered_mask.squeeze(-1)  # [B, H, W]
    target = target_fg_mask.float()

    diff = (rendered - target).abs()  # [B, H, W]

    if valid_bg_mask is not None:
        # Only penalize in background regions (pixel3dmm convention)
        diff = diff * valid_bg_mask.float()

    return diff.mean()

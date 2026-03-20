"""Silhouette loss.

Reference: VHAP tracker.py (sil loss), Pixel3DMM tracker.py (fg mask L1)
"""

import torch


def silhouette_loss(rendered_mask: torch.Tensor,
                    target_mask: torch.Tensor) -> torch.Tensor:
    """L1 foreground mask difference.

    Args:
        rendered_mask: [B, H, W, 1] from renderer
        target_mask: [B, H, W] binary face mask

    Returns:
        scalar loss
    """
    rendered = rendered_mask.squeeze(-1)  # [B, H, W]
    target = target_mask.float()
    return (rendered - target).abs().mean()

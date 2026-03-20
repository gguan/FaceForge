"""Normal consistency loss.

Reference: pixel3dmm tracker.py L975-996 (normal loss with outlier filtering)
"""

import torch
import torch.nn.functional as F


def normal_loss(rendered_normals: torch.Tensor, predicted_normals: torch.Tensor,
                face_mask: torch.Tensor,
                eye_mask: torch.Tensor | None = None,
                delta_n: float = 0.15,
                eye_dilate_kernel: int = 5) -> torch.Tensor:
    """Normal consistency with outlier filtering and eye exclusion.

    Args:
        rendered_normals: [B, H, W, 3] from renderer
        predicted_normals: [B, 3, H, W] from Pixel3DMM → permuted to [B, H, W, 3]
        face_mask: [B, H, W] binary
        eye_mask: [B, H, W] optional eye region mask (BiSeNet class 4+5)
        delta_n: outlier threshold (normals differing more are invalid)
        eye_dilate_kernel: Gaussian blur kernel for eye dilation

    Returns:
        scalar loss
    """
    pred_n = predicted_normals.permute(0, 2, 3, 1)  # [B, H, W, 3]

    # Outlier filtering: large differences are invalid
    l_map = rendered_normals - pred_n
    valid = (l_map.abs().sum(dim=-1) / 3.0) < delta_n  # [B, H, W]

    # Eye region dilation and exclusion
    if eye_mask is not None and eye_dilate_kernel > 0:
        eye_f = eye_mask.float().unsqueeze(1)  # [B, 1, H, W]
        k = eye_dilate_kernel
        kernel = torch.ones(1, 1, k, k, device=eye_f.device) / (k * k)
        dilated = F.conv2d(eye_f, kernel, padding=k // 2)
        valid = valid & (dilated.squeeze(1) < 0.5)

    mask = face_mask.bool() & valid
    mask_sum = mask.sum().clamp(min=1)

    # Cosine distance
    cos_sim = (rendered_normals * pred_n).sum(dim=-1)  # [B, H, W]
    loss = ((1 - cos_sim) * mask.float()).sum() / mask_sum
    return loss

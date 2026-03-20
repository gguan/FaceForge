"""Normal consistency loss.

Reference: pixel3dmm tracker.py L975-996 (normal loss with outlier filtering)
"""

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def normal_loss(rendered_normals: torch.Tensor, predicted_normals: torch.Tensor,
                face_mask: torch.Tensor,
                cam_rotation: torch.Tensor | None = None,
                eye_mask: torch.Tensor | None = None,
                delta_n: float = 0.33,
                eye_dilate_kernel: int = 13) -> torch.Tensor:
    """Normal consistency with outlier filtering and eye exclusion.

    Uses L1 of raw normal differences (not cosine distance) to match the
    pixel3dmm reference.  Rendered normals are rotated back to FLAME canonical
    space before comparison (ref: pixel3dmm tracker.py L984-986).

    Args:
        rendered_normals: [B, H, W, 3] from renderer (camera space)
        predicted_normals: [B, 3, H, W] from Pixel3DMM (FLAME canonical space)
        face_mask: [B, H, W] binary
        cam_rotation: [B, 3, 3] camera rotation matrix (R from extrinsics).
            If provided, rendered normals are rotated from camera space to
            world/FLAME space: n_world = R^T @ n_cam.
        eye_mask: [B, H, W] optional eye region mask (BiSeNet class 4+5)
        delta_n: outlier threshold (mean per-channel absolute diff)
        eye_dilate_kernel: Gaussian blur kernel size for eye dilation

    Returns:
        scalar loss
    """
    pred_n = predicted_normals.permute(0, 2, 3, 1)  # [B, H, W, 3]

    # Rotate rendered normals from camera space back to FLAME canonical space
    # so they are comparable with Pixel3DMM predictions.
    # n_world = R^T @ n_cam  (R is orthogonal)
    rend_n = rendered_normals
    if cam_rotation is not None:
        B, H, W, _ = rend_n.shape
        rend_flat = rend_n.reshape(B, -1, 3)  # [B, HW, 3]
        # R^T rotates camera→world: [B, 3, 3]^T @ [B, HW, 3]^T = [B, 3, HW]
        rend_flat = torch.bmm(cam_rotation.transpose(1, 2), rend_flat.transpose(1, 2))
        rend_n = rend_flat.transpose(1, 2).reshape(B, H, W, 3)

    # L1 difference (ref: pixel3dmm uses .abs().mean(), not cosine distance)
    l_map = rend_n - pred_n  # [B, H, W, 3]

    # Outlier filtering: large differences are invalid
    valid = (l_map.abs().sum(dim=-1) / 3.0) < delta_n  # [B, H, W]

    # Eye region dilation and exclusion (Gaussian blur, kernel=13 per reference)
    if eye_mask is not None and eye_dilate_kernel > 0:
        eye_f = eye_mask.float().unsqueeze(1)  # [B, 1, H, W]
        k = eye_dilate_kernel
        # Gaussian blur dilation — matches pixel3dmm tracker.py L979-982:
        # gaussian_blur(ops['mask_images_eyes'], [ksize, ksize], sigma=[ksize, ksize]) > 0
        # Any pixel with nonzero Gaussian response from the eye region is excluded.
        dilated = gaussian_blur(eye_f, [k, k], sigma=[k, k])  # [B, 1, H, W]
        valid = valid & (dilated.squeeze(1) <= 0.0)

    mask = face_mask.bool() & valid

    # L1 loss over valid pixels, .mean() over full tensor including zeros
    # to match pixel3dmm reference normalization (ref: losses.py L122-126)
    loss = (l_map.abs() * mask.unsqueeze(-1).float()).mean()
    return loss

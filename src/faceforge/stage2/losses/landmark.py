"""Landmark losses matching pixel3dmm tracker.py exactly.

pixel3dmm optimize_color (opt_pre) uses:
  - lmk_eye2:        eye contours (36:48)   × w_lmks × 5      = 5000  (L2/MSE)
  - lmk_eye_closure: upper-lower eyelid gap  × w_lmks_lid × 500 = 500000 (L2/MSE)
  - lmk_iris_left/right: iris vertices       × w_lmks_iris × 50  = 50000  (L2/MSE)

pixel3dmm optimize_camera (first half) uses:
  - lmk68: all 68 landmarks × 3000 (L2/MSE)

Normalization: pixel3dmm scale_lmks divides by [1/w, 1/h] → range [0, 1].
Loss: (diff^2 * mask).mean()  — squared L2 (MSE), NOT L1.

Ref: pixel3dmm tracking/util.py L88-156, tracking/tracker.py L884-910
"""

import torch


# pixel3dmm iris vertex indices (tracker.py L58-59)
LEFT_IRIS_FLAME = [4597, 4542, 4510, 4603, 4570]
RIGHT_IRIS_FLAME = [4051, 3996, 3964, 3932, 4028]


def _scale_lmks(pred: torch.Tensor, target: torch.Tensor,
                image_size: tuple[int, int] | int):
    """pixel3dmm scale_lmks: normalize by image dimensions.

    Ref: pixel3dmm tracking/util.py L88-93
    """
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size
    scale = torch.tensor([1.0 / w, 1.0 / h], device=pred.device, dtype=pred.dtype)
    return pred * scale, target * scale


def lmk_loss_l2(pred: torch.Tensor, target: torch.Tensor,
                image_size: int, lmk_mask: torch.Tensor | None = None) -> torch.Tensor:
    """pixel3dmm lmk_loss: squared L2 (MSE) after normalization.

    Ref: pixel3dmm tracking/util.py L96-102
    ``(diff^2 * mask).mean()``
    """
    pred_n, target_n = _scale_lmks(pred, target, image_size)
    diff_sq = (pred_n - target_n).pow(2)
    if lmk_mask is not None:
        diff_sq = diff_sq * lmk_mask
    return diff_sq.mean()


def eye_closure_loss(pred_68: torch.Tensor, target_68: torch.Tensor,
                     image_size: int,
                     lmk_mask: torch.Tensor | None = None) -> torch.Tensor:
    """pixel3dmm eye_closure_lmk_loss: penalize eyelid gap mismatch.

    Measures (upper - lower) distance in both pred and target, then MSE.
    Ref: pixel3dmm tracking/util.py L146-156
    """
    upper = [37, 38, 43, 44]
    lower = [41, 40, 47, 46]
    pred_n, target_n = _scale_lmks(pred_68, target_68, image_size)
    gap_pred = pred_n[:, upper, :] - pred_n[:, lower, :]
    gap_target = target_n[:, upper, :] - target_n[:, lower, :]
    diff_sq = (gap_pred - gap_target).pow(2)
    if lmk_mask is not None:
        diff_sq = diff_sq * lmk_mask[:, upper, :]
    return diff_sq.mean()


def iris_loss(proj_vertices: torch.Tensor, target_iris: torch.Tensor,
              iris_indices: list[int], image_size: int,
              iris_mask: torch.Tensor | None = None) -> torch.Tensor:
    """pixel3dmm iris landmark loss: MSE on projected iris vertex vs detected iris.

    pixel3dmm uses only the first iris vertex ([:1]) per eye.
    Ref: pixel3dmm tracker.py L905-910
    """
    pred_iris = proj_vertices[:, iris_indices[:1], :2]
    pred_n, target_n = _scale_lmks(pred_iris, target_iris, image_size)
    diff_sq = (pred_n - target_n).pow(2)
    if iris_mask is not None:
        diff_sq = diff_sq * iris_mask
    return diff_sq.mean()


# ---------------------------------------------------------------------------
# Combined landmark loss for Stage 2 (matching pixel3dmm opt_pre)
# ---------------------------------------------------------------------------

def landmark_loss_p3m(pred_68: torch.Tensor,
                      target_68: torch.Tensor,
                      image_size: int,
                      lmk_mask: torch.Tensor | None = None,
                      proj_vertices: torch.Tensor | None = None,
                      target_iris_left: torch.Tensor | None = None,
                      target_iris_right: torch.Tensor | None = None,
                      iris_mask_left: torch.Tensor | None = None,
                      iris_mask_right: torch.Tensor | None = None,
                      w_lmks: float = 1000.0,
                      w_lmks_lid: float = 1000.0,
                      w_lmks_iris: float = 1000.0,
                      ) -> torch.Tensor:
    """Combined landmark loss matching pixel3dmm optimize_color exactly.

    Returns:
        scalar loss (already weighted)
    """
    loss = torch.tensor(0.0, device=pred_68.device)

    # Eye contour (36:48) — w_lmks * 5
    eye_mask = lmk_mask[:, 36:48, :] if lmk_mask is not None else None
    loss = loss + w_lmks * 5.0 * lmk_loss_l2(
        pred_68[:, 36:48, :2], target_68[:, 36:48, :],
        image_size, eye_mask)

    # Eye closure — w_lmks_lid * 500
    loss = loss + w_lmks_lid * 500.0 * eye_closure_loss(
        pred_68, target_68, image_size, lmk_mask)

    # Iris — w_lmks_iris * 50 each
    if proj_vertices is not None and target_iris_left is not None:
        loss = loss + w_lmks_iris * 50.0 * iris_loss(
            proj_vertices, target_iris_left,
            LEFT_IRIS_FLAME, image_size, iris_mask_left)
    if proj_vertices is not None and target_iris_right is not None:
        loss = loss + w_lmks_iris * 50.0 * iris_loss(
            proj_vertices, target_iris_right,
            RIGHT_IRIS_FLAME, image_size, iris_mask_right)

    return loss


def landmark_loss_camera(pred_68: torch.Tensor,
                         target_68: torch.Tensor,
                         image_size: int,
                         lmk_mask: torch.Tensor | None = None,
                         ) -> torch.Tensor:
    """Landmark loss for optimize_camera first half: all 68 pts × 3000.

    Ref: pixel3dmm tracker.py L649
    """
    return 3000.0 * lmk_loss_l2(
        pred_68[:, :, :2], target_68, image_size, lmk_mask)


# ---------------------------------------------------------------------------
# Legacy functions (kept for backward compat, unused in pixel3dmm_compat)
# ---------------------------------------------------------------------------

def build_landmark_weights(device: torch.device, nose_weight: float = 3.0) -> torch.Tensor:
    """Build per-landmark weights [68] (multi-source blend)."""
    w = torch.ones(68, device=device)
    w[17:] = 1.5
    w[0:17] = 0.75
    w[0:3] = 0.5
    w[14:17] = 0.5
    w[27:31] = nose_weight
    w[31:36] = 0.75
    w[36:48] = 3.0
    w[49:] = 2.0
    return w


def build_landmark_weights_pixel3dmm(device: torch.device) -> torch.Tensor:
    """Build pixel3dmm-style landmark weights [68]."""
    w = torch.zeros(68, device=device)
    w[36:48] = 1.0
    return w

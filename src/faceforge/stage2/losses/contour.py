"""Contour-aware loss (baseline).

Reference: HRN models/losses.py L187-205 contour_aware_loss
NOTE: This is boundary-checking, NOT Chamfer distance.
"""

import torch
import torch.nn.functional as F


def extract_row_boundaries(face_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-row left and right face boundaries from mask.

    Args:
        face_mask: [B, H, W] binary mask

    Returns:
        left_points: [B, H] leftmost face pixel x per row
        right_points: [B, H] rightmost face pixel x per row
    """
    B, H, W = face_mask.shape
    device = face_mask.device

    # For each row, find leftmost and rightmost True pixel
    x_coords = torch.arange(W, device=device).float()  # [W]

    mask_f = face_mask.float()  # [B, H, W]

    # Left: first nonzero from left → argmax works on binary mask
    left = torch.argmax(mask_f, dim=2).float()  # [B, H]

    # Right: first nonzero from right
    right = (W - 1) - torch.argmax(mask_f.flip(2), dim=2).float()  # [B, H]

    # For rows with no face pixels, set boundaries to center (no penalty)
    row_has_face = mask_f.sum(dim=2) > 0
    center = W / 2.0
    left = torch.where(row_has_face, left, torch.full_like(left, center))
    right = torch.where(row_has_face, right, torch.full_like(right, center))

    return left, right


def contour_loss(projected_vertices: torch.Tensor,
                 boundary_vertex_indices: torch.Tensor,
                 face_mask: torch.Tensor,
                 image_size: int = 512) -> torch.Tensor:
    """Contour-aware loss: penalize vertices outside face boundary.

    Ref: HRN losses.py L187-205 (reimplemented, not Chamfer)

    Args:
        projected_vertices: [B, V, 2] all projected FLAME vertices
        boundary_vertex_indices: [N] indices of jaw/boundary vertices
        face_mask: [B, H, W] binary face mask
        image_size: int

    Returns:
        scalar loss
    """
    B = projected_vertices.shape[0]
    W = image_size

    left_points, right_points = extract_row_boundaries(face_mask)  # [B, H]

    # Get boundary vertices
    verts = projected_vertices[:, boundary_vertex_indices]  # [B, N, 2]
    verts_x = verts[:, :, 0]  # [B, N]
    # projected_vertices y is already in image y-down (row index, 0=top)
    verts_y_idx = torch.clamp(verts[:, :, 1].long(), 0, image_size - 1)  # [B, N]

    # Look up left/right boundaries for each vertex's row
    batch_idx = torch.arange(B, device=verts.device).unsqueeze(1).expand(-1, verts_y_idx.shape[1])
    verts_left = left_points[batch_idx, verts_y_idx]    # [B, N]
    verts_right = right_points[batch_idx, verts_y_idx]  # [B, N]

    # Boundary check: dist = (left - x)(right - x) / width²
    # negative → inside, positive → outside
    dist = (verts_left - verts_x) / W * (verts_right - verts_x) / W  # [B, N]

    # Normalize by max distance to boundary (ref: HRN L202)
    max_dist = torch.max((verts_left - verts_x).abs() / W,
                         (verts_right - verts_x).abs() / W).clamp(min=1e-6)
    dist = dist / max_dist

    # Only penalize outside (dist > 0), with small offset to avoid zero grad
    dist = dist + 0.01
    dist = F.relu(dist)
    dist = dist - 0.01
    loss = dist.abs().mean()

    return loss

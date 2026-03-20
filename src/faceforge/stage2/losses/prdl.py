"""Part Re-projection Distance Loss (PRDL).

Alternative to contour + region_weight + silhouette (when config.use_prdl=True).

Reference: 3DDFA-V3 (Wang et al., CVPR 2024), arXiv:2312.00311
"""

import torch
import torch.nn.functional as F


# BiSeNet 19-class → FLAME region mapping
SEG_TO_REGION = {
    'nose': [10],
    'left_eye': [4],
    'right_eye': [5],
    'lips': [12, 13],
    'skin_lower': [1],  # jaw/chin area (lower half of skin)
}

# Corresponding FLAME_masks.pkl region names
FLAME_REGION_MAP = {
    'nose': 'nose',
    'left_eye': 'left_eye_region',
    'right_eye': 'right_eye_region',
    'lips': 'lips',
    'skin_lower': 'boundary',  # or jaw-related mask
}


class PRDLLoss:
    """Part Re-projection Distance Loss.

    For each semantic face region:
    1. BiSeNet mask → 2D target point set
    2. FLAME region vertices → project to 2D → predicted point set
    3. Grid-based statistical distance between the two sets
    """

    def __init__(self, flame_model, grid_size: int = 16, device: torch.device = None):
        self.device = device or torch.device('cuda:0')
        self.grid_size = grid_size

        # Pre-extract FLAME region vertex indices
        self.flame_regions = {}
        for part_name, flame_key in FLAME_REGION_MAP.items():
            try:
                self.flame_regions[part_name] = flame_model.get_region_indices(flame_key).to(self.device)
            except KeyError:
                pass

    def compute(self, projected_vertices: torch.Tensor,
                face_segmentation: torch.Tensor,
                image_size: int = 512) -> torch.Tensor:
        """Compute PRDL loss.

        Args:
            projected_vertices: [B, V, 2] projected FLAME vertices
            face_segmentation: [B, H, W] BiSeNet 19-class labels
            image_size: int

        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_parts = 0

        for part_name, seg_classes in SEG_TO_REGION.items():
            if part_name not in self.flame_regions:
                continue

            # Target: mask → 2D point set
            part_mask = torch.zeros_like(face_segmentation, dtype=torch.bool)
            for c in seg_classes:
                part_mask = part_mask | (face_segmentation == c)

            # For skin_lower, only take bottom half
            if part_name == 'skin_lower':
                part_mask[:, :image_size // 2, :] = False

            target_coords = part_mask[0].nonzero().float()  # [N_t, 2] (y, x)
            if target_coords.shape[0] < 10:
                continue
            target_pts = target_coords[:, [1, 0]]  # swap to (x, y)

            # Predicted: FLAME region vertices → 2D
            region_idx = self.flame_regions[part_name]
            pred_pts = projected_vertices[:, region_idx, :2].squeeze(0)  # [N_p, 2]

            if pred_pts.shape[0] < 3:
                continue

            # Grid statistical distance
            part_loss = self._grid_distance(pred_pts, target_pts, image_size)
            total_loss = total_loss + part_loss
            n_parts += 1

        if n_parts > 0:
            total_loss = total_loss / n_parts

        return total_loss

    def _grid_distance(self, pred_pts: torch.Tensor, target_pts: torch.Tensor,
                       image_size: int) -> torch.Tensor:
        """Compute grid-based statistical distance between two point sets.

        Args:
            pred_pts: [N_p, 2] predicted (differentiable)
            target_pts: [N_t, 2] target (fixed)
            image_size: int

        Returns:
            scalar distance
        """
        G = self.grid_size
        device = pred_pts.device

        # Create uniform grid anchors
        grid_x = torch.linspace(0, image_size, G, device=device)
        grid_y = torch.linspace(0, image_size, G, device=device)
        anchors = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).reshape(-1, 2)  # [G*G, 2]

        # Distance from each anchor to each point set
        # pred: [G*G, N_p] → mean dist per anchor
        d_pred = torch.cdist(anchors.unsqueeze(0), pred_pts.unsqueeze(0)).squeeze(0)  # [G*G, N_p]
        mean_d_pred = d_pred.min(dim=1).values  # [G*G] nearest distance

        d_target = torch.cdist(anchors.unsqueeze(0), target_pts.unsqueeze(0)).squeeze(0)
        mean_d_target = d_target.min(dim=1).values

        # Statistical distance: compare the two distance profiles
        loss = (mean_d_pred - mean_d_target).pow(2).mean() / (image_size ** 2)
        return loss

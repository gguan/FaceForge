"""Pixel3DMM dense UV correspondence loss.

Reference: pixel3dmm tracking/losses.py L46-139 (UVLoss class)
"""

import torch
import torch.nn.functional as F


class Pixel3DMMUVLoss:
    """Dense UV correspondence loss with two-stage threshold refinement.

    Establishes per-vertex correspondences between FLAME canonical UV coords
    and Pixel3DMM predicted per-pixel UV, then penalizes 2D projection distance.
    """

    def __init__(self, flame_uv_coords: torch.Tensor,
                 uv_valid_verts: torch.Tensor,
                 device: torch.device):
        """
        Args:
            flame_uv_coords: [5023, 2] FLAME canonical UV (V-flipped)
            uv_valid_verts: [N] int64 valid vertex indices
            device: torch device
        """
        self.valid_verts = uv_valid_verts.to(device)
        self.can_uv = flame_uv_coords[self.valid_verts].unsqueeze(0).to(device)  # [1, N, 2]
        self.device = device

        # Will be set by compute_correspondences
        self.target_pixel_coords = None  # [N, 2]
        self.correspondence_mask = None  # [N] bool
        self.delta_uv = None
        self.dist_uv = None
        self._knn_dists = None

    def compute_correspondences(self, uv_prediction: torch.Tensor,
                                delta_uv: float = 0.00005,
                                dist_uv: float = 15.0,
                                image_size: int = 512):
        """Build KNN correspondences from FLAME UV to predicted UV.

        Ref: pixel3dmm losses.py L88-96

        Args:
            uv_prediction: [1, 2, H, W] Pixel3DMM UV output in [0,1]
            delta_uv: max UV-space KNN distance threshold
            dist_uv: max 2D pixel distance threshold
            image_size: int
        """
        self.delta_uv = delta_uv
        self.dist_uv = dist_uv

        H, W = uv_prediction.shape[2], uv_prediction.shape[3]

        # Flatten predicted UV to pixel set [1, H*W, 2]
        gt_uv = uv_prediction.permute(0, 2, 3, 1).reshape(1, -1, 2)  # [1, HW, 2]

        # KNN: for each FLAME vertex UV, find nearest pixel UV
        try:
            from pytorch3d.ops import knn_points
            knn_result = knn_points(self.can_uv, gt_uv)
            knn_dists = knn_result.dists.squeeze(-1)   # [1, N]
            knn_idx = knn_result.idx.squeeze(-1)       # [1, N]
        except ImportError:
            # Fallback: manual distance computation (slower)
            dists = torch.cdist(self.can_uv, gt_uv)  # [1, N, HW]
            knn_dists, knn_idx = dists.min(dim=-1)    # [1, N]

        # Convert flat index to pixel coordinates
        pixel_y = knn_idx // W  # [1, N]
        pixel_x = knn_idx % W   # [1, N]
        self.target_pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1).squeeze(0).float()  # [N, 2]

        # Filter by UV distance threshold
        self.correspondence_mask = (knn_dists.squeeze(0) < delta_uv)  # [N]

        self._knn_dists = knn_dists.squeeze(0)

    def tighten_thresholds(self, delta_fine: float, dist_fine: float):
        """Tighten correspondence thresholds for medium+ stages.

        Ref: pixel3dmm losses.py L70-74 finish_stage1()
        """
        if self._knn_dists is not None:
            self.delta_uv = delta_fine
            self.dist_uv = dist_fine
            self.correspondence_mask = (self._knn_dists < delta_fine)

    def compute_loss(self, projected_vertices_2d: torch.Tensor,
                     image_size: int = 512,
                     visibility_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute UV correspondence loss.

        Ref: pixel3dmm losses.py L99-127

        Args:
            projected_vertices_2d: [B, 5023, 2] projected pixel coords
            image_size: for normalization
            visibility_mask: [B, 5023] optional occlusion mask

        Returns:
            scalar loss
        """
        if self.target_pixel_coords is None:
            return torch.tensor(0.0, device=self.device)

        B = projected_vertices_2d.shape[0]
        pred = projected_vertices_2d[:, self.valid_verts]  # [B, N, 2]
        target = self.target_pixel_coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # 2D distance
        dist_2d = (pred - target).abs().sum(dim=-1)  # [B, N]

        # Combined mask: UV threshold + 2D distance threshold
        mask = self.correspondence_mask.unsqueeze(0).expand(B, -1)  # [B, N]
        mask = mask & (dist_2d < self.dist_uv)

        # Occlusion filtering
        if visibility_mask is not None:
            vis = visibility_mask[:, self.valid_verts]
            mask = mask & vis

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # L1, normalized by image size (ref: pixel3dmm losses.py L117-126)
        loss = ((pred - target).abs() / image_size * mask.unsqueeze(-1).float()).sum()
        loss = loss / (mask.sum() * 2 + 1e-8)  # *2 for x,y
        return loss

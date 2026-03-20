"""Identity loss via ArcFace cosine similarity.

Reference: MICA (ECCV 2022) ArcFace encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """Identity loss: 1 - cos_sim(arcface(rendered), target_feat).

    ArcFace is frozen; gradients flow through the rendered image only.
    """

    def __init__(self, arcface_model: nn.Module):
        super().__init__()
        self.arcface = arcface_model
        self.arcface.eval()
        for p in self.arcface.parameters():
            p.requires_grad_(False)

    def forward(self, rendered_image: torch.Tensor,
                target_arcface_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rendered_image: [B, H, W, 3] rendered face (0-1 range, BHWC)
            target_arcface_feat: [B, 512] precomputed L2-normalized

        Returns:
            scalar loss
        """
        # Center crop and resize to 112×112 for ArcFace
        face_crop = self._crop_face(rendered_image, 112)  # [B, 3, 112, 112]

        # ArcFace normalization: (pixel*255 - 127.5) / 127.5 → [-1, 1]
        face_norm = (face_crop * 255.0 - 127.5) / 127.5

        # Encode (ArcFace frozen, but input has gradients)
        feat = self.arcface(face_norm)
        feat = F.normalize(feat, dim=-1)

        # Cosine distance
        cos_sim = F.cosine_similarity(feat, target_arcface_feat, dim=-1)
        return (1 - cos_sim).mean()

    @staticmethod
    def _crop_face(image: torch.Tensor, target_size: int) -> torch.Tensor:
        """Center crop rendered image to face region and resize.

        Args:
            image: [B, H, W, 3] BHWC format, values in [0, 1]
            target_size: output size (112 for ArcFace)

        Returns:
            [B, 3, target_size, target_size] BCHW
        """
        B, H, W, C = image.shape

        # Center crop (keep ~80% to focus on face)
        margin = int(H * 0.1)
        cropped = image[:, margin:H - margin, margin:W - margin, :]

        # BHWC → BCHW for interpolate
        cropped = cropped.permute(0, 3, 1, 2)

        # Resize to ArcFace input size
        return F.interpolate(cropped, size=(target_size, target_size),
                             mode='bilinear', align_corners=False)

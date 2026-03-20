"""Pixel3DMM inference wrapper for Stage 2 preprocessing.

Runs once per image (not in optimization loop) to predict dense UV + Normal maps.

Reference: pixel3dmm scripts/network_inference.py L115-158
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from faceforge._paths import PROJECT_ROOT
from .config import Stage2Config


class Pixel3DMMInference:
    """Pixel3DMM UV/Normal prediction."""

    def __init__(self, config: Stage2Config):
        self.config = config
        self.device = config.device
        self._uv_model = None
        self._normal_model = None

    def _ensure_loaded(self):
        if self._uv_model is not None:
            return

        code_base = str(PROJECT_ROOT / self.config.pixel3dmm_code_base)
        src_path = f'{code_base}/src'
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Patch env_paths to avoid .env dependency
        import pixel3dmm.env_paths as env_paths
        env_paths.CODE_BASE = code_base

        from pixel3dmm.lightning.p3dmm_system import P3DMMSystem

        uv_ckpt = str(PROJECT_ROOT / self.config.pixel3dmm_uv_ckpt)
        normal_ckpt = str(PROJECT_ROOT / self.config.pixel3dmm_normal_ckpt)

        self._uv_model = P3DMMSystem.load_from_checkpoint(uv_ckpt).to(self.device).eval()
        self._normal_model = P3DMMSystem.load_from_checkpoint(normal_ckpt).to(self.device).eval()

    @torch.no_grad()
    def predict(self, aligned_image: torch.Tensor,
                face_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict UV and Normal maps.

        Args:
            aligned_image: [1, 3, 512, 512] uint8 or [0,255] float
            face_mask: [1, 512, 512] int (19-class BiSeNet)

        Returns:
            uv_map: [1, 2, 512, 512] in [0, 1]
            normal_map: [1, 3, 512, 512] unit normals
        """
        self._ensure_loaded()

        # Prepare image: [0, 1], BHWC format for pixel3dmm
        image = aligned_image.float() / 255.0 if aligned_image.max() > 1.0 else aligned_image.float()
        image = image.permute(0, 2, 3, 1)  # [1, H, W, 3]
        image = image.unsqueeze(1)  # [1, 1, H, W, 3]

        # Build mask: skin + brow + eye + nose + lip, exclude mouth interior
        # Ref: network_inference.py mask construction
        seg = face_mask.squeeze(0)  # [H, W]
        mask = ((seg == 1) | (seg == 2) | (seg == 3) |   # skin, brows
                (seg == 4) | (seg == 5) |                 # eyes
                (seg == 10) |                             # nose
                (seg == 12) | (seg == 13))                # lips
        mask = mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, H, W, 1]

        # Build batch dict
        batch = {
            'image': image.to(self.device),
            'mask': mask.to(self.device),
        }

        # Original + horizontally flipped, average for stability
        batch_flip = {
            'image': image.flip(3).to(self.device),
            'mask': mask.flip(3).to(self.device),
        }

        # UV prediction
        uv_pred = self._uv_model.net(batch)
        uv_pred_flip = self._uv_model.net(batch_flip)
        # Flip-back: reverse spatial W dim (dim -2 for channels-last [...,H,W,C]).
        # Raw output is in [-1,1]; horizontal flip reverses u-axis → negate u channel.
        uv_pred_flip = uv_pred_flip.flip(-2).clone()
        uv_pred_flip[..., 0] = -uv_pred_flip[..., 0]
        uv_avg = (uv_pred + uv_pred_flip) / 2
        uv_map = torch.clamp((uv_avg + 1) / 2, 0, 1)  # [-1,1] → [0,1]

        # Reshape to [1, 2, H, W]
        if uv_map.dim() == 5:
            uv_map = uv_map.squeeze(1)  # remove seq dim
        if uv_map.shape[-1] == 2:
            uv_map = uv_map.permute(0, 3, 1, 2)  # BHWC → BCHW

        # Normal prediction
        normal_pred = self._normal_model.net(batch)
        normal_pred_flip = self._normal_model.net(batch_flip)
        # Flip-back: reverse spatial W dim; negate x-normal (horizontal mirror)
        normal_pred_flip = normal_pred_flip.flip(-2)
        normal_pred_flip = normal_pred_flip.clone()
        normal_pred_flip[..., 0] = -normal_pred_flip[..., 0]
        normal_avg = (normal_pred + normal_pred_flip) / 2

        if normal_avg.dim() == 5:
            normal_avg = normal_avg.squeeze(1)
        if normal_avg.shape[-1] == 3:
            normal_avg = normal_avg.permute(0, 3, 1, 2)  # → [1, 3, H, W]

        # Coordinate convention swap: [x, 1-z, 1-y], then re-normalize.
        # Normalization must come AFTER the swap: 1-z maps [-1,1]→[2,0], breaking
        # unit length. Re-normalize to restore unit normals.
        normal_prenorm = F.normalize(normal_avg, dim=1)
        normal_swapped = torch.stack([
            normal_prenorm[:, 0],
            1 - normal_prenorm[:, 2],
            1 - normal_prenorm[:, 1],
        ], dim=1)
        normal_map_swapped = F.normalize(normal_swapped, dim=1)

        return uv_map, normal_map_swapped

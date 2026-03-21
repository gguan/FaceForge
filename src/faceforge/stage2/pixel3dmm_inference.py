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

        import os
        code_base = str(PROJECT_ROOT / self.config.pixel3dmm_code_base)
        src_path = f'{code_base}/src'
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Set env vars BEFORE importing pixel3dmm (env_paths reads them at import time)
        os.environ.setdefault('PIXEL3DMM_CODE_BASE', code_base)
        os.environ.setdefault('PIXEL3DMM_PREPROCESSED_DATA', str(PROJECT_ROOT / 'output'))
        os.environ.setdefault('PIXEL3DMM_TRACKING_OUTPUT', str(PROJECT_ROOT / 'output'))

        import pixel3dmm.env_paths as env_paths
        env_paths.CODE_BASE = code_base

        from pixel3dmm.lightning.p3dmm_system import system as P3DMMSystem

        uv_ckpt = str(PROJECT_ROOT / self.config.pixel3dmm_uv_ckpt)
        normal_ckpt = str(PROJECT_ROOT / self.config.pixel3dmm_normal_ckpt)

        # weights_only=False needed for omegaconf-based checkpoints (PyTorch >= 2.6)
        # strict=False because checkpoints contain extra FLAME/camera keys not in the model
        # NOTE: pixel3dmm's network_inference.py intentionally does NOT call .eval()
        # (L110: "# model.eval()").  BatchNorm in train mode uses per-batch stats
        # which produces different (higher contrast) outputs than eval mode's running
        # stats.  We replicate this to match pixel3dmm's reference outputs exactly.
        self._uv_model = P3DMMSystem.load_from_checkpoint(
            uv_ckpt, map_location=self.device, weights_only=False, strict=False,
        ).to(self.device)
        self._normal_model = P3DMMSystem.load_from_checkpoint(
            normal_ckpt, map_location=self.device, weights_only=False, strict=False,
        ).to(self.device)

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

        # Build mask: match pixel3dmm reference exactly
        # Ref: network_inference.py L127
        # (seg == 2) | ((seg > 3) & (seg < 14)) & ~(seg == 11)
        # = classes 2, 4-10, 12, 13 (excludes 0=bg, 1=skin, 3=r_brow, 11=mouth, 14+=neck/cloth/hair)
        seg = face_mask.squeeze(0)  # [H, W]
        mask = ((seg == 2) | ((seg > 3) & (seg < 14))) & ~(seg == 11)
        mask = mask.long().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] matching ref shape

        # Build batch dict — keys must match pixel3dmm Network.forward expectations
        batch = {
            'tar_rgb': image.to(self.device),
            'tar_msk': mask.to(self.device),
        }

        # Original + horizontally flipped, average for stability (TTA)
        batch_flip = {
            'tar_rgb': image.flip(3).to(self.device),  # flip W dim in BHWC
            'tar_msk': mask.flip(3).to(self.device),    # flip W dim in B1HW
        }

        # UV prediction
        # model.net() returns (output_dict, conf) where output_dict values
        # are channels-first: [B, seq, C, H, W]
        uv_output, _ = self._uv_model.net(batch)
        uv_output_flip, _ = self._uv_model.net(batch_flip)

        uv_pred = uv_output['uv_map']        # [B, seq, 2, H, W]
        uv_pred_flip = uv_output_flip['uv_map']

        # Flip-back: reverse spatial W dim (dim=4 for [B,seq,C,H,W])
        # Raw output is in [-1,1]; horizontal flip reverses u-axis → negate u channel
        uv_pred_flip = uv_pred_flip.flip(dims=[4]).clone()
        uv_pred_flip[:, :, 0, :, :] *= -1
        uv_pred_flip[:, :, 0, :, :] += 2 * 0.0075  # sub-pixel centering correction (ref L149)
        uv_avg = (uv_pred + uv_pred_flip) / 2
        uv_map = torch.clamp((uv_avg + 1) / 2, 0, 1)  # [-1,1] → [0,1]

        # Reshape to [1, 2, H, W]: remove seq dim
        if uv_map.dim() == 5:
            uv_map = uv_map.squeeze(1)  # [B, 2, H, W] already channels-first

        # Normal prediction
        normal_output, _ = self._normal_model.net(batch)
        normal_output_flip, _ = self._normal_model.net(batch_flip)

        normal_pred = normal_output['normals']        # [B, seq, 3, H, W]
        normal_pred_flip = normal_output_flip['normals']

        # Flip-back: reverse spatial W dim; negate x-normal (horizontal mirror)
        normal_pred_flip = normal_pred_flip.flip(dims=[4]).clone()
        normal_pred_flip[:, :, 0, :, :] *= -1
        normal_avg = (normal_pred + normal_pred_flip) / 2

        # Remove seq dim → [B, 3, H, W]
        if normal_avg.dim() == 5:
            normal_avg = normal_avg.squeeze(1)

        # Match pixel3dmm reference exactly:
        # network_inference.py saves normals as PNG in [0,1] after coordinate swap,
        # tracker.py loads them back via (pixel/255 - 0.5)*2 → [-1,1].
        # We replicate the full save→load chain in one pass:
        #   1. L2 normalize raw output
        #   2. Map [-1,1] → [0,1]: (x+1)/2
        #   3. Coordinate swap in [0,1] space: [x, 1-z, 1-y]
        #   4. Map back [0,1] → [-1,1]: x*2-1
        # This is equivalent to what the tracker sees in batch['normals'].
        normal_unit = F.normalize(normal_avg, dim=1)                     # [B, 3, H, W]
        normal_01 = torch.clamp((normal_unit + 1) / 2, 0, 1)            # [-1,1] → [0,1]
        normal_swapped_01 = torch.stack([
            normal_01[:, 0],          # x stays
            1 - normal_01[:, 2],      # 1-z in [0,1] space
            1 - normal_01[:, 1],      # 1-y in [0,1] space
        ], dim=1)
        # Map back to [-1,1] (matching tracker.py L1543: (pixel/255 - 0.5)*2)
        normal_map = normal_swapped_01 * 2 - 1

        return uv_map, normal_map

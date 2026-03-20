"""
MICA inference: RetinaFace 5-point → ArcFace alignment → shape code extraction.

Uses submodules/MICA as the canonical MICA implementation.
All processing in memory — no intermediate JPEG files.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Stage1Config
from .paths import (
    ensure_file_matches,
    ensure_inspect_getargspec,
    ensure_numpy_legacy_aliases,
)
from faceforge._paths import PROJECT_ROOT


# ArcFace standard 5-point target coordinates (112×112 space)
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """ArcFace standard alignment using similarity transform.

    Args:
        img: BGR uint8 [H, W, 3]
        landmark: [5, 2] five-point landmarks
        image_size: Output size (default 112)

    Returns:
        Aligned image [image_size, image_size, 3] uint8
    """
    from skimage import transform as trans

    assert landmark.shape == (5, 2)
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class MICAInference:
    """MICA model for 3D face shape reconstruction."""

    def __init__(self, config: Stage1Config):
        self.device = config.device
        self.model_path = config.mica_model_path

        # Add MICA submodule to path
        project_root = str(PROJECT_ROOT)
        mica_root = os.path.join(project_root, 'submodules', 'MICA')
        if mica_root not in sys.path:
            sys.path.insert(0, mica_root)

        ensure_inspect_getargspec()
        ensure_numpy_legacy_aliases()

        from configs.config import get_cfg_defaults
        from micalib.models.mica import MICA

        cfg = get_cfg_defaults()
        cfg.model.testing = True
        cfg.pretrained_model_path = os.path.join(project_root, self.model_path)

        # Override FLAME model paths — MICA config resolves these relative to
        # submodules/MICA/ but generic_model.pkl is at data/pretrained/FLAME2020/
        flame_pkl = os.path.join(project_root, config.flame_model_path)
        cfg.model.flame_model_path = flame_pkl

        # MICA's Masking module hardcodes path relative to its own directory:
        #   ROOT_DIR/data/FLAME2020/generic_model.pkl
        # Mirror the file into the expected location so it can find the file.
        mica_flame_pkl = os.path.join(mica_root, 'data', 'FLAME2020', 'generic_model.pkl')
        if os.path.exists(flame_pkl):
            ensure_file_matches(os.path.abspath(flame_pkl), mica_flame_pkl)

        # Patch torch.load for PyTorch >= 2.6 compatibility and cross-device loading
        original_load = torch.load
        def patched_load(f, *args, **kwargs):
            kwargs.setdefault('weights_only', False)
            kwargs.setdefault('map_location', self.device)
            return original_load(f, *args, **kwargs)
        torch.load = patched_load
        try:
            self.model = MICA(cfg, self.device)
            self.model.eval()
        finally:
            torch.load = original_load

    def get_arcface_input(self, image_bgr: np.ndarray, kps_5pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prepare ArcFace input from BGR image and 5-point landmarks.

        Args:
            image_bgr: BGR uint8 [H, W, 3]
            kps_5pt: [5, 2] RetinaFace native 5-point landmarks

        Returns:
            (blob [1, 3, 112, 112], aligned_img [112, 112, 3] uint8)
        """
        input_mean = 127.5
        input_std = 127.5
        aimg = _norm_crop(image_bgr, landmark=kps_5pt)
        blob = cv2.dnn.blobFromImages(
            [aimg],
            1.0 / input_std,
            (112, 112),
            (input_mean, input_mean, input_mean),
            swapRB=True,
        )
        return blob[0], aimg

    @torch.no_grad()
    def encode(self, image_rgb: np.ndarray, arcface_blob: np.ndarray) -> dict:
        """Run MICA encoding: image + ArcFace blob → shape code + vertices.

        Args:
            image_rgb: RGB float32 or uint8 [H, W, 3]
            arcface_blob: [1, 3, 112, 112] normalized ArcFace input

        Returns:
            dict with 'shape_code' [1, 300], 'vertices' [1, 5023, 3],
            'arcface_feat' [1, 512]
        """
        # Prepare image: RGB → [0,1] → resize 224 → tensor [1, 3, 224, 224]
        img = image_rgb.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = cv2.resize(img, (224, 224))
        image_tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # ArcFace blob: [3, 112, 112] → [1, 3, 112, 112]
        arcface_tensor = torch.tensor(arcface_blob).unsqueeze(0).to(self.device)

        # MICA encode + decode
        codedict = self.model.encode(image_tensor, arcface_tensor)
        opdict = self.model.decode(codedict)

        return {
            'shape_code': opdict['pred_shape_code'].detach().cpu(),             # [1, 300]
            'vertices': opdict['pred_canonical_shape_vertices'].detach().cpu(),  # [1, 5023, 3]
            'arcface_feat': codedict['arcface'].detach().cpu(),                  # [1, 512]
        }

    @torch.no_grad()
    def run(self, image_bgr: np.ndarray, retinaface_kps: np.ndarray) -> dict:
        """Complete MICA pipeline: 5-point → ArcFace align → encode.

        Args:
            image_bgr: BGR uint8 [H, W, 3]
            retinaface_kps: [5, 2] RetinaFace native 5-point landmarks

        Returns:
            dict with shape_code, vertices, arcface_feat, arcface_img
        """
        arcface_blob, arcface_img = self.get_arcface_input(image_bgr, retinaface_kps)

        # Convert BGR to RGB for MICA image input
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.encode(image_rgb, arcface_blob)
        result['arcface_img'] = arcface_img  # [112, 112, 3] BGR uint8
        return result

"""
DECA inference: expression, pose, texture, lighting extraction.

Uses decalib submodule.
"""

import os
import sys

import numpy as np
import torch

from .config import Stage1Config
from faceforge._paths import PROJECT_ROOT


class DECAInference:
    """DECA model for FLAME expression/pose/texture/lighting estimation."""

    def __init__(self, config: Stage1Config):
        self.device = config.device

        # Add project root to sys.path so decalib's internal imports work.
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from submodules.decalib.deca import DECA
        from submodules.decalib.deca_utils.config import cfg as deca_cfg

        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = True

        # Resolve model paths
        models_dir = os.path.join(project_root, 'data', 'pretrained')

        # DECA pretrained checkpoint
        deca_cfg.pretrained_modelpath = os.path.join(models_dir, 'deca_model.tar')

        # DECA-specific texture/displacement files
        deca_cfg.model.dense_template_path = os.path.join(models_dir, 'texture_data_256.npy')
        deca_cfg.model.fixed_displacement_path = os.path.join(models_dir, 'fixed_displacement_256.npy')
        deca_cfg.model.face_mask_path = os.path.join(models_dir, 'uv_face_mask.png')
        deca_cfg.model.face_eye_mask_path = os.path.join(models_dir, 'uv_face_eye_mask.png')
        deca_cfg.model.mean_tex_path = os.path.join(models_dir, 'mean_texture.jpg')
        deca_cfg.model.tex_path = os.path.join(models_dir, 'FLAME_albedo_from_BFM.npz')

        # FLAME model files
        flame_dir = os.path.join(models_dir, 'FLAME2020')
        deca_cfg.model.topology_path = os.path.join(flame_dir, 'head_template.obj')
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(flame_dir, 'landmark_embedding.npy')
        deca_cfg.model.flame_model_path = os.path.join(
            project_root, config.flame_model_path,
        )

        # Patch torch.load for PyTorch >= 2.6 compatibility and cross-device loading
        original_load = torch.load
        def patched_load(f, *args, **kwargs):
            kwargs.setdefault('weights_only', False)
            kwargs.setdefault('map_location', self.device)
            return original_load(f, *args, **kwargs)
        torch.load = patched_load
        try:
            self.model = DECA(config=deca_cfg, device=self.device)
        finally:
            torch.load = original_load

    @torch.no_grad()
    def run(self, image_rgb: np.ndarray) -> dict:
        """Run DECA inference on an RGB image.

        DECA internally uses FAN for face detection and cropping.

        Args:
            image_rgb: RGB uint8 [H, W, 3]

        Returns:
            dict with:
                'exp': [1, 50] expression coefficients
                'pose': [1, 6] pose (head_pose[:3] + jaw_pose[3:])
                'tex': [1, 50] texture coefficients
                'light': [1, 9, 3] SH lighting coefficients
                'deca_crop': np.ndarray 224x224 crop (for debug)
        """
        # DECA crop_image expects uint8 RGB, normalizes to [0,1] internally
        image_tensor = self.model.crop_image(image_rgb).to(self.device)
        deca_dict = self.model.encode(image_tensor[None])

        # Extract crop image for debug (convert tensor back to numpy)
        deca_crop = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        return {
            'exp': deca_dict['exp'].detach().cpu(),      # [1, 50]
            'pose': deca_dict['pose'].detach().cpu(),     # [1, 6]
            'tex': deca_dict['tex'].detach().cpu(),       # [1, 50]
            'light': deca_dict['light'].detach().cpu(),   # [1, 9, 3]
            'deca_crop': deca_crop,                        # [224, 224, 3] uint8
        }

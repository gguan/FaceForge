"""
DECA inference: expression, pose, texture, lighting extraction.

Uses decalib submodule.
"""

import os
import sys

import numpy as np
import torch

from .config import Stage1Config
from .paths import get_deca_topology_path
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
        deca_cfg.model.topology_path = get_deca_topology_path()
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
                'cam': [1, 3] orthographic camera [scale, tx, ty]
                'tex': [1, 50] texture coefficients
                'light': [1, 9, 3] SH lighting coefficients
                'deca_crop': np.ndarray 224x224 crop (for debug)
                'crop_tform': [3, 3] similarity transform (original → 224 crop)
        """
        # Crop image and capture the similarity transform (original → 224 crop)
        image_tensor, crop_tform = self._crop_image_with_tform(image_rgb)
        image_tensor = image_tensor.to(self.device)
        deca_dict = self.model.encode(image_tensor[None])

        # Extract crop image for debug (convert tensor back to numpy)
        deca_crop = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        return {
            'exp': deca_dict['exp'].detach().cpu(),      # [1, 50]
            'pose': deca_dict['pose'].detach().cpu(),     # [1, 6]
            'cam': deca_dict['cam'].detach().cpu(),       # [1, 3]
            'tex': deca_dict['tex'].detach().cpu(),       # [1, 50]
            'light': deca_dict['light'].detach().cpu(),   # [1, 9, 3]
            'deca_crop': deca_crop,                        # [224, 224, 3] uint8
            'crop_tform': crop_tform,                      # [3, 3] ndarray
        }

    def _crop_image_with_tform(self, image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Crop face for DECA and return the similarity transform matrix.

        Replicates DECA's crop_image logic but also returns the 3x3
        similarity transform (original image → 224x224 crop).

        Args:
            image: RGB uint8 [H, W, 3]

        Returns:
            (image_tensor [3, 224, 224], tform_matrix [3, 3])
        """
        from skimage.transform import estimate_transform, warp

        h, w, _ = image.shape
        bbox, bbox_type = self.model.face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left, right, top, bottom = 0, h - 1, 0, w - 1
        else:
            left, right = bbox[0], bbox[2]
            top, bottom = bbox[1], bbox[3]

        old_size, center = self.model.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size * 1.25)
        src_pts = np.array([
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ])
        DST_PTS = np.array([[0, 0], [0, 223], [223, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        dst_image = warp(image / 255.0, tform.inverse, output_shape=(224, 224))
        dst_image = dst_image.transpose(2, 0, 1)
        return torch.tensor(dst_image).float(), tform.params.astype(np.float64)

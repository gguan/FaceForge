"""Loss aggregator with stage dispatch matching pixel3dmm tracker.

Coordinates all loss functions and handles stage-dependent activation.
pixel3dmm stages:
  coarse_lmk  = optimize_camera first half  → landmark only (all 68, ×3000)
  coarse_uv   = optimize_camera second half → UV only
  medium      = optimize_color              → all losses
"""

import os
import sys

import torch
import torch.nn as nn

from faceforge._paths import PROJECT_ROOT
from ..config import Stage2Config
from .landmark import landmark_loss_p3m, landmark_loss_camera
from .regularization import regularization_loss
from .normal import normal_loss
from .contour import contour_loss
from .silhouette import silhouette_loss
from .photometric import photometric_loss
from .identity import IdentityLoss
from .prdl import PRDLLoss
from .mirror import load_mirror_order, mirror_symmetry_loss


class LossAggregator:
    """Computes weighted total loss with stage-dependent activation."""

    def __init__(self, config: Stage2Config, flame_model,
                 mica_init_shape: torch.Tensor,
                 arcface_model=None):
        self.config = config
        self.device = torch.device(config.device)

        # MICA initial shape for regularization
        self.mica_init_shape = mica_init_shape.detach().to(self.device)

        # UV loss (initialized per-image via setup_uv_loss)
        self.uv_losses = []

        # Mirror symmetry vertex reindexing
        self._mirror_order = None
        if config.w_mirror > 0:
            try:
                self._mirror_order = load_mirror_order(
                    config.pixel3dmm_code_base).to(self.device)
            except FileNotFoundError:
                pass

        # Identity loss (skip if w=0)
        self.identity_loss = None
        if config.w_identity > 0 and arcface_model is not None:
            self.identity_loss = IdentityLoss(arcface_model)

        # PRDL or baseline (skip if w=0)
        self.prdl_loss = None
        if config.use_prdl and config.w_prdl > 0:
            self.prdl_loss = PRDLLoss(flame_model, device=self.device)

        # Boundary vertex indices for contour loss
        try:
            self.boundary_indices = flame_model.get_region_indices('boundary')
        except KeyError:
            self.boundary_indices = flame_model.get_region_indices('face')

    def setup_uv_loss(self, flame_model, n_images: int):
        """Initialize UV loss instances for each image."""
        code_base = str(PROJECT_ROOT / self.config.pixel3dmm_code_base)
        src_path = f'{code_base}/src'
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        os.environ.setdefault('PIXEL3DMM_CODE_BASE', code_base)
        os.environ.setdefault('PIXEL3DMM_PREPROCESSED_DATA', str(PROJECT_ROOT / 'output'))
        os.environ.setdefault('PIXEL3DMM_TRACKING_OUTPUT', str(PROJECT_ROOT / 'output'))

        from pixel3dmm.tracking.losses import UVLoss as P3DMMUVLoss

        self.uv_losses = []
        for _ in range(n_images):
            uv_loss = P3DMMUVLoss(
                stricter_mask=False,
                delta_uv=self.config.uv_delta_coarse,
                dist_uv=self.config.uv_dist_coarse,
            )
            self.uv_losses.append(uv_loss)

    def compute(self, stage: str, image_idx: int = 0, **kwargs) -> tuple[torch.Tensor, dict]:
        """Compute total loss for a stage.

        Args:
            stage: one of 'coarse_lmk', 'coarse_uv', 'medium', 'fine_pca', 'fine_detail'
            image_idx: which image's UV loss to use
            **kwargs: loss inputs

        Returns:
            (total_loss, {loss_name: value} dict)
        """
        c = self.config
        losses = {}
        image_size = c.render_size

        # Principal point regularization (always active, pixel3dmm tracker.py L647)
        pp = kwargs.get('principal_point')
        if pp is not None:
            losses['pp_reg'] = pp.pow(2).sum()

        # === coarse_lmk: optimize_camera first half → all 68 lmks × 3000 ===
        if stage == 'coarse_lmk':
            losses['landmark'] = landmark_loss_camera(
                kwargs['pred_lmks_68'], kwargs['target_lmks_68'],
                image_size, lmk_mask=kwargs.get('lmk_mask'))

        # === coarse_uv: optimize_camera second half → UV only ===
        # pixel3dmm uses weight 1000 in camera phase (tracker.py L655), not 2000
        if stage == 'coarse_uv' and image_idx < len(self.uv_losses):
            losses['uv'] = 1000.0 * self.uv_losses[image_idx].compute_loss(
                kwargs['projected_vertices'],
                is_visible_verts_idx=kwargs.get('visibility_mask'),
            )

        # === medium+ = optimize_color: all losses ===
        if stage in ('medium', 'fine_pca', 'fine_detail'):
            # Landmark (pixel3dmm style: eye contour + eye_closure + iris)
            losses['landmark'] = landmark_loss_p3m(
                kwargs['pred_lmks_68'], kwargs['target_lmks_68'],
                image_size,
                lmk_mask=kwargs.get('lmk_mask'),
                proj_vertices=kwargs.get('projected_vertices'),
                target_iris_left=kwargs.get('target_iris_left'),
                target_iris_right=kwargs.get('target_iris_right'),
                iris_mask_left=kwargs.get('iris_mask_left'),
                iris_mask_right=kwargs.get('iris_mask_right'),
                w_lmks=c.w_lmks,
                w_lmks_lid=c.w_lmks_lid,
                w_lmks_iris=c.w_lmks_iris,
            )

            # UV
            if c.w_pixel3dmm_uv > 0 and image_idx < len(self.uv_losses):
                losses['uv'] = c.w_pixel3dmm_uv * self.uv_losses[image_idx].compute_loss(
                    kwargs['projected_vertices'],
                    is_visible_verts_idx=kwargs.get('visibility_mask'),
                )

            # Normal
            if c.w_normal > 0 and 'rendered_normals' in kwargs and 'predicted_normals' in kwargs:
                losses['normal'] = c.w_normal * normal_loss(
                    kwargs['rendered_normals'], kwargs['predicted_normals'],
                    kwargs['face_mask'],
                    cam_rotation=kwargs.get('cam_rotation'),
                    eye_mask=kwargs.get('eye_mask'),
                    delta_n=c.normal_delta_threshold,
                    eye_dilate_kernel=c.normal_eye_dilate_kernel,
                    use_l2=c.normal_l2,
                )

            # Silhouette
            if c.w_sil > 0 and 'rendered_mask' in kwargs:
                losses['sil'] = c.w_sil * silhouette_loss(
                    kwargs['rendered_mask'],
                    kwargs.get('sil_fg_mask', kwargs['face_mask']),
                    valid_bg_mask=kwargs.get('sil_valid_bg_mask'),
                )

            # Mirror symmetry (pixel3dmm: 5000 × mirror_loss)
            if c.w_mirror > 0 and self._mirror_order is not None:
                verts_canonical = kwargs.get('vertices_canonical')
                if verts_canonical is not None:
                    losses['mirror'] = c.w_mirror * mirror_symmetry_loss(
                        verts_canonical, self._mirror_order)

            # Regularization (base: shape, expression, jaw)
            has_reg = (c.w_reg_shape_to_mica > 0 or c.w_reg_shape_to_zero > 0 or
                       c.w_reg_expression > 0 or c.w_reg_jaw > 0 or
                       c.w_reg_sh_mono > 0 or c.w_reg_tex_tv > 0)
            if has_reg:
                losses['reg'] = regularization_loss(
                    kwargs['shape'], kwargs['expression'],
                    kwargs['jaw_6d'], kwargs['lighting'],
                    self.mica_init_shape, c,
                    texture_displacement=kwargs.get('texture_displacement'),
                )

            # pixel3dmm eye/neck regularization (tracker.py L914-919)
            eyes_6d = kwargs.get('eyes_6d')
            neck_6d = kwargs.get('neck_6d')
            if eyes_6d is not None:
                from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
                I6d = matrix_to_rotation_6d(
                    torch.eye(3, device=self.device).unsqueeze(0))
                right_eye, left_eye = eyes_6d[:, :6], eyes_6d[:, 6:]
                # Eye symmetry: pixel3dmm L914  reg/sym = 0.1
                losses['eye_sym'] = c.w_eye_sym * (right_eye - left_eye).pow(2).sum(-1).mean()
                # Eye-to-identity: pixel3dmm L918-919  reg/eye_left=0.01, reg/eye_right=0.01
                losses['eye_l_reg'] = c.w_eye_reg * (I6d - left_eye).pow(2).sum(-1).mean()
                losses['eye_r_reg'] = c.w_eye_reg * (I6d - right_eye).pow(2).sum(-1).mean()
            if neck_6d is not None:
                from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
                I6d = matrix_to_rotation_6d(
                    torch.eye(3, device=self.device).unsqueeze(0))
                # Neck-to-identity: pixel3dmm L916  reg/neck = w_neck (0.1)
                losses['neck_reg'] = c.w_neck * (I6d - neck_6d).pow(2).sum(-1).mean()

            # Contour / PRDL (A/B switch)
            if c.use_prdl and c.w_prdl > 0 and self.prdl_loss is not None:
                losses['prdl'] = c.w_prdl * self.prdl_loss.compute(
                    kwargs['projected_vertices'],
                    kwargs.get('face_segmentation', kwargs['face_mask']),
                    image_size=image_size,
                )
            elif c.w_contour > 0:
                losses['contour'] = c.w_contour * contour_loss(
                    kwargs['projected_vertices'],
                    self.boundary_indices,
                    kwargs['face_mask'],
                    image_size=image_size,
                )

        # === Fine+ losses (require rendered output) ===
        if stage in ('fine_pca', 'fine_detail') and 'rendered_image' in kwargs:
            if c.w_photometric > 0:
                losses['photo'] = c.w_photometric * photometric_loss(
                    kwargs['rendered_image'], kwargs['target_image'],
                    kwargs['face_mask'],
                    region_weight_map=kwargs.get('region_weight_map'),
                )
            if c.w_identity > 0 and self.identity_loss is not None and 'target_arcface_feat' in kwargs:
                losses['identity'] = c.w_identity * self.identity_loss(
                    kwargs['rendered_image'], kwargs['target_arcface_feat'],
                )

        # === Total ===
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        total = sum(losses.values())

        # NaN guard
        if self.config.use_nan_guard:
            total = torch.nan_to_num(total, nan=0.0, posinf=1e5)

        return total, {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

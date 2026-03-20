"""Loss aggregator with 5-stage dispatch and A/B switching.

Coordinates all loss functions and handles stage-dependent activation.
"""

import torch
import torch.nn as nn

from ..config import Stage2Config
from .landmark import landmark_loss, build_landmark_weights
from .regularization import regularization_loss
from .pixel3dmm_uv import Pixel3DMMUVLoss
from .normal import normal_loss
from .contour import contour_loss
from .silhouette import silhouette_loss
from .photometric import photometric_loss
from .identity import IdentityLoss
from .prdl import PRDLLoss


class LossAggregator:
    """Computes weighted total loss with stage-dependent activation."""

    def __init__(self, config: Stage2Config, flame_model,
                 mica_init_shape: torch.Tensor,
                 arcface_model=None):
        self.config = config
        self.device = torch.device(config.device)

        # Landmark weights
        self.lmk_weights = build_landmark_weights(self.device, config.nose_landmark_weight)

        # MICA initial shape for regularization
        self.mica_init_shape = mica_init_shape.detach().to(self.device)

        # UV loss (initialized per-image via setup_uv_loss)
        self.uv_losses: list[Pixel3DMMUVLoss] = []

        # Identity loss
        self.identity_loss = None
        if arcface_model is not None:
            self.identity_loss = IdentityLoss(arcface_model)

        # PRDL or baseline
        self.prdl_loss = None
        if config.use_prdl:
            self.prdl_loss = PRDLLoss(flame_model, device=self.device)

        # Boundary vertex indices for contour loss
        try:
            self.boundary_indices = flame_model.get_region_indices('boundary')
        except KeyError:
            # Fallback to face region boundary
            self.boundary_indices = flame_model.get_region_indices('face')

    def setup_uv_loss(self, flame_model, n_images: int):
        """Initialize UV loss instances for each image."""
        self.uv_losses = []
        for _ in range(n_images):
            uv_loss = Pixel3DMMUVLoss(
                flame_model.flame_uv_coords,
                flame_model.uv_valid_verts,
                self.device,
            )
            self.uv_losses.append(uv_loss)

    def compute(self, stage: str, image_idx: int = 0, **kwargs) -> tuple[torch.Tensor, dict]:
        """Compute total loss for a stage.

        Args:
            stage: one of 'coarse_lmk', 'coarse_uv', 'medium', 'fine_pca', 'fine_detail'
            image_idx: which image's UV loss to use
            **kwargs: loss inputs (pred_lmks_68, pred_lmks_eyes, target_lmks_68,
                      target_lmks_eyes, projected_vertices, proj_depths,
                      rendered_image, rendered_normals, rendered_mask,
                      target_image, predicted_normals, face_mask, eye_mask,
                      face_segmentation, target_arcface_feat, visibility_mask,
                      shape, expression, jaw_pose, lighting, texture_displacement)

        Returns:
            (total_loss, {loss_name: value} dict)
        """
        c = self.config
        losses = {}
        image_size = c.render_size

        use_l2 = stage in ('coarse_lmk', 'coarse_uv')

        # === Landmark (all stages) ===
        losses['landmark'] = c.w_landmark * landmark_loss(
            kwargs['pred_lmks_68'], kwargs['pred_lmks_eyes'],
            kwargs['target_lmks_68'], kwargs['target_lmks_eyes'],
            self.lmk_weights, image_size,
            use_l2=use_l2,
            visibility_mask_68=kwargs.get('visibility_mask_68'),
        )

        # === UV (coarse_uv+) ===
        if stage != 'coarse_lmk' and image_idx < len(self.uv_losses):
            losses['uv'] = c.w_pixel3dmm_uv * self.uv_losses[image_idx].compute_loss(
                kwargs['projected_vertices'],
                image_size=image_size,
                visibility_mask=kwargs.get('visibility_mask'),
            )

        # === Medium+ losses ===
        if stage in ('medium', 'fine_pca', 'fine_detail'):
            # Normal
            if 'rendered_normals' in kwargs and 'predicted_normals' in kwargs:
                losses['normal'] = c.w_normal * normal_loss(
                    kwargs['rendered_normals'], kwargs['predicted_normals'],
                    kwargs['face_mask'],
                    eye_mask=kwargs.get('eye_mask'),
                    delta_n=c.normal_delta_threshold,
                    eye_dilate_kernel=c.normal_eye_dilate_kernel,
                )

            # Silhouette
            if 'rendered_mask' in kwargs:
                losses['sil'] = c.w_sil * silhouette_loss(
                    kwargs['rendered_mask'], kwargs['face_mask'],
                )

            # Contour / PRDL (A/B switch)
            if c.use_prdl and self.prdl_loss is not None:
                losses['prdl'] = c.w_prdl * self.prdl_loss.compute(
                    kwargs['projected_vertices'],
                    kwargs.get('face_segmentation', kwargs['face_mask']),
                    image_size=image_size,
                )
            else:
                losses['contour'] = c.w_contour * contour_loss(
                    kwargs['projected_vertices'],
                    self.boundary_indices,
                    kwargs['face_mask'],
                    image_size=image_size,
                )

            # Regularization
            losses['reg'] = regularization_loss(
                kwargs['shape'], kwargs['expression'],
                kwargs['jaw_pose'], kwargs['lighting'],
                self.mica_init_shape, c,
                texture_displacement=kwargs.get('texture_displacement'),
            )

        # === Fine+ losses (require rendered output) ===
        if stage in ('fine_pca', 'fine_detail') and 'rendered_image' in kwargs:
            # Photometric
            losses['photo'] = c.w_photometric * photometric_loss(
                kwargs['rendered_image'], kwargs['target_image'],
                kwargs['face_mask'],
                region_weight_map=kwargs.get('region_weight_map'),
            )

            # Identity
            if self.identity_loss is not None and 'target_arcface_feat' in kwargs:
                losses['identity'] = c.w_identity * self.identity_loss(
                    kwargs['rendered_image'], kwargs['target_arcface_feat'],
                )

        # === Total ===
        total = sum(losses.values())

        # NaN guard
        if self.config.use_nan_guard:
            total = torch.nan_to_num(total, nan=0.0, posinf=1e5)

        return total, {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

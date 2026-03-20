"""5-stage coarse-to-fine optimizer with dynamic LR decay and early stopping.

Reference: pixel3dmm tracker.py L1076-1098 (LR decay),
           VHAP tracker.py (optimize_stage / get_train_parameters)
"""

from typing import Callable

import torch

from .config import Stage2Config
from .data_types import SharedParams, PerImageParams


# Which parameters are optimized at each stage
STAGE_PARAMS = {
    'coarse_lmk': {
        'shared': [],
        'per_image': ['head_pose', 'jaw_pose', 'translation'],
        'shared_if_single': ['focal_length'],
    },
    'coarse_uv': {
        'shared': [],
        'per_image': ['head_pose', 'jaw_pose', 'translation'],
        'shared_if_single': ['focal_length'],
    },
    'medium': {
        'shared': ['shape'],
        'per_image': ['expression', 'head_pose', 'jaw_pose', 'translation'],
        'shared_if_single': ['focal_length'],
    },
    'fine_pca': {
        'shared': ['shape', 'texture'],
        'per_image': ['expression', 'head_pose', 'jaw_pose', 'translation', 'lighting'],
        'shared_if_single': ['focal_length'],
    },
    'fine_detail': {
        'shared': ['shape', 'texture'],
        'per_image': ['expression', 'head_pose', 'jaw_pose', 'translation', 'lighting'],
        'shared_if_single': ['focal_length'],
    },
}

# Large params decay faster (pose/camera), small params decay slower (shape/exp)
LARGE_PARAMS = {'head_pose', 'jaw_pose', 'translation', 'focal_length'}

STAGE_ORDER = ['coarse_lmk', 'coarse_uv', 'medium', 'fine_pca', 'fine_detail']


class Stage2Optimizer:
    """Manages 5-stage optimization with per-stage param groups, LR decay, and early stopping."""

    def __init__(self, config: Stage2Config):
        self.config = config
        self.lr_map = {
            'shape': config.lr_shape,
            'expression': config.lr_expression,
            'head_pose': config.lr_head_pose,
            'jaw_pose': config.lr_jaw_pose,
            'focal_length': config.lr_focal,
            'translation': config.lr_translation,
            'texture': config.lr_texture,
            'lighting': config.lr_lighting,
        }

    def get_steps(self, stage: str) -> int:
        steps_map = {
            'coarse_lmk': self.config.coarse_lmk_steps,
            'coarse_uv': self.config.coarse_uv_steps,
            'medium': self.config.medium_steps,
            'fine_pca': self.config.fine_pca_steps,
            'fine_detail': self.config.fine_detail_steps,
        }
        return steps_map[stage]

    def create_optimizer(self, stage: str,
                         shared_params: SharedParams,
                         per_image_params: list[PerImageParams],
                         selected_indices: list[int],
                         lr_scale: float = 1.0) -> torch.optim.Adam:
        """Create Adam optimizer for a stage.

        Args:
            stage: stage name
            shared_params: SharedParams instance
            per_image_params: list of PerImageParams
            selected_indices: which images to optimize (for multi-image)
            lr_scale: global LR multiplier (e.g., 0.1 for global phase)

        Returns:
            Adam optimizer
        """
        param_groups = []
        stage_def = STAGE_PARAMS[stage]

        # Shared params
        for name in stage_def['shared'] + stage_def['shared_if_single']:
            tensor = getattr(shared_params, name, None)
            if tensor is None:
                continue
            tensor.requires_grad_(True)
            lr = self.lr_map.get(name, 0.001) * lr_scale
            param_groups.append({'params': [tensor], 'lr': lr, 'name': name, 'initial_lr': lr})

        # Per-image params
        for idx in selected_indices:
            pip = per_image_params[idx]
            for name in stage_def['per_image']:
                tensor = getattr(pip, name, None)
                if tensor is None:
                    continue
                tensor.requires_grad_(True)
                lr = self.lr_map.get(name, 0.001) * lr_scale
                param_groups.append({'params': [tensor], 'lr': lr, 'name': name, 'initial_lr': lr})

        # Freeze everything else
        self._freeze_non_optimized(stage, shared_params, per_image_params)

        if not param_groups:
            # Dummy optimizer if nothing to optimize
            dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            param_groups = [{'params': [dummy], 'lr': 1e-10}]

        return torch.optim.Adam(param_groups)

    def _freeze_non_optimized(self, stage: str,
                              shared_params: SharedParams,
                              per_image_params: list[PerImageParams]):
        """Set requires_grad=False for params not in this stage."""
        stage_def = STAGE_PARAMS[stage]
        active_shared = set(stage_def['shared'] + stage_def['shared_if_single'])
        active_per = set(stage_def['per_image'])

        for name in ['shape', 'texture', 'focal_length']:
            tensor = getattr(shared_params, name, None)
            if tensor is not None:
                tensor.requires_grad_(name in active_shared)

        for pip in per_image_params:
            for name in ['expression', 'head_pose', 'jaw_pose', 'translation', 'lighting']:
                tensor = getattr(pip, name, None)
                if tensor is not None:
                    tensor.requires_grad_(name in active_per)

    @staticmethod
    def adjust_lr(optimizer: torch.optim.Optimizer, progress: float):
        """Dynamic LR decay based on progress within a stage.

        Ref: pixel3dmm tracker.py L1076-1098

        Args:
            optimizer: Adam optimizer
            progress: step / total_steps (0.0 to 1.0)
        """
        for group in optimizer.param_groups:
            base_lr = group.get('initial_lr', group['lr'])
            name = group.get('name', '')

            if name in LARGE_PARAMS:
                if progress > 0.9:
                    scale = 0.01
                elif progress > 0.75:
                    scale = 0.02
                elif progress > 0.5:
                    scale = 0.1
                else:
                    scale = 1.0
            else:
                if progress > 0.9:
                    scale = 0.2
                elif progress > 0.75:
                    scale = 0.5
                elif progress > 0.5:
                    scale = 0.5
                else:
                    scale = 1.0

            group['lr'] = base_lr * scale

    @staticmethod
    def check_early_stopping(loss_history: list[float], window: int = 10,
                             delta: float = 1e-5) -> bool:
        """Check if optimization has stagnated.

        Ref: pixel3dmm tracker.py L1169-1176

        Returns:
            True if should stop early
        """
        if len(loss_history) < 2 * window:
            return False
        recent = sum(loss_history[-window:]) / window
        prev = sum(loss_history[-2 * window:-window]) / window
        return (prev - recent) < delta

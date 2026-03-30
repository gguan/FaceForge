"""P3M Pipeline: faithful reimplementation of pixel3dmm's tracking flow.

Replicates pixel3dmm tracker.py's optimize_camera() + optimize_color() + joint phase
as a baseline for comparison with our Stage 2 implementation.

Key differences from Stage 2:
  1. FLAME called in canonical mode (no rot_params), rotation applied externally.
  2. R stored as 6D rotation (same as pixel3dmm).
  3. Landmark loss: eye contours only (36:48) × w_lmks × 5.
  4. LR decay only in joint phase (at 50%, 75%, 90%).
  5. Early stopping in per-frame phase only.
  6. UV loss: pixel3dmm UVLoss with finish_stage1() before joint.

Ref: pixel3dmm/src/pixel3dmm/tracking/tracker.py
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from faceforge.stage1.data_types import Stage1Output
from faceforge.stage2.data_types import (
    PreprocessedData, PerImageParams, Stage2Output,
)
from faceforge.stage2.camera import build_intrinsics, project_points
from faceforge.stage2.renderer import NvdiffrastRenderer
from faceforge.stage2.pixel3dmm_inference import Pixel3DMMInference
from .config import P3MConfig


# ---------------------------------------------------------------------------
# Internal parameter store
# ---------------------------------------------------------------------------

@dataclass
class _ImageParams:
    """Per-image optimization parameters (pixel3dmm style)."""
    R_6d: torch.Tensor           # [1, 6]  6D head rotation
    t: torch.Tensor              # [1, 3]  translation
    exp: torch.Tensor            # [1, 100] expression
    jaw_6d: torch.Tensor         # [1, 6]  6D jaw rotation
    focal_length: torch.Tensor   # [1, 1]
    principal_point: torch.Tensor  # [1, 2]


# ---------------------------------------------------------------------------
# pixel3dmm-compatible FLAME config adapter
# ---------------------------------------------------------------------------

class _P3DMMConfig:
    use_flame2023 = False
    num_shape_params = 300
    num_exp_params = 100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _aa_to_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle [B, 3] → 6D rotation [B, 6]."""
    from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
    R = _batch_rodrigues(axis_angle)
    return matrix_to_rotation_6d(R)


def _batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Axis-angle [N, 3] → rotation matrices [N, 3, 3]."""
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def _p3m_lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask):
    """Reimplementation of pixel3dmm's util.lmk_loss (L2, normalized by image size).

    Ref: pixel3dmm/src/pixel3dmm/tracking/util.py L96-102
    """
    h, w = image_size
    size = torch.tensor([1.0 / w, 1.0 / h],
                        device=opt_lmks.device, dtype=opt_lmks.dtype)
    opt_scaled = opt_lmks * size
    tgt_scaled = target_lmks * size
    diff = (opt_scaled - tgt_scaled) ** 2
    return (diff * lmk_mask).mean()


def _compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex normals from mesh.

    Args:
        vertices: [B, V, 3]
        faces: [F, 3] long indices

    Returns:
        normals: [B, V, 3] unit normals
    """
    B, V, _ = vertices.shape
    F_ = faces.shape[0]
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [B, F, 3]
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):
        vertex_normals.scatter_add_(
            1,
            faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, F_, 3),
            face_normals,
        )
    return F.normalize(vertex_normals, dim=-1)


# ---------------------------------------------------------------------------
# P3M Pipeline
# ---------------------------------------------------------------------------

class P3MPipeline:
    """Faithful reimplementation of pixel3dmm tracking for baseline comparison.

    Replicates pixel3dmm's full tracking flow:
      1. optimize_camera(): 500 steps — first half landmark-only, second half UV-only
      2. optimize_color():  500 steps — all losses active, early stopping
      3. Joint phase:       finish_stage1() then 5000 steps with LR decay (multi-image only)

    Key conventions (matching pixel3dmm):
      - FLAME called in canonical mode (no internal head rotation)
      - Head rotation applied EXTERNALLY: verts_world = R @ verts_can + t
      - Rotation stored as 6D rotation representation
      - Normal loss: R^T @ rendered_normals (to convert world → canonical space)
      - Landmark loss: eye contours only (indices 36:48), weight w_lmks × 5
    """

    def __init__(self, config: P3MConfig | None = None,
                 visualizer=None):
        self.config = config or P3MConfig()
        self.device = torch.device(self.config.device)
        self.visualizer = visualizer

        # Configure pixel3dmm asset paths
        from faceforge.stage2._pixel3dmm_paths import configure_pixel3dmm_paths
        configure_pixel3dmm_paths(self.config)

        # pixel3dmm FLAME (canonical mode - no internal rotation)
        from pixel3dmm.tracking.flame.FLAME import FLAME as P3DFLAME
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
        self.flame = P3DFLAME(_P3DMMConfig()).to(self.device)

        # Fixed eye/neck/eyelid parameters (not optimized)
        i6d = matrix_to_rotation_6d(torch.eye(3, device=self.device).unsqueeze(0))
        self._eyes_fixed = i6d.repeat(1, 2)
        self._neck_fixed = i6d
        self._eyelids_fixed = torch.zeros(1, 2, device=self.device)

        # Renderer
        self.renderer = NvdiffrastRenderer(
            image_size=self.config.render_size,
            use_opengl=self.config.use_opengl,
        )

        # pixel3dmm inference (UV + normals)
        self._pixel3dmm = None

    @classmethod
    def from_stage2_config(cls, s2_config, visualizer=None) -> 'P3MPipeline':
        """Create P3MPipeline from a Stage2Config, copying shared asset paths.

        Allows stageP3M to be used as a drop-in replacement for Stage2Pipeline
        without duplicating model path settings in the YAML config.

        P3M-specific optimization hyperparameters (iters, lr, loss weights)
        keep their default values from P3MConfig; only asset paths and
        device/render settings are inherited from s2_config.
        """
        _PATH_FIELDS = (
            'flame_model_path', 'flame_masks_path', 'flame_lmk_embedding_path',
            'flame_uv_coords_path', 'flame_uv_valid_verts_path',
            'pixel3dmm_uv_ckpt', 'pixel3dmm_normal_ckpt', 'pixel3dmm_code_base',
        )
        _COMMON_FIELDS = ('render_size', 'use_opengl', 'device')

        overrides = {f: getattr(s2_config, f)
                     for f in _PATH_FIELDS + _COMMON_FIELDS
                     if hasattr(s2_config, f)}
        for f in ('output_dir', 'save_debug'):
            if hasattr(s2_config, f):
                overrides[f] = getattr(s2_config, f)

        cfg = P3MConfig(**overrides)
        return cls(cfg, visualizer=visualizer)

        # pixel3dmm FLAME (canonical mode — no internal rotation)
        from pixel3dmm.tracking.flame.FLAME import FLAME as P3DFLAME
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
        self.flame = P3DFLAME(_P3DMMConfig()).to(self.device)

        # Fixed eye/neck/eyelid parameters (not optimized)
        I6D = matrix_to_rotation_6d(torch.eye(3, device=self.device).unsqueeze(0))  # [1, 6]
        self._eyes_fixed = I6D.repeat(1, 2)   # [1, 12]
        self._neck_fixed = I6D                  # [1, 6]
        self._eyelids_fixed = torch.zeros(1, 2, device=self.device)

        # Renderer
        self.renderer = NvdiffrastRenderer(
            image_size=self.config.render_size,
            use_opengl=self.config.use_opengl,
        )

        # pixel3dmm inference (UV + normals)
        self._pixel3dmm = None

    @property
    def pixel3dmm(self):
        if self._pixel3dmm is None:
            self._pixel3dmm = Pixel3DMMInference(self.config)
        return self._pixel3dmm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, stage1_outputs: list[Stage1Output]) -> Stage2Output:
        """Run pixel3dmm-style tracking.

        Args:
            stage1_outputs: list of N Stage1Output instances (same subject)

        Returns:
            Stage2Output with optimized FLAME parameters
        """
        if len(stage1_outputs) == 1:
            return self._run_single(stage1_outputs[0])
        else:
            return self._run_multi(stage1_outputs)

    # ------------------------------------------------------------------
    # Single-image path
    # ------------------------------------------------------------------

    def _run_single(self, s1out: Stage1Output) -> Stage2Output:
        """Single image: optimize_camera + optimize_color (shape not frozen)."""
        prep = self._preprocess(s1out)
        shape = s1out.shape.detach().to(self.device).requires_grad_(True)
        mica_shape = shape.detach().clone()
        pip = self._init_image_params(s1out)

        # UV loss (one per image)
        from pixel3dmm.tracking.losses import UVLoss
        uv_loss_fn = UVLoss(
            stricter_mask=False,
            delta_uv=self.config.delta_uv_coarse,
            dist_uv=self.config.dist_uv_coarse,
        )

        # Phase 1: optimize_camera (500 steps)
        # Ref: tracker.py run() L1658: optimize_camera(batch, steps=500, is_first_frame=True)
        self._optimize_camera(shape, pip, prep, uv_loss_fn, is_first_frame=True)

        # Phase 2: optimize_color (500 steps, shape unfrozen for single image)
        # Ref: tracker.py run() L1662: freeze_id=not(MAX_STEPS==1) = False → don't freeze
        loss_history = self._optimize_color(
            shape, [pip], [prep], [uv_loss_fn],
            mica_shape=mica_shape,
            is_joint=False, is_first_step=True, freeze_shape=False,
        )

        if self.visualizer is not None:
            self.visualizer.save_stage_progression()

        return self._build_output(shape, [pip], loss_history)

    # ------------------------------------------------------------------
    # Multi-image path
    # ------------------------------------------------------------------

    def _run_multi(self, s1outs: list[Stage1Output]) -> Stage2Output:
        """Multi-image: sequential per-frame then joint optimization.

        Ref: tracker.py run() L1633-1735
        """
        N = len(s1outs)
        preps = [self._preprocess(s1out) for s1out in s1outs]

        # Shared shape (median of all MICA shapes)
        all_shapes = torch.stack([s.shape for s in s1outs])
        shape = torch.median(all_shapes, dim=0).values.detach().to(self.device).requires_grad_(True)
        mica_shape = shape.detach().clone()

        # Per-image params
        pips = [self._init_image_params(s1out) for s1out in s1outs]

        # Per-image UV loss
        from pixel3dmm.tracking.losses import UVLoss
        uv_loss_fns = []
        for i in range(N):
            uv = UVLoss(
                stricter_mask=False,
                delta_uv=self.config.delta_uv_coarse,
                dist_uv=self.config.dist_uv_coarse,
            )
            uv_loss_fns.append(uv)

        all_loss_history = {}

        # Phase 1: Per-frame sequential optimization
        # Ref: tracker.py L1658-1674: for each timestep, optimize_camera + optimize_color(freeze_id=True)
        print("\n<<<<<<<< P3M: PER-FRAME PHASE >>>>>>>>")
        for i in range(N):
            print(f"  Frame {i+1}/{N}")
            is_first_frame = (i == 0)
            steps = self.config.camera_steps if is_first_frame else self.config.camera_steps_extra
            self._optimize_camera(
                shape, pips[i], preps[i], uv_loss_fns[i],
                is_first_frame=is_first_frame, steps=steps,
            )
            h = self._optimize_color(
                shape, [pips[i]], [preps[i]], [uv_loss_fns[i]],
                mica_shape=mica_shape,
                is_joint=False, is_first_step=is_first_frame,
                freeze_shape=True,  # freeze shape in per-frame (multi-image)
            )
            for k, v in h.items():
                all_loss_history.setdefault(f'frame{i}_{k}', []).extend(v)

            # Ref: tracker.py L1675: self.uv_loss_fn.is_next()
            uv_loss_fns[i].is_next()

            if self.visualizer is not None:
                self._snapshot(f'frame{i}', shape, pips[i], preps[i])

        # Tighten UV thresholds before joint phase
        # Ref: tracker.py L1720: if config.uv_map_super > 0: self.uv_loss_fn.finish_stage1()
        for uv_fn in uv_loss_fns:
            uv_fn.finish_stage1(
                delta_uv_fine=self.config.delta_uv_fine,
                dist_uv_fine=self.config.dist_uv_fine,
            )
            self._prime_joint_uv_loss(uv_fn)

        # Phase 2: Joint global optimization
        # Ref: tracker.py L1735: optimize_color(is_joint=True) with iters=global_iters
        if N > 1:
            print("\n<<<<<<<< P3M: JOINT GLOBAL PHASE >>>>>>>>")
            h = self._optimize_color(
                shape, pips, preps, uv_loss_fns,
                mica_shape=mica_shape,
                is_joint=True, is_first_step=False, freeze_shape=False,
                total_steps=self.config.global_iters,
            )
            all_loss_history.update(h)

        if self.visualizer is not None:
            self.visualizer.save_stage_progression()

        return self._build_output(shape, pips, all_loss_history)

    @staticmethod
    def _prime_joint_uv_loss(uv_loss_fn) -> None:
        """Reuse cached single-frame UV correspondences for joint optimization.

        In this pipeline each ``UVLoss`` instance belongs to exactly one image.
        After ``finish_stage1()``, pixel3dmm stores that image's cached
        correspondences in ``verts_2d`` and resets ``gt_2_verts`` via
        ``is_next()``. Joint mode should reuse the cached tensor instead of
        calling ``compute_corresp()`` again, which would try to append to a
        tensor-backed cache.
        """
        cached = getattr(uv_loss_fn, 'verts_2d', None)
        if uv_loss_fn.gt_2_verts is None and isinstance(cached, torch.Tensor):
            uv_loss_fn.gt_2_verts = cached[:1]

    # ------------------------------------------------------------------
    # optimize_camera
    # ------------------------------------------------------------------

    def _optimize_camera(self, shape, pip: _ImageParams, prep: PreprocessedData,
                         uv_loss_fn, is_first_frame: bool = True, steps: int = None):
        """Replicate pixel3dmm's optimize_camera().

        Ref: tracker.py L582-685
          - First half: landmark loss only (eye contours, 36:48) × 3000
          - At step 0: compute_corresp(uv_map)
          - Second half: UV loss × 1000
          - Principal point regularization throughout
          - Scheduler: StepLR(step_size=steps*0.75, gamma=0.1)
        """
        steps = steps if steps is not None else self.config.camera_steps
        h, w = self.config.render_size, self.config.render_size
        cfg = self.config

        # Parameters to optimize
        pip.R_6d.requires_grad_(True)
        pip.t.requires_grad_(True)
        pip.principal_point.requires_grad_(True)

        params = [
            {'params': [pip.t], 'lr': cfg.lr_t},
            {'params': [pip.R_6d], 'lr': cfg.lr_R},
            {'params': [pip.principal_point], 'lr': cfg.lr_pp},
        ]
        if is_first_frame:
            pip.focal_length.requires_grad_(True)
            params.append({'params': [pip.focal_length], 'lr': cfg.lr_focal})
        else:
            pip.focal_length.requires_grad_(False)

        # Freeze params not being optimized
        shape.requires_grad_(False)
        pip.exp.requires_grad_(False)
        pip.jaw_6d.requires_grad_(False)

        optimizer = torch.optim.Adam(params)
        # Ref: tracker.py L604: scheduler StepLR(step_size=int(steps*0.75), gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(steps * 0.75), gamma=0.1)

        for k in range(steps):
            # --- FLAME forward (canonical, no rot_params) ---
            verts_can, lmk68_can = self._flame_canonical(shape, pip)

            # Apply external rotation
            from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
            R_mat = rotation_6d_to_matrix(pip.R_6d)   # [1, 3, 3]
            lmk68_world = torch.einsum('bny,bxy->bnx', lmk68_can, R_mat) + pip.t.unsqueeze(1)
            verts_world = torch.einsum('bny,bxy->bnx', verts_can, R_mat) + pip.t.unsqueeze(1)

            # Project to screen space (OpenCV convention, z > 0 in front)
            K = build_intrinsics(pip.focal_length, pip.principal_point, self.config.render_size)
            RT = torch.eye(4, device=self.device, dtype=pip.t.dtype).unsqueeze(0)
            lmk68_screen, _ = project_points(lmk68_world, K, RT, self.config.render_size)
            verts_screen_2d, verts_depth = project_points(verts_world, K, RT, self.config.render_size)
            # UVLoss.compute_loss expects [B, N, 3]: (x_screen, y_screen, z_cam)
            # Ref: pixel3dmm tracker.py project_points_screen_space output format
            verts_screen = torch.cat([verts_screen_2d, verts_depth.unsqueeze(-1)], dim=-1)

            losses = {}
            losses['pp_reg'] = torch.sum(pip.principal_point ** 2)

            # First half: landmark loss (eye contours 36:48)
            # Ref: tracker.py L648: if k <= steps // 2: losses['lmk68'] = ... * 3000
            if k <= steps // 2:
                lmk_mask = torch.ones(1, 12, 2, device=self.device)
                losses['lmk_eye'] = cfg.w_lmks_camera * _p3m_lmk_loss(
                    lmk68_screen[:, 36:48, :2],
                    prep.target_lmks_68[:, 36:48, :].to(self.device),
                    [h, w], lmk_mask,
                )

            # UV correspondence (computed once at step 0)
            # Ref: tracker.py L651: if k == 0: self.uv_loss_fn.compute_corresp(uv_map)
            if k == 0:
                uv_loss_fn.compute_corresp(prep.pixel3dmm_uv)

            # Second half: UV loss
            # Ref: tracker.py L653: if k > steps // 2: losses['uv_loss'] = uv_loss * 1000
            if k > steps // 2:
                uv_loss = uv_loss_fn.compute_loss(verts_screen)
                losses['uv_loss'] = cfg.w_uv_camera * uv_loss

            total = sum(losses.values())
            if cfg.use_nan_guard:
                total = torch.nan_to_num(total, nan=0.0, posinf=1e5)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Clamp focal length
            with torch.no_grad():
                pip.focal_length.clamp_(1.5, 8.0)

    # ------------------------------------------------------------------
    # optimize_color
    # ------------------------------------------------------------------

    def _optimize_color(self, shape: torch.Tensor,
                        pips: list[_ImageParams],
                        preps: list[PreprocessedData],
                        uv_loss_fns: list,
                        mica_shape: torch.Tensor,
                        is_joint: bool,
                        is_first_step: bool,
                        freeze_shape: bool = False,
                        total_steps: int = None) -> dict:
        """Replicate pixel3dmm's optimize_color().

        Ref: tracker.py L1009-1200
          - All losses active from step 0
          - Early stopping: per-frame only (non-joint, not first step)
          - LR decay: joint phase only, at 50%, 75%, 90%
          - Losses: eye landmarks, UV, silhouette, normal, regularization
        """
        cfg = self.config
        N = len(pips)
        steps = total_steps if total_steps is not None else cfg.iters
        h, w = cfg.render_size, cfg.render_size
        I6D = self._get_I6D()

        # Build parameter groups
        # Ref: tracker.py clone_params_keyframes_all (per-frame params)
        param_groups = []
        for i, pip in enumerate(pips):
            pip.R_6d.requires_grad_(True)
            pip.t.requires_grad_(True)
            pip.exp.requires_grad_(True)
            pip.jaw_6d.requires_grad_(True)
            pip.focal_length.requires_grad_(not freeze_shape or is_joint)
            pip.principal_point.requires_grad_(True)
            param_groups += [
                {'params': [pip.R_6d], 'lr': cfg.lr_R, 'name': 'R'},
                {'params': [pip.t], 'lr': cfg.lr_t, 'name': 't'},
                {'params': [pip.exp], 'lr': cfg.lr_exp, 'name': 'exp'},
                {'params': [pip.jaw_6d], 'lr': cfg.lr_jaw, 'name': 'jaw'},
                {'params': [pip.focal_length], 'lr': cfg.lr_focal, 'name': 'focal'},
                {'params': [pip.principal_point], 'lr': cfg.lr_pp, 'name': 'pp'},
            ]

        if not freeze_shape:
            shape.requires_grad_(True)
            # Ref: tracker.py optimize_color() joint: optimizer_id optimizes shape separately
            # For simplicity, include shape in main optimizer
            param_groups.append({'params': [shape], 'lr': cfg.lr_shape, 'name': 'shape'})
        else:
            shape.requires_grad_(False)

        optimizer = torch.optim.Adam(param_groups)

        # Tracking for early stopping and LR decay
        past_k_steps = np.array([100.0 for _ in range(cfg.early_stopping_window)])
        prev_loss = None
        loss_history = []

        iterator = tqdm(range(steps), desc='P3M optimize_color', leave=True, miniters=100)

        for p in iterator:
            # LR decay (joint phase only)
            # Ref: tracker.py optimize_color() L1075-1101
            if is_joint:
                # Ref: tracker.py L1079–1101: decay at 50%, 75%, 90% of total steps
                if p == int(steps * 0.5):
                    self._decay_lr(optimizer, {'R', 't', 'jaw'}, '0.5')
                elif p == int(steps * 0.75):
                    self._decay_lr(optimizer, {'R', 't', 'jaw'}, '0.75')
                elif p == int(steps * 0.9):
                    self._decay_lr(optimizer, {'R', 't', 'jaw'}, '0.9')

            # Sample batch (joint: random; per-frame: all)
            if is_joint:
                batch_size = min(N, cfg.batch_size)
                selected = random.sample(range(N), batch_size)
            else:
                selected = list(range(N))

            total_loss_val = torch.tensor(0.0, device=self.device)

            for i in selected:
                pip = pips[i]
                prep = preps[i]
                uv_fn = uv_loss_fns[i]

                loss_i = self._compute_color_loss(
                    shape, pip, prep, uv_fn, mica_shape, I6D,
                    is_joint=is_joint, is_first_step=is_first_step, step=p,
                )
                total_loss_val = total_loss_val + loss_i

            total_loss_val = total_loss_val / len(selected)

            if cfg.use_nan_guard:
                total_loss_val = torch.nan_to_num(total_loss_val, nan=0.0, posinf=1e5)

            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for pip in pips:
                    pip.focal_length.clamp_(1.5, 8.0)

            loss_val = total_loss_val.item()
            loss_history.append(loss_val)
            iterator.set_description(f'P3M loss={loss_val:.4f}')

            # Early stopping (per-frame only, not first step)
            # Ref: tracker.py L1181-1190
            if not is_joint and not is_first_step:
                if prev_loss is not None:
                    past_k_steps[p % cfg.early_stopping_window] = abs(loss_val - prev_loss)
                if p > cfg.early_stopping_window and np.mean(past_k_steps) < cfg.early_stopping_delta:
                    print(f'  Early stopping at step {p}')
                    break
            prev_loss = loss_val

        return {'optimize_color': loss_history}

    def _compute_color_loss(self, shape, pip: _ImageParams, prep: PreprocessedData,
                            uv_loss_fn, mica_shape: torch.Tensor, I6D: torch.Tensor,
                            is_joint: bool, is_first_step: bool, step: int) -> torch.Tensor:
        """Compute all optimize_color losses for one image.

        Ref: tracker.py opt_pre() + opt_post()
        """
        from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
        from torchvision.transforms.functional import gaussian_blur
        cfg = self.config
        h, w = cfg.render_size, cfg.render_size

        # --- FLAME forward (canonical, no rot_params) ---
        verts_can, lmk68_can = self._flame_canonical(shape, pip)
        R_mat = rotation_6d_to_matrix(pip.R_6d)   # [1, 3, 3]

        # Apply external rotation + translation
        lmk68_world = torch.einsum('bny,bxy->bnx', lmk68_can, R_mat) + pip.t.unsqueeze(1)
        verts_world = torch.einsum('bny,bxy->bnx', verts_can, R_mat) + pip.t.unsqueeze(1)

        # Project to screen space
        K = build_intrinsics(pip.focal_length, pip.principal_point, cfg.render_size)
        RT = torch.eye(4, device=self.device, dtype=pip.t.dtype).unsqueeze(0)
        lmk68_screen, _ = project_points(lmk68_world, K, RT, cfg.render_size)
        verts_screen_2d, verts_depth = project_points(verts_world, K, RT, cfg.render_size)
        # UVLoss.compute_loss expects [B, N, 3]: (x_screen, y_screen, z_cam)
        verts_screen = torch.cat([verts_screen_2d, verts_depth.unsqueeze(-1)], dim=-1)

        losses = {}

        # --- Landmark loss: eye contours only (36:48) × w_lmks × 5 ---
        # Ref: tracker.py opt_pre() L876: * config.w_lmks * 5
        lmk_mask = torch.ones(1, 12, 2, device=self.device)
        losses['lmk_eye'] = cfg.w_lmks * 5.0 * _p3m_lmk_loss(
            lmk68_screen[:, 36:48, :2],
            prep.target_lmks_68[:, 36:48, :].to(self.device),
            [h, w], lmk_mask,
        )

        # --- Regularization ---
        # Ref: tracker.py opt_pre() L895-907
        losses['reg_exp'] = torch.sum(pip.exp ** 2, dim=-1).mean() * cfg.w_exp
        losses['reg_shape'] = torch.sum((shape - mica_shape) ** 2, dim=-1).mean() * cfg.w_shape
        losses['reg_shape_general'] = torch.sum(shape ** 2, dim=-1).mean() * cfg.w_shape_general
        losses['reg_jaw'] = torch.sum((I6D - pip.jaw_6d) ** 2, dim=-1).mean() * cfg.w_jaw
        losses['reg_pp'] = torch.sum(pip.principal_point ** 2, dim=-1).mean()

        # --- Render ---
        vertex_normals_world = _compute_vertex_normals(verts_world, self.flame.faces.long())
        render_out = {}
        if self.renderer.available:
            render_out = self.renderer.render(
                verts_world, self.flame.faces, K, RT, vertex_normals_world,
            )

        if render_out:
            # --- Visibility mask ---
            vis_mask = None
            if 'depth' in render_out:
                vis_mask = self.renderer.compute_visibility_mask(
                    render_out['depth'], verts_screen_2d, verts_depth,
                    cfg.render_size, 0.01,
                )

            # --- UV loss ---
            # Ref: tracker.py opt_post() L972-982
            if uv_loss_fn.gt_2_verts is None:
                uv_loss_fn.compute_corresp(prep.pixel3dmm_uv)
            uv_loss = uv_loss_fn.compute_loss(
                verts_screen, is_visible_verts_idx=vis_mask)
            losses['uv'] = cfg.uv_map_super * uv_loss

            # --- Silhouette loss ---
            # Ref: tracker.py opt_post() L939-952
            # Apply sil at all steps (is_joint or not is_first_step)
            if 'mask' in render_out:
                # valid_bg: background+neck classes (seg <= 1 is background)
                seg = prep.face_segmentation.to(self.device)
                valid_bg = (seg <= 1).float()  # [1, H, W]
                fg_mask = prep.face_mask.to(self.device)  # [1, H, W]
                rendered_mask = render_out['mask']
                if rendered_mask.dim() == 4:
                    rendered_mask = rendered_mask[..., 0]  # [1, H, W]
                sil_scale = 1.0 if (is_joint or not is_first_step) else 0.1
                losses['sil'] = sil_scale * cfg.sil_super * (
                    valid_bg * (fg_mask - rendered_mask).abs()
                ).mean()

            # --- Normal loss ---
            # Ref: tracker.py opt_post() L955-977
            if 'normal' in render_out and prep.pixel3dmm_normals is not None:
                rendered_normals = render_out['normal']  # [1, 3, H, W] or [1, H, W, 3]
                # Ensure [1, 3, H, W]
                if rendered_normals.dim() == 4 and rendered_normals.shape[-1] == 3:
                    rendered_normals = rendered_normals.permute(0, 3, 1, 2)

                # Undo external rotation: R^T @ rendered_normals (world → canonical)
                # Ref: tracker.py L984: pred_normals_flame_space = einsum('bxy,bxhw->byhw', rot_mat, pred_normals)
                pred_normals_can = torch.einsum(
                    'bxy,bxhw->byhw', R_mat, rendered_normals)

                predicted_normals = prep.pixel3dmm_normals.to(self.device)  # [1, 3, H, W]
                if predicted_normals.dim() == 4 and predicted_normals.shape[-1] == 3:
                    predicted_normals = predicted_normals.permute(0, 3, 1, 2)

                # Dilate eye mask to exclude eye region from normal loss
                # Ref: tracker.py L963-967
                if 'eye_mask' in render_out:
                    eye_mask = render_out['eye_mask']
                    if eye_mask.dim() == 4:
                        eye_mask = eye_mask[..., 0]
                    dilated_eye = (gaussian_blur(
                        eye_mask.float().unsqueeze(1),
                        [cfg.normal_mask_ksize, cfg.normal_mask_ksize],
                        sigma=[cfg.normal_mask_ksize, cfg.normal_mask_ksize],
                    ) > 0).float().squeeze(1)
                    dilated_eye_mask = (1 - dilated_eye).unsqueeze(1)
                else:
                    dilated_eye_mask = torch.ones(1, 1, cfg.render_size, cfg.render_size,
                                                  device=self.device)

                # Normal mask (face region)
                normal_mask = prep.face_mask.to(self.device).unsqueeze(1)  # [1, 1, H, W]

                # Threshold filter
                l_map = predicted_normals - pred_normals_can
                valid = ((l_map.abs().sum(dim=1) / 3) < cfg.delta_n).unsqueeze(1).float()

                normal_loss = (l_map * valid * normal_mask * dilated_eye_mask).abs().mean()
                losses['normal'] = cfg.normal_super * normal_loss

        total = sum(losses.values())
        return total

    # ------------------------------------------------------------------
    # FLAME canonical forward
    # ------------------------------------------------------------------

    def _flame_canonical(self, shape: torch.Tensor,
                         pip: _ImageParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Call pixel3dmm FLAME in canonical mode (no internal rotation).

        Returns:
            verts_can: [B, V, 3] canonical vertices
            lmk68_can: [B, 68, 3] canonical landmarks
        """
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d, rotation_6d_to_matrix
        B = shape.shape[0]

        # rot_params_lmk_shift = inv(R_head) for correct contour landmark selection
        # Ref: tracker.py opt_pre() L853-854
        R_mat = rotation_6d_to_matrix(pip.R_6d)  # [1, 3, 3]
        lmk_shift = matrix_to_rotation_6d(R_mat.transpose(-1, -2))  # [1, 6]

        I3 = torch.eye(3, device=self.device, dtype=shape.dtype).unsqueeze(0).expand(B, -1, -1)

        verts_can, lmk68, _, _, _ = self.flame(
            shape_params=shape,
            cameras=I3,                  # = inv(R_base[0]) = I (R_base[0]=I)
            # rot_params NOT passed → canonical output (no internal rotation)
            jaw_pose_params=pip.jaw_6d.expand(B, -1),
            expression_params=pip.exp.expand(B, -1),
            rot_params_lmk_shift=lmk_shift.expand(B, -1),
            eye_pose_params=self._eyes_fixed.expand(B, -1).to(self.device),
            neck_pose_params=self._neck_fixed.expand(B, -1).to(self.device),
            eyelid_params=self._eyelids_fixed.expand(B, -1).to(self.device),
        )
        return verts_can, lmk68

    # ------------------------------------------------------------------
    # LR decay helper
    # ------------------------------------------------------------------

    def _decay_lr(self, optimizer, large_param_names: set, milestone: str):
        """Apply LR decay at a given milestone (called at the right step from the loop).

        Ref: tracker.py optimize_color() L1076-1101
          milestone '0.5'  → large /10, small /2
          milestone '0.75' → large /5,  small /2
          milestone '0.9'  → large /2,  small /5
        """
        for group in optimizer.param_groups:
            name = group.get('name', '')
            if name in large_param_names:
                # Large params (R, t, jaw): decay aggressively
                if milestone == '0.5':
                    group['lr'] /= 10
                elif milestone == '0.75':
                    group['lr'] /= 5
                elif milestone == '0.9':
                    group['lr'] /= 2
            else:
                # Small params (shape, exp, etc.): decay more gradually
                if milestone == '0.5':
                    group['lr'] /= 2
                elif milestone == '0.75':
                    group['lr'] /= 2
                elif milestone == '0.9':
                    group['lr'] /= 5

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _snapshot(self, stage: str, shape: torch.Tensor,
                  pip: _ImageParams, prep: PreprocessedData):
        """Save visualization snapshot for one image/stage."""
        if self.visualizer is None:
            return
        from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix

        with torch.no_grad():
            verts_can, lmk68_can = self._flame_canonical(shape, pip)
            R_mat = rotation_6d_to_matrix(pip.R_6d)
            lmk68_world = torch.einsum('bny,bxy->bnx', lmk68_can, R_mat) + pip.t.unsqueeze(1)
            verts_world = torch.einsum('bny,bxy->bnx', verts_can, R_mat) + pip.t.unsqueeze(1)

            K = build_intrinsics(pip.focal_length, pip.principal_point, self.config.render_size)
            RT = torch.eye(4, device=self.device, dtype=pip.t.dtype).unsqueeze(0)
            lmk68_screen, _ = project_points(lmk68_world, K, RT, self.config.render_size)

            rendered_image = None
            rendered_mask = None
            if self.renderer.available:
                normals_world = _compute_vertex_normals(verts_world, self.flame.faces.long())
                render_out = self.renderer.render(verts_world, self.flame.faces, K, RT, normals_world)
                rendered_image = render_out.get('image')
                rendered_mask = render_out.get('mask')

        self.visualizer.save_stage_snapshot(
            stage=stage,
            target_image=prep.target_image,
            target_lmks_68=prep.target_lmks_68,
            pred_lmks_68=lmk68_screen,
            rendered_image=rendered_image,
            rendered_mask=rendered_mask,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, s1out: Stage1Output) -> PreprocessedData:
        """Run pixel3dmm inference and build PreprocessedData."""
        uv_map, normal_map = self.pixel3dmm.predict(s1out.aligned_image, s1out.face_mask)

        seg_raw = s1out.face_mask
        if seg_raw.dim() == 2:
            seg_raw = seg_raw.unsqueeze(0)
        face_segmentation = seg_raw.long()

        seg = seg_raw.squeeze(0)
        binary_mask = ((seg > 0) & (seg <= 13)).float()

        d = self.device
        return PreprocessedData(
            pixel3dmm_uv=uv_map,
            pixel3dmm_normals=normal_map,
            face_mask=binary_mask.unsqueeze(0).to(d),
            face_segmentation=face_segmentation.to(d),
            target_image=s1out.aligned_image.to(d),
            target_lmks_68=s1out.lmks_68.to(d),
            target_lmks_eyes=s1out.lmks_eyes.to(d),
            arcface_feat=s1out.arcface_feat.to(d),
        )

    def _init_image_params(self, s1out: Stage1Output) -> _ImageParams:
        """Initialize per-image optimization parameters from Stage 1 outputs."""
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d

        head_pose = s1out.head_pose.detach().to(self.device)
        jaw_pose = s1out.jaw_pose.detach().to(self.device)
        focal_length = s1out.focal_length.detach().to(self.device)

        R_6d = matrix_to_rotation_6d(_batch_rodrigues(head_pose))  # [1, 6]
        jaw_6d = matrix_to_rotation_6d(_batch_rodrigues(jaw_pose))  # [1, 6]

        # Initial translation: place face at depth z = focal_length * 0.4
        # OpenCV convention (z > 0 in front), consistent with our project_points
        z_init = float(focal_length) * 0.4
        t_init = torch.tensor([[0.0, 0.0, z_init]], device=self.device)

        return _ImageParams(
            R_6d=R_6d.requires_grad_(True),
            t=t_init.requires_grad_(True),
            exp=s1out.expression.detach().to(self.device).requires_grad_(True),
            jaw_6d=jaw_6d.requires_grad_(True),
            focal_length=focal_length.requires_grad_(True),
            principal_point=torch.zeros(1, 2, device=self.device).requires_grad_(True),
        )

    def _get_I6D(self) -> torch.Tensor:
        """6D representation of identity rotation."""
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
        return matrix_to_rotation_6d(torch.eye(3, device=self.device).unsqueeze(0))

    def _build_output(self, shape: torch.Tensor,
                      pips: list[_ImageParams],
                      loss_history: dict) -> Stage2Output:
        """Build Stage2Output from optimized parameters."""
        from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix

        with torch.no_grad():
            # Neutral expression mesh in canonical space
            pip0 = pips[0]
            R_mat = rotation_6d_to_matrix(pip0.R_6d)

            # Identity jaw for neutral mesh
            I6D = self._get_I6D()
            neutral_pip = _ImageParams(
                R_6d=pip0.R_6d,
                t=pip0.t,
                exp=torch.zeros(1, 100, device=self.device),
                jaw_6d=I6D,
                focal_length=pip0.focal_length,
                principal_point=pip0.principal_point,
            )
            verts_can, lmk68_can = self._flame_canonical(shape, neutral_pip)
            vertices = verts_can  # canonical neutral mesh

        # Build PerImageParams (axis-angle format for Stage 2 compatibility)
        per_image_out = []
        for pip in pips:
            R_mat = rotation_6d_to_matrix(pip.R_6d.detach())
            # Convert 6D back to axis-angle approximation
            # Note: exact inversion not needed — just for output structure compatibility
            head_pose_aa = torch.zeros(1, 3, device='cpu')
            jaw_6d_mat = rotation_6d_to_matrix(pip.jaw_6d.detach())
            jaw_pose_aa = torch.zeros(1, 3, device='cpu')

            per_image_out.append(PerImageParams(
                expression=pip.exp.detach().cpu(),
                head_pose=head_pose_aa,
                jaw_pose=jaw_pose_aa,
                translation=pip.t.detach().cpu(),
                lighting=torch.zeros(1, 9, 3),
            ))

        return Stage2Output(
            shape=shape.detach().cpu(),
            texture=torch.zeros(1, 50),
            focal_length=pips[0].focal_length.detach().cpu(),
            vertices=vertices.detach().cpu(),
            landmarks_3d=lmk68_can.detach().cpu(),
            per_image_params=per_image_out,
            loss_history=loss_history,
        )

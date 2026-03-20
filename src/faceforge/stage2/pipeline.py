"""Stage 2 Pipeline: Dense constraint optimization.

Takes Stage 1 outputs and iteratively optimizes FLAME parameters via
differentiable rendering + multi-source dense constraints.

Reference: src/faceforge/stage1/pipeline.py (run_single/run_multi pattern)
"""

import random
from typing import Optional

import torch
import torch.nn as nn

from faceforge._paths import PROJECT_ROOT
from faceforge.stage1.data_types import Stage1Output
from .config import Stage2Config
from .data_types import PreprocessedData, SharedParams, PerImageParams, Stage2Output
from .flame_model import FLAMEModel
from .camera import build_intrinsics, build_extrinsics, project_points, world_to_camera
from .renderer import NvdiffrastRenderer
from .pixel3dmm_inference import Pixel3DMMInference
from .losses.total import LossAggregator
from .optimizer import Stage2Optimizer, STAGE_ORDER
from .visualization import Stage2Visualizer


class Stage2Pipeline:
    """Stage 2: Dense constraint optimization pipeline."""

    def __init__(self, config: Stage2Config | None = None,
                 mica_model=None,
                 visualizer: Stage2Visualizer | None = None):
        """
        Args:
            config: Stage 2 configuration
            mica_model: Optional MICA model for ArcFace identity loss.
                        If None, identity loss is disabled.
            visualizer: Optional visualizer for saving debug images.
        """
        self.config = config or Stage2Config()
        self.device = torch.device(self.config.device)
        self.visualizer = visualizer

        # Load FLAME
        self.flame = FLAMEModel(self.config).to(self.device)

        # Load renderer
        self.renderer = NvdiffrastRenderer(
            image_size=self.config.render_size,
            use_opengl=self.config.use_opengl,
        )

        # Pixel3DMM (lazy loaded)
        self._pixel3dmm = None

        # ArcFace for identity loss
        self.arcface_model = None
        if mica_model is not None:
            try:
                self.arcface_model = mica_model.arcface
            except AttributeError:
                pass

        # Optimizer manager
        self.opt_manager = Stage2Optimizer(self.config)

    @property
    def pixel3dmm(self):
        if self._pixel3dmm is None:
            self._pixel3dmm = Pixel3DMMInference(self.config)
        return self._pixel3dmm

    def run(self, stage1_outputs: list[Stage1Output]) -> Stage2Output:
        """Main entry point.

        Args:
            stage1_outputs: list of N Stage1Output instances (same person)

        Returns:
            Stage2Output with optimized shape + neutral mesh
        """
        if len(stage1_outputs) == 1:
            return self._run_single(stage1_outputs[0])
        else:
            return self._run_multi(stage1_outputs)

    # ------------------------------------------------------------------
    # Single-image optimization
    # ------------------------------------------------------------------

    def _run_single(self, s1out: Stage1Output) -> Stage2Output:
        """Single image: straightforward 5-stage optimization."""
        preprocessed = [self._preprocess(s1out)]
        shared, per_image = self._init_params([s1out])

        # Setup UV correspondences
        loss_agg = self._create_loss_aggregator(shared)
        loss_agg.setup_uv_loss(self.flame, 1)
        loss_agg.uv_losses[0].compute_correspondences(
            preprocessed[0].pixel3dmm_uv,
            self.config.uv_delta_coarse, self.config.uv_dist_coarse,
        )

        loss_history = self._run_optimization(
            shared, per_image, preprocessed, loss_agg,
            selected_fn=lambda: [0], lr_scale=1.0,
        )

        return self._build_output(shared, per_image, loss_history)

    # ------------------------------------------------------------------
    # Multi-image joint optimization
    # ------------------------------------------------------------------

    def _run_multi(self, s1outs: list[Stage1Output]) -> Stage2Output:
        """Multi-image: sequential per-image → global joint optimization."""
        N = len(s1outs)
        preprocessed = [self._preprocess(s1out) for s1out in s1outs]
        shared, per_image = self._init_params(s1outs)

        loss_agg = self._create_loss_aggregator(shared)
        loss_agg.setup_uv_loss(self.flame, N)

        # Setup UV correspondences for each image
        for i in range(N):
            loss_agg.uv_losses[i].compute_correspondences(
                preprocessed[i].pixel3dmm_uv,
                self.config.uv_delta_coarse, self.config.uv_dist_coarse,
            )

        # Phase 1: Sequential (freeze shape, per-image optimization)
        shared.shape.requires_grad_(False)
        for i in range(N):
            self._run_sequential_stage(shared, per_image, preprocessed, loss_agg, i)
        shared.shape.requires_grad_(True)

        # Phase 2: Global joint optimization
        batch_size = min(N, self.config.multi_image_batch_size)

        def sample_fn():
            return random.sample(range(N), batch_size)

        loss_history = self._run_optimization(
            shared, per_image, preprocessed, loss_agg,
            selected_fn=sample_fn, lr_scale=self.config.global_lr_scale,
        )

        return self._build_output(shared, per_image, loss_history)

    def _run_sequential_stage(self, shared, per_image, preprocessed, loss_agg, idx):
        """Quick per-image alignment (coarse + medium subset)."""
        for stage in ['coarse_lmk', 'coarse_uv', 'medium']:
            steps = {
                'coarse_lmk': self.config.sequential_coarse_steps // 2,
                'coarse_uv': self.config.sequential_coarse_steps // 2,
                'medium': self.config.sequential_medium_steps,
            }[stage]

            optimizer = self.opt_manager.create_optimizer(
                stage, shared, per_image, [idx], lr_scale=1.0)

            for step in range(steps):
                optimizer.zero_grad()
                loss, _ = self._forward_step(
                    shared, per_image[idx], preprocessed[idx],
                    loss_agg, stage, idx)
                loss.backward()
                optimizer.step()
                self._apply_constraints(shared)

    # ------------------------------------------------------------------
    # Core optimization loop
    # ------------------------------------------------------------------

    def _run_optimization(self, shared, per_image, preprocessed, loss_agg,
                          selected_fn, lr_scale) -> dict:
        """Run 5-stage optimization loop."""
        all_loss_history = {}

        stages = list(STAGE_ORDER)
        if not self.config.enable_fine_detail:
            stages.remove('fine_detail')

        for stage in stages:
            total_steps = self.opt_manager.get_steps(stage)
            stage_history = []

            # Tighten UV thresholds at medium stage
            if stage == 'medium':
                for uv_loss in loss_agg.uv_losses:
                    uv_loss.tighten_thresholds(
                        self.config.uv_delta_fine, self.config.uv_dist_fine)

            selected = selected_fn()
            optimizer = self.opt_manager.create_optimizer(
                stage, shared, per_image, selected, lr_scale)

            for step in range(total_steps):
                # Re-sample images periodically (multi-image)
                if step > 0 and step % 50 == 0 and len(per_image) > 1:
                    selected = selected_fn()
                    optimizer = self.opt_manager.create_optimizer(
                        stage, shared, per_image, selected, lr_scale)

                optimizer.zero_grad()

                total_loss = torch.tensor(0.0, device=self.device)
                for i in selected:
                    loss_i, log_i = self._forward_step(
                        shared, per_image[i], preprocessed[i],
                        loss_agg, stage, i)
                    total_loss = total_loss + loss_i

                total_loss = total_loss / len(selected)
                total_loss.backward()
                optimizer.step()

                # LR decay
                progress = step / max(total_steps - 1, 1)
                self.opt_manager.adjust_lr(optimizer, progress)

                # Safety constraints
                self._apply_constraints(shared)

                stage_history.append(total_loss.item())

                # Early stopping
                if self.config.use_early_stopping:
                    if self.opt_manager.check_early_stopping(
                            stage_history,
                            self.config.early_stopping_window,
                            self.config.early_stopping_delta):
                        break

            all_loss_history[stage] = stage_history

            # --- Visualization: snapshot at end of each stage ---
            if self.visualizer is not None:
                self._snapshot_stage(stage, shared, per_image, preprocessed, selected)

        # Final progression image
        if self.visualizer is not None:
            self.visualizer.save_stage_progression()

        return all_loss_history

    def _forward_step(self, shared, per_img, preprocessed, loss_agg, stage, img_idx):
        """Single forward: FLAME → render → project → losses."""
        # FLAME forward
        vertices, lmks_68, lmks_eyes = self.flame(
            shared.shape, per_img.expression,
            per_img.head_pose, per_img.jaw_pose,
        )
        vertex_normals = self.flame.get_vertex_normals(vertices)

        # Camera: head_pose is already applied inside FLAME as global_orient,
        # so camera extrinsics use zero rotation (translation only).
        K = build_intrinsics(shared.focal_length,
                             torch.zeros(1, 2, device=self.device),
                             self.config.render_size)
        RT = build_extrinsics(torch.zeros_like(per_img.head_pose), per_img.translation)

        # Project landmarks
        proj_lmks_68, _ = project_points(lmks_68, K, RT, self.config.render_size)
        proj_lmks_eyes, _ = project_points(lmks_eyes, K, RT, self.config.render_size)
        proj_verts, proj_depths = project_points(vertices, K, RT, self.config.render_size)

        # Render
        render_out = {}
        need_render = stage in ('medium', 'fine_pca', 'fine_detail')
        if need_render and self.renderer.available:
            render_out = self.renderer.render(
                vertices, self.flame.faces, K, RT,
                vertex_normals,
                sh_coefficients=per_img.lighting,
            )

        # Visibility mask
        visibility_mask = None
        if self.config.use_occlusion_filter and 'depth' in render_out:
            visibility_mask = self.renderer.compute_visibility_mask(
                render_out['depth'], proj_verts, proj_depths,
                self.config.render_size, self.config.occlusion_depth_eps,
            )

        # Build loss kwargs
        loss_kwargs = dict(
            pred_lmks_68=proj_lmks_68,
            pred_lmks_eyes=proj_lmks_eyes,
            target_lmks_68=preprocessed.target_lmks_68.to(self.device),
            target_lmks_eyes=preprocessed.target_lmks_eyes.to(self.device),
            projected_vertices=proj_verts,
            proj_depths=proj_depths,
            visibility_mask=visibility_mask,
            face_mask=preprocessed.face_mask.to(self.device),  # [1, H, W] = [B, H, W]
            shape=shared.shape,
            expression=per_img.expression,
            jaw_pose=per_img.jaw_pose,
            lighting=per_img.lighting,
        )

        if render_out:
            loss_kwargs.update(
                rendered_image=render_out.get('image'),
                rendered_normals=render_out.get('normal'),
                rendered_mask=render_out.get('mask'),
                target_image=preprocessed.target_image.to(self.device),
                predicted_normals=preprocessed.pixel3dmm_normals.to(self.device),
                target_arcface_feat=preprocessed.arcface_feat.to(self.device),
                cam_rotation=RT[:, :3, :3],  # for normal_loss: rotate rendered normals back to FLAME space
            )

        return loss_agg.compute(stage, image_idx=img_idx, **loss_kwargs)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _snapshot_stage(self, stage: str, shared: SharedParams,
                        per_image: list[PerImageParams],
                        preprocessed: list[PreprocessedData],
                        selected: list[int]):
        """Take a visualization snapshot at the end of a stage.

        Uses the first selected image for the snapshot panels.
        """
        idx = selected[0]
        pip = per_image[idx]
        prep = preprocessed[idx]

        # FLAME forward with current params
        vertices, lmks_68, _ = self.flame(
            shared.shape, pip.expression, pip.head_pose, pip.jaw_pose,
        )
        vertex_normals = self.flame.get_vertex_normals(vertices)

        # Camera (same as _forward_step)
        K = build_intrinsics(shared.focal_length,
                             torch.zeros(1, 2, device=self.device),
                             self.config.render_size)
        RT = build_extrinsics(torch.zeros_like(pip.head_pose), pip.translation)

        # Project landmarks
        proj_lmks_68, _ = project_points(lmks_68, K, RT, self.config.render_size)

        # Render (if available)
        rendered_image = None
        rendered_mask = None
        if self.renderer.available:
            render_out = self.renderer.render(
                vertices, self.flame.faces, K, RT,
                vertex_normals,
                sh_coefficients=pip.lighting,
            )
            rendered_image = render_out.get('image')
            rendered_mask = render_out.get('mask')

        self.visualizer.save_stage_snapshot(
            stage=stage,
            target_image=prep.target_image,
            target_lmks_68=prep.target_lmks_68,
            pred_lmks_68=proj_lmks_68,
            rendered_image=rendered_image,
            rendered_mask=rendered_mask,
            image_idx=idx,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, s1out: Stage1Output) -> PreprocessedData:
        """Run Pixel3DMM and assemble PreprocessedData."""
        uv_map, normal_map = self.pixel3dmm.predict(
            s1out.aligned_image, s1out.face_mask)

        # Convert face_mask to binary (skin + facial features)
        seg = s1out.face_mask.squeeze(0) if s1out.face_mask.dim() > 2 else s1out.face_mask
        binary_mask = ((seg > 0) & (seg <= 13)).float()

        return PreprocessedData(
            pixel3dmm_uv=uv_map,
            pixel3dmm_normals=normal_map,
            face_mask=binary_mask.unsqueeze(0),
            target_image=s1out.aligned_image,
            target_lmks_68=s1out.lmks_68,
            target_lmks_eyes=s1out.lmks_eyes,
            arcface_feat=s1out.arcface_feat,
        )

    def _init_params(self, s1outs: list[Stage1Output]) -> tuple[SharedParams, list[PerImageParams]]:
        """Initialize optimization parameters from Stage 1 outputs."""
        N = len(s1outs)

        # Shared: shape = median of all, texture/focal from first
        if N > 1:
            all_shapes = torch.stack([s.shape for s in s1outs])
            median_shape = torch.median(all_shapes, dim=0).values
        else:
            median_shape = s1outs[0].shape.clone()

        shared = SharedParams(
            shape=median_shape.detach().to(self.device).requires_grad_(True),
            texture=s1outs[0].texture.detach().to(self.device).requires_grad_(True),
            focal_length=s1outs[0].focal_length.detach().to(self.device).requires_grad_(True),
        )

        # Per-image params
        # Estimate initial camera z so the face projects to ~50% of image width.
        # Formula: z_init = fx * face_width_m / target_face_px
        #        = (focal_norm * render_size) * 0.2 / (0.5 * render_size)
        #        = focal_norm * 0.4
        z_init = float(s1outs[0].focal_length) * 0.4
        per_image = []
        for s1out in s1outs:
            t_init = torch.tensor([[0.0, 0.0, z_init]], device=self.device)
            pip = PerImageParams(
                expression=s1out.expression.detach().to(self.device).requires_grad_(True),
                head_pose=s1out.head_pose.detach().to(self.device).requires_grad_(True),
                jaw_pose=s1out.jaw_pose.detach().to(self.device).requires_grad_(True),
                translation=t_init.requires_grad_(True),
                lighting=s1out.lighting.detach().to(self.device).requires_grad_(True),
            )
            per_image.append(pip)

        return shared, per_image

    def _create_loss_aggregator(self, shared: SharedParams) -> LossAggregator:
        return LossAggregator(
            self.config, self.flame,
            mica_init_shape=shared.shape.detach().clone(),
            arcface_model=self.arcface_model,
        )

    def _apply_constraints(self, shared: SharedParams):
        """Apply safety constraints after optimizer step."""
        with torch.no_grad():
            shared.focal_length.clamp_(
                self.config.focal_length_min, self.config.focal_length_max)

    def _build_output(self, shared: SharedParams,
                      per_image: list[PerImageParams],
                      loss_history: dict) -> Stage2Output:
        """Build final output: neutral expression mesh + per-image params."""
        with torch.no_grad():
            vertices, lmks_68, _ = self.flame(
                shared.shape,
                torch.zeros(1, 100, device=self.device),
                torch.zeros(1, 3, device=self.device),
                torch.zeros(1, 3, device=self.device),
            )

        # Detach per-image params for output
        per_image_out = []
        for pip in per_image:
            per_image_out.append(PerImageParams(
                expression=pip.expression.detach().cpu(),
                head_pose=pip.head_pose.detach().cpu(),
                jaw_pose=pip.jaw_pose.detach().cpu(),
                translation=pip.translation.detach().cpu(),
                lighting=pip.lighting.detach().cpu(),
            ))

        return Stage2Output(
            shape=shared.shape.detach().cpu(),
            texture=shared.texture.detach().cpu(),
            focal_length=shared.focal_length.detach().cpu(),
            vertices=vertices.detach().cpu(),
            landmarks_3d=lmks_68.detach().cpu(),
            per_image_params=per_image_out,
            loss_history=loss_history,
        )

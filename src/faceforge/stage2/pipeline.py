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
from .flame_wrapper import FLAMEWrapper, batch_rodrigues
from .camera import build_intrinsics, build_intrinsics_standard, build_extrinsics, project_points, world_to_camera
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

        # Configure pixel3dmm asset paths before creating any pixel3dmm objects
        from ._pixel3dmm_paths import configure_pixel3dmm_paths
        configure_pixel3dmm_paths(self.config)

        # Load FLAME (delegates to pixel3dmm's FLAME internally)
        self.flame = FLAMEWrapper(self.config).to(self.device)

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
        if self.visualizer is not None:
            self.visualizer.save_preprocessing(preprocessed)
        shared, per_image = self._init_params([s1out])

        # Setup UV correspondences
        loss_agg = self._create_loss_aggregator(shared)
        loss_agg.setup_uv_loss(self.flame, 1)
        loss_agg.uv_losses[0].compute_corresp(preprocessed[0].pixel3dmm_uv)

        loss_history = self._run_optimization(
            shared, per_image, preprocessed, loss_agg,
            selected_fn=lambda: [0], lr_scale=1.0,
        )

        self._save_final_previews(shared, per_image, preprocessed)
        return self._build_output(shared, per_image, loss_history)

    # ------------------------------------------------------------------
    # Multi-image joint optimization
    # ------------------------------------------------------------------

    def _run_multi(self, s1outs: list[Stage1Output]) -> Stage2Output:
        """Multi-image: sequential per-image → global joint optimization."""
        N = len(s1outs)
        preprocessed = [self._preprocess(s1out) for s1out in s1outs]
        if self.visualizer is not None:
            self.visualizer.save_preprocessing(preprocessed)
        shared, per_image = self._init_params(s1outs)

        loss_agg = self._create_loss_aggregator(shared)
        loss_agg.setup_uv_loss(self.flame, N)

        # Setup UV correspondences for each image
        for i in range(N):
            loss_agg.uv_losses[i].compute_corresp(preprocessed[i].pixel3dmm_uv)

        # Phase 1: Sequential (freeze shape, per-image optimization)
        shared.shape.requires_grad_(False)
        for i in range(N):
            self._run_sequential_stage(shared, per_image, preprocessed, loss_agg, i)
        shared.shape.requires_grad_(True)

        # Phase 2: Global joint optimization (pixel3dmm: global_iters=5000)
        batch_size = min(N, self.config.multi_image_batch_size)

        def sample_fn():
            return random.sample(range(N), batch_size)

        loss_history = self._run_optimization(
            shared, per_image, preprocessed, loss_agg,
            selected_fn=sample_fn, lr_scale=self.config.global_lr_scale,
            steps_override={'medium': self.config.global_iters},
        )

        self._save_final_previews(shared, per_image, preprocessed)
        return self._build_output(shared, per_image, loss_history)

    def _run_sequential_stage(self, shared, per_image, preprocessed, loss_agg, idx):
        """Quick per-image alignment (coarse camera + medium subset).

        Coarse: single loop with first half lmk, second half UV (matching pixel3dmm).
        """
        coarse_steps = self.config.sequential_coarse_steps
        optimizer = self.opt_manager.create_optimizer(
            'coarse_lmk', shared, per_image, [idx], lr_scale=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(coarse_steps * 0.75), gamma=0.1)

        for step in range(coarse_steps):
            optimizer.zero_grad()
            stage = 'coarse_lmk' if step <= coarse_steps // 2 else 'coarse_uv'
            loss, _ = self._forward_step(
                shared, per_image[idx], preprocessed[idx],
                loss_agg, stage, idx)
            loss.backward()
            optimizer.step()
            scheduler.step()
            self._apply_constraints(shared)

        # Medium subset
        medium_steps = self.config.sequential_medium_steps
        optimizer = self.opt_manager.create_optimizer(
            'medium', shared, per_image, [idx], lr_scale=1.0)

        for step in range(medium_steps):
            optimizer.zero_grad()
            loss, _ = self._forward_step(
                shared, per_image[idx], preprocessed[idx],
                loss_agg, 'medium', idx)
            loss.backward()
            optimizer.step()
            self._apply_constraints(shared)

    # ------------------------------------------------------------------
    # Core optimization loop
    # ------------------------------------------------------------------

    def _run_coarse(self, shared, per_image, preprocessed, loss_agg,
                    selected_fn):
        """Run coarse stage = pixel3dmm optimize_camera.

        Single loop, shared optimizer+scheduler:
          - First half: landmark loss only (coarse_lmk)
          - Second half: UV loss only (coarse_uv)
        Ref: pixel3dmm tracker.py L619-688
        """
        total_steps = self.config.coarse_lmk_steps  # 500 = full camera phase
        selected = selected_fn()

        # Same optimizer for both halves (continuous Adam momentum)
        optimizer = self.opt_manager.create_optimizer(
            'coarse_lmk', shared, per_image, selected, 1.0)
        # pixel3dmm: StepLR at 75% of steps with gamma=0.1
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(total_steps * 0.75), gamma=0.1)

        history = []
        for step in range(total_steps):
            optimizer.zero_grad()

            # First half = coarse_lmk, second half = coarse_uv
            stage = 'coarse_lmk' if step <= total_steps // 2 else 'coarse_uv'

            total_loss = torch.tensor(0.0, device=self.device)
            for i in selected:
                loss_i, _ = self._forward_step(
                    shared, per_image[i], preprocessed[i],
                    loss_agg, stage, i)
                total_loss = total_loss + loss_i

            total_loss = total_loss / len(selected)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            self._apply_constraints(shared)
            history.append(total_loss.item())

        # Visualization snapshot at end of coarse
        if self.visualizer is not None:
            for vis_idx in range(len(per_image)):
                self._snapshot_stage('coarse_lmk', shared, per_image, preprocessed, vis_idx)

        return history

    def _run_optimization(self, shared, per_image, preprocessed, loss_agg,
                          selected_fn, lr_scale,
                          steps_override: dict | None = None) -> dict:
        """Run optimization: coarse (camera) + medium+ stages.

        Args:
            steps_override: optional dict mapping stage name → step count,
                            e.g. {'medium': 5000} for joint phase.
        """
        all_loss_history = {}

        # === Coarse = pixel3dmm optimize_camera (single loop) ===
        coarse_steps = self.config.coarse_lmk_steps
        if coarse_steps > 0:
            all_loss_history['coarse'] = self._run_coarse(
                shared, per_image, preprocessed, loss_agg, selected_fn)

        # === Medium+ stages ===
        stages = ['medium']
        if self.config.fine_pca_steps > 0:
            stages.append('fine_pca')
        if self.config.enable_fine_detail and self.config.fine_detail_steps > 0:
            stages.append('fine_detail')

        for stage in stages:
            total_steps = (steps_override or {}).get(
                stage, self.opt_manager.get_steps(stage))
            if total_steps <= 0:
                continue
            stage_history = []

            # Tighten UV thresholds at medium stage
            if stage == 'medium':
                for uv_loss in loss_agg.uv_losses:
                    uv_loss.finish_stage1(
                        delta_uv_fine=self.config.uv_delta_fine,
                        dist_uv_fine=self.config.uv_dist_fine)

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

                # LR decay (only in joint phase)
                progress = step / max(total_steps - 1, 1)
                self.opt_manager.adjust_lr(optimizer, progress)

                self._apply_constraints(shared)
                stage_history.append(total_loss.item())

                # Early stopping — only in joint multi-image phase
                # pixel3dmm: no early stopping in single-image optimize_color
                # (only in per-frame sequential within multi-image)
                # We disable it here; it's handled in _run_sequential_stage if needed.

            all_loss_history[stage] = stage_history

            # Visualization snapshot
            if self.visualizer is not None:
                for vis_idx in range(len(per_image)):
                    self._snapshot_stage(stage, shared, per_image, preprocessed, vis_idx)

        # Final progression image
        if self.visualizer is not None:
            self.visualizer.save_stage_progression()

        return all_loss_history

    def _forward_step(self, shared, per_img, preprocessed, loss_agg, stage, img_idx):
        """Single forward: FLAME(canonical) → external R@v+t → project → losses.

        Matches pixel3dmm tracker.py exactly:
          1. FLAME outputs canonical vertices (no head rotation)
          2. External rotation: vertices = R @ v_canonical + t
          3. Project to 2D and render
        Ref: pixel3dmm tracker.py L865-879
        """
        # FLAME forward — canonical output, all 6D rotations
        verts_can, lmks_68_can, lmks_eyes_can, R_head = self.flame(
            shared.shape, per_img.expression,
            per_img.R_6d, per_img.jaw_6d,
            eyes_6d=per_img.eyes_6d, neck_6d=per_img.neck_6d,
            eyelids=per_img.eyelids,
        )

        # External rotation + translation (pixel3dmm tracker.py L865-871):
        #   v_world = R @ v_can + flame_t
        # Then projection uses t_base=[0,0,-1] as fixed camera offset.
        t = per_img.translation.unsqueeze(1)  # [B, 1, 3] optimizable (starts at 0)
        vertices = torch.einsum('bny,bxy->bnx', verts_can, R_head) + t
        lmks_68 = torch.einsum('bny,bxy->bnx', lmks_68_can, R_head) + t
        lmks_eyes = torch.einsum('bny,bxy->bnx', lmks_eyes_can, R_head) + t

        vertex_normals = self.flame.get_vertex_normals(vertices)

        # Camera: fixed t_base=[0,0,-1] in extrinsics (pixel3dmm dreifus convention)
        # This decouples depth from focal_length — pixel3dmm optimizes flame_t (small)
        # while t_base provides the base depth, preventing focal-depth coupling.
        K = build_intrinsics(shared.focal_length, per_img.principal_point,
                             self.config.render_size)
        t_base = torch.tensor([[0., 0., -1.]], device=self.device)
        RT = build_extrinsics(torch.zeros(1, 3, device=self.device), t_base)

        # Project landmarks
        proj_lmks_68, _ = project_points(lmks_68, K, RT, self.config.render_size)
        proj_lmks_eyes, _ = project_points(lmks_eyes, K, RT, self.config.render_size)
        proj_verts, proj_depths = project_points(vertices, K, RT, self.config.render_size)

        # Render (uses standard K without hack, matching pixel3dmm renderer convention)
        render_out = {}
        need_render = stage in ('medium', 'fine_pca', 'fine_detail')
        if need_render and self.renderer.available:
            K_render = build_intrinsics_standard(shared.focal_length, self.config.render_size)
            render_out = self.renderer.render(
                vertices, self.flame.faces, K_render, RT,
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
            target_lmks_68=preprocessed.target_lmks_68,
            target_lmks_eyes=preprocessed.target_lmks_eyes,
            projected_vertices=proj_verts,
            proj_depths=proj_depths,
            visibility_mask=visibility_mask,
            face_mask=preprocessed.face_mask,
            shape=shared.shape,
            expression=per_img.expression,
            jaw_6d=per_img.jaw_6d,
            lighting=per_img.lighting,
            vertices_canonical=verts_can,
            principal_point=per_img.principal_point,
            lmk_mask=preprocessed.lmk_mask,
            target_iris_left=preprocessed.target_iris_left,
            target_iris_right=preprocessed.target_iris_right,
            iris_mask_left=preprocessed.iris_mask_left,
            iris_mask_right=preprocessed.iris_mask_right,
            eyes_6d=per_img.eyes_6d,
            neck_6d=per_img.neck_6d,
        )

        if render_out:
            # Silhouette masks (pixel3dmm tracker.py L1526-1533)
            seg = preprocessed.face_segmentation
            sil_fg = ((seg == 1) | (seg == 2) | (seg == 4) | (seg == 5) |
                      (seg == 6) | (seg == 7) | (seg == 10) |
                      (seg == 12) | (seg == 13)).float()
            sil_bg = (seg <= 1).float()

            # Normal loss: rendered normals are in world space (= rotated by R_head).
            # pixel3dmm compares with canonical-space normals using R^T @ rendered.
            # Ref: pixel3dmm tracker.py L984-986
            loss_kwargs.update(
                rendered_image=render_out.get('image'),
                rendered_normals=render_out.get('normal'),
                rendered_mask=render_out.get('mask'),
                target_image=preprocessed.target_image,
                predicted_normals=preprocessed.pixel3dmm_normals,
                target_arcface_feat=preprocessed.arcface_feat,
                cam_rotation=R_head,
                sil_fg_mask=sil_fg,
                sil_valid_bg_mask=sil_bg,
            )

        return loss_agg.compute(stage, image_idx=img_idx, **loss_kwargs)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_final_previews(self, shared: SharedParams,
                             per_image: list[PerImageParams],
                             preprocessed: list[PreprocessedData]):
        """Save pixel3dmm-style preview for each image after final optimization.

        Preview: [original | normal blend | rendered normal]
        Ref: pixel3dmm tracker.py L1410-1413
        """
        if self.visualizer is None or not self.renderer.available:
            return

        for idx in range(len(per_image)):
            pip = per_image[idx]
            prep = preprocessed[idx]

            # FLAME forward + external rotation
            verts_can, lmks_68_can, _, R_head = self.flame(
                shared.shape, pip.expression, pip.R_6d, pip.jaw_6d,
                eyes_6d=pip.eyes_6d, neck_6d=pip.neck_6d, eyelids=pip.eyelids,
            )
            t = pip.translation.unsqueeze(1)
            vertices = torch.einsum('bny,bxy->bnx', verts_can, R_head) + t
            vertex_normals = self.flame.get_vertex_normals(vertices)

            K = build_intrinsics(shared.focal_length, pip.principal_point,
                                 self.config.render_size)
            t_base = torch.tensor([[0., 0., -1.]], device=self.device)
            RT = build_extrinsics(torch.zeros(1, 3, device=self.device), t_base)

            K_render = build_intrinsics_standard(shared.focal_length, self.config.render_size)
            render_out = self.renderer.render(
                vertices, self.flame.faces, K_render, RT,
                vertex_normals, sh_coefficients=pip.lighting,
            )

            self.visualizer.save_p3m_preview(
                image_idx=idx,
                target_image=prep.target_image,
                rendered_normals=render_out['normal'],
                rendered_mask=render_out['mask'],
                cam_rotation=R_head,
                predicted_normals=prep.pixel3dmm_normals,
            )

    @torch.no_grad()
    def _snapshot_stage(self, stage: str, shared: SharedParams,
                        per_image: list[PerImageParams],
                        preprocessed: list[PreprocessedData],
                        idx: int):
        """Take a visualization snapshot at the end of a stage for one image."""
        pip = per_image[idx]
        prep = preprocessed[idx]

        # FLAME forward (canonical) + external rotation (same as _forward_step)
        verts_can, lmks_68_can, _, R_head = self.flame(
            shared.shape, pip.expression, pip.R_6d, pip.jaw_6d,
            eyes_6d=pip.eyes_6d, neck_6d=pip.neck_6d, eyelids=pip.eyelids,
        )
        t = pip.translation.unsqueeze(1)
        vertices = torch.einsum('bny,bxy->bnx', verts_can, R_head) + t
        lmks_68 = torch.einsum('bny,bxy->bnx', lmks_68_can, R_head) + t
        vertex_normals = self.flame.get_vertex_normals(vertices)

        K = build_intrinsics(shared.focal_length, pip.principal_point,
                             self.config.render_size)
        t_base = torch.tensor([[0., 0., -1.]], device=self.device)
        RT = build_extrinsics(torch.zeros(1, 3, device=self.device), t_base)

        proj_lmks_68, _ = project_points(lmks_68, K, RT, self.config.render_size)

        # Render (if available)
        rendered_image = None
        rendered_mask = None
        if self.renderer.available:
            K_render = build_intrinsics_standard(shared.focal_length, self.config.render_size)
            render_out = self.renderer.render(
                vertices, self.flame.faces, K_render, RT,
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
        seg_raw = s1out.parsing_map if s1out.parsing_map is not None else s1out.face_mask
        uv_map, normal_map = self.pixel3dmm.predict(
            s1out.aligned_image, seg_raw)

        # Preserve raw 19-class segmentation for pixel3dmm-style mask construction
        # (needed for silhouette fg/bg split in _forward_step)
        if seg_raw.dim() == 2:
            seg_raw = seg_raw.unsqueeze(0)   # [1, H, W]
        elif seg_raw.dim() == 3:
            pass                             # already [1, H, W]
        face_segmentation = seg_raw.long()   # [1, H, W] int

        binary_mask = s1out.face_mask
        if binary_mask.dim() == 2:
            binary_mask = binary_mask.unsqueeze(0)

        d = self.device

        # Landmark mask: non-zero landmarks are valid
        # Ref: pixel3dmm tracker.py L1489
        lmks_68 = s1out.lmks_68  # [1, 68, 2]
        lmk_mask = ~(lmks_68.sum(2, keepdim=True) == 0)  # [1, 68, 1]

        # Iris landmarks: pixel3dmm FLAME convention:
        #   left_iris_flame = [4597,...] = FLAME's LEFT eye (viewer's RIGHT)
        #   right_iris_flame = [4051,...] = FLAME's RIGHT eye (viewer's LEFT)
        # MediaPipe convention:
        #   R_EYE_MP_LMKS = [468,...] = viewer's RIGHT eye
        #   L_EYE_MP_LMKS = [473,...] = viewer's LEFT eye
        # Mapping: pixel3dmm "left" (4597) = MediaPipe "right" (468)
        #          pixel3dmm "right" (4051) = MediaPipe "left" (473)
        lmks_eyes = s1out.lmks_eyes  # [1, 10, 2] = [right_5, left_5]
        target_iris_left = lmks_eyes[:, 0:1, :]   # MediaPipe right = pixel3dmm left
        target_iris_right = lmks_eyes[:, 5:6, :]   # MediaPipe left = pixel3dmm right
        iris_mask_left = ~(target_iris_left.sum(2, keepdim=True) == 0)
        iris_mask_right = ~(target_iris_right.sum(2, keepdim=True) == 0)

        return PreprocessedData(
            pixel3dmm_uv=uv_map,
            pixel3dmm_normals=normal_map,
            face_mask=binary_mask.float().to(d),
            face_segmentation=face_segmentation.to(d),
            target_image=s1out.aligned_image.to(d),
            target_lmks_68=lmks_68.to(d),
            target_lmks_eyes=lmks_eyes.to(d),
            lmk_mask=lmk_mask.to(d),
            target_iris_left=target_iris_left.to(d),
            target_iris_right=target_iris_right.to(d),
            iris_mask_left=iris_mask_left.to(d),
            iris_mask_right=iris_mask_right.to(d),
            arcface_feat=s1out.arcface_feat.to(d) if s1out.arcface_feat is not None else None,
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
        # pixel3dmm: flame_t starts at zeros. The base depth comes from
        # t_base=[0,0,-1] in the projection extrinsics (not in flame_t).
        # This prevents focal-depth coupling during optimization.
        z_init = 0.0
        per_image = []
        for s1out in s1outs:
            t_init = torch.tensor([[0.0, 0.0, z_init]], device=self.device)
            # pixel3dmm initializes expression, head_pose, jaw_pose to ZEROS
            # (tracker.py L464-469). Using DECA pre-estimates causes different
            # local minima and instability.
            # pixel3dmm: eyes/neck/eyelids all start at identity/zeros
            from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
            I6d = matrix_to_rotation_6d(torch.eye(3, device=self.device).unsqueeze(0))

            pip = PerImageParams(
                expression=torch.zeros(1, 100, device=self.device).requires_grad_(True),
                R_6d=I6d.clone().requires_grad_(True),            # [1, 6] head rotation
                jaw_6d=I6d.clone().requires_grad_(True),          # [1, 6] jaw rotation
                translation=t_init.requires_grad_(True),
                lighting=s1out.lighting.detach().to(self.device).requires_grad_(True),
                principal_point=torch.zeros(1, 2, device=self.device).requires_grad_(True),
                eyes_6d=I6d.repeat(1, 2).requires_grad_(True),  # [1, 12]
                neck_6d=I6d.clone().requires_grad_(True),        # [1, 6]
                eyelids=torch.zeros(1, 2, device=self.device).requires_grad_(True),
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
        from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d
        I6d = matrix_to_rotation_6d(torch.eye(3, device=self.device).unsqueeze(0))

        with torch.no_grad():
            vertices, lmks_68, _, _ = self.flame(
                shared.shape,
                torch.zeros(1, 100, device=self.device),
                I6d,   # identity head rotation (6D)
                I6d,   # identity jaw (6D)
            )

        # Detach per-image params for output
        per_image_out = []
        for pip in per_image:
            per_image_out.append(PerImageParams(
                expression=pip.expression.detach().cpu(),
                R_6d=pip.R_6d.detach().cpu(),
                jaw_6d=pip.jaw_6d.detach().cpu(),
                translation=pip.translation.detach().cpu(),
                lighting=pip.lighting.detach().cpu(),
                principal_point=pip.principal_point.detach().cpu() if pip.principal_point is not None else None,
                eyes_6d=pip.eyes_6d.detach().cpu() if pip.eyes_6d is not None else None,
                neck_6d=pip.neck_6d.detach().cpu() if pip.neck_6d is not None else None,
                eyelids=pip.eyelids.detach().cpu() if pip.eyelids is not None else None,
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

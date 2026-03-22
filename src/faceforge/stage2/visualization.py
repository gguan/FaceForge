"""Stage 2 debug visualization.

Saves per-stage snapshots during optimization and a final progression strip
showing how the fit improves from coarse_lmk → fine_detail.

Each stage snapshot is a 3-panel row:
  [Landmark overlay]  [Rendered mesh]  [Render vs Target blend]

Final output: a wide image with all stages side-by-side.

Reference: src/faceforge/stage1/visualization.py (directory structure pattern)
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


# Landmark connections for drawing (68-pt markup)
_LMK_CONTOURS = [
    list(range(0, 17)),     # jawline
    list(range(17, 22)),    # left eyebrow
    list(range(22, 27)),    # right eyebrow
    list(range(27, 31)),    # nose bridge
    list(range(31, 36)),    # nose bottom
    list(range(36, 42)),    # left eye
    list(range(42, 48)),    # right eye
    list(range(48, 60)),    # outer lip
    list(range(60, 68)),    # inner lip
]
_LMK_CLOSED = {36, 42, 48, 60}  # contours that should be closed


STAGE_LABELS = {
    'coarse_lmk':   'Coarse LMK',
    'coarse_uv':    'Coarse UV',
    'medium':       'Medium',
    'fine_pca':     'Fine PCA',
    'fine_detail':  'Fine Detail',
}


class Stage2Visualizer:
    """Save debug visualizations during Stage 2 optimization."""

    PANEL_SIZE = 256  # side length of each panel

    def __init__(self, output_dir: str, subject_name: str):
        self.base_dir = Path(output_dir) / subject_name / 'stage2'
        self.preprocess_dir = self.base_dir / '01_preprocessing'
        self.optim_dir = self.base_dir / '02_optimization'
        self.result_dir = self.base_dir / '03_result'
        for d in [self.preprocess_dir, self.optim_dir, self.result_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Accumulate per-stage snapshot panels for final progression image
        self._stage_snapshots: dict[tuple[str, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def save_preprocessing(self, preprocessed_list, image_indices=None):
        """Save Pixel3DMM UV/Normal maps."""
        try:
            from PIL import Image
        except ImportError:
            return

        for i, p in enumerate(preprocessed_list):
            idx = image_indices[i] if image_indices else i

            # UV map: [1, 2, H, W] → colorize
            uv = p.pixel3dmm_uv[0].cpu().numpy()  # [2, H, W]
            uv_vis = np.zeros((uv.shape[1], uv.shape[2], 3), dtype=np.uint8)
            uv_vis[:, :, 0] = (uv[0] * 255).clip(0, 255).astype(np.uint8)
            uv_vis[:, :, 1] = (uv[1] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(uv_vis).save(self.preprocess_dir / f'pixel3dmm_uv_{idx}.png')

            # Normal map: [1, 3, H, W] → RGB
            n = p.pixel3dmm_normals[0].cpu().numpy()  # [3, H, W]
            n_vis = ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(n_vis).save(self.preprocess_dir / f'pixel3dmm_normals_{idx}.png')

    # ------------------------------------------------------------------
    # Per-stage snapshot (called at end of each optimization stage)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def save_stage_snapshot(
        self,
        stage: str,
        target_image: torch.Tensor,          # [1, 3, H, W] or [1, H, W, 3]
        target_lmks_68: torch.Tensor,         # [1, 68, 2] pixel coords
        pred_lmks_68: torch.Tensor,           # [1, 68, 2] pixel coords
        rendered_image: Optional[torch.Tensor] = None,  # [1, H, W, 3] from renderer
        rendered_mask: Optional[torch.Tensor] = None,    # [1, H, W, 1]
        image_idx: int = 0,
    ) -> np.ndarray:
        """Create and save a 3-panel snapshot for one optimization stage.

        Panels:
            1. Landmark overlay: target image with green=target lmks, red=predicted lmks
            2. Rendered mesh: renderer output (normal-colored if no texture/lighting)
            3. Render vs Target: alpha blend of rendered image over target (50/50)

        Args:
            stage: stage name (e.g. 'coarse_lmk')
            target_image: aligned input image
            target_lmks_68: target 2D landmarks [1, 68, 2]
            pred_lmks_68: predicted projected landmarks [1, 68, 2]
            rendered_image: renderer output [1, H, W, 3] float [0,1]
            rendered_mask: renderer mask [1, H, W, 1] float
            image_idx: which image (for multi-image)

        Returns:
            snapshot: RGB uint8 [PANEL_SIZE, PANEL_SIZE * 3, 3]
        """
        S = self.PANEL_SIZE

        # --- Prepare target image as numpy HWC uint8 ---
        target_np = self._tensor_to_hwc_uint8(target_image)
        H, W = target_np.shape[:2]

        # Scale factor for drawing on resized image
        sx, sy = S / W, S / H

        # --- Panel 1: Landmark overlay ---
        lmk_panel = cv2.resize(target_np.copy(), (S, S))
        gt_lmks = target_lmks_68[0].cpu().numpy()   # [68, 2]
        pr_lmks = pred_lmks_68[0].cpu().numpy()      # [68, 2]
        self._draw_landmarks(lmk_panel, gt_lmks * [sx, sy], color=(0, 255, 0), label='GT')
        self._draw_landmarks(lmk_panel, pr_lmks * [sx, sy], color=(0, 0, 255), label='Pred')

        # --- Panel 2: Rendered mesh ---
        if rendered_image is not None:
            render_np = self._tensor_to_hwc_uint8(rendered_image)
            mesh_panel = cv2.resize(render_np, (S, S))
        else:
            # No renderer available — show placeholder with text
            mesh_panel = np.zeros((S, S, 3), dtype=np.uint8)
            cv2.putText(mesh_panel, 'No Render', (S // 4, S // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # --- Panel 3: Render vs Target blend ---
        if rendered_image is not None and rendered_mask is not None:
            blend_panel = self._create_blend_panel(target_np, rendered_image, rendered_mask, S)
        elif rendered_image is not None:
            # No mask, just 50/50 blend
            render_np = self._tensor_to_hwc_uint8(rendered_image)
            target_resized = cv2.resize(target_np, (S, S))
            render_resized = cv2.resize(render_np, (S, S))
            blend_panel = cv2.addWeighted(target_resized, 0.5, render_resized, 0.5, 0)
        else:
            blend_panel = cv2.resize(target_np, (S, S))

        # --- Add stage label to each panel ---
        label = STAGE_LABELS.get(stage, stage)
        for panel in [lmk_panel, mesh_panel, blend_panel]:
            cv2.putText(panel, label, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        lineType=cv2.LINE_AA)

        # Concatenate horizontally
        snapshot = np.concatenate([lmk_panel, mesh_panel, blend_panel], axis=1)

        # Save individual stage snapshot
        fname = f'stage_{stage}_img{image_idx}.png'
        cv2.imwrite(
            str(self.optim_dir / fname),
            cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR),
        )

        # Store for progression image: key = (stage, image_idx)
        self._stage_snapshots[(stage, image_idx)] = snapshot
        return snapshot

    def save_stage_progression(self) -> None:
        """Save per-image progression strips.

        For each image, stacks all stage snapshots vertically:
            coarse_lmk:  [lmk | mesh | blend]
            coarse_uv:   [lmk | mesh | blend]
            medium:      [lmk | mesh | blend]
            ...

        Output files: stage_progression_img{idx}.png  (one per input image)
        Also saves: stage_progression.png  (first image, for backwards compat)
        """
        if not self._stage_snapshots:
            return

        # Collect unique image indices
        image_indices = sorted({idx for (_, idx) in self._stage_snapshots.keys()})
        stage_order = ['coarse_lmk', 'coarse_uv', 'medium', 'fine_pca', 'fine_detail']

        for img_idx in image_indices:
            rows = []
            for stage in stage_order:
                snap = self._stage_snapshots.get((stage, img_idx))
                if snap is not None:
                    rows.append(snap)
            if not rows:
                continue

            progression = np.concatenate(rows, axis=0)
            fname = f'stage_progression_img{img_idx}.png'
            cv2.imwrite(
                str(self.optim_dir / fname),
                cv2.cvtColor(progression, cv2.COLOR_RGB2BGR),
            )

        # Also save first image as stage_progression.png (backwards compat)
        first_idx = image_indices[0]
        rows = []
        for stage in stage_order:
            snap = self._stage_snapshots.get((stage, first_idx))
            if snap is not None:
                rows.append(snap)
        if rows:
            cv2.imwrite(
                str(self.optim_dir / 'stage_progression.png'),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR),
            )

    # ------------------------------------------------------------------
    # Progress during optimization (every N steps)
    # ------------------------------------------------------------------

    def save_progress(self, step: int, rendered_image: torch.Tensor | None,
                      target_image: torch.Tensor | None, loss_dict: dict):
        """Save optimization progress every N steps."""
        try:
            from PIL import Image
        except ImportError:
            return

        if rendered_image is not None:
            img = rendered_image[0].detach().cpu().numpy()  # [H, W, 3]
            img = (img * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(self.optim_dir / f'progress_step_{step:04d}.png')

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------

    def save_loss_curves(self, loss_history: dict):
        """Save loss curves as PNG — one subplot per stage, all loss terms overlaid."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # loss_history: {'coarse_lmk': [float,...], ...} for total loss
        # or enhanced: {'coarse_lmk': {'total': [...], 'lmk_68': [...], ...}, ...}
        is_detailed = any(isinstance(v, dict) for v in loss_history.values())

        if is_detailed:
            n_stages = len(loss_history)
            fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 4),
                                     squeeze=False)
            axes = axes[0]
            for ax, (stage_name, terms) in zip(axes, loss_history.items()):
                if isinstance(terms, dict):
                    for term_name, values in terms.items():
                        if values:
                            ax.plot(values, label=term_name, alpha=0.8)
                else:
                    ax.plot(terms, label='total', alpha=0.8)
                ax.set_title(stage_name)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_yscale('log')
                ax.legend(fontsize=7)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            for stage_name, history in loss_history.items():
                ax.plot(history, label=stage_name, alpha=0.8)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Stage 2 Optimization Loss')
            ax.legend()
            ax.set_yscale('log')

        fig.tight_layout()
        fig.savefig(self.optim_dir / 'loss_curves.png', dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    def save_result(self, stage2_output, vertices_before=None):
        """Save final optimized parameters as npz."""
        np.savez(
            self.result_dir / 'flame_params_optimized.npz',
            shape=stage2_output.shape.cpu().numpy(),
            texture=stage2_output.texture.cpu().numpy(),
            focal_length=stage2_output.focal_length.cpu().numpy(),
            vertices=stage2_output.vertices[0].cpu().numpy(),
            landmarks_3d=stage2_output.landmarks_3d.cpu().numpy(),
        )

    def save_mesh_obj(self, vertices: torch.Tensor, faces: torch.Tensor,
                      filename: str = 'mesh_optimized.obj'):
        """Save mesh as OBJ file."""
        verts = vertices[0].detach().cpu().numpy() * 1000  # m → mm
        f = faces.cpu().numpy()
        path = self.result_dir / filename
        with open(path, 'w') as fp:
            for v in verts:
                fp.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
            for face in f:
                fp.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
        """Convert image tensor to [H, W, 3] uint8 RGB.

        Handles:
          [1, 3, H, W] CHW float → HWC uint8
          [1, H, W, 3] HWC float → HWC uint8
          [H, W, 3]    HWC float → HWC uint8
        """
        x = t.detach().cpu()
        if x.dim() == 4:
            x = x[0]  # remove batch
        if x.shape[0] == 3 and x.dim() == 3:
            x = x.permute(1, 2, 0)  # CHW → HWC
        arr = x.numpy()
        if arr.max() <= 1.0 + 1e-3:
            arr = arr * 255
        return arr.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _draw_landmarks(img: np.ndarray, lmks: np.ndarray,
                        color: tuple = (0, 255, 0), label: str = ''):
        """Draw 68-point landmarks with connections on an image.

        Args:
            img: [H, W, 3] uint8 RGB (modified in-place)
            lmks: [68, 2] pixel coords
            color: BGR tuple for points and lines
        """
        # Draw contour lines
        for contour in _LMK_CONTOURS:
            for j in range(len(contour) - 1):
                p1 = tuple(lmks[contour[j]].astype(int))
                p2 = tuple(lmks[contour[j + 1]].astype(int))
                cv2.line(img, p1, p2, color, 1, lineType=cv2.LINE_AA)
            # Close loops for eyes and lips
            if contour[0] in _LMK_CLOSED:
                p1 = tuple(lmks[contour[-1]].astype(int))
                p2 = tuple(lmks[contour[0]].astype(int))
                cv2.line(img, p1, p2, color, 1, lineType=cv2.LINE_AA)

        # Draw points
        for pt in lmks:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _create_blend_panel(
        target_np: np.ndarray,
        rendered_image: torch.Tensor,
        rendered_mask: torch.Tensor,
        size: int,
    ) -> np.ndarray:
        """Create a checkerboard blend panel of target vs rendered image.

        Uses the render mask: where mesh is visible, show rendered; elsewhere show target.
        Alpha blend at boundary for smooth transition.
        """
        render_np = Stage2Visualizer._tensor_to_hwc_uint8(rendered_image)
        mask_np = rendered_mask[0].detach().cpu().numpy()  # [H, W, 1]
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Resize all to panel size
        target_r = cv2.resize(target_np, (size, size))
        render_r = cv2.resize(render_np, (size, size))
        mask_r = cv2.resize(mask_np, (size, size))

        # Create checkerboard pattern (16px squares)
        checker = np.zeros((size, size), dtype=np.float32)
        block = 16
        for y in range(0, size, block):
            for x in range(0, size, block):
                if ((y // block) + (x // block)) % 2 == 0:
                    checker[y:y+block, x:x+block] = 1.0

        # Where mask > 0.5: alternate between render and target in checkerboard
        alpha = mask_r[:, :, np.newaxis] if mask_r.ndim == 2 else mask_r
        alpha = np.clip(alpha, 0, 1)
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]

        checker_3 = checker[:, :, np.newaxis]
        # Inside mesh: checkerboard alternates render/target
        inside = (checker_3 * render_r + (1 - checker_3) * target_r).astype(np.uint8)
        # Final: blend inside (where mask) with target (outside mask)
        blend = (alpha * inside.astype(np.float32) + (1 - alpha) * target_r.astype(np.float32))
        return blend.clip(0, 255).astype(np.uint8)

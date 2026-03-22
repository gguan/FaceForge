"""P3M baseline visualization.

Saves per-stage comparison images showing target vs. predicted result.
Output format mirrors Stage 2 visualization for easy side-by-side comparison.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


STAGE_LABELS = {
    'camera': 'Camera Opt',
    'per_frame': 'Per-Frame',
    'joint': 'Joint',
}


class P3MVisualizer:
    """Save P3M baseline visualizations."""

    PANEL_SIZE = 256

    def __init__(self, output_dir: str, subject_name: str):
        self.base_dir = Path(output_dir) / subject_name / 'stageP3M'
        self.optim_dir = self.base_dir / 'optimization'
        for d in [self.optim_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._stage_snapshots: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def save_stage_snapshot(
        self,
        stage: str,
        target_image: torch.Tensor,
        target_lmks_68: torch.Tensor,
        pred_lmks_68: torch.Tensor,
        rendered_image: Optional[torch.Tensor] = None,
        rendered_mask: Optional[torch.Tensor] = None,
        image_idx: int = 0,
    ) -> np.ndarray:
        """Create and save a 3-panel snapshot for one optimization stage."""
        S = self.PANEL_SIZE

        target_np = self._tensor_to_hwc_uint8(target_image)
        H, W = target_np.shape[:2]
        sx, sy = S / W, S / H

        # Panel 1: Landmark overlay
        lmk_panel = cv2.resize(target_np.copy(), (S, S))
        gt_lmks = target_lmks_68[0].cpu().numpy()
        pr_lmks = pred_lmks_68[0].cpu().numpy()
        self._draw_landmarks(lmk_panel, gt_lmks * [sx, sy], (0, 255, 0))
        self._draw_landmarks(lmk_panel, pr_lmks * [sx, sy], (0, 0, 255))

        # Panel 2: Rendered mesh
        if rendered_image is not None:
            render_np = self._tensor_to_hwc_uint8(rendered_image)
            mesh_panel = cv2.resize(render_np, (S, S))
        else:
            mesh_panel = np.zeros((S, S, 3), dtype=np.uint8)
            cv2.putText(mesh_panel, 'No Render', (S // 4, S // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Panel 3: Blend
        if rendered_image is not None and rendered_mask is not None:
            render_np = self._tensor_to_hwc_uint8(rendered_image)
            target_resized = cv2.resize(target_np, (S, S))
            render_resized = cv2.resize(render_np, (S, S))
            blend_panel = cv2.addWeighted(target_resized, 0.5, render_resized, 0.5, 0)
        else:
            blend_panel = cv2.resize(target_np.copy(), (S, S))

        # Combine panels
        row = np.concatenate([lmk_panel, mesh_panel, blend_panel], axis=1)

        # Add stage label
        label = STAGE_LABELS.get(stage.split('_')[0], stage)
        cv2.putText(row, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self._stage_snapshots[f'{stage}_{image_idx}'] = row

        key = f'{stage}_{image_idx}'
        out_path = self.optim_dir / f'{key}.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        return row

    def save_stage_progression(self) -> Optional[np.ndarray]:
        """Save all stages in one vertical strip."""
        if not self._stage_snapshots:
            return None

        rows = list(self._stage_snapshots.values())
        progression = np.concatenate(rows, axis=0)
        out_path = self.optim_dir / 'progression.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(progression, cv2.COLOR_RGB2BGR))
        return progression

    # ------------------------------------------------------------------
    # Final artifact helpers (interface parity with Stage2Visualizer)
    # ------------------------------------------------------------------

    def save_loss_curves(self, loss_history: dict):
        """Save loss curves as a PNG plot.

        Args:
            loss_history: dict mapping stage/key → list[float]
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        for key, vals in loss_history.items():
            if vals:
                ax.plot(vals, label=key, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('P3M Loss Curves')
        ax.legend(fontsize=7)
        ax.set_yscale('log')
        fig.tight_layout()
        out = self.base_dir / 'loss_curves.png'
        fig.savefig(str(out), dpi=120)
        plt.close(fig)

    def save_mesh_obj(self, vertices: torch.Tensor, faces: torch.Tensor,
                      filename: str = 'mesh_optimized.obj'):
        """Save neutral mesh as OBJ file.

        Args:
            vertices: [1, V, 3] or [V, 3]
            faces:    [F, 3] long
            filename: output filename (placed in base_dir/result/)
        """
        result_dir = self.base_dir / 'result'
        result_dir.mkdir(parents=True, exist_ok=True)
        out = result_dir / filename

        v = vertices.squeeze(0).detach().cpu().numpy() if vertices.dim() == 3 else vertices.detach().cpu().numpy()
        f = faces.detach().cpu().numpy()

        with open(out, 'w') as fp:
            for vx, vy, vz in v:
                fp.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
            for fi in f:
                fp.write(f'f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n')

    def save_result(self, s2_output):
        """Save summary of Stage2Output (no-op placeholder for interface parity)."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor_to_hwc_uint8(self, t: torch.Tensor) -> np.ndarray:
        if t is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        t = t.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        if t.dim() == 3 and t.shape[-1] == 1:
            t = t.repeat(1, 1, 3)
        if t.shape[-1] == 4:
            t = t[:, :, :3]
        arr = t.float().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        return arr.astype(np.uint8)

    def _draw_landmarks(self, img: np.ndarray, lmks: np.ndarray,
                        color: tuple, radius: int = 2):
        """Draw landmark dots on image."""
        S = img.shape[0]
        for x, y in lmks:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < S and 0 <= yi < S:
                cv2.circle(img, (xi, yi), radius, color, -1)

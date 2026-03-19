"""
Debug visualization outputs for Stage 1.
"""

import json
import os

import cv2
import numpy as np


# Color palette for 19-class face parsing visualization
PARSING_COLORS = np.array([
    [0, 0, 0],        # 0: background
    [204, 0, 0],      # 1: skin
    [76, 153, 0],     # 2: l_brow
    [204, 204, 0],    # 3: r_brow
    [51, 51, 255],    # 4: l_eye
    [204, 0, 204],    # 5: r_eye
    [0, 255, 255],    # 6: eye_g
    [255, 204, 204],  # 7: l_ear
    [102, 51, 0],     # 8: r_ear
    [255, 0, 0],      # 9: ear_r
    [102, 204, 0],    # 10: nose
    [255, 255, 0],    # 11: mouth
    [0, 0, 153],      # 12: u_lip
    [0, 0, 204],      # 13: l_lip
    [255, 51, 153],   # 14: neck
    [0, 204, 204],    # 15: neck_l
    [0, 51, 0],       # 16: cloth
    [255, 153, 51],   # 17: hair
    [0, 204, 0],      # 18: hat
], dtype=np.uint8)


class Stage1Visualizer:
    """Saves debug visualizations for each Stage 1 step."""

    def __init__(self, output_dir: str, subject_name: str):
        self.base_dir = os.path.join(output_dir, subject_name, 'stage1')
        self.dirs = {}
        for name in ['01_detection', '02_alignment', '03_segmentation',
                      '04_mica', '05_deca', '06_merged']:
            d = os.path.join(self.base_dir, name)
            os.makedirs(d, exist_ok=True)
            self.dirs[name] = d

    def save_detection(
        self,
        image_rgb: np.ndarray,
        lmks_dense: np.ndarray,
        lmks_68: np.ndarray,
        retinaface_kps: np.ndarray | None,
    ):
        """Save detection debug images."""
        d = self.dirs['01_detection']
        cv2.imwrite(
            os.path.join(d, 'input.png'),
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        )

        # MediaPipe 478 landmarks
        vis = image_rgb.copy()
        for pt in lmks_dense:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        cv2.imwrite(
            os.path.join(d, 'mediapipe_478.png'),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )

        # 68 landmarks with indices
        vis = image_rgb.copy()
        for i, pt in enumerate(lmks_68):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            if i % 5 == 0:
                cv2.putText(vis, str(i), (int(pt[0]) + 3, int(pt[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imwrite(
            os.path.join(d, 'landmarks_68.png'),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )

        # Save 68 landmarks as npy
        np.save(os.path.join(d, 'landmarks_68.npy'), lmks_68)

        # RetinaFace 5 points
        if retinaface_kps is not None:
            vis = image_rgb.copy()
            labels = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']
            for i, pt in enumerate(retinaface_kps):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
                cv2.putText(vis, labels[i], (int(pt[0]) + 5, int(pt[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.imwrite(
                os.path.join(d, 'retinaface_5pt.png'),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

    def save_alignment(
        self,
        original_image: np.ndarray,
        aligned_image: np.ndarray,
        lmks_68_before: np.ndarray,
    ):
        """Save alignment debug images."""
        d = self.dirs['02_alignment']
        cv2.imwrite(
            os.path.join(d, 'aligned_512.png'),
            cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR),
        )

        # Side-by-side comparison
        h = max(original_image.shape[0], aligned_image.shape[0])
        orig_resized = cv2.resize(original_image, (int(original_image.shape[1] * h / original_image.shape[0]), h))
        aligned_resized = cv2.resize(aligned_image, (int(aligned_image.shape[1] * h / aligned_image.shape[0]), h))

        # Draw landmarks on original
        vis_orig = orig_resized.copy()
        scale_x = orig_resized.shape[1] / original_image.shape[1]
        scale_y = orig_resized.shape[0] / original_image.shape[0]
        for pt in lmks_68_before:
            cv2.circle(vis_orig, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 2, (0, 255, 0), -1)

        grid = np.concatenate([vis_orig, aligned_resized], axis=1)
        cv2.imwrite(
            os.path.join(d, 'alignment_grid.png'),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )

    def save_segmentation(
        self,
        aligned_image: np.ndarray,
        parsing: np.ndarray,
        face_mask: np.ndarray,
    ):
        """Save segmentation debug images."""
        d = self.dirs['03_segmentation']

        # 19-class colored parsing
        parsing_vis = PARSING_COLORS[parsing]
        cv2.imwrite(os.path.join(d, 'parsing_vis.png'), parsing_vis[:, :, ::-1])

        # Binary face mask
        mask_img = (face_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(d, 'face_mask.png'), mask_img)

        # Mask overlay
        overlay = aligned_image.copy()
        overlay[~face_mask] = (overlay[~face_mask] * 0.3).astype(np.uint8)
        cv2.imwrite(
            os.path.join(d, 'mask_overlay.png'),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    def save_mica(
        self,
        arcface_img: np.ndarray,
        shape_code: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray | None,
    ):
        """Save MICA debug outputs."""
        d = self.dirs['04_mica']

        # ArcFace 112 input (BGR)
        cv2.imwrite(os.path.join(d, 'arcface_112.png'), arcface_img)

        # Shape code
        np.save(os.path.join(d, 'shape_code.npy'), shape_code)

        # Mesh OBJ (vertices in mm)
        if vertices is not None and faces is not None:
            verts_mm = vertices * 1000  # meters → mm
            _save_obj(os.path.join(d, 'mesh.obj'), verts_mm, faces)

    def save_deca(self, deca_crop: np.ndarray, deca_params: dict):
        """Save DECA debug outputs."""
        d = self.dirs['05_deca']

        # DECA crop
        cv2.imwrite(
            os.path.join(d, 'deca_crop_224.png'),
            cv2.cvtColor(deca_crop, cv2.COLOR_RGB2BGR),
        )

        # Parameters as JSON
        params_json = {}
        for k, v in deca_params.items():
            if hasattr(v, 'numpy'):
                params_json[k] = v.detach().cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                params_json[k] = v.tolist()
        with open(os.path.join(d, 'params.json'), 'w') as f:
            json.dump(params_json, f, indent=2)

    def save_merged(
        self,
        flame_params: dict,
        vertices: np.ndarray | None,
        faces: np.ndarray | None,
        arcface_feat: np.ndarray | None,
        input_image: np.ndarray | None,
    ):
        """Save merged output debug files."""
        d = self.dirs['06_merged']

        # Save FLAME parameters as npz
        params_np = {}
        for k, v in flame_params.items():
            if hasattr(v, 'numpy'):
                params_np[k] = v.detach().cpu().numpy()
            else:
                params_np[k] = np.array(v)
        np.savez(os.path.join(d, 'flame_params.npz'), **params_np)

        # Final mesh
        if vertices is not None and faces is not None:
            verts_mm = vertices * 1000  # meters → mm
            _save_obj(os.path.join(d, 'mesh_final.obj'), verts_mm, faces)

    def save_summary(
        self,
        input_image: np.ndarray,
        lmk_image: np.ndarray | None,
        aligned_image: np.ndarray,
        mask_overlay: np.ndarray | None,
    ):
        """Save 6-panel summary image."""
        size = 256
        panels = []

        def _resize(img):
            return cv2.resize(img, (size, size))

        panels.append(_resize(input_image))

        if lmk_image is not None:
            panels.append(_resize(lmk_image))
        else:
            panels.append(_resize(input_image))

        panels.append(_resize(aligned_image))

        if mask_overlay is not None:
            panels.append(_resize(mask_overlay))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        # Placeholder for MICA mesh and final overlay (would need rendering)
        panels.append(np.zeros((size, size, 3), dtype=np.uint8))
        panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        row1 = np.concatenate(panels[:3], axis=1)
        row2 = np.concatenate(panels[3:], axis=1)
        summary = np.concatenate([row1, row2], axis=0)

        cv2.imwrite(
            os.path.join(self.base_dir, 'summary.png'),
            cv2.cvtColor(summary, cv2.COLOR_RGB2BGR),
        )


def _save_obj(filepath: str, vertices: np.ndarray, faces: np.ndarray):
    """Save a simple OBJ mesh file.

    Args:
        filepath: Output path
        vertices: [N, 3] vertex positions
        faces: [M, 3] face indices (0-based)
    """
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            # OBJ uses 1-based indexing
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

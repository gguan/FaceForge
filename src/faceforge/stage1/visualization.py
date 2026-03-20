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

    def __init__(self, output_dir: str, subject_name: str, image_name: str | None = None):
        if image_name:
            self.base_dir = os.path.join(output_dir, subject_name, 'stage1', image_name)
        else:
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
        """Save alignment debug images.

        Outputs:
            - aligned_512.png: clean aligned crop
            - aligned_annotated.png: aligned crop with landmarks and bbox overlay
            - alignment_grid.png: side-by-side original (with bbox+lmks) vs aligned
        """
        d = self.dirs['02_alignment']

        # Clean aligned image
        cv2.imwrite(
            os.path.join(d, 'aligned_512.png'),
            cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR),
        )

        # Compute face bounding box from landmarks (on original image)
        lmks = lmks_68_before[:, :2]
        x_min, y_min = lmks.min(axis=0)
        x_max, y_max = lmks.max(axis=0)
        # Add padding (20% of bbox size)
        bw, bh = x_max - x_min, y_max - y_min
        pad_x, pad_y = bw * 0.2, bh * 0.2
        bbox = [
            max(0, int(x_min - pad_x)),
            max(0, int(y_min - pad_y)),
            min(original_image.shape[1], int(x_max + pad_x)),
            min(original_image.shape[0], int(y_max + pad_y)),
        ]

        # Annotated original: landmarks + bbox, cropped to face region
        vis_orig = original_image.copy()
        cv2.rectangle(vis_orig, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 200, 255), 2)
        for i, pt in enumerate(lmks):
            cv2.circle(vis_orig, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        # Crop to bbox for the annotated view
        crop_orig = vis_orig[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Annotated aligned image: re-detect landmarks in aligned space isn't trivial,
        # so we just save the clean aligned + the annotated original crop
        cv2.imwrite(
            os.path.join(d, 'original_annotated.png'),
            cv2.cvtColor(crop_orig, cv2.COLOR_RGB2BGR),
        )

        # Save bbox and landmarks as metadata
        np.savez(
            os.path.join(d, 'alignment_meta.npz'),
            lmks_68=lmks_68_before,
            bbox=np.array(bbox),
        )

        # Side-by-side grid: cropped original (with annotations) vs aligned
        size = aligned_image.shape[0]  # 512
        crop_resized = cv2.resize(crop_orig, (size, size))
        grid = np.concatenate([crop_resized, aligned_image], axis=1)
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
        aligned_image: np.ndarray,
        parsing: np.ndarray | None,
        lmks_68: np.ndarray | None,
        vertices: np.ndarray | None,
        faces: np.ndarray | None,
        lmks_3d_68: np.ndarray | None = None,
    ) -> np.ndarray:
        """Save a horizontal summary strip (flame-head-tracker / pixel3dmm style).

        Panels: aligned | segmentation | landmarks overlay | mesh wireframe overlay

        Args:
            aligned_image: [H, W, 3] RGB uint8 aligned crop
            parsing: [H, W] int32 19-class parsing map (optional)
            lmks_68: [68, 2] landmarks in aligned image coords (optional)
            vertices: [N, 3] posed FLAME mesh vertices in meters (optional)
            faces: [M, 3] FLAME mesh face indices (optional)
            lmks_3d_68: [68, 3] FLAME 3D landmark positions (optional, for camera fitting)

        Returns:
            The summary strip as RGB uint8 [size, size*4, 3]
        """
        size = 256
        panels = []

        def _resize(img):
            return cv2.resize(img, (size, size))

        # Panel 1: Aligned face
        panels.append(_resize(aligned_image))

        # Panel 2: Segmentation parsing (colored)
        if parsing is not None:
            parsing_vis = PARSING_COLORS[parsing]
            panels.append(_resize(parsing_vis))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        # Panel 3: Landmarks on aligned image
        if lmks_68 is not None:
            lmk_vis = aligned_image.copy()
            for i, pt in enumerate(lmks_68):
                cv2.circle(lmk_vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            # Draw contour connections
            contours = [
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
            for contour in contours:
                for j in range(len(contour) - 1):
                    p1 = lmks_68[contour[j]]
                    p2 = lmks_68[contour[j + 1]]
                    cv2.line(lmk_vis, (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
                # Close loops for eyes and lips
                if contour[0] in (36, 42, 48, 60):
                    p1 = lmks_68[contour[-1]]
                    p2 = lmks_68[contour[0]]
                    cv2.line(lmk_vis, (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
            panels.append(_resize(lmk_vis))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        # Panel 4: Mesh wireframe overlay (posed FLAME mesh)
        if vertices is not None and faces is not None:
            mesh_vis = aligned_image.copy()
            if lmks_3d_68 is not None and lmks_68 is not None:
                # Use landmark-fitted orthographic projection for proper alignment
                pts_2d = _project_vertices_fitted(
                    vertices, lmks_3d_68, lmks_68, aligned_image.shape[0],
                )
            else:
                # Fallback to naive fit-to-bounds projection
                pts_2d = _project_vertices_ortho(vertices, aligned_image.shape[0])
            _draw_wireframe(mesh_vis, pts_2d, faces, color=(0, 255, 255), thickness=1)
            panels.append(_resize(mesh_vis))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        summary = np.concatenate(panels, axis=1)

        cv2.imwrite(
            os.path.join(self.base_dir, 'summary.png'),
            cv2.cvtColor(summary, cv2.COLOR_RGB2BGR),
        )
        return summary


def save_summary_grid(
    summaries: list[np.ndarray],
    output_path: str,
):
    """Vertically concatenate per-image summary strips into one grid image.

    Args:
        summaries: List of RGB uint8 summary strips (same width)
        output_path: Where to save the combined image
    """
    if not summaries:
        return
    grid = np.concatenate(summaries, axis=0)
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _project_vertices_fitted(
    vertices: np.ndarray,
    lmks_3d: np.ndarray,
    lmks_2d: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Project FLAME vertices using orthographic camera fitted to 2D landmarks.

    Solves for [scale, tx, ty] that best maps FLAME 3D landmarks to detected
    2D landmarks in the aligned image, following the same orthographic model
    as DECA (batch_orth_proj): u = scale * x + tx, v = scale * (-y) + ty.

    Args:
        vertices: [N, 3] FLAME mesh vertices in meters
        lmks_3d: [68, 3] FLAME 3D landmark positions in meters
        lmks_2d: [68, 2] detected 2D landmarks in aligned image pixel coords
        image_size: aligned image dimension (square)

    Returns:
        [N, 2] projected 2D coordinates in pixel space
    """
    # FLAME: x=right, y=up, z=forward
    # Image: u=right, v=down
    # Orthographic model: u = scale * x + tx, v = scale * (-y) + ty
    #
    # Use only interior landmarks (17-67: brows, eyes, nose, mouth) for fitting.
    # Jawline landmarks (0-16) use dynamic contour in 2D detection but static
    # positions in FLAME's seletec_3d68, causing mismatch at non-frontal poses.
    interior = slice(17, 68)
    x_3d = lmks_3d[interior, 0]
    y_3d = -lmks_3d[interior, 1]  # flipped y

    u_2d = lmks_2d[interior, 0]
    v_2d = lmks_2d[interior, 1]

    # Solve least-squares: [scale, tx] from u = scale * x_3d + tx
    #                      [scale, ty] from v = scale * y_3d + ty
    # Combined system: minimize || A @ [scale, tx, ty]^T - b ||^2
    n = len(x_3d)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)

    A[:n, 0] = x_3d       # scale * x_3d
    A[:n, 1] = 1.0        # tx
    A[n:, 0] = y_3d       # scale * (-y_3d)
    A[n:, 2] = 1.0        # ty
    b[:n] = u_2d
    b[n:] = v_2d

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, tx, ty = params

    # Apply to all vertices
    vx = vertices[:, 0]
    vy = -vertices[:, 1]
    pts_2d = np.stack([
        scale * vx + tx,
        scale * vy + ty,
    ], axis=1)
    return pts_2d


def _project_vertices_ortho(
    vertices: np.ndarray,
    image_size: int,
    padding: float = 0.1,
) -> np.ndarray:
    """Orthographic projection of FLAME vertices to 2D image coordinates.

    FLAME canonical vertices: x=right, y=up, z=forward, units=meters.
    Projects by taking (x, -y) and scaling to fit image_size with padding.

    Args:
        vertices: [N, 3] FLAME mesh vertices in meters
        image_size: Target image dimension (square)
        padding: Fraction of image to leave as border

    Returns:
        [N, 2] projected 2D coordinates in pixel space
    """
    x = vertices[:, 0]
    y = -vertices[:, 1]  # flip y (FLAME y-up → image y-down)

    # Scale to fit image with padding
    xy = np.stack([x, y], axis=1)
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = (xy_max - xy_min).max()

    usable = image_size * (1.0 - 2 * padding)
    scale = usable / xy_range
    offset = image_size / 2.0 - (xy_min + xy_max) / 2.0 * scale

    pts_2d = xy * scale + offset
    return pts_2d


def _draw_wireframe(
    image: np.ndarray,
    pts_2d: np.ndarray,
    faces: np.ndarray,
    color: tuple = (0, 255, 255),
    thickness: int = 1,
    max_edge_len: float = 50.0,
):
    """Draw mesh wireframe edges on image.

    Skips edges longer than max_edge_len pixels to avoid stray lines
    from back-facing or boundary triangles.

    Args:
        image: [H, W, 3] image to draw on (modified in-place)
        pts_2d: [N, 2] projected vertex positions
        faces: [M, 3] triangle face indices (0-based)
        color: BGR/RGB line color
        thickness: Line thickness
        max_edge_len: Skip edges longer than this (pixels)
    """
    # Collect unique edges to avoid drawing duplicates
    edges = set()
    for f in faces:
        for i in range(3):
            e = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
            edges.add(e)

    for i, j in edges:
        p1 = pts_2d[i]
        p2 = pts_2d[j]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if dx * dx + dy * dy > max_edge_len * max_edge_len:
            continue
        cv2.line(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color,
            thickness,
            cv2.LINE_AA,
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

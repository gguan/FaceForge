"""
Debug visualization outputs for Stage 1.
"""

import json
import os

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


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

# DECA-style multi-directional lighting for mesh rendering
# 5 lights with intensity 1.7, same as DECA's render_shape()
DECA_LIGHT_DIRS = [
    [-1, 1, 1],
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, 1],
    [0, 0, 1],
]
DECA_LIGHT_INTENSITY = 1.7
DECA_MESH_COLOR = 180.0 / 255.0  # gray (0.706)


class Stage1Visualizer:
    """Saves debug visualizations for each Stage 1 step."""

    def __init__(self, output_dir: str, subject_name: str, image_name: str | None = None):
        if image_name:
            self.base_dir = os.path.join(output_dir, subject_name, 'stage1', image_name)
        else:
            self.base_dir = os.path.join(output_dir, subject_name, 'stage1')
        self.dirs = {}
        for name in ['01_detection', '02_alignment', '03_segmentation',
                      '04_mica', '05_deca']:
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

    def save_deca(
        self,
        deca_crop: np.ndarray,
        deca_params: dict,
        flame_params: dict | None = None,
        canonical_vertices: np.ndarray | None = None,
        posed_vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
    ):
        """Save DECA outputs, merged FLAME params, and meshes.

        Args:
            deca_crop: [224, 224, 3] uint8 DECA crop image
            deca_params: dict with exp, pose, tex, light (for debug JSON)
            flame_params: merged FLAME parameters dict (optional)
            canonical_vertices: [V, 3] canonical mesh (shape only, no pose/exp)
            posed_vertices: [V, 3] posed mesh (shape + expression + pose)
            faces: [F, 3] FLAME face indices
        """
        d = self.dirs['05_deca']

        # DECA crop
        cv2.imwrite(
            os.path.join(d, 'deca_crop_224.png'),
            cv2.cvtColor(deca_crop, cv2.COLOR_RGB2BGR),
        )

        # DECA raw parameters as JSON
        params_json = {}
        for k, v in deca_params.items():
            if hasattr(v, 'numpy'):
                params_json[k] = v.detach().cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                params_json[k] = v.tolist()
        with open(os.path.join(d, 'params.json'), 'w') as f:
            json.dump(params_json, f, indent=2)

        # Merged FLAME parameters (MICA shape + DECA exp/pose/tex/light)
        if flame_params is not None:
            params_np = {}
            for k, v in flame_params.items():
                if hasattr(v, 'numpy'):
                    params_np[k] = v.detach().cpu().numpy()
                else:
                    params_np[k] = np.array(v)
            np.savez(os.path.join(d, 'flame_params.npz'), **params_np)

        # Canonical mesh (shape only, identity pose, zero expression)
        if canonical_vertices is not None and faces is not None:
            verts_mm = canonical_vertices * 1000  # meters → mm
            _save_obj(os.path.join(d, 'mesh_canonical.obj'), verts_mm, faces)

        # Posed mesh (shape + expression + pose)
        if posed_vertices is not None and faces is not None:
            verts_mm = posed_vertices * 1000  # meters → mm
            _save_obj(os.path.join(d, 'mesh_posed.obj'), verts_mm, faces)

    def save_summary(
        self,
        aligned_image: np.ndarray,
        parsing: np.ndarray | None,
        lmks_68: np.ndarray | None,
        vertices: np.ndarray | None,
        faces: np.ndarray | None,
        flame_lmks_3d: np.ndarray | None = None,
        device: str | None = None,
        deca_cam: np.ndarray | None = None,
        deca_crop_tform: np.ndarray | None = None,
        align_M: np.ndarray | None = None,
    ) -> np.ndarray:
        """Save a horizontal summary strip (flame-head-tracker / pixel3dmm style).

        Panels: aligned | segmentation | landmarks overlay | mesh overlay

        Args:
            aligned_image: [H, W, 3] RGB uint8 aligned crop
            parsing: [H, W] int32 19-class parsing map (optional)
            lmks_68: [68, 2] landmarks in aligned image coords (optional)
            vertices: [N, 3] posed FLAME mesh vertices in meters (optional)
            faces: [M, 3] FLAME mesh face indices (optional)
            flame_lmks_3d: [68, 3] FLAME 3D landmarks (for projection fitting)
            device: torch device string for PyTorch3D rendering (optional)
            deca_cam: [1, 3] DECA orthographic camera [scale, tx, ty]
            deca_crop_tform: [3, 3] original image -> DECA crop transform
            align_M: [3, 3] original image -> aligned image transform

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
        #   Green = MediaPipe detected 2D landmarks
        #   Red   = FLAME 3D landmarks projected to aligned image (diagnostic)
        if lmks_68 is not None:
            lmk_vis = aligned_image.copy()
            # Draw detected landmarks (green)
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

        # Panel 4: Phong-shaded mesh aligned to the image via DECA projection
        if vertices is not None and faces is not None and device is not None:
            img_size = aligned_image.shape[0]
            pts_2d = None
            if deca_cam is not None and deca_crop_tform is not None and align_M is not None:
                pts_2d = _project_vertices_deca(
                    vertices, deca_cam, deca_crop_tform, align_M,
                )
            elif flame_lmks_3d is not None and lmks_68 is not None:
                pts_2d = _project_vertices_similarity(
                    vertices, flame_lmks_3d, lmks_68,
                )
            if pts_2d is not None:
                mesh_vis = _render_mesh_aligned(
                    vertices, faces, pts_2d, img_size, device,
                )
            else:
                mesh_vis = _render_mesh_front_view(vertices, faces, device)
            panels.append(_resize(mesh_vis))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        summary = np.concatenate(panels, axis=1)

        cv2.imwrite(
            os.path.join(self.base_dir, 'summary.png'),
            cv2.cvtColor(summary, cv2.COLOR_RGB2BGR),
        )
        return summary


@torch.no_grad()
def _render_mesh_aligned(
    vertices: np.ndarray,
    faces: np.ndarray,
    pts_2d: np.ndarray,
    image_size: int,
    device: str,
) -> np.ndarray:
    """Render mesh with PyTorch3D, aligned to the image via DECA projection.

    Uses pts_2d (DECA-projected pixel positions) to fit an OrthographicCameras
    transform, then renders with SoftPhongShader for proper lighting.

    Args:
        vertices: [V, 3] FLAME vertices in meters
        faces: [F, 3] face indices
        pts_2d: [V, 2] projected pixel coords in aligned image
        image_size: aligned image dimension (e.g. 512)
        device: torch device string

    Returns:
        rendered_img [image_size, image_size, 3] uint8 RGB
    """
    verts_t = torch.from_numpy(vertices).float().to(device)
    faces_world_t = torch.from_numpy(faces).long().to(device)
    # x-flip + y-flip = even number of flips → handedness preserved, no winding change
    faces_t = faces_world_t

    # Convert pts_2d (pixels) to PyTorch3D NDC: [-1, 1], y-up
    # pixels: (0,0)=top-left, (img,img)=bottom-right
    # NDC:    (-1,-1)=bottom-left, (1,1)=top-right
    # PyTorch3D screen: x-left, y-up; pixels: x-right, y-down
    ndc_x = -(pts_2d[:, 0] / image_size * 2.0 - 1.0)  # flip x
    ndc_y = -(pts_2d[:, 1] / image_size * 2.0 - 1.0)   # flip y

    # Fit 2D affine: world (x, y) → NDC (ndc_x, ndc_y)
    # Using least-squares on FLAME xy → projected NDC xy
    wx = vertices[:, 0]
    wy = vertices[:, 1]
    # ndc_x ≈ a * wx + b * wy + c
    # ndc_y ≈ d * wx + e * wy + f
    A = np.column_stack([wx, wy, np.ones(len(wx))])
    params_x, _, _, _ = np.linalg.lstsq(A, ndc_x, rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, ndc_y, rcond=None)

    # Construct verts_ndc using the fitted affine + z for depth
    fitted_ndc_x = A @ params_x
    fitted_ndc_y = A @ params_y

    # z depth: scale world z so closer vertices (larger z) have smaller NDC z
    # (PyTorch3D: smaller z = closer to camera)
    z_world = vertices[:, 2]
    z_range = z_world.max() - z_world.min()
    # Map z to [0.1, 1.0] range, with larger world z → smaller NDC z
    ndc_z = 0.1 + 0.9 * (z_world.max() - z_world) / max(z_range, 1e-8)

    verts_ndc = torch.from_numpy(
        np.stack([fitted_ndc_x, fitted_ndc_y, ndc_z], axis=1)
    ).float().to(device)

    # Compute lighting in world space (normals make sense here), bake into vertex colors
    normals = _compute_vertex_normals(verts_t, faces_world_t)
    vertex_color = _compute_deca_vertex_colors(normals, device=device)

    mesh = Meshes(verts=verts_ndc[None], faces=faces_t[None])
    mesh.textures = TexturesVertex(verts_features=vertex_color[None])

    # Identity camera — vertices are already in NDC
    cameras = OrthographicCameras(device=device)

    # Ambient-only: shading is pre-baked in vertex colors
    lights = PointLights(
        device=device, location=[[0.0, 0.0, 0.0]],
        ambient_color=[[1.0, 1.0, 1.0]],
        diffuse_color=[[0.0, 0.0, 0.0]],
        specular_color=[[0.0, 0.0, 0.0]],
    )

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
        cull_backfaces=True,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        ),
    )

    image = renderer(mesh)
    image = (image[0, ..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return image


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



def _blend_overlay_on_mask(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    foreground_mask: np.ndarray,
    base_weight: float = 0.4,
    overlay_weight: float = 0.6,
) -> np.ndarray:
    """Blend overlay only on the rendered foreground region."""
    blended = base_image.copy()
    mask = foreground_mask.astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if not np.any(mask):
        return blended

    base_f = base_image.astype(np.float32)
    overlay_f = overlay_image.astype(np.float32)
    mixed = (base_f * base_weight + overlay_f * overlay_weight).clip(0, 255).astype(np.uint8)
    blended[mask] = mixed[mask]
    return blended


def _enhance_mesh_contours(
    rendered_image: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """Darken only the mesh boundary to improve shape readability."""
    enhanced = rendered_image.copy()
    mask = foreground_mask.astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if not np.any(mask):
        return enhanced

    kernel = np.ones((MESH_EDGE_WIDTH * 2 + 1, MESH_EDGE_WIDTH * 2 + 1), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    contour = mask.astype(bool) & ~eroded.astype(bool)
    if not np.any(contour):
        return enhanced

    enhanced_f = enhanced.astype(np.float32)
    enhanced_f[contour] *= MESH_EDGE_DARKEN
    return enhanced_f.clip(0, 255).astype(np.uint8)


def _project_vertices_similarity(
    vertices: np.ndarray,
    flame_lmks_3d: np.ndarray,
    lmks_2d: np.ndarray,
) -> np.ndarray | None:
    """Project FLAME vertices to 2D by fitting an affine transform.

    Fits a full affine (6 DOF: scale_x, scale_y, rotation, shear, tx, ty)
    from FLAME 3D landmarks (x, -y) to detected 2D landmarks.
    Full affine handles non-frontal faces better than similarity (4 DOF)
    because it can capture perspective foreshortening as anisotropic scaling.

    Args:
        vertices: [V, 3] FLAME mesh vertices in meters
        flame_lmks_3d: [68, 3] FLAME 3D landmark positions
        lmks_2d: [68, 2] detected 2D landmarks in aligned image

    Returns:
        [V, 2] projected 2D pixel coordinates, or None on failure
    """
    # Use interior landmarks only (17-67) to avoid jawline contour mismatch.
    interior = slice(17, 68)

    # FLAME: x=right, y=up → image: x=right, y=down
    src = np.stack([
        flame_lmks_3d[interior, 0],
        -flame_lmks_3d[interior, 1],
    ], axis=1).astype(np.float32)

    dst = lmks_2d[interior, :2].astype(np.float32)

    # Diagnostic: check input ranges
    print(f'[Vis] Affine fit: src range x=[{src[:,0].min():.4f}, {src[:,0].max():.4f}], '
          f'y=[{src[:,1].min():.4f}, {src[:,1].max():.4f}]')
    print(f'[Vis] Affine fit: dst range x=[{dst[:,0].min():.1f}, {dst[:,0].max():.1f}], '
          f'y=[{dst[:,1].min():.1f}, {dst[:,1].max():.1f}]')

    # Full affine (6 DOF) with RANSAC — handles perspective foreshortening
    M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
    if M is None:
        print('[Vis] WARNING: Affine fit FAILED (M is None)')
        return None

    # Diagnostic: check fit quality
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    src_h = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
    projected = (M @ src_h.T).T
    residual = np.sqrt(((projected - dst) ** 2).sum(axis=1)).mean()
    print(f'[Vis] Affine fit: inliers={n_inliers}/{len(src)}, mean_residual={residual:.1f}px, '
          f'M=[[{M[0,0]:.1f}, {M[0,1]:.1f}, {M[0,2]:.1f}], [{M[1,0]:.1f}, {M[1,1]:.1f}, {M[1,2]:.1f}]]')

    if residual > 20.0:
        print(f'[Vis] WARNING: High residual ({residual:.1f}px) — projection may be inaccurate')

    # Apply to all vertices
    verts_xy = np.stack([vertices[:, 0], -vertices[:, 1]], axis=1).astype(np.float32)
    ones = np.ones((verts_xy.shape[0], 1), dtype=np.float32)
    verts_h = np.concatenate([verts_xy, ones], axis=1)  # [V, 3]
    pts_2d = (M @ verts_h.T).T  # [V, 2]
    return pts_2d


def _compute_ndc_from_pixels_direct(
    vertices: np.ndarray,
    pts_2d: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Convert projected 2D pixels + FLAME z to NDC for PyTorch3D rendering.

    Uses the fitted similarity transform scale for z-depth computation.

    Args:
        vertices: [V, 3] FLAME world-space vertices
        pts_2d: [V, 2] projected pixel coordinates in aligned image
        image_size: aligned image dimension

    Returns:
        [V, 3] NDC coordinates
    """
    ndc_x = pts_2d[:, 0] / image_size * 2.0 - 1.0
    # Flip y: pixel y-down → NDC y-up
    ndc_y = -(pts_2d[:, 1] / image_size * 2.0 - 1.0)
    # Estimate effective scale from the projection (pixels per meter)
    verts_x_range = vertices[:, 0].max() - vertices[:, 0].min()
    pts_x_range = pts_2d[:, 0].max() - pts_2d[:, 0].min()
    eff_scale = pts_x_range / max(verts_x_range, 1e-8)
    # z: PyTorch3D requires z > 0 for visible geometry.
    # Use FLAME z for relative depth ordering, shifted to be all-positive.
    z_scaled = eff_scale * vertices[:, 2] / image_size * 2.0
    ndc_z = z_scaled - z_scaled.min() + 1.0  # ensure all z > 0
    return np.stack([ndc_x, ndc_y, ndc_z], axis=1)


def _compute_vertex_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute per-vertex normals by accumulating face normals.

    Ported from flame-head-tracker utils/graphics_utils.py.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] face indices

    Returns:
        [V, 3] unit-length vertex normals
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals


def _compute_deca_vertex_colors(
    normals: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute per-vertex shading using DECA's multi-directional lighting.

    Faithfully ported from DECA's add_directionlight() + render_shape():
    - 5 directional lights, each with per-channel intensity 1.7
    - Per-light: shading_l = clamp(n·l, 0, 1) * intensity  → [V, 3]
    - Final shading = mean over 5 lights                    → [V, 3]
    - vertex_color = albedo * shading
    - No normal flipping (DECA handles backfaces via pos_mask on pixels)
    """
    n_verts = normals.shape[0]

    # Base gray color: DECA uses RGB(180, 180, 180) / 255
    albedo = torch.full((n_verts, 3), DECA_MESH_COLOR, device=device)

    # 5 directional lights — exactly as DECA's render_shape()
    light_dirs = torch.tensor(DECA_LIGHT_DIRS, dtype=torch.float32, device=device)  # [5, 3]
    light_dirs = torch.nn.functional.normalize(light_dirs, dim=1)
    light_int = torch.full((5, 3), DECA_LIGHT_INTENSITY, device=device)  # [5, 3]

    # Per-light Lambert: n_dot_l [V, 5], then broadcast with intensity [5, 3]
    # shading_per_light = n_dot_l[:, :, None] * intensity[None, :, :]  → [V, 5, 3]
    n_dot_l = torch.clamp(
        torch.einsum('vc,lc->vl', normals, light_dirs), min=0.0, max=1.0,
    )  # [V, 5]
    shading = (n_dot_l[:, :, None] * light_int[None, :, :]).mean(dim=1)  # [V, 3]

    vertex_color = torch.clamp(albedo * shading, 0.0, 1.0)
    return vertex_color


@torch.no_grad()
def _render_mesh_front_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str,
    render_size: int = 512,
) -> np.ndarray:
    """Render Phong-shaded mesh from the front using PyTorch3D.

    Uses FoVPerspectiveCameras + look_at_view_transform for a clean
    front-view render, matching the approach in test_render_mesh.py.

    Args:
        vertices: [V, 3] FLAME vertices in meters
        faces: [F, 3] face indices
        device: torch device string
        render_size: output image size

    Returns:
        rendered_img [H, W, 3] uint8 RGB
    """
    # Convert meters → mm for camera distance calculation
    verts_mm = vertices * 1000.0
    verts_t = torch.from_numpy(verts_mm).float().to(device)
    faces_t = torch.from_numpy(faces).long().to(device)

    # Gray material
    gray_color = torch.ones_like(verts_t)[None] * 0.7
    mesh = Meshes(verts=verts_t[None], faces=faces_t[None])
    mesh.textures = TexturesVertex(verts_features=gray_color)

    # Camera: look at face center from front, framed like aligned image.
    # Use front-facing vertices (z > median) to find face center,
    # then set distance so the face span fills ~70% of the frame.
    z_med = verts_t[:, 2].median()
    front_mask = verts_t[:, 2] > z_med
    face_verts = verts_t[front_mask]
    face_center = face_verts.mean(0)
    face_span = (face_verts.max(0).values - face_verts.min(0).values).max().item()

    # FoV → distance = span / (2 * tan(fov/2)) / fill_ratio
    import math
    fov = 30.0
    fill_ratio = 0.85
    cam_dist = face_span / (2.0 * math.tan(math.radians(fov / 2.0))) / fill_ratio

    # Shift look-at point slightly down to match FFHQ-style framing
    # (aligned images center between eyes, with more forehead than chin)
    cx = face_center[0].item()
    cy = face_center[1].item() - face_span * 0.08
    cz = face_center[2].item()
    eye = (cx, cy, cz + cam_dist)
    at = (cx, cy, cz)
    R, T = look_at_view_transform(eye=(eye,), at=(at,), up=((0, 1, 0),))
    cameras = FoVPerspectiveCameras(
        device=device, R=R.to(device), T=T.to(device),
        fov=fov, znear=1.0, zfar=cam_dist * 3,
    )

    # Lighting: in front of face, slightly above
    lights = PointLights(
        device=device,
        location=[[eye[0], eye[1] + face_span * 0.3, eye[2]]],
    )

    raster_settings = RasterizationSettings(
        image_size=render_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        ),
    )

    image = renderer(mesh)
    image = (image[0, ..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return image


def _compute_ndc_from_pixels(
    vertices: np.ndarray,
    pts_2d: np.ndarray,
    deca_scale: float,
    image_size: int,
) -> np.ndarray:
    """Convert projected 2D pixel coords + FLAME z to NDC for PyTorch3D.

    Args:
        vertices: [V, 3] FLAME world-space vertices (need z for depth)
        pts_2d: [V, 2] projected pixel coordinates in aligned image
        deca_scale: DECA camera scale parameter
        image_size: aligned image dimension

    Returns:
        [V, 3] NDC coordinates for PyTorch3D rendering
    """
    ndc_xy = pts_2d / image_size * 2.0 - 1.0
    # z: closer vertices (larger FLAME z) should have smaller NDC z
    ndc_z = -deca_scale * vertices[:, 2]
    return np.stack([ndc_xy[:, 0], ndc_xy[:, 1], ndc_z], axis=1)


def _project_vertices_deca(
    vertices: np.ndarray,
    deca_cam: np.ndarray,
    deca_crop_tform: np.ndarray,
    align_M: np.ndarray,
) -> np.ndarray:
    """Project FLAME vertices to aligned image space via DECA's camera.

    Uses the same orthographic model as DECA (batch_orth_proj), then chains
    through the DECA crop → original image → aligned image transforms.

    Projection chain:
        1. FLAME 3D → NDC via batch_orth_proj: ndc = scale * (xy + [tx, ty]), flip y
        2. NDC [-1, 1] → DECA crop pixels [0, 224]
        3. DECA crop pixels → original image pixels (inverse of crop tform)
        4. Original image pixels → aligned image pixels (align_M)

    Args:
        vertices: [N, 3] FLAME mesh vertices in meters
        deca_cam: [1, 3] orthographic camera [scale, tx, ty]
        deca_crop_tform: [3, 3] similarity transform (original → 224 crop)
        align_M: [3, 3] projective transform (original → aligned image)

    Returns:
        [N, 2] projected 2D coordinates in aligned image pixel space
    """
    scale, tx, ty = deca_cam[0, 0], deca_cam[0, 1], deca_cam[0, 2]

    # Step 1: batch_orth_proj — translate then scale, same as DECA
    # ndc_x = scale * (x + tx)
    # ndc_y = scale * (y + ty)
    xy = vertices[:, :2].copy()
    xy[:, 0] += tx
    xy[:, 1] += ty
    ndc = xy * scale

    # Flip y (and z, though we only use x,y) — same as DECA's decode:
    #   trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
    ndc[:, 1] = -ndc[:, 1]

    # Step 2: NDC [-1, 1] → DECA crop pixels [0, 224]
    crop_size = 224.0
    crop_px = (ndc + 1.0) / 2.0 * crop_size  # [N, 2]

    # Step 3: DECA crop pixels → original image pixels
    # deca_crop_tform maps original → crop, so we need its inverse
    crop_to_orig = np.linalg.inv(deca_crop_tform)  # [3, 3]

    # Step 4: original → aligned image
    # align_M is a 3x3 projective matrix

    # Compose: crop_pixels → original → aligned
    # We can compose crop_to_orig and align_M into a single matrix
    composed = align_M @ crop_to_orig  # [3, 3]

    # Apply composed transform to crop pixel coordinates
    N = crop_px.shape[0]
    ones = np.ones((N, 1))
    crop_h = np.concatenate([crop_px, ones], axis=1)  # [N, 3]
    aligned_h = (composed @ crop_h.T).T  # [N, 3]

    # Perspective divide
    pts_2d = aligned_h[:, :2] / aligned_h[:, 2:3]
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

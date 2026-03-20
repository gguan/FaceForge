"""
Debug visualization outputs for Stage 1.
"""

import json
import os

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
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
        flame_lmks_3d: np.ndarray | None = None,
        device: str | None = None,
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
        #   Red   = FLAME 3D landmarks projected via DECA cam (diagnostic)
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
            # Diagnostic: project FLAME 3D landmarks via similarity fit (red)
            # Red dots = FLAME landmarks projected to aligned image
            # Green dots = MediaPipe detected landmarks
            # Overlap means projection is correct
            if flame_lmks_3d is not None:
                flame_lmks_2d = _project_vertices_similarity(
                    flame_lmks_3d, flame_lmks_3d, lmks_68,
                )
                if flame_lmks_2d is not None:
                    for pt in flame_lmks_2d:
                        cv2.circle(lmk_vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
            panels.append(_resize(lmk_vis))
        else:
            panels.append(np.zeros((size, size, 3), dtype=np.uint8))

        # Panel 4: Solid mesh overlay (flame-head-tracker style Phong shading)
        if vertices is not None and faces is not None:
            img_size = aligned_image.shape[0]
            can_render = (
                flame_lmks_3d is not None
                and lmks_68 is not None
                and device is not None
            )
            if can_render:
                # Direct projection: fit similarity transform from FLAME 3D
                # landmarks to detected 2D landmarks, apply to all vertices
                pts_2d = _project_vertices_similarity(
                    vertices, flame_lmks_3d, lmks_68,
                )
                if pts_2d is not None:
                    verts_ndc = _compute_ndc_from_pixels_direct(
                        vertices, pts_2d, img_size,
                    )
                    rendered, _ = _render_mesh_phong(
                        vertices, verts_ndc, faces, device, render_size=img_size,
                    )
                    mesh_vis = cv2.addWeighted(aligned_image, 0.4, rendered, 0.6, 0)
                else:
                    mesh_vis = aligned_image.copy()
            else:
                # Fallback: wireframe
                mesh_vis = aligned_image.copy()
                pts_2d = _project_vertices_ortho(vertices, img_size)
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



def _project_vertices_similarity(
    vertices: np.ndarray,
    flame_lmks_3d: np.ndarray,
    lmks_2d: np.ndarray,
) -> np.ndarray | None:
    """Project FLAME vertices to 2D by fitting a similarity transform.

    Fits scale + rotation + translation from FLAME 3D landmarks (x, -y)
    to detected 2D landmarks using cv2.estimateAffinePartial2D (RANSAC).
    Then applies the same transform to all mesh vertices.

    This is the simplest correct approach: no DECA camera chain needed,
    just direct 3D-landmark-to-2D-landmark correspondence.

    Args:
        vertices: [V, 3] FLAME mesh vertices in meters
        flame_lmks_3d: [68, 3] FLAME 3D landmark positions
        lmks_2d: [68, 2] detected 2D landmarks in aligned image

    Returns:
        [V, 2] projected 2D pixel coordinates, or None on failure
    """
    # Use interior landmarks only (17-67) to avoid jawline contour mismatch.
    # FLAME's seletec_3d68 gives static jawline positions, but 2D detectors
    # use dynamic contour (visible silhouette), causing mismatch at non-frontal.
    interior = slice(17, 68)

    # FLAME: x=right, y=up → image: x=right, y=down
    src = np.stack([
        flame_lmks_3d[interior, 0],
        -flame_lmks_3d[interior, 1],
    ], axis=1).astype(np.float32)

    dst = lmks_2d[interior, :2].astype(np.float32)

    # estimateAffinePartial2D: 4 DOF (scale, rotation, tx, ty) with RANSAC
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        return None

    # Apply to all vertices
    verts_xy = np.stack([vertices[:, 0], -vertices[:, 1]], axis=1).astype(np.float32)
    # M is [2, 3]: dst = M @ [x, y, 1]^T
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
    ndc_xy = pts_2d / image_size * 2.0 - 1.0
    # Estimate effective scale from the projection (pixels per meter)
    # Use the range of projected x to estimate
    verts_x_range = vertices[:, 0].max() - vertices[:, 0].min()
    pts_x_range = pts_2d[:, 0].max() - pts_2d[:, 0].min()
    eff_scale = pts_x_range / max(verts_x_range, 1e-8)
    # z: closer vertices (larger FLAME z) → smaller NDC z
    ndc_z = -eff_scale * vertices[:, 2] / image_size * 2.0
    return np.stack([ndc_xy[:, 0], ndc_xy[:, 1], ndc_z], axis=1)


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
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals


@torch.no_grad()
def _render_mesh_phong(
    vertices: np.ndarray,
    verts_ndc: np.ndarray,
    faces: np.ndarray,
    device: str,
    render_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Render solid Phong-shaded mesh via PyTorch3D.

    Ported from flame-head-tracker utils/general_utils.py render_geometry().

    Args:
        vertices: [V, 3] world-space FLAME vertices (for normal computation)
        verts_ndc: [V, 3] NDC coordinates (for rasterization position)
        faces: [F, 3] face indices
        device: torch device string (e.g. 'cuda:0')
        render_size: output image size

    Returns:
        (rendered_img [H, W, 3] uint8 RGB, foreground_mask [H, W] uint8)
    """
    faces_t = torch.from_numpy(faces).to(device)
    verts_t = torch.from_numpy(vertices).float().to(device)
    verts_ndc_t = torch.from_numpy(verts_ndc).float().to(device)

    # Compute vertex normals from world-space geometry
    normals = _compute_vertex_normals(verts_t, faces_t)

    # Per-vertex Phong shading (same as flame-head-tracker)
    light_pos = torch.tensor([0.0, 0.0, 10.0], device=device)
    camera_pos = torch.tensor([0.0, 0.0, 10.0], device=device)
    ambient_color = torch.tensor([0.1, 0.1, 0.1], device=device)
    diffuse_color = torch.tensor([0.8, 0.8, 0.8], device=device)
    base_color = torch.ones_like(verts_t)

    light_dir = light_pos - verts_t
    light_dir = light_dir / (light_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # Ambient + diffuse (no specular, same as flame-head-tracker)
    ambient = ambient_color * base_color
    diff = torch.clamp((normals * light_dir).sum(-1, keepdim=True), min=0.0)
    diffuse = diffuse_color * base_color * diff
    vertex_color = torch.clamp(ambient + diffuse, 0.0, 1.0)

    # Build PyTorch3D mesh with baked vertex colors
    mesh = Meshes(verts=verts_ndc_t[None], faces=faces_t[None])
    mesh.textures = TexturesVertex(verts_features=vertex_color[None])

    # OrthographicCameras: identity with x,y flipped (same as flame-head-tracker)
    R = torch.eye(3, device=device)
    R[0, 0] = -1.0
    R[1, 1] = -1.0
    cameras = OrthographicCameras(device=device, R=R[None], T=torch.zeros(1, 3, device=device))

    raster_settings = RasterizationSettings(
        image_size=512, blur_radius=0.0, faces_per_pixel=1,
    )
    # Full ambient lights (shading already baked into vertex colors)
    lights = PointLights(
        device=device, location=[[0.0, 0.0, 0.0]],
        ambient_color=[[1.0, 1.0, 1.0]],
        diffuse_color=[[0.0, 0.0, 0.0]],
        specular_color=[[0.0, 0.0, 0.0]],
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    image = renderer(mesh)
    alpha = image[0, ..., 3].cpu().numpy()
    foreground_mask = (alpha > 0).astype(np.uint8)
    image = (image[0, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    image = cv2.resize(image, (render_size, render_size))
    foreground_mask = cv2.resize(foreground_mask, (render_size, render_size))
    return image, foreground_mask


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

"""
REALY benchmark evaluation: FLAME mesh → HIFI3D++ topology conversion and regional error metrics.

Implements the full REALY evaluation pipeline:
1. Extract 85 keypoints from FLAME mesh using barycentric coordinates
2. Global alignment using 7-point Procrustes
3. Regional ICP alignment + NMSE computation for nose/mouth/forehead/cheek
"""

import os
import json

import numpy as np
import trimesh


# Region definitions for 85 keypoints
KEYPOINTS_REGION_MAP = {
    'forehead': list(range(36, 48)) + list(range(17, 27)),  # 22 points
    'nose': list(range(27, 36)),                              # 9 points
    'mouth': list(range(48, 61)) + [64],                      # 14 points
    'cheek': list(range(0, 17)) + list(range(61, 64)) + list(range(65, 85)),  # remaining
}

# 7 keypoints for global alignment (eye corners×4, nose tip, mouth corners×2)
SEVEN_KEYPOINTS = [36, 39, 42, 45, 33, 48, 54]


def _load_barycentric_coords(txt_path: str) -> list[tuple[int, float, float]]:
    """Load barycentric coordinate file (85 lines: triangle_id, w1, w2)."""
    coords = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                tri_id = int(parts[0])
                w1 = float(parts[1])
                w2 = float(parts[2])
                coords.append((tri_id, w1, w2))
    return coords


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Rigid Procrustes alignment (rotation + translation + uniform scale).

    Args:
        source: [N, 3] points to align
        target: [N, 3] reference points

    Returns:
        (aligned_source, scale, R, t)
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_centered = source - mu_s
    tgt_centered = target - mu_t

    # Compute optimal rotation via SVD
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    # Compute scale
    scale = np.trace(R @ H) / np.trace(src_centered.T @ src_centered)

    # Compute translation
    t = mu_t - scale * R @ mu_s

    # Apply
    aligned = scale * (source @ R.T) + t
    return aligned, scale, R, t


class REALYEvaluator:
    """REALY benchmark evaluator for FLAME mesh quality assessment."""

    def __init__(self, realy_data_dir: str = 'data/realy'):
        self.data_dir = realy_data_dir

        # Load FLAME template mesh
        flame_obj_path = os.path.join(realy_data_dir, 'FLAME.obj')
        if os.path.exists(flame_obj_path):
            self.flame_mesh = trimesh.load(flame_obj_path, process=False)
        else:
            self.flame_mesh = None

        # Load HIFI3D template mesh
        hifi3d_obj_path = os.path.join(realy_data_dir, 'HIFI3D.obj')
        if os.path.exists(hifi3d_obj_path):
            self.hifi3d_mesh = trimesh.load(hifi3d_obj_path, process=False)
        else:
            self.hifi3d_mesh = None

        # Load barycentric coordinates for both topologies
        flame_txt = os.path.join(realy_data_dir, 'FLAME.txt')
        hifi3d_txt = os.path.join(realy_data_dir, 'HIFI3D.txt')

        self.flame_bary = _load_barycentric_coords(flame_txt) if os.path.exists(flame_txt) else None
        self.hifi3d_bary = _load_barycentric_coords(hifi3d_txt) if os.path.exists(hifi3d_txt) else None

        # Load metrical scale factors
        scale_path = os.path.join(realy_data_dir, 'metrical_scale.txt')
        self.metrical_scales = {}
        if os.path.exists(scale_path):
            with open(scale_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.metrical_scales[parts[0]] = float(parts[1])

    def extract_85_keypoints(
        self,
        vertices: np.ndarray,
        topology: str = 'FLAME',
    ) -> np.ndarray:
        """Extract 85 keypoints from mesh vertices using barycentric coordinates.

        Args:
            vertices: [V, 3] mesh vertices
            topology: 'FLAME' or 'HIFI3D'

        Returns:
            [85, 3] keypoint coordinates
        """
        if topology == 'FLAME':
            bary = self.flame_bary
            faces = np.array(self.flame_mesh.faces)
        elif topology == 'HIFI3D':
            bary = self.hifi3d_bary
            faces = np.array(self.hifi3d_mesh.faces)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        if bary is None:
            raise FileNotFoundError(f"Barycentric coordinates for {topology} not loaded")

        keypoints = np.zeros((85, 3))
        for i, (tri_id, w1, w2) in enumerate(bary):
            w3 = 1.0 - w1 - w2
            v1, v2, v3 = faces[tri_id]
            keypoints[i] = w1 * vertices[v1] + w2 * vertices[v2] + w3 * vertices[v3]

        return keypoints

    def global_align(
        self,
        predicted_vertices: np.ndarray,
        predicted_kps_85: np.ndarray,
        gt_kps_85: np.ndarray,
    ) -> np.ndarray:
        """Global rigid alignment using 7 keypoints (Procrustes).

        Args:
            predicted_vertices: [V, 3] predicted mesh vertices
            predicted_kps_85: [85, 3] predicted 85 keypoints
            gt_kps_85: [85, 3] ground truth 85 keypoints

        Returns:
            [V, 3] aligned predicted vertices
        """
        # Select 7 keypoints for alignment
        src_7 = predicted_kps_85[SEVEN_KEYPOINTS]
        tgt_7 = gt_kps_85[SEVEN_KEYPOINTS]

        # Procrustes alignment
        _, scale, R, t = _procrustes_align(src_7, tgt_7)

        # Apply transform to all vertices
        aligned = scale * (predicted_vertices @ R.T) + t
        return aligned

    def regional_align_and_eval(
        self,
        predicted_vertices: np.ndarray,
        predicted_faces: np.ndarray,
        gt_scan_regions: dict,
        predicted_kps_85: np.ndarray,
        gt_kps_85: np.ndarray,
    ) -> dict:
        """Regional ICP alignment + point-to-surface error.

        For each of nose/mouth/forehead/cheek:
        1. Use region keypoints for initial local alignment
        2. Run ICP for fine alignment
        3. Compute point-to-surface distance

        Args:
            predicted_vertices: [V, 3] globally aligned vertices
            predicted_faces: [F, 3] face indices
            gt_scan_regions: dict mapping region name → trimesh.Trimesh
            predicted_kps_85: [85, 3] keypoints from predicted mesh
            gt_kps_85: [85, 3] keypoints from GT scan

        Returns:
            dict with per-region NMSE values and 'all' average
        """
        try:
            import open3d as o3d
        except ImportError:
            print("[REALY] open3d not available, skipping regional evaluation")
            return {}

        errors = {}
        predicted_mesh = trimesh.Trimesh(vertices=predicted_vertices, faces=predicted_faces, process=False)

        for region_name, kp_indices in KEYPOINTS_REGION_MAP.items():
            if region_name not in gt_scan_regions:
                continue

            gt_region = gt_scan_regions[region_name]

            # Local alignment using region keypoints
            src_kps = predicted_kps_85[kp_indices]
            tgt_kps = gt_kps_85[kp_indices]
            _, scale, R, t = _procrustes_align(src_kps, tgt_kps)

            # Apply local transform
            local_verts = scale * (predicted_vertices @ R.T) + t
            local_mesh = trimesh.Trimesh(vertices=local_verts, faces=predicted_faces, process=False)

            # ICP refinement using open3d
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(local_mesh.vertices)

            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(np.array(gt_region.vertices))

            threshold = 10.0  # mm
            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            # Apply ICP transform
            icp_verts = np.array(local_mesh.vertices)
            icp_verts_h = np.hstack([icp_verts, np.ones((len(icp_verts), 1))])
            icp_verts_aligned = (icp_result.transformation @ icp_verts_h.T).T[:, :3]

            # Compute point-to-surface distance
            # From GT to predicted (for coverage)
            gt_points = np.array(gt_region.vertices)
            pred_mesh_aligned = trimesh.Trimesh(
                vertices=icp_verts_aligned, faces=predicted_faces, process=False,
            )
            closest_points, distances, _ = pred_mesh_aligned.nearest.on_surface(gt_points)
            errors[region_name] = float(np.mean(distances))

        if errors:
            errors['all'] = float(np.mean(list(errors.values())))

        return errors

    def evaluate(
        self,
        flame_vertices: np.ndarray,
        subject_id: str,
        realy_scan_dir: str,
    ) -> dict:
        """Full REALY evaluation pipeline.

        Args:
            flame_vertices: [5023, 3] FLAME mesh vertices (meters)
            subject_id: REALY subject identifier
            realy_scan_dir: Directory containing GT scans

        Returns:
            dict with per-region errors in mm
        """
        # Convert to mm
        verts_mm = flame_vertices * 1000

        # Extract 85 keypoints from predicted mesh
        pred_kps_85 = self.extract_85_keypoints(verts_mm, 'FLAME')

        # Load GT scan and extract its 85 keypoints
        gt_scan_path = os.path.join(realy_scan_dir, f'{subject_id}.obj')
        if not os.path.exists(gt_scan_path):
            raise FileNotFoundError(f"GT scan not found: {gt_scan_path}")

        gt_scan = trimesh.load(gt_scan_path, process=False)
        gt_kps_85 = self.extract_85_keypoints(np.array(gt_scan.vertices), 'HIFI3D')

        # Global alignment
        flame_faces = np.array(self.flame_mesh.faces) if self.flame_mesh is not None else None
        aligned_verts = self.global_align(verts_mm, pred_kps_85, gt_kps_85)

        # Re-extract keypoints after alignment
        aligned_kps_85 = self.extract_85_keypoints(aligned_verts, 'FLAME')

        # Load regional GT meshes
        gt_regions = {}
        for region in ['nose', 'mouth', 'forehead', 'cheek']:
            region_path = os.path.join(realy_scan_dir, f'{subject_id}_{region}.obj')
            if os.path.exists(region_path):
                gt_regions[region] = trimesh.load(region_path, process=False)

        if not gt_regions:
            return {'error': 'No GT region meshes found'}

        # Regional evaluation
        errors = self.regional_align_and_eval(
            aligned_verts, flame_faces, gt_regions,
            aligned_kps_85, gt_kps_85,
        )

        # Apply metrical scale
        if subject_id in self.metrical_scales:
            scale = self.metrical_scales[subject_id]
            errors = {k: v * scale for k, v in errors.items()}

        return errors

    def save_results(
        self,
        errors: dict,
        output_dir: str,
        aligned_verts: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        kps_85: np.ndarray | None = None,
    ):
        """Save evaluation results to output directory.

        Args:
            errors: dict of per-region errors
            output_dir: Directory for saving results
            aligned_verts: Aligned mesh vertices (optional, for saving mesh)
            faces: Mesh faces (optional)
            kps_85: 85 keypoints (optional)
        """
        realy_dir = os.path.join(output_dir, '07_realy_eval')
        os.makedirs(realy_dir, exist_ok=True)

        # Save errors
        with open(os.path.join(realy_dir, 'region_errors.json'), 'w') as f:
            json.dump(errors, f, indent=2)

        # Save keypoints
        if kps_85 is not None:
            np.save(os.path.join(realy_dir, 'flame_85kps.npy'), kps_85)

        # Save aligned mesh
        if aligned_verts is not None and faces is not None:
            from .visualization import _save_obj
            _save_obj(os.path.join(realy_dir, 'aligned_mesh.obj'), aligned_verts, faces)

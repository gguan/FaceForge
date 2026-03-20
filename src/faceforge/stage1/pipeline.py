"""
Stage 1 Pipeline: Identity-aware initialization.

Extracts FLAME parameters from input images using MICA (shape) + DECA (expression/pose/texture/lighting).
"""

import os

import cv2
import numpy as np
import torch

from faceforge._paths import PROJECT_ROOT

from .config import Stage1Config
from .data_types import Stage1Output, DetectionResult
from .detection import MediaPipeDetector, RetinaFaceDetector, detect_all
from .alignment import image_align
from .segmentation import FaceParser
from .mica_inference import MICAInference
from .deca_inference import DECAInference
from .merge import merge_params
from .aggregation import aggregate_shapes
from .visualization import Stage1Visualizer


class Stage1Pipeline:
    """Complete Stage 1 pipeline: image → FLAME parameters."""

    def __init__(self, config: Stage1Config | None = None):
        if config is None:
            config = Stage1Config()
        self.config = config

        print("[Stage1] Initializing models...")

        # Detectors
        self.mp_detector = MediaPipeDetector(config.mediapipe_model_path)
        self.retina_detector = RetinaFaceDetector(config.device)

        # Face parser (BiSeNet)
        face_parsing_weights = str(PROJECT_ROOT / 'data' / 'pretrained' / '79999_iter.pth')
        if os.path.exists(face_parsing_weights):
            self.face_parser = FaceParser(face_parsing_weights, config.device)
        else:
            print(f"[Stage1] Warning: Face parsing weights not found at {face_parsing_weights}")
            self.face_parser = None

        # MICA
        self.mica = MICAInference(config)

        # DECA
        self.deca = DECAInference(config)

        print("[Stage1] All models loaded.")

    @torch.no_grad()
    def run_single(
        self,
        image_rgb: np.ndarray,
        subject_name: str = 'default',
        image_name: str | None = None,
    ) -> tuple['Stage1Output', np.ndarray | None]:
        """Run Stage 1 pipeline on a single image.

        Args:
            image_rgb: RGB uint8 image [H, W, 3]
            subject_name: Subject name for output directory
            image_name: Optional name for this image (used for per-image output dirs)

        Returns:
            (Stage1Output, summary_strip or None) — summary is RGB uint8 if debug enabled
        """
        config = self.config

        # Optional visualizer
        vis = None
        if config.save_debug:
            vis = Stage1Visualizer(config.output_dir, subject_name, image_name=image_name)

        # Step 1: Dual-detector face detection
        detection = detect_all(image_rgb, self.mp_detector, self.retina_detector)
        if detection is None:
            raise ValueError("No face detected in image")

        if vis:
            vis.save_detection(
                image_rgb, detection.lmks_dense, detection.lmks_68,
                detection.retinaface_kps,
            )

        # Step 2: Face alignment (also get the transform matrix for landmark projection)
        aligned_img, align_M = image_align(
            image_rgb,
            detection.lmks_68,
            output_size=config.align_output_size,
            transform_size=config.align_transform_size,
            scale_factor=config.align_scale_factor,
            padding_mode='constant',
            return_transform=True,
        )

        # Project 68 landmarks into aligned image coordinates
        lmks_68_orig = detection.lmks_68[:, :2].astype(np.float64)
        ones = np.ones((lmks_68_orig.shape[0], 1), dtype=np.float64)
        lmks_h = np.concatenate([lmks_68_orig, ones], axis=1)  # [68, 3]
        lmks_aligned = (align_M @ lmks_h.T).T  # [68, 3]
        lmks_68_aligned = lmks_aligned[:, :2] / lmks_aligned[:, 2:3]  # perspective divide

        if vis:
            vis.save_alignment(image_rgb, aligned_img, detection.lmks_68)

        # Step 3: Face segmentation
        parsing = None
        face_mask = None
        if self.face_parser is not None:
            parsing = self.face_parser.parse(aligned_img)
            face_mask = FaceParser.extract_face_mask(parsing)

            if vis:
                vis.save_segmentation(aligned_img, parsing, face_mask)

        # Step 4a: MICA inference (uses RetinaFace 5-point)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if detection.retinaface_kps is not None:
            mica_result = self.mica.run(image_bgr, detection.retinaface_kps)
        else:
            raise ValueError("RetinaFace detection failed — required for MICA ArcFace alignment")

        # Fetch FLAME faces once for all subsequent use
        faces = self._get_flame_faces()

        if vis:
            vis.save_mica(
                mica_result['arcface_img'],
                mica_result['shape_code'].numpy(),
                mica_result['vertices'].squeeze(0).numpy(),
                faces,
            )

        # Step 4b: DECA inference
        deca_result = self.deca.run(image_rgb)

        # Step 5: Merge parameters
        flame_params = merge_params(mica_result, deca_result)

        # Canonical vertices (shape only, no pose/expression)
        canonical_verts = mica_result['vertices'].squeeze(0).numpy()  # [5023, 3]

        # Posed vertices (shape + expression + pose)
        posed_verts = self._get_posed_vertices(flame_params)
        if posed_verts is not None:
            verts_for_vis = posed_verts
            print(f'[Stage1] Posed vertices OK, range: x=[{posed_verts[:,0].min():.4f}, {posed_verts[:,0].max():.4f}], '
                  f'y=[{posed_verts[:,1].min():.4f}, {posed_verts[:,1].max():.4f}]')
        else:
            verts_for_vis = canonical_verts
            print('[Stage1] WARNING: Using canonical vertices (no pose) — mesh overlay will NOT match head pose!')

        summary_strip = None
        if vis:
            vis.save_deca(
                deca_result['deca_crop'],
                {
                    'exp': deca_result['exp'],
                    'pose': deca_result['pose'],
                    'tex': deca_result['tex'],
                    'light': deca_result['light'],
                },
                flame_params=flame_params,
                canonical_vertices=canonical_verts,
                posed_vertices=verts_for_vis,
                faces=faces,
            )

            flame_lmks_3d = self._get_flame_landmarks_3d(verts_for_vis)
            if flame_lmks_3d is not None:
                print(f'[Stage1] FLAME 3D landmarks OK, range: '
                      f'x=[{flame_lmks_3d[:,0].min():.4f}, {flame_lmks_3d[:,0].max():.4f}], '
                      f'y=[{flame_lmks_3d[:,1].min():.4f}, {flame_lmks_3d[:,1].max():.4f}]')
            else:
                print('[Stage1] WARNING: FLAME 3D landmarks FAILED — no mesh overlay possible')

            summary_strip = vis.save_summary(
                aligned_image=aligned_img,
                parsing=parsing,
                lmks_68=lmks_68_aligned,
                vertices=verts_for_vis,
                faces=faces,
                flame_lmks_3d=flame_lmks_3d,
                device=config.device,
            )

        # Build Stage1Output
        # Camera initialization
        render_size = config.render_size
        init_focal = 2000.0 * (render_size / 512)
        focal_length = init_focal / render_size

        # Convert aligned image to tensor [1, 3, 512, 512]
        aligned_tensor = torch.tensor(
            aligned_img.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)

        # Face mask tensor
        if face_mask is not None:
            mask_tensor = torch.tensor(face_mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = torch.ones(1, config.align_output_size, config.align_output_size)

        output = Stage1Output(
            shape=flame_params['shape'],
            expression=flame_params['exp'],
            head_pose=flame_params['head_pose'],
            jaw_pose=flame_params['jaw_pose'],
            texture=flame_params['tex'],
            lighting=flame_params['light'],
            arcface_feat=mica_result['arcface_feat'],
            aligned_image=aligned_tensor,
            face_mask=mask_tensor,
            lmks_68=torch.tensor(detection.lmks_68).unsqueeze(0),
            lmks_dense=torch.tensor(detection.lmks_dense).unsqueeze(0),
            lmks_eyes=torch.tensor(detection.lmks_eyes).unsqueeze(0),
            focal_length=torch.tensor([[focal_length]]),
            principal_point=torch.tensor([[0.0, 0.0]]),
        )
        return output, summary_strip

    @torch.no_grad()
    def run_multi(
        self,
        images_rgb: list[np.ndarray],
        subject_name: str = 'default',
        image_names: list[str] | None = None,
    ) -> tuple['Stage1Output', np.ndarray | None]:
        """Run Stage 1 with multi-image shape aggregation.

        Uses the first image for alignment/segmentation/DECA,
        aggregates shape codes across all images.

        Args:
            images_rgb: List of RGB uint8 images of the same person
            subject_name: Subject name for output directory
            image_names: Optional list of names for per-image debug dirs

        Returns:
            (Stage1Output with aggregated shape, summary_strip or None)
        """
        if len(images_rgb) == 1:
            name = image_names[0] if image_names else None
            return self.run_single(images_rgb[0], subject_name, image_name=name)

        # Aggregate shape codes across all images
        aggregated_shape, per_image_outputs = aggregate_shapes(
            images_rgb, self.mica, self.retina_detector,
            method=self.config.aggregation_method,
        )

        # Run full pipeline on first image (reuse its existing debug dir if named)
        first_name = image_names[0] if image_names else None
        result, summary_strip = self.run_single(images_rgb[0], subject_name, image_name=first_name)

        # Replace shape with aggregated version
        result.shape = aggregated_shape.unsqueeze(0)  # [1, 300]

        # Save aggregation debug info
        if self.config.save_debug:
            agg_dir = os.path.join(self.config.output_dir, subject_name, 'stage1', 'aggregation')
            os.makedirs(agg_dir, exist_ok=True)

            # Save all shape codes
            all_codes = torch.stack(
                [o['shape_code'].squeeze(0) for o in per_image_outputs], dim=0
            ).numpy()
            np.save(os.path.join(agg_dir, 'shape_codes_all.npy'), all_codes)
            np.save(os.path.join(agg_dir, 'shape_median.npy'), aggregated_shape.numpy())

        return result, summary_strip

    def _get_flame_faces(self) -> np.ndarray | None:
        """Get FLAME mesh faces from MICA's FLAME model."""
        try:
            faces = self.mica.model.flame.faces_tensor.cpu().numpy()
            return faces
        except Exception:
            return None

    @torch.no_grad()
    def _get_posed_vertices(self, flame_params: dict) -> np.ndarray | None:
        """Run FLAME forward pass with merged params to get posed vertices.

        Args:
            flame_params: dict with 'shape', 'exp', 'head_pose', 'jaw_pose'

        Returns:
            [5023, 3] posed vertices in meters, or None on failure
        """
        import traceback
        try:
            flame = self.mica.model.flame
            device = next(flame.parameters()).device

            shape = flame_params['shape'].to(device)  # [1, 300]

            # Expression dimension lives in different places across FLAME wrappers.
            # Prefer explicit config when present, otherwise infer from shapedirs.
            n_exp = None
            if hasattr(flame, 'cfg'):
                model_cfg = getattr(flame.cfg, 'model', None)
                if model_cfg is not None and hasattr(model_cfg, 'n_exp'):
                    n_exp = model_cfg.n_exp
                elif hasattr(flame.cfg, 'n_exp'):
                    n_exp = flame.cfg.n_exp
            if n_exp is None and hasattr(flame, 'shapedirs') and hasattr(flame, 'n_shape'):
                total_dims = flame.shapedirs.shape[-1]
                n_exp = max(0, int(total_dims) - int(flame.n_shape))
            if n_exp is None or n_exp <= 0:
                n_exp = flame_params['exp'].shape[1]

            exp = flame_params['exp'][:, :n_exp].to(device)

            # Reconstruct pose [1, 6] = head_pose + jaw_pose
            pose = torch.cat([
                flame_params['head_pose'].to(device),
                flame_params['jaw_pose'].to(device),
            ], dim=1)

            print(f'[Stage1] FLAME forward: shape={shape.shape}, exp={exp.shape}, pose={pose.shape}, device={device}')

            verts, _, _ = flame(
                shape_params=shape,
                expression_params=exp,
                pose_params=pose,
            )
            return verts.squeeze(0).cpu().numpy()  # [5023, 3]
        except Exception as e:
            print(f'[Stage1] FLAME posed vertices FAILED:')
            traceback.print_exc()
            return None

    @torch.no_grad()
    def _get_flame_landmarks_3d(self, vertices: np.ndarray) -> np.ndarray | None:
        """Extract 68 3D landmark positions from FLAME vertices.

        Args:
            vertices: [5023, 3] posed FLAME vertices in meters

        Returns:
            [68, 3] 3D landmark positions, or None on failure
        """
        try:
            flame = self.mica.model.flame
            device = next(flame.parameters()).device
            verts_t = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
            lmks_3d = flame.seletec_3d68(verts_t)  # [1, 68, 3]
            return lmks_3d.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f'[Stage1] Warning: FLAME 3D landmarks failed ({e}).')
            return None


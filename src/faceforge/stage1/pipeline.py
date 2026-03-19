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
    ) -> Stage1Output:
        """Run Stage 1 pipeline on a single image.

        Args:
            image_rgb: RGB uint8 image [H, W, 3]
            subject_name: Subject name for output directory

        Returns:
            Stage1Output with all FLAME parameters and features
        """
        config = self.config

        # Optional visualizer
        vis = None
        if config.save_debug:
            vis = Stage1Visualizer(config.output_dir, subject_name)

        # Step 1: Dual-detector face detection
        detection = detect_all(image_rgb, self.mp_detector, self.retina_detector)
        if detection is None:
            raise ValueError("No face detected in image")

        if vis:
            vis.save_detection(
                image_rgb, detection.lmks_dense, detection.lmks_68,
                detection.retinaface_kps,
            )

        # Step 2: Face alignment
        aligned_img = image_align(
            image_rgb,
            detection.lmks_68,
            output_size=config.align_output_size,
            transform_size=config.align_transform_size,
            scale_factor=config.align_scale_factor,
            padding_mode='constant',
        )

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

        if vis:
            # Get FLAME faces for mesh saving
            faces = self._get_flame_faces()
            vis.save_mica(
                mica_result['arcface_img'],
                mica_result['shape_code'].numpy(),
                mica_result['vertices'].squeeze(0).numpy(),
                faces,
            )

        # Step 4b: DECA inference
        deca_result = self.deca.run(image_rgb)

        if vis:
            vis.save_deca(deca_result['deca_crop'], {
                'exp': deca_result['exp'],
                'pose': deca_result['pose'],
                'tex': deca_result['tex'],
                'light': deca_result['light'],
            })

        # Step 5: Merge parameters
        flame_params = merge_params(mica_result, deca_result)

        if vis:
            faces = self._get_flame_faces()
            vis.save_merged(
                flame_params,
                mica_result['vertices'].squeeze(0).numpy(),
                faces,
                mica_result['arcface_feat'].numpy(),
                image_rgb,
            )

            # Build landmark image for summary
            lmk_img = image_rgb.copy()
            for pt in detection.lmks_68:
                cv2.circle(lmk_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            mask_overlay = None
            if face_mask is not None:
                mask_overlay = aligned_img.copy()
                mask_overlay[~face_mask] = (mask_overlay[~face_mask] * 0.3).astype(np.uint8)

            vis.save_summary(image_rgb, lmk_img, aligned_img, mask_overlay)

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

        return Stage1Output(
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

    @torch.no_grad()
    def run_multi(
        self,
        images_rgb: list[np.ndarray],
        subject_name: str = 'default',
    ) -> Stage1Output:
        """Run Stage 1 with multi-image shape aggregation.

        Uses the first image for alignment/segmentation/DECA,
        aggregates shape codes across all images.

        Args:
            images_rgb: List of RGB uint8 images of the same person
            subject_name: Subject name for output directory

        Returns:
            Stage1Output with aggregated shape
        """
        if len(images_rgb) == 1:
            return self.run_single(images_rgb[0], subject_name)

        # Aggregate shape codes across all images
        aggregated_shape, per_image_outputs = aggregate_shapes(
            images_rgb, self.mica, self.retina_detector,
            method=self.config.aggregation_method,
        )

        # Run full pipeline on first image
        result = self.run_single(images_rgb[0], subject_name)

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

        return result

    def _get_flame_faces(self) -> np.ndarray | None:
        """Get FLAME mesh faces from MICA's FLAME model."""
        try:
            faces = self.mica.model.flame.faces_tensor.cpu().numpy()
            return faces
        except Exception:
            return None

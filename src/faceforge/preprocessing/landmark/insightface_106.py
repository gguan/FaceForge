"""SCRFD face detection + InsightFace 106-point landmarks.

Wraps :class:`insightface.app.FaceAnalysis` with only the two modules we
need (``detection`` + ``landmark_2d_106``), avoiding the cost of ArcFace
recognition, gender/age, and 3D landmarks during preprocessing.

A single ``FaceAnalysis`` instance runs both models; one image pass yields
bbox, 5-point ArcFace-order landmarks, and 106-point dense landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseLandmarkDetector, LandmarkResult
from .visualize import draw_landmarks


@dataclass
class InsightFace106Config:
    pack_name: str = 'antelopev2'
    det_size: int = 640
    device: str = 'cuda:0'
    pick_center_face: bool = True   # False → pick largest face by bbox area


class InsightFace106Detector(BaseLandmarkDetector):
    """SCRFD detector + InsightFace 106-point landmark refiner."""

    name = 'insightface_106'

    def __init__(self, config: InsightFace106Config | None = None):
        from insightface.app import FaceAnalysis

        self.config = config or InsightFace106Config()

        providers = (
            ['CUDAExecutionProvider'] if 'cuda' in self.config.device
            else ['CPUExecutionProvider']
        )
        ctx_id = 0 if 'cuda' in self.config.device else -1

        self.app = FaceAnalysis(
            name=self.config.pack_name,
            allowed_modules=['detection', 'landmark_2d_106'],
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(self.config.det_size, self.config.det_size))

    def run(self, image_rgb: np.ndarray) -> LandmarkResult:
        # FaceAnalysis expects BGR.
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(image_bgr)
        if len(faces) == 0:
            raise ValueError("no face detected")

        face = self._select_face(faces, image_rgb.shape[:2], self.config.pick_center_face)
        if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
            raise RuntimeError(
                "InsightFace returned a face without landmark_2d_106 — "
                "is the 2d106det.onnx model present in the pack?"
            )

        return LandmarkResult(
            landmarks=face.landmark_2d_106.astype(np.float32),
            bbox=face.bbox.astype(np.float32),
            confidence=float(face.det_score),
            n_points=106,
            scheme='insightface_106',
            kps_5pt=face.kps.astype(np.float32),
        )

    def visualize(self, image_rgb: np.ndarray, result: LandmarkResult) -> np.ndarray:
        return draw_landmarks(
            image_rgb,
            landmarks=result.landmarks,
            bbox=result.bbox,
            kps_5pt=result.kps_5pt,
            title=f'{result.scheme}  conf={result.confidence:.2f}',
        )

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _select_face(faces, image_hw, pick_center_face: bool):
        if len(faces) == 1:
            return faces[0]

        if pick_center_face:
            h, w = image_hw
            img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            best, best_dist = None, float('inf')
            for f in faces:
                bbox = f.bbox
                center = np.array(
                    [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                    dtype=np.float32,
                )
                dist = float(np.linalg.norm(center - img_center))
                if dist < best_dist:
                    best, best_dist = f, dist
            return best

        # Largest area.
        best, best_area = None, -1.0
        for f in faces:
            bbox = f.bbox
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if area > best_area:
                best, best_area = f, area
        return best

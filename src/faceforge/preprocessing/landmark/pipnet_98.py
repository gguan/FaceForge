"""PIPNet WFLW 98-point landmark detection (preprocessing component).

Wraps the existing :class:`faceforge.stage1.pipnet_inference.PIPNetLandmarkDetector`,
which itself defers to the original PIPNet code shipped with the
pixel3dmm/MonoNPHM submodule. The wrapper adds a face bounding box (from
the same FaceBoxesV2 detector PIPNet already uses internally) and a
visualization method.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseLandmarkDetector, LandmarkResult
from .visualize import draw_landmarks


@dataclass
class PIPNet98Config:
    code_base: str = 'submodules/pixel3dmm'
    device: str = 'cuda:0'
    experiment_path: str = 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py'


class PIPNet98Detector(BaseLandmarkDetector):
    """PIPNet on WFLW (98 landmarks) + FaceBoxesV2 bbox."""

    name = 'pipnet_98'

    def __init__(self, config: PIPNet98Config | None = None):
        self.config = config or PIPNet98Config()
        from faceforge.stage1.pipnet_inference import PIPNetLandmarkDetector
        self._impl = PIPNetLandmarkDetector(
            code_base=self.config.code_base,
            device=self.config.device,
            experiment_path=self.config.experiment_path,
        )

    def run(self, image_rgb: np.ndarray) -> LandmarkResult:
        # Face detection (FaceBoxesV2 — already loaded by the impl).
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detections, _ = self._impl.detector.detect(image_bgr, self._impl.threshold, 1)
        detections = [d for d in detections if d[0] == 'face']
        if not detections:
            raise ValueError("no face detected")
        detections.sort(key=lambda x: -x[1])
        det = detections[0]
        x1, y1, w, h = int(det[2]), int(det[3]), int(det[4]), int(det[5])
        bbox = np.array([x1, y1, x1 + w - 1, y1 + h - 1], dtype=np.float32)
        confidence = float(det[1])

        landmarks = self._impl.predict(image_rgb)
        if landmarks is None:
            raise ValueError("PIPNet inference failed")

        return LandmarkResult(
            landmarks=landmarks.astype(np.float32),
            bbox=bbox,
            confidence=confidence,
            n_points=int(landmarks.shape[0]),
            scheme='wflw_98',
            kps_5pt=None,
        )

    def visualize(self, image_rgb: np.ndarray, result: LandmarkResult) -> np.ndarray:
        return draw_landmarks(
            image_rgb,
            landmarks=result.landmarks,
            bbox=result.bbox,
            kps_5pt=result.kps_5pt,
            title=f'{result.scheme}  conf={result.confidence:.2f}',
        )

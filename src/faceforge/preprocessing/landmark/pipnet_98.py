"""PIPNet WFLW 98-point landmark detection (preprocessing component).

Wraps the existing :class:`faceforge.stage1.pipnet_inference.PIPNetLandmarkDetector`,
which itself defers to the original PIPNet code shipped with the
pixel3dmm/MonoNPHM submodule. The wrapper adds a face bounding box (from
the same FaceBoxesV2 detector PIPNet already uses internally) and a
visualization method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseLandmarkDetector, LandmarkResult
from .visualize import draw_landmarks


def _wflw98_to_arcface_5pt(lmks_98: np.ndarray) -> np.ndarray:
    """Derive 5-point ArcFace-order landmarks from WFLW 98pt.

    WFLW canonical indices (subject's perspective; matches ``WFLW_98``
    annotation README):

      * 60..67 — subject's RIGHT eye contour (=image-LEFT side of frontal portrait)
      * 68..75 — subject's LEFT eye contour
      * 96      — subject's RIGHT iris
      * 97      — subject's LEFT iris
      * 54      — nose tip
      * 76      — subject's RIGHT mouth corner
      * 82      — subject's LEFT mouth corner

    ArcFace 5pt order: ``[eye_r, eye_l, nose, mouth_r, mouth_l]`` (subject
    perspective). Use the iris points (96, 97) for eye centres — they're
    more stable than the eye-contour averages for off-frontal poses.
    """
    eye_r = lmks_98[96]
    eye_l = lmks_98[97]
    nose  = lmks_98[54]
    mouth_r = lmks_98[76]
    mouth_l = lmks_98[82]
    return np.stack([eye_r, eye_l, nose, mouth_r, mouth_l], axis=0).astype(np.float32)


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
        # One detection pass, reused for both the bbox we surface and the
        # crop PIPNet operates on.
        det = self._impl.detect_face(image_rgb)
        if det is None:
            raise ValueError("no face detected")
        x1, y1, w, h = int(det[2]), int(det[3]), int(det[4]), int(det[5])
        bbox = np.array([x1, y1, x1 + w - 1, y1 + h - 1], dtype=np.float32)
        confidence = float(det[1])

        landmarks = self._impl.predict_from_detection(image_rgb, det).astype(np.float32)

        return LandmarkResult(
            landmarks=landmarks,
            bbox=bbox,
            confidence=confidence,
            n_points=int(landmarks.shape[0]),
            scheme='wflw_98',
            kps_5pt=_wflw98_to_arcface_5pt(landmarks),
        )

    def visualize(self, image_rgb: np.ndarray, result: LandmarkResult) -> np.ndarray:
        return draw_landmarks(
            image_rgb,
            landmarks=result.landmarks,
            bbox=result.bbox,
            kps_5pt=result.kps_5pt,
            title=f'{result.scheme}  conf={result.confidence:.2f}',
        )

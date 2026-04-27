"""FAN 2D 68-point landmark detection (preprocessing component).

Wraps the ``face_alignment`` package's 2D 68pt detector (used by MICA,
MonoNPHM, HRN, and other FLAME-fit pipelines for landmark loss). FAN
produces iBUG-68 layout — a different scheme from PIPNet's WFLW-98 — so
this is exposed as its own backend for callers that need 68pt
specifically (e.g., MonoNPHM's ``kpt/{i:05d}.npy``).

Note: ``face_alignment`` does its own face detection via S3FD by default
and runs both networks in one ``get_landmarks_from_image`` call. The
returned bbox here is derived from the landmark min/max — there's no
detector-confidence concept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseLandmarkDetector, LandmarkResult
from .visualize import draw_landmarks


@dataclass
class FAN68Config:
    device: str = 'cuda'           # ``cuda`` or ``cpu`` — face_alignment's API
    flip_input: bool = False       # ensemble with mirrored input (slower, more robust)


class FAN68Detector(BaseLandmarkDetector):
    """face_alignment 2D 68-point landmark detector."""

    name = 'fan_68'

    def __init__(self, config: FAN68Config | None = None):
        self.config = config or FAN68Config()
        import face_alignment
        # Newer face_alignment renames _2D → TWO_D; tolerate both.
        landmarks_type = getattr(
            face_alignment.LandmarksType, 'TWO_D',
            getattr(face_alignment.LandmarksType, '_2D', None),
        )
        if landmarks_type is None:
            raise RuntimeError("face_alignment LandmarksType (2D) not found")
        self._fa = face_alignment.FaceAlignment(
            landmarks_type,
            flip_input=self.config.flip_input,
            device=self.config.device,
        )

    def run(self, image_rgb: np.ndarray) -> LandmarkResult:
        result = self._fa.get_landmarks_from_image(image_rgb)
        if result is None or len(result) == 0:
            raise ValueError("FAN: no face detected")
        # Highest-confidence is implicit — face_alignment returns multiple
        # if S3FD finds multiple; pick the largest by bbox area.
        best = max(result, key=lambda lm: (lm[:, 0].ptp() * lm[:, 1].ptp()))
        landmarks = best.astype(np.float32)
        bbox = np.array([
            landmarks[:, 0].min(), landmarks[:, 1].min(),
            landmarks[:, 0].max(), landmarks[:, 1].max(),
        ], dtype=np.float32)
        return LandmarkResult(
            landmarks=landmarks,
            bbox=bbox,
            confidence=1.0,                # FAN doesn't expose a score
            n_points=68,
            scheme='ibug_68',
            kps_5pt=None,
        )

    def visualize(self, image_rgb: np.ndarray, result: LandmarkResult) -> np.ndarray:
        return draw_landmarks(
            image_rgb, landmarks=result.landmarks, bbox=result.bbox,
            kps_5pt=None, title=f'{result.scheme}',
        )

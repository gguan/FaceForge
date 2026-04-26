"""FFHQ-style aligned head crop, driven by SCRFD 5-point landmarks.

This component wires the preprocessing surface (Config / Result / run /
visualize) around :mod:`._geometry`, which holds the pure FFHQ alignment
math. Detection is lazy: if the caller passes a precomputed
:class:`LandmarkResult`, no detector is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..landmark.base import LandmarkResult
from ..landmark.insightface_106 import InsightFace106Config, InsightFace106Detector
from ._geometry import project_points, standardize_crop
from .base import BaseCropper, CropResult
from .visualize import draw_crop_overlay


@dataclass
class FFHQCropConfig:
    output_size: int = 512
    transform_size: int = 1024
    scale_factor: float = 1.3      # 1.0 = FFHQ tight, 1.3 = tracking-style head
    padding_mode: str = 'constant'

    # Used only when no pre-detected landmarks are passed to ``run()``.
    detector_pack: str = 'antelopev2'
    detector_det_size: int = 640
    detector_device: str = 'cuda:0'
    pick_center_face: bool = True


class FFHQCropper(BaseCropper):
    """SCRFD 5pt → FFHQ alignment quad → 512×512 head crop."""

    name = 'ffhq_scale_crop'

    def __init__(self, config: FFHQCropConfig | None = None):
        self.config = config or FFHQCropConfig()
        self._detector: Optional[InsightFace106Detector] = None  # lazily built

    @property
    def detector(self) -> InsightFace106Detector:
        if self._detector is None:
            cfg = self.config
            self._detector = InsightFace106Detector(
                InsightFace106Config(
                    pack_name=cfg.detector_pack,
                    det_size=cfg.detector_det_size,
                    device=cfg.detector_device,
                    pick_center_face=cfg.pick_center_face,
                )
            )
        return self._detector

    def run(
        self,
        image_rgb: np.ndarray,
        landmark_result: Optional[LandmarkResult] = None,
    ) -> CropResult:
        if landmark_result is None:
            landmark_result = self.detector.run(image_rgb)
        if landmark_result.kps_5pt is None:
            raise ValueError(
                f"FFHQCropper requires kps_5pt, but the supplied "
                f"LandmarkResult ({landmark_result.scheme}) has none. "
                "Use a backend that provides 5-point landmarks "
                "(e.g., insightface_106)."
            )

        cfg = self.config
        aligned, M, quad = standardize_crop(
            image_rgb,
            kps_5pt=landmark_result.kps_5pt,
            output_size=cfg.output_size,
            transform_size=cfg.transform_size,
            scale_factor=cfg.scale_factor,
            padding_mode=cfg.padding_mode,
        )

        landmarks_aligned: dict[str, np.ndarray] = {
            'kps_5pt': project_points(landmark_result.kps_5pt, M),
            landmark_result.scheme: project_points(landmark_result.landmarks, M),
        }

        return CropResult(
            aligned_image=aligned,
            transform=M.astype(np.float32),
            crop_quad=quad.astype(np.float32),
            landmarks_aligned=landmarks_aligned,
            scheme=f'ffhq_scale{cfg.scale_factor}',
            source_landmarks=landmark_result,
        )

    def visualize(self, image_rgb: np.ndarray, result: CropResult) -> np.ndarray:
        return draw_crop_overlay(image_rgb, result)

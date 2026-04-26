"""Cropping/alignment result + base class."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..base import BasePreprocessor, ComponentResult
from ..landmark.base import LandmarkResult


@dataclass
class CropResult(ComponentResult):
    """Standardized aligned head crop + the math to project back and forth.

    Attributes:
        aligned_image: [output_size, output_size, 3] RGB uint8 — the crop.
        transform: [3, 3] perspective matrix mapping source → aligned coords.
        crop_quad:  [4, 2] quad in *source* coords whose corners get warped
                    to (TL, BL, BR, TR) of the output square.
        landmarks_aligned: dict of name → [N, 2] arrays — input keypoints
                    re-projected into aligned-image coordinates. Convenient
                    for downstream consumers that don't have access to the
                    source image.
        scheme:     backend identifier (e.g., 'ffhq_scale1.3').
    """

    aligned_image: np.ndarray
    transform: np.ndarray
    crop_quad: np.ndarray
    landmarks_aligned: dict[str, np.ndarray]
    scheme: str
    source_landmarks: Optional[LandmarkResult] = None


class BaseCropper(BasePreprocessor):
    """Common surface for face cropping / alignment backends.

    Backends accept either a raw image (running their own internal detector)
    or a precomputed :class:`LandmarkResult`. The two-argument form lets a
    pipeline share one detection pass across multiple downstream consumers.
    """

    @abstractmethod
    def run(
        self,
        image_rgb: np.ndarray,
        landmark_result: Optional[LandmarkResult] = None,
    ) -> CropResult:
        ...

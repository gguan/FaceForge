"""Landmark detection result + abstract base class."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..base import BasePreprocessor, ComponentResult


@dataclass
class LandmarkResult(ComponentResult):
    """Output of a landmark detection backend (single image).

    Attributes:
        landmarks: [N, 2] landmark pixel coords in source-image frame
        bbox:      [4] (x1, y1, x2, y2) face bbox in source coords
        confidence: detection score in [0, 1] if available
        kps_5pt:   [5, 2] optional ArcFace-order 5-point landmarks
                   (eye_r, eye_l, nose, mouth_r, mouth_l), if the backend
                   produces them. None otherwise.
        n_points:  number of dense landmarks (e.g., 98 / 106 / 68)
        scheme:    string identifying the landmark layout
                   (e.g., 'wflw_98', 'insightface_106', 'ibug_68')
    """

    landmarks: np.ndarray
    bbox: np.ndarray
    confidence: float
    n_points: int
    scheme: str
    kps_5pt: Optional[np.ndarray] = None


class BaseLandmarkDetector(BasePreprocessor):
    """Common surface for all landmark backends."""

    @abstractmethod
    def run(self, image_rgb: np.ndarray) -> LandmarkResult:
        ...

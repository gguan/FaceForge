"""Facial segmentation result + base class + class-name catalogs."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from ..base import BasePreprocessor, ComponentResult


# BiSeNet (CelebAMask-HQ) 19-class label scheme.
BISENET_CLASSES = {
    0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye',
    5: 'r_eye',     6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r',
    10: 'nose',     11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck',
    15: 'neck_l',   16: 'cloth', 17: 'hair', 18: 'hat',
}

# facer FaRL/CelebM 11-class label scheme (also widely used by MonoNPHM).
FACER_CELEBM_CLASSES = {
    0: 'background', 1: 'neck', 2: 'face', 3: 'cloth', 4: 'rr',
    5: 'lr',         6: 'rb',   7: 'lb',   8: 're',    9: 'le',
    10: 'nose',      11: 'imouth', 12: 'llip', 13: 'ulip',
    14: 'hair',      15: 'eyeg',   16: 'hat',   17: 'earr', 18: 'neck_l',
}


@dataclass
class SegmentationResult(ComponentResult):
    """Per-image facial segmentation output.

    Attributes:
        seg_map:    [H, W] int class map (uint8)
        face_mask:  [H, W] bool — face region (skin + brows + eyes + nose + lips)
        n_classes:  number of classes the backend produces
        scheme:     'bisenet_19' or 'facer_celebm_19'
    """

    seg_map: np.ndarray
    face_mask: np.ndarray
    n_classes: int
    scheme: str


class BaseSegmenter(BasePreprocessor):
    """Common surface for facial segmentation backends."""

    @abstractmethod
    def run(self, image_rgb: np.ndarray) -> SegmentationResult:
        ...

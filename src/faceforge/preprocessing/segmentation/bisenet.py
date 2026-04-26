"""BiSeNet (CelebAMask-HQ 19-class) segmentation component."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceforge._paths import PROJECT_ROOT

from .base import BaseSegmenter, SegmentationResult
from .visualize import colorize_seg, overlay_face_mask


@dataclass
class BiSeNetConfig:
    weights_path: str = str(PROJECT_ROOT / 'data' / 'pretrained' / '79999_iter.pth')
    device: str = 'cuda:0'
    # Network is trained at 512×512; if input differs we resize for inference
    # and resize the parsing map back to the source resolution.
    input_size: int = 512


class BiSeNetSegmenter(BaseSegmenter):
    """BiSeNet wrapper exposed as a preprocessing component."""

    name = 'bisenet'

    def __init__(self, config: BiSeNetConfig | None = None):
        self.config = config or BiSeNetConfig()
        from faceforge.stage1.segmentation import FaceParser
        self._impl = FaceParser(self.config.weights_path, self.config.device)

    def run(self, image_rgb: np.ndarray) -> SegmentationResult:
        h, w = image_rgb.shape[:2]
        size = self.config.input_size

        if (h, w) != (size, size):
            img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            img = image_rgb

        parsing_small = self._impl.parse(img)  # [size, size] int
        if (h, w) != (size, size):
            parsing = cv2.resize(
                parsing_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
            )
        else:
            parsing = parsing_small.astype(np.uint8)

        face_mask = self._impl.extract_face_mask(parsing)
        return SegmentationResult(
            seg_map=parsing,
            face_mask=face_mask,
            n_classes=19,
            scheme='bisenet_19',
        )

    def visualize(
        self,
        image_rgb: np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:
        seg_color = colorize_seg(result.seg_map, n_classes=result.n_classes)
        masked = overlay_face_mask(image_rgb, result.face_mask)

        # Side-by-side: [seg color | source w/ face-mask outline]
        return np.concatenate([seg_color, masked], axis=1)

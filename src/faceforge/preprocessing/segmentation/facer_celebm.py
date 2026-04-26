"""facer (FaRL/CelebM) face parsing component.

Optional backend used by MonoNPHM. Requires the ``facer`` package and a
network connection on first use (it pulls FaRL weights). Detection +
segmentation are run in one shot, like MonoNPHM's ``run_facer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseSegmenter, SegmentationResult
from .visualize import colorize_seg, overlay_face_mask


@dataclass
class FacerConfig:
    detector_name: str = 'retinaface/mobilenet'
    parser_name: str = 'farl/celebm/448'
    device: str = 'cuda:0'
    # Long edge above this threshold gets downscaled before inference,
    # matching MonoNPHM's preprocessing behaviour.
    max_long_edge: int = 1000


# CelebM "face" core classes (skin + brows + eyes + nose + lips). Hair, hat,
# glasses, neck, cloth, ears are excluded. Indices follow the FaRL/CelebM
# label set documented in the facer repo.
_FACE_CORE_CLASSES = {2, 6, 7, 8, 9, 10, 11, 12, 13}


class FacerSegmenter(BaseSegmenter):
    """facer FaceDetector + FaceParser bundled into a single component."""

    name = 'facer_celebm'

    def __init__(self, config: FacerConfig | None = None):
        self.config = config or FacerConfig()
        import facer
        import torch

        self._facer = facer
        self._torch = torch
        self.detector = facer.face_detector(self.config.detector_name, device=self.config.device)
        self.parser = facer.face_parser(self.config.parser_name, device=self.config.device)
        self._n_classes: Optional[int] = None

    def run(self, image_rgb: np.ndarray) -> SegmentationResult:
        torch = self._torch
        facer = self._facer

        h, w = image_rgb.shape[:2]
        long_edge = max(h, w)
        scale = 1.0
        if long_edge > self.config.max_long_edge:
            scale = self.config.max_long_edge / long_edge
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            import cv2
            inp = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            inp = image_rgb

        # facer.hwc2bchw: HWC uint8 → 1×3×H×W on device.
        image_tensor = facer.hwc2bchw(
            torch.from_numpy(inp[..., :3])
        ).to(self.config.device)

        with torch.inference_mode():
            faces = self.detector(image_tensor)
            faces = self.parser(image_tensor, faces)

        if 'seg' not in faces or len(faces.get('image_ids', [])) == 0:
            raise ValueError("facer detected no faces")

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        n_classes = int(seg_probs.shape[1])
        self._n_classes = n_classes

        # When multiple faces, take the first one with highest area heuristic.
        # The simplest correct policy is: pick the face index with the largest
        # probability mass over the foreground (i.e., largest non-bg area).
        argmax = seg_probs.argmax(dim=1)
        seg_classes = argmax.detach().cpu().numpy().astype(np.uint8)
        if seg_classes.shape[0] > 1:
            non_bg = (seg_classes > 0).reshape(seg_classes.shape[0], -1).sum(axis=1)
            best = int(np.argmax(non_bg))
            seg_small = seg_classes[best]
        else:
            seg_small = seg_classes[0]

        if scale != 1.0:
            import cv2
            seg = cv2.resize(seg_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            seg = seg_small

        face_mask = np.isin(seg, list(_FACE_CORE_CLASSES))

        return SegmentationResult(
            seg_map=seg,
            face_mask=face_mask,
            n_classes=n_classes,
            scheme='facer_celebm_19',
        )

    def visualize(
        self,
        image_rgb: np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray:
        seg_color = colorize_seg(result.seg_map, n_classes=result.n_classes)
        masked = overlay_face_mask(image_rgb, result.face_mask)
        return np.concatenate([seg_color, masked], axis=1)

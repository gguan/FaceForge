"""
Multi-image shape aggregation: median/mean of shape codes across images.
"""

import cv2
import numpy as np
import torch

from .mica_inference import MICAInference
from .detection import RetinaFaceDetector


@torch.no_grad()
def aggregate_shapes(
    images_rgb: list[np.ndarray],
    mica: MICAInference,
    retina_detector: RetinaFaceDetector,
    method: str = 'median',
) -> tuple[torch.Tensor, list[dict]]:
    """Aggregate shape codes from multiple images of the same person.

    Args:
        images_rgb: List of RGB uint8 images
        mica: MICA inference model
        retina_detector: RetinaFace detector
        method: 'median' or 'mean'

    Returns:
        Tuple of:
            - Aggregated shape code [300]
            - List of per-image MICA outputs
    """
    shape_codes = []
    per_image_outputs = []

    for img_rgb in images_rgb:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        kps = retina_detector.detect_5pt(img_bgr)
        if kps is None:
            continue

        result = mica.run(img_bgr, kps)
        shape_codes.append(result['shape_code'].squeeze(0))  # [300]
        per_image_outputs.append(result)

    if len(shape_codes) == 0:
        raise ValueError("No faces detected in any of the input images")

    stacked = torch.stack(shape_codes, dim=0)  # [N, 300]

    if method == 'median':
        aggregated = torch.median(stacked, dim=0).values  # [300]
    elif method == 'mean':
        aggregated = stacked.mean(dim=0)  # [300]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregated, per_image_outputs

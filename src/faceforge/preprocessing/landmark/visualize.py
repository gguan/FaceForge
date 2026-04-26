"""Landmark visualization helpers."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# SCRFD 5pt order: eye_r, eye_l, nose, mouth_r, mouth_l (BGR colors).
_KPS_5PT_COLORS_BGR = [
    (0, 0, 255),       # eye_r — red
    (0, 255, 0),       # eye_l — green
    (255, 0, 0),       # nose  — blue
    (0, 255, 255),     # mouth_r — yellow
    (255, 0, 255),     # mouth_l — magenta
]


def draw_landmarks(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    kps_5pt: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> np.ndarray:
    """Render dense landmarks (+ optional bbox / 5pt / title) on a copy of the image.

    Args:
        image_rgb: source RGB uint8 image
        landmarks: [N, 2] dense landmark pixel coords
        bbox: optional [4] face bbox (x1, y1, x2, y2)
        kps_5pt: optional [5, 2] ArcFace-order 5-point landmarks
        title: optional caption rendered top-left

    Returns:
        RGB uint8 debug image (same H×W as input).
    """
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]
    diag = float(np.hypot(h, w))
    point_r = max(2, int(round(diag * 0.0015)))
    kps_r = max(4, int(round(diag * 0.004)))
    line_t = max(1, int(round(diag * 0.0008)))

    if bbox is not None:
        b = bbox.astype(int)
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]),
                      (0, 200, 200), max(1, line_t * 2), lineType=cv2.LINE_AA)

    for x, y in landmarks:
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   point_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))),
                   point_r, (0, 128, 0), line_t, lineType=cv2.LINE_AA)

    if kps_5pt is not None:
        for (x, y), color in zip(kps_5pt, _KPS_5PT_COLORS_BGR):
            cv2.circle(canvas, (int(round(x)), int(round(y))),
                       kps_r, color, -1, lineType=cv2.LINE_AA)

    if title is not None:
        font_scale = max(0.5, diag * 0.0008)
        font_thick = max(1, int(round(diag * 0.0015)))
        cv2.putText(canvas, title, (10, max(20, int(diag * 0.025))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thick + 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (10, max(20, int(diag * 0.025))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), font_thick, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

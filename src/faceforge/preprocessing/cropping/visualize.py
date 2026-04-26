"""Crop / alignment visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np

from ..landmark.visualize import _KPS_5PT_COLORS_BGR


def _draw_kps_5pt_bgr(canvas_bgr: np.ndarray, kps_5pt: np.ndarray, radius: int) -> None:
    for (x, y), color in zip(kps_5pt, _KPS_5PT_COLORS_BGR):
        cv2.circle(canvas_bgr, (int(round(x)), int(round(y))),
                   radius, color, -1, lineType=cv2.LINE_AA)


def _draw_dense_bgr(canvas_bgr: np.ndarray, lmks: np.ndarray, radius: int) -> None:
    for x, y in lmks:
        cv2.circle(canvas_bgr, (int(round(x)), int(round(y))),
                   radius, (0, 200, 0), -1, lineType=cv2.LINE_AA)


def draw_crop_overlay(image_rgb: np.ndarray, result) -> np.ndarray:
    """[source w/ bbox+quad+lmks | aligned crop | aligned crop w/ lmks].

    Args:
        image_rgb: source RGB uint8 image.
        result: a :class:`CropResult` instance.

    Returns:
        RGB uint8 triptych. All three panels are rescaled to the output size.
    """
    out_size = result.aligned_image.shape[0]

    # ----- panel 1: source overlay -----
    src_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = src_bgr.shape[:2]
    diag = float(np.hypot(h, w))
    line_t = max(1, int(round(diag * 0.0008)))
    point_r = max(2, int(round(diag * 0.0015)))
    kps_r = max(4, int(round(diag * 0.004)))

    if result.source_landmarks is not None:
        b = result.source_landmarks.bbox.astype(int)
        cv2.rectangle(src_bgr, (b[0], b[1]), (b[2], b[3]),
                      (0, 200, 200), max(1, line_t * 2), lineType=cv2.LINE_AA)
        _draw_dense_bgr(src_bgr, result.source_landmarks.landmarks, point_r)
        if result.source_landmarks.kps_5pt is not None:
            _draw_kps_5pt_bgr(src_bgr, result.source_landmarks.kps_5pt, kps_r)

    cv2.polylines(src_bgr, [result.crop_quad.astype(np.int32)],
                  isClosed=True, color=(0, 255, 0),
                  thickness=max(1, line_t * 2), lineType=cv2.LINE_AA)
    src_overlay = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)

    # ----- panel 2: aligned crop (clean) -----
    aligned_clean = result.aligned_image

    # ----- panel 3: aligned crop with landmarks -----
    al_bgr = cv2.cvtColor(aligned_clean, cv2.COLOR_RGB2BGR).copy()
    al_point_r = max(2, out_size // 256)
    al_kps_r = max(4, out_size // 96)
    for name, pts in result.landmarks_aligned.items():
        if name == 'kps_5pt':
            _draw_kps_5pt_bgr(al_bgr, pts, al_kps_r)
        else:
            _draw_dense_bgr(al_bgr, pts, al_point_r)
    aligned_overlay = cv2.cvtColor(al_bgr, cv2.COLOR_BGR2RGB)

    def _resize_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        new_w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    panels = [_resize_to_h(p, out_size) for p in (src_overlay, aligned_clean, aligned_overlay)]
    return np.concatenate(panels, axis=1)


def make_crop_summary_strip(image_rgb: np.ndarray, result) -> np.ndarray:
    """Alias for :func:`draw_crop_overlay` — kept for naming symmetry with
    the other components' ``make_*_summary_strip`` helpers."""
    return draw_crop_overlay(image_rgb, result)

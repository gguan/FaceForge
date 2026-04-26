"""Segmentation visualization helpers."""

from __future__ import annotations

import numpy as np


# Stable, distinguishable colors. Index 0 is reserved for background (black).
# Generated with distinctipy; hard-coded to avoid the runtime dependency.
_PALETTE_RGB = np.array([
    [  0,   0,   0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
    [  0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240,  50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
    [220, 190, 255], [170, 110, 40], [255, 250, 200], [128,   0,   0],
    [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128],
    [128, 128, 128], [255, 255, 255],
], dtype=np.uint8)


def colorize_seg(seg_map: np.ndarray, n_classes: int) -> np.ndarray:
    """Map an integer parsing map to an RGB color visualization.

    Args:
        seg_map: [H, W] integer segmentation map
        n_classes: number of classes (so palette length is sized correctly)

    Returns:
        RGB uint8 image [H, W, 3]
    """
    h, w = seg_map.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    palette = _PALETTE_RGB
    if n_classes > len(palette):
        # Tile the palette rather than fail.
        reps = (n_classes + len(palette) - 1) // len(palette)
        palette = np.tile(palette, (reps, 1))[:n_classes]

    for cls_idx in range(n_classes):
        mask = seg_map == cls_idx
        if mask.any():
            canvas[mask] = palette[cls_idx]
    return canvas


def overlay_face_mask(
    image_rgb: np.ndarray,
    face_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Tint the face region green; everything else is darkened.

    Args:
        image_rgb: source RGB uint8 [H, W, 3]
        face_mask: bool [H, W] — True for face pixels
        alpha: how strongly to tint the masked region

    Returns:
        RGB uint8 image [H, W, 3]
    """
    canvas = image_rgb.astype(np.float32)
    bg = ~face_mask

    # Darken outside-face
    canvas[bg] *= 0.35

    # Tint inside-face green
    tint = np.array([20, 220, 20], dtype=np.float32)
    canvas[face_mask] = (1.0 - alpha) * canvas[face_mask] + alpha * tint

    return np.clip(canvas, 0, 255).astype(np.uint8)

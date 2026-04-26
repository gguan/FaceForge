"""Matting visualization helpers."""

from __future__ import annotations

import numpy as np


def composite_alpha(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite *image_rgb* over a flat background using *alpha*.

    Args:
        image_rgb: RGB uint8 [H, W, 3]
        alpha: float32 [H, W] in [0, 1]
        background_rgb: tuple — solid color to composite over

    Returns:
        RGB uint8 image [H, W, 3]
    """
    a = alpha[..., None].astype(np.float32)
    bg = np.broadcast_to(np.asarray(background_rgb, dtype=np.float32), image_rgb.shape)
    out = a * image_rgb.astype(np.float32) + (1.0 - a) * bg
    return np.clip(out, 0, 255).astype(np.uint8)


def alpha_to_rgb(alpha: np.ndarray) -> np.ndarray:
    """Render an alpha map as a 3-channel grayscale image (0–255 RGB)."""
    a = np.clip(alpha, 0.0, 1.0)
    a8 = (a * 255.0).astype(np.uint8)
    return np.stack([a8, a8, a8], axis=-1)


def side_by_side_matte(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """[source | alpha-as-gray | foreground-on-white] horizontal triptych."""
    panels = [
        image_rgb,
        alpha_to_rgb(alpha),
        composite_alpha(image_rgb, alpha, (255, 255, 255)),
    ]
    return np.concatenate(panels, axis=1)

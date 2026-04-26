"""
FFHQ-style alignment geometry.

The 5-point SCRFD output (eye_r, eye_l, nose, mouth_r, mouth_l) anchors an
oriented quadrilateral in source-image coordinates following the same
recipe as NVlabs/ffhq-dataset, parameterized so callers can pick the FFHQ
(scale=1.0) or tracking-style (scale=1.3) crop. ``standardize_crop()`` then
warps that quad onto a square output and returns the forward perspective
transform.

This module is *purely geometric* — no model loading, no detection. It is
imported both by :class:`FFHQCropper` (the preprocessing component) and
by anyone who wants to reuse the math directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import PIL.Image
import scipy.ndimage


def _ffhq_quad_from_eye_mouth(
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    mouth_left: np.ndarray,
    mouth_right: np.ndarray,
    scale_factor: float,
) -> tuple[np.ndarray, float]:
    """Compute the FFHQ oriented crop quad from eye+mouth keypoints.

    Returns (quad [4, 2] in source coords, qsize) where the quad corners
    correspond to output (TL, BL, BR, TR) in that order.
    """
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1.0, 1.0]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) * scale_factor
    y = np.flipud(x) * [-1.0, 1.0]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = float(np.hypot(*x) * 2.0)
    return quad, qsize


def standardize_crop(
    image_rgb: np.ndarray,
    kps_5pt: np.ndarray,
    output_size: int = 512,
    transform_size: int = 1024,
    scale_factor: float = 1.3,
    padding_mode: str = 'constant',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFHQ-style aligned crop driven by SCRFD 5-point landmarks.

    Args:
        image_rgb: RGB uint8 [H, W, 3]
        kps_5pt: [5, 2] in SCRFD/ArcFace order (eye_r, eye_l, nose, mouth_r, mouth_l)
        output_size: final square resolution
        transform_size: internal working resolution
        scale_factor: 1.0 = FFHQ tight crop, 1.3 = tracking-style head crop
        padding_mode: 'constant' (white fill) or 'reflect'

    Returns:
        aligned_image [output_size, output_size, 3] uint8 RGB,
        transform_M [3, 3] perspective: source coords → aligned coords,
        crop_quad [4, 2] in source coords (TL, BL, BR, TR).
    """
    if kps_5pt.shape != (5, 2):
        raise ValueError(f"kps_5pt must be [5, 2], got {kps_5pt.shape}")

    eye_right, eye_left = kps_5pt[0].astype(np.float64), kps_5pt[1].astype(np.float64)
    mouth_right, mouth_left = kps_5pt[3].astype(np.float64), kps_5pt[4].astype(np.float64)

    quad, qsize = _ffhq_quad_from_eye_mouth(
        eye_left=eye_left,
        eye_right=eye_right,
        mouth_left=mouth_left,
        mouth_right=mouth_right,
        scale_factor=scale_factor,
    )
    quad_orig = quad.copy()

    img_pil = PIL.Image.fromarray(image_rgb)

    # Shrink (anti-aliasing for very large source images)
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img_pil.size[0]) / shrink)),
            int(np.rint(float(img_pil.size[1]) / shrink)),
        )
        img_pil = img_pil.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img_pil.size[0]),
        min(crop[3] + border, img_pil.size[1]),
    )
    if crop[2] - crop[0] < img_pil.size[0] or crop[3] - crop[1] < img_pil.size[1]:
        img_pil = img_pil.crop(crop)
        quad -= crop[0:2]

    # Pad
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img_pil.size[0] + border, 0),
        max(pad[3] - img_pil.size[1] + border, 0),
    )
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        if padding_mode == 'reflect':
            img_np = np.pad(
                np.float32(img_pil),
                ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                'reflect',
            )
        else:
            img_np = np.pad(
                np.float32(img_pil),
                ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                'constant',
                constant_values=255,
            )
        h, w, _ = img_np.shape
        y_grid, x_grid, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x_grid) / pad[0], np.float32(w - 1 - x_grid) / pad[2]),
            1.0 - np.minimum(np.float32(y_grid) / pad[1], np.float32(h - 1 - y_grid) / pad[3]),
        )
        blur = qsize * 0.02
        img_np += (scipy.ndimage.gaussian_filter(img_np, [blur, blur, 0]) - img_np) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img_np += (np.median(img_np, axis=(0, 1)) - img_np) * np.clip(mask, 0.0, 1.0)
        img_pil = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_np), 0, 255)), 'RGB')
        quad += pad[:2]

    # Warp
    img_pil = img_pil.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img_pil = img_pil.resize((output_size, output_size), PIL.Image.LANCZOS)

    aligned = np.asarray(img_pil)

    # Compute the source → aligned perspective transform from the *original*
    # quad (we operated on a shrunk/cropped/padded copy purely to make the
    # warp numerically clean; the math here works on source coords).
    # PIL.Image.QUAD output corners: (0,0), (0,h), (w,h), (w,0)
    src_pts = quad_orig.astype(np.float32)
    dst_pts = np.array(
        [[0, 0], [0, output_size], [output_size, output_size], [output_size, 0]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return aligned, M, quad_orig.astype(np.float32)


def project_points(points_xy: np.ndarray, transform_M: np.ndarray) -> np.ndarray:
    """Apply a 3×3 perspective transform to 2D points.

    Use this to map source-image landmarks (or any 2D point) into the
    aligned-image frame after :func:`standardize_crop` runs, or — given
    ``np.linalg.inv(M)`` — to warp aligned-frame outputs back into
    source-image coordinates.

    Args:
        points_xy: [N, 2] (or [N, ≥2] — only the first two columns are used).
        transform_M: [3, 3] perspective matrix.

    Returns:
        [N, 2] float32 transformed points.
    """
    pts = points_xy[:, :2].astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_warp = (transform_M @ pts_h.T).T
    return (pts_warp[:, :2] / pts_warp[:, 2:3]).astype(np.float32)

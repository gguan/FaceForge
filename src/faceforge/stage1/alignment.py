"""
Face alignment/cropping based on 68-point landmarks.

Ported from: flame-head-tracker/utils/image_utils.py image_align()
Original: FFHQ dataset pre-processing step (NVlabs)
"""

import cv2
import numpy as np
import PIL.Image
import scipy.ndimage


def image_align(
    img: np.ndarray,
    face_landmarks: np.ndarray,
    output_size: int = 512,
    transform_size: int = 1024,
    scale_factor: float = 1.3,
    padding_mode: str = 'constant',
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Align and crop face image using 68-point landmarks.

    Args:
        img: RGB uint8 image [H, W, 3]
        face_landmarks: [68, 2] or [68, 3] dlib-style landmarks
        output_size: Final output resolution (default 512)
        transform_size: Internal working resolution (default 1024)
        scale_factor: Crop scale (1.3 for tracking, 1.0 for FFHQ)
        padding_mode: 'constant' (white fill) or 'reflect'
        return_transform: If True, also return the 3x3 perspective matrix
            that maps original image coords → aligned image coords.

    Returns:
        Aligned image [output_size, output_size, 3] uint8.
        If return_transform=True, returns (aligned_image, transform_matrix).
    """
    lm = np.array(face_landmarks)
    lm_eye_left = lm[36:42, :2]
    lm_eye_right = lm[42:48, :2]
    lm_mouth_outer = lm[48:60, :2]

    # Auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) * scale_factor
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    # Save original quad for transform computation
    quad_orig = quad.copy()
    qsize = np.hypot(*x) * 2

    # Convert to PIL
    img_pil = PIL.Image.fromarray(img)

    # Shrink
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

    # Transform
    img_pil = img_pil.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img_pil = img_pil.resize((output_size, output_size), PIL.Image.LANCZOS)

    aligned = np.asarray(img_pil)

    if not return_transform:
        return aligned

    # Compute perspective transform: original image coords → aligned image coords.
    # PIL QUAD maps output corners → source (quad) corners.
    # quad_orig holds source positions of the 4 output corners:
    #   quad_orig[0] → output (0, 0)         top-left
    #   quad_orig[1] → output (0, out_sz)    bottom-left
    #   quad_orig[2] → output (out_sz, out_sz) bottom-right
    #   quad_orig[3] → output (out_sz, 0)    top-right
    src_pts = quad_orig.astype(np.float32)
    dst_pts = np.array([
        [0, 0],
        [0, output_size],
        [output_size, output_size],
        [output_size, 0],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return aligned, M

"""
Build the ``face_mask`` and ``head_mask`` HRN-head expects, using our
own segmentation backends in place of modelscope's TF1 .pb mattes.

modelscope HRN-head uses two TF1 GraphDef models that we choose not to
depend on:

  * ``segment_face.pb``         → face alpha matte (skin only).
  * ``Matting_headparser_6_18.pb`` → full head alpha (hair + ears + neck).

Both are consumed at full image resolution and later cropped/resized via
``align_img`` to the 224×224 working frame. We substitute them with our
existing :class:`BiSeNetSegmenter` / :class:`FacerSegmenter`, taking
class-union masks that approximate the same regions.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from faceforge.preprocessing.segmentation.base import (
    BISENET_CLASSES, FACER_CELEBM_CLASSES, BaseSegmenter, SegmentationResult,
)


# Class IDs that count as "face" (skin + eyes + brows + nose + lips), per scheme.
_FACE_CLASSES = {
    'bisenet_19':       {1, 2, 3, 4, 5, 10, 11, 12, 13},          # skin/brows/eyes/nose/mouth/lips
    'facer_celebm_19':  {2, 6, 7, 8, 9, 10, 11, 12, 13},          # face/brows/eyes/nose/mouth/lips
}

# Class IDs that count as "head" (face ∪ hair ∪ ears ∪ neck) — used as the
# target silhouette for the Dice loss in the fitting loop.
_HEAD_CLASSES = {
    'bisenet_19':       {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18},
    'facer_celebm_19':  {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18},
}


def _union_mask(seg_map: np.ndarray, class_ids: Iterable[int]) -> np.ndarray:
    """Return a uint8 [H, W] mask in {0, 255} for pixels whose class is in
    ``class_ids``. The 0/255 range matches what modelscope's TF mattes
    produce, so downstream code (``align_img``, ``read_data``) doesn't
    need to be tweaked."""
    ids = np.asarray(list(class_ids), dtype=seg_map.dtype)
    mask = np.isin(seg_map, ids)
    return (mask.astype(np.uint8) * 255)


def derive_face_and_head_masks(
    segmentation: SegmentationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(face_mask_u8, head_mask_u8)`` in image-source resolution.

    Args:
        segmentation: a :class:`SegmentationResult` from any registered
            segmentation backend whose ``scheme`` is ``'bisenet_19'`` or
            ``'facer_celebm_19'``.

    Returns:
        ``(face_mask, head_mask)`` — both uint8 in [0, 255], shape [H, W].
    """
    scheme = segmentation.scheme
    if scheme not in _FACE_CLASSES:
        raise ValueError(
            f"unknown segmentation scheme {scheme!r} for HRN-head mask "
            f"adapter; supported: {sorted(_FACE_CLASSES)}"
        )
    seg_map = np.asarray(segmentation.seg_map)
    if seg_map.ndim != 2:
        raise ValueError(f"seg_map must be [H, W], got {seg_map.shape}")

    face_mask = _union_mask(seg_map, _FACE_CLASSES[scheme])
    head_mask = _union_mask(seg_map, _HEAD_CLASSES[scheme])
    return face_mask, head_mask


def run_segmentation_for_hrn_head(
    image_rgb: np.ndarray,
    segmenter: Optional[BaseSegmenter] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: lazily build a :class:`BiSeNetSegmenter` if no
    segmenter is provided, run it on the image, and return masks.

    Use the explicit ``segmenter`` arg from a pipeline that already keeps
    a segmenter around so the model isn't reloaded per call.
    """
    if segmenter is None:
        from faceforge.preprocessing.segmentation.bisenet import (
            BiSeNetConfig, BiSeNetSegmenter,
        )
        segmenter = BiSeNetSegmenter(BiSeNetConfig())
    seg_result = segmenter.run(image_rgb)
    return derive_face_and_head_masks(seg_result)


__all__ = [
    'derive_face_and_head_masks',
    'run_segmentation_for_hrn_head',
]

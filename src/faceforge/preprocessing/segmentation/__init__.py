"""Facial segmentation backends."""

from .base import SegmentationResult, BaseSegmenter, BISENET_CLASSES, FACER_CELEBM_CLASSES
from .bisenet import BiSeNetSegmenter, BiSeNetConfig
from .visualize import colorize_seg, overlay_face_mask

__all__ = [
    'SegmentationResult',
    'BaseSegmenter',
    'BISENET_CLASSES',
    'FACER_CELEBM_CLASSES',
    'BiSeNetSegmenter',
    'BiSeNetConfig',
    'colorize_seg',
    'overlay_face_mask',
]


def make_segmenter(backend: str, **kwargs) -> 'BaseSegmenter':
    backend = backend.lower()
    if backend in ('bisenet', 'celeba_19'):
        return BiSeNetSegmenter(BiSeNetConfig(**kwargs))
    if backend in ('facer', 'farl', 'facer_celebm'):
        from .facer_celebm import FacerSegmenter, FacerConfig
        return FacerSegmenter(FacerConfig(**kwargs))
    raise ValueError(f"unknown segmentation backend: {backend!r}")

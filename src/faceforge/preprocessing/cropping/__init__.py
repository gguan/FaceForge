"""Face cropping / alignment backends.

Cropping sits at the boundary between landmark detection and the rest of
the preprocessing chain: it consumes detected keypoints and produces an
aligned head crop in a known canonical frame, plus the forward transform
needed to project landmarks/masks into that frame.

Right now only FFHQ-style alignment driven by 5-point landmarks is
exposed. Other recipes (ArcFace 112×112 norm-crop, MICA-style ArcFace
template, custom 68-point alignment) can plug in by adding new backends
here.
"""

from ._geometry import project_points, standardize_crop
from .base import CropResult, BaseCropper
from .ffhq import FFHQCropper, FFHQCropConfig
from .visualize import draw_crop_overlay, make_crop_summary_strip

__all__ = [
    'CropResult',
    'BaseCropper',
    'FFHQCropper',
    'FFHQCropConfig',
    'project_points',
    'standardize_crop',
    'draw_crop_overlay',
    'make_crop_summary_strip',
]


def make_cropper(backend: str, **kwargs) -> 'BaseCropper':
    backend = backend.lower()
    if backend in ('ffhq', 'ffhq_5pt'):
        return FFHQCropper(FFHQCropConfig(**kwargs))
    raise ValueError(f"unknown cropping backend: {backend!r}")

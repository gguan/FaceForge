"""
Composable preprocessing components for face/avatar pipelines.

Each component is a self-contained module with a small dataclass for its
result type, a `run()` entry point, and a `visualize()` debug renderer.
Components can be wired into different pipelines (MonoNPHM-style sequence
preprocessing, pixel3dmm tracking, custom flows) by instantiating only the
ones a given pipeline needs.

Available components
--------------------
landmark/        SCRFD + 106 (InsightFace) or FaceBoxes + 98 (PIPNet/WFLW)
cropping/        FFHQ-style aligned head crop driven by 5-point landmarks
segmentation/    BiSeNet (CelebAMask-HQ 19-class), facer/FaRL CelebM
matting/         MODNet portrait alpha matte
flame_tracking/  metrical-tracker FLAME parameter optimization
"""

from .base import BasePreprocessor, ComponentResult

__all__ = [
    'BasePreprocessor',
    'ComponentResult',
]

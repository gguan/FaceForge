"""Landmark detection backends."""

from .base import LandmarkResult, BaseLandmarkDetector
from .insightface_106 import InsightFace106Detector, InsightFace106Config
from .visualize import draw_landmarks

__all__ = [
    'LandmarkResult',
    'BaseLandmarkDetector',
    'InsightFace106Detector',
    'InsightFace106Config',
    'draw_landmarks',
]


def make_landmark_detector(backend: str, **kwargs) -> 'BaseLandmarkDetector':
    """Factory: dispatch to the requested backend.

    Args:
        backend: 'insightface_106' or 'pipnet_98'
        **kwargs: forwarded to the backend's Config

    Returns:
        a BaseLandmarkDetector instance
    """
    backend = backend.lower()
    if backend in ('insightface_106', 'scrfd_106', 'insightface'):
        return InsightFace106Detector(InsightFace106Config(**kwargs))
    if backend in ('pipnet_98', 'pipnet'):
        from .pipnet_98 import PIPNet98Detector, PIPNet98Config
        return PIPNet98Detector(PIPNet98Config(**kwargs))
    raise ValueError(f"unknown landmark backend: {backend!r}")

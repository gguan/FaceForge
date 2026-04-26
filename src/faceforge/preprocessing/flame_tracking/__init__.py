"""FLAME tracking backends.

Unlike landmark/segmentation/matting, FLAME tracking is *sequence-based*:
it consumes a folder of images plus precomputed landmarks/seg/matte and
produces a per-frame stream of FLAME parameters + a fitted mesh. Hence
the base class here is :class:`BaseFlameTracker`, not
:class:`BasePreprocessor` — the single-image surface doesn't fit.
"""

from .base import (
    FlameTrackingResult,
    BaseFlameTracker,
    SequenceInputs,
)
from .metrical_tracker import MetricalTracker, MetricalTrackerConfig
from .visualize import load_tracker_overlay, make_tracker_summary_strip

__all__ = [
    'FlameTrackingResult',
    'BaseFlameTracker',
    'SequenceInputs',
    'MetricalTracker',
    'MetricalTrackerConfig',
    'load_tracker_overlay',
    'make_tracker_summary_strip',
]

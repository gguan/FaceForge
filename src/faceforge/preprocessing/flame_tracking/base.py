"""FLAME tracking sequence types + base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SequenceInputs:
    """Filesystem layout that a sequence-based FLAME tracker consumes.

    A pipeline is responsible for populating these directories — typically
    by running the per-frame components (landmark / segmentation / matting)
    plus a one-off MICA shape estimate — before invoking the tracker.

    All paths are absolute. ``mica_identity_npy`` is the file produced by
    MICA's ``demo.py`` (a [300] shape coefficient vector).
    """

    seq_root: Path                    # holds source/, bboxes/, pipnet/, seg/, ...
    intrinsics_provided: bool = True
    keyframes: list[int] = field(default_factory=lambda: [0])

    # The tracker resolves the rest by convention:
    #   <seq_root>/source/         RGB frames
    #   <seq_root>/bboxes/         per-frame bbox.npy
    #   <seq_root>/pipnet/         per-frame 98-pt landmarks
    #   <seq_root>/seg/            per-frame parsing maps
    #   <seq_root>/matting/        per-frame alpha mattes
    #   <seq_root>/identity.npy    MICA shape code
    mica_identity_npy: Optional[Path] = None


@dataclass
class FlameTrackingResult:
    """Output of a sequence-level FLAME tracker.

    Attributes:
        output_dir: where the tracker wrote its artefacts (camera/, mesh/,
            video.avi, checkpoint/, etc.)
        per_frame_params: dict mapping field name to [N_frames, ...] arrays
            (filled in lazily by `BaseFlameTracker.read_results`).
        flame_shape: [300] aggregated identity shape (if optimized).
        rendered_video: optional path to the tracker's preview video.
    """

    output_dir: Path
    per_frame_params: dict[str, np.ndarray] = field(default_factory=dict)
    flame_shape: Optional[np.ndarray] = None
    rendered_video: Optional[Path] = None


class BaseFlameTracker(ABC):
    """Common surface for sequence-based FLAME trackers."""

    name: str = 'flame_tracker'

    @abstractmethod
    def run_sequence(self, inputs: SequenceInputs) -> FlameTrackingResult:
        """Optimize FLAME params over a full sequence."""

    @abstractmethod
    def visualize(
        self,
        result: FlameTrackingResult,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Return an RGB uint8 debug overlay for the given output frame."""

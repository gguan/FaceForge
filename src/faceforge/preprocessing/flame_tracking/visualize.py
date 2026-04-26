"""FLAME-tracking visualization helpers.

The metrical-tracker dumps several viewable artefacts under its
``save_folder``: ``video/`` (RGB w/ landmarks), ``mesh/`` (textured shape
overlays), ``initialization/`` (first-pass overlays). These helpers just
load and arrange the relevant frames so callers can sanity-check tracking.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_tracker_overlay(
    output_dir: Path,
    frame_idx: int = 0,
    subdir: str = 'video',
) -> np.ndarray:
    """Load one of the tracker's per-frame overlay images.

    Args:
        output_dir: the tracker's ``save_folder`` (the directory containing
            ``video/`` and ``mesh/``).
        frame_idx: zero-based frame index.
        subdir: which artefact subfolder to read from. ``'video'`` is the
            RGB+landmarks composite; other options include ``'mesh'`` and
            ``'initialization'``.

    Returns:
        RGB uint8 image.
    """
    output_dir = Path(output_dir)
    candidates = sorted((output_dir / subdir).glob('*.jpg')) + \
                 sorted((output_dir / subdir).glob('*.png'))
    if not candidates:
        raise FileNotFoundError(
            f"no images under {output_dir / subdir} — has the tracker run?"
        )
    if frame_idx < 0 or frame_idx >= len(candidates):
        raise IndexError(
            f"frame_idx {frame_idx} out of range (have {len(candidates)} frames)"
        )
    return _read_rgb(candidates[frame_idx])


def make_tracker_summary_strip(
    output_dir: Path,
    frame_idx: int = 0,
    subdirs: tuple[str, ...] = ('video', 'mesh'),
) -> np.ndarray:
    """Concatenate one frame from each requested subdirectory horizontally."""
    output_dir = Path(output_dir)
    panels: list[np.ndarray] = []
    target_h: int | None = None
    for sub in subdirs:
        try:
            img = load_tracker_overlay(output_dir, frame_idx=frame_idx, subdir=sub)
        except FileNotFoundError:
            continue
        if target_h is None:
            target_h = img.shape[0]
        elif img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = max(1, int(round(img.shape[1] * scale)))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        panels.append(img)
    if not panels:
        raise FileNotFoundError(
            f"no overlay images found under {output_dir} for subdirs={subdirs}"
        )
    return np.concatenate(panels, axis=1)

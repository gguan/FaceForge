"""Background matting result + base class."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from ..base import BasePreprocessor, ComponentResult


@dataclass
class MatteResult(ComponentResult):
    """Per-image alpha matte.

    Attributes:
        alpha: [H, W] float32 in [0, 1] — 1 is foreground, 0 is background.
        scheme: backend identifier (e.g., 'modnet').
    """

    alpha: np.ndarray
    scheme: str


class BaseMatter(BasePreprocessor):
    """Common surface for matting backends."""

    @abstractmethod
    def run(self, image_rgb: np.ndarray) -> MatteResult:
        ...

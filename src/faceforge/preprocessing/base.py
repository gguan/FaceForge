"""
Common abstractions for preprocessing components.

Every component in this package should:

  1. Define a ``Config`` dataclass describing its tunables.
  2. Define a ``Result`` dataclass (subclass of :class:`ComponentResult`)
     holding what the component produces for a single image.
  3. Subclass :class:`BasePreprocessor` and implement
     ``run(image_rgb) -> Result`` and ``visualize(image_rgb, result) -> RGB``.

The base class only standardizes the surface — it does not impose anything
heavy. Components remain free to expose backend-specific extras (e.g.
batch APIs, mid-network features) on top of this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentResult:
    """Marker base class for per-image preprocessing results."""


class BasePreprocessor(ABC):
    """Single-image preprocessing component.

    Subclasses implement two methods:

    - ``run(image_rgb)``     : compute the result for one RGB uint8 image.
    - ``visualize(image_rgb, result)`` : render an RGB debug overlay.

    Pipelines compose components by calling ``run()`` and stashing the
    result; visualization is opt-in and only used for debugging.
    """

    name: str = 'preprocessor'

    @abstractmethod
    def run(self, image_rgb: np.ndarray) -> ComponentResult:
        """Process a single RGB uint8 image."""

    @abstractmethod
    def visualize(self, image_rgb: np.ndarray, result: ComponentResult) -> np.ndarray:
        """Return an RGB uint8 debug image for the given (image, result) pair."""

    def run_batch(self, images_rgb: list[np.ndarray]) -> list[ComponentResult]:
        """Default batch implementation — subclasses may override for speed."""
        return [self.run(img) for img in images_rgb]

"""Background matting backends."""

from .base import MatteResult, BaseMatter
from .modnet import MODNetMatter, MODNetConfig
from .visualize import composite_alpha, side_by_side_matte

__all__ = [
    'MatteResult',
    'BaseMatter',
    'MODNetMatter',
    'MODNetConfig',
    'composite_alpha',
    'side_by_side_matte',
]


def make_matter(backend: str, **kwargs) -> 'BaseMatter':
    backend = backend.lower()
    if backend in ('modnet',):
        return MODNetMatter(MODNetConfig(**kwargs))
    raise ValueError(f"unknown matting backend: {backend!r}")

"""MonoNPHM monocular Neural Parametric Head Model wrapper."""

from faceforge.models.registry import register_model

from .wrapper import MonoNPHMConfig, MonoNPHMModel

register_model('mononphm', MonoNPHMModel)

__all__ = ['MonoNPHMConfig', 'MonoNPHMModel']

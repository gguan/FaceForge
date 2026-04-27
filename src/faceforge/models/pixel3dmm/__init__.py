"""pixel3dmm tracker wrapper for the unified model interface."""

from faceforge.models.registry import register_model

from .wrapper import Pixel3DMMConfig, Pixel3DMMModel

register_model('pixel3dmm', Pixel3DMMModel)

__all__ = ['Pixel3DMMConfig', 'Pixel3DMMModel']

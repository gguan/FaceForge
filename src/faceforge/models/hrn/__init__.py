"""HRN single-image reconstruction wrappers.

Two flavours:

* :class:`HRNModel` (``'hrn'``) — face-only HRN bundled in
  ``submodules/HRN``. Outputs a face mesh + BFM coefficients.
* :class:`HRNHeadModel` (``'hrn_head'``) — full-head HRN-head from the
  modelscope ``cv_HRN_head-reconstruction`` model. Outputs a complete
  textured head mesh (face + scalp + ears + neck) ready to OBJ-export.
"""

from faceforge.models.registry import register_model

from .head_wrapper import HRNHeadConfig, HRNHeadModel
from .wrapper import HRNConfig, HRNModel

register_model('hrn', HRNModel)
register_model('hrn_head', HRNHeadModel)

__all__ = [
    'HRNConfig', 'HRNModel',
    'HRNHeadConfig', 'HRNHeadModel',
]

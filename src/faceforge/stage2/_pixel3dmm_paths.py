"""Configure pixel3dmm env_paths from Stage2Config.

pixel3dmm components (FLAME, UVLoss) resolve asset paths via the
``pixel3dmm.env_paths`` module, which normally reads from
``~/.config/pixel3dmm/.env``.  This helper patches those paths at
runtime so they point to assets managed by FaceForge's own config,
avoiding the need for an external .env file.

Call ``configure_pixel3dmm_paths(config)`` once at pipeline init.
"""

import os
import sys
from pathlib import Path

from faceforge._paths import PROJECT_ROOT


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def configure_pixel3dmm_paths(config) -> None:
    """Patch ``pixel3dmm.env_paths`` so FLAME assets are resolved
    from *config* rather than from the .env file.

    This must be called **before** constructing any pixel3dmm object
    (FLAME, UVLoss, …).
    """
    code_base = str(_resolve(config.pixel3dmm_code_base))
    src_path = f'{code_base}/src'
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    os.environ.setdefault('PIXEL3DMM_CODE_BASE', code_base)
    os.environ.setdefault('PIXEL3DMM_PREPROCESSED_DATA', str(PROJECT_ROOT / 'output'))
    os.environ.setdefault('PIXEL3DMM_TRACKING_OUTPUT', str(PROJECT_ROOT / 'output'))

    import pixel3dmm.env_paths as env_paths

    flame_dir = str(_resolve(config.flame_model_path).parent)  # .../FLAME2020

    # FLAME model assets (used by pixel3dmm.tracking.flame.FLAME)
    # FLAME_ASSETS must point to the *parent* of FLAME2020/
    env_paths.FLAME_ASSETS = str(_resolve(config.flame_model_path).parent.parent)

    # UV / vertex assets (used by pixel3dmm.tracking.losses.UVLoss)
    env_paths.FLAME_UV_COORDS = str(_resolve(config.flame_uv_coords_path))
    env_paths.VALID_VERTS = str(_resolve(config.flame_uv_valid_verts_path))

    # Keep CODE_BASE pointing at the submodule root for any other
    # pixel3dmm code that may need it (e.g. head_template.obj).
    env_paths.CODE_BASE = code_base

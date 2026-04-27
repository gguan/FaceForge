"""
Tiny bridge to the geometry utilities bundled in ``submodules/HRN``.

The face-HRN repo at ``submodules/HRN/util/`` ships the same
``read_obj`` / ``estimate_normals`` / ``align_img`` / ``align_for_lm`` /
``load_lm3d`` helpers that modelscope's HRN-head pipeline needs. Rather
than re-vendor them, we import-from-path the existing copies. The path
patching matches the existing ``faceforge.models.hrn.wrapper._patched_sys``
contract.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from faceforge._paths import PROJECT_ROOT


_HRN_ROOT = (PROJECT_ROOT / 'submodules' / 'HRN').resolve()


def _patch_numpy_compat():
    """``submodules/HRN/util/preprocess.py`` filters
    ``np.VisibleDeprecationWarning`` at import time, but numpy >= 1.25
    removed that symbol. Restore a stand-in (a plain DeprecationWarning
    subclass) so the warnings-filter call doesn't blow up."""
    import numpy as np
    if not hasattr(np, 'VisibleDeprecationWarning'):
        class VisibleDeprecationWarning(DeprecationWarning):
            pass
        np.VisibleDeprecationWarning = VisibleDeprecationWarning  # type: ignore[attr-defined]


def _stub_torchsummary():
    """``submodules/HRN/facelandmark/nets/large_base_lmks_net.py`` does
    ``from torchsummary import summary`` at the top, but only uses it inside
    a ``if __name__ == '__main__':`` block we never run. Inject a no-op
    module so the import succeeds without pulling a dep we don't need."""
    import sys
    import types
    if 'torchsummary' in sys.modules:
        return
    mod = types.ModuleType('torchsummary')
    mod.summary = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules['torchsummary'] = mod


@contextlib.contextmanager
def _hrn_on_path():
    """Add ``submodules/HRN`` to sys.path for the duration of an import.

    The HRN repo's modules import each other with bare names
    (``from util.preprocess import align_img``), so its root must be on
    sys.path *and* must take precedence over any other ``util`` package.
    """
    _patch_numpy_compat()
    _stub_torchsummary()
    root = str(_HRN_ROOT)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted and root in sys.path:
            sys.path.remove(root)


def _import_hrn_helpers():
    with _hrn_on_path():
        from util.load_mats import load_lm3d as _load_lm3d
        from util.preprocess import (
            POS as _POS,
            align_img as _align_img,
            align_for_lm as _align_for_lm,
            extract_5p as _extract_5p,
            resize_n_crop_img as _resize_n_crop_img,
        )
        from util.util_ import (
            estimate_normals as _estimate_normals,
            read_obj as _read_obj,
            resize_on_long_side as _resize_on_long_side,
        )
    return {
        'POS': _POS,
        'align_img': _align_img,
        'align_for_lm': _align_for_lm,
        'extract_5p': _extract_5p,
        'resize_n_crop_img': _resize_n_crop_img,
        'load_lm3d': _load_lm3d,
        'estimate_normals': _estimate_normals,
        'read_obj': _read_obj,
        'resize_on_long_side': _resize_on_long_side,
    }


_helpers = _import_hrn_helpers()
POS = _helpers['POS']
_hrn_align_img = _helpers['align_img']
align_for_lm = _helpers['align_for_lm']
extract_5p = _helpers['extract_5p']
_hrn_resize_n_crop_img = _helpers['resize_n_crop_img']
load_lm3d = _helpers['load_lm3d']
estimate_normals = _helpers['estimate_normals']
resize_on_long_side = _helpers['resize_on_long_side']

_hrn_read_obj = _helpers['read_obj']


def _safe_resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    """Numpy-≥1.25-safe resize + crop.

    The bundled ``submodules/HRN/util/preprocess.resize_n_crop_img`` does
    ``float(np.ndarray_size_1)``; this raises ``TypeError`` on modern numpy
    because ``POS`` returns a column vector (shape ``(2, 1)``). The
    modelscope copy of the same function squeezes ``t`` upstream — we
    do the same defensively here.
    """
    import numpy as np
    from PIL import Image

    t = np.asarray(t).reshape(-1)
    s = float(s)

    w0, h0 = img.size
    w = int(np.round(w0 * s))
    h = int(np.round(h0 * s))
    left = int(np.round(w / 2 - target_size / 2 + (t[0] - w0 / 2) * s))
    right = int(left + target_size)
    up = int(np.round(h / 2 - target_size / 2 + (h0 / 2 - t[1]) * s))
    below = int(up + target_size)

    new_img = img.resize((w, h), resample=Image.BICUBIC).crop((left, up, right, below))
    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC).crop((left, up, right, below))

    new_lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    new_lm = new_lm - np.array([w / 2 - target_size / 2,
                                h / 2 - target_size / 2]).reshape(1, 2)
    return new_img, new_lm, mask


def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """Numpy-safe replacement for ``submodules/HRN/util/preprocess.align_img``.

    Same algorithm (POS-based 5-pt similarity → resize + crop), but with
    the column-vector squeeze that newer numpy needs to keep
    ``resize_n_crop_img`` from blowing up on ``float(arr_size_1)``.
    """
    import numpy as np

    w0, h0 = img.size
    lm5p = extract_5p(lm) if lm.shape[0] != 5 else lm
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    t = np.asarray(t).reshape(-1)
    s = rescale_factor / float(s)
    img_new, lm_new, mask_new = _safe_resize_n_crop_img(
        img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])
    return trans_params, img_new, lm_new, mask_new


def resize_n_crop_img(*args, **kwargs):
    """Public alias of the safe resize+crop. Kept for symmetry with the
    HRN/modelscope helper signature."""
    return _safe_resize_n_crop_img(*args, **kwargs)


def read_obj(obj_path, print_shape=False):
    """Wrap submodules/HRN's read_obj to normalize keys to modelscope's
    lowercase convention (``UVs`` → ``uvs``, ``vns`` → ``normals``)."""
    mesh = _hrn_read_obj(obj_path, print_shape=print_shape)
    if 'UVs' in mesh and 'uvs' not in mesh:
        mesh['uvs'] = mesh['UVs']
    if 'vns' in mesh and 'normals' not in mesh:
        mesh['normals'] = mesh['vns']
    return mesh


__all__ = [
    'POS', 'align_img', 'align_for_lm', 'extract_5p', 'resize_n_crop_img',
    'load_lm3d', 'estimate_normals', 'read_obj', 'resize_on_long_side',
]

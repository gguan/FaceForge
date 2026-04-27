"""
Compatibility shims auto-installed when ``_pytorch3d_shim/`` is on sys.path.

Python imports any module called ``sitecustomize`` once at startup, so
dropping this file alongside the pytorch3d shim is enough to monkey-patch
the third-party packages that MonoNPHM imports without us editing the
upstream tree.

Patches:

* numpy ≥ 2.0 removed the capitalized ``np.NaN`` / ``np.NAN`` aliases.
  MonoNPHM still uses ``np.NaN`` (e.g. ``utils/renderer.py``). Restore it.
* chumpy reaches for ``inspect.getargspec`` on first import — that helper
  was removed in Python 3.11. Forward to ``getfullargspec``.
"""

import inspect


def _patch_inspect_getargspec():
    if not hasattr(inspect, 'getargspec'):
        # chumpy only reads .args off the result; forward for that case.
        def _shim(func):
            spec = inspect.getfullargspec(func)
            return type('ArgSpec', (), {
                'args':     spec.args,
                'varargs':  spec.varargs,
                'keywords': spec.varkw,
                'defaults': spec.defaults,
            })
        inspect.getargspec = _shim   # type: ignore[attr-defined]


def _patch_numpy_capitalized():
    try:
        import numpy as np
    except ImportError:
        return
    for attr in ('NaN', 'NAN', 'Inf', 'INF', 'Infinity', 'PINF', 'NINF'):
        if not hasattr(np, attr):
            setattr(np, attr, getattr(np, attr.lower(), None))
    # Python's complex builtin replaces deprecated np.complex/np.float/np.int
    for old, new in (('complex', complex), ('float', float),
                     ('int', int), ('bool', bool), ('object', object)):
        if not hasattr(np, old):
            setattr(np, old, new)


_patch_inspect_getargspec()
_patch_numpy_capitalized()

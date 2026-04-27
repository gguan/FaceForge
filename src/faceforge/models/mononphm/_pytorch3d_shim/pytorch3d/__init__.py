"""
Minimal pytorch3d shim — only the bits MonoNPHM's inference path uses.

We don't take a hard pytorch3d build dependency because the upstream
package needs to be CUDA-compiled from source on Windows + recent CUDA,
which is a long, fragile build. The MonoNPHM ``rec.py`` path actually
touches only three pytorch3d functions:

  * ``pytorch3d.transforms.so3_exp_map``  (rendering.py, tracking.py)
  * ``pytorch3d.transforms.so3_log_map``  (tracking.py)
  * ``pytorch3d.ops.knn_points``          (canonical_space.py, reconstruction.py)

This shim provides drop-in replacements with the same signatures and
return conventions, in <100 LoC of pure torch. Inject into ``sys.path``
before importing ``mononphm``.
"""

__version__ = '0.0.0-shim'

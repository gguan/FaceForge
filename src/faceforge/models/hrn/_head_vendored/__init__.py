"""
Vendored modelscope HRN-head sources.

These files are adapted from
https://github.com/modelscope/modelscope/tree/master/modelscope/models/cv/head_reconstruction
(Apache License 2.0). Adaptations:

* Imports rewritten from ``modelscope.models.cv...`` → relative imports
  inside this subpackage.
* Asset/checkpoint paths take an explicit ``model_dir`` argument instead
  of the modelscope hub-download convention.
* The TensorFlow-1 .pb mattes (``segment_face.pb``,
  ``Matting_headparser_6_18.pb``) are NOT loaded here — the orchestrator
  in :mod:`faceforge.models.hrn.head_pipeline` substitutes them with our
  own BiSeNet/facer segmentation outputs (see :mod:`head_mask_adapter`).

The face HRN repo bundled at ``submodules/HRN`` provides the smaller
utilities (alignment, load_lm3d, RetinaFace, 106-pt landmarker) — those
are reused in-place rather than re-vendored.
"""

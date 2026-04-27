"""Smoke tests for the self-contained HRN-head wrapper.

These verify the module imports + asset list integrity + registry. The
full forward pass needs CUDA + nvdiffrast + the HRN submodule, gated
behind ``integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'hrn_head_model'


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_nvdiffrast() -> bool:
    try:
        import nvdiffrast  # noqa: F401
        return True
    except ImportError:
        return False


def test_wrapper_module_imports():
    from faceforge.models.hrn import head_wrapper
    assert hasattr(head_wrapper, 'HRNHeadModel')
    assert hasattr(head_wrapper, 'HRNHeadConfig')


def test_pipeline_module_imports():
    from faceforge.models.hrn import head_pipeline
    assert hasattr(head_pipeline, 'HRNHeadPipeline')
    assert hasattr(head_pipeline, 'HRNHeadPipelineConfig')


def test_registry_exposes_hrn_head():
    from faceforge.models.registry import list_models
    assert 'hrn_head' in list_models()


@pytest.mark.skipif(not DATA_DIR.exists(), reason="hrn_head_model assets not staged")
def test_required_assets_present():
    from faceforge.models.hrn.head_pipeline import _REQUIRED_FILES
    missing = [f for f in _REQUIRED_FILES if not (DATA_DIR / f).exists()]
    assert not missing, f"missing assets: {missing}"


def test_mask_adapter_class_unions():
    """Sanity: face/head class-id sets cover the right BiSeNet classes."""
    from faceforge.models.hrn.head_mask_adapter import (
        _FACE_CLASSES, _HEAD_CLASSES,
    )
    bn_face = _FACE_CLASSES['bisenet_19']
    bn_head = _HEAD_CLASSES['bisenet_19']
    # face must be a subset of head
    assert bn_face.issubset(bn_head)
    # head must include hair (17), neck (14), ears (7, 8)
    assert {7, 8, 14, 17}.issubset(bn_head)


@pytest.mark.integration
@pytest.mark.skipif(not DATA_DIR.exists(), reason="requires hrn_head_model assets")
@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA for nvdiffrast + HRN encoder")
@pytest.mark.skipif(not _has_nvdiffrast(), reason="requires nvdiffrast")
def test_end_to_end_on_cortis_image(tmp_path):
    """Reconstruct one cortis image and dump an OBJ. Skip if assets/cortis is empty."""
    cortis_img = PROJECT_ROOT / 'assets' / 'cortis' / '1.jpg'
    if not cortis_img.exists():
        pytest.skip("assets/cortis/1.jpg missing")

    import cv2
    from faceforge.models.hrn import HRNHeadConfig, HRNHeadModel
    from faceforge.pipeline.types import PreparedInputs

    bgr = cv2.imread(str(cortis_img))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model = HRNHeadModel(HRNHeadConfig(model_dir=str(DATA_DIR)))
    out = model.run(PreparedInputs(image_rgb=rgb, image_id='cortis_1'))

    assert out.mesh_vertices is not None and out.mesh_vertices.ndim == 2
    assert out.mesh_faces is not None and out.mesh_faces.shape[1] == 3
    assert 'texture_map' in out.extras
    assert out.extras['texture_map'].shape == (4096, 4096, 3)

    obj_path = model.write_obj(out, tmp_path / 'head.obj')
    assert obj_path.exists()
    assert obj_path.with_suffix('.mtl').exists()
    assert obj_path.with_suffix('.png').exists()

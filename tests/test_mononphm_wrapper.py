"""Smoke + integration tests for the MonoNPHM wrapper.

Quick tests verify imports + cmd construction + the bundled pytorch3d
shim's correctness. The full reconstruction is gated behind ``integration``
because it needs CUDA + the pretrained checkpoint + ~5 min of compute.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TRACKING = PROJECT_ROOT / 'data' / 'mononphm' / 'tracking_input'
EXP_DIR = PROJECT_ROOT / 'data' / 'pretrained'
DEMO_SEQ = DATA_TRACKING / '00059'


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_wrapper_module_imports():
    from faceforge.models.mononphm.wrapper import MonoNPHMConfig, MonoNPHMModel
    assert MonoNPHMConfig is not None
    assert MonoNPHMModel.name == 'mononphm'


def test_pytorch3d_shim_so3_roundtrip():
    """``so3_log_map ∘ so3_exp_map`` round-trips on random axis-angles."""
    import sys
    shim = (PROJECT_ROOT / 'src' / 'faceforge' / 'models' / 'mononphm'
            / '_pytorch3d_shim')
    if str(shim) not in sys.path:
        sys.path.insert(0, str(shim))

    import torch
    from pytorch3d.transforms import so3_exp_map, so3_log_map

    torch.manual_seed(0)
    v = torch.randn(8, 3) * 0.5
    R = so3_exp_map(v)
    v2 = so3_log_map(R)
    assert (v - v2).abs().max().item() < 1e-5


def test_pytorch3d_shim_knn_points_matches_brute_force():
    import sys
    shim = (PROJECT_ROOT / 'src' / 'faceforge' / 'models' / 'mononphm'
            / '_pytorch3d_shim')
    if str(shim) not in sys.path:
        sys.path.insert(0, str(shim))

    import numpy as np
    import torch
    from pytorch3d.ops import knn_points

    torch.manual_seed(1)
    p1 = torch.randn(1, 5, 3)
    p2 = torch.randn(1, 50, 3)
    res = knn_points(p1, p2, K=4)

    p1_np, p2_np = p1[0].numpy(), p2[0].numpy()
    d = ((p1_np[:, None, :] - p2_np[None, :, :]) ** 2).sum(-1)
    expected_idx = np.argsort(d, axis=-1)[:, :4]
    assert (res.idx[0].numpy() == expected_idx).all()


@pytest.mark.skipif(not EXP_DIR.exists(), reason='checkpoints not staged')
def test_wrapper_init_validates_assets():
    """Constructor should validate that the .tar + configs.yaml exist."""
    from faceforge.models.mononphm.wrapper import MonoNPHMConfig, MonoNPHMModel
    cfg = MonoNPHMConfig(seq_name='00059')
    m = MonoNPHMModel(cfg)
    assert 'pretrained_mononphm' in m._build_cmd(stage2=False)
    assert m._build_env()['MONONPHM_EXPERIMENT_DIR'].endswith('pretrained')


def test_wrapper_cmd_matches_readme_demo_invocation():
    """CLI flags must match the README's single-image FFHQ demo line:
       rec.py --model_type nphm --exp_name pretrained_mononphm --ckpt 2500
              --seq_name FFHQ_ID --no-intrinsics_provided
              --downsample_factor 0.33 --no-is_video
    """
    from faceforge.models.mononphm.wrapper import MonoNPHMConfig, MonoNPHMModel
    if not (EXP_DIR / 'pretrained_mononphm' / 'configs.yaml').exists():
        pytest.skip('checkpoint dir incomplete')
    cfg = MonoNPHMConfig(seq_name='00059', is_video=False,
                         intrinsics_provided=False, downsample_factor=0.33)
    m = MonoNPHMModel(cfg)
    cmd = m._build_cmd(stage2=False)
    assert '--model_type' in cmd and 'nphm' in cmd
    assert '--exp_name' in cmd and 'pretrained_mononphm' in cmd
    assert '--ckpt' in cmd and '2500' in cmd
    assert '--seq_name' in cmd and '00059' in cmd
    assert '--no-intrinsics_provided' in cmd
    assert '--no-is_video' in cmd
    assert '--no-is_stage2' in cmd
    assert '--downsample_factor' in cmd and '0.33' in cmd


@pytest.mark.integration
@pytest.mark.skipif(not _has_cuda(), reason='requires CUDA')
@pytest.mark.skipif(not DEMO_SEQ.exists(), reason='requires demo data 00059')
@pytest.mark.skipif(not (EXP_DIR / 'pretrained_mononphm' / 'configs.yaml').exists(),
                    reason='requires pretrained_mononphm checkpoints')
def test_demo_00059_end_to_end(tmp_path):
    """Reproduce the README's single-image FFHQ demo on 00059.

    Heavy: ~5 minutes on RTX-class hardware. Verifies the full pipeline
    (vendored pytorch3d shim + numpy/chumpy compat + wrapper CLI/env)
    produces the expected per-frame artefacts.
    """
    from faceforge.models.mononphm.wrapper import MonoNPHMConfig, MonoNPHMModel
    from faceforge.pipeline.types import PreparedInputs
    import numpy as np

    cfg = MonoNPHMConfig(
        seq_name='00059',
        is_video=False,
        intrinsics_provided=False,
        downsample_factor=0.33,
        clear_existing_output=True,
        tracking_output=str(tmp_path / 'mononphm_out'),
    )
    m = MonoNPHMModel(cfg)
    placeholder = [PreparedInputs(image_rgb=np.zeros((1, 1, 3), 'uint8'),
                                  image_id='00000')]
    outs = m.run_sequence(placeholder)

    assert len(outs) == 1
    out = outs[0]
    assert out.mesh_obj_path is not None and out.mesh_obj_path.exists()
    assert out.mesh_obj_path.suffix == '.ply'
    assert out.rendered_overlay is not None
    for key in ('z_geo', 'z_app', 'z_exp', 'rot', 'trans', 'scale'):
        assert key in out.extras, f'missing extras key: {key}'

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


def _make_stage1_output():
    from faceforge.stage1.data_types import Stage1Output

    aligned = torch.linspace(0, 1, 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    face_mask = torch.tensor(
        [[[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    parsing_map = torch.tensor(
        [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
        dtype=torch.long,
    )
    lmks_68 = torch.zeros(1, 68, 2)
    lmks_98 = torch.stack(
        [
            torch.linspace(0.0, 3.0, 98, dtype=torch.float32),
            torch.linspace(1.0, 4.0, 98, dtype=torch.float32),
        ],
        dim=1,
    ).unsqueeze(0)

    return Stage1Output(
        shape=torch.ones(1, 300),
        expression=torch.zeros(1, 100),
        head_pose=torch.zeros(1, 3),
        jaw_pose=torch.zeros(1, 3),
        texture=torch.zeros(1, 50),
        lighting=torch.zeros(1, 9, 3),
        arcface_feat=torch.zeros(1, 512),
        aligned_image=aligned,
        face_mask=face_mask,
        parsing_map=parsing_map,
        lmks_68=lmks_68,
        lmks_98=lmks_98,
        lmks_dense=torch.zeros(1, 478, 2),
        lmks_eyes=torch.zeros(1, 10, 2),
        focal_length=torch.ones(1, 1),
        principal_point=torch.zeros(1, 2),
    )


def test_write_preprocessed_uses_parsing_map_and_real_98_landmarks():
    from faceforge.stageP3M._data_bridge import write_preprocessed

    stage1_output = _make_stage1_output()

    class DummyInference:
        def predict(self, aligned_image, face_segmentation):
            assert torch.equal(face_segmentation, stage1_output.parsing_map)
            return torch.zeros(1, 2, 4, 4), torch.zeros(1, 3, 4, 4)

    with TemporaryDirectory() as tmp_dir:
        write_preprocessed(
            [stage1_output],
            DummyInference(),
            tmp_dir=tmp_dir,
            video_name='actor',
            render_size=4,
        )

        base = Path(tmp_dir) / 'actor'
        seg = cv2.imread(str(base / 'seg_og' / '00000.png'), cv2.IMREAD_UNCHANGED)
        lmks = np.load(base / 'PIPnet_landmarks' / '00000.npy')

        assert np.array_equal(seg, stage1_output.parsing_map.squeeze(0).numpy().astype(np.uint8))
        np.testing.assert_allclose(
            lmks,
            stage1_output.lmks_98.squeeze(0).numpy() / 4.0,
            atol=1e-6,
        )


def test_build_tracker_config_keeps_reference_defaults(tmp_path):
    from faceforge.stageP3M._data_bridge import build_tracker_config

    cfg = build_tracker_config(
        video_name='actor',
        preprocessed_dir=str(tmp_path / 'preprocessed'),
        output_dir=str(tmp_path / 'output'),
        n_frames=20,
        render_size=256,
        code_base='submodules/pixel3dmm',
    )
    base_cfg = OmegaConf.load('submodules/pixel3dmm/configs/tracking.yaml')

    assert cfg.video_name == 'actor'
    assert cfg.output_folder == str(tmp_path / 'output')
    assert cfg.size == 256
    assert list(cfg.image_size) == [256, 256]
    assert cfg.is_discontinuous == base_cfg.is_discontinuous
    assert cfg.iters == base_cfg.iters
    assert cfg.batch_size == base_cfg.batch_size


def test_build_tracker_config_uses_even_batch_for_odd_frame_counts(tmp_path):
    from faceforge.stageP3M._data_bridge import build_tracker_config

    cfg = build_tracker_config(
        video_name='actor',
        preprocessed_dir=str(tmp_path / 'preprocessed'),
        output_dir=str(tmp_path / 'output'),
        n_frames=7,
        render_size=256,
        code_base='submodules/pixel3dmm',
    )

    assert cfg.batch_size == 6
    assert cfg.is_discontinuous is False


def test_pipeline_preserves_native_tracking_outputs(tmp_path, monkeypatch):
    from faceforge.stageP3M.config import P3MConfig
    from faceforge.stageP3M.pipeline import P3MPipeline
    from faceforge.stageP3M.visualization import P3MVisualizer
    import faceforge.stage2._pixel3dmm_paths as pixel3dmm_paths
    import faceforge.stageP3M.pipeline as pipeline_module

    monkeypatch.setattr(pixel3dmm_paths, 'configure_pixel3dmm_paths', lambda cfg: None)

    env_paths = ModuleType('pixel3dmm.env_paths')
    env_paths.PREPROCESSED_DATA = 'unset_preprocessed'
    env_paths.TRACKING_OUTPUT = 'unset_tracking'

    tracker_module = ModuleType('pixel3dmm.tracking.tracker')
    tracker_module.COMPILE = True

    def _project_points_screen_space(*args, **kwargs):
        return None

    tracker_module.project_points_screen_space = _project_points_screen_space

    class FakeTracker:
        def __init__(self, config, device='cpu'):
            self.config = config
            self.device = device
            self.output_folder = str(Path(config.output_folder) / 'fake_actor')

        def run(self):
            base = Path(self.output_folder)
            for subdir, filename in [
                ('joint_initialization', '00000.png'),
                ('checkpoint', '00000.frame'),
                ('mesh', 'canonical.ply'),
            ]:
                target = base / subdir
                target.mkdir(parents=True, exist_ok=True)
                (target / filename).write_text('artifact', encoding='utf-8')

    tracker_module.Tracker = FakeTracker

    pixel3dmm_pkg = ModuleType('pixel3dmm')
    tracking_pkg = ModuleType('pixel3dmm.tracking')
    pixel3dmm_pkg.env_paths = env_paths
    pixel3dmm_pkg.tracking = tracking_pkg
    tracking_pkg.tracker = tracker_module

    monkeypatch.setitem(__import__('sys').modules, 'pixel3dmm', pixel3dmm_pkg)
    monkeypatch.setitem(__import__('sys').modules, 'pixel3dmm.env_paths', env_paths)
    monkeypatch.setitem(__import__('sys').modules, 'pixel3dmm.tracking', tracking_pkg)
    monkeypatch.setitem(__import__('sys').modules, 'pixel3dmm.tracking.tracker', tracker_module)

    monkeypatch.setattr(pipeline_module, 'write_preprocessed', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        'build_tracker_config',
        lambda **kwargs: SimpleNamespace(output_folder=kwargs['output_dir']),
    )

    sentinel = object()
    monkeypatch.setattr(pipeline_module, 'read_tracker_results', lambda *args, **kwargs: sentinel)

    visualizer = P3MVisualizer(str(tmp_path / 'output'), 'subject')
    pipeline = P3MPipeline(P3MConfig(device='cpu'), visualizer=visualizer)
    pipeline._pixel3dmm = object()

    result = pipeline.run([object()])

    tracking_dir = visualizer.base_dir / 'tracking' / 'fake_actor'
    assert result is sentinel
    assert (tracking_dir / 'joint_initialization' / '00000.png').is_file()
    assert (tracking_dir / 'checkpoint' / '00000.frame').is_file()
    assert (tracking_dir / 'mesh' / 'canonical.ply').is_file()

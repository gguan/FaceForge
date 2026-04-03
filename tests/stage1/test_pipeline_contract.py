from unittest.mock import MagicMock

import numpy as np
import torch


def test_run_single_preserves_parsing_map_and_pipnet_landmarks(monkeypatch):
    from faceforge.stage1.config import Stage1Config
    from faceforge.stage1.data_types import DetectionResult
    from faceforge.stage1.pipeline import Stage1Pipeline
    from faceforge.stage1.segmentation import FaceParser
    import faceforge.stage1.pipeline as pipeline_module

    image_rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    aligned = np.full((8, 8, 3), 200, dtype=np.uint8)
    parsing = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 1, 2, 3],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17, 18, 1],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ], dtype=np.int64)
    face_mask = FaceParser.extract_face_mask(parsing).astype(np.float32)
    lmks_68 = np.stack([np.linspace(1, 6, 68), np.linspace(2, 7, 68)], axis=1).astype(np.float32)
    lmks_dense = np.stack([np.linspace(0, 7, 478), np.linspace(1, 8, 478)], axis=1).astype(np.float32)
    lmks_eyes = np.stack([np.linspace(1, 5, 10), np.linspace(2, 6, 10)], axis=1).astype(np.float32)
    lmks_98 = np.stack([np.linspace(0.5, 7.5, 98), np.linspace(1.5, 8.5, 98)], axis=1).astype(np.float32)

    detection = DetectionResult(
        lmks_dense=lmks_dense,
        lmks_68=lmks_68,
        lmks_eyes=lmks_eyes,
        blend_scores=np.zeros(52, dtype=np.float32),
        retinaface_kps=np.zeros((5, 2), dtype=np.float32),
    )

    mica_result = {
        'shape_code': torch.ones(1, 300),
        'arcface_feat': torch.ones(1, 512),
        'arcface_img': np.zeros((112, 112, 3), dtype=np.uint8),
        'vertices': torch.zeros(1, 5023, 3),
    }
    deca_result = {
        'exp': torch.zeros(1, 50),
        'pose': torch.zeros(1, 6),
        'tex': torch.zeros(1, 50),
        'light': torch.zeros(1, 9, 3),
        'deca_crop': np.zeros((224, 224, 3), dtype=np.uint8),
        'cam': torch.zeros(1, 3),
        'crop_tform': np.eye(3, dtype=np.float32),
    }
    flame_params = {
        'shape': torch.ones(1, 300),
        'exp': torch.zeros(1, 100),
        'head_pose': torch.zeros(1, 3),
        'jaw_pose': torch.zeros(1, 3),
        'tex': torch.zeros(1, 50),
        'light': torch.zeros(1, 9, 3),
    }

    pipeline = Stage1Pipeline.__new__(Stage1Pipeline)
    pipeline.config = Stage1Config(save_debug=False, align_output_size=8, render_size=8)
    pipeline.mp_detector = object()
    pipeline.retina_detector = object()
    pipeline.face_parser = MagicMock()
    pipeline.face_parser.parse.return_value = parsing
    pipeline.pipnet = MagicMock()
    pipeline.pipnet.predict.return_value = lmks_98
    pipeline.mica = MagicMock()
    pipeline.mica.run.return_value = mica_result
    pipeline.deca = MagicMock()
    pipeline.deca.run.return_value = deca_result
    pipeline._get_flame_faces = MagicMock(return_value=None)
    pipeline._get_posed_vertices = MagicMock(return_value=None)
    pipeline._get_flame_landmarks_3d = MagicMock(return_value=None)

    monkeypatch.setattr(pipeline_module, 'detect_all', lambda *_args, **_kwargs: detection)
    monkeypatch.setattr(
        pipeline_module,
        'image_align',
        lambda *_args, **_kwargs: (aligned, np.eye(3, dtype=np.float64)),
    )
    monkeypatch.setattr(pipeline_module, 'merge_params', lambda *_args, **_kwargs: flame_params)

    output, summary = pipeline.run_single(image_rgb, subject_name='pytest')

    assert summary is None
    assert torch.equal(output.parsing_map, torch.from_numpy(parsing).unsqueeze(0))
    assert torch.equal(output.lmks_98, torch.from_numpy(lmks_98).unsqueeze(0))
    assert set(output.face_mask.unique().tolist()) <= {0.0, 1.0}


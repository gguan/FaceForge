from types import SimpleNamespace

import torch


def _make_stage1_output():
    from faceforge.stage1.data_types import Stage1Output

    parsing_map = torch.tensor([
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    ], dtype=torch.long)
    face_mask = torch.tensor([
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ], dtype=torch.float32)
    return Stage1Output(
        shape=torch.zeros(1, 300),
        expression=torch.zeros(1, 100),
        head_pose=torch.zeros(1, 3),
        jaw_pose=torch.zeros(1, 3),
        texture=torch.zeros(1, 50),
        lighting=torch.zeros(1, 9, 3),
        arcface_feat=torch.zeros(1, 512),
        aligned_image=torch.zeros(1, 3, 4, 4),
        face_mask=face_mask,
        parsing_map=parsing_map,
        lmks_68=torch.zeros(1, 68, 2),
        lmks_98=torch.zeros(1, 98, 2),
        lmks_dense=torch.zeros(1, 478, 2),
        lmks_eyes=torch.zeros(1, 10, 2),
        focal_length=torch.ones(1, 1),
        principal_point=torch.zeros(1, 2),
    )


def test_preprocess_prefers_parsing_map_for_pixel3dmm_masks():
    from faceforge.stage2.pipeline import Stage2Pipeline

    pipeline = Stage2Pipeline.__new__(Stage2Pipeline)
    pipeline.device = torch.device('cpu')
    pipeline._pixel3dmm = SimpleNamespace(
        predict=lambda aligned_image, face_segmentation: (
            torch.zeros(1, 2, 4, 4),
            torch.zeros(1, 3, 4, 4),
        )
    )

    preprocessed = pipeline._preprocess(_make_stage1_output())

    assert torch.equal(
        preprocessed.face_segmentation.cpu(),
        torch.tensor(
            [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        preprocessed.face_mask.cpu(),
        torch.tensor(
            [[[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
    )


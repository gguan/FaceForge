"""
Integration tests for Stage 1 — require real model weights and a test image.

Setup:
    export FACEFORGE_TEST_IMAGE=/path/to/face.jpg
    export FACEFORGE_DEVICE=cpu          # or cuda:0
    export FACEFORGE_DATA_DIR=/path/to/data  # defaults to <project_root>/data

Run all integration tests:
    pytest tests/stage1/test_integration.py -v

Run only fast integration tests (skip full pipeline):
    pytest tests/stage1/test_integration.py -v -m "integration and not slow"
"""

import os

import cv2
import numpy as np
import pytest
import torch


pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetection:

    def test_detection_returns_result(self, detection_result):
        assert detection_result is not None

    def test_dense_landmarks_shape(self, detection_result):
        assert detection_result.lmks_dense.shape == (478, 2), \
            'MediaPipe dense landmarks must be [478, 2]'

    def test_68_landmarks_shape(self, detection_result):
        assert detection_result.lmks_68.shape == (68, 2), \
            'Converted dlib landmarks must be [68, 2]'

    def test_eye_landmarks_shape(self, detection_result):
        assert detection_result.lmks_eyes.shape == (10, 2), \
            'Eye landmarks must be [10, 2]'

    def test_blendshapes_shape(self, detection_result):
        assert detection_result.blend_scores.shape == (52,), \
            'Blendshapes must be [52]'

    def test_retinaface_kps_shape(self, detection_result):
        assert detection_result.retinaface_kps is not None, \
            'RetinaFace must return 5-point kps'
        assert detection_result.retinaface_kps.shape == (5, 2), \
            'RetinaFace kps must be [5, 2]'

    def test_landmarks_within_image_bounds(self, test_image_rgb, detection_result):
        h, w = test_image_rgb.shape[:2]
        lmks = detection_result.lmks_68
        assert lmks[:, 0].min() >= 0 and lmks[:, 0].max() <= w
        assert lmks[:, 1].min() >= 0 and lmks[:, 1].max() <= h

    def test_blendshapes_sum_roughly_one(self, detection_result):
        """Blendshape scores are probabilities; their sum should be reasonable."""
        total = detection_result.blend_scores.sum()
        assert 0.0 <= total <= 52.0, 'blendshape scores out of expected range'


# ─────────────────────────────────────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignment:

    def test_output_shape(self, aligned_image):
        assert aligned_image.shape == (512, 512, 3)

    def test_output_dtype(self, aligned_image):
        assert aligned_image.dtype == np.uint8

    def test_output_value_range(self, aligned_image):
        assert aligned_image.min() >= 0 and aligned_image.max() <= 255

    def test_face_roughly_centred(self, aligned_image):
        """Mean pixel value in the centre quadrant should be brighter than the corners
        for a typical face image (simple sanity check, not a hard guarantee)."""
        h, w = aligned_image.shape[:2]
        centre = aligned_image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        assert centre.mean() > 10, 'centre of aligned image appears empty'


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentation:

    @pytest.fixture(scope='class')
    def face_parser(self, stage1_config, data_dir):
        weights = os.path.join(data_dir, 'pretrained', '79999_iter.pth')
        if not os.path.exists(weights):
            pytest.skip(f'Missing BiSeNet weights: {weights}')
        from faceforge.stage1.segmentation import FaceParser
        return FaceParser(weights, stage1_config.device)

    def test_parsing_shape(self, face_parser, aligned_image):
        parsing = face_parser.parse(aligned_image)
        assert parsing.shape == (512, 512)

    def test_parsing_classes_in_range(self, face_parser, aligned_image):
        parsing = face_parser.parse(aligned_image)
        assert parsing.min() >= 0 and parsing.max() <= 18, \
            'parsing classes must be in [0, 18]'

    def test_face_present_in_mask(self, face_parser, aligned_image):
        from faceforge.stage1.segmentation import FaceParser
        parsing = face_parser.parse(aligned_image)
        mask = FaceParser.extract_face_mask(parsing)
        face_pixels = mask.sum()
        total_pixels = mask.size
        ratio = face_pixels / total_pixels
        assert ratio > 0.05, f'face mask covers only {ratio:.1%} of image — parsing may have failed'
        assert ratio < 0.95, f'face mask covers {ratio:.1%} — mask seems too large'


# ─────────────────────────────────────────────────────────────────────────────
# MICA inference
# ─────────────────────────────────────────────────────────────────────────────

class TestMICAInference:

    @pytest.fixture(scope='class')
    def mica(self, stage1_config):
        if not os.path.exists(stage1_config.mica_model_path):
            pytest.skip(f'Missing MICA weights: {stage1_config.mica_model_path}')
        from faceforge.stage1.mica_inference import MICAInference
        return MICAInference(stage1_config)

    def test_shape_code_shape(self, mica, test_image_rgb, detection_result):
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert result['shape_code'].shape == (1, 300)

    def test_vertices_shape(self, mica, test_image_rgb, detection_result):
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert result['vertices'].shape == (1, 5023, 3)

    def test_arcface_feat_shape(self, mica, test_image_rgb, detection_result):
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert result['arcface_feat'].shape == (1, 512)

    def test_arcface_img_shape(self, mica, test_image_rgb, detection_result):
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert result['arcface_img'].shape == (112, 112, 3)

    def test_shape_code_finite(self, mica, test_image_rgb, detection_result):
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert torch.isfinite(result['shape_code']).all(), 'shape_code contains NaN/Inf'

    def test_shape_code_reasonable_range(self, mica, test_image_rgb, detection_result):
        """FLAME shape PCA codes are typically small — outside ±10 is unusual."""
        image_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
        result = mica.run(image_bgr, detection_result.retinaface_kps)
        assert result['shape_code'].abs().max() < 10.0, 'shape code has unusually large values'


# ─────────────────────────────────────────────────────────────────────────────
# DECA inference
# ─────────────────────────────────────────────────────────────────────────────

class TestDECAInference:

    @pytest.fixture(scope='class')
    def deca(self, stage1_config, data_dir):
        deca_ckpt = os.path.join(data_dir, 'pretrained', 'deca_model.tar')
        if not os.path.exists(deca_ckpt):
            pytest.skip(f'Missing DECA checkpoint: {deca_ckpt}')
        from faceforge.stage1.deca_inference import DECAInference
        return DECAInference(stage1_config)

    def test_expression_shape(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        assert result['exp'].shape == (1, 50)

    def test_pose_shape(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        assert result['pose'].shape == (1, 6)

    def test_texture_shape(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        assert result['tex'].shape == (1, 50)

    def test_lighting_shape(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        light = result['light']
        # DECA may return [1, 27] or [1, 9, 3]
        assert light.numel() == 27, f'lighting should have 27 values, got {light.numel()}'

    def test_deca_crop_shape(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        assert result['deca_crop'].shape == (224, 224, 3)

    def test_outputs_finite(self, deca, test_image_rgb):
        result = deca.run(test_image_rgb)
        for key in ('exp', 'pose', 'tex', 'light'):
            assert torch.isfinite(result[key]).all(), f'{key} contains NaN/Inf'


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (slow)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestFullPipeline:

    @pytest.fixture(scope='class')
    def pipeline(self, stage1_config):
        from faceforge.stage1 import Stage1Pipeline
        return Stage1Pipeline(stage1_config)

    @pytest.fixture(scope='class')
    def output(self, pipeline, test_image_rgb):
        return pipeline.run_single(test_image_rgb, subject_name='pytest')

    # ── tensor shapes ────────────────────────────────────────────────────────

    def test_shape_tensor(self, output):
        assert output.shape.shape == (1, 300)

    def test_expression_tensor(self, output):
        assert output.expression.shape == (1, 100)

    def test_head_pose_tensor(self, output):
        assert output.head_pose.shape == (1, 3)

    def test_jaw_pose_tensor(self, output):
        assert output.jaw_pose.shape == (1, 3)

    def test_texture_tensor(self, output):
        assert output.texture.shape == (1, 50)

    def test_lighting_tensor(self, output):
        assert output.lighting.shape == (1, 9, 3)

    def test_arcface_feat_tensor(self, output):
        assert output.arcface_feat.shape == (1, 512)

    def test_aligned_image_tensor(self, output):
        assert output.aligned_image.shape == (1, 3, 512, 512)

    def test_face_mask_tensor(self, output):
        assert output.face_mask.shape[1:] == torch.Size([512, 512])

    def test_lmks_68_tensor(self, output):
        assert output.lmks_68.shape == (1, 68, 2)

    def test_lmks_dense_tensor(self, output):
        assert output.lmks_dense.shape == (1, 478, 2)

    def test_lmks_eyes_tensor(self, output):
        assert output.lmks_eyes.shape == (1, 10, 2)

    def test_focal_length_tensor(self, output):
        assert output.focal_length.shape == (1, 1)

    def test_principal_point_tensor(self, output):
        assert output.principal_point.shape == (1, 2)

    # ── value sanity ─────────────────────────────────────────────────────────

    def test_aligned_image_range(self, output):
        assert output.aligned_image.min() >= 0.0
        assert output.aligned_image.max() <= 1.0

    def test_face_mask_binary(self, output):
        vals = output.face_mask.unique()
        for v in vals:
            assert v.item() in (0.0, 1.0), f'face_mask contains non-binary value {v}'

    def test_expression_padding_zeros(self, output):
        assert torch.all(output.expression[:, 50:] == 0), \
            'last 50 dims of expression should be zeros'

    def test_all_tensors_finite(self, output):
        fields = ('shape', 'expression', 'head_pose', 'jaw_pose',
                  'texture', 'lighting', 'arcface_feat',
                  'aligned_image', 'focal_length', 'principal_point')
        for field in fields:
            t = getattr(output, field)
            assert torch.isfinite(t).all(), f'output.{field} contains NaN/Inf'


@pytest.mark.slow
class TestMultiImagePipeline:

    @pytest.fixture(scope='class')
    def pipeline(self, stage1_config):
        from faceforge.stage1 import Stage1Pipeline
        return Stage1Pipeline(stage1_config)

    def test_multi_same_as_single_for_one_image(self, pipeline, test_image_rgb):
        single = pipeline.run_single(test_image_rgb, subject_name='single')
        multi = pipeline.run_multi([test_image_rgb], subject_name='multi1')
        assert torch.allclose(single.shape, multi.shape), \
            'run_multi with 1 image should equal run_single'

    def test_multi_two_images_shape_aggregated(self, pipeline, test_image_rgb):
        result = pipeline.run_multi([test_image_rgb, test_image_rgb], subject_name='multi2')
        assert result.shape.shape == (1, 300), 'aggregated shape must be [1, 300]'

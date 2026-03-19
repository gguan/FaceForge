"""
Unit tests for Stage 1 — no model weights or GPU required.

Run with:
    pytest tests/stage1/test_unit.py -v
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# merge_params
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeParams:

    def _make_inputs(self):
        mica = {'shape_code': torch.randn(1, 300)}
        deca = {
            'exp':   torch.randn(1, 50),
            'pose':  torch.randn(1, 6),
            'tex':   torch.randn(1, 50),
            'light': torch.randn(1, 9, 3),
        }
        return mica, deca

    def test_output_shapes(self):
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        out = merge_params(mica, deca)

        assert out['shape'].shape    == (1, 300), 'shape should be [1, 300]'
        assert out['exp'].shape      == (1, 100), 'expression should be padded to [1, 100]'
        assert out['head_pose'].shape == (1, 3),  'head_pose should be [1, 3]'
        assert out['jaw_pose'].shape  == (1, 3),  'jaw_pose should be [1, 3]'
        assert out['tex'].shape      == (1, 50),  'texture should be [1, 50]'
        assert out['light'].shape    == (1, 9, 3), 'lighting should be [1, 9, 3]'

    def test_shape_comes_from_mica(self):
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        out = merge_params(mica, deca)
        assert torch.allclose(out['shape'], mica['shape_code']), \
            'shape should equal mica shape_code unchanged'

    def test_expression_first_50_from_deca(self):
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        out = merge_params(mica, deca)
        assert torch.allclose(out['exp'][:, :50], deca['exp']), \
            'first 50 dims of expression should come from DECA'

    def test_expression_last_50_zeros(self):
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        out = merge_params(mica, deca)
        assert torch.all(out['exp'][:, 50:] == 0), \
            'last 50 dims of expression should be zeros'

    def test_pose_split(self):
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        deca['pose'] = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        out = merge_params(mica, deca)
        assert torch.allclose(out['head_pose'], torch.tensor([[1., 2., 3.]])), \
            'head_pose = pose[:3]'
        assert torch.allclose(out['jaw_pose'],  torch.tensor([[4., 5., 6.]])), \
            'jaw_pose = pose[3:]'

    def test_light_already_3d(self):
        """[1, 9, 3] lighting passes through unchanged."""
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        out = merge_params(mica, deca)
        assert out['light'].shape == (1, 9, 3)

    def test_light_reshape_from_flat(self):
        """[1, 27] flat lighting is reshaped to [1, 9, 3]."""
        from faceforge.stage1.merge import merge_params
        mica, deca = self._make_inputs()
        deca['light'] = torch.arange(27, dtype=torch.float32).unsqueeze(0)  # [1, 27]
        out = merge_params(mica, deca)
        assert out['light'].shape == (1, 9, 3)
        # Verify values preserved after reshape
        assert out['light'][0, 8, 2] == 26.0


# ─────────────────────────────────────────────────────────────────────────────
# FaceParser.extract_face_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFaceMask:

    def _make_parsing(self, fill: int, shape=(512, 512)) -> np.ndarray:
        return np.full(shape, fill, dtype=np.int32)

    def test_background_excluded(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(0)  # 0 = background
        mask = FaceParser.extract_face_mask(parsing)
        assert not mask.any(), 'background should be fully masked out'

    def test_skin_included(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(1)  # 1 = skin
        mask = FaceParser.extract_face_mask(parsing)
        assert mask.all(), 'skin should be fully included'

    def test_hair_excluded(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(17)  # 17 = hair
        mask = FaceParser.extract_face_mask(parsing)
        assert not mask.any(), 'hair should be excluded'

    def test_cloth_excluded(self):
        from faceforge.stage1.segmentation import FaceParser
        for label in (16, 17, 18):  # cloth, hair, hat
            parsing = self._make_parsing(label)
            mask = FaceParser.extract_face_mask(parsing)
            assert not mask.any(), f'label {label} should be excluded'

    def test_glasses_excluded_by_default(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(6)  # 6 = eye_g (glasses)
        mask = FaceParser.extract_face_mask(parsing)
        assert not mask.any(), 'glasses should be excluded by default'

    def test_ears_excluded_by_default(self):
        from faceforge.stage1.segmentation import FaceParser
        for label in (7, 8, 9):  # l_ear, r_ear, ear_r
            parsing = self._make_parsing(label)
            mask = FaceParser.extract_face_mask(parsing)
            assert not mask.any(), f'ear label {label} excluded by default'

    def test_ears_included_when_flag_set(self):
        from faceforge.stage1.segmentation import FaceParser
        for label in (7, 8, 9):
            parsing = self._make_parsing(label)
            mask = FaceParser.extract_face_mask(parsing, keep_ears=True)
            assert mask.all(), f'ear label {label} should be included when keep_ears=True'

    def test_neck_excluded_by_default(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(14)  # 14 = neck
        mask = FaceParser.extract_face_mask(parsing)
        assert not mask.any(), 'neck excluded by default'

    def test_neck_included_when_flag_set(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(14)
        mask = FaceParser.extract_face_mask(parsing, keep_neck=True)
        assert mask.all(), 'neck included when keep_neck=True'

    def test_mouth_included_by_default(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(11)  # 11 = mouth
        mask = FaceParser.extract_face_mask(parsing)
        assert mask.all(), 'mouth included by default'

    def test_mouth_excluded_when_flag_unset(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = self._make_parsing(11)
        mask = FaceParser.extract_face_mask(parsing, keep_mouth=False)
        assert not mask.any(), 'mouth excluded when keep_mouth=False'

    def test_output_dtype_bool(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = np.random.randint(0, 19, size=(512, 512), dtype=np.int32)
        mask = FaceParser.extract_face_mask(parsing)
        assert mask.dtype == bool, 'mask dtype should be bool'

    def test_output_shape_preserved(self):
        from faceforge.stage1.segmentation import FaceParser
        parsing = np.zeros((256, 256), dtype=np.int32)
        mask = FaceParser.extract_face_mask(parsing)
        assert mask.shape == (256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# mp2dlib landmark conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestMp2Dlib:

    def test_output_shape(self):
        from faceforge.stage1.mp2dlib import convert_landmarks_mediapipe_to_dlib
        lmks_478 = np.random.rand(478, 2).astype(np.float32) * 512
        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_478)
        assert lmks_68.shape == (68, 2), 'output must be [68, 2]'

    def test_output_dtype(self):
        from faceforge.stage1.mp2dlib import convert_landmarks_mediapipe_to_dlib
        lmks_478 = np.random.rand(478, 2).astype(np.float32)
        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_478)
        assert lmks_68.dtype == np.float32

    def test_single_index_equals_input(self):
        """A dlib landmark mapped from a single MP index must equal that MP point."""
        from faceforge.stage1.mp2dlib import convert_landmarks_mediapipe_to_dlib, mp2dlib_correspondence

        lmks_478 = np.zeros((478, 2), dtype=np.float32)
        # Plant unique values at each MP index
        for i in range(478):
            lmks_478[i] = [float(i), float(i) * 2]

        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_478)

        for dlib_idx, mp_indices in enumerate(mp2dlib_correspondence):
            if len(mp_indices) == 1:
                mp_idx = mp_indices[0]
                assert np.allclose(lmks_68[dlib_idx], lmks_478[mp_idx]), \
                    f'dlib {dlib_idx} (MP {mp_idx}) mismatch'

    def test_dual_index_is_average(self):
        """A dlib landmark mapped from two MP indices must be their average."""
        from faceforge.stage1.mp2dlib import convert_landmarks_mediapipe_to_dlib, mp2dlib_correspondence

        lmks_478 = np.zeros((478, 2), dtype=np.float32)
        for i in range(478):
            lmks_478[i] = [float(i) * 3, float(i) * 7]

        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_478)

        for dlib_idx, mp_indices in enumerate(mp2dlib_correspondence):
            if len(mp_indices) == 2:
                expected = (lmks_478[mp_indices[0]] + lmks_478[mp_indices[1]]) / 2
                assert np.allclose(lmks_68[dlib_idx], expected), \
                    f'dlib {dlib_idx} (MP {mp_indices}) not averaged correctly'


# ─────────────────────────────────────────────────────────────────────────────
# _norm_crop (ArcFace alignment)
# ─────────────────────────────────────────────────────────────────────────────

class TestNormCrop:

    def _make_frontal_kps(self) -> np.ndarray:
        """Approximate 5-point frontal-face landmarks in a 256×256 image."""
        return np.array([
            [85.0,  100.0],   # right eye
            [170.0, 100.0],   # left eye
            [128.0, 145.0],   # nose tip
            [95.0,  185.0],   # right mouth corner
            [160.0, 185.0],   # left mouth corner
        ], dtype=np.float32)

    def test_output_shape_default(self):
        from faceforge.stage1.mica_inference import _norm_crop
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = _norm_crop(img, self._make_frontal_kps())
        assert out.shape == (112, 112, 3), 'default output size should be 112×112×3'

    def test_output_shape_custom_size(self):
        from faceforge.stage1.mica_inference import _norm_crop
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = _norm_crop(img, self._make_frontal_kps(), image_size=224)
        assert out.shape == (224, 224, 3)

    def test_output_dtype(self):
        from faceforge.stage1.mica_inference import _norm_crop
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = _norm_crop(img, self._make_frontal_kps())
        assert out.dtype == np.uint8

    def test_wrong_landmark_shape_raises(self):
        from faceforge.stage1.mica_inference import _norm_crop
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bad_kps = np.zeros((4, 2), dtype=np.float32)  # should be (5, 2)
        with pytest.raises(AssertionError):
            _norm_crop(img, bad_kps)


# ─────────────────────────────────────────────────────────────────────────────
# image_align (FFHQ alignment)
# ─────────────────────────────────────────────────────────────────────────────

class TestImageAlign:

    def _make_frontal_68(self, img_w=512, img_h=512) -> np.ndarray:
        """Synthesize plausible 68-point frontal-face landmarks."""
        lmks = np.zeros((68, 2), dtype=np.float32)
        cx, cy = img_w / 2, img_h / 2

        # Jaw (0-16): rough ellipse along the bottom half
        for i in range(17):
            angle = np.pi - i * np.pi / 16
            lmks[i] = [cx + 0.35 * img_w * np.cos(angle),
                       cy + 0.35 * img_h * np.sin(angle) * 0.6 + 0.05 * img_h]

        # Right brow (17-21)
        for i, x in enumerate(np.linspace(cx - 0.25 * img_w, cx - 0.05 * img_w, 5)):
            lmks[17 + i] = [x, cy - 0.22 * img_h]

        # Left brow (22-26)
        for i, x in enumerate(np.linspace(cx + 0.05 * img_w, cx + 0.25 * img_w, 5)):
            lmks[22 + i] = [x, cy - 0.22 * img_h]

        # Nose (27-35)
        for i in range(9):
            lmks[27 + i] = [cx + (i - 4) * 0.02 * img_w,
                             cy - 0.15 * img_h + i * 0.04 * img_h]

        # Right eye (36-41)
        for i, x in enumerate(np.linspace(cx - 0.22 * img_w, cx - 0.07 * img_w, 6)):
            lmks[36 + i] = [x, cy - 0.10 * img_h]

        # Left eye (42-47)
        for i, x in enumerate(np.linspace(cx + 0.07 * img_w, cx + 0.22 * img_w, 6)):
            lmks[42 + i] = [x, cy - 0.10 * img_h]

        # Mouth (48-67)
        for i in range(20):
            angle = -np.pi + i * 2 * np.pi / 20
            lmks[48 + i] = [cx + 0.13 * img_w * np.cos(angle),
                             cy + 0.18 * img_h + 0.05 * img_h * np.sin(angle)]

        return lmks

    def test_output_shape_default(self):
        from faceforge.stage1.alignment import image_align
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = image_align(img, self._make_frontal_68())
        assert out.shape == (512, 512, 3)

    def test_output_shape_custom_size(self):
        from faceforge.stage1.alignment import image_align
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = image_align(img, self._make_frontal_68(), output_size=256, transform_size=512)
        assert out.shape == (256, 256, 3)

    def test_output_dtype(self):
        from faceforge.stage1.alignment import image_align
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = image_align(img, self._make_frontal_68())
        assert out.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_shapes (mocked MICA / RetinaFace)
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateShapes:

    def _make_mock_mica(self, shape_codes: list[torch.Tensor]):
        """Return a mock MICAInference that yields successive shape codes."""
        calls = iter(shape_codes)

        def mock_run(img_bgr, kps):
            code = next(calls)
            return {
                'shape_code': code.unsqueeze(0),   # [1, 300]
                'vertices':   torch.zeros(1, 5023, 3),
                'arcface_feat': torch.zeros(1, 512),
                'arcface_img':  np.zeros((112, 112, 3), dtype=np.uint8),
            }

        mica = MagicMock()
        mica.run.side_effect = mock_run
        return mica

    def _make_mock_retina(self, n_images: int):
        retina = MagicMock()
        kps = np.random.rand(5, 2).astype(np.float32)
        retina.detect_5pt.return_value = kps
        return retina

    def _make_images(self, n: int):
        return [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n)]

    def test_median_aggregation(self):
        from faceforge.stage1.aggregation import aggregate_shapes

        codes = [torch.tensor([1.0, 2.0, 3.0] * 100),   # 300 dims
                 torch.tensor([3.0, 4.0, 5.0] * 100),
                 torch.tensor([2.0, 3.0, 4.0] * 100)]   # median

        mica = self._make_mock_mica(codes)
        retina = self._make_mock_retina(3)
        images = self._make_images(3)

        agg, per_img = aggregate_shapes(images, mica, retina, method='median')

        assert agg.shape == (300,), 'aggregated shape must be [300]'
        assert len(per_img) == 3
        # median of [1,3,2] = 2, median of [2,4,3] = 3, median of [3,5,4] = 4
        expected = torch.tensor([2.0, 3.0, 4.0] * 100)
        assert torch.allclose(agg, expected), 'median not computed correctly'

    def test_mean_aggregation(self):
        from faceforge.stage1.aggregation import aggregate_shapes

        codes = [torch.ones(300) * 1.0,
                 torch.ones(300) * 3.0]

        mica = self._make_mock_mica(codes)
        retina = self._make_mock_retina(2)
        images = self._make_images(2)

        agg, _ = aggregate_shapes(images, mica, retina, method='mean')

        assert agg.shape == (300,)
        assert torch.allclose(agg, torch.ones(300) * 2.0), 'mean not computed correctly'

    def test_unknown_method_raises(self):
        from faceforge.stage1.aggregation import aggregate_shapes

        mica = self._make_mock_mica([torch.zeros(300)])
        retina = self._make_mock_retina(1)

        with pytest.raises(ValueError, match='Unknown aggregation method'):
            aggregate_shapes(self._make_images(1), mica, retina, method='bogus')

    def test_no_faces_raises(self):
        from faceforge.stage1.aggregation import aggregate_shapes

        mica = MagicMock()
        retina = MagicMock()
        retina.detect_5pt.return_value = None  # no face in any image

        with pytest.raises(ValueError, match='No faces detected'):
            aggregate_shapes(self._make_images(3), mica, retina)

    def test_output_count_matches_detected(self):
        """Only images where a face is found should appear in per_image_outputs."""
        from faceforge.stage1.aggregation import aggregate_shapes

        codes = [torch.zeros(300), torch.ones(300)]

        mica = self._make_mock_mica(codes)
        retina = MagicMock()
        # Face found in first and third images; none in second
        retina.detect_5pt.side_effect = [
            np.random.rand(5, 2).astype(np.float32),
            None,
            np.random.rand(5, 2).astype(np.float32),
        ]

        _, per_img = aggregate_shapes(self._make_images(3), mica, retina)
        assert len(per_img) == 2, 'skipped image should not appear in outputs'

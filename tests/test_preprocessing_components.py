"""Smoke tests for the preprocessing components on a real cortis frame."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from faceforge._paths import PROJECT_ROOT
from faceforge.preprocessing.landmark import (
    InsightFace106Detector,
    InsightFace106Config,
    LandmarkResult,
    make_landmark_detector,
)
from faceforge.preprocessing.segmentation import (
    BiSeNetSegmenter,
    BiSeNetConfig,
    SegmentationResult,
    make_segmenter,
)
from faceforge.preprocessing.matting import (
    MatteResult,
    MODNetConfig,
    MODNetMatter,
    make_matter,
)
from faceforge.preprocessing.cropping import (
    CropResult,
    FFHQCropConfig,
    FFHQCropper,
    make_cropper,
    project_points,
    standardize_crop,
)


SAMPLE_IMAGE = (
    PROJECT_ROOT / 'data' / 'mononphm' / 'tracking_input'
    / 'cortis' / 'source' / '00000.png'
)


@pytest.fixture(scope='module')
def image_rgb() -> np.ndarray:
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"sample image not present: {SAMPLE_IMAGE}")
    bgr = cv2.imread(str(SAMPLE_IMAGE))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- landmark


def test_insightface_106_run_and_visualize(image_rgb):
    det = InsightFace106Detector(InsightFace106Config(device='cpu'))
    result = det.run(image_rgb)

    assert isinstance(result, LandmarkResult)
    assert result.scheme == 'insightface_106'
    assert result.n_points == 106
    assert result.landmarks.shape == (106, 2)
    assert result.bbox.shape == (4,)
    assert result.kps_5pt is not None and result.kps_5pt.shape == (5, 2)
    assert 0.0 < result.confidence <= 1.0

    vis = det.visualize(image_rgb, result)
    assert vis.shape == image_rgb.shape
    assert vis.dtype == np.uint8
    # Visualization must differ from input (we drew on it).
    assert not np.array_equal(vis, image_rgb)


def test_landmark_factory_dispatches_insightface():
    det = make_landmark_detector('insightface_106', device='cpu')
    assert det.name == 'insightface_106'


def test_landmark_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_landmark_detector('nonexistent_backend')


# ------------------------------------------------------------- segmentation


def test_bisenet_run_and_visualize(image_rgb):
    seg = BiSeNetSegmenter(BiSeNetConfig(device='cpu'))
    result = seg.run(image_rgb)

    h, w = image_rgb.shape[:2]
    assert isinstance(result, SegmentationResult)
    assert result.scheme == 'bisenet_19'
    assert result.seg_map.shape == (h, w)
    assert result.face_mask.shape == (h, w)
    assert result.face_mask.dtype == np.bool_
    assert result.seg_map.max() <= 18
    # Sanity: a face should occupy a reasonable fraction of the image.
    assert result.face_mask.mean() > 0.02

    vis = seg.visualize(image_rgb, result)
    assert vis.dtype == np.uint8
    # Side-by-side: width is 2× source.
    assert vis.shape[0] == h
    assert vis.shape[1] == 2 * w


def test_segmenter_factory_dispatches_bisenet():
    seg = make_segmenter('bisenet', device='cpu')
    assert seg.name == 'bisenet'


# ------------------------------------------------------------------ matting


def test_modnet_run_and_visualize(image_rgb):
    try:
        matter = MODNetMatter(MODNetConfig(device='cpu'))
    except FileNotFoundError as exc:
        pytest.skip(f"MODNet not available: {exc}")

    result = matter.run(image_rgb)

    h, w = image_rgb.shape[:2]
    assert isinstance(result, MatteResult)
    assert result.scheme == 'modnet'
    assert result.alpha.shape == (h, w)
    assert result.alpha.dtype == np.float32
    assert 0.0 <= result.alpha.min() and result.alpha.max() <= 1.0
    # The portrait should yield substantial foreground.
    assert result.alpha.mean() > 0.05

    vis = matter.visualize(image_rgb, result)
    # Triptych: width is 3× source.
    assert vis.shape[0] == h
    assert vis.shape[1] == 3 * w


# ----------------------------------------------------------------- cropping


def test_ffhq_cropper_runs_with_internal_detector(image_rgb):
    cropper = FFHQCropper(FFHQCropConfig(detector_device='cpu', output_size=512))
    result = cropper.run(image_rgb)

    assert isinstance(result, CropResult)
    assert result.aligned_image.shape == (512, 512, 3)
    assert result.transform.shape == (3, 3)
    assert result.crop_quad.shape == (4, 2)
    assert 'kps_5pt' in result.landmarks_aligned
    assert result.landmarks_aligned['kps_5pt'].shape == (5, 2)
    assert result.scheme.startswith('ffhq_scale')


def test_ffhq_cropper_reuses_passed_landmark_result(image_rgb):
    """Passing a pre-computed LandmarkResult must skip the internal detector."""
    det = InsightFace106Detector(InsightFace106Config(device='cpu'))
    lm = det.run(image_rgb)

    cropper = FFHQCropper(FFHQCropConfig(detector_device='cpu'))
    result = cropper.run(image_rgb, landmark_result=lm)

    # The cropper's lazy detector should NOT have been instantiated.
    assert cropper._detector is None
    assert result.aligned_image.shape == (512, 512, 3)
    assert 'insightface_106' in result.landmarks_aligned
    assert result.landmarks_aligned['insightface_106'].shape == (106, 2)


def test_ffhq_cropper_visualize_returns_triptych(image_rgb):
    cropper = FFHQCropper(FFHQCropConfig(detector_device='cpu', output_size=256))
    result = cropper.run(image_rgb)
    vis = cropper.visualize(image_rgb, result)
    assert vis.dtype == np.uint8
    # Triptych at output_size height; width is 3 panels each ≥ output_size.
    assert vis.shape[0] == 256
    assert vis.shape[1] >= 3 * 256


def test_cropper_factory_dispatches_ffhq():
    c = make_cropper('ffhq', detector_device='cpu')
    assert c.name == 'ffhq_scale_crop'


def test_ffhq_transform_round_trips_5pt_landmarks(image_rgb):
    """The stored transform must reproduce kps_5pt_aligned from the source kps."""
    cropper = FFHQCropper(FFHQCropConfig(detector_device='cpu'))
    result = cropper.run(image_rgb)
    src = result.source_landmarks.kps_5pt
    reprojected = project_points(src, result.transform)
    np.testing.assert_allclose(
        reprojected, result.landmarks_aligned['kps_5pt'], atol=1e-3,
    )


def test_ffhq_eyes_above_mouth_in_aligned_crop(image_rgb):
    """Sanity: in the standardized crop, eyes sit above the mouth (smaller y)."""
    cropper = FFHQCropper(FFHQCropConfig(detector_device='cpu'))
    result = cropper.run(image_rgb)

    eye_r, eye_l, _, mouth_r, mouth_l = result.landmarks_aligned['kps_5pt']
    eye_y = (eye_r[1] + eye_l[1]) / 2
    mouth_y = (mouth_r[1] + mouth_l[1]) / 2
    assert eye_y < mouth_y, (
        f"eyes ({eye_y}) should be above mouth ({mouth_y}) after alignment"
    )


def test_standardize_crop_pure_function_runs():
    """Geometry function should work on synthetic 5pt input without a detector."""
    h, w = 256, 256
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    kps = np.array(
        [[100, 110], [156, 110], [128, 140], [108, 175], [148, 175]],
        dtype=np.float32,
    )
    aligned, M, quad = standardize_crop(image, kps, output_size=128, transform_size=256)
    assert aligned.shape == (128, 128, 3)
    assert M.shape == (3, 3)
    assert quad.shape == (4, 2)


def test_project_points_inverse_round_trip():
    """Forward + inverse transform should reconstruct the original points."""
    h, w = 256, 256
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    kps = np.array(
        [[100, 110], [156, 110], [128, 140], [108, 175], [148, 175]],
        dtype=np.float32,
    )
    _, M, _ = standardize_crop(image, kps, output_size=128, transform_size=256)
    M_inv = np.linalg.inv(M)
    aligned = project_points(kps, M)
    back = project_points(aligned, M_inv)
    np.testing.assert_allclose(back, kps, atol=1e-2)


def test_matter_factory_dispatches_modnet():
    try:
        m = make_matter('modnet', device='cpu')
    except FileNotFoundError as exc:
        pytest.skip(f"MODNet not available: {exc}")
    assert m.name == 'modnet'

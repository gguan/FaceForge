"""
Shared pytest fixtures for Stage 1 tests.

Environment variables:
    FACEFORGE_TEST_IMAGE   Path to a test face image (required for integration tests).
    FACEFORGE_DATA_DIR     Project data directory, defaults to 'data'.
    FACEFORGE_DEVICE       Torch device, defaults to 'cpu'.
"""

import os
import pytest
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _data_dir() -> str:
    """Resolve the data/ directory relative to the project root."""
    # tests/ sits two levels below project root (tests/stage1/conftest.py)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
    return os.environ.get('FACEFORGE_DATA_DIR', os.path.join(project_root, 'data'))


def _model_exists(rel_path: str) -> bool:
    return os.path.exists(os.path.join(_data_dir(), rel_path))


# ─── marks ────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: requires real model weights and a test image')
    config.addinivalue_line('markers', 'slow: tests that take >10 s (full pipeline)')


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def device() -> str:
    return os.environ.get('FACEFORGE_DEVICE', 'cpu')


@pytest.fixture(scope='session')
def data_dir() -> str:
    return _data_dir()


@pytest.fixture(scope='session')
def test_image_path() -> str:
    path = os.environ.get('FACEFORGE_TEST_IMAGE', '')
    if not path or not os.path.exists(path):
        pytest.skip('Set FACEFORGE_TEST_IMAGE to a face image path to run integration tests')
    return path


@pytest.fixture(scope='session')
def test_image_rgb(test_image_path) -> np.ndarray:
    import cv2
    img = cv2.imread(test_image_path)
    assert img is not None, f'Could not read image: {test_image_path}'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope='session')
def stage1_config(device, data_dir):
    """Stage1Config pointing at the real data/ directory."""
    from faceforge.stage1 import Stage1Config

    return Stage1Config(
        mica_model_path=os.path.join(data_dir, 'pretrained', 'mica.tar'),
        mediapipe_model_path=os.path.join(data_dir, 'pretrained', 'mediapipe', 'face_landmarker.task'),
        flame_model_path=os.path.join(data_dir, 'pretrained', 'FLAME2020', 'generic_model.pkl'),
        flame_masks_path=os.path.join(data_dir, 'pretrained', 'FLAME2020', 'FLAME_masks.pkl'),
        device=device,
        save_debug=False,
    )


@pytest.fixture(scope='session')
def mp_detector(stage1_config):
    """MediaPipe detector (requires face_landmarker.task)."""
    if not os.path.exists(stage1_config.mediapipe_model_path):
        pytest.skip(f'Missing: {stage1_config.mediapipe_model_path}')
    from faceforge.stage1.detection import MediaPipeDetector
    return MediaPipeDetector(stage1_config.mediapipe_model_path)


@pytest.fixture(scope='session')
def retina_detector(stage1_config):
    """RetinaFace detector (auto-downloads antelopev2 on first run)."""
    from faceforge.stage1.detection import RetinaFaceDetector
    return RetinaFaceDetector(stage1_config.device)


@pytest.fixture(scope='session')
def detection_result(test_image_rgb, mp_detector, retina_detector):
    """Pre-computed detection result for integration tests."""
    from faceforge.stage1.detection import detect_all
    result = detect_all(test_image_rgb, mp_detector, retina_detector)
    assert result is not None, 'No face detected in test image'
    return result


@pytest.fixture(scope='session')
def aligned_image(test_image_rgb, detection_result):
    """512×512 FFHQ-aligned face crop."""
    from faceforge.stage1.alignment import image_align
    return image_align(test_image_rgb, detection_result.lmks_68)

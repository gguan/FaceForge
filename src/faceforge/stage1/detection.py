"""
Face detection: MediaPipe (478 dense landmarks) + RetinaFace (native 5-point).
"""

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .mp2dlib import convert_landmarks_mediapipe_to_dlib
from .data_types import DetectionResult

# Eye landmark indices in MediaPipe 478-point model
R_EYE_MP_LMKS = [468, 469, 470, 471, 472]
L_EYE_MP_LMKS = [473, 474, 475, 476, 477]


class MediaPipeDetector:
    """MediaPipe FaceLandmarker: 478 dense → 68 dlib → eye landmarks + blendshapes."""

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> DetectionResult | None:
        """Run MediaPipe face detection.

        Args:
            image_rgb: RGB uint8 image [H, W, 3]

        Returns:
            DetectionResult or None if no face detected.
        """
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)

        if len(result.face_blendshapes) == 0:
            return None

        # Blendshape scores [52]
        blend_scores = np.array(
            [bs.score for bs in result.face_blendshapes[0]], dtype=np.float32
        )

        # Dense landmarks [478, 2] in pixel coordinates
        lmks_dense = np.array(
            [[lm.x, lm.y] for lm in result.face_landmarks[0]], dtype=np.float32
        )
        lmks_dense[:, 0] *= w
        lmks_dense[:, 1] *= h

        # Convert to 68-point dlib landmarks
        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_dense)

        # Eye landmarks [10, 2] (right eye 5 + left eye 5)
        lmks_eyes = lmks_dense[R_EYE_MP_LMKS + L_EYE_MP_LMKS]

        return DetectionResult(
            lmks_dense=lmks_dense.astype(np.float32),
            lmks_68=lmks_68.astype(np.float32),
            lmks_eyes=lmks_eyes.astype(np.float32),
            blend_scores=blend_scores,
        )


class RetinaFaceDetector:
    """RetinaFace (InsightFace): native 5-point landmarks for ArcFace alignment."""

    def __init__(self, device: str = 'cuda:0'):
        from insightface.app import FaceAnalysis

        providers = ['CUDAExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_5pt(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Detect face and return native 5-point landmarks.

        Args:
            image_bgr: BGR uint8 image [H, W, 3]

        Returns:
            [5, 2] landmarks or None if no face detected.
        """
        faces = self.app.get(image_bgr)
        if len(faces) == 0:
            return None

        if len(faces) > 1:
            # Pick face closest to image center
            h, w = image_bgr.shape[:2]
            img_center = np.array([w / 2, h / 2])
            best_idx = 0
            best_dist = float('inf')
            for i, face in enumerate(faces):
                bbox = face.bbox  # [x1, y1, x2, y2]
                center = (bbox[:2] + bbox[2:4]) / 2
                dist = np.linalg.norm(center - img_center)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            return faces[best_idx].kps.astype(np.float32)

        return faces[0].kps.astype(np.float32)


def detect_all(
    image_rgb: np.ndarray,
    mp_detector: MediaPipeDetector,
    retina_detector: RetinaFaceDetector,
) -> DetectionResult | None:
    """Run both detectors and merge results.

    Args:
        image_rgb: RGB uint8 image [H, W, 3]
        mp_detector: MediaPipe detector
        retina_detector: RetinaFace detector

    Returns:
        DetectionResult with all fields populated, or None.
    """
    # MediaPipe detection (RGB input)
    mp_result = mp_detector.detect(image_rgb)
    if mp_result is None:
        return None

    # RetinaFace detection (BGR input)
    import cv2
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    retinaface_kps = retina_detector.detect_5pt(image_bgr)

    mp_result.retinaface_kps = retinaface_kps
    return mp_result

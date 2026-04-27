"""
HRN-head end-to-end pipeline orchestrator.

Replaces modelscope's ``HeadReconstructionPipeline`` — but does NOT
depend on ``modelscope`` or ``tensorflow``. Re-uses everything that was
already vendored or available in our tree:

  * Detection      : RetinaFace (PyTorch) from ``submodules/HRN/retinaface/``
  * 106-pt landmark: ``LargeBaseLmkInfer`` from ``submodules/HRN/facelandmark/``
  * 68-pt FAN      : the ``face_alignment`` PyPI package
                     (FAN weights cached at ``data/hrn_head_model/face_alignment/``)
  * Face/head masks: BiSeNet/facer via :mod:`head_mask_adapter`
                     (replaces TF1 ``segment_face.pb`` + ``Matting_headparser_6_18.pb``)
  * BFM regressor + fitting + texture baking + template composite
                   : :mod:`._head_vendored.headrecon_model` + :mod:`.tex_processor`

The fat-face warp is skipped — we feed the original (non-fat) image as
the photometric target. This is the only intentional behavioural drift
from upstream; it weakens the chin/jaw silhouette pull but keeps the
pipeline numpy/torch-only.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import PIL.Image
import torch

from faceforge._paths import PROJECT_ROOT
from faceforge.preprocessing.segmentation.base import BaseSegmenter

from ._head_vendored.headrecon_model import HeadReconModel
from ._head_vendored.tex_processor import TexProcesser
from ._head_vendored.utils_bridge import _HRN_ROOT, _hrn_on_path, align_img, load_lm3d
from .head_mask_adapter import derive_face_and_head_masks


_DEFAULT_MODEL_DIR = PROJECT_ROOT / 'data' / 'hrn_head_model'

_REQUIRED_FILES = (
    'configuration.json',
    'pytorch_model.pt',
    'large_base_net.pth',
    'retinaface_resnet50_2020-07-20_old_torch.pth',
    'face_alignment/s3fd-619a316812.pth',
    'face_alignment/3DFAN4-4a694010b9.zip',
    'face_alignment/depth-6c4283c0e0.zip',
    'assets/3dmm/BFM/BFM_model_front.mat',
    'assets/3dmm/BFM/ourRefineBFMEye0504_model.mat',
    'assets/3dmm/BFM/ourRefineFull_model.mat',
    'assets/3dmm/template_mesh/template_ourFull_bfmEyes.obj',
    'assets/3dmm/inds/bfm_keep_inds.npy',
    'assets/3dmm/inds/ours_hair_area_inds.npy',
    'assets/3dmm/inds/ours_head_face_inds.npy',
    'assets/3dmm/inds/bfm_withou_forehead_inds.npy',
    'assets/3dmm/inds/eye_corner_inds.npy',
    'assets/3dmm/inds/eye_corner_lines.npy',
    'assets/3dmm/inds/our_refine0223_basis_withoutEyes_withUV_keypoints_inds.npy',
    'assets/3dmm/adjust_part/our_full_bfmEyes/145_nose.obj',
    'assets/3dmm/adjust_part/our_full_bfmEyes/146_neck.obj',
    'assets/3dmm/adjust_part/our_full_bfmEyes/147_neckSlim2.obj',
    'assets/3dmm/adjust_part/our_full_bfmEyes/148_neckLength.obj',
    'assets/3dmm/adjust_part/our_full/145_nose.obj',
    'assets/3dmm/adjust_part/our_full/154_neck.obj',
    'assets/3dmm/adjust_part/our_full/our_mean_adjust_eyes.obj',
    'assets/texture/template_bald_tex_2.jpg',
    'assets/texture/template_withHair_tex.jpg',
    'assets/texture/hair_mask_male.png',
    'assets/texture/jaw_edge_mask2.png',
    'assets/texture/fg_mask.png',
    'assets/texture/face_mask_singleview.jpg',
    'assets/texture/cheek_area_mask.png',
    'assets/similarity_Lm3D_all.mat',
)


def check_assets(model_dir: Path) -> None:
    missing = [f for f in _REQUIRED_FILES if not (model_dir / f).exists()]
    if not missing:
        return
    head = '\n  - '.join(missing[:10])
    tail = f"\n  ... ({len(missing) - 10} more)" if len(missing) > 10 else ''
    raise FileNotFoundError(
        f"HRN head-recon assets missing under {model_dir}:\n  - {head}{tail}"
    )


@dataclass
class HRNHeadPipelineConfig:
    """Tunables for :class:`HRNHeadPipeline`."""

    model_dir: str = str(_DEFAULT_MODEL_DIR)
    hair_tex: bool = False
    device: str = 'cuda'

    # Cap the long edge of huge inputs so the BFM regressor + fitting
    # loop stay within memory (modelscope does the same at 1500).
    max_long_side: int = 1500

    # Reject inputs whose predicted Euler angles exceed this many degrees
    # on any axis. modelscope ships 30°; raising the threshold lets you
    # try profile/three-quarter shots at the cost of fitting stability.
    # Set very large (e.g. 180.0) to disable the check.
    pose_threshold_deg: float = 30.0


def _stage_face_alignment_weights(model_dir: Path) -> None:
    """Copy the cached FAN weights into ``torch.hub`` so face_alignment
    finds them locally instead of trying to download.

    Idempotent — skips files already in place.
    """
    try:
        from torch.hub import get_dir
    except Exception:
        from torch.hub import _get_torch_home as get_dir   # type: ignore

    hub_dir = Path(get_dir())
    cp_dir = hub_dir / 'checkpoints'
    cp_dir.mkdir(parents=True, exist_ok=True)

    src = model_dir / 'face_alignment'
    for fname in ('s3fd-619a316812.pth',
                  '3DFAN4-4a694010b9.zip',
                  'depth-6c4283c0e0.zip'):
        dst = cp_dir / fname
        if not dst.exists() and (src / fname).exists():
            shutil.copy(src / fname, dst)


@contextlib.contextmanager
def _torch_load_legacy():
    """Temporarily make ``torch.load`` default to ``weights_only=False``.

    PyTorch 2.6+ defaults ``weights_only=True``, which rejects HRN's
    pickled numpy scalars. The HRN/RetinaFace checkpoints under
    ``data/hrn_head_model/`` are first-party (downloaded from modelscope
    hub), so loading with the legacy unpickler is safe.
    """
    import torch
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


@contextlib.contextmanager
def _isolated_cwd(target: Path):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(old)


class HRNHeadPipeline:
    """End-to-end HRN-head pipeline.

    Heavy in __init__ (loads RetinaFace + 106-pt + FAN + BFM regressor +
    extended .mat models), so cache the instance.
    """

    def __init__(
        self,
        config: HRNHeadPipelineConfig | None = None,
        segmenter: Optional[BaseSegmenter] = None,
    ):
        self.config = config or HRNHeadPipelineConfig()
        self.model_dir = Path(self.config.model_dir).resolve()
        check_assets(self.model_dir)

        self.device = torch.device(
            self.config.device if (
                'cuda' in self.config.device and torch.cuda.is_available())
            else 'cpu')
        self.device_name = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Optional shared segmenter — lazy-built on first run() if not passed.
        self._segmenter = segmenter

        # ---------------- 106-pt landmarker (PyTorch, from submodules/HRN) ---
        with _hrn_on_path():
            from facelandmark.large_base_lmks_infer import LargeBaseLmkInfer
            from retinaface.predict_single import Model as RetinaFaceModel
        self._LargeBaseLmkInfer = LargeBaseLmkInfer
        with _torch_load_legacy():
            self.large_base_lmks_model = LargeBaseLmkInfer.model_preload(
                str(self.model_dir / 'large_base_net.pth'),
                self.device_name == 'cuda',
            )

        # ---------------- RetinaFace face detector --------------------------
        self.detector = RetinaFaceModel(max_size=512, device=self.device_name)
        with _torch_load_legacy():
            det_state = torch.load(
                str(self.model_dir / 'retinaface_resnet50_2020-07-20_old_torch.pth'),
                map_location='cpu')
        self.detector.load_state_dict(det_state)
        self.detector.eval()

        # ---------------- HeadReconModel (BFM regressor + fitting + bake) ---
        self.model = HeadReconModel(
            model_dir=str(self.model_dir),
            pose_threshold_radians=np.deg2rad(self.config.pose_threshold_deg),
        )
        self.model.set_device(self.device)
        with _torch_load_legacy():
            self.model.setup(str(self.model_dir / 'pytorch_model.pt'))
        self.model.eval()
        self.model.set_render(image_res=1024)

        # ---------------- 68-pt FAN -----------------------------------------
        _stage_face_alignment_weights(self.model_dir)
        import face_alignment
        # FAN's API renamed _3D → THREE_D in newer releases.
        landmarks_type = getattr(
            face_alignment.LandmarksType, 'THREE_D',
            getattr(face_alignment.LandmarksType, '_3D', None))
        if landmarks_type is None:
            raise RuntimeError("face_alignment LandmarksType (3D) not found")
        self.lm_sess = face_alignment.FaceAlignment(landmarks_type, flip_input=False)

        # ---------------- 5pt → 3D BFM std landmarks ------------------------
        self.lm3d_std = load_lm3d(str(self.model_dir / 'assets'))

        # ---------------- 4096² template texture compositor -----------------
        self.tex_processor = TexProcesser(model_root=str(self.model_dir))

    # ------------------------------------------------------------------ utils

    def _ensure_segmenter(self) -> BaseSegmenter:
        if self._segmenter is None:
            from faceforge.preprocessing.segmentation.bisenet import (
                BiSeNetConfig, BiSeNetSegmenter,
            )
            self._segmenter = BiSeNetSegmenter(BiSeNetConfig(device=self.config.device))
        return self._segmenter

    def _detect_face_boxes(self, img_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.predict_jsons(rgb)
        boxes = []
        for anno in results:
            if anno['score'] == -1:
                break
            boxes.append({
                'x1': anno['bbox'][0],
                'y1': anno['bbox'][1],
                'x2': anno['bbox'][2],
                'y2': anno['bbox'][3],
            })
        return boxes

    def _infer_106(self, img_bgr: np.ndarray, boxes: list[dict]) -> list[np.ndarray]:
        """For each detection, run the LargeBaseLmkInfer two-pass crop and
        return its 106-point landmark set in source image coordinates."""
        INPUT_SIZE = 224
        ENLARGE = 1.35
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        landmarks = []

        for det in boxes:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            sz = max(y2 - y1 + 1, x2 - x1 + 1) * ENLARGE
            base_lmks = self._crop_and_predict(rgb, cx, cy, sz, INPUT_SIZE, h_img, w_img)
            # second pass: use detected landmarks to retighten the crop
            x1n, y1n = np.min(base_lmks, axis=0)
            x2n, y2n = np.max(base_lmks, axis=0)
            cx2, cy2 = (x1n + x2n) / 2, (y1n + y2n) / 2
            sz2 = max(y2n - y1n + 1, x2n - x1n + 1) * ENLARGE
            base_lmks = self._crop_and_predict(rgb, cx2, cy2, sz2, INPUT_SIZE, h_img, w_img)
            landmarks.append(base_lmks)
        return landmarks

    def _crop_and_predict(self, rgb, cx, cy, sz, INPUT_SIZE, H, W):
        x1 = cx - sz / 2
        y1 = cy - sz / 2
        x2 = x1 + sz
        y2 = y1 + sz
        dx = max(0, -x1); dy = max(0, -y1)
        x1c, y1c = max(0, x1), max(0, y1)
        edx = max(0, x2 - W); edy = max(0, y2 - H)
        x2c, y2c = min(W, x2), min(H, y2)

        crop = rgb[int(y1c):int(y2c), int(x1c):int(x2c)]
        if dx or dy or edx or edy:
            crop = cv2.copyMakeBorder(
                crop, int(dy), int(edy), int(dx), int(edx),
                cv2.BORDER_CONSTANT, value=(103.94, 116.78, 123.68))
        crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))

        base_lmks = self._LargeBaseLmkInfer.infer_img(
            crop, self.large_base_lmks_model, self.device_name == 'cuda')
        inv_scale = sz / INPUT_SIZE
        affine = np.zeros((106, 2), dtype=np.float32)
        for idx in range(106):
            affine[idx, 0] = base_lmks[0][idx * 2 + 0] * inv_scale + x1
            affine[idx, 1] = base_lmks[0][idx * 2 + 1] * inv_scale + y1
        return affine

    def _prepare_68(self, img_bgr: np.ndarray, lm_106: np.ndarray) -> np.ndarray:
        """Crop tightly around the 5 points sampled from the 106-pt result,
        then run FAN to get 68 landmarks back in the source frame.

        ``submodules/HRN/util/preprocess.align_for_lm`` reads its
        BBRegressor .mat from the hardcoded relative path
        ``'util/BBRegressorParam_r.mat'``, so we ``chdir`` into the HRN
        root for the duration of the call.
        """
        five = lm_106[[74, 83, 54, 84, 90]]
        with _hrn_on_path(), _isolated_cwd(_HRN_ROOT):
            from util.preprocess import align_for_lm as _align_for_lm
            input_img, scale, bbox = _align_for_lm(img_bgr, five.copy())
        if scale == 0:
            return None
        input_img = np.reshape(input_img, [1, 224, 224, 3]).astype(np.float32)
        input_img = input_img[0, :, :, ::-1]
        landmark = self.lm_sess.get_landmarks_from_image(input_img)[0]
        landmark = landmark[:, :2] / scale
        landmark[:, 0] += bbox[0]
        landmark[:, 1] += bbox[1]
        return landmark

    # ------------------------------------------------------------------ I/O

    def _read_data(
        self,
        img_bgr: np.ndarray,
        lm_68: np.ndarray,
        face_mask_full: np.ndarray,
        head_mask_full: np.ndarray,
        image_res: int = 1024,
        rescale_factor: float = 75.0,
    ) -> dict:
        """Build the tensor dict that :meth:`HeadReconModel.set_input` expects."""
        # Modelscope flips y to v-direction at this stage.
        im = PIL.Image.fromarray(img_bgr[..., ::-1])
        W, H = im.size
        lm = lm_68.copy()
        lm[:, -1] = H - 1 - lm[:, -1]

        head_mask_pil = PIL.Image.fromarray(head_mask_full)

        _, im_lr_coeff, lm_lr_coeff, _ = align_img(im, lm, self.lm3d_std)
        _, im_lr, lm_lr, mask_lr_head = align_img(
            im, lm, self.lm3d_std, mask=head_mask_pil, rescale_factor=rescale_factor)
        _, im_hd, lm_hd, _ = align_img(
            im, lm, self.lm3d_std,
            target_size=image_res, rescale_factor=rescale_factor * image_res / 224)

        # face_mask: warp the FULL-RES face_mask via the same align_img
        face_mask_pil = PIL.Image.fromarray(face_mask_full)
        _, _, _, mask_lr = align_img(
            im, lm.copy(), self.lm3d_std, mask=face_mask_pil, rescale_factor=rescale_factor)

        def _to_im_t(pil):
            return torch.tensor(
                np.array(pil) / 255., dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0)

        def _to_mask_t(pil):
            arr = np.array(pil)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return torch.tensor(arr / 255., dtype=torch.float32)[None, None, :, :]

        return {
            'imgs':       _to_im_t(im_lr),
            'imgs_hd':    _to_im_t(im_hd),
            'imgs_fat_hd': _to_im_t(im_hd),    # no fat-face; reuse hd
            'lms':        torch.tensor(lm_lr).unsqueeze(0),
            'lms_hd':     torch.tensor(lm_hd).unsqueeze(0),
            'face_mask':  _to_mask_t(mask_lr),
            'head_mask':  _to_mask_t(mask_lr_head),
            'imgs_coeff': _to_im_t(im_lr_coeff),
            'lms_coeff':  torch.tensor(lm_lr_coeff).unsqueeze(0),
        }

    # ------------------------------------------------------------------ run

    def run(self, image_rgb: np.ndarray) -> dict:
        """Reconstruct one image. Returns dict with mesh + texture map.

        The dict has these keys:
            ``vertices``    : [N, 3] float32, world coords (z mirrored)
            ``triangles``   : [F, 3] int32, 0-indexed
            ``uvs``         : [N, 2] float32
            ``faces_uv``    : [F, 3] int32, 1-indexed (OBJ-style)
            ``normals``     : [N, 3] float32
            ``texture_map`` : [4096, 4096, 3] BGR float32 in [0, 255]
            ``coeffs``      : dict of BFM coeffs (id/exp/tex/angle/gamma/trans)
        """
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if max(bgr.shape[0], bgr.shape[1]) > self.config.max_long_side:
            from ._head_vendored.utils_bridge import resize_on_long_side
            bgr, _ = resize_on_long_side(bgr, self.config.max_long_side)

        boxes = self._detect_face_boxes(bgr)
        if not boxes:
            raise ValueError("no face detected by RetinaFace")
        lm_106s = self._infer_106(bgr, boxes)
        if not lm_106s:
            raise ValueError("106-pt landmark inference returned no faces")
        lm_106 = lm_106s[0]    # use highest-confidence face

        lm_68 = self._prepare_68(bgr, lm_106)
        if lm_68 is None:
            raise ValueError("68-pt FAN landmark inference failed")

        segmenter = self._ensure_segmenter()
        seg_result = segmenter.run(image_rgb if bgr.shape == image_rgb.shape
                                   else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        face_mask_full, head_mask_full = derive_face_and_head_masks(seg_result)

        data = self._read_data(bgr, lm_68, face_mask_full, head_mask_full)
        self.model.set_input(data)
        output = self.model.forward()
        if output is None:
            rx, ry, rz = self.model.last_predicted_pose
            thr_deg = self.config.pose_threshold_deg
            raise ValueError(
                f"HRN-head rejected pose: rx={np.rad2deg(rx):+.1f}° "
                f"ry={np.rad2deg(ry):+.1f}° rz={np.rad2deg(rz):+.1f}° "
                f"(threshold ±{thr_deg:.0f}°). Pass pose_threshold_deg=... "
                f"to relax (e.g. 60° for 3/4 shots; 90° to disable)."
            )

        tex_map = np.asarray(output['tex_map'], dtype=np.float32)
        tex_map = self.tex_processor.post_process_texture(
            tex_map, hair_tex=self.config.hair_tex)

        return {
            'vertices':    output['vertices'],
            'triangles':   output['triangles'],
            'uvs':         output['uvs'],
            'faces_uv':    output['faces_uv'],
            'normals':     output['normals'],
            'texture_map': tex_map,
            'coeffs':      {k: v.detach().cpu().numpy() for k, v in output['coeffs'].items()},
        }

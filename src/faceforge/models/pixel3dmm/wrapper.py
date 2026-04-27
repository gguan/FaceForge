"""
pixel3dmm tracker wrapper — clean preprocessing-component-driven pipeline.

The official pixel3dmm pipeline is a 3-step recipe (see
``submodules/pixel3dmm/README.md`` §2):

  1. ``run_preprocessing.py``   PIPNet cropping + facer segmentation + MICA
  2. ``network_inference.py``   normal map + UV map prediction (×2)
  3. ``track.py``               iterative FLAME fit (single-image: ``iters=800``)

Tracker (``submodules/pixel3dmm/src/pixel3dmm/tracking/tracker.py``) reads
this on-disk layout:

    ${PIXEL3DMM_PREPROCESSED_DATA}/${video_name}/
      cropped/{i:05d}.{jpg|png}      face crop
      seg_og/{i:05d}.png             facer 19-class parsing (raw IDs)
      mica/<sub>/identity.npy        MICA shape (averaged across subdirs)
      PIPnet_landmarks/{i:05d}.npy   98-pt landmarks normalized to [0, 1]
      p3dmm/normals/{i:05d}.png      network_inference normals
      p3dmm/uv_map/{i:05d}.png       network_inference uv

Our wrapper produces the same on-disk layout from FaceForge's
preprocessing components:

  * cropped/         → ``prepared.aligned_image`` (FFHQCropper)
  * seg_og/          → ``prepared.segmentation`` (BiSeNet/facer; auto-translated to facer IDs)
  * PIPnet_landmarks → ``prepared.landmarks['wflw_98']`` (PIPNet98Detector)
                       projected to aligned coords + normalized to [0, 1]
  * mica/0/          → ``prepared.identity_shape`` (MICAIdentityEstimator)
  * p3dmm/normals + uv → in-process ``Pixel3DMMInference`` (faceforge.stage2)

Then we hand over to pixel3dmm's ``Tracker`` for the actual fit. Tunable
hyperparameters (``iters``, ``uv_map_super``, ``normal_super``, ``sil_super``,
``ignore_mica``, ``use_flame2023`` …) are exposed on ``Pixel3DMMConfig``
mirroring ``configs/tracking.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from faceforge._paths import PROJECT_ROOT
from faceforge.models.base import BaseModel, ModelOutput, ModelRequirements
from faceforge.pipeline.types import PreparedInputs


class _OverlayPreserver:
    """Capture stageP3M's tracker output dir before its tempdir cleanup.

    stageP3M.P3MPipeline calls ``visualizer.preserve_tracking_outputs(path)``
    inside its ``finally:`` block right before deleting the tempdir. We
    copy the relevant per-frame artefact subfolders (``video/``, ``mesh/``,
    ``initialization/``) into a long-lived scratch directory so the
    wrapper can read them back to build each frame's
    :class:`ModelOutput`.

    Args:
        preserve_root: directory where preserved snapshots land. Each
            run creates a new ``pixel3dmm_overlay_<rand>/`` subdir under it.
    """

    def __init__(self, preserve_root: Path):
        self.preserve_root = preserve_root
        self.preserved_dir: Optional[Path] = None

    def preserve_tracking_outputs(self, tracker_output_dir: str) -> None:
        import shutil
        import tempfile
        src = Path(tracker_output_dir)
        if not src.exists():
            return
        root = self.preserve_root
        root.mkdir(parents=True, exist_ok=True)
        dst = Path(tempfile.mkdtemp(prefix='pixel3dmm_overlay_', dir=str(root)))
        for sub in ('video', 'mesh', 'initialization', 'joint_initialization'):
            s = src / sub
            if s.exists():
                shutil.copytree(s, dst / sub, dirs_exist_ok=True)
        self.preserved_dir = dst


@dataclass
class Pixel3DMMConfig:
    """Tunables for :class:`Pixel3DMMModel`.

    Mirrors pixel3dmm's ``configs/tracking.yaml``. Most defaults match the
    YAML; key overrides:

      * ``iters=800`` (was 200) — README recommends 500-800 for single-image.
      * ``is_discontinuous=True`` — lets multi-image batches with N=1 avoid
        the joint-stage assertion that needs an even temporal window.

    To match the README's "Multi-Image Inference" recipe set::

        iters=1500, include_neck=False, w_exp=0.1, use_mouth_lmk=False,
        w_shape=0.01, w_shape_general=0.001, normal_super=2000.0,
        sil_super=1000.0, use_flame2023=True, ignore_mica=True,
        is_discontinuous=True
    """

    # Pixel3DMM source layout
    pixel3dmm_code_base: str = 'submodules/pixel3dmm'
    pixel3dmm_uv_ckpt: str = 'data/pretrained/uv.ckpt'
    pixel3dmm_normal_ckpt: str = 'data/pretrained/normals.ckpt'
    flame_model_path: str = 'data/pretrained/FLAME2020/generic_model.pkl'

    # Working resolution — Tracker reads ``cropped/`` at this size.
    render_size: int = 256
    device: str = 'cuda:0'

    # ----------------- Tracker hyperparameters (mirror tracking.yaml) -----
    iters: int = 800                        # online stage iters per frame
    global_iters: int = 5000                # video-only joint stage iters

    use_flame2023: bool = False
    ignore_mica: bool = False
    include_neck: bool = True

    # Loss weights (key tunable)
    uv_map_super: float = 2000.0
    normal_super: float = 1000.0
    sil_super: float = 500.0

    use_mouth_lmk: bool = True
    w_shape: float = 0.2
    w_shape_general: float = 0.05
    w_exp: float = 0.05

    # Multi-image (uncoupled frames) vs video sequence
    is_discontinuous: bool = False
    early_stopping_delta: float = 5.0

    # Free-form override dict — keys here trump everything above and the
    # underlying YAML defaults. Use for params not surfaced explicitly.
    tracker_overrides: dict = field(default_factory=dict)

    # Where preserved per-frame overlays land (mesh.ply, joint_initialization
    # /00000.png, etc.). Co-located with final outputs by default — caller
    # can point this somewhere else if they want a separate preserve path.
    overlay_preserve_root: str = str(PROJECT_ROOT / 'output' / 'pixel3dmm' / '_overlays')


class Pixel3DMMModel(BaseModel):
    """pixel3dmm tracker exposed through the unified model interface.

    Driven entirely by FaceForge's preprocessing components — no DECA, no
    synthetic landmarks. The tracker initialises its own
    expression/jaw/pose/texture/lighting and optimises them against
    photometric + normal + uv + silhouette + landmark losses.
    """

    name = 'pixel3dmm'
    requirements = ModelRequirements(
        mode='sequence',
        needs_aligned_image=True,        # FFHQCropper output drives the cropped/ dir
        needs_landmarks=['wflw_98'],     # PIPNet98Detector — Tracker reads PIPnet_landmarks/
        needs_segmentation='bisenet_19', # auto-translated to facer IDs in the bridge
        needs_matte=False,
        needs_identity_shape=True,       # MICA → mica/0/identity.npy
    )

    def __init__(self, config: Pixel3DMMConfig | None = None):
        self.config = config or Pixel3DMMConfig()
        self._p3m: Any = None
        self._overlay_preserver: Optional[_OverlayPreserver] = None

    # ------------------------------------------------------------------ run

    def run(self, prepared: PreparedInputs) -> ModelOutput:
        return self.run_sequence([prepared])[0]

    def run_sequence(
        self,
        prepared_list: list[PreparedInputs],
    ) -> list[ModelOutput]:
        from faceforge.stage1.data_types import Stage1Output
        from faceforge.stageP3M.config import P3MConfig
        from faceforge.stageP3M.pipeline import P3MPipeline

        if self._p3m is None:
            tracker_overrides = self._build_tracker_overrides()
            p3m_cfg = P3MConfig(
                pixel3dmm_code_base=self.config.pixel3dmm_code_base,
                pixel3dmm_uv_ckpt=self.config.pixel3dmm_uv_ckpt,
                pixel3dmm_normal_ckpt=self.config.pixel3dmm_normal_ckpt,
                flame_model_path=self.config.flame_model_path,
                render_size=self.config.render_size,
                device=self.config.device,
                tracker_overrides=tracker_overrides,
            )
            preserve_root = Path(self.config.overlay_preserve_root).resolve()
            preserve_root.mkdir(parents=True, exist_ok=True)
            preserver = _OverlayPreserver(preserve_root=preserve_root)
            self._p3m = P3MPipeline(p3m_cfg, visualizer=preserver)
            self._overlay_preserver = preserver

        stage1_list = [self._build_stage1_output(p) for p in prepared_list]
        stage2_output = self._p3m.run(stage1_list)
        return self._convert_to_model_outputs(stage2_output, prepared_list)

    # ----------------------------------------------------------- visualize

    def visualize(
        self,
        prepared: PreparedInputs,
        output: ModelOutput,
    ) -> np.ndarray:
        """Composite [aligned input | rendered overlay] horizontally."""
        panels: list[np.ndarray] = []
        if prepared.aligned_image is not None:
            panels.append(prepared.aligned_image)
        if output.rendered_overlay is not None:
            panels.append(output.rendered_overlay)
        if not panels:
            return prepared.image_rgb.copy()

        target_h = max(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            if p.shape[0] != target_h:
                scale = target_h / p.shape[0]
                p = cv2.resize(
                    p, (max(1, int(p.shape[1] * scale)), target_h),
                    interpolation=cv2.INTER_AREA,
                )
            resized.append(p)
        return np.concatenate(resized, axis=1)

    # ------------------------------------------------------------ helpers

    def _build_tracker_overrides(self) -> dict:
        """Translate :class:`Pixel3DMMConfig` into a YAML-merge override dict.

        Only fields the user has expressed an opinion about land here; the
        rest of pixel3dmm's ``configs/tracking.yaml`` defaults are kept
        intact.
        """
        cfg = self.config
        overrides: dict[str, Any] = dict(cfg.tracker_overrides or {})
        overrides.setdefault('iters', cfg.iters)
        overrides.setdefault('global_iters', cfg.global_iters)
        overrides.setdefault('use_flame2023', cfg.use_flame2023)
        overrides.setdefault('ignore_mica', cfg.ignore_mica)
        overrides.setdefault('include_neck', cfg.include_neck)
        overrides.setdefault('uv_map_super', cfg.uv_map_super)
        overrides.setdefault('normal_super', cfg.normal_super)
        overrides.setdefault('sil_super', cfg.sil_super)
        overrides.setdefault('use_mouth_lmk', cfg.use_mouth_lmk)
        overrides.setdefault('w_shape', cfg.w_shape)
        overrides.setdefault('w_shape_general', cfg.w_shape_general)
        overrides.setdefault('w_exp', cfg.w_exp)
        overrides.setdefault('is_discontinuous', cfg.is_discontinuous)
        overrides.setdefault('early_stopping_delta', cfg.early_stopping_delta)
        return overrides

    def _build_stage1_output(self, prepared: PreparedInputs):
        """Pack a PreparedInputs into the on-disk-format-compatible Stage1Output.

        Only the fields the tracker actually reads off disk are populated:
        - ``aligned_image``      → cropped/
        - ``parsing_map``        → seg_og/
        - ``lmks_98``            → PIPnet_landmarks/
        - ``shape``              → mica/0/identity.npy

        Everything else (expression / pose / texture / lighting / 68pt /
        eye landmarks) is zero-filled — the tracker initialises its own
        and optimises from scratch. This is the cleanest mapping of our
        preprocessing components onto pixel3dmm's expected inputs.
        """
        from faceforge.preprocessing.cropping import project_points
        from faceforge.stage1.data_types import Stage1Output

        if prepared.aligned_image is None:
            raise RuntimeError(
                'pixel3dmm wrapper needs an aligned image — enable cropping '
                'in PreprocessingConfig'
            )
        if prepared.identity_shape is None:
            raise RuntimeError(
                'pixel3dmm wrapper needs MICA identity shape — enable identity '
                'in PreprocessingConfig'
            )
        if prepared.crop_transform is None:
            raise RuntimeError(
                'pixel3dmm wrapper needs crop_transform — enable cropping'
            )

        # PIPNet 98 — the actual landmark backend pixel3dmm trains against.
        lm_pip = prepared.landmarks.get('wflw_98')
        if lm_pip is None:
            raise RuntimeError(
                "pixel3dmm wrapper requires PIPNet 98pt landmarks "
                "(landmark_backend='pipnet_98' in PreprocessingConfig)"
            )

        # Project source-frame 98pt to aligned-frame, then normalize to [0, 1]
        # — the data_bridge writes them as fractions and tracker re-multiplies
        # by config.size at load time (see tracker.py L1553).
        lmks_98_aligned = project_points(lm_pip.landmarks, prepared.crop_transform)
        aligned_size = prepared.aligned_image.shape[0]
        lmks_98_aligned_t = torch.tensor(lmks_98_aligned).unsqueeze(0)

        # Aligned image as [1, 3, H, W] float in [0, 1].
        aligned_t = torch.tensor(
            prepared.aligned_image.astype(np.float32) / 255.0,
        ).permute(2, 0, 1).unsqueeze(0)

        # Face mask + parsing map.
        if prepared.segmentation is not None:
            mask_t = torch.tensor(
                prepared.segmentation.face_mask.astype(np.float32),
            ).unsqueeze(0)
            parsing_t = torch.tensor(
                prepared.segmentation.seg_map.astype(np.int64),
            ).unsqueeze(0)
        else:
            mask_t = torch.ones(1, aligned_size, aligned_size)
            parsing_t = None

        shape_t = torch.as_tensor(
            prepared.identity_shape, dtype=torch.float32,
        ).reshape(1, -1)

        # The tracker doesn't use these initialization values for
        # expression/pose/jaw/texture/lighting — it constructs its own
        # parameters and optimises from scratch. Zero-fill the slots so
        # Stage1Output's dataclass is well-formed.
        zero_exp = torch.zeros(1, 50)
        zero_pose = torch.zeros(1, 3)
        zero_tex = torch.zeros(1, 50)
        zero_light = torch.zeros(1, 9, 3)

        # Camera init (matches old Stage1Pipeline.run_single).
        init_focal = 2000.0 * (aligned_size / 512.0)
        focal_length = init_focal / aligned_size

        return Stage1Output(
            shape=shape_t,
            expression=zero_exp,
            head_pose=zero_pose,
            jaw_pose=zero_pose,
            texture=zero_tex,
            lighting=zero_light,
            arcface_feat=torch.zeros(1, 512),  # unused
            aligned_image=aligned_t,
            face_mask=mask_t,
            # 68 / dense / eyes are tracker-internal init slots that the
            # on-disk path doesn't read; pass zero-shaped placeholders.
            lmks_68=torch.zeros(1, 68, 2),
            lmks_dense=lmks_98_aligned_t,
            lmks_eyes=torch.zeros(1, 10, 2),
            focal_length=torch.tensor([[focal_length]]),
            principal_point=torch.tensor([[0.0, 0.0]]),
            parsing_map=parsing_t,
            lmks_98=lmks_98_aligned_t,
        )

    def _convert_to_model_outputs(
        self,
        stage2_output,
        prepared_list: list[PreparedInputs],
    ) -> list[ModelOutput]:
        outputs: list[ModelOutput] = []
        n = len(prepared_list)

        tracker_dir = (
            self._overlay_preserver.preserved_dir
            if self._overlay_preserver else None
        )

        per_frame_overlays: list[Optional[np.ndarray]] = [None] * n
        if tracker_dir is not None:
            for sub in ('joint_initialization', 'video', 'initialization'):
                cand = tracker_dir / sub
                if not cand.exists():
                    continue
                files = sorted(cand.glob('*.jpg')) + sorted(cand.glob('*.png'))
                if len(files) >= n:
                    per_frame_overlays = [
                        cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                        for f in files[:n]
                    ]
                    break

        shape_np = (
            stage2_output.shape.detach().cpu().numpy().reshape(-1)
            if hasattr(stage2_output, 'shape') else None
        )

        for i, _ in enumerate(prepared_list):
            flame_params: dict[str, np.ndarray] = {}
            if shape_np is not None:
                flame_params['shape'] = shape_np

            per_img = getattr(stage2_output, 'per_image_params', None)
            if per_img is not None and i < len(per_img):
                pi = per_img[i]
                for fname in ('expression', 'R_6d', 'jaw_6d', 'translation',
                              'lighting', 'principal_point', 'eyes_6d',
                              'neck_6d', 'eyelids'):
                    val = getattr(pi, fname, None)
                    if val is not None:
                        flame_params[fname] = (
                            val.detach().cpu().numpy()
                            if hasattr(val, 'detach') else np.asarray(val)
                        )

            outputs.append(ModelOutput(
                name=self.name,
                flame_params=flame_params or None,
                rendered_overlay=per_frame_overlays[i],
                extras={'tracker_dir': str(tracker_dir) if tracker_dir else ''},
            ))

        return outputs

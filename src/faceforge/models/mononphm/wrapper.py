"""
MonoNPHM wrapper.

MonoNPHM is a two-stage optimisation-based head reconstruction. This
wrapper drives ``submodules/MonoNPHM/scripts/inference/rec.py`` as a
subprocess; we don't import MonoNPHM into the FaceForge process because
its package layout (``mononphm.env_paths`` + tyro CLI + global config)
is incompatible with the rest of the framework.

The wrapper assumes the per-sequence preprocessing has already been
written to disk under ``<data_tracking>/<seq_name>/`` in the layout
MonoNPHM expects:

    <data_tracking>/<seq_name>/
        source/{i:05d}.png         RGB frames
        pipnet/test.npy            [N, 98, 2] normalized landmarks
        bboxes/test.npy            [N, 5]
        kpt/{i:05d}.npy            [68, 2] pixel coords
        seg/{i}.png                facer 19-class parsing
        matting/{i:05d}.png        alpha matte
        identity.npy               MICA shape coefficients
        metrical_tracker/<seq_name>/checkpoint/{i:05d}_cam_params_opencv.npz

For our cortis sequence, ``scripts/prepare_mononphm_cortis.py`` builds
this structure from FaceForge's stage1 outputs. The official FFHQ demo
sequences (00059, 00103, …) ship pre-staged.

Two operating modes mirror the upstream README:

* ``is_video=False`` (single-image FFHQ demo)
    - matches: ``rec.py … --no-intrinsics_provided --downsample_factor 0.33 --no-is_video``
    - per-frame stage1 only, no stage2
* ``is_video=True``  (Kinect-style sequence tracking)
    - matches: ``rec.py … --intrinsics_provided --is_video [--is_stage2]``
    - stage1 followed by optional stage2 (``run_stage2=True``) for the
      finalised per-frame meshes
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np

from faceforge._paths import PROJECT_ROOT
from faceforge.models.base import BaseModel, ModelOutput, ModelRequirements
from faceforge.pipeline.types import PreparedInputs


_DEFAULT_CODE = PROJECT_ROOT / 'submodules' / 'MonoNPHM'
_DEFAULT_EXP_DIR = PROJECT_ROOT / 'data' / 'pretrained'
_DEFAULT_DATA_TRACKING = PROJECT_ROOT / 'data' / 'mononphm' / 'tracking_input'
_DEFAULT_TRACKING_OUTPUT = PROJECT_ROOT / 'output' / 'mononphm'

# Bundled minimal pytorch3d shim — see _pytorch3d_shim/pytorch3d/ for
# implementations of the only three functions MonoNPHM's inference path
# needs (knn_points, so3_exp_map, so3_log_map). Avoids the painful
# CUDA-source build of upstream pytorch3d on Windows + recent CUDA.
_PYTORCH3D_SHIM = Path(__file__).resolve().parent / '_pytorch3d_shim'


@dataclass
class MonoNPHMConfig:
    """Tunables for :class:`MonoNPHMModel`.

    Mirrors rec.py's CLI surface; defaults match the README's "single-image
    FFHQ demo" flavour.

    Attributes:
        code_root:        ``submodules/MonoNPHM`` (where rec.py lives).
        experiment_dir:   ``MONONPHM_EXPERIMENT_DIR`` — parent of ``exp_name``.
        exp_name:         which subfolder of ``experiment_dir`` to use; must
                          contain ``configs.yaml`` and ``checkpoints/checkpoint_epoch_<ckpt>.tar``.
        ckpt:             checkpoint epoch number (``2500`` is the released).
        model_type:       ``'nphm'`` (full) or ``'global'`` (mlp baseline).
        data_tracking:    ``MONONPHM_DATA_TRACKING`` — parent of ``seq_name``.
        seq_name:         the per-sequence subfolder name. Required.
        intrinsics_provided / is_video: control crop + camera-init mode.
            Single-image FFHQ demo wants both False;
            Kinect/video tracking wants both True.
        run_stage2:       if True (and is_video), run stage2 after stage1.
        downsample_factor: photometric pyramid factor; README uses 0.33
            for single-image, 1/6 for video.
        tracking_output:  where rec.py writes results; mirrors
            ``MONONPHM_TRACKING_OUTPUT``.
        python_executable: which python to invoke. Default = sys.executable.
        extra_env:        merged into the subprocess env (last write wins).
    """

    code_root: str = str(_DEFAULT_CODE)
    experiment_dir: str = str(_DEFAULT_EXP_DIR)
    exp_name: str = 'pretrained_mononphm'
    ckpt: int = 2500
    model_type: Literal['nphm', 'global'] = 'nphm'

    data_tracking: str = str(_DEFAULT_DATA_TRACKING)
    seq_name: str = ''

    intrinsics_provided: bool = False
    is_video: bool = False
    run_stage2: bool = False
    downsample_factor: float = 0.33

    tracking_output: str = str(_DEFAULT_TRACKING_OUTPUT)

    # rec.py auto-skips frames whose ``z_geo.npy`` already exists (so it
    # can resume long video runs). For single-image demos and clean reruns
    # this is usually wrong: it'll skip frame 0 and try to track frame 1
    # which may not exist. Set True to wipe the seq output dir first.
    clear_existing_output: bool = False

    python_executable: str = sys.executable
    extra_env: dict = field(default_factory=dict)


class MonoNPHMModel(BaseModel):
    """MonoNPHM tracking driver — subprocesses ``rec.py``.

    Lifecycle:
        1. Construct with a :class:`MonoNPHMConfig` (validates code_root +
           ckpt + configs.yaml exist; does NOT load weights).
        2. Call :meth:`run_sequence` to run stage1 (and optionally stage2)
           on the on-disk sequence at ``<data_tracking>/<seq_name>``.
        3. Per-frame outputs are read back from
           ``<tracking_output>/<exp_name>/stage1/<seq_name>/<NNNNN>/``.
    """

    name = 'mononphm'
    requirements = ModelRequirements(
        mode='sequence',
        needs_aligned_image=True,
        needs_landmarks=['wflw_98'],
        needs_segmentation='facer_celebm_19',
        needs_matte=True,
        needs_identity_shape=True,
    )

    def __init__(self, config: MonoNPHMConfig | None = None):
        self.config = config or MonoNPHMConfig()
        self.code_root = Path(self.config.code_root).resolve()
        self.experiment_dir = Path(self.config.experiment_dir).resolve()
        self.data_tracking = Path(self.config.data_tracking).resolve()
        self.tracking_output = Path(self.config.tracking_output).resolve()

        if not (self.code_root / 'scripts' / 'inference' / 'rec.py').exists():
            raise FileNotFoundError(
                f"MonoNPHM not found at {self.code_root} — initialize the submodule"
            )

        exp_root = self.experiment_dir / self.config.exp_name
        cfg_yaml = exp_root / 'configs.yaml'
        ckpt_path = exp_root / 'checkpoints' / f'checkpoint_epoch_{self.config.ckpt}.tar'
        for p in (cfg_yaml, ckpt_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"MonoNPHM model file missing: {p}\n"
                    f"Expected layout under MONONPHM_EXPERIMENT_DIR={self.experiment_dir}/:\n"
                    f"  {self.config.exp_name}/configs.yaml\n"
                    f"  {self.config.exp_name}/checkpoints/checkpoint_epoch_<ckpt>.tar"
                )

    # ------------------------------------------------------------------ env

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment.

        ``mononphm.env_paths`` reads SIX env vars at import time; if any
        are unset it raises immediately, even ones we don't actually use
        at inference (TRAINING_SUPERVISION, DATA). Set sane defaults for
        all of them and let ``extra_env`` override.
        """
        env = os.environ.copy()
        env.setdefault('MONONPHM_CODE_BASE', str(self.code_root))
        env.setdefault('MONONPHM_EXPERIMENT_DIR', str(self.experiment_dir))
        env.setdefault('MONONPHM_DATA_TRACKING', str(self.data_tracking))
        env.setdefault('MONONPHM_TRACKING_OUTPUT', str(self.tracking_output))
        # Inference-only path: set placeholders for the training vars so
        # env_paths.py loads cleanly. They're not read by rec.py.
        env.setdefault('MONONPHM_TRAINING_SUPERVISION', str(self.code_root / 'unused'))
        env.setdefault('MONONPHM_DATA', str(self.code_root / 'unused'))

        # Make MonoNPHM's package importable from ``submodules/MonoNPHM/src``
        # without requiring `pip install -e`. Also prepend the bundled
        # pytorch3d shim so MonoNPHM imports the lightweight reimplementation
        # instead of needing the full CUDA build.
        existing = env.get('PYTHONPATH', '')
        prepend = (
            str(_PYTORCH3D_SHIM) + os.pathsep
            + str(self.code_root / 'src')
        )
        env['PYTHONPATH'] = (prepend + os.pathsep + existing) if existing else prepend

        env.update({k: str(v) for k, v in self.config.extra_env.items()})
        return env

    # ----------------------------------------------------------------- cmd

    def _build_cmd(self, stage2: bool) -> list[str]:
        """Translate :class:`MonoNPHMConfig` into rec.py CLI flags."""
        c = self.config
        cmd = [
            c.python_executable,
            str(Path('scripts') / 'inference' / 'rec.py'),
            '--seq_name', c.seq_name,
            '--model_type', c.model_type,
            '--exp_name', c.exp_name,
            '--ckpt', str(c.ckpt),
            '--downsample_factor', str(c.downsample_factor),
            '--intrinsics_provided' if c.intrinsics_provided else '--no-intrinsics_provided',
            '--is_video' if c.is_video else '--no-is_video',
            '--is_stage2' if stage2 else '--no-is_stage2',
        ]
        return cmd

    # ----------------------------------------------------------------- run

    def run(self, prepared: PreparedInputs) -> ModelOutput:
        """Single-frame entry: defers to :meth:`run_sequence`."""
        return self.run_sequence([prepared])[0]

    def run_sequence(self, prepared_list: list[PreparedInputs]) -> list[ModelOutput]:
        if not self.config.seq_name:
            raise RuntimeError(
                "MonoNPHMConfig.seq_name is required — set it to the on-disk "
                f"directory name under {self.data_tracking}/."
            )
        seq_root = self.data_tracking / self.config.seq_name
        if not seq_root.exists():
            raise FileNotFoundError(f"sequence root not found: {seq_root}")
        if not (seq_root / 'source').exists():
            raise FileNotFoundError(
                f"missing {seq_root / 'source'}/. Run the preprocessing "
                f"pipeline (e.g. scripts/prepare_mononphm_cortis.py) first."
            )

        env = self._build_env()
        self.tracking_output.mkdir(parents=True, exist_ok=True)

        if self.config.clear_existing_output:
            import shutil
            stage1_seq = self._stage1_dir()
            if stage1_seq.exists():
                shutil.rmtree(stage1_seq)
            stage2_seq = (self.tracking_output / self.config.exp_name
                          / 'stage2' / self.config.seq_name)
            if stage2_seq.exists():
                shutil.rmtree(stage2_seq)

        # Stage 1 — always.
        subprocess.run(
            self._build_cmd(stage2=False),
            cwd=str(self.code_root),
            env=env,
            check=True,
        )

        # Stage 2 — only for video sequences AND when the user asks.
        if self.config.run_stage2:
            if not self.config.is_video:
                raise ValueError(
                    "stage2 is only valid for video sequences "
                    "(MonoNPHMConfig.is_video must be True)."
                )
            subprocess.run(
                self._build_cmd(stage2=True),
                cwd=str(self.code_root),
                env=env,
                check=True,
            )

        return self._collect_outputs(prepared_list)

    # ----------------------------------------------------------- visualize

    def visualize(
        self,
        prepared: PreparedInputs,
        output: ModelOutput,
    ) -> np.ndarray:
        """[source ‖ progress overlay] strip — falls back to source-only."""
        src = prepared.image_rgb
        overlay = output.rendered_overlay
        if overlay is None:
            return src.copy()

        target_h = src.shape[0]
        if overlay.shape[0] != target_h:
            scale = target_h / overlay.shape[0]
            new_w = max(1, int(round(overlay.shape[1] * scale)))
            overlay = cv2.resize(overlay, (new_w, target_h),
                                 interpolation=cv2.INTER_AREA)
        return np.concatenate([src, overlay], axis=1)

    # ------------------------------------------------------------ helpers

    def _stage1_dir(self) -> Path:
        return (self.tracking_output / self.config.exp_name
                / 'stage1' / self.config.seq_name)

    def _collect_outputs(
        self, prepared_list: list[PreparedInputs],
    ) -> list[ModelOutput]:
        """Read per-frame artefacts back into :class:`ModelOutput` records.

        rec.py per-frame output layout:

            stage1/<seq_name>/<NNNNN>/
                mesh.ply                       triangle mesh (textured-color via colorA/b + sh_coeffs)
                z_geo.npy / z_app.npy / z_exp.npy   latent codes
                rot.npy / trans.npy / scale.npy     rigid params
                colorA.npy / colorb.npy             color decoder weights
                sh_coeffs.npy                       lighting
                progress/epoch<E>_0_view0_variance00000.png
        """
        stage1_root = self._stage1_dir()
        outputs: list[ModelOutput] = []
        for i, _ in enumerate(prepared_list):
            frame_dir = stage1_root / f'{i:05d}'
            mesh_ply = frame_dir / 'mesh.ply' if (frame_dir / 'mesh.ply').exists() else None

            # Pick the highest-epoch progress png as the rendered overlay.
            rendered = None
            progress_dir = frame_dir / 'progress'
            if progress_dir.exists():
                pngs = sorted(progress_dir.glob('epoch*_0_view0_variance*.png'))
                if pngs:
                    bgr = cv2.imread(str(pngs[-1]))
                    if bgr is not None:
                        rendered = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            extras: dict[str, Any] = {'frame_dir': str(frame_dir)}
            for key in ('z_geo', 'z_app', 'z_exp', 'rot', 'trans', 'scale',
                        'colorA', 'colorb', 'sh_coeffs'):
                p = frame_dir / f'{key}.npy'
                if p.exists():
                    extras[key] = np.load(p)

            outputs.append(ModelOutput(
                name=self.name,
                mesh_obj_path=mesh_ply,
                rendered_overlay=rendered,
                extras=extras,
            ))
        return outputs

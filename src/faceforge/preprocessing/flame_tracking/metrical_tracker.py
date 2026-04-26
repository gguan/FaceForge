"""Wrapper around the metrical-tracker bundled in the MonoNPHM submodule.

The original tracker is a CLI (``python tracker.py --cfg <yml>``) that reads
a sequence directory laid out the way MonoNPHM's preprocessing scripts
produce. This component:

  * builds the per-sequence YAML config on the fly,
  * launches the tracker as a subprocess,
  * walks the output checkpoint folder and exposes per-frame FLAME params
    + camera intrinsics/extrinsics in a friendly dict-of-arrays form.

It does *not* reimplement the optimization. If the user prefers a different
tracker (e.g., pixel3dmm's tracker — see ``stageP3M``) they can write a
parallel subclass of :class:`BaseFlameTracker`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from faceforge._paths import PROJECT_ROOT

from .base import BaseFlameTracker, FlameTrackingResult, SequenceInputs
from .visualize import load_tracker_overlay


_DEFAULT_TRACKER_ROOT = (
    PROJECT_ROOT / 'submodules' / 'MonoNPHM' / 'src' / 'mononphm'
    / 'preprocessing' / 'metrical-tracker'
)


@dataclass
class MetricalTrackerConfig:
    tracker_root: str = str(_DEFAULT_TRACKER_ROOT)
    save_subdir: str = 'metrical_tracker'         # written under <seq_root>/
    optimize_shape: bool = True
    optimize_jaw: bool = True
    begin_frames: int = 0
    python_executable: str = sys.executable
    extra_env: dict = field(default_factory=dict)


class MetricalTracker(BaseFlameTracker):
    """Subprocess-based wrapper around metrical-tracker."""

    name = 'metrical_tracker'

    def __init__(self, config: MetricalTrackerConfig | None = None):
        self.config = config or MetricalTrackerConfig()
        self.tracker_root = Path(self.config.tracker_root).resolve()
        if not (self.tracker_root / 'tracker.py').exists():
            raise FileNotFoundError(
                f"metrical-tracker not found at {self.tracker_root} "
                "— did you initialize the MonoNPHM submodule?"
            )

    # ------------------------------------------------------------------ run

    def run_sequence(self, inputs: SequenceInputs) -> FlameTrackingResult:
        seq_root = inputs.seq_root.resolve()
        seq_name = seq_root.name

        config_yml = self._write_config(inputs, seq_name)
        save_dir = (seq_root / self.config.save_subdir / seq_name).resolve()

        if not save_dir.exists():
            self._launch_tracker(config_yml)

        return self.read_results(save_dir)

    def read_results(self, save_dir: Path) -> FlameTrackingResult:
        """Load per-frame .frame checkpoints written by the tracker."""
        save_dir = Path(save_dir)
        ckpt_dir = save_dir / 'checkpoint'
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"no checkpoint directory at {ckpt_dir}")

        import torch

        frame_files = sorted(ckpt_dir.glob('[0-9]*.frame'))
        if not frame_files:
            raise FileNotFoundError(f"no .frame files under {ckpt_dir}")

        per_frame: dict[str, list[np.ndarray]] = {}
        flame_shape: Optional[np.ndarray] = None

        for path in frame_files:
            frame = torch.load(str(path), map_location='cpu', weights_only=False)
            for k, v in frame['flame'].items():
                if hasattr(v, 'detach'):
                    v = v.detach().cpu().numpy()
                per_frame.setdefault(f'flame_{k}', []).append(np.asarray(v))
            for k, v in frame['camera'].items():
                if hasattr(v, 'detach'):
                    v = v.detach().cpu().numpy()
                per_frame.setdefault(f'camera_{k}', []).append(np.asarray(v))

            if flame_shape is None and 'shape' in frame['flame']:
                fs = frame['flame']['shape']
                if hasattr(fs, 'detach'):
                    fs = fs.detach().cpu().numpy()
                flame_shape = np.asarray(fs).reshape(-1)

        per_frame_arr = {k: np.stack(v, axis=0) for k, v in per_frame.items()}

        video_path = save_dir / 'video.avi'

        return FlameTrackingResult(
            output_dir=save_dir,
            per_frame_params=per_frame_arr,
            flame_shape=flame_shape,
            rendered_video=video_path if video_path.exists() else None,
        )

    # ----------------------------------------------------------- visualize

    def visualize(
        self,
        result: FlameTrackingResult,
        frame_idx: int = 0,
    ) -> np.ndarray:
        return load_tracker_overlay(result.output_dir, frame_idx=frame_idx)

    # ------------------------------------------------------------ helpers

    def _write_config(self, inputs: SequenceInputs, seq_name: str) -> Path:
        seq_root = inputs.seq_root.resolve()
        save_root = (seq_root / self.config.save_subdir).resolve()

        config_dir = self.tracker_root / 'configs' / 'actors'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f'{seq_name}.yml'

        contents = (
            f"# Auto-generated by faceforge.preprocessing.flame_tracking\n"
            f"actor: '{seq_root.as_posix()}/'\n"
            f"save_folder: '{save_root.as_posix()}/'\n"
            f"optimize_shape: {'true' if self.config.optimize_shape else 'false'}\n"
            f"optimize_jaw: {'true' if self.config.optimize_jaw else 'false'}\n"
            f"begin_frames: {self.config.begin_frames}\n"
            f"keyframes: [{', '.join(str(k) for k in inputs.keyframes)}]\n"
            f"intrinsics_provided: {'true' if inputs.intrinsics_provided else 'false'}\n"
        )
        config_path.write_text(contents, encoding='utf-8')
        return config_path

    def _launch_tracker(self, config_yml: Path) -> None:
        env = os.environ.copy()
        env.update(self.config.extra_env)
        cmd = [self.config.python_executable, 'tracker.py', '--cfg', str(config_yml)]
        subprocess.run(
            cmd,
            cwd=str(self.tracker_root),
            env=env,
            check=True,
        )

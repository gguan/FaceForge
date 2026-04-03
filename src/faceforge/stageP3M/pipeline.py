"""StageP3M Pipeline: pixel3dmm tracker wrapper over Stage1 outputs."""

import shutil
import tempfile
import warnings
from pathlib import Path

import torch

from faceforge._paths import PROJECT_ROOT
from faceforge.stage1.data_types import Stage1Output
from faceforge.stage2.data_types import Stage2Output
from .config import P3MConfig
from ._data_bridge import write_preprocessed, build_tracker_config, read_tracker_results


class P3MPipeline:
    """StageP3M: pixel3dmm tracker baseline wrapper."""

    def __init__(self, config: P3MConfig | None = None,
                 visualizer=None):
        self.config = config or P3MConfig()
        self.device = torch.device(self.config.device)
        self.visualizer = visualizer
        self._tracker = None  # set after run()

        # Configure pixel3dmm asset paths
        from faceforge.stage2._pixel3dmm_paths import configure_pixel3dmm_paths
        configure_pixel3dmm_paths(self.config)

        # Pixel3DMM inference (lazy loaded for UV/normal prediction)
        self._pixel3dmm = None

    @classmethod
    def from_stage2_config(cls, s2_config, visualizer=None) -> 'P3MPipeline':
        """Create P3MPipeline from Stage2Config (for CLI compatibility)."""
        cfg = P3MConfig(
            flame_model_path=getattr(s2_config, 'flame_model_path', P3MConfig.flame_model_path),
            pixel3dmm_code_base=getattr(s2_config, 'pixel3dmm_code_base', P3MConfig.pixel3dmm_code_base),
            pixel3dmm_uv_ckpt=getattr(s2_config, 'pixel3dmm_uv_ckpt', P3MConfig.pixel3dmm_uv_ckpt),
            pixel3dmm_normal_ckpt=getattr(s2_config, 'pixel3dmm_normal_ckpt', P3MConfig.pixel3dmm_normal_ckpt),
            device=getattr(s2_config, 'device', 'auto'),
        )
        return cls(cfg, visualizer=visualizer)

    @property
    def pixel3dmm(self):
        if self._pixel3dmm is None:
            from faceforge.stage2.pixel3dmm_inference import Pixel3DMMInference
            self._pixel3dmm = Pixel3DMMInference(self.config)
        return self._pixel3dmm

    @property
    def flame(self):
        """Expose tracker's FLAME model (for mesh export in run_stage2.py)."""
        if self._tracker is not None:
            return self._tracker.flame
        return None

    @staticmethod
    def _disable_tracker_compile():
        """Disable torch.compile in pixel3dmm's tracker module.

        pixel3dmm does module-level torch.compile (tracker.py L128-129)
        which fails on Windows without Triton. We suppress dynamo errors
        so torch.compile becomes a no-op fallback, then disable COMPILE
        for subsequent compilations inside run().
        """
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

    def run(self, stage1_outputs: list[Stage1Output]) -> Stage2Output:
        """Run pixel3dmm tracker as baseline.

        Args:
            stage1_outputs: list of N Stage1Output instances

        Returns:
            Stage2Output with optimized shape + neutral mesh
        """
        import pixel3dmm.env_paths as env_paths

        # Disable torch.compile in pixel3dmm tracker (fails without Triton on Windows).
        # Must happen before first import of tracker module.
        self._disable_tracker_compile()

        N = len(stage1_outputs)
        video_name = 'p3m_baseline'
        code_base = str(PROJECT_ROOT / self.config.pixel3dmm_code_base)
        render_size = self.config.render_size

        # Create temp directory for preprocessed data
        tmp_dir = tempfile.mkdtemp(prefix='faceforge_p3m_')
        output_dir = str(Path(tmp_dir) / 'output')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save/restore env_paths and COMPILE flag
        old_preproc = env_paths.PREPROCESSED_DATA
        old_output = env_paths.TRACKING_OUTPUT
        tracker_output_dir = None

        # Disable torch.compile before importing tracker.
        # pixel3dmm does module-level torch.compile (tracker.py L128-129)
        # which fails on Windows without Triton.
        import pixel3dmm.tracking.tracker as tracker_module
        # If torch.compile already ran, the compiled version has an __wrapped__
        # attribute pointing at the original. Restore it.
        _pps = tracker_module.project_points_screen_space
        if hasattr(_pps, '_torchdynamo_orig_callable'):
            tracker_module.project_points_screen_space = _pps._torchdynamo_orig_callable
        elif hasattr(_pps, '__wrapped__'):
            tracker_module.project_points_screen_space = _pps.__wrapped__
        old_compile = tracker_module.COMPILE

        try:
            # 1. Write Stage1 data to disk in pixel3dmm format
            print(f'  [P3M] Writing {N} images to temp dir...')
            write_preprocessed(
                stage1_outputs, self.pixel3dmm,
                tmp_dir, video_name, render_size,
            )

            # 2. Build tracker config
            cfg = build_tracker_config(
                video_name=video_name,
                preprocessed_dir=tmp_dir,
                output_dir=output_dir,
                n_frames=N,
                render_size=render_size,
                code_base=code_base,
                overrides=self.config.tracker_overrides or None,
            )

            # 3. Patch pixel3dmm env_paths
            env_paths.PREPROCESSED_DATA = tmp_dir
            env_paths.TRACKING_OUTPUT = output_dir

            # 4. Disable torch.compile (may fail on Windows)
            tracker_module.COMPILE = False

            # 5. Instantiate and run tracker
            print(f'  [P3M] Running pixel3dmm tracker (render_size={render_size})...')
            tracker = tracker_module.Tracker(cfg, device=str(self.device))
            tracker_output_dir = getattr(tracker, 'output_folder', None)
            try:
                tracker.run()
            except Exception as e:
                # mediapy.write_video may fail without ffmpeg — that's OK,
                # all optimization and checkpoints are done before video export.
                if 'ffmpeg' in str(e) or 'write_video' in str(e) or 'mediapy' in str(e):
                    warnings.warn(f'Video export failed (non-critical): {e}')
                else:
                    raise

            self._tracker = tracker

            # 6. Extract results
            print(f'  [P3M] Extracting results...')
            result = read_tracker_results(tracker, stage1_outputs, self.device)

            return result

        finally:
            if self.visualizer is not None and tracker_output_dir:
                try:
                    self.visualizer.preserve_tracking_outputs(tracker_output_dir)
                except Exception as exc:
                    warnings.warn(f'Failed to preserve tracker outputs: {exc}')

            # Restore env_paths
            env_paths.PREPROCESSED_DATA = old_preproc
            env_paths.TRACKING_OUTPUT = old_output
            tracker_module.COMPILE = old_compile

            # Clean up temp dir (best-effort on Windows)
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

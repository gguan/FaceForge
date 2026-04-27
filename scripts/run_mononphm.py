"""Run MonoNPHM tracking on a pre-staged sequence.

Mirrors the official README commands:

  # FFHQ-style single-image demo (e.g. seq_name=00059):
  python scripts/run_mononphm.py 00059
  # equivalent to:
  #   rec.py --model_type nphm --exp_name pretrained_mononphm --ckpt 2500 \
  #          --seq_name 00059 --no-intrinsics_provided --downsample_factor 0.33 \
  #          --no-is_video

  # Kinect-style video tracking with stage2:
  python scripts/run_mononphm.py 510_seq_4 --is-video --intrinsics-provided --stage2

Outputs land at  output/mononphm/<exp_name>/stage1/<seq_name>/<NNNNN>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from faceforge.models.mononphm import MonoNPHMConfig, MonoNPHMModel
from faceforge.pipeline.types import PreparedInputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq_name', type=str,
                    help='subdir under data/mononphm/tracking_input/ to track')
    ap.add_argument('--exp-name', default='pretrained_mononphm')
    ap.add_argument('--ckpt', type=int, default=2500)
    ap.add_argument('--model-type', choices=['nphm', 'global'], default='nphm')
    ap.add_argument('--intrinsics-provided', action='store_true')
    ap.add_argument('--is-video', action='store_true',
                    help='Kinect-style sequence; also a prerequisite for stage2')
    ap.add_argument('--stage2', action='store_true',
                    help='also run stage2 (requires --is-video)')
    ap.add_argument('--downsample-factor', type=float, default=None,
                    help='photometric pyramid factor (default: 0.33 if not --is-video, else 1/6)')
    ap.add_argument('--data-tracking', type=Path, default=None)
    ap.add_argument('--experiment-dir', type=Path, default=None)
    ap.add_argument('--tracking-output', type=Path, default=None)
    ap.add_argument('--clear-existing', action='store_true',
                    help='wipe stage1/stage2 output dirs for this seq before running')
    args = ap.parse_args()

    if args.downsample_factor is None:
        args.downsample_factor = 1 / 6 if args.is_video else 0.33

    cfg_kwargs = dict(
        seq_name=args.seq_name,
        exp_name=args.exp_name,
        ckpt=args.ckpt,
        model_type=args.model_type,
        intrinsics_provided=args.intrinsics_provided,
        is_video=args.is_video,
        run_stage2=args.stage2,
        downsample_factor=args.downsample_factor,
        clear_existing_output=args.clear_existing,
    )
    for opt_name, opt_val in [
        ('data_tracking', args.data_tracking),
        ('experiment_dir', args.experiment_dir),
        ('tracking_output', args.tracking_output),
    ]:
        if opt_val is not None:
            cfg_kwargs[opt_name] = str(opt_val.resolve())

    cfg = MonoNPHMConfig(**cfg_kwargs)
    model = MonoNPHMModel(cfg)

    seq_root = Path(cfg.data_tracking) / cfg.seq_name
    n_frames = len(list((seq_root / 'source').glob('*.png')))
    print(f"running MonoNPHM on {cfg.seq_name} ({n_frames} frame(s)) ...")
    print(f"  experiment_dir={cfg.experiment_dir}/{cfg.exp_name}")
    print(f"  tracking_output={cfg.tracking_output}/{cfg.exp_name}/stage1/{cfg.seq_name}/")
    print(f"  is_video={cfg.is_video}  intrinsics={cfg.intrinsics_provided}  "
          f"stage2={cfg.run_stage2}  downsample={cfg.downsample_factor}")

    placeholder = [PreparedInputs(image_rgb=__import__('numpy').zeros((1, 1, 3), 'uint8'),
                                  image_id=f'{i:05d}') for i in range(n_frames)]
    outs = model.run_sequence(placeholder)

    print(f"\n=== summary: {len(outs)} frame(s) ===")
    for i, out in enumerate(outs):
        mesh = out.mesh_obj_path
        rendered = 'yes' if out.rendered_overlay is not None else 'no'
        keys = sorted(k for k in out.extras if k != 'frame_dir')
        print(f"  [{i:05d}]  mesh={mesh.name if mesh else '-':10s}  "
              f"render={rendered}  extras=[{','.join(keys)}]")


if __name__ == '__main__':
    main()

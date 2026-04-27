"""Run HRN-head reconstruction on one or more images.

Usage:
    python scripts/run_hrn_head.py <image_path> [<image_path> ...] \
        [--out <dir>] [--hair-tex] [--device cuda]

Outputs (under <out>/<image-stem>/):
    head.obj   — head mesh w/ UVs + normals
    head.mtl   — material referencing head.png
    head.png   — baked 4096² head texture (BGR-saved → load as BGR)
    viz.jpg    — [source ‖ texture] preview strip
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from faceforge.models.hrn import HRNHeadConfig, HRNHeadModel
from faceforge.pipeline.types import PreparedInputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', type=Path, nargs='+',
                    help='input image(s) — anything cv2.imread accepts')
    ap.add_argument('--out', type=Path, default=Path('output/hrn_head'))
    ap.add_argument('--hair-tex', action='store_true',
                    help='use hair template instead of bald (default: bald)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model-dir', type=Path, default=None,
                    help='override data/hrn_head_model location')
    ap.add_argument('--pose-threshold', type=float, default=90.0,
                    help='reject poses with |Euler angle| > this many degrees '
                         '(default 90° — effectively disabled; modelscope ships '
                         'with 30° but rejects most 3/4 + profile shots)')
    args = ap.parse_args()

    cfg_kwargs = {
        'hair_tex': args.hair_tex,
        'device': args.device,
        'pose_threshold_deg': args.pose_threshold,
    }
    if args.model_dir is not None:
        cfg_kwargs['model_dir'] = str(args.model_dir.resolve())
    print(f"loading HRN-head model (device={args.device}, hair_tex={args.hair_tex}) ...")
    t0 = time.time()
    model = HRNHeadModel(HRNHeadConfig(**cfg_kwargs))
    print(f"loaded in {time.time() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for img_path in args.images:
        print(f"\n=== {img_path}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  SKIP (cv2.imread failed)")
            summary.append((img_path.name, 'skip', 0.0))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        prepared = PreparedInputs(
            image_rgb=rgb, image_id=img_path.stem, image_path=str(img_path))

        t0 = time.time()
        try:
            out = model.run(prepared)
        except Exception as e:
            print(f"  FAIL ({type(e).__name__}): {e}")
            traceback.print_exc()
            summary.append((img_path.name, 'fail', time.time() - t0))
            continue
        elapsed = time.time() - t0

        out_dir = args.out / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Source copy — makes side-by-side comparison straightforward.
        cv2.imwrite(str(out_dir / 'source.jpg'),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        obj_path = model.write_obj(out, out_dir / 'head.obj')
        viz = model.visualize(prepared, out)
        cv2.imwrite(str(out_dir / 'viz.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

        print(f"  vertices: {out.mesh_vertices.shape}")
        print(f"  faces:    {out.mesh_faces.shape}")
        print(f"  texture:  {np.asarray(out.extras['texture_map']).shape}")
        print(f"  → {obj_path}  (took {elapsed:.1f}s)")
        summary.append((img_path.name, 'ok', elapsed))

    print("\n=== summary ===")
    for name, status, secs in summary:
        print(f"  {name:24s}  {status:5s}  {secs:6.1f}s")


if __name__ == '__main__':
    main()

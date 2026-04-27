"""Run pixel3dmm tracker on one or more images.

Mirrors the official README:

  # Single-image FFHQ-style fitting (e.g. slamdunk/1.jpg):
  python scripts/run_pixel3dmm.py assets/slamdunk/1.jpg

  # Multi-image (independent stills) — README §2.4.2:
  python scripts/run_pixel3dmm.py assets/slamdunk/*.jpg --multi-image

End-to-end pipeline (per the official 3-step recipe):
  1. PreprocessingConfig drives our preprocessing components:
     - FFHQCropper (cropping/ffhq) → aligned 512×512
     - PIPNet98Detector (landmark/pipnet_98) → wflw_98 landmarks
     - BiSeNetSegmenter (segmentation/bisenet) → 19-class parsing
     - MICAIdentityEstimator (identity/mica) → 300-dim shape
  2. Pixel3DMMInference (in-process) runs UV + normal map heads
  3. pixel3dmm.tracking.Tracker iteratively fits FLAME params

Outputs:
  output/pixel3dmm/<seq>/
    head.obj  — neutral FLAME mesh
    viz.jpg   — [aligned input | tracker overlay] preview strip

Note: requires pytorch3d at runtime (ops/io/structures/transforms);
see ``submodules/pixel3dmm/README.md`` §1.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from faceforge.models.pixel3dmm import Pixel3DMMConfig, Pixel3DMMModel
from faceforge.pipeline.types import PreparedInputs, PreprocessingConfig


class _PreprocessingBackends:
    """Hold PIPNet/FFHQ/BiSeNet/MICA singletons so we don't re-instantiate
    them per frame. PIPNet in particular performs a sys.path/sys.modules
    dance on first import that can't be cleanly repeated — re-instantiating
    it triggers ``ModuleNotFoundError: utils.nms``."""

    def __init__(self):
        from faceforge.preprocessing.cropping import FFHQCropConfig, FFHQCropper
        from faceforge.preprocessing.identity import MICAConfig, MICAIdentityEstimator
        from faceforge.preprocessing.landmark.pipnet_98 import (
            PIPNet98Config, PIPNet98Detector,
        )
        from faceforge.preprocessing.segmentation import (
            BiSeNetConfig, BiSeNetSegmenter,
        )
        self.pipnet = PIPNet98Detector(PIPNet98Config())
        self.cropper = FFHQCropper(FFHQCropConfig(output_size=512, scale_factor=1.3))
        self.segmenter = BiSeNetSegmenter(BiSeNetConfig())
        self.mica = MICAIdentityEstimator(MICAConfig())


def _build_prepared(
    image_path: Path,
    backends: '_PreprocessingBackends',
) -> PreparedInputs:
    """Run our preprocessing components on one image and pack a PreparedInputs."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f'failed to read {image_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f'  detecting & cropping ...')
    landmark = backends.pipnet.run(rgb)
    crop = backends.cropper.run(rgb, landmark)

    print(f'  parsing (bisenet) ...')
    seg = backends.segmenter.run(crop.aligned_image)

    print(f'  identity (mica) ...')
    mica = backends.mica.run(rgb, landmark_result=landmark)

    return PreparedInputs(
        image_rgb=rgb,
        image_id=image_path.stem,
        image_path=str(image_path),
        aligned_image=crop.aligned_image,
        crop_transform=crop.transform,
        crop_quad=crop.crop_quad,
        landmarks={'wflw_98': landmark},
        segmentation=seg,
        matte=None,
        identity_shape=mica.shape_code,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', type=Path, nargs='+',
                    help='input image(s)')
    ap.add_argument('--out', type=Path, default=Path('output/pixel3dmm'))

    # Tunables (mirror tracking.yaml + README recipes)
    ap.add_argument('--iters', type=int, default=800,
                    help='per-frame online stage iters (README single-image: 800)')
    ap.add_argument('--multi-image', action='store_true',
                    help='README §2.4.2 recipe: multi-image independent fitting')
    ap.add_argument('--use-flame2023', action='store_true')
    ap.add_argument('--ignore-mica', action='store_true')
    ap.add_argument('--no-include-neck', action='store_false', dest='include_neck')
    ap.add_argument('--uv-map-super', type=float, default=2000.0)
    ap.add_argument('--normal-super', type=float, default=1000.0)
    ap.add_argument('--sil-super', type=float, default=500.0)
    ap.add_argument('--w-shape', type=float, default=0.2)
    ap.add_argument('--w-exp', type=float, default=0.05)
    ap.add_argument('--render-size', type=int, default=256)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    # Multi-image recipe overrides — README §2.4.2 verbatim defaults.
    if args.multi_image:
        args.iters = max(args.iters, 1500)
        args.include_neck = False
        args.use_flame2023 = True
        args.ignore_mica = True
        args.uv_map_super = 2000.0
        args.normal_super = 2000.0
        args.sil_super = 1000.0
        args.w_shape = 0.01
        args.w_exp = 0.1

    cfg = Pixel3DMMConfig(
        render_size=args.render_size,
        device=args.device,
        iters=args.iters,
        use_flame2023=args.use_flame2023,
        ignore_mica=args.ignore_mica,
        include_neck=args.include_neck,
        uv_map_super=args.uv_map_super,
        normal_super=args.normal_super,
        sil_super=args.sil_super,
        use_mouth_lmk=not args.multi_image,
        w_shape=args.w_shape,
        w_exp=args.w_exp,
        is_discontinuous=args.multi_image or len(args.images) == 1,
    )
    print(f'loading pixel3dmm model (device={args.device}, iters={args.iters}) ...')
    t0 = time.time()
    model = Pixel3DMMModel(cfg)
    print(f'loaded in {time.time() - t0:.1f}s')

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    print('\nbuilding preprocessing backends (PIPNet/FFHQ/BiSeNet/MICA) ...')
    backends = _PreprocessingBackends()

    prepared_list = []
    for img_path in args.images:
        print(f'\n=== preprocess: {img_path}')
        try:
            prepared = _build_prepared(img_path, backends)
            prepared_list.append(prepared)
        except Exception as e:
            print(f'  FAIL ({type(e).__name__}): {e}')
            traceback.print_exc()
            summary.append((img_path.name, 'preprocess-fail', 0.0))

    if not prepared_list:
        print('\nno images preprocessed successfully — aborting')
        return

    print(f'\n=== tracking {len(prepared_list)} frame(s) ===')
    t0 = time.time()
    try:
        outputs = model.run_sequence(prepared_list)
    except Exception as e:
        print(f'  TRACKING FAIL: {type(e).__name__}: {e}')
        traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f'  tracker done in {elapsed:.1f}s')

    for prepared, output in zip(prepared_list, outputs):
        out_dir = args.out / prepared.image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        viz = model.visualize(prepared, output)
        cv2.imwrite(str(out_dir / 'viz.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        if output.flame_params:
            np.savez(out_dir / 'flame_params.npz',
                     **{k: np.asarray(v) for k, v in output.flame_params.items()})
        summary.append((prepared.image_id, 'ok', elapsed / max(1, len(prepared_list))))

    print('\n=== summary ===')
    for name, status, secs in summary:
        print(f'  {name:24s}  {status:18s}  {secs:6.1f}s')


if __name__ == '__main__':
    main()

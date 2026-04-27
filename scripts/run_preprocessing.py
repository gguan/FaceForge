"""Run all preprocessing components on a set of images and write a shared
on-disk layout that downstream model wrappers (pixel3dmm, mononphm, …)
can read without re-running their own preprocessing.

Usage:
    python scripts/run_preprocessing.py assets/slamdunk/*.jpg \
        --out output/_preprocessing/slamdunk

Output layout:
    output/_preprocessing/<seq>/
      <id>/
        source.png                  # full-res RGB input
        cropped.jpg                 # FFHQ-aligned 512² crop
        crop_meta.npz               # transform M [3,3] + crop_quad [4,2]
        bbox.npy                    # [x1, y1, x2, y2, conf]
        lmks_pipnet98.npy           # [98, 2] source coords
        lmks_pipnet98_aligned.npy   # [98, 2] aligned coords (matches cropped.jpg)
        lmks_pipnet98_normalized.npy # [98, 2] aligned ÷ 512 ∈ [0, 1]
        lmks_kps5pt.npy             # [5, 2] ArcFace-order, source coords
        lmks_fan68.npy              # [68, 2] FAN iBUG-68, source coords (only if --with-fan68)
        seg_facer.png               # facer 19-class parsing on cropped (if --seg-on=cropped)
        seg_facer_full.png          # facer 19-class parsing on full image (mononphm wants this)
        face_mask.png               # binary face mask (uint8 0/255)
        matte_modnet.png            # MODNet alpha (uint8 0..255) — only if --with-matte
        identity_mica.npy           # [300] MICA shape — only if --with-mica
      summary.json                  # which artifacts were produced per id

Downstream wrappers can either:
  A. consume PreparedInputs they build from this dir (pixel3dmm wrapper)
  B. convert into their model-specific layout via the helper below
     (mononphm wrapper consumes ``data/mononphm/tracking_input/<seq>/``).

The layout converters live alongside this script as small functions
(``as_pixel3dmm_layout``, ``as_mononphm_layout``) — invoke them directly
or via the per-model run scripts' ``--from-preprocessing`` flag.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def _build_backends(with_fan68: bool, with_matte: bool, with_mica: bool):
    """Lazy-construct the heavy preprocessing backends — once per process."""
    from faceforge.preprocessing.cropping import FFHQCropConfig, FFHQCropper
    from faceforge.preprocessing.landmark.pipnet_98 import (
        PIPNet98Config, PIPNet98Detector,
    )
    from faceforge.preprocessing.segmentation.facer_celebm import (
        FacerConfig, FacerSegmenter,
    )

    print('  loading PIPNet98 ...')
    pipnet = PIPNet98Detector(PIPNet98Config())
    print('  loading FFHQCropper ...')
    cropper = FFHQCropper(FFHQCropConfig(output_size=512, scale_factor=1.3))
    print('  loading facer (FaRL/CelebM) ...')
    facer = FacerSegmenter(FacerConfig())

    fan, matter, mica = None, None, None
    if with_fan68:
        print('  loading FAN 68pt ...')
        from faceforge.preprocessing.landmark.fan_68 import (
            FAN68Config, FAN68Detector,
        )
        fan = FAN68Detector(FAN68Config())
    if with_matte:
        print('  loading MODNet matter ...')
        from faceforge.preprocessing.matting import MODNetConfig, MODNetMatter
        matter = MODNetMatter(MODNetConfig())
    if with_mica:
        print('  loading MICA ...')
        from faceforge.preprocessing.identity import (
            MICAConfig, MICAIdentityEstimator,
        )
        mica = MICAIdentityEstimator(MICAConfig())

    return {
        'pipnet': pipnet, 'cropper': cropper, 'facer': facer,
        'fan': fan, 'matter': matter, 'mica': mica,
    }


def _process_one(
    image_path: Path,
    out_dir: Path,
    backends: dict,
) -> dict:
    """Run all backends on one image and dump artifacts to ``out_dir``.

    Returns a small status dict the caller stitches into ``summary.json``.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f'failed to read {image_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / 'source.png'), bgr)

    status: dict[str, str] = {}

    # PIPNet 98pt (also auto-derives 5pt + bbox).
    landmark = backends['pipnet'].run(rgb)
    np.save(out_dir / 'lmks_pipnet98.npy', landmark.landmarks)
    np.save(out_dir / 'lmks_kps5pt.npy', landmark.kps_5pt)
    np.save(out_dir / 'bbox.npy',
            np.concatenate([landmark.bbox, [landmark.confidence]]).astype(np.float32))
    status['pipnet98'] = 'ok'

    # FAN 68pt (optional).
    if backends['fan'] is not None:
        try:
            fan_lm = backends['fan'].run(rgb)
            np.save(out_dir / 'lmks_fan68.npy', fan_lm.landmarks)
            status['fan68'] = 'ok'
        except Exception as e:
            status['fan68'] = f'fail: {e}'

    # FFHQ crop using PIPNet's auto-derived 5pt.
    crop = backends['cropper'].run(rgb, landmark)
    cv2.imwrite(str(out_dir / 'cropped.jpg'),
                cv2.cvtColor(crop.aligned_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez(out_dir / 'crop_meta.npz',
             M=crop.transform, quad=crop.crop_quad)

    # PIPNet 98 in aligned space (handy for tracker landmark loss).
    from faceforge.preprocessing.cropping import project_points
    lmks_aligned = project_points(landmark.landmarks, crop.transform)
    np.save(out_dir / 'lmks_pipnet98_aligned.npy', lmks_aligned)
    np.save(out_dir / 'lmks_pipnet98_normalized.npy',
            (lmks_aligned / float(crop.aligned_image.shape[0])).astype(np.float32))
    status['cropping'] = 'ok'

    # facer segmentation on BOTH cropped (for pixel3dmm tracker which
    # consumes seg_og at render_size) AND full image (mononphm tracker
    # which reads at full res). Fall back to BiSeNet (translated to facer
    # IDs) when facer's RetinaFace detector misses on stylised inputs.
    def _seg_with_fallback(image, label):
        try:
            res = backends['facer'].run(image)
            return res.seg_map.astype(np.uint8), res.face_mask, 'facer'
        except Exception as e:
            print(f'    facer {label} failed ({type(e).__name__}); '
                  f'falling back to BiSeNet+translate')
            from faceforge.preprocessing.segmentation.bisenet import (
                BiSeNetConfig, BiSeNetSegmenter,
            )
            from faceforge.preprocessing.segmentation.translate import translate
            if 'bisenet' not in backends or backends.get('bisenet') is None:
                backends['bisenet'] = BiSeNetSegmenter(BiSeNetConfig())
            res = backends['bisenet'].run(image)
            seg = translate(res.seg_map.astype(np.uint8), 'bisenet_19',
                            'facer_celebm_19')
            return seg, res.face_mask, 'bisenet→facer'

    seg_full_map, mask_full, src_full = _seg_with_fallback(rgb, 'full')
    cv2.imwrite(str(out_dir / 'seg_facer_full.png'), seg_full_map)
    cv2.imwrite(str(out_dir / 'face_mask_full.png'),
                (mask_full.astype(np.uint8) * 255))

    seg_crop_map, mask_crop, src_crop = _seg_with_fallback(
        crop.aligned_image, 'cropped')
    cv2.imwrite(str(out_dir / 'seg_facer.png'), seg_crop_map)
    cv2.imwrite(str(out_dir / 'face_mask.png'),
                (mask_crop.astype(np.uint8) * 255))
    status['facer'] = f'{src_crop} (full={src_full})'

    # MODNet alpha matte (mononphm-only).
    if backends['matter'] is not None:
        try:
            matte = backends['matter'].run(rgb)
            cv2.imwrite(str(out_dir / 'matte_modnet.png'),
                        (matte.alpha * 255).astype(np.uint8))
            status['modnet'] = 'ok'
        except Exception as e:
            status['modnet'] = f'fail: {e}'

    # MICA identity.
    if backends['mica'] is not None:
        try:
            mica = backends['mica'].run(rgb, landmark_result=landmark)
            np.save(out_dir / 'identity_mica.npy',
                    np.asarray(mica.shape_code).reshape(-1).astype(np.float32))
            status['mica'] = 'ok'
        except Exception as e:
            status['mica'] = f'fail: {e}'

    return status


# ---------------------------------------------------------------------------- LAYOUTS

def as_pixel3dmm_layout(preproc_dir: Path, target_dir: Path,
                        render_size: int = 256) -> None:
    """Symlink/copy a shared-preprocessing dir into pixel3dmm's expected
    ``${PIXEL3DMM_PREPROCESSED_DATA}/<vid_name>/`` layout:

        rgb/{i:05d}.jpg
        cropped/{i:05d}.jpg
        seg_og/{i:05d}.png         (facer 19-class on cropped, render_size)
        PIPnet_landmarks/{i:05d}.npy  (normalized aligned 98pt)
        mica/0/identity.npy
    """
    import shutil
    image_ids = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir()])

    for sub in ('rgb', 'cropped', 'seg_og', 'PIPnet_landmarks', 'mica/0'):
        (target_dir / sub).mkdir(parents=True, exist_ok=True)

    # Identity is averaged across frames per pixel3dmm convention.
    ids = []
    for i, image_id in enumerate(image_ids):
        src = preproc_dir / image_id
        dst_rgb = target_dir / 'rgb' / f'{i:05d}.jpg'
        cv2.imwrite(str(dst_rgb), cv2.imread(str(src / 'source.png')),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # cropped — resize to render_size since tracker resizes anyway and
        # we'd rather keep the on-disk file small.
        cropped = cv2.imread(str(src / 'cropped.jpg'))
        if cropped.shape[0] != render_size:
            cropped = cv2.resize(cropped, (render_size, render_size))
        cv2.imwrite(str(target_dir / 'cropped' / f'{i:05d}.jpg'), cropped)

        # seg_og — same render_size, NEAREST.
        seg = cv2.imread(str(src / 'seg_facer.png'), cv2.IMREAD_UNCHANGED)
        if seg.shape[0] != render_size:
            seg = cv2.resize(seg, (render_size, render_size),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(target_dir / 'seg_og' / f'{i:05d}.png'), seg)

        # Landmarks — pixel3dmm tracker expects fractional [0, 1].
        lm = np.load(src / 'lmks_pipnet98_normalized.npy')
        np.save(target_dir / 'PIPnet_landmarks' / f'{i:05d}.npy', lm)

        if (src / 'identity_mica.npy').exists():
            ids.append(np.load(src / 'identity_mica.npy'))

    if ids:
        np.save(target_dir / 'mica' / '0' / 'identity.npy',
                np.mean(np.stack(ids), axis=0).astype(np.float32))


def as_mononphm_layout(preproc_dir: Path, target_dir: Path,
                       seq_name: str | None = None) -> None:
    """Symlink/copy a shared-preprocessing dir into MonoNPHM's expected
    ``${MONONPHM_DATA_TRACKING}/<seq>/`` layout:

        source/{i:05d}.png
        pipnet/test.npy           — [N, 98, 2] normalized aligned coords
        bboxes/test.npy           — [N, 5]
        kpt/{i:05d}.npy           — [68, 2] FAN coords (must have run with --with-fan68)
        seg/{i}.png               — facer (no leading zero!)
        matting/{i:05d}.png       — MODNet alpha (must have run with --with-matte)
        identity.npy              — averaged MICA shape

    Notes:
      * MonoNPHM also needs ``metrical_tracker/<seq>/checkpoint/{i:05d}_cam_params_opencv.npz``
        which this converter does NOT produce — that's a separate step.
        Run metrical-tracker upstream and copy the result in, or set
        ``MONONPHM_DATA_TRACKING`` to a sequence that already has it
        (e.g., ``data/mononphm/tracking_input/00059``).
    """
    seq_name = seq_name or target_dir.name
    image_ids = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir()])

    for sub in ('source', 'pipnet', 'bboxes', 'kpt', 'seg', 'matting'):
        (target_dir / sub).mkdir(parents=True, exist_ok=True)

    # Stack pipnet + bbox into mononphm's [N, ...] arrays.
    all_pipnet, all_bboxes = [], []
    ids = []

    for i, image_id in enumerate(image_ids):
        src = preproc_dir / image_id
        # source — copy the full-res RGB.
        full_bgr = cv2.imread(str(src / 'source.png'))
        cv2.imwrite(str(target_dir / 'source' / f'{i:05d}.png'), full_bgr)

        # PIPNet 98 — mononphm uses normalized but on FULL image, not
        # aligned crop (its tracker reads source/{i:05d}.png at full res).
        lmks_full = np.load(src / 'lmks_pipnet98.npy')
        h, w = full_bgr.shape[:2]
        all_pipnet.append((lmks_full / np.array([[w, h]], dtype=np.float32)).astype(np.float32))

        bbox = np.load(src / 'bbox.npy')   # [x1, y1, x2, y2, conf]
        # mononphm bbox normalised: [x_norm, y_norm, w_norm, h_norm, conf]
        x1, y1, x2, y2, conf = bbox
        all_bboxes.append(np.array([
            x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h, conf,
        ], dtype=np.float32))

        # FAN 68pt → kpt/.
        if (src / 'lmks_fan68.npy').exists():
            np.save(target_dir / 'kpt' / f'{i:05d}.npy',
                    np.load(src / 'lmks_fan68.npy'))

        # facer seg on FULL image (mononphm reads at full res).
        seg_full = cv2.imread(str(src / 'seg_facer_full.png'), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(target_dir / 'seg' / f'{i}.png'), seg_full)  # no leading zero

        # MODNet matte.
        matte_src = src / 'matte_modnet.png'
        if matte_src.exists():
            cv2.imwrite(str(target_dir / 'matting' / f'{i:05d}.png'),
                        cv2.imread(str(matte_src), cv2.IMREAD_UNCHANGED))

        if (src / 'identity_mica.npy').exists():
            ids.append(np.load(src / 'identity_mica.npy'))

    np.save(target_dir / 'pipnet' / 'test.npy', np.stack(all_pipnet))
    np.save(target_dir / 'bboxes' / 'test.npy', np.stack(all_bboxes))

    if ids:
        np.save(target_dir / 'identity.npy',
                np.mean(np.stack(ids), axis=0).astype(np.float32))


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True,
                    help='shared preprocessing root, e.g. output/_preprocessing/slamdunk')

    ap.add_argument('--with-fan68', action='store_true', default=True,
                    help='run FAN 68pt (mononphm needs this; pixel3dmm does not)')
    ap.add_argument('--no-fan68', action='store_false', dest='with_fan68')
    ap.add_argument('--with-matte', action='store_true', default=True,
                    help='run MODNet alpha matte (mononphm needs this)')
    ap.add_argument('--no-matte', action='store_false', dest='with_matte')
    ap.add_argument('--with-mica', action='store_true', default=True,
                    help='run MICA identity (both pipelines need this)')
    ap.add_argument('--no-mica', action='store_false', dest='with_mica')

    # Layout output (optional convenience — produce model-ready dirs in
    # one shot rather than calling the converters separately).
    ap.add_argument('--emit-pixel3dmm', type=Path, default=None,
                    help='also write pixel3dmm-ready layout to this dir')
    ap.add_argument('--emit-mononphm', type=Path, default=None,
                    help='also write mononphm-ready layout to this dir')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'shared preprocessing root: {args.out}')

    backends = _build_backends(args.with_fan68, args.with_matte, args.with_mica)

    summary = {}
    for img_path in args.images:
        image_id = img_path.stem
        img_dir = args.out / image_id
        print(f'\n=== {img_path}  →  {img_dir}')
        try:
            t0 = time.time()
            status = _process_one(img_path, img_dir, backends)
            status['elapsed_s'] = round(time.time() - t0, 2)
            summary[image_id] = status
            print(f'  done in {status["elapsed_s"]}s  '
                  f'({", ".join(f"{k}={v}" for k, v in status.items() if k != "elapsed_s")})')
        except Exception as e:
            print(f'  FAIL ({type(e).__name__}): {e}')
            traceback.print_exc()
            summary[image_id] = {'error': f'{type(e).__name__}: {e}'}

    with open(args.out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {args.out / "summary.json"}')

    if args.emit_pixel3dmm is not None:
        print(f'\nemitting pixel3dmm layout → {args.emit_pixel3dmm}')
        as_pixel3dmm_layout(args.out, args.emit_pixel3dmm)

    if args.emit_mononphm is not None:
        print(f'\nemitting mononphm layout → {args.emit_mononphm}')
        as_mononphm_layout(args.out, args.emit_mononphm)


if __name__ == '__main__':
    main()

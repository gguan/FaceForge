"""
Run the preprocessing components (landmark / segmentation / matting) on
every image in cortis/source/ and dump per-component visualizations.

  output/preprocessing_cortis/
    <stem>_landmark.jpg     — InsightFace 106 + 5pt + bbox
    <stem>_cropping.jpg     — [src+quad+lmks | aligned crop | aligned w/ lmks]
    <stem>_segmentation.jpg — [color seg | face-mask overlay]
    <stem>_matting.jpg      — [source | alpha | foreground-on-white]
    <stem>_summary.jpg      — vertical stack of all four

FLAME tracking is NOT exercised here: it is sequence-based, depends on the
upstream MICA shape, and runs metrical-tracker as a heavy subprocess. See
``faceforge.preprocessing.flame_tracking.MetricalTracker`` for the API.

Usage:
    PYTHONPATH=src python scripts/run_preprocessing_cortis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from faceforge._paths import PROJECT_ROOT
from faceforge.preprocessing.landmark import (
    InsightFace106Config,
    InsightFace106Detector,
)
from faceforge.preprocessing.segmentation import (
    BiSeNetConfig,
    BiSeNetSegmenter,
)
from faceforge.preprocessing.matting import MODNetConfig, MODNetMatter
from faceforge.preprocessing.cropping import FFHQCropConfig, FFHQCropper


CORTIS_DIR = PROJECT_ROOT / 'data' / 'mononphm' / 'tracking_input' / 'cortis' / 'source'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'preprocessing_cortis'


def _save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _resize_to_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] == w:
        return img
    scale = w / img.shape[1]
    new_h = max(1, int(round(img.shape[0] * scale)))
    return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    images = sorted(
        list(CORTIS_DIR.glob('*.jpg')) + list(CORTIS_DIR.glob('*.png'))
    )
    if not images:
        print(f"[preprocessing] no images under {CORTIS_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[preprocessing] {len(images)} images → {OUTPUT_DIR}")

    print("[preprocessing] loading landmark detector (SCRFD + 106)…")
    landmark_det = InsightFace106Detector(InsightFace106Config())

    print("[preprocessing] loading FFHQ cropper…")
    # Disable internal detector — we reuse landmark_det's output below.
    cropper = FFHQCropper(FFHQCropConfig(scale_factor=1.3, output_size=512))

    print("[preprocessing] loading BiSeNet segmenter…")
    segmenter = BiSeNetSegmenter(BiSeNetConfig())

    print("[preprocessing] loading MODNet matter…")
    try:
        matter = MODNetMatter(MODNetConfig())
        matter_ok = True
    except Exception as exc:
        print(f"[preprocessing] MODNet unavailable ({exc}) — skipping matting")
        matter = None
        matter_ok = False

    for idx, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [{idx}] {img_path.name}: cv2.imread failed")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        stem = img_path.stem

        # --- landmark ---
        lm = None
        try:
            lm = landmark_det.run(rgb)
            lm_vis = landmark_det.visualize(rgb, lm)
            _save_rgb(OUTPUT_DIR / f'{stem}_landmark.jpg', lm_vis)
        except Exception as exc:
            print(f"  [{idx}] {img_path.name}: landmark failed ({exc})")
            lm_vis = None

        # --- cropping (reuses landmark_det output) ---
        crop_vis = None
        if lm is not None:
            try:
                crop = cropper.run(rgb, landmark_result=lm)
                crop_vis = cropper.visualize(rgb, crop)
                _save_rgb(OUTPUT_DIR / f'{stem}_cropping.jpg', crop_vis)
                _save_rgb(OUTPUT_DIR / f'{stem}_aligned.png', crop.aligned_image)
            except Exception as exc:
                print(f"  [{idx}] {img_path.name}: cropping failed ({exc})")

        # --- segmentation ---
        try:
            seg = segmenter.run(rgb)
            seg_vis = segmenter.visualize(rgb, seg)
            _save_rgb(OUTPUT_DIR / f'{stem}_segmentation.jpg', seg_vis)
        except Exception as exc:
            print(f"  [{idx}] {img_path.name}: segmentation failed ({exc})")
            seg_vis = None

        # --- matting ---
        mat_vis = None
        if matter_ok:
            try:
                matte = matter.run(rgb)
                mat_vis = matter.visualize(rgb, matte)
                _save_rgb(OUTPUT_DIR / f'{stem}_matting.jpg', mat_vis)
            except Exception as exc:
                print(f"  [{idx}] {img_path.name}: matting failed ({exc})")

        # --- per-image summary stack ---
        panels = [p for p in (lm_vis, crop_vis, seg_vis, mat_vis) if p is not None]
        if panels:
            target_w = max(p.shape[1] for p in panels)
            panels_resized = [_resize_to_w(p, target_w) for p in panels]
            summary = np.concatenate(panels_resized, axis=0)
            _save_rgb(OUTPUT_DIR / f'{stem}_summary.jpg', summary)

        print(f"  [{idx}] {img_path.name}: written")

    print(f"[preprocessing] done → {OUTPUT_DIR}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

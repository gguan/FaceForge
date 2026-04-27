"""Stitch per-model outputs into side-by-side comparison images.

After running per-model scripts (run_hrn_head.py, run_pixel3dmm.py,
run_mononphm.py …) on the same image set, this aggregator reads each
model's per-image ``viz.jpg`` (and falls back to ``source.jpg`` for the
input column) and writes one comparison strip per image, plus an
``index.html`` browsable summary.

Usage:
    python scripts/compare.py \
        --models hrn_head=output/hrn_head pixel3dmm=output/pixel3dmm \
        --out output/compare/slamdunk

    # Or auto-discover from output/<model>/<id>/:
    python scripts/compare.py \
        --root output \
        --models hrn_head pixel3dmm \
        --out output/compare/slamdunk

Each <model>=<dir> argument tells the script: the column labelled
``<model>`` reads its viz from ``<dir>/<image_id>/viz.jpg``.

Output layout:
    <out>/<image_id>.jpg     — [source | model1 viz | model2 viz | ...]
    <out>/index.html         — clickable thumbnail grid
    <out>/summary.csv        — image_id × model presence/absence
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import cv2
import numpy as np


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    new_w = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _label_band(text: str, width: int, band_h: int = 30) -> np.ndarray:
    """White band with centred text — used to label each model column."""
    band = np.full((band_h, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, (width - tw) // 2)
    y = (band_h + th) // 2
    cv2.putText(band, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return band


def _load_viz(model_dir: Path, image_id: str) -> tuple[np.ndarray | None, str]:
    """Return ``(rgb image | None, status string)``.

    Tries ``viz.jpg`` first, falls back to ``overlay.png`` and finally
    ``source.jpg``. Status reflects which file was loaded.
    """
    for fname in ('viz.jpg', 'overlay.png', 'source.jpg'):
        p = model_dir / image_id / fname
        if p.exists():
            bgr = cv2.imread(str(p))
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fname
    return None, 'missing'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True,
                    help='space-separated list of <name>=<dir> pairs, '
                         'or just <name> when --root is given')
    ap.add_argument('--root', type=Path, default=None,
                    help='if set, treat each --models entry as <name> and '
                         'read from <root>/<name>')
    ap.add_argument('--out', type=Path, required=True,
                    help='where to write the compare strips + report')
    ap.add_argument('--source-from', type=str, default=None,
                    help='which model dir to pull source.jpg from '
                         '(default: first model). Useful when a model failed '
                         'to even preprocess and source.jpg is missing.')
    ap.add_argument('--row-height', type=int, default=512,
                    help='target row height (px) for the comparison strip')
    args = ap.parse_args()

    models: list[tuple[str, Path]] = []
    for spec in args.models:
        if '=' in spec:
            name, _, path = spec.partition('=')
            models.append((name, Path(path)))
        elif args.root is not None:
            models.append((spec, args.root / spec))
        else:
            ap.error(
                f'invalid --models entry {spec!r}: use <name>=<dir> '
                f'or pass --root'
            )

    # Discover all image_ids by union across model dirs.
    image_ids: set[str] = set()
    for name, mdir in models:
        if not mdir.exists():
            print(f'  WARN: {mdir} does not exist (skipping {name!r})')
            continue
        for sub in mdir.iterdir():
            if sub.is_dir() and not sub.name.startswith('_'):
                image_ids.add(sub.name)
    image_ids_sorted = sorted(image_ids)

    if not image_ids_sorted:
        print('no per-image dirs found across the given model dirs')
        return

    args.out.mkdir(parents=True, exist_ok=True)

    source_source = args.source_from or models[0][0]
    print(f'comparing {len(models)} model(s) on {len(image_ids_sorted)} image(s)')
    print(f'  source.jpg pulled from: {source_source}')
    print(f'  output: {args.out}')

    # Build summary CSV header
    summary_rows: list[dict] = []

    for image_id in image_ids_sorted:
        # Source column.
        source_dir = next((d for n, d in models if n == source_source), None)
        if source_dir is None:
            print(f'  [{image_id}] source provider {source_source!r} not in models')
            continue
        source_path = source_dir / image_id / 'source.jpg'
        source_rgb = None
        if source_path.exists():
            bgr = cv2.imread(str(source_path))
            if bgr is not None:
                source_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Per-model viz columns.
        cols: list[tuple[str, np.ndarray | None, str]] = [('source', source_rgb, 'source.jpg')]
        row = {'image_id': image_id}
        for name, mdir in models:
            viz, status = _load_viz(mdir, image_id)
            cols.append((name, viz, status))
            row[name] = status

        # Drop entirely-missing rows.
        if all(c[1] is None for c in cols):
            print(f'  [{image_id}] all columns missing — skipping')
            continue

        # Resize each column to target row height; fill missing with grey.
        target_h = args.row_height
        resized = []
        labels = []
        for name, img, status in cols:
            if img is None:
                ph = np.full((target_h, target_h, 3), 200, dtype=np.uint8)
                cv2.putText(ph, 'missing', (target_h // 4, target_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2,
                            cv2.LINE_AA)
                resized.append(ph)
            else:
                resized.append(_resize_to_height(img, target_h))
            labels.append(name)

        # Stitch: label band over each col, then concat horizontally.
        labelled_cols = []
        for name, panel in zip(labels, resized):
            band = _label_band(name, panel.shape[1])
            labelled_cols.append(np.concatenate([band, panel], axis=0))
        strip = np.concatenate(labelled_cols, axis=1)

        out_path = args.out / f'{image_id}.jpg'
        cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        summary_rows.append(row)
        print(f'  [{image_id}] wrote {out_path.name}  '
              f'(' + ', '.join(f'{n}={s}' for n, _, s in cols) + ')')

    # CSV summary.
    if summary_rows:
        keys = ['image_id'] + [n for n, _ in models]
        with open(args.out / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)

    # HTML report — one row per image, click-through to full-size jpg.
    html_path = args.out / 'index.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8">'
                f'<title>Compare — {args.out.name}</title>'
                '<style>body{font-family:system-ui;background:#222;color:#ddd;'
                'margin:0;padding:16px}h1{font-size:18px;margin:0 0 16px}'
                '.row{display:block;margin:8px 0;border:1px solid #444;'
                'background:#333}.row img{width:100%;display:block}'
                '.row .id{padding:4px 8px;font-size:13px;color:#aaa}</style></head>'
                f'<body><h1>{html.escape(args.out.name)} — {len(summary_rows)} image(s)</h1>')
        for row in summary_rows:
            iid = row['image_id']
            f.write(f'<a class="row" href="{iid}.jpg">'
                    f'<div class="id">{html.escape(iid)}</div>'
                    f'<img src="{iid}.jpg" loading="lazy"></a>')
        f.write('</body></html>')
    print(f'\nwrote {html_path}  (open in a browser to scroll through results)')


if __name__ == '__main__':
    main()

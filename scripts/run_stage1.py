#!/usr/bin/env python3
"""
Quick-test script for the Stage 1 pipeline.

Usage:
    python scripts/run_stage1.py --images path/to/face.jpg
    python scripts/run_stage1.py --images img1.jpg img2.jpg img3.jpg --device cuda:0 --debug
    python scripts/run_stage1.py --image-dir assets/tom/
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FaceForge Stage 1: image → FLAME parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--images', nargs='+', metavar='PATH',
        help='Input face image path(s).',
    )
    group.add_argument(
        '--image-dir', metavar='DIR',
        help='Directory of input images.',
    )
    parser.add_argument(
        '--output', default='output', metavar='DIR',
        help='Output directory.',
    )
    parser.add_argument(
        '--subject', default='test', metavar='NAME',
        help='Subject name (used as output subdirectory).',
    )
    parser.add_argument(
        '--device', default='cpu', metavar='DEVICE',
        help='PyTorch device: "cpu", "cuda:0", etc.',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Save per-step debug images to <output>/<subject>/stage1/',
    )
    parser.add_argument(
        '--data-dir', default=None, metavar='DIR',
        help='Data directory (default: <project_root>/data).',
    )
    parser.add_argument(
        '--aggregation', choices=['median', 'mean'], default=None,
        help='Shape aggregation method when multiple images are supplied (overrides config.yaml).',
    )
    parser.add_argument(
        '--config', default=None, metavar='PATH',
        help='Path to config.yaml (default: project root config.yaml).',
    )
    return parser.parse_args()


def _resolve_data_dir(data_dir_arg: str | None) -> str:
    if data_dir_arg:
        return data_dir_arg
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(project_root, 'data')


def _expand_paths(paths: list[str]) -> list[str]:
    """Expand directories to individual image file paths."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    result.append(os.path.join(p, f))
        else:
            result.append(p)
    if not result:
        print('[ERROR] No image files found in the given paths.', file=sys.stderr)
        sys.exit(1)
    return result


def _load_images(paths: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Load images and return (images_rgb, paths)."""
    images = []
    valid_paths = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'[WARN] Could not read image, skipping: {path}', file=sys.stderr)
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        valid_paths.append(path)
        print(f'  Loaded: {path}  ({img.shape[1]}×{img.shape[0]})')
    if not images:
        print('[ERROR] No valid images loaded.', file=sys.stderr)
        sys.exit(1)
    return images, valid_paths


def _print_output(result) -> None:
    print('\n── Stage1Output ─────────────────────────────────────────')
    fields = [
        ('shape',          'FLAME shape code'),
        ('expression',     'Expression coefficients'),
        ('head_pose',      'Head pose (rot-vec)'),
        ('jaw_pose',       'Jaw pose (rot-vec)'),
        ('texture',        'Texture coefficients'),
        ('lighting',       'SH lighting'),
        ('arcface_feat',   'ArcFace identity feature'),
        ('aligned_image',  'Aligned image tensor'),
        ('face_mask',      'Face mask tensor'),
        ('lmks_68',        '68-pt landmarks'),
        ('lmks_dense',     '478-pt dense landmarks'),
        ('lmks_eyes',      'Eye landmarks'),
        ('focal_length',   'Focal length'),
        ('principal_point','Principal point'),
    ]
    for attr, label in fields:
        t = getattr(result, attr)
        print(f'  {label:<28} {tuple(t.shape)}  dtype={t.dtype}')

    print()
    print(f'  shape_code range:  [{result.shape.min():.3f}, {result.shape.max():.3f}]')
    print(f'  focal_length:       {result.focal_length.item():.4f}')
    print(f'  aligned_image range: [{result.aligned_image.min():.3f}, {result.aligned_image.max():.3f}]')
    print('─────────────────────────────────────────────────────────')


def main():
    args = parse_args()
    data_dir = _resolve_data_dir(args.data_dir)

    print('FaceForge — Stage 1')
    print(f'  device:     {args.device}')
    print(f'  images:     {args.images or args.image_dir}')
    print(f'  output:     {args.output}')
    print(f'  debug:      {args.debug}')
    print(f'  data_dir:   {data_dir}')
    print()

    # ── expand & load images ────────────────────────────────────────────────
    print('Loading images...')
    raw_paths = args.images if args.images else [args.image_dir]
    image_paths = _expand_paths(raw_paths)
    images, image_paths = _load_images(image_paths)
    print(f'  Total images: {len(images)}')

    # ── build config ─────────────────────────────────────────────────────────
    from faceforge.stage1 import Stage1Config, Stage1Pipeline
    from faceforge._config_loader import load_stage1_overrides

    # CLI overrides: only apply if explicitly set by user
    cli_overrides = {
        'mica_model_path': os.path.join(data_dir, 'pretrained', 'mica.tar'),
        'mediapipe_model_path': os.path.join(data_dir, 'pretrained', 'mediapipe', 'face_landmarker.task'),
        'flame_model_path': os.path.join(data_dir, 'pretrained', 'FLAME2020', 'generic_model.pkl'),
        'flame_masks_path': os.path.join(data_dir, 'pretrained', 'FLAME2020', 'FLAME_masks.pkl'),
        'device': args.device,
        'output_dir': args.output,
        'save_debug': args.debug,
        **(({'aggregation_method': args.aggregation}) if args.aggregation else {}),
    }

    config = Stage1Config(**{**load_stage1_overrides(args.config), **cli_overrides})

    # ── initialise pipeline ──────────────────────────────────────────────────
    print('\nInitialising models (first run downloads InsightFace weights)...')
    t0 = time.perf_counter()
    pipeline = Stage1Pipeline(config)
    print(f'  Init time: {time.perf_counter() - t0:.1f}s')

    # ── run pipeline ─────────────────────────────────────────────────────────
    print('\nRunning Stage 1...')
    t0 = time.perf_counter()

    results = []
    summaries = []
    img_names = [f'{i+1:03d}' for i in range(len(images))]

    for img, img_name, path in zip(images, img_names, image_paths):
        print(f'\n  [{img_name}] {os.path.basename(path)}')
        try:
            r, summary = pipeline.run_single(
                img, subject_name=args.subject, image_name=img_name,
            )
            results.append(r)
            if summary is not None:
                summaries.append(summary)
            _print_output(r)
        except ValueError as e:
            print(f'  [WARN] Skipping {img_name}: {e}')

    if not results:
        print('[ERROR] No images processed successfully.', file=sys.stderr)
        sys.exit(1)

    result = results[0]
    if len(results) > 1:
        result, _ = pipeline.run_multi(
            images, subject_name=args.subject, image_names=img_names,
        )
        print('\n── Aggregated result ──')
        _print_output(result)

    elapsed = time.perf_counter() - t0
    print(f'\n  Total inference time: {elapsed:.2f}s')

    if args.debug:
        debug_dir = os.path.join(args.output, args.subject, 'stage1')
        print(f'\nDebug output saved to: {debug_dir}')

        # Concatenate all per-image summaries into one grid
        if len(summaries) > 1:
            from faceforge.stage1.visualization import save_summary_grid
            grid_path = os.path.join(debug_dir, 'summary_grid.png')
            save_summary_grid(summaries, grid_path)
            print(f'  Summary grid: {grid_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()

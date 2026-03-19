#!/usr/bin/env python3
"""
Quick-test script for the Stage 1 pipeline.

Usage:
    python scripts/run_stage1.py --image path/to/face.jpg
    python scripts/run_stage1.py --image path/to/face.jpg --device cuda:0 --debug
    python scripts/run_stage1.py --image img1.jpg img2.jpg img3.jpg  # multi-image
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run FaceForge Stage 1: image → FLAME parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--image', nargs='+', required=True, metavar='PATH',
        help='Input face image(s). Provide multiple images for shape aggregation.',
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
        '--aggregation', choices=['median', 'mean'], default='median',
        help='Shape aggregation method when multiple images are supplied.',
    )
    return parser.parse_args()


def _resolve_data_dir(data_dir_arg: str | None) -> str:
    if data_dir_arg:
        return data_dir_arg
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(project_root, 'data')


def _load_images(paths: list[str]) -> list[np.ndarray]:
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'[ERROR] Could not read image: {path}', file=sys.stderr)
            sys.exit(1)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f'  Loaded: {path}  ({img.shape[1]}×{img.shape[0]})')
    return images


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
    print(f'  images:     {args.image}')
    print(f'  output:     {args.output}')
    print(f'  debug:      {args.debug}')
    print(f'  data_dir:   {data_dir}')
    print()

    # ── load images ──────────────────────────────────────────────────────────
    print('Loading images...')
    images = _load_images(args.image)

    # ── build config ─────────────────────────────────────────────────────────
    from faceforge.stage1 import Stage1Config, Stage1Pipeline

    config = Stage1Config(
        mica_model_path=os.path.join(data_dir, 'pretrained', 'mica.tar'),
        mediapipe_model_path=os.path.join(data_dir, 'pretrained', 'mediapipe', 'face_landmarker.task'),
        flame_model_path=os.path.join(data_dir, 'pretrained', 'FLAME2020', 'generic_model.pkl'),
        flame_masks_path=os.path.join(data_dir, 'pretrained', 'FLAME2020', 'FLAME_masks.pkl'),
        device=args.device,
        output_dir=args.output,
        save_debug=args.debug,
        aggregation_method=args.aggregation,
    )

    # ── initialise pipeline ──────────────────────────────────────────────────
    print('\nInitialising models (first run downloads InsightFace weights)...')
    t0 = time.perf_counter()
    pipeline = Stage1Pipeline(config)
    print(f'  Init time: {time.perf_counter() - t0:.1f}s')

    # ── run pipeline ─────────────────────────────────────────────────────────
    print('\nRunning Stage 1...')
    t0 = time.perf_counter()
    if len(images) == 1:
        result = pipeline.run_single(images[0], subject_name=args.subject)
    else:
        result = pipeline.run_multi(images, subject_name=args.subject)
    elapsed = time.perf_counter() - t0
    print(f'  Inference time: {elapsed:.2f}s')

    # ── print results ────────────────────────────────────────────────────────
    _print_output(result)

    if args.debug:
        debug_dir = os.path.join(args.output, args.subject, 'stage1')
        print(f'\nDebug output saved to: {debug_dir}')

    print('\nDone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run Stage 2: Dense constraint optimization.

Usage:
    python scripts/run_stage2.py --images face1.jpg face2.jpg --device cuda:0 --debug
    python scripts/run_stage2.py --image-dir assets/tom/ --device cuda:0
    python scripts/run_stage2.py --images face.jpg --use-prdl
    python scripts/run_stage2.py --images face.jpg --pipeline p3m --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Dense constraint optimization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images', nargs='+', help='Input image paths')
    group.add_argument('--image-dir', type=str, help='Directory of input images')
    parser.add_argument('--subject', type=str, default='default', help='Subject name')
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    parser.add_argument('--debug', action='store_true', help='Save debug visualizations')
    parser.add_argument('--use-prdl', action='store_true', help='Use PRDL instead of baseline')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (default: project root config.yaml)')
    parser.add_argument('--pipeline', choices=['stage2', 'p3m'], default='stage2',
                        help='Pipeline to run (default: stage2). '
                             'p3m = faithful pixel3dmm reimplementation as baseline.')
    args = parser.parse_args()

    # Collect image paths
    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        )

    if not image_paths:
        print('No images found.')
        sys.exit(1)

    print(f'Found {len(image_paths)} images')

    # Load images
    images_rgb = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f'Warning: cannot read {p}, skipping')
            continue
        images_rgb.append(img[:, :, ::-1].copy())  # BGR → RGB

    # Load project config
    from faceforge._config_loader import load_stage1_overrides, load_stage2_overrides
    s1_overrides = load_stage1_overrides(args.config)
    s2_overrides = load_stage2_overrides(args.config)

    # Stage 1
    from faceforge.stage1 import Stage1Pipeline, Stage1Config
    s1_config = Stage1Config(
        **{
            **s1_overrides,
            **(({'device': args.device}) if args.device else {}),
            **(({'output_dir': args.output_dir}) if args.output_dir else {}),
            'save_debug': args.debug,
        }
    )
    s1 = Stage1Pipeline(s1_config)

    print('Running Stage 1...')
    s1_outputs = []
    for i, img in enumerate(images_rgb):
        name = image_paths[i].stem if i < len(image_paths) else f'img_{i}'
        try:
            result, _ = s1.run_single(img, args.subject, image_name=name)
            s1_outputs.append(result)
            print(f'  Stage 1 [{i+1}/{len(images_rgb)}]: {name}')
        except ValueError as e:
            print(f'  [WARN] Skipping {i+1:03d} ({name}): {e}')

    if not s1_outputs:
        print('No images processed successfully in Stage 1. Exiting.')
        sys.exit(1)

    # Build Stage2Config (shared by both pipelines for asset paths)
    from faceforge.stage2 import Stage2Config

    s2_config = Stage2Config(
        **{
            **s2_overrides,
            **(({'device': args.device}) if args.device else {}),
            **(({'output_dir': args.output_dir}) if args.output_dir else {}),
            'save_debug': args.debug,
            'use_prdl': args.use_prdl,
        }
    )

    # Pass MICA model for identity loss (stage2 only; p3m doesn't use it)
    mica_model = getattr(s1, 'mica', None)
    mica_inner = getattr(mica_model, 'mica', None) if mica_model else None

    # --- Pipeline factory ---
    output_dir = args.output_dir or 'output'
    if args.pipeline == 'p3m':
        from faceforge.stageP3M import P3MPipeline
        from faceforge.stageP3M.visualization import P3MVisualizer
        visualizer = P3MVisualizer(output_dir, args.subject) if args.debug else None
        pipeline_label = 'P3M (pixel3dmm baseline)'
        print(f'\nRunning {pipeline_label}...')
        s2 = P3MPipeline.from_stage2_config(s2_config, visualizer=visualizer)
    else:
        from faceforge.stage2.pipeline import Stage2Pipeline
        from faceforge.stage2.visualization import Stage2Visualizer
        visualizer = Stage2Visualizer(output_dir, args.subject) if args.debug else None
        pipeline_label = 'Stage 2'
        print(f'\nRunning {pipeline_label}...')
        s2 = Stage2Pipeline(s2_config, mica_model=mica_inner, visualizer=visualizer)

    s2_output = s2.run(s1_outputs)

    print(f'{pipeline_label} complete!')
    print(f'  Output shape:        {s2_output.shape.shape}')
    print(f'  Vertices:            {s2_output.vertices.shape}')
    print(f'  Loss history stages: {list(s2_output.loss_history.keys())}')

    # Save final debug artifacts (same interface for both pipelines)
    if args.debug and visualizer is not None:
        visualizer.save_loss_curves(s2_output.loss_history)
        visualizer.save_mesh_obj(s2_output.vertices, s2.flame.faces, 'mesh_optimized.obj')
        visualizer.save_result(s2_output)
        subdir = 'stageP3M' if args.pipeline == 'p3m' else 'stage2'
        print(f'  Debug output:      {output_dir}/{args.subject}/{subdir}/')


if __name__ == '__main__':
    main()

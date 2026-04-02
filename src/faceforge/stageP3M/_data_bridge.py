"""Bridge between FaceForge Stage1Output and pixel3dmm Tracker.

Writes Stage1 outputs to disk in pixel3dmm's expected preprocessed format,
builds tracker config, and converts tracker results back to Stage2Output.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from faceforge._paths import PROJECT_ROOT
from faceforge.stage1.data_types import Stage1Output
from faceforge.stage2.data_types import Stage2Output, PerImageParams

# pixel3dmm's WFLW→iBUG68 mapping (tracker.py L117-120)
WFLW_2_iBUG68 = np.array([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33,
    34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
])


def _lmk68_to_98(lmks_68: np.ndarray) -> np.ndarray:
    """Convert 68-point dlib landmarks to 98-point WFLW format.

    Places each dlib landmark at the correct WFLW index.
    Unoccupied positions (including iris 96,97) are zero,
    which pixel3dmm auto-masks via ``lmk_mask = ~(lmk.sum(2) == 0)``.

    Args:
        lmks_68: [68, 2] pixel coordinates

    Returns:
        lmks_98: [98, 2] in WFLW layout
    """
    lmks_98 = np.zeros((98, 2), dtype=np.float32)
    for ibug_idx, wflw_idx in enumerate(WFLW_2_iBUG68):
        lmks_98[wflw_idx] = lmks_68[ibug_idx]
    return lmks_98


def write_preprocessed(
    stage1_outputs: list[Stage1Output],
    pixel3dmm_inference,
    tmp_dir: str,
    video_name: str,
    render_size: int = 256,
) -> None:
    """Write Stage1 outputs to disk in pixel3dmm's expected format.

    Creates directory structure:
        {tmp_dir}/{video_name}/
            cropped/{i:05d}.png
            seg_og/{i:05d}.png
            PIPnet_landmarks/{i:05d}.npy
            mica/0/identity.npy
            p3dmm/normals/{i:05d}.png
            p3dmm/uv_map/{i:05d}.png
    """
    base = Path(tmp_dir) / video_name
    for subdir in ['cropped', 'seg_og', 'PIPnet_landmarks', 'mica/0',
                   'p3dmm/normals', 'p3dmm/uv_map']:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    # MICA shape: average all shapes, save once
    all_shapes = torch.stack([s.shape for s in stage1_outputs])  # [N, 1, 300]
    mean_shape = all_shapes.mean(dim=0).squeeze(0).cpu().numpy()  # [300]
    np.save(base / 'mica' / '0' / 'identity.npy', mean_shape)

    for i, s1out in enumerate(stage1_outputs):
        # --- RGB image ---
        img = s1out.aligned_image.detach().cpu()  # [1, 3, H, W]
        if img.max() > 1.0:
            img = img / 255.0
        img_np = (img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        # Resize to render_size
        img_resized = cv2.resize(img_np, (render_size, render_size))
        cv2.imwrite(
            str(base / 'cropped' / f'{i:05d}.png'),
            cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR),
        )

        # --- Segmentation mask ---
        seg = s1out.face_mask.detach().cpu().squeeze().numpy().astype(np.uint8)
        seg_resized = cv2.resize(seg, (render_size, render_size),
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(base / 'seg_og' / f'{i:05d}.png'), seg_resized)

        # --- Landmarks (98-point WFLW format) ---
        # pixel3dmm read_data() does: lms = np.load(...) * config.size
        # So we save as fractional [0, 1] coordinates
        lmks_68 = s1out.lmks_68[0].detach().cpu().numpy()  # [68, 2] pixel coords
        lmks_98 = _lmk68_to_98(lmks_68)
        # Source landmarks are at render_size=512 (Stage1 align resolution).
        # pixel3dmm loads them and scales by config.size (render_size).
        # So save as fraction of 512, then at load time: fraction * render_size.
        # But Stage1 aligned at 512, and tracker might use 256.
        # Safest: save as fraction (÷512), then pixel3dmm multiplies by its render_size.
        # This means landmark positions are scaled proportionally.
        src_size = s1out.aligned_image.shape[-1]  # typically 512
        lmks_98_frac = lmks_98 / float(src_size)
        np.save(base / 'PIPnet_landmarks' / f'{i:05d}.npy', lmks_98_frac)

        # --- UV and Normal maps via Pixel3DMM inference ---
        uv_map, normal_map = pixel3dmm_inference.predict(
            s1out.aligned_image, s1out.face_mask)

        # UV: [1, 2, H, W] in [0, 1] → save as 3-channel uint8 PNG
        uv = uv_map[0].detach().cpu()  # [2, H, W]
        # Pad to 3 channels (pixel3dmm loads 3-channel PNG, uses first 2)
        uv_3ch = torch.cat([uv, torch.zeros_like(uv[:1])], dim=0)  # [3, H, W]
        uv_np = (uv_3ch.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        uv_resized = cv2.resize(uv_np, (render_size, render_size))
        cv2.imwrite(
            str(base / 'p3dmm' / 'uv_map' / f'{i:05d}.png'),
            cv2.cvtColor(uv_resized, cv2.COLOR_RGB2BGR),
        )

        # Normal: [1, 3, H, W] in [-1, 1] → save as uint8 PNG in [0, 1]
        # pixel3dmm's read_data does: (pixel/255 - 0.5)*2 to get back to [-1,1]
        # So we save as: (normal + 1) / 2 * 255
        n = normal_map[0].detach().cpu()  # [3, H, W]
        n_01 = ((n + 1) / 2).clamp(0, 1)
        n_np = (n_01.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        n_resized = cv2.resize(n_np, (render_size, render_size))
        cv2.imwrite(
            str(base / 'p3dmm' / 'normals' / f'{i:05d}.png'),
            cv2.cvtColor(n_resized, cv2.COLOR_RGB2BGR),
        )


def build_tracker_config(
    video_name: str,
    preprocessed_dir: str,
    output_dir: str,
    n_frames: int,
    render_size: int,
    code_base: str,
    overrides: dict | None = None,
):
    """Build OmegaConf config for pixel3dmm Tracker.

    Returns:
        cfg: OmegaConf config matching tracking.yaml schema
    """
    from omegaconf import OmegaConf

    # Load pixel3dmm's default config
    yaml_path = Path(code_base) / 'configs' / 'tracking.yaml'
    base_cfg = OmegaConf.load(str(yaml_path))

    # Apply our overrides
    our_overrides = OmegaConf.create({
        'video_name': video_name,
        'size': render_size,
        'image_size': [render_size, render_size],
        'batch_size': min(16, n_frames),
        'start_frame': 0,
        'is_discontinuous': True,  # independent images, not video
        'save_meshes': True,
        'delete_preprocessing': False,
        'output_folder': output_dir,
        'iters': 500,  # pixel3dmm default is 200 for video, 500 for images
        'early_exit': False,
    })

    if overrides:
        user_overrides = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(base_cfg, our_overrides, user_overrides)
    else:
        cfg = OmegaConf.merge(base_cfg, our_overrides)

    return cfg


def read_tracker_results(
    tracker,
    stage1_outputs: list[Stage1Output],
    device: torch.device,
) -> Stage2Output:
    """Extract optimized parameters from Tracker and convert to Stage2Output.

    Args:
        tracker: pixel3dmm Tracker instance (after .run())
        stage1_outputs: original Stage1 inputs (for texture/lighting)
        device: torch device

    Returns:
        Stage2Output
    """
    from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix

    N = len(stage1_outputs)

    # Shape from tracker
    shape = tracker.shape.detach().cpu()  # [1, 300]

    # Focal length
    focal_length = tracker.focal_length.detach().cpu()  # [1, 1]

    # Run FLAME with neutral expression for canonical mesh
    with torch.no_grad():
        verts_neutral, lmks_68, _, _, _ = tracker.flame(
            cameras=torch.eye(3, device=tracker.device).unsqueeze(0),
            shape_params=tracker.shape,
        )

    # Per-image params from intermediate storage
    per_image_params = []
    for i in range(N):
        # Expression
        exp_i = tracker.intermediate_exprs[i].detach().cpu()  # [1, 100]

        # Head rotation: 6D → axis-angle
        R_6d = tracker.intermediate_Rs[i].detach()  # [1, 6]
        R_mat = rotation_6d_to_matrix(R_6d)  # [1, 3, 3]
        head_pose = _rotation_matrix_to_axis_angle(R_mat).cpu()  # [1, 3]

        # Translation
        t_i = tracker.intermediate_ts[i].detach().cpu()  # [1, 3]

        # Jaw: 6D → axis-angle
        jaw_6d = tracker.intermediate_jaws[i].detach()  # [1, 6]
        jaw_mat = rotation_6d_to_matrix(jaw_6d)
        jaw_pose = _rotation_matrix_to_axis_angle(jaw_mat).cpu()

        # Lighting from Stage1 (tracker doesn't optimize lighting)
        lighting = stage1_outputs[i].lighting.detach().cpu()

        per_image_params.append(PerImageParams(
            expression=exp_i,
            R_6d=R_6d.cpu(),
            jaw_6d=jaw_6d.cpu(),
            translation=t_i,
            lighting=lighting,
        ))

    return Stage2Output(
        shape=shape,
        texture=stage1_outputs[0].texture.detach().cpu(),
        focal_length=focal_length,
        vertices=verts_neutral.detach().cpu(),
        landmarks_3d=lmks_68.detach().cpu(),
        per_image_params=per_image_params,
        loss_history={},  # pixel3dmm tracker doesn't expose structured loss history
    )


def _rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert [B, 3, 3] rotation matrix to [B, 3] axis-angle."""
    # Using Rodrigues formula inverse
    B = R.shape[0]
    result = torch.zeros(B, 3, device=R.device, dtype=R.dtype)

    for i in range(B):
        r = R[i]
        theta = torch.acos(((r.trace() - 1) / 2).clamp(-1, 1))
        if theta.abs() < 1e-6:
            continue  # identity rotation
        axis = torch.stack([
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ]) / (2 * torch.sin(theta) + 1e-8)
        result[i] = axis * theta

    return result

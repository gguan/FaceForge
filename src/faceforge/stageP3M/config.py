"""P3M baseline config — thin wrapper around pixel3dmm's tracking.yaml.

All optimization hyperparameters (lr, loss weights, steps, etc.) are owned
by pixel3dmm's tracking.yaml.  This config only holds FaceForge-specific
settings like asset paths and rendering resolution.
"""
from dataclasses import dataclass, field
from faceforge._paths import default_device


@dataclass
class P3MConfig:
    # === Model paths (same as Stage2Config) ===
    flame_model_path: str = 'data/pretrained/FLAME2020/generic_model.pkl'
    flame_masks_path: str = 'data/pretrained/FLAME2020/FLAME_masks.pkl'
    flame_lmk_embedding_path: str = 'data/pretrained/FLAME2020/landmark_embedding.npy'
    flame_uv_coords_path: str = 'data/pretrained/FLAME2020/flame_uv_coords.npy'
    flame_uv_valid_verts_path: str = 'data/pretrained/FLAME2020/flame_uv_valid_verts.npy'
    pixel3dmm_uv_ckpt: str = 'data/pretrained/uv.ckpt'
    pixel3dmm_normal_ckpt: str = 'data/pretrained/normals.ckpt'
    pixel3dmm_code_base: str = 'submodules/pixel3dmm'

    # === Rendering ===
    render_size: int = 256   # match pixel3dmm default (256)
    use_opengl: bool = False

    # === Multi-image ===
    batch_size: int = 16     # joint phase batch size (tracking.yaml default)

    # === Escape hatch: arbitrary overrides to tracking.yaml ===
    tracker_overrides: dict = field(default_factory=dict)

    # === Output ===
    output_dir: str = 'output'
    save_debug: bool = True
    device: str = field(default_factory=default_device)

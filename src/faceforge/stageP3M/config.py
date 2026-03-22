"""P3M baseline config — matches pixel3dmm default hyperparameters.

All values reference pixel3dmm/src/pixel3dmm/tracking/tracker.py.
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
    render_size: int = 512
    use_opengl: bool = False

    # === Optimization steps (pixel3dmm reference values) ===
    # Ref: tracker.py run() — optimize_camera(steps=500, is_first_frame=True)
    camera_steps: int = 500          # optimize_camera() first-frame steps
    camera_steps_extra: int = 10     # optimize_camera() subsequent-frame steps
    # Ref: tracker.py optimize_color() — iters = config.iters = 500
    iters: int = 500                 # optimize_color() per-frame steps
    # Ref: tracker.py run() — config.iters = config.global_iters = 5000
    global_iters: int = 5000        # joint phase steps

    # === Learning rates (pixel3dmm tracker.py defaults) ===
    # Ref: tracker.py optimize_camera() param groups
    lr_t: float = 0.001
    lr_R: float = 0.005
    lr_focal: float = 0.02
    lr_pp: float = 0.0001
    # Ref: tracker.py clone_params_keyframes_all (not shown, use reasonable defaults)
    lr_exp: float = 0.005
    lr_jaw: float = 0.005
    lr_shape: float = 0.002         # used in joint Adam optimizer (optimizer_id)

    # === Loss weights ===
    # Ref: tracker.py optimize_camera() L652: * 3000
    w_lmks_camera: float = 3000.0
    # Ref: tracker.py optimize_camera() L656: * 1000
    w_uv_camera: float = 1000.0
    # Ref: tracker.py opt_pre() L876: * w_lmks * lmk_scale * 5
    w_lmks: float = 1000.0          # eye landmark weight (× 5 applied in code)
    # Ref: tracker.py opt_post() — * config.uv_map_super
    uv_map_super: float = 2000.0
    # Ref: tracker.py opt_post() — * config.sil_super
    sil_super: float = 500.0
    # Ref: tracker.py opt_post() — * config.normal_super
    normal_super: float = 1000.0

    # === Regularization (pixel3dmm defaults) ===
    # Ref: tracker.py opt_pre() — * config.w_shape / w_shape_general / w_exp / w_jaw
    w_shape: float = 0.2            # shape-to-MICA regularization
    w_shape_general: float = 0.05   # shape-to-zero regularization
    w_exp: float = 0.05             # expression regularization
    w_jaw: float = 0.01             # jaw regularization (distance from identity)

    # === UV loss thresholds ===
    # Ref: stage2/config.py (same defaults)
    delta_uv_coarse: float = 0.00005
    dist_uv_coarse: float = 15.0
    delta_uv_fine: float = 0.00002
    dist_uv_fine: float = 8.0

    # === Normal loss ===
    # Ref: tracker.py config.delta_n / config.normal_mask_ksize
    delta_n: float = 0.33
    normal_mask_ksize: int = 13

    # === Multi-image ===
    batch_size: int = 4             # joint phase batch size

    # === Early stopping (per-frame optimize_color only) ===
    # Ref: tracker.py optimize_color() — stagnant_window_size=10
    early_stopping_window: int = 10
    # Ref: tracker.py optimize_color() — np.mean(past_k_steps) < config.early_stopping_delta
    early_stopping_delta: float = 1e-4

    # === Output ===
    output_dir: str = 'output'
    save_debug: bool = True
    use_nan_guard: bool = True
    device: str = field(default_factory=default_device)

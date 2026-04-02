"""Stage 2 configuration.

Default values follow pixel3dmm multi-image tracking recommendations.
All fields can be overridden via config.yaml (see project root) or
by passing keyword arguments directly.

Usage::

    from faceforge._config_loader import load_stage2_overrides
    from faceforge.stage2.config import Stage2Config

    cfg = Stage2Config(**{**load_stage2_overrides(), 'device': 'cuda:0'})
"""

from dataclasses import dataclass, field

from faceforge._paths import default_device


@dataclass
class Stage2Config:
    # === 模型路径 ===
    flame_model_path: str = 'data/pretrained/FLAME2020/generic_model.pkl'
    flame_masks_path: str = 'data/pretrained/FLAME2020/FLAME_masks.pkl'
    flame_lmk_embedding_path: str = 'data/pretrained/FLAME2020/landmark_embedding.npy'
    flame_uv_coords_path: str = 'data/pretrained/FLAME2020/flame_uv_coords.npy'
    flame_uv_valid_verts_path: str = 'data/pretrained/FLAME2020/flame_uv_valid_verts.npy'
    pixel3dmm_uv_ckpt: str = 'data/pretrained/uv.ckpt'
    pixel3dmm_normal_ckpt: str = 'data/pretrained/normals.ckpt'
    pixel3dmm_code_base: str = 'submodules/pixel3dmm'

    # === 渲染 ===
    render_size: int = 512
    use_opengl: bool = False

    # === 5 阶段步数 ===
    # pixel3dmm 用单一循环（config.iters=500），所有 loss 从第 0 步全开，
    # 无 coarse_lmk / coarse_uv 分段。只有 LR 在 50%/75%/90% 处衰减。
    # 我们保留分段结构方便 debug，但 medium（= pixel3dmm 主循环）设为 500。
    # coarse_lmk / coarse_uv 是我们自己加的预热阶段，pixel3dmm 中不存在。
    coarse_lmk_steps: int = 1000  # pixel3dmm: 500 but we need more for focal convergence
    coarse_uv_steps: int = 500    # optimize_camera 后半段：UV only（landmark 关闭），仅 R/t/focal
    medium_steps: int = 500       # optimize_color(is_joint=False)：所有 loss 全开
    fine_pca_steps: int = 0
    fine_detail_steps: int = 0
    enable_fine_detail: bool = False

    # === 学习率 (ref: pixel3dmm defaults) ===
    lr_shape: float = 0.002       # pixel3dmm: lr_id = 0.002
    lr_expression: float = 0.005  # pixel3dmm: lr_exp = 0.005
    lr_R: float = 0.005           # pixel3dmm: lr_R = 0.005 (head rotation 6D)
    lr_jaw: float = 0.005         # pixel3dmm: lr_jaw = 0.005
    lr_translation: float = 0.001 # pixel3dmm: lr_t = 0.001
    lr_focal: float = 0.02        # pixel3dmm: lr_f = 0.02 (coarse only, frozen in medium)
    lr_focal_medium: float = 0.002 # reduced lr for medium stage (pixel3dmm freezes it)
    lr_texture: float = 0.005
    lr_lighting: float = 0.01
    lr_texture_disp: float = 0.001

    # === Loss 权重 (ref: pixel3dmm tracking.yaml + tracker.py) ===
    # Landmark weights (individual components, not combined)
    # pixel3dmm: w_lmks=1000, eye×5=5000, lid×500=500000, iris×50=50000
    w_lmks: float = 1000.0            # eye contour base weight
    w_lmks_lid: float = 1000.0        # eye closure base weight
    w_lmks_iris: float = 1000.0       # iris base weight (×50 in loss)
    w_lmk_camera: float = 3000.0      # optimize_camera landmark weight (all 68)
    # Dense losses
    w_pixel3dmm_uv: float = 2000.0
    w_normal: float = 1000.0
    w_sil: float = 500.0
    w_contour: float = 0.0
    w_region_weight: float = 0.0
    w_photometric: float = 0.0
    w_identity: float = 0.0
    # Mirror symmetry (pixel3dmm: 5000)
    w_mirror: float = 5000.0
    # Eye/neck regularization (pixel3dmm tracker.py L914-919)
    w_eye_sym: float = 0.1       # eye symmetry: (right-left)^2
    w_eye_reg: float = 0.01      # each eye → identity
    w_neck: float = 0.1          # neck → identity
    # Normal loss mode
    normal_l2: bool = False       # pixel3dmm: configurable L1/L2

    # === PRDL (experimental) ===
    use_prdl: bool = False
    w_prdl: float = 0.0

    # === 正则化 (ref: pixel3dmm tracking.yaml) ===
    w_reg_shape_to_mica: float = 0.2
    w_reg_shape_to_zero: float = 0.05
    w_reg_expression: float = 0.05
    w_reg_jaw: float = 0.01
    w_reg_sh_mono: float = 0.0
    w_reg_tex_tv: float = 0.0

    # === UV 对应阈值 ===
    uv_delta_coarse: float = 0.00005
    uv_dist_coarse: float = 15.0
    uv_delta_fine: float = 0.00002
    uv_dist_fine: float = 8.0

    # === Normal loss ===
    normal_delta_threshold: float = 0.33
    normal_eye_dilate_kernel: int = 13

    # === Landmark ===
    # (weights are now in w_lmks, w_lmks_lid, w_lmks_iris above)

    # === LR 衰减 ===
    lr_decay_milestones: tuple = (0.5, 0.75, 0.9)

    # === 梯度安全 ===
    use_nan_guard: bool = True
    focal_length_min: float = 2.0
    focal_length_max: float = 6.0

    # === 遮挡过滤 ===
    use_occlusion_filter: bool = True
    occlusion_depth_eps: float = 0.01

    # === 早停 ===
    use_early_stopping: bool = True
    early_stopping_window: int = 10
    early_stopping_delta: float = 1e-5

    # === 多图优化 ===
    # pixel3dmm: per-frame iters=500, joint global_iters=5000
    multi_image_batch_size: int = 4
    sequential_coarse_steps: int = 100
    sequential_medium_steps: int = 200
    global_iters: int = 5000         # joint 阶段总步数 (= pixel3dmm config.global_iters)
    global_lr_scale: float = 0.1

    # === pixel3dmm 兼容 ===
    # landmark/loss 现在默认匹配 pixel3dmm，无需额外开关

    # === 输出 ===
    output_dir: str = 'output'
    save_debug: bool = True
    device: str = field(default_factory=default_device)

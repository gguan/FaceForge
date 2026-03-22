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
    coarse_lmk_steps: int = 0    # pixel3dmm 无此阶段，设 0 跳过
    coarse_uv_steps: int = 0     # pixel3dmm 无此阶段，设 0 跳过
    medium_steps: int = 500      # 对应 pixel3dmm config.iters = 500
    fine_pca_steps: int = 0
    fine_detail_steps: int = 0
    enable_fine_detail: bool = False

    # === 学习率 (ref: pixel3dmm defaults) ===
    lr_shape: float = 0.002
    lr_expression: float = 0.005
    lr_head_pose: float = 0.005
    lr_jaw_pose: float = 0.005
    lr_translation: float = 0.001
    lr_focal: float = 0.02
    lr_texture: float = 0.005
    lr_lighting: float = 0.01
    lr_texture_disp: float = 0.001

    # === Loss 权重 (ref: pixel3dmm tracker.py) ===
    w_landmark: float = 5000.0
    w_pixel3dmm_uv: float = 2000.0
    w_normal: float = 1000.0
    w_sil: float = 500.0
    w_contour: float = 0.0
    w_region_weight: float = 0.0
    w_photometric: float = 0.0
    w_identity: float = 0.0

    # === PRDL (experimental) ===
    use_prdl: bool = False
    w_prdl: float = 0.0

    # === 正则化 (ref: pixel3dmm defaults) ===
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
    nose_landmark_weight: float = 3.0

    # === LR 衰减 ===
    lr_decay_milestones: tuple = (0.5, 0.75, 0.9)

    # === 梯度安全 ===
    use_nan_guard: bool = True
    focal_length_min: float = 1.5
    focal_length_max: float = 8.0

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

    # === pixel3dmm 兼容模式 ===
    # True: 仅使用眼部关键点权重（与 pixel3dmm tracker 完全一致）
    # False: 使用完整 68 点权重（鼻子额外加权）
    pixel3dmm_compat: bool = False

    # === 输出 ===
    output_dir: str = 'output'
    save_debug: bool = True
    device: str = field(default_factory=default_device)

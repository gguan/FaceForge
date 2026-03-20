from dataclasses import dataclass, field


@dataclass
class Stage2Config:
    # === pixel3dmm 兼容模式 ===
    # True: 仅使用 pixel3dmm 的 loss/权重/LR/步数，屏蔽其他来源的 loss
    # False: 使用完整的多源 loss 组合
    pixel3dmm_compat: bool = True

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
    coarse_lmk_steps: int = 100
    coarse_uv_steps: int = 100
    medium_steps: int = 500
    fine_pca_steps: int = 150
    fine_detail_steps: int = 150
    enable_fine_detail: bool = False

    # === 学习率 ===
    lr_shape: float = 0.005
    lr_expression: float = 0.01
    lr_head_pose: float = 0.01
    lr_jaw_pose: float = 0.01
    lr_focal: float = 0.02
    lr_translation: float = 0.001
    lr_texture: float = 0.005
    lr_lighting: float = 0.01
    lr_texture_disp: float = 0.001

    # === Loss 权重 ===
    w_landmark: float = 1.0
    w_pixel3dmm_uv: float = 50.0
    w_normal: float = 10.0
    w_sil: float = 1.0
    w_contour: float = 5.0
    w_region_weight: float = 2.0
    w_photometric: float = 2.0
    w_identity: float = 0.5

    # === PRDL A/B 切换 ===
    use_prdl: bool = False
    w_prdl: float = 5.0

    # === 正则化 ===
    w_reg_shape_to_mica: float = 1e-4
    w_reg_shape_to_zero: float = 1e-6
    w_reg_expression: float = 1e-3
    w_reg_jaw: float = 1e-3
    w_reg_sh_mono: float = 1.0
    w_reg_tex_tv: float = 100.0

    # === UV 对应阈值 ===
    uv_delta_coarse: float = 0.00005
    uv_dist_coarse: float = 15.0
    uv_delta_fine: float = 0.00002
    uv_dist_fine: float = 8.0

    # === 遮挡过滤 ===
    use_occlusion_filter: bool = True
    occlusion_depth_eps: float = 0.01

    # === Normal loss ===
    normal_delta_threshold: float = 0.33   # ref: pixel3dmm default
    normal_eye_dilate_kernel: int = 13     # ref: pixel3dmm Gaussian blur kernel

    # === Landmark ===
    nose_landmark_weight: float = 3.0

    # === LR 衰减 ===
    lr_decay_milestones: tuple = (0.5, 0.75, 0.9)

    # === 梯度安全 ===
    use_nan_guard: bool = True
    focal_length_min: float = 1.5
    focal_length_max: float = 8.0

    # === 早停 ===
    use_early_stopping: bool = True
    early_stopping_window: int = 10
    early_stopping_delta: float = 1e-5

    # === 多图优化 ===
    multi_image_batch_size: int = 4
    sequential_coarse_steps: int = 100
    sequential_medium_steps: int = 200
    global_lr_scale: float = 0.1

    # === 输出 ===
    output_dir: str = 'output'
    save_debug: bool = True
    device: str = 'cuda:0'

    def __post_init__(self):
        if self.pixel3dmm_compat:
            self._apply_pixel3dmm_compat()

    def _apply_pixel3dmm_compat(self):
        """Override all weights/LR/steps to match pixel3dmm tracker defaults.

        Ref: pixel3dmm tracking/tracker.py + configs/tracking.yaml
        """
        # --- Loss weights (pixel3dmm per-frame online stage) ---
        self.w_landmark = 5000.0        # lmk_eye2 (only eyes 36-48)
        self.w_pixel3dmm_uv = 2000.0    # uv correspondence
        self.w_normal = 1000.0          # L1 normal
        self.w_sil = 500.0              # silhouette

        # Disable non-pixel3dmm losses
        self.w_contour = 0.0            # HRN
        self.w_region_weight = 0.0      # HiFace
        self.w_photometric = 0.0        # flame-head-tracker
        self.w_identity = 0.0           # ArcFace
        self.use_prdl = False
        self.w_prdl = 0.0               # 3DDFA-V3

        # --- Regularization (pixel3dmm defaults) ---
        self.w_reg_shape_to_mica = 0.2
        self.w_reg_shape_to_zero = 0.05
        self.w_reg_expression = 0.05
        self.w_reg_jaw = 0.01
        self.w_reg_sh_mono = 0.0        # pixel3dmm doesn't use
        self.w_reg_tex_tv = 0.0         # pixel3dmm doesn't use

        # --- Learning rates (pixel3dmm defaults) ---
        self.lr_shape = 0.002
        self.lr_expression = 0.005
        self.lr_head_pose = 0.005
        self.lr_jaw_pose = 0.005
        # lr_translation = 0.001 (already matches)
        # lr_focal = 0.02 (already matches)

        # --- Steps (pixel3dmm: 500 camera + 200 online) ---
        self.coarse_lmk_steps = 500
        self.coarse_uv_steps = 100
        self.medium_steps = 200
        self.fine_pca_steps = 0         # skip (pixel3dmm has no separate fine stage)
        self.fine_detail_steps = 0      # skip

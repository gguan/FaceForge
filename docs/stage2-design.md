# Stage 2: 稠密约束优化 — 详细设计文档

## 1. 目标

通过可微渲染 + 多源稠密约束, 对 Stage 1 输出的 FLAME 参数进行迭代优化, 突破 MICA 回归精度上限。

**输入**: N 张同一人照片的 `Stage1Output` 列表 (单图时 N=1)

**输出**: 优化后的 shape (身份) + 中性表情正面 mesh → 送 Stage 3

### 1.1 参数分类

**共享参数** (所有图使用同一份, 编码身份):

| 参数 | 维度 | 初始值来源 |
|------|------|-----------|
| shape | 300 | MICA (多图取 median) |
| texture | 50 | DECA |
| focal_length | 1 | Stage 1 相机参数 |

**每图独立参数** (编码该图的状态):

| 参数 | 维度 | 初始值来源 |
|------|------|-----------|
| expression[i] | 100 | DECA(前50) + 零(后50) |
| head_pose[i] | 3 | DECA |
| jaw_pose[i] | 3 | DECA |
| translation[i] | 3 | [0, 0, z] |
| lighting[i] | 9×3 SH | DECA |

> 单图时退化为全部参数属于同一张图, 无共享/独立之分。

## 2. 总体流程

```
N 张照片 → Stage 1 × N → N 个 Stage1Output
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: Pixel3DMM 预处理 (每图独立, 一次性)            │
│  aligned_image[i] + face_mask[i]                      │
│  → DINOv2 → UV map[i] [512,512,2]                    │
│  →        → Normal map[i] [512,512,3]                 │
└─────────┬────────────────────────────────────────────┘
          │  PreprocessedData[0..N-1]
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 2: 参数初始化                                    │
│  shared.shape = median(Stage1Output[i].shape_code)    │
│  per_image[i] = {exp, pose, jaw, trans, light}        │
│  + 每图独立 KNN 建立 UV 对应关系                        │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 3: Sequential 优化 (多图时, 逐图, 冻结 shape)    │
│  for each image i:                                    │
│    Coarse-LMK (100 steps) + Coarse-UV (100 steps)    │
│    + Medium (200 steps, 无 identity)                  │
│  目的: 让每图 pose/exp 基本对齐                         │
│  (单图时跳过此步)                                      │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: Global 联合优化 (核心)                        │
│  解冻 shape, 每步随机采样 min(N,4) 张图                 │
│  shape 梯度 = 多图梯度聚合                              │
│                                                      │
│  ┌─ Coarse-LMK (100 steps)                           │
│  │  优化: pose + jaw + translation + focal            │
│  │  Loss: L_landmark (L2)                            │
│  │                                                   │
│  ├─ Coarse-UV (100 steps)                            │
│  │  Loss: + L_pixel3dmm_uv (宽松阈值)                │
│  │                                                   │
│  ├─ Medium (500 steps)                               │
│  │  优化: + shape + expression                       │
│  │  Loss: + L_normal + L_contour/L_prdl + L_sil      │
│  │        + L_reg (landmark 切换为 L1)                │
│  │  UV 对应收紧阈值                                   │
│  │                                                   │
│  ├─ Fine-PCA (150 steps)                             │
│  │  优化: + texture code + lighting                  │
│  │  Loss: + L_photometric + L_identity               │
│  │                                                   │
│  └─ Fine-Detail (150 steps, 可选)                     │
│     优化: + texture displacement map                  │
│     Loss: 同上, 更低 LR                               │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: 最终输出                                      │
│  shape(优化后) + expression=zeros + pose=zeros         │
│  → FLAME forward → 中性表情正面 mesh [5023, 3]         │
│  → Stage 3 (HRN displacement)                        │
└──────────────────────────────────────────────────────┘
```

## 3. Pixel3DMM 预处理

### 3.1 概述

**论文**: Pixel3DMM (Kirschstein et al., CVPR 2025)

逐图运行 (不在优化循环内), 为每张图预测稠密 UV + Normal map。**唯一需要 import 子模块源码的地方** (其余均在 stage2 内重新实现)。

**权重**: `data/pretrained/uv.ckpt` + `data/pretrained/normals.ckpt`

### 3.2 推理流程

```python
# 每张图独立推理
for i in range(N):
    # 1. 图像 [0,1], reshape [1, 1, 512, 512, 3]
    image = stage1_outputs[i].aligned_image / 255.0

    # 2. 分割 mask: skin + brow + eye + nose + lip, 排除 mouth interior
    mask = (seg == 2) | ((seg > 3) & (seg < 14)) & ~(seg == 11)

    # 3. 原图 + 水平翻转各推理一次, 取平均 (增强稳定性)
    pred = (model.net(batch_orig) + flip_back(model.net(batch_flip))) / 2

    # 4. UV: [-1,1] → [0,1]
    uv_map[i] = torch.clamp((pred_uv + 1) / 2, 0, 1)  # [1, 2, 512, 512]

    # 5. Normal: 归一化 + 坐标约定变换 [x, 1-z, 1-y]
    normal_map[i] = F.normalize(pred_normal)  # [1, 3, 512, 512]
```

### 3.3 环境设置

运行时 patch `pixel3dmm.env_paths` 避免 .env 文件依赖:
```python
import pixel3dmm.env_paths as env_paths
env_paths.CODE_BASE = Path('submodules/pixel3dmm')
```

## 4. 独立 FLAME 模型

### 4.1 加载方式

从 `data/pretrained/FLAME2020/generic_model.pkl` 直接加载, 参考 pixel3dmm `tracking/flame/FLAME.py` 的 lbs 算法, 在 stage2 内重新实现。

**注册 Buffer**:

| Buffer | 形状 | 含义 |
|--------|------|------|
| `v_template` | [5023, 3] | 平均脸模板 |
| `shapedirs` | [5023, 3, 400] | PCA 基底 (前300=shape, 后100=expression) |
| `posedirs` | [36, 5023×3] | 姿态 blend shapes |
| `J_regressor` | [5, 5023] | 关节回归矩阵 |
| `parents` | [5] | 骨骼父节点 |
| `lbs_weights` | [5023, 5] | LBS 蒙皮权重 |
| `f` | [13776, 3] | 面片索引 |

### 4.2 可微前向传播

```python
def forward(shape_params, expression_params, head_pose, jaw_pose):
    # 1. Shape + Expression blend
    betas = cat([shape_params, expression_params], dim=1)  # [B, 400]
    v_shaped = v_template + einsum('bk,vnk->bvn', betas, shapedirs)

    # 2. Joint locations
    J = einsum('jv,bvn->bjn', J_regressor, v_shaped)

    # 3. Pose → rotation matrices (rodrigues, 可微)
    full_pose = cat([head_pose, jaw_pose, zeros(B,3), zeros(B,3)])
    rot_mats = rodrigues(full_pose.reshape(-1, 3)).reshape(B, 4, 3, 3)

    # 4. Pose blend shapes
    pose_offsets = (rot_mats.flatten() - eye(3).flatten()) @ posedirs
    v_posed = v_shaped + pose_offsets.reshape(B, 5023, 3)

    # 5. LBS skinning
    T = einsum('vj,bjmn->bvmn', lbs_weights, transforms)
    vertices = einsum('bvij,bvj->bvi', T[:,:,:3,:3], v_posed) + T[:,:,:3,3]
    return vertices  # [B, 5023, 3]
```

### 4.3 Landmark 提取

**68 点** (从 `data/pretrained/FLAME2020/landmark_embedding.npy`):
- Static 51pt (idx 17-67): barycentric interpolation, 不随姿态变化
- Dynamic 17pt (idx 0-16): 下颌轮廓, 根据 head_pose yaw 插值选取

**10 点眼球** (硬编码 FLAME 顶点索引):
```python
R_EYE = [4597, 4543, 4511, 4479, 4575]
L_EYE = [4051, 3997, 3965, 3933, 4020]
```

### 4.4 区域 Mask

从 `data/pretrained/FLAME2020/FLAME_masks.pkl` 加载 dict: `face, nose, left_eye_region, right_eye_region, lips, forehead, ...` 每个 key 对应一组顶点索引。

### 4.5 UV 坐标

从 pixel3dmm 或 FLAME 数据中加载 `flame_uv_coords` [5023, 2]。

**关键: V 轴翻转**:
```python
flame_uv_coords[..., 1] = (flame_uv_coords[..., 1] * -1) + 1
```
不翻转会导致 KNN 对应上下颠倒。

### 4.6 顶点法线

```python
def get_vertex_normals(vertices, faces):
    v0, v1, v2 = vertices[:, faces[:, 0]], vertices[:, faces[:, 1]], vertices[:, faces[:, 2]]
    face_normals = cross(v1 - v0, v2 - v0)
    vertex_normals = zeros(B, 5023, 3)
    vertex_normals.scatter_add_(1, faces_expanded, face_normals_repeated)
    return F.normalize(vertex_normals, dim=-1)
```

## 5. 相机模型

### 5.1 参数化

| 参数 | 初始值 | 说明 |
|------|--------|------|
| `focal_length` | Stage 1 焦距 (≈2000/512 归一化) | 共享, 可优化 |
| `translation[i]` | [0, 0, z] | 每图独立 |
| `head_pose[i]` | DECA | 同时作为 FLAME 旋转和外参 |

### 5.2 投影

```python
# 内参 K
fx = fy = focal_length * image_size
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

# 外参 RT
R = rodrigues(head_pose)
RT = [[R, translation], [0, 0, 0, 1]]

# 投影
points_cam = R @ points_3d.T + translation
points_2d = K @ points_cam; points_2d = points_2d[:2] / points_2d[2]
```

### 5.3 安全约束

```python
# FOV clamp: 防止退化投影 (参考 flame-head-tracker)
focal_length = torch.clamp(focal_length, min=1.5, max=8.0)  # 归一化范围
```

## 6. nvdiffrast 可微渲染器

### 6.1 渲染管线

参考 VHAP `render_nvdiffrast.py` 逻辑, 在 stage2 内重新实现:

```python
def render(self, vertices, faces, K, RT, vertex_colors=None, sh_coefficients=None):
    # 1. World → Camera → Clip
    verts_clip = project_to_clip(vertices, K, RT)  # [B, V, 4]

    # 2. 光栅化
    rast_out, rast_db = dr.rasterize(self.glctx, verts_clip, faces, (H, W))

    # 3. 属性插值 (可微)
    normals_interp, _ = dr.interpolate(vertex_normals, rast_out, faces)
    positions_interp, _ = dr.interpolate(verts_cam, rast_out, faces)

    # 4. SH 光照
    shading = compute_sh_shading(normals_interp, sh_coefficients)
    image = colors_interp * shading

    # 5. 前景 mask + 抗锯齿
    mask = (rast_out[..., 3:] > 0).float()
    image = dr.antialias(image, rast_out, verts_clip, faces)

    # 6. Y 轴翻转 (OpenGL → 图像坐标)
    image, mask, normals_interp = [x.flip(1) for x in [image, mask, normals_interp]]

    return {
        'image': image,          # [B, H, W, 3]
        'mask': mask,            # [B, H, W, 1]
        'normal': normals_interp,# [B, H, W, 3]
        'depth': depth,          # [B, H, W, 1] ← 用于遮挡过滤
    }
```

### 6.2 SH 光照

DECA 的 9 系数 (2 阶) SH, 每通道独立, 共 9×3=27 参数:

```python
def compute_sh_shading(normals, sh_coeffs):
    # sh_coeffs: [B, 9, 3]
    # 9 个 SH 基函数: [1, y, z, x, xy, yz, 3z²-1, xz, x²-y²]
    sh_basis = compute_sh_basis(normals)  # [B, H, W, 9]
    return einsum('bhwk,bkc->bhwc', sh_basis, sh_coeffs)
```

### 6.3 遮挡过滤 (from Pixel3DMM)

每步优化时, 用深度 buffer 过滤不可见顶点:

```python
def compute_visibility_mask(depth_buffer, projected_vertices, proj_verts_2d):
    """返回 [B, V] bool mask, True = 可见"""
    # 在投影位置采样深度图
    pixel_depth = grid_sample(depth_buffer, proj_verts_2d)
    proj_z = projected_vertices[:, :, 2]
    # 如果顶点深度 > 像素深度 + eps, 则被遮挡
    return pixel_depth < (proj_z + 1e-2)
```

**用途**: 传入 UV loss、landmark loss、contour loss, 仅计算可见顶点。侧脸时避免不可见的鼻翼/耳部顶点产生错误梯度。

### 6.4 梯度流

- **通过属性插值**: loss → interpolated attr → vertex attr → FLAME params (L_photometric, L_normal)
- **通过 antialias**: loss → edge pixels → clip positions → vertices (silhouette 边缘)
- **通过投影**: loss → 2D position → 3D vertex → FLAME params (L_landmark, L_uv, 不经过光栅化)

## 7. 损失函数

### 7.1 L_landmark — Landmark 对齐

**来源**: flame-head-tracker 拟合策略

68pt FLAME landmarks + 10pt 眼球, 分区域加权。

**区域权重**:

| 区域 | 索引 | 权重 | 来源 |
|------|------|------|------|
| 鼻梁+鼻翼 | 27-35 | **3.0** | VHAP (最稳定点, 10x) + 折中 |
| 眼轮廓 | 36-47 | 3.0 | flame-head-tracker |
| 嘴轮廓 | 49-67 | 2.0 | flame-head-tracker |
| 面部其余 | 17-26 | 1.5 | flame-head-tracker |
| 下颌线 | 3-13 | 0.75 | flame-head-tracker |
| 下颌两端 | 0-2, 14-16 | 0.5 | flame-head-tracker |
| 眼球 | 10pt | 3.0 | flame-head-tracker |

**距离度量** (按阶段切换):
- Coarse-LMK / Coarse-UV: **L2** (对大偏差施加更大梯度, 加速收敛)
- Medium / Fine: **L1** (对离群点更鲁棒, 避免过拟合噪声 landmark)

```python
def landmark_loss(pred_68, pred_eyes, target_68, target_eyes,
                  weights, use_l2=False, visibility_mask=None):
    # 归一化到 [-1, 1]
    pred_norm = pred_68 / (image_size / 2) - 1
    target_norm = target_68 / (image_size / 2) - 1

    if visibility_mask is not None:  # 遮挡过滤
        weights = weights * visibility_mask

    diff = pred_norm - target_norm
    if use_l2:
        loss_68 = (weights.unsqueeze(-1) * diff.pow(2)).mean()
    else:
        loss_68 = (weights.unsqueeze(-1) * diff.abs()).mean()

    loss_eyes = 3.0 * (pred_eyes_norm - target_eyes_norm).abs().mean()
    return loss_68 + loss_eyes
```

### 7.2 L_pixel3dmm_uv — 稠密 UV 对应

**来源**: Pixel3DMM (CVPR 2025)

核心稠密约束, 将约束点从 68 个 landmark 扩展到数千个顶点。

**两阶段 KNN 精化** (from Pixel3DMM):

```python
class Pixel3DMMUVLoss:
    def __init__(self, flame_uv_coords, valid_verts_mask):
        self.flame_uv = flame_uv_coords[valid_verts_mask]  # 排除嘴内/眼内
        self.valid_indices = valid_verts_mask

    def compute_correspondences(self, uv_prediction, delta_uv, dist_uv):
        """KNN 建立对应, delta_uv/dist_uv 控制宽松度"""
        pixels_uv = uv_prediction.permute(0,2,3,1).reshape(-1, 2)
        dists, indices = knn(self.flame_uv, pixels_uv, k=1)
        # 过滤: KNN距离 < delta_uv 且 2D距离 < dist_uv
        valid = (dists < delta_uv)
        pixel_coords = idx_to_xy(indices)
        self.target_coords = pixel_coords[valid]
        self.valid_verts = valid

    def tighten_thresholds(self, delta_fine, dist_fine):
        """Medium 阶段开始时收紧阈值"""
        self.compute_correspondences(..., delta_fine, dist_fine)

    def compute_loss(self, projected_vertices_2d, visibility_mask=None):
        pred = projected_vertices_2d[:, self.valid_indices]
        pred = pred[:, self.valid_verts]
        if visibility_mask is not None:  # 遮挡过滤
            vis = visibility_mask[:, self.valid_indices][:, self.valid_verts]
            pred, target = pred[vis], self.target_coords[vis]
        # L1, 除以 image_size 归一化 (from Pixel3DMM)
        return (pred - target).abs().mean() / image_size
```

**阈值**:
- Coarse: `delta_uv=0.00005, dist_uv=15px` (宽松)
- Medium+: `delta_uv=0.00002, dist_uv=8px` (严格)

### 7.3 L_normal — 法线一致性

**来源**: Pixel3DMM (CVPR 2025)

```python
def normal_loss(rendered_normals, predicted_normals, face_mask,
                eye_mask=None, delta_n=0.15):
    pred_n = predicted_normals.permute(0, 2, 3, 1)

    # 1. Cosine distance
    l_map = rendered_normals - pred_n

    # 2. 异常值过滤 (差异过大的像素视为无效)
    valid = (l_map.abs().sum(dim=-1) / 3) < delta_n

    # 3. 眼部膨胀排除 (眼球法线不连续)
    if eye_mask is not None:
        dilated_eye = gaussian_blur(eye_mask, kernel_size=5)
        valid = valid & (dilated_eye < 0.5)

    mask = face_mask & valid
    loss = (1 - (rendered_normals * pred_n).sum(dim=-1)) * mask
    return loss.sum() / (mask.sum() + 1e-8)
```

### 7.4 L_contour — 轮廓约束 (基线方案)

**来源**: HRN (CVPR 2023) contour_aware_loss

**不是 Chamfer Distance**, 而是 boundary-checking: 只惩罚投影顶点"越界", 不对已在边界内的顶点施力。

```python
def contour_loss(projected_vertices, flame_masks, face_mask_2d):
    """
    对 face_mask_2d 的每行 y:
    - left[y] = 该行最左面部像素 x
    - right[y] = 该行最右面部像素 x

    对每个边界区域顶点:
    - dist = (left - x) * (right - x) / width²
    - dist < 0 → 在边界内, 无惩罚
    - dist > 0 → 越界, 惩罚 |dist|
    """
    boundary_verts = projected_vertices[:, flame_masks['boundary']]
    left, right = extract_row_boundaries(face_mask_2d)

    verts_y = boundary_verts[:, :, 1].long()
    verts_x = boundary_verts[:, :, 0]
    l = left[verts_y]; r = right[verts_y]

    dist = (l - verts_x) * (r - verts_x) / (width ** 2)
    loss = torch.clamp(dist, min=0).mean()  # 只惩罚越界
    return loss
```

**条件**: 仅 `config.use_prdl=False`

### 7.5 L_sil — 轮廓 Silhouette 损失

**来源**: VHAP + Pixel3DMM (两者均使用)

当前设计之前遗漏, 与 L_contour 互补。

```python
def silhouette_loss(rendered_mask, target_mask):
    """
    rendered_mask: [B, H, W, 1] 渲染前景 mask (from nvdiffrast)
    target_mask: [B, H, W] BiSeNet 面部分割 mask
    L1 距离, 惩罚渲染轮廓与图像轮廓的不一致
    """
    return (rendered_mask.squeeze(-1) - target_mask.float()).abs().mean()
```

**激活阶段**: Medium+ (需要基本对齐后才有效)

### 7.6 L_region_weight — 区域加权 (基线方案)

**来源**: HiFace (ICCV 2023) 启发

不是独立 loss, 而是**修饰器**: 为 L_photometric 提供逐像素区域权重。

```python
region_weights = {
    'nose': 3.0, 'eye_region': 2.5, 'lips': 2.0,
    'jaw': 1.5, 'cheeks': 1.0, 'forehead': 0.5,
}
# FLAME 顶点 → 权重 → nvdiffrast 光栅化 → 2D 权重图 [B, H, W]
```

**条件**: 仅 `config.use_prdl=False`

### 7.7 L_prdl — Part Re-projection Distance Loss (备选方案)

**来源**: 3DDFA-V3 (Wang et al., CVPR 2024), [arXiv:2312.00311](https://arxiv.org/abs/2312.00311)

统一替代 L_contour + L_region_weight + L_sil。

```python
class PRDLLoss:
    SEG_TO_FLAME = {
        'nose': [10], 'left_eye': [4], 'right_eye': [5],
        'lips': [12, 13], 'jaw': [1],  # skin 下半部
    }

    def compute(self, projected_vertices, face_segmentation, flame_masks):
        total = 0
        for part in self.SEG_TO_FLAME:
            target_pts = mask_to_points(segmentation[part])
            pred_pts = projected_vertices[:, flame_masks[part]]
            total += grid_statistical_distance(pred_pts, target_pts)
        return total
```

| 维度 | 基线 (contour + region + sil) | PRDL |
|------|------------------------------|------|
| 轮廓 | boundary-checking | 全区域分布匹配 |
| 区域加权 | 手动调权 | 按语义自然分区 |
| 梯度 | 依赖光栅化 | 纯几何投影, 更清晰 |

**条件**: 仅 `config.use_prdl=True`

### 7.8 L_photometric — 光度损失

**来源**: VHAP / flame-head-tracker

```python
def photometric_loss(rendered_image, target_image, mask, region_weight_map=None):
    diff = (rendered - target).abs()  # [B, H, W, 3]
    if region_weight_map is not None:
        diff = diff * region_weight_map.unsqueeze(-1)
    # 归一化: 除以 (mask面积 × 通道数), 使 loss 与面部大小无关
    C = 3
    loss = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * C + 1e-8)
    return loss
```

### 7.9 L_identity — 身份一致性

**来源**: MICA (ECCV 2022) ArcFace

```python
class IdentityLoss:
    def __init__(self, arcface_model):
        self.arcface = arcface_model  # frozen, no grad

    def compute(self, rendered_image, target_arcface_feat):
        # rendered → crop 112×112 → (x*255-127.5)/127.5 → arcface → L2 norm
        feat = F.normalize(self.arcface(face_crop), dim=-1)
        return 1 - F.cosine_similarity(feat, target_arcface_feat, dim=-1).mean()
```

**多图时**: 每图有独立 `arcface_feat[i]`, 渲染图与该图对应的 feat 比较。

> ArcFace frozen, 梯度通过 rendered_image → nvdiffrast → FLAME params。

### 7.10 L_reg — 参数正则化

**来源**: Pixel3DMM (shape→MICA) + HRN (SH 光照) + VHAP (纹理 TV)

```python
def regularization_loss(params, mica_init_shape, config):
    # 1. Shape 约束到 MICA 初始值 (不是零!)
    loss_shape = config.w_reg_shape_to_mica * (params.shape - mica_init_shape).pow(2).mean()
    # 辅助: 约束到零 (更低权重)
    loss_shape += config.w_reg_shape_to_zero * params.shape.pow(2).mean()

    # 2. Expression + Jaw 约束到零
    loss_exp = config.w_reg_expression * params.expression.pow(2).mean()
    loss_jaw = config.w_reg_jaw * params.jaw_pose.pow(2).mean()

    # 3. SH 光照单色正则 (防止不自然的彩色光照)
    sh_mean = params.lighting.mean(dim=-1, keepdim=True)
    loss_sh = config.w_reg_sh_mono * (params.lighting - sh_mean).pow(2).mean()

    # 4. 纹理 TV 正则 (防止尖刺纹理, 仅 Fine-Detail 阶段)
    if hasattr(params, 'texture_displacement'):
        td = params.texture_displacement
        loss_tv = config.w_reg_tex_tv * (
            (td[:,:,1:] - td[:,:,:-1]).pow(2).mean() +
            (td[:,1:,:] - td[:,:-1,:]).pow(2).mean()
        )

    return loss_shape + loss_exp + loss_jaw + loss_sh + loss_tv
```

## 8. 损失聚合与 A/B 切换

### 8.1 阶段调度

```python
class LossAggregator:
    def compute(self, stage, **kwargs):
        losses = {}

        # === Coarse-LMK ===
        if stage == 'coarse_lmk':
            losses['landmark'] = w_lmk * landmark_loss(..., use_l2=True)

        # === Coarse-UV ===
        elif stage == 'coarse_uv':
            losses['landmark'] = w_lmk * landmark_loss(..., use_l2=True)
            losses['uv'] = w_uv * uv_loss(...)

        # === Medium ===
        elif stage == 'medium':
            losses['landmark'] = w_lmk * landmark_loss(..., use_l2=False)  # 切换 L1
            losses['uv'] = w_uv * uv_loss(...)  # 收紧后的阈值
            losses['normal'] = w_normal * normal_loss(...)
            losses['sil'] = w_sil * silhouette_loss(...)
            losses['reg'] = regularization_loss(...)

            if self.config.use_prdl:
                losses['prdl'] = w_prdl * prdl_loss(...)
            else:
                losses['contour'] = w_contour * contour_loss(...)
                # region_weight 修饰 photometric, 此阶段暂无

        # === Fine-PCA / Fine-Detail ===
        elif stage in ('fine_pca', 'fine_detail'):
            # 继承 medium 的全部 loss
            losses['landmark'] = w_lmk * landmark_loss(..., use_l2=False)
            losses['uv'] = w_uv * uv_loss(...)
            losses['normal'] = w_normal * normal_loss(...)
            losses['sil'] = w_sil * silhouette_loss(...)
            losses['reg'] = regularization_loss(...)
            if self.config.use_prdl:
                losses['prdl'] = w_prdl * prdl_loss(...)
            else:
                losses['contour'] = w_contour * contour_loss(...)
            # 新增
            losses['photo'] = w_photo * photometric_loss(..., region_weight_map)
            losses['identity'] = w_identity * identity_loss(...)

        total = sum(losses.values())
        total = torch.nan_to_num(total, nan=0.0, posinf=1e5)  # 梯度安全
        return total, losses
```

### 8.2 权重表

| Loss | 权重 | 阶段 | 条件 |
|------|------|------|------|
| L_landmark | 1.0 | 全部 | 始终 |
| L_pixel3dmm_uv | 50.0 | coarse_uv+ | 始终 |
| L_normal | 10.0 | medium+ | 始终 |
| L_sil | 1.0 | medium+ | 始终 |
| L_contour | 5.0 | medium+ | use_prdl=False |
| L_region_weight | 2.0 | fine+ | use_prdl=False |
| L_prdl | 5.0 | medium+ | use_prdl=True |
| L_photometric | 2.0 | fine+ | 始终 |
| L_identity | 0.5 | fine+ | 始终 |
| L_reg_shape_to_mica | 1e-4 | medium+ | 始终 |
| L_reg_shape_to_zero | 1e-6 | medium+ | 始终 |
| L_reg_expression | 1e-3 | medium+ | 始终 |
| L_reg_jaw | 1e-3 | medium+ | 始终 |
| L_reg_sh_mono | 1.0 | fine+ | 始终 |
| L_reg_tex_tv | 100.0 | fine_detail | 始终 |

## 9. 五阶段 Coarse-to-Fine 优化

### 9.1 阶段总览

| 阶段 | 步数 | 优化参数 | 距离度量 | 关键事件 |
|------|------|---------|---------|---------|
| Coarse-LMK | 100 | pose, jaw, trans, focal | L2 | 仅 landmark 粗对齐 |
| Coarse-UV | 100 | pose, jaw, trans, focal | L2→L1 | 引入 UV (宽松阈值) |
| Medium | 500 | + shape, expression | L1 | UV 阈值收紧; +normal/contour/sil/reg |
| Fine-PCA | 150 | + texture code, lighting | L1 | +photometric/identity |
| Fine-Detail | 150 | + texture displacement | L1 | 高频纹理细节 (可选) |

### 9.2 学习率

```python
param_lr = {
    'shape':         0.005,
    'expression':    0.01,
    'head_pose':     0.01,
    'jaw_pose':      0.01,
    'focal_length':  0.02,
    'translation':   0.001,
    'texture':       0.005,
    'lighting':      0.01,
    'texture_disp':  0.001,  # Fine-Detail 专用, 更保守
}
```

### 9.3 阶段内动态 LR 衰减

每个阶段内部, 按进度分段衰减 (from Pixel3DMM + flame-head-tracker):

```python
def adjust_lr(optimizer, progress, param_groups):
    """
    progress: step / total_steps_in_this_stage
    大参数 (pose, translation, jaw): 衰减更激进
    小参数 (shape, expression): 衰减更温和
    """
    for group in param_groups:
        base_lr = group['initial_lr']
        if group['name'] in ('head_pose', 'jaw_pose', 'translation', 'focal_length'):
            if progress > 0.9:   scale = 0.01
            elif progress > 0.75: scale = 0.02
            elif progress > 0.5:  scale = 0.1
            else:                 scale = 1.0
        else:
            if progress > 0.9:   scale = 0.2
            elif progress > 0.75: scale = 0.5
            elif progress > 0.5:  scale = 0.5
            else:                 scale = 1.0
        group['lr'] = base_lr * scale
```

### 9.4 早停

```python
# 监控最近 window 步的 loss 变化, 低于阈值时提前结束当前阶段
if len(loss_history) >= 2 * window:
    recent = mean(loss_history[-window:])
    prev = mean(loss_history[-2*window:-window])
    if prev - recent < early_stopping_delta:
        break
```

### 9.5 多图联合优化循环

```python
for stage in ['coarse_lmk', 'coarse_uv', 'medium', 'fine_pca', 'fine_detail']:
    steps = config.get_steps(stage)
    shared_opt, per_image_opts = create_optimizers(stage, shared_params, per_image_params)

    # Medium 阶段开始时收紧 UV 阈值
    if stage == 'medium':
        for uv_loss in uv_losses:
            uv_loss.tighten_thresholds(config.uv_delta_fine, config.uv_dist_fine)

    for step in range(steps):
        # 1. 采样图片 (单图时 selected=[0])
        selected = random.sample(range(N), min(N, config.multi_image_batch_size))
        total_loss = 0

        for i in selected:
            # 2. FLAME forward: 共享 shape + 第 i 图的 exp/pose
            vertices, lmks_68, lmks_eyes = flame(
                shared_params.shape,
                per_image_params[i].expression,
                per_image_params[i].head_pose,
                per_image_params[i].jaw_pose)

            # 3. 渲染
            render_out = renderer.render(vertices, faces,
                K, build_RT(per_image_params[i]),
                sh_coefficients=per_image_params[i].lighting)

            # 4. 遮挡过滤
            visibility = compute_visibility_mask(
                render_out['depth'], projected_verts)

            # 5. 计算 loss
            loss_i, log_i = loss_aggregator.compute(
                stage=stage,
                visibility_mask=visibility,
                target_image=preprocessed[i].target_image,
                target_lmks_68=preprocessed[i].target_lmks_68,
                ...)
            total_loss += loss_i

        total_loss = total_loss / len(selected)

        # 6. 反向传播 + 更新
        total_loss.backward()
        shared_opt.step(); shared_opt.zero_grad()
        for i in selected:
            per_image_opts[i].step(); per_image_opts[i].zero_grad()

        # 7. LR 衰减
        adjust_lr(shared_opt, step / steps, ...)
        for i in selected:
            adjust_lr(per_image_opts[i], step / steps, ...)

        # 8. 安全 clamp
        with torch.no_grad():
            shared_params.focal_length.clamp_(1.5, 8.0)
```

## 10. 最终输出

```python
def extract_final_model(shared_params, flame):
    """联合优化完成后, 输出中性表情正面 mesh"""
    vertices, lmks, _ = flame(
        shape_params=shared_params.shape,
        expression_params=torch.zeros(1, 100),
        head_pose=torch.zeros(1, 3),
        jaw_pose=torch.zeros(1, 3),
    )
    return Stage2Output(
        shape=shared_params.shape,
        texture=shared_params.texture,
        focal_length=shared_params.focal_length,
        vertices=vertices,
        landmarks_3d=lmks,
        loss_history=loss_history,
    )
```

每图独立优化的 expression/pose/lighting 仅作为解耦身份与状态的工具, 不进入最终输出。

## 11. 调试输出

### 11.1 输出目录

```
output/{subject_name}/
├── stage2/
│   ├── 01_preprocessing/
│   │   ├── pixel3dmm_uv_{i}.png        # 每图 UV map
│   │   └── pixel3dmm_normals_{i}.png   # 每图 Normal map
│   ├── 02_optimization/
│   │   ├── progress_step_{n}.png       # 每 50 步: 最佳视角渲染叠加
│   │   └── loss_curves.png             # 各 loss 项曲线
│   ├── 03_result/
│   │   ├── flame_params_optimized.npz  # 优化后共享参数
│   │   ├── mesh_optimized.obj          # 中性表情 mesh
│   │   ├── render_final.png            # 最终渲染
│   │   ├── render_overlay.png          # 渲染叠加
│   │   └── before_after.png            # Stage 1 vs Stage 2
│   └── summary.png                     # input → S1 → S2 → overlay
```

### 11.2 关键检查项

| 环节 | 检查 | 文件 |
|------|------|------|
| Pixel3DMM | UV/Normal 与面部对齐 | `pixel3dmm_uv_*.png` |
| Coarse | mesh 粗略对齐到人脸 | `progress_step_200.png` |
| Medium | 鼻/下巴形状更贴合 | `progress_step_700.png` |
| Fine | 颜色匹配, overlay "像" | `render_overlay.png` |
| Loss | 各项持续下降, 无振荡 | `loss_curves.png` |

## 12. 与 Stage 1/3 的接口

### 12.1 Stage 1 → Stage 2

```python
# Pipeline 入口
class Stage2Pipeline:
    def run(self, stage1_outputs: list[Stage1Output],
            mica_model=None) -> Stage2Output:
        if len(stage1_outputs) == 1:
            return self._run_single(stage1_outputs[0])
        else:
            return self._run_multi(stage1_outputs)
```

消费 Stage1Output 字段:

| 字段 | 用途 |
|------|------|
| `shape_code` | SharedParams.shape 初始值 (多图取 median) |
| `expression, head_pose, jaw_pose` | PerImageParams 初始值 |
| `texture, lighting` | SharedParams / PerImageParams 初始值 |
| `arcface_feat` | L_identity target |
| `aligned_image` | Pixel3DMM 输入 + L_photometric target |
| `face_mask` | Pixel3DMM mask + loss masking |
| `lmks_68, lmks_eyes` | L_landmark target |
| `focal_length, principal_point` | 相机初始化 |

### 12.2 Stage 2 → Stage 3

```python
@dataclass
class Stage2Output:
    shape: Tensor           # [1, 300] 联合优化后
    texture: Tensor         # [1, 50]
    focal_length: Tensor    # [1, 1]
    vertices: Tensor        # [1, 5023, 3] 中性表情 mesh
    landmarks_3d: Tensor    # [1, 68, 3]
    loss_history: dict
```

Stage 3 (HRN displacement) 在 `vertices` 基础上叠加高频细节。

## 13. 数据类

```python
@dataclass
class PreprocessedData:
    """每图一份, 一次性预处理结果"""
    pixel3dmm_uv: torch.Tensor        # [1, 2, 512, 512]
    pixel3dmm_normals: torch.Tensor    # [1, 3, 512, 512]
    face_mask: torch.Tensor            # [1, 512, 512]
    target_image: torch.Tensor         # [1, 3, 512, 512]
    target_lmks_68: torch.Tensor       # [1, 68, 2]
    target_lmks_eyes: torch.Tensor     # [1, 10, 2]
    arcface_feat: torch.Tensor         # [1, 512]

@dataclass
class SharedParams:
    """所有图共享 (身份)"""
    shape: torch.Tensor          # [1, 300]
    texture: torch.Tensor        # [1, 50]
    focal_length: torch.Tensor   # [1, 1]

@dataclass
class PerImageParams:
    """每图独立 (状态)"""
    expression: torch.Tensor     # [1, 100]
    head_pose: torch.Tensor      # [1, 3]
    jaw_pose: torch.Tensor       # [1, 3]
    translation: torch.Tensor    # [1, 3]
    lighting: torch.Tensor       # [1, 9, 3]
```

## 14. 代码结构

```
src/faceforge/stage2/
├── __init__.py
├── config.py                    # Stage2Config (完整, 见下方)
├── data_types.py                # PreprocessedData, SharedParams, PerImageParams, Stage2Output
├── flame_model.py               # 独立 FLAME (lbs, landmark, region masks, UV coords)
├── camera.py                    # 内参/外参/投影/rodrigues
├── renderer.py                  # nvdiffrast 渲染 + 遮挡过滤
├── pixel3dmm_inference.py       # Pixel3DMM UV/Normal 推理
├── losses/
│   ├── __init__.py
│   ├── landmark.py              # L_landmark (78pt, 区域加权, L1/L2 切换)
│   ├── pixel3dmm_uv.py          # L_pixel3dmm_uv (两阶段 KNN + 遮挡过滤)
│   ├── normal.py                # L_normal (cosine + 异常值过滤 + 眼部排除)
│   ├── contour.py               # L_contour (HRN boundary-checking, 基线)
│   ├── silhouette.py            # L_sil (前景 mask L1)
│   ├── region_weight.py         # L_region_weight (光栅化权重图, 基线)
│   ├── prdl.py                  # L_prdl (3DDFA-V3 统计距离, 备选)
│   ├── photometric.py           # L_photometric (masked L1, 面积归一化)
│   ├── identity.py              # L_identity (ArcFace cosine sim)
│   ├── regularization.py        # L_reg (shape→MICA, SH 单色, 纹理 TV)
│   └── total.py                 # LossAggregator (5 阶段调度 + A/B 切换)
├── optimizer.py                 # 5 阶段优化 + 动态 LR 衰减 + 早停
├── pipeline.py                  # Stage2Pipeline (_run_single / _run_multi)
└── visualization.py             # 调试可视化
```

## 15. 完整 Stage2Config

```python
@dataclass
class Stage2Config:
    # === 模型路径 ===
    flame_model_path: str = 'data/pretrained/FLAME2020/generic_model.pkl'
    flame_masks_path: str = 'data/pretrained/FLAME2020/FLAME_masks.pkl'
    flame_lmk_embedding_path: str = 'data/pretrained/FLAME2020/landmark_embedding.npy'
    flame_uv_coords_path: str = 'data/pretrained/FLAME2020/flame_uv_coords.npy'
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
    enable_fine_detail: bool = False   # 可选, 需要 texture displacement

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
    normal_delta_threshold: float = 0.15
    normal_eye_dilate_kernel: int = 5

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
```

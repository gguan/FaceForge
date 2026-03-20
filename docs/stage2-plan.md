# Stage 2 开发计划

> 本文档是给 Claude Code 的开发指令。每个 Task 是一个独立可验证的开发单元。
> 技术细节参考 `docs/stage2-design.md`。

## 前置条件

- 远程 GPU 机器: CUDA + PyTorch
- 额外依赖: `pip install nvdiffrast pytorch_lightning timm einops`
- 预训练权重:
  - `data/pretrained/FLAME2020/generic_model.pkl`
  - `data/pretrained/FLAME2020/FLAME_masks.pkl`
  - `data/pretrained/uv.ckpt`, `data/pretrained/normals.ckpt`
- Landmark embedding: `submodules/MICA/data/FLAME2020/landmark_embedding.npy` (需复制到 `data/pretrained/FLAME2020/`)
- UV 资产: `submodules/pixel3dmm/assets/flame_uv_coords.npy` 和 `uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy`

## 代码位置

所有代码写在 `src/faceforge/stage2/` 下。创建 `src/faceforge/stage2/__init__.py`。

## 参考文件约定

复用子模块的算法逻辑, 在 stage2/ 中**重新实现** (不直接 import), 唯一例外: Pixel3DMM 网络加载。

---

## Task 1: 项目骨架 + 数据类 + 资产复制

**创建文件**: `src/faceforge/stage2/__init__.py`, `src/faceforge/stage2/config.py`, `src/faceforge/stage2/data_types.py`

**前置操作**: 复制所需资产文件到统一位置:
```bash
cp submodules/MICA/data/FLAME2020/landmark_embedding.npy data/pretrained/FLAME2020/
cp submodules/pixel3dmm/assets/flame_uv_coords.npy data/pretrained/FLAME2020/
cp submodules/pixel3dmm/assets/uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy data/pretrained/FLAME2020/flame_uv_valid_verts.npy
```

**config.py**: 按 `src/faceforge/stage1/config.py` (L4-30) 的 dataclass 模式, 实现 `Stage2Config`。完整字段参考 `stage2-design.md` Section 15。

**data_types.py**: 定义 `PreprocessedData`, `SharedParams`, `PerImageParams`, `Stage2Output`。参考 `stage2-design.md` Section 13。字段命名和 tensor 形状注释与 `src/faceforge/stage1/data_types.py` (L19-45) 保持一致风格。

**验证**: `python -c "from faceforge.stage2.config import Stage2Config; print(Stage2Config())"`

---

## Task 2: 独立 FLAME 模型

**创建文件**: `src/faceforge/stage2/flame_model.py`

从 `generic_model.pkl` 直接加载 FLAME, 实现可微前向传播。

**参考源码** (读取算法逻辑, 在 stage2 内重新实现):
- FLAME 加载: `submodules/pixel3dmm/src/pixel3dmm/tracking/flame/FLAME.py` L75-97 — pickle 加载 + register_buffer
- LBS 蒙皮: `submodules/pixel3dmm/src/pixel3dmm/tracking/flame/lbs.py` L170-263 — `lbs()` 函数, 含 shape blend, pose blend, skinning
- batch_rodrigues: 同文件 L310-341 或 `submodules/pixel3dmm/src/pixel3dmm/lightning/p3dmm_system.py` L23-57 (更简洁的实现)
- Landmark embedding: `submodules/pixel3dmm/src/pixel3dmm/tracking/flame/FLAME.py` L110-117 — 加载 `landmark_embedding.npy`, 提取 static/dynamic lmk_faces_idx + lmk_bary_coords
- Dynamic landmark: `submodules/flame-head-tracker/submodules/flame_lib/FLAME.py` `_find_dynamic_lmk_idx_and_bcoords()` — 根据 yaw 角选取下颌轮廓

```python
class FLAMEModel(nn.Module):
    def __init__(self, config: Stage2Config):
        # 从 generic_model.pkl 加载 8 个 buffer (参考 pixel3dmm FLAME.py L75-97):
        # v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, faces
        #
        # 从 landmark_embedding.npy 加载 (参考 L110-117):
        # static_lmk_faces_idx [51], static_lmk_bary_coords [51, 3]
        # dynamic_lmk_faces_idx [79, 17], dynamic_lmk_bary_coords [79, 17, 3]
        #
        # 从 FLAME_masks.pkl 加载: region 顶点索引 dict
        #
        # 从 flame_uv_coords.npy 加载: [5023, 2], V 轴翻转
        # flame_uv[..., 1] = (flame_uv[..., 1] * -1) + 1
        #
        # 从 flame_uv_valid_verts.npy 加载: UV loss 有效顶点 mask
        #
        # 硬编码眼球顶点:
        # R_EYE = [4597, 4543, 4511, 4479, 4575]
        # L_EYE = [4051, 3997, 3965, 3933, 4020]

    def forward(self, shape_params, expression_params, head_pose, jaw_pose):
        # 实现 lbs 蒙皮 (参考 pixel3dmm lbs.py L170-263)
        # 返回 (vertices [B, 5023, 3], landmarks_68 [B, 68, 3], landmarks_eyes [B, 10, 3])

    def get_vertex_normals(self, vertices):
        # cross product + scatter_add + normalize
```

**验证**:
1. `python -c "from faceforge.stage2.flame_model import FLAMEModel; ..."` 加载成功
2. 零参数前向传播 → vertices.shape == [1, 5023, 3]
3. 与 MICA 的 FLAME 输出比较 (相同零参数), 误差 < 1e-5

---

## Task 3: 相机参数 + Rodrigues

**创建文件**: `src/faceforge/stage2/camera.py`

**参考源码**:
- 投影管线: `submodules/VHAP/vhap/util/render_nvdiffrast.py` L117-160 `projection_from_intrinsics()` — 从 K 矩阵构建 clip-space 投影
- batch_rodrigues: `submodules/pixel3dmm/src/pixel3dmm/lightning/p3dmm_system.py` L23-57

```python
def rodrigues(r: Tensor) -> Tensor:
    """axis-angle [B, 3] → rotation matrix [B, 3, 3], 可微"""
    # 参考 pixel3dmm p3dmm_system.py L23-57

def build_intrinsics(focal_length, principal_point, image_size) -> Tensor:
    """→ [B, 3, 3] 内参矩阵"""

def build_extrinsics(head_pose, translation) -> Tensor:
    """→ [B, 4, 4] 外参矩阵, 使用 rodrigues"""

def project_points(points_3d, K, RT, image_size) -> Tensor:
    """3D → 2D 像素坐标 [B, N, 2]"""

def project_to_clip(vertices, K, image_size, znear=0.1, zfar=10.0) -> Tensor:
    """Camera space → Clip space [B, V, 4], 供 nvdiffrast 使用"""
    # 参考 VHAP render_nvdiffrast.py L117-160
```

**验证**: 构造已知 3D 点 + 已知 K/RT, 投影结果与手算一致。rodrigues 正反互逆。

---

## Task 4: nvdiffrast 可微渲染器

**创建文件**: `src/faceforge/stage2/renderer.py`

**参考源码** (读取逻辑, 在 stage2 重新实现):
- 光栅化管线: `submodules/VHAP/vhap/util/render_nvdiffrast.py` L216-254 — world_to_camera → camera_to_clip → dr.rasterize → dr.interpolate → dr.antialias
- SH 光照: 同文件 L19-53 `get_SH_shading()` — 9 个 SH 基函数计算 + shading

```python
class NvdiffrastRenderer(nn.Module):
    def __init__(self, image_size=512, use_opengl=False):
        # dr.RasterizeCudaContext() 或 dr.RasterizeGLContext()

    def render(self, vertices, faces, K, RT, vertex_normals,
               vertex_colors=None, sh_coefficients=None) -> dict:
        # 1. project_to_clip (Task 3)
        # 2. dr.rasterize()
        # 3. dr.interpolate() — normals, positions, colors
        # 4. compute_sh_shading() — 参考 VHAP L19-53
        # 5. dr.antialias()
        # 6. flip(1) — OpenGL y-up 转图像 y-down
        # 返回 {'image', 'mask', 'normal', 'depth'}

    def compute_visibility_mask(self, depth_buffer, projected_verts, proj_z) -> Tensor:
        """遮挡过滤: grid_sample depth at proj positions, 比较 proj_z"""
        # 参考 pixel3dmm tracker.py 遮挡过滤逻辑
```

**验证**:
1. 加载 FLAME (Task 2), 零参数 → 渲染正面图 → 保存 PNG, 目视检查
2. 渲染 mask 非全零, depth 值合理 (0.1~10 范围)

---

## Task 5: Pixel3DMM 推理封装

**创建文件**: `src/faceforge/stage2/pixel3dmm_inference.py`

**参考源码** (此 Task 需要 import pixel3dmm):
- 环境 patch: `submodules/pixel3dmm/src/pixel3dmm/env_paths.py` L1-34 — 需 patch `CODE_BASE`
- 模型加载: `submodules/pixel3dmm/src/pixel3dmm/lightning/p3dmm_system.py` — `P3DMMSystem.load_from_checkpoint()`
- 推理流程: `submodules/pixel3dmm/scripts/network_inference.py` L115-158 — 图像归一化 + mask 构造 + 原图/翻转平均 + UV/Normal 后处理

```python
class Pixel3DMMInference:
    def __init__(self, config: Stage2Config):
        # 1. sys.path.insert(0, config.pixel3dmm_code_base + '/src')
        # 2. Patch env_paths.CODE_BASE = config.pixel3dmm_code_base
        #    参考 env_paths.py L13: CODE_BASE 是所有路径的根
        # 3. from pixel3dmm.lightning.p3dmm_system import P3DMMSystem
        # 4. 加载 UV 和 Normal 两个模型

    @torch.no_grad()
    def predict(self, aligned_image, face_mask) -> tuple[Tensor, Tensor]:
        # 参考 network_inference.py L115-158:
        # 1. image / 255.0 → [0,1], reshape [1, 1, 512, 512, 3]
        # 2. mask 构造: (seg==2)|((seg>3)&(seg<14))&~(seg==11)
        # 3. 原图 + flip 各推理一次, 取平均
        # 4. UV: clamp((pred+1)/2, 0, 1)
        # 5. Normal: normalize + 坐标变换 [x, 1-z, 1-y]
        # 返回 (uv_map [1,2,H,W], normal_map [1,3,H,W])
```

**验证**: 输入 Stage 1 的 aligned_image + face_mask → 输出 UV/Normal map → 保存可视化, 确认面部区域有合理 UV 分布。

---

## Task 6: L_landmark + L_reg

**创建文件**: `src/faceforge/stage2/losses/__init__.py`, `src/faceforge/stage2/losses/landmark.py`, `src/faceforge/stage2/losses/regularization.py`

**参考源码**:
- 区域权重: `submodules/flame-head-tracker/tracker_base.py` L626-633 — 权重赋值模式
- L2 距离: `submodules/flame-head-tracker/utils/loss_utils.py` `compute_l2_distance_per_sample()` — `sqrt(sum((v1-v2)^2, dim=2))` then mean
- L1 距离: 同文件 `compute_l1_distance_per_sample()` — `sum(abs(v1-v2), dim=2)` then mean
- 鼻部权重 10x: `submodules/VHAP/vhap/model/tracker.py` 中鼻部 landmark 权重 (折中取 3.0x)

**landmark.py**:
```python
def landmark_loss(pred_68, pred_eyes, target_68, target_eyes,
                  use_l2=False, visibility_mask=None):
    """
    权重: 鼻 27-35=3.0, 眼 36-47=3.0, 嘴 49-67=2.0,
          面部 17-26=1.5, 下颌 3-13=0.75, 下颌两端 0-2/14-16=0.5
    眼球 10pt=3.0
    use_l2: Coarse 阶段 True, Medium+ 阶段 False
    visibility_mask: 遮挡过滤 (from renderer)
    """
```

**regularization.py**:
```python
def regularization_loss(params, mica_init_shape, config):
    """
    shape: 约束到 MICA (不是零!) — from pixel3dmm tracker.py
    + expression L2 + jaw L2
    + SH 单色正则 — from HRN losses.py (w_gamma)
    + 纹理 TV — from VHAP config (reg_tex_tv)
    """
```

**验证**: pred=target → loss≈0; pred=target+noise → loss>0; backward() 无报错。

---

## Task 7: L_pixel3dmm_uv + L_normal

**创建文件**: `src/faceforge/stage2/losses/pixel3dmm_uv.py`, `src/faceforge/stage2/losses/normal.py`

**参考源码**:
- UVLoss 全部逻辑: `submodules/pixel3dmm/src/pixel3dmm/tracking/losses.py` L46-139
  - L56-59: valid_verts 加载 (两种 strictness)
  - L60-61: flame_uv_coords 加载 + V 轴翻转
  - L88: `knn_points(can_uv, gt_uv)` 建立对应
  - L99-127: loss 计算 + delta/dist 过滤
  - L70-74: `finish_stage1()` 收紧阈值
- Normal loss 异常值过滤: `submodules/pixel3dmm/src/pixel3dmm/tracking/tracker.py` L975-996

**pixel3dmm_uv.py**:
```python
class Pixel3DMMUVLoss:
    def __init__(self, flame_uv_coords, valid_verts_mask):
        # 参考 losses.py L52-66: 加载 UV + valid mask
        # V 轴翻转 (L60-61)

    def compute_correspondences(self, uv_prediction, delta_uv, dist_uv):
        # 参考 losses.py L88: knn_points
        # 参考 L99-127: delta/dist 过滤

    def tighten_thresholds(self, delta_fine, dist_fine):
        # 参考 L70-74: finish_stage1()

    def compute_loss(self, projected_vertices_2d, visibility_mask=None):
        # L1, 除以 image_size 归一化 (参考 L117-126)
```

**normal.py**:
```python
def normal_loss(rendered_normals, predicted_normals, face_mask,
                eye_mask=None, delta_n=0.15):
    # 参考 tracker.py L975-996:
    # 1. l_map = predicted - rendered
    # 2. valid = (l_map.abs().sum / 3) < delta_n
    # 3. 眼部膨胀排除
    # 4. (1 - cosine_sim) * valid_mask
```

**验证**: 对齐的 UV/Normal → loss≈0; 加噪声 → loss>0; 梯度存在。

---

## Task 8: L_contour + L_sil + L_region_weight (基线方案)

**创建文件**: `src/faceforge/stage2/losses/contour.py`, `src/faceforge/stage2/losses/silhouette.py`, `src/faceforge/stage2/losses/region_weight.py`

**参考源码**:
- contour_aware_loss: `submodules/HRN/models/losses.py` L187-205 — **完整实现**, boundary-checking (不是 chamfer!)
  - L193: `verts_y = width - 1 - verts_int[:, :, 1]`
  - L197-198: 查询 left_points / right_points
  - L200: `dist = (left - x) / width * (right - x) / width`
  - L202: 除以 max(|left-x|, |right-x|) 归一化
  - L203-205: relu + 0.01 offset 防止零梯度
- silhouette loss: `submodules/VHAP/vhap/model/tracker.py` — `|fg_mask - rendered_fg|`

**contour.py**: 从 HRN L187-205 移植 `contour_aware_loss`, 输入改为使用 BiSeNet mask 提取 left/right boundary。

**silhouette.py**: 简单的 `|rendered_mask - target_mask|.abs().mean()`。

**region_weight.py**: FLAME_masks 顶点权重 → nvdiffrast 光栅化 → 2D 权重图。

**验证**: contour loss 对越界顶点产生正 loss, 对界内顶点产生零 loss。渲染权重图可视化, 确认鼻/眼高亮。

---

## Task 9: L_photometric + L_identity

**创建文件**: `src/faceforge/stage2/losses/photometric.py`, `src/faceforge/stage2/losses/identity.py`

**参考源码**:
- photometric: `submodules/flame-head-tracker/utils/loss_utils.py` L37-63 `compute_batch_pixelwise_l1_loss()` — masked L1, **除以 mask面积×通道数** 归一化
- identity: `submodules/MICA/micalib/models/mica.py` 中 ArcFace encoder 的 forward — 输入 112×112 [-1,1], 输出 512-dim L2 normalized

**photometric.py**:
```python
def photometric_loss(rendered, target, mask, region_weight_map=None):
    # 参考 flame-head-tracker loss_utils.py L37-63:
    # l1 = |rendered - target| * mask
    # loss = l1.sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) * C + 1e-8)
    # 可选乘以 region_weight_map
```

**identity.py**:
```python
class IdentityLoss:
    def __init__(self, arcface_model):
        # 从 MICA 提取 ArcFace, frozen
    def compute(self, rendered_image, target_arcface_feat):
        # rendered → crop 112×112 → (x*255-127.5)/127.5 → arcface → L2 norm
        # loss = 1 - cosine_similarity
```

**验证**: photometric: rendered=target → loss≈0; identity: 同人 → cosine sim 高。

---

## Task 10: L_prdl (备选方案)

**创建文件**: `src/faceforge/stage2/losses/prdl.py`

**来源**: 3DDFA-V3 (CVPR 2024), [arXiv:2312.00311](https://arxiv.org/abs/2312.00311)

```python
class PRDLLoss:
    """统一替代 contour + region_weight + sil (use_prdl=True 时)
    BiSeNet 19-class → FLAME region 映射
    per-part: mask_to_points vs projected_vertices → grid_statistical_distance
    """
```

**验证**: `use_prdl=True` 和 `False` 两条路径均可执行, loss 值合理。

---

## Task 11: Loss 聚合器

**创建文件**: `src/faceforge/stage2/losses/total.py`

```python
class LossAggregator:
    def __init__(self, config, flame_masks, mica_init_shape, arcface_model=None):
        # 初始化所有 loss 子模块

    def compute(self, stage: str, **kwargs) -> tuple[Tensor, dict]:
        """
        5 阶段调度: coarse_lmk / coarse_uv / medium / fine_pca / fine_detail
        A/B 切换: use_prdl

        stage='coarse_lmk': landmark(L2) only
        stage='coarse_uv':  + uv
        stage='medium':     + normal + sil + contour/prdl + reg (landmark→L1, UV收紧)
        stage='fine_pca':   + photometric + identity
        stage='fine_detail': 同上

        NaN guard: torch.nan_to_num(total, nan=0.0, posinf=1e5)
        返回 (total_loss, {name: value} dict)
        """
```

**验证**: 5 个 stage 分别调用, 确认激活的 loss 项和距离度量正确。

---

## Task 12: 五阶段优化器

**创建文件**: `src/faceforge/stage2/optimizer.py`

**参考源码**:
- coarse-to-fine 模式: `submodules/VHAP/vhap/model/tracker.py` optimize_stage / get_train_parameters
- LR 衰减: `submodules/pixel3dmm/src/pixel3dmm/tracking/tracker.py` L1076-1098 — 50%/75%/90% 分段衰减
- 早停: 同文件 L1169-1176 — 监控 loss 变化窗口

```python
STAGES = {
    'coarse_lmk':  {'params': ['head_pose','jaw_pose','translation','focal_length'], 'steps_key': 'coarse_lmk_steps'},
    'coarse_uv':   {'params': ['head_pose','jaw_pose','translation','focal_length'], 'steps_key': 'coarse_uv_steps'},
    'medium':      {'params': ['shape','expression','head_pose','jaw_pose','translation','focal_length'], 'steps_key': 'medium_steps'},
    'fine_pca':    {'params': ['shape','expression','head_pose','jaw_pose','translation','focal_length','texture','lighting'], 'steps_key': 'fine_pca_steps'},
    'fine_detail': {'params': ['shape','expression','head_pose','jaw_pose','translation','focal_length','texture','lighting','texture_disp'], 'steps_key': 'fine_detail_steps'},
}

class Stage2Optimizer:
    def create_optimizer(self, stage, shared_params, per_image_params, selected_indices):
        """创建 Adam, 共享参数 + 选中图的独立参数, 按 param_lr 分组"""

    def adjust_lr(self, optimizer, progress):
        """参考 pixel3dmm L1076-1098: 大参数/小参数分段衰减"""

    def check_early_stopping(self, loss_history, window, delta):
        """参考 pixel3dmm L1169-1176"""
```

**验证**: coarse_lmk 阶段只有 pose/camera requires_grad; medium 加入 shape/exp。

---

## Task 13: Pipeline 主入口

**创建文件**: `src/faceforge/stage2/pipeline.py`

**参考源码**: `src/faceforge/stage1/pipeline.py` L27-150 — `__init__` 加载模型 + `run_single`/`run_multi` 模式

```python
class Stage2Pipeline:
    def __init__(self, config: Stage2Config = None, mica_model=None):
        # 加载 FLAMEModel (Task 2)
        # 加载 NvdiffrastRenderer (Task 4)
        # 加载 Pixel3DMMInference (Task 5, lazy)
        # 初始化 LossAggregator (Task 11)
        # 从 mica_model 提取 ArcFace (可选)

    def run(self, stage1_outputs: list[Stage1Output]) -> Stage2Output:
        if len(stage1_outputs) == 1:
            return self._run_single(stage1_outputs[0])
        else:
            return self._run_multi(stage1_outputs)

    def _run_single(self, s1out: Stage1Output) -> Stage2Output:
        # 1. Pixel3DMM 预处理 → PreprocessedData
        # 2. 初始化参数 (requires_grad)
        # 3. UV KNN 对应 (宽松阈值)
        # 4. 五阶段优化循环 (Task 12)
        #    medium 开始时 tighten UV
        # 5. 最终输出: shape + zeros exp/pose → mesh

    def _run_multi(self, s1outs: list[Stage1Output]) -> Stage2Output:
        # 1. 每图 Pixel3DMM 预处理
        # 2. 共享参数: shape=median, texture=s1outs[0].texture
        # 3. 每图独立参数初始化
        # 4. Sequential: 逐图冻结 shape 优化 (coarse+medium 子集)
        # 5. Global: 解冻 shape, 联合五阶段优化
        #    每步采样 min(N, batch_size) 张图
        #    loss = mean over sampled images
        # 6. 最终输出: shared.shape + zeros → mesh
```

**验证** (需要 GPU):
```python
from faceforge.stage1 import Stage1Pipeline
from faceforge.stage2 import Stage2Config, Stage2Pipeline

s1 = Stage1Pipeline()
s1_out = s1.run_single(image_rgb, 'test')

s2 = Stage2Pipeline(Stage2Config(), mica_model=s1.mica.mica)
s2_out = s2.run([s1_out])
assert s2_out.vertices.shape == (1, 5023, 3)
assert len(s2_out.loss_history) > 0
```

---

## Task 14: 调试可视化

**创建文件**: `src/faceforge/stage2/visualization.py`

**参考**: `src/faceforge/stage1/visualization.py` — Stage 1 的可视化模式, 保持目录结构风格一致。

```python
class Stage2Visualizer:
    def __init__(self, output_dir, subject_name):
        # output/{subject}/stage2/01_preprocessing/ 02_optimization/ 03_result/

    def save_preprocessing(self, preprocessed_list):
        # pixel3dmm_uv_{i}.png, pixel3dmm_normals_{i}.png

    def save_progress(self, step, rendered, target, loss_dict):
        # 每 50 步保存渲染叠加图

    def save_loss_curves(self, loss_history):
        # matplotlib 各 loss 项曲线

    def save_result(self, stage2_output, target_images):
        # mesh_optimized.obj, render_overlay.png, before_after.png

    def save_summary(self):
        # 4-panel: input → S1 mesh → S2 mesh → overlay
```

**验证**: 运行完整 pipeline (Task 13), 检查输出目录图片是否正确生成。

---

## Task 15: 测试

**创建文件**: `tests/stage2/test_unit.py`, `tests/stage2/test_integration.py`

**单元测试** (不需要 GPU/权重):
- `test_config_defaults`: Stage2Config 默认值合理
- `test_rodrigues_invertible`: rodrigues(r) 逆运算正确
- `test_project_points`: 已知 3D→2D 正确
- `test_landmark_weights`: 鼻部 3.0, 眼 3.0, 嘴 2.0 ...
- `test_landmark_l1_vs_l2`: use_l2 切换正确
- `test_regularization_to_mica`: shape 约束到 mica_init 而非零
- `test_loss_aggregator_5stages`: 5 阶段激活正确的 loss 集合
- `test_prdl_switch`: use_prdl=True/False 切换
- `test_uv_v_flip`: flame_uv_coords V 轴翻转

**集成测试** (需要 GPU + 权重, 标记 `@pytest.mark.integration`):
- `test_flame_model_parity`: 独立 FLAME vs MICA FLAME, 误差 < 1e-5
- `test_renderer_output_shapes`: nvdiffrast 输出形状正确
- `test_pixel3dmm_inference`: UV/Normal 形状和值范围正确
- `test_full_pipeline_single`: 单图端到端, loss 下降
- `test_full_pipeline_multi`: 多图端到端, shape 收敛

---

## Task 16: CLI 脚本

**创建文件**: `scripts/run_stage2.py`

```python
"""
用法:
  python scripts/run_stage2.py --images face1.jpg face2.jpg face3.jpg --device cuda:0 --debug
  python scripts/run_stage2.py --images face.jpg --use-prdl
  python scripts/run_stage2.py --image-dir assets/tom/ --debug

流程:
  1. 加载所有图片
  2. Stage 1 pipeline × N → N 个 Stage1Output
  3. Stage 2 pipeline (多图联合) → Stage2Output
  4. 保存结果到 output/{subject}/stage2/
"""
```

**验证**: `python scripts/run_stage2.py --image-dir assets/tom/ --debug`, 检查输出。

# Stage 1: 身份感知初始化 — 详细设计文档

## 1. 目标

从输入图像中提取 FLAME 全参数初始值，为 Stage 2 优化提供高质量起点。核心输出：

| 参数 | 维度 | 来源 | 精度要求 |
|------|------|------|---------|
| shape | 300 | MICA | 决定身份相似度上限 |
| expression | 100 | DECA (前50维) + 零填充 | 合理即可，Stage 2 会优化 |
| head_pose | 3 | DECA | 合理即可 |
| jaw_pose | 3 | DECA | 合理即可 |
| texture | 50 | DECA | 用于后续光度优化的起点 |
| lighting | 27 (9×3 SH) | DECA | 用于后续光度优化的起点 |

## 2. 总体流程

```
输入图像 (任意尺寸, RGB)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: 双检测器人脸检测                              │
│  ├─ MediaPipe → 478点landmark + 68点(转换) + bbox     │
│  └─ RetinaFace → 原生5点landmark (用于 MICA ArcFace)  │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 2: 人脸对齐裁剪 → 512×512                       │
│  (基于 MediaPipe 68 点, flame-head-tracker 对齐算法)   │
└─────────┬────────────────────────────────────────────┘
          │  aligned_img_512
          ▼
┌──────────────────────────────────────────────────────┐
│  Step 3: 人脸分割 (BiSeNet)                           │
│  → skin/eye/nose/lip/hair 区域 mask                   │
└─────────┬────────────────────────────────────────────┘
          │  face_mask [512, 512, 19类]
          ▼
┌───────────────────────┬──────────────────────────────┐
│  Step 4a: MICA 推理    │  Step 4b: DECA 推理           │
│  RetinaFace 原生5点    │  FAN 检测裁剪                  │
│  → norm_crop 112×112   │  224×224 → [0,1]             │
│  → [-1,1] ArcFace     │  → ResNet50 编码              │
│  → 512维 identity      │  → exp/pose/tex/light 参数    │
│  → 300维 shape code    │                              │
│  (全程内存, 不存 JPEG)  │                              │
└───────────┬────────────┴──────────────┬──────────────┘
            │                           │
            ▼                           ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: 参数合并                                     │
│  shape=MICA(完整300维), exp/pose/tex/light=DECA       │
│  + 多图中位数聚合 (可选)                               │
└──────────────────────────────────────────────────────┘
            │
            ▼
      FLAME 初始参数 + face_mask + landmarks
```

## 3. Step 1: 人脸检测与 Landmark

### 3.1 双检测器方案

本 pipeline 使用**两个检测器**, 各取所长:

| 检测器 | 用途 | 原因 |
|--------|------|------|
| **MediaPipe FaceLandmarker** | 478 点稠密 landmark → 68 点转换 → 对齐裁剪 + Stage 2 约束 | 点数最多, 覆盖全脸 |
| **RetinaFace (InsightFace)** | 原生 5 点 landmark → MICA ArcFace 对齐 | ArcFace 训练时使用的检测器, 对齐质量与模型期望最匹配 |

> **设计决策**: 对比发现 flame-head-tracker 用 MediaPipe 478→68→5 点转换链做 ArcFace 对齐, pixel3dmm 用 RetinaFace 原生 5 点。由于 ArcFace 模型是在 RetinaFace 5 点对齐的数据上训练的, 用相同检测器能减少 domain gap, 产出更准确的 identity code。

### 3.2 MediaPipe FaceLandmarker

**模型文件**: `data/pretrained/mediapipe/face_landmarker.task` (3.6MB)

内含 3 个子模型:
- `face_detector.tflite` (229KB) — 人脸检测
- `face_landmarks_detector.tflite` (2.5MB) — 478 点稠密 landmark
- `face_blendshapes.tflite` (955KB) — 52 个 FACS blendshape 分数

**初始化** (参考 `flame-head-tracker/tracker_base.py` L100-106):

```python
import mediapipe as mp
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.python.BaseOptions(
        model_asset_path='data/pretrained/mediapipe/face_landmarker.task'
    ),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
mp_detector = vision.FaceLandmarker.create_from_options(options)
```

**输出**:

| 输出 | 形状 | 用途 |
|------|------|------|
| `lmks_dense` | [478, 2] | 稠密 landmark, 用于 Step 2 对齐 + Stage 2 约束 |
| `lmks_68` | [68, 2] | 通过 `mp2dlib.py` 从 478→68 转换, 用于对齐裁剪 |
| `lmks_eyes` | [10, 2] | 左右眼各 5 点 (MediaPipe idx: 右[468-472], 左[473-477]) |
| `blend_scores` | [52] | FACS blendshape, 可辅助表情初始化 |

### 3.3 RetinaFace (InsightFace)

**参考**: `pixel3dmm/preprocessing/MICA/utils/landmark_detector.py` L34-55

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 检测: 直接返回原生 5 点
faces = app.get(img_bgr)
kps = faces[i].kps  # [5, 2] — 原生 5 点, 无需从 68 点转换
```

**原生 5 点含义** (与 ArcFace 训练对齐标准一致):
- 左眼中心, 右眼中心, 鼻尖, 左嘴角, 右嘴角

### 3.4 478→68 Landmark 转换

**参考**: `flame-head-tracker/utils/mp2dlib.py` L37-126

转换规则: 大部分 68 点取单个 MediaPipe 索引, 部分取两个索引的平均值:
- 脸颊轮廓 #4: avg(MP[132], MP[58])
- 鼻梁 #28: avg(MP[168], MP[6])
- 眼角、嘴角等: 单点对应

### 3.5 多人脸选择

当检测到多张人脸时, 选择最接近图像中心的人脸 (参考 pixel3dmm `util.py` L92-106):

```python
def get_center(bboxes, img):
    img_center = np.array([img.shape[1]//2, img.shape[0]//2])
    centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2
    dists = np.linalg.norm(centers - img_center, axis=1)
    return np.argmin(dists)
```

## 4. Step 2: 人脸对齐裁剪

### 4.1 对齐算法

**参考**: `flame-head-tracker/tracker_base.py` 中的 `image_align()` (L133-233)

基于 68 点 landmark 的仿射对齐, 关键参数:

| 参数 | 值 | 来源 |
|------|-----|------|
| `output_size` | 512 | 标准裁剪分辨率 |
| `transform_size` | 1024 | 内部工作分辨率 (先算 1024 再缩到 512) |
| `scale_factor` | 1.3 | tracking 标准 (vs 1.0 for FFHQ 标准) |
| `padding_mode` | 'constant' (255) 或 'reflect' | 边界填充 |
| `blur` | qsize × 0.02 | 填充区域高斯模糊 sigma |
| `border` | max(round(qsize × 0.1), 3) | 像素边框 |

**处理流程**:

1. 从 68 点提取: 下巴轮廓, 眉毛, 鼻子, 眼睛, 嘴巴子集
2. 计算眼睛中心、眼间距向量
3. 计算嘴巴中心、眼嘴距向量
4. 确定旋转角度和裁剪矩形
5. `shrink = floor(qsize / output_size * 0.5)`
6. 边界填充 + 高斯模糊混合
7. PIL `Image.QUAD` 透视变换 + `Image.BILINEAR` 插值
8. 最终 resize 到 512×512 (`Image.ANTIALIAS`)

### 4.2 pixel3dmm 的裁剪方案 (对比参考)

pixel3dmm 使用更简单的裁剪:

1. FaceBoxesV2 检测 bbox, 置信度阈值 0.6
2. bbox 扩展: `det_box_scale = 1.42`
3. 强制正方形: 扩展短边到与长边等长
4. 边界 clamp 后 `cv2.resize` 到 512×512

**差异**: pixel3dmm 裁剪更紧 (1.42x), flame-head-tracker 更松 (1.3x + 旋转对齐)。flame-head-tracker 的对齐质量更高, 因为做了旋转矫正。

### 4.3 建议

采用 flame-head-tracker 的 `image_align()` 方案, 原因:
- 包含旋转矫正, 大姿态下对齐质量更好
- MICA 论文的 ArcFace 对齐也需要正面化处理, 上游对齐越好, 下游越准

## 5. Step 3: 人脸分割

### 5.1 分割模型

**参考**: `flame-head-tracker/tracker_base.py` 使用 BiSeNet

**输出**: 19 类语义分割图 [512, 512]:

```
0: background    1: skin           2: l_brow         3: r_brow
4: l_eye         5: r_eye          6: eye_g(lasses)  7: l_ear
8: r_ear         9: ear_r(ing)     10: nose          11: mouth
12: u_lip        13: l_lip         14: neck          15: neck_l(ace)
16: cloth        17: hair          18: hat
```

**预处理**: 输入 resize 到 512×512, ImageNet 归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 5.2 Face Mask 提取

用于后续 Stage 2/3 的光度损失遮罩:

```python
# flame-head-tracker 的默认策略
face_mask = (parsing > 0) & (parsing <= 13)  # 保留: skin, brow, eye, nose, lip
# 排除: neck(14-15), cloth(16), hair(17), hat(18), background(0), glasses(6)
# 可选: keep_mouth=False 时额外排除 mouth(11)
```

### 5.3 pixel3dmm 的分割方案 (对比参考)

pixel3dmm 使用 **facer** 库 (基于 FARL):
- 通过 RetinaFace 检测 + FARL face parser
- 输出格式不同但语义类似

**建议**: 两个分割器都可用。BiSeNet (flame-head-tracker 中已有) 更轻量; facer (pixel3dmm) 精度可能略高。初版用 BiSeNet, 可后续对比。

## 6. Step 4a: MICA 推理 (Shape 提取)

### 6.1 ArcFace 输入准备

**参考**: `pixel3dmm/preprocessing/MICA/datasets/creation/util.py` L42-45

**关键流程**:

1. **5 点 landmark**: 直接使用 Step 1 中 RetinaFace 的**原生 5 点输出**, 不从 68 点转换

   > **设计决策**: flame-head-tracker 从 MediaPipe 478→68→5 点转换, pixel3dmm 用 RetinaFace 原生 5 点。本 pipeline 采用 pixel3dmm 方案, 因为 ArcFace 模型是在 RetinaFace 5 点对齐的数据上训练的, 用相同检测器消除 domain gap。
   >
   > flame-head-tracker 的 5 点提取方式 (参考 `mica_inference_utils.py` L42-56):
   > ```python
   > # flame-head-tracker: 从 68 点手动提取 (不采用)
   > lmk5[0] = mean(lmks_68[36:42])  # 左眼=6点均值
   > lmk5[1] = mean(lmks_68[42:48])  # 右眼=6点均值
   > ```
   > 这与 RetinaFace 原生检测的 5 点位置有微小偏差, 会影响 ArcFace 对齐质量。

2. **ArcFace 标准对齐** (insightface `norm_crop`):
   - 目标 5 点坐标 (112×112 空间):
     ```
     [38.2946, 51.6963]   # 左眼
     [73.5318, 51.5014]   # 右眼
     [56.0252, 71.7366]   # 鼻尖
     [41.5493, 92.3655]   # 左嘴角
     [70.7299, 92.2041]   # 右嘴角
     ```
   - 计算 SimilarityTransform (旋转+缩放+平移), 无剪切
   - `cv2.warpAffine()` 插值, `borderValue=0.0` (黑色填充)

3. **归一化到 [-1, 1]**:
   ```python
   blob = cv2.dnn.blobFromImages(
       [aligned_img],
       scalefactor=1.0 / 127.5,       # ÷127.5
       size=(112, 112),
       mean=(127.5, 127.5, 127.5),    # -127.5
       swapRB=True                     # BGR→RGB
   )
   # 最终: (pixel - 127.5) / 127.5 → [-1, 1]
   ```

### 6.2 MICA 模型加载

**模型文件**: `data/pretrained/mica.tar` (502MB, PyTorch checkpoint ZIP)

内含两组权重:
- `arcface` — IResNet50 (identity encoder, 512 维输出)
- `flameModel` — Generator/MappingNetwork (512→300 shape decoder) + FLAME 模型

**项目中有 3 份 MICA 代码副本** (flame-head-tracker 和 pixel3dmm 各嵌套了一份):

| 副本 | 权重路径 | 问题 |
|------|---------|------|
| `submodules/MICA` | 无预设路径 | 主副本, 代码最完整 |
| `flame-head-tracker/submodules/MICA` | `data/pretrained/mica.tar` | **路径不存在**, 需手动 symlink |
| `pixel3dmm/.../preprocessing/MICA` | `data/pretrained/mica.tar` | 已有 **symlink** → 共享权重 |

两个项目的加载代码也有差异:

```python
# flame-head-tracker: 直接构造 (缺少 weights_only=False, 新版 PyTorch 会报错)
mica = MICA(cfg, device)  # 内部调用 torch.load(model_path)

# pixel3dmm: 动态查找 + 安全加载
mica = util.find_model_using_name(...)(cfg, device)
load_checkpoint(args, mica)  # torch.load(..., weights_only=False)
```

**本 pipeline 的加载方案**:

```python
import sys
sys.path.insert(0, 'submodules/MICA')  # 直接用主副本, 不用嵌套副本
from micalib.models.mica import MICA
from configs.config import get_cfg_defaults

def load_mica(device, model_path='data/pretrained/mica.tar'):
    cfg = get_cfg_defaults()
    cfg.model.testing = True
    cfg.pretrained_model_path = model_path  # 统一指向共享权重
    mica = MICA(cfg, device)
    # 注意: 需确保内部 torch.load() 使用 weights_only=False
    mica.eval()
    return mica
```

**设计决策**:
- 统一使用 `submodules/MICA` 主副本, 避免嵌套副本的路径/版本不一致问题
- 权重路径统一指向 `data/pretrained/mica.tar`
- 需 patch MICA 的 `load_model()` 添加 `weights_only=False` (兼容 PyTorch >= 2.6)

### 6.3 ArcFace 编码器 (mica.tar 中的 arcface 权重)

**架构**: IResNet50 变体 (block 配置 [3, 13, 30, 3])

```
Input: 112×112×3 (RGB, [-1,1])
  → Conv2d(3→64, k=3, s=1, p=1) + BN + PReLU     → 112×112×64
  → Layer1: 3 blocks, stride=2                     → 56×56×64
  → Layer2: 13 blocks, stride=2                    → 28×28×128
  → Layer3: 30 blocks, stride=2                    → 14×14×256
  ────────── 以上层冻结 (no grad) ──────────
  → Layer4: 3 blocks, stride=2                     → 7×7×512
  → BN2d(512) → Flatten(25088) → Dropout
  → FC(25088→512) → BN1d(512)                      → 512维
```

**输出**: 512 维向量, 经 `F.normalize()` 后为 L2 单位向量

### 6.4 MappingNetwork (mica.tar 中的 flameModel 权重, 512→300)

**架构**: 4 层 MLP

```
Input: 512维 (L2 normalized ArcFace features)
  → Linear(512→300) → LeakyReLU(0.2)
  → Linear(300→300) → LeakyReLU(0.2)
  → Linear(300→300) → LeakyReLU(0.2)
  → Linear(300→300)                        → 300维 shape code (无激活)
```

- 权重初始化: Kaiming normal (a=0.2, fan_in, leaky_relu)
- 最后一层权重额外缩放 ×0.25
- hidden=3 时无 skip connection

### 6.5 FLAME Shape → Mesh

```python
# shape_code: [B, 300], expression 默认全零 [B, 100]
betas = torch.cat([shape_code, zeros_expression], dim=1)  # [B, 400]
vertices = v_template + shapedirs @ betas                  # [B, 5023, 3]
# v_template: FLAME 平均脸模板
# shapedirs: [5023, 3, 400] PCA 基底 (前300维=shape, 后100维=expression)
```

**输出**: 5023 个顶点, 单位为米 (导出时 ×1000 转为 mm)

### 6.6 影响相似度的关键细节

1. **ArcFace 5 点来源**: 使用 RetinaFace 原生 5 点 (与 ArcFace 训练数据一致), 而非从 68 点转换。对齐偏差 1px 即可导致 identity code 显著变化。

2. **L2 归一化**: `F.normalize(arcface_output)` 是必须的, 确保 identity code 在单位球面上, MappingNetwork 训练时依赖这个前提。

3. **Shape code 保留完整 300 维**: flame-head-tracker 截断到 100 维, 丢失了后 200 维的个人化细节 (鼻翼宽度微调、颧骨弧度等)。本 pipeline 保留完整 300 维 (与 pixel3dmm 一致)。

4. **全程内存处理**: pixel3dmm 将对齐图存为 JPEG 再读回 (有损), 本 pipeline 全程内存操作, 避免压缩损失。

5. **MICA 训练时的区域权重**: 训练损失对不同区域加权不同:
   - face: 150.0, nose: 50.0, lips: 50.0, forehead: 50.0
   - eyes: 0.01, ears: 0.01 (极低)
   - **含义**: MICA 的 shape 输出在眼部和耳部区域精度较差, 需要 Stage 2 补偿

## 7. Step 4b: DECA 推理 (Expression/Pose/Texture/Lighting)

### 7.1 DECA 输入准备

**参考**: `flame-head-tracker/submodules/decalib/deca.py` 的 `crop_image()`

1. FAN 检测器获取 bbox
2. bbox 扩展: `size = int(old_size * 1.25)`
3. SimilarityTransform 裁剪到 224×224
4. 归一化: `/255.0` → [0, 1]
5. 输出: tensor [3, 224, 224]

### 7.2 DECA 编码器

**架构**: ResNet50 + 3层回归 MLP

```
Input: 224×224×3 (RGB, [0,1])
  → ResNet50 backbone                              → 2048维
  → Linear(2048→1024) → Linear(1024→1024)
  → Linear(1024→236)                               → 236维 (总参数)
```

**参数分解** (deca.py L164-176):

```
236 维 = 100 (shape) + 50 (tex) + 50 (exp) + 6 (pose) + 3 (cam) + 27 (light)
```

| 参数 | 维度 | 取值 | 备注 |
|------|------|------|------|
| shape | 100 | **丢弃** (用 MICA 的 300 维) | DECA shape 精度低于 MICA |
| exp | 50 | 填入 recon_dict[:, :50] | 100 维中后 50 维填零 |
| pose | 6 | 前3=head_pose, 后3=jaw_pose | 轴角表示 |
| tex | 50 | 填入 recon_dict[:, :50] | FLAME 纹理空间系数 |
| cam | 3 | **丢弃** | 用 Stage 2 自己的相机模型 |
| light | 27 | reshape 为 [9, 3] SH 系数 | Spherical Harmonics |

### 7.3 影响相似度的关键细节

1. **DECA shape 被丢弃**: DECA 只用 100 维 shape (vs MICA 300 维), 且未做身份编码, 精度远低于 MICA。

2. **Expression 只用前 50 维**: FLAME 支持 100 维表情, DECA 只预测 50 维, 后 50 维填零。对于张嘴、眨眼等需要后 50 维表达的表情, 初始化不够准确, 但 Stage 2 会优化。

3. **DECA 的 FAN 检测器**: DECA 内部用 FAN (face_alignment 库) 做人脸检测裁剪, 与我们外部的 MediaPipe 检测**独立**。不一致可能导致裁剪区域微小差异。

## 8. Step 5: 参数合并

### 8.1 合并逻辑

**参考**: `flame-head-tracker/tracker_base.py` L337-373 `run_reconstruction_models()`

```python
def merge_params(mica_output, deca_output):
    params = {}
    # Shape: 100% 来自 MICA
    params['shape'] = mica_output['shape_code'][:, :300]     # [1, 300]

    # Expression: DECA 的前 50 维, 后 50 维填零
    params['exp'] = torch.zeros([1, 100])
    params['exp'][:, :50] = deca_output['exp']                # [1, 100]

    # Pose: 100% 来自 DECA
    params['head_pose'] = deca_output['pose'][:, :3]          # [1, 3]
    params['jaw_pose'] = deca_output['pose'][:, 3:]           # [1, 3]

    # Texture: DECA 的前 50 维
    params['tex'] = torch.zeros([1, 50])
    params['tex'][:, :50] = deca_output['tex']                # [1, 50]

    # Lighting: 100% 来自 DECA
    params['light'] = deca_output['light'].reshape(1, 9, 3)   # [1, 9, 3]

    return params
```

### 8.2 多图中位数聚合

当输入多张同一人的图像时:

```python
def aggregate_shapes(images, mica_model, method='median'):
    shape_codes = []
    for img in images:
        code = mica_inference(mica_model, img)  # [1, 300]
        shape_codes.append(code.squeeze())

    stacked = torch.stack(shape_codes, dim=0)   # [N, 300]

    if method == 'median':
        return torch.median(stacked, dim=0).values  # [300]
    elif method == 'mean':
        return stacked.mean(dim=0)                   # [300]
```

**flame-head-tracker 的策略**: 前 3 帧取 mean (tracker_video.py L67-83)
**本项目建议**: 全部图像取 median (更鲁棒, 抗离群值)

### 8.3 相机参数初始化

**参考**: pixel3dmm `tracker.py` L405-457

```python
# 初始焦距 (像素单位, 基于 512×512 渲染分辨率)
init_focal = 2000.0 * (render_size / 512)
focal_length = init_focal / render_size           # 归一化焦距

# 主点: 图像中心
principal_point = [0.0, 0.0]                      # 归一化坐标

# FOV: 约 20° (flame-head-tracker 默认)
DEFAULT_FOV = 20.0
```

## 9. 影响相似度的设计决策汇总

### 9.1 高影响

| 决策 | 本 pipeline 选择 | 对相似度的影响 |
|------|---------------|-------------|
| **ArcFace 5 点来源** | RetinaFace 原生 5 点 (非 68→5 转换) | 与 ArcFace 训练数据一致, 消除 domain gap |
| **Shape code 维度** | 完整 300 维 (非截断 100 维) | 保留后 200 维个人化细节 |
| **多图聚合** | median (非 mean) | 抗离群值, 消除单图偏差 |
| **MICA vs DECA shape** | 始终用 MICA 300 维 | DECA shape 在 NoW 上差 MICA 约 30% |

### 9.2 中等影响

| 决策 | 细节 | 对相似度的影响 |
|------|------|-------------|
| **裁剪 scale factor** | 1.3 (flame-head-tracker) vs 1.42 (pixel3dmm) | 过紧裁剪可能切掉下巴/发际线, 过松降低有效分辨率 |
| **裁剪旋转矫正** | flame-head-tracker 做了, pixel3dmm 没做 | 大姿态下旋转矫正可改善 MICA shape 估计 |
| **MICA 眼耳区域精度低** | 训练权重 eyes=0.01, ears=0.01 | 眼眶深度和耳部形态需要 Stage 2 补偿 |

### 9.3 低影响 (但需注意正确性)

| 决策 | 细节 |
|------|------|
| ArcFace 归一化必须为 [-1,1] | (x-127.5)/127.5, 不是 [0,1] |
| DECA 归一化为 [0,1] | /255.0, 不是 [-1,1] |
| ArcFace 输入为 RGB | cv2 读入 BGR 需 swapRB=True |
| MICA 输出单位为米 | 导出网格需 ×1000 转 mm |
| F.normalize() 不可省略 | MappingNetwork 依赖 L2 单位向量输入 |
| 全程内存处理 | 不存中间 JPEG, 避免有损压缩 (pixel3dmm 的文件 I/O 方案不采用) |

## 10. 调试输出

每个环节输出可视化结果, 方便逐步检查 pipeline 质量。

### 10.1 输出目录结构

```
output/{subject_name}/
├── stage1/
│   ├── 01_detection/
│   │   ├── input.png                    # 原始输入图 (留底)
│   │   ├── mediapipe_478.png            # 478 点 landmark 绘制在原图上
│   │   ├── landmarks_68.png             # 68 点 landmark (从 478 转换) 绘制在原图上
│   │   ├── retinaface_5pt.png           # RetinaFace 原生 5 点绘制在原图上
│   │   └── landmarks_68.npy             # 68 点坐标 [68, 2]
│   │
│   ├── 02_alignment/
│   │   ├── aligned_512.png              # 对齐裁剪后的 512×512 图
│   │   └── alignment_grid.png           # 对齐前后对比 (左原图+landmark, 右对齐图+landmark)
│   │
│   ├── 03_segmentation/
│   │   ├── parsing_vis.png              # 19 类分割图 (彩色渲染)
│   │   ├── face_mask.png                # 二值 face mask (白=面部, 黑=非面部)
│   │   └── mask_overlay.png             # face mask 半透明叠加在对齐图上
│   │
│   ├── 04_mica/
│   │   ├── arcface_112.png              # ArcFace 对齐后的 112×112 输入图
│   │   ├── shape_code.npy               # 300 维 shape code
│   │   ├── mesh.obj                     # FLAME mesh (5023 顶点, mm 单位)
│   │   ├── mesh_front.png               # 网格正面渲染 (灰色 shading)
│   │   ├── mesh_side.png                # 网格侧面渲染 (45°)
│   │   └── mesh_overlay.png             # 网格线框叠加在输入图上
│   │
│   ├── 05_deca/
│   │   ├── deca_crop_224.png            # DECA 内部 FAN 裁剪的 224×224 输入
│   │   └── params.json                  # DECA 输出参数 (exp, pose, tex, light 的值)
│   │
│   ├── 06_merged/
│   │   ├── flame_params.npz             # 合并后的全部 FLAME 参数
│   │   ├── mesh_final.obj               # 最终 FLAME mesh (含表情/姿态)
│   │   ├── render_front.png             # 最终网格正面渲染
│   │   ├── render_overlay.png           # 最终网格叠加在输入图上 (关键检查图)
│   │   └── identity_similarity.txt      # ArcFace cosine similarity (渲染 vs 输入)
│   │
│   └── summary.png                      # 一图总览 (6 格: 输入→landmark→对齐→分割→MICA mesh→最终叠加)
```

### 10.2 多图聚合时的额外输出

```
output/{subject_name}/
├── stage1/
│   ├── per_image/
│   │   ├── img_001/                     # 每张图的完整 01-06 输出
│   │   ├── img_002/
│   │   └── img_003/
│   ├── aggregation/
│   │   ├── shape_codes_all.npy          # 所有图的 shape code [N, 300]
│   │   ├── shape_median.npy             # 中位数聚合结果 [300]
│   │   ├── shape_variance.png           # 各维度方差图 (高方差维度=不稳定)
│   │   └── mesh_comparison.png          # 各图 mesh vs 聚合 mesh 叠加对比
│   └── 06_merged/                       # 使用聚合 shape 的最终输出
```

### 10.3 各环节关键检查项

| 环节 | 检查什么 | 看哪个文件 |
|------|---------|----------|
| 检测 | landmark 是否准确落在五官上, 无漂移 | `mediapipe_478.png` |
| 检测 | RetinaFace 5 点是否在眼/鼻/嘴正确位置 | `retinaface_5pt.png` |
| 对齐 | 人脸是否正面化, 无明显旋转残留, 下巴/额头未被裁掉 | `aligned_512.png` |
| 分割 | mask 边界是否贴合面部轮廓, 头发/脖子是否正确排除 | `mask_overlay.png` |
| MICA | ArcFace 112 图是否正面、清晰、归一化正确 | `arcface_112.png` |
| MICA | mesh 形状是否大致匹配人脸 (鼻子宽度、下巴弧度) | `mesh_overlay.png` |
| DECA | DECA 裁剪区域是否合理, 未切掉关键区域 | `deca_crop_224.png` |
| 合并 | **最终叠加图是否"像"目标人物** | `render_overlay.png` |
| 合并 | ArcFace similarity 数值 (>0.5 基本合格, >0.7 良好) | `identity_similarity.txt` |

### 10.4 输出控制

```python
class Stage1Config:
    save_debug: bool = True          # 是否保存调试输出
    save_mesh: bool = True           # 是否保存 .obj 文件
    save_summary: bool = True        # 是否生成 summary.png 一图总览
    output_dir: str = 'output'       # 输出根目录
    render_size: int = 512           # 渲染预览图分辨率
```

设置 `save_debug=False` 可跳过所有中间输出, 仅保留 `06_merged/` 最终结果。

## 11. 与 Stage 2 的接口

Stage 1 输出传给 Stage 2 的完整数据:

```python
@dataclass
class Stage1Output:
    # FLAME 参数
    shape: Tensor           # [1, 300] — MICA, 作为 Stage 2 优化初始值
    expression: Tensor      # [1, 100] — DECA(前50)+零(后50), 可优化
    head_pose: Tensor       # [1, 3]   — DECA, 可优化
    jaw_pose: Tensor        # [1, 3]   — DECA, 可优化
    texture: Tensor         # [1, 50]  — DECA, 用于光度优化
    lighting: Tensor        # [1, 9, 3] — DECA SH 系数

    # 预计算特征 (Stage 2 损失函数用)
    arcface_feat: Tensor    # [1, 512] — L2 归一化, 用于 L_identity

    # 图像数据
    aligned_image: Tensor   # [1, 3, 512, 512] — 对齐裁剪后的输入图
    face_mask: Tensor       # [1, 512, 512] — 语义分割 mask (19类)

    # Landmark
    lmks_68: Tensor         # [1, 68, 2] — 68 点 landmark (图像坐标)
    lmks_dense: Tensor      # [1, 478, 2] — MediaPipe 稠密 landmark
    lmks_eyes: Tensor       # [1, 10, 2] — 眼部 landmark

    # 相机
    focal_length: Tensor    # [1, 1] — 归一化焦距
    principal_point: Tensor # [1, 2] — 主点
```

## 12. 代码结构

```
src/faceforge/stage1/
├── __init__.py
├── pipeline.py              # Stage 1 总入口: run_stage1(images) → Stage1Output
├── detection.py             # 双检测器: MediaPipe (478点) + RetinaFace (原生5点)
├── mp2dlib.py               # 478→68 landmark 转换 (从 flame-head-tracker 移植)
├── alignment.py             # image_align() 人脸对齐裁剪 (从 flame-head-tracker 移植)
├── segmentation.py          # BiSeNet 人脸分割
├── mica_inference.py        # MICA 推理: RetinaFace 5点 → ArcFace 对齐 → shape 解码 (全程内存)
├── deca_inference.py         # DECA 推理: FAN 裁剪 + 编码 + 参数分解
└── aggregation.py           # 多图 shape 聚合 (median/mean)
```

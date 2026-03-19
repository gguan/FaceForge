# FaceForge 头像重建 Pipeline 技术方案

## 1. 项目目标

设计一个综合性头像重建 pipeline，融合 MICA、DECA、HRN、Pixel3DMM、VHAP、flame-head-tracker 六个子模块的技术优势，**核心目标是最大化生成头像与目标人物的相似度**。

## 2. 子模块技术分析

### 2.1 MICA (ECCV 2022)

- **方法**: ArcFace 身份编码器 (512维) → FLAME shape 解码器 (300维)
- **核心优势**: 唯一将人脸身份特征显式编码到 3DMM 形状回归的方法
- **输出**: FLAME shape parameters + 3D mesh vertices (5023点, mm尺度)
- **基准**: NoW benchmark 顶尖

### 2.2 HRN (CVPR 2023)

- **方法**: 分层表示网络, ResNet50 backbone + displacement maps
- **核心优势**: 高频几何细节恢复 (皱纹、毛孔), 轮廓感知损失
- **输出**: 基础网格 + displacement map 叠加的精细网格
- **基准**: REALY benchmark 第一

### 2.3 Pixel3DMM (CVPR 2025)

- **方法**: DINOv2 特征 → 逐像素 UV 坐标图 + 表面法线图 (512×512)
- **核心优势**: 稠密屏幕空间约束, 将几何约束从 68 个 landmark 扩展到数万像素
- **输出**: per-pixel UV maps + normal maps → 可微优化 FLAME 参数
- **流程**: 预处理 → 网络推理 → 在线优化 + 全局精细化

### 2.4 VHAP

- **方法**: 可微网格光栅化 (nvdiffrast) + 自适应扰动机制
- **核心优势**: 遮挡感知 (头发、耳朵、颈部), 区域自适应外观先验
- **输出**: 优化后的 FLAME 参数 + 纹理贴图

### 2.5 DECA (SIGGRAPH 2021)

- **方法**: ResNet50 编码器 → FLAME 表情/姿态/纹理/光照参数
- **核心优势**: 表情、姿态、光照估计成熟; 包含 detail decoder 用于位移细节
- **输出**: expression (50维), head pose (3维), jaw pose (3维), texture (50维), lighting (27维, 9×3 SH)
- **在本 pipeline 中的角色**: 与 MICA 互补 — MICA 提供 shape, DECA 提供表情/姿态/光照初始化

### 2.6 flame-head-tracker

- **方法**: MICA + DECA 混合初始化 → Landmark 拟合 (~0.9s/帧) + 光度拟合 (~1.9s/帧)
- **核心优势**: 轻量成熟, 表情/姿态/相机参数联合优化
- **初始化策略**: MICA 负责 shape (300维), DECA 负责 expression/pose/texture/lighting
- **输出**: NPZ 格式的 FLAME 全参数 (shape, expression, pose, texture, lighting, camera)

## 3. Pipeline 架构

### 3.1 总体设计: 三阶段递进

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: 身份感知初始化                        [MICA + DECA]    │
│  Input Images → Face Detection → MICA → FLAME shape init         │
│  + DECA → expression/pose/texture/lighting init                  │
│  + 多图中位数聚合 (可选)                                          │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: 稠密约束优化              [Pixel3DMM + HRN + HiFace]   │
│  Pixel3DMM → per-pixel UV + normal maps (稠密几何约束)            │
│  + 98点 PIPNet landmark + 不确定性加权 (HiFace 启发)              │
│  + HRN contour-aware loss (下颌/颧骨轮廓)                        │
│  + ArcFace identity loss (身份相似度直接优化)                     │
│  + 投影空间区域加权 (鼻3x/眼2.5x/嘴2x) (HiFace 启发)            │
│  → nvdiffrast 可微渲染迭代优化 FLAME 参数                        │
└───────────────────────────┬──────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3: 细节恢复 + 外观对齐           [HRN + HiFace + VHAP]   │
│  HRN displacement maps → 顶点张力加权精炼 (HiFace 启发)          │
│  + ArcFace identity 校验 (防止 detail 破坏身份)                  │
│  + VHAP 光度对齐 → 纹理/外观一致性                               │
│  → 最终精细化网格 + 纹理                                         │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage 1: 身份感知初始化 (MICA + DECA 混合)

**目的**: 利用 MICA 的身份编码能力获取高质量形状初始值, 同时通过 DECA 获取表情、姿态和光照参数。

**流程**:

1. **人脸检测与裁剪**: FaceBoxesV2 检测 (置信度阈值 0.6), 选取最接近图像中心的人脸, 1.42x 缩放裁剪至 512×512
2. **ArcFace 预处理**: InsightFace RetinaFace 检测 → 112×112 blob (归一化至 [-1, 1])
3. **MICA 推理**: ArcFace blob → 512维身份特征 → 300维 FLAME shape code → 5023 顶点网格
4. **DECA 推理**: 输入图 → ResNet50 编码器 → expression (50维) + pose (6维) + texture (50维) + lighting (27维)

**参数初始化分工** (沿用 flame-head-tracker 的验证策略):

| 参数 | 来源 | 维度 | 原因 |
|------|------|------|------|
| Shape | MICA | 300维 | MICA 的身份编码更准确, NoW benchmark 验证 |
| Expression | DECA | 50维 | DECA 的表情估计成熟稳定 |
| Head Pose | DECA | 3维 | DECA 的姿态估计可靠 |
| Jaw Pose | DECA | 3维 | DECA 提供下颌旋转初始值 |
| Texture | DECA | 50维 | DECA 提供 FLAME 纹理空间系数 |
| Lighting | DECA | 27维 | DECA 的 SH 光照估计用于后续光度优化 |

**多图聚合增强**: 当有多张输入图时, 对每张图独立推理 MICA shape code, 取中位数, 零成本消除单张图的姿态/光照偏差。

```python
shape_codes = [mica(img) for img in images]
robust_shape = torch.median(torch.stack(shape_codes), dim=0).values
```

### 3.3 Stage 2: 稠密约束优化

**目的**: 通过可微渲染 + 多源稠密约束迭代优化 FLAME 参数, 突破 MICA 回归精度的上限。

**可优化参数**:

- FLAME shape (300维, 以 MICA 输出为初始值)
- FLAME expression (100维)
- 头部姿态 (全局旋转 + 下颌旋转)
- 相机参数 (焦距 + 平移)

**损失函数设计**:

```
L_total = λ₁·L_landmark       # 98点 PIPNet landmark + 不确定性加权 (HiFace 启发)
        + λ₂·L_pixel3dmm_uv   # Pixel3DMM 稠密 UV 对应约束
        + λ₃·L_normal          # Pixel3DMM 法线一致性
        + λ₄·L_contour         # HRN 启发的轮廓损失 (下颌线/颧骨)
        + λ₅·L_photometric     # 像素级光度损失 (渲染 vs 输入)
        + λ₆·L_identity        # ArcFace cosine similarity (渲染→ArcFace vs 输入→ArcFace)
        + λ₇·L_region_weight   # 投影空间区域加权 (HiFace 启发, 鼻/眼区域光度+landmark 加权)
        + λ_reg·L_reg          # FLAME 正则化 (防止偏离合理分布)
```

**各损失项技术来源与作用**:

| 损失项 | 技术来源 | 作用 |
|--------|---------|------|
| `L_landmark` | PIPNet + **HiFace 启发** | 98 点 landmark + 不确定性加权 (比 68 点多 30 个脸颊/瞳孔约束) |
| `L_pixel3dmm_uv` | Pixel3DMM | 稠密几何约束, 数万像素级 UV 对应 |
| `L_normal` | Pixel3DMM | 表面朝向一致性, 补充深度信息 |
| `L_contour` | HRN | 下颌线和颧骨轮廓加权, 人脸辨识度最高区域 |
| `L_photometric` | VHAP / flame-head-tracker | 像素颜色一致性, 全局外观约束 |
| `L_identity` | MICA (ArcFace) | **直接优化人脸识别层面的相似度** |
| `L_region_weight` | **HiFace 启发** | 投影空间区域加权: 鼻/眼区域光度+landmark 损失 3x 权重, 无需 3D GT |
| `L_reg` | 通用 | 参数正则化, 约束 shape/expression 在合理分布内 |

**关键设计 — L_region_weight (投影空间区域加权损失)**:

不同面部区域对身份辨识的贡献不均匀——鼻形、眼眶形态对"像不像"的影响远大于额头。本设计在 **投影 (2D) 空间**施加区域加权, 不需要 3D GT 顶点, 纯推理时可用:

```python
# 1. 利用 FLAME_masks.pkl 获取各区域的顶点索引
# data/pretrained/FLAME2020/FLAME_masks.pkl — 项目中已有
region_weights = {
    'nose':       3.0,   # 鼻形对身份最关键 (鼻翼宽度、鼻梁高度、鼻尖形态)
    'eye_region': 2.5,   # 眼眶深度、眼睑弧度
    'lips':       2.0,   # 嘴型、唇厚度
    'jaw':        1.5,   # 下颌角度
    'cheeks':     1.0,   # 脸颊 (基准权重)
    'forehead':   0.5,   # 额头相对平坦, 身份信息密度低
}

# 2. 将 FLAME 区域 mask 光栅化为 2D 权重图 (通过可微渲染)
weight_map = rasterize_region_weights(flame_mesh, camera, region_weights, flame_masks)
# weight_map: [H, W], 每个像素的权重由其对应的面部区域决定

# 3. 将区域权重应用到 L_photometric 和 L_landmark 上
L_photometric_weighted = weight_map * |I_rendered - I_input|   # 鼻/眼区域光度误差权重 3x
L_landmark_weighted = Σ region_weights[region_of(lmk_i)] * ||μ_i - μ̂_i||² / (2*σ_i²)
```

这个设计使优化器在鼻部/眼部区域分配更多梯度, 即使在光度损失主导全局收敛的中后期, 仍然持续细化高身份信息区域的几何精度。零额外数据, 仅需 `FLAME_masks.pkl` (项目中已有), **无需 3D GT 顶点**。

**关键设计 — L_identity (身份保持损失)**:

```python
# 将当前优化的网格通过可微渲染得到图像
rendered = differentiable_render(flame_mesh, texture, camera)
# 通过 ArcFace 提取渲染图的身份特征
feat_rendered = arcface(rendered)
# 与输入图的身份特征计算 cosine similarity
feat_input = arcface(input_image)  # 预计算, 固定
L_identity = 1 - cosine_similarity(feat_rendered, feat_input)
```

这是提升相似度最关键的设计: 将"像不像"从主观感受转化为可微分的数值目标。

### 3.4 Stage 3: 细节恢复 + 外观对齐

**目的**: 在优化后的 FLAME 基础网格上叠加高频细节, 并生成高质量纹理。确保细节只增不减相似度。

**几何细节 (HRN + HiFace 启发的张力加权)**:

1. HRN displacement 网络预测原始 displacement map `D_hrn`
2. 计算顶点张力 (vertex tension): 当前网格 vs FLAME 中性网格的边长比
3. 按张力加权衰减表情活跃区域的 displacement 幅度 → `D_refined`
4. 叠加至 FLAME mesh 法线方向, 恢复皱纹、法令纹、眼袋等个人化微细节

> **外部依赖**: HRN 预训练权重需从 Google Drive 额外下载 (`pretrained_models/hrn_v1.1/epoch_10.pth` 等, 参见 `submodules/HRN/README.md`)。项目中暂未包含。
>
> **降级方案**: 若 HRN 权重不可用, 可使用 DECA detail decoder 作为 fallback。flame-head-tracker 中已集成 DECA (含 detail decoder), 开箱可用。但注意 DECA-d 在 REALY 上精度 (2.210mm) 远低于 HRN (1.537mm), 且加 detail 后精度反而恶化, 因此**强烈建议获取 HRN 权重**。

**身份安全校验 (HiFace 启发)**:

- displacement 叠加后通过可微渲染 + ArcFace 计算 identity loss
- 若 identity loss 超过阈值, 自动缩小 displacement 幅度
- 确保不出现 DECA-d 式的"加 detail 反而降精度"

**纹理对齐 (来自 VHAP)**:

- 自适应扰动机制处理遮挡区域 (头发、耳朵、颈部)
- 光度优化生成高质量 UV 纹理贴图
- Total-variation 正则化保证纹理平滑性

## 4. 对相似度贡献排序

| 排名 | 设计 | 来源 | 作用阶段 | 原因 |
|------|------|------|---------|------|
| 1 | ArcFace identity loss | MICA | Stage 2 | 直接优化人脸识别层面的相似度, 端到端对齐 |
| 2 | Pixel3DMM 稠密 UV 约束 | Pixel3DMM | Stage 2 | 约束从 68 点扩展到数万像素, 几何精度质变 |
| 3 | 投影空间区域加权 | **HiFace 启发** | Stage 2 | 鼻/眼区域光度+landmark 3x 权重, 无需 3D GT, 纯推理可用 |
| 4 | 轮廓加权损失 | HRN | Stage 2 | 下颌线是区分"像 vs 不像"的人眼第一感知区域 |
| 5 | 顶点张力加权 displacement | **HiFace 启发** | Stage 3 | 抑制表情区域位移噪声, 防止加 detail 降精度 (零数据) |
| 6 | 98 点稠密 landmark + 不确定性加权 | **HiFace 启发** | Stage 2 | 比 68 点多 30 个约束, 遮挡降权更鲁棒 (已有 PIPNet) |
| 7 | Stage 3 identity loss 校验 | **HiFace 启发** | Stage 3 | 确保 displacement 不破坏身份, 避免 DECA-d 式恶化 |
| 8 | 多图中位数聚合 | MICA | Stage 1 | 零成本消除单图偏差, 提升初始化鲁棒性 |
| 9 | 光度优化纹理 | VHAP | Stage 3 | 外观层面的相似度补充, 视觉完整性 |

## 5. 实施路线

### Phase 1: 最小可行 Pipeline (预计 1-2 天)

- MICA 初始化 + 多图聚合
- flame-head-tracker 的 landmark fitting
- 基础可微渲染框架搭建

### Phase 2: 核心相似度提升

- 集成 Pixel3DMM 网络推理 (UV + normal 预测)
- 实现稠密约束优化循环
- 加入 ArcFace identity loss

### Phase 3: 精细化

- HRN displacement map 生成
- 顶点张力加权 displacement 精炼 (HiFace 启发)
- Stage 3 ArcFace identity loss 校验 (HiFace 启发)
- VHAP 纹理优化
- 端到端 pipeline 打通与参数调优

## 6. HiFace 技术调研 (ICCV 2023)

> 本节为 HiFace 论文的调研记录。可融合的思路已纳入 Stage 2/3 设计, 此处仅保留论文摘要与可行性结论。

**核心创新**: SD-DeTail 模块将面部细节解耦为静态 (身份特有纹理) + 动态 (表情驱动皱纹), 通过 PCA displacement basis + 顶点张力插值生成。是唯一"加 detail 后精度还提升"的方法 (REALY 1.275mm, 对比 DECA-d 2.210mm 恶化, HRN 1.537mm)。

**数据依赖 (本项目不可用)**:

| 需求 | 可获取性 |
|------|---------|
| 332 真实扫描 → PCA displacement basis | Microsoft 内部 / FaceScape 需申请, 格式不兼容 |
| 200k 合成图 + GT displacement maps | 合成管线未开源 |

**已融合到本 pipeline 的思路** (均为零数据, 纯代码):

- Stage 2: 98 点 PIPNet landmark + 不确定性加权, 投影空间区域加权损失
- Stage 3: 顶点张力加权 displacement 精炼, ArcFace identity 校验

**不可行 / 不融合**:

| 技术 | 原因 |
|------|------|
| SD-DeTail 完整复现 | 缺少 PCA basis + 合成数据 |
| 合成数据 GT 监督训练 | 合成管线未开源 |
| 年龄感知知识蒸馏 | 依赖 SD-DeTail 静态系数 |
| ResNet50 编码器 / coarse shape 回归 / SH 光照 | 已有更优方案覆盖 |

## 7. 技术栈

- **参数化模型**: FLAME 2020 (5023 顶点, 固定拓扑)
- **可微渲染**: nvdiffrast
- **深度学习**: PyTorch
- **人脸检测**: FaceBoxesV2 / InsightFace RetinaFace
- **身份编码**: ArcFace (预训练)
- **稠密特征**: DINOv2 (Pixel3DMM 网络)
- **Landmark 检测**: PIPNet 98 点 WFLW (已有预训练权重)

## 8. 代码结构

```
src/faceforge/
├── detection/
│   ├── face_detector.py          # FaceBoxesV2 人脸检测
│   └── landmark_detector.py      # 98点 landmark 检测 (PIPNet) + 不确定性加权
├── preprocessing/
│   ├── cropper.py                # 人脸裁剪 (512×512)
│   ├── segmenter.py              # 人脸分割 (facer)
│   └── arcface_preprocess.py     # ArcFace 对齐与特征提取
├── initialization/
│   ├── mica_model.py             # MICA 编码器/解码器
│   └── multi_image_aggregator.py # 多图中位数聚合
├── optimization/
│   ├── differentiable_renderer.py  # nvdiffrast 可微渲染
│   ├── losses.py                   # 全部损失函数
│   ├── identity_loss.py            # ArcFace identity loss
│   ├── region_weight.py            # 投影空间区域加权 (FLAME_masks → 2D weight map)
│   └── optimizer.py                # 迭代优化循环
├── refinement/
│   ├── vertex_tension.py         # 顶点张力计算 (HiFace 启发, 零数据)
│   ├── displacement_net.py       # HRN displacement map 预测
│   ├── displacement_refiner.py   # 张力加权 + identity 校验 displacement 精炼
│   └── texture_optimizer.py      # VHAP 纹理光度优化
├── pixel3dmm/
│   ├── network.py                # UV + normal 预测网络
│   └── dense_constraints.py      # 稠密约束计算
└── pipeline.py                   # 端到端 pipeline 入口
```

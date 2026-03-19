# Stage 1 开发计划

> 本文档是给 Claude Code 的开发指令。每个 Task 是一个独立可验证的开发单元。
> 技术细节参考 `docs/stage1-design.md`。

## 前置条件

- 项目根目录: `/Users/guanguan/Projects/FaceForge`
- Python 环境已安装 PyTorch, mediapipe, insightface, opencv-python, scipy, Pillow, trimesh
- 预训练权重: `data/pretrained/mica.tar` (502MB), `data/pretrained/mediapipe/face_landmarker.task` (3.6MB), `data/pretrained/FLAME2020/generic_model.pkl`
- 测试图片: `data/test_images/` 目录下有测试用人脸图片
- 子模块: `submodules/MICA`, `submodules/flame-head-tracker`

## 代码位置

所有代码写在 `src/faceforge/stage1/` 下。创建 `src/faceforge/__init__.py` 和 `src/faceforge/stage1/__init__.py`。

---

## Task 1: 项目骨架 + 数据类定义

**创建文件**: `src/faceforge/stage1/config.py`, `src/faceforge/stage1/data_types.py`

**config.py**:
```python
from dataclasses import dataclass, field

@dataclass
class Stage1Config:
    # 模型路径
    mica_model_path: str = 'data/pretrained/mica.tar'
    mediapipe_model_path: str = 'data/pretrained/mediapipe/face_landmarker.task'
    flame_model_path: str = 'data/pretrained/FLAME2020/generic_model.pkl'
    flame_masks_path: str = 'data/pretrained/FLAME2020/FLAME_masks.pkl'

    # 处理参数
    align_scale_factor: float = 1.3       # flame-head-tracker tracking 标准
    align_output_size: int = 512
    align_transform_size: int = 1024
    arcface_input_size: int = 112
    deca_input_size: int = 224
    render_size: int = 512

    # 多图聚合
    aggregation_method: str = 'median'    # 'median' 或 'mean'

    # 输出控制
    output_dir: str = 'output'
    save_debug: bool = True
    save_mesh: bool = True
    save_summary: bool = True

    # 设备
    device: str = 'cuda:0'
```

**data_types.py**: 定义 `Stage1Output` dataclass (参考 stage1-design.md 第 11 节的完整字段定义, 包含 shape/expression/pose/texture/lighting/arcface_feat/aligned_image/face_mask/landmarks/camera 所有字段)。同时定义 `DetectionResult` dataclass 存放检测结果 (lmks_dense, lmks_68, lmks_eyes, blend_scores, retinaface_kps, bbox)。

**验证**: `python -c "from faceforge.stage1.config import Stage1Config; print(Stage1Config())"`

---

## Task 2: MediaPipe 检测 + 478→68 转换

**创建文件**: `src/faceforge/stage1/detection.py`, `src/faceforge/stage1/mp2dlib.py`

**mp2dlib.py**: 从 `submodules/flame-head-tracker/utils/mp2dlib.py` 移植 `mp2dlib_correspondence` 映射表和 `convert_landmarks_mediapipe_to_dlib()` 函数。保持逻辑不变。

**detection.py**: 实现 `MediaPipeDetector` 类:

```python
class MediaPipeDetector:
    def __init__(self, model_path: str):
        # 初始化 MediaPipe FaceLandmarker
        # 参考 flame-head-tracker/tracker_base.py L100-106
        # output_face_blendshapes=True, num_faces=1

    def detect(self, image_rgb: np.ndarray) -> DetectionResult:
        # 1. MediaPipe 检测 → 478 点 landmark (像素坐标)
        # 2. 调用 mp2dlib 转换 → 68 点
        # 3. 提取眼部 landmark (MediaPipe idx 468-472, 473-477)
        # 4. 提取 52 个 blendshape 分数
        # 5. 多人脸时选择最接近图像中心的 (参考 pixel3dmm get_center 逻辑)
        # 返回 DetectionResult
```

**验证**: 加载测试图片, 运行检测, 打印 68 点坐标, 确认形状 [68, 2] 且值在图像范围内。

---

## Task 3: RetinaFace 检测 (MICA 专用 5 点)

**修改文件**: `src/faceforge/stage1/detection.py`

新增 `RetinaFaceDetector` 类:

```python
class RetinaFaceDetector:
    def __init__(self, device: str = 'cuda:0'):
        # 初始化 InsightFace FaceAnalysis
        # 参考 pixel3dmm/preprocessing/MICA/utils/landmark_detector.py L39-40
        # app = FaceAnalysis(name='antelopev2', providers=[...])
        # app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_5pt(self, image_bgr: np.ndarray) -> np.ndarray:
        # 返回 [5, 2] 原生 5 点 landmark
        # 多人脸时选最接近中心的
```

新增统一入口 `detect_all()`:
```python
def detect_all(image_rgb, mp_detector, retina_detector) -> DetectionResult:
    # 1. MediaPipe 检测 → 478/68/eyes/blendshapes
    # 2. RetinaFace 检测 → 原生 5 点
    # 3. 合并到 DetectionResult (新增 retinaface_kps 字段)
```

**验证**: 同一张图片, 打印 MediaPipe 68→5 点转换结果 vs RetinaFace 原生 5 点, 确认两者接近但不完全相同。

---

## Task 4: 人脸对齐裁剪 (image_align)

**创建文件**: `src/faceforge/stage1/alignment.py`

从 `submodules/flame-head-tracker/tracker_base.py` 的 `image_align()` 函数 (L133-233) 移植。这是一个独立的纯函数, 依赖 numpy, PIL, scipy。

```python
def image_align(
    img: np.ndarray,           # RGB uint8 [H, W, 3]
    face_landmarks: np.ndarray, # [68, 2] 或 [68, 3]
    output_size: int = 512,
    transform_size: int = 1024,
    scale_factor: float = 1.3,  # tracking 标准
    padding_mode: str = 'constant',
) -> tuple[np.ndarray, np.ndarray]:
    # 返回 (aligned_image [512,512,3], transform_matrix [3,3])
```

关键实现细节 (参考 stage1-design.md 第 4 节):
- 从 68 点提取眼/嘴子集, 计算旋转角度
- shrink = floor(qsize / output_size * 0.5)
- border = max(round(qsize * 0.1), 3)
- 填充 + 高斯模糊 (sigma = qsize * 0.02)
- PIL Image.QUAD + BILINEAR 插值
- 最终 resize 到 output_size

**验证**: 对齐一张侧脸图片, 输出 512×512 图应该是接近正面的、居中的人脸。保存到文件目视检查。

---

## Task 5: 人脸分割 (BiSeNet)

**创建文件**: `src/faceforge/stage1/segmentation.py`

```python
class FaceParser:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        # 加载 BiSeNet 模型
        # 参考 flame-head-tracker 的 face_parsing 子模块

    def parse(self, image_rgb_512: np.ndarray) -> np.ndarray:
        # 输入: 512×512 RGB 图
        # 预处理: ImageNet 归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        # 返回: [512, 512] int, 0-18 类别标签

    @staticmethod
    def extract_face_mask(parsing: np.ndarray, keep_mouth: bool = True) -> np.ndarray:
        # 保留: skin(1), brow(2-3), eye(4-5), nose(10), lip(12-13)
        # 可选保留: mouth(11)
        # 排除: background(0), glasses(6), ear(7-9), neck(14-15), cloth(16), hair(17), hat(18)
        # 返回: [512, 512] bool
```

**注意**: BiSeNet 权重文件位置需要确认。检查 `submodules/flame-head-tracker/` 中的 face_parsing 模块, 找到权重文件路径, 如果不存在需要记录。

**验证**: 对齐图 → 分割 → 可视化 mask overlay, 确认面部区域正确。

---

## Task 6: MICA 推理

**创建文件**: `src/faceforge/stage1/mica_inference.py`

```python
class MICAInference:
    def __init__(self, config: Stage1Config):
        # 1. 添加 submodules/MICA 到 sys.path (用主副本, 不用嵌套副本)
        # 2. 加载 MICA 模型:
        #    from micalib.models.mica import MICA
        #    from configs.config import get_cfg_defaults
        #    cfg.model.testing = True
        #    cfg.pretrained_model_path = config.mica_model_path
        #    注意: 需 patch torch.load 添加 weights_only=False (PyTorch >= 2.6)
        # 3. model.eval()

    def get_arcface_input(self, image_bgr: np.ndarray, kps_5pt: np.ndarray) -> np.ndarray:
        # RetinaFace 原生 5 点 → insightface norm_crop → 112×112
        # cv2.dnn.blobFromImages: (x - 127.5) / 127.5 → [-1, 1], swapRB=True
        # 返回: blob [1, 3, 112, 112]

    def encode(self, image_rgb: np.ndarray, arcface_blob: np.ndarray) -> dict:
        # image_rgb → /255.0 → resize 224×224 → tensor [1, 3, 224, 224]
        # arcface_blob → tensor [1, 1, 3, 112, 112]
        # mica.encode(images, arcface) → F.normalize(arcface_output)
        # mica.decode(codedict)
        # 返回: {'shape_code': [1, 300], 'vertices': [1, 5023, 3], 'arcface_feat': [1, 512]}

    def run(self, image_bgr: np.ndarray, retinaface_kps: np.ndarray) -> dict:
        # 完整流程: get_arcface_input → encode → 返回结果
        # 全程内存, 不存中间 JPEG
```

**关键**: `arcface_feat` (512 维 L2 归一化向量) 要一并返回, Stage 2 的 L_identity 需要它。

**验证**:
1. 打印 shape_code 形状 [1, 300], 值范围应在 [-3, 3] 内
2. 打印 vertices 形状 [1, 5023, 3]
3. 保存 mesh.obj, 用 MeshLab 打开确认是合理的人脸形状

---

## Task 7: DECA 推理

**创建文件**: `src/faceforge/stage1/deca_inference.py`

```python
class DECAInference:
    def __init__(self, config: Stage1Config):
        # 添加 flame-head-tracker 路径到 sys.path
        # 从 flame-head-tracker/submodules/decalib 加载 DECA
        # 参考 flame-head-tracker/utils/deca_inference_utils.py

    def run(self, image_rgb: np.ndarray) -> dict:
        # DECA 内部会用 FAN 检测+裁剪 (无法绕过)
        # 返回:
        # {
        #   'exp': [1, 50],      # expression
        #   'pose': [1, 6],      # 前3=head, 后3=jaw
        #   'tex': [1, 50],      # texture
        #   'light': [1, 27],    # SH coefficients
        #   'cam': [1, 3],       # (丢弃)
        #   'deca_crop': np.ndarray  # 224×224 裁剪图 (用于调试输出)
        # }
```

**注意**: DECA 依赖 face_alignment 库 (FAN)。确认 flame-head-tracker 中 decalib 的 import 路径, 可能需要调整 sys.path。

**验证**: 打印 exp/pose/tex/light 的形状和值范围。pose 的值应在 [-1, 1] 弧度量级。

---

## Task 8: 参数合并 + 多图聚合

**创建文件**: `src/faceforge/stage1/merge.py`, `src/faceforge/stage1/aggregation.py`

**merge.py**:
```python
def merge_params(mica_output: dict, deca_output: dict) -> dict:
    # shape: MICA 的完整 300 维
    # exp: [1, 100] — DECA 前 50 维, 后 50 维填零
    # head_pose: DECA pose[:, :3]
    # jaw_pose: DECA pose[:, 3:]
    # tex: [1, 50] — DECA
    # light: [1, 9, 3] — DECA reshape
    # 参考 stage1-design.md 第 8.1 节的精确逻辑
```

**aggregation.py**:
```python
def aggregate_shapes(
    images: list[np.ndarray],
    mica: MICAInference,
    retina_detector: RetinaFaceDetector,
    method: str = 'median'
) -> torch.Tensor:
    # 对每张图: RetinaFace 5点 → MICA → shape_code
    # 堆叠 [N, 300] → median 或 mean → [300]
```

**验证**: 用 3 张同一人不同角度的图片, 打印各图 shape_code 的 L2 距离, 确认聚合后更稳定。

---

## Task 9: 调试输出 (可视化)

**创建文件**: `src/faceforge/stage1/visualization.py`

```python
class Stage1Visualizer:
    def __init__(self, output_dir: str, subject_name: str):
        # 创建 output/{subject_name}/stage1/01_detection/ ... 06_merged/ 目录结构

    def save_detection(self, image, detection_result):
        # 保存 input.png, mediapipe_478.png, landmarks_68.png, retinaface_5pt.png
        # 在图上绘制 landmark 点 (绿色圆点, 带编号)

    def save_alignment(self, original_image, aligned_image, lmks_before, lmks_after):
        # 保存 aligned_512.png, alignment_grid.png (左右对比)

    def save_segmentation(self, aligned_image, parsing, face_mask):
        # 保存 parsing_vis.png (19类彩色), face_mask.png, mask_overlay.png

    def save_mica(self, arcface_img, shape_code, vertices, faces, aligned_image):
        # 保存 arcface_112.png, shape_code.npy, mesh.obj
        # 渲染 mesh_front.png, mesh_side.png, mesh_overlay.png
        # mesh 渲染: 用 pyrender 或简单的深度 buffer 即可

    def save_deca(self, deca_crop, deca_params):
        # 保存 deca_crop_224.png, params.json

    def save_merged(self, flame_params, vertices, faces, aligned_image, arcface_feat, input_image):
        # 保存 flame_params.npz, mesh_final.obj
        # 渲染 render_front.png, render_overlay.png
        # 计算并保存 identity_similarity.txt (arcface cosine sim)

    def save_summary(self):
        # 拼接 6 格总览图: 输入 → landmark → 对齐 → 分割 → MICA mesh → 最终叠加
```

**mesh 渲染方案**: 使用 `pyrender` 或 `trimesh` 的 offscreen 渲染。如果 GPU 渲染不可用, 退化到 `trimesh.scene.Scene` 的软件渲染。

**验证**: 运行完整 pipeline, 检查输出目录下所有图片是否正确生成。

---

## Task 10: Pipeline 总入口

**创建文件**: `src/faceforge/stage1/pipeline.py`

```python
class Stage1Pipeline:
    def __init__(self, config: Stage1Config = None):
        # 初始化所有模型:
        # - MediaPipeDetector
        # - RetinaFaceDetector
        # - FaceParser (BiSeNet)
        # - MICAInference
        # - DECAInference
        # - Stage1Visualizer (if save_debug)

    def run_single(self, image_rgb: np.ndarray, subject_name: str = 'default') -> Stage1Output:
        # Step 1: detect_all() → DetectionResult
        # Step 2: image_align() → aligned_512
        # Step 3: face_parser.parse() → parsing, face_mask
        # Step 4a: mica.run() → shape_code, vertices, arcface_feat
        # Step 4b: deca.run() → exp, pose, tex, light
        # Step 5: merge_params() → flame_params
        # 调试输出: visualizer.save_xxx() (if save_debug)
        # 构造并返回 Stage1Output

    def run_multi(self, images_rgb: list[np.ndarray], subject_name: str = 'default') -> Stage1Output:
        # 对每张图运行 run_single 的检测+MICA 部分
        # aggregate_shapes() 聚合 shape
        # 用第一张图的 DECA/alignment/segmentation 结果
        # 用聚合后的 shape 替换
        # 保存 per_image/ 和 aggregation/ 调试输出
        # 返回 Stage1Output
```

**验证**:
```python
from faceforge.stage1.pipeline import Stage1Pipeline
pipeline = Stage1Pipeline()

# 单图测试
result = pipeline.run_single(cv2.imread('data/test_images/face1.jpg')[:,:,::-1], 'test_subject')
print(result.shape.shape)  # torch.Size([1, 300])

# 多图测试
images = [cv2.imread(p)[:,:,::-1] for p in glob('data/test_images/same_person_*.jpg')]
result = pipeline.run_multi(images, 'test_multi')
```

检查 `output/test_subject/stage1/summary.png` 和 `output/test_subject/stage1/06_merged/render_overlay.png`。

---

## Task 11: FLAME → HIFI3D++ 转换 (REALY 评测用)

**创建文件**: `src/faceforge/stage1/realy_eval.py`

**背景**: REALY benchmark 在 HIFI3D++ 拓扑上评测, 需要将 FLAME mesh 转换到 HIFI3D++ 空间。REALY 提供了预定义的 85 个关键点和对应的重心坐标文件, 用于不同拓扑间的对齐。

**数据文件** (从 https://github.com/czh-98/REALY 获取):
- `data/realy/FLAME.obj` — FLAME 模板 mesh
- `data/realy/FLAME.txt` — FLAME 拓扑上 85 个关键点的重心坐标 (格式: `[triangle_id, w1, w2]`, 85 行)
- `data/realy/HIFI3D.obj` — HIFI3D++ 模板 mesh
- `data/realy/HIFI3D.txt` — HIFI3D++ 拓扑上 85 个关键点的重心坐标
- `data/realy/metrical_scale.txt` — 每个 subject 的尺度因子

**85 关键点的区域分布**:
```python
keypoints_region_map = {
    'forehead': list(range(36, 48)) + list(range(17, 27)),  # 22 点
    'nose':     list(range(27, 36)),                         # 9 点
    'mouth':    list(range(48, 61)) + [64],                  # 14 点
    'cheek':    [...],                                       # 27 点 (多区域合并)
    'seven_keypoints': [36, 39, 42, 45, 33, 48, 54],        # 全局对齐用 7 点
}
```

**实现**:

```python
class REALYEvaluator:
    def __init__(self, realy_data_dir: str = 'data/realy'):
        # 加载 FLAME.obj, FLAME.txt, HIFI3D.obj, HIFI3D.txt, metrical_scale.txt

    def extract_85_keypoints(self, vertices: np.ndarray, topology: str = 'FLAME') -> np.ndarray:
        """从 mesh 顶点中用重心坐标提取 85 个关键点。

        Args:
            vertices: [5023, 3] FLAME 顶点 或 HIFI3D 顶点
            topology: 'FLAME' 或 'HIFI3D', 决定用哪个 .txt 重心坐标文件
        Returns:
            keypoints: [85, 3] 关键点坐标

        原理: 对每个关键点, .txt 文件提供 (triangle_id, w1, w2)
              w3 = 1 - w1 - w2
              三角形的三个顶点 v1, v2, v3 = faces[triangle_id]
              keypoint = w1 * vertices[v1] + w2 * vertices[v2] + w3 * vertices[v3]
        """

    def global_align(self, predicted_kps_85: np.ndarray, gt_kps_85: np.ndarray) -> np.ndarray:
        """用 7 个关键点 (眼角×4 + 鼻尖 + 嘴角×2) 做刚性对齐 (Procrustes)。

        Args:
            predicted_kps_85: [85, 3] 预测 mesh 的 85 关键点
            gt_kps_85: [85, 3] GT scan 的 85 关键点
        Returns:
            aligned_vertices: 对齐后的预测 mesh 顶点
        """

    def regional_align_and_eval(
        self,
        predicted_mesh: dict,        # {'v': [N,3], 'f': [M,3]}
        gt_scan_regions: dict,       # {'nose': mesh, 'mouth': mesh, ...}
        gt_kps_85: np.ndarray,
    ) -> dict:
        """REALY 的区域对齐 + 评测。

        对 nose/mouth/forehead/cheek 四个区域:
        1. 用该区域的关键点子集做局部 ICP 对齐
        2. 计算双向 NMSE (normalized mean squared error)

        Returns:
            {'nose': float, 'mouth': float, 'forehead': float, 'cheek': float, 'all': float}
        """

    def evaluate(
        self,
        flame_vertices: np.ndarray,  # [5023, 3] Stage 1 输出
        subject_id: int,             # REALY subject ID
        realy_scan_dir: str,         # REALY GT scan 目录
    ) -> dict:
        """完整评测流程:
        1. extract_85_keypoints(flame_vertices, 'FLAME')
        2. 加载 GT scan 和 GT 85 关键点
        3. global_align()
        4. regional_align_and_eval()
        5. 乘以 metrical_scale 转为 mm
        """
```

**REALY 评测管线 (参考 REALY/main.py)**:

```
FLAME mesh (5023 顶点)
    │
    ▼
extract_85_keypoints(vertices, 'FLAME')  ← FLAME.txt 重心坐标
    │  [85, 3] 关键点
    ▼
global_align() ← 7 点刚性对齐 (Procrustes)
    │  全局对齐后的 mesh
    ▼
regional_align_and_eval() ← 4 区域局部 ICP
    │  nose/mouth/forehead/cheek NMSE
    ▼
× metrical_scale → mm 单位误差
```

**ICP 实现**: 从 REALY 仓库移植 `utils/gICP.py` (全局 ICP) 和 `utils/rICP.py` (区域 ICP)。或者用 `open3d.pipelines.registration` 替代 (更稳定, API 更清晰)。

**输出** (追加到 `output/{subject}/stage1/`):

```
07_realy_eval/
├── flame_85kps.npy          # FLAME mesh 上提取的 85 关键点
├── aligned_mesh.obj         # 全局对齐后的 mesh
├── region_errors.json       # {'nose': 1.05, 'mouth': 1.46, 'forehead': 1.33, 'cheek': 1.29, 'all': 1.28}
└── region_vis/
    ├── nose_error_map.png   # 鼻部误差热力图
    ├── mouth_error_map.png
    ├── forehead_error_map.png
    └── cheek_error_map.png
```

**验证**:
1. 用 MICA 原始输出跑 REALY 评测, 确认数值与 MICA 论文报告的 NoW/REALY 指标接近
2. 对比 REALY 官方脚本的输出, 确认误差数值一致

**注意**:
- REALY benchmark 的 GT scan 数据需要从 Headspace 数据集申请下载, 评测时才需要
- 85 关键点提取和全局对齐可以在没有 GT 的情况下独立运行和调试
- `FLAME.txt` 的格式是每行 `[triangle_id, w1, w2]`, 共 85 行, w3 隐式 = 1 - w1 - w2

---

## Task 依赖关系

```
Task 1 (骨架)
  ├── Task 2 (MediaPipe) ──┐
  ├── Task 3 (RetinaFace) ─┤
  │                        ├── Task 4 (对齐)
  │                        │     └── Task 5 (分割)
  │                        ├── Task 6 (MICA) ───┐
  │                        └── Task 7 (DECA) ───┤
  │                                             ├── Task 8 (合并+聚合)
  │                                             │     └── Task 10 (Pipeline)
  └── Task 9 (可视化) ──────────────────────────┘
                                                      │
                                          Task 11 (REALY 评测) ← 依赖 Task 6 的 FLAME mesh 输出
```

Task 2/3 可并行, Task 6/7 可并行, Task 9 可在任意时候开发, Task 11 只依赖 MICA 输出。

## 注意事项

1. **sys.path 管理**: MICA 和 DECA 的 import 依赖 sys.path。在每个 inference 模块的 `__init__` 中添加路径, 使用后不要污染全局。考虑用 `contextlib` 或在模块顶部处理。

2. **weights_only=False**: MICA 的 `mica.tar` 需要 `torch.load(..., weights_only=False)` 才能在 PyTorch >= 2.6 下加载。直接 patch `submodules/MICA/micalib/models/mica.py` 的 `load_model()` 方法。

3. **BGR vs RGB**: cv2 读入为 BGR。MediaPipe 需要 RGB, RetinaFace 需要 BGR, DECA 内部处理 RGB, MICA ArcFace 用 swapRB=True。在每个函数签名中明确标注输入的色彩空间。

4. **FLAME 模型文件**: MICA 内部会加载 `generic_model.pkl`。确认 MICA config 中的 `flame_model_path` 指向 `data/pretrained/FLAME2020/generic_model.pkl`。

5. **GPU 内存**: MICA + DECA + MediaPipe + RetinaFace 同时加载约需 2-3GB 显存。如果显存不足, 可以在 MICA 推理完后释放, 再加载 DECA。

6. **测试图片**: 使用 `data/test_images/` 下的图片进行验证。如果不存在, 用任意包含人脸的 JPEG 图片即可。

7. **REALY 数据**: Task 11 需要从 https://github.com/czh-98/REALY/tree/master/data 下载 `FLAME.obj`, `FLAME.txt`, `HIFI3D.obj`, `HIFI3D.txt`, `metrical_scale.txt` 放到 `data/realy/`。GT scan 数据需从 Headspace 数据集申请, 仅评测时需要。

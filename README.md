# FaceForge

Identity-aware 3D face reconstruction pipeline.
**Stage 1** extracts FLAME parameters from one or more face images using MICA (shape) and DECA (expression / pose / texture / lighting).

---

## Table of contents

- [Project structure](#project-structure)
- [Environment setup](#environment-setup)
- [Model weights](#model-weights)
- [Running the pipeline](#running-the-pipeline)
- [Testing](#testing)

---

## Project structure

```
FaceForge/
├── src/faceforge/
│   └── stage1/            # Stage 1 pipeline
│       ├── pipeline.py    # Stage1Pipeline (entry point)
│       ├── config.py      # Stage1Config dataclass
│       ├── data_types.py  # DetectionResult, Stage1Output
│       ├── detection.py   # MediaPipe + RetinaFace detectors
│       ├── alignment.py   # FFHQ-style face alignment
│       ├── segmentation.py# BiSeNet face parsing
│       ├── mica_inference.py  # MICA shape reconstruction
│       ├── deca_inference.py  # DECA expression/pose/tex/light
│       ├── merge.py       # Combine MICA + DECA outputs
│       └── aggregation.py # Multi-image shape aggregation
├── submodules/
│   ├── MICA/              # MICA submodule
│   ├── decalib/           # DECA submodule
│   └── face_parsing/      # BiSeNet submodule
├── data/pretrained/       # Model weights (not tracked by git)
├── scripts/
│   └── run_stage1.py      # Quick-test CLI script
├── tests/
│   └── stage1/
│       ├── test_unit.py        # Unit tests (no GPU required)
│       └── test_integration.py # Integration tests (require weights)
└── pyproject.toml
```

---

## Environment setup

```bash
# 1. Clone and initialise submodules
git clone <repo> FaceForge
cd FaceForge
git submodule update --init --recursive

# 2. Create a conda environment (Python 3.10+)
conda create -n faceforge python=3.10 -y
conda activate faceforge

# 3. PyTorch — pick the right CUDA version for your system
#    CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Core dependencies
pip install opencv-python numpy scipy scikit-image Pillow
pip install mediapipe
pip install insightface onnxruntime       # CPU
# pip install insightface onnxruntime-gpu  # GPU

# 5. DECA renderer dependencies
pip install pytorch3d kornia

# 6. (Optional) REALY benchmark evaluation
pip install open3d

# 7. Install this package in editable mode
pip install -e .
```

---

## Model weights

Download the following files into `data/pretrained/` before running:

| File | Source |
|------|--------|
| `mica.tar` | [MICA releases](https://github.com/Zielon/MICA/releases) |
| `deca_model.tar` + texture data | [DECA data.zip](https://github.com/yfeng95/DECA) |
| `79999_iter.pth` | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| `FLAME2020/generic_model.pkl` | [FLAME website](https://flame.is.tue.mpg.de/) (registration required) |
| `FLAME2020/head_template.obj` | included in DECA data.zip |
| `FLAME2020/landmark_embedding.npy` | included in DECA data.zip |
| `FLAME_albedo_from_BFM.npz` | included in DECA data.zip |
| `mean_texture.jpg` | included in DECA data.zip |
| `texture_data_256.npy` | included in DECA data.zip |
| `fixed_displacement_256.npy` | included in DECA data.zip |
| `uv_face_mask.png` | included in DECA data.zip |
| `uv_face_eye_mask.png` | included in DECA data.zip |
| `mediapipe/face_landmarker.task` | [MediaPipe models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models) |

InsightFace's **antelopev2** model is downloaded automatically on first run into `~/.insightface/`.

Expected directory layout after downloading:

```
data/pretrained/
├── mica.tar
├── deca_model.tar
├── 79999_iter.pth
├── FLAME_albedo_from_BFM.npz
├── mean_texture.jpg
├── texture_data_256.npy
├── fixed_displacement_256.npy
├── uv_face_mask.png
├── uv_face_eye_mask.png
├── FLAME2020/
│   ├── generic_model.pkl
│   ├── head_template.obj
│   └── landmark_embedding.npy
└── mediapipe/
    └── face_landmarker.task
```

---

## Running the pipeline

### Quick test with the CLI script

```bash
# Single image (CPU) — using the bundled test assets
python scripts/run_stage1.py --image assets/tom/1.jpeg

# Single image (GPU) with debug output
python scripts/run_stage1.py \
    --image assets/tom/1.jpeg \
    --device cuda:0 \
    --debug \
    --subject tom

# Multi-image shape aggregation (3 images of the same person)
python scripts/run_stage1.py \
    --image assets/tom/1.jpeg assets/tom/2.jpeg assets/tom/3.jpeg \
    --subject tom \
    --aggregation median
```

The script prints all output tensor shapes and value ranges, and optionally saves debug images to `output/<subject>/stage1/`.

### Python API

```python
import cv2
from faceforge.stage1 import Stage1Config, Stage1Pipeline

config = Stage1Config(device='cpu', save_debug=True, output_dir='output')
pipeline = Stage1Pipeline(config)

img = cv2.cvtColor(cv2.imread('face.jpg'), cv2.COLOR_BGR2RGB)
result = pipeline.run_single(img, subject_name='alice')

print(result.shape.shape)        # [1, 300]  — MICA shape code
print(result.expression.shape)   # [1, 100]  — expression (50 DECA + 50 zeros)
print(result.head_pose.shape)    # [1, 3]    — head rotation
print(result.jaw_pose.shape)     # [1, 3]    — jaw rotation
print(result.arcface_feat.shape) # [1, 512]  — identity feature
```

### Debug output

When `save_debug=True` the pipeline writes per-step visualisations:

```
output/alice/stage1/
├── 01_detection/     landmarks overlaid on the input image
├── 02_alignment/     FFHQ-style aligned crop
├── 03_segmentation/  BiSeNet parsing map + binary face mask
├── 04_mica/          ArcFace-aligned crop + shape code stats
├── 05_deca/          DECA crop + parameter values
├── 06_merged/        merged FLAME parameters
└── summary.jpg       6-panel overview
```

---

## Testing

### Unit tests (no GPU or model weights needed)

Tests pure Python logic: parameter merging, face mask extraction, landmark conversion, ArcFace crop, FFHQ alignment, shape aggregation.

```bash
pytest tests/stage1/test_unit.py -v
```

### Integration tests (require weights + a test image)

```bash
# The bundled assets/tom/ images work as test images
export FACEFORGE_TEST_IMAGE=assets/tom/1.jpeg
export FACEFORGE_DEVICE=cpu          # or cuda:0

# All integration tests (detection → segmentation → MICA → DECA)
pytest tests/stage1/test_integration.py -v -m integration

# Skip the slow full-pipeline tests
pytest tests/stage1/test_integration.py -v -m "integration and not slow"

# Full pipeline including multi-image
pytest tests/stage1/test_integration.py -v -m "integration or slow"
```

Available environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FACEFORGE_TEST_IMAGE` | — | Path to a test face image (**required** for integration tests) |
| `FACEFORGE_DEVICE` | `cpu` | PyTorch device |
| `FACEFORGE_DATA_DIR` | `<project_root>/data` | Data directory |

### Run everything

```bash
# Unit tests only (fast, no setup required)
pytest tests/stage1/test_unit.py

# Everything — uses bundled test image
export FACEFORGE_TEST_IMAGE=assets/tom/1.jpeg
pytest tests/ -v
```

### Expected output (unit tests)

```
tests/stage1/test_unit.py::TestMergeParams::test_output_shapes PASSED
tests/stage1/test_unit.py::TestMergeParams::test_shape_comes_from_mica PASSED
...
tests/stage1/test_unit.py::TestAggregateShapes::test_median_aggregation PASSED
tests/stage1/test_unit.py::TestAggregateShapes::test_no_faces_raises PASSED

========== 30 passed in 0.8s ==========
```

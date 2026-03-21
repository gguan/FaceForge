from dataclasses import dataclass, field

from faceforge._paths import default_device


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

    # 设备 (自动选择 cuda → mps → cpu)
    device: str = field(default_factory=default_device)

"""PIPNet 98-point landmark inference using pixel3dmm's original code path."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

from faceforge._paths import PROJECT_ROOT


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _ensure_pixel3dmm_src_path(code_base: str) -> list[str]:
    """Add pixel3dmm's src directory to sys.path for package imports."""
    code_base_path = _resolve(code_base)
    src_path = code_base_path / 'src'

    added: list[str] = []
    entry_str = str(src_path)
    if entry_str not in sys.path:
        sys.path.insert(0, entry_str)
        added.append(entry_str)
    return added


def _ensure_scipy_simps_compat() -> None:
    """Backfill scipy.integrate.simps on newer SciPy releases."""
    import scipy.integrate

    if not hasattr(scipy.integrate, 'simps') and hasattr(scipy.integrate, 'simpson'):
        scipy.integrate.simps = scipy.integrate.simpson


def _purge_faceboxes_alias_modules(faceboxes_root: Path) -> None:
    """Remove script-style FaceBoxes aliases from sys.modules.

    ``faceboxes_detector.py`` imports ``detector`` and ``utils.*`` as top-level
    modules. We only need those during import; keeping them around shadows other
    projects like MICA that also import ``utils``.
    """
    faceboxes_root = faceboxes_root.resolve()
    for name, module in list(sys.modules.items()):
        if name != 'detector' and not name.startswith('utils'):
            continue
        module_file = getattr(module, '__file__', None)
        if module_file is None:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue
        if module_path == faceboxes_root or faceboxes_root in module_path.parents:
            sys.modules.pop(name, None)


def _import_faceboxes_detector_class(code_base: str):
    """Import FaceBoxesDetector while keeping script-style aliases isolated."""
    code_base_path = _resolve(code_base)
    faceboxes_root = code_base_path / 'src' / 'pixel3dmm' / 'preprocessing' / 'PIPNet' / 'FaceBoxesV2'

    added: list[str] = []
    injected_modules: list[str] = []
    faceboxes_entry = str(faceboxes_root)
    if faceboxes_entry not in sys.path:
        sys.path.insert(0, faceboxes_entry)
        added.append(faceboxes_entry)

    try:
        try:
            importlib.import_module('utils.nms.cpu_nms')
        except ModuleNotFoundError as exc:
            if exc.name != 'utils.nms.cpu_nms':
                raise
            from utils.nms.py_cpu_nms import py_cpu_nms

            cpu_nms_module = types.ModuleType('utils.nms.cpu_nms')
            cpu_nms_module.cpu_nms = py_cpu_nms
            cpu_nms_module.cpu_soft_nms = py_cpu_nms
            sys.modules['utils.nms.cpu_nms'] = cpu_nms_module
            injected_modules.append('utils.nms.cpu_nms')

        from pixel3dmm.preprocessing.PIPNet.FaceBoxesV2.faceboxes_detector import FaceBoxesDetector
        return FaceBoxesDetector
    finally:
        for entry in reversed(added):
            if entry in sys.path:
                sys.path.remove(entry)
        for name in injected_modules:
            sys.modules.pop(name, None)
        _purge_faceboxes_alias_modules(faceboxes_root)


class PIPNetLandmarkDetector:
    """Run the original WFLW PIPNet model on a single image."""

    def __init__(
        self,
        code_base: str = 'submodules/pixel3dmm',
        device: str = 'cpu',
        experiment_path: str = 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py',
    ):
        self.code_base = str(_resolve(code_base))
        self.pixel3dmm_paths = _ensure_pixel3dmm_src_path(self.code_base)
        _ensure_scipy_simps_compat()

        os.environ.setdefault('PIXEL3DMM_CODE_BASE', self.code_base)
        os.environ.setdefault('PIXEL3DMM_PREPROCESSED_DATA', str(PROJECT_ROOT / 'output'))
        os.environ.setdefault('PIXEL3DMM_TRACKING_OUTPUT', str(PROJECT_ROOT / 'output'))

        import pixel3dmm.env_paths as env_paths

        env_paths.CODE_BASE = self.code_base

        from pixel3dmm.preprocessing.PIPNet.lib.functions import forward_pip, get_meanface
        from pixel3dmm.preprocessing.PIPNet.lib.mobilenetv3 import mobilenetv3_large
        from pixel3dmm.preprocessing.PIPNet.lib.networks import (
            Pip_mbnetv2,
            Pip_mbnetv3,
            Pip_resnet18,
            Pip_resnet50,
            Pip_resnet101,
        )
        FaceBoxesDetector = _import_faceboxes_detector_class(self.code_base)

        self.forward_pip = forward_pip
        self.get_meanface = get_meanface
        self.mobilenetv3_large = mobilenetv3_large
        self.network_fns = {
            'resnet18': lambda cfg: Pip_resnet18(
                tv_models.resnet18(pretrained=cfg.pretrained),
                cfg.num_nb,
                num_lms=cfg.num_lms,
                input_size=cfg.input_size,
                net_stride=cfg.net_stride,
            ),
            'resnet50': lambda cfg: Pip_resnet50(
                tv_models.resnet50(pretrained=cfg.pretrained),
                cfg.num_nb,
                num_lms=cfg.num_lms,
                input_size=cfg.input_size,
                net_stride=cfg.net_stride,
            ),
            'resnet101': lambda cfg: Pip_resnet101(
                tv_models.resnet101(pretrained=cfg.pretrained),
                cfg.num_nb,
                num_lms=cfg.num_lms,
                input_size=cfg.input_size,
                net_stride=cfg.net_stride,
            ),
            'mobilenet_v2': lambda cfg: Pip_mbnetv2(
                tv_models.mobilenet_v2(pretrained=cfg.pretrained),
                cfg.num_nb,
                num_lms=cfg.num_lms,
                input_size=cfg.input_size,
                net_stride=cfg.net_stride,
            ),
            'mobilenet_v3': lambda cfg: self._build_mobilenet_v3(cfg, Pip_mbnetv3),
        }

        experiment_name = experiment_path.split('/')[-1][:-3]
        data_name = experiment_path.split('/')[-2]
        config_path = f'.experiments.{data_name}.{experiment_name}'
        my_config = importlib.import_module(config_path, package='pixel3dmm.preprocessing.PIPNet')
        Config = getattr(my_config, 'Config')
        self.cfg = Config()
        self.cfg.experiment_name = experiment_name
        self.cfg.data_name = data_name

        use_gpu = 'cuda' in device and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.detector = FaceBoxesDetector(
            'FaceBoxes',
            f'{self.code_base}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth',
            use_gpu,
            self.device,
        )
        self.threshold = 0.6
        self.det_box_scale = 1.2
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.cfg.input_size, self.cfg.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        meanface_path = os.path.join(
            self.code_base,
            'src',
            'pixel3dmm',
            'preprocessing',
            'PIPNet',
            'data',
            self.cfg.data_name,
            'meanface.txt',
        )
        _, reverse_index1, reverse_index2, max_len = self.get_meanface(meanface_path, self.cfg.num_nb)
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2
        self.max_len = max_len

        self.net = self.network_fns[self.cfg.backbone](self.cfg).to(self.device)
        save_dir = os.path.join(
            self.code_base,
            'src',
            'pixel3dmm',
            'preprocessing',
            'PIPNet',
            'snapshots',
            self.cfg.data_name,
            self.cfg.experiment_name,
        )
        weight_file = os.path.join(save_dir, f'epoch{self.cfg.num_epochs - 1}.pth')
        state_dict = torch.load(weight_file, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def _build_mobilenet_v3(self, cfg, builder):
        model = self.mobilenetv3_large()
        if cfg.pretrained:
            model.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        return builder(
            model,
            cfg.num_nb,
            num_lms=cfg.num_lms,
            input_size=cfg.input_size,
            net_stride=cfg.net_stride,
        )

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Predict 98 WFLW landmarks in pixel coordinates."""
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_height, image_width = image_bgr.shape[:2]

        detections, _ = self.detector.detect(image_bgr, self.threshold, 1)
        detections = [det for det in detections if det[0] == 'face']
        detections.sort(key=lambda x: -1 * x[1])
        if not detections:
            return None
        if detections[0][1] < 0.99:
            return None

        det = detections[0]
        det_xmin = int(det[2])
        det_ymin = int(det[3])
        det_width = int(det[4])
        det_height = int(det[5])
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (self.det_box_scale - 1) / 2)
        det_ymin += int(det_height * (self.det_box_scale - 1) / 2)
        det_xmax += int(det_width * (self.det_box_scale - 1) / 2)
        det_ymax += int(det_height * (self.det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width - 1)
        det_ymax = min(det_ymax, image_height - 1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        det_crop = image_bgr[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (self.cfg.input_size, self.cfg.input_size))
        inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
        inputs = self.preprocess(inputs).unsqueeze(0).to(self.device)

        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, _, _ = self.forward_pip(
            self.net,
            inputs,
            self.preprocess,
            self.cfg.input_size,
            self.cfg.net_stride,
            self.cfg.num_nb,
        )
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(
            self.cfg.num_lms,
            self.max_len,
        )
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(
            self.cfg.num_lms,
            self.max_len,
        )
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()

        pred_export = np.zeros((self.cfg.num_lms, 2), dtype=np.float32)
        for i in range(self.cfg.num_lms):
            x_pred = lms_pred_merge[i * 2] * det_width
            y_pred = lms_pred_merge[i * 2 + 1] * det_height
            pred_export[i, 0] = x_pred + det_xmin
            pred_export[i, 1] = y_pred + det_ymin

        return pred_export

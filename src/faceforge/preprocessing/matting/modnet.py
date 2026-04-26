"""MODNet portrait matting (preprocessing component).

Wraps the MODNet network bundled in the MonoNPHM submodule. The original
``demo/image_matting/colab/inference.py`` script is a CLI; this is a thin,
in-process equivalent so we can call MODNet directly from a pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faceforge._paths import PROJECT_ROOT

from .base import BaseMatter, MatteResult
from .visualize import composite_alpha, side_by_side_matte


_DEFAULT_MODNET_ROOT = (
    PROJECT_ROOT / 'submodules' / 'MonoNPHM' / 'src' / 'mononphm'
    / 'preprocessing' / 'MODNet'
)
_DEFAULT_CKPT = _DEFAULT_MODNET_ROOT / 'pretrained' / 'modnet_webcam_portrait_matting.ckpt'


@dataclass
class MODNetConfig:
    modnet_root: str = str(_DEFAULT_MODNET_ROOT)
    ckpt_path: str = str(_DEFAULT_CKPT)
    device: str = 'cuda:0'
    ref_size: int = 512   # MODNet's recommended reference resolution


class MODNetMatter(BaseMatter):
    """In-process MODNet wrapper, mirroring the colab inference script."""

    name = 'modnet'

    def __init__(self, config: MODNetConfig | None = None):
        self.config = config or MODNetConfig()

        modnet_root = Path(self.config.modnet_root).resolve()
        if not modnet_root.exists():
            raise FileNotFoundError(f"MODNet root not found: {modnet_root}")
        ckpt = Path(self.config.ckpt_path).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"MODNet checkpoint not found: {ckpt}")

        # MODNet's models package uses absolute "src.models.modnet" imports,
        # so we put the MODNet root on sys.path and remove it again after the
        # import so we don't pollute the global namespace.
        modnet_root_str = str(modnet_root)
        added = False
        if modnet_root_str not in sys.path:
            sys.path.insert(0, modnet_root_str)
            added = True
        try:
            from src.models.modnet import MODNet
        finally:
            if added:
                sys.path.remove(modnet_root_str)

        import torch
        import torchvision.transforms as T

        self._torch = torch
        self._F = torch.nn.functional
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        device = torch.device(self.config.device if torch.cuda.is_available()
                              and 'cuda' in self.config.device else 'cpu')
        self.device = device

        # The shipped checkpoint was saved through ``DataParallel``, so its
        # state-dict keys are prefixed with ``module.``. Build the network
        # the same way on CUDA so the keys match; on CPU strip the prefix
        # since DataParallel requires a CUDA device.
        modnet = MODNet(backbone_pretrained=False)
        weights = torch.load(str(ckpt), map_location=device)
        if device.type == 'cuda':
            modnet = torch.nn.DataParallel(modnet).to(device)
            modnet.load_state_dict(weights)
        else:
            stripped = {
                (k[len('module.'):] if k.startswith('module.') else k): v
                for k, v in weights.items()
            }
            modnet.load_state_dict(stripped)
            modnet = modnet.to(device)
        modnet.eval()
        self.modnet = modnet

    def run(self, image_rgb: np.ndarray) -> MatteResult:
        torch = self._torch
        F = self._F
        cfg = self.config

        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb] * 3, axis=-1)
        if image_rgb.shape[-1] == 4:
            image_rgb = image_rgb[..., :3]

        h, w = image_rgb.shape[:2]
        # Match the original MODNet inference script: short edge → ref_size,
        # then both dims rounded down to a multiple of 32.
        if max(h, w) < cfg.ref_size or min(h, w) > cfg.ref_size:
            if w >= h:
                rh, rw = cfg.ref_size, int(w / h * cfg.ref_size)
            else:
                rh, rw = int(h / w * cfg.ref_size), cfg.ref_size
        else:
            rh, rw = h, w
        rw = max(32, rw - rw % 32)
        rh = max(32, rh - rh % 32)

        from PIL import Image
        im = self.transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        im_resized = F.interpolate(im, size=(rh, rw), mode='area')

        with torch.no_grad():
            _, _, matte = self.modnet(im_resized, True)

        matte = F.interpolate(matte, size=(h, w), mode='area')
        alpha = matte[0, 0].detach().cpu().numpy().astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)

        return MatteResult(alpha=alpha, scheme='modnet')

    def visualize(
        self,
        image_rgb: np.ndarray,
        result: MatteResult,
    ) -> np.ndarray:
        return side_by_side_matte(image_rgb, result.alpha)

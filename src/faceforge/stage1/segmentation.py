"""
Face segmentation using BiSeNet (19-class CelebAMask-HQ).

Ported from: flame-head-tracker/submodules/face_parsing/FaceParsingUtil.py
"""

import sys

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from faceforge._paths import PROJECT_ROOT


class FaceParser:
    """BiSeNet face parsing: 19-class semantic segmentation."""

    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.device = device

        # Import BiSeNet from face_parsing submodule.
        # model.py uses absolute import "from submodules.face_parsing.resnet import Resnet18",
        # so the project root must be on sys.path.
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from submodules.face_parsing.model import BiSeNet

        self.net = BiSeNet(n_classes=19).to(device).eval()
        self.net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def parse(self, image_rgb_512: np.ndarray) -> np.ndarray:
        """Run face parsing on a 512x512 RGB image.

        Args:
            image_rgb_512: RGB uint8 [512, 512, 3]

        Returns:
            Parsing map [512, 512] int, values 0-18.
        """
        img = cv2.resize(image_rgb_512, (512, 512))
        img_tensor = self.to_tensor(img).unsqueeze(0).to(self.device)
        out = self.net(img_tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing

    @staticmethod
    def extract_face_mask(
        parsing: np.ndarray,
        keep_mouth: bool = True,
        keep_ears: bool = False,
        keep_neck: bool = False,
    ) -> np.ndarray:
        """Extract binary face mask from parsing map.

        Classes:
            0:background 1:skin 2:l_brow 3:r_brow 4:l_eye 5:r_eye
            6:eye_g 7:l_ear 8:r_ear 9:ear_r 10:nose
            11:mouth 12:u_lip 13:l_lip 14:neck 15:neck_l
            16:cloth 17:hair 18:hat

        Default keeps: skin, brows, eyes, nose, lips.
        Excludes: background, glasses, ears, neck, cloth, hair, hat.

        Returns:
            Boolean mask [512, 512]
        """
        # Start with all foreground (non-background)
        face_mask = parsing > 0

        # Remove non-face regions: neck_l(15), cloth(16), hair(17), hat(18)
        face_mask = face_mask & (parsing <= 14)

        # Remove glasses
        face_mask = face_mask & (parsing != 6)

        if not keep_mouth:
            face_mask = face_mask & (parsing != 11)

        if not keep_ears:
            face_mask = face_mask & ~((parsing >= 7) & (parsing <= 9))

        if not keep_neck:
            face_mask = face_mask & (parsing != 14)

        return face_mask

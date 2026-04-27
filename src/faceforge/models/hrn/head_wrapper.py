"""
HRN-head wrapper around our self-contained reimplementation.

Earlier this module wrapped modelscope's ``HeadReconstructionPipeline``;
that path required ``pip install modelscope tensorflow``. We now drive
:class:`HRNHeadPipeline` directly — vendored modelscope source under
:mod:`._head_vendored` plus our own preprocessing for the face/head
mattes (no TF needed).

Output: textured head mesh (35709 + scalp/neck/ears/eyes verts), 4096²
baked texture, plus a small helper to dump it as an OBJ + MTL + PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from faceforge._paths import PROJECT_ROOT
from faceforge.models.base import BaseModel, ModelOutput, ModelRequirements
from faceforge.pipeline.types import PreparedInputs
from faceforge.preprocessing.segmentation.base import BaseSegmenter

from .head_pipeline import HRNHeadPipeline, HRNHeadPipelineConfig


_DEFAULT_MODEL_DIR = PROJECT_ROOT / 'data' / 'hrn_head_model'


@dataclass
class HRNHeadConfig:
    """Tunables for :class:`HRNHeadModel`.

    Attributes:
        model_dir: directory holding the staged HRN-head asset tree.
            Default: ``<project>/data/hrn_head_model``.
        hair_tex: ``False`` (default) bakes texture onto a bald template;
            ``True`` uses the hair template (less stable).
        device: torch device for the BFM regressor + 250-iter fitting.
        max_long_side: cap input long edge before reconstruction (default
            1500, matching modelscope's behaviour).
    """

    model_dir: str = str(_DEFAULT_MODEL_DIR)
    hair_tex: bool = False
    device: str = 'cuda'
    max_long_side: int = 1500
    pose_threshold_deg: float = 30.0   # see HRNHeadPipelineConfig


class HRNHeadModel(BaseModel):
    """Single-image full-head reconstruction (vendored HRN-head)."""

    name = 'hrn_head'
    requirements = ModelRequirements(
        mode='single',
        needs_aligned_image=False,    # internal RetinaFace + LargeBaseLmkInfer
        needs_landmarks=[],           # internal 106 + 68 (FAN)
        needs_segmentation=None,      # uses BiSeNet internally for mattes
        needs_matte=False,
        needs_identity_shape=False,
    )

    def __init__(
        self,
        config: HRNHeadConfig | None = None,
        segmenter: Optional[BaseSegmenter] = None,
    ):
        self.config = config or HRNHeadConfig()
        self.model_dir = Path(self.config.model_dir).resolve()
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"HRN head-recon model dir not found: {self.model_dir}")

        self._pipeline = HRNHeadPipeline(
            config=HRNHeadPipelineConfig(
                model_dir=str(self.model_dir),
                hair_tex=self.config.hair_tex,
                device=self.config.device,
                max_long_side=self.config.max_long_side,
                pose_threshold_deg=self.config.pose_threshold_deg,
            ),
            segmenter=segmenter,
        )

    def run(self, prepared: PreparedInputs) -> ModelOutput:
        result = self._pipeline.run(prepared.image_rgb)

        vertices = np.asarray(result['vertices'], dtype=np.float32)
        faces = np.asarray(result['triangles'], dtype=np.int32)
        uvs = np.asarray(result['uvs'], dtype=np.float32)
        faces_uv = np.asarray(result['faces_uv'], dtype=np.int64)
        # template_ourFull_bfmEyes.obj is 1-indexed in source; subtract for 0-index.
        if faces_uv.min() >= 1:
            faces_uv = faces_uv - 1
        faces_uv = faces_uv.astype(np.int32)

        extras: dict[str, Any] = {
            'texture_map': np.asarray(result['texture_map']),
            'uvs': uvs,
            'faces_uv': faces_uv,
            'normals': np.asarray(result['normals'], dtype=np.float32),
        }

        bfm_params = {k: np.asarray(v) for k, v in result['coeffs'].items()}

        return ModelOutput(
            name=self.name,
            mesh_vertices=vertices,
            mesh_faces=faces,
            bfm_params=bfm_params,
            rendered_overlay=None,
            extras=extras,
        )

    def visualize(
        self,
        prepared: PreparedInputs,
        output: ModelOutput,
    ) -> np.ndarray:
        """[source ‖ baked head texture] strip at the source image's height."""
        src = prepared.image_rgb
        tex = output.extras.get('texture_map')
        if tex is None:
            return src.copy()

        if tex.dtype != np.uint8:
            tex = np.clip(tex, 0, 255).astype(np.uint8)
        if tex.ndim == 3 and tex.shape[2] == 3:
            tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)

        target_h = src.shape[0]
        if tex.shape[0] != target_h:
            scale = target_h / tex.shape[0]
            new_w = max(1, int(round(tex.shape[1] * scale)))
            tex = cv2.resize(tex, (new_w, target_h), interpolation=cv2.INTER_AREA)
        return np.concatenate([src, tex], axis=1)

    def write_obj(self, output: ModelOutput, obj_path: str | Path) -> Path:
        """Dump the head mesh as ``<name>.obj`` + ``.mtl`` + ``.png`` triplet."""
        obj_path = Path(obj_path)
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        v = output.mesh_vertices
        f = output.mesh_faces
        if v is None or f is None:
            raise ValueError("output has no mesh — was run() called?")

        uvs = output.extras.get('uvs')
        f_uv = output.extras.get('faces_uv')
        normals = output.extras.get('normals')
        tex = output.extras.get('texture_map')

        mtl_path = obj_path.with_suffix('.mtl')
        tex_path = obj_path.with_suffix('.png')

        if tex is not None:
            tex_uint8 = (tex if tex.dtype == np.uint8
                         else np.clip(tex, 0, 255).astype(np.uint8))
            cv2.imwrite(str(tex_path), tex_uint8)   # cv2 writes BGR

        with open(mtl_path, 'w', encoding='utf-8') as fh:
            fh.write('newmtl head\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n')
            if tex is not None:
                fh.write(f'map_Kd {tex_path.name}\n')

        with open(obj_path, 'w', encoding='utf-8') as fh:
            fh.write(f'mtllib {mtl_path.name}\nusemtl head\n')
            for vx, vy, vz in v:
                fh.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
            if uvs is not None:
                for u, vt in uvs:
                    fh.write(f'vt {u:.6f} {vt:.6f}\n')
            if normals is not None:
                for nx, ny, nz in normals:
                    fh.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')

            for tri_idx, tri in enumerate(f):
                a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
                if uvs is not None and f_uv is not None:
                    ua = int(f_uv[tri_idx][0]) + 1
                    ub = int(f_uv[tri_idx][1]) + 1
                    uc = int(f_uv[tri_idx][2]) + 1
                    if normals is not None:
                        fh.write(f'f {a}/{ua}/{a} {b}/{ub}/{b} {c}/{uc}/{c}\n')
                    else:
                        fh.write(f'f {a}/{ua} {b}/{ub} {c}/{uc}\n')
                else:
                    fh.write(f'f {a} {b} {c}\n')

        return obj_path

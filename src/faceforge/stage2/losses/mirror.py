"""Mirror symmetry loss matching pixel3dmm tracker.py L858-863.

Computes the difference between canonical vertices and their mirrored version
(x-flipped, reindexed via mirror_order). Encourages left-right face symmetry.

Ref: pixel3dmm tracker.py L858-863
    verts_mirrored = vertices_can[:, mirror_order, :]
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    mirrored[:, :, 1:] = mirrored[:, :, 1:]
    mirror_loss = (mirrored - vertices_can).square().sum(-1).mean()
"""

import numpy as np
import torch

from faceforge._paths import PROJECT_ROOT


def load_mirror_order(pixel3dmm_code_base: str = 'submodules/pixel3dmm') -> torch.Tensor:
    """Load FLAME mirror vertex indices from pixel3dmm assets."""
    path = PROJECT_ROOT / pixel3dmm_code_base / 'assets' / 'flame_mirror_index.npy'
    return torch.from_numpy(np.load(str(path))).long()


def mirror_symmetry_loss(vertices_canonical: torch.Tensor,
                         mirror_order: torch.Tensor) -> torch.Tensor:
    """Mirror symmetry loss on canonical (unrotated) FLAME vertices.

    Args:
        vertices_canonical: [B, V, 3] vertices in canonical FLAME space
            (no head rotation applied)
        mirror_order: [V] vertex reindexing for left-right mirror

    Returns:
        scalar loss
    """
    mirror_order = mirror_order.to(vertices_canonical.device)
    mirrored = vertices_canonical[:, mirror_order, :].clone()
    mirrored[:, :, 0] = -mirrored[:, :, 0]  # flip x
    # mirrored[:, :, 1:] stays the same (y, z unchanged)
    return (mirrored - vertices_canonical).square().sum(-1).mean()

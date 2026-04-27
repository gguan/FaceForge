"""Brute-force batched k-NN — replaces ``pytorch3d.ops.knn_points``."""

from dataclasses import dataclass

import torch


@dataclass
class _KNNResult:
    """Subset of pytorch3d's KNN return type used by MonoNPHM.

    Attributes:
        dists: ``[B, P1, K]`` squared L2 distances.
        idx:   ``[B, P1, K]`` long tensor — indices into the second
               ``[B, P2, D]`` argument's points dimension.
    """
    dists: torch.Tensor
    idx: torch.Tensor


def knn_points(p1: torch.Tensor, p2: torch.Tensor, K: int = 1, **_) -> _KNNResult:
    """Brute-force batched k-NN matching pytorch3d's ``knn_points`` shape.

    Args:
        p1: ``[B, P1, D]`` queries.
        p2: ``[B, P2, D]`` anchors to search.
        K:  number of neighbors.

    Returns:
        :class:`_KNNResult` with ``.dists`` (squared distances) and
        ``.idx`` (anchor indices), each ``[B, P1, K]``.
    """
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError(
            f"p1, p2 must be [B, N, D]; got {tuple(p1.shape)}, {tuple(p2.shape)}")
    if p1.shape[0] != p2.shape[0] or p1.shape[2] != p2.shape[2]:
        raise ValueError("p1 and p2 must agree on batch and feature dims")

    # Pairwise squared L2: ||a-b||² = ||a||² + ||b||² - 2 a·b
    a2 = (p1 * p1).sum(dim=-1, keepdim=True)            # [B, P1, 1]
    b2 = (p2 * p2).sum(dim=-1).unsqueeze(-2)            # [B, 1, P2]
    ab = torch.bmm(p1, p2.transpose(1, 2))              # [B, P1, P2]
    dists = (a2 + b2 - 2.0 * ab).clamp_min_(0.0)        # [B, P1, P2]

    K = min(K, p2.shape[1])
    knn_dists, knn_idx = torch.topk(dists, k=K, dim=-1, largest=False, sorted=True)
    return _KNNResult(dists=knn_dists, idx=knn_idx.long())

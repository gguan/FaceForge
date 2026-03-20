"""Region weight map for photometric loss (baseline).

Reference: stage2-design.md Section 7.6 (HiFace inspired)
"""

import torch


# Default region weights from technical-design.md
DEFAULT_REGION_WEIGHTS = {
    'nose': 3.0,
    'left_eye_region': 2.5,
    'right_eye_region': 2.5,
    'lips': 2.0,
    'forehead': 0.5,
}


def build_vertex_weights(flame_model, region_weights: dict | None = None) -> torch.Tensor:
    """Assign per-vertex weights based on face region.

    Args:
        flame_model: FLAMEModel instance (has get_region_indices)
        region_weights: dict of region_name → weight. Defaults to standard weights.

    Returns:
        vertex_weights: [5023] per-vertex weight
    """
    if region_weights is None:
        region_weights = DEFAULT_REGION_WEIGHTS

    weights = torch.ones(5023, device=flame_model.v_template.device)
    for region_name, weight in region_weights.items():
        try:
            indices = flame_model.get_region_indices(region_name)
            weights[indices] = weight
        except KeyError:
            pass
    return weights

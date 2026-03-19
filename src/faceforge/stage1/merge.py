"""
Parameter merging: combine MICA shape with DECA expression/pose/texture/lighting.
"""

import torch


def merge_params(mica_output: dict, deca_output: dict) -> dict:
    """Merge MICA and DECA outputs into unified FLAME parameters.

    Shape comes entirely from MICA (300 dims).
    Expression, pose, texture, lighting come from DECA.

    Args:
        mica_output: dict with 'shape_code' [1, 300]
        deca_output: dict with 'exp' [1, 50], 'pose' [1, 6],
                     'tex' [1, 50], 'light' [1, 9, 3]

    Returns:
        dict with merged FLAME parameters:
            'shape': [1, 300]
            'exp': [1, 100] (DECA 50 + 50 zeros)
            'head_pose': [1, 3]
            'jaw_pose': [1, 3]
            'tex': [1, 50]
            'light': [1, 9, 3]
    """
    params = {}

    # Shape: 100% from MICA (full 300 dims)
    params['shape'] = mica_output['shape_code'][:, :300]  # [1, 300]

    # Expression: DECA first 50 dims, pad to 100
    params['exp'] = torch.zeros(1, 100)
    params['exp'][:, :50] = deca_output['exp'][:, :50]  # [1, 100]

    # Pose: from DECA
    params['head_pose'] = deca_output['pose'][:, :3]  # [1, 3]
    params['jaw_pose'] = deca_output['pose'][:, 3:]   # [1, 3]

    # Texture: DECA first 50 dims
    params['tex'] = deca_output['tex'][:, :50]  # [1, 50]

    # Lighting: DECA, reshape to [1, 9, 3]
    light = deca_output['light']
    if light.dim() == 2:
        light = light.reshape(1, 9, 3)
    params['light'] = light  # [1, 9, 3]

    return params

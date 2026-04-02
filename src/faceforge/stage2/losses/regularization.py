"""Parameter regularization losses.

Reference: pixel3dmm tracker.py (shape→MICA reg),
           HRN losses.py (SH monochrome w_gamma=10),
           VHAP config (texture TV reg)
"""

import torch

from ..config import Stage2Config


def regularization_loss(shape: torch.Tensor, expression: torch.Tensor,
                        jaw_6d: torch.Tensor, lighting: torch.Tensor,
                        mica_init_shape: torch.Tensor,
                        config: Stage2Config,
                        texture_displacement: torch.Tensor | None = None) -> torch.Tensor:
    """Combined regularization loss.

    pixel3dmm: torch.sum(x**2, dim=-1).mean() — sum over params, mean over batch.
    Jaw regularization: (I6D - jaw)^2 towards identity 6D rotation.
    Ref: pixel3dmm tracker.py L913-922

    Args:
        shape: [B, 300] current shape params
        expression: [B, 100]
        jaw_6d: [B, 6] jaw rotation in 6D format
        lighting: [B, 9, 3] SH coefficients
        mica_init_shape: [1, 300] MICA initial shape (frozen target)
        config: Stage2Config
    """
    # Shape: constrain to MICA initial value
    loss = config.w_reg_shape_to_mica * (shape - mica_init_shape).pow(2).sum(dim=-1).mean()
    loss = loss + config.w_reg_shape_to_zero * shape.pow(2).sum(dim=-1).mean()

    # Expression
    loss = loss + config.w_reg_expression * expression.pow(2).sum(dim=-1).mean()

    # Jaw: pixel3dmm uses (I6D - jaw)^2 — penalize deviation from identity
    # I6D = [1, 0, 0, 0, 1, 0] for 6D rotation
    I6d = torch.tensor([1., 0., 0., 0., 1., 0.], device=jaw_6d.device, dtype=jaw_6d.dtype)
    loss = loss + config.w_reg_jaw * (I6d - jaw_6d).pow(2).sum(dim=-1).mean()

    # SH monochrome: penalize RGB channel divergence
    sh_mean = lighting.mean(dim=-1, keepdim=True)  # [B, 9, 1]
    loss = loss + config.w_reg_sh_mono * (lighting - sh_mean).pow(2).mean()

    # Texture TV (only in fine_detail stage when displacement exists)
    if texture_displacement is not None:
        td = texture_displacement
        tv_h = (td[:, :, 1:, :] - td[:, :, :-1, :]).pow(2).mean()
        tv_w = (td[:, :, :, 1:] - td[:, :, :, :-1]).pow(2).mean()
        loss = loss + config.w_reg_tex_tv * (tv_h + tv_w)

    return loss

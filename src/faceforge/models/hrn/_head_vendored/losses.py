"""HRN-head losses (vendored from modelscope, Apache 2.0)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry import warp_affine


def resize_n_crop(image, M, dsize=112):
    return warp_affine(image, M, dsize=(dsize, dsize))


def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]


def photo_loss(imageA, imageB, mask, eps=1e-6):
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss


def landmark_loss(predict_lm, gt_lm, weight=None):
    if not weight:
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm) ** 2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


def reg_loss(coeffs_dict, w_id=1, w_exp=1, w_tex=1):
    creg_loss = (w_id * torch.sum(coeffs_dict['id'] ** 2)
                 + w_exp * torch.sum(coeffs_dict['exp'] ** 2)
                 + w_tex * torch.sum(coeffs_dict['tex'] ** 2))
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]
    gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)
    return creg_loss, gamma_loss


def reflectance_loss(texture, mask):
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask) ** 2) / (texture.shape[0] * torch.sum(mask))
    return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        h_x, w_x = x.size()[2], x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class TVLoss_std(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x, w_x = x.size()[2], x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        h_tv = ((h_tv - torch.mean(h_tv)) ** 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        w_tv = ((w_tv - torch.mean(w_tv)) ** 2).sum()
        return self.TVLoss_weight * 2 * (h_tv + w_tv) / batch_size


def points_loss_horizontal(verts, left_points, right_points, width=224):
    verts_int = torch.ceil(verts[0]).long().clamp(0, width - 1)
    verts_left = left_points[width - 1 - verts_int[:, 1]].float()
    verts_right = right_points[width - 1 - verts_int[:, 1]].float()
    verts_x = verts[0, :, 0]
    dist = (verts_left - verts_x) / width * (verts_right - verts_x) / width
    dist /= torch.max(torch.abs((verts_left - verts_x) / width),
                      torch.abs((verts_right - verts_x) / width))
    edge_inds = torch.where(dist > 0)[0]
    dist += 0.01
    dist = torch.nn.functional.relu(dist).clone()
    dist -= 0.01
    dist = torch.abs(dist)
    return torch.mean(dist), edge_inds


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=1, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict + target, dim=1)
        loss = 1 - (2 * num + self.smooth) / (den + self.smooth)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        raise Exception(f'Unexpected reduction {self.reduction}')

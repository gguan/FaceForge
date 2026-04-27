"""
HeadReconModel — per-image fitting + final mesh + texture baking.

Vendored from modelscope (Apache 2.0). Internals adapted to our local
package layout. The high-level flow:

    forward()
      1. with no_grad: net_recon(input_img_coeff) → 257-d BFM coeffs
      2. fitting_nonlinear():  250-iter Adam
           - first 150 iters update only `trans` (pose)
           - remaining 100 update id/exp/tex/gamma + face/head shape
             & texture offset UV maps (300² + 100²)
           - losses: photo (face+head crops), landmark, BFM regularizers,
             TV + TV-std on offsets, Dice (occ_head vs head_mask),
             horizontal-points (silhouette pull-in)
      3. eye-open boost on exp[16]/[17]/[19]
      4. compute_for_render_head() with neck/nose adjustments → final
         vertex set
      5. renderer.pred_shape_and_texture() → bakes texture into a
         tex_size² UV map (the orchestrator runs TexProcesser on top)
      6. returns dict with vertices / triangles / uvs / faces_uv /
         normals / tex_map / tex_mask / coeffs
"""

import os

import cv2
import numpy as np
import torch

from . import opt
from .bfm import ParametricFaceModel
from .losses import (
    BinaryDiceLoss, TVLoss, TVLoss_std,
    landmark_loss, photo_loss, points_loss_horizontal, reg_loss,
)
from .nv_diffrast import MeshRenderer
from .utils_bridge import estimate_normals, read_obj


class HeadReconModel:
    """HRN-head reconstruction model.

    Standalone (no modelscope ``TorchModel`` superclass needed). Caller
    is responsible for placing the model on a device via :meth:`set_device`,
    loading the checkpoint via :meth:`setup`, then calling
    :meth:`set_input` and :meth:`forward` per image.
    """

    def __init__(self, model_dir, pose_threshold_radians: float = 3.14 / 6):
        self.model_dir = model_dir
        opt.bfm_folder = os.path.join(model_dir, 'assets')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.pose_threshold_radians = float(pose_threshold_radians)

        # Lazy import of networks to avoid a hard dep on torchvision when
        # someone imports the package just to read the docs.
        from . import networks
        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=None)

        self.headmodel = ParametricFaceModel(
            assets_root=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal, center=opt.center,
            is_train=self.isTrain,
            default_name='ourRefineBFMEye0504_model.mat')

        self.headmodel_for_fitting = ParametricFaceModel(
            assets_root=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal, center=opt.center,
            is_train=self.isTrain,
            default_name='ourRefineFull_model.mat')

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(rasterize_fov=fov, znear=opt.z_near,
                                     zfar=opt.z_far, rasterize_size=int(2 * opt.center))
        self.renderer_fitting = MeshRenderer(rasterize_fov=fov, znear=opt.z_near,
                                             zfar=opt.z_far, rasterize_size=int(2 * opt.center))

        template_obj_path = os.path.join(
            model_dir, 'assets/3dmm/template_mesh/template_ourFull_bfmEyes.obj')
        self.template_output_mesh = read_obj(template_obj_path)
        self.nonlinear_UVs = self.template_output_mesh['uvs']
        self.nonlinear_UVs = torch.from_numpy(self.nonlinear_UVs)

        self.jaw_edge_mask = cv2.imread(
            os.path.join(model_dir, 'assets/texture/jaw_edge_mask2.png')
        )[..., 0].astype(np.float32) / 255.0
        self.jaw_edge_mask = cv2.resize(self.jaw_edge_mask, (300, 300))[..., None]

        self.compute_color_loss = photo_loss
        self.compute_lm_loss = landmark_loss
        self.compute_reg_loss = reg_loss

    # ------------------------------------------------------------------ setup

    def set_device(self, device):
        self.device = device
        self.net_recon = self.net_recon.to(device)
        self.headmodel.to(device)
        self.headmodel_for_fitting.to(device)
        self.nonlinear_UVs = self.nonlinear_UVs.to(device)
        self.renderer = self.renderer.to(device)
        self.renderer_fitting = self.renderer_fitting.to(device)

    def setup(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        net = self.net_recon
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict['net_recon'], strict=False)

    def eval(self):
        self.net_recon.eval()

    def set_render(self, image_res=1024):
        fov = 2 * np.arctan(self.opt.center / self.opt.focal) * 180 / np.pi
        if image_res is None:
            image_res = int(2 * self.opt.center)
        self.renderer = MeshRenderer(rasterize_fov=fov, znear=self.opt.z_near,
                                     zfar=self.opt.z_far, rasterize_size=image_res).to(self.device)

    def set_input(self, input_dict):
        d = self.device
        self.input_img = input_dict['imgs'].to(d)
        self.input_img_hd = input_dict['imgs_hd'].to(d) if 'imgs_hd' in input_dict else None
        self.input_fat_img_hd = (
            input_dict['imgs_fat_hd'].to(d) if input_dict.get('imgs_fat_hd') is not None
            else self.input_img_hd
        )
        self.gt_lm = input_dict['lms'].to(d) if 'lms' in input_dict else None
        self.gt_lm_hd = input_dict['lms_hd'].to(d) if 'lms_hd' in input_dict else None
        self.face_mask = input_dict['face_mask'].to(d) if 'face_mask' in input_dict else None
        self.head_mask = input_dict['head_mask'].to(d) if 'head_mask' in input_dict else None
        self.input_img_coeff = (input_dict['imgs_coeff'].to(d)
                                if 'imgs_coeff' in input_dict else None)
        self.gt_lm_coeff = (input_dict['lms_coeff'].to(d)
                            if 'lms_coeff' in input_dict else None)

    # -------------------------------------------------------------- pose checks

    def check_head_pose(self, coeffs):
        thr = self.pose_threshold_radians
        for i in (224, 225, 226):
            if coeffs[0, i] > thr or coeffs[0, i] < -thr:
                return False
        return True

    def head_pose_angles(self, coeffs) -> tuple[float, float, float]:
        """Return predicted Euler (rx, ry, rz) in radians for diagnostics."""
        return (float(coeffs[0, 224]), float(coeffs[0, 225]), float(coeffs[0, 226]))

    # ----------------------------------------------------------- mask helpers

    def get_fusion_mask(self, keep_forehead=True):
        self.without_forehead_inds = torch.from_numpy(np.load(
            os.path.join(self.model_dir, 'assets/3dmm/inds/bfm_withou_forehead_inds.npy')
        )).long().to(self.device)

        h, w = self.shape_offset_uv.shape[1:3]
        self.fusion_mask = torch.zeros((h, w)).to(self.device).float()
        if keep_forehead:
            UVs_coords = self.nonlinear_UVs.clone()[:35709][self.without_forehead_inds]
        else:
            UVs_coords = self.nonlinear_UVs.clone()[:35709]
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords).long()
        self.fusion_mask[h - 1 - UVs_coords_int[:, 1], UVs_coords_int[:, 0]] = 1

        m = self.fusion_mask.cpu().numpy()
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
        m = cv2.blur(m, (17, 17))
        self.fusion_mask = torch.from_numpy(m).float().to(self.device)

    def get_edge_mask(self):
        h, w = self.shape_offset_uv.shape[1:3]
        self.edge_mask = torch.zeros((h, w)).to(self.device).float()
        UVs_coords = self.nonlinear_UVs.clone()[self.edge_points_inds]
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords).long()
        self.edge_mask[h - 1 - UVs_coords_int[:, 1], UVs_coords_int[:, 0]] = 1

        m = self.edge_mask.cpu().numpy()
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
        m = cv2.blur(m, (5, 5))
        self.edge_mask = torch.from_numpy(m).float().to(self.device)

    def blur_shape_offset_uv(self, global_blur=False, blur_size=3):
        if self.edge_mask is not None:
            sov = self.shape_offset_uv[0].detach().cpu().numpy()
            sov = cv2.blur(sov, (15, 15))
            sov = torch.from_numpy(sov).float().to(self.device)[None, ...]
            self.shape_offset_uv = (sov * self.edge_mask[None, ..., None]
                                    + self.shape_offset_uv * (1 - self.edge_mask[None, ..., None]))
        self.shape_offset_uv = self.shape_offset_uv * self.fusion_mask[None, ..., None]

        if global_blur and blur_size > 0:
            sov = self.shape_offset_uv[0].detach().cpu().numpy()
            sov = cv2.blur(sov, (blur_size, blur_size))
            self.shape_offset_uv = torch.from_numpy(sov).float().to(self.device)[None, ...]

    def blur_offset_edge(self):
        sov = self.shape_offset_uv[0].detach().cpu().numpy()
        sov_head = self.shape_offset_uv_head[0].detach().cpu().numpy()
        sov_head = cv2.resize(sov_head, (300, 300))
        sov_head = sov_head * (1 - self.jaw_edge_mask) + sov * self.jaw_edge_mask
        sov_head = cv2.resize(sov_head, (100, 100))
        self.shape_offset_uv_head = torch.from_numpy(sov_head).float().to(self.device)[None, ...]

    # -------------------------------------------------------------- fitting

    def fitting_nonlinear(self, coeff, n_iters=250):
        out = self.headmodel_for_fitting.split_coeff(coeff.detach().clone())
        for k in ('id', 'exp', 'tex', 'angle', 'gamma', 'trans'):
            out[k].requires_grad = True

        d = self.device
        self.shape_offset_uv = torch.zeros((1, 300, 300, 3), dtype=torch.float32, device=d)
        self.shape_offset_uv.requires_grad = True
        self.texture_offset_uv = torch.zeros((1, 300, 300, 3), dtype=torch.float32, device=d)
        self.texture_offset_uv.requires_grad = True
        self.shape_offset_uv_head = torch.zeros((1, 100, 100, 3), dtype=torch.float32, device=d)
        self.shape_offset_uv_head.requires_grad = True
        self.texture_offset_uv_head = torch.zeros((1, 100, 100, 3), dtype=torch.float32, device=d)
        self.texture_offset_uv_head.requires_grad = True

        head_face_inds = np.load(os.path.join(
            self.model_dir, 'assets/3dmm/inds/ours_head_face_inds.npy'))
        head_face_inds = torch.from_numpy(head_face_inds).to(d)
        head_faces = self.headmodel_for_fitting.face_buf[head_face_inds]

        opt_params = [self.shape_offset_uv, self.texture_offset_uv,
                      self.shape_offset_uv_head, self.texture_offset_uv_head,
                      out['id'], out['exp'], out['tex'], out['gamma']]
        optim = torch.optim.Adam(opt_params, lr=1e-3)
        optim_pose = torch.optim.Adam([out['trans']], lr=1e-1)

        self.get_edge_points_horizontal()

        for i in range(n_iters):
            (self.pred_vertex_head, self.pred_tex, self.pred_color_head, self.pred_lm,
             face_shape, face_shape_offset, self.verts_proj_head) = \
                self.headmodel_for_fitting.compute_for_render_head_fitting(
                    out, self.shape_offset_uv, self.texture_offset_uv,
                    self.shape_offset_uv_head, self.texture_offset_uv_head,
                    self.nonlinear_UVs)
            self.pred_vertex = self.pred_vertex_head[:, :35241]
            self.pred_color = self.pred_color_head[:, :35241]
            self.verts_proj = self.verts_proj_head[:, :35241]
            self.pred_mask_head, _, self.pred_head, self.occ_head = self.renderer_fitting(
                self.pred_vertex_head, head_faces, feat=self.pred_color_head)
            self.pred_mask, _, self.pred_face, self.occ_face = self.renderer_fitting(
                self.pred_vertex, self.headmodel_for_fitting.face_buf[:69732],
                feat=self.pred_color)

            self.pred_coeffs_dict = self.headmodel_for_fitting.split_coeff(out)
            self.compute_losses_fitting()

            if i < 150:
                optim_pose.zero_grad()
                (self.loss_lm + self.loss_color * 0.1).backward()
                optim_pose.step()
            else:
                optim.zero_grad()
                self.loss_all.backward()
                optim.step()

        out_coeff = self.headmodel_for_fitting.merge_coeff(out)

        self.get_edge_mask()
        self.get_fusion_mask(keep_forehead=False)
        self.blur_shape_offset_uv(global_blur=True)
        self.blur_offset_edge()
        return out_coeff

    def compute_losses_fitting(self):
        face_mask = self.pred_mask.detach()
        self.loss_color = self.opt.w_color * self.compute_color_loss(
            self.pred_face, self.input_img, face_mask)

        loss_reg, loss_gamma = self.compute_reg_loss(
            self.pred_coeffs_dict, w_id=self.opt.w_id,
            w_exp=self.opt.w_exp, w_tex=self.opt.w_tex)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(
            self.pred_lm, self.gt_lm) * 0.1

        self.loss_smooth_offset = TVLoss()(self.shape_offset_uv.permute(0, 3, 1, 2)) * 10000
        self.loss_reg_textureOff = torch.mean(torch.abs(self.texture_offset_uv)) * 10
        self.loss_smooth_offset_std = TVLoss_std()(
            self.shape_offset_uv.permute(0, 3, 1, 2)) * 50000

        self.loss_points_horizontal, self.edge_points_inds = points_loss_horizontal(
            self.verts_proj, self.left_points, self.right_points)
        self.loss_points_horizontal *= 20

        self.loss_all = (self.loss_color + self.loss_lm + self.loss_reg + self.loss_gamma
                        + self.loss_smooth_offset + self.loss_smooth_offset_std
                        + self.loss_reg_textureOff + self.loss_points_horizontal)

        head_mask_pred = self.pred_mask_head.detach()
        self.loss_color_head = self.opt.w_color * self.compute_color_loss(
            self.pred_head, self.input_img, head_mask_pred)
        self.loss_smooth_offset_head = TVLoss()(
            self.shape_offset_uv_head.permute(0, 3, 1, 2)) * 100
        self.loss_smooth_offset_std_head = TVLoss_std()(
            self.shape_offset_uv_head.permute(0, 3, 1, 2)) * 500
        self.loss_mask = BinaryDiceLoss()(self.occ_head, self.head_mask) * 20

        self.loss_all = (self.loss_all + self.loss_mask + self.loss_color_head
                        + self.loss_smooth_offset_head + self.loss_smooth_offset_std_head)

    def get_edge_points_horizontal(self):
        left_points, right_points = [], []
        for i in range(self.face_mask.shape[2]):
            inds = torch.where(self.face_mask[0, 0, i, :] > 0.5)
            if len(inds[0]) > 0:
                left_points.append(int(inds[0][0]) + 1)
                right_points.append(int(inds[0][-1]))
            else:
                left_points.append(0)
                right_points.append(self.face_mask.shape[3] - 1)
        self.left_points = torch.tensor(left_points).long().to(self.device)
        self.right_points = torch.tensor(right_points).long().to(self.device)

    # ----------------------------------------------------------- forward

    def forward(self):
        with torch.no_grad():
            output_coeff = self.net_recon(self.input_img_coeff)
        self.last_predicted_pose = self.head_pose_angles(output_coeff)

        if not self.check_head_pose(output_coeff):
            return None

        with torch.enable_grad():
            output_coeff = self.fitting_nonlinear(output_coeff)

        coef_dict = self.headmodel.split_coeff(output_coeff)
        eye_sum = coef_dict['exp'][0, 16] + coef_dict['exp'][0, 17] + coef_dict['exp'][0, 19]
        degree = 0.5 if eye_sum > 1.0 else 1.0
        coef_dict['exp'][0, 16] += 1 * degree
        coef_dict['exp'][0, 17] += 1 * degree
        coef_dict['exp'][0, 19] += 1.5 * degree
        output_coeff = self.headmodel.merge_coeff(coef_dict)

        self.pred_vertex, _, _, _, face_shape_ori, face_shape, _ = \
            self.headmodel.compute_for_render_head(
                output_coeff,
                self.shape_offset_uv.detach(), self.texture_offset_uv.detach(),
                self.shape_offset_uv_head.detach() * 0,    # head offset killed at output
                self.texture_offset_uv_head.detach(),
                self.nonlinear_UVs,
                nose_coeff=0.1, neck_coeff=0.3,
                neckSlim_coeff=0.5, neckStretch_coeff=0.5)

        UVs = np.array(self.template_output_mesh['uvs'])
        UVs_tensor = torch.tensor(UVs, dtype=torch.float32).unsqueeze(0).to(self.pred_vertex.device)

        target_img = self.input_fat_img_hd.permute(0, 2, 3, 1)
        face_buf = self.headmodel.face_buf

        with torch.enable_grad():
            pred_mask, _, pred_face, texture_map, texture_mask = \
                self.renderer.pred_shape_and_texture(
                    self.pred_vertex, face_buf, UVs_tensor, target_img, None)

        self.pred_coeffs_dict = self.headmodel.split_coeff(output_coeff)

        recon_shape = face_shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1]   # camera → world
        recon_shape = recon_shape.cpu().numpy()[0]
        tri = self.headmodel.face_buf.cpu().numpy()

        return {
            'flag': 0,
            'tex_map': texture_map,
            'tex_mask': texture_mask * 255.0,
            'coeffs': self.pred_coeffs_dict,
            'vertices': recon_shape,
            'triangles': tri,
            'uvs': UVs,
            'faces_uv': self.template_output_mesh['faces_uv'],
            'normals': estimate_normals(recon_shape, tri),
        }

    __call__ = forward

import torch
from torch import nn
import utils.rend_util as rend_util
import utils.general as utils
from utils.ssim import SSIM
import math
import pdb
import torchvision.transforms

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    
    
class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss,
                 eikonal_weight, 
                 normal_smooth_weight = 0.005,
                 patch_depth_smooth_weight = 0.005,
                 patch_normal_smooth_weight = 0.005,
                 entropy_weight=0.001,
                 entropy_log_scaling=False,
                 entropy_acc_thresh=0.01,
                 patch_rgb_ncc_weight=0.01,
                 patch_rgb_loss='l1',
                 warped_rgb_weight=1,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 min_visibility = 1e-3,
                 end_step = -1,
                 start_reg_step=500):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.normal_smooth_weight = normal_smooth_weight
        self.patch_depth_smooth_weight = patch_depth_smooth_weight
        self.patch_normal_smooth_weight = patch_normal_smooth_weight
        self.entropy_weight = entropy_weight
        self.entropy_log_scaling = entropy_log_scaling
        self.entropy_acc_thresh = entropy_acc_thresh
        self.patch_rgb_ncc_weight = patch_rgb_ncc_weight
        self.patch_rgb_loss = patch_rgb_loss
        self.warped_rgb_weight = warped_rgb_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.min_visibility = min_visibility
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        self.h_patch_size = 0
        self.patch_offset = None
        self.ssim = SSIM(self.h_patch_size)
        
        print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} NSM_{self.normal_smooth_weight} PDSM_{self.patch_depth_smooth_weight} PNSM_{self.patch_normal_smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step
        self.start_reg_step = start_reg_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_normal_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_patch_depth_smooth_loss(self, model_outputs, patch_size):
        depth = model_outputs['patch_depth_values'].reshape(-1, patch_size, patch_size) # (B, PS, PS)
        acc = model_outputs['patch_acc'].reshape(-1, patch_size, patch_size)
        mask = (acc > 0.95).detach()
        
        d00 = depth[:, :-1, :-1]
        d01 = depth[:, :-1, 1:]
        d10 = depth[:, 1:, :-1]

        m00 = mask[:, :-1, :-1]
        m01 = mask[:, :-1, 1:]
        m10 = mask[:, 1:, :-1]
        
        # patch_depth_loss = (((d00 - d01) ** 2) + ((d00 - d10) ** 2))
        patch_depth_loss = torch.abs(d00 - d01) * torch.mul(m00, m01) + torch.abs(d00 - d10) * torch.mul(m00, m10)

        if model_outputs['unseen_patch_depth_values'] is not None:
            unseen_depth = model_outputs['unseen_patch_depth_values'].reshape(-1, patch_size, patch_size) # (B, PS, PS)
            acc = model_outputs['unseen_patch_acc'].reshape(-1, patch_size, patch_size)
            mask = (acc > 0.95).detach()
            
            ud00 = unseen_depth[:, :-1, :-1]
            ud01 = unseen_depth[:, :-1, 1:]
            ud10 = unseen_depth[:, 1:, :-1]

            m00 = mask[:, :-1, :-1]
            m01 = mask[:, :-1, 1:]
            m10 = mask[:, 1:, :-1]
            
            # unseen_patch_depth_loss = (((ud00 - ud01) ** 2) + ((ud00 - ud10) ** 2))
            unseen_patch_depth_loss = torch.abs(ud00 - ud01) * torch.mul(m00, m01) + torch.abs(ud00 - ud10) * torch.mul(m00, m10)
            patch_depth_loss = torch.cat([patch_depth_loss, unseen_patch_depth_loss], dim=0)

        patch_depth_loss = patch_depth_loss.mean()
        return patch_depth_loss
    
    def get_patch_normal_smooth_loss(self, model_outputs, patch_size):
        normal = model_outputs['patch_normal_map'].reshape(-1, patch_size, patch_size, 3) # (B, PS, PS, 3)
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)

        acc = model_outputs['patch_acc'].reshape(-1, patch_size, patch_size)
        mask = (acc > 0.95).detach()
        
        n00 = torch.nn.functional.normalize(normal[:, :-1, :-1], p=2, dim=-1)
        n01 = torch.nn.functional.normalize(normal[:, :-1, 1:], p=2, dim=-1)
        n10 = torch.nn.functional.normalize(normal[:, 1:, :-1], p=2, dim=-1)

        m00 = mask[:, :-1, :-1]
        m01 = mask[:, :-1, 1:]
        m10 = mask[:, 1:, :-1]
        # n01 = normal[:, :-1, 1:]
        # n10 = normal[:, 1:, :-1]
        
        # n00 = n00 / (n00.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
        # n01 = n01 / (n01.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
        # n10 = n10 / (n10.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
        
        # normal_loss01 = torch.norm(n00 - n01, dim=-1)
        # normal_loss10 = torch.norm(n00 - n10, dim=-1)
        # patch_normal_loss = (normal_loss01 + normal_loss10)

        normal_loss01 = (torch.abs(n00 - n01).sum(dim=-1) + (1. - torch.sum(n00 * n01, dim=-1))) * torch.mul(m00, m01)
        normal_loss10 = (torch.abs(n00 - n10).sum(dim=-1) + (1. - torch.sum(n00 * n10, dim=-1))) * torch.mul(m00, m10)
        patch_normal_loss = (normal_loss01 + normal_loss10)

        if model_outputs['unseen_patch_normal_map'] is not None:
            unseen_normal = model_outputs['unseen_patch_normal_map'].reshape(-1, patch_size, patch_size, 3) # (B, PS, PS, 3)
            
            # un00 = unseen_normal[:, :-1, :-1]
            # un01 = unseen_normal[:, :-1, 1:]
            # un10 = unseen_normal[:, 1:, :-1]
            
            # un00 = un00 / (un00.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
            # un01 = un01 / (un01.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
            # un10 = un10 / (un10.norm(2,dim=-1).unsqueeze(-1) + 1e-5)
            
            # unseen_normal_loss01 = torch.norm(un00 - un01, dim=-1)
            # unseen_normal_loss10 = torch.norm(un00 - un10, dim=-1)
            # unseen_patch_normal_loss = (unseen_normal_loss01 + unseen_normal_loss10)

            unseen_normal = torch.nn.functional.normalize(unseen_normal, p=2, dim=-1)

            acc = model_outputs['unseen_patch_acc'].reshape(-1, patch_size, patch_size)
            mask = (acc > 0.95).detach()
            
            un00 = torch.nn.functional.normalize(unseen_normal[:, :-1, :-1], p=2, dim=-1)
            un01 = torch.nn.functional.normalize(unseen_normal[:, :-1, 1:], p=2, dim=-1)
            un10 = torch.nn.functional.normalize(unseen_normal[:, 1:, :-1], p=2, dim=-1)

            m00 = mask[:, :-1, :-1]
            m01 = mask[:, :-1, 1:]
            m10 = mask[:, 1:, :-1]

            unseen_normal_loss01 = (torch.abs(un00 - un01).sum(dim=-1) + (1. - torch.sum(un00 * un01, dim=-1))) * torch.mul(m00, m01)
            unseen_normal_loss10 = (torch.abs(un00 - un10).sum(dim=-1) + (1. - torch.sum(un00 * un10, dim=-1))) * torch.mul(m00, m10)
            unseen_patch_normal_loss = (unseen_normal_loss01 + unseen_normal_loss10)
            
            patch_normal_loss = torch.cat([patch_normal_loss, unseen_patch_normal_loss], dim=0)

        patch_normal_loss = patch_normal_loss.mean()
        return patch_normal_loss

    def get_entropy_loss(self, model_outputs):
        # seen
        sigma = model_outputs['sigma']
        acc = model_outputs['acc']

        ray_prob = sigma / (torch.sum(sigma, -1).unsqueeze(-1) + 1e-10)
        ray_entropy = self.entropy(ray_prob)

        # thershold by accuracy
        mask = (acc > self.entropy_acc_thresh).detach().unsqueeze(-1)
        ray_entropy *= mask
        
        # unseen
        if model_outputs['unseen_sigma'] is not None:
            unseen_sigma = model_outputs['unseen_sigma']
            unseen_acc = model_outputs['unseen_acc']

            unseen_ray_prob = unseen_sigma / (torch.sum(unseen_sigma, -1).unsqueeze(-1) + 1e-10)
            unseen_ray_entropy = self.entropy(unseen_ray_prob)

            # thershold by accuracy
            mask = (unseen_acc > self.entropy_acc_thresh).detach().unsqueeze(-1)
            unseen_ray_entropy *= mask

            ray_entropy = torch.cat([ray_entropy, unseen_ray_entropy], dim=0)

        if self.entropy_log_scaling:
            entropy_loss = torch.log(ray_entropy.mean() + 1e-10)
        else:
            entropy_loss = ray_entropy.mean()
        return entropy_loss

    def entropy(self, prob):
        return -1*prob*torch.log2(prob+1e-10)
        return prob*torch.log2(1-prob)

    """
    def get_patch_rgb_ncc_loss(self, model_outputs, rgb_gt, img_res, patch_size):
        rgb_gt = rgb_gt.permute(0, 2, 1).reshape(-1, 3, img_res[0], img_res[1])
        patch_grid = model_outputs['unseen_patch_grid_uv']
        patch_mask = model_outputs['unseen_patch_grid_mask']
        sampled_rgb_values = torch.nn.functional.grid_sample(rgb_gt, patch_grid, align_corners=True).squeeze(-1).permute(0, 2, 1).squeeze(0)

        unseen_rgb = model_outputs['unseen_patch_rgb_values'].reshape(-1, patch_size, patch_size, 3)
        sampled_rgb_values = sampled_rgb_values.reshape(-1, patch_size, patch_size, 3)
        patch_mask = patch_mask.reshape(-1, patch_size, patch_size)

        # ncc_loss = 1 - self.normalized_cross_corrleation(self.rgb_to_gray(sampled_rgb_values), self.rgb_to_gray(unseen_rgb))
        if patch_mask.sum() == 0:
            ncc_loss = torch.tensor(0.0).cuda().float()
        else:
            ncc_loss = torch.abs(unseen_rgb - sampled_rgb_values)[patch_mask]
            ncc_loss = ncc_loss.mean()
        return ncc_loss, unseen_rgb.detach(), sampled_rgb_values.detach()

    def rgb_to_gray(self, img):
        # img: [N, H, W, 3]
        rgb_img = img.permute(0, 3, 1, 2)
        rgb2gray = torchvision.transforms.Grayscale()
        gray_img = rgb2gray(rgb_img)
        return gray_img

    def normalized_cross_corrleation(self, patch1, patch2):
        N_patch, img_res = patch1.shape[0], patch1.shape[1:]
        patch1 = patch1.view(N_patch, -1)
        patch2 = patch2.view(N_patch, -1)
        
        product = torch.mean((patch1 - torch.mean(patch1, dim=-1, keepdim=True)) * (patch2 - torch.mean(patch2, dim=-1, keepdim=True)), dim=-1)
        std = torch.std(patch1, dim=-1) * torch.std(patch2, dim=-1)
        std = torch.clamp(std, 1e-8)
        ncc = product / std

        return ncc
    """
        
    
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), (depth_gt * 50 + 0.5).reshape(1, 32, 32), mask.reshape(1, 32, 32))
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos

    def masked_patch_loss(self, rgb_values, rgb_gt, warp_mask):
        npx, nsrc, npatch, _ = rgb_values.shape

        warp_mask = warp_mask.float()

        if self.patch_rgb_loss == "l1":
            num = torch.sum(warp_mask.unsqueeze(-1).unsqueeze(-1) * torch.abs(rgb_values - rgb_gt.unsqueeze(1)),
                            dim=1).sum(dim=1).sum(dim=1) / npatch

        elif self.patch_rgb_loss == "ssim":
            self.ssim = self.ssim.to(rgb_values.device)
            num = torch.sum(warp_mask * self.ssim(rgb_values, rgb_gt), dim=1)

        else:
            raise NotImplementedError("Patch loss + " + self.patch_rgb_loss)

        denom = torch.sum(warp_mask, dim=1)

        valids = denom > self.min_visibility
        
        return torch.sum(num[valids] / denom[valids]), valids

    def masked_pixel_loss(self,rgb_values, rgb_gt, warp_mask):
        npx, nsrc, _ = rgb_values.shape

        if warp_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.ones_like(warp_mask[:, 0])

        warp_mask = warp_mask.float()

        # pdb.set_trace()

        num = torch.sum(warp_mask.unsqueeze(2) * torch.abs(rgb_values - rgb_gt.unsqueeze(1)), dim=1).sum(dim=1)
        denom = torch.sum(warp_mask, dim=1)

        valids = denom > self.min_visibility
        rgb_loss = torch.sum(num[valids] / denom[valids])

        return rgb_loss, valids

    def set_patch_offset(self, h_patch_size):
        self.h_patch_size = h_patch_size
        self.ssim = SSIM(self.h_patch_size)
        self.patch_offset = rend_util.build_patch_offset(self.h_patch_size)
        
    def forward(self, model_input, model_outputs, ground_truth, reg_patch_size, img_res):
        rgb_gt = ground_truth['rgb'].cuda()
        full_rgb_gt = ground_truth['full_rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)

        warped_rgb_loss = torch.tensor(0.0).cuda().float()
        if model_outputs['warped_rgb_vals'] is not None:
            warped_rgb_val = model_outputs['warped_rgb_vals']
            patch_loss = len(warped_rgb_val.shape) == 4
            uv = model_input['uv']
            num_pixels = uv.shape[1]
            i, j = uv[0, :, 1].long(), uv[0, :, 0].long()
            full_rgb_gt = ground_truth['full_rgb'].cuda()
            full_rgb_gt = full_rgb_gt.permute(0, 2, 1).reshape(-1, 3, img_res[0], img_res[1])
            self.patch_offset = self.patch_offset.to(uv.device)
            
            if patch_loss:
                rgb_patches_gt = full_rgb_gt[0, :, i.unsqueeze(1) + self.patch_offset[..., 1],
                                            j.unsqueeze(1)+ self.patch_offset[..., 0]].permute(1, 2, 0)
            else:
                rgb_gt = full_rgb_gt[0, :, i, j].view(3, -1).t()

            warp_mask = model_outputs['warping_mask']
            occlusion_mask = model_outputs['occlusion_mask']
            if occlusion_mask is not None:
                mask = warp_mask * (1 - occlusion_mask)
            else:
                mask = warp_mask
            
            if patch_loss:
                mask = mask * model_outputs['valid_hom_mask']
                warped_rgb_loss, _ = self.masked_patch_loss(warped_rgb_val, rgb_patches_gt, mask)
            else:
                warped_rgb_loss, _ = self.masked_pixel_loss(warped_rgb_val, rgb_gt, mask)
            warped_rgb_loss /= num_pixels
        
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask)
        # depth_loss = 0.0
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)

        # neighbour normal smooth loss
        normal_smooth_loss = self.get_normal_smooth_loss(model_outputs)

        # patch depth / normal smooth loss (seen / unseen pose)
        patch_depth_smooth_loss = torch.tensor(0.0).cuda().float()
        patch_normal_smooth_loss = torch.tensor(0.0).cuda().float()
        if model_outputs['patch_depth_values'] is not None:
            patch_depth_smooth_loss = self.get_patch_depth_smooth_loss(model_outputs, reg_patch_size)
        if model_outputs['patch_normal_map'] is not None:
            patch_normal_smooth_loss = self.get_patch_normal_smooth_loss(model_outputs, reg_patch_size)

        # patch rgb normalized cross correlation loss (seen to unseen pose)
        # if model_outputs['unseen_patch_grid_uv'] is not None:
        #     patch_rgb_ncc_loss, unseen_rgb, sampled_rgb = self.get_patch_rgb_ncc_loss(model_outputs, full_rgb_gt, img_res, reg_patch_size)

        # ray entropy loss (seen / unseen pose)
        entropy_loss = self.get_entropy_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-(self.step - self.start_reg_step) / (self.end_step - self.start_reg_step) * 10.)
        else:
            decay = 1.0

        if self.step < self.start_reg_step:
            # patch_depth_smooth_loss = torch.tensor(0.0).cuda().float()
            # patch_normal_smooth_loss = torch.tensor(0.0).cuda().float()
            # entropy_loss = torch.tensor(0.0).cuda().float()
            # # patch_rgb_ncc_loss = torch.tensor(0.0).cuda().float()
            # warped_rgb_loss = torch.tensor(0.0).cuda().float()
            entropy_weight = 0
            patch_normal_smooth_weight = 0
            patch_depth_smooth_weight = 0
            # patch_rgb_ncc_weight = 0
            warped_rgb_weight = 0
        else:
            entropy_weight = self.entropy_weight
            patch_normal_smooth_weight = self.patch_normal_smooth_weight
            patch_depth_smooth_weight = self.patch_depth_smooth_weight
            # patch_rgb_ncc_weight = self.patch_rgb_ncc_weight
            warped_rgb_weight = self.warped_rgb_weight


        # detach loss which set weight as zero
        if self.eikonal_weight == 0:
            eikonal_loss = eikonal_loss.detach()
        if self.normal_smooth_weight == 0:
            normal_smooth_loss = normal_smooth_loss.detach()
        if patch_depth_smooth_weight == 0:
            patch_depth_smooth_loss = patch_depth_smooth_loss.detach()
        if patch_normal_smooth_weight == 0:
            patch_normal_smooth_loss = patch_normal_smooth_loss.detach()
        if entropy_weight == 0:
            entropy_loss = entropy_loss.detach()
        # if patch_rgb_ncc_weight == 0:
        #     patch_rgb_ncc_loss = patch_rgb_ncc_loss.detach()
        if warped_rgb_weight == 0:
            warped_rgb_loss = warped_rgb_loss.detach()
        if self.depth_weight == 0:
            depth_loss = depth_loss.detach()
        if self.normal_l1_weight == 0:
            normal_l1 = normal_l1.detach()
        if self.normal_cos_weight == 0:
            normal_cos = normal_cos.detach()
        
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.normal_smooth_weight * normal_smooth_loss +\
               patch_depth_smooth_weight * patch_depth_smooth_loss +\
               patch_normal_smooth_weight * patch_normal_smooth_loss +\
               entropy_weight * entropy_loss +\
               decay * warped_rgb_weight * warped_rgb_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'normal_smooth_loss': normal_smooth_loss,
            'patch_depth_smooth_loss': patch_depth_smooth_loss,
            'patch_normal_smooth_loss': patch_normal_smooth_loss,
            'entropy_loss': entropy_loss,
            # 'patch_rgb_ncc_loss': patch_rgb_ncc_loss,
            'warped_rgb_loss': warped_rgb_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos,
            # 'unseen_rgb': unseen_rgb,
            # 'sampled_rgb': sampled_rgb
        }

        return output

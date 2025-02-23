import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth

import torch.distributed as dist
import torchvision.utils as vutils
import pdb

class MonoSDFTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        self.expcommnet = self.conf.get_string('train.expcomment')
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            self.timestamp = self.timestamp + '_' + self.expcommnet
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'depth'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'normal'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'rendering'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'warp'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'merge'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'surface'))

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        # if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=0)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        self.h_patch_size = self.model.h_patch_size
        self.loss.set_patch_offset(self.h_patch_size)

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.reg_patch_size = self.conf.get_int('train.reg_patch_size')
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.start_reg_step = self.conf.get_int('train.start_reg_step', default=0)
        self.warp_pixel_patch_both = self.conf.get_int('train.warp_pixel_patch_both', default=True)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        print("training...")
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()
                self.model.module.h_patch_size = 0

                self.train_dataset.change_sampling_idx(-1, self.h_patch_size)
                self.train_dataset.change_sampling_patch_idx(-1, self.h_patch_size)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach(),
                         'depth_values': out['depth_values'].detach(),
                         'warped_rgb_vals': out['warped_rgb_vals_pixel'].detach(),
                         'warping_mask': out['warping_mask_pixel'].detach(),
                         'occlusion_mask': out['occlusion_mask_pixel'].detach()}
                    if 'rgb_un_values' in out:
                        d['rgb_un_values'] = out['rgb_un_values'].detach()
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])

                plt.plot(self.model.module.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.img_res,
                        **self.plot_conf
                        )

                self.model.train()
                self.model.module.h_patch_size = self.h_patch_size
                del model_outputs
                del res
                torch.cuda.empty_cache()

            # if epoch <= self.start_reg_step:
            if epoch <= self.nepochs / 2:
                # pixel warp
                if epoch == 0:
                    tmp_h_patch_size = self.h_patch_size
                    self.h_patch_size = 0
                    self.model.module.update_h_patch_size(0)
                start_reg = False
            else:
                self.h_patch_size = tmp_h_patch_size
                self.model.module.update_h_patch_size(self.h_patch_size)
                self.model.module.warp_pixel_patch_both = self.warp_pixel_patch_both
                start_reg = True
                
            self.train_dataset.change_sampling_idx(self.num_pixels, self.h_patch_size)
            self.train_dataset.change_sampling_patch_idx(self.num_pixels, self.reg_patch_size)
            self.train_dataset.change_unseen_sampling_idx(self.num_pixels, self.h_patch_size)
            self.train_dataset.change_unseen_sampling_patch_idx(self.num_pixels, self.reg_patch_size)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_input["patch_uv"] = model_input["patch_uv"].cuda()
                model_input["unseen_uv"] = model_input["unseen_uv"].cuda()
                model_input["unseen_pose"] = model_input["unseen_pose"].cuda()
                model_input["unseen_patch_uv"] = model_input["unseen_patch_uv"].cuda()
                model_input["inv_pose"] = model_input["inv_pose"].cuda()
                model_input["inv_intrinsics"] = model_input["inv_intrinsics"].cuda()
                model_input["src_rgb"] = model_input["src_rgb"].cuda()
                model_input["src_pose"] = model_input["src_pose"].cuda()
                model_input["src_inv_pose"] = model_input["src_inv_pose"].cuda()
                model_input["src_intrinsics"] = model_input["src_intrinsics"].cuda()
                model_input["src_inv_intrinsics"] = model_input["src_inv_intrinsics"].cuda()
                # model_input["relative_pose"] = model_input["relative_pose"].cuda()
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(model_input, indices, self.img_res)
                
                loss_output = self.loss(model_input, model_outputs, ground_truth, self.reg_patch_size, self.img_res, reg=start_reg)
                loss = loss_output['loss']
                loss.backward()
                self.optimizer.step()
                
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                del model_outputs
                self.iter_step += 1                
                
                if self.GPU_INDEX == 0:
                    print(
                        '{0}_{1} [{2}/{11}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}, patch_size={12}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item(),
                                    indices[0],
                                    self.h_patch_size))
                    
                    self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
                    self.writer.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/normal_smooth_loss', loss_output['normal_smooth_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/patch_depth_smooth_loss', loss_output['patch_depth_smooth_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/patch_normal_smooth_loss', loss_output['patch_normal_smooth_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/entropy_loss', loss_output['entropy_loss'].item(), self.iter_step)
                    # self.writer.add_scalar('Loss/patch_rgb_ncc_loss', loss_output['patch_rgb_ncc_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/warped_rgb_loss_patch', loss_output['warped_rgb_loss_patch'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/warped_rgb_loss_pixel', loss_output['warped_rgb_loss_pixel'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), self.iter_step)
                    self.writer.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), self.iter_step)

                    # if self.iter_step % 50 == 0:
                    #     unseen_rgb = loss_output['unseen_rgb'][:4].permute(0,3,1,2)
                    #     sampled_rgb = loss_output['sampled_rgb'][:4].permute(0,3,1,2)
                    #     output_img = torch.cat([unseen_rgb, sampled_rgb], dim=0)
                    #     output_img = vutils.make_grid(output_img, 4)
                    #     self.writer.add_image('patch_rgb', output_img, self.iter_step)
                    
                    self.writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item(), self.iter_step)
                    self.writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item(), self.iter_step)
                    self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                    
                    if self.Grid_MLP:
                        self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                        self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                        self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                
                self.train_dataset.change_sampling_idx(self.num_pixels, self.h_patch_size)
                self.train_dataset.change_sampling_patch_idx(self.num_pixels, self.reg_patch_size)
                self.train_dataset.change_unseen_sampling_idx(self.num_pixels, self.h_patch_size)
                self.train_dataset.change_unseen_sampling_patch_idx(self.num_pixels, self.reg_patch_size)
                self.scheduler.step()

        self.save_checkpoints(epoch)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        # depth_map = depth_map * scale + shift

        warp_rgb = model_outputs['warped_rgb_vals']
        nsrc = warp_rgb.shape[1]
        warp_rgb = warp_rgb.reshape(batch_size, num_samples, nsrc, 3)
        warp_mask = model_outputs[f"warping_mask"].unsqueeze(0) * warp_rgb + (1 - model_outputs[f"warping_mask"]).unsqueeze(0) * torch.tensor([0, 1, 0]).to(warp_rgb.device).float()
        # warp_mask = model_outputs[f"warping_mask"].unsqueeze(0) * torch.tensor([0, 0, 1]).to(warp_rgb.device).float() + (1 - model_outputs[f"warping_mask"]).unsqueeze(0) * torch.tensor([0, 1, 0]).to(warp_rgb.device).float()
        if f"occlusion_mask" in model_outputs:
            occlusion_mask = model_outputs[f"occlusion_mask"]
            inv_occlusion_mask = 1 - occlusion_mask
            warp_mask = inv_occlusion_mask.unsqueeze(0) * warp_mask + occlusion_mask.unsqueeze(0) * torch.tensor([0, 0, 1]).to(warp_rgb.device).float()
        
        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
            "warp_rgb": warp_rgb[:, :, 0, :],
            "warp_mask": warp_mask[:, :, 0, :],
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()

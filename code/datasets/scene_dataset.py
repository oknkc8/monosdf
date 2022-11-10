import os
import pdb
from requests import patch
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 num_views=-1,  
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        # used a fake depth image and normal image
        self.depth_images = []
        self.normal_images = []

        for path in image_paths:
            depth = np.ones_like(rgb[:, :1])
            self.depth_images.append(torch.from_numpy(depth).float())
            normal = np.ones_like(rgb)
            self.normal_images.append(torch.from_numpy(normal).float())
            
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
            
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["mask"] = torch.ones_like(self.depth_images[idx][self.sampling_idx, :])
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 near_pose_type='rot_from_origin',
                 near_pose_rot=5, # degree
                 near_pose_trans=0.1
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.sampling_patch_idx = None
        
        self.unseen_sampling_idx = None
        self.unseen_sampling_patch_idx = None
        self.get_near_pose = utils.GetNearPose(near_pose_type, near_pose_rot, near_pose_trans)
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        self.inv_intrinsics_all = []
        self.inv_pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            # pdb.set_trace()

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.inv_intrinsics_all.append(torch.inverse(torch.from_numpy(intrinsics).float()))
            self.inv_pose_all.append(torch.inverse(torch.from_numpy(pose).float()))

        # pdb.set_trace()
        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            _, self.H, self.W = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        # return self.n_images
        return self.n_images if self.num_views < 0 else self.num_views

    def __getitem__(self, idx):
        src_idxs = None
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            image_ids = [42, 23, 13, 40, 44, 48, 0, 8, 13][:self.num_views] # for dtu scan 24
            # idx = image_ids[random.randint(0, self.num_views - 1)]
            # idx = image_ids[idx % self.num_views]
            
            # src_idx = idx
            # while src_idx == idx:
            #     src_idx = random.randint(0, self.num_views - 1)

            src_idxs = image_ids[:idx] + image_ids[idx+1:]
            idx = image_ids[idx]
            
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "patch_uv": uv
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]
            
        if self.sampling_patch_idx is not None:
            sample["patch_uv"] = uv[self.sampling_patch_idx, :]

        sample["unseen_pose"] = torch.eye(4, dtype=torch.float32)    
        sample["unseen_pose"][:3, :4] = self.get_near_pose(sample["pose"])
        if self.unseen_sampling_idx is not None:
            sample["unseen_uv"] = uv[self.unseen_sampling_idx, :]
            
        if self.unseen_sampling_patch_idx is not None:
            sample["unseen_patch_uv"] = uv[self.unseen_sampling_patch_idx, :]

        if src_idxs is not None:
            sample["inv_pose"] = self.inv_pose_all[idx]
            sample["inv_intrinsics"] = self.inv_intrinsics_all[idx]
            
            sample["src_rgb"] = torch.stack([self.rgb_images[i].transpose(1,0).reshape(3, self.H, self.W) for i in src_idxs])
            sample["src_pose"] = torch.stack([self.pose_all[i] for i in src_idxs])
            sample["src_inv_pose"] = torch.stack([self.inv_pose_all[i] for i in src_idxs])
            sample["src_intrinsics"] = torch.stack([self.intrinsics_all[i] for i in src_idxs])
            sample["src_inv_intrinsics"] = torch.stack([self.inv_intrinsics_all[i] for i in src_idxs])

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size, h_patch_size=0):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            if h_patch_size:
                idx_img = torch.arange(self.total_pixels).view(self.img_res[0], self.img_res[1])
                if h_patch_size > 0:
                    idx_img = idx_img[h_patch_size:-h_patch_size, h_patch_size:-h_patch_size]
                idx_img = idx_img.reshape(-1)
                self.sampling_idx = idx_img[torch.randperm(idx_img.shape[0])[:sampling_size]]
            else:
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
            
    def change_unseen_sampling_idx(self, sampling_size, h_patch_size=0):
        if sampling_size == -1:
            self.unseen_sampling_idx = None
        else:
            if h_patch_size:
                idx_img = torch.arange(self.total_pixels).view(self.img_res[0], self.img_res[1])
                if h_patch_size > 0:
                    idx_img = idx_img[h_patch_size:-h_patch_size, h_patch_size:-h_patch_size]
                idx_img = idx_img.reshape(-1)
                self.unseen_sampling_idx = idx_img[torch.randperm(idx_img.shape[0])[:sampling_size]]
            else:
                self.unseen_sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
    

    # borrowed from RegNeRF
    def change_sampling_patch_idx(self, sampling_size, reg_patch_size=8):
        if sampling_size == -1:
            self.sampling_patch_idx = None
        else:
            n_patches = sampling_size // (reg_patch_size ** 2)
            
            # Sample start locations
            x0 = np.random.randint(0, self.img_res[1] - reg_patch_size + 1, size=(n_patches, 1, 1))
            y0 = np.random.randint(0, self.img_res[0] - reg_patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_xy = xy0 + np.stack(np.meshgrid(np.arange(reg_patch_size), np.arange(reg_patch_size), indexing='xy'), axis=-1).reshape(1, -1, 2)
            
            patch_idx = patch_xy[..., 1] * self.img_res[1] + patch_xy[..., 0]
            self.sampling_patch_idx = torch.tensor(patch_idx, dtype=torch.int64).reshape(-1)
            
    def change_unseen_sampling_patch_idx(self, sampling_size, reg_patch_size=8):
        if sampling_size == -1:
            self.unseen_sampling_patch_idx = None
        else:
            n_patches = sampling_size // (reg_patch_size ** 2)
            
            # Sample start locations
            x0 = np.random.randint(0, self.img_res[1] - reg_patch_size + 1, size=(n_patches, 1, 1))
            y0 = np.random.randint(0, self.img_res[0] - reg_patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_xy = xy0 + np.stack(np.meshgrid(np.arange(reg_patch_size), np.arange(reg_patch_size), indexing='xy'), axis=-1).reshape(1, -1, 2)
            
            patch_idx = patch_xy[..., 1] * self.img_res[1] + patch_xy[..., 0]
            self.unseen_sampling_patch_idx = torch.tensor(patch_idx, dtype=torch.int64).reshape(-1)
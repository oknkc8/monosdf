import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
import numpy as np
import pdb

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_time():
    torch.cuda.synchronize()
    return time.time()

trans_topil = transforms.ToPILImage()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

# borrowed from InfoNeRF
class GetNearPose:
    def __init__(self,
                 near_pose_type='rot_from_origin',
                 near_pose_rot=5,
                 near_pose_trans=0.1):
        super(GetNearPose, self).__init__()
        self.near_pose_type = near_pose_type
        self.near_pose_rot = near_pose_rot
        self.near_pose_trans = near_pose_trans
    
    def __call__(self, c2w):
        c2w = c2w[:3, :4]
        assert (c2w.shape == (3,4))
        
        if self.near_pose_type == 'rot_from_origin':
            return self.rot_from_origin(c2w)
        elif self.near_pose_type == 'random_pos':
            return self.random_pos(c2w)
        elif self.near_pose_type == 'random_dir':
            return self.random_dir(c2w)
   
    def random_pos(self, c2w):     
        c2w[:, -1:] += self.near_pose_trans*torch.randn(3).unsqueeze(-1)
        return c2w 
    
    def random_dir(self, c2w):
        rot = c2w[:3,:3]
        pos = c2w[:3,-1:]
        rot_mat = self.get_rotation_matrix()
        rot = torch.mm(rot_mat, rot)
        c2w = torch.cat((rot, pos), -1)
        return c2w
    
    def rot_from_origin(self, c2w):
        rot = c2w[:3,:3]
        pos = c2w[:3,-1:]
        rot_mat = self.get_rotation_matrix()
        pos = torch.mm(rot_mat, pos)
        rot = torch.mm(rot_mat, rot)
        c2w = torch.cat((rot, pos), -1)
        return c2w

    def get_rotation_matrix(self):
        rotation = self.near_pose_rot

        phi = (rotation*(np.pi / 180.))
        x = np.random.uniform(-phi, phi)
        y = np.random.uniform(-phi, phi)
        z = np.random.uniform(-phi, phi)
        
        rot_x = torch.Tensor([
                    [1,0,0],
                    [0,np.cos(x),-np.sin(x)],
                    [0,np.sin(x), np.cos(x)]
                    ])
        rot_y = torch.Tensor([
                    [np.cos(y),0,-np.sin(y)],
                    [0,1,0],
                    [np.sin(y),0, np.cos(y)]
                    ])
        rot_z = torch.Tensor([
                    [np.cos(z),-np.sin(z),0],
                    [np.sin(z),np.cos(z),0],
                    [0,0,1],
                    ])
        rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
        return rot_mat
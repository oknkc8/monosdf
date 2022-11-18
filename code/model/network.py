import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


from hashencoder.hashgrid import _hash_encode, HashEncoder
class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]

        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x


class MonoSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP:
            self.implicit_network = ImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        sampling_method = conf.get_string('sampling_method', default="errorbounded")
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        
        self.use_patch_reg = conf.get_bool('use_patch_reg', default=False)
        self.use_unseen_pose = conf.get_bool('use_unseen_pose', default=False)  
        self.use_warped_colors = conf.get_bool('use_warped_colors', default=False)
        self.use_occ_detector = conf.get_bool('use_occ_detector', default=False)
        # self.occ_min_distance = conf.get_float('occ_min_distance', default=0.01)

        self.h_patch_size = conf.get_int('h_patch_size', default=False)
        self.warp_pixel_patch_both = False
        self.plane_dist_thresh = 1e-3
        self.z_axis = torch.tensor([0, 0, 1]).float()
        self.offsets = rend_util.build_patch_offset(self.h_patch_size)

    def update_h_patch_size(self, h_patch_size):
        self.h_patch_size = h_patch_size
        self.offsets = rend_util.build_patch_offset(self.h_patch_size)

    def forward_raw(self, uv, pose, intrinsics, indices, eikonal=True, output_rgb=True, warping_params=None):
        
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)


        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        weights, alpha, all_cumulated, dists, density = self.volume_rendering(z_vals, sdf)
        acc_map = torch.sum(weights, -1)

        rgb = None
        rgb_values = None
        if output_rgb:
            rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
            rgb = rgb_flat.reshape(-1, N_samples, 3)
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # white background assumption
        if self.white_bkgd and output_rgb:
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        warped_rgb_vals_patch = None
        warping_mask_patch = None
        valid_hom_mask_patch = None
        warped_rgb_vals_pixel = None
        warping_mask_pixel = None
        valid_hom_mask_pixel = None
        occlusion_mask_patch = None
        occlusion_mask_pixel = None
        if warping_params is not None:
            if self.h_patch_size > 0:
                with torch.no_grad():
                    sampled_dists = torch.norm(points - cam_loc.unsqueeze(1), dim=-1).reshape(-1, 1)

                    N_rays, N_sampled = points.shape[:2]
                    N_pts = N_rays * N_sampled
                    N_src = warping_params["src_intr"].shape[0]
                    
                    normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
                    
                    rot_normals = warping_params["R_ref"] @ normals.unsqueeze(-1)
                    points_in_ref = warping_params["R_ref"] @ points_flat.unsqueeze(-1) + warping_params["t_ref"]
                    d1 = torch.sum(rot_normals * points_in_ref, dim=1).unsqueeze(1)
                    d2 = torch.sum(rot_normals.unsqueeze(1) 
                                * (-warping_params["R_rel"].transpose(1,2) @ warping_params['t_rel']), dim=2)

                    valid_hom = (torch.abs(d1) > self.plane_dist_thresh) & (
                                torch.abs(d1 - d2) > self.plane_dist_thresh) & ((d2 / d1) < 1)

                    valid_hom = valid_hom.view(N_rays, N_sampled, N_src)
                    valid_hom_mask_patch = torch.sum(weights.unsqueeze(1) * valid_hom.transpose(1, 2).float(), dim=2)
                    valid_hom_mask_patch += torch.ones_like(valid_hom_mask_patch) * all_cumulated.unsqueeze(1)

                    d1 = d1.squeeze()
                    sign = torch.sign(d1)
                    sign[sign == 0] = 1
                    d = torch.clamp(torch.abs(d1), 1e-8) * sign

                    H = warping_params["src_intr"].unsqueeze(1) @ (
                            warping_params["R_rel"].unsqueeze(1) 
                        + warping_params["t_rel"].unsqueeze(1) @ rot_normals.view(1, N_pts, 1, 3) 
                        / d.view(1, N_pts, 1, 1)
                    ) @ warping_params["inv_ref_intr"].view(1, 1, 3, 3)

                    # replace invalid homs with fronto-parallel homographies
                    H_invalid = warping_params["src_intr"].unsqueeze(1) @ (
                            warping_params["R_rel"].unsqueeze(1) 
                        + warping_params["t_rel"].unsqueeze(1) @ rot_normals.view(1, N_pts, 1, 3) 
                        / sampled_dists.view(1, N_pts, 1, 1)
                    ) @ warping_params["inv_ref_intr"].view(1, 1, 3, 3)

                    tmp_m = ~valid_hom.view(-1, N_src).t()
                    H[tmp_m] = H_invalid[tmp_m]

                pixels = uv.view(N_rays, 1, 2) + self.offsets.float().to(uv.device)
                N_pixels = pixels.shape[1]
                grid, warp_mask_full = self.patch_homography(H, pixels)

                h, w = warping_params['src_img'].shape[-2:]
                warp_mask_full = warp_mask_full & (grid[..., 0] < (w - self.h_patch_size)) & (grid[..., 1] < (h - self.h_patch_size)) & (grid >= self.h_patch_size).all(dim=-1)
                warp_mask_full = warp_mask_full.view(N_src, N_rays, N_sampled, N_pixels)
                
                grid = torch.clamp(rend_util.normalize(grid, h, w), -10, 10)

                sampled_rgb_val = F.grid_sample(warping_params['src_img'].squeeze(0), grid.view(N_src, -1, 1, 2), align_corners=True).squeeze(-1).transpose(1, 2)
                sampled_rgb_val = sampled_rgb_val.view(N_src, N_rays, N_sampled, N_pixels, 3)
                sampled_rgb_val[~warp_mask_full, :] = 0.5

                warping_mask_patch = warp_mask_full.float().mean(dim=-1)
                warping_mask_patch = torch.sum(weights.unsqueeze(1) * warping_mask_patch.permute(1, 0, 2).float(), dim=2)
                warping_mask_patch += torch.ones_like(warping_mask_patch) * all_cumulated.unsqueeze(1)

                warped_rgb_vals_patch = torch.sum(
                    weights.unsqueeze(-1).unsqueeze(-1) * sampled_rgb_val, dim=2
                ).transpose(0, 1)

            if self.h_patch_size == 0 or self.warp_pixel_patch_both:
                N_rays, N_sampled = points.shape[:2]
                N_pts = N_rays * N_sampled
                N_src = warping_params["src_intr"].shape[0]
                h, w = warping_params['src_img'].shape[-2:]

                # pdb.set_trace()                
                grid_px, in_front = self.project(points_flat, warping_params["inv_src_pose"][:, :3], warping_params["src_intr"])
                grid = rend_util.normalize(grid_px.squeeze(0), h, w, clamp=10)

                warping_mask_full = (in_front.squeeze(0) & (grid < 1).all(dim=-1) & (grid > -1).all(dim=-1))
                # warping_mask_full = ((grid < 1).all(dim=-1) & (grid > -1).all(dim=-1))

                # pdb.set_trace()
                sampled_rgb_vals = F.grid_sample(warping_params['src_img'].squeeze(0), grid.unsqueeze(1), align_corners=True).squeeze(2).transpose(1, 2)
                sampled_rgb_vals[~warping_mask_full, :] = 0.5  # set pixels out of image to grey
                sampled_rgb_vals = sampled_rgb_vals.view(N_src, N_rays, -1, 3)
                warped_rgb_vals_pixel = torch.sum(weights.unsqueeze(-1).unsqueeze(0) * sampled_rgb_vals, dim=2).transpose(0, 1)

                # pdb.set_trace()
                warping_mask_full = warping_mask_full.view(N_src, N_rays, -1).permute(1, 2, 0).float()
                warping_mask_pixel = torch.sum(weights.unsqueeze(-1) * warping_mask_full, dim=1)
                warping_mask_pixel += torch.ones_like(warping_mask_pixel) * all_cumulated.unsqueeze(1)
                if warping_mask_pixel.isnan().sum() != 0:
                    pdb.set_trace()
                valid_hom_mask_pixel = None

            # occlusion mask
            if self.use_occ_detector:
                intersection_points = torch.sum(weights.unsqueeze(-1) * points, dim=1)
                intersection_points += all_cumulated[:, None] * points[:, -1] # background point is the last point (i.e. intersection with world sphere)
                network_object_mask = (all_cumulated < 0.5) # no intersection if background contribution is more than half
                with torch.no_grad():
                    N_src = warping_params['src_pose'].shape[0]
                    N_rays = intersection_points.shape[0]
                    cam_locs_src = warping_params['src_pose'][:, :3, 3]
                    ray_dirs_src = intersection_points.unsqueeze(1) - cam_locs_src.unsqueeze(0)
                    max_dist = torch.norm(ray_dirs_src, dim=-1).reshape(-1) - 0.01
                    ray_dirs_src = F.normalize(ray_dirs_src, dim=-1)

                    cam_locs_src = cam_locs_src.unsqueeze(0).repeat(N_rays, 1, 1).reshape(-1, 3)
                    ray_dirs_src = ray_dirs_src.reshape(-1, 3)
                    # max_dist = max_dist.reshape(-1) - 0.01

                    z_vals_src, _ = self.ray_sampler.get_z_vals(ray_dirs_src, cam_locs_src, self, max_dist)
                    N_samples_src = z_vals.shape[1]

                    points_src = cam_locs_src.unsqueeze(1) + z_vals_src.unsqueeze(2) * ray_dirs_src.unsqueeze(1)
                    points_flat_src = points_src.reshape(-1, 3)

                    dirs_src = ray_dirs_src.unsqueeze(1).repeat(1,N_samples_src,1)
                    dirs_flat_src = dirs_src.reshape(-1, 3)

                    sdf_src = self.implicit_network(points_flat_src)[:,:1]
                    _, alpha_src, _, _, _ = self.volume_rendering(z_vals_src, sdf_src, occ_mask=True)

                    occlusion_mask = 1 - torch.prod(1 - alpha_src, dim=-1)
                    occlusion_mask = occlusion_mask.reshape(N_rays, N_src)
                    if occlusion_mask.isnan().sum() != 0:
                        pdb.set_trace()
                    if warping_mask_patch is not None:
                        occlusion_mask_patch = occlusion_mask.reshape(N_rays, N_src)
                        valid_mask_patch = network_object_mask.unsqueeze(-1) & (warping_mask_patch > 0.5)
                        occlusion_mask_patch[~valid_mask_patch] = 0
                    if warping_mask_pixel is not None:
                        occlusion_mask_pixel = occlusion_mask.reshape(N_rays, N_src)
                        valid_mask_pixel = network_object_mask.unsqueeze(-1) & (warping_mask_pixel > 0.5)
                        occlusion_mask_pixel[~valid_mask_pixel] = 0

        
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
            
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'sigma': density,
            'acc': acc_map,
            'alpha': alpha,
            'dists': dists,
            'normal_map': normal_map,
            'warped_rgb_vals_patch': warped_rgb_vals_patch,
            'warping_mask_patch': warping_mask_patch,
            'valid_hom_mask_patch': valid_hom_mask_patch,
            'warped_rgb_vals_pixel': warped_rgb_vals_pixel,
            'warping_mask_pixel': warping_mask_pixel,
            'valid_hom_mask_pixel': valid_hom_mask_pixel,
            'occlusion_mask_patch': occlusion_mask_patch,
            'occlusion_mask_pixel': occlusion_mask_pixel,
        }
        
        if self.training and eikonal:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
                   
            grad_theta = self.implicit_network.gradient(eikonal_points)
            
            # split gradient to eikonal points and neighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        return output

    def forward(self, input, indices, img_res=None):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        warping_params = None
        
        if self.use_warped_colors:
            # ref_intrinsics = input["intrinsics"]
            src_intr = input['src_intrinsics'][0][:, :3, :3]
            inv_ref_intr = input['inv_intrinsics'][0][:3, :3]
            src_img = input['src_rgb']

            ref_pose = input["pose"][0]
            src_pose = input["src_pose"][0]
            inv_src_pose = input["src_inv_pose"][0]
            inv_ref_pose = input["inv_pose"][0]

            relative_proj = inv_src_pose @ ref_pose
            R_rel = relative_proj[:, :3, :3]
            t_rel = relative_proj[:, :3, 3:]
            R_ref = inv_ref_pose[:3, :3]
            t_ref = inv_ref_pose[:3, 3:]

            warping_params = {
                "src_img": src_img,
                "src_intr": src_intr,
                "inv_ref_intr": inv_ref_intr,
                "ref_pose": ref_pose,
                "src_pose": src_pose,
                "inv_src_pose": inv_src_pose,
                "inv_ref_pose": inv_ref_pose,
                "relative_proj": relative_proj,
                "R_rel": R_rel,
                "t_rel": t_rel,
                "R_ref": R_ref,
                "t_ref": t_ref,
            }
    
        output = self.forward_raw(input["uv"], input["pose"], input["intrinsics"], indices, warping_params=warping_params)
        
        ################################
        #  forward for patched pixels
        ################################
        output['patch_depth_values'] = None
        output['patch_normal_map'] = None
        if self.training and self.use_patch_reg:
            output_patch = self.forward_raw(input["patch_uv"], input["pose"], input["intrinsics"], indices, eikonal=False, output_rgb=False)
            output['patch_depth_values'] = output_patch['depth_values']
            output['patch_normal_map'] = output_patch['normal_map']
            output['patch_acc'] = output_patch['acc']
            del output_patch
        
        
        ################################
        #  forward for unseen pose
        ################################
        output['unseen_depth_values'] = None
        output['unseen_normal_map'] = None
        output['unseen_sdf'] = None
        output['unseen_sigma'] = None
        output['unseen_acc'] = None
        output['unseen_weights'] = None
        output['unseen_alpha'] = None
        output['unseen_dists'] = None
        output['unseen_grad_theta'] = None
        output['unseen_grad_theta_nei'] = None
        output['unseen_patch_depth_values'] = None
        output['unseen_patch_normal_map'] = None
        output['unseen_patch_acc'] = None
        if self.training and self.use_unseen_pose:
            # for entropy loss
            output_unseen = self.forward_raw(input["unseen_uv"], input["unseen_pose"], input["intrinsics"], indices, eikonal=True, output_rgb=False)
            output['unseen_depth_values'] = output_unseen['depth_values']
            output['unseen_normal_map'] = output_unseen['normal_map']
            output['unseen_sdf'] = output_unseen['sdf']
            output['unseen_sigma'] = output_unseen['sigma']
            output['unseen_acc'] = output_unseen['acc']
            output['unseen_weights'] = output_unseen['weights']
            output['unseen_alpha'] = output_unseen['alpha']
            output['unseen_dists'] = output_unseen['dists']
            output['unseen_grad_theta'] = output_unseen['grad_theta']
            output['unseen_grad_theta_nei'] = output_unseen['grad_theta_nei']
            
            if self.use_patch_reg:
                output_patch_unseen = self.forward_raw(input["unseen_patch_uv"], input["unseen_pose"], input["intrinsics"], indices, eikonal=False, output_rgb=True)
                output['unseen_patch_depth_values'] = output_patch_unseen['depth_values']
                output['unseen_patch_normal_map'] = output_patch_unseen['normal_map']
                output['unseen_patch_acc'] = output_patch_unseen['acc']

        return output

    def volume_rendering(self, z_vals, sdf, occ_mask=False):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        if not occ_mask:
            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        else:
            density = density[:, :-1]
            # dists = torch.cat([dist, dist])

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        no_shift_transmittance = torch.exp(-torch.cumsum(free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here
        all_cumulated = no_shift_transmittance[:, -1]

        # alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        # transmittance = torch.cumprod((1 - alpha), dim=1).roll(1, dims=1)
        # all_cumulated = transmittance[:, 0].clone()
        # transmittance[:, 0] = 1
        # weights = alpha * transmittance

        return weights, alpha, all_cumulated, dists, density

    def project(self, points, pose, intr):
        xyz = (intr.unsqueeze(1) @ pose.unsqueeze(1) @ rend_util.add_hom(points).unsqueeze(-1))[..., :3, 0]
        in_front = xyz[..., 2] > 0
        grid = xyz[..., :2] / torch.clamp(xyz[..., 2:], 1e-8)
        return grid, in_front

    def patch_homography(self, H, uv):
        N, Npx = uv.shape[:2]
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        hom_uv = rend_util.add_hom(uv)

        # einsum is 30 times faster
        # tmp = (H.view(Nsrc, N, -1, 1, 3, 3) @ hom_uv.view(1, N, 1, -1, 3, 1)).squeeze(-1).view(Nsrc, -1, 3)
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv).reshape(Nsrc, -1, 3)

        grid = tmp[..., :2] / torch.clamp(tmp[..., 2:], 1e-8)
        mask = tmp[..., 2] > 0
        return grid, mask
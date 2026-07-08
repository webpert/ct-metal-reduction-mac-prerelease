#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from mac_gaussian.utils.general_utils import t2a
from mac_gaussian.utils.system_utils import mkdir_p
from mac_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    inverse_softmax,
    strip_symmetric,
    build_scaling_rotation,
)


EPS = 1e-5



class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()
        self.density_inverse_activation = inverse_softplus
        self.density_res_activation = torch.nn.Sigmoid()
        self.density_res_inverse_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self._density_res = torch.empty(0)
        self._global_density_res = torch.empty(0)   # material basis ratio [1, M]
        self._global_density_control = torch.empty(0)   # global density control for metal gaussians

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.is_metal = torch.empty(0)
        self.vol_mask = torch.empty(0)
        self.vol_contrib_cnt = torch.empty(0)   # int
        self.vol_contrib_cnt_metal = torch.empty(0)   # int
        self.vol_contrib_move_vector = torch.empty(0)   # float, 3
        self.vol_contrib_move_vector_metal = torch.empty(0)   # float, 3
        self.is_splittable = torch.empty(0)
        self.offset_metal = torch.empty(0)
        self.offset_nonmetal = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound                
        self.iteration = 0
        self.densify_until_iter = 0
        self.mac_basis = None
        self.bhc_eta_count = 0      # number of energy bins for eta
        self.bhc_basis_count = 0    # number of basis functions for MAC
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._density_res,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._density_res,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        density = self.density_activation(self._density)       # [P]
        # metal_density_scale = self.get_global_density_control  # [1]
        # metal_mask = self.is_metal.view(-1, 1).float()  # [P] for metal

        # # Mask blending (no in-place op)
        # density = ((1 - metal_mask) + metal_mask * metal_density_scale) * raw_density

        return density

    @property
    def get_global_density_control(self):
        return self.density_activation(self._global_density_control)  # [1]


    @property
    def get_density_res(self):
        return self.density_res_activation(self._density_res)

    
    @property
    def get_global_density_res(self):
        return self.density_res_activation(self._global_density_res)  # [1, M-2]    

    def get_density_res_fused(self, is_metal_fitting_period=False):

        metal_density_res_pad = torch.zeros((1, self.bhc_basis_count)).float().cuda()
        metal_density_res = self.get_global_density_res  # [1, M-2]
        metal_mask = self.is_metal.view(-1, 1).float()  # [P, 1], 1 for metal
        metal_density_res_pad[0,2:] = metal_density_res

        if is_metal_fitting_period:
            nonmetal_density_res_pad = torch.zeros((1, self.bhc_basis_count)).float().cuda()
            nonmetal_density_res_pad[0,0] = 1.0
        else:
            nonmetal_density_res_pad = torch.zeros((self._density_res.shape[0], self.bhc_basis_count)).float().cuda()
            nonmetal_density_res_pad[:,1:3] = self.get_density_res   # [P, 2]            
            
        density_res_fused = (1 - metal_mask) * nonmetal_density_res_pad + metal_mask * metal_density_res_pad

        return density_res_fused

    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )
    
    def cache_from_pcd(self, xyz, density, spatial_lr_scale: float, bhc_density_init_scale: float = 1.0):
        self.cached_xyz = xyz
        self.cached_density = density
        self.cached_spatial_lr_scale = spatial_lr_scale
        self.cached_density_init_scale = bhc_density_init_scale

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float, bhc_density_init_scale: float = 1.0):
        self.spatial_lr_scale = spatial_lr_scale
        
        fused_point_cloud = torch.tensor(xyz).float().cuda()
        P = fused_point_cloud.shape[0]
        print(
            "Initialize gaussians from {} estimated points".format(
                P
            )
        )
        # kschoi, init with some perturbation
        fused_density = (
            self.density_inverse_activation(torch.tensor(density) * bhc_density_init_scale).float().cuda()
        )        
    
        # reference basis: 0:constant, 1:water, 2:al, 3:ti, 4:fe
        init_density_res = 0.5 * torch.ones((P, 1), device="cuda", dtype=torch.float32)
        fused_density_res = (
            self.density_res_inverse_activation(init_density_res)
        )

        init_density_res = torch.exp(-10.0 * torch.ones((1, self.bhc_basis_count), device="cuda", dtype=torch.float32))
        init_density_res[:, 0] = 1.0    # default: aluminum basis, don't care. not used
        fused_global_density_res = (self.density_res_inverse_activation(init_density_res))
        
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((P, 4), device="cuda")
        rots[:, 0] = 1

        new_max_radii2D = torch.zeros((P), device="cuda")
        new_is_metal = torch.zeros((P), dtype=torch.bool, device="cuda")
        new_is_splittable = torch.zeros((P), dtype=torch.bool, device="cuda")
        new_offset_metal = torch.zeros((P, 3), dtype=torch.float, device="cuda")
        new_offset_nonmetal = torch.zeros((P, 3), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self._density_res = nn.Parameter(fused_density_res.requires_grad_(True))
        self._global_density_res = nn.Parameter(fused_global_density_res.requires_grad_(True))
        self._global_density_control = nn.Parameter(
            self.density_inverse_activation(torch.tensor(1.0).requires_grad_(True)).float().cuda()
        )
        self.max_radii2D = new_max_radii2D
        self.is_metal = new_is_metal
        self.is_splittable = new_is_splittable
        self.offset_metal = new_offset_metal
        self.offset_nonmetal = new_offset_nonmetal


    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale * self.scale_bound[-1],
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._density_res],
                "lr": training_args.density_res_lr_init * self.spatial_lr_scale,
                "name": "density_res",
            },
            {
                "params": [self._global_density_control],
                "lr": training_args.global_density_control_lr_init * self.spatial_lr_scale,
                "name": "global_density_control",
            },
            {
                "params": [self._global_density_res],
                "lr": training_args.global_density_res_lr_init * self.spatial_lr_scale,
                "name": "global_density_res",
            },      
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * self.scale_bound[-1],
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * self.scale_bound[-1],
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.global_density_control_scheduler_args = get_expon_lr_func(
            lr_init=training_args.global_density_control_lr_init * self.spatial_lr_scale,
            lr_final=training_args.global_density_control_lr_final * self.spatial_lr_scale,
            max_steps=training_args.global_density_control_lr_max_steps,
        )
        self.global_density_res_scheduler_args = get_expon_lr_func(
            lr_init=training_args.global_density_res_lr_init * self.spatial_lr_scale,
            lr_final=training_args.global_density_res_lr_final * self.spatial_lr_scale,
            max_steps=training_args.global_density_res_lr_max_steps,
        )
        self.density_res_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_res_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_res_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_res_lr_max_steps,
        )        
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

        self.densify_until_iter = training_args.densify_until_iter


    def update_learning_rate(self, iteration):
        self.iteration = iteration
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "global_density_control":
                lr = self.global_density_control_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "global_density_res":
                lr = self.global_density_res_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density_res":
                lr = self.density_res_scheduler_args(iteration)
                param_group["lr"] = lr                
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def update_learning_rate_with_keyword(self, iteration, keyword):
        self.iteration = iteration
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" and keyword == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density" and keyword == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "global_density_control" and keyword == "global_density_control":
                lr = self.global_density_control_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "global_density_res" and keyword == "global_density_res":
                lr = self.global_density_res_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density_res" and keyword == "density_res":
                lr = self.density_res_scheduler_args(iteration)
                param_group["lr"] = lr                
            if param_group["name"] == "scaling" and keyword == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation" and keyword == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr        

    def pause_learning_rate(self, iteration):
        self.iteration = iteration
        lr = 0.0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = lr
            if param_group["name"] == "density":
                param_group["lr"] = lr
            if param_group["name"] == "global_density_control":
                param_group["lr"] = lr
            if param_group["name"] == "global_density_res":
                param_group["lr"] = lr
            if param_group["name"] == "density_res":
                param_group["lr"] = lr                
            if param_group["name"] == "scaling":
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                param_group["lr"] = lr        

    def pause_learning_rate_with_keyword(self, iteration, keyword):
        self.iteration = iteration
        lr = 0.0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" and keyword =="xyz":
                param_group["lr"] = lr
            if param_group["name"] == "density" and keyword =="density":
                param_group["lr"] = lr
            if param_group["name"] == "global_density_control" and keyword =="global_density_control":
                param_group["lr"] = lr
            if param_group["name"] == "global_density_res" and keyword =="global_density_res":
                param_group["lr"] = lr
            if param_group["name"] == "density_res" and keyword =="density_res":
                param_group["lr"] = lr                
            if param_group["name"] == "scaling" and keyword =="scaling":
                param_group["lr"] = lr
            if param_group["name"] == "rotation" and keyword =="rotation":
                param_group["lr"] = lr       

    def set_volume_info(self, volume_origin, nVoxel, sVoxel, scene_scale):
        self.volume_origin = volume_origin
        self.nVoxel = nVoxel
        self.sVoxel = sVoxel
        self.scene_scale = scene_scale

    def _dilation3d_torch(self, vol_mask: torch.Tensor, kernel_size=3, iterations=3):
        import numpy as np
        from scipy.ndimage import binary_dilation

        vol_mask_np = vol_mask.detach().cpu().numpy()

        kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
        dilated_mask_np = binary_dilation(
            vol_mask_np, structure=kernel, iterations=iterations
        ).astype(np.uint8)

        dilated_mask = torch.from_numpy(dilated_mask_np).to(vol_mask.device).float()

        return dilated_mask


    def set_metal_volume(self, volume, metal_mask_threshold=0.08):
        # set metal mask from volume
        # volume can be vol_fdk or vol_mask
        vol_mask_boolean = (volume > metal_mask_threshold)
        tmp_vol_mask = vol_mask_boolean.float() # volume should be torch tensor on cuda
        
        self.vol_mask = tmp_vol_mask
        self.vol_mask_boolean = vol_mask_boolean
        self.vol_mask_shell = self._dilation3d_torch(tmp_vol_mask, kernel_size=3, iterations=3) - tmp_vol_mask
        
    
    def set_metal_gaussians(self, aabb_ratio=0.5, k=3.0):
        """
        Check whether each Gaussian (mu, scale, rotation) overlaps with mask==1 voxels.
        Returns a boolean tensor [B] indicating intersection for each Gaussian.
        """
        dVoxel = torch.tensor(self.sVoxel, device='cuda') / torch.tensor(self.nVoxel, device='cuda')
        mu = self.get_xyz              # [B,3], p_origin
        scale = self.get_scaling       # [B,3], scale
        quat = self.get_rotation       # [B,4], (r, x, y, z)

        # --- 1. Quaternion → Rotation matrix [B,3,3]
        q = quat / quat.norm(dim=-1, keepdim=True)
        r, x, y, z = q.unbind(-1)

        # standard Hamilton convention
        R = torch.stack([
            1 - 2*(y**2 + z**2), 2*(x*y - z*r),     2*(x*z + y*r),
            2*(x*y + z*r),     1 - 2*(x**2 + z**2), 2*(y*z - x*r),
            2*(x*z - y*r),     2*(y*z + x*r),     1 - 2*(x**2 + y**2)
        ], dim=-1).reshape(-1,3,3)

        # --- 2. Compute rotated AABB half-extent
        absR = torch.abs(R)  # rotation matrix absolute value for projection
        s = scale
        half_extent = k * torch.stack([
            absR[:,0,0]*s[:,0] + absR[:,0,1]*s[:,1] + absR[:,0,2]*s[:,2],
            absR[:,1,0]*s[:,0] + absR[:,1,1]*s[:,1] + absR[:,1,2]*s[:,2],
            absR[:,2,0]*s[:,0] + absR[:,2,1]*s[:,1] + absR[:,2,2]*s[:,2],
        ], dim=-1)  # [B,3]

        # --- 3. Translate to world coordinates
        mu_translated = mu - torch.tensor(self.volume_origin, device='cuda') + torch.tensor(self.sVoxel, device='cuda') * 0.5

        AABB_min = mu_translated - half_extent
        AABB_max = mu_translated + half_extent

        # --- 4. Convert to voxel indices
        idx_min = torch.floor(AABB_min / dVoxel)
        idx_max = torch.floor(AABB_max / dVoxel)

        # --- 5. Clamp to grid boundaries
        grid_dim = torch.tensor(self.nVoxel, device='cuda').view(1, 3) - 1

        idx_min_clamped = torch.clamp(idx_min, min=0)
        idx_min_clamped = torch.minimum(idx_min_clamped, grid_dim)
        idx_max_clamped = torch.clamp(idx_max, min=0)
        idx_max_clamped = torch.minimum(idx_max_clamped, grid_dim)

        idx_min = idx_min_clamped.long()
        idx_max = idx_max_clamped.long()

        # --- 6. Check overlap with mask==1
        B = mu.shape[0]        
        results = torch.zeros(B, dtype=torch.bool, device='cuda')
        debug_volume_size = torch.zeros(B, device='cuda')

        for b in range(B):
            x0, y0, z0 = idx_min[b]
            x1, y1, z1 = idx_max[b]
            aabb_volume_size = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
            debug_volume_size[b] = aabb_volume_size
            if aabb_volume_size > 1000:
                continue
            submask = self.vol_mask[x0:x1+1, y0:y1+1, z0:z1+1]
            if torch.mean(submask) > aabb_ratio:
                results[b] = True

        self.is_metal = results


    def set_metal_gaussians_from_cnt(self, dVoxel):        
        # dVoxel: [3]
        M = self.vol_contrib_cnt_metal  # [P]
        N = self.vol_contrib_cnt        # [P]
        eps = 1e-10

        criterion = M / (M + N + eps)
        if self._density.shape[0] > 0:
            hetero_thershold_min = 0.2
            hetero_thershold_max = 0.5
            is_splittable = torch.logical_and(criterion > hetero_thershold_min, criterion < hetero_thershold_max)
            self.is_metal = (criterion >= 0.9)
            self.offset_metal = self.vol_contrib_move_vector_metal / (M.unsqueeze(1) + eps) * dVoxel.cuda().unsqueeze(0)              # [P, 3]
            self.offset_nonmetal = self.vol_contrib_move_vector / (N.unsqueeze(1) + eps) * dVoxel.cuda().unsqueeze(0)     # [P, 3]
        else:
            is_splittable = torch.zeros_like(self.is_metal)

        print("Set metal Gaussians")
        self.is_splittable = is_splittable

    def set_metal_gaussians_from_distance(self, dVoxel):        
        # dVoxel: [3]
        min_dist_to_metal_voxel = self.vol_contrib_move_vector[:,0]
        M = self.vol_contrib_cnt_metal  # [P]
        N = self.vol_contrib_cnt        # [P]
        eps = 1e-10

        criterion = M / (M + N + eps)        

        if self._density.shape[0] > 0:
            self.is_metal = torch.logical_and(min_dist_to_metal_voxel <= 0.5, criterion > 0.5)
        
        self.is_splittable = torch.zeros_like(self.is_metal)
        print("Set metal Gaussians")

    def set_metal_gaussians_all(self):
        self.is_metal = torch.ones((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        self.is_splittable = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")

    
    def update_gaussian_properties(self):
        if self._xyz.grad is not None:
            self._xyz.grad[self.is_metal] = 0.0
        if self._density.grad is not None:
            self._density.grad[self.is_metal] = 0.0
        if self._scaling.grad is not None:
            self._scaling.grad[self.is_metal] = 0.0
        if self._rotation.grad is not None:
            self._rotation.grad[self.is_metal] = 0.0

    




    def get_erank(self):
        scales = self.get_scaling       # [N, 3]
        S = scales.abs().sum(axis=-1)   # [N, 1]
        q = scales / S.unsqueeze(-1).repeat(1,3)     # [N, 3]
        erank = torch.exp(-torch.sum(q * torch.log(q), axis=-1))                    # [N, 1]
        return erank

    def get_sparsity_loss(self):
        h = self.get_density_res

        if len(h) == 0:
            return torch.tensor(0.0, device="cuda")
            

        loss = (h * ((h-1)**2)).mean()
    
        return loss
    
    def get_lower_basis_loss(self):
        h = (self.get_density_res - 1.0) ** 2

        if len(h) == 0:
            return torch.tensor(0.0, device="cuda")

        loss = h.mean()
    
        return loss
    

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        N = xyz.shape[0]
        normals = np.zeros_like(xyz)
        f_dc_o = torch.ones((N, 3, 1)).float()
        f_rest_o = torch.zeros((N, 3, 15)).float()
        f_dc = f_dc_o.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest_o.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc_o.shape[1]*f_dc_o.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest_o.shape[1]*f_rest_o.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        densities_res_new = self.density_res_inverse_activation(
            torch.ones((self.get_density.shape[0], self.bhc_basis_count)) / self.bhc_basis_count
            ).float().cuda()

        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_res_new, "density_res")
        self._density_res = optimizable_tensors["density_res"]    

    def reset_density_res(self):
        with torch.no_grad():
            densities_res_new = self.density_res_inverse_activation(
                torch.ones((self.get_density.shape[0], self.bhc_basis_count)) / self.bhc_basis_count
            ).float().cuda()      
            self._density_res[:] = densities_res_new
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_res_new, "density_res")
        self._density_res = optimizable_tensors["density_res"]  


    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        try:
            self._density_res = nn.Parameter(
                torch.tensor(
                    data["density_res"], dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )        
        except Exception as e:
            self._density_res = self.density_res_inverse_activation(torch.ones((self._density.shape[0], self.bhc_basis_count)) / self.bhc_basis_count).float().cuda()
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.scale_bound = data["scale_bound"]

        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("global_"): # kschoi
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._density_res = optimizable_tensors["density_res"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.is_metal = self.is_metal[valid_points_mask]
        self.is_splittable = self.is_splittable[valid_points_mask]
        self.offset_metal = self.offset_metal[valid_points_mask]
        self.offset_nonmetal = self.offset_nonmetal[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"].startswith("global_"): # kschoi
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_densities_res,
        new_scaling,
        new_rotation,
        new_max_radii2D,
        new_is_metal,
        new_is_splittable,
        new_offset_metal,
        new_offset_nonmetal,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "density_res": new_densities_res,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._density_res = optimizable_tensors["density_res"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)
        self.is_metal = torch.cat([self.is_metal, new_is_metal], dim=-1)
        self.is_splittable = torch.cat([self.is_splittable, new_is_splittable], dim=-1)
        self.offset_metal = torch.cat([self.offset_metal, new_offset_metal], dim=0)
        self.offset_nonmetal = torch.cat([self.offset_nonmetal, new_offset_nonmetal], dim=0)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):        
        
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_negated = ~selected_pts_mask

        # default split for large gaussians
        selected_pts_mask_default = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )
        stds = self.get_scaling[selected_pts_mask_default].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask_default]).repeat(N, 1, 1)
        new_xyz_default = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask_default
        ].repeat(N, 1)

        # new split for metal vs non-metal
        selected_pts_mask_material = torch.logical_and(selected_pts_mask_negated, self.is_splittable)
        metal_xyz = self.offset_metal[selected_pts_mask_material] + self.get_xyz[selected_pts_mask_material]
        nonmetal_xyz = self.offset_nonmetal[selected_pts_mask_material] + self.get_xyz[selected_pts_mask_material]
        new_xyz_material = torch.cat([metal_xyz, nonmetal_xyz], dim=0)

        # concat of two splits
        new_xyz = torch.cat([new_xyz_default, new_xyz_material], dim=0)

        # postprocess new attributes (TODO: mask should be computed accordingly)
        new_scaling = self.scaling_inverse_activation(
            torch.cat([
                self.get_scaling[selected_pts_mask_default].repeat(N, 1) / (0.8 * N),
                self.get_scaling[selected_pts_mask_material].repeat(2, 1) / (0.8 * 2)
            ], dim=0)            
        )
        new_rotation = torch.cat([
            self._rotation[selected_pts_mask_default].repeat(N, 1),
            self._rotation[selected_pts_mask_material].repeat(2, 1)
        ], dim=0)        
        new_density = self.density_inverse_activation(
            torch.cat([
                self.get_density[selected_pts_mask_default].repeat(N, 1) * (1 / N),
                self.get_density[selected_pts_mask_material].repeat(N, 1) * (1 / N)
            ], dim=0)
        )
        new_density_res = torch.cat([
            self._density_res[selected_pts_mask_default].repeat(N, 1),
            self._density_res[selected_pts_mask_material].repeat(2, 1)
        ], dim=0)
        new_max_radii2D = torch.cat([
            self.max_radii2D[selected_pts_mask_default].repeat(N),
            self.max_radii2D[selected_pts_mask_material].repeat(2)
        ], dim=0)
        new_is_metal = torch.cat([
            self.is_metal[selected_pts_mask_default].repeat(N),
            self.is_metal[selected_pts_mask_material].repeat(2)
        ], dim=0)
        new_is_splittable = torch.cat([
            self.is_splittable[selected_pts_mask_default].repeat(N),
            self.is_splittable[selected_pts_mask_material].repeat(2)
        ], dim=0)
        new_offset_metal = torch.cat([
            self.offset_metal[selected_pts_mask_default].repeat(N, 1),
            self.offset_metal[selected_pts_mask_material].repeat(2, 1)
        ], dim=0)
        new_offset_nonmetal = torch.cat([
            self.offset_nonmetal[selected_pts_mask_default].repeat(N, 1),
            self.offset_nonmetal[selected_pts_mask_material].repeat(2, 1)
        ], dim=0)


        self.densification_postfix(
            new_xyz,
            new_density,
            new_density_res,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_is_metal,
            new_is_splittable,
            new_offset_metal,
            new_offset_nonmetal
        )

        selected_pts_mask_sum = torch.logical_or(selected_pts_mask_default, selected_pts_mask_material)
        prune_filter = torch.cat(
            (
                selected_pts_mask_sum,
                torch.zeros(N * selected_pts_mask_sum.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):    # 0.005, 4.0
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_densities_res = self._density_res[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        new_is_metal = self.is_metal[selected_pts_mask]
        new_is_splittable = self.is_splittable[selected_pts_mask]
        new_offset_metal = self.offset_metal[selected_pts_mask]
        new_offset_nonmetal = self.offset_nonmetal[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_densities_res,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_is_metal,
            new_is_splittable,
            new_offset_metal,
            new_offset_nonmetal
        )

    def shrink(self):
        selected_pts_mask = self.is_splittable
        
        new_scales = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask] * 0.5
        )
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 8.0
        )
        
        invalid_densities_idx = torch.isinf(new_densities)
        past_densities = self._density[selected_pts_mask]
        new_densities[invalid_densities_idx] = past_densities[invalid_densities_idx]

        self._scaling[selected_pts_mask] = new_scales
        self._density[selected_pts_mask] = new_densities


    def densify_and_prune(
        self,
        max_grad,   # 5e-5
        min_density,    # 1e-5
        max_screen_size,    # None
        max_scale,  # None
        max_num_gaussians,  # 500000
        densify_scale_threshold,    # 4.0
        bbox=None,  # -20,-20,-20, 20,20,20
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # grads.max() 0.0007
        if False:
            scaled_max_grad = max_grad * self.scale_bound[1]    # kschoi, 20250502
        else:
            scaled_max_grad = max_grad * 100.0  # scaled_max_grad(0.005), default scale: 100.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold: # 4.0
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, scaled_max_grad, densify_scale_threshold)
                self.densify_and_split(grads, scaled_max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def prune_in_forbidden(
        self
    ):
        # Prune gaussians with too small density
        prune_mask = (self.is_metal == False).squeeze()
        gaussian_center_in_forbidden = (self.vol_contrib_cnt_metal == 1).squeeze()
        prune_mask = torch.logical_and(prune_mask, gaussian_center_in_forbidden)
        if prune_mask.sum() > 0:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

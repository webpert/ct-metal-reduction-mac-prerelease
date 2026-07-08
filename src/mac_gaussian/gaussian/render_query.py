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
import sys
import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from mac_gaussian.gaussian.gaussian_model import GaussianModel
from mac_gaussian.dataset.projector import Projector
from mac_gaussian.dataset.cameras import Camera
from mac_gaussian.arguments import PipelineParams

def has_invalid_values(input):
    has_nan_or_inf = torch.isnan(input).any() or torch.isinf(input).any()

    return has_nan_or_inf

def query(
    pc: GaussianModel,
    projector: Projector,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    scene_scale = 1.0,        
    nonmetal_only=False,
):
    """
    Query a volume with voxelization.
    """
    pc.vol_contrib_cnt = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.int, device="cuda")
    pc.vol_contrib_cnt_metal = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.int, device="cuda")
    # pc.vol_contrib_move_vector = torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float, device="cuda")
    pc.vol_contrib_move_vector = torch.ones((pc.get_xyz.shape[0], 3), dtype=torch.float, device="cuda") * 1e4
    pc.vol_contrib_move_vector_metal = torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float, device="cuda")
    vol_mask = (pc.vol_mask > 0.9)

    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        mac_basis=torch.as_tensor(pc.mac_basis, device='cuda', dtype=torch.float32),   # cvpr
        vol_mask=vol_mask,
        vol_contrib_cnt=pc.vol_contrib_cnt,
        vol_contrib_cnt_metal=pc.vol_contrib_cnt_metal,
        vol_contrib_move_vector=pc.vol_contrib_move_vector,
        vol_contrib_move_vector_metal=pc.vol_contrib_move_vector_metal,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density_raw = pc.get_density
    if False:
        density = (1 - pc.is_metal.unsqueeze(-1).float()) * density_raw
    else:
        density = density_raw
    density_res = pc.get_density_res

    assert(has_invalid_values(density) == False)
    assert(has_invalid_values(density_res) == False)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,    # P x 3
        opacities=density,  # P
        opacities_res=density_res,  # P x BHC_MAC_BASIS_COUNT
        scales=scales,     # P x 3
        rotations=rotations,    # P x 4
        cov3D_precomp=cov3D_precomp,
    )

    lower_index, higher_index, lower_weight, higher_weight = projector.get_effective_spectrum_index()
    vol_eff = lower_weight * vol_pred[lower_index, ...] + higher_weight * vol_pred[higher_index, ...]

    # "vol": vol_pred[pc.bhc_eta_count // 2 - 1, ...],

    return {
        "vol": vol_eff,
        "vol_lac": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    projector: Projector,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    scene_scale=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        scene_scale_inverse = 1.0 / scene_scale,
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        mac_basis=torch.as_tensor(pc.mac_basis, device='cuda', dtype=torch.float32),   # cvpr
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density
    density_res = pc.get_density_res
    bhc_eta = projector.get_bhc_eta
    bhc_gamma = projector.get_bhc_gamma

    assert(has_invalid_values(density) == False)
    assert(has_invalid_values(density_res) == False)
    assert(has_invalid_values(bhc_eta) == False)
    assert(has_invalid_values(bhc_gamma) == False)    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, debug = rasterizer(
        bhc_gamma=bhc_gamma,
        bhc_eta=bhc_eta,
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        opacities_res=density_res,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "debug": debug,
    }

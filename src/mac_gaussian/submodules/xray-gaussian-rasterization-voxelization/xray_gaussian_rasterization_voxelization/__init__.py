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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    bhc_gamma,
    bhc_eta,
    means3D,
    means2D,
    opacities,
    opacities_res,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        bhc_gamma,
        bhc_eta,
        means3D,
        means2D,
        opacities,
        opacities_res,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        bhc_gamma,
        bhc_eta,
        means3D,
        means2D,
        opacities,
        opacities_res,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (            
            bhc_gamma,
            bhc_eta,
            means3D,
            opacities,
            opacities_res,
            scales,
            rotations,
            raster_settings.scene_scale_inverse,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.mac_basis,   # cvpr
            raster_settings.prefiltered,
            raster_settings.mode,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                num_rendered, color, A, B_k, C_k, D, radii, geomBuffer, binningBuffer, imgBuffer = (
                    _C.rasterize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, color, A, B_k, C_k, D, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.rasterize_gaussians(*args)
            )

        if torch.isnan(color).any() or torch.isinf(color).any() \
            or torch.isnan(A).any() or torch.isinf(A).any() \
            or torch.isnan(B_k).any() or torch.isinf(B_k).any() \
            or torch.isnan(C_k).any() or torch.isinf(C_k).any():
            assert(False)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.mode = raster_settings.mode
        ctx.save_for_backward(
            bhc_gamma,
            bhc_eta,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            A,
            B_k,
            C_k, 
            D,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, D

    @staticmethod
    def backward(ctx, grad_out_color, *_):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        mode = ctx.mode
        (
            bhc_gamma,
            bhc_eta,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            A,
            B_k,
            C_k,
            D,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors


        # Restructure args as C++ method expects them
        args = (
            bhc_gamma,
            bhc_eta,
            means3D,
            A,
            B_k,
            C_k,
            D,
            radii,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            raster_settings.campos,
            raster_settings.mac_basis,   # cvpr
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            mode,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_bhc_gamma,
                    grad_bhc_eta,
                    grad_means2D,
                    grad_opacities,
                    grad_opacities_res,
                    _,  # grad_mu
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_bhc_gamma,
                grad_bhc_eta,
                grad_means2D,
                grad_opacities,
                grad_opacities_res,
                _,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.rasterize_gaussians_backward(*args)

        # means3D,
        # means2D,
        # opacities,
        # opacities_res,
        # scales,
        # rotations,
        # cov3Ds_precomp,
        # raster_settings,            
        grads = (
            grad_bhc_gamma,
            grad_bhc_eta,
            grad_means3D,
            grad_means2D,
            grad_opacities,
            grad_opacities_res,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


def getBhcEtaCount():
    return _C.get_bhc_eta_count()


def getBhcMacBasisCount():
    return _C.get_bhc_mac_basis_count()


# Change according to line 45 gaussian_renderer/__init__.py
class GaussianRasterizationSettings(NamedTuple):
    scene_scale_inverse: float
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    mac_basis: torch.Tensor     # cvpr
    prefiltered: bool
    mode: int
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible
    
    def forward(
        self,
        bhc_gamma,
        bhc_eta,
        means3D,
        means2D,
        opacities,
        opacities_res,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            bhc_gamma,
            bhc_eta,
            means3D,
            means2D,
            opacities,
            opacities_res,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )


class GaussianVoxelizationSettings(NamedTuple):
    scale_modifier: float
    nVoxel_x: int
    nVoxel_y: int
    nVoxel_z: int
    sVoxel_x: float
    sVoxel_y: float
    sVoxel_z: float
    center_x: float
    center_y: float
    center_z: float
    prefiltered: bool
    mac_basis: torch.Tensor
    vol_mask: torch.Tensor
    vol_contrib_cnt: torch.Tensor
    vol_contrib_cnt_metal: torch.Tensor
    vol_contrib_move_vector: torch.Tensor
    vol_contrib_move_vector_metal: torch.Tensor
    debug: bool


def voxelize_gaussians(
    means3D, opacities, opacities_res, scales, rotations, cov3Ds_precomp, voxel_settings
):
    return _VoxelizeGaussians.apply(
        means3D,
        opacities,
        opacities_res,  # cvpr
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings,
    )


class _VoxelizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        opacities,
        opacities_res,  # cvpr
        scales,
        rotations,
        cov3Ds_precomp,
        voxel_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            opacities,
            opacities_res,  # cvpr
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.prefiltered,
            voxel_settings.mac_basis,
            voxel_settings.vol_mask,
            voxel_settings.vol_contrib_cnt,
            voxel_settings.vol_contrib_cnt_metal,
            voxel_settings.vol_contrib_move_vector,
            voxel_settings.vol_contrib_move_vector_metal,
            voxel_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                num_rendered, fields, radii, geomBuffer, binningBuffer, imgBuffer = (
                    _C.voxelize_gaussians(*args)
                )
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            num_rendered, fields, radii, geomBuffer, binningBuffer, imgBuffer = (
                _C.voxelize_gaussians(*args)
            )

        # Keep relevant tensors for backward
        ctx.voxel_settings = voxel_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            opacities_res,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )

        return fields, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        voxel_settings = ctx.voxel_settings
        (
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            opacities_res,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        args = (
            means3D,
            radii,
            scales,
            rotations,
            voxel_settings.scale_modifier,
            cov3Ds_precomp,
            opacities_res,
            grad_out_color,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            voxel_settings.mac_basis,
            voxel_settings.nVoxel_x,
            voxel_settings.nVoxel_y,
            voxel_settings.nVoxel_z,
            voxel_settings.sVoxel_x,
            voxel_settings.sVoxel_y,
            voxel_settings.sVoxel_z,
            voxel_settings.center_x,
            voxel_settings.center_y,
            voxel_settings.center_z,
            voxel_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if voxel_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_opacities,
                    grad_opacities_res,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_scales,
                    grad_rotations,
                ) = _C.voxelize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_opacities,
                grad_opacities_res,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_scales,
                grad_rotations,
            ) = _C.voxelize_gaussians_backward(*args)
        grads = (
            grad_means3D,
            grad_opacities,
            grad_opacities_res,   # grad_opacities_res, if total variation  is used, we can implement it
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianVoxelizer(nn.Module):
    def __init__(self, voxel_settings):
        super().__init__()
        self.voxel_settings = voxel_settings

    def forward(
        self,
        means3D,
        opacities,
        opacities_res,  # cvpr
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):

        voxel_settings = self.voxel_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return voxelize_gaussians(
            means3D,
            opacities,
            opacities_res,  # cvpr
            scales,
            rotations,
            cov3D_precomp,
            voxel_settings,
        )

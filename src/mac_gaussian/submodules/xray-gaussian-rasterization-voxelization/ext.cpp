/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "voxelize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("voxelize_gaussians", &VoxelizeGaussiansCUDA);
  m.def("voxelize_gaussians_backward", &VoxelizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("get_bhc_eta_count", &getBhcEtaCount);  
  m.def("get_bhc_mac_basis_count", &getBhcMacBasisCount);   // cvpr
}
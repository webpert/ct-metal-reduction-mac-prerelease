/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * Modified from code base https://github.com/graphdeco-inria/diff-gaussian-rasterization
 * by Tao Jun Lin
 * 
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
    class Rasterizer
	{
	public:
        
        // Basic Function
		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);
		
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width, int height,
			const float* bhc_gamma,
			const float* bhc_eta,
			const float* means3D,
			const float* opacities,
			const float* opacities_res,
			const float* scales,
			const float scene_scale_inverse,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float* mac_basis,   // cvpr
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			const int mode,
			float* out_color,
			float* out_A,
			float* out_B,
			float* out_C_k,
			float* out_D,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R,
			const int width, int height,
			const float* bhc_gamma,
			const float* bhc_eta,			
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* mac_basis,   // cvpr
			const float tan_fovx, float tan_fovy,
			const float* A,
			const float* B_k,
			const float* C_k,
			const float* D,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dopacity_res,
			float* dL_dmu,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dbhc_eta,
			float* dL_dbhc_gamma,
			const int mode,
			bool debug);
	};

};

#endif
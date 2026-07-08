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
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.append("./")
from mac_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from mac_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from mac_gaussian.utils.general_utils import safe_state
from mac_gaussian.utils.cfg_utils import load_config
from mac_gaussian.utils.log_utils import prepare_output_and_logger
from mac_gaussian.dataset import Scene, Projector
from mac_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from mac_gaussian.utils.image_utils import metric_vol, metric_proj
from mac_gaussian.utils.plot_utils import show_two_slice
from xray_gaussian_rasterization_voxelization import getBhcEtaCount, getBhcMacBasisCount

def sample_tiny_volume_center_from_mask(
        vol_mask, 
        vol_mask_shell, 
        vol_fdk_metal,
        bbox_min, 
        bbox_max, 
        tv_vol_nVoxel,
        tv_vol_sVoxel, 
        voxel_size):
    """
    vol_mask 내 metal voxel 중에서 tiny volume 중심을 랜덤하게 선택합니다.
    단, tiny volume이 bbox 안에 완전히 포함되도록 제한합니다.

    Args:
        vol_mask (torch.Tensor): [D, H, W] 형태의 binary mask (metal=1, nonmetal=0)
        bbox_min (torch.Tensor): [3] 최소 좌표 (x_min, y_min, z_min)
        bbox_max (torch.Tensor): [3] 최대 좌표 (x_max, y_max, z_max)
        tv_vol_sVoxel (float): tiny volume 한 변의 물리적 길이
        voxel_size (float): 한 voxel의 물리적 크기 (같은 단위, 예: mm/voxel)
    Returns:
        torch.Tensor: [3] tiny volume 중심 좌표 (x, y, z)
    """

    # metal voxel 인덱스 추출
    metal_indices = torch.nonzero(vol_mask_shell > 0, as_tuple=False)  # (N, 3)
    if metal_indices.numel() == 0:
        raise ValueError("vol_mask에 metal voxel이 없습니다.")

    # voxel 인덱스를 실제 좌표로 변환
    # (보통 voxel center를 기준으로 변환)
    metal_coords = bbox_min + (metal_indices + 0.5) * voxel_size  # (N, 3)

    # tiny volume 중심이 bbox 내부에 완전히 포함되도록 가능한 좌표 범위 계산
    valid_min = bbox_min + tv_vol_sVoxel / 2
    valid_max = bbox_max - tv_vol_sVoxel / 2

    # 유효 영역 안에 완전히 들어가는 metal voxel만 필터링
    valid_mask = (
        (metal_coords[:, 0] >= valid_min[0]) & (metal_coords[:, 0] <= valid_max[0]) &
        (metal_coords[:, 1] >= valid_min[1]) & (metal_coords[:, 1] <= valid_max[1]) &
        (metal_coords[:, 2] >= valid_min[2]) & (metal_coords[:, 2] <= valid_max[2])
    )

    valid_indices = metal_indices[valid_mask]
    valid_coords = metal_coords[valid_mask]
    if valid_coords.numel() == 0:
        raise ValueError("bbox 내부에 완전히 포함되는 metal voxel이 없습니다.")

    # 무작위로 하나 선택
    rand_idx = torch.randint(0, valid_coords.shape[0], (1,))
    tv_vol_center = valid_coords[rand_idx, :].squeeze(0)
    center_index = valid_indices[rand_idx, :].squeeze(0)

    tv_vol_mask = vol_mask[
        center_index[0] - torch.div(tv_vol_nVoxel[0], 2, rounding_mode='trunc') :
        center_index[0] + torch.div(tv_vol_nVoxel[0], 2, rounding_mode='trunc'),
        center_index[1] - torch.div(tv_vol_nVoxel[1], 2, rounding_mode='trunc') :
        center_index[1] + torch.div(tv_vol_nVoxel[1], 2, rounding_mode='trunc'),
        center_index[2] - torch.div(tv_vol_nVoxel[2], 2, rounding_mode='trunc') :
        center_index[2] + torch.div(tv_vol_nVoxel[2], 2, rounding_mode='trunc'),
    ]
    tv_vol_fdk_metal = vol_fdk_metal[
        center_index[0] - torch.div(tv_vol_nVoxel[0], 2, rounding_mode='trunc') :
        center_index[0] + torch.div(tv_vol_nVoxel[0], 2, rounding_mode='trunc'),
        center_index[1] - torch.div(tv_vol_nVoxel[1], 2, rounding_mode='trunc') :
        center_index[1] + torch.div(tv_vol_nVoxel[1], 2, rounding_mode='trunc'),
        center_index[2] - torch.div(tv_vol_nVoxel[2], 2, rounding_mode='trunc') :
        center_index[2] + torch.div(tv_vol_nVoxel[2], 2, rounding_mode='trunc'),
    ]
    # tv_vol_mask = vol_mask[center_index[0]-tv_vol_nVoxel[0]//2:center_index[0]+tv_vol_nVoxel[0]//2,
    #                        center_index[1]-tv_vol_nVoxel[1]//2:center_index[1]+tv_vol_nVoxel[1]//2,
    #                        center_index[2]-tv_vol_nVoxel[2]//2:center_index[2]+tv_vol_nVoxel[2]//2]
    # tv_vol_fdk_metal = vol_fdk_metal[center_index[0]-tv_vol_nVoxel[0]//2:center_index[0]+tv_vol_nVoxel[0]//2,
    #                        center_index[1]-tv_vol_nVoxel[1]//2:center_index[1]+tv_vol_nVoxel[1]//2,
    #                        center_index[2]-tv_vol_nVoxel[2]//2:center_index[2]+tv_vol_nVoxel[2]//2]

    return tv_vol_center, tv_vol_mask, tv_vol_fdk_metal

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x, y: query(
        x, 
        y,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        scene_scale=scene.scene_scale
    )

    # kschoi, Set up projector
    projector = Projector(dataset)
    scene.projector = projector
    projector.training_setup(opt)

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, opt, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if scene.vol_mask is None:
        scene.vol_mask = (scene.vol_fdk > dataset.bhc_metal_mask_threshold).float()    
    gaussians.set_metal_volume(scene.vol_mask, metal_mask_threshold=0.5)
    vol_fdk_metal_raw = scene.vol_fdk * gaussians.vol_mask
    metal_mu = vol_fdk_metal_raw[gaussians.vol_mask_boolean].mean()
    vol_fdk_metal = gaussians.vol_mask * metal_mu
    vol_mask_boolean = gaussians.vol_mask_boolean
    vol_mask_shell = gaussians.vol_mask_shell

    gaussians.set_volume_info(
        scanner_cfg["offOrigin"], 
        scanner_cfg["nVoxel"], 
        scanner_cfg["sVoxel"], 
        scene.scene_scale)
    
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
    tv_vol_size = opt.tv_vol_size
    tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
    tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        if iteration < 2000:
            gaussians.update_learning_rate(iteration)
            projector.pause_learning_rate(iteration)
        else:
            gaussians.update_learning_rate(iteration)
            projector.update_learning_rate(iteration)        

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, projector, pipe, scene_scale=scene.scene_scale)
        image, viewspace_point_tensor, visibility_filter, radii, _ = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["debug"],
        )
        
        loss = {"total": 0.0}
        gt_image = viewpoint_cam.original_image.cuda()        
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim

        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_tot = query(
                gaussians,
                projector,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                scene_scale=scene.scene_scale
            )
            vol_pred = vol_tot["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )

            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                projector.optimizer.step()
                projector.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            for param_group in projector.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]                
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, z: render(x, y, z, pipe, scene_scale=scene.scene_scale),
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar(f"train/bhc_gamma", scene.projector.get_bhc_gamma.detach().cpu().item(), iteration)    # kschoi
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        scene.projector,
                    )
                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )

        vol_tot = queryFunc(scene.gaussians, scene.projector)
        vol_pred = vol_tot["vol"]

        vol_gt = scene.vol_gt
        vol_mask = scene.vol_mask

        M = vol_mask
        G = torch.where(M > 0.5, 0, vol_gt)
        V = torch.where(M > 0.5, 0, vol_pred)
        psnr_3d, _ = metric_vol(G, V, "psnr", pixel_max=G.max())
        ssim_3d, ssim_3d_axis = metric_vol(G, V, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            vmax = np.percentile(vol_gt.cpu().detach().numpy(), scene.view_percentile)
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vmax,
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)

            projector = scene.projector
            bhc_eta = projector.get_bhc_eta.detach().cpu().numpy()
            bhc_eta_gamma = projector.get_bhc_gamma.detach().cpu().numpy()
            bhc_eta_xaxis = np.linspace(1-bhc_eta_gamma, 1+bhc_eta_gamma, bhc_eta.shape[0])
            gt_spectrum = projector.get_gt_spectrum_view.astype(np.float32)
            bhc_gt = gt_spectrum[1,:]
            bhc_gt_xaxis = gt_spectrum[0,:]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(bhc_eta_xaxis[:,0], bhc_eta[:,0], color='blue', marker='o', label='Ours')
            ax.scatter(bhc_gt_xaxis, bhc_gt, color='black', marker='x', label='GT')
            ax.set_xlabel('Norm.E')
            ax.set_ylabel('Resp.')
            ax.legend()
            ax.grid(True)
            ax.set_title('Poly. Resp.')
            tb_writer.add_figure("reconstruction/bhc_eta", fig, iteration)
            plt.close(fig)

            gaussians = scene.gaussians
            global_density_res = gaussians.get_global_density_res.squeeze().detach().cpu().numpy()
            material_idices = np.arange(global_density_res.shape[0])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(material_idices, global_density_res, color='red', marker='o', label='Metal')
            ax.set_xlabel('Material Index')
            ax.set_ylabel('Weights')
            ax.legend()
            ax.grid(True)
            ax.set_title('Metal weights')
            tb_writer.add_figure("reconstruction/material_weights", fig, iteration)            
            plt.close(fig)

            global_density_control = gaussians.get_global_density_control.item()
            tb_writer.add_scalar("reconstruction/global_density_control", global_density_control, iteration)


        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off+
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--src_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    if args.src_path is not None:
        args.source_path = args.src_path
    args.bhc_eta_count = getBhcEtaCount()
    args.bhc_mac_basis_count = getBhcMacBasisCount()    # cvpr
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")

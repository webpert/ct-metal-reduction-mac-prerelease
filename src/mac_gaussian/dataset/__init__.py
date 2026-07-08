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
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from mac_gaussian.gaussian import GaussianModel
from mac_gaussian.arguments import ModelParams
from mac_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from mac_gaussian.utils.camera_utils import cameraList_from_camInfos
from mac_gaussian.utils.general_utils import t2a
from mac_gaussian.dataset.projector import Projector
from mac_gaussian.utils.system_utils import mkdir_p


class Scene:
    gaussians: GaussianModel
    projector: Projector

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}
        self.view_percentile = args.view_percentile

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."
    
        try:            
            self.ref_idx = np.load(osp.join(args.source_path, "ref_idx.npy"))
            self.ref_mu_gt = np.load(osp.join(args.source_path, "ref_mu_gt.npy"))
        except Exception as e:
            print('reference atten. coeff. info is not available!')
            self.ref_idx = None
            self.ref_mu_gt = None

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.vol_mask = scene_info.vol_mask
        self.vol_fdk = scene_info.vol_fdk
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        
        mkdir_p(point_cloud_path)
        self.projector.save(osp.join(point_cloud_path, "eta_pred.npy"))
        if queryfunc is not None:
            # vol_pred = queryfunc(self.gaussians, self.projector)["vol"]
            vol_tot = queryfunc(self.gaussians, self.projector)
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            optimal_weight = self.projector.get_optimal_weight_for_b
            # vol_pred = vol_tot["vol"] + (vol_tot["vol_res"]*optimal_weight.item())
            vol_pred = vol_tot["vol"]
            np.save(
                osp.join(point_cloud_path, "vol_center.npy"),
                t2a(vol_pred),
            )
            save_total = False
            if save_total:
                vol_lac = vol_tot['vol_lac']
                np.save(
                    osp.join(point_cloud_path, "vol_lac.npy"),
                    t2a(vol_lac),
                )

            # np.savez(
            #     osp.join(point_cloud_path, "vol_pred.npz"),
            #     a = t2a(vol_tot["vol"]),
            #     b = t2a(vol_tot["vol_res"]),
            #     gamma = t2a(self.projector.get_bhc_gamma),
            #     weight_b = optimal_weight
            # )            

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

import sys
import torch
from torch import nn
import numpy as np
import os
import re
from glob import glob

sys.path.append("./")
from mac_gaussian.utils.gaussian_utils import get_expon_lr_func, inverse_softplus
from mac_gaussian.utils.general_utils import t2a
from mac_gaussian.arguments import ModelParams, OptimizationParams
from xray_gaussian_rasterization_voxelization import getBhcEtaCount
import os.path as osp
from enum import IntEnum

class ETA_MODE(IntEnum):
    ETA_NO_OPT = 0          # no optimization (fixed parameter)

def find_spektr(folder_path: str):
    pattern = os.path.join(folder_path, "spectrum_*_*_*.txt")
    matched_files = glob(pattern)

    if len(matched_files) == 1:
        filepath = matched_files[0]
        filename = os.path.basename(matched_files[0])
        match = re.search(r"spectrum_(\d+)_.*?\.txt$", filename)
        if match:
            return filepath, int(match.group(1))
    return None, None

class Projector(nn.Module):
    def __init__(
        self,
        args: ModelParams,
    ):
        super(Projector, self).__init__()

        self.optimizer = None
        self.eta_mode = args.eta_mode
        self._bhc_eta_mode0 = torch.empty(0)

        self.bhc_eta_count = self.get_bhc_eta_count
        self._bhc_gamma = torch.empty(0)
        
        bhc_eta_gt_path, eta_high = find_spektr(args.source_path)
        if bhc_eta_gt_path is not None:
            e_low = 10
            e_high = eta_high
            e_low_norm = 2*e_low/(e_low+e_high)
            e_high_norm = 2*e_high/(e_low+e_high)
            Es = np.linspace(e_low_norm, e_high_norm, e_high-e_low, True)
            spec_data = np.loadtxt(bhc_eta_gt_path, max_rows=150)            
            spectrum = spec_data[e_low-1:e_high-1, 1]
            s = spectrum / np.sum(spectrum)
            self.gt_spectrum = np.stack([Es, s], axis=0).astype(np.float32)
        else:
            self.gt_spectrum = None

        self.bhc_gamma_activation = torch.nn.Identity()
        self.iteration = 0
        

    @property
    def get_bhc_eta(self):
        bhc_eta = self._bhc_eta_mode0
        return bhc_eta

    @property
    def get_bhc_gamma(self):
        return self.bhc_gamma_activation(self._bhc_gamma)
    
    @property
    def get_bhc_eta_count(self):
        bhc_eta_count = getBhcEtaCount()
        return bhc_eta_count    # BHC_ETA_COUNT
    
    @property
    def get_gt_spectrum_view(self):
        gt_spectrum_view = self.gt_spectrum.copy()
        gt_spectrum_view[1,:] = gt_spectrum_view[1,:] * gt_spectrum_view.shape[1] / self.bhc_eta_count
        return gt_spectrum_view

    @property
    def get_optimal_weight_for_b(self):
        optimal_weight = np.array([1.0]).astype(np.float32)
        return optimal_weight
                    
    def get_effective_spectrum_index(self):
        N = self.gt_spectrum.shape[1]
        s = self.gt_spectrum[1,:]
        M = self.bhc_eta_count
        effective_spectrum_index = (s * np.arange(0, N)).sum() / (N-1) * (M-1)
        lower_index = np.floor(effective_spectrum_index).astype(int)
        higher_index = lower_index + 1
        higher_weight = effective_spectrum_index - lower_index
        lower_weight = 1 - higher_weight

        return lower_index, higher_index, lower_weight, higher_weight       

    
    def training_setup(self, training_args):
        assert self.gt_spectrum is not None, "gt_spectrum is not available"
        bhc_gamma_init = 1 - self.gt_spectrum[0,0]
        bhc_eta_Es = np.linspace(1-bhc_gamma_init, 1+bhc_gamma_init, self.bhc_eta_count)
        self._bhc_gamma = nn.Parameter(
            torch.tensor([bhc_gamma_init]).float().cuda().requires_grad_(True)
        )

        if training_args.bhc_eta_mode01_initialize_with_gt:
            bhc_eta_s = np.interp(bhc_eta_Es, self.gt_spectrum[0,:], self.gt_spectrum[1,:])
            bhc_eta_s = bhc_eta_s / bhc_eta_s.sum()
            bhc_eta_init = torch.from_numpy(bhc_eta_s).unsqueeze(-1).float().cuda()

        l = [
            {
                "params": [self._bhc_gamma],
                "lr": training_args.bhc_gamma_lr_init,
                "name": "bhc_gamma",
            },       
        ]
        self._bhc_eta_mode0 = bhc_eta_init

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.bhc_gamma_scheduler_args = get_expon_lr_func(
            lr_init=training_args.bhc_gamma_lr_init,
            lr_final=training_args.bhc_gamma_lr_final,
            max_steps=training_args.bhc_gamma_lr_max_steps,
        )      

    def update_learning_rate(self, iteration):
        self.iteration = iteration
        """Learning rate scheduling per step"""
           
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bhc_gamma":
                lr = self.bhc_gamma_scheduler_args(iteration)
                param_group["lr"] = lr         
    
    def pause_learning_rate(self, iteration):
        self.iteration = iteration
        lr = 0.0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bhc_gamma":
                param_group["lr"] = lr         


    def save(self, path):
        bhc_gamma = self.get_bhc_gamma.item()
        bhc_eta_count = self.get_bhc_eta_count
        bhc_axis = np.linspace(1 - bhc_gamma, 1 + bhc_gamma, bhc_eta_count)
        bhc_eta = t2a(self.get_bhc_eta.squeeze())
        eta_pred = np.stack([bhc_axis, bhc_eta], axis=0)
        np.save(path, eta_pred)

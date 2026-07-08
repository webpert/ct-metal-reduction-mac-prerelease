import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from mac_gaussian.gaussian.gaussian_model import GaussianModel
from mac_gaussian.arguments import ModelParams, OptimizationParams
from mac_gaussian.utils.graphics_utils import fetchPly
from mac_gaussian.utils.system_utils import searchForMaxIteration


def do_resampling_batch(x, y, xmin, xmax, sampling_count):
    # x: (M,), y: (N, M)
    N = y.shape[0]
    L = sampling_count
    xnew = np.linspace(xmin, xmax, L)
    ynew = np.empty((N, L), dtype=y.dtype)

    for i in range(N):
        ynew[i] = np.interp(xnew, x, y[i])

    return xnew, ynew



def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, opt: OptimizationParams, loaded_iter=None):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        

        data = np.load(args.mac_basis_path)
        energy_in_kev = data['energy']
        basis_mac = data['mac'] * 0.1     # convert from cm^3/g to mm^2/kg
        
        gaussians.bhc_eta_count = args.bhc_eta_count
        gaussians.bhc_basis_count = basis_mac.shape[0]
        assert gaussians.bhc_basis_count == args.bhc_mac_basis_count, \
            f"Python basis count {gaussians.bhc_basis_count} should be equal to CUDA basis count {args.bhc_mac_basis_count}"

        # resample to bhc_eta_count
        _, tmp_basis = do_resampling_batch(
                energy_in_kev, 
                basis_mac, 
                args.bhc_photon_energy_min, 
                args.bhc_photon_energy_max, 
                gaussians.bhc_eta_count
            )
        
        if True:
            eff_idx = gaussians.bhc_eta_count // 2
            # tmp_center_basis = tmp_basis[:, eff_idx]   # assumption: bhc_eta_count is odd
            # norm_basis = tmp_basis / np.tile(tmp_center_basis[:, np.newaxis], (1, gaussians.bhc_eta_count))

            center_value = tmp_basis[1,eff_idx]
            norm_basis = tmp_basis / center_value

            # norm_basis = tmp_basis

            # norm_leftmost_basis = norm_basis[:, 0]
            # sorted_idx = np.argsort(norm_leftmost_basis)
            # const_basis = np.ones((1, gaussians.bhc_eta_count), dtype=np.float32)
            # gaussians.mac_basis = np.concatenate([const_basis, norm_basis[sorted_idx, :]], axis=0)
            gaussians.mac_basis = norm_basis
        else:
            tmp_basis = tmp_basis / tmp_basis.sum(axis=1, keepdims=True)
            const_basis = np.ones((1, gaussians.bhc_eta_count), dtype=np.float32) / gaussians.bhc_eta_count
            gaussians.mac_basis = np.concatenate([const_basis, tmp_basis], axis=0)

        if False:
            # just for testing
            # gaussians.mac_basis[:, gaussians.bhc_eta_count // 2:] = 1.0   # set high energy part to 1.0
            # gaussians.mac_basis[1:, :gaussians.bhc_eta_count // 2] = gaussians.mac_basis[1:, :gaussians.bhc_eta_count // 2] * 10.0
            gaussians.mac_basis[1:, :eff_idx] = (gaussians.mac_basis[1:, :eff_idx] - 1.0) * 10.0 + 1.0
        # gaussians.mac_basis = norm_basis[sorted_idx, :]
        
        print(f"resample the basis MAC to shape: {gaussians.mac_basis.shape}")

        gaussians.create_from_pcd(xyz, density, 1.0, opt.bhc_density_init_scale)



    return loaded_iter

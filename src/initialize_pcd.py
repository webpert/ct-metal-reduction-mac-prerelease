import numpy as np
import sys
import argparse
import os.path as osp
from typing import Optional

sys.path.append("./")
from mac_gaussian.arguments import ParamGroup, ModelParams
from mac_gaussian.dataset import Scene

np.random.seed(0)


class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fdk"
        self.n_points = 50000
        self.density_thresh = 0.005
        self.density_rescale = 0.15
        self.random_density_max = 1.0  # Parameters for random mode
        super().__init__(parser, "Initialization Parameters")


def resolve_output_path(data_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path

    if osp.exists(osp.join(data_path, "meta_data.json")):
        return osp.join(data_path, "init_" + osp.basename(data_path) + ".npy")
    if data_path.split(".")[-1] in ["pickle", "pkl"]:
        return osp.join(
            osp.dirname(data_path),
            "init_" + osp.basename(data_path).split(".")[0] + ".npy",
        )
    raise ValueError(f"Could not recognize scene type: {data_path}.")


def init_pcd(scene: Scene, args: InitParams, save_path: str):
    "Initialize Gaussians."
    recon_method = args.recon_method
    n_points = args.n_points
    scanner_cfg = scene.scanner_cfg
    assert recon_method in ["random", "fdk"], "--recon_method not supported."

    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = (
            np.random.rand(
                n_points,
            )
            * args.random_density_max
        )
    else:
        print(f"Initialize from {recon_method} volume.")
        vol = scene.vol_fdk.detach().cpu().numpy()

        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def main(args, init_args: InitParams, model_args: ModelParams):
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)
    save_path = resolve_output_path(data_path, args.output)
    init_pcd(scene=scene, args=init_args, save_path=save_path)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate initialization parameters")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    parser.add_argument("--data", type=str, help="Path to data.")
    parser.add_argument("--output", default=None, type=str, help="Path to output.")
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args))

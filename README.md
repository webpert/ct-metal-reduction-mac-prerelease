# Splat-Based Metal Artifact Reduction in Cone-Beam CT via Compact Attenuation Modeling
X-ray computed tomography (CT) suffers from severe metal artifacts when high-attenuation objects such as dental fillings or orthopedic implants are present. These artifacts originate from the polychromatic nature of X-rays, where attenuation varies strongly with photon energy and material composition, breaking the monochromatic assumption used by conventional reconstruction algorithms. Recent neural rendering approaches attempt to address this mismatch through differentiable polychromatic projection models, but they still struggle with smoothness bias, loss of fine structures, and prohibitive computation when extended to largescale cone-beam CT. We introduce a splat-based metal artifact reduction framework that incorporates a physically grounded polychromatic forward model into a continuous Gaussian representation for cone-beam CT. Each Gaussian encodes the energy-dependent attenuation of the underlying material using a compact material parameterization, which enables efficient joint optimization of geometric and material properties without relying on a metal mask. This compact attenuation formulation captures the essential variation across biological tissues and metallic implants, allowing our model to explain metal-induced nonlinearity while preserving high-frequency structure. Experiments on simulated and real cone-beam CT scans show that our method converges significantly faster and suppresses metal artifacts more effectively than existing reconstruction and neural field-based approaches.

### [Project page](https://vclab.kaist.ac.kr/cvpr2026p1/index.html) | [Paper](https://vclab.kaist.ac.kr/cvpr2026p1/cbct_mar_main.pdf) | [Supplemental](https://vclab.kaist.ac.kr/cvpr2026p1/cbct_mar_supp.pdf)
[Kiseok Choi](https://sites.google.com/view/kiseokchoi), 
[Jaemin Cho](http://vclab.kaist.ac.kr/jmcho/index.html), 
[Inchul Kim](https://inchul-kim.github.io/), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html)

This repository is organized around the execution pipeline defined in `.vscode/launch.json`.

- `initialize_pcd.py`: Initializes the Gaussian point cloud
- `train.py`: Runs CBCT reconstruction with metal artifact reduction

## Tested Environment
```
CPU: Intel Xeon 4214R @ 2.4Ghz
GPU: NVIDIA A6000 48GB
RAM: 256GB
OS: Ubuntu 22.04
Docker: 24.0.2
CUDA: 13.1 (Driver: 590.48.01)
```

## Setup
Ensure that your system supports CUDA and Docker. Then clone this repository and move into the project directory.
```
git clone https://github.com/KAIST-VCLAB/ct-metal-reduction-mac.git
cd ct-metal-reduction-mac
```

Build the Docker image using Dockerfile. This process may take several minutes.
```
docker build -t r2gs_bhc_mac:cuda118 .
```
Launch the Docker container. Modify the options below according to your environment.
```
docker run -it --gpus all \
  --name r2gs_bhc_mac \
  -v /mnt/datassd/kschoi/data:/workspace/data \
  -p 20000:20000 \
  r2gs_bhc_mac:cuda118
```

## Data Preparation
Download one of the datasets from [link](https://drive.google.com/drive/folders/1l4noH0qe3abyq17l8Ex3BiDFcygj9hLs?usp=drive_link) and extract it to your preferred directory.

## Gaussian Initialization
Run the following command to initialize the Gaussian representation.<br>
($SCENE_PATH can be ``real_walnut`` or something).
```
python initialize_pcd.py --data $SCENE_PATH
```
This step can be skipped if the dataset directory already contains initialized Gaussians (e.g., ``init_$SCENE_PATH.npy``).

## Reconstruction
Modify ``source_path`` in ``./config/default.yaml`` and start reconstruction using:
```
python train.py --config $CONFIG_FILE_PATH
```
After optimization completes (20,000 iterations), the reconstructed volume will be saved to:
```
./output/$CONFIG_FILE_NAME_MM-DD-hh-mm-ss/point_cloud/iteration_20000/vol_center.npy
```

## Citation
```	
@InProceedings{Choi_2026p1_CVPR,
   author = {Choi, Kiseok and Cho, Jaemin and Kim, Inchul and Kim, Min H.},
   title = {Splat-Based Metal Artifact Reduction in Cone-Beam CT 
            via Compact Attenuation Modeling},
   booktitle = {IEEE Conference on Computer Vision and 
      Pattern Recognition (CVPR)},
   month = {June},
   year = {2026}
} 
```

## License

This project is released under the MIT License.

This code is built upon and contains modifications of the [R2-Gaussian](https://github.com/ruyi-zha/r2_gaussian) project, which is also distributed under the MIT License. We gratefully acknowledge the authors for making their code publicly available.

See the LICENSE file for details.

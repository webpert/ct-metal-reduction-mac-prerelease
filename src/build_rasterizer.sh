rm -rf "./mac_gaussian/submodules/xray-gaussian-rasterization-voxelization/build"
pip uninstall -y xray_gaussian_rasterization_voxelization
pip install -e mac_gaussian/submodules/xray-gaussian-rasterization-voxelization

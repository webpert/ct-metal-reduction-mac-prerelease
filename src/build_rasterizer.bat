rd ".\mac_gaussian\submodules\xray-gaussian-rasterization-voxelization\build\" /S /Q
pip uninstall -y xray_gaussian_rasterization_voxelization
pip install -e mac_gaussian/submodules/xray-gaussian-rasterization-voxelization

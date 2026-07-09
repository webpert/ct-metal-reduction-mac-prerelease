import io, sys, os, contextlib
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

@contextlib.contextmanager
def quiet_build():
    buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    try:
        yield
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        print("Build failed:")
        print(buf.getvalue())
        raise e
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

with quiet_build():
    setup(
        name="xray_gaussian_rasterization_voxelization",
        packages=["xray_gaussian_rasterization_voxelization"],
        ext_modules=[
            CUDAExtension(
                name="xray_gaussian_rasterization_voxelization._C",
                sources=[
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "rasterize_points.cu",
                    "cuda_voxelizer/voxelizer_impl.cu",
                    "cuda_voxelizer/forward.cu",
                    "cuda_voxelizer/backward.cu",
                    "voxelize_points.cu",
                    "ext.cpp",
                ],
                extra_compile_args={
                    "nvcc": [
                        "-w",
                        "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    ]
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

"""Build script for radfield3d-nn.

Combines the pure-Python ``radfield3dnn`` package (plus ``tasks``,
``callbacks``, ``loggers`` helpers) with a CUDA/pybind11 C++ extension built
via CMake.  The C++ target ``radfield3dnn`` in ``CMakeLists.txt`` is installed
as ``radfield3dnn.radfield3dnn`` (used by ``radfield3dnn.models.nerf_cpp``).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import Extension, find_namespace_packages, find_packages, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    """A setuptools Extension whose build is delegated to CMake.

    ``sources`` is intentionally empty — the real source list lives in
    ``CMakeLists.txt``. setuptools only needs ``name`` to compute the install
    path inside the wheel.
    """

    def __init__(self, name: str, sourcedir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Drives a CMake (Ninja) build and drops the resulting .so where
    setuptools expects to find each Extension's output."""

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        # The only native extension is the tiny-cuda-nn module
        # (radfield3dnn.radfield3dnn, used by models/nerf_cpp). It needs CUDA +
        # tiny-cuda-nn + libtorch and is OPTIONAL — tcnn is deactivated by
        # default, so a plain `pip install` is a pure-Python install (the
        # cpp-backed models in nerf_cpp are simply unavailable, and their tests
        # self-skip). Enable the native build explicitly with RFNN_WITH_TCNN=1.
        if os.environ.get("RFNN_WITH_TCNN", "").strip().lower() not in ("1", "on", "true", "yes"):
            print("radfield3d-nn: tcnn deactivated — skipping the native "
                  "radfield3dnn.radfield3dnn build (set RFNN_WITH_TCNN=1 to enable it). "
                  "Pure-Python install.")
            return

        # extdir: the directory inside the wheel where the .so for this
        # Extension must land. For ext.name == "radfield3dnn.radfield3dnn"
        # this is ``build/lib.<plat>-<py>/radfield3dnn/``.
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)

        # Use a persistent build/ dir so incremental rebuilds are fast.
        build_temp = HERE / "build"
        build_temp.mkdir(parents=True, exist_ok=True)

        # CUDA architectures are deliberately NOT forced here. Forcing
        # TCNN_CUDA_ARCHITECTURES (e.g. to the live GPU's "120" on Blackwell)
        # overrides tiny-cuda-nn's own architecture handling and routed the
        # fused MLP into a precompiled/cuBLAS path that aborts at runtime on
        # CUDA 13.x ("unspecified launch failure" in the fused backward). The
        # working build lets tcnn auto-detect and set TCNN_CUDA_ARCHITECTURES
        # itself (CMakeLists then adopts it for our targets); tcnn JIT-compiles
        # the fused kernels for the live GPU via RTC at runtime. An explicit
        # arch can still be supplied out-of-band via the CMAKE_CUDA_ARCHITECTURES
        # environment variable if ever needed.
        env_arch = os.environ.get("CMAKE_CUDA_ARCHITECTURES")

        cmake_args: List[str] = [
            # Ninja gives correct incremental builds for CUDA separable
            # compilation; Unix Makefiles has produced .so files missing
            # symbols in this project in the past, so do not let CMake pick
            # whatever it likes from the environment.
            "-G", "Ninja",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_ROOT_DIR={sys.exec_prefix}",
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
            # Build the tcnn module + its python bindings. RFNN_WITH_TCNN is
            # forced here (not left to the CMake cache) so a persistent build/
            # dir previously configured for the deploy lib — which leaves
            # RFNN_WITH_TCNN=OFF, making the `radfield3dnn` target vanish
            # ("ninja: error: unknown target 'radfield3dnn'") — still produces
            # the target. The whole native build is gated by the opt-in check
            # above, so a plain `pip install` never reaches here.
            "-DRFNN_WITH_TCNN=ON",
            "-DBuild_PyBindings=ON",
            # tiny-cuda-nn's JIT-fused MLP device-code path is __half-only;
            # turning this OFF disables the JIT and the fused model errors
            # with "Use JIT!". This is a hard constraint of the C++ path.
            "-DTCNN_HALF_PRECISION=ON",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
            "-DBUILD_RADFIELDNN_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
        ]
        if env_arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={env_arch}")

        build_args: List[str] = [
            "--config", self.build_type,
            "--target", "radfield3dnn",
            "-j", str(os.cpu_count() or 1),
        ]

        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=build_temp)

    @property
    def build_type(self) -> str:
        # Honour CMAKE_BUILD_TYPE env if the user sets it; otherwise Release.
        return os.environ.get("CMAKE_BUILD_TYPE", "Release")


# Source layout (flat — no python/ prefix):
#   radfield3dnn/...             → installed as the `radfield3dnn` package
#   tasks/, callbacks/, loggers/ → top-level helper packages

packages = find_packages(include=["radfield3dnn*"]) + sorted(set(
    find_namespace_packages(include=["tasks*", "callbacks*", "loggers*"])
))

LICENSE_TEXT = (HERE / "LICENSE").read_text() if (HERE / "LICENSE").exists() else ""

setup(
    name="radfield3d-nn",
    version=os.environ.get("CI_COMMIT_TAG") or os.environ.get("CI_COMMIT_REF_NAME") or "1.0.0",
    author="Felix Lehner",
    author_email="felix.lehner@ptb.de",
    license=LICENSE_TEXT,
    description="Neural networks for spatially-resolved x-ray flux/spectra prediction.",
    python_requires=">=3.12",
    install_requires=[
        "rich>=14.1.0",
        "torch>=2.9.1",
        "torchvision",
        "numpy>=2.0.0",
        "lightning>=2.5.6",
        "pandas>=2.3.3",
        "pyyaml>=6.0",
    ],
    packages=packages,
    package_data={"radfield3dnn": ["*.pyi", "py.typed"]},
    # The tiny-cuda-nn native extension is OPTIONAL and OFF by default (tcnn deactivated → a
    # pure-Python install). It is declared ONLY when RFNN_WITH_TCNN is enabled, so setuptools
    # never expects a .so that the build skipped. Build it with `RFNN_WITH_TCNN=1 pip install -e .`.
    ext_modules=(
        [CMakeExtension("radfield3dnn.radfield3dnn", sourcedir=str(HERE))]
        if os.environ.get("RFNN_WITH_TCNN", "").strip().lower() in ("1", "on", "true", "yes")
        else []
    ),
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

"""Build script for radfield3d-nn.

Combines the pure-Python ``radfield3dnn`` package (plus ``tasks``, ``callbacks``, ``loggers``
helpers) with CMake-built C++ extensions:

* ``radfield3dnn.deploy.rfnn_deploy`` — the ONNX deployment runtime (RF3M loader + field
  predictors). CPU-only, no CUDA, built BY DEFAULT so ``pip install`` yields a working
  load/store/infer path. Opt out with ``RFNN_WITH_DEPLOY=0``.
* ``radfield3dnn.radfield3dnn`` — the tiny-cuda-nn fused models (used by ``models/nerf_cpp``).
  Needs CUDA + tiny-cuda-nn + libtorch; OFF by default, enable with ``RFNN_WITH_TCNN=1``.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import Extension, find_namespace_packages, find_packages, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).parent.resolve()


def _truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "on", "true", "yes")


def _ensure_generator(build_dir: Path, generator: str) -> None:
    """CMake refuses to switch generators on an existing build dir. If ``build_dir`` was configured
    with a different generator (e.g. an older Makefiles build), wipe it so the requested generator
    can be used on a clean cache."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith("CMAKE_GENERATOR:"):
            if line.split("=", 1)[1].strip() != generator:
                shutil.rmtree(build_dir, ignore_errors=True)
            return


def _conda_toolchain():
    """The conda compiler pair, if this is a conda env that ships one. The deploy binding must link
    the SAME libstdc++ the interpreter loads at runtime; in a conda env that is the conda toolchain,
    and building with the system g++ instead yields a binding whose std::string ABI mismatches the
    interpreter (garbage metadata / crashes on load). Returns (cc, cxx) or (None, None)."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None, None
    cc = Path(prefix) / "bin" / "x86_64-conda-linux-gnu-gcc"
    cxx = Path(prefix) / "bin" / "x86_64-conda-linux-gnu-g++"
    return (str(cc), str(cxx)) if cc.exists() and cxx.exists() else (None, None)


class CMakeExtension(Extension):
    """A setuptools Extension whose build is delegated to CMake.

    ``sources`` is intentionally empty — the real source list lives in ``CMakeLists.txt``.
    ``kind`` selects which CMake target this Extension drives (``deploy`` or ``tcnn``).
    """

    def __init__(self, name: str, kind: str, sourcedir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.kind = kind
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Drives a CMake (Ninja) build and drops the resulting .so where setuptools expects it."""

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)
        if ext.kind == "deploy":
            return self._build_deploy(ext)
        return self._build_tcnn(ext)

    # ── deployment runtime (default; CPU-only, no CUDA) ──────────────────────────────────────────
    def _build_deploy(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)

        # Dedicated build dir: the deploy config (RFNN_WITH_TCNN=OFF) is incompatible with the
        # tcnn build's cache, so the two never share one.
        build_temp = HERE / "build_deploy"
        _ensure_generator(build_temp, "Ninja")
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args: List[str] = [
            "-G", "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_ROOT_DIR={sys.exec_prefix}",
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
            "-DBuild_DeployPyBindings=ON",
            "-DRFNN_WITH_TCNN=OFF",
            "-DRFNN_BACKEND_VULKAN=OFF",
            "-DBUILD_RADFIELDNN_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
        ]

        cc, cxx = _conda_toolchain()
        if cxx:
            cmake_args += [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]

        # Reuse an already-fetched ONNX Runtime if present (it is a large prebuilt download).
        ort_src = HERE / "build" / "_deps" / "fetch_onnxruntime-src"
        if ort_src.exists():
            cmake_args.append(f"-DFETCHCONTENT_SOURCE_DIR_FETCH_ONNXRUNTIME={ort_src}")

        os.environ.setdefault("PYTHON_EXECUTABLE", sys.executable)
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "rfnn_deploy",
                               "-j", str(os.cpu_count() or 1)], cwd=build_temp)

        # CMake outputs rfnn_deploy into <repo>/lib (where the Python loader resolves it for an
        # editable install). Mirror it into the wheel's package dir so non-editable installs ship
        # it too.
        for so in (HERE / "lib").glob("rfnn_deploy*.so"):
            shutil.copy2(so, extdir / so.name)

    # ── tiny-cuda-nn fused models (opt-in: RFNN_WITH_TCNN=1) ──────────────────────────────────────
    def _build_tcnn(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)

        # Persistent build/ dir so incremental rebuilds are fast.
        build_temp = HERE / "build"
        build_temp.mkdir(parents=True, exist_ok=True)

        # CUDA architectures are deliberately NOT forced. Forcing TCNN_CUDA_ARCHITECTURES overrides
        # tiny-cuda-nn's own handling and routes the fused MLP into a precompiled/cuBLAS path that
        # aborts at runtime on CUDA 13.x; letting tcnn auto-detect lets it JIT-compile the fused
        # kernels for the live GPU via RTC. An explicit arch can be supplied via
        # CMAKE_CUDA_ARCHITECTURES if ever needed.
        env_arch = os.environ.get("CMAKE_CUDA_ARCHITECTURES")

        cmake_args: List[str] = [
            "-G", "Ninja",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_ROOT_DIR={sys.exec_prefix}",
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
            "-DRFNN_WITH_TCNN=ON",
            "-DBuild_PyBindings=ON",
            # tiny-cuda-nn's JIT-fused MLP device-code path is __half-only; OFF disables the JIT and
            # the fused model errors with "Use JIT!". A hard constraint of the C++ path.
            "-DTCNN_HALF_PRECISION=ON",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
            "-DBUILD_RADFIELDNN_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
        ]
        if env_arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={env_arch}")

        os.environ.setdefault("PYTHON_EXECUTABLE", sys.executable)
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", self.build_type,
                               "--target", "radfield3dnn", "-j", str(os.cpu_count() or 1)],
                              cwd=build_temp)

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

# The deployment runtime builds by default (CPU-only); opt out with RFNN_WITH_DEPLOY=0. The
# tiny-cuda-nn extension is opt-in (needs CUDA); enable with RFNN_WITH_TCNN=1.
ext_modules: List[Extension] = []
if _truthy("RFNN_WITH_DEPLOY", "1"):
    ext_modules.append(CMakeExtension("radfield3dnn.deploy.rfnn_deploy", kind="deploy",
                                      sourcedir=str(HERE)))
if _truthy("RFNN_WITH_TCNN"):
    ext_modules.append(CMakeExtension("radfield3dnn.radfield3dnn", kind="tcnn", sourcedir=str(HERE)))

setup(
    name="radfield3d-nn",
    version=os.environ.get("CI_COMMIT_TAG") or os.environ.get("CI_COMMIT_REF_NAME") or "1.0.0",
    author="Felix Lehner",
    author_email="felix.lehner@ptb.de",
    license=LICENSE_TEXT,
    description="Neural networks for spatially-resolved x-ray flux/spectra prediction.",
    python_requires=">=3.12",
    install_requires=[
        "rich>=15.0.0",
        "RadFiled3D>=1.3.4",
        "torch>=2.10.0",
        "numpy>=2.4.6",
        "pyyaml>=6.0.3",
        "lightning>=2.6.4",
        "pandas>=3.0.3",
        "plotly>=6.7.0",
        "joblib>=1.5.3",
        "optuna>=4.8.0",
        "optuna-integration[lightning]>=4.8.0",
        "onnx>=1.21.0",
        "onnxscript>=0.7.0",
        "onnxruntime>=1.26.0",
    ],
    packages=packages,
    package_data={
        "radfield3dnn": ["*.pyi", "py.typed"],
        "radfield3dnn.deploy": ["*.pyi"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

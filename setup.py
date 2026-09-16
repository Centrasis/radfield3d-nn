"""Build script for radfield3d-nn: the pure-Python ``radfield3dnn`` package plus the ``tasks``,
``callbacks`` and ``loggers`` helpers. Nothing here is compiled.

Deployment (the ``.rf3m`` container, the inference runtime, the graphics interop) is the separate
``radfiled3d-nn`` distribution, installed as an ordinary dependency.
"""

import os
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


HERE = Path(__file__).parent.resolve()


def _version() -> str:
    """Package version from the CI toolchain.

    On a GitHub Actions tag build the runner sets ``GITHUB_REF_TYPE=tag`` and ``GITHUB_REF_NAME``
    to the tag itself. Release tags are plain ``x.y.z`` (see ``.github/workflows/release.yml``), so
    the tag *is* the version — no prefix to strip. Branch builds and local installs fall back to a
    dev version (never published).
    """
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ.get("GITHUB_REF_NAME", "").strip()
        if tag:
            return tag
    return os.environ.get("RFNN_VERSION", "0.0.0.dev0")


# `visualizers` carries no __init__.py, so it is found as a NAMESPACE package like the other three.
# It is not optional: `tasks.train` imports `callbacks.plotter`, which imports three modules from it.
packages = find_packages(include=["radfield3dnn*"]) + sorted(set(
    find_namespace_packages(include=["tasks*", "callbacks*", "loggers*", "visualizers*"])
))

LICENSE_TEXT = (HERE / "LICENSE").read_text() if (HERE / "LICENSE").exists() else ""

setup(
    name="radfield3d-nn",
    version=_version(),
    author="Felix Lehner",
    author_email="felix.lehner@ptb.de",
    license=LICENSE_TEXT,
    description="Neural networks for spatially-resolved x-ray flux/spectra prediction.",
    python_requires=">=3.12",
    install_requires=[
        "rich>=15.0.0",
        "RadFiled3D>=1.3.6",
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
        "onnxconverter-common>=1.14.0",   # fp16 ONNX conversion for export_fp16 packaging
        # The .rf3m container + inference runtime, imported as `RadFiled3D.nn` — 1.4 is where that
        # layout (nn.deploy / nn.inference / nn.memory) replaced the flat `radfiled3d_nn` module.
        # A LOCAL branch build of that repo reports 0.0.0 by design and does not satisfy this, so a
        # `pip install -e .` here would replace it with a released wheel. Install this repo with
        # --no-deps while developing the two together.
        "radfiled3d-nn>=1.4.2",
    ],
    packages=packages,
    package_data={
        "radfield3dnn": ["*.pyi", "py.typed"],
    },
    zip_safe=False,
)

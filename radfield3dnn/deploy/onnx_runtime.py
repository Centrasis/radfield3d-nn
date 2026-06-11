"""Python entry-point to the C++ ONNX deployment runtime (rfnn_deploy bindings).

This is the Python side of the *deployment* path: it loads the compiled ``rfnn_deploy`` module (the
pybind bindings over ``rfnn::io::V1::ModelFactory`` + the ONNX field predictors) and runs an RF3M
package's exported ONNX graphs through ONNX Runtime — no torch / CUDA needed. It is the counterpart
to :class:`~radfield3dnn.deploy.model_packager.ModelPackager`, which writes the RF3M package this
loads.

    from radfield3dnn.deploy import load_rf3m
    model = load_rf3m("PBRFNet.rf3m")            # LoadedModel
    pred  = model.build(use_cuda=False)          # Volume/VoxelFieldPredictor
    out   = pred.predict_volume(beam, (48,48,48))

Build the bindings once with::

    cd build_novk && PYTHON_EXECUTABLE=$(which python) cmake .. -DBuild_DeployPyBindings=ON \
        -DPython_EXECUTABLE=$(which python) && cmake --build . --target rfnn_deploy -j

The compiled ``rfnn_deploy*.so`` lands in ``<repo>/lib``; ONNX Runtime's shared lib is fetched by
CMake under the build tree. This module finds both and makes the import work regardless of cwd.
"""
from __future__ import annotations

import ctypes
import glob
import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis bridge: the runtime module is loaded dynamically (importlib from the built
    # .so), which type checkers cannot follow. These imports resolve against the persistent stub
    # `typings/rfnn_deploy.pyi` (Pylance's default stubPath), so every re-exported name below — and
    # the return type of load_rf3m — carries the full mirrored C++ API in the IDE.
    from rfnn_deploy import (  # noqa: F401
        BeamParameters,
        BeamParameterSpec,
        EncodedBeam,
        ExecutionOptions,
        ModelDomain,
        ModelFactory,
        ModelProvenance,
        ParameterRange,
        PredictorType,
        VolumeFieldPredictor,
        VoxelFieldPredictor,
    )

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB = os.path.join(_REPO, "lib")


def _linked_ort_version(so_path: str) -> str | None:
    """The ONNX Runtime version the rfnn_deploy .so was linked against (e.g. '1.20.1'), read from
    the fetch-dir string baked into the binary — so we load the SAME ABI it needs."""
    import re
    try:
        with open(so_path, "rb") as f:
            blob = f.read()
        m = re.search(rb"onnxruntime-linux-x64-gpu[_a-z0-9]*-(\d+\.\d+\.\d+)", blob)
        return m.group(1).decode() if m else None
    except Exception:
        return None


def _find_onnxruntime(want_version: str | None) -> str | None:
    """Locate libonnxruntime.so, preferring the exact version the binding was linked against
    (mismatched ABIs raise `version VERS_x not found` at import)."""
    candidates = []
    for base in (os.path.join(_REPO, "build_novk"), os.path.join(_REPO, "build_cmake"),
                 os.path.join(_REPO, "build"), "/tmp"):
        candidates += glob.glob(os.path.join(base, "**", "libonnxruntime.so*"), recursive=True)
    if not candidates:
        return None
    if want_version:
        exact = [p for p in candidates if f"-{want_version}/" in p or p.endswith(f".so.{want_version}")]
        if exact:
            exact.sort(key=lambda p: (p.count("."), len(p)), reverse=True)
            return exact[0]
    candidates.sort(key=lambda p: (p.count("."), len(p)), reverse=True)
    return candidates[0]


def _ensure_loaded():
    """Import the compiled rfnn_deploy module, pre-loading ONNX Runtime so its SONAME resolves."""
    if "rfnn_deploy" in sys.modules:
        return sys.modules["rfnn_deploy"]
    so = glob.glob(os.path.join(_LIB, "rfnn_deploy*.so"))
    if not so:
        raise ImportError(
            "rfnn_deploy native module not found in ./lib. Build it:\n"
            "  cd build_novk && PYTHON_EXECUTABLE=$(which python) cmake .. -DBuild_DeployPyBindings=ON "
            "-DPython_EXECUTABLE=$(which python) && cmake --build . --target rfnn_deploy -j")
    ort = _find_onnxruntime(_linked_ort_version(so[0]))
    if ort:
        # RTLD_GLOBAL so the predictor lib's undefined ORT symbols bind to this handle.
        ctypes.CDLL(ort, mode=ctypes.RTLD_GLOBAL)
    spec = importlib.util.spec_from_file_location("rfnn_deploy", so[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["rfnn_deploy"] = mod
    return mod


# Eager-load so the names below are real classes (and IDEs resolve them via lib/rfnn_deploy.pyi).
rfnn_deploy = _ensure_loaded()

# re-export the binding surface (mirrors the C++ structure: rfnn::io::V1::ModelFactory + the
# radfield3dnn:: predictor hierarchy; package metadata lives ON the loaded predictor). The names
# are typed by the TYPE_CHECKING block above; these runtime assignments are invisible to linters.
if not TYPE_CHECKING:
    BeamParameters = rfnn_deploy.BeamParameters
    ExecutionOptions = rfnn_deploy.ExecutionOptions
    EncodedBeam = rfnn_deploy.EncodedBeam
    PredictorType = rfnn_deploy.PredictorType
    VolumeFieldPredictor = rfnn_deploy.VolumeFieldPredictor
    VoxelFieldPredictor = rfnn_deploy.VoxelFieldPredictor
    ModelFactory = rfnn_deploy.ModelFactory
    ModelDomain = rfnn_deploy.ModelDomain
    ModelProvenance = rfnn_deploy.ModelProvenance
    BeamParameterSpec = rfnn_deploy.BeamParameterSpec
    ParameterRange = rfnn_deploy.ParameterRange


def load_rf3m(path: str, use_cuda: bool = False) -> "VoxelFieldPredictor | VolumeFieldPredictor":
    """Load an RF3M package STRAIGHT to the runnable predictor (:class:`VoxelFieldPredictor` for
    per-voxel models, :class:`VolumeFieldPredictor` for field-wise ones). The package metadata is
    attached to the predictor: ``.domain``, ``.provenance``, ``.metrics``, ``.graph_names``."""
    return rfnn_deploy.ModelFactory.load(path, use_cuda=use_cuda)


__all__ = [
    "load_rf3m", "ModelFactory", "BeamParameters", "ExecutionOptions", "EncodedBeam",
    "PredictorType", "VolumeFieldPredictor", "VoxelFieldPredictor", "ModelDomain",
    "ModelProvenance", "BeamParameterSpec", "ParameterRange", "rfnn_deploy",
]

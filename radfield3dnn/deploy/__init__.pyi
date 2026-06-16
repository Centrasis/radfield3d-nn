"""Typed public surface of :mod:`radfield3dnn.deploy` — the ONNX deployment / inference API.

This inline stub (shipped with the package; see ``py.typed``) is the single source of truth for
the types every consumer needs to run inference from an RF3M package, and it is what a linter
reads for ``from radfield3dnn.deploy import BeamParameters, load_rf3m, ...``. The names are produced
lazily by the runtime ``__getattr__`` (re-exporting the compiled ``rfnn_deploy`` bindings) so the
package imports without the native module built; this stub mirrors the bindings 1:1 so member
access on the inference types is fully type-hinted regardless of the editor's stub-path config.

C++ → Python mapping:
    radfield3dnn::VolumeFieldPredictor   → VolumeFieldPredictor
    radfield3dnn::VoxelFieldPredictor    → VoxelFieldPredictor  (inherits VolumeFieldPredictor)
    radfield3dnn::BeamParameters         → BeamParameters
    radfield3dnn::EncodedBeam            → EncodedBeam
    radfield3dnn::ExecutionOptions       → ExecutionOptions
    rfnn::io::V1::ModelStore             → ModelStore  (loads an RF3M STRAIGHT to a predictor)
    rfnn::io::ModelDomain / ModelProvenance / BeamParameterSpec / ParameterRange → metadata classes

Quick start::

    from radfield3dnn.deploy import load_rf3m, BeamParameters
    pred = load_rf3m("PBRFNet.rf3m")                 # VoxelFieldPredictor | VolumeFieldPredictor
    pred.domain.beam_parameters                      # input layout + valid ranges
    pred.metrics                                     # stored test metrics
    beam = BeamParameters(direction=[0, 0, -1], origin=[0.5, 0.5, 0.5], spectrum=raw_tube_histogram)
    vol  = pred.predict_volume(beam, (48, 48, 48))   # {"flux": np[D,H,W], "spectrum": np[...,bins]}
"""
from __future__ import annotations

import enum
from typing import Mapping, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

from radfield3dnn.deploy.model_packager import ModelPackager as ModelPackager

# ── inference result ──────────────────────────────────────────────────────────────
class FieldPrediction(TypedDict):
    """Returned by every predict_* call. ``flux``: ``[D,H,W]`` (volume mode) or ``[N]`` (voxel
    mode); ``spectrum``: ``[D,H,W,n_bins]`` / ``[N,n_bins]``; both float32."""
    flux: npt.NDArray[np.float32]
    spectrum: npt.NDArray[np.float32]
    dims: tuple[int, int, int]
    n_bins: int
    inference_ms: float

# ── beam conditioning (radfield3dnn::BeamParameters / EncodedBeam) ─────────────────
class BeamParameters:
    """Beam geometry + spectrum conditioning a prediction. ``origin`` is field-relative ([0,1]³,
    centre 0.5); ``direction`` a unit vector; ``spectrum`` the RAW tube histogram (the beam
    encoder's input width, e.g. 150 — NOT the output bins); ``rect`` collimation in metres.
    Metric inputs (the source distance derived from ``origin``) are clipped+normalised to the
    model's stored parameter ranges before encoding — pass PHYSICAL values."""
    direction: tuple[float, float, float]
    origin: tuple[float, float, float]
    spectrum: list[float]
    rect: tuple[float, float]
    def __init__(
        self,
        direction: Sequence[float],
        origin: Sequence[float] = ...,
        spectrum: Sequence[float] = ...,
        rect: Sequence[float] = ...,
    ) -> None: ...

class EncodedBeam:
    """The beam encoded ONCE into the trunk-conditioning latent (from
    :meth:`VoxelFieldPredictor.encode_beam`); reuse it across many per-voxel queries."""
    @property
    def is_encoded(self) -> bool: ...
    @property
    def latent(self) -> list[float]: ...

class ExecutionOptions:
    """ONNX Runtime execution-provider request (TensorRT → CUDA → CPU fallback)."""
    use_gpu: bool
    use_tensorrt: bool
    fp16: bool
    device_id: int
    engine_cache_dir: str
    def __init__(self) -> None: ...

class PredictorType(enum.Enum):
    VolumeField = ...
    VoxelField = ...

# ── package metadata (rfnn::io) — attached to the loaded predictor ─────────────────
class ParameterRange:
    """Valid [min, max] + physical unit of one beam-parameter segment."""
    min: float
    max: float
    unit: str
    def __init__(self, min: float = ..., max: float = ..., unit: str = ...) -> None: ...

class BeamParameterSpec:
    """One ordered entry of the model's beam-parameter input vector (name, slot count, range)."""
    name: str
    count: int
    range: ParameterRange
    def __init__(self, name: str, count: int, range: ParameterRange = ...) -> None: ...

class ModelDomain:
    """The model's fixed I/O domain in metric units: spectrum bins + max energy and the ordered
    beam-parameter input layout (what each input slot means and its valid range)."""
    spectrum_bins: int
    spectrum_max_energy_ev: float
    beam_parameters: list[BeamParameterSpec]
    def __init__(self, spectrum_bins: int = ..., spectrum_max_energy_ev: float = ...,
                 beam_parameters: list[BeamParameterSpec] = ...) -> None: ...

class ModelProvenance:
    """Lightweight training provenance (dataset name, simulation software/physics)."""
    dataset_name: str
    software_version: str
    physics: str
    def __init__(self, dataset_name: str = ..., software_version: str = ..., physics: str = ...) -> None: ...

# ── predictor hierarchy (radfield3dnn::) ────────────────────────────────────────────
class VolumeFieldPredictor:
    """Runs one exported ONNX trunk graph through ONNX Runtime. Field-wise models emit the whole
    D×H×W volume in a single Run(). Base of the predictor hierarchy.

    When constructed by :meth:`ModelStore.load`, the RF3M package metadata is attached:
    ``domain`` / ``provenance`` / ``metrics`` / ``graph_names``.
    """
    # package metadata (present when loaded via ModelStore)
    domain: ModelDomain
    provenance: ModelProvenance
    metrics: Mapping[str, float]
    graph_names: list[str]

    def __init__(self, onnx_path: str, use_cuda: bool = ...) -> None: ...
    @property
    def type(self) -> PredictorType: ...
    @property
    def is_voxelwise(self) -> bool: ...
    @property
    def spectrum_bins(self) -> int: ...
    def predict_volume(
        self,
        beam: BeamParameters,
        dims: tuple[int, int, int] | Sequence[int],
        max_inner_batch: int = ...,
    ) -> FieldPrediction:
        """Whole-field prediction on the ``i/(dims-1)`` voxel grid (matches training)."""
        ...

class VoxelFieldPredictor(VolumeFieldPredictor):
    """Per-voxel implicit model (MLP/NeRF). IS-A :class:`VolumeFieldPredictor` — inherits
    ``predict_volume`` (assembles the volume by tiling per-voxel queries) and all metadata — and
    adds arbitrary point queries against a cached encoded beam."""
    def encode_beam(self, beam: BeamParameters) -> EncodedBeam:
        """Encode the beam ONCE into the trunk-conditioning latent; cache + reuse."""
        ...
    def predict_voxelwise(
        self,
        positions: npt.NDArray[np.float32],
        encoded_beam: EncodedBeam,
    ) -> FieldPrediction:
        """Query arbitrary points. ``positions``: (N,3) float32 in [0,1]³ (zero-copy bind)."""
        ...

# ── factory (rfnn::io::V1::ModelStore) ────────────────────────────────────────────
class ModelStore:
    """Loads an RF3M model package STRAIGHT to the runnable predictor; the byte layout's single
    source of truth on the save side (used by the Python ModelPackager)."""
    @staticmethod
    def load(path: str, use_cuda: bool = ...) -> VoxelFieldPredictor | VolumeFieldPredictor:
        """Per-voxel trunk → :class:`VoxelFieldPredictor` (wired with the beam-encoder graph);
        field-wise trunk → :class:`VolumeFieldPredictor`. Metric beam-parameter inputs are
        normalised using the package's stored ranges (matches training)."""
        ...
    @staticmethod
    def load_from_memory(data: bytes, use_cuda: bool = ...) -> VoxelFieldPredictor | VolumeFieldPredictor: ...


def load_rf3m(path: str, use_cuda: bool = ...) -> VoxelFieldPredictor | VolumeFieldPredictor:
    """Load an RF3M package straight to the runnable predictor (per-voxel → VoxelFieldPredictor,
    field-wise → VolumeFieldPredictor), with the package metadata attached to the returned object."""
    ...


__all__ = [
    "ModelPackager",
    "load_rf3m",
    "FieldPrediction",
    "BeamParameters",
    "BeamParameterSpec",
    "EncodedBeam",
    "ExecutionOptions",
    "ModelDomain",
    "ModelProvenance",
    "ModelStore",
    "ParameterRange",
    "PredictorType",
    "VolumeFieldPredictor",
    "VoxelFieldPredictor",
]

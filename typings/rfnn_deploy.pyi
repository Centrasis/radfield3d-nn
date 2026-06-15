"""Type stubs for the rfnn_deploy native module — a 1:1 mirror of the C++ deployment runtime
(relevant methods and fields only).

C++ → Python mapping:
    radfield3dnn::VolumeFieldPredictor      → VolumeFieldPredictor
    radfield3dnn::VoxelFieldPredictor       → VoxelFieldPredictor  (inherits VolumeFieldPredictor)
    radfield3dnn::BeamParameters            → BeamParameters
    radfield3dnn::EncodedBeam               → EncodedBeam
    radfield3dnn::ExecutionOptions          → ExecutionOptions
    rfnn::io::V1::ModelStore              → ModelStore  (loads an RF3M STRAIGHT to a predictor)
    rfnn::io::ModelDomain / ModelProvenance / BeamParameter / ParameterRange → metadata classes

There is deliberately NO LoadedModel: ``ModelStore.load(path)`` parses the RF3M container AND
builds the runnable predictor in one step, returning the :class:`VoxelFieldPredictor` (per-voxel
models) or :class:`VolumeFieldPredictor` (field-wise models) directly, with the package metadata
exposed as read-only properties (``.domain``, ``.provenance``, ``.metrics``, ``.graph_names``).

Quick start::

    import rfnn_deploy as rd
    pred = rd.ModelStore.load("PBRFNet.rf3m")     # VoxelFieldPredictor | VolumeFieldPredictor
    pred.domain.beam_parameters                      # explore the model's input layout + ranges
    pred.metrics                                     # the stored test metrics
    beam = rd.BeamParameters(direction=[0,0,-1], origin=[0.5,0.5,0.5], spectrum=raw_tube_histogram)
    vol  = pred.predict_volume(beam, (48,48,48))     # {"flux": np[D,H,W], "spectrum": np[...,bins]}
    enc  = pred.encode_beam(beam)                    # per-voxel models: encode the beam ONCE
    out  = pred.predict_voxelwise(positions, enc)    # (N,3) float32 positions in [0,1]^3
"""
from __future__ import annotations

from typing import Sequence, Mapping, TypedDict
import enum
import numpy as np
import numpy.typing as npt

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

# ── module-level save (the C++ serialiser; used by ModelPackager) ───────────────────
def save_to_memory(
    graphs: Mapping[str, bytes],
    domain: ModelDomain,
    provenance: ModelProvenance,
    metrics: Mapping[str, float],
) -> bytes:
    """Serialise an RF3M package to bytes. ``graphs`` maps a graph name ('trunk', 'beam_encoder',
    ...) to its ONNX protobuf bytes."""
    ...

def save(
    path: str,
    graphs: Mapping[str, bytes],
    domain: ModelDomain,
    provenance: ModelProvenance,
    metrics: Mapping[str, float],
) -> None: ...

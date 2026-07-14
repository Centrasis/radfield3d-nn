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
from typing import Mapping, Sequence, TypedDict, overload

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
    user_compute_stream: int
    """Caller's CUDA stream as an integer handle (torch: current_stream().cuda_stream); 0 = ORT-owned."""
    def __init__(self) -> None: ...

class PredictorType(enum.Enum):
    VolumeField = ...
    VoxelField = ...

# ── zero-copy binding (radfield3dnn::model_binding.h) ─────────────────────────────
class DeviceKind(enum.Enum):
    Cpu = ...
    Cuda = ...
    Vulkan = ...

class ElementType(enum.Enum):
    F32 = ...
    F16 = ...

class Unit(enum.Enum):
    """The unit a caller's buffer is in. Bound buffers reach the graph VERBATIM, so they must hold
    normalised values; ParameterNormalizer converts a metric buffer into that space."""
    Normalized = ...
    Metres = ...
    Millimetres = ...
    Degrees = ...
    Radians = ...

class SessionMemory:
    """What a session consumes with NO copy — the yardstick every binding is checked against."""
    device: DeviceKind
    dtype: ElementType
    device_id: int
    provider: str

class BindingError(Exception):
    """A bound buffer does not match what the session can consume, and no conversion was asked for.
    The message names what mismatched, what a conversion would cost, and the three ways out."""

class IncompleteSetupError(Exception):
    """predict_volume() was called before the model was set up. Lists EVERY gap at once."""

class ParameterNormalizer:
    """Produced by a model (``pred.parameter_normalizer()``): only the model knows the ranges its
    inputs were normalised against."""
    def write(self, flag: ModelInput, values: Sequence[float] | npt.NDArray[np.float32],
              unit: Unit = ...) -> None:
        """Convert metric values into the network's normalised space (using this model's own domain)
        and write them into the buffer bound for ``flag``."""

# ── deployment interface (radfield3dnn::model_interface.h) ────────────────────────
# WHAT a stored model consumes / produces. Bit flags: combine with ``|``, test with ``&``.
# Defined in C++ (include/radfield3d-nn/model_interface.h) and bound here — never redeclared in
# Python. Bit positions are append-only ABI.

class ModelInput(enum.IntFlag):
    NONE: int
    POSITION: int                   # xyz. SET => per-voxel model; CLEAR => whole-volume model
    QUERY_DIRECTION: int
    TUBE_SPECTRUM: int              # bin count comes from ModelDomain.spectrum_bins
    BEAM_DIRECTION: int
    SOURCE_DISTANCE: int
    SOURCE_ORIGIN_3D: int           # alternative to SOURCE_DISTANCE; never both
    BEAM_COLLIMATION: int           # the KIND comes from ModelDomain.collimation
    PATIENT_TRANSLATION_3D: int
    PATIENT_ROTATION_3D: int
    GEOMETRY_MAP: int
    ANODE_ANGLE: int

class ModelOutput(enum.IntFlag):
    NONE: int
    FLUX: int
    SPECTRUM: int                   # needs ModelDomain.spectrum_bins
    ANGULAR_FLUX: int               # needs ModelDomain.angular_{phi,theta}_segments
    ERROR: int
    AIR_KERMA: int
    SEPARATE_CHANNELS: int          # scatter and direct beam emitted separately, not joined

class ModelInterface:
    """What a stored model consumes and produces, plus how it must be driven.

    Written when a model is STORED and read when it is LOADED, to instantiate the executable
    predictor that implements it. Describes data, never a network: two architectures with the same
    I/O are interchangeable to a consumer. Every RESOLUTION (spectrum bins, angular segments,
    collimation kind) lives in ModelDomain, not here — validate() cross-checks the two.
    """
    inputs: ModelInput
    outputs: ModelOutput
    resolution_aware: bool          # the location encoder must be configured per queried grid
    region_state_dims: int          # width of the trunk's region_state input (0 = absent)
    region_width_frame: float       # region_width = frame / N for an N^3 query grid
    @property
    def id(self) -> int:
        """Packed (inputs << 32) | outputs. Exact equality selects the executable predictor."""
    @property
    def is_voxelwise(self) -> bool:
        """True if the model is queried at explicit positions (the POSITION bit)."""
    def validate(self, domain: ModelDomain) -> None:
        """Reject reserved bits; check the domain carries every resolution the flags need.

        Raises UnsupportedInterfaceError. Called on store and again on load.
        """
    def takes(self, flag: ModelInput) -> bool: ...
    def gives(self, flag: ModelOutput) -> bool: ...
    @staticmethod
    def make_id(inputs: ModelInput, outputs: ModelOutput) -> int: ...

# ── package metadata (rfnn::io) — attached to the loaded predictor ─────────────────
class ParameterRangeType(enum.IntEnum):
    MinMax: int                     # a [min, max] interval + unit
    Spectrum: int                   # a histogram range + bin_width (bins = (max-min)/bin_width)
    Map: int                        # a nested name -> range map

class CollimationType(enum.IntEnum):
    None_: int
    Rectangle: int                  # 2 parameters
    Cone: int                       # 1
    Ellipsoid: int                  # 2

class ParameterRange:
    """The valid range of one beam parameter — a tagged variant; which fields carry meaning
    depends on `type`. Build one with the factories, not the constructor."""
    type: ParameterRangeType
    min: float
    max: float
    bin_width: float                # Spectrum only
    unit: str
    children: list[tuple[str, ParameterRange]]   # Map only
    @staticmethod
    def min_max(min: float, max: float, unit: str) -> ParameterRange: ...
    @staticmethod
    def spectrum(min: float, max: float, bin_width: float, unit: str) -> ParameterRange: ...
    @staticmethod
    def nested(children: list[tuple[str, ParameterRange]]) -> ParameterRange: ...

class BeamParameterSpec:
    """One ordered entry of the model's beam-parameter input vector. The slot count is implied by
    the range (a Spectrum range's bin count, a MinMax range's arity), not stored separately."""
    name: str
    range: ParameterRange
    def __init__(self, name: str, range: ParameterRange) -> None: ...

class ModelDomain:
    """The model's fixed I/O domain in metric units. Every RESOLUTION a ModelInterface flag depends
    on lives here — ModelInterface.validate(domain) checks the two agree, on store and on load."""
    spectrum_bins: int                      # OUTPUT spectrum histogram bins
    spectrum_max_energy_ev: float
    field_dimensions_m: list[float]         # metric box the normalised [0,1]^3 positions map into
    angular_phi_segments: int               # AngularFlux resolution (0 = no angular output)
    angular_theta_segments: int
    collimation: CollimationType            # kind of the BeamCollimation input
    beam_parameters: list[BeamParameterSpec]
    @property
    def collimation_dims(self) -> int:
        """Width of the BeamCollimation input tensor, implied by `collimation`."""
    def __init__(self, spectrum_bins: int = ..., spectrum_max_energy_ev: float = ...,
                 field_dimensions_m: list[float] = ...,
                 beam_parameters: list[BeamParameterSpec] = ...) -> None: ...

class PackageMetadata:
    """The RF3M header alone — what read_metadata() returns (no ONNX Runtime session)."""
    interface: ModelInterface
    domain: ModelDomain
    provenance: ModelProvenance
    metrics: dict[str, float]

def read_metadata(path: str) -> PackageMetadata:
    """Read ONLY the RF3M header: interface + domain + provenance + metrics. No graphs are parsed
    and no inference session is built. Use this instead of parsing the container in Python."""
    ...

class ModelProvenance:
    """Lightweight training provenance (dataset name, simulation software/physics)."""
    dataset_name: str
    software_version: str
    physics: str
    def __init__(self, dataset_name: str = ..., software_version: str = ..., physics: str = ...) -> None: ...

# ── predictor hierarchy (radfield3dnn::) ────────────────────────────────────────────
# Any `buffer` below is a numpy array or a torch.Tensor (CPU or CUDA). Its memory is bound
# DIRECTLY — nothing is copied and no result is allocated.
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
    @property
    def session_memory(self) -> SessionMemory:
        """What this session consumes with no copy (device / precision / execution provider)."""

    @overload
    def predict_volume(
        self,
        beam: BeamParameters,
        dims: tuple[int, int, int] | Sequence[int],
        max_inner_batch: int = ...,
    ) -> FieldPrediction:
        """Allocating path: returns fresh arrays and normalises ``beam`` against the domain."""
    @overload
    def predict_volume(self) -> None:
        """Zero-copy path: runs into the BOUND buffers and returns nothing.

        Raises :class:`IncompleteSetupError` — listing every gap at once — if the grid or a required
        binding is missing.
        """

    # ── zero-copy binding: the caller owns the memory ────────────────────────────
    def set_voxel_grid(self, voxel_counts: tuple[int, int, int] | Sequence[int]) -> None:
        """The grid the next :meth:`predict_volume` fills. Chosen per call — nothing about it is
        baked into the graph."""
    @property
    def voxel_grid(self) -> tuple[int, int, int]: ...
    def bind_global_parameter(
        self,
        flag: ModelInput,
        buffer: npt.NDArray[np.float32] | object,   # numpy array or torch.Tensor (CPU/CUDA)
        convert_buffer: bool = False,
    ) -> None:
        """Bind a per-field input's memory for ``flag``.

        The buffer is re-read on every run, so editing it IN PLACE is picked up by the next
        inference — no rebinding. Values reach the graph verbatim: it must hold NORMALISED values
        (see :meth:`parameter_normalizer`).

        Strict by default: a device/precision mismatch with the session raises :class:`BindingError`
        explaining the cost. ``convert_buffer=True`` accepts it and SNAPSHOTS the buffer at bind
        time — later in-place edits are then no longer seen.
        """
    def bind_output_layer(
        self,
        flag: ModelOutput,
        buffer: npt.NDArray[np.float32] | object,
        convert_buffer: bool = False,
    ) -> None:
        """Bind the caller's memory for an output — the model writes STRAIGHT into it, allocating
        nothing. Call :meth:`set_voxel_grid` first (it fixes the required size)."""
    def clear_bindings(self) -> None: ...
    def required_elements(self, flag: ModelInput | ModelOutput) -> int:
        """How many elements a buffer for this flag must hold at the current grid."""
    def parameter_normalizer(self) -> ParameterNormalizer:
        """A converter that knows THIS model's domain: metric values in, normalised values written
        into the bound buffers."""

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
    "read_metadata",
    "FieldPrediction",
    "BeamParameters",
    "BeamParameterSpec",
    "CollimationType",
    "EncodedBeam",
    "ExecutionOptions",
    "ModelDomain",
    "ModelInput",
    "ModelInterface",
    "ModelOutput",
    "ModelProvenance",
    "ModelStore",
    "PackageMetadata",
    "ParameterRange",
    "ParameterRangeType",
    "PredictorType",
    "SpectrumInputLayout",
    "VolumeFieldPredictor",
    "VoxelFieldPredictor",
]

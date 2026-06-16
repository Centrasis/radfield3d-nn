"""Type stubs for the ``rfnn_deploy`` native module (the compiled C++ deployment bindings).

The full, authoritative definitions of the inference types live in the package's own shipped stub,
``radfield3dnn/deploy/__init__.pyi`` (so consumers get them via PEP 561 without this dev-only
stub-path). This file just re-exports them under the bare ``rfnn_deploy`` name — which is how the
module is imported internally (``radfield3dnn/deploy/onnx_runtime.py`` loads the ``.so`` by path) —
and adds the two module-level serialiser functions that are native-only (not part of the public
:mod:`radfield3dnn.deploy` surface).
"""
from __future__ import annotations

from typing import Mapping

from radfield3dnn.deploy import (
    BeamParameters as BeamParameters,
    BeamParameterSpec as BeamParameterSpec,
    EncodedBeam as EncodedBeam,
    ExecutionOptions as ExecutionOptions,
    FieldPrediction as FieldPrediction,
    ModelDomain as ModelDomain,
    ModelProvenance as ModelProvenance,
    ModelStore as ModelStore,
    ParameterRange as ParameterRange,
    PredictorType as PredictorType,
    VolumeFieldPredictor as VolumeFieldPredictor,
    VoxelFieldPredictor as VoxelFieldPredictor,
)

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

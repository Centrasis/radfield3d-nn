"""Spectra-encoder factory: build a spectrum encoder from a ``{"type": ..., **kwargs}`` dict.

Mirrors :mod:`factory` (location/direction encoders): the config selects one self-contained encoder
class by ``type`` and supplies that class's own attributes. This replaces the flat
``in_spectra_dim`` / ``encoded_spectra_dims`` / ``use_spectra_encoding`` trio, e.g.::

    "spectra_encoding": {"type": "projector", "in_spectra_dim": 150, "out_spectra_dim": 150}
    "spectra_encoding": {"type": "simple",    "in_spectra_dim": 150, "encoded_spectra_dims": 64}
    "spectra_encoding": {"type": "conv",      "in_spectra_dim": 150, "encoded_spectra_dims": 64}

Every spectra encoder exposes ``encoded_dims`` (its output width); the backbone reads it instead of
recomputing the width from flags. ``projector`` keeps the raw-spectrum dim (the old
``use_spectra_encoding=False`` path); ``simple`` is the MLP bottleneck (old ``True``).
"""
from typing import Any

from .spectra_encoder import SimpleSpectraEncoder, SpectraProjector, SpectraEncoder

# type -> spectra encoder class. Each class keeps its own constructor signature.
SPECTRA_ENCODER_REGISTRY = {
    "simple": SimpleSpectraEncoder,
    "projector": SpectraProjector,
    "conv": SpectraEncoder,
}


def build_spectra_encoding(params: dict):
    """Construct a spectra encoder from ``params`` (mandatory ``type`` + class-specific kwargs).

    Returns the encoder; read ``encoder.encoded_dims`` for its output width.
    """
    if not isinstance(params, dict) or "type" not in params:
        raise ValueError(f"spectra encoding params must be a dict with a 'type' key, got {params!r}")
    p: dict[str, Any] = {k: v for k, v in params.items() if k != "type"}
    etype = params["type"]
    if etype not in SPECTRA_ENCODER_REGISTRY:
        raise ValueError(f"Unknown spectra encoding type {etype!r}. Valid: {list(SPECTRA_ENCODER_REGISTRY)}.")
    return SPECTRA_ENCODER_REGISTRY[etype](**p)

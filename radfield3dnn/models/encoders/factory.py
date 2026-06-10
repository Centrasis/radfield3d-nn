"""Encoder factory: build a location/direction encoder from a ``{"type": ..., **kwargs}`` dict.

Each encoder is its own self-contained class; the config selects one by ``type`` and supplies that
class's own attributes. This replaces the flat ``location_encoding_type`` + ``hash_*`` + ``rff_*``
parameter soup with one nested dict per encoder, e.g.::

    "location_encoding": {"type": "rff", "num_features": 96, "sigma": 2.0, "append_input": true}
    "location_encoding": {"type": "sinusoidal", "pos_enc_dim": 12, "append_input": true}
    "location_encoding": {"type": "hash", "n_levels": 16, "features_per_level": 2}

The spatial input dim (``d_input``/``in_dim``) is injected by the caller's ``default_d_input`` when the
chosen class needs it and the dict omits it, so configs stay terse.
"""
import inspect
from typing import Any

from .sinusoidal_encoding import SinusoidalFrequencyEncoding
from .hash_encoding import HashGridEncoding
from .rff_encoding import RandomFourierFeatures
from .spherical_hamonics import SphericalHarmonics

# type -> encoder class. Each class keeps its own constructor signature (its "real attributes").
ENCODER_REGISTRY = {
    "sinusoidal": SinusoidalFrequencyEncoding,
    "hash": HashGridEncoding,
    "rff": RandomFourierFeatures,
    "spherical_harmonics": SphericalHarmonics,
}


def build_encoding(params: dict, default_d_input: int = 3):
    """Construct an encoder from ``params`` (a dict with a mandatory ``type`` + class-specific kwargs).

    The spatial-dim kwarg (``d_input`` or ``in_dim``) is injected from ``default_d_input`` if the class
    accepts it and the dict does not provide it.
    """
    if not isinstance(params, dict) or "type" not in params:
        raise ValueError(f"encoding params must be a dict with a 'type' key, got {params!r}")
    p: dict[str, Any] = {k: v for k, v in params.items() if k != "type"}
    etype = params["type"]
    if etype not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoding type {etype!r}. Valid: {list(ENCODER_REGISTRY)}.")
    cls = ENCODER_REGISTRY[etype]
    sig = inspect.signature(cls.__init__).parameters
    for dim_key in ("d_input", "in_dim"):
        if dim_key in sig and dim_key not in p:
            p[dim_key] = default_d_input
    return cls(**p)

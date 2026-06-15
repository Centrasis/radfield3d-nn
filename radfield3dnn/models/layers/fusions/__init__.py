"""Two-vector feature fusions (merge a main/location vector with a conditioning/beam vector).

All concrete fusions share the FusionBase contract: forward(x, cond) -> y with dim(y) == dim(x),
and are budgeted as one logical trunk layer each.
"""
from .base import FusionBase
from .film import FiLM, ResidualFiLM
from .gates import GatedFusion, ModulativeSigmoidGate, ResidualAdditiveTanhGate
from .concat import ConcatLinear
from .attention import CrossAttentionFusion, TokenCrossAttentionFusion

__all__ = [
    "FusionBase",
    "FiLM", "ResidualFiLM",
    "GatedFusion", "ModulativeSigmoidGate", "ResidualAdditiveTanhGate",
    "ConcatLinear",
    "CrossAttentionFusion",
    "TokenCrossAttentionFusion",
]

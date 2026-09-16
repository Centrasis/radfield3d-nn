from torch import nn, Tensor
import torch

from .fusions import (
    FusionBase,
    FiLM, ResidualFiLM,
    GatedFusion, ModulativeSigmoidGate, ResidualAdditiveTanhGate,
    ConcatLinear,
    CrossAttentionFusion,
    TokenCrossAttentionFusion,
)


class Concat(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.cat(x, dim=self.dim) if len(x) > 1 else x[0]

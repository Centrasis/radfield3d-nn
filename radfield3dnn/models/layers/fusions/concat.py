from torch import nn, Tensor
import torch
from typing import Literal

from .base import FusionBase


class ConcatLinear(FusionBase):
    """Early-fusion: concatenate the main and conditioning vectors and project back to ``out_channels``.

    ``y = act( W · [norm(x) ‖ cond] + b )``  →  one Linear, so it counts as a single trunk layer.

    Unlike FiLM (which restricts the beam to a *per-channel affine* γ·x+β), the conditioning vector
    enters the **same full-rank linear remix** as x: it can add an arbitrary cond-derived vector to
    every output channel. That is the cheapest way to give the beam a full additive voice over a voxel
    (FiLM- and ConcatLinear-equal in params, ~``out·(out+cond)``), and the direct test of whether
    FiLM's affine restriction is what starves beam→voxel sensitivity.

    Identity-startable: the x-half of ``W`` is initialised to identity and the cond-half to zero, so at
    init ``y ≈ act(norm(x))`` — i.e. it begins like an ordinary norm+Linear+act trunk layer and *learns*
    to mix in the beam, matching FiLM's identity-at-start stability.
    """

    def __init__(self, condition_channels: int, out_channels: int,
                 non_linearity: type[nn.Module] | None = nn.SiLU,
                 norm: Literal["layer", "none"] = "none"):
        super().__init__(condition_channels, out_channels)
        assert non_linearity is None or isinstance(non_linearity, type), \
            "non_linearity must be a class/type or None."
        self.norm_x = nn.LayerNorm(out_channels) if norm == "layer" else nn.Identity()
        self.proj = nn.Linear(out_channels + condition_channels, out_channels)
        self.non_linearity = non_linearity(inplace=True) if non_linearity is not None else None
        self.initialize()

    def initialize(self) -> None:
        with torch.no_grad():
            self.proj.weight.zero_()
            # x-half = identity (start ≈ norm(x)), cond-half = 0 (no beam at init).
            self.proj.weight[:, :self.out_channels].copy_(torch.eye(self.out_channels))
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        self._check(x, cond)
        h = torch.cat([self.norm_x(x), cond], dim=-1)
        y = self.proj(h)
        return self.non_linearity(y) if self.non_linearity is not None else y

from torch import nn
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Literal


class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) layer from the paper "FiLM: Visual Reasoning with a General Conditioning Layer" (2017).
    """
    def __init__(self, condition_channels: int, out_channels: int, non_linearity: type[nn.Module] | None = nn.ReLU, norm: Literal["layer", "batch2d", "batch3d", "none"] = "layer"):
        """
        Args:
            condition_channels (int): Number of channels in the conditioning input
            out_channels (int): Number of channels in the input to be modulated
            non_linearity (nn.Module | None): Non-linearity to apply after modulation. If None, no non-linearity is applied.
        """
        super().__init__()
        assert non_linearity is None or isinstance(non_linearity, type), "non_linearity must be a class/type or None."
        self.gamma_beta = nn.Linear(condition_channels, 2 * out_channels)  # gamma || beta
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.non_linearity = non_linearity(inplace=True) if non_linearity is not None else None
        if norm == "layer":
            self.norm_x = nn.LayerNorm(out_channels)
        elif norm == "batch2d":
            self.norm_x = nn.BatchNorm2d(out_channels)
        elif norm == "batch3d":
            self.norm_x = nn.BatchNorm3d(out_channels)
        elif norm == "none":
            self.norm_x = nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

        # Initialize the last layer to produce zeros, so that FiLM starts as an identity function
        self.initialize()

    def initialize(self):
        """
        Re-initialize the FiLM layer to its initial state.
        """
        with torch.no_grad():
            nn.init.normal_(self.gamma_beta.weight.data, mean=0.0, std=1e-3)
            nn.init.zeros_(self.gamma_beta.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        assert x.size(0) == condition.size(0), "Batch size of input and condition must match."

        assert x.size(1) == self.out_channels, f"Expected input tensor with {self.out_channels} channels, but got {x.shape[1]}."
        assert condition.shape[1] == self.condition_channels, f"Expected condition tensor with {self.condition_channels} channels, but got {condition.shape[1]}."
        gamma_beta = self.gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        x = self.norm_x(x)

        shape = (x.size(0), self.out_channels) + (1,) * (x.ndim - 2)
        gamma = 1.0 + nn.functional.tanh(gamma.view(shape)) # identity at the start and bound gain
        beta = beta.view(shape)

        modulated = x * gamma + beta  # apply FiLM modulation and all normalized
        return self.non_linearity(modulated) if self.non_linearity is not None else modulated


class ResidualFiLM(FiLM):
    """
    Residual FiLM layer that adds the input to the modulated output.
    """
    def __init__(self, condition_channels: int, out_channels: int, non_linearity: type[nn.Module] | None = nn.ReLU):
        super().__init__(condition_channels, out_channels, None)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.outer_non_linearity = non_linearity(inplace=True) if non_linearity is not None else None

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        modulated = super().forward(x, condition)
        out = x + self.alpha * modulated
        return self.outer_non_linearity(out) if self.outer_non_linearity is not None else out

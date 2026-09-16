import torch
from torch import nn
from torch import Tensor


class GradientConservingClamping(nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        clamped_x = torch.clamp(x, min=self.min_value, max=self.max_value)
        return x + (clamped_x - x).detach()


class ArcTan(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (4.0 / torch.pi) * torch.atan(x) # Scaled to [-1, 1] for x in [-1, 1]


class SymmetricSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(x) - 1.0  # Scaled to [-1, 1] for x in [-inf, inf]


class SmoothedTanh(nn.Module):
    def __init__(self, scale: float = 1.2):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x / self.scale) * self.scale  # Scaled tanh for smoother transitions

from .linear import LinearNormalizer
from typing import Union
from torch import Tensor
import torch
import torch.nn as nn


class AsinhNormalizer(LinearNormalizer):
    def __init__(self, range: tuple[float, float] = (0.0, 1.0), input_scale: float = 1.0):
        super().__init__(range)
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        assert self.range[1] > self.range[0], "Invalid range for AsinhNormalizer."

    def get_type(self) -> str:
        exponent = int(torch.log10(self.input_scale).item())
        sign_prefix = "+" if exponent > 0 else ""
        scale_str = f"_1e{sign_prefix}{exponent}"
        return "asinh" + scale_str
    
    def __repr__(self):
        return f"AsinhNormalizer(range={self.range}, input_scale={self.input_scale.item()})"
    
    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for normalization.
        Ensures that x is non-negative.
        Raises an error if the conditions are not met.
        """
        min_val = 0.0
        if abs(self.range[0]) > 1e-7:
            min_val = 1e-7
        x_min = x.min()
        if x_min < min_val:
            raise ValueError(f"Input to AsinhNormalizer must be above {min_val}. Minimum: {x_min.item()}.")

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None] = None) -> Tensor:
        with torch.no_grad():
            x_min = torch.amin(x, dim=tuple(range(1, x.ndim)), keepdim=True) if respect_to is not None else torch.ones_like(x)
            x_min = torch.min(x_min, torch.tensor(0.0, device=x.device))
            x_max = torch.amax(x, dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-12)
            x = x - x_min  # shift to make non-negative
            x = x / x_max  # scale to [0..1]
            
            y = torch.asinh(x / self.input_scale) / torch.asinh((1.0 / self.input_scale)) # map to [0..1]

            a, b = self.range
            y = y * (b-a)
            y = y + a  # map to target range
            return y

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None] = None) -> Tensor:
        with torch.no_grad():
            a, b = self.range
            x = x - a  # shift to make non-negative
            x = x / (b - a)  # scale to [0..1]

            x = torch.sinh(x * torch.asinh(1.0 / self.input_scale)) * self.input_scale  # inverse asinh
            if respect_to is not None:
                x_max = torch.amax(respect_to, dim=tuple(range(1, respect_to.ndim)), keepdim=True)
                x = x * x_max  # scale back
            return x


class LearnableAsinhNormalizer(AsinhNormalizer):
    def __init__(self, range: tuple[float, float] = (0.0, 1.0), exponent_range: tuple[float, float] = (2, 4), base: float = 10.0):
        super().__init__(range=range)
        self.exponent_range = exponent_range
        self.register_buffer("base", torch.tensor(base, dtype=torch.float32))
        self.input_scale_exponent = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    @property
    def input_scale(self) -> Tensor:
        exp = self.sigmoid(self.input_scale_exponent) * (self.exponent_range[1] - self.exponent_range[0]) + self.exponent_range[0]
        return torch.pow(self.base, -exp)

    def get_type(self) -> str:
        return "learnable_asinh"

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None] = None) -> Tensor:
        x_min = min(0.0, x.min() if respect_to is None else respect_to.min())
        x = x - x_min  # shift to make non-negative
        x_max = torch.amax(x, dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-12)
        x = x / x_max  # scale to [0..1]
        
        y = torch.asinh(x / self.input_scale) / torch.asinh((1.0 / self.input_scale)) # map to [0..1]

        a, b = self.range
        y = y * (b-a)
        y = y + a  # map to target range
        return y

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None] = None) -> Tensor:
        a, b = self.range
        x = x - a  # shift to make non-negative
        x = x / (b - a)  # scale to [0..1]

        x = torch.sinh(x * torch.asinh(1.0 / self.input_scale)) * self.input_scale  # inverse asinh
        if respect_to is not None:
            x_max = torch.amax(respect_to, dim=tuple(range(1, respect_to.ndim)), keepdim=True).clamp_min(1e-12)
            x = x * x_max  # scale back
        return x

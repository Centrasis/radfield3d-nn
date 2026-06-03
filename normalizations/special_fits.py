import torch
from .base import Normalizer
from typing import Union
from torch import Tensor


class SpecialPolynomialNormalizer(Normalizer):
    def __init__(self):
        super().__init__()
        self.coeffs = torch.tensor([
            58.0/7.0,
            -28.0,
            344.0/7.0,
            -320.0/7.0,
            128.0/7.0
        ], dtype=torch.float32, requires_grad=False)  # Coefficients for the polynomial fit

    def get_type(self) -> str:
        return "special_polynomial"
    
    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for normalization.
        Ensures that x is non-negative and has a sufficient range of values.
        Raises an error if the conditions are not met.
        """
        x_min = x.min()
        if x_min < 0.0:
            raise ValueError(f"Input to SpecialPolynomialNormalizer must be non-negative. Minimum: {x_min.item()}.")
        min_2nd = x[x > x_min].min() if (x > x_min).any() else x_min
        if min_2nd < 1e-8:
            raise ValueError(f"Input to SpecialPolynomialNormalizer has too small values. Minimum: {x_min.item()}, 2nd minimum: {min_2nd.item()}. Consider using a different normalizer.")
        
    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            assert x.min() >= 0.0, "Input to SpecialPolynomialNormalizer must be non-negative."
            assert x.max() <= 1.0, "Input to SpecialPolynomialNormalizer must be in the range [0, 1]."
            x_orig_shape = x.shape
            if self.coeffs.device != x.device:
                self.coeffs = self.coeffs.to(x.device)

            # f(x) = -1 + sum(c_i * x^i) for i in [0, 4]
            x_powers = torch.stack([x**(i+1) for i in range(len(self.coeffs))], dim=1)  # Shape: (B, 5, ...)
            poly = torch.tensor(-1.0, dtype=x.dtype, device=x.device).expand_as(x).clone()  # Start with -1.0, shape: (B, ...)
            for i in range(len(self.coeffs)):  # Sum up the polynomial terms in a loop to save memory
                poly += self.coeffs[i] * x_powers[:, i]
            poly = poly.squeeze(1)

            assert torch.isfinite(poly).all(), "Normalization resulted in non-finite values."
            assert poly.min() >= -1.0 and poly.max() <= 1.0, "Normalization resulted in values outside [-1, 1]."
            return poly.view(x_orig_shape)

    def apply_inverse_transformation(self, x, respect_to):
        with torch.no_grad():
            assert x.min() >= -1.0 and x.max() <= 1.0, f"Input to inverse SpecialPolynomialNormalizer must be in [-1, 1], but is in [{x.min().item()}, {x.max().item()}]."
            
            coeffs = torch.tensor([
                58.0/7.0,
                -28.0,
                344.0/7.0,
                -320.0/7.0,
                128.0/7.0
            ], dtype=torch.float32, requires_grad=False, device=x.device)  # Coefficients for the polynomial fit (cant use the buffer directly in calculations, as pytorch is setting all coeffs to 0 during training otherwise)

            # Use Newton-Raphson method to find roots of f(y) - x = 0
            def poly_func(y: Tensor):
                y = y.unsqueeze(1)
                y_powers = torch.stack([y**(i+1) for i in range(len(coeffs))], dim=1)  # Shape: (B, 5, ...)
                return -1.0 + torch.tensordot(coeffs, y_powers, dims=([0], [1])).squeeze(1)

            def poly_deriv(y: Tensor):
                y = y.unsqueeze(1)
                deriv_coeffs = torch.tensor([ (i+1)*coeffs[i] for i in range(len(coeffs)) ], dtype=y.dtype, device=y.device)
                y_powers = torch.stack([y**i for i in range(len(deriv_coeffs))], dim=1)  # Shape: (B, 5, ...)
                return torch.tensordot(deriv_coeffs, y_powers, dims=([0], [1])).squeeze(1)

            y = torch.clamp(x, min=0.0, max=1.0)  # Initial guess
            for _ in range(10):  # 10 iterations should be sufficient
                f_y = poly_func(y)
                f_prime_y = poly_deriv(y)
                y = y - (f_y - x) / (f_prime_y + 1e-6)  # Avoid division by zero
                y = torch.clamp(y, min=0.0, max=1.0)  # Keep within bounds

            assert torch.isfinite(y).all(), "Inverse normalization resulted in non-finite values."
            assert y.min() >= 0.0 and y.max() <= 1.0, "Inverse normalization resulted in values outside [0, 1]."
            return y

    def clone(self) -> "SpecialPolynomialNormalizer":
        new_inst: SpecialPolynomialNormalizer = super().clone()
        new_inst.coeffs = self.coeffs.clone().detach()
        return new_inst


class LearnableLogNorm(Normalizer):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.range = (-1.0, 1.0)    # needed for SRBFNets to select proper activation functions

    def get_type(self) -> str:
        return "learnable_log"
    
    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for normalization.
        Ensures that x is non-negative and has a sufficient range of values.
        Raises an error if the conditions are not met.
        """
        x_min = x.min()
        if x_min < 0.0:
            raise ValueError(f"Input to LearnableLogNorm must be non-negative. Minimum: {x_min.item()}.")
        min_2nd = x[x > x_min].min() if (x > x_min).any() else x_min
        if min_2nd < 1e-8:
            raise ValueError(f"Input to LearnableLogNorm has too small values. Minimum: {x_min.item()}, 2nd minimum: {min_2nd.item()}. Consider using a different normalizer.")
        
    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        assert x.min() >= 0.0, "Input to LearnableLogNorm must be non-negative."
        x = (x + 0.03) / ((torch.max(x) if respect_to is None else torch.max(respect_to)) + 0.03)  # Normalize to [0, 1] with small offset to avoid log(0)
        assert x.max() <= 1.0, "Input to LearnableLogNorm must be in the range [0, 1]."
        x_orig_shape = x.shape
        y = torch.log1p(10 ** self.m * torch.min(x, torch.tensor(1.0, dtype=x.dtype, device=x.device))) / torch.log1p(10 ** self.m)
        y = (y * 2.0) - 1.0  # Scale to [-1, 1]
        assert torch.isfinite(y).all(), "Normalization resulted in non-finite values."
        assert y.min() >= -1.0 and y.max() <= 1.0, "Normalization resulted in values outside [-1, 1]."
        return y.view(x_orig_shape)
    
    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        assert (x.min().isclose(torch.tensor(-1.0), atol=1e-6) or x.min() >= -1.0) and (x.max().isclose(torch.tensor(1.0), atol=1e-6) or x.max() <= 1.0), f"Input to inverse LearnableLogNorm must be in [-1, 1], but is in [{x.min().item()}, {x.max().item()}]."
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = (x + 1.0) / 2.0  # Scale to [0, 1]
        x_orig_shape = x.shape
        y = (torch.expm1(x * torch.log1p(10 ** self.m))) / (10 ** self.m)
        assert torch.isfinite(y).all(), "Inverse normalization resulted in non-finite values."
        y = (y - 0.03) / (1.0 - 0.03)  # Scale back to [0, 1]
        assert y.min() >= 0.0, "Inverse normalization resulted in negative values."
        return y.view(x_orig_shape)

    def clone(self):
        new_inst = super().clone()
        new_inst.m = torch.nn.Parameter(self.m.clone())
        return new_inst

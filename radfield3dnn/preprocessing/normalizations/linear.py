import torch
from .base import Normalizer
from typing import Union
from torch import Tensor


class LinearNormalizer(Normalizer):
    def __init__(self, range: tuple[float, float] = (0.0, 1.0), always_normalize: bool = False):
        super().__init__()
        self.range = range
        self.min_input = 1e-8 if (range[1] - range[0]) <= 1.0 else 1e-6
        self.always_normalize = always_normalize

    def get_type(self) -> str:
        return f"linear{int(self.range[0])}_{int(self.range[1])}"

    def validate_range(self, x: Tensor):
        """
        Validate that the input tensor x is suitable for normalization.
        Ensures that x is non-negative and has a sufficient range of values.
        Raises an error if the conditions are not met.
        """
        valid_mask = torch.isfinite(x)
        if not valid_mask.all():
            x = x[valid_mask]
        x_min = x.min()
        if x_min < 0.0:
            raise ValueError(f"Input to LinearNormalizer must be non-negative. Minimum: {x_min.item()}.")
        min_2nd = x[x > x_min].min() if (x > x_min).any() else x_min
        if min_2nd < self.min_input:
            raise ValueError(f"Input to LinearNormalizer has too small values. Minimum: {x_min.item()}, 2nd minimum: {min_2nd.item()}. Consider using a different normalization range or a different normalizer.")

    @staticmethod
    def _masked_amin(t: Tensor, valid: Tensor, dims: tuple[int, ...]) -> Tensor:
        return torch.amin(torch.where(valid, t, torch.full_like(t, float("inf"))), dim=dims, keepdim=True)

    @staticmethod
    def _masked_amax(t: Tensor, valid: Tensor, dims: tuple[int, ...]) -> Tensor:
        return torch.amax(torch.where(valid, t, torch.full_like(t, float("-inf"))), dim=dims, keepdim=True)

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            valid_mask = torch.isfinite(x)
            if not valid_mask.any():
                return torch.full_like(x, self.range[0])
            reduce_dims = tuple(range(1, x.ndim))

            # shape-preserving per-sample reductions over finite voxels (boolean indexing would flatten the batch).
            if self.always_normalize and respect_to is None:
                sample_min = self._masked_amin(x, valid_mask, reduce_dims)
                x = torch.where(valid_mask, x - sample_min, x)

            if respect_to is not None:
                r_valid = torch.isfinite(respect_to)
                r_dims = tuple(range(1, respect_to.ndim))
                x_min = self._masked_amin(respect_to, r_valid, r_dims)
                max = self._masked_amax(respect_to, r_valid, r_dims).clamp_min(1e-12)
            else:
                x_min = torch.zeros((), device=x.device, dtype=x.dtype)
                max = self._masked_amax(x, valid_mask, reduce_dims).clamp_min(1e-12)

            finite_min = self._masked_amin(x, valid_mask, reduce_dims)
            assert bool((finite_min >= x_min - 1e-6).all()), \
                f"Input to LinearNormalizer must be >= {x_min}, but min finite value is {finite_min.min().item()}."

            normalized = x / max
            normalized = normalized * (self.range[1] - self.range[0]) + self.range[0]
            normalized = torch.where(valid_mask, normalized, torch.full_like(normalized, self.range[0]))
            assert torch.isfinite(normalized).all(), "Normalization resulted in non-finite values."
            return normalized

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            valid_mask = torch.isfinite(x)
            finite = x[valid_mask]   # validate over finite voxels only (the -inf mask sentinel would fail the assert)
            if finite.numel():
                assert finite.min() >= self.range[0] - 1e-6 and finite.max() <= self.range[1] + 1e-6, \
                    f"Input to inverse LinearNormalizer must be in [{self.range[0]}, {self.range[1]}], " \
                    f"but finite values span [{finite.min().item()}, {finite.max().item()}]."
            out = (x - self.range[0]) / (self.range[1] - self.range[0])
            if respect_to is not None:
                r_valid = torch.isfinite(respect_to)
                r_dims = tuple(range(1, respect_to.ndim))
                max = self._masked_amax(respect_to, r_valid, r_dims)
            else:
                max = 1.0
            out = out * max
            return torch.where(valid_mask, out, x)

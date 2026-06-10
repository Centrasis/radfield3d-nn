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

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            valid_mask = torch.isfinite(x)
            if not valid_mask.all():
                orig_values = x.clone()
                x = x[valid_mask]
                if x.numel() == 0:
                    # The whole field is non-finite (e.g. a degenerate/empty field
                    # or one the importance sampler fully dropped). There is
                    # nothing to normalise — map to the low end of the output
                    # range so downstream metrics see a well-formed zero field
                    # instead of crashing on amin() over an empty tensor.
                    return torch.full_like(orig_values, self.range[0])
            if self.always_normalize:
                x = x - torch.amin(x, dim=tuple(range(1, x.ndim)), keepdim=True)   # ensure min = 0

            x_min = 0.0 if respect_to is None else torch.amin(respect_to, dim=tuple(range(1, respect_to.ndim)), keepdim=True)
            assert x.min() >= x_min, f"Input to LinearNormalizer must be >= {x_min}, but minimum value is {x.min().item()}."
            max = torch.amax(respect_to, dim=tuple(range(1, respect_to.ndim)), keepdim=True).clamp_min(1e-12) if respect_to is not None else torch.amax(x, dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-12)
            if respect_to is None:
                x = x - x_min
                max = max - x_min
            normalized = x / max
            normalized = normalized * (self.range[1] - self.range[0]) + self.range[0]
            assert torch.isfinite(normalized).all(), "Normalization resulted in non-finite values."
            if not valid_mask.all():
                orig_values[valid_mask] = normalized
                normalized = orig_values
            return normalized

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        with torch.no_grad():
            assert x.min() >= self.range[0] and x.max() <= self.range[1], f"Input to inverse LinearNormalizer must be in [{self.range[0]}, {self.range[1]}], but is in [{x.min().item()}, {x.max().item()}]."
            valid_mask = torch.isfinite(x)
            if not valid_mask.all():
                orig_values = x
                x = x[valid_mask]
            x = (x - self.range[0])  # Shift to [0, range]
            x = x / (self.range[1] - self.range[0])  # Scale to [0, 1]
            max = torch.amax(respect_to, dim=tuple(range(1, respect_to.ndim)), keepdim=True) if respect_to is not None else 1.0
            x = x * max
            if not valid_mask.all():
                orig_values[valid_mask] = x
                x = orig_values
            return x


class JointLinearNormalizer(LinearNormalizer):
    """Shared-scale linear normaliser for multi-channel (two-head) fields.

    The base ``LinearNormalizer`` dispatch (``base.py`` recurses into ``scatter_field`` and
    ``direct_beam`` independently) would divide each channel by its OWN per-field max — destroying
    the scatter:direct magnitude relation exactly like ``ChannelsSplitRelative`` did. This subclass
    divides BOTH flux channels by the SAME per-field max (over scatter ∪ direct), so the relation is
    preserved in the [0,1] targets (direct ≈ peak 1, scatter at its true small fraction). The tiny
    scatter then gets its gradient footing back from ``ChannelMaxBalancedLoss`` (in the loss, not the
    data). Single-channel inputs fall straight through to ``LinearNormalizer``.

    Inverse is inherited: with ``respect_to=None`` it is per-channel identity (max=1.0), leaving both
    channels on the shared [0,1] scale so ``ChannelsJoin`` recombines them correctly.
    """

    def get_type(self) -> str:
        return "linear_joint"

    @staticmethod
    def _shared_max(a: Tensor, b: Tensor) -> Tensor:
        fa = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        fb = torch.where(torch.isfinite(b), b, torch.zeros_like(b))
        ma = fa.amax(dim=tuple(range(1, a.ndim)), keepdim=True)
        mb = fb.amax(dim=tuple(range(1, b.ndim)), keepdim=True)
        return torch.maximum(ma, mb).clamp_min(1e-12)

    def forward(self, x, respect_to=None):
        from radfield3dnn.rftypes import (RadiationField, rf3RadiationField,
                                          TrainingInputData, rf3TrainingInputData)
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth,
                                          respect_to.ground_truth if respect_to is not None else None),
                original_ground_truth=getattr(x, "original_ground_truth", None),
            )
        if (isinstance(x, (RadiationField, rf3RadiationField))
                and x.scatter_field is not None and x.direct_beam is not None):
            scale = self._shared_max(x.scatter_field.flux, x.direct_beam.flux).detach()
            return RadiationField(
                scatter_field=x.scatter_field._replace(flux=x.scatter_field.flux / scale),
                direct_beam=x.direct_beam._replace(flux=x.direct_beam.flux / scale),
                geometry=getattr(x, "geometry", None),
            )
        return super().forward(x, respect_to)

    def clone(self) -> "LinearNormalizer":
        new_int: LinearNormalizer = super().clone()
        new_int.range = (
            self.range[0],
            self.range[1]
        )
        new_int.min_input = self.min_input
        return new_int

import torch
from .base import Normalizer
from typing import Union
from torch import Tensor


class LinearNormalizer(Normalizer):
    def __init__(self, range: tuple[float, float] = (0.0, 1.0), always_normalize: bool = False,
                 dataset_scale: Union[float, None] = None):
        super().__init__()
        if dataset_scale is not None:
            if always_normalize:
                raise ValueError("dataset_scale and always_normalize are mutually exclusive: subtracting a "
                                 "per-field minimum would undo the one scale shared by every field.")
            if not (dataset_scale > 0.0):
                raise ValueError(f"dataset_scale must be positive, got {dataset_scale}.")
        self.range = range
        self.min_input = 1e-8 if (range[1] - range[0]) <= 1.0 else 1e-6
        self.always_normalize = always_normalize
        # One constant for the whole dataset (the largest fluence any field reaches) instead of each
        # field's own maximum. Keeps the RELATIVE magnitude between fields in the target, which a
        # per-field maximum discards, while still giving training and evaluation identical numbers.
        self.dataset_scale = float(dataset_scale) if dataset_scale is not None else None

    def get_type(self) -> str:
        return f"linear{int(self.range[0])}_{int(self.range[1])}"

    def deploy_params(self) -> list:
        # [min, max] of the PHYSICAL interval the normalized range maps onto. Without a
        # dataset_scale that maximum is each field's own, so there is no constant to hand out.
        return [0.0, self.dataset_scale] if self.dataset_scale is not None else []

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

    # No torch.no_grad() here: the transforms also run on PREDICTIONS (two-head models join their
    # prediction via inverse -> ChannelsJoin -> forward), so a no_grad wrapper would detach the flux
    # from the loss. Ground-truth tensors carry no grad, so nothing is built for them either way.
    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        valid_mask = torch.isfinite(x)
        dims = tuple(range(1, x.ndim))
        if self.always_normalize:
            x = x - self._masked_amin(x, valid_mask, dims, empty=0.0)
        if self.dataset_scale is not None:
            x_min = 0.0
            max = torch.full((1,) * x.ndim, self.dataset_scale, dtype=x.dtype, device=x.device)
        elif respect_to is None:
            x_min = 0.0
            max = self._masked_amax(x, valid_mask, dims, empty=1.0)
        else:
            respect_mask = torch.isfinite(respect_to)
            respect_dims = tuple(range(1, respect_to.ndim))
            x_min = self._masked_amin(respect_to, respect_mask, respect_dims, empty=0.0)
            max = self._masked_amax(respect_to, respect_mask, respect_dims, empty=1.0)
        assert (x[valid_mask] >= (x_min if isinstance(x_min, float) else x_min.expand_as(x)[valid_mask])).all(), \
            "Input to LinearNormalizer must not be smaller than the minimum of the reference tensor."
        # -inf sentinels divide to -inf and stay sentinels, so the masked voxels a ROI sampler
        # dropped survive the transform untouched.
        normalized = x / max.clamp_min(1e-12)
        normalized = normalized * (self.range[1] - self.range[0]) + self.range[0]
        assert torch.isfinite(normalized[valid_mask]).all(), "Normalization resulted in non-finite values."
        return normalized

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        x = (x - self.range[0]) / (self.range[1] - self.range[0])
        if respect_to is None:
            # With one scale for the dataset the inverse is exact without a reference field, so a
            # prediction returns to physical fluence here rather than staying in [0,1].
            return x * self.dataset_scale if self.dataset_scale is not None else x
        respect_mask = torch.isfinite(respect_to)
        respect_dims = tuple(range(1, respect_to.ndim))
        return x * self._masked_amax(respect_to, respect_mask, respect_dims, empty=1.0)

    @staticmethod
    def _masked_amax(x: Tensor, valid_mask: Tensor, dims: tuple[int, ...], empty: float) -> Tensor:
        """Largest finite value of every sample, keeping the batch dimension.

        `empty` replaces the reduction of a sample that holds no finite voxel at all, whose amax
        would be -inf and would turn the whole sample into NaN.
        """
        reduced = torch.where(valid_mask, x, torch.full_like(x, float("-inf"))).amax(dim=dims, keepdim=True)
        return torch.where(valid_mask.any(dim=dims, keepdim=True), reduced, torch.full_like(reduced, empty))

    @staticmethod
    def _masked_amin(x: Tensor, valid_mask: Tensor, dims: tuple[int, ...], empty: float) -> Tensor:
        """Smallest finite value of every sample, keeping the batch dimension."""
        reduced = torch.where(valid_mask, x, torch.full_like(x, float("inf"))).amin(dim=dims, keepdim=True)
        return torch.where(valid_mask.any(dim=dims, keepdim=True), reduced, torch.full_like(reduced, empty))

from torch import Tensor
import math
import torch
from typing import Union
from .base import Normalizer


class LogScaleNormalizer(Normalizer):
    """Non-normalized base-10 log compressor with true-zero support.

    Unlike LogDecadeNormalizer, this normalizer does NOT affinely squeeze the
    result into ``[0, 1]``: it returns values in the *real log domain*
    ``[log10(x_min), log10(x_max)]`` (default ``[-8, 0]``). This is intended
    for training where the loss already operates in log-space (e.g. plain
    ``L1Loss`` on these targets is equivalent to ``L1LogLoss`` on the raw
    flux), and where the network's output stage clamps directly to the
    log domain.

    True-zero handling: a flux of exactly ``0`` is a real value in the
    dataset (fully occluded voxels). ``log10(0) = -inf`` is unusable, and
    clamping ``0`` up to ``x_min`` would lose the distinction. This
    normalizer reserves a *zero sentinel* (``zero_floor``, default
    ``-9.0``) below ``log10(x_min)``: zeros forward to that sentinel and
    inverse maps anything below ``log10(x_min) - 0.5`` back to true ``0``.
    The gap ``(zero_floor, log10(x_min))`` is a dead zone the network
    learns to avoid (no training target lands there).

    Transform (per-element)::

        forward:  y = zero_floor                       if x == 0
                  y = log10(clamp(x, x_min, x_max))    else   -> [log10(x_min), log10(x_max)]
        inverse:  x = 0                                if y < log10(x_min) - 0.5
                  x = 10 ** clamp(y, log10(x_min), log10(x_max))   else

    Pair with PBRFNetCPP using::

        normalizer="log_scale", flux_activation="clamp",
        flux_clamp_min=zero_floor, flux_clamp_max=log10(x_max),
        flux_offset=(zero_floor + log10(x_max)) / 2

    so the network can emit the zero sentinel as well as any value in the
    log domain. Consider switching the flux loss to a plain ``L1Loss``
    when using this normalizer (the log step is now in the data, not the
    loss).
    """

    def __init__(self, x_min: float = 1e-8, x_max: float = 1.0,
                 zero_floor: float = -9.0):
        super().__init__()
        assert x_min > 0.0 and x_max > x_min, \
            f"Require 0 < x_min < x_max, got x_min={x_min}, x_max={x_max}."
        assert zero_floor < math.log10(x_min), \
            f"zero_floor ({zero_floor}) must be strictly below log10(x_min) ({math.log10(x_min)})."
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.zero_floor = float(zero_floor)

    @property
    def log_min(self) -> float:
        return math.log10(self.x_min)

    @property
    def log_max(self) -> float:
        return math.log10(self.x_max)

    @property
    def inverse_zero_threshold(self) -> float:
        # Midpoint between the zero sentinel and log10(x_min); anything
        # below this rounds back to true zero on inverse.
        return 0.5 * (self.zero_floor + self.log_min)

    def get_type(self) -> str:
        if self.x_min == 1e-8 and self.x_max == 1.0 and self.zero_floor == -9.0:
            return "log_scale"
        lo, hi = int(round(self.log_min)), int(round(self.log_max))
        return f"logscale_1e{lo}_1e{hi}_zf{int(self.zero_floor)}"

    def validate_range(self, x: Tensor):
        finite = torch.isfinite(x)
        xf = x[finite] if not finite.all() else x
        if xf.numel() and xf.min() < 0.0:
            raise ValueError(
                f"Input to LogScaleNormalizer must be non-negative. "
                f"Minimum: {xf.min().item()}."
            )

    def apply_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        if respect_to is not None and not isinstance(respect_to, Tensor):
            raise TypeError("respect_to must be a Tensor when normalizing a Tensor.")
        with torch.no_grad():
            in_dtype = x.dtype
            finite = torch.isfinite(x)
            if not finite.all():
                out = x.clone()
                xv = x[finite]
            else:
                out = None
                xv = x
            self.validate_range(xv)
            # Promote to fp32 — fp16 underflows 1e-8 to 0 and log10(0) = -inf.
            xv32 = xv.to(torch.float32)
            zero_mask = xv32 == 0.0
            c = torch.clamp(xv32, min=self.x_min, max=self.x_max)
            y = torch.log10(c)
            if zero_mask.any():
                y = torch.where(zero_mask, torch.full_like(y, self.zero_floor), y)
            assert torch.isfinite(y).all(), "Normalization resulted in non-finite values."
            y = y.to(in_dtype)
            if out is None:
                return y
            out[finite] = y
            return out

    def apply_inverse_transformation(self, x: Tensor, respect_to: Union[Tensor, None]) -> Tensor:
        if respect_to is not None and not isinstance(respect_to, Tensor):
            raise TypeError("respect_to must be a Tensor when normalizing a Tensor.")
        in_dtype = x.dtype
        finite = torch.isfinite(x)
        if not finite.all():
            out = x.clone()
            xv = x[finite]
        else:
            out = None
            xv = x
        xv32 = xv.to(torch.float32)
        zero_mask = xv32 < self.inverse_zero_threshold
        y = torch.clamp(xv32, min=self.log_min, max=self.log_max)
        recon = torch.pow(torch.tensor(10.0, dtype=torch.float32, device=xv32.device), y)
        if zero_mask.any():
            recon = torch.where(zero_mask, torch.zeros_like(recon), recon)
        assert torch.isfinite(recon).all(), "Inverse normalization resulted in non-finite values."
        recon = recon.to(in_dtype)
        if out is None:
            return recon
        out[finite] = recon
        return out

    def __repr__(self):
        return (f"LogScaleNormalizer(x_min={self.x_min}, "
                f"x_max={self.x_max}, zero_floor={self.zero_floor})")

    def clone(self) -> "LogScaleNormalizer":
        return LogScaleNormalizer(
            x_min=self.x_min,
            x_max=self.x_max,
            zero_floor=self.zero_floor,
        )

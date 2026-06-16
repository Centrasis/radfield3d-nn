from torch import Tensor
import math
import torch
from typing import Union
from .base import Normalizer


class AsinhTonemapNormalizer(Normalizer):
    """Smooth bounded HDR tonemap: ``y = asinh(x / sigma) / asinh(1 / sigma)``.

    The tonemap is linear near zero (``y ≈ x / sigma``), logarithmic for ``x >> sigma``,
    smooth everywhere, and bounded to ``[0, 1]``; inversion is closed-form. Unlike a raw
    log scale it has no zero sentinel (``asinh(0) = 0`` exact) and a per-element error
    budget bounded by 1, so it is fp16-safe under L1 / SSIM.

    σ is the "noise floor": values much smaller than σ map to ~0, values much larger map
    logarithmically towards 1.
    """

    def __init__(self, sigma: float = 1e-3):
        super().__init__()
        assert sigma > 0.0, f"Require sigma > 0, got sigma={sigma}."
        self.sigma = float(sigma)
        # The tonemap codomain is the closed [0, 1] interval (y = asinh(x/σ)/asinh(1/σ) with
        # x ∈ [0, 1] after per-field max-normalisation). Declaring it lets the (0,1)-codomain flux
        # heads — sigmoid / softclip — engage on asinh targets, not just LinearNormalizer(0,1).
        self.range = (0.0, 1.0)
        # Precompute the constant denominator asinh(1/sigma) once. Stored as
        # a python float; tensor versions are constructed per-call to match
        # the input tensor's device/dtype.
        self._scale = float(math.asinh(1.0 / self.sigma))

    @property
    def scale(self) -> float:
        return self._scale

    def get_type(self) -> str:
        # Compact tag in the form "asinh_3e-03" so wandb/MLFlow run
        # tagging stays human-readable for distinct sigma values.
        return f"asinh_{self.sigma:.0e}"

    def validate_range(self, x: Tensor):
        finite = torch.isfinite(x)
        xf = x[finite] if not finite.all() else x
        if xf.numel() and xf.min() < 0.0:
            raise ValueError(
                f"Input to AsinhTonemapNormalizer must be non-negative. "
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
            # Promote to fp32 for the asinh — fp16 asinh of tiny inputs
            # underflows. The final cast back to in_dtype is bounded in
            # [0, 1] and always representable in fp16.
            xv32 = xv.to(torch.float32)
            y = torch.asinh(xv32 / self.sigma) / self._scale
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
        # Clamp into the tonemap codomain before inverting — predictions
        # slightly outside [0, 1] are forced to the valid range; this also
        # makes the inverse safe against fp16 saturation at exactly 1.0.
        y = torch.clamp(xv32, min=0.0, max=1.0)
        recon = self.sigma * torch.sinh(y * self._scale)
        # Numerical guard: at y=1 the inverse is exactly 1.0 by construction
        # but accumulated fp32 error can drift by ~1e-7; clamp to [0, 1] to
        # make the round-trip exact at the endpoints.
        recon = torch.clamp(recon, min=0.0, max=1.0)
        assert torch.isfinite(recon).all(), "Inverse normalization resulted in non-finite values."
        recon = recon.to(in_dtype)
        if out is None:
            return recon
        out[finite] = recon
        return out

    def __repr__(self):
        return f"AsinhTonemapNormalizer(sigma={self.sigma})"

    def clone(self) -> "AsinhTonemapNormalizer":
        return AsinhTonemapNormalizer(sigma=self.sigma)

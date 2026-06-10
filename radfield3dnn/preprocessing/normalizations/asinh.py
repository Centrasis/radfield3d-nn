from torch import Tensor
import math
import torch
from typing import Union
from .base import Normalizer


class AsinhTonemapNormalizer(Normalizer):
    """Smooth bounded HDR tonemap: ``y = asinh(x / sigma) / asinh(1 / sigma)``.

    Designed for the per-field-normalised relative flux contract used by
    the experimental two-head PBRFNetCPP. After ``ChannelsSplitRelative``
    each flux head's target is in ``[0, 1]``. The tonemap is linear near
    zero (``y ≈ x / sigma``), logarithmic for ``x >> sigma``, smooth
    everywhere, and bounded to ``[0, 1]``; inversion is closed-form.

    Replaces the raw ``LogScaleNormalizer`` for the new stack. Two failure
    modes the old normaliser suffered from on DS03 (see architecture-
    change.md §3, run ``efmzkarx``):

    * ``-9`` zero sentinel is a learned discontinuity — there is no
      smooth target signal that crosses it.
    * Per-element error budget in raw log domain is ~9; with L1+SSIM the
      late-training Adam step can overflow fp16 weight updates and NaN
      the model (observed at epoch ~45 on ``efmzkarx``).

    asinh has neither problem: ``asinh(0) = 0`` exact (no sentinel), and
    the per-element error budget is bounded by 1 in any L1 / SSIM setup.

    Tonemap design: σ is the "noise floor" — values much smaller than σ
    map to ~0 (the network is asked to produce zero), values much larger
    than σ map logarithmically to [...→ 1]. Tune σ per channel from the
    per-field-normalised noise floor (DS03 picks: σ_scatter ≈ 3e-3,
    σ_direct ≈ 1e-3, see architecture-change.md §4 and the per-channel
    constructor classmethods below).

    Pair with PBRFNetCPP using::

        normalizer="asinh_scatter" (σ=3e-3) | "asinh_direct" (σ=1e-3)
        flux_activation="clamp", flux_clamp_min=0.0, flux_clamp_max=1.0,
        flux_offset=0.3
    """

    def __init__(self, sigma: float = 1e-3):
        super().__init__()
        assert sigma > 0.0, f"Require sigma > 0, got sigma={sigma}."
        self.sigma = float(sigma)
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


class SplitChannelAsinhNormalizer(Normalizer):
    """Per-channel asinh tonemap for the two-head PBRFNetCPP stack.

    Holds two independent ``AsinhTonemapNormalizer`` instances (scatter,
    direct) and dispatches by channel when applied to a ``RadiationField``.
    For ``RadiationFieldChannel`` / ``Tensor`` inputs (already joined or
    single-channel) it falls back to the scatter normaliser since that's
    the only sensible default; callers normalising a single tensor should
    pass the per-channel object explicitly.

    The per-field max normalisation contract (``ChannelsSplitRelative``)
    already maps each channel into ``[0, 1]``. The asinh tonemap then
    compresses each channel from its own physical noise floor (per-channel
    σ) up to 1.0, with the codomain bounded in ``[0, 1]``.

    Tunable from data via :meth:`from_dataset` (recommended) or
    constructed with hand-picked σ. Defaults are the DS03-empirically-tuned
    values (σ_scatter = 3e-3, σ_direct = 1e-3).
    """

    def __init__(self, scatter_sigma: float = 3e-3, direct_sigma: float = 1e-3):
        super().__init__()
        self.scatter = AsinhTonemapNormalizer(sigma=scatter_sigma)
        self.direct = AsinhTonemapNormalizer(sigma=direct_sigma)

    @property
    def scatter_sigma(self) -> float:
        return self.scatter.sigma

    @property
    def direct_sigma(self) -> float:
        return self.direct.sigma

    def get_type(self) -> str:
        return f"asinh_split_s{self.scatter.sigma:.0e}_d{self.direct.sigma:.0e}"

    def validate_range(self, x: Tensor):
        # Delegated to the scatter normaliser; both sub-normalisers share
        # the same validity rule (non-negative).
        self.scatter.validate_range(x)

    def forward(self, x, respect_to=None):
        """Override the base class dispatcher so RadiationField uses
        per-channel sigmas instead of the same normaliser on both
        sub-channels."""
        from radfield3dnn.rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, AirKermaField
        if isinstance(x, TrainingInputData):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth, respect_to.ground_truth if respect_to is not None else None),
                original_ground_truth=x.original_ground_truth,
            )
        if isinstance(x, RadiationField):
            return RadiationField(
                scatter_field=self.scatter.forward(x.scatter_field, respect_to=respect_to.scatter_field if respect_to is not None else None) if x.scatter_field is not None else None,
                direct_beam=self.direct.forward(x.direct_beam, respect_to=respect_to.direct_beam if respect_to is not None else None) if x.direct_beam is not None else None,
                geometry=x.geometry if hasattr(x, "geometry") else None,
            )
        # Tensor / RadiationFieldChannel / AirKermaField fall through to
        # the scatter normaliser. The two-head pipeline always supplies
        # RadiationField at the boundary; the fallback exists only so
        # diagnostic paths and inference helpers that join channels first
        # don't crash.
        return self.scatter.forward(x, respect_to=respect_to)

    def inverse(self, x, respect_to=None):
        from radfield3dnn.rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, AirKermaField
        if isinstance(x, TrainingInputData):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.inverse(x.ground_truth, respect_to.ground_truth if respect_to is not None else None),
                original_ground_truth=x.original_ground_truth,
            )
        if isinstance(x, RadiationField):
            return RadiationField(
                scatter_field=self.scatter.inverse(x.scatter_field, respect_to=respect_to.scatter_field if respect_to is not None else None) if x.scatter_field is not None else None,
                direct_beam=self.direct.inverse(x.direct_beam, respect_to=respect_to.direct_beam if respect_to is not None else None) if x.direct_beam is not None else None,
                geometry=x.geometry if hasattr(x, "geometry") else None,
            )
        return self.scatter.inverse(x, respect_to=respect_to)

    @classmethod
    def from_dataset(cls, dataset, max_fields: int = 50,
                     scatter_quantile: float = 0.10,
                     direct_quantile: float = 0.90,
                     min_sigma: float = 1e-6, max_sigma: float = 1e-1) -> "SplitChannelAsinhNormalizer":
        """Pick per-channel σ empirically from up to ``max_fields`` raw
        volumes drawn from ``dataset``.

        Rule:

        * For each volume, divide each channel by its own per-volume max
          (the same per-field relative normalisation
          ``ChannelsSplitRelative`` will apply at training time).
        * Aggregate non-zero per-volume-normalised values across all
          sampled volumes.
        * ``σ_channel = clamp(percentile(values, q_channel), min_sigma, max_sigma)``.

        Rationale (DS03-validated, see architecture-change.md §4.2):

        * **Scatter** is unimodal in log-space — the bottom decile is MC
          noise floor; ``q_scatter = 0.10`` places noise at y ≈ 0 in
          tonemap space and the median signal voxel near y ≈ 0.5.
        * **Direct beam** is bimodal — the bottom 90 % of nonzero voxels
          are MC leakage outside the cone, the top 1 % is the cone
          itself. ``q_direct = 0.90`` puts the noise floor at y ≈ 0 and
          gives the cone the full upper half of the tonemap codomain.

        ``dataset`` is expected to expose iterable batches whose items
        have either a ``ground_truth: RadiationField`` (training
        pipeline) or, equivalently, an attribute pair
        ``scatter_field.flux`` / ``direct_beam.flux``. The method
        tolerates either layout and silently skips items it cannot
        interpret.
        """
        import numpy as np
        sc_vals: list[np.ndarray] = []
        dr_vals: list[np.ndarray] = []

        def _flatten(t) -> "np.ndarray | None":
            try:
                arr = t.detach().cpu().to(torch.float32).numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
            except Exception:
                return None
            return arr.ravel()

        n_seen = 0
        for item in dataset:
            if n_seen >= max_fields:
                break
            gt = None
            if hasattr(item, "ground_truth"):
                gt = item.ground_truth
            elif isinstance(item, tuple) and len(item) >= 2 and hasattr(item[1], "scatter_field"):
                gt = item[1]
            elif hasattr(item, "scatter_field"):
                gt = item
            if gt is None or not hasattr(gt, "scatter_field") or not hasattr(gt, "direct_beam"):
                continue
            if gt.scatter_field is None or gt.direct_beam is None:
                continue
            sc = _flatten(gt.scatter_field.flux)
            dr = _flatten(gt.direct_beam.flux)
            if sc is None or dr is None:
                continue
            sc_max = float(sc.max()) if sc.size else 0.0
            dr_max = float(dr.max()) if dr.size else 0.0
            if sc_max > 0:
                sc_n = sc / sc_max
                sc_vals.append(sc_n[sc_n > 0])
            if dr_max > 0:
                dr_n = dr / dr_max
                dr_vals.append(dr_n[dr_n > 0])
            n_seen += 1

        if not sc_vals or not dr_vals:
            # Cannot compute — fall back to the empirically-tuned DS03 defaults.
            return cls()

        sc_all = np.concatenate(sc_vals)
        dr_all = np.concatenate(dr_vals)
        sigma_sc = float(np.clip(np.quantile(sc_all, scatter_quantile), min_sigma, max_sigma))
        sigma_dr = float(np.clip(np.quantile(dr_all, direct_quantile),  min_sigma, max_sigma))
        return cls(scatter_sigma=sigma_sc, direct_sigma=sigma_dr)

    def __repr__(self):
        return (f"SplitChannelAsinhNormalizer(scatter_sigma={self.scatter.sigma}, "
                f"direct_sigma={self.direct.sigma})")

    def clone(self) -> "SplitChannelAsinhNormalizer":
        return SplitChannelAsinhNormalizer(
            scatter_sigma=self.scatter.sigma,
            direct_sigma=self.direct.sigma,
        )

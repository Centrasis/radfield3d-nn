import math
from .base import Loss
from torch import Tensor, nn
from radfield3dnn.rftypes import TrainingInputData
import torch
from radfield3dnn.metrics.base import weight_field_by_statistical_error
from torch.nn import functional as F


class StdLossWeighted(Loss):
    def __init__(self, loss_fn: nn.Module, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__()
        self.weight_with_error = weight_with_error
        self.loss_fn = loss_fn
        self.scale = scale

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if len(prediction.shape) != len(target.shape):
            if len(prediction.shape) - 1 == len(target.shape) and target.shape[0] != 1:
                target = target.unsqueeze(0) if prediction.shape[0] == 1 else target.unsqueeze(-1)
            elif len(prediction.shape) == len(target.shape) - 1 and prediction.shape[0] != 1:
                prediction = prediction.unsqueeze(0) if target.shape[0] == 1 else prediction.unsqueeze(-1)
            else:
                raise ValueError(f"Prediction shape {prediction.shape} and target shape {target.shape} do not match.")

        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        all_valid = bool(valid_mask.all())
        if not all_valid:
            target = target.masked_fill(~valid_mask, 0.0)
            prediction = prediction.masked_fill(~valid_mask, 0.0)

        losses: Tensor = self.loss_fn(target=target, input=prediction)
        losses = torch.nan_to_num(losses, nan=1.0, posinf=1.0, neginf=1.0)

        if len(losses.shape) == 5:
            losses = torch.mean(losses, dim=1)
            if not all_valid and valid_mask.ndim == 5:
                valid_mask = valid_mask.all(dim=1)

        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        if self.scale != 1.0:
            losses = losses * self.scale

        if len(losses.shape) <= 1:
            return losses
        reduce_dims = [x for x in range(1, len(losses.shape))]
        if not all_valid and valid_mask.shape == losses.shape:
            valid_f = valid_mask.to(losses.dtype)
            denom = valid_f.sum(dim=reduce_dims).clamp(min=1.0)
            return (losses * valid_f).sum(dim=reduce_dims) / denom
        return torch.mean(losses, dim=reduce_dims)


class L1LossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(nn.L1Loss(reduction='none'), weight_with_error, scale)


class MagnitudeWeightedL1Loss(Loss):
    """L1 on **log-space** flux targets, weighted by physical flux magnitude.

    For an HDR field normalised by ``LogScaleNormalizer`` the targets live in
    log10 space (~[-9, 0]) and ~99% of voxels are near-zero background. A plain
    L1 then weights every voxel equally, so the rare high-flux beam voxels — the
    ones that dominate air-kerma — are drowned out and badly under-fit (the model
    learns the *shape* (high SSIM) but the peak magnitude collapses, e.g. predicts
    max flux 0.11 vs GT 0.97, wrecking the relative/air-kerma accuracy).

    This loss multiplies each voxel's log-space L1 by ``(10**target + c)**gamma``,
    i.e. by (a floor-shifted power of) its **physical** flux. Air-kerma is
    proportional to flux, so this makes the training objective emphasise exactly
    the voxels the accuracy metric cares about, while the floor ``c`` keeps the
    background represented (preserving structure). Intended for log-space targets
    only (pairs with ``normalizer="log_scale"``).
    """

    def __init__(self, c: float = 0.05, gamma: float = 1.0, weight_with_error: bool = False):
        super().__init__()
        self.c = float(c)
        self.gamma = float(gamma)
        self.weight_with_error = weight_with_error

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)

        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        all_valid = bool(valid_mask.all())
        if not all_valid:
            target = target.masked_fill(~valid_mask, 0.0)
            prediction = prediction.masked_fill(~valid_mask, 0.0)

        per_voxel = (prediction - target).abs()
        # Physical flux magnitude of the GT (target is log10 flux). The zero
        # sentinel (~-9) maps to ~1e-9 -> ~floor weight, which is what we want.
        with torch.no_grad():
            physical = torch.pow(torch.full_like(target, 10.0), target).clamp(0.0, 1.0)
            w = (physical + self.c) ** self.gamma
        losses = per_voxel * w
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)

        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)

        if losses.ndim <= 1:
            return losses
        reduce_dims = tuple(range(1, losses.ndim))
        wv = (w * valid_mask.to(w.dtype)) if not all_valid else w
        denom = wv.sum(dim=reduce_dims).clamp(min=1e-6)
        return losses.sum(dim=reduce_dims) / denom


class PhysicalSpaceL1Loss(Loss):
    """L1 measured in **physical flux space** for log-space-normalised targets.

    Air-kerma is proportional to *physical* flux, so the accuracy metric cares
    about **additive** error at the high-flux beam, not the multiplicative (log)
    error that a log-space L1 penalises. A 20% peak under-prediction is only
    ~0.1 in log-L1 but ~0.2 in physical-L1 — which is exactly why the old
    ``LinearNormalizer`` (whose normalised L1 *is* a physical-space L1) reached
    ~84% scatter air-kerma accuracy while ``log_scale`` + log-space L1 plateaued
    near ~65%.

    This loss keeps the network's stable **log-space output** but moves the
    *error measurement* into physical space: it exponentiates both prediction
    and target (``10**x``, the LogScaleNormalizer inverse, clamped to
    ``[0, 1]``) and takes their L1. A small log-space term (``beta``) is added
    back so the ~99% near-zero background — whose physical L1 gradient vanishes —
    still receives a structure-preserving signal.

        loss = |10**pred - 10**target| + beta * |pred - target|

    Pairs with ``normalizer="log_scale"``. ``beta`` trades peak air-kerma
    accuracy (low beta) against background structure/SSIM (higher beta).
    """

    def __init__(self, beta: float = 0.1, weight_with_error: bool = False):
        super().__init__()
        self.beta = float(beta)
        self.weight_with_error = weight_with_error

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)

        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        all_valid = bool(valid_mask.all())
        if not all_valid:
            target = target.masked_fill(~valid_mask, 0.0)
            prediction = prediction.masked_fill(~valid_mask, 0.0)

        ten = torch.full_like(target, 10.0)
        pred_phys = torch.pow(ten, prediction).clamp(0.0, 1.0)
        tgt_phys = torch.pow(ten, target).clamp(0.0, 1.0)
        per_voxel = (pred_phys - tgt_phys).abs() + self.beta * (prediction - target).abs()
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)

        if self.weight_with_error:
            per_voxel = weight_field_by_statistical_error(per_voxel, input=input)

        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        if not all_valid and valid_mask.shape == per_voxel.shape:
            vf = valid_mask.to(per_voxel.dtype)
            denom = vf.sum(dim=reduce_dims).clamp(min=1.0)
            return (per_voxel * vf).sum(dim=reduce_dims) / denom
        return per_voxel.mean(dim=reduce_dims)


class PlainL1Loss(Loss):
    """Plain L1 on the (already physical) normalised targets — no log, no weighting.

    For a field normalised by ``LinearNormalizer(0,1)`` the normalised value IS the
    physical flux (÷ per-field max), so a plain ``|pred − target|`` is a physical-space
    L1. The high-flux beam voxels carry the largest absolute errors and so dominate the
    objective *automatically*, which is exactly why the published ``LinearNormalizer``
    config reached ~84% scatter air-kerma accuracy. (Do NOT pair with ``log_scale`` —
    use ``L1Physical``/``L1MagWeighted`` there instead.)
    """

    def __init__(self, weight_with_error: bool = False):
        super().__init__()
        self.weight_with_error = weight_with_error

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        all_valid = bool(valid_mask.all())
        if not all_valid:
            target = target.masked_fill(~valid_mask, 0.0)
            prediction = prediction.masked_fill(~valid_mask, 0.0)
        per_voxel = (prediction - target).abs()
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight_with_error:
            per_voxel = weight_field_by_statistical_error(per_voxel, input=input)
        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        if not all_valid and valid_mask.shape == per_voxel.shape:
            vf = valid_mask.to(per_voxel.dtype)
            return (per_voxel * vf).sum(dim=reduce_dims) / vf.sum(dim=reduce_dims).clamp(min=1.0)
        return per_voxel.mean(dim=reduce_dims)


class PlainL2Loss(Loss):
    """Plain L2 (MSE) on the (already normalised) targets — no log, no weighting.

    The squared error weights the high-error voxels even harder than L1, so it is the most
    beam/peak-dominated of the plain cores in physical (linear) space, but in a tonemapped space
    (asinh / log) it spreads more evenly across the dynamic range. Used in the loss-effectiveness
    study as the ``L2_abs`` core (e.g. paired with the asinh normalizer).
    """

    def __init__(self, weight_with_error: bool = False):
        super().__init__()
        self.weight_with_error = weight_with_error

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        all_valid = bool(valid_mask.all())
        if not all_valid:
            target = target.masked_fill(~valid_mask, 0.0)
            prediction = prediction.masked_fill(~valid_mask, 0.0)
        per_voxel = (prediction - target) ** 2
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight_with_error:
            per_voxel = weight_field_by_statistical_error(per_voxel, input=input)
        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        if not all_valid and valid_mask.shape == per_voxel.shape:
            vf = valid_mask.to(per_voxel.dtype)
            return (per_voxel * vf).sum(dim=reduce_dims) / vf.sum(dim=reduce_dims).clamp(min=1.0)
        return per_voxel.mean(dim=reduce_dims)


class SMAPERegionBalancedLoss(Loss):
    """Metric-targeted, region-balanced SMAPE loss (the pipeline-audit P0 fix).

    Trains the SAME functional the evaluation scores (per-voxel SMAPE = 2|p−t|/(|p|+|t|+eps)) and
    rebalances the gradient across the three regions the metrics actually score, instead of letting
    voxel counts decide (DS03: bulk:ring:beam ≈ 92% : 3% : <0.1%):

      * beam  — t ≥ beam_rel · max(t)   (drives top90 + the GPR high-dose criterion)
      * ring  — ring_rel ≤ t < beam_rel (the legacy bright-ring scatter metric region)
      * bulk  — t < ring_rel AND statistically reliable (drives the noise-aware scatter metric)
      * noise — MC-noise voxels (joined error ≥ err_threshold) get a ONE-SIDED HINGE instead of a
        fit: zero cost while the prediction stays below hinge_rel · max(t), SMAPE-style cost above.
        Masked-out ≠ unconstrained: with no term at all the smooth MLP freely extrapolated radiation
        blobs into the noise region (observed: box-corner hallucinations, |rel err| ≈ 1.0 at the
        corners). The hinge pins the noise region down ("we don't know the exact value, but we know
        it is small") without fitting the MC noise itself.

    Each region's per-voxel cost is averaged separately and the region means are averaged, so every
    region receives the same total gradient mass per field. After ChannelsJoin the bulk error sits at
    ≈0.5 (reliable scatter + leakage direct) and true noise at ≈1.0 — hence the 0.75 default
    threshold. SMAPE is scale-invariant, so under LinearNormalizer(0,1) this optimizes physical
    relative accuracy directly — exactly what SMAPE-accuracy and the gamma pass-rate reward.
    """

    def __init__(self, eps: float = 1e-4, beam_rel: float = 5e-2, ring_rel: float = 5e-3,
                 err_threshold: float = 0.75, hinge_rel: float = None, weight_with_error: bool = False,
                 core: str = "smape"):
        super().__init__()
        # eps must sit AT/BELOW the reliable-bulk median (~6e-5 normalized), not above it: with
        # eps=1e-3 the bulk denominator was eps-dominated -> near-absolute (damped) treatment -> the
        # bulk level was never anchored and oscillated with the LR (observed: val_loss monotone down
        # while the bulk-dominated noise-aware scatter metric swung 0.19->0.08). Noise amplification
        # is already guarded by the reliability mask + hinge, so eps only needs to bound the deepest
        # reliable voxels (x10 at t=1e-5).
        self.eps = float(eps)
        # Per-voxel core:
        #   "smape"    — 2|p−t|/(|p|+|t|+eps): the metric's own functional, BUT saturates on the
        #                over-prediction side (grad ≈ 4t/(p+t)² → ~4e-4 for a ghost at p=0.3, t=1e-4)
        #                so misplaced/rotated bright structure is nearly FREE to keep. Observed at
        #                ep153: bright-region IoU 0.29, 63% of predicted beam mass outside the GT
        #                beam (ghost/rotated beams).
        #   "logratio" — |log10((p+eps)/(t+eps))| clamped to 6 decades: non-saturating relative
        #                error; gradient ~ 1/(p+eps) pushes ghosts down at full strength (~1.4 at
        #                p=0.3 — ~3500x SMAPE's) while keeping strong under-prediction gradients.
        #                Monotone in relative error, so it still optimizes the SMAPE metrics.
        assert core in ("smape", "logratio"), f"Unknown core {core!r}"
        self.core = core
        self.beam_rel = float(beam_rel)
        self.ring_rel = float(ring_rel)
        self.err_threshold = float(err_threshold)
        # Hinge threshold for the noise region (relative to per-field max). Defaults to ring_rel:
        # a noise voxel predicting above the bright-ring threshold is unambiguously wrong.
        self.hinge_rel = float(hinge_rel) if hinge_rel is not None else float(ring_rel)
        self.weight_with_error = weight_with_error  # kept for interface parity; masking supersedes it

    def _gt_error(self, input: TrainingInputData, like: Tensor):
        gt = getattr(input, "ground_truth", None) if input is not None else None
        err = None
        if gt is not None:
            if hasattr(gt, "scatter_field") and gt.scatter_field is not None and getattr(gt.scatter_field, "error", None) is not None:
                err = gt.scatter_field.error
                if hasattr(gt, "direct_beam") and gt.direct_beam is not None and getattr(gt.direct_beam, "error", None) is not None:
                    err = (err + gt.direct_beam.error) / 2.0
            elif getattr(gt, "error", None) is not None:
                err = gt.error
        if err is None:
            return None
        if err.shape != like.shape and err.numel() == like.numel():
            err = err.reshape(like.shape)
        elif err.shape != like.shape:
            return None
        return err

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        t = target.masked_fill(~valid, 0.0)
        p = prediction.masked_fill(~valid, 0.0)

        if self.core == "logratio":
            # |log10 ratio|, clamped at 6 decades (NaN guard only; never active from the median-bias
            # init, which starts <4 decades from the beam peak).
            smape = (torch.log10(p.abs() + self.eps) - torch.log10(t.abs() + self.eps)).abs().clamp(max=6.0)
        else:
            smape = (2.0 * (p - t).abs()) / (p.abs() + t.abs() + self.eps)   # per-voxel metric core, [0,2]
        smape = torch.nan_to_num(smape, nan=0.0, posinf=0.0, neginf=0.0)

        if smape.ndim <= 1:                      # row mode (no spatial structure): plain mean
            return smape

        reduce_dims = tuple(range(1, smape.ndim))
        tmax = t.amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
        rel = t / tmax

        err = self._gt_error(input, t)
        reliable = (err < self.err_threshold) if err is not None else torch.ones_like(valid)

        beam = valid & (rel >= self.beam_rel)
        ring = valid & (rel >= self.ring_rel) & (rel < self.beam_rel)
        bulk = valid & (rel < self.ring_rel) & reliable
        noise = valid & (rel < self.ring_rel) & ~reliable

        # One-sided hinge for the noise region: free below tau = hinge_rel * max(t), cost above.
        # Prevents the "masked = unconstrained" corner hallucination while still not fitting the MC
        # noise values themselves. Core-matched: log-ratio hinge under the logratio core (gradient
        # ~1/p — full-strength ghost suppression), SMAPE-style otherwise.
        tau = self.hinge_rel * tmax
        if self.core == "logratio":
            hinge = (torch.log10(p.abs() + self.eps) - torch.log10(tau + self.eps)).clamp(min=0.0, max=6.0)
        else:
            over = (p - tau).clamp(min=0.0)
            hinge = (2.0 * over) / (p.abs() + tau + self.eps)
        hinge = torch.nan_to_num(hinge, nan=0.0, posinf=0.0, neginf=0.0)

        def region_mean(values, mask):
            cnt = mask.to(values.dtype).sum(dim=reduce_dims)
            s = (values * mask.to(values.dtype)).sum(dim=reduce_dims)
            return s / cnt.clamp(min=1.0), (cnt > 0)

        means, present = zip(*(region_mean(v, m) for v, m in
                               ((smape, beam), (smape, ring), (smape, bulk), (hinge, noise))))
        means = torch.stack(means, dim=0)                    # [4, B]
        present = torch.stack(present, dim=0).to(smape.dtype)
        # average over the regions PRESENT in each field (equal gradient mass per present region)
        return (means * present).sum(dim=0) / present.sum(dim=0).clamp(min=1.0)


class MuLawL2Loss(Loss):
    """μ-law tone-mapped L2 — the standard HDR-reconstruction training loss (Kalantari &
    Ramamoorthi, SIGGRAPH 2017, "Deep High Dynamic Range Imaging of Dynamic Scenes"; used across
    single-image HDR reconstruction, e.g. ExpandNet, HDR-GAN):

        L = ( T(p) − T(t) )²   with   T(x) = log(1 + μ·x) / log(1 + μ)

    One smooth formula, NO branches, NO masks, NO per-field statistics — differentiable everywhere.
    T compresses the HDR range so every decade contributes (T linear near 0, log above 1/μ), and the
    gradient  ∂L/∂p ∝ μ/(1+μ·p)  is non-saturating for over-prediction (≈1/p — ghosts/rotated beams
    are pushed down at full strength) and bounded by μ at p→0 (no explosion). μ=5000 spreads the
    DS03 normalized range [1e-4, 1] over T ∈ [0.05, 1]."""

    def __init__(self, mu: float = 5000.0, weight_with_error: bool = False):
        super().__init__()
        self.mu = float(mu)
        self.weight_with_error = weight_with_error

    def _tonemap(self, x: Tensor) -> Tensor:
        return torch.log1p(self.mu * x.clamp_min(0.0)) / math.log1p(self.mu)

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        t = target.masked_fill(~valid, 0.0)
        p = prediction.masked_fill(~valid, 0.0)
        per_voxel = (self._tonemap(p) - self._tonemap(t)) ** 2
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        vf = valid.to(per_voxel.dtype)
        return (per_voxel * vf).sum(dim=reduce_dims) / vf.sum(dim=reduce_dims).clamp(min=1.0)


class RawNeRFLoss(Loss):
    """RawNeRF HDR loss (Mildenhall et al., CVPR 2022, "NeRF in the Dark"): linear-space L2 weighted
    by a stop-gradient relative factor, ``((p − t) / (sg(p) + eps))²``. Behaves like a relative error
    (every decade gets gradient) while staying in LINEAR space — the proven recipe for training
    multi-decade HDR radiance without a log transform and without the log-recipe peak underfit.
    Pair with LinearNormalizer(0,1); eps in normalized units."""

    def __init__(self, eps: float = 1e-3, weight_with_error: bool = False):
        super().__init__()
        self.eps = float(eps)
        self.weight_with_error = weight_with_error

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        # -inf = masked / not-sampled. Excluded from the loss AND the reduction (vf below), so a
        # masked voxel never contributes — robust to the MCFloorCut / ROIbasedSampler sentinel.
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        t = target.masked_fill(~valid, 0.0)
        p = prediction.masked_fill(~valid, 0.0)
        # Self-normalizing HDR weight (Mildenhall et al.) is ``1/(sg(p)+eps)`` — but at p→0 with a
        # nonzero target it blows up to (t/eps)² (≈700 on DS03, unbounded as eps→0), which the
        # study diagnostics flagged (loss(zero-pred)=681, |grad|=16) and which destabilised DB-MTL
        # and the rawnerfsharp run. FIX: floor the denominator by the (detached) TARGET scale,
        # ``1/(max(sg(p), |t|)+eps)``. This keeps the cross-decade relative weighting (when p≈t,
        # w≈1/t) but BOUNDS every term to ≤1: under-prediction p→0 → ((0−t)/(|t|+eps))²≤1,
        # over-prediction p≫t → ((p−t)/p)²≤1. The weight stays a pure stop-gradient factor.
        scale = torch.maximum(p.detach().abs(), t.detach().abs())
        w = 1.0 / (scale + self.eps)
        per_voxel = ((p - t) * w) ** 2
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        vf = valid.to(per_voxel.dtype)
        return (per_voxel * vf).sum(dim=reduce_dims) / vf.sum(dim=reduce_dims).clamp(min=1.0)


class RawNeRFSharpLoss(RawNeRFLoss):
    """RawNeRF + α·plain-L1 — the beam-sharpening hybrid (reconstruction-eval finding, 2026-06-11).

    The deployed-model eval measured RawNeRF's failure mode: the beam is PLACED correctly but stays
    blurred (high-dose region 10–34× too large, peak 0.05–0.2× as sharp) because the self-weight
    ``1/(sg(p)+ε)²`` anneals ∝p⁻² as the beam forms — the refinement gradient vanishes exactly where
    sharpening is needed. Plain absolute L1 has the complementary profile (the published sharp-beam
    recipe): its per-voxel gradient is ±1 forever, the near-fit bulk's sign-gradients CANCEL while
    the beam's coherent error survives — so it keeps polishing the beam to convergence, but alone it
    starves the scatter field. The sum keeps both: RawNeRF trains every decade + erases ghosts;
    α·L1 restores the non-annealing beam-sharpening pressure.

    α=10 puts the L1 term's beam gradient at parity with RawNeRF's at the measured converged-blur
    state (|p−t|≈0.5, p≈0.3 → RawNeRF grad ≈ 11/N vs α·L1 ≈ 10/N) and lets L1 dominate the beam as
    RawNeRF anneals further. One smooth formula, no branches, fully differentiable.
    """

    def __init__(self, eps: float = 1e-3, alpha: float = 10.0, weight_with_error: bool = False):
        super().__init__(eps=eps, weight_with_error=weight_with_error)
        self.alpha = float(alpha)
        self._l1 = PlainL1Loss(weight_with_error=False)

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        return super().forward(target, prediction, input) \
            + self.alpha * self._l1.forward(target, prediction, input)


class ChannelMaxBalancedLoss(Loss):
    """Wrap a per-channel flux loss so its GRADIENT behaves as if the channel were individually
    (per-field-max) normalised — *without* normalising the data, so the model still predicts in the
    shared physical scale and the scatter:direct magnitude relation is preserved at the output.

    Mechanism (the implicit relation-preservation approach, validated in
    ``tests/test_split_loss_weighting.py``): divide both prediction and target by a **detached**
    per-field max of the target before the base loss. For any homogeneous loss this multiplies the
    raw-space gradient by 1/max — exactly the gradient individual normalisation would produce —
    while the prediction the model emits stays raw (so summing the two heads recombines correctly).
    The tiny scatter channel (small in the shared scale) thus gets the same gradient footing as the
    large direct channel, which is the whole reason per-channel normalisation existed.
    """

    def __init__(self, base_loss: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.base_loss = base_loss
        self.eps = eps

    def _per_field_scale(self, target: Tensor) -> Tensor:
        finite = torch.isfinite(target)
        tf = torch.where(finite, target, torch.zeros_like(target))
        dims = tuple(range(1, target.ndim)) if target.ndim > 1 else (0,)
        return tf.amax(dim=dims).clamp(min=self.eps).detach()   # [B] (or scalar)

    @staticmethod
    def _scale(t: Tensor, s_flat: Tensor) -> Tensor:
        if t.ndim >= 1 and s_flat.ndim == 1 and t.shape[0] == s_flat.shape[0]:
            return t / s_flat.view((s_flat.shape[0],) + (1,) * (t.ndim - 1))
        return t / s_flat

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        s = self._per_field_scale(target)
        return self.base_loss.forward(target=self._scale(target, s),
                                      prediction=self._scale(prediction, s), input=input)


class WassersteinLossWeighted(StdLossWeighted):
    def __init__(self, dim: int = -1, weight_with_error: bool = False):
        super().__init__(None, weight_with_error)
        self.dim = dim

    def forward(self, target, prediction, input, dim: int | None = None):
        d = self.dim if dim is None else dim
        wasserstein = torch.abs(torch.cumsum(prediction, dim=d) - torch.cumsum(target, dim=d))
        losses = torch.mean(wasserstein, dim=d)

        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)

        return torch.mean(losses)


class StructuralSimilarity3DLoss(StdLossWeighted):
    """3D SSIM loss for volumetric fields (B, C, D, H, W). Returns 1 - SSIM."""

    def __init__(
        self,
        window_size: int = 7,
        sigma: float = 1.5,
        data_range: float | None = None,
        C1: float | None = None,
        C2: float | None = None,
        channel_average: bool = True,
        size_average: bool = True,
        clamp_ssim: bool = True,
        weight_with_error: bool = False
    ):
        super().__init__(loss_fn=None, weight_with_error=weight_with_error)
        assert window_size % 2 == 1, "window_size must be odd."
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        if self.data_range is not None:
            self.C1 = C1 if C1 is not None else (0.01 * self.data_range) ** 2
            self.C2 = C2 if C2 is not None else (0.03 * self.data_range) ** 2
        else:
            self.C1 = C1
            self.C2 = C2
        self.channel_average = channel_average
        self.size_average = size_average
        self.clamp_ssim = clamp_ssim
        self._kernel_cache = {}

    @staticmethod
    def _create_gaussian_kernel3d(window_size: int, sigma: float, channels: int, device, dtype):
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel_3d = kernel_3d / kernel_3d.sum()
        kernel_3d = kernel_3d.view(1, 1, window_size, window_size, window_size)
        return kernel_3d.repeat(channels, 1, 1, 1, 1)

    def _get_kernel(self, channels: int, device, dtype):
        key = (channels, device, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._create_gaussian_kernel3d(
                self.window_size, self.sigma, channels, device, dtype
            )
        return self._kernel_cache[key]

    def _ssim_3d(self, x: Tensor, y: Tensor, C1: Tensor, C2: Tensor, mask: Tensor | None = None) -> Tensor:
        B, C, D, H, W = x.shape
        if D < self.window_size or H < self.window_size or W < self.window_size:
            return 1.0 - torch.mean(torch.abs(x - y))

        kernel = self._get_kernel(C, x.device, x.dtype)
        padding = self.window_size // 2

        def conv(v):
            return F.conv3d(v, kernel, groups=C, padding=padding)

        mu_x = conv(x)
        mu_y = conv(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = conv(x * x) - mu_x2
        sigma_y2 = conv(y * y) - mu_y2
        sigma_xy = conv(x * y) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = numerator / (denominator + 1e-12)

        if self.clamp_ssim:
            ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)

        if mask is not None:
            if mask.shape != x.shape:
                if mask.shape == (B, 1, D, H, W):
                    mask = mask.repeat(1, C, 1, 1, 1)
                else:
                    raise ValueError("Mask shape inconsistent.")
            ssim_map = ssim_map * mask
            denom = mask.sum().clamp(min=1.0)
        else:
            denom = torch.tensor(ssim_map.numel(), device=ssim_map.device, dtype=ssim_map.dtype)

        if self.channel_average:
            ssim_val = ssim_map.view(B, C, -1).sum(-1) / (denom / (B * C))
            ssim_val = ssim_val.mean() if self.size_average else ssim_val
        else:
            ssim_val = ssim_map.sum() / denom

        return ssim_val

    def _compute_dynamic_constants(self, target: Tensor, prediction: Tensor, valid_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if valid_mask is not None:
            vb = valid_mask > 0.5
            if vb.any():
                vals = torch.cat([prediction[vb], target[vb]], dim=0)
            else:
                vals = torch.tensor([0.0, 1.0], device=prediction.device, dtype=prediction.dtype)
        else:
            vals = torch.cat([prediction.reshape(-1), target.reshape(-1)], dim=0)

        data_range = torch.clamp(vals.max() - vals.min(), min=1e-12)
        C1_val = self.C1 if self.C1 is not None else (0.01 * data_range) ** 2
        C2_val = self.C2 if self.C2 is not None else (0.03 * data_range) ** 2
        return (
            torch.as_tensor(C1_val, device=prediction.device, dtype=prediction.dtype),
            torch.as_tensor(C2_val, device=prediction.device, dtype=prediction.dtype),
        )

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        assert prediction.shape == target.shape, f"Shape mismatch {prediction.shape} vs {target.shape}"

        neginf_mask = torch.isneginf(prediction) | torch.isneginf(target)
        if neginf_mask.any():
            min_non_masked = torch.min(target[~neginf_mask]) if (~neginf_mask).any() else torch.tensor(0.0, device=target.device, dtype=target.dtype)
            prediction = prediction.masked_fill(neginf_mask, min_non_masked)
            target = target.masked_fill(neginf_mask, min_non_masked)

        valid_mask = (~neginf_mask).float() if neginf_mask.any() else None

        if self.data_range is None:
            C1_t, C2_t = self._compute_dynamic_constants(target, prediction, valid_mask)
        else:
            C1_t = torch.as_tensor(self.C1, device=prediction.device, dtype=prediction.dtype)
            C2_t = torch.as_tensor(self.C2, device=prediction.device, dtype=prediction.dtype)

        ssim_val = self._ssim_3d(prediction, target, C1=C1_t, C2=C2_t, mask=valid_mask)
        return 1.0 - ssim_val


class RelativeWeighting(nn.Module):
    def __init__(self, eps: float = 1e-6, max_ratio: float = 100.0):
        super().__init__()
        self.eps = eps
        self.max_ratio = max_ratio

    def forward(self, target: Tensor, prediction: Tensor) -> Tensor:
        rel = (prediction - target).abs() / (target.abs() + self.eps)
        return rel.clamp(max=self.max_ratio).mean()


class FluxLoss(StdLossWeighted):
    """Huber-like flux loss combining per-voxel L1+L2 with (optional) 3D SSIM.

    ``ssim_weight`` is the convex weight on the structural term; the per-voxel
    core gets ``1 - ssim_weight``. Set ``ssim_weight=0.0`` to drop SSIM
    entirely (the conv is then never computed). This is the correct setting
    whenever the target volume is sparsified by voxel-dropout (the
    ``ErrorbasedImportanceSampler``): random holes destroy the local
    neighbourhood SSIM relies on, so a purely point-wise core is the right
    pairing. Combine with ``log_scale=True`` for HDR flux fields.
    """

    def __init__(self, weight_with_error: bool = False, log_scale: bool = False,
                 relative_weighting: bool = False, focal_r: bool = False,
                 focal_r_beta: float = 20.0, focal_r_gamma: float = 1.0,
                 ssim_weight: float = 0.34):
        super().__init__(None, weight_with_error)
        self.ssim_weight = float(ssim_weight)
        self.ssim = StructuralSimilarity3DLoss(weight_with_error=False) if self.ssim_weight > 0.0 else None
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.log_scale = log_scale
        self.relative_weighting = relative_weighting
        self.rel_weight = RelativeWeighting() if relative_weighting else None
        self.focal_r = focal_r
        self.focal_r_beta = float(focal_r_beta)
        self.focal_r_gamma = float(focal_r_gamma)

    def _focal_r_weight(self, err: Tensor) -> Tensor:
        with torch.no_grad():
            w = torch.sigmoid(self.focal_r_beta * err.abs()) ** self.focal_r_gamma
        return w

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        # SSIM is skipped entirely when ssim_weight == 0 (e.g. under voxel-dropout
        # importance sampling, where the structural term is meaningless).
        ssim = self.ssim(target, prediction, input) if self.ssim is not None \
            else torch.zeros((), device=target.device, dtype=target.dtype)

        invalid_mask = ~(torch.isfinite(target) & torch.isfinite(prediction))
        if invalid_mask.any():
            target = target.masked_fill(invalid_mask, 0.0)
            prediction = prediction.masked_fill(invalid_mask, 0.0)

        if self.log_scale:
            if target.min() < 0.0 and target.max() > 0.0:
                target = target + 1.0
                prediction = prediction + 1.0
            target = torch.log1p(target)
            prediction = torch.log1p(prediction)

        err = prediction - target
        if self.rel_weight is not None:
            per_voxel = err.abs() / (target.abs() + self.rel_weight.eps)
            per_voxel = per_voxel.clamp(max=self.rel_weight.max_ratio)
        else:
            l1 = self.l1_loss(target, prediction)
            l2 = self.mse_loss(target, prediction)
            per_voxel = 0.5 * (l1 + l2)

        if self.focal_r:
            per_voxel = per_voxel * self._focal_r_weight(err)

        if invalid_mask.any():
            valid = (~invalid_mask).to(per_voxel.dtype)
            reduce_dims = tuple(range(1, per_voxel.ndim))
            denom = valid.sum(dim=reduce_dims).clamp(min=1.0)
            core = (per_voxel * valid).sum(dim=reduce_dims) / denom
        else:
            core = per_voxel.mean(dim=tuple(range(1, per_voxel.ndim))) if per_voxel.ndim > 1 \
                else per_voxel.mean()
        return core * (1.0 - self.ssim_weight) + ssim * self.ssim_weight


class L1WithSSIM3DLoss(StdLossWeighted):
    """α·L1 + (1−α)·SSIM3D — safe for log-space output domains."""

    def __init__(self, weight_with_error: bool = False,
                 l1_weight: float = 0.66, ssim_weight: float = 0.34):
        super().__init__(None, weight_with_error)
        self.ssim = StructuralSimilarity3DLoss(weight_with_error=False)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)

    @staticmethod
    def _is_volumetric(t: Tensor) -> bool:
        return t.ndim == 5

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        use_ssim = self._is_volumetric(target) and self._is_volumetric(prediction)
        ssim = self.ssim(target, prediction, input) if use_ssim else \
            torch.zeros((), device=target.device, dtype=target.dtype)

        invalid_mask = ~(torch.isfinite(target) & torch.isfinite(prediction))
        if invalid_mask.any():
            target = target.masked_fill(invalid_mask, 0.0)
            prediction = prediction.masked_fill(invalid_mask, 0.0)

        per_voxel = self.l1_loss(target, prediction)

        if invalid_mask.any():
            valid = (~invalid_mask).to(per_voxel.dtype)
            reduce_dims = tuple(range(1, per_voxel.ndim)) if per_voxel.ndim > 1 else (0,)
            denom = valid.sum(dim=reduce_dims).clamp(min=1.0)
            core = (per_voxel * valid).sum(dim=reduce_dims) / denom
        else:
            core = per_voxel.mean(dim=tuple(range(1, per_voxel.ndim))) \
                if per_voxel.ndim > 1 else per_voxel.mean()

        return core * self.l1_weight + ssim * self.ssim_weight


class HotspotAwareFluxLoss(Loss):
    """Composite loss for HDR flux volumes where ~90% of voxels are low
    (1e-2..1e-4) but sparse high voxels must be reconstructed accurately.

    Term 1: power-weighted relative L2 (prediction-normalized, stop-grad):
            (pred-target)^2 / (sg(pred)^(2-beta) + eps)
            beta=0   -> NRC relative-L2 (per-decade equal; low voxels dominate
                        when they are 90% of the volume)
            beta=2   -> plain L2 (hotspots dominate)
            beta~0.5-1 -> biased toward high voxels but background still learns
            eps      -> regime boundary: below ~eps^(1/(2-beta)) errors count
                        absolutely. Set near max(MC noise floor, 1e-2 normalized),
                        NOT an arbitrary tiny number.
    Term 2: optional hotspot L1 on voxels above a target-percentile threshold
            (pins maxima; analogous to Dmax/D1% in dosimetry QA).

    Inputs assumed in linear space, normalized so max/median is O(1)."""
    def __init__(self, beta=0.75, eps=1e-2,
                 hotspot_quantile=0.995, hotspot_weight=0.1):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.hq, self.hw = hotspot_quantile, hotspot_weight

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        denom = prediction.detach().clamp_min(0).pow(2.0 - self.beta) + self.eps
        loss = ((prediction - target) ** 2 / denom).mean()
        if self.hw > 0:
            thr = torch.quantile(target.detach().flatten().float(), self.hq)
            mask = target >= thr
            if mask.any():
                loss = loss + self.hw * (prediction[mask] - target[mask]).abs().mean()
        return loss


class TwoROIGammaLoss(nn.Module):
    """Loss over the shared beam / scatter / floor ROIs (radfield3dnn.roi.compute_roi_masks),
    matched 1:1 to the air-kerma scatter metric and the ROI voxel sampler.

    The three ROIs are derived FROM THE TARGET (the joined flux the model is trained on; in the
    beam, joined ≈ direct so the beam ROI agrees with the metric's direct-channel definition):
      beam    = target >= beam_rel  * max(target)                  (≈0.55% of voxels at 0.05)
      scatter = NOT beam AND target >= scatter_lo * max(target)    (≈80% at 5e-5; SMAPE surrogate)
      floor   = the rest (the MC-noise floor; light smoothness glue)

    Each term is the MEAN over its ROI's voxels, i.e. normalised by the number of TARGET voxels in
    that ROI — so a ~700-voxel beam and an ~80000-voxel scatter region contribute EQUALLY to the
    total (per-ROI influence equalised by the target voxel count). Non-sampled voxels carry -inf
    (the ROI-sampler / masking convention shared with the other losses) and are excluded everywhere.

    beam term = L1 (absolute, matches the beam air-kerma SMAPE); scatter/floor = bounded symmetric
    SMAPE |p−t|/(|p|+|t|+eps) (≤1 per voxel — the metric's own functional; v2, see __init__ note).
    Optional 1-voxel-DTA soft gamma (w_gamma>0, stage3). Expects pred/target as full volumes
    (B, X, Y, Z) in linear, normalised units (needed for the spatial shifts).
    """
    def __init__(self, beam_rel=0.05, scatter_lo=5e-5,
                 w_beam=1.0, w_scatter=1.0, w_floor=0.05, w_gamma=0.0,
                 gamma_crit_beam=0.03, gamma_crit_scatter=0.10,
                 eps_scatter=5e-5, softmin_tau=0.1):
        # v2 core (2026-06-12, after concat-tworoi v1 hard-stuck at val scatter ~0.05 with the
        # flux loss oscillating 5..125): the v1 scatter/floor term |p-t|/(sg(p)+1e-5) was
        # UNBOUNDED in value (t/eps ≈ 5e3 per voxel when under-predicting) with a constant 1/eps
        # gradient on the whole under-prediction side and a weak over-prediction side → violent
        # up-pumping then drift, no convergence. v2 uses the bounded symmetric SMAPE core
        # |p-t|/(p+t+eps) (≤1 per voxel — the metric's own functional), eps_scatter at the ROI
        # floor (5e-5 ≈ scatter_lo·max in normalised units), and w_floor back to 0.05 (the floor
        # is ~60% MC noise; v1's 1.0 forced noise-fitting at full weight).
        super().__init__()
        self.beam_rel, self.scatter_lo = beam_rel, scatter_lo
        self.w = dict(beam=w_beam, scatter=w_scatter, floor=w_floor, gamma=w_gamma)
        self.cb, self.cs = gamma_crit_beam, gamma_crit_scatter
        self.eps, self.tau = eps_scatter, softmin_tau
        self.stage3 = False

    @staticmethod
    def _shifted_stack(x):
        """All 27 shifts within ±1 voxel (includes identity). x: (B,X,Y,Z)."""
        xp = F.pad(x.unsqueeze(1), (1,1,1,1,1,1), mode='replicate').squeeze(1)
        return torch.stack([xp[:, 1+dx:xp.shape[1]-1+dx,
                                  1+dy:xp.shape[2]-1+dy,
                                  1+dz:xp.shape[3]-1+dz]
                            for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)],
                           dim=0)                                   # (27,B,X,Y,Z)

    def _soft_gamma(self, pred, target, mask, crit, ref):
        """Soft 1-voxel-DTA gamma: per ref voxel, soft-min over shifted
        predictions of |pred_shift - target| / (crit*ref); hinge at 1."""
        if not mask.any():
            return pred.new_zeros(())
        shifts = self._shifted_stack(pred)                          # (27,B,...)
        g = (shifts - target.unsqueeze(0)).abs() / (crit * ref + 1e-12)
        g = -self.tau * torch.logsumexp(-g / self.tau, dim=0)       # soft-min
        return F.relu(g[mask] - 1.0).mean()                         # only failures

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        from radfield3dnn.roi import compute_roi_masks
        finite = torch.isfinite(target)   # -inf = not-sampled / masked -> excluded from every ROI
        # ROIs FROM THE TARGET (joined flux). Pass it as both 'direct' and 'joined': the joined-max
        # beam ≈ the metric's direct-channel beam (direct dominates the joined beam). Zero-fill the
        # -inf voxels so the per-field max stays finite; they are re-excluded via `finite` below.
        safe = torch.where(finite, target, torch.zeros_like(target))
        m_beam, m_scatter, m_floor = compute_roi_masks(safe, safe, self.beam_rel, self.scatter_lo)
        m_beam, m_scatter, m_floor = m_beam & finite, m_scatter & finite, m_floor & finite

        loss = prediction.new_zeros(())
        # Per-term breakdown for the training debug probe (callbacks/debug_probe.py); floats only.
        self.last_terms: dict[str, float] = {}
        if self.w['beam'] > 0 and m_beam.any():
            # absolute L1 — matches the beam air-kerma SMAPE
            term = (prediction[m_beam] - target[m_beam]).abs().mean()
            self.last_terms['beam'] = float(term.detach())
            self.last_terms['n_beam'] = int(m_beam.sum())
            loss = loss + self.w['beam'] * term
        for name, m in (('scatter', m_scatter), ('floor', m_floor)):
            if self.w[name] > 0 and m.any():
                # bounded symmetric SMAPE core (≤1 per voxel) — the metric's own functional;
                # differentiable through BOTH p and t paths, no unbounded under-prediction value.
                p, t = prediction[m], target[m]
                rel = (p - t).abs() / (p.abs() + t.abs() + self.eps)
                # .mean() normalises by THIS ROI's target voxel count -> equal per-ROI influence
                term = rel.mean()
                self.last_terms[name] = float(term.detach())
                self.last_terms[f'n_{name}'] = int(m.sum())
                loss = loss + self.w[name] * term
        if self.stage3 and self.w['gamma'] > 0:
            tmax = safe.detach().amax()
            loss = loss + self.w['gamma'] * (
                self._soft_gamma(prediction, target, m_beam,    self.cb, tmax)
              + self._soft_gamma(prediction, target, m_scatter, self.cs,
                                 target.detach().clamp_min(self.scatter_lo)))
        return loss

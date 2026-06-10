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

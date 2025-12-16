from .base import Loss
from torch import Tensor, nn
from radfield3dnn import TrainingInputData
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
        # get a mask from all values that are -inf
        mask = torch.isinf(prediction) & (prediction < 0)
        prediction = prediction.masked_fill(mask, 0.0)
        target = target.masked_fill(mask, 0.0)

        losses: Tensor = self.loss_fn(target=target, input=prediction)
        losses = torch.nan_to_num(losses, nan=1.0, posinf=1.0, neginf=1.0)
        if len(losses.shape) == 5:
            losses = torch.mean(losses, dim=1)
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        if self.scale != 1.0:
            losses = losses * self.scale
        return torch.mean(losses, dim=[x for x in range(1, len(losses.shape))])


class L1LossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(nn.L1Loss(reduction='none'), weight_with_error, scale)


class L1LossLogSpace(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(None, weight_with_error, scale)
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        losses = self.l1(torch.log(target + 1e-8), torch.log(prediction + 1e-8))
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        return torch.mean(losses)


class PoissonNLLLoss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(None, weight_with_error, scale)
        self.nll = nn.PoissonNLLLoss(reduction='none')

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        losses = self.nll(torch.log(prediction + 1e-8), target)
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        return losses.mean(dim=tuple(range(1, len(losses.shape))))


class ZeroInflatedMSELoss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0, eps: float = 1e-9):
        super().__init__(None, weight_with_error, scale)
        self.eps = eps

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        is_zero = (target <= self.eps).float()
        is_pos  = 1.0 - is_zero

        loss_zero = (prediction ** 2) * is_zero

        # Positiv-Teil: MSE
        loss_pos = ((prediction - target) ** 2) * is_pos

        loss = loss_zero + loss_pos

        if self.weight_with_error:
            loss = weight_field_by_statistical_error(loss, input=input)

        return torch.mean(loss)
    

class ZeroInflatedPoissonLoss(StdLossWeighted):
    """
    Zero-Inflated Poisson loss for 5D tensors (B, C, D, W, H).

    Prediction:
        prediction[:, 0, ...] -> logit for structural zero probability p0
        prediction[:, 1, ...] -> raw rate (will be softplus'ed to ensure λ > 0)

    Target:
        target shape (B, 1, D, W, H), normalized in [0, 1].

    Optional scaling to convert normalized target to approximate integer counts:
        If count_scale is not None: counts = round(target * count_scale)
        Else: use target * rate_scale as (possibly fractional) expected counts.

    Args:
        weight_with_error: apply statistical error weighting
        scale: global multiplicative factor on the final loss
        count_scale: (int/float|None) if provided, convert normalized target to counts
        rate_scale: (float) multiplicative factor applied to target before treating as counts if count_scale is None
        zero_threshold: values <= threshold are treated as zeros
        eps: numerical stability constant
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        weight_with_error: bool = False,
        scale: float = 1.0,
        count_scale: float | None = None,
        rate_scale: float = 1.0,
        zero_threshold: float = 1e-8,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__(loss_fn=None, weight_with_error=weight_with_error, scale=scale)
        self.count_scale = count_scale
        self.rate_scale = rate_scale
        self.zero_threshold = zero_threshold
        self.eps = eps
        self.reduction = reduction

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        """
        target: (B, 1, D, W, H)
        prediction: (B, 2, D, W, H)
        """
        if prediction.shape[1] != 2:
            raise ValueError(f"Expected prediction with 2 channels (logit_p0, rate_raw), got {prediction.shape}")

        # Broadcast/shape checks
        if target.shape[1] != 1:
            raise ValueError(f"Expected target channel dimension == 1, got {target.shape}")

        logit_p0 = prediction[:, 0:1, ...]        # (B,1,D,W,H)
        rate_raw = prediction[:, 1:2, ...]        # (B,1,D,W,H)

        # Structural zero probability
        p0 = torch.sigmoid(logit_p0)              # in (0,1)

        # Positive rate λ (softplus is stable & ensures > 0)
        lambda_ = F.softplus(rate_raw) + self.eps  # (B,1,D,W,H)

        # Prepare target counts
        if self.count_scale is not None:
            # Discrete approximation
            counts = torch.round(target.clamp(min=0.0) * self.count_scale)
        else:
            # Continuous relaxation
            counts = (target * self.rate_scale).clamp(min=0.0)

        # Identify structural zeros
        is_zero = (counts <= self.zero_threshold)

        # log P(Y=0) = log( p0 + (1-p0)*exp(-λ) )
        log_p_zero = torch.log(p0 + (1.0 - p0) * torch.exp(-lambda_) + self.eps)

        # For Y > 0:
        # log P(Y=k) = log(1-p0) + k * log λ - λ - log(k!)
        positive_mask = ~is_zero
        k_pos = counts[positive_mask]
        lambda_pos = lambda_[positive_mask]
        p0_pos = p0[positive_mask]

        # log(1 - p0) stable
        log1m_p0_pos = torch.log1p(-p0_pos + self.eps)
        # log factorial via lgamma(k+1)
        log_fact = torch.lgamma(k_pos + 1.0)
        log_p_pos = log1m_p0_pos + k_pos * torch.log(lambda_pos + self.eps) - lambda_pos - log_fact

        # Assemble per-voxel log likelihood
        log_prob = torch.empty_like(counts)
        log_prob[is_zero] = log_p_zero[is_zero]
        log_prob[positive_mask] = log_p_pos

        nll = -log_prob  # negative log likelihood per voxel

        # Optionally weight by statistical error
        if self.weight_with_error:
            nll = weight_field_by_statistical_error(nll, input=input)

        # Apply global scale
        if self.scale != 1.0:
            nll = nll * self.scale

        # Reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class FocalMSELoss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, gamma: float = 2.0, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__(None, weight_with_error=weight_with_error)
        self.gamma = gamma
        self.eps = eps

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        error = (torch.log(prediction + self.eps) - torch.log(target + self.eps)) ** 2
        weights = error.pow(self.gamma / 2)  # gamma=2 -> quad Gewichtung
        loss = weights * error

        if self.weight_with_error:
            loss = weight_field_by_statistical_error(loss, input=input)

        return loss.mean()


class FocalL1Loss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, gamma: float = 2.0, alpha: float = 0.75, eps: float = 1e-8):
        super().__init__(None, weight_with_error=weight_with_error)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        pos_weight = (1 - target).pow(self.gamma)
        pos_mask = (target > 0).float()
        neg_mask = 1.0 - pos_mask
        # Alpha for positives, (1-alpha) for zeros
        weights = self.alpha * pos_mask * pos_weight + (1 - self.alpha) * neg_mask
        l1 = torch.abs(torch.log(prediction + self.eps) - torch.log(target + self.eps))
        loss = weights * l1

        if self.weight_with_error:
            loss = weight_field_by_statistical_error(loss, input=input)

        return loss.mean()


class FocalSmoothL1Loss(StdLossWeighted):
    """
    Combines SmoothL1 (Huber) with focal weighting: more stable for noisy high values.
    delta: Huber transition point.
    """
    def __init__(
        self,
        delta: float = 0.01,
        gamma: float = 2.0,
        zero_threshold: float = 1e-9,
        alpha: float = 0.25,
        weight_with_error: bool = False
    ):
        super().__init__(loss_fn=None, weight_with_error=weight_with_error)
        self.delta = delta
        self.gamma = gamma
        self.zero_threshold = zero_threshold
        self.alpha = alpha
        self.eps = 1e-12

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        diff = prediction - target
        adiff = diff.abs()
        huber = torch.where(adiff < self.delta, 0.5 * adiff.pow(2) / self.delta, adiff - 0.5 * self.delta)

        pt = torch.exp(-huber)               # good fit -> pt≈1
        focal = (1 - pt).pow(self.gamma)

        pos_mask = (target > self.zero_threshold).float()
        alpha_weight = pos_mask * self.alpha + (1 - pos_mask) * (1 - self.alpha)

        loss = huber * focal * alpha_weight

        if self.weight_with_error:
            loss = weight_field_by_statistical_error(loss, input=input)

        return loss.mean()


class StructuralSimilarity3DLoss(StdLossWeighted):
    """
    3D SSIM für Volumen (B, C, D, H, W).
    Annahme: Werte in [0, data_range] oder beliebig skaliert.
    Wenn data_range=None, wird der Wertebereich dynamisch aus (target, prediction) bestimmt.
    Liefert 1 - SSIM. Optional: mask (gleiche Form wie target/pred) -> ignoriert Bereiche (mask==0).
    """
    def __init__(
        self,
        window_size: int = 7,
        sigma: float = 1.5,
        data_range: float | None = None,  # None -> dynamic data range
        C1: float | None = None,
        C2: float | None = None,
        channel_average: bool = True,
        size_average: bool = True,
        clamp_ssim: bool = True,
        weight_with_error: bool = False
    ):
        super().__init__(loss_fn=None, weight_with_error=weight_with_error)
        assert window_size % 2 == 1, "window_size muss ungerade sein."
        self.window_size = window_size
        self.sigma = sigma
        # If data_range is None, compute dynamically per forward
        self.data_range = data_range
        # Only precompute C1/C2 if data_range is fixed and C1/C2 not explicitly given
        if self.data_range is not None:
            self.C1 = C1 if C1 is not None else (0.01 * self.data_range) ** 2
            self.C2 = C2 if C2 is not None else (0.03 * self.data_range) ** 2
        else:
            # Defer to runtime computation
            self.C1 = C1  # may be None -> will be computed from dynamic range
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
        kernel_1d = g
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d = kernel_3d / kernel_3d.sum()
        kernel_3d = kernel_3d.view(1, 1, window_size, window_size, window_size)
        kernel_3d = kernel_3d.repeat(channels, 1, 1, 1, 1)
        return kernel_3d

    def _get_kernel(self, channels: int, device, dtype):
        key = (channels, device, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = StructuralSimilarity3DLoss._create_gaussian_kernel3d(
                self.window_size, self.sigma, channels, device, dtype
            )
        return self._kernel_cache[key]

    def _ssim_3d(self, x: Tensor, y: Tensor, C1: Tensor, C2: Tensor, mask: Tensor | None = None) -> Tensor:
        B, C, D, H, W = x.shape
        if D < self.window_size or H < self.window_size or W < self.window_size:
            # Fallback: L1 wenn Volumen kleiner als Fenster
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

        # SSIM Map mit dynamischen Konstanten
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = numerator / (denominator + 1e-12)

        if self.clamp_ssim:
            ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)

        if mask is not None:
            # Broadcast mask to (B,C,D,H,W)
            if mask.shape != x.shape:
                if mask.shape == (B, 1, D, H, W):
                    mask = mask.repeat(1, C, 1, 1, 1)
                else:
                    raise ValueError("Mask shape inkonsistent.")
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
        # Determine data range from valid voxels of both tensors
        if valid_mask is not None:
            vb = valid_mask > 0.5
            if vb.any():
                vals = torch.cat([prediction[vb], target[vb]], dim=0)
            else:
                # No valid voxels -> fallback
                vals = torch.tensor([0.0, 1.0], device=prediction.device, dtype=prediction.dtype)
        else:
            vals = torch.cat([prediction.reshape(-1), target.reshape(-1)], dim=0)

        vmin = vals.min()
        vmax = vals.max()
        data_range = torch.clamp(vmax - vmin, min=1e-12)

        # If user provided explicit C1/C2, respect them; otherwise compute from dynamic range
        C1_val = self.C1 if self.C1 is not None else (0.01 * data_range) ** 2
        C2_val = self.C2 if self.C2 is not None else (0.03 * data_range) ** 2

        # Ensure tensors on correct device/dtype
        C1_t = torch.as_tensor(C1_val, device=prediction.device, dtype=prediction.dtype)
        C2_t = torch.as_tensor(C2_val, device=prediction.device, dtype=prediction.dtype)
        return C1_t, C2_t

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        # Erwartete Form: (B,C,D,H,W)
        assert prediction.shape == target.shape, f"Shape mismatch {prediction.shape} vs {target.shape}"

        # Mask invalid -inf values
        neginf_mask = torch.isneginf(prediction) | torch.isneginf(target)
        if neginf_mask.any():
            # replace masked with minimum non-masked target value to avoid NaNs
            min_non_masked = torch.min(target[~neginf_mask]) if (~neginf_mask).any() else torch.tensor(0.0, device=target.device, dtype=target.dtype)
            prediction = prediction.masked_fill(neginf_mask, min_non_masked)
            target = target.masked_fill(neginf_mask, min_non_masked)

        # valid_mask: 1 for valid, 0 for invalid
        valid_mask = (~neginf_mask).float() if neginf_mask.any() else None

        # Compute C1/C2 dynamically if requested
        if self.data_range is None:
            C1_t, C2_t = self._compute_dynamic_constants(target, prediction, valid_mask)
        else:
            # Use precomputed or user provided fixed constants
            C1_t = torch.as_tensor(self.C1, device=prediction.device, dtype=prediction.dtype)
            C2_t = torch.as_tensor(self.C2, device=prediction.device, dtype=prediction.dtype)

        ssim_val = self._ssim_3d(prediction, target, C1=C1_t, C2=C2_t, mask=valid_mask)
        loss = 1.0 - ssim_val  # SSIM -> Loss
        return loss


class MultiScaleStructuralSimilarity3DLoss(StdLossWeighted):
    """
    MS-SSIM 3D. Verkleinert iterativ per AvgPool3d und kombiniert Skalen.
    Gewichte: nach Originalpaper oder benutzerdefiniert.
    Wenn data_range=None, wird pro Skala ein dynamischer Bereich verwendet.
    """
    def __init__(
        self,
        window_size: int = 7,
        sigma: float = 1.5,
        data_range: float | None = None,  # None -> dynamic
        scales: int = 3,
        scale_weights: list[float] | None = None,
        weight_with_error: bool = False
    ):
        super().__init__(loss_fn=None, weight_with_error=weight_with_error)
        self.base = StructuralSimilarity3DLoss(
            window_size=window_size,
            sigma=sigma,
            data_range=data_range,  # pass through (None -> dynamic)
            weight_with_error=False
        )
        self.scales = scales
        if scale_weights is None:
            # normalize
            self.scale_weights = torch.tensor([0.3, 0.5, 0.2][:scales], dtype=torch.float32)
            self.scale_weights /= self.scale_weights.sum()
        else:
            assert len(scale_weights) == scales
            self.scale_weights = torch.tensor(scale_weights, dtype=torch.float32)
            self.scale_weights /= self.scale_weights.sum()

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        x = prediction
        y = target
        vals = []
        for s in range(self.scales):
            v = 1.0 - self.base(y, x, input)  # SSIM Wert
            vals.append(v)
            if s < self.scales - 1:
                # Downsample (B,C,D,H,W) -> stride 2
                x = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
                y = F.avg_pool3d(y, kernel_size=2, stride=2, ceil_mode=True)
        ssim_stack = torch.stack(vals)  # (S,)
        ms_ssim = (ssim_stack * self.scale_weights.to(ssim_stack.device)).sum()
        loss = 1.0 - ms_ssim
        return loss


class TweedieLoss(nn.Module):
    def __init__(self, p: float = 1.5, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        assert 1 < p < 2, "Use 1 < p < 2 for zero-inflated continuous targets"
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        mu = prediction.clamp(min=self.eps)  # ensure positivity
        y = target

        term1 = y * mu.pow(1 - self.p) / (1 - self.p)
        term2 = mu.pow(2 - self.p) / (2 - self.p)
        loss = term1 - term2
        nll = -loss

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


class MSELossLogSpace(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(None, weight_with_error, scale)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        losses = self.mse(torch.log(target + 1e-8), torch.log(prediction + 1e-8))
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        return torch.mean(losses)


class AmplifiedL1LossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, scale: float = 1.0):
        super().__init__(None, weight_with_error, scale)
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        losses = self.l1(target, prediction)
        losses *= target.abs().clamp(min=1e-6)
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        return torch.mean(losses)


class MSELossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False):
        super().__init__(nn.MSELoss(reduction='none'), weight_with_error)


class KLDivLossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False):
        super().__init__(nn.KLDivLoss(reduction='none'), weight_with_error)

    def forward(self, target, prediction, input):
        return super().forward(target, torch.log(prediction), input)


class CrossEntropyLossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False):
        super().__init__(nn.CrossEntropyLoss(reduction='none'), weight_with_error)


class CosineSimilarityLossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False):
        super().__init__(nn.CosineSimilarity(reduction='none'), weight_with_error)

    def forward(self, target, prediction, input):
        return 1.0 - torch.clamp(super().forward(target, prediction, input), min=0.0, max=1.0)


class WassersteinLossWeighted(StdLossWeighted):
    def __init__(self, dim: int = -1, weight_with_error: bool = False):
        super().__init__(None, weight_with_error)
        self.dim = dim

    def forward(self, target, prediction, input):
        wasserstein = torch.abs(torch.cumsum(prediction, dim=self.dim) - torch.cumsum(target, dim=self.dim))
        losses = torch.mean(wasserstein, dim=self.dim)

        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)

        return torch.mean(losses)


class HuberLoss(Loss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        return self.l1_loss(target, prediction) * 0.5 + self.l2_loss(target, prediction) * 0.5


class TotalVariationDistanceLoss(nn.Module):
    def forward(self, x, y):
        difference = torch.abs(x - y)
        if len(difference.shape) > 2:
            difference = difference.reshape(difference.shape[0], -1, difference.shape[1])
            sum_difference = torch.sum(difference, dim=-1)
            sum_difference = torch.mean(sum_difference, dim=-1)
        else:
            sum_difference = torch.sum(difference, dim=-1)

        batch_mean = torch.mean(sum_difference)

        tvd = 0.5 * batch_mean

        return tvd

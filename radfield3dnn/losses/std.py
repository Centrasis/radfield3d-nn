from .base import Loss
from torch import Tensor, nn
from radfield3dnn.rftypes import TrainingInputData
import torch
from torch.nn import functional as F
from radfield3dnn.metrics.base import weight_field_by_statistical_error


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


class PlainL1Loss(Loss):
    """Plain L1 on the (already physical) normalised targets — no log, no weighting.

    For a field normalised by ``LinearNormalizer(0,1)`` the normalised value IS the
    physical flux (÷ per-field max), so a plain ``|pred − target|`` is a physical-space
    L1. The high-flux beam voxels carry the largest absolute errors and so dominate the
    objective automatically.
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


class RawNeRFLoss(Loss):
    """RawNeRF HDR loss (Mildenhall et al., CVPR 2022, "NeRF in the Dark"): linear-space L2 weighted
    by a stop-gradient relative factor, ``((p − t) / (sg(p) + eps))²``. Behaves like a relative error
    (every decade gets gradient) while staying in LINEAR space — for training multi-decade HDR
    radiance without a log transform. Pair with LinearNormalizer(0,1); eps in normalized units."""

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
        # -inf = masked / not-sampled. Excluded from the loss AND the reduction so a masked voxel
        # never contributes.
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        t = target.masked_fill(~valid, 0.0)
        p = prediction.masked_fill(~valid, 0.0)
        # Self-normalizing HDR weight, floored by the (detached) target scale so every term is
        # bounded to ≤1: ``1/(max(sg(p), |t|)+eps)``. Keeps the cross-decade relative weighting
        # (when p≈t, w≈1/t) without the p→0 blow-up of the bare ``1/(sg(p)+eps)``.
        scale = torch.maximum(p.detach().abs(), t.detach().abs())
        w = 1.0 / (scale + self.eps)
        per_voxel = ((p - t) * w) ** 2
        per_voxel = torch.nan_to_num(per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        if per_voxel.ndim <= 1:
            return per_voxel
        reduce_dims = tuple(range(1, per_voxel.ndim))
        vf = valid.to(per_voxel.dtype)
        return (per_voxel * vf).sum(dim=reduce_dims) / vf.sum(dim=reduce_dims).clamp(min=1.0)


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

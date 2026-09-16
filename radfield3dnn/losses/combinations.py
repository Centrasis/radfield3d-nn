from .std import WassersteinLossWeighted, StdLossWeighted, StructuralSimilarity3DLoss, L1LossWeighted
from radfield3dnn.metrics.base import weight_field_by_statistical_error
from torch import nn
from .base import Loss
from torch import Tensor
from radfield3dnn import TrainingInputData
import torch


class HistogramLoss(Loss):
    """
    Loss function for comparing two histograms.
    """
    def __init__(self, bin_dim: int = -1, weight_with_error: bool = False, penalize_out_of_range: bool = False, calc_moments: bool = False):
        super().__init__()
        self.weight_with_error = weight_with_error
        self.penalize_out_of_range = penalize_out_of_range
        self.l1_loss = L1LossWeighted(weight_with_error)
        self.wasserstein_loss = WassersteinLossWeighted(bin_dim, weight_with_error)
        self.bin_dim = bin_dim
        self.calc_moments = calc_moments

    def compute_moments(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.arange(dist.size(self.bin_dim), dtype=dist.dtype, device=dist.device)
        # expand x to match the size of dist tensor, with the range in the bin_dim dimension
        if self.bin_dim < 0:
            self.bin_dim += dist.dim()
        dims = [1] * dist.dim()
        dims[self.bin_dim] = dist.size(self.bin_dim)
        x = x.view(*dims)
        # repeat x to match the batch size if dist has more than 1 batch dimension
        dims = [dist.size(i) for i in range(dist.dim())]
        dims[self.bin_dim] = 1
        x = x.repeat(*dims)
        mean = torch.sum(dist * x, dim=self.bin_dim, keepdim=True)
        var = torch.sum(dist * (x - mean)**2, dim=self.bin_dim, keepdim=True)
        return mean, var

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        hist_size = target.size(self.bin_dim)
        mask = ~(torch.isfinite(target) & torch.isfinite(prediction))

        if mask.any():
            assert self.weight_with_error == False, "HistogramLoss does not support weighting with error when there are masked values."
            target = target[~mask]
            prediction = prediction[~mask]
            self.wasserstein_loss.dim = -1  # after masking, histograms are now 1D
            self.l1_loss.dim = -1
            target = target.view(-1, hist_size)
            prediction = prediction.view(-1, hist_size)
        
        sum_hist = torch.clamp(torch.sum(target, dim=self.bin_dim, keepdim=True), min=1e-8)
        sum_hist_zero = sum_hist == 0.0
        if sum_hist_zero.any():
            sum_hist[sum_hist_zero] = 1.0 / target.size(self.bin_dim)

        sum_hist = torch.clamp(torch.sum(prediction, dim=self.bin_dim, keepdim=True), min=1e-8)
        sum_hist_zero = sum_hist == 0.0
        if sum_hist_zero.any():
            prediction[sum_hist_zero] = 1.0 / target.size(self.bin_dim)

        ws = self.wasserstein_loss(target, prediction, input)
        l1 = self.l1_loss(target, prediction, input)

        if self.penalize_out_of_range:
            # get the highest index in the target histogram that has values > 1e-8
            target_max_index = (target > 1e-8).nonzero(as_tuple=True)[0].max().item() if (target > 1e-8).any() else 0
            # get the highest index in the prediction histogram that has values > 1e-8
            prediction_max_index = (prediction > 1e-8).nonzero(as_tuple=True)[0].max().item() if (prediction > 1e-8).any() else 0

            distance = abs(target_max_index - prediction_max_index)
            if distance > 0:
                ws *= 1.0 + distance * 0.1

        if self.calc_moments:
            bin_count = prediction.size(self.bin_dim) if len(prediction.shape) > self.bin_dim else 1
            pred_mean, pred_var = self.compute_moments(prediction)
            target_mean, target_var = self.compute_moments(target)
            moments_loss = (nn.functional.l1_loss(pred_mean, target_mean) / bin_count) + (nn.functional.l1_loss(pred_var, target_var) / bin_count)
            moments_loss = torch.clamp(moments_loss, max=1.0, min=0.0)  # avoid exploding moments loss
            return ws * 0.33 + l1 * 0.33 + moments_loss * 0.34
        else:
            return ws * 0.7 + l1 * 0.3


class L1L2LossWeighted(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False):
        super().__init__(None, weight_with_error)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        invalid_mask = torch.isneginf(target) | torch.isneginf(prediction)
        if invalid_mask.any():
            target = target[~invalid_mask]
            prediction = prediction[~invalid_mask]

        l1_loss = self.l1_loss(target, prediction)
        mse_loss = self.mse_loss(target, prediction)
        losses = (l1_loss + mse_loss) * 0.5
        if self.weight_with_error:
            losses = weight_field_by_statistical_error(losses, input=input)
        return torch.mean(losses) if invalid_mask.any() else torch.mean(losses, dim=tuple(range(1, losses.ndim)))  # mean over all but batch dim


class FluxLoss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, log_scale: bool = True):
        super().__init__(None, weight_with_error)
        self.ssim = StructuralSimilarity3DLoss(weight_with_error=False)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.log_scale = log_scale

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        ssim = self.ssim(target, prediction, input)

        invalid_mask = ~(torch.isfinite(target) & torch.isfinite(prediction))
        if invalid_mask.any():
            target = target[~invalid_mask]
            prediction = prediction[~invalid_mask]

        if self.log_scale:
            if target.min() < 0.0 and target.max() > 0.0:
                target = target + 1.0
                prediction = prediction + 1.0
            target = torch.log1p(target)
            prediction = torch.log1p(prediction)

        l1 = self.l1_loss(target, prediction)
        l2 = self.mse_loss(target, prediction)

        l1l2: Tensor = 0.5 * (l1 + l2)  # Huber loss with delta=1.0
        l1l2 = l1l2.mean()
        return l1l2 * 0.66 + ssim * 0.34


class FluxMultiScaleLoss(StdLossWeighted):
    def __init__(self, weight_with_error: bool = False, levels: int = 3):
        super().__init__(None, weight_with_error)
        self.l1l2 = L1L2LossWeighted(weight_with_error=weight_with_error)
        self.levels = levels
        assert self.levels >= 1, "levels must be at least 1."

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        loss = torch.tensor(0.0, dtype=target.dtype, device=target.device)
        for i in range(self.levels - 1):
            target_s = nn.functional.avg_pool3d(target, kernel_size=i*2, stride=2, padding=0, ceil_mode=True) if i > 0 else target
            prediction_s = nn.functional.avg_pool3d(prediction, kernel_size=i*2, stride=2, padding=0, ceil_mode=True) if i > 0 else prediction
            loss += self.l1l2(target_s, prediction_s, input) * (1.0 / self.levels)
        return loss

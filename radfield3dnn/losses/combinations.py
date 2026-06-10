from .std import WassersteinLossWeighted, L1LossWeighted, FluxLoss, L1WithSSIM3DLoss, StructuralSimilarity3DLoss
from .base import Loss
from torch import Tensor
from radfield3dnn.rftypes import TrainingInputData
import torch
from torch import nn


class HistogramLoss(Loss):
    """Loss for comparing spectral histograms: Wasserstein + L1."""

    def __init__(self, bin_dim: int = -1, weight_with_error: bool = False,
                 penalize_out_of_range: bool = False, calc_moments: bool = False):
        super().__init__()
        self.weight_with_error = weight_with_error
        self.penalize_out_of_range = penalize_out_of_range
        self.l1_loss = L1LossWeighted(weight_with_error)
        self.wasserstein_loss = WassersteinLossWeighted(bin_dim, weight_with_error)
        self.bin_dim = bin_dim
        self.calc_moments = calc_moments

    def compute_moments(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.arange(dist.size(self.bin_dim), dtype=dist.dtype, device=dist.device)
        if self.bin_dim < 0:
            self.bin_dim += dist.dim()
        dims = [1] * dist.dim()
        dims[self.bin_dim] = dist.size(self.bin_dim)
        x = x.view(*dims)
        dims = [dist.size(i) for i in range(dist.dim())]
        dims[self.bin_dim] = 1
        x = x.repeat(*dims)
        mean = torch.sum(dist * x, dim=self.bin_dim, keepdim=True)
        var = torch.sum(dist * (x - mean) ** 2, dim=self.bin_dim, keepdim=True)
        return mean, var

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        hist_size = target.size(self.bin_dim)
        mask = ~(torch.isfinite(target) & torch.isfinite(prediction))

        # Bin axis for the Wasserstein term. When masked values force a reshape
        # to (N, hist_size) the bin axis becomes the last one; pass it per call
        # rather than mutating the shared sub-module's `.dim` (which used to leak
        # into every later unmasked batch — a latent state bug).
        ws_dim = self.bin_dim
        if mask.any():
            assert not self.weight_with_error, "HistogramLoss does not support weighting with error when there are masked values."
            ws_dim = -1
            # Drop the masked (e.g. importance-sampled, -inf) voxels and reshape
            # to (n_valid_voxels, hist_size). CRITICAL: the bin axis must be moved
            # to LAST *before* the boolean-mask flatten. With bin_dim != last
            # (here bin_dim=1 on a volumetric (N, bins, H, W, D) spectrum) the old
            # `x[~mask].view(-1, hist_size)` interleaved different voxels' bins —
            # e.g. row0 became [A_bin0, B_bin0, A_bin1, B_bin1] instead of A's
            # histogram. Since importance sampling drops voxels every training
            # step, this scrambled EVERY histogram on EVERY step and pinned the
            # spectrum head at a trivial ~constant (val spectrum_accuracy frozen
            # ≈0.47 across 100→200 epochs). Permuting bins to last keeps each
            # voxel's histogram contiguous so the reshape is correct.
            bd = self.bin_dim if self.bin_dim >= 0 else self.bin_dim + target.dim()
            perm = [d for d in range(target.dim()) if d != bd] + [bd]
            mask_p = mask.permute(*perm)
            target = target.permute(*perm)[~mask_p].view(-1, hist_size)
            prediction = prediction.permute(*perm)[~mask_p].view(-1, hist_size)

        target_sum = torch.clamp(torch.sum(target, dim=ws_dim, keepdim=True), min=1e-8)
        target = target / target_sum
        pred_sum = torch.clamp(torch.sum(prediction, dim=ws_dim, keepdim=True), min=1e-8)
        prediction = prediction / pred_sum

        ws = self.wasserstein_loss(target, prediction, input, dim=ws_dim)
        l1 = self.l1_loss(target, prediction, input)

        if self.penalize_out_of_range:
            target_max_index = (target > 1e-8).nonzero(as_tuple=True)[0].max().item() if (target > 1e-8).any() else 0
            prediction_max_index = (prediction > 1e-8).nonzero(as_tuple=True)[0].max().item() if (prediction > 1e-8).any() else 0
            distance = abs(target_max_index - prediction_max_index)
            if distance > 0:
                ws *= 1.0 + distance * 0.1

        if self.calc_moments:
            bin_count = prediction.size(self.bin_dim) if len(prediction.shape) > self.bin_dim else 1
            pred_mean, pred_var = self.compute_moments(prediction)
            target_mean, target_var = self.compute_moments(target)
            moments_loss = (nn.functional.l1_loss(pred_mean, target_mean) / bin_count) + \
                           (nn.functional.l1_loss(pred_var, target_var) / bin_count)
            moments_loss = torch.clamp(moments_loss, max=1.0, min=0.0)
            return ws * 0.33 + l1 * 0.33 + moments_loss * 0.34
        else:
            return ws * 0.7 + l1 * 0.3

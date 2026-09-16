from torch.nn import Module
from torch import Tensor
import torch
from torch import nn


class HistogramNormalize(Module):
    def __init__(self, dim=0, enforce_positivity=True):
        super().__init__()
        self.dim = dim
        self.positivity = nn.ReLU() if enforce_positivity else nn.Identity()

    def forward(self, hists: Tensor) -> Tensor:
        hists = self.positivity(hists)

        if torch.isnan(hists).any() or torch.isinf(hists).any():
            hists = torch.where(torch.isfinite(hists), hists, torch.zeros_like(hists))

        sum = torch.sum(hists, dim=self.dim, keepdim=True)
        sum = torch.clamp(sum, min=1e-8)
        return hists / sum

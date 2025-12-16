import torch
import torch.nn as nn
from .smape import SMAPEAccuracy
from .histogram_accuracy import HistogramOverlapAccuracy
from .base import MetricBase


class TotalVariationDistanceAccuracy(nn.Module):
    def __init__(self, clamp: bool = False, epsilon: float = 1e-10):
        super(TotalVariationDistanceAccuracy, self).__init__()
        self.clamp = clamp
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)

    def forward(self, x, y):
        if self.epsilon.device != x.device:
            self.epsilon = self.epsilon.to(x.device)
        x = torch.clamp(x, min=self.epsilon)
        y = torch.clamp(y, min=self.epsilon)
        difference = torch.abs(x - y)
        if len(difference.shape) > 2:
            difference = difference.reshape(difference.shape[0], -1, difference.shape[1])
            sum_difference = torch.sum(difference, dim=-1)
            sum_difference = torch.mean(sum_difference, dim=-1)
        else:
            sum_difference = torch.sum(difference, dim=-1)

        batch_mean = torch.mean(sum_difference)

        tvd = 0.5 * batch_mean

        accuracy = 1.0 - tvd
        if self.clamp:
            accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
        return accuracy


class CosineSimilarityAccuracy(nn.Module):
    def __init__(self):
        super(CosineSimilarityAccuracy, self).__init__()

    def forward(self, x, y):
        accuracy = torch.mean((1.0 + torch.cosine_similarity(x, y)) / 2.0)
        return accuracy

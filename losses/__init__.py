import torch
from .base import Loss
from torch import Tensor, nn
from rftypes import TrainingInputData


class BalancedL1L2Loss(Loss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        return self.l1_loss(target, prediction) * 0.5 + self.l2_loss(target, prediction) * 0.5

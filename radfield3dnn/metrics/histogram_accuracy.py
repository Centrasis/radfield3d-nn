from .base import MetricBase
from torch import Tensor
import torch
from typing import Union, Literal


class HistogramOverlapAccuracy(MetricBase):
    def __init__(self, epsilon: float = 1e-8, weight_with_error: bool = False, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean'):
        super().__init__(layer_name='spectrum', reduction=reduction, weight_with_error=weight_with_error, eps=epsilon)

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        if not valid_mask.any():
            return torch.zeros(0, device=target.device, dtype=target.dtype)

        target = target[valid_mask]
        prediction = prediction[valid_mask]

        prediction = prediction + self.eps
        target = target + self.eps
        intersection = torch.min(prediction, target)
        union = (prediction + target) - intersection

        reduce_dims = tuple(range(1, prediction.ndim))
        iou = intersection.sum(dim=reduce_dims) / union.sum(dim=reduce_dims)

        return iou if self.weight_with_error else self.reduction_fn(iou)

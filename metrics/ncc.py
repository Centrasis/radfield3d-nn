from .base import MetricBase
import torch
from torch import Tensor
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel
from typing import Union, Literal


class NCCAccuracy(MetricBase):
    def __init__(self, layer_name: str = None, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean', weight_with_error: bool = False, importance_threshold: float = 0.0, keep_dim: bool = False):
        super().__init__(layer_name=layer_name, reduction=reduction, weight_with_error=weight_with_error)
        self.importance_threshold = torch.tensor(importance_threshold)
        self.keep_dim = keep_dim
        self.eps = torch.tensor(1e-9)

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        if self.eps.device != target.device:
            self.eps = self.eps.to(target.device)
            self.importance_threshold = self.importance_threshold.to(target.device)

        # Build mask: valid numbers only
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)

        # Optionally exclude very small values via importance threshold
        max_val = self.importance_threshold * target[valid_mask].max() if valid_mask.any() else torch.tensor(0.0, device=target.device, dtype=target.dtype)
        importance_mask = valid_mask & ((target >= max_val) | (prediction >= max_val))

        if self.keep_dim:
            acc_field = torch.full_like(target, -torch.inf)

        target = target[importance_mask]
        prediction = prediction[importance_mask]

        target_mean = target.mean(dim=tuple(range(1, target.ndim)), keepdim=True)
        prediction_mean = prediction.mean(dim=tuple(range(1, prediction.ndim)), keepdim=True)

        target_centered = target - target_mean
        prediction_centered = prediction - prediction_mean

        numerator = (target_centered * prediction_centered).sum(dim=tuple(range(1, target.ndim)))
        denominator = torch.sqrt((target_centered ** 2).sum(dim=tuple(range(1, target.ndim))) * (prediction_centered ** 2).sum(dim=tuple(range(1, target.ndim)))) + self.eps

        ncc = numerator / denominator

        if self.keep_dim:
            acc_field[importance_mask] = ncc
            return acc_field
        else:
            return self.reduction_fn(ncc)

    def forward(self, target: Union[RadiationFieldChannel, Tensor], prediction: Union[RadiationFieldChannel, Tensor], input: TrainingInputData = None) -> Tensor:
        if self.importance_threshold.device != target.device:
            self.eps = self.eps.to(target.device)
            self.importance_threshold = self.importance_threshold.to(target.device)

        if self.weight_with_error:
            if self.importance_threshold > 0.0:
                max_val = self.importance_threshold * target.max()
                target_data = self.extract_tensor_from(target)
                prediction_data = self.extract_tensor_from(prediction)
                input = TrainingInputData(
                    input=input.input,
                    ground_truth=RadiationField(
                        scatter_field=RadiationFieldChannel(
                            spectrum=input.ground_truth.scatter_field.spectrum,
                            flux=input.ground_truth.scatter_field.flux,
                            error=torch.where(
                                (target_data >= max_val) | (prediction_data >= max_val),
                                input.ground_truth.scatter_field.error,
                                torch.zeros_like(input.ground_truth.scatter_field.error)
                            )
                        ) if input.ground_truth.scatter_field is not None else None,
                        direct_beam=RadiationFieldChannel(
                            spectrum=input.ground_truth.direct_beam.spectrum,
                            flux=input.ground_truth.direct_beam.flux,
                            error=torch.where(
                                (target_data >= max_val) | (prediction_data >= max_val),
                                input.ground_truth.direct_beam.error,
                                torch.zeros_like(input.ground_truth.direct_beam.error)
                            )
                        ) if input.ground_truth.direct_beam is not None else None
                    ) if isinstance(input.ground_truth, RadiationField) else RadiationFieldChannel(
                        spectrum=input.ground_truth.spectrum,
                        flux=input.ground_truth.flux,
                        error=torch.where(
                            (target_data >= max_val) | (prediction_data >= max_val),  # fixed precedence with parentheses
                            input.ground_truth.error,
                            torch.zeros_like(input.ground_truth.error)
                        )
                    )
                )
        return super().forward(target, prediction, input)

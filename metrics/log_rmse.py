from .base import MetricBase
import torch
from torch import Tensor
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel
from typing import Union


class LogRMSEAccuracy(MetricBase):
    def __init__(self, layer_name: str = None, clamp: bool = False, weight_with_error: bool = False, zero_eps: float = 1e-8, importance_threshold: float = 0.0, keep_dim: bool = False):
        super().__init__(layer_name=layer_name, reduction="mean", weight_with_error=weight_with_error, eps=zero_eps)
        self.keep_dim = keep_dim

        self.importance_threshold = torch.tensor(importance_threshold, dtype=torch.float32, requires_grad=False)

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

        # Select only positions to evaluate
        t = target[importance_mask]
        p = prediction[importance_mask]

        if t.numel() == 0:
            # Nothing to evaluate
            if self.keep_dim:
                return acc_field
            return torch.zeros(0, device=target.device, dtype=target.dtype)

        t = torch.log(torch.clamp(t, min=self.eps))
        p = torch.log(torch.clamp(p, min=self.eps))
        rmse_log = torch.sqrt(torch.mean((p - t) ** 2))
        # Convert to accuracy-like score
        acc_map = 1.0 / (1.0 + rmse_log)

        if self.keep_dim:
            acc_field[importance_mask] = acc_map
            return acc_field
        else:
            return acc_map

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
                            fluence=input.ground_truth.scatter_field.fluence,
                            error=torch.where(
                                (target_data >= max_val) | (prediction_data >= max_val),
                                input.ground_truth.scatter_field.error,
                                torch.zeros_like(input.ground_truth.scatter_field.error)
                            )
                        ) if input.ground_truth.scatter_field is not None else None,
                        xray_beam=RadiationFieldChannel(
                            spectrum=input.ground_truth.xray_beam.spectrum,
                            fluence=input.ground_truth.xray_beam.fluence,
                            error=torch.where(
                                (target_data >= max_val) | (prediction_data >= max_val),
                                input.ground_truth.xray_beam.error,
                                torch.zeros_like(input.ground_truth.xray_beam.error)
                            )
                        ) if input.ground_truth.xray_beam is not None else None
                    ) if isinstance(input.ground_truth, RadiationField) else RadiationFieldChannel(
                        spectrum=input.ground_truth.spectrum,
                        fluence=input.ground_truth.fluence,
                        error=torch.where(
                            (target_data >= max_val) | (prediction_data >= max_val),  # fixed precedence with parentheses
                            input.ground_truth.error,
                            torch.zeros_like(input.ground_truth.error)
                        )
                    )
                )
        return super().forward(target, prediction, input)


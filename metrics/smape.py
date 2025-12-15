from .base import MetricBase
import torch
from torch import Tensor
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel
from typing import Union, Literal


class SMAPEAccuracy(MetricBase):
    def __init__(self, layer_name: Union[Literal['fluence'], Literal['spectrum'], Literal['error'], None] = None, clamp: bool = False, zero_eps: float = 1e-8, weight_with_error: bool = False, importance_threshold: float = 0.0, keep_dim: bool = False):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE) as accuracy metric.
        SMAPE is defined as: SMAPE = |prediction - target| / ((|target| + |prediction|) / 2)
        It is a measure of accuracy based on relative errors, scaled to the range [0, 2].
        :param layer_name: The name of the layer to compute the metric on. If None, the entire input tensor is used.
        :param clamp: If True, clamps the accuracy to the range [0, 1].
        :param zero_eps: A small value to avoid division by zero.
        :param weight_with_error: If True, weights the accuracy by the statistical error of the ground truth.
        :param importance_threshold: Relative threshold for importance weighting. If the ground truth value is below this relative threshold, the error is set to zero/the accuracy is not counted.
        :param keep_dim: If True, keeps the original dimensions of the input tensors. All voxels outside the importance mask are set to -inf.
        :return: None
        """
        super().__init__(layer_name=layer_name, reduction="mean", weight_with_error=weight_with_error, eps=zero_eps)
        self.keep_dim = keep_dim
        self.clamp = clamp
        assert 0.0 <= importance_threshold <= 1.0, "importance_threshold must be in the range [0, 1]."
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

        # Exclude trivial denom near zero to avoid background dominating
        denom = torch.abs(t) + torch.abs(p)
        non_trivial = denom > self.eps
        t = t[non_trivial]
        p = p[non_trivial]

        if t.numel() == 0:
            # Nothing to evaluate
            if self.keep_dim:
                return acc_field
            return torch.zeros(0, device=target.device, dtype=target.dtype)

        denom = denom[non_trivial]

        # SMAPE in [0, 2]: 2*|p - t| / (|p| + |t|)
        smape = (2.0 * torch.abs(p - t)) / denom

        # Accuracy from SMAPE. If clamp=True, bound to [0,1]; else keep legacy 1 - SMAPE (can be negative).
        if self.clamp:
            acc_map = (1.0 - 0.5 * smape).clamp(0.0, 1.0)
        else:
            acc_map = 1.0 - smape

        # Replace NaNs/Infs defensively
        acc_map = torch.nan_to_num(acc_map, nan=0.0, posinf=0.0, neginf=0.0)

        if self.keep_dim:
            # re-create a mask at evaluated positions after non_trivial filter
            tmp_mask = torch.zeros_like(importance_mask, dtype=torch.bool)
            tmp_mask[importance_mask] = non_trivial
            acc_field[tmp_mask] = acc_map
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


class EnergyWeightedSMAPEAccuracy(SMAPEAccuracy):
    def __init__(self, layer_name = None, clamp = False, zero_eps = 1e-8, weight_with_error = False, importance_threshold = 0, keep_dim = False):
        super().__init__(layer_name, clamp, zero_eps, weight_with_error, importance_threshold, keep_dim=True)
        self._keep_dim_weighted_result = keep_dim

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        smape = super()._calc_metric(target, prediction)
        valid_mask = torch.isfinite(smape)
        if valid_mask.all():
            weigths = target
        else:
            weigths = target[valid_mask]
            smape = smape[valid_mask]
        denom = torch.sum(weigths)
        wsmape = torch.sum(smape * weigths) / denom if denom > 0 else torch.tensor(0.0, device=target.device, dtype=target.dtype)

        if self._keep_dim_weighted_result and not valid_mask.all():
            result = torch.full_like(smape, -torch.inf)
            result[valid_mask] = wsmape
            return result
        else:
            return wsmape

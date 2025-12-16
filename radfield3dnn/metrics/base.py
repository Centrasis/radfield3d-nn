from torch import nn
from torch import Tensor
import torch
from radfield3dnn import TrainingInputData, RadiationField, RadiationFieldChannel
from typing import Union, Literal


def weight_field_by_statistical_error(tensor: Tensor, input: TrainingInputData, eps: Union[float, Tensor] = 1e-8) -> Tensor:
    """
    Weights the tensor by the statistical error of the ground truth.
    The statistical error is calculated as the average of the errors in the scatter field and x-ray beam.
    The confidence is calculated as 1.0 - (error / 2.0), where error is the average of the scatter field and x-ray beam errors.
    The confidence is then normalized to sum to 1 across the spatial dimensions per batch.
    The final result is a weighted across the batch dimension. The weights are calculated according to ratio of the accumulated confidence in each field.
    :param tensor: The tensor to be weighted.
    :param input: The input data containing ground truth errors.
    :return: The weighted tensor.
    """
    if isinstance(input.ground_truth, RadiationField):
        confidence = torch.clamp(1.0 - ((input.ground_truth.scatter_field.error + input.ground_truth.xray_beam.error) / 2.0), min=0.0, max=1.0)
    elif isinstance(input.ground_truth, RadiationFieldChannel):
        confidence = torch.clamp(1.0 - input.ground_truth.error, min=0.0, max=1.0)
    else:
        raise ValueError("Unsupported ground truth type: {}".format(type(input.ground_truth)))
    if len(confidence.shape) - 1 == len(tensor.shape):
        # If confidence has one more dimension than tensor, squeeze the error channel to match tensor's dimensions
        confidence = confidence.squeeze(1)

    if len(tensor.shape) > 1:
        spatial_dims = tuple(range(1, len(tensor.shape))) # Exclude the batch dimension (0th dimension)
        # Sum along the first dimension (batch dimension) and keep the rest
        confidence_spatial_sum = torch.sum(confidence, dim=spatial_dims, keepdim=True)
        confidence_spatial_sum = torch.clamp(confidence_spatial_sum, min=eps)
        max_sum = torch.sum(torch.ones_like(confidence), dim=spatial_dims, keepdim=True)
        spatial_normalization_factor = max_sum / confidence_spatial_sum
        spatial_normalization_factor = spatial_normalization_factor.expand_as(tensor)
    else:
        spatial_normalization_factor = 1.0
    
    # Apply the confidence to the tensor and correct for the down scaling from the confidence as if it was not applied
    # This ensures that computed gradients are comparable to unweighted tensors
    weighted_tensor = confidence * tensor
    weighted_tensor = weighted_tensor * spatial_normalization_factor
    return weighted_tensor

def identity_reduction(x: Tensor, dim=None) -> Tensor:
    return x

class MetricBase(nn.Module):
    """
    Base class for a metric method.
    """
    def __init__(self, layer_name: Union[Literal['fluence'], Literal['spectrum'], Literal['error'], None] = None, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean', weight_with_error: bool = False, eps: float = 1e-8):
        super().__init__()
        self.layer_name = layer_name
        self.weight_with_error = weight_with_error
        self.eps = torch.tensor(eps, dtype=torch.float32)
        self.reduction_fn = None
        if reduction == 'mean':
            self.reduction_fn = torch.mean
        elif reduction == 'median':
            self.reduction_fn = torch.median
        elif reduction == 'none':
            self.reduction_fn = identity_reduction
        else:
            raise ValueError("Unsupported reduction type: {}".format(reduction))

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        """
        Calculate the metric.
        Args:
            target (Tensor): The ground truth tensor.
            prediction (Tensor): The predicted tensor.
        Returns:
            Tensor: The calculated metric as single valued tensor per batch of shape (B,) or (B, 1). If weight_with_error is true, return a voxelwise metric field of shape (B, 1, D, W, H) or (B, D, W, H).
        """
        ...

    def extract_tensor_from(self, x: Union[RadiationFieldChannel, Tensor]) -> Tensor:
        if self.layer_name == 'fluence':
            return x.fluence if isinstance(x, RadiationFieldChannel) else x
        elif self.layer_name == 'spectrum':
            return x.spectrum if isinstance(x, RadiationFieldChannel) else x
        elif self.layer_name == 'error':
            return x.error if isinstance(x, RadiationFieldChannel) else x
        elif self.layer_name is None and isinstance(x, Tensor):
            return x
        else:
            raise ValueError("Unsupported layer name: {}".format(self.layer_name))

    def forward(self, target: Union[RadiationFieldChannel, Tensor], prediction: Union[RadiationFieldChannel, Tensor], input: TrainingInputData = None) -> Tensor:
        target_data = self.extract_tensor_from(target)
        prediction_data = self.extract_tensor_from(prediction)

        metric = self._calc_metric(target_data, prediction_data)
        if self.weight_with_error and input is not None:
            metric = self.weight_by_statistical_error(metric, input=input)

        if len(metric.shape) > 1 or (len(metric.shape) > 0 and metric.shape[0] > 1):
            metric = metric.view(-1)
            valid_mask = torch.isfinite(metric)
            if valid_mask.any():
                metric = metric[valid_mask]
                metric = self.reduction_fn(metric)
            else:
                metric = torch.tensor(torch.nan, device=metric.device, dtype=metric.dtype)
        return metric

    def weight_by_statistical_error(self, tensor: Tensor, input: TrainingInputData) -> Tensor:
        """
        Weights the tensor by the statistical error of the ground truth.
        The statistical error is calculated as the average of the errors in the scatter field and x-ray beam.
        The confidence is calculated as 1.0 - (error / 2.0), where error is the average of the scatter field and x-ray beam errors.
        The confidence is then normalized to sum to 1 across the spatial dimensions per batch.
        The final result is a weighted across the batch dimension. The weights are calculated according to ratio of the accumulated confidence in each field.
        :param tensor: The tensor to be weighted.
        :param input: The input data containing ground truth errors.
        :return: The weighted tensor.
        """
        return weight_field_by_statistical_error(tensor, input, eps=self.eps)

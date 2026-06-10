from RadFiled3D.pytorch.types import RadiationField as rf3RadiationField, RadiationFieldChannel, TrainingInputData as rf3TrainingInputData, DirectionalInput as DirectionalInput, PositionalInput as PositionalInput
from typing import NamedTuple, Union
from torch import Tensor


class RadiationField(NamedTuple):
    scatter_field: RadiationFieldChannel
    direct_beam: RadiationFieldChannel
    geometry: Union[Tensor, None] = None  # Optional geometry tensor associated with the radiation field


class AirKermaField(NamedTuple):
    air_kerma: Tensor
    geometry: Union[Tensor, None] = None  # Optional geometry tensor associated with the air kerma field


class TrainingInputData(NamedTuple):
    input: Union[DirectionalInput, PositionalInput, Tensor]
    ground_truth: Union[rf3RadiationField, RadiationFieldChannel, RadiationField, AirKermaField]
    original_ground_truth: Union[rf3RadiationField, RadiationFieldChannel, RadiationField, AirKermaField, None] = None  # Optional original ground truth for reference

from rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, rf3RadiationField, rf3TrainingInputData
from typing import Union
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing


class ChannelsJoin(DataProcessing):
    def join_channels(self, field: RadiationField) -> RadiationFieldChannel:
        if field.xray_beam is None:
            return field.scatter_field
        elif field.scatter_field is None:
            return field.xray_beam

        total_fluence = field.scatter_field.fluence + field.xray_beam.fluence
        scatter_fluence = field.scatter_field.fluence
        beam_fluence = field.xray_beam.fluence
        valid_mask = torch.isfinite(total_fluence)
        if not valid_mask.all():
            orig_values = total_fluence.clone()
            total_fluence = total_fluence[valid_mask]
            scatter_fluence = scatter_fluence[valid_mask]
            beam_fluence = field.xray_beam.fluence[valid_mask]
        if len(total_fluence.shape) < len(field.xray_beam.spectrum.shape):
            assert len(total_fluence.shape) == len(field.scatter_field.spectrum.shape) - 1, f"Fluence and spectrum dimensions do not match: {total_fluence.shape} vs {field.scatter_field.spectrum.shape}"
            if len(total_fluence.shape) == 3:
                total_fluence = total_fluence.unsqueeze(0)
                scatter_fluence = scatter_fluence.unsqueeze(0)
                beam_fluence = beam_fluence.unsqueeze(0)
            elif len(total_fluence.shape) == 4:
                total_fluence = total_fluence.unsqueeze(1)
                scatter_fluence = scatter_fluence.unsqueeze(1)
                beam_fluence = beam_fluence.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected shape for total_fluence: {total_fluence.shape}. Expected 3 or 4 dimensions.")
        ratio_beam = (beam_fluence + 1e-8) / (total_fluence  + 1e-8)
        ratio_scatter = (scatter_fluence + 1e-8) / (total_fluence + 1e-8)
        scatter_component = ratio_scatter * field.scatter_field.spectrum
        beam_component = ratio_beam * field.xray_beam.spectrum
        spectrum = scatter_component + beam_component
        if len(spectrum.shape) == 1:
            spectrum_sum = torch.clamp(torch.sum(spectrum), min=1e-8)
        elif len(spectrum.shape) == 4:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=0, keepdim=True), min=1e-8)
        else:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=1, keepdim=True), min=1e-8)
        spectrum = spectrum / spectrum_sum

        total_error = field.scatter_field.error
        if total_error is None:
            total_error = field.xray_beam.error
        elif total_error is not None and field.xray_beam.error is not None:
            total_error = (total_error + field.xray_beam.error) / 2.0

        if not valid_mask.all():
            orig_values[valid_mask] = total_fluence
            total_fluence = orig_values
        return RadiationFieldChannel(
            spectrum=spectrum,
            fluence=total_fluence,
            error=total_error
        )
    
    def forward(self, x: Union[TrainingInputData, RadiationField]) -> Union[TrainingInputData, RadiationFieldChannel]:
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth),
                original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None
            )
        elif isinstance(x, (RadiationField, rf3RadiationField)):
            return self.join_channels(x)
        elif isinstance(x, RadiationFieldChannel):
            return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected TrainingInputData, RadiationField, or RadiationFieldChannel.")

    @classmethod
    def create_from_config(cls, config: dict) -> "ChannelsJoin":
        return ChannelsJoin()

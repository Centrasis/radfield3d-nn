from radfield3dnn import RadiationField, TrainingInputData, RadiationFieldChannel, rf3RadiationField, rf3TrainingInputData
from typing import Union
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing


class ChannelsJoin(DataProcessing):
    def join_channels(self, field: RadiationField) -> RadiationFieldChannel:
        if field.direct_beam is None:
            return field.scatter_field
        elif field.scatter_field is None:
            return field.direct_beam

        total_flux = field.scatter_field.flux + field.direct_beam.flux
        scatter_flux = field.scatter_field.flux
        beam_flux= field.direct_beam.flux
        valid_mask = torch.isfinite(total_flux)
        if not valid_mask.all():
            orig_values = total_flux.clone()
            total_flux = total_flux[valid_mask]
            scatter_flux = scatter_flux[valid_mask]
            beam_flux = field.direct_beam.flux[valid_mask]
        if len(total_flux.shape) < len(field.direct_beam.spectrum.shape):
            assert len(total_flux.shape) == len(field.scatter_field.spectrum.shape) - 1, f"flux and spectrum dimensions do not match: {total_flux.shape} vs {field.scatter_field.spectrum.shape}"
            if len(total_flux.shape) == 3:
                total_flux = total_flux.unsqueeze(0)
                scatter_flux = scatter_flux.unsqueeze(0)
                beam_flux = beam_flux.unsqueeze(0)
            elif len(total_flux.shape) == 4:
                total_flux = total_flux.unsqueeze(1)
                scatter_flux = scatter_flux.unsqueeze(1)
                beam_flux = beam_flux.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected shape for total_flux: {total_flux.shape}. Expected 3 or 4 dimensions.")
        ratio_beam = (beam_flux + 1e-8) / (total_flux  + 1e-8)
        ratio_scatter = (scatter_flux + 1e-8) / (total_flux + 1e-8)
        scatter_component = ratio_scatter * field.scatter_field.spectrum
        beam_component = ratio_beam * field.direct_beam.spectrum
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
            total_error = field.direct_beam.error
        elif total_error is not None and field.direct_beam.error is not None:
            total_error = (total_error + field.direct_beam.error) / 2.0

        if not valid_mask.all():
            orig_values[valid_mask] = total_flux
            total_flux = orig_values
        return RadiationFieldChannel(
            spectrum=spectrum,
            flux=total_flux,
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

from radfield3dnn.rftypes import RadiationField, TrainingInputData, RadiationFieldChannel, rf3RadiationField, rf3TrainingInputData
from typing import Union
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing


class ChannelsJoin(DataProcessing):
    def join_channels(self, field: RadiationField) -> RadiationFieldChannel:
        if field.direct_beam is None:
            return field.scatter_field
        elif field.scatter_field is None:
            return field.direct_beam

        scatter_flux = field.scatter_field.flux
        beam_flux = field.direct_beam.flux

        # Valid datasets are required to be NaN/Inf-free; surface any
        # violation as a hard error here so the pipeline can't silently
        # corrupt training data downstream.
        assert torch.isfinite(scatter_flux).all() and torch.isfinite(beam_flux).all(), \
            "ChannelsJoin: non-finite values in scatter/direct flux — dataset is invalid."

        total_flux = scatter_flux + beam_flux

        # The spectrum tensor carries the histogram bin as either dim 0
        # (single field, shape (C, D, H, W)) or dim 1 (batched, shape
        # (B, C, D, H, W)). Insert a length-1 bin axis into the flux
        # tensors so `ratio * spectrum` broadcasts correctly.
        spec_ndim = field.scatter_field.spectrum.ndim
        flux_ndim = total_flux.ndim
        if spec_ndim > flux_ndim:
            assert spec_ndim - flux_ndim == 1, \
                f"Flux/spectrum dim mismatch: flux={total_flux.shape} spectrum={field.scatter_field.spectrum.shape}"
            bin_axis = 0 if spec_ndim == 4 else 1
            total_flux_b   = total_flux.unsqueeze(bin_axis)
            scatter_flux_b = scatter_flux.unsqueeze(bin_axis)
            beam_flux_b    = beam_flux.unsqueeze(bin_axis)
        else:
            total_flux_b   = total_flux
            scatter_flux_b = scatter_flux
            beam_flux_b    = beam_flux

        # eps in numerator AND denominator gives a stable ratio when both
        # channels are zero. The downstream `empty_mask` then zeroes the
        # spectrum for those voxels entirely (B-16) so the spectrum loss
        # does not waste gradient on the ~13% empty-voxel mass.
        eps = 1e-8
        ratio_beam    = (beam_flux_b    + eps) / (total_flux_b + eps)
        ratio_scatter = (scatter_flux_b + eps) / (total_flux_b + eps)
        spectrum = ratio_scatter * field.scatter_field.spectrum + ratio_beam * field.direct_beam.spectrum

        # Re-normalise along the histogram bin axis. ratio_beam +
        # ratio_scatter ≈ 1 (within the eps-introduced bias), so the
        # rescale is a near-noop in the non-empty mass.
        if spec_ndim == 1:
            spectrum_sum = torch.clamp(torch.sum(spectrum), min=eps)
        elif spec_ndim == 4:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=0, keepdim=True), min=eps)
        else:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=1, keepdim=True), min=eps)
        spectrum = spectrum / spectrum_sum

        # B-16: where the joined flux is exactly 0 (both channels empty),
        # the eps-stabilised ratio above produces a near-uniform spectrum
        # — the HistogramLoss would pull the predicted spectrum toward
        # that arbitrary uniform target on the ~13% empty-voxel mass.
        # Zero those spectra out so the loss treats them as a "no
        # constraint" target. Downstream HistogramLoss already handles
        # sum=0 voxels by clamping.
        empty_mask = (total_flux_b <= 0)
        if empty_mask.any():
            spectrum = torch.where(empty_mask.expand_as(spectrum),
                                    torch.zeros_like(spectrum),
                                    spectrum)

        total_error = field.scatter_field.error
        if total_error is None:
            total_error = field.direct_beam.error
        elif field.direct_beam.error is not None:
            total_error = (total_error + field.direct_beam.error) / 2.0

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

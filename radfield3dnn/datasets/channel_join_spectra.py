"""ChannelsJoinSpectra — the two-head channel transform.

The two-headed PBRFNet predicts the scatter and direct-beam **flux** with two
separate heads but a **single, shared spectrum** head. Its target therefore needs
the two fluxes kept apart while the per-channel spectra are merged into one.
Neither of the existing transforms does this: ``ChannelsJoin`` sums the flux too
(single-head target), and ``ChannelsSplitRelative`` re-encodes the direct channel
relative to scatter. This transform fills that gap.

Output (a `RadiationField`, mirroring the two-head model's own output structure):

* ``scatter_field``: original scatter flux (kept) + the **joined spectrum**
  (flux-weighted average of the two channel spectra, exactly as ``ChannelsJoin``
  computes it) + scatter error.
* ``direct_beam``:  original direct flux (kept) + a **zero spectrum placeholder**
  (the model emits zeros on the direct spectrum slot; a zero GT makes the direct
  spectrum loss a no-grad constant the MTL balancer skips) + direct error.

So the loss sees: scatter-flux vs scatter head, direct-flux vs direct head, and
the one joined spectrum vs the shared spectrum head — the intended two-head
contract.
"""

from typing import Union
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import (
    RadiationField, TrainingInputData, RadiationFieldChannel,
    rf3RadiationField, rf3TrainingInputData,
)


class ChannelsJoinSpectra(DataProcessing):
    def join_spectrum(self, field: RadiationField) -> torch.Tensor:
        """Flux-weighted average of the two channel spectra (cf. ChannelsJoin)."""
        scatter_flux = field.scatter_field.flux
        beam_flux = field.direct_beam.flux
        assert torch.isfinite(scatter_flux).all() and torch.isfinite(beam_flux).all(), \
            "ChannelsJoinSpectra: non-finite values in scatter/direct flux — dataset is invalid."

        total_flux = scatter_flux + beam_flux
        spec = field.scatter_field.spectrum
        spec_ndim = spec.ndim
        flux_ndim = total_flux.ndim
        if spec_ndim > flux_ndim:
            assert spec_ndim - flux_ndim == 1, \
                f"Flux/spectrum dim mismatch: flux={total_flux.shape} spectrum={spec.shape}"
            bin_axis = 0 if spec_ndim == 4 else 1
            total_flux_b = total_flux.unsqueeze(bin_axis)
            scatter_flux_b = scatter_flux.unsqueeze(bin_axis)
            beam_flux_b = beam_flux.unsqueeze(bin_axis)
        else:
            bin_axis = None
            total_flux_b = total_flux
            scatter_flux_b = scatter_flux
            beam_flux_b = beam_flux

        eps = 1e-8
        ratio_beam = (beam_flux_b + eps) / (total_flux_b + eps)
        ratio_scatter = (scatter_flux_b + eps) / (total_flux_b + eps)
        spectrum = ratio_scatter * spec + ratio_beam * field.direct_beam.spectrum

        if spec_ndim == 1:
            spectrum_sum = torch.clamp(torch.sum(spectrum), min=eps)
        elif spec_ndim == 4:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=0, keepdim=True), min=eps)
        else:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=1, keepdim=True), min=eps)
        spectrum = spectrum / spectrum_sum

        # Zero the spectrum where both channels are empty (no constraint),
        # matching ChannelsJoin (B-16).
        empty_mask = (total_flux_b <= 0)
        if empty_mask.any():
            spectrum = torch.where(empty_mask.expand_as(spectrum),
                                   torch.zeros_like(spectrum), spectrum)
        return spectrum

    def join_channels(self, field: RadiationField) -> RadiationField:
        # Degenerate single-channel inputs: nothing to join.
        if field.direct_beam is None or field.scatter_field is None:
            return field

        joined_spectrum = self.join_spectrum(field)
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                flux=field.scatter_field.flux,           # kept split
                spectrum=joined_spectrum,                # merged spectrum
                error=field.scatter_field.error,
            ),
            direct_beam=RadiationFieldChannel(
                flux=field.direct_beam.flux,             # kept split
                spectrum=torch.zeros_like(joined_spectrum),  # placeholder slot
                error=field.direct_beam.error,
            ),
        )

    def forward(self, x: Union[TrainingInputData, RadiationField]) -> Union[TrainingInputData, RadiationField]:
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth),
                original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None,
            )
        elif isinstance(x, (RadiationField, rf3RadiationField)):
            return self.join_channels(x)
        elif isinstance(x, RadiationFieldChannel):
            return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected TrainingInputData or RadiationField.")

    @classmethod
    def create_from_config(cls, config: dict) -> "ChannelsJoinSpectra":
        return ChannelsJoinSpectra()

"""TwoHeadHDRNet — a clean dual-head (scatter + direct) PBRFNet derived from the DS03 HDR
diagnosis (claude-notes/hdr-analysis/PBRFNet_HDR_diagnosis.html). Complete overhaul of the old
stripped two-head path (which did not learn).

Why two heads (from the diagnosis):
  * §3/§4 the *joined* field is normalised by the beam peak, crushing scatter into the bottom of
    the output (only ~4% of the relative gradient). Predicting scatter on its OWN head, with the
    per-channel `asinh_split` normaliser (scatter ÷ scatter-peak, direct ÷ direct-peak), gives
    scatter the full output range.
  * §9/§10 the direct beam is LEARNED (the analytic beam is >10% off MC) and owns the only hard
    edge (the penumbra, 61× sharper than scatter) — so it gets its own head and an extra
    high-frequency positional injection, while the scatter head stays smooth (§8: no hard edges,
    ~55% radially symmetric → low-frequency features suffice, real-time).
  * §5/§7 the capacity that matters is the beam→scatter conditioning (~13–28 mode manifold), which
    rides the shared FiLM-conditioned trunk + beam encoder (run once per field → real-time).

Pair with: normalizer "asinh_split", flux_loss "FluxLossRelative", importance sampling on.
The framework already scores both `scatter_field` and `direct_beam`; the old head just emitted
`direct_beam=None`. Here both channels are produced with per-channel asinh-range clamping.
"""
from typing import Literal, Union
import torch
from torch import nn, Tensor

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping
from radfield3dnn.rftypes import PositionalInput, RadiationField, RadiationFieldChannel


class TwoHeadHDRNet(PBRFNet):
    """Dual-head PBRFNet: separate learned scatter and direct-beam heads, per-channel asinh range."""
    __model_name__ = "TwoHeadHDRNet"

    @property
    def output_head_markers(self) -> tuple:
        # DB-MTL excludes params whose name contains any marker from the *shared* trunk Jacobian.
        # The parent's "flux_decoder"/"spectra_decoder" already catch flux_decoder_direct (substring),
        # but the direct head's xyz projection is also task-specific (direct-flux only) and must be
        # excluded too — otherwise DB-MTL mis-attributes it to the shared representation.
        return super().output_head_markers + ("direct_xyz_proj",)

    class BackboneModel(PBRFNet.BackboneModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            d = self.d_model
            loc_dims = self.positional_location_encoding.encoded_dims
            # Direct-beam heads (own flux + spectrum). The direct head also receives a direct
            # projection of the positional encoding (high-frequency detail for the sharp penumbra
            # edge, §10) added to the FiLM-conditioned trunk feature.
            self.direct_xyz_proj = nn.Linear(loc_dims, d)
            self.flux_decoder_direct = nn.Sequential(nn.Linear(d, d), nn.SiLU(True), nn.Linear(d, 1))
            # NOTE: no separate direct spectrum head — calculate_metrics scores ONLY the direct
            # FLUX (the direct beam shares the joined spectrum in the scatter slot). A direct
            # spectrum head would receive no real-loss gradient (dead weight).
            # asinh_split maps each channel to [0,1]; clamp(0,1) is the matching flux activation.
            self.flux_activation_direct = GradientConservingClamping(0.0, 1.0)

        def _decode_two(self, trunk_feat: Tensor, xyz_enc: Tensor) -> RadiationField:
            # Scatter channel: reuse the parent single-head decode (scatter flux + the joined spectrum).
            scatter = super().decode_results(trunk_feat).scatter_field
            # Direct channel: inject positional detail (penumbra), own flux head; share the spectrum.
            df = trunk_feat + self.direct_xyz_proj(xyz_enc)
            d_flux = self.flux_decoder_direct(df).squeeze(-1)
            # No additive offset: the direct flux head's final Linear bias learns any shift.
            d_flux = self.flux_activation_direct(d_flux)
            return RadiationField(
                scatter_field=scatter,
                direct_beam=RadiationFieldChannel(spectrum=scatter.spectrum, flux=d_flux.float(), error=None),
            )

        def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
            position = batch.position.to(self._compute_dtype)
            xyz_enc = self.positional_location_encoding(position)
            params_enc = (self.encode_additional_parameters(batch)
                          if global_parameters is None else global_parameters.to(self._compute_dtype))
            x0 = xyz_enc if self.use_conditioning else torch.cat((xyz_enc, params_enc), dim=-1)
            x1 = self.block1(x0)
            if self.use_conditioning:
                x = self.beam_conditioner1(x1, params_enc)
                if self._xyz_concat_skip:
                    x = torch.cat((x, xyz_enc), dim=-1)
            else:
                x1 = self.activation_fn(x1)
                x = torch.cat((x1, x0), dim=-1)
            x = self.block2(x)
            if self.use_conditioning:
                x = self.beam_conditioner2(x, params_enc)
            else:
                x = self.activation_fn(x)
            x2_1 = x + x1
            return self._decode_two(x2_1, xyz_enc)

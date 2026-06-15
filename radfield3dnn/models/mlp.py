from .feedforward import FeedforwardPointwiseModel
from radfield3dnn.rftypes import PositionalInput, Union, Tensor, RadiationField, RadiationFieldChannel
from radfield3dnn.models.layers import ConcatLinear, Concat
from radfield3dnn.models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
from radfield3dnn.models.encoders.spherical_hamonics import SphericalHarmonics
from torch import nn
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.models.activations.flux_activations import LogitSigmoid
import torch
from radfield3dnn.models.encoders.spectra_factory import build_spectra_encoding
from radfield3dnn.losses.std import RawNeRFLoss


class SimpleMLP(FeedforwardPointwiseModel):
    __model_name__ = "SimpleMLP"

    @property
    def _compute_dtype(self) -> torch.dtype:
        return torch.float16 if self._precision == "fp16" else torch.float32

    def __init__(self, d_model: int = 128, precision: str = "fp32", learning_rate = 0.001, normalizer=None,
                 flux_loss: str = "FluxLoss", spectrum_loss: str = "HistogramLoss"):
        super().__init__(learning_rate, True, False, normalizer)
        # Losses are config-wired (ModuleBuilder registry names, e.g. "FluxLoss",
        # "TwoROIGammaLoss", "HotspotAwareFluxLoss", "RawNeRFSharp") so the SimpleMLP loss study
        # can sweep them from JSON configs exactly like PBRFNet.
        from radfield3dnn.models.base import ModuleBuilder
        self.flux_loss_name = flux_loss
        self.spectrum_loss_name = spectrum_loss
        self._flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)
        self.d_model = d_model
        self._precision = precision
        self.location_encoding  = SinusoidalFrequencyEncoding(10, 3, True)
        self.direction_encoding = SphericalHarmonics(3, False)
        self.block1 = ConcatLinear(
            self.location_encoding.encoded_dims,
            self.d_model
        )
        self.block2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(True),
            nn.Linear(d_model // 2, 32 + 1),
        )
        self.spectrum_act = HistogramNormalize(dim=-1)
        # PER-VOXEL flux activation. The previous nn.Softmax() normalized ACROSS the (N,) voxel
        # batch — every inner-batch chunk's fluxes were forced to sum to 1.0 (~1.5e-5 each at
        # 65536 voxels), so the field was near-uniform BY CONSTRUCTION and chunk-dependent: no
        # loss could produce structure through it. LogitSigmoid maps each voxel independently to
        # (0,1) with a gradient-conserving ±30 logit clamp (same head as PBRFNet's linear0_1 recipe).
        self.flux_act     = LogitSigmoid(logit_range=30.0)

        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(True),
            nn.Linear(16, 16),
            nn.SiLU(True)
        )
        self.spectra_encoder = build_spectra_encoding(
            {
                "type": "simple",
                "in_spectra_dim": 32,
                "encoded_spectra_dims": 16
            }
        )

        self.beam_encoder = nn.Sequential(
            Concat(dim=-1),
            nn.Linear(self.direction_encoding.encoded_dims + 16 + 16, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model)
        )

        self._init_flux_bias()

    def _init_flux_bias(self):
        # The flux output is element 32 of the final decoder Linear (32 spectrum + 1 flux). Bias it
        # by the NORMALIZER family: linear → just above the floor (anti-collapse), log → midpoint.
        # Default PyTorch init starts the bias ~0 → sigmoid(0)=0.5, the collapse-prone midpoint.
        if self._normalizer is None:
            return
        from radfield3dnn.models.activations.flux_activations import flux_head_init_bias
        out = self.decoder[-1]
        if isinstance(out, nn.Linear) and out.bias is not None:
            with torch.no_grad():
                out.bias[32] = flux_head_init_bias(self.flux_act, self._normalizer)

    def get_custom_parameters(self) -> dict:
        return {
            "d_model": self.d_model,
            "precision": self._precision,
            "flux_loss": self.flux_loss_name,
            "spectrum_loss": self.spectrum_loss_name,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None,
        }

    def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
        dtype = self._compute_dtype
        assert batch.origin.shape[-1] == 1, f"Origin must be a single distance value for PBRFNet. Got shape: {batch.origin.shape}"

        dir_enc = self.direction_encoding(batch.direction.to(dtype))
        origin_enc = self.distance_encoder(batch.origin.to(dtype))
        spectrum = self.spectra_encoder(batch.spectrum.to(dtype))
        enc = [dir_enc, origin_enc, spectrum]
        beam = self.beam_encoder(enc)
        return beam
    
    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
        position = batch.position.to(self._compute_dtype)
        xyz_enc = self.location_encoding(position)
        params_enc = self.encode_additional_parameters(batch) if global_parameters is None else global_parameters.to(self._compute_dtype)

        x0 = self.block1(params_enc, xyz_enc)
        x1 = self.block2(x0) + x0
        x2 = self.block3(x1) + x1
        x3 = self.decoder(x2)

        spectrum = self.spectrum_act(x3[:, :32]).float()
        flux     = self.flux_act(x3[:, 32]).float()

        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=spectrum,
                flux=flux,
                error=None
            ),
            direct_beam=None
        )

import torch
import torch.nn as nn
from torch import Tensor

from .nerf import PBRFNet
from .encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
from radfield3dnn.rftypes import PositionalInput, positional_like, TranslationalPositionalInput


class TPBRFNet(PBRFNet):
    """
    Translational Parametric Beam Radiation Field Network
    PBRFNet with one additional beam parameter: the patient translation (vec3, in metres) read from
    the field metadata. Only the in-plane (x, y) components are encoded — z (couch height) is held
    constant across the dataset. Requires a TranslationalInput; the network is otherwise identical
    to PBRFNet, so configs/normalizer/losses carry over unchanged.

    The translation is SPATIAL — the field shifts with the patient, so the output depends on it at
    high frequency, the regime where Fourier features demonstrably help low-dim coordinate inputs
    (Tancik et al. 2020; NeRF's positional encoding). It is therefore encoded with the SAME
    sinusoidal frequency encoding family as the query position, then projected to
    scalar_encoding_dims. The remaining beam parameters keep their proven encoders: distance and
    rectangle collimation are smooth GLOBAL scales (Fourier encoding did not help distance
    empirically, consistent with conditioning practice in parametric neural fields), direction
    keeps spherical harmonics.

    ``conditioning_params["translation_encoding"]`` selects the encoder (folded into
    conditioning_params like ``use_beam_shape``):
        {"type": "fourier", "n_frequencies": 6, "append_input": True}   (default)
        {"type": "mlp"}   — the pre-Fourier Linear/SiLU stack; set this in the stored config of
                            checkpoints trained before the Fourier default to keep them loadable.
        {"type": "relative", "n_frequencies": 6, "append_input": True}
            DUAL-FRAME mode. The physics: the field stays ISOCENTRIC (source/beam fixed in the
            world frame); the translation moves the PATIENT, i.e. what changes is the
            intersection of the fixed beam with the shifted geometry. The natural inductive bias
            is therefore not "translation as a global beam parameter" but a second PER-POINT
            coordinate: the trunk sees the world-frame position AND the patient-frame coordinate
            ``position_xy − frame(translation_xy)`` (Fourier-encoded, appended to the location
            features). ``frame`` is a learnable per-axis affine (init scale 1, shift −0.5) that
            maps the [0,1]-normalized translation onto the position's coordinate scale — the two
            normalizations use different physical ranges, so the alignment is learned. In this
            mode the translation LEAVES the beam vector (the beam encoder is exactly PBRFNet's),
            and the deploy package exports as a single graph (the split trunk would need the
            translation as a third input, which the C++ runtime does not bind yet).
    Expects the translation NORMALIZED to [0, 1] (TranslationNormalization, built from the dataset
    definition file) so the frequency ladder 2^i·π·x spans the intended band.
    """
    __model_name__ = "TPBRFNet"

    class BackboneModel(PBRFNet.BackboneModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trans_enc_params = dict(self._conditioning_params.get(
                "translation_encoding", {"type": "fourier", "n_frequencies": 6, "append_input": True}))
            enc_type = trans_enc_params.get("type", "fourier")
            self._relative_translation = (enc_type == "relative")
            if enc_type == "relative":
                # DUAL-FRAME: patient-frame coordinate per query point, appended to the trunk's
                # location features; the beam vector stays PBRFNet's (super() configured it).
                self.translation_encoder = None
                self.relative_translation_encoding = SinusoidalFrequencyEncoding(
                    pos_enc_dim=int(trans_enc_params.get("n_frequencies", 6)),
                    d_input=2,  # (x, y); z is constant across the dataset
                    append_input=bool(trans_enc_params.get("append_input", True)),
                )
                # Learnable per-axis frame alignment: translation is [0,1]-normalized over its
                # GENERATION range, the position over the FIELD extent — the mapping between the
                # two scales is two affine coefficients per axis, learned (init: identity scale,
                # −0.5 shift = center the normalized translation).
                self.translation_frame_scale = nn.Parameter(torch.ones(2))
                self.translation_frame_shift = nn.Parameter(torch.full((2,), -0.5))
                self._translation_encoding_params = {"type": enc_type, **{k: v for k, v in trans_enc_params.items() if k != "type"}}
                self._build_trunk_blocks(self.location_feature_dims)   # widen block1/block2
                return
            if enc_type == "fourier":
                fourier = SinusoidalFrequencyEncoding(
                    pos_enc_dim=int(trans_enc_params.get("n_frequencies", 6)),
                    d_input=2,  # (x, y) of the vec3 translation; z is constant
                    append_input=bool(trans_enc_params.get("append_input", True)),
                )
                self.translation_encoder = nn.Sequential(
                    fourier,
                    nn.Linear(fourier.calc_encoded_dim(), self.scalar_encoding_dims),
                    nn.SiLU(True),
                    nn.Linear(self.scalar_encoding_dims, self.scalar_encoding_dims),
                    nn.SiLU(True)
                )
            elif enc_type == "mlp":
                self.translation_encoder = nn.Sequential(
                    nn.Linear(2, self.scalar_encoding_dims),  # (x, y) of the vec3 translation; z is constant
                    nn.SiLU(True),
                    nn.Linear(self.scalar_encoding_dims, self.scalar_encoding_dims),
                    nn.SiLU(True)
                )
            else:
                raise ValueError(f"Unknown translation_encoding type: {enc_type!r}. Valid: 'fourier', 'mlp', 'relative'.")
            self._translation_encoding_params = {"type": enc_type, **{k: v for k, v in trans_enc_params.items() if k != "type"}}
            # Same re-configure pattern as SPERFNet/PBRFNet: extend the beam vector by the new
            # encoder's width and rebuild the beam encoding/fusion for the wider input.
            beam_param_dims = self.positional_direction_encoding.encoded_dims + self.d_encoded_spectra + 2 * self.scalar_encoding_dims
            if self.use_beam_shape:
                beam_param_dims += self.scalar_encoding_dims
            token_component_dims = [self.positional_direction_encoding.encoded_dims, self.scalar_encoding_dims, self.d_encoded_spectra]
            if self.use_beam_shape:
                token_component_dims.append(self.scalar_encoding_dims)
            token_component_dims.append(self.scalar_encoding_dims)
            self.configure_beam_encoding(
                self.conditioning,
                beam_param_dims,
                self.d_model,
                token_component_dims=token_component_dims if self._token_attention else None,
            )

        @property
        def location_feature_dims(self) -> int:
            base = super().location_feature_dims
            if getattr(self, "_relative_translation", False):
                return base + self.relative_translation_encoding.encoded_dims
            return base

        def encode_location(self, batch: PositionalInput, region_state: Tensor | None = None) -> Tensor:
            base = super().encode_location(batch, region_state)
            if not getattr(self, "_relative_translation", False):
                return base
            dtype = self._compute_dtype
            translation = getattr(batch, "translation", None)
            assert translation is not None, "TPBRFNet requires a TranslationalInput with a translation tensor."
            if translation.ndim == 1:
                translation = translation.unsqueeze(0)
            t = translation[:, :2].to(dtype)
            if t.shape[0] != batch.position.shape[0]:
                assert t.shape[0] == 1, (
                    f"translation rows ({t.shape[0]}) must match the query positions "
                    f"({batch.position.shape[0]}) or be a single field")
                t = t.expand(batch.position.shape[0], -1)
            # patient-frame coordinate of the query point (learned frame alignment, see __init__)
            rel = batch.position[:, :2].to(dtype) - (t * self.translation_frame_scale.to(dtype) + self.translation_frame_shift.to(dtype))
            return torch.cat([base, self.relative_translation_encoding(rel)], dim=-1)

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
            if getattr(self, "_relative_translation", False):
                # translation is consumed per-point in encode_location; the beam vector is
                # exactly PBRFNet's. Still require the translation up front so a wrong dataset
                # fails at the beam-encoding stage, not mid-trunk.
                assert getattr(batch, "translation", None) is not None, \
                    "TPBRFNet requires a TranslationalInput with a translation tensor."
                return super().encode_additional_parameters(batch)
            dtype = self._compute_dtype
            assert batch.origin.shape[-1] == 1, f"Origin must be a single distance value for TPBRFNet. Got shape: {batch.origin.shape}"
            translation = getattr(batch, "translation", None)
            assert translation is not None, "TPBRFNet requires a TranslationalInput with a translation tensor."
            if translation.ndim == 1:
                translation = translation.unsqueeze(0)  # single (3,) -> (1, 3)

            dir_enc = self.positional_direction_encoding(batch.direction.to(dtype))
            origin_enc = self.distance_encoder(batch.origin.to(dtype))
            spectrum = self.spectra_encoder(batch.spectrum.to(dtype))
            translation_enc = self.translation_encoder(translation[:, :2].to(dtype))  # (x, y) only
            opening_angle = None
            if self.use_beam_shape:
                opening_angle = self.opening_angle_encoder(self._beam_shape_features(batch.beam_shape_parameters.to(dtype))).view(batch.spectrum.shape[0], -1)

            enc = [dir_enc, origin_enc, spectrum, opening_angle, translation_enc] if opening_angle is not None else [dir_enc, origin_enc, spectrum, translation_enc]
            if getattr(self, "_token_attention", False):
                return torch.cat(enc, dim=-1)

            return self.beam_encoder(enc)

    def _augment_random_input(self, x, device, batch_size: int):
        """Random inputs must carry a translation: every consumer of them (batch-size search,
        ONNX export, plot callbacks) runs encode_additional_parameters directly. Sampled in the
        NORMALIZED [0, 1] domain the TranslationNormalization delivers."""
        return positional_like(x, **x._asdict(),
                               translation=torch.rand(batch_size, 3, device=device))

    def get_custom_parameters(self):
        # Persist the RESOLVED translation encoding into the stored config: a checkpoint saved
        # today reloads identically even if the class default changes later.
        params = super().get_custom_parameters()
        cp = dict(params["conditioning_params"] or {})
        cp["translation_encoding"] = dict(self.get_core_model()._translation_encoding_params)
        params["conditioning_params"] = cp
        return params

    def deploy_interface(self):
        """PBRFNet's interface plus the one input this model adds."""
        from radfield3dnn.deploy import ModelInput
        iface = super().deploy_interface()
        iface.inputs |= ModelInput.PATIENT_TRANSLATION_3D
        return iface

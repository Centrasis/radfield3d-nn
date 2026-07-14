import torch
import torch.nn as nn
from torch import Tensor

from .nerf import PBRFNet
from radfield3dnn.rftypes import PositionalInput
# The patient translation (vec3, in metres) is loaded from field metadata by RadFiled3D's
# RadField3DTranslationDataset, which yields this TranslationalInput.
from RadFiled3D.pytorch.types import TranslationalInput


class TPBRFNet(PBRFNet):
    """
    Translational Parametric Beam Radiation Field Network
    PBRFNet with one additional beam parameter: the patient translation (vec3, in metres) read from
    the field metadata. Only the in-plane (x, y) components are encoded — z (couch height) is held
    constant across the dataset. Requires a TranslationalInput; the network is otherwise identical
    to PBRFNet, so configs/normalizer/losses carry over unchanged.
    """
    __model_name__ = "TPBRFNet"

    class BackboneModel(PBRFNet.BackboneModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.translation_encoder = nn.Sequential(
                nn.Linear(2, self.scalar_encoding_dims),  # (x, y) of the vec3 translation; z is constant
                nn.SiLU(True),
                nn.Linear(self.scalar_encoding_dims, self.scalar_encoding_dims),
                nn.SiLU(True)
            )
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

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
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

    def deploy_interface(self):
        """PBRFNet's interface plus the one input this model adds."""
        from radfield3dnn.deploy import ModelInput
        iface = super().deploy_interface()
        iface.inputs |= ModelInput.PATIENT_TRANSLATION_3D
        return iface

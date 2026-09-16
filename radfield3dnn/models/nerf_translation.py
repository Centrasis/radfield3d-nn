import torch
import torch.nn as nn
from torch import Tensor

from .nerf import PBRFNet
from .encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding
from .encoders.factory import build_encoding
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
        {"type": "relative", "encoding": {"type": "hash", "input_domain": [-0.5, 1.5], ...}}
            DUAL-FRAME mode. The physics: the field stays ISOCENTRIC (source/beam fixed in the
            world frame); the translation moves the PATIENT, i.e. what changes is the
            intersection of the fixed beam with the shifted geometry. The natural inductive bias
            is therefore not "translation as a global beam parameter" but a second PER-POINT
            coordinate: the trunk sees the world-frame position AND the patient-frame coordinate
            ``position_xy − frame(translation_xy)`` (encoded, appended to the location features).
            The patient-frame encoder is any registry type (``encoders/factory.py``) given as
            the nested ``encoding`` dict; ``n_frequencies`` / ``append_input`` are the shorthand
            for the sinusoidal default and cannot be combined with ``encoding``. A hash grid
            here is a learned 2-D feature map that moves WITH the patient — the phantom's
            footprint stored once, in its own frame. It sees ``rel`` with the frame at init in
            [−0.5, 1.5], hence its ``input_domain``. ``frame`` is a learnable per-axis affine (init scale 1, shift −0.5) that
            maps the [0,1]-normalized translation onto the position's coordinate scale — the two
            normalizations use different physical ranges, so the alignment is learned. In this
            mode the translation LEAVES the beam vector (the beam encoder is exactly PBRFNet's),
            and the deploy package exports as a single graph (the split trunk would need the
            translation as a third input, which the C++ runtime does not bind yet).
    ``conditioning_params["beam_geometry"]`` (PBRFNet's block, see ``PBRFNet.BackboneModel.
    _configure_beam_geometry``) keeps all of the above inputs and additionally relates them INSIDE
    the graph: per query point the beam-frame coordinate (depth along the central ray, lateral
    offsets in collimated half-widths) and the patient-frame position ``x − t``; per field the
    central ray's angle/distance to the patient origin (a beam-vector component); plus an analytic
    free-air direct-beam prior gated by a learned transmission. It needs the physical ranges the
    inputs were normalised with (``definition_file`` at build time, persisted as ``physics``) and
    supersedes the ``relative`` mode — the two cannot be combined.
    Expects the translation NORMALIZED to [0, 1] (TranslationNormalization, built from the dataset
    definition file) so the frequency ladder 2^i·π·x spans the intended band.
    """
    __model_name__ = "TPBRFNet"

    class BackboneModel(PBRFNet.BackboneModel):
        provides_translation = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trans_enc_params = dict(self._conditioning_params.get(
                "translation_encoding", {"type": "fourier", "n_frequencies": 6, "append_input": True}))
            enc_type = trans_enc_params.get("type", "fourier")
            self._relative_translation = (enc_type == "relative")
            if enc_type == "relative":
                if self.beam_geometry is not None:
                    raise ValueError("translation_encoding 'relative' and beam_geometry both derive the patient-frame "
                                     "coordinate; beam_geometry.patient_frame supersedes the relative mode — pick one.")
                # DUAL-FRAME: patient-frame coordinate per query point, appended to the trunk's
                # location features; the beam vector stays PBRFNet's (super() configured it).
                self.translation_encoder = None
                self.relative_translation_encoding = build_encoding(
                    self._relative_encoding_spec(trans_enc_params),
                    default_d_input=2,  # (x, y); z is constant across the dataset
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
            # the beam vector gained the translation component: rebuild the beam encoding/fusion
            self._configure_beam_encoding_from_components()

        @staticmethod
        def _translation_rows(batch: PositionalInput) -> Tensor:
            translation = getattr(batch, "translation", None)
            assert translation is not None, "TPBRFNet requires a TranslationalInput with a translation tensor."
            return translation.unsqueeze(0) if translation.ndim == 1 else translation

        def _geometry_translation(self, batch: PositionalInput) -> Tensor:
            return self._translation_rows(batch)

        @property
        def beam_component_dims(self) -> list[int]:
            dims = super().beam_component_dims
            # absent while PBRFNet's __init__ configures the beam encoding, before this class adds its encoder
            if getattr(self, "translation_encoder", None) is not None:
                dims.append(self.scalar_encoding_dims)
            return dims

        def _beam_components(self, batch: PositionalInput) -> list[Tensor]:
            # Require the translation up front in every mode so a wrong dataset fails at the
            # beam-encoding stage, not mid-trunk.
            translation = self._translation_rows(batch)
            enc = super()._beam_components(batch)
            if getattr(self, "translation_encoder", None) is not None:
                enc.append(self.translation_encoder(translation[:, :2].to(self._compute_dtype)))  # (x, y) only
            return enc

        @property
        def trunk_consumes_beam_inputs(self) -> bool:
            return getattr(self, "_relative_translation", False) or super().trunk_consumes_beam_inputs

        @staticmethod
        def _relative_encoding_spec(trans_enc_params: dict) -> dict:
            """Encoder spec for the patient-frame coordinate: the nested ``encoding`` dict, or the
            sinusoidal shorthand (``n_frequencies`` / ``append_input``). Both at once is
            ambiguous and refused."""
            spec = trans_enc_params.get("encoding")
            shorthand = [k for k in ("n_frequencies", "append_input") if k in trans_enc_params]
            if spec is not None and shorthand:
                raise ValueError(
                    f"translation_encoding: 'encoding' cannot be combined with {shorthand} "
                    f"(those are the sinusoidal shorthand); put the encoder's own keys inside 'encoding'.")
            if spec is None:
                return {"type": "sinusoidal",
                        "pos_enc_dim": int(trans_enc_params.get("n_frequencies", 6)),
                        "append_input": bool(trans_enc_params.get("append_input", True))}
            return dict(spec)

        @property
        def location_feature_dims(self) -> int:
            base = super().location_feature_dims
            if getattr(self, "_relative_translation", False):
                return base + self.relative_translation_encoding.encoded_dims
            return base

        def encode_location(self, batch: PositionalInput, region_state: Tensor | None = None, geometry=None) -> Tensor:
            base = super().encode_location(batch, region_state, geometry)
            if not getattr(self, "_relative_translation", False):
                return base
            dtype = self._compute_dtype
            t = self._translation_rows(batch)[:, :2].to(dtype)
            if t.shape[0] != batch.position.shape[0]:
                assert t.shape[0] == 1, (
                    f"translation rows ({t.shape[0]}) must match the query positions "
                    f"({batch.position.shape[0]}) or be a single field")
                t = t.expand(batch.position.shape[0], -1)
            # patient-frame coordinate of the query point (learned frame alignment, see __init__)
            rel = batch.position[:, :2].to(dtype) - (t * self.translation_frame_scale.to(dtype) + self.translation_frame_shift.to(dtype))
            return torch.cat([base, self.relative_translation_encoding(rel)], dim=-1)

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
        inputs, outputs = super().deploy_interface()
        # Metres in the field frame; the model encodes x/y only (z is constant across the dataset).
        inputs["patient_translation"] = dict(semantic="patient_translation", unit="m")
        return inputs, outputs

import torch
import torch.nn as nn
from torch import Tensor
from radfield3dnn.models.encoders.factory import build_encoding
from radfield3dnn.models.encoders.spectra_factory import build_spectra_encoding
from .feedforward import FeedforwardPointwiseModel
from radfield3dnn.rftypes import AirKermaField, RadiationField, RadiationFieldChannel, PositionalInput, DirectionalInput
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from .base import ModuleBuilder
from radfield3dnn.optim import OptimizerBehaviour, CosineWithWarmup
from typing import Union, Literal
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping, SoftClip, LogitSigmoid, IdentityFlux
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
from radfield3dnn.preprocessing.normalizations.asinh import AsinhTonemapNormalizer
from radfield3dnn.models.layers import FiLM, Concat, GatedFusion, ResidualFiLM, ConcatLinear, CrossAttentionFusion, TokenCrossAttentionFusion
from radfield3dnn.preprocessing.normalizations.base import Normalizer


# Variance-preserving init gain for SiLU hidden layers: 1/sqrt(E[silu(x)^2]) for x~N(0,1)
# (E[silu(x)^2] = 0.35577, measured to 5e-5; analogue of ReLU's sqrt(2)).
SILU_GAIN = 1.6765


class RFNetBase(FeedforwardPointwiseModel):
    @property
    def output_head_markers(self) -> tuple[str, ...]:
        # Shared backbone feeds two output heads on the backbone model.
        return ("spectra_decoder", "flux_decoder")

    def _init_weights(self, module: nn.Module):
        # Hidden layers: SiLU-optimal gain (the trunk/decoder hidden activations are SiLU),
        # biases plainly zero everywhere. The only special-cased bias is the flux OUTPUT
        # layer, set from the flux activation in apply_weights_init.
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=SILU_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _maybe_cast_to_precision(self):
        # Init runs in fp32, then round to fp16. self.half() also casts the encoding
        # buffers (frequencies / SH coefficients), which is acceptable.
        if getattr(self, "_precision", "fp32") == "fp16":
            self.half()

    # fp16 fp32-master-weight RUNTIME hooks now live on BaseNeuralRadFieldModel; the master *setup*
    # is wired by the optimizer behaviour (radfield3dnn/optim) below.
    @property
    def optimizer_behaviour(self) -> OptimizerBehaviour:
        """The encapsulated optimizer + LR-schedule behaviour (default: linear warmup → cosine). It
        builds the configure_optimizers() result and sets up the fp16 fp32-master-weights; the
        matching per-step hooks live on BaseNeuralRadFieldModel."""
        if getattr(self, "_optimizer_behaviour", None) is None:
            self._optimizer_behaviour = CosineWithWarmup()
        return self._optimizer_behaviour

    def configure_optimizers(self):
        return self.optimizer_behaviour.configure(self)



class RFBackboneModel(nn.Module):
    def __init__(self, d_model=256, out_spectra_dim=32, normalizer=None, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                 location_encoding_params: dict = None,
                 direction_encoding_params: dict = None,
                 conditioning_params: dict = None,
                 trunk_depth: int = 4,
                 flux_head_hidden: int = 1):
        super().__init__()
        self._flux_head_hidden = int(flux_head_hidden)
        self.d_model = d_model
        self.out_spectra_dim = out_spectra_dim
        self._precision = precision
        self._flux_activation_kind = flux_activation

        # Location/direction encoders are built from self-contained ``{"type", **kwargs}`` dicts via the
        # encoder factory (each diverging encoder is its own class); a None dict falls back to the
        # default sinusoidal/spherical-harmonic encoders.
        if location_encoding_params is None:
            location_encoding_params = {"type": "sinusoidal", "pos_enc_dim": 10, "append_input": True}
        if direction_encoding_params is None:
            direction_encoding_params = {"type": "spherical_harmonics", "degree": 4, "append_input": True}
        self._location_encoding_params = dict(location_encoding_params)
        self._direction_encoding_params = dict(direction_encoding_params)
        self._location_encoding_type = location_encoding_params["type"]

        self.positional_location_encoding = build_encoding(location_encoding_params, default_d_input=3)
        self.positional_direction_encoding = build_encoding(direction_encoding_params, default_d_input=3)

        # Conditioning (beam->trunk fusion) is selected by ``conditioning_params["type"]``; the dict may
        # also carry ``use_beam_shape`` (read by PBRFNet.BackboneModel, ignored at this level).
        if conditioning_params is None:
            conditioning_params = {"type": "None"}
        self._conditioning_params = dict(conditioning_params)
        conditioning = conditioning_params["type"]

        self._normalizer: Normalizer = normalizer
        self.activation_fn = nn.SiLU(inplace=True)
        self.configure_beam_encoding(conditioning, self.positional_direction_encoding.encoded_dims, self.d_model)

        self.decoder_in_dim = d_model

        self.block1 = nn.Sequential(
            nn.Linear(self.positional_location_encoding.encoded_dims, d_model) if self.use_conditioning else nn.Linear(self.positional_location_encoding.encoded_dims + d_model, d_model),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model),  # LayerNorm only if not using FiLM
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.SiLU(True) if self.use_conditioning else nn.Identity()
        )

        if self.use_layer_norm:
            _block2_in = d_model * 2 + self.positional_location_encoding.encoded_dims
        else:
            _block2_in = d_model

        if int(trunk_depth) < 2:
            raise ValueError(f"trunk_depth must be >= 2, got {trunk_depth}")
        _block2_layers: list[nn.Module] = [
            nn.Linear(_block2_in, d_model),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model),
            nn.SiLU(True),
        ]
        for _ in range(int(trunk_depth) - 2):
            _block2_layers += [nn.Linear(d_model, d_model), nn.SiLU(True)]
        _block2_layers += [
            nn.Linear(d_model, d_model),
            nn.SiLU(True) if self.use_conditioning else nn.Identity(),
        ]
        self.block2 = nn.Sequential(*_block2_layers)

        self.spectra_decoder = nn.Sequential(
            nn.Linear(self.decoder_in_dim, d_model // 2),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model // 2),
            nn.SiLU(True),
            nn.Linear(d_model // 2, self.out_spectra_dim)
        )
        # Flux head depth = ``flux_head_hidden`` SiLU-separated hidden Linears before the scalar
        # projection. Default 1 (Linear+SiLU+Linear): the HDR flux distribution (~87% empty +
        # sharp peaks) is too non-linear for a single matrix-vector projection; one SiLU lets the
        # head model the bimodal "beam present / not present" decision. ``flux_head_hidden=0``
        # gives a plain single-Linear head.
        _flux_layers: list[nn.Module] = []
        for _ in range(self._flux_head_hidden):
            _flux_layers += [nn.Linear(d_model, d_model), nn.SiLU(True)]
        _flux_layers += [nn.Linear(d_model, 1)]
        self.flux_decoder = nn.Sequential(*_flux_layers)
        
        self.spectra_activation = HistogramNormalize(dim=-1)  # nn.Softmax(dim=-1)
        # Flux activations. The output-layer bias is decided by the chosen activation
        # (its `init_bias` = the pre-activation putting the initial output at the codomain
        # midpoint / max-gradient point); see apply_weights_init.
        # Selectable flux activations:
        #   "clamp"   — gradient-conserving hard clamp matching the normalizer's codomain
        #               endpoints. Can lock into "predict 0 forever" once z is pushed below
        #               the lower clamp.
        #   "softclip" — 0.5 * (tanh(z) + 1), smooth in (0,1). Avoids the lock-in (gradient
        #               nonzero everywhere on R). Not valid for the [-1,1] codomain; falls
        #               back to clamp there with a warning.
        #   "sigmoid"  — y = sigmoid(z), logit clamped (grad-conserving) to ±30. Spans the
        #               full (0,1) codomain so crushed scatter AND the peak stay representable
        #               with a recovery gradient everywhere. Requires the (0,1) codomain.
        # The sigmoid / softclip heads only need a (0,1) codomain — any normalizer that declares
        # range=(0,1) qualifies (LinearNormalizer(0,1) AND the asinh tonemap, whose codomain is
        # [0,1]), so gate on the declared range rather than the concrete normalizer class.
        if flux_activation in ("none", "identity"):
            # Unbounded raw-logit head: no clamp/squash. The target may be the asinh tonemap
            # ([0,1] codomain) but the head is left free to emit unbounded logits.
            self.flux_activation = IdentityFlux()
        elif flux_activation == "sigmoid":
            if getattr(self._normalizer, "range", None) == (0.0, 1.0):
                self.flux_activation = LogitSigmoid(logit_range=30.0)
            else:
                print(f"Warning: sigmoid requires normalizer.range=(0,1); falling back to clamp for {self._normalizer.__class__}.")
                self.flux_activation = GradientConservingClamping(0.0, 1.0)
        elif flux_activation == "softclip":
            if getattr(self._normalizer, "range", None) == (0.0, 1.0):
                self.flux_activation = SoftClip(k=1.0)
            else:
                print(f"Warning: SoftClip requires normalizer.range=(0,1); falling back to clamp for {self._normalizer.__class__}.")
                self.flux_activation = GradientConservingClamping(0.0, 1.0)
        elif issubclass(self._normalizer.__class__, LinearNormalizer):
            if self._normalizer.range[0] == -1.0 and self._normalizer.range[1] == 1.0:
                self.flux_activation = GradientConservingClamping(-1.0, 1.0)
            elif self._normalizer.range[0] == 0.0 and self._normalizer.range[1] == 1.0:
                self.flux_activation = GradientConservingClamping(0.0, 1.0)
            else:
                raise ValueError(f"Unsupported normalization range for LinearNormalizer: {self._normalizer.range}")
        elif isinstance(self._normalizer, AsinhTonemapNormalizer):
            # asinh tonemap codomain is [0,1] (y = asinh(x/σ)/asinh(1/σ), bounded), so the head clamps
            # to [0,1] — same range as linear0_1, but the targets are tonemapped (HDR-spread). The
            # activation's init_bias (0.5, midpoint) starts the head in its high-gradient interior.
            self.flux_activation = GradientConservingClamping(0.0, 1.0)
        else:
            print(f"Warning: Using default flux activation (0.0, 1.0) clamping for unknown normalizer: {self._normalizer.__class__}.")
            self.flux_activation = GradientConservingClamping(0.0, 1.0)

    @property
    def _compute_dtype(self) -> torch.dtype:
        return torch.float16 if self._precision == "fp16" else torch.float32

    def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
        direction = batch.direction.to(self._compute_dtype)
        dir_enc = self.positional_direction_encoding(direction)
        beam_encoded = self.beam_encoder([dir_enc])
        return beam_encoded

    # Beam→trunk fusion registry: name -> factory(d_model, activation_fn) building one fusion that
    # merges a trunk feature (x) with the beam latent (cond), returning dim(x) (FusionBase contract).
    # Two of these become beam_conditioner1/2 in forward(); each counts as one logical trunk layer.
    _FUSION_FACTORIES = {
        "FiLM":      lambda d, act: FiLM(d, d, non_linearity=act),
        "ResFiLM":   lambda d, act: ResidualFiLM(d, d, non_linearity=act),
        "Gated":     lambda d, act: GatedFusion(d, d, hidden=d, non_linearity=act),
        "Concat":    lambda d, act: ConcatLinear(d, d, non_linearity=act),
        "Attention": lambda d, act: CrossAttentionFusion(d, d, n_heads=4, n_tokens=4),
    }

    def configure_beam_encoding(self, conditioning: Literal["None", "FiLM", "ResFiLM", "Gated", "Concat", "Attention", "TokenAttention"], beam_param_dims: int, d_model: int, token_component_dims: list[int] = None):
        self.conditioning = conditioning
        activation_fn = type(self.activation_fn)
        self.beam_conditioner1 = None
        self.beam_conditioner2 = None
        self.use_layer_norm = False
        self.beam_encoder = nn.Sequential(
            Concat(dim=-1),
            nn.Linear(beam_param_dims, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model)
        )
        if conditioning == "None":
            self.use_conditioning = False
            self.use_layer_norm = True
        elif conditioning == "TokenAttention":
            self.use_conditioning = True
            if token_component_dims is not None:
                self.beam_conditioner1 = TokenCrossAttentionFusion(token_component_dims, d_model, n_heads=4)
                self.beam_conditioner2 = TokenCrossAttentionFusion(token_component_dims, d_model, n_heads=4)
        elif conditioning in self._FUSION_FACTORIES:
            self.use_conditioning = True
            make = self._FUSION_FACTORIES[conditioning]
            self.beam_conditioner1 = make(d_model, activation_fn)
            self.beam_conditioner2 = make(d_model, activation_fn)
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning!r}. "
                             f"Valid: 'None', 'TokenAttention', {', '.join(repr(k) for k in self._FUSION_FACTORIES)}.")
        
    def decode_results(self, model_output: Tensor) -> RadiationField | AirKermaField:
        spectra = self.spectra_decoder(model_output)
        flux = self.flux_decoder(model_output).squeeze(-1)

        spectra = self.spectra_activation(spectra)
        # No additive flux offset: the flux_decoder's final Linear bias learns any
        # required shift, so the output prior is set by initialization, not a constant.
        flux = self.flux_activation(flux)

        # Boundary cast: losses, normalizers and metrics downstream of the model
        # consume fp32 unconditionally. Cast here so the precision switch is
        # invisible past this point.
        spectra = spectra.float()
        flux = flux.float()

        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=spectra,
                flux=flux,
                error=None
            ),
            direct_beam=None
        )

    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
        position = batch.position.to(self._compute_dtype)
        xyz_enc = self.positional_location_encoding(position)
        params_enc = self.encode_additional_parameters(batch) if global_parameters is None else global_parameters.to(self._compute_dtype)

        x0 = xyz_enc if self.use_conditioning else torch.cat((xyz_enc, params_enc), dim=-1)
        x1 = self.block1(x0)

        # Second block with skip connection
        if self.use_conditioning:
            x1_1 = self.beam_conditioner1(x1, params_enc)
        else:
            x1 = self.activation_fn(x1)
            x1_1 = torch.cat((x1, x0), dim=-1)

        x2 = self.block2(x1_1)

        if self.use_conditioning:
            x2 = self.beam_conditioner2(x2, params_enc)
        else:
            x2 = self.activation_fn(x2)

        # Terminal additive skip connection.
        x2_1 = x2 + x1
        return self.decode_results(x2_1)


class SRBFNet(RFNetBase):
    """
    Static Rotatable Beam Field Network
    A NeRF-based architecture for learning implicit radiation fields with static radiation field, but rotated beam.
    """
    __model_name__ = "SRBFNet"
        
    class IndexSelectableList(list):
        """
        IndexSelectableList is a decorator for an interatable (preferably a list) of tensors.
        This decorator is needed to allow for the 'index_select' call as required by FeedforwardPointwiseModel.forward method.
        """
        def __init__(self, inner_list: list):
            self.inner_list = inner_list

        @staticmethod
        def recursive_index_select(dim: int, batch_idx: int, target: list | Tensor) -> list | Tensor:
            if isinstance(target, Tensor):
                return target.index_select(dim, batch_idx)
            else:
                return [
                    SRBFNet.IndexSelectableList.recursive_index_select(dim, batch_idx, tl)
                    for tl in target
                ]

        def index_select(self, dim: int, batch_idx: int) -> list:
            return SRBFNet.IndexSelectableList.recursive_index_select(dim, batch_idx, self.inner_list)
        
        def __getitem__(self, idx):
            return self.inner_list[idx]
        
        def __len__(self):
            return len(self.inner_list)
        
        def __iter__(self):
            return self.inner_list.__iter__()
        

    def forward2volume(self, x: DirectionalInput, voxel_counts, spectra_bins = 32, mask: Union[Tensor, None] = None):
        assert spectra_bins == self.out_spectra_dim, f"Output spectra bins must match the model's output dimension. Given: {spectra_bins}, expected: {self.out_spectra_dim}"
        # drop geometry if present to speed up training, as this network is learning only implicit geometry
        x = DirectionalInput(
            direction=x.direction,
            spectrum=x.spectrum,
            geometry=None,
            origin=x.origin,
            beam_shape_parameters=x.beam_shape_parameters,
            beam_shape_type=x.beam_shape_type
        )
        global_parameters = self.backbone_model.encode_additional_parameters(x)
        return super().forward2volume(x, voxel_counts, self.out_spectra_dim, mask=mask, global_parameters=global_parameters)

    def __init__(self, d_model=256, out_spectra_dim=32, flux_loss="SMAPEBalanced", spectrum_loss="HistogramLoss", normalizer=None, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                 location_encoding_params: dict = None, direction_encoding_params: dict = None, conditioning_params: dict = None, training_params: dict = None, trunk_depth: int = 4, flux_head_hidden: int = 1):
        # Optimization/sampling knobs (learning_rate, max_lr, voxel-sampling flags) are folded into one
        # ``training_params`` dict; the individual values are unpacked here and threaded into the shared
        # FeedforwardPointwiseModel base (whose signature is kept stable for the other model families).
        if training_params is None:
            training_params = {}
        learning_rate = training_params.get("learning_rate", 1e-3)
        randomize_voxel_location_in_training = training_params.get("randomize_voxel_location_in_training", True)
        voxels_centered_around_origin = training_params.get("voxels_centered_around_origin", True)
        max_lr = training_params.get("max_lr", 5e-4)
        super().__init__(
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            normalizer=normalizer
        )
        self._training_params = {
            "learning_rate": learning_rate,
            "randomize_voxel_location_in_training": randomize_voxel_location_in_training,
            "voxels_centered_around_origin": voxels_centered_around_origin,
            "max_lr": max_lr,
        }

        if conditioning_params is None:
            conditioning_params = {"type": "None"}
        self.out_spectra_dim = out_spectra_dim
        self._conditioning_params = dict(conditioning_params)
        self.conditioning = conditioning_params["type"]
        self.d_model = d_model
        self._precision = precision
        self._max_lr = float(max_lr)
        self._flux_activation_kind = flux_activation
        self._location_encoding_params = location_encoding_params
        self._direction_encoding_params = direction_encoding_params
        self._trunk_depth = int(trunk_depth)
        self._flux_head_hidden = int(flux_head_hidden)

        self.flux_loss_name = flux_loss
        self.spectrum_loss_name = spectrum_loss

        self._flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)

        self.backbone_model = RFBackboneModel(
            d_model=d_model,
            out_spectra_dim=out_spectra_dim,
            conditioning_params=conditioning_params,
            normalizer=normalizer,
            precision=precision,
            flux_activation=flux_activation,
            location_encoding_params=location_encoding_params,
            direction_encoding_params=direction_encoding_params,
            trunk_depth=trunk_depth,
            flux_head_hidden=flux_head_hidden,
        )

        self.apply_weights_init()
        self._maybe_cast_to_precision()

    def get_core_model(self) -> nn.Module:
        return self.backbone_model
        
    def apply_weights_init(self):
        # One uniform scheme: SiLU-gain weights + zero bias everywhere (_init_weights). The
        # FINAL Linear of each decoder is an output projection (no SiLU after it), so it gets
        # the standard gain-1 init instead of the SiLU gain.
        self.apply(self._init_weights)
        for head in (self.backbone_model.spectra_decoder, self.backbone_model.flux_decoder):
            out_linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            if out_linears:
                nn.init.xavier_uniform_(out_linears[-1].weight, gain=1.0)

        # Flux OUTPUT bias is decided by the actual flux activation: each activation reports the
        # pre-activation value (init_bias) that places the initial output at its high-gradient
        # point. A zero bias on a clamp(0,1) head would start at the clamp FLOOR, where the
        # peak-crushed scatter band gets no gradient -> predict-0 lock-in. The exact value also
        # depends on the NORMALIZER (linear → just above the floor; log-like → codomain midpoint);
        # see flux_activations.flux_head_init_bias.
        flux_linears = [m for m in self.backbone_model.flux_decoder.modules() if isinstance(m, nn.Linear)]
        if flux_linears and flux_linears[-1].bias is not None:
            from radfield3dnn.models.activations.flux_activations import flux_head_init_bias
            bias = flux_head_init_bias(self.backbone_model.flux_activation, self._normalizer)
            nn.init.constant_(flux_linears[-1].bias, bias)

        if self.backbone_model.beam_conditioner1 is not None:
            self.backbone_model.beam_conditioner1.initialize()
        if self.backbone_model.beam_conditioner2 is not None:
            self.backbone_model.beam_conditioner2.initialize()

    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
        return self.backbone_model.forward(batch, global_parameters=global_parameters)
    
    def get_custom_parameters(self):
        return {
            "d_model": self.d_model,
            "out_spectra_dim": self.out_spectra_dim,
            "conditioning_params": self._conditioning_params,
            "flux_loss": self.flux_loss_name,
            "spectrum_loss": self.spectrum_loss_name,
            "training_params": self._training_params,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None,
            "precision": self._precision,
            "flux_activation": self._flux_activation_kind,
            "location_encoding_params": self._location_encoding_params,
            "direction_encoding_params": self._direction_encoding_params,
            "trunk_depth": self._trunk_depth,
            "flux_head_hidden": self._flux_head_hidden,
        }


class SPERFNet(SRBFNet):
    """
    Spectral Enhanced Radiation Field Network
    A NeRF-based architecture for learning implicit radiation fields with spectral encoding and beam rotation.
    """
    __model_name__ = "SPERFNet"

    class BackboneModel(RFBackboneModel):
        def __init__(self, d_model=256, out_spectra_dim=32, normalizer=None, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                     location_encoding_params: dict = None, direction_encoding_params: dict = None,
                     spectra_encoding_params: dict = None, conditioning_params: dict = None, trunk_depth: int = 4, flux_head_hidden: int = 1):
            super().__init__(d_model=d_model, out_spectra_dim=out_spectra_dim, normalizer=normalizer, conditioning_params=conditioning_params, precision=precision, flux_activation=flux_activation, location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden)
            # Spectrum encoder built from a self-contained ``{"type", **kwargs}`` dict via the spectra
            # factory; the chosen encoder exposes its output width as ``encoded_dims`` (``projector``
            # keeps the raw-spectrum dim; ``simple`` bottlenecks).
            if spectra_encoding_params is None:
                spectra_encoding_params = {"type": "projector", "in_spectra_dim": 150, "out_spectra_dim": 150}
            self._spectra_encoding_params = dict(spectra_encoding_params)
            self.spectra_encoder = build_spectra_encoding(spectra_encoding_params)
            self.d_encoded_spectra = self.spectra_encoder.encoded_dims
            d_beam_parameters_features = self.positional_direction_encoding.encoded_dims + self.d_encoded_spectra
            self.configure_beam_encoding(
                self.conditioning,
                d_beam_parameters_features,
                d_model
            )

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
            spectrum = self.spectra_encoder(batch.spectrum.to(self._compute_dtype))
            dir_enc = self.positional_direction_encoding(batch.direction.to(self._compute_dtype))
            beam_params = self.beam_encoder([dir_enc, spectrum])
            return beam_params

    def __init__(self, d_model=256, out_spectra_dim=32, flux_loss="SMAPEBalanced", spectrum_loss="HistogramLoss", normalizer=None, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                 location_encoding_params: dict = None, direction_encoding_params: dict = None, spectra_encoding_params: dict = None, conditioning_params: dict = None, training_params: dict = None, trunk_depth: int = 4, flux_head_hidden: int = 1):
        super().__init__(
            d_model=d_model,
            out_spectra_dim=out_spectra_dim,
            flux_loss=flux_loss,
            spectrum_loss=spectrum_loss,
            training_params=training_params,
            conditioning_params=conditioning_params,
            normalizer=normalizer,
            precision=precision,
            flux_activation=flux_activation,
            location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden,
        )

        self.backbone_model = SPERFNet.BackboneModel(
            d_model=d_model,
            conditioning_params=conditioning_params,
            spectra_encoding_params=spectra_encoding_params,
            normalizer=normalizer,
            out_spectra_dim=out_spectra_dim,
            precision=precision,
            flux_activation=flux_activation,
            location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden,
        )
        self.d_encoded_spectra = self.backbone_model.d_encoded_spectra
        self.d_beam_parameters_features = self.backbone_model.positional_direction_encoding.encoded_dims + self.d_encoded_spectra

        self.apply_weights_init()
        self._maybe_cast_to_precision()

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        params["spectra_encoding_params"] = self.backbone_model._spectra_encoding_params
        return params


class PBRFNet(SPERFNet):
    """
    Parametric Beam Radiation Field Network
    A NeRF-based architecture for learning implicit radiation fields with spectral encoding and parametric beam modeling and rotation.
    """
    __model_name__ = "PBRFNet"

    class BackboneModel(SPERFNet.BackboneModel):
        def __init__(self, d_model=256, out_spectra_dim=32, normalizer=None, scalar_encoding_dims=16, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                     location_encoding_params: dict = None, direction_encoding_params: dict = None,
                     spectra_encoding_params: dict = None, conditioning_params: dict = None, trunk_depth: int = 4, flux_head_hidden: int = 1):
            super().__init__(d_model=d_model, out_spectra_dim=out_spectra_dim, normalizer=normalizer, conditioning_params=conditioning_params, spectra_encoding_params=spectra_encoding_params, precision=precision, flux_activation=flux_activation, location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden)
            # use_beam_shape lives in conditioning_params (folded with the fusion ``type``).
            use_beam_shape = bool(self._conditioning_params.get("use_beam_shape", False))
            self.opening_angle_encoder = nn.Sequential(
                nn.Linear(1, scalar_encoding_dims),
                nn.SiLU(True),
                nn.Linear(scalar_encoding_dims, scalar_encoding_dims),
                nn.SiLU(True)
            ) if use_beam_shape else None
            self.distance_encoder = nn.Sequential(
                nn.Linear(1, scalar_encoding_dims),
                nn.SiLU(True),
                nn.Linear(scalar_encoding_dims, scalar_encoding_dims),
                nn.SiLU(True)
            )
            self.scalar_encoding_dims = scalar_encoding_dims
            beam_param_dims = self.positional_direction_encoding.encoded_dims + self.d_encoded_spectra + self.scalar_encoding_dims
            if use_beam_shape:
                beam_param_dims += scalar_encoding_dims
            self.use_beam_shape = use_beam_shape

            # Per-component encoded widths in the SAME order encode_additional_parameters concatenates
            # them (direction-SH, distance-MLP, spectrum-encoder, [opening-angle]). For TokenAttention the
            # conditioners (TokenCrossAttentionFusion, built inside configure_beam_encoding) split the
            # concatenated beam encodings by these widths and attend over the per-parameter tokens.
            self._token_attention = (self.conditioning == "TokenAttention")
            token_component_dims = [self.positional_direction_encoding.encoded_dims, self.scalar_encoding_dims, self.d_encoded_spectra]
            if use_beam_shape:
                token_component_dims.append(self.scalar_encoding_dims)
            self.configure_beam_encoding(
                self.conditioning,
                beam_param_dims,
                d_model,
                token_component_dims=token_component_dims if self._token_attention else None,
            )

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
            dtype = self._compute_dtype
            assert batch.origin.shape[-1] == 1, f"Origin must be a single distance value for PBRFNet. Got shape: {batch.origin.shape}"

            dir_enc = self.positional_direction_encoding(batch.direction.to(dtype))
            origin_enc = self.distance_encoder(batch.origin.to(dtype))
            spectrum = self.spectra_encoder(batch.spectrum.to(dtype))
            opening_angle = None
            if self.use_beam_shape:
                opening_angle = self.opening_angle_encoder(batch.beam_shape_parameters[:, 0].unsqueeze(-1).to(dtype)).view(batch.spectrum.shape[0], -1)

            enc = [dir_enc, origin_enc, spectrum, opening_angle] if opening_angle is not None else [dir_enc, origin_enc, spectrum]
            if getattr(self, "_token_attention", False):
                return torch.cat(enc, dim=-1)

            beam = self.beam_encoder(enc)
            return beam

    def __init__(self, d_model=256, out_spectra_dim=32, scalar_encoding_dims=16, flux_loss="SMAPEBalanced", spectrum_loss="HistogramLoss", normalizer=None, precision: Literal["fp32", "fp16"] = "fp32", flux_activation: Literal["clamp", "softclip", "sigmoid", "none"] = "clamp",
                 location_encoding_params: dict = None, direction_encoding_params: dict = None, spectra_encoding_params: dict = None, conditioning_params: dict = None, training_params: dict = None, trunk_depth: int = 4, flux_head_hidden: int = 1):
        if conditioning_params is None:
            conditioning_params = {"type": "None"}
        conditioning_params = dict(conditioning_params)
        conditioning_params.setdefault("use_beam_shape", False)
        super().__init__(d_model=d_model, out_spectra_dim=out_spectra_dim, flux_loss=flux_loss, spectrum_loss=spectrum_loss, training_params=training_params, conditioning_params=conditioning_params, normalizer=normalizer, precision=precision, flux_activation=flux_activation, location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, spectra_encoding_params=spectra_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden)
        self.scalar_encoding_dims = scalar_encoding_dims
        self.use_beam_shape = bool(conditioning_params.get("use_beam_shape", True))

        # `self.BackboneModel` resolves via MRO to the most-derived nested BackboneModel, so
        # subclasses (e.g. HDRScatterNet) get their own backbone without re-implementing __init__.
        self.backbone_model = self.BackboneModel(
            d_model=d_model,
            conditioning_params=conditioning_params,
            spectra_encoding_params=spectra_encoding_params,
            normalizer=normalizer,
            out_spectra_dim=out_spectra_dim,
            scalar_encoding_dims=scalar_encoding_dims,
            precision=precision,
            flux_activation=flux_activation,
            location_encoding_params=location_encoding_params, direction_encoding_params=direction_encoding_params, trunk_depth=trunk_depth, flux_head_hidden=flux_head_hidden,
        )

        self.apply_weights_init()
        self._maybe_cast_to_precision()

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        params["scalar_encoding_dims"] = self.scalar_encoding_dims
        return params

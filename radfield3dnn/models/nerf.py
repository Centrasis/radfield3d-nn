import math
import torch
import torch.nn as nn
from torch import Tensor
from radfield3dnn.encodings.spherical_hamonics import SphericalHarmonics
from radfield3dnn.encodings.sinusoidal_encoding import SinusoidalFrequencyEncoding
from .feedforward import FeedforwardPointwiseModel
from radfield3dnn import AirKermaField, RadiationField, RadiationFieldChannel, PositionalInput, DirectionalInput
from radfield3dnn.activations.HistogramNormalize import HistogramNormalize
from .base import ModuleBuilder
from typing import Union, Literal
from radfield3dnn.activations.fluence_activations import GradientConservingClamping
from .encoders.spectra_encoder import SpectraProjector, SimpleSpectraEncoder
from radfield3dnn.normalizations.linear import LinearNormalizer
from radfield3dnn.layers import FiLM, Concat, GatedFusion, ResidualFiLM
from radfield3dnn.normalizations.base import Normalizer


class RFNetBase(FeedforwardPointwiseModel):
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def configure_optimizers(self):
        effective_lr = max(float(self._lr), 1e-5)

        # Collect param ids of all LayerNorms (no weight decay)
        ln_param_ids = set()
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    ln_param_ids.add(id(p))

        # Separate parameters by component
        encoding_params = []
        mlp_params = []
        no_decay = []
        
        params = self.named_parameters()
        for name, param in params:
            if not param.requires_grad:
                continue

            if 'positional_location_encoding.encoding.params' in name or 'positional_direction_encoding.encoding.params' in name:
                encoding_params.append(param)
            elif ("_normalizer" in name) or ("_normalizer.m" in name) or (name.endswith(".bias")) or (id(param) in ln_param_ids):
                no_decay.append(param)
            else:
                mlp_params.append(param)

        assert len(encoding_params) + len(mlp_params) + len(no_decay) == len(list(self.parameters())), "Parameter separation error"
        
        optimizer = torch.optim.AdamW([
                {'params': encoding_params, 'lr': 1e-2, 'initial_lr': 1e-2, "weight_decay": 0.0, "eps": 1e-8},
                {'params': mlp_params, 'lr': effective_lr, 'initial_lr': effective_lr, 'weight_decay': 1e-4, "eps": 1e-8},
                {'params': no_decay, 'lr': effective_lr, 'initial_lr': effective_lr, 'weight_decay': 0.0, "eps": 1e-8},
            ],
            betas=(0.9, 0.99)
        )

        # method to extract the number of elements effectively used as one batch
        def get_accumulate_grad_batches(trainer) -> int:
            try:
                for cb in getattr(trainer, "callbacks", []):
                    if cb.__class__.__name__ == "GradientAccumulationScheduler":
                        sched = getattr(cb, "scheduling", None)
                        if isinstance(sched, dict) and sched:
                            keys = sorted(int(k) for k in sched.keys())
                            return int(sched.get(0, sched[keys[0]]))
            except Exception:
                pass
            return 1
        
        default_warmup_steps = 1000
        warmup_lr = 1e-5
        # Step-wise warmup followed by cosine decay (avoid mixing step/epoch schedulers)
        total_opt_steps = int(self.trainer.estimated_stepping_batches)
        max_epochs = int(max(self.trainer.max_epochs, 1))
        acc_batches = max(1, get_accumulate_grad_batches(self.trainer))
        if not torch.isfinite(torch.tensor(total_opt_steps)) or total_opt_steps <= 0:
            total_opt_steps = default_warmup_steps  # fallback to avoid zero division
            max_epochs = 1

        total_opt_steps /= acc_batches
        default_warmup_steps /= acc_batches
        default_warmup_steps = int(max(1, default_warmup_steps))
        steps_per_epoch = int(math.ceil(total_opt_steps / max_epochs))
        warmup_epochs = int(max(default_warmup_steps / steps_per_epoch, 1))
        warmup_steps = int(min(warmup_epochs * steps_per_epoch, max(1, total_opt_steps - 1)))
        cosine_steps = int(max(1, total_opt_steps - warmup_steps))

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_lr / effective_lr, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=5e-6
        )
        schedule = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        
        return [optimizer], [{
            "scheduler": schedule,
            "interval": "step",
            "monitor": "train_loss",
            "name": "warmup+cosine"
        }]


class RFBackboneModel(nn.Module):
    def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, out_spectra_dim=32, normalizer=None, conditioning: Literal["None", "FiLM", "ResFiLM", "AttentionConditioning", "Hypernetwork", "Gated"] = "None"):
        super().__init__()
        self.d_model = d_model
        self.out_spectra_dim = out_spectra_dim

        self.positional_location_encoding = SinusoidalFrequencyEncoding(location_encoding_dims, d_input=3, append_input=True)
        self.positional_direction_encoding = SphericalHarmonics(direction_encoding_dims, append_input=True)

        self._normalizer: Normalizer = normalizer
        self.activation_fn = nn.SiLU(inplace=True)
        self.configure_beam_encoding(conditioning, self.positional_direction_encoding.encoded_dims, self.d_model)

        if self.xyz_decoder_skip:
            if self.use_conditioning:
                self.decoder_in_dim = d_model
            else:
                self.decoder_in_dim = d_model * 2
            self.decoder_in_dim += self.positional_location_encoding.encoded_dims
        else:
            self.decoder_in_dim = d_model

        self.block1 = nn.Sequential(
            nn.Linear(self.positional_location_encoding.encoded_dims, d_model) if self.use_conditioning else nn.Linear(self.positional_location_encoding.encoded_dims + d_model, d_model),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model),  # LayerNorm only if not using FiLM
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.Identity() if not self.use_conditioning else nn.SiLU(True) # no activation and no normalization here, when using FiLM
        )

        self.block2 = nn.Sequential(
            nn.Linear(d_model, d_model) if not self.use_layer_norm else nn.Linear(d_model * 2 + self.positional_location_encoding.encoded_dims, d_model),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.SiLU(True),
            nn.Linear(d_model, d_model),
            nn.Identity() if not self.use_conditioning else nn.SiLU(True) # no activation and no normalization here, as FiLM will be applied after
        )

        self.spectra_decoder = nn.Sequential(
            nn.Linear(self.decoder_in_dim, d_model // 2),
            nn.Identity() if not self.use_layer_norm else nn.LayerNorm(d_model // 2),
            nn.SiLU(True),
            nn.Linear(d_model // 2, self.out_spectra_dim)
        )
        self.fluence_decoder = nn.Linear(d_model, 1)
        
        self.spectra_activation = HistogramNormalize(dim=-1)  # nn.Softmax(dim=-1) 
        if issubclass(self._normalizer.__class__, LinearNormalizer):
            if self._normalizer.range[0] == -1.0 and self._normalizer.range[1] == 1.0:
                self.fluence_activation = GradientConservingClamping(-1.0, 1.0)
            elif self._normalizer.range[0] == 0.0 and self._normalizer.range[1] == 1.0:
                self.fluence_activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported normalization range for LinearNormalizer: {self._normalizer.range}")
        else:
            print(f"Warning: Using default fluence activation (0.0, 1.0) clamping for unknown normalizer: {self._normalizer.__class__}.")
            self.fluence_activation = GradientConservingClamping(0.0, 1.0)

    def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
        dir_enc = self.positional_direction_encoding(batch.direction)
        beam_encoded = self.beam_encoder([dir_enc])
        return beam_encoded

    def configure_beam_encoding(self, conditioning: Literal["None", "FiLM", "ResFiLM", "AttentionConditioning", "Hypernetwork", "Gated"], beam_param_dims: int, d_model: int):
        self.conditioning = conditioning

        activation_fn = type(self.activation_fn)
        self.xyz_decoder_skip = False
        self.beam_conditioner1 = None
        self.beam_conditioner2 = None
        self.use_layer_norm = False
        self.beam_encoder = nn.Sequential(
            Concat(dim=-1),
            nn.Linear(beam_param_dims, d_model),
            nn.LayerNorm(d_model),  # to stabilize training because of the frequency encoding
            nn.SiLU(True),
            nn.Linear(d_model, d_model)
        )
        if conditioning == "FiLM":
            self.use_conditioning = True
            self.first_layer_xyz_only = True
            self.xyz_decoder_skip = False
            self.beam_conditioner1 = FiLM(d_model, d_model, non_linearity=activation_fn)
            self.beam_conditioner2 = FiLM(d_model, d_model, non_linearity=activation_fn)

        elif conditioning == "ResFiLM":
            self.use_conditioning = True
            self.first_layer_xyz_only = True
            self.beam_conditioner1 = ResidualFiLM(d_model, d_model, non_linearity=activation_fn)
            self.beam_conditioner2 = ResidualFiLM(d_model, d_model, non_linearity=activation_fn)
            
        elif conditioning == "Gated":
            self.first_layer_xyz_only = True
            self.use_conditioning = True
            self.beam_conditioner1 = GatedFusion(d_model, d_model, hidden=d_model, non_linearity=activation_fn)
            self.beam_conditioner2 = GatedFusion(d_model, d_model, hidden=d_model, non_linearity=activation_fn)

        elif conditioning == "None":
            self.first_layer_xyz_only = False
            self.use_conditioning = False
            self.use_layer_norm = True
            self.xyz_decoder_skip = False

        else:
            raise ValueError(f"Unknown conditioning type: {conditioning}")
        
    def decode_results(self, model_output: Tensor) -> RadiationField | AirKermaField:
        spectra = self.spectra_decoder(model_output)
        fluence = self.fluence_decoder(model_output).squeeze(-1)
        
        spectra = self.spectra_activation(spectra)
        fluence = self.fluence_activation(fluence)

        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=spectra,
                fluence=fluence,
                error=None
            ),
            xray_beam=None
        )
    
    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
        xyz_enc = self.positional_location_encoding(batch.position)
        params_enc = self.encode_additional_parameters(batch) if global_parameters is None else global_parameters

        x0 = xyz_enc if self.first_layer_xyz_only else torch.cat((xyz_enc, params_enc), dim=-1)
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

        # Add the skip connection and reinforce location and beam encoding
        if self.xyz_decoder_skip:
            x2_1 = torch.cat([x2, x0], dim=-1)
        else:
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

    def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, out_spectra_dim=32, fluence_loss="L1LogLoss", spectrum_loss="HistogramLoss", learning_rate: float=1e-3, randomize_voxel_location_in_training: bool = True, voxels_centered_around_origin: bool = True, normalizer=None, conditioning: Literal["None", "FiLM", "ResFiLM", "AttentionConditioning", "Hypernetwork", "Gated"] = "None"):
        super().__init__(
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            normalizer=normalizer
        )

        self.location_encoding_dims = location_encoding_dims
        self.direction_encoding_dims = direction_encoding_dims
        self.out_spectra_dim = out_spectra_dim
        self.conditioning = conditioning
        self.d_model = d_model

        self.fluence_loss_name = fluence_loss
        self.spectrum_loss_name = spectrum_loss

        self._fluence_loss_fn = ModuleBuilder.ConstructLoss_fn(fluence_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)

        self.backbone_model = RFBackboneModel(
            location_encoding_dims=location_encoding_dims,
            direction_encoding_dims=direction_encoding_dims,
            d_model=d_model,
            out_spectra_dim=out_spectra_dim,
            conditioning=conditioning,
            normalizer=normalizer
        )

        self.apply_weights_init()

    def get_core_model(self) -> nn.Module:
        return self.backbone_model
        
    def apply_weights_init(self):
        self.apply(self._init_weights)

        def _init_decoders(module):
            if isinstance(module, nn.Linear):
                # Use smaller initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.backbone_model.spectra_decoder.apply(_init_decoders)
        self.backbone_model.fluence_decoder.apply(_init_decoders)

        if self.backbone_model.beam_conditioner1 is not None:
            self.backbone_model.beam_conditioner1.initialize()
        if self.backbone_model.beam_conditioner2 is not None:
            self.backbone_model.beam_conditioner2.initialize()

    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None):
        return self.backbone_model.forward(batch, global_parameters=global_parameters)
    
    def get_custom_parameters(self):
        return {
            "location_encoding_dims": self.location_encoding_dims,
            "direction_encoding_dims": self.direction_encoding_dims,
            "d_model": self.d_model,
            "out_spectra_dim": self.out_spectra_dim,
            "conditioning": self.conditioning,
            "fluence_loss": self.fluence_loss_name,
            "spectrum_loss": self.spectrum_loss_name,
            "randomize_voxel_location_in_training": self.randomize_voxel_location_in_training,
            "voxels_centered_around_origin": self.voxels_centered_around_origin,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None
        }


class SPERFNet(SRBFNet):
    """
    Spectral Enhanced Radiation Field Network
    A NeRF-based architecture for learning implicit radiation fields with spectral encoding and beam rotation.
    """
    __model_name__ = "SPERFNet"

    class BackboneModel(RFBackboneModel):
        def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, in_spectra_dim=32, d_encoded_spectra=16, use_spectra_encoding=True, out_spectra_dim=32, normalizer=None, conditioning = "None"):
            super().__init__(location_encoding_dims, direction_encoding_dims, d_model, out_spectra_dim, normalizer, conditioning)
            # Redefine beam encoder to include spectra encoding
            self.spectra_encoder = SimpleSpectraEncoder(in_spectra_dim, d_encoded_spectra) if use_spectra_encoding else SpectraProjector(in_spectra_dim, d_encoded_spectra)
            d_beam_parameters_features = self.positional_direction_encoding.encoded_dims + d_encoded_spectra
            self.configure_beam_encoding(
                conditioning,
                d_beam_parameters_features,
                d_model
            )

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
            spectrum = self.spectra_encoder(batch.spectrum)
            dir_enc = self.positional_direction_encoding(batch.direction)
            beam_params = self.beam_encoder([dir_enc, spectrum])
            return beam_params

    def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, out_spectra_dim=32, in_spectra_dim=150, encoded_spectra_dims=64, use_spectra_encoding=False, fluence_loss="L1LogLoss", spectrum_loss="HistogramLoss", learning_rate: float=1e-3, randomize_voxel_location_in_training: bool = True, voxels_centered_around_origin: bool = True, conditioning: Literal["None", "FiLM", "ResFiLM", "AttentionConditioning", "Hypernetwork", "Gated"] = "None", normalizer=None):
        super().__init__(
            location_encoding_dims,
            direction_encoding_dims,
            d_model,
            out_spectra_dim=out_spectra_dim,
            fluence_loss=fluence_loss,
            spectrum_loss=spectrum_loss,
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            conditioning=conditioning,
            normalizer=normalizer
        )
        self.in_spectra_dim = in_spectra_dim
        self.d_encoded_spectra = encoded_spectra_dims if use_spectra_encoding else in_spectra_dim
        self.d_beam_parameters_features = self.backbone_model.positional_direction_encoding.encoded_dims + self.d_encoded_spectra
        self.use_spectra_encoding = use_spectra_encoding

        self.backbone_model = SPERFNet.BackboneModel(
            location_encoding_dims=location_encoding_dims,
            direction_encoding_dims=direction_encoding_dims,
            d_encoded_spectra=self.d_encoded_spectra,
            conditioning=conditioning,
            d_model=d_model,
            in_spectra_dim=in_spectra_dim,
            normalizer=normalizer,
            out_spectra_dim=out_spectra_dim,
            use_spectra_encoding=use_spectra_encoding
        )

        self.apply_weights_init()

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        params["encoded_spectra_dims"] = self.d_encoded_spectra
        params["use_spectra_encoding"] = self.use_spectra_encoding
        params["in_spectra_dim"] = self.in_spectra_dim
        params["conditioning"] = self.conditioning
        return params


class PBRFNet(SPERFNet):
    """
    Parametric Beam Radiation Field Network
    A NeRF-based architecture for learning implicit radiation fields with spectral encoding and parametric beam modeling and rotation.
    """
    __model_name__ = "PBRFNet"

    class BackboneModel(SPERFNet.BackboneModel):
        def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, in_spectra_dim=32, d_encoded_spectra=16, use_spectra_encoding=True, out_spectra_dim=32, normalizer=None, conditioning = "None", use_beam_shape=False, scalar_encoding_dims=16):
            super().__init__(location_encoding_dims=location_encoding_dims, direction_encoding_dims=direction_encoding_dims, d_model=d_model, out_spectra_dim=out_spectra_dim, normalizer=normalizer, conditioning=conditioning, in_spectra_dim=in_spectra_dim, d_encoded_spectra=d_encoded_spectra, use_spectra_encoding=use_spectra_encoding)
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
            beam_param_dims = self.positional_direction_encoding.encoded_dims + d_encoded_spectra + scalar_encoding_dims
            if use_beam_shape:
                beam_param_dims += scalar_encoding_dims
            self.use_beam_shape = use_beam_shape

            self.configure_beam_encoding(
                conditioning,
                beam_param_dims,
                d_model
            )

        def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
            dir_enc = self.positional_direction_encoding(batch.direction)
            assert batch.origin.shape[-1] == 1, f"Origin must be a single distance value for PBRFNet. Got shape: {batch.origin.shape}"

            origin_enc = self.distance_encoder(batch.origin)
            spectrum = self.spectra_encoder(batch.spectrum)
            if self.use_beam_shape:
                opening_angle = self.opening_angle_encoder(batch.beam_shape_parameters[:, 0].unsqueeze(-1)).view(batch.spectrum.shape[0], -1)
                enc = [dir_enc, spectrum, opening_angle, origin_enc]
            else:
                enc = [dir_enc, spectrum, origin_enc]
            beam = self.beam_encoder(enc)
            return beam

    def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, out_spectra_dim=32, in_spectra_dim=150, encoded_spectra_dims=64, scalar_encoding_dims=16, use_spectra_encoding=False, fluence_loss="L1LogLoss", spectrum_loss="HistogramLoss", learning_rate = 0.001, randomize_voxel_location_in_training = True, voxels_centered_around_origin = True, conditioning: Literal["None", "FiLM", "ResFiLM", "AttentionConditioning", "Gated"] = "None", use_beam_shape: bool = True, normalizer=None):
        super().__init__(location_encoding_dims, direction_encoding_dims, d_model, out_spectra_dim, in_spectra_dim, encoded_spectra_dims, use_spectra_encoding, fluence_loss, spectrum_loss, learning_rate, randomize_voxel_location_in_training, voxels_centered_around_origin, conditioning=conditioning, normalizer=normalizer)
        self.scalar_encoding_dims = scalar_encoding_dims
        self.use_beam_shape = use_beam_shape

        self.backbone_model = PBRFNet.BackboneModel(
            location_encoding_dims=location_encoding_dims,
            direction_encoding_dims=direction_encoding_dims,
            d_encoded_spectra=self.d_encoded_spectra,
            conditioning=conditioning,
            d_model=d_model,
            in_spectra_dim=in_spectra_dim,
            normalizer=normalizer,
            out_spectra_dim=out_spectra_dim,
            use_spectra_encoding=use_spectra_encoding,
            scalar_encoding_dims=scalar_encoding_dims,
            use_beam_shape=use_beam_shape
        )

        self.apply_weights_init()

    def get_custom_parameters(self):
        params = super().get_custom_parameters()
        params["scalar_encoding_dims"] = self.scalar_encoding_dims
        params["use_beam_shape"] = self.use_beam_shape
        return params

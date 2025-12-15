import torch
import torch.nn as nn
from torch import Tensor
from normalizations.linear import LinearNormalizer
from .base import BaseNeuralRadFieldModel
from typing import Type
from rftypes import RadiationField, RadiationFieldChannel, TrainingInputData
from activations.HistogramNormalize import HistogramNormalize
from RadFiled3D.pytorch.types import PositionalInput
from activations.fluence_activations import GradientConservingClamping
from models.encoders.spectra_encoder import SimpleSpectraEncoder
from layers.film import FiLM


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: Type[nn.Module] = nn.LeakyReLU, normalize: bool = True, raw_last_layer: bool = False):
        super().__init__()
        keep_kernel, keep_padding = 3, 1  # prevents collapse on 1×1×1 inputs
        up_kernel = 2 if stride == 2 else kernel_size
        up_padding = 0 if stride == 2 else padding
        self.block = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=keep_kernel,
                    stride=1,
                    padding=keep_padding,
                    output_padding=0
                ),
                nn.BatchNorm3d(out_channels) if normalize else nn.Identity(),
                activation() if activation is not None else nn.Identity(),
                nn.ConvTranspose3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=up_kernel,
                    stride=stride,
                    padding=up_padding,
                    output_padding=0
                ),
                nn.BatchNorm3d(out_channels) if normalize and not raw_last_layer else nn.Identity(),
                activation() if activation is not None and not raw_last_layer else nn.Identity(),
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualDeConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 0, activation: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self.block = nn.Sequential(
            *(
                [
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm3d(out_channels)
                ] + ([activation()] if activation is not None else [])
            )
        )
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: Type[nn.Module] = nn.LeakyReLU, normalize: bool = True, raw_last_layer: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm3d(out_channels) if normalize else nn.Identity(),
            activation() if activation is not None else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels) if normalize and not raw_last_layer else nn.Identity(),
            activation() if activation is not None and not raw_last_layer else nn.Identity(),
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvBase(BaseNeuralRadFieldModel):
    def __init__(self, pos_enc_dim=10, d_model=256, spectra_bins: int = 150, out_dims: tuple[int, int, int] = (50, 50, 50), learning_rate: float = 1e-3, normalizer = None):
        super().__init__(direction_encoding_dims=pos_enc_dim, d_model=d_model, learning_rate=learning_rate, normalizer=normalizer)
        self.spectra_bins = spectra_bins
        self.out_dims = out_dims
        self.pos_enc_dim = pos_enc_dim

    def forward2volume_from_training_input(self, batch: TrainingInputData, voxel_counts: Tensor, spectra_bins: int = 32) -> RadiationField:
        if not isinstance(batch, TrainingInputData):
            return self._generate_random_ground_truth(batch.direction.device)[:batch.direction.shape[0]]

        voxel_counts = batch.ground_truth.scatter_field.fluence.shape[2:] if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.fluence.shape[2:]
        spectra_bins = batch.ground_truth.scatter_field.spectrum.shape[1] if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.spectrum.shape[1]

        gt_fluence = (batch.ground_truth.scatter_field.fluence + (batch.ground_truth.xray_beam.fluence if batch.ground_truth.xray_beam is not None else 0.0) if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.fluence)
        mask = ~torch.isfinite(gt_fluence)
        if not mask.any():
            mask = None

        return self.forward2volume(batch.input, voxel_counts, spectra_bins, mask=mask)

    def generate_deconvolution(self, out_channels: int, target_shape: tuple[int, int, int], activation: Type[nn.Module] = nn.LeakyReLU, raw_last_layer: bool = False) -> nn.ModuleList:
        deconv_layers = []
        in_channels = self.d_model
        
        target_size = min(target_shape)
        num_layers = 0
        curr_size = 1
        while curr_size < target_size:
            curr_size *= 2
            num_layers += 1
        
        for i in range(num_layers - 1):
            # Gradually reduce number of channels
            out_ch = in_channels // 2 if i < num_layers - 1 else out_channels
            
            # Add deconvolution layer
            deconv_layers.append(
                DeConvBlock3D(
                    in_channels,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation=activation,
                    normalize=len(deconv_layers) >= 2
                )
            )
            
            in_channels = out_ch
        deconv_layers.append(
            DeConvBlock3D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=None,
                normalize=True
            )
        )
        
        return nn.ModuleList(deconv_layers)
    
    def generate_convolution(self, in_channels: int, in_shape: tuple[int, int, int], activation: Type[nn.Module] = nn.LeakyReLU) -> nn.ModuleList:
        conv_layers = []
        max_out_channels = self.d_model

        target_size = min(in_shape)
        num_layers = 0
        curr_size = 1
        while curr_size < target_size:
            curr_size *= 2
            num_layers += 1
        
        for i in range(num_layers - 1):
            # Gradually increase number of channels
            out_ch = in_channels * 2 if i < num_layers - 1 else max_out_channels
            out_ch = min(out_ch, max_out_channels)
            
            # Add convolution layer
            conv_layers.append(
                nn.Sequential(
                    *([nn.Conv3d(in_channels, out_ch, kernel_size=3, stride=2, padding=1)]
                     + ([nn.BatchNorm3d(out_ch)] if i < num_layers - 2 else [])
                     + [activation()]
                    )
                )
            )
            
            in_channels = out_ch
            conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, max_out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(max_out_channels),
                    activation()
                )
            )
        
        return nn.ModuleList(conv_layers)

    def forward2volume(self, x, voxel_counts: Tensor, spectra_bins: int = 32, mask: Tensor | None = None) -> RadiationField:
        pred_field: RadiationField = self(x)

        if mask is not None:
            inf_fluence = torch.full_like(pred_field.scatter_field.fluence, -torch.inf)
            inf_spectrum = torch.full_like(pred_field.scatter_field.spectrum, -torch.inf)
            spec_mask = mask.unsqueeze(1).expand_as(pred_field.scatter_field.spectrum) if len(mask.shape) == 4 else mask.expand_as(pred_field.scatter_field.spectrum)
            pred_field = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=torch.where(spec_mask, inf_spectrum, pred_field.scatter_field.spectrum) if mask is not None else pred_field.scatter_field.spectrum,
                    fluence=torch.where(mask, inf_fluence, pred_field.scatter_field.fluence) if mask is not None else pred_field.scatter_field.fluence,
                    error=pred_field.scatter_field.error
                ),
                xray_beam=RadiationFieldChannel(
                    spectrum=torch.where(spec_mask, inf_spectrum, pred_field.xray_beam.spectrum) if mask is not None else pred_field.xray_beam.spectrum,
                    fluence=torch.where(mask, inf_fluence, pred_field.xray_beam.fluence) if mask is not None else pred_field.xray_beam.fluence,
                    error=pred_field.xray_beam.error
                ) if pred_field.xray_beam is not None else None,
                geometry=pred_field.geometry
            )

        return pred_field
    
    def evaluate_forward(self, batch: TrainingInputData):
        batch = self._normalizer.forward(batch)

        if not isinstance(batch, TrainingInputData):
            return self._generate_random_ground_truth(batch.direction.device)[:batch.direction.shape[0]]

        pred_field = self.forward2volume_from_training_input(batch, None, None)
        return self.calculate_metrics(pred_field, batch.ground_truth, batch)

    def get_custom_parameters(self) -> dict:
        return {
            "spectra_bins": self.spectra_bins,
            "out_dims": self.out_dims,
            "pos_enc_dim": self.pos_enc_dim,
            "d_model": self.d_model
        }


class Beam2ScatterUNet(ConvBase):
    __model_name__ = "Beam2ScatterUNet"

    def __init__(self, d_model: int = 256, out_spectra_bins: int = 32, in_spectra_bins: int = 32, out_dims: tuple[int, int, int] = (50, 50, 50), learning_rate: float = 1e-3, normalizer=None, use_spectra: bool = True):
        super().__init__(1, d_model=d_model, spectra_bins=in_spectra_bins, out_dims=out_dims, learning_rate=learning_rate, normalizer=normalizer)
        self.out_dims = out_dims
        self.out_spectra_bins = out_spectra_bins
        self.use_spectra = use_spectra
        self.down_sampler, self.up_sampler, self.midpoint_channels = self.generate_UNet_layers(4, out_spectra_bins + 1)
        self.spectra_encoder = SimpleSpectraEncoder(in_spectra_bins, self.midpoint_channels) if use_spectra else nn.Identity()
        self.mid_spectra_film = FiLM(self.midpoint_channels, self.midpoint_channels, norm="batch3d", non_linearity=nn.LeakyReLU) if use_spectra else nn.Identity()
        self.skip_films = nn.ModuleList([
            FiLM(self.midpoint_channels, self.up_sampler[i].out_channels, norm="batch3d", non_linearity=nn.LeakyReLU) for i in range(len(self.up_sampler) - 1)
        ]) if use_spectra else nn.ModuleList([
            nn.Identity() for _ in range(len(self.up_sampler) - 1)
        ])

        self.spectra_activation = HistogramNormalize(dim=-1)
        if issubclass(self._normalizer.__class__, LinearNormalizer):
            if self._normalizer.range[0] == -1.0 and self._normalizer.range[1] == 1.0:
                self.fluence_activation = GradientConservingClamping(-1.0, 1.0) #SmoothedTanh() # nn.Softplus()  # to allow for high dynamic range
            elif self._normalizer.range[0] == 0.0 and self._normalizer.range[1] == 1.0:
                self.fluence_activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported normalization range for LinearNormalizer: {self._normalizer.range}")
        else:
            print(f"Warning: Using default fluence activation (0.0, 1.0) clamping for unknown normalizer: {self._normalizer.__class__}.")
            self.fluence_activation = GradientConservingClamping(0.0, 1.0)

        self.apply(self._init_weights)

        # final layer initialization
        last_up = self.up_sampler[-1]
        if isinstance(last_up, DeConvBlock3D):
            deconv = last_up.block[0]
            if isinstance(deconv, nn.ConvTranspose3d):
                nn.init.normal_(deconv.weight, mean=0.0, std=1e-3)
                if deconv.bias is not None:
                    nn.init.zeros_(deconv.bias)

    def _init_weights(self, m: nn.Module | None = None):
        neg_slope = 0.01
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=neg_slope, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _generate_random_ground_truth(self, device) -> RadiationField:
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=torch.randn(2, self.out_spectra_bins, *self.out_dims, device=device),
                fluence=torch.randn(2, 1, *self.out_dims, device=device),
                error=torch.randn(2, 1, *self.out_dims, device=device)
            ),
            xray_beam=RadiationFieldChannel(
                spectrum=torch.randn(2, self.out_spectra_bins, *self.out_dims, device=device),
                fluence=torch.randn(2, 1, *self.out_dims, device=device),
                error=torch.randn(2, 1, *self.out_dims, device=device)
            )
        )

    def _generate_random_input(self, device: torch.device) -> TrainingInputData:
        return TrainingInputData(
            input=PositionalInput(
                direction=torch.randn(2, 3, device=device),
                spectrum=torch.randn(2, self.spectra_bins, device=device),
                position=torch.randn(2, 3, device=device),
                origin=torch.randn(2, 3, device=device),
                geometry=None,
                beam_shape_parameters=torch.randn(2, 1, device=device),
                beam_shape_type=torch.randint(0, 2, (2, 1), device=device, dtype=torch.float32)
            ),
            ground_truth=self._generate_random_ground_truth(device)
        )

    def _search_optimal_batch_size(self):
        self.max_inner_batch_size = 16

    def forward(self, batch: TrainingInputData) -> RadiationField:
        assert isinstance(batch.ground_truth, RadiationField), "Ground truth must be of type RadiationField. Disable channel joining."
        next_power_of_two = lambda x: 1 << (x - 1).bit_length()
        padding_target_dims = torch.tensor(tuple(next_power_of_two(dim) for dim in batch.ground_truth.xray_beam.fluence.shape[2:]), device=batch.input.direction.device)
        padding_difference = padding_target_dims - torch.tensor(batch.ground_truth.xray_beam.fluence.shape[2:], device=batch.input.direction.device)
        padding_l = padding_difference // 2
        padding_r = padding_difference - padding_l

        padded_xray_fluence = nn.functional.pad(batch.ground_truth.xray_beam.fluence, (
            padding_l[2], padding_r[2], padding_l[1], padding_r[1], padding_l[0], padding_r[0]
        ), mode='constant', value=0.0)
        voxel_map = self.generate_voxelmap3d(padding_target_dims, None, batch.input.direction.device)
        voxel_map = (voxel_map * 2.0) - 1.0  # Normalize to [-1, 1]
        voxel_map = voxel_map.unsqueeze(0).expand(batch.input.direction.shape[0], -1, -1, -1, -1).permute(0, 4, 1, 2, 3)
        x = torch.cat([padded_xray_fluence, voxel_map], dim=1)
        if self.use_spectra:
            spectra = self.spectra_encoder(batch.input.spectrum)
        results = []
        for layer in self.down_sampler:
            x = layer(x)
            results.append(x)

        if self.use_spectra:
            x = self.mid_spectra_film(x, spectra)

        for i, layer in enumerate(self.up_sampler):
            layer: DeConvBlock3D = layer
            current_downsample: Tensor = results.pop()
            x = torch.cat([x, current_downsample], dim=1)
            x: Tensor = layer(x)
            if i < len(self.up_sampler) - 1:    # only apply FiLM to all but last layer
                x = self.skip_films[i](x, spectra)

        crop_start = padding_l.to(dtype=torch.long)
        crop_end = (padding_target_dims - padding_r).to(dtype=torch.long)
        x = x[:, :, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        fluence: Tensor = self.fluence_activation(x[:, self.out_spectra_bins, :, :, :]).unsqueeze(1)
        if self.out_spectra_bins > 0:
            spectra = self.spectra_activation(x[:, :self.out_spectra_bins, :, :, :])
        else:
            spectra = None

        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=spectra,
                fluence=fluence,
                error=None
            ),
            xray_beam=None
        )

    def forward2volume_from_training_input(self, batch: TrainingInputData, voxel_counts: Tensor, spectra_bins: int = 32) -> RadiationField:
        voxel_counts = batch.ground_truth.scatter_field.fluence.shape[2:] if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.fluence.shape[2:]
        spectra_bins = self.out_spectra_bins if spectra_bins is not None and spectra_bins <= 0 else (batch.ground_truth.scatter_field.spectrum.shape[1] if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.spectrum.shape[1])
        gt_fluence = (batch.ground_truth.scatter_field.fluence + (batch.ground_truth.xray_beam.fluence if batch.ground_truth.xray_beam is not None else 0.0) if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth.fluence)
        mask = ~torch.isfinite(gt_fluence)
        if not mask.any():
            mask = None
        return self.forward2volume(batch, voxel_counts, spectra_bins, mask=mask)

    def generate_UNet_layers(self, in_channels: int, out_channels: int, activation: Type[nn.Module] = nn.LeakyReLU, midpoint_inject_dims: int = 0) -> tuple[nn.ModuleList, nn.ModuleList, int]:
        """
        Generates the layers for a UNet architecture and returns a tuple for the down and up-sampling part.
        @param in_channels: Number of input channels
        @param out_channels: Number of output channels
        @param activation: Activation function to use
        @param midpoint_inject_dims: Number of dimensions to inject at the midpoint (default is 0, meaning no injection)
        @return: Tuple of down-sampling and up-sampling layers as nn.ModuleList objects (down, up, midpoint_channels)
        """
        
        max_channels = self.d_model

        # Calculate number of layers needed to downsample to approximately 1x1x1
        target_size = min(self.out_dims)
        num_layers = 1
        curr_size = target_size
        while curr_size > 1:
            curr_size = curr_size // 2  # Each layer halves the size with stride=2
            num_layers += 1
        
        # Ensure we have at least one layer
        num_layers = max(1, num_layers)

        # Initialize lists for down and up sampling layers
        down_sampler: list[ConvBlock3D] = []
        up_sampler: list[DeConvBlock3D] = []

        # Calculate channel progression for downsampling
        current_in_channels = in_channels
        channel_progression = []  # Store channel sizes for each layer
        
        for i in range(num_layers):
            # Double channels each layer, but cap at max_channels
            if i == 0:
                current_out_channels = min(32, max_channels)  # Start with reasonable base
            else:
                current_out_channels = min(channel_progression[-1] * 2, max_channels)
            
            channel_progression.append(current_out_channels)
            
            # Create downsampling layer
            down_sampler.append(
                ConvBlock3D(
                    current_in_channels,
                    current_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation=activation,
                    normalize=i < num_layers - 1,  # Normalize all but last layer
                    raw_last_layer=(i == num_layers - 1) and self.use_spectra
                )
            )
            
            current_in_channels = current_out_channels

        midpoint_channels = current_out_channels

        # Create upsampling layers (reverse the downsampling)
        for i in range(num_layers - 1):  # Go backwards through layers
            # For upsampling, we go from higher channels back down
            current_in_channels = channel_progression[-(i+1)]
            current_out_channels = channel_progression[-(i+2)]

            up_sampler.append(DeConvBlock3D(
                in_channels=current_in_channels * 2,  # Because of skip connections
                out_channels=current_out_channels,
                kernel_size=3 if i > 0 else 2, # Use kernel size 2 for the first upsample to allow exact doubling
                stride=2,
                padding=1 if i > 0 else 0,
                activation=activation,
                normalize=i > 0,  # Normalize all but first layer,
                raw_last_layer=True
            ))

        # Final upsampling layer to get desired output channels
        up_sampler.append(DeConvBlock3D(
            in_channels=current_out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=None  # No activation for final layer
        ))
        
        return nn.ModuleList(down_sampler), nn.ModuleList(up_sampler), midpoint_channels

    def get_custom_parameters(self):
        return {
            "spectra_bins": self.spectra_bins,
            "out_dims": self.out_dims,
            "d_model": self.d_model,
            "out_spectra_bins": self.out_spectra_bins,
            "normalization_type": self._normalizer.get_type(),
            "use_spectra": self.use_spectra
        }

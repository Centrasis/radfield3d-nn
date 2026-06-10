import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Type

from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel, TrainingInputData
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping
from radfield3dnn.models.encoders.spectra_encoder import SimpleSpectraEncoder
from radfield3dnn.models.layers.fusions.film import FiLM


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: Type[nn.Module] = nn.LeakyReLU,
                 normalize: bool = True, raw_last_layer: bool = False):
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


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: Type[nn.Module] = nn.LeakyReLU,
                 normalize: bool = True, raw_last_layer: bool = False):
        super().__init__()
        keep_kernel, keep_padding = 3, 1
        up_kernel = 2 if stride == 2 else kernel_size
        up_padding = 0 if stride == 2 else padding
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, keep_kernel, stride=1, padding=keep_padding),
            nn.BatchNorm3d(out_channels) if normalize else nn.Identity(),
            activation() if activation is not None else nn.Identity(),
            nn.ConvTranspose3d(out_channels, out_channels, up_kernel, stride=stride, padding=up_padding),
            nn.BatchNorm3d(out_channels) if normalize and not raw_last_layer else nn.Identity(),
            activation() if activation is not None and not raw_last_layer else nn.Identity(),
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Beam2ScatterUNet(BaseNeuralRadFieldModel):
    """3D U-Net mapping direct-beam flux + voxel coordinates to scatter field.

    Input: 4 channels (1 direct-beam flux + 3 normalised voxel coordinates).
    Output: scatter spectrum (out_spectra_bins channels) + scatter flux (1 channel).
    """

    __model_name__ = "Beam2ScatterUNet"

    def __init__(self, d_model: int = 256, out_spectra_bins: int = 32,
                 in_spectra_bins: int = 32, out_dims: tuple[int, int, int] = (50, 50, 50),
                 learning_rate: float = 1e-3, normalizer=None, use_spectra: bool = True):
        super().__init__(direction_encoding_dims=1, d_model=d_model,
                         learning_rate=learning_rate, normalizer=normalizer)
        self.out_dims = out_dims
        self.out_spectra_bins = out_spectra_bins
        self.use_spectra = use_spectra
        self.spectra_bins = in_spectra_bins

        self.down_sampler, self.up_sampler, self.midpoint_channels = self._generate_unet_layers(
            4, out_spectra_bins + 1
        )

        if use_spectra:
            self.spectra_encoder = SimpleSpectraEncoder(in_spectra_bins, self.midpoint_channels)
            self.mid_spectra_film = FiLM(self.midpoint_channels, self.midpoint_channels,
                                         norm="batch3d", non_linearity=nn.LeakyReLU)
            self.skip_films = nn.ModuleList([
                FiLM(self.midpoint_channels, self.up_sampler[i].out_channels,
                     norm="batch3d", non_linearity=nn.LeakyReLU)
                for i in range(len(self.up_sampler) - 1)
            ])
        else:
            self.spectra_encoder = nn.Identity()
            self.mid_spectra_film = nn.Identity()
            self.skip_films = nn.ModuleList([nn.Identity() for _ in range(len(self.up_sampler) - 1)])

        self.spectra_activation = HistogramNormalize(dim=1)
        if isinstance(self._normalizer, LinearNormalizer):
            if self._normalizer.range == (-1.0, 1.0):
                self.flux_activation = GradientConservingClamping(-1.0, 1.0)
            elif self._normalizer.range == (0.0, 1.0):
                self.flux_activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported LinearNormalizer range: {self._normalizer.range}")
        else:
            self.flux_activation = GradientConservingClamping(0.0, 1.0)

        self.apply(self._init_weights)
        # Re-initialize final output layer with small std for stable early training.
        last_up = self.up_sampler[-1]
        if isinstance(last_up, DeConvBlock3D):
            deconv = last_up.block[3]  # block[3] is the stride-2 upsampling ConvTranspose3d
            if isinstance(deconv, nn.ConvTranspose3d):
                nn.init.normal_(deconv.weight, mean=0.0, std=1e-2)
                if deconv.bias is not None:
                    nn.init.zeros_(deconv.bias)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _generate_unet_layers(
        self, in_channels: int, out_channels: int,
        activation: Type[nn.Module] = nn.LeakyReLU
    ) -> tuple[nn.ModuleList, nn.ModuleList, int]:
        """Build symmetric down/up paths. First up-sampler layer has no skip concat."""
        max_channels = self.d_model
        target_size = min(self.out_dims)
        num_layers = 1
        curr_size = target_size
        while curr_size > 1:
            curr_size //= 2
            num_layers += 1
        num_layers = max(1, num_layers)

        down_sampler: list[ConvBlock3D] = []
        channel_progression: list[int] = []
        cur_in = in_channels

        for i in range(num_layers):
            cur_out = min(32 if i == 0 else channel_progression[-1] * 2, max_channels)
            channel_progression.append(cur_out)
            down_sampler.append(ConvBlock3D(
                cur_in, cur_out,
                kernel_size=3, stride=2, padding=1,
                activation=activation,
                normalize=i < num_layers - 1,
                raw_last_layer=(i == num_layers - 1) and self.use_spectra,
            ))
            cur_in = cur_out

        midpoint_channels = cur_in
        up_sampler: list[DeConvBlock3D] = []

        # First up-sampler: no skip concat → input channels NOT doubled.
        # Subsequent layers receive concatenated skip → input channels doubled.
        for i in range(num_layers - 1):
            cin = channel_progression[-(i + 1)]
            cout = channel_progression[-(i + 2)]
            in_w = cin if i == 0 else cin * 2
            up_sampler.append(DeConvBlock3D(
                in_channels=in_w,
                out_channels=cout,
                kernel_size=3 if i > 0 else 2,
                stride=2,
                padding=1 if i > 0 else 0,
                activation=activation,
                normalize=i > 0,
                raw_last_layer=True,
            ))

        # Final layer: receives skip concat → doubled input
        cout_final = channel_progression[-(num_layers - 1 + 1)] if num_layers > 1 else channel_progression[0]
        in_final = cout_final * 2 if num_layers > 1 else channel_progression[0] * 2
        up_sampler.append(DeConvBlock3D(
            in_channels=in_final,
            out_channels=out_channels,
            kernel_size=3, stride=2, padding=1,
            activation=None,
        ))

        return nn.ModuleList(down_sampler), nn.ModuleList(up_sampler), midpoint_channels

    def forward(self, batch: TrainingInputData) -> RadiationField:
        assert isinstance(batch.ground_truth, RadiationField), \
            "Ground truth must be a RadiationField. Disable channel joining."

        def next_pow2(x: int) -> int:
            return 1 << (x - 1).bit_length()

        dev = batch.input.direction.device
        spatial = batch.ground_truth.direct_beam.flux.shape[2:]
        pad_dims = torch.tensor([next_pow2(d) for d in spatial], device=dev)
        diff = pad_dims - torch.tensor(list(spatial), device=dev)
        pad_l = diff // 2
        pad_r = diff - pad_l

        padded_flux = nn.functional.pad(
            batch.ground_truth.direct_beam.flux,
            (pad_l[2], pad_r[2], pad_l[1], pad_r[1], pad_l[0], pad_r[0]),
            mode='constant', value=0.0
        )
        voxel_map = self.generate_voxelmap3d(pad_dims, None, dev)
        voxel_map = (voxel_map * 2.0) - 1.0
        B = batch.input.direction.shape[0]
        voxel_map = voxel_map.unsqueeze(0).expand(B, -1, -1, -1, -1).permute(0, 4, 1, 2, 3)
        x = torch.cat([padded_flux, voxel_map], dim=1)

        spectra_enc = self.spectra_encoder(batch.input.spectrum) if self.use_spectra else None

        # Downsampling — collect all but the bottleneck for skip connections
        results: list[Tensor] = []
        for i, layer in enumerate(self.down_sampler):
            x = layer(x)
            if i < len(self.down_sampler) - 1:
                results.append(x)

        if self.use_spectra:
            x = self.mid_spectra_film(x, spectra_enc)

        # Upsampling — first layer has no skip, subsequent layers concat skip
        for i, layer in enumerate(self.up_sampler):
            if i > 0:
                skip = results.pop()
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
            if i < len(self.up_sampler) - 1 and self.use_spectra:
                x = self.skip_films[i](x, spectra_enc)

        # Crop padding and split channels
        cs = pad_l.long()
        ce = (pad_dims - pad_r).long()
        x = x[:, :, cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]

        flux = self.flux_activation(x[:, self.out_spectra_bins]).unsqueeze(1)
        spectra = self.spectra_activation(x[:, :self.out_spectra_bins]) if self.out_spectra_bins > 0 else None

        return RadiationField(
            scatter_field=RadiationFieldChannel(spectrum=spectra, flux=flux, error=None),
            direct_beam=None,
        )

    def forward2volume_from_training_input(
        self, batch: TrainingInputData, voxel_counts: Tensor = None, spectra_bins: int = 32
    ) -> RadiationField:
        gt = batch.ground_truth
        gt_flux = gt.scatter_field.flux + (gt.direct_beam.flux if gt.direct_beam is not None else 0.0)
        mask = ~torch.isfinite(gt_flux)
        mask = mask if mask.any() else None
        return self.forward2volume(batch, voxel_counts or gt.scatter_field.flux.shape[2:],
                                   self.out_spectra_bins, mask=mask)

    def forward2volume(self, x, voxel_counts, spectra_bins: int = 32,
                       mask: Tensor | None = None) -> RadiationField:
        pred: RadiationField = self(x)
        if mask is not None:
            inf_flux = torch.full_like(pred.scatter_field.flux, -torch.inf)
            inf_spec = torch.full_like(pred.scatter_field.spectrum, -torch.inf)
            spec_mask = mask.expand_as(pred.scatter_field.spectrum)
            pred = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=torch.where(spec_mask, inf_spec, pred.scatter_field.spectrum),
                    flux=torch.where(mask, inf_flux, pred.scatter_field.flux),
                    error=None,
                ),
                direct_beam=None,
            )
        return pred

    def evaluate_forward(self, batch: TrainingInputData):
        batch = self._normalizer.forward(batch)
        pred = self.forward2volume_from_training_input(batch, None, None)
        return self.calculate_metrics(pred, batch.ground_truth, batch)

    def _generate_random_ground_truth(self, device):
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=torch.randn(2, self.out_spectra_bins, *self.out_dims, device=device),
                flux=torch.randn(2, 1, *self.out_dims, device=device),
                error=torch.randn(2, 1, *self.out_dims, device=device),
            ),
            direct_beam=RadiationFieldChannel(
                spectrum=torch.randn(2, self.out_spectra_bins, *self.out_dims, device=device),
                flux=torch.randn(2, 1, *self.out_dims, device=device),
                error=torch.randn(2, 1, *self.out_dims, device=device),
            ),
        )

    def _generate_random_input(self, device):
        from radfield3dnn.rftypes import PositionalInput
        return TrainingInputData(
            input=PositionalInput(
                direction=torch.randn(2, 3, device=device),
                spectrum=torch.randn(2, self.spectra_bins, device=device),
                position=torch.randn(2, 3, device=device),
                origin=torch.randn(2, 3, device=device),
                geometry=None,
                beam_shape_parameters=torch.randn(2, 1, device=device),
                beam_shape_type=torch.randint(0, 2, (2, 1), device=device, dtype=torch.float32),
            ),
            ground_truth=self._generate_random_ground_truth(device),
        )

    def _search_optimal_batch_size(self):
        self.max_inner_batch_size = 16

    def get_custom_parameters(self) -> dict:
        params = {
            "spectra_bins": self.spectra_bins,
            "out_dims": self.out_dims,
            "d_model": self.d_model,
            "out_spectra_bins": self.out_spectra_bins,
            "normalization_type": self._normalizer.get_type(),
            "use_spectra": self.use_spectra,
        }
        seed = os.environ.get("PL_GLOBAL_SEED", None)
        if seed is not None:
            params["training_seed"] = int(seed)
        return params

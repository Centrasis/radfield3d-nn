from .base import BaseNeuralRadFieldModel, ModuleBuilder, AngularSinusoidalFrequencyEncoding
import torch
from torch import Tensor
from torch import nn
from typing import Type
import os
from radfield3dnn.encodings.sinusoidal_encoding import SinusoidalFrequencyEncoding
from radfield3dnn.rftypes import AirKermaField, RadiationField, PositionalInput, RadiationFieldChannel, DirectionalInput, PositionalInput
from typing import Union


class Reduce(nn.Module):
    """
    Reduce the input tensor along the specified dimension.
    For residual connections, the input and output dimensions have different sizes.
    Uses summation to reduce the tensor. If the tensor input and output dimensions are not a multiple of each other, the remaining dimensions are just added
    """
    def __init__(self, axis: int, output_dim: int):
        super(Reduce, self).__init__()
        self.axis = axis
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        input_dim = x.shape[self.axis]
        group_size = input_dim // self.output_dim
        remainder = input_dim % self.output_dim

        # Calculate the number of elements to sum in each group
        if group_size == 0:
            group_size = 1
            self.output_dim = input_dim

        # Reshape the tensor to group elements
        dims = list(x.shape)
        dims.pop(self.axis)
        summed_elements = torch.tensor(dims, dtype=torch.int32).prod() * self.output_dim * group_size
        output = torch.zeros(*dims, self.output_dim, device=x.device)
        output_sum = torch.sum(x.flatten()[:summed_elements].reshape(*dims, self.output_dim, group_size), dim=-1)
        output_sum = output_sum / group_size

        if remainder > 0:
            sum_elements_embedded = torch.zeros_like(output)
            output_slice = [slice(None, None) for _ in range(len(output.shape))]

            output_slice[self.axis] = slice(0, x.shape[self.axis] - remainder)
            sum_elements_embedded[*tuple(output_slice)] = output_sum
            output = output + sum_elements_embedded

            left_elements = x.flatten()[summed_elements:]
            left_elements_shape = list(output.shape)
            left_elements_shape[self.axis] = remainder
            left_elements = left_elements.reshape(*left_elements_shape)

            output_slice[self.axis] = slice(int(output.shape[self.axis] - remainder), None)
            left_elements_embedded = torch.zeros_like(output)
            left_elements_embedded[*tuple(output_slice)] = left_elements

            output = output + left_elements_embedded
        else:
            output = output + output_sum
        return output


class LinearBlock(nn.Module):
    def __init__(self, d_model: int, activation: Type[nn.Module] | None, normalization: Type[nn.Module] | None = nn.BatchNorm1d, num_layers: int = 1, d_model_out: int | None = None):
        super(LinearBlock, self).__init__()
        assert num_layers > 0, "Number of layers must be greater than 0."
        if d_model_out is None:
            d_model_out = d_model
        layers = []

        for i in range(num_layers):
            layer_out = d_model_out if i == num_layers - 1 else d_model
            layers += [
                nn.Linear(d_model, layer_out),
            ]
            if normalization is not None:
                layers += [
                    normalization(layer_out)
                ]
            if activation is not None and i < num_layers - 1:
                layers += [
                    activation()
                ]

        self.model = nn.Sequential(
            *layers
        )
        self.activation = activation() if activation is not None else nn.Identity()
        self.reduction = nn.Identity() if d_model_out == d_model else Reduce(-1, d_model_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.model(x))


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.SiLU(inplace=False)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.activation(x)
        return x


class FeedforwardPointwiseModel(BaseNeuralRadFieldModel):
    def __init__(self, location_encoding_dims=10, direction_encoding_dims=10, d_model=256, learning_rate: float = 1e-3, randomize_voxel_location_in_training: bool = True, voxels_centered_around_origin: bool = True, normalizer=None):
        """
        Feedforward model that processes each voxel independently.
        :param location_encoding_dims: Number of frequency bands for positional encoding of the location.
        :param direction_encoding_dims: Number of frequency bands for positional encoding of the direction.
        :param d_model: Dimension of the model.
        :param learning_rate: Learning rate for the optimizer.
        :param randomize_voxel_location_in_training: Whether to randomize voxel centers during training within the voxel extent.
        :param voxels_centered_around_origin: Whether the voxel grid is centered around the origin ([-1, 1]) or starts from the origin ([0, 1]).
        """
        super().__init__(location_encoding_dims, direction_encoding_dims, d_model, learning_rate=learning_rate, normalizer=normalizer)
        self.randomize_voxel_location_in_training = randomize_voxel_location_in_training
        self.voxels_centered_around_origin = voxels_centered_around_origin
        self.relevance_discriminator: "FeedforwardPointwiseModel" | None = None
        self._base_voxel_map = None

    def forward(self, x: Union[DirectionalInput, PositionalInput], global_parameters: Tensor | None = None) -> RadiationField:
        raise NotImplementedError("This method must be implemented by the subclass.")

    def on_fit_start(self):
        if self.relevance_discriminator is not None:
            self.relevance_discriminator = self.relevance_discriminator.eval()
        super().on_fit_start()

    def jacobian_norm(self, x: PositionalInput) -> Tensor:
        x = PositionalInput(
            position=x.position.clone().detach().requires_grad_(True),
            direction=x.direction.clone().detach().requires_grad_(True),
            spectrum=x.spectrum.clone().detach().requires_grad_(True),
            geometry=x.geometry.clone().detach().requires_grad_(True) if x.geometry is not None else None,
            origin=x.origin.clone().detach().requires_grad_(True) if x.origin is not None else None,
            beam_shape_parameters=x.beam_shape_parameters.clone().detach().requires_grad_(True) if x.beam_shape_parameters is not None else None,
            beam_shape_type=x.beam_shape_type.clone().detach().requires_grad_(True) if x.beam_shape_type is not None else None
        )
        field = self.forward(x)
        y = torch.stack([field.scatter_field.flux, field.scatter_field.spectrum.sum(dim=-1)], dim=-1)
        batch_size = y.size(0)
        norms = torch.empty(batch_size, device=x.position.device)
        for i in range(batch_size):
            grads = torch.autograd.grad(
                y[i].sum(), 
                [
                    x.position,
                    x.direction,
                    x.spectrum,
                    x.geometry,
                    x.origin,
                    x.beam_shape_parameters,
                    x.beam_shape_type
                ],
                create_graph=False, 
                retain_graph=False,
                allow_unused=True
            )[0][i]
            norms[i] = grads.norm()  # L2 norm of gradient row i
        return norms.mean()

    @staticmethod
    def write_linear_output_to_volume(output: RadiationFieldChannel, target_volume: RadiationFieldChannel, indices: Tensor):
        """
        Write the output of a linear model to the target volume at the specified indices.
        :param output: RadiationFieldChannel containing the output flux and spectrum.
        :param target_volume: RadiationFieldChannel where the output will be written.
        :param indices: Tensor containing the indices where the output should be written.
        """
        target_volume.flux[:, indices[:, 0], indices[:, 1], indices[:, 2]] = output.flux.to(torch.float32).unsqueeze(0) if len(output.flux.shape) == 1 else output.flux
        if output.spectrum is not None:
            target_volume.spectrum[:, indices[:, 0], indices[:, 1], indices[:, 2]] = output.spectrum.to(torch.float32).permute(-1, 0)

    def get_submodels(self):
        return [self.relevance_discriminator] if self.relevance_discriminator is not None else []
    
    @property
    def base_voxel_map(self):
        if self._base_voxel_map is not None:
            if self._base_voxel_map.requires_grad != self.training:
                self._base_voxel_map = None
        
        if self._base_voxel_map is None:
            if self.training:
                self._base_voxel_map = self.generate_voxelmap3d(self.voxel_counts, None, self.device)
            else:
                with torch.no_grad():
                    self._base_voxel_map = self.generate_voxelmap3d(self.voxel_counts, None, self.device)
        return self._base_voxel_map

    def prepare_linear_input_batches(self, x: DirectionalInput, voxel_counts: Tensor, mask: Tensor | None = None):
        # Generate base voxel map once increase voxel_counts by 1 in each dimension to account for randomization
        batched_input = len(x.direction.shape) == 2
        
        if not batched_input:
            x = DirectionalInput(
                direction=x.direction.unsqueeze(0),
                spectrum=x.spectrum.unsqueeze(0) if x.spectrum is not None else None,
                geometry=x.geometry.unsqueeze(0) if x.geometry is not None else None,
                origin=x.origin.unsqueeze(0) if x.origin is not None else None,
                beam_shape_parameters=x.beam_shape_parameters.unsqueeze(0) if x.beam_shape_parameters is not None else None,
                beam_shape_type=x.beam_shape_type.unsqueeze(0) if x.beam_shape_type is not None else None
            )
        batch_size = x.direction.shape[0]

        # Calculate total voxels per batch item
        total_voxels_per_batch = int(voxel_counts.prod().item())

        # Create voxel map for all batch items efficiently
        if self.randomize_voxel_location_in_training and self.training:
            # Add noise to each batch item separately
            voxel_map = self.base_voxel_map.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
            voxel_extent = 1.0 / voxel_counts.float().view(1, 1, 1, 1, -1)
            noise = torch.rand_like(voxel_map[..., :3]) * voxel_extent
            voxel_map[..., :3] = voxel_map[..., :3] + noise
        else:
            voxel_map = self.base_voxel_map.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        # allocate memory for the full voxel map an make it contiguous
        voxel_map = voxel_map.clone().contiguous()

        if self.voxels_centered_around_origin:
            # map xyz from [0,1] to [-1,1] for the model
            voxel_map[..., :3] = (voxel_map[..., :3] * 2.0) - 1.0

        # Flatten to (batch_size * total_voxels_per_batch, 4)
        voxel_list = voxel_map.reshape(-1, voxel_map.shape[-1])
        batch_indices = torch.arange(batch_size, device=x.direction.device).repeat_interleave(total_voxels_per_batch)
        linear_indices = torch.arange(voxel_list.shape[0], device=x.direction.device)

        # Create direction list by repeating each direction for all voxels in that batch
        direction_list = x.direction.repeat_interleave(total_voxels_per_batch, dim=0)

        if mask is not None:
            mask = mask.reshape(-1)
            keep = ~mask

            voxel_list = voxel_list[keep]
            direction_list = direction_list[keep]
            batch_indices = batch_indices[keep]
            linear_indices = linear_indices[keep]

        # Create split iterators
        voxel_splits = torch.split(voxel_list, self.max_inner_batch_size)
        direction_splits = torch.split(direction_list, self.max_inner_batch_size)
        batch_idx_splits = torch.split(batch_indices, self.max_inner_batch_size)
        index_splits = torch.split(linear_indices, self.max_inner_batch_size)

        return voxel_splits, direction_splits, batch_idx_splits, index_splits

    def forward2volume(self, x: DirectionalInput, voxel_counts: Union[Tensor, torch.Size], spectra_bins: int = 32, mask: Tensor | None = None, global_parameters: Tensor | None = None) -> RadiationField | AirKermaField:
        """
        Forward pass of the model to generate a volume from the input data.
        :param x: DirectionalInput containing direction and optional spectrum. Shape (B, 3) or (3,) for direction and (B, C) or (C,) for spectrum.
        :param voxel_counts: Tensor or torch.Size specifying the dimensions of the output volume.
        :param spectra_bins: Number of bins for the spectrum.
        :return: RadiationField containing the generated volume.
        """
        voxel_counts = torch.tensor(voxel_counts, dtype=torch.int32, device=x.direction.device) if not isinstance(voxel_counts, Tensor) else voxel_counts
        self.voxel_counts = voxel_counts

        if self.max_inner_batch_size is None:
            self.on_fit_start()

        # Generate base voxel map once increase voxel_counts by 1 in each dimension to account for randomization
        batched_input = len(x.direction.shape) == 2
        batch_size = x.direction.shape[0] if batched_input else 1
        # Calculate total voxels per batch item
        total_voxels_per_batch = int(voxel_counts.prod().item())
        
        # Process in chunks respecting max_inner_batch_size
        full_field: RadiationField = None
        voxel_splits, direction_splits, batch_idx_splits, index_splits = self.prepare_linear_input_batches(x, voxel_counts, mask)

        for voxels, directions, batch_idx, indices in zip(voxel_splits, direction_splits, batch_idx_splits, index_splits):
            pred_field = self.forward(
                PositionalInput(
                    direction=directions,
                    position=voxels[:, 0:3],
                    spectrum=x.spectrum.index_select(0, batch_idx) if x.spectrum is not None else None,
                    geometry=x.geometry.index_select(0, batch_idx) if x.geometry is not None else None,
                    origin=x.origin.index_select(0, batch_idx) if x.origin is not None else None,
                    beam_shape_parameters=x.beam_shape_parameters.index_select(0, batch_idx) if x.beam_shape_parameters is not None else None,
                    beam_shape_type=x.beam_shape_type.index_select(0, batch_idx) if x.beam_shape_type is not None else None
                ),
                global_parameters=global_parameters.index_select(0, batch_idx) if global_parameters is not None else None
            )

            if full_field is None:
                total_samples = batch_size * total_voxels_per_batch
                if isinstance(pred_field, RadiationField):
                    full_field = RadiationField(
                        scatter_field=RadiationFieldChannel(
                            spectrum=torch.empty(total_samples, spectra_bins, device=x.direction.device, dtype=torch.float32) if pred_field.scatter_field.spectrum is not None else None,
                            flux=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32)
                        ) if pred_field.scatter_field is not None else None,
                        direct_beam=RadiationFieldChannel(
                            spectrum=torch.empty(total_samples, spectra_bins, device=x.direction.device, dtype=torch.float32) if pred_field.direct_beam.spectrum is not None else None,
                            flux=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32)
                        ) if pred_field.direct_beam is not None else None,
                        geometry=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32) if pred_field.geometry is not None else None
                    )
                elif isinstance(pred_field, AirKermaField):
                    full_field = AirKermaField(
                        air_kerma=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32),
                        geometry=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32) if pred_field.geometry is not None else None
                    )
            if isinstance(pred_field, RadiationField):
                if pred_field.scatter_field is not None:
                    full_field.scatter_field.flux[indices] = pred_field.scatter_field.flux.to(torch.float32).unsqueeze(0) if len(pred_field.scatter_field.flux.shape) == 1 else pred_field.scatter_field.flux
                    if pred_field.scatter_field.spectrum is not None:
                        full_field.scatter_field.spectrum[indices] = pred_field.scatter_field.spectrum.to(torch.float32)
                if pred_field.direct_beam is not None:
                    full_field.direct_beam.flux[indices] = pred_field.direct_beam.flux.to(torch.float32).unsqueeze(0) if len(pred_field.direct_beam.flux.shape) == 1 else pred_field.direct_beam.flux
                    if pred_field.direct_beam.spectrum is not None:
                        full_field.direct_beam.spectrum[indices] = pred_field.direct_beam.spectrum.to(torch.float32)
                if pred_field.geometry is not None:
                    full_field.geometry[indices] = pred_field.geometry.to(torch.float32).unsqueeze(0) if len(pred_field.geometry.shape) == 1 else pred_field.geometry
            elif isinstance(pred_field, AirKermaField):
                full_field.air_kerma[indices] = pred_field.air_kerma.to(torch.float32).unsqueeze(0) if len(pred_field.air_kerma.shape) == 1 else pred_field.air_kerma
                if pred_field.geometry is not None:
                    full_field.geometry[indices] = pred_field.geometry.to(torch.float32).unsqueeze(0) if len(pred_field.geometry.shape) == 1 else pred_field.geometry
            else:
                raise ValueError("Unknown field type returned by the model.")

        # Reshape to final volume dimensions
        if batched_input:
            target_flux_shape = (batch_size, 1, voxel_counts[0], voxel_counts[1], voxel_counts[2])
            target_spectra_shape = (batch_size, spectra_bins, voxel_counts[0], voxel_counts[1], voxel_counts[2])
        else:
            target_flux_shape = (1, voxel_counts[0], voxel_counts[1], voxel_counts[2])
            target_spectra_shape = (spectra_bins, voxel_counts[0], voxel_counts[1], voxel_counts[2])

        if isinstance(full_field, RadiationField):
            full_field = RadiationField(
                scatter_field=RadiationFieldChannel(
                    flux=full_field.scatter_field.flux.view(*target_flux_shape) if full_field.scatter_field.flux is not None else None,
                    spectrum=full_field.scatter_field.spectrum.view(batch_size, total_voxels_per_batch, spectra_bins).permute(0, 2, 1).view(*target_spectra_shape) if full_field.scatter_field.spectrum is not None else None
                ) if full_field.scatter_field is not None else None,
                direct_beam=RadiationFieldChannel(
                    flux=full_field.direct_beam.flux.view(*target_flux_shape) if full_field.direct_beam.flux is not None else None,
                    spectrum=full_field.direct_beam.spectrum.view(batch_size, total_voxels_per_batch, spectra_bins).permute(0, 2, 1).view(*target_spectra_shape) if full_field.direct_beam.spectrum is not None else None
                ) if full_field.direct_beam is not None else None,
                geometry=full_field.geometry.view(*target_flux_shape) if full_field.geometry is not None else None
            )
            if mask is not None and mask.any():
                if full_field.scatter_field is not None and full_field.scatter_field.flux is not None:
                    full_field.scatter_field.flux[mask] = -torch.inf
                    spec_mask = mask.expand_as(full_field.scatter_field.spectrum) if full_field.scatter_field.spectrum is not None else None
                    if spec_mask is not None:
                        full_field.scatter_field.spectrum[spec_mask] = -torch.inf
                if full_field.direct_beam is not None and full_field.direct_beam.flux is not None:
                    full_field.direct_beam.flux[mask] = -torch.inf
                    spec_mask = mask.expand_as(full_field.direct_beam.spectrum) if full_field.direct_beam.spectrum is not None else None
                    if spec_mask is not None:
                        full_field.direct_beam.spectrum[spec_mask] = -torch.inf
        elif isinstance(full_field, AirKermaField):
            full_field: AirKermaField = AirKermaField(
                air_kerma=full_field.air_kerma.view(*target_flux_shape) if full_field.air_kerma is not None else None,
                geometry=full_field.geometry.view(*target_flux_shape) if full_field.geometry is not None else None
            )
            if mask is not None and mask.any():
                if full_field.air_kerma is not None:
                    full_field.air_kerma[mask] = -torch.inf
        else:
            raise ValueError("Unknown field type returned by the model.")
        
        return full_field

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]


class ParametricFeedforwardModel(FeedforwardPointwiseModel):
    __model_name__ = "ParametricFeedforwardModel"

    def __init__(self, location_encoding_dims=10, direction_encoding_dims=4, d_model=256, num_layers=5, out_spectra_dims: int = 32, activation_fn: str = "ReLU", spectra_loss_fn = "L1Loss", flux_loss_fn = "L1Loss", position_encoding_type: str = "SinusoidalFrequency", layer_normalization: str | None = "LayerNorm", use_residuals: bool = True):
        super().__init__(location_encoding_dims, direction_encoding_dims, d_model)
        self.num_layers = num_layers
        self.activation_fn_name = activation_fn
        self.spectra_loss_fn_name = spectra_loss_fn
        self.flux_loss_fn_name = flux_loss_fn
        self.position_encoding_type = position_encoding_type
        self.num_layers = num_layers
        self.out_spectra_dims = out_spectra_dims
        self.use_residuals = use_residuals
        self.layer_normalization_name = layer_normalization
        self.activation_fn = ModuleBuilder.ConstructActivation_fn(activation_fn)
        self.spectra_loss_fn = ModuleBuilder.ConstructLoss_fn(spectra_loss_fn)
        self.flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss_fn)
        self.layer_normalization = ModuleBuilder.ConstructNormalization_fn(layer_normalization) if layer_normalization is not None else None
        self.positional_location_encoding = None
        if position_encoding_type == "SinusoidalFrequency":
            self.preprocess_location = nn.Identity()
            self.preprocess_direction = nn.Identity()
            self.positional_location_encoding = SinusoidalFrequencyEncoding(location_encoding_dims, 3, append_input=True)
            self.positional_direction_encoding = AngularSinusoidalFrequencyEncoding(direction_encoding_dims, append_input=True)
        else:
            raise ValueError(f"Unknown position encoding type: {position_encoding_type}")

        self.hidden_layers_and_output = [
            ResidualBlock(d_model, self.activation_fn, self.layer_normalization) if use_residuals else LinearBlock(d_model, self.activation_fn, self.layer_normalization)
            for _ in range(num_layers - 1)
        ] + [
            ResidualBlock(d_model, self.activation_fn, self.layer_normalization, d_model_out=d_model//2) if use_residuals else LinearBlock(d_model, self.activation_fn, self.layer_normalization, d_model_out=d_model//2)
        ]

        self.model = nn.Sequential(
            nn.Linear(self.positional_location_encoding.encoded_dims + self.positional_direction_encoding.encoded_dims, d_model),
            *self.hidden_layers_and_output
        )

        self.flux_activation = nn.Sigmoid()
        self.spectra_activation = nn.Softmax(dim=1)

    def forward(self, x: PositionalInput) -> RadiationField:
        loc = self.preprocess_location(x.position)
        loc_enc = self.positional_location_encoding(loc)
        dir = self.preprocess_direction(x.direction)
        dir_enc = self.positional_direction_encoding(dir)
        x = torch.cat((loc_enc, dir_enc), dim=-1)
        x = self.model(x)
        flux = self.flux_activation(x[:, -1])
        spectra = self.spectra_activation(x[:, :self.out_spectra_dims])
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                flux=flux,
                spectrum=spectra
            ),
            direct_beam=None
        )

    def get_custom_parameters(self):
        params = {
            "location_encoding_dims": self.positional_location_encoding.encoded_dims,
            "direction_encoding_dims": self.positional_direction_encoding.encoded_dims,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "out_spectra_dims": self.out_spectra_dims,
            "activation_fn": self.activation_fn_name,
            "spectra_loss_fn": self.spectra_loss_fn_name,
            "flux_loss_fn": self.flux_loss_fn_name,
            "position_encoding_type": self.position_encoding_type,
            "layer_normalization": self.layer_normalization_name,
            "use_residuals": self.use_residuals,
        }
        seed = os.environ.get("PL_GLOBAL_SEED", None)
        if seed is not None:
            params["training_seed"] = int(seed)
        return params

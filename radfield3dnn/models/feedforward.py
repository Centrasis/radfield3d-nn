from .base import BaseNeuralRadFieldModel
import torch
from torch import Tensor
from torch import nn
from radfield3dnn import AirKermaField, RadiationField, PositionalInput, RadiationFieldChannel, DirectionalInput, PositionalInput
from typing import Union
from radfield3dnn.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear


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
        y = torch.stack([field.scatter_field.fluence, field.scatter_field.spectrum.sum(dim=-1)], dim=-1)
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
        :param output: RadiationFieldChannel containing the output fluence and spectrum.
        :param target_volume: RadiationFieldChannel where the output will be written.
        :param indices: Tensor containing the indices where the output should be written.
        """
        target_volume.fluence[:, indices[:, 0], indices[:, 1], indices[:, 2]] = output.fluence.to(torch.float32).unsqueeze(0) if len(output.fluence.shape) == 1 else output.fluence
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
                            fluence=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32)
                        ) if pred_field.scatter_field is not None else None,
                        xray_beam=RadiationFieldChannel(
                            spectrum=torch.empty(total_samples, spectra_bins, device=x.direction.device, dtype=torch.float32) if pred_field.xray_beam.spectrum is not None else None,
                            fluence=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32)
                        ) if pred_field.xray_beam is not None else None,
                        geometry=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32) if pred_field.geometry is not None else None
                    )
                elif isinstance(pred_field, AirKermaField):
                    full_field = AirKermaField(
                        air_kerma=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32),
                        geometry=torch.empty(total_samples, device=x.direction.device, dtype=torch.float32) if pred_field.geometry is not None else None
                    )
            if isinstance(pred_field, RadiationField):
                if pred_field.scatter_field is not None:
                    full_field.scatter_field.fluence[indices] = pred_field.scatter_field.fluence.to(torch.float32).unsqueeze(0) if len(pred_field.scatter_field.fluence.shape) == 1 else pred_field.scatter_field.fluence
                    if pred_field.scatter_field.spectrum is not None:
                        full_field.scatter_field.spectrum[indices] = pred_field.scatter_field.spectrum.to(torch.float32)
                if pred_field.xray_beam is not None:
                    full_field.xray_beam.fluence[indices] = pred_field.xray_beam.fluence.to(torch.float32).unsqueeze(0) if len(pred_field.xray_beam.fluence.shape) == 1 else pred_field.xray_beam.fluence
                    if pred_field.xray_beam.spectrum is not None:
                        full_field.xray_beam.spectrum[indices] = pred_field.xray_beam.spectrum.to(torch.float32)
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
            target_fluence_shape = (batch_size, 1, voxel_counts[0], voxel_counts[1], voxel_counts[2])
            target_spectra_shape = (batch_size, spectra_bins, voxel_counts[0], voxel_counts[1], voxel_counts[2])
        else:
            target_fluence_shape = (1, voxel_counts[0], voxel_counts[1], voxel_counts[2])
            target_spectra_shape = (spectra_bins, voxel_counts[0], voxel_counts[1], voxel_counts[2])

        if isinstance(full_field, RadiationField):
            full_field = RadiationField(
                scatter_field=RadiationFieldChannel(
                    fluence=full_field.scatter_field.fluence.view(*target_fluence_shape) if full_field.scatter_field.fluence is not None else None,
                    spectrum=full_field.scatter_field.spectrum.view(batch_size, total_voxels_per_batch, spectra_bins).permute(0, 2, 1).view(*target_spectra_shape) if full_field.scatter_field.spectrum is not None else None
                ) if full_field.scatter_field is not None else None,
                xray_beam=RadiationFieldChannel(
                    fluence=full_field.xray_beam.fluence.view(*target_fluence_shape) if full_field.xray_beam.fluence is not None else None,
                    spectrum=full_field.xray_beam.spectrum.view(batch_size, total_voxels_per_batch, spectra_bins).permute(0, 2, 1).view(*target_spectra_shape) if full_field.xray_beam.spectrum is not None else None
                ) if full_field.xray_beam is not None else None,
                geometry=full_field.geometry.view(*target_fluence_shape) if full_field.geometry is not None else None
            )
            if mask is not None and mask.any():
                if full_field.scatter_field is not None and full_field.scatter_field.fluence is not None:
                    full_field.scatter_field.fluence[mask] = -torch.inf
                    spec_mask = mask.expand_as(full_field.scatter_field.spectrum) if full_field.scatter_field.spectrum is not None else None
                    if spec_mask is not None:
                        full_field.scatter_field.spectrum[spec_mask] = -torch.inf
                if full_field.xray_beam is not None and full_field.xray_beam.fluence is not None:
                    full_field.xray_beam.fluence[mask] = -torch.inf
                    spec_mask = mask.expand_as(full_field.xray_beam.spectrum) if full_field.xray_beam.spectrum is not None else None
                    if spec_mask is not None:
                        full_field.xray_beam.spectrum[spec_mask] = -torch.inf
        elif isinstance(full_field, AirKermaField):
            full_field: AirKermaField = AirKermaField(
                air_kerma=full_field.air_kerma.view(*target_fluence_shape) if full_field.air_kerma is not None else None,
                geometry=full_field.geometry.view(*target_fluence_shape) if full_field.geometry is not None else None
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


class WangFCN(FeedforwardPointwiseModel):
    """
    Wang et al. (2025) "Research on 3D Radiation Fields Reconstruction based on FNN"
    """
    __model_name__ = "WangFCN"

    def __init__(self, out_spectra_dims: int = 32, in_spectra_bins: int = 32):
        super().__init__(location_encoding_dims=3, direction_encoding_dims=2, d_model=1)
        self.out_spectra_dims = out_spectra_dims
        self.in_spectra_bins = in_spectra_bins
        self.model = nn.Sequential(
            nn.Linear(3 + 2, 100), # + in_spectra_bins
            nn.SiLU(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2000),
            nn.Tanh(),
            nn.Linear(2000, 500),
            nn.Tanh(),
            nn.Linear(500, 200),
            nn.Tanh()
        )
        self.spectra_decoder = nn.Linear(200, out_spectra_dims)
        self.fluence_decoder = nn.Linear(200, 1)
        self.spectra_activation_fn = HistogramNormalize(dim=-1)
        self.fluence_activation_fn = nn.Sigmoid()

    def forward(self, x: PositionalInput) -> RadiationField:
        loc = x.position
        if x.spectrum is not None:
            spectrum = resample_histogram_bilinear(x.spectrum, self.in_spectra_bins)
        else:
            spectrum = torch.zeros(loc.shape[0], self.in_spectra_bins, device=loc.device, dtype=torch.float32)

        #x = torch.cat((loc, dir, spectrum), dim=-1)
        x = torch.cat((loc, dir), dim=-1)
        x = self.model(x)

        fluence = self.fluence_decoder(x)
        fluence = self.fluence_activation_fn(fluence)
        #spectra = self.spectra_decoder(x)
        #spectra = self.spectra_activation_fn(spectra)
        spectra = None

        return RadiationField(
            scatter_field=RadiationFieldChannel(
                fluence=fluence,
                spectrum=spectra
            ),
            xray_beam=None
        )

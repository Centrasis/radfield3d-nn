from .base import BaseNeuralRadFieldModel
import torch
from torch import Tensor
from torch import nn
from radfield3dnn.rftypes import AirKermaField, RadiationField, PositionalInput, RadiationFieldChannel, DirectionalInput, PositionalInput
from typing import Union
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear


class FeedforwardPointwiseModel(BaseNeuralRadFieldModel):
    def __init__(self, learning_rate: float = 1e-3, randomize_voxel_location_in_training: bool = True, voxels_centered_around_origin: bool = True, normalizer=None):
        """
        Feedforward model that processes each voxel independently.
        :param learning_rate: Learning rate for the optimizer.
        :param randomize_voxel_location_in_training: Whether to randomize voxel centers during training within the voxel extent.
        :param voxels_centered_around_origin: Whether the voxel grid is centered around the origin ([-1, 1]) or starts from the origin ([0, 1]).
        """
        super().__init__(learning_rate=learning_rate, normalizer=normalizer)
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
        # The normalised voxel-coordinate grid depends only on the grid shape and
        # device — never on grad (positions carry no gradient, and
        # generate_voxelmap3d already builds it under torch.no_grad). Cache it
        # and rebuild only when the shape or device changes. Per-step location
        # randomisation happens on a clone in prepare_linear_input_batches, so
        # caching the un-noised base map across steps is correct.
        vc = self.voxel_counts
        target_shape = (int(vc[0]), int(vc[1]), int(vc[2]))
        if (self._base_voxel_map is None
                or self._base_voxel_map.device != self.device
                or tuple(self._base_voxel_map.shape[:3]) != target_shape):
            self._base_voxel_map = self.generate_voxelmap3d(vc, None, self.device)
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

        # Every chunk is padded to a CONSTANT row count (max_inner_batch_size)
        # before it reaches the tcnn modules. tiny-cuda-nn's GPUMemoryArena is
        # a high-water-mark allocator backed by CUDA virtual memory that is
        # never returned to the driver; a variable input size (short last
        # split, data-dependent mask filtering, per-call 256-padding) makes it
        # allocate a fresh, larger arena for every new shape and keep all the
        # old ones, so VRAM ratchets up every epoch until cuMemCreate OOMs.
        # Fixed shape => one arena, reused forever. Padded rows are sliced off
        # (`[:n]`) before anything is scattered into full_field, so they never
        # affect the result or the gradient.
        fixed_chunk = self.max_inner_batch_size
        for voxels, directions, batch_idx, indices in zip(voxel_splits, direction_splits, batch_idx_splits, index_splits):
            n = voxels.shape[0]
            pad = fixed_chunk - n
            if pad > 0:
                voxels = nn.functional.pad(voxels, (0, 0, 0, pad))
                directions = nn.functional.pad(directions, (0, 0, 0, pad))
                # index 0 is always valid; these rows are discarded via [:n].
                batch_idx = nn.functional.pad(batch_idx, (0, pad))

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

            # Drop the padded rows before they reach full_field. RadiationField
            # / RadiationFieldChannel / AirKermaField are NamedTuples, so
            # rebuild via _replace rather than mutating. Slicing keeps the
            # autograd path for the real rows intact.
            if pad > 0:
                def _trim_channel(ch):
                    if ch is None:
                        return None
                    return ch._replace(
                        flux=ch.flux[:n],
                        spectrum=ch.spectrum[:n] if ch.spectrum is not None else None
                    )
                if isinstance(pred_field, RadiationField):
                    pred_field = pred_field._replace(
                        scatter_field=_trim_channel(pred_field.scatter_field),
                        direct_beam=_trim_channel(pred_field.direct_beam),
                        geometry=pred_field.geometry[:n] if pred_field.geometry is not None else None
                    )
                elif isinstance(pred_field, AirKermaField):
                    pred_field = pred_field._replace(
                        air_kerma=pred_field.air_kerma[:n],
                        geometry=pred_field.geometry[:n] if pred_field.geometry is not None else None
                    )

            if full_field is None:
                # Pre-fill prediction-side flux/spectrum slots with `-inf`
                # so any positions NOT scatter-written below (because
                # `ErrorbasedImportanceSampler` filtered them out of the
                # forward via `prepare_linear_input_batches`'s mask) carry
                # a known sentinel matching the target convention rather
                # than `torch.empty`'s uninitialised garbage. Downstream
                # `StdLossWeighted` / `HistogramLoss` mask both sides on
                # `isfinite`, so consistent `-inf` here is what lets the
                # loss-side mask exclude dropped voxels correctly. Geometry
                # is carried-through metadata (no `-inf` convention) and
                # stays as zero-init.
                total_samples = batch_size * total_voxels_per_batch
                _dev, _dt = x.direction.device, torch.float32
                neg_inf = float("-inf")
                if isinstance(pred_field, RadiationField):
                    full_field = RadiationField(
                        scatter_field=RadiationFieldChannel(
                            spectrum=torch.full((total_samples, spectra_bins), neg_inf, device=_dev, dtype=_dt) if pred_field.scatter_field.spectrum is not None else None,
                            flux=torch.full((total_samples,), neg_inf, device=_dev, dtype=_dt)
                        ) if pred_field.scatter_field is not None else None,
                        direct_beam=RadiationFieldChannel(
                            spectrum=torch.full((total_samples, spectra_bins), neg_inf, device=_dev, dtype=_dt) if pred_field.direct_beam.spectrum is not None else None,
                            flux=torch.full((total_samples,), neg_inf, device=_dev, dtype=_dt)
                        ) if pred_field.direct_beam is not None else None,
                        geometry=torch.zeros(total_samples, device=_dev, dtype=_dt) if pred_field.geometry is not None else None
                    )
                elif isinstance(pred_field, AirKermaField):
                    full_field = AirKermaField(
                        air_kerma=torch.full((total_samples,), neg_inf, device=_dev, dtype=_dt),
                        geometry=torch.zeros(total_samples, device=_dev, dtype=_dt) if pred_field.geometry is not None else None
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
            # masked voxels were never forwarded (keep = ~mask), so they already hold the -inf pre-fill.
        elif isinstance(full_field, AirKermaField):
            full_field: AirKermaField = AirKermaField(
                air_kerma=full_field.air_kerma.view(*target_flux_shape) if full_field.air_kerma is not None else None,
                geometry=full_field.geometry.view(*target_flux_shape) if full_field.geometry is not None else None
            )
        else:
            raise ValueError("Unknown field type returned by the model.")
        
        return full_field

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

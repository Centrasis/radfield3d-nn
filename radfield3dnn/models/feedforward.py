from .base import BaseNeuralRadFieldModel
import torch
from torch import Tensor
from torch import nn
from radfield3dnn.rftypes import AirKermaField, RadiationField, PositionalInput, RadiationFieldChannel, DirectionalInput, positional_like
from typing import Union
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear


class FeedforwardPointwiseModel(BaseNeuralRadFieldModel):
    def __init__(self, learning_rate: float = 1e-3, voxel_supersampling: int = 1, voxels_centered_around_origin: bool = True, normalizer=None):
        """
        Feedforward model that processes each voxel independently.
        :param learning_rate: Learning rate for the optimizer.
        :param voxel_supersampling: K in-voxel samples per voxel during TRAINING; the K predictions
            are AVERAGED per voxel before the loss, matching the voxel-averaged ground truth
            (Monte-Carlo integral supervision; zip-NeRF-style multisampling). 1 = single node-grid
            sample (no jitter). Supervision stays strictly per-voxel, so importance-sampling masks
            compose freely (no neighbor voxels needed).
        :param voxels_centered_around_origin: Whether the voxel grid is centered around the origin ([-1, 1]) or starts from the origin ([0, 1]).
        """
        super().__init__(learning_rate=learning_rate, normalizer=normalizer)
        self.voxel_supersampling = max(int(voxel_supersampling), 1)
        self.voxels_centered_around_origin = voxels_centered_around_origin
        self.relevance_discriminator: "FeedforwardPointwiseModel" | None = None
        self._base_voxel_map = None

    def forward(self, x: Union[DirectionalInput, PositionalInput], global_parameters: Tensor | None = None,
                region_state: Tensor | None = None) -> RadiationField:
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

    @property
    def encoder_frame_extent(self) -> float:
        """Extent of the coordinate frame the location encoder sees: [-1,1] -> 2, [0,1] -> 1.

        The single source of this convention. One voxel of an N^3 grid is `extent / N` wide, which
        is the region a region-aware encoder integrates over — so the deployed model carries this
        number and the C++ runtime derives the voxel width for whatever grid it picks.
        """
        return 2.0 if self.voxels_centered_around_origin else 1.0

    @property
    def location_encoder(self):
        """The encoder the queried POSITION goes through (None if the core has none)."""
        return getattr(self.get_core_model(), "positional_location_encoding", None)

    def voxel_width_in_encoder_frame(self, voxel_counts: Tensor) -> float:
        """Width of ONE voxel in the coordinate frame the location encoder sees."""
        n = float(torch.as_tensor(voxel_counts).float().mean())
        return self.encoder_frame_extent / n

    def estimate_step_activation_bytes(self, batch_size: int, voxel_counts) -> int:
        """Rough activation memory ONE optimizer step retains for the backward pass, in bytes.

        A whole-field step predicts every voxel of every field in the batch. forward2volume splits
        that into chunks of ``max_inner_batch_size``, but all chunks belong to the SAME loss, so
        every chunk's activations stay alive until backward: the retained memory scales with
        batch_size x voxels, NOT with the chunk size. Lowering max_inner_batch_size therefore does
        not fix an out-of-memory here — fewer voxels per step does (smaller batch_size, with
        `effective_batch_size` restoring the effective batch via gradient accumulation; a smaller
        voxel_resolution; or an importance sampler, which drops most voxels before the forward).

        Counts one tensor per Linear output in the trunk and heads plus the encoded location and
        the two skip/fusion tensors — an estimate, not an allocator-accurate figure.
        """
        core = self.get_core_model()
        floats_per_point = 0
        for name in ("block1", "block2", "spectra_decoder", "flux_decoder"):
            module = getattr(core, name, None)
            if module is not None:
                floats_per_point += sum(m.out_features for m in module.modules() if isinstance(m, nn.Linear))
        enc = getattr(core, "positional_location_encoding", None)
        floats_per_point += int(getattr(enc, "encoded_dims", 0)) + 2 * int(getattr(core, "d_model", 0))

        counts = [int(c) for c in (voxel_counts.tolist() if isinstance(voxel_counts, Tensor) else voxel_counts)]
        voxels = 1
        for c in counts:
            voxels *= c
        points = int(batch_size) * voxels * max(int(self.voxel_supersampling), 1)
        bytes_per_float = 2 if getattr(self, "_precision", "fp32") == "fp16" else 4
        return points * floats_per_point * bytes_per_float

    def deploy_interface(self):
        """The beam interface of the base, plus the two things only a per-voxel model can say:
        that it is queried at POSITIONS, and how its location encoder must be configured for the
        grid the consumer chooses."""
        from radfield3dnn.deploy import ModelInput
        iface = super().deploy_interface()
        iface.inputs |= ModelInput.POSITION
        enc = self.location_encoder
        dims = int(enc.region_state_dims) if enc is not None else 0
        iface.resolution_aware = dims > 0
        iface.region_state_dims = dims
        iface.region_width_frame = self.encoder_frame_extent
        return iface

    def region_state_for_grid(self, voxel_counts: Tensor, device, dtype) -> Tensor | None:
        """Region-configuration vector for the grid being queried, or None if the location encoder
        is region-agnostic. Cached: recomputed only when the grid width actually changes."""
        enc = self.location_encoder
        if enc is None or enc.region_state_dims == 0:
            return None
        width = self.voxel_width_in_encoder_frame(voxel_counts)
        if getattr(self, "_region_state_width", None) != width or getattr(self, "_region_state", None) is None:
            self._region_state = enc.compute_region_state(torch.tensor(width, device=device, dtype=torch.float32))
            self._region_state_width = width
        return self._region_state.to(dtype)

    def set_query_region(self, region_width: float | None):
        """Announce the spatial extent each queried point represents to the location encoder.

        Region-aware encoders (integrated/anti-aliased) prefilter their features with it; point
        encoders ignore it. forward2volume calls this automatically for the grid it assembles;
        callers that query the model DIRECTLY at single voxels / arbitrary positions must call it
        themselves (e.g. ``model.set_query_region(model.voxel_width_in_encoder_frame(counts))``),
        otherwise a region-aware encoder has no way to know the scale it is being asked about.
        """
        enc = getattr(self.get_core_model(), "positional_location_encoding", None)
        if enc is not None:
            enc.set_query_region(region_width)

    def prepare_linear_input_batches(self, x: DirectionalInput, voxel_counts: Tensor, mask: Tensor | None = None):
        batched_input = len(x.direction.shape) == 2
        
        if not batched_input:
            # unsqueeze every tensor field; _replace keeps the concrete input type (e.g. TranslationalInput)
            x = x._replace(**{f: v.unsqueeze(0) for f, v in zip(x._fields, x) if isinstance(v, Tensor)})
        batch_size = x.direction.shape[0]

        # Calculate total voxels per batch item
        total_voxels_per_batch = int(voxel_counts.prod().item())

        # K in-voxel samples per voxel while TRAINING (their predictions are voxel-mean-reduced in
        # forward2volume); eval always queries the single node-grid position.
        k = self.voxel_supersampling if self.training else 1

        # Create voxel map for all batch items efficiently
        voxel_map = self.base_voxel_map.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        if k > 1:
            # Rescale the i/(N-1) node grid to voxel lower corners i/N so the always-positive
            # in-voxel jitter (added per replica below) fills cell i = [i/N, (i+1)/N) without
            # overflowing the box.
            voxel_extent = 1.0 / voxel_counts.float().view(1, 1, 1, 1, -1)
            voxel_map = voxel_map.clone()
            voxel_map[..., :3] = voxel_map[..., :3] * (1.0 - voxel_extent)

        # allocate memory for the full voxel map an make it contiguous
        voxel_map = voxel_map.contiguous()

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

        if k > 1:
            # Replicate AFTER the importance mask so every KEPT voxel gets exactly k in-voxel
            # samples (supervision is per-voxel — no neighbor voxels are involved — so this
            # composes with any voxel-dropping sampler). linear_indices keep the ORIGINAL voxel
            # id per replica: forward2volume mean-reduces by these ids.
            voxel_list = voxel_list.repeat_interleave(k, dim=0).clone()
            direction_list = direction_list.repeat_interleave(k, dim=0)
            batch_indices = batch_indices.repeat_interleave(k, dim=0)
            linear_indices = linear_indices.repeat_interleave(k, dim=0)
            voxel_extent_flat = (1.0 / voxel_counts.float()).to(voxel_list.device).view(1, -1)
            voxel_list[:, :3] = voxel_list[:, :3] + torch.rand_like(voxel_list[:, :3]) * voxel_extent_flat

        if self.voxels_centered_around_origin:
            # map xyz from [0,1] to [-1,1] for the model
            voxel_list = voxel_list.clone() if k == 1 else voxel_list
            voxel_list[:, :3] = (voxel_list[:, :3] * 2.0) - 1.0

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
        # Announce the query geometry: each assembled sample represents one voxel of this grid.
        # Region-aware location encoders prefilter with it; point encoders ignore it.
        self.set_query_region(self.voxel_width_in_encoder_frame(voxel_counts))
        region_state = self.region_state_for_grid(voxel_counts, x.direction.device, torch.float32)

        if self.max_inner_batch_size is None:
            self.on_fit_start()

        batched_input = len(x.direction.shape) == 2
        batch_size = x.direction.shape[0] if batched_input else 1
        # Calculate total voxels per batch item
        total_voxels_per_batch = int(voxel_counts.prod().item())
        
        # Process in chunks respecting max_inner_batch_size
        full_field: RadiationField = None
        # With voxel_supersampling k>1 (training only), every kept voxel contributes exactly k
        # jittered rows sharing the SAME linear index; the scatters below become index_add_
        # accumulations and the volume is divided by k afterwards (per-voxel mean prediction,
        # matching the voxel-averaged ground truth).
        k = self.voxel_supersampling if self.training else 1
        written = None
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

            # positional_like (not a bare PositionalInput): the field-level input may carry beam
            # parameters PositionalInput has no field for — TranslationalInput's patient
            # translation — and dropping them here leaves the chunk unencodable by the model's own
            # beam encoder whenever global_parameters is not precomputed.
            _translation = getattr(x, "translation", None)
            pred_field = self.forward(
                positional_like(
                    x,
                    direction=directions,
                    position=voxels[:, 0:3],
                    spectrum=x.spectrum.index_select(0, batch_idx) if x.spectrum is not None else None,
                    geometry=x.geometry.index_select(0, batch_idx) if x.geometry is not None else None,
                    origin=x.origin.index_select(0, batch_idx) if x.origin is not None else None,
                    beam_shape_parameters=x.beam_shape_parameters.index_select(0, batch_idx) if x.beam_shape_parameters is not None else None,
                    beam_shape_type=x.beam_shape_type.index_select(0, batch_idx) if x.beam_shape_type is not None else None,
                    translation=_translation.index_select(0, batch_idx) if _translation is not None else None,
                ),
                global_parameters=global_parameters.index_select(0, batch_idx) if global_parameters is not None else None,
                region_state=region_state,
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
                # k>1 accumulates via index_add_, so start from zero and re-apply the -inf
                # sentinel to never-written (sampler-dropped) voxels after the loop.
                neg_inf = 0.0 if k > 1 else float("-inf")
                if k > 1:
                    written = torch.zeros(total_samples, dtype=torch.bool, device=_dev)
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
            def _write(dst: Tensor, src: Tensor):
                src = src.to(torch.float32)
                if k > 1:
                    dst.index_add_(0, indices, src)
                else:
                    dst[indices] = src.unsqueeze(0) if (dst.dim() > 1 and src.dim() == 1) else src

            if isinstance(pred_field, RadiationField):
                if pred_field.scatter_field is not None:
                    _write(full_field.scatter_field.flux, pred_field.scatter_field.flux)
                    if pred_field.scatter_field.spectrum is not None:
                        _write(full_field.scatter_field.spectrum, pred_field.scatter_field.spectrum)
                if pred_field.direct_beam is not None:
                    _write(full_field.direct_beam.flux, pred_field.direct_beam.flux)
                    if pred_field.direct_beam.spectrum is not None:
                        _write(full_field.direct_beam.spectrum, pred_field.direct_beam.spectrum)
                if pred_field.geometry is not None:
                    _write(full_field.geometry, pred_field.geometry)
            elif isinstance(pred_field, AirKermaField):
                _write(full_field.air_kerma, pred_field.air_kerma)
                if pred_field.geometry is not None:
                    _write(full_field.geometry, pred_field.geometry)
            else:
                raise ValueError("Unknown field type returned by the model.")
            if k > 1:
                written[indices] = True

        if k > 1:
            # per-voxel mean over the k replicas; sampler-dropped voxels get the -inf sentinel
            def _finalize(t: Tensor):
                if t is None:
                    return None
                t = t / k
                t[~written] = float("-inf")
                return t
            if isinstance(full_field, RadiationField):
                full_field = full_field._replace(
                    scatter_field=full_field.scatter_field._replace(
                        flux=_finalize(full_field.scatter_field.flux),
                        spectrum=_finalize(full_field.scatter_field.spectrum)) if full_field.scatter_field is not None else None,
                    direct_beam=full_field.direct_beam._replace(
                        flux=_finalize(full_field.direct_beam.flux),
                        spectrum=_finalize(full_field.direct_beam.spectrum)) if full_field.direct_beam is not None else None,
                    geometry=(full_field.geometry / k) if full_field.geometry is not None else None,
                )
            elif isinstance(full_field, AirKermaField):
                full_field = full_field._replace(
                    air_kerma=_finalize(full_field.air_kerma),
                    geometry=(full_field.geometry / k) if full_field.geometry is not None else None,
                )

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

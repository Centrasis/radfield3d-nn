import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch import Tensor
from typing import Type
from radfield3dnn.normalizations import Normalizer, LinearNormalizer, NormalizerConstructor
from radfield3dnn.encodings.sinusoidal_encoding import SinusoidalFrequencyEncoding
from radfield3dnn.encodings.hash_encoding import HashGridEncoding
from radfield3dnn.rftypes import AirKermaField, RadiationField, PositionalInput, TrainingInputData, RadiationFieldChannel, DirectionalInput, PositionalInput
import gc
from rich import print
from typing import Union
from radfield3dnn.losses.base import Loss
from radfield3dnn.metrics.types import TrainingMetrics, ChannelMetrics
import radfield3dnn.losses.std as std
import radfield3dnn.losses.combinations as comb_loss
import torch.nn.functional as F
from radfield3dnn.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.datasets.channel_join import ChannelsJoin


class ModuleBuilder:
    @staticmethod
    def ConstructLoss_fn(loss_fn_name: str) -> nn.Module:
        if loss_fn_name == "L1Loss":
            return std.L1LossWeighted(False)
        elif loss_fn_name == "MSELoss":
            return std.MSELossWeighted(False)
        elif loss_fn_name == "CrossEntropyLoss":
            return std.CrossEntropyLossWeighted(False)
        elif loss_fn_name == "KLDivLoss":
            return std.KLDivLossWeighted(False)
        elif loss_fn_name == "BCELoss":
            raise NotImplementedError("BCELoss is not supported.")
        elif loss_fn_name == "BCEWithLogitsLoss":
            raise NotImplementedError("BCEWithLogitsLoss is not supported.")
        elif loss_fn_name == "SmoothL1Loss":
            raise NotImplementedError("SmoothL1Loss is not supported.")
        elif loss_fn_name == "L1LogLoss":
            return std.L1LossLogSpace(False)
        elif loss_fn_name == "WassersteinLoss":
            return std.WassersteinLossWeighted(dim=1, weight_with_error=False)
        elif loss_fn_name == "HistogramLoss":
            return comb_loss.HistogramLoss(bin_dim=1, weight_with_error=False, penalize_out_of_range=False, calc_moments=False)
        elif loss_fn_name == "MSELogLoss":
            return std.MSELossLogSpace(weight_with_error=False)
        elif loss_fn_name == "PoissonNLLLoss":
            return std.PoissonNLLLoss(weight_with_error=False)
        elif loss_fn_name == "FocalMSELoss":
            return std.FocalMSELoss(weight_with_error=False)
        elif loss_fn_name == "FocalSmoothL1Loss":
            return std.FocalSmoothL1Loss(weight_with_error=False)
        elif loss_fn_name == "StructuralSimilarity3DLoss":
            return std.StructuralSimilarity3DLoss(weight_with_error=False)
        elif loss_fn_name == "MultiScaleStructuralSimilarity3DLoss":
            return std.MultiScaleStructuralSimilarity3DLoss(weight_with_error=False)
        elif loss_fn_name == "FluxLoss":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False)
        elif loss_fn_name == "FluxLogLoss":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=True)
        elif loss_fn_name == "FluxMultiScaleLoss":
            return comb_loss.FluxMultiScaleLoss(weight_with_error=False)
        else:
            raise ValueError(f"Invalid loss function name: {loss_fn_name}")
        
    @staticmethod
    def ConstructActivation_fn(activation_fn_name: str) -> Type[nn.Module]:
        if activation_fn_name == "ReLU":
            return nn.ReLU
        elif activation_fn_name == "LeakyReLU":
            return nn.LeakyReLU
        elif activation_fn_name == "Sigmoid":
            return nn.Sigmoid
        elif activation_fn_name == "Softmax":
            return nn.Softmax
        elif activation_fn_name == "Tanh":
            return nn.Tanh
        elif activation_fn_name == "Identity":
            return nn.Identity
        elif activation_fn_name == "ELU":
            return nn.ELU
        elif activation_fn_name == "SELU":
            return nn.SELU
        elif activation_fn_name == "GELU":
            return nn.GELU
        elif activation_fn_name == "SiLU":
            return nn.SiLU
        else:
            raise ValueError(f"Invalid activation function name: {activation_fn_name}")

    @staticmethod
    def ConstructNormalization_fn(normalization_fn_name: str) -> Type[nn.Module]:
        if normalization_fn_name == "BatchNorm1d":
            return nn.BatchNorm1d
        elif normalization_fn_name == "BatchNorm2d":
            return nn.BatchNorm2d
        elif normalization_fn_name == "BatchNorm3d":
            return nn.BatchNorm3d
        elif normalization_fn_name == "LayerNorm":
            return nn.LayerNorm
        elif normalization_fn_name == "InstanceNorm1d":
            return nn.InstanceNorm1d
        elif normalization_fn_name == "InstanceNorm2d":
            return nn.InstanceNorm2d
        elif normalization_fn_name == "InstanceNorm3d":
            return nn.InstanceNorm3d
        elif normalization_fn_name == "GroupNorm":
            return nn.GroupNorm
        elif normalization_fn_name == "LocalResponseNorm":
            return nn.LocalResponseNorm
        else:
            raise ValueError(f"Invalid normalization function name: {normalization_fn_name}")
        
    @staticmethod
    def ConstructEncoding_fn(encoding_fn_name: str) -> Type[nn.Module]:
        if encoding_fn_name == "SinusoidalFrequencyEncoding":
            return SinusoidalFrequencyEncoding
        elif encoding_fn_name == "HashGridEncoding":
            return HashGridEncoding
        else:
            raise ValueError(f"Invalid encoding function name: {encoding_fn_name}")


class AngularSinusoidalFrequencyEncoding(SinusoidalFrequencyEncoding):
    def __init__(self, pos_enc_dim, append_input = False, dim = -1):
        super().__init__(pos_enc_dim, 2, append_input, dim)

    def forward(self, x):
        assert x.shape[-1] == 3, f"Input tensor last dim should be 3, got {x.shape[-1]}"
        x = AngularSinusoidalFrequencyEncoding.map_direction_vector2spherical_coords(x)
        return super().forward(x)
    
    @staticmethod
    def map_direction_vector2spherical_coords(direction: Tensor) -> Tensor:
        """Convert direction vectors (x,y,z) to spherical coordinates (theta, phi).
        This version uses torch.atan2 for better numerical stability.
        Args:
            direction: Tensor of shape (..., 3) containing cartesian direction vectors
        Returns:
            Tensor of shape (..., 2) containing spherical coordinates (theta, phi)
        """
        # Normalize the vectors
        direction = F.normalize(direction, dim=-1, p=2)
        
        # Extract x, y, z components
        x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
        
        # Convert to spherical coordinates
        # theta: angle from z-axis (0 to π)
        # phi: angle in xy-plane from x-axis (0 to 2π)
        theta = torch.acos(torch.clamp(z, -1.0 + 1e-8, 1.0 - 1e-8))
        phi = torch.atan2(y, x)
        
        # Normalize to [0, 1] range for more stable training
        theta = theta / torch.pi
        phi = (phi + torch.pi) / (2 * torch.pi)
        
        return torch.stack([theta, phi], dim=-1)


class BaseNeuralRadFieldModel(pl.LightningModule):
    __model_name__: str = None

    lr = property(lambda x: x.get_lr(), lambda x, v: x.set_lr(v))
    learning_rate = property(lambda x: x.get_lr(), lambda x, v: x.set_lr(v))

    def __init__(self, location_encoding_dims=10, direction_encoding_dims=4, d_model=256, normalizer: Union[Normalizer, str] = LinearNormalizer(), learning_rate: float=1e-3):
        super().__init__()
        self.logging_prefix = ""
        
        self.positional_location_encoding = SinusoidalFrequencyEncoding(location_encoding_dims, 3, append_input=True)
        self.positional_direction_encoding = AngularSinusoidalFrequencyEncoding(direction_encoding_dims, append_input=True)
        self.d_model = d_model
        self._lr = learning_rate
        self._flux_loss_fn: Loss = comb_loss.FluxLoss(weight_with_error=False)
        self._spectrum_loss_fn: Loss = comb_loss.HistogramLoss(bin_dim=1, weight_with_error=False, penalize_out_of_range=False)
        self._normalizer = normalizer if not isinstance(normalizer, str) else NormalizerConstructor.construct_by_name(normalizer)
        self.max_inner_batch_size = None
        self.indices: Tensor = None
        self.grid_dims: Tensor = None
        self.batch_size = 1
        self._channels_join = ChannelsJoin()
        assert isinstance(self._normalizer, Normalizer), f"normalizer must be an instance of Normalizer, got {type(self._normalizer)}"
        self.save_hyperparameters(ignore=["indices", "grid_dims", "_flux_loss_fn", "_spectrum_loss_fn", "normalizer", "_channels_join"])

    def _generate_random_ground_truth(self, device) -> RadiationField:
        input = self._generate_random_input(device=device)
        return self.forward(input)
    
    def get_submodels(self) -> list["BaseNeuralRadFieldModel"]:
        return []

    def _generate_random_input(self, device, batch_size=2) -> PositionalInput:
        return PositionalInput(
            direction=torch.rand(batch_size, 3, device=device),
            spectrum=HistogramNormalize(dim=-1)(torch.rand(batch_size, 150, device=device)),
            position=torch.rand(batch_size, 3, device=device),
            origin=torch.rand(batch_size, 3, device=device),
            geometry=None,
            beam_shape_parameters=torch.rand(batch_size, 1, device=device),
            beam_shape_type=torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
        )

    def _search_optimal_batch_size(self):
        """
        This method will simulate mutliple iterations of this network using a forward and a backward pass with incresing the batch size each.
        Thus, this method will execute the model for n² batch sizes until a CUDA out of memory error occurs or the memory is filled to 90% capacity.
        The biggest possible batch size will be stored to self.max_inner_batch_size.
        NOTE: In order to skip the execution of this method set max_inner_batch_size to a value != None and > 0.
        """
        print(f"[yellow]Try finding max inner batch_size...")
        self.max_inner_batch_size = 2
        device = next(self.parameters()).device
        safety_margin = 0.9
        if device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(device).total_memory

        y_base = self._generate_random_ground_truth(device=device)
        y_base = RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=y_base.scatter_field.spectrum[0] if y_base.scatter_field.spectrum is not None else None,
                flux=y_base.scatter_field.flux[0],
                error=torch.zeros_like(y_base.scatter_field.flux[0])
            ),
            direct_beam=RadiationFieldChannel(
                spectrum=y_base.scatter_field.spectrum[0] if y_base.scatter_field.spectrum is not None else None,
                flux=y_base.scatter_field.flux[0],
                error=torch.zeros_like(y_base.scatter_field.flux[0])
            )
        )
        train_in = TrainingInputData(
            input=self._generate_random_input(device=device),
            ground_truth=y_base
        )

        # calculate scaling factor for the spectra and flux loss functions
        num_losses = 1000
        if y_base.scatter_field.spectrum is not None:
            random_spectra1 = torch.rand((num_losses, y_base.scatter_field.spectrum.shape[0]), device=device)
            random_spectra1 = random_spectra1 / random_spectra1.sum(dim=1, keepdim=True)
        else:
            random_spectra1 = None
        random_flux1 = torch.rand((num_losses, 1), device=device)
        if y_base.scatter_field.spectrum is not None:
            random_spectra2 = torch.rand((num_losses, y_base.scatter_field.spectrum.shape[0]), device=device)
            random_spectra2 = random_spectra2 / random_spectra2.sum(dim=1, keepdim=True)
        else:
            random_spectra2 = None
        random_flux2 = torch.rand((num_losses, 1), device=device)

        loss_test_in = TrainingInputData(
            input=train_in.input,
            ground_truth=RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=random_spectra1,
                    flux=random_flux1,
                    error=torch.zeros_like(random_flux1)
                ),
                direct_beam=RadiationFieldChannel(
                    spectrum=random_spectra2,
                    flux=random_flux2,
                    error=torch.zeros_like(random_flux2)
                )
            )
        )

        _ = self._spectrum_loss_fn.forward(prediction=random_spectra1, target=random_spectra2, input=loss_test_in) if random_spectra1 is not None else torch.tensor(0.0, device=device)
        _ = self._flux_loss_fn.forward(prediction=random_flux1, target=random_flux2, input=loss_test_in) 

        while True:
            try:
                torch.cuda.empty_cache()
                gc.collect()

                if device.type == 'cuda':
                    before_memory = torch.cuda.memory_allocated()

                scatter_channel = RadiationFieldChannel(
                    spectrum=y_base.scatter_field.spectrum,
                    flux=y_base.scatter_field.flux,
                    error=y_base.scatter_field.error
                )
                xray_channel = RadiationFieldChannel(
                    spectrum=y_base.direct_beam.spectrum,
                    flux=y_base.direct_beam.flux,
                    error=y_base.direct_beam.error
                ) if y_base.direct_beam is not None else scatter_channel
                y = RadiationField(
                    scatter_field=scatter_channel,
                    direct_beam=xray_channel,
                    geometry=torch.zeros_like(y_base.scatter_field.flux) if y_base.geometry is not None else None
                )
                batch_size = self.max_inner_batch_size * 2
                y_scatter_flu = y.scatter_field.flux
                y_scatter_spec = y.scatter_field.spectrum
                y_xray_flu = y.direct_beam.flux
                y_xray_spec = y.direct_beam.spectrum
                if y.scatter_field.spectrum is not None:
                    y_scatter_spec = y.scatter_field.spectrum.unsqueeze(0)
                    y_xray_spec = y.direct_beam.spectrum.unsqueeze(0)

                y_scatter_flu = y.scatter_field.flux.unsqueeze(0)
                y_xray_flu = y.direct_beam.flux.unsqueeze(0)

                xray_err = torch.rand_like(y_xray_flu).expand(batch_size, *([-1] * (len(y_xray_flu.shape) - 1)))
                if len(xray_err.shape) == 1:
                    xray_err = xray_err.unsqueeze(1)
                scatter_err = torch.rand_like(y_scatter_flu).expand(batch_size, *([-1] * (len(y_scatter_flu.shape) - 1)))
                if len(scatter_err.shape) == 1:
                    scatter_err = scatter_err.unsqueeze(1)
                y = RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=y_scatter_spec.expand(batch_size, *([-1] * (len(y_scatter_spec.shape) - 1))) if y_scatter_spec is not None else None,
                        flux=y_scatter_flu.expand(batch_size, *([-1] * (len(y_scatter_flu.shape) - 1))),
                        error=scatter_err
                    ),
                    direct_beam=RadiationFieldChannel(
                        spectrum=y_xray_spec.expand(batch_size, *([-1] * (len(y_xray_spec.shape) - 1))) if y_xray_spec is not None else None,
                        flux=y_xray_flu.expand(batch_size, *([-1] * (len(y_xray_flu.shape) - 1))),
                        error=xray_err
                    ),
                    geometry=torch.zeros_like(y_scatter_flu).expand(batch_size, *([-1] * (len(y_scatter_flu.shape) - 1))) if y.geometry is not None else None
                )
                x = self._generate_random_input(device=device, batch_size=batch_size)

                batch = TrainingInputData(input=x, ground_truth=y)
                _ = self._search_optimal_batch_size_evaluate_forward(batch)

                if device.type == 'cuda':
                    after_memory = torch.cuda.memory_allocated()
                    memory_used = after_memory - before_memory
                    available_memory = total_memory * safety_margin - after_memory
                    if available_memory < memory_used * 2:
                        raise torch.cuda.OutOfMemoryError("Not enough memory available for more batches.")

                print(f"[blue]{self.max_inner_batch_size} ", end="")

                gc.collect()
                torch.cuda.empty_cache()
                self.max_inner_batch_size = batch_size
            except torch.cuda.OutOfMemoryError:
                print(f"[green]{self.max_inner_batch_size}")
                break
            except Exception as e:
                error_msg = str(e)
                if "DefaultCPUAllocator" in error_msg or "CUDA out of memory" in error_msg:
                    print(f"[green]{self.max_inner_batch_size}")
                else:
                    raise e
                break

    def _search_optimal_batch_size_evaluate_forward(self, batch: TrainingInputData):
        return self.evaluate_forward(batch)

    def on_fit_start(self):
        if self.max_inner_batch_size is not None:
            return
        self._search_optimal_batch_size()

    def get_indices_map(self, voxel_counts: Tensor, device: str) -> Tensor:
        if self.indices is None:
            self.generate_voxelmap3d(voxel_counts, torch.zeros(1, device=device), device)
        return self.indices

    def assert_model_on_gpu(self):
        for name, param in self.named_parameters():
            if param.device.type != "cuda":
                print("PARAM WARNUNG: {name} liegt auf {param.device}!")
        for name, buf in self.named_buffers():
            if buf.device.type != "cuda":
                print(f"BUFFER WARNUNG: {name} liegt auf {buf.device}!")


    def generate_voxelmap3d(self, voxel_counts: Union[Tensor, tuple], constant_voxel_vector: Tensor, device: str) -> Tensor:
        """
        Generates a 3D voxel map based on the input voxel counts and a constant voxel vector.
        The voxel map will consist of the normalized 3D center positions (3,) of each voxel and optionally has the constant vector (C,) concatenated to it.
        Args:
            voxel_counts: The counts of voxels in each dimension.
            constant_voxel_vector: A constant vector to be concatenated to the voxel map.
            device: The device to which the tensors should be moved.
        Returns:
            A 3D voxel map tensor (D, H, W, 3 + C)
        """
        with torch.no_grad():
            voxel_counts = voxel_counts.to(device) if isinstance(voxel_counts, torch.Tensor) else voxel_counts
            constant_voxel_vector = constant_voxel_vector.to(device) if constant_voxel_vector is not None else None
            if constant_voxel_vector is not None and len(constant_voxel_vector.shape) > 1:    # ensure that the constant_voxel_vector is stripped of batch dimension == 1
                constant_voxel_vector = constant_voxel_vector.squeeze(0)
            if self.indices is None or self.grid_dims is None or self.indices.shape[0] != voxel_counts[0] or self.indices.shape[1] != voxel_counts[1] or self.indices.shape[2] != voxel_counts[2]:
                self.indices = torch.stack(torch.meshgrid(torch.arange(voxel_counts[0], device=device), torch.arange(voxel_counts[1], device=device), torch.arange(voxel_counts[2], device=device), indexing='ij'), dim=-1)
                # normalize indices to maximum in each dimension between 0 and 1
                self.grid_dims = torch.tensor(voxel_counts, dtype=torch.float32, device=device) if not isinstance(voxel_counts, torch.Tensor) else voxel_counts.to(torch.float32)
            elif self.indices.device != device:
                self.indices = self.indices.to(device)
                self.grid_dims = self.grid_dims.to(device)
            normalized_indices = torch.clamp((self.indices.to(torch.float32) / (self.grid_dims - 1.0)), 0, 1) #  + (1.0 / (2 * self.grid_dims))
            if constant_voxel_vector is not None:
                if len(constant_voxel_vector.shape) == 1:
                    voxel_map = torch.cat((normalized_indices, constant_voxel_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*voxel_counts, constant_voxel_vector.shape[-1])), dim=-1)
                elif len(constant_voxel_vector.shape) >= 2:
                    normalized_indices = normalized_indices.unsqueeze(0).expand(constant_voxel_vector.shape[0], *normalized_indices.shape)
                    constant_voxel_vector = constant_voxel_vector.view(constant_voxel_vector.shape[0], 1, 1, 1, *constant_voxel_vector.shape[1:])
                    constant_voxel_vector = constant_voxel_vector.expand(constant_voxel_vector.shape[0], *voxel_counts, constant_voxel_vector.shape[-1])
                    voxel_map = torch.cat((normalized_indices, constant_voxel_vector), dim=-1)
                else:
                    raise ValueError(f"Invalid constant_voxel_vector shape: {constant_voxel_vector.shape}. It must be a 1D or 2D tensor.")
            else:
                voxel_map = normalized_indices
            return voxel_map

    @staticmethod
    def create_radiationfield_like(field: RadiationField, batch_size: int = None) -> RadiationField:
        """
        Creates a new RadiationField with the same structure as the provided field, but with new and empty tensors.
        Args:
            field (RadiationField): The RadiationField to use as a template. Field shall not have a batch dimension.
            batch_size (int, optional): If provided, the new RadiationField will have this batch size. If None, the batch size will be inferred from the field's flux tensor shape.
        Raises:
            AssertionError: If the scatter_field.flux tensor is not 4D (i.e., not batched).
        Returns:
            RadiationField: A new RadiationField with empty tensors.
        """
        assert len(field.scatter_field.flux.shape) == 4, "The scatter_field.flux must be a 4D tensor (not batched)."

        flux_field_shape = [batch_size] + list(field.scatter_field.flux.shape) if batch_size is not None else list(field.scatter_field.flux.shape)
        spectrum_field_shape = [batch_size] + list(field.scatter_field.spectrum.shape) if field.scatter_field.spectrum is not None and batch_size is not None else list(field.scatter_field.spectrum.shape) if field.scatter_field.spectrum is not None else None
        error_field_shape = [batch_size] + list(field.scatter_field.error.shape) if field.scatter_field.error is not None and batch_size is not None else list(field.scatter_field.error.shape) if field.scatter_field.error is not None else None
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                spectrum=torch.empty(spectrum_field_shape, dtype=field.scatter_field.spectrum.dtype, device=field.scatter_field.spectrum.device) if field.scatter_field.spectrum is not None else None,
                flux=torch.empty(flux_field_shape, dtype=field.scatter_field.flux.dtype, device=field.scatter_field.flux.device) if field.scatter_field.flux is not None else None,
                error=torch.empty(error_field_shape, dtype=field.scatter_field.error.dtype, device=field.scatter_field.error.device) if field.scatter_field.error is not None else None
            ) if field.scatter_field is not None else None,
            direct_beam=RadiationFieldChannel(
                spectrum=torch.empty(spectrum_field_shape, dtype=field.direct_beam.spectrum.dtype, device=field.direct_beam.spectrum.device) if field.direct_beam.spectrum is not None else None,
                flux= torch.empty(flux_field_shape, dtype=field.direct_beam.flux.dtype, device=field.direct_beam.flux.device) if field.direct_beam.flux is not None else None,
                error=torch.empty(error_field_shape, dtype=field.direct_beam.error.dtype, device=field.direct_beam.error.device) if field.direct_beam.error is not None else None
            ) if field.direct_beam is not None else None,
            geometry=torch.empty(flux_field_shape, dtype=field.geometry.dtype, device=field.geometry.device) if field.geometry is not None else None
        )

    def forward2volume_from_training_input(self, batch: TrainingInputData, voxel_counts: Tensor = None, spectra_bins: int = 32) -> RadiationField:
        sample_field = batch.ground_truth.scatter_field.flux if isinstance(batch.ground_truth, RadiationField) else (batch.ground_truth.flux if isinstance(batch.ground_truth, RadiationFieldChannel) else batch.ground_truth.air_kerma)
        is_complete_volume = (sample_field is not None) and (len(sample_field.shape) == 4 or (len(sample_field.shape) == 5 and sample_field.shape[1] == 1))
        if is_complete_volume:
            if len(sample_field.shape) == 5 and sample_field.shape[1] == 1:
                voxel_counts = sample_field.shape[2:]
            elif len(sample_field.shape) == 4:
                voxel_counts = sample_field.shape[1:]
            else:
                raise ValueError(f"Invalid y_hits shape: {len(sample_field.shape)}.")
            
        if voxel_counts is None:
            raise ValueError("voxel_counts must be provided when using forward2volume_from_training_input.")

        drop_mask = None
        if isinstance(batch.ground_truth, RadiationField):
            drop_mask = torch.isneginf(batch.ground_truth.scatter_field.flux)
            if batch.ground_truth.direct_beam is not None and batch.ground_truth.direct_beam.flux is not None:
                drop_mask = drop_mask | torch.isneginf(batch.ground_truth.direct_beam.flux)
        elif isinstance(batch.ground_truth, RadiationFieldChannel):
            drop_mask = torch.isneginf(batch.ground_truth.flux)
        elif isinstance(batch.ground_truth, AirKermaField):
            drop_mask = torch.isneginf(batch.ground_truth.air_kerma)
        else:
            raise ValueError("Ground truth must be of type RadiationField, RadiationFieldChannel, or AirKermaField.")
        pred_field = self.forward2volume(batch.input, voxel_counts, spectra_bins=spectra_bins, mask=drop_mask)
        return pred_field

    def extract_input_from_batch(self, input: DirectionalInput, batch_idx: int) -> DirectionalInput:
        """
        Extracts a single batch element from the input DirectionalInput.
        Overload, if the input for self.forward2volume(...) is different (e.g., PositionalInput, GeometricInput, ...).
        Args:
            input (DirectionalInput): The input containing direction and spectrum tensors.
            batch_idx (int): The index of the batch to extract.
        Returns:
            DirectionalInput: A new DirectionalInput instance containing the specified batch.
        """
        assert isinstance(input, DirectionalInput), "Input must be of type DirectionalInput."
        return DirectionalInput(
            direction=input.direction[batch_idx],
            spectrum=input.spectrum[batch_idx],
            geometry=input.geometry[batch_idx] if input.geometry is not None else None,
            origin=input.origin[batch_idx] if input.origin is not None else None,
            beam_shape_parameters=input.beam_shape_parameters[batch_idx] if input.beam_shape_parameters is not None else None,
            beam_shape_type=input.beam_shape_type[batch_idx] if input.beam_shape_type is not None else None
        )

    def forward2volume(self, x: DirectionalInput, voxel_counts: Tensor, spectra_bins: int = 32, mask: Tensor | None = None) -> RadiationField:
        raise NotImplementedError("This method must be implemented by the subclass.")

    def forward(self, x: Union[DirectionalInput, PositionalInput]) -> RadiationField:
        raise NotImplementedError("This method must be implemented by the subclass.")

    def evaluate_forward(self, batch: TrainingInputData) -> TrainingMetrics:
        batch = self._normalizer.forward(batch)
        y = batch.ground_truth
        x = batch.input
        self.batch_size = int(x.direction.shape[0])
        if isinstance(y, RadiationField) or isinstance(y, RadiationFieldChannel):
            has_multi_channel = y.direct_beam is not None if isinstance(y, RadiationField) else False
            scatter_field_gt = y.scatter_field if isinstance(y, RadiationField) else y
            is_complete_volume = (len(scatter_field_gt.flux.shape) == 4 or (len(scatter_field_gt.flux.shape) == 5 and scatter_field_gt.flux.shape[1] == 1)) and len(scatter_field_gt.spectrum.shape) == 5
        elif isinstance(y, AirKermaField):
            has_multi_channel = False
            is_complete_volume = (len(y.air_kerma.shape) == 4 or (len(y.air_kerma.shape) == 5 and y.air_kerma.shape[1] == 1))
            scatter_field_gt = batch.original_ground_truth.scatter_field if batch.original_ground_truth is not None and isinstance(batch.original_ground_truth, RadiationField) else None
        else:
            raise ValueError("Ground truth must be of type RadiationField, RadiationFieldChannel, or AirKermaField.")

        if is_complete_volume:
            pred_field: RadiationField | AirKermaField = self.forward2volume_from_training_input(batch, spectra_bins=scatter_field_gt.spectrum.shape[1] if scatter_field_gt.spectrum is not None else 32)
        else:
            pred_field: RadiationField | AirKermaField = self(x)

        if isinstance(y, RadiationField) or isinstance(y, RadiationFieldChannel):
            if has_multi_channel and (pred_field.scatter_field is None and pred_field.direct_beam is None):
                raise ValueError("The model should not return both scatter_field and direct_beam in the same forward pass when has_multi_channel is True.")
            elif not has_multi_channel and ((pred_field.scatter_field is not None) and (pred_field.direct_beam is not None)):
                raise ValueError("The model should return either scatter_field or direct_beam, but not both when has_multi_channel is False.")

        return self.calculate_metrics(pred_field, y, batch)

    def calculate_metrics(self, pred_field: RadiationField | AirKermaField, y: RadiationField | AirKermaField, batch: TrainingInputData, ignore_scatter: bool = False, ignore_direct_beam: bool = False) -> TrainingMetrics:
        scatter_metrics: ChannelMetrics = None
        direct_beam_metrics: ChannelMetrics = None

        if isinstance(pred_field, RadiationField):
            has_multi_channel = y.direct_beam is not None if isinstance(y, RadiationField) else False
            scatter_field_gt = y.scatter_field if isinstance(y, RadiationField) else y

            if not has_multi_channel and (pred_field.scatter_field is not None and pred_field.direct_beam is not None):
                raise ValueError("The model should return either scatter_field or direct_beam, but not both when has_multi_channel is False.")
            
            if has_multi_channel and (pred_field.scatter_field is None or pred_field.direct_beam is None):
                # if network is only predicting one channel, join channels for loss calculation
                batch = self._normalizer.inverse(batch)
                y = self._normalizer.inverse(y)
                batch = self._channels_join(batch)
                batch = self._normalizer.forward(batch)
                y = self._channels_join(y)
                y = self._normalizer.forward(y)
                scatter_field_gt = y
                has_multi_channel = False

            if pred_field.direct_beam is not None and has_multi_channel and not ignore_direct_beam:
                loss_spec = None
                if pred_field.direct_beam.spectrum is not None:
                    loss_spec = self._spectrum_loss_fn.forward(prediction=pred_field.direct_beam.spectrum, target=y.direct_beam.spectrum, input=batch)

                loss_flux = self._flux_loss_fn.forward(prediction=pred_field.direct_beam.flux, target=y.direct_beam.flux, input=batch)

                direct_beam_metrics = ChannelMetrics(
                    flux_loss=loss_flux,
                    spectrum_loss=loss_spec if loss_spec is not None else None,
                )

            if pred_field.scatter_field is not None and not ignore_scatter:
                loss_spec = None
                if pred_field.scatter_field.spectrum is not None:
                    loss_spec = self._spectrum_loss_fn.forward(prediction=pred_field.scatter_field.spectrum, target=scatter_field_gt.spectrum, input=batch)
                    if not torch.isfinite(loss_spec).all():
                        print(f"[red] Spectrum loss is not finite. Setting to 1.")
                        raise ValueError("Spectrum loss is not finite.")

                loss_flux = self._flux_loss_fn.forward(prediction=pred_field.scatter_field.flux, target=scatter_field_gt.flux, input=batch)

                if not torch.isfinite(loss_flux).all():
                    print(f"[red] Flux loss is not finite. Setting to 1.")
                    raise ValueError("Flux loss is not finite.")

                if direct_beam_metrics is not None and y.direct_beam is not None:
                    sum_flux_ratio = y.direct_beam.flux.sum() / scatter_field_gt.flux.sum()
                    loss_flux = loss_flux * sum_flux_ratio

                scatter_metrics = ChannelMetrics(
                    flux_loss=loss_flux,
                    spectrum_loss=loss_spec
                )

            return TrainingMetrics(
                scatter_field=scatter_metrics,
                direct_beam=direct_beam_metrics
            )
        elif isinstance(pred_field, AirKermaField):
            assert isinstance(y, AirKermaField), "Ground truth must be of type AirKermaField when predicting AirKermaField."
            loss_airkerma = self._flux_loss_fn.forward(prediction=pred_field.air_kerma, target=y.air_kerma, input=batch)
            return TrainingMetrics(
                airkerma_field=loss_airkerma
            )
        else:
            raise ValueError("pred_field must be of type RadiationField or AirKermaField.")
    
    def process_metrics(self, metrics: TrainingMetrics, stage: str) -> Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        on_epoch = stage in ["val", "test"]

        if len(self.logging_prefix) > 0:
            stage = stage + "." + self.logging_prefix

        on_step = not on_epoch
        if metrics.scatter_field is not None:
            total_loss = metrics.scatter_field.flux_loss
            if metrics.scatter_field.spectrum_loss is not None:
                loss_weight = 0.5
                total_loss = total_loss * loss_weight + metrics.scatter_field.spectrum_loss * (1 - loss_weight)
                self.log(f'{stage}_scatter_spectrum_loss', metrics.scatter_field.spectrum_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
            self.log(f'{stage}_scatter_flux_loss', metrics.scatter_field.flux_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        if metrics.direct_beam is not None:
            if metrics.direct_beam.spectrum_loss is not None:
                loss_weight = 0.5
                total_loss = total_loss + metrics.direct_beam.flux_loss * loss_weight + metrics.direct_beam.spectrum_loss * (1 - loss_weight)
                self.log(f'{stage}_direct_beam_spectrum_loss', metrics.direct_beam.spectrum_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
            else:
                total_loss = total_loss + metrics.direct_beam.flux_loss
            self.log(f'{stage}_direct_beam_flux_loss', metrics.direct_beam.flux_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        if metrics.airkerma_field is not None:
            total_loss = total_loss + metrics.airkerma_field
            self.log(f'{stage}_airkerma_loss', metrics.airkerma_field.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        if not torch.isfinite(total_loss).all():
            print(f"[red] Loss is not finite. Setting to 1.")
            total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        self.log(f'{stage}_loss', total_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
        return total_loss.mean()

    def training_step(self, batch: TrainingInputData, batch_idx):
        metrics = self.evaluate_forward(batch)
        return self.process_metrics(metrics, "train")
    
    def validation_step(self, batch: TrainingInputData, batch_idx):
        metrics = self.evaluate_forward(batch)
        return self.process_metrics(metrics, "val")
    
    def test_step(self, batch: TrainingInputData, batch_idx):
        metrics = self.evaluate_forward(batch)
        return self.process_metrics(metrics, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, fused=False, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            self._lr = param_group['lr']
        return self._lr

    def set_lr(self, lr):
        for param_group in self.optimizers().param_groups:
            param_group['lr'] = lr
        self._lr = lr

    def get_model_config(self) -> dict:
        return {
            "model_name": self.__class__.__model_name__,
            "parameters": self.get_custom_parameters()
        }

    def get_custom_parameters(self) -> dict:
        return {}

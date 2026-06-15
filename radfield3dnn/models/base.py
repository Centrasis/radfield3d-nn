import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch import Tensor
from typing import Type
from radfield3dnn.preprocessing.normalizations import Normalizer, LinearNormalizer, NormalizerConstructor
from radfield3dnn.models.encoders.sinusoidal_encoding import SinusoidalFrequencyEncoding, AngularSinusoidalFrequencyEncoding
from radfield3dnn.models.encoders.hash_encoding import HashGridEncoding
from radfield3dnn.models.encoders.spherical_hamonics import SphericalHarmonics
from radfield3dnn.rftypes import AirKermaField, RadiationField, PositionalInput, TrainingInputData, RadiationFieldChannel, DirectionalInput, PositionalInput
import gc
from rich import print
from typing import Union
from radfield3dnn.losses.base import Loss
from radfield3dnn.metrics.types import TrainingMetrics, ChannelMetrics
import radfield3dnn.losses.std as std
import radfield3dnn.losses.combinations as comb_loss
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.datasets.channel_join import ChannelsJoin
from radfield3dnn.losses.mtl.mtl import MultiTaskLossBalancer


class ModuleBuilder:
    @staticmethod
    def ConstructLoss_fn(loss_fn_name: str) -> nn.Module:
        if loss_fn_name == "L1Loss":
            return std.L1LossWeighted(False)
        elif loss_fn_name == "L2Loss":
            # Plain L2 (MSE) on the normalised targets. In a tonemapped space (asinh/log) it spreads
            # gradient across the dynamic range; pairs with asinh in the loss-effectiveness study.
            return std.PlainL2Loss(weight_with_error=False)
        elif loss_fn_name == "LogRatioBalanced":
            # Region-balanced LOG-RATIO L1 (SMAPEBalanced's regions/masks/hinge with a non-saturating
            # relative core): fixes SMAPE's over-prediction saturation, which left misplaced/rotated
            # ghost beams nearly gradient-free (ep153 evidence: bright-IoU 0.29, 63% ghost mass).
            return std.SMAPERegionBalancedLoss(core="logratio")
        elif loss_fn_name == "SMAPEBalanced":
            # Metric-targeted region-balanced SMAPE (pipeline-audit P0): trains the metric's own
            # functional with equal gradient mass on beam / bright-ring / reliable-bulk regions and
            # MC-noise voxels masked out. Pair with normalizer="linear0_1" (SMAPE is scale-invariant).
            return std.SMAPERegionBalancedLoss()
        elif loss_fn_name == "RawNeRF":
            # RawNeRF HDR loss (Mildenhall CVPR 2022): linear-space L2 / (sg(pred)+eps)^2 — the
            # relative-weighted linear recipe between the linear+abs and log+abs corners.
            return std.RawNeRFLoss()
        elif loss_fn_name == "RawNeRFSharp":
            # RawNeRF + alpha*L1 beam-sharpening hybrid (reconstruction-eval finding: RawNeRF's
            # self-weight anneals ~p^-2 -> blurred beam; the L1 term restores the non-annealing
            # coherent beam gradient of the published sharp-beam recipe).
            return std.RawNeRFSharpLoss(alpha=10.0)
        elif loss_fn_name == "MuLawL2":
            # mu-law tone-mapped L2 (Kalantari & Ramamoorthi, SIGGRAPH 2017) — the standard HDR-
            # reconstruction loss: one smooth branch-free formula, non-saturating ghost suppression.
            return std.MuLawL2Loss(mu=5000.0)
        elif loss_fn_name == "L1MagWeighted":
            # Log-space L1 weighted by physical flux magnitude (air-kerma-aligned).
            # Fixes the HDR imbalance where the near-zero background drowns out the
            # high-flux beam and the peak collapses. Pairs with normalizer="log_scale".
            return std.MagnitudeWeightedL1Loss(c=0.01, gamma=1.0)
        elif loss_fn_name == "L1Plain":
            # Plain physical-space L1 for LinearNormalizer(0,1) targets — the
            # published config that reached ~84% scatter air-kerma accuracy.
            return std.PlainL1Loss(weight_with_error=False)
        elif loss_fn_name == "L1Physical":
            # L1 measured in PHYSICAL flux space (10**target) for log-space targets.
            # Air-kerma is ∝ physical flux, so additive peak error is what the
            # accuracy metric rewards — this recovers the ~84% scatter accuracy the
            # old LinearNormalizer reached, which log-space (multiplicative) L1 lost.
            # Pairs with normalizer="log_scale".
            return std.PhysicalSpaceL1Loss(beta=0.1)
        elif loss_fn_name == "WassersteinLoss":
            return std.WassersteinLossWeighted(dim=1, weight_with_error=False)
        elif loss_fn_name == "HistogramLoss":
            return comb_loss.HistogramLoss(bin_dim=1, weight_with_error=False, penalize_out_of_range=False, calc_moments=False)
        elif loss_fn_name == "SpectrumWasserstein":
            # PURE Earth-Mover (no L1 term) spectrum loss. Routed through HistogramLoss so it inherits
            # the -inf/ROI-mask-safe bin-permute path (the bare WassersteinLossWeighted scrambles
            # histograms under masking). = HistogramLoss with the W:L1 split set to (1.0, 0.0).
            return comb_loss.HistogramLoss(bin_dim=1, weight_with_error=False, penalize_out_of_range=False,
                                           calc_moments=False, ws_weight=1.0, l1_weight=0.0)
        elif loss_fn_name == "StructuralSimilarity3DLoss":
            return std.StructuralSimilarity3DLoss(weight_with_error=False)
        elif loss_fn_name == "L1ChannelBalanced":
            # Two-head: plain physical-space L1 whose gradient is per-channel-max-balanced (as if
            # individually normalized) WITHOUT normalizing the data — so the model predicts raw and
            # the scatter:direct relation is preserved. Pair with normalizer="linear_joint" + raw
            # split. See ChannelMaxBalancedLoss / tests/test_split_loss_weighting.py.
            return std.ChannelMaxBalancedLoss(std.PlainL1Loss(weight_with_error=False))
        elif loss_fn_name == "FluxChannelBalanced":
            return std.ChannelMaxBalancedLoss(comb_loss.FluxLoss(weight_with_error=False, log_scale=False))
        elif loss_fn_name == "FluxLoss":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False)
        elif loss_fn_name == "FluxLossNoSSIM":
            # Ablation A1: the published FluxLoss core (Huber = 0.5*(L1+L2)) with the SSIM3D
            # structural term removed (ssim_weight=0) — isolates the only spatial/sharpness term.
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False, ssim_weight=0.0)
        elif loss_fn_name == "FluxLossRelative":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False, relative_weighting=True)
        elif loss_fn_name == "FluxLossFocalR":
            # Huber core × Focal-R modulator. Use for absolute-error work
            # where the 99% near-zero background otherwise dominates the loss.
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False, focal_r=True)
        elif loss_fn_name == "HotspotAwareFluxLoss":
            return std.HotspotAwareFluxLoss()
        elif loss_fn_name == "TwoROIGammaLoss":
            # ROI thresholds MATCH the air-kerma scatter metric + the ROI sampler
            # (radfield3dnn.roi): beam = >=0.05*max, scatter floor = 5e-5*max (DS03 sweep,
            # 2026-06-12). Per-ROI means equalise beam/scatter/floor influence by target voxel count.
            from radfield3dnn.roi import BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
            return std.TwoROIGammaLoss(beam_rel=BEAM_REL_DEFAULT, scatter_lo=SCATTER_LO_DEFAULT)
        elif loss_fn_name == "FluxLossRelativeFocalR":
            # Relative-error core × Focal-R modulator. Use when relative
            # accuracy in [0,1] codomain matters AND the dataset is imbalanced
            # (e.g. dosimetry on a phantom with a sparse beam core).
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=False,
                                         relative_weighting=True, focal_r=True)
        elif loss_fn_name == "FluxLossHDR":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=True, ssim_weight=0.0)
        elif loss_fn_name == "FluxLogLoss":
            return comb_loss.FluxLoss(weight_with_error=False, log_scale=True)
        elif loss_fn_name == "L1WithSSIM3DLoss":
            # FluxLoss recipe with the L2 component removed. fp16-safe at
            # raw log-space outputs (LogScaleNormalizer in [-9, 0]); SSIM3D
            # breaks the median-collapse plateau that plain L1 suffers on
            # sparse-target fields. Recommended pairing with the
            # log_scale stack — see handoff §6 S-17.
            return comb_loss.L1WithSSIM3DLoss(weight_with_error=False)
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
        elif encoding_fn_name == "SphericalHarmonics":
            return SphericalHarmonics
        else:
            raise ValueError(f"Invalid encoding function name: {encoding_fn_name}")


class BaseNeuralRadFieldModel(pl.LightningModule):
    __model_name__: str = None

    lr = property(lambda x: x.get_lr(), lambda x, v: x.set_lr(v))
    learning_rate = property(lambda x: x.get_lr(), lambda x, v: x.set_lr(v))

    def __init__(self, normalizer: Union[Normalizer, str] = LinearNormalizer(), learning_rate: float=1e-3):
        super().__init__()
        self.logging_prefix = ""

        self._lr = learning_rate
        self._flux_loss_fn: Loss = comb_loss.FluxLoss(weight_with_error=False)
        self._spectrum_loss_fn: Loss = comb_loss.HistogramLoss(bin_dim=1) # comb_loss.HistogramLoss(bin_dim=1, weight_with_error=False, penalize_out_of_range=False) # std.KLDivLossWeighted(True)
        # Learnable log-variance weights for the multi-task (flux, spectrum)
        # objective — Kendall & Gal 2018 ("Multi-Task Learning Using Uncertainty
        # to Weigh Losses for Scene Geometry and Semantics", arXiv:1705.07115).
        # Effective loss is exp(-s) * L + s per task; the +s prevents both
        # weights from collapsing to zero. Initialised at 0 → equal weights
        # (the historic 0.5/0.5 split) on the first step, then re-balanced
        # automatically by the optimizer. Logged to wandb under
        # `loss_logvar_flux` / `loss_logvar_spectrum` so the rebalancing is
        # observable across training. Not in save_hyperparameters since
        # nn.Parameters are serialised by state_dict anyway.
        # Multi-task balancing is handled by DB-MTL (Lin et al. 2023,
        # "Dual-Balancing for Multi-Task Learning", arXiv:2308.12029),
        # encapsulated in `MultiTaskLossBalancer`. It is non-parametric (no
        # learnable loss weights), so the only model state needed is a counter
        # of non-finite-loss events for observability (findings.md §3.7).
        # DB-MTL replaces the previous Kendall uncertainty weighting, which
        # converged to weight ∝ 1/loss and starved the high-magnitude flux head
        # (findings.md §3.2/§4).
        self._mtl = MultiTaskLossBalancer()
        self.register_buffer("_nonfinite_loss_count", torch.zeros(1))
        # Generic seam for subclass-specific auxiliary task losses. Subclasses
        # populate this dict (name -> per-sample loss tensor) from their own
        # forward/evaluate overrides; process_metrics folds it into the DB-MTL
        # combine and clears it. base.py stays agnostic to what's in it.
        self._extra_task_losses: dict[str, Tensor] = {}
        self._normalizer = normalizer if not isinstance(normalizer, str) else NormalizerConstructor.construct_by_name(normalizer)
        self.max_inner_batch_size = None
        self.indices: Tensor = None
        self.grid_dims: Tensor = None
        self.batch_size = 1
        self._channels_join = ChannelsJoin()
        assert isinstance(self._normalizer, Normalizer), f"normalizer must be an instance of Normalizer, got {type(self._normalizer)}"
        self.save_hyperparameters(ignore=["indices", "grid_dims", "_flux_loss_fn", "_spectrum_loss_fn", "normalizer", "_channels_join"])

    def get_core_model(self) -> nn.Module:
        raise NotImplementedError("Please implement get_core_model()")

    def _generate_random_ground_truth(self, device) -> RadiationField:
        input = self._generate_random_input(device=device)
        return self._normalizer.inverse(self.forward(input))
    
    def get_submodels(self) -> list["BaseNeuralRadFieldModel"]:
        return []

    def _generate_random_input(self, device, batch_size=2) -> PositionalInput:
        return PositionalInput(
            direction=torch.rand(batch_size, 3, device=device),
            spectrum=HistogramNormalize(dim=-1)(torch.rand(batch_size, 150, device=device)),
            position=torch.rand(batch_size, 3, device=device),
            # origin is the per-sample source distance (a scalar) — [B, 1].
            # PBRFNet(CPP)'s beam encoder asserts origin.shape[-1] == 1; the
            # other models are shape-agnostic about it.
            origin=torch.rand(batch_size, 1, device=device),
            geometry=None,
            beam_shape_parameters=torch.rand(batch_size, 1, device=device),
            beam_shape_type=torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
        )

    def _search_optimal_batch_size(self):
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
        random_flux1 = torch.abs(torch.rand((num_losses, 1), device=device))
        random_flux1 /= random_flux1.max()
        if y_base.scatter_field.spectrum is not None:
            random_spectra2 = torch.rand((num_losses, y_base.scatter_field.spectrum.shape[0]), device=device)
            random_spectra2 = random_spectra2 / random_spectra2.sum(dim=1, keepdim=True)
        else:
            random_spectra2 = None
        random_flux2 = torch.abs(torch.rand((num_losses, 1), device=device))
        random_flux2 /= random_flux1.max()

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
                direct_channel = RadiationFieldChannel(
                    spectrum=y_base.direct_beam.spectrum,
                    flux=y_base.direct_beam.flux,
                    error=y_base.direct_beam.error
                ) if y_base.direct_beam is not None else scatter_channel
                y = RadiationField(
                    scatter_field=scatter_channel,
                    direct_beam=direct_channel,
                    geometry=torch.zeros_like(y_base.scatter_field.flux) if y_base.geometry is not None else None
                )
                batch_size = self.max_inner_batch_size * 2
                y_scatter_flu = y.scatter_field.flux
                y_scatter_spec = y.scatter_field.spectrum
                y_direct_flu = y.direct_beam.flux
                y_direct_spec = y.direct_beam.spectrum
                if y.scatter_field.spectrum is not None:
                    y_scatter_spec = y.scatter_field.spectrum.unsqueeze(0)
                    y_direct_spec = y.direct_beam.spectrum.unsqueeze(0)

                y_scatter_flu = y.scatter_field.flux.unsqueeze(0)
                y_direct_flu = y.direct_beam.flux.unsqueeze(0)

                direct_err = torch.rand_like(y_direct_flu).expand(batch_size, *([-1] * (len(y_direct_flu.shape) - 1)))
                if len(direct_err.shape) == 1:
                    direct_err = direct_err.unsqueeze(1)
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
                        spectrum=y_direct_spec.expand(batch_size, *([-1] * (len(y_direct_spec.shape) - 1))) if y_direct_spec is not None else None,
                        flux=y_direct_flu.expand(batch_size, *([-1] * (len(y_direct_flu.shape) - 1))),
                        error=direct_err
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

        # The drop mask (voxels the importance sampler removed, marked with the
        # -inf sentinel) is a TRAINING-only optimisation: it skips computing the
        # dropped voxels so the loss ignores them. During eval / inference we must
        # predict the FULL volume — otherwise validation gets a sparse (and, when
        # a field is fully dropped, EMPTY) prediction that breaks the metrics.
        drop_mask = None
        if self.training:
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

    # ── channel-eval dispatch ────────────────────────────────────────────────────────────────
    # How many channels the model PREDICTS (1 = single/joined, 2 = scatter+direct, AirKermaField)
    # is a fixed property of the architecture, so it is determined ONCE by a tiny test run of
    # forward and the matching plain eval method is bound for the lifetime of the instance —
    # replacing the per-step isinstance/None-check maze the old calculate_metrics carried.
    _calc_metrics_impl = None   # bound on first use (test-run probe or first real prediction)

    def _probe_prediction_channels(self, batch: TrainingInputData, is_complete_volume: bool):
        """TEST RUN of the model's forward — a tiny no_grad pass on the REAL batch inputs
        (a 2x2x2 volume for volume-trained models, the first 2 rows for pointwise ones) whose
        only purpose is to observe what the model emits (one channel, two channels, or an
        AirKermaField). Returns that probe prediction."""
        with torch.no_grad():
            if is_complete_volume:
                bins = self.out_spectra_dim if hasattr(self, "out_spectra_dim") else 32
                dims = torch.tensor([2, 2, 2], dtype=torch.int32, device=batch.input.direction.device)
                return self.forward2volume(batch.input, dims, spectra_bins=bins)
            x = batch.input
            x2 = type(x)(*[(v[:2] if isinstance(v, Tensor) else v) for v in x])
            return self(x2)

    def _select_metrics_impl(self, pred_field, y):
        """Pick the plain eval method matching (predicted channels, GT channels)."""
        if isinstance(pred_field, AirKermaField):
            return self._calculate_metrics_airkerma
        if not isinstance(pred_field, RadiationField):
            raise ValueError("pred_field must be of type RadiationField or AirKermaField.")
        pred_two = pred_field.scatter_field is not None and pred_field.direct_beam is not None
        gt_two = isinstance(y, RadiationField) and y.direct_beam is not None
        if pred_two and gt_two:
            return self._calculate_metrics_two_channel
        if pred_two:                       # two-head model, single/joined GT -> join the prediction
            return self._calculate_metrics_join_pred
        if gt_two:                         # single-head model, split GT -> join the GT
            return self._calculate_metrics_join_gt
        return self._calculate_metrics_single_channel

    def _join_field(self, field):
        """Physically join a split RadiationField (inverse -> sum channels -> re-normalise)."""
        field = self._normalizer.inverse(field)
        field = self._channels_join(field)
        return self._normalizer.forward(field)

    def _calculate_metrics_single_channel(self, pred_field: RadiationField, y, batch: TrainingInputData) -> TrainingMetrics:
        """Plain single-channel eval: one predicted channel vs a single-channel GT."""
        scatter_field_gt = y.scatter_field if isinstance(y, RadiationField) else y

        loss_spec = None
        if pred_field.scatter_field.spectrum is not None:
            loss_spec = self._spectrum_loss_fn.forward(prediction=pred_field.scatter_field.spectrum, target=scatter_field_gt.spectrum, input=batch)
            if not torch.isfinite(loss_spec).all():
                print(f"[red] Spectrum loss is not finite — letting downstream fallback set total to 1.")

        loss_flux = self._flux_loss_fn.forward(prediction=pred_field.scatter_field.flux, target=scatter_field_gt.flux, input=batch)
        if not torch.isfinite(loss_flux).all():
            print(f"[red] Flux loss is not finite — letting downstream fallback set total to 1.")

        # Debug-probe capture seam (callbacks/debug_probe.py): stash detached references to
        # exactly what went through the loss this step. Enabled by the TrainingDebugProbe
        # callback (YAML `training: debug_probe: true`); zero cost when disabled.
        if getattr(self, "_debug_probe_enabled", False):
            self._debug_capture = {
                "pred_flux": pred_field.scatter_field.flux.detach(),
                "target_flux": scatter_field_gt.flux.detach(),
                "pred_spectrum": pred_field.scatter_field.spectrum.detach() if pred_field.scatter_field.spectrum is not None else None,
                "flux_loss": float(loss_flux.detach().mean()),
                "spectrum_loss": float(loss_spec.detach().mean()) if loss_spec is not None else None,
                "flux_loss_terms": dict(getattr(self._flux_loss_fn, "last_terms", {})),
                "batch_input": batch.input,
            }

        return TrainingMetrics(
            scatter_field=ChannelMetrics(flux_loss=loss_flux, spectrum_loss=loss_spec),
            direct_beam=None,
        )

    def _calculate_metrics_join_gt(self, pred_field: RadiationField, y: RadiationField, batch: TrainingInputData) -> TrainingMetrics:
        """Single-channel prediction vs a SPLIT GT: physically join the GT (and the batch the
        loss receives as `input`), then evaluate as plain single-channel."""
        batch = self._normalizer.forward(self._channels_join(self._normalizer.inverse(batch)))
        y = self._join_field(y)
        return self._calculate_metrics_single_channel(pred_field, y, batch)

    def _calculate_metrics_join_pred(self, pred_field: RadiationField, y, batch: TrainingInputData) -> TrainingMetrics:
        """Two-head prediction vs a single-channel GT: physically join the PREDICTION
        (inverse -> scatter+direct -> re-normalise), then evaluate as plain single-channel."""
        inv = self._normalizer.inverse(pred_field)
        joined = RadiationField(
            scatter_field=RadiationFieldChannel(
                flux=inv.scatter_field.flux + inv.direct_beam.flux,
                spectrum=inv.scatter_field.spectrum,
                error=inv.scatter_field.error,
            ),
            direct_beam=None,
        )
        return self._calculate_metrics_single_channel(self._normalizer.forward(joined), y, batch)

    def _calculate_metrics_two_channel(self, pred_field: RadiationField, y: RadiationField, batch: TrainingInputData) -> TrainingMetrics:
        """Plain two-channel eval: scatter (flux+spectrum) and direct (flux) scored separately.

        NOTE: a manual `loss_flux *= direct_flux.sum() / scatter_flux.sum()` rescaling used to
        live here. It assumed a log-space normaliser where the per-channel flux sums are large
        and same-signed; under `asinh_split` the scatter sum crosses zero, so the ratio blew up
        to inf/NaN and poisoned the DB-MTL surrogate. The cross-task flux balancing it tried to
        do is exactly what `MultiTaskLossBalancer` now does (scale-invariantly, in log space).
        """
        single = self._calculate_metrics_single_channel(pred_field, y, batch)
        loss_direct = self._flux_loss_fn.forward(prediction=pred_field.direct_beam.flux, target=y.direct_beam.flux, input=batch)
        return TrainingMetrics(
            scatter_field=single.scatter_field,
            direct_beam=ChannelMetrics(flux_loss=loss_direct, spectrum_loss=None),
        )

    def _calculate_metrics_airkerma(self, pred_field: AirKermaField, y, batch: TrainingInputData) -> TrainingMetrics:
        assert isinstance(y, AirKermaField), "Ground truth must be of type AirKermaField when predicting AirKermaField."
        loss_airkerma = self._flux_loss_fn.forward(prediction=pred_field.air_kerma, target=y.air_kerma, input=batch)
        return TrainingMetrics(airkerma_field=loss_airkerma)

    def evaluate_forward(self, batch: TrainingInputData) -> TrainingMetrics:
        batch = self._normalizer.forward(batch)
        y = batch.ground_truth
        x = batch.input
        self.batch_size = int(x.direction.shape[0])
        if isinstance(y, (RadiationField, RadiationFieldChannel)):
            scatter_field_gt = y.scatter_field if isinstance(y, RadiationField) else y
            is_complete_volume = (len(scatter_field_gt.flux.shape) == 4 or (len(scatter_field_gt.flux.shape) == 5 and scatter_field_gt.flux.shape[1] == 1)) and len(scatter_field_gt.spectrum.shape) == 5
        elif isinstance(y, AirKermaField):
            scatter_field_gt = batch.original_ground_truth.scatter_field if batch.original_ground_truth is not None and isinstance(batch.original_ground_truth, RadiationField) else None
            is_complete_volume = (len(y.air_kerma.shape) == 4 or (len(y.air_kerma.shape) == 5 and y.air_kerma.shape[1] == 1))
        else:
            raise ValueError("Ground truth must be of type RadiationField, RadiationFieldChannel, or AirKermaField.")

        # One-time test run of forward: count the predicted channels and bind the matching plain
        # eval method for the rest of the instance's lifetime.
        if self._calc_metrics_impl is None:
            probe = self._probe_prediction_channels(batch, is_complete_volume)
            self._calc_metrics_impl = self._select_metrics_impl(probe, y)

        if is_complete_volume:
            pred_field: RadiationField | AirKermaField = self.forward2volume_from_training_input(batch, spectra_bins=scatter_field_gt.spectrum.shape[1] if scatter_field_gt.spectrum is not None else 32)
        else:
            pred_field: RadiationField | AirKermaField = self(x)

        return self._calc_metrics_impl(pred_field, y, batch)

    def calculate_metrics(self, pred_field: RadiationField | AirKermaField, y: RadiationField | AirKermaField, batch: TrainingInputData) -> TrainingMetrics:
        """Public seam (subclasses with their own evaluate_forward call this with their
        prediction). Binds the matching plain eval method on first use — here the first real
        prediction IS the test run — then dispatches straight to it."""
        if self._calc_metrics_impl is None:
            self._calc_metrics_impl = self._select_metrics_impl(pred_field, y)
        return self._calc_metrics_impl(pred_field, y, batch)
    
    @property
    def output_head_markers(self) -> tuple[str, ...]:
        """Substrings of parameter names belonging to task-specific output heads.

        `_shared_parameters` excludes these from DB-MTL's gradient-magnitude
        balancing. The base model declares no heads — subclasses that add output
        heads override this property, so base.py stays decoupled from any
        specific head layout.
        """
        return ()

    @property
    def use_lr_finder(self) -> bool:
        """Whether Lightning's LR finder should run before training.

        Default True. Fused fp16 models (PBRFNetCPP) override to False — the
        finder's high-LR sweep overflows their fp16 fused weights to NaN, and
        their optimizer clamps the LR to ``max_lr`` regardless, so the sweep is
        both harmful and pointless.

        Can also be disabled per-run via the instance attribute
        ``_use_lr_finder`` (set from the YAML ``training: lr_finder: false``) —
        the finder picks a different LR per seed, a major source of run-to-run
        variance, so a fixed configured LR is preferred for reproducible runs.
        """
        return getattr(self, "_use_lr_finder", True)

    @property
    def mtl_gradient_balancing(self) -> bool:
        """Whether DB-MTL's gradient-magnitude balancing (step 2) is applied.

        It requires per-task gradients w.r.t. the shared trunk, obtained with
        one ``autograd.grad`` backward per task. Disable for **fused / black-box
        models** (e.g. the tcnn-fused PBRFNetCPP) where: (a) there is no
        Python-visible shared representation between trunk and heads to take a
        cheap head-only Jacobian, and (b) repeated full-network backward through
        the fused C++ autograd Function is costly and may not be re-entrant.
        Those models fall back to **loss-scale balancing only** (step 1), which
        is still scale-invariant and needs a single backward.

        The per-task `autograd.grad(loss, shared_trunk)` is a FULL-network backward
        EACH (N_tasks extra backwards/step). That cost is now **amortised by caching**:
        the balancer recomputes the weights only every `update_every` steps and reuses
        the cache in between (see `MultiTaskLossBalancer`), so DB-MTL stays ON by default
        cheaply. Set `model._mtl_gradient_balancing = False` for a single-backward
        fallback (fused/black-box models override this to False).
        """
        return getattr(self, "_mtl_gradient_balancing", True)

    def _shared_parameters(self) -> list[nn.Parameter]:
        """Trunk/encoder parameters shared across the task heads.

        Used by DB-MTL's gradient-magnitude balancing: excludes the per-task
        output heads (see `output_head_markers`) so the balancing reflects how
        each task pulls on the *shared* representation.
        """
        # Cache the list: the parameter set is fixed after construction, so rebuild it
        # once instead of iterating named_parameters() (+ string-matching) every step.
        cached = getattr(self, "_shared_params_cache", None)
        if cached is not None:
            return cached
        markers = self.output_head_markers
        cached = [p for n, p in self.named_parameters()
                  if p.requires_grad and not any(m in n for m in markers)]
        self._shared_params_cache = cached
        return cached

    def _shared_parameter_limits(self, shared_params: list[nn.Parameter]) -> list[int | None]:
        """Per-parameter element limit aligned with ``shared_params`` for DB-MTL.

        Default ``None`` for every parameter (the whole gradient enters the norm).
        Fused models (PBRFNetCPP) override this to return the TRUNK element count
        for their single fused-weights tensor, so the per-task norm excludes the
        output heads living in the same blob (heads can't be name-excluded via
        ``output_head_markers`` the way pure-Python models do).
        """
        return [None] * len(shared_params)

    def _dbmtl_combine(self, task_losses: dict[str, Tensor], stage: str, on_step: bool, on_epoch: bool) -> Tensor:
        """Combine task losses with the DB-MTL balancer and log its weights.

        Delegates the balancing to `self._mtl` (`MultiTaskLossBalancer`).
        Gradient-magnitude balancing (step 2) needs per-task gradients w.r.t. the
        shared trunk — supplied only when the model can expose one
        (`mtl_gradient_balancing`); fused/black-box models (PBRFNetCPP) pass
        ``None`` and fall back to loss-scale balancing (step 1). The per-task
        weights ``α_i`` and gradient norms are logged (``{stage}_dbmtl_*``) so the
        balancing can be validated in wandb.
        """
        use_grad_balancing = self.mtl_gradient_balancing and self.training
        balance_params = self._shared_parameters() if use_grad_balancing else None
        balance_limits = self._shared_parameter_limits(balance_params) if use_grad_balancing else None
        total = self._mtl.combine(task_losses, balance_params, balance_limits)

        for n, w in self._mtl.last_weights.items():
            self.log(f'{stage}_dbmtl_weight_{n}', w, on_step=on_step, on_epoch=on_epoch, logger=True, batch_size=self.batch_size)
        for n, gn in self._mtl.last_gradnorms.items():
            self.log(f'{stage}_dbmtl_gradnorm_{n}', gn, on_step=on_step, on_epoch=on_epoch, logger=True, batch_size=self.batch_size)
        # Surface the B-hardening guard: 1.0 whenever a task's trunk-gradient norm
        # overflowed to inf/NaN this step (the guard fired). A non-zero value here
        # means a real fp16 overflow was masked — investigate, don't ignore.
        for n, of in getattr(self._mtl, "last_overflowed", {}).items():
            self.log(f'{stage}_dbmtl_overflow_{n}', float(of), on_step=on_step, on_epoch=on_epoch, logger=True, batch_size=self.batch_size)
        return total

    def process_metrics(self, metrics: TrainingMetrics, stage: str) -> Tensor:
        on_epoch = stage in ["val", "test"]

        if len(self.logging_prefix) > 0:
            stage = stage + "." + self.logging_prefix

        on_step = not on_epoch

        # Gather every active task loss into one dict; DB-MTL balances them
        # jointly (a single combine call instead of the old per-channel Kendall).
        task_losses: dict[str, Tensor] = {}
        if metrics.scatter_field is not None:
            task_losses["scatter_flux"] = metrics.scatter_field.flux_loss
            self.log(f'{stage}_scatter_flux_loss', metrics.scatter_field.flux_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
            if metrics.scatter_field.spectrum_loss is not None:
                # Fixed spectrum-task weight (YAML `training: spectrum_loss_weight`). The DB-MTL
                # experiment showed ADAPTIVE balancing is incompatible with self-weighted flux
                # losses (their magnitude IS their imbalance correction); a FIXED multiplier is the
                # safe way to lift the undertrained spectrum head (flux/spectrum scale gap ~40-1000x),
                # which leaks into every air-kerma metric via the mu_tr-weighted integral.
                spec_w = float(getattr(self, "_spectrum_loss_weight", 1.0))
                task_losses["scatter_spectrum"] = metrics.scatter_field.spectrum_loss * spec_w
                self.log(f'{stage}_scatter_spectrum_loss', metrics.scatter_field.spectrum_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        if metrics.direct_beam is not None:
            task_losses["direct_flux"] = metrics.direct_beam.flux_loss
            self.log(f'{stage}_direct_beam_flux_loss', metrics.direct_beam.flux_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        if metrics.airkerma_field is not None:
            task_losses["airkerma"] = metrics.airkerma_field
            self.log(f'{stage}_airkerma_loss', metrics.airkerma_field.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        # Subclass-supplied auxiliary task losses (generic seam; base.py does not
        # know what they are — e.g. PBRFNet's two-head max-ratio). Folded into
        # the DB-MTL combine and then cleared.
        for name, loss in self._extra_task_losses.items():
            task_losses[name] = loss
            self.log(f'{stage}_{name}_loss', loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
        self._extra_task_losses = {}

        if not task_losses:
            return torch.zeros((), device=self.device, requires_grad=True)

        # 3.7: monitor on the raw (positive, scale-meaningful) sum of task
        # losses — NOT the DB-MTL surrogate. Used for checkpoint/early-stop.
        raw_total = torch.stack([l.mean() for l in task_losses.values()]).sum()
        self.log(f'{stage}_raw_loss', raw_total.detach(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)

        # MTL ABLATION: with `_use_mtl=False` (YAML `training: mtl_balancing: false`) the task
        # losses are combined by a plain EQUAL-WEIGHT sum — no DB-MTL loss-scale or
        # gradient-magnitude balancing — so the effectiveness of the balancing can be scored
        # against the default. The equal-weight gradient carrier IS the raw sum.
        if not getattr(self, "_use_mtl", True):
            self.log(f'{stage}_loss', raw_total.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
            return raw_total.mean()

        # DB-MTL surrogate: lives in log space, so its *value* can be negative.
        # It is only meaningful as a gradient carrier.
        surrogate = self._dbmtl_combine(task_losses, stage, on_step=on_step, on_epoch=on_epoch)

        # Straight-through estimator: the returned scalar carries the DB-MTL
        # surrogate's *gradient* but takes the *value* of the positive raw loss.
        # This is essential — a negative log-space loss breaks Lightning's LR
        # finder (its divergence test `loss > k * best_loss` misfires on negative
        # losses), gradient/loss monitoring and checkpoint selection.
        if torch.isfinite(surrogate).all():
            total_loss = surrogate - surrogate.detach() + raw_total.detach()
        else:
            # 3.7: keep training alive on a non-finite surrogate, but make it
            # observable (a silent swap previously masked real divergence).
            self._nonfinite_loss_count += 1
            print(f"[red] Non-finite loss (event #{int(self._nonfinite_loss_count.item())}) — falling back to the raw loss to continue.")
            self.log(f'{stage}_nonfinite_loss_count', self._nonfinite_loss_count, on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
            total_loss = raw_total if torch.isfinite(raw_total).all() else torch.tensor(1.0, device=self.device, requires_grad=True)

        self.log(f'{stage}_loss', total_loss.mean(), on_epoch=on_epoch, on_step=on_step, logger=True, batch_size=self.batch_size)
        return total_loss.mean()

    def training_step(self, batch: TrainingInputData, batch_idx):
        metrics = self.evaluate_forward(batch)
        loss = self.process_metrics(metrics, "train")
        # fp16 loss scaling: scale so the small HDR-flux-background gradients survive fp16's ~6e-8
        # underflow floor through the deep backbone backward (no Trainer GradScaler on the self.half()
        # path). The scale is undone when grads are transferred to the fp32 masters in
        # `on_after_backward`, so the optimiser sees true-magnitude gradients. No-op in fp32.
        if getattr(self, "_fp32_masters", None) is not None:
            loss = loss * self._loss_scale
        return loss

    # ── fp16 fp32-master-weight runtime hooks ─────────────────────────────────
    # The masters + loss scale are *created* by the optimizer behaviour
    # (radfield3dnn/optim) at configure_optimizers time; these hooks drive them each step. All
    # are no-ops when `_fp32_masters` is absent (fp32 training).
    @torch.no_grad()
    def _sync_masters_to_fp16(self):
        masters = getattr(self, "_fp32_masters", None)
        if not masters:
            return
        pdict = dict(self.named_parameters())
        for name, master in masters.items():
            pdict[name].copy_(master.to(pdict[name].dtype))

    def on_train_batch_start(self, batch, batch_idx):
        # Make the fp16 forward read the freshly Adam-updated fp32 masters.
        self._sync_masters_to_fp16()

    def on_after_backward(self):
        # Transfer fp16 weight grads → fp32 master grads (cast to fp32, accumulate across micro-
        # batches), undoing the loss scaling, then clear the fp16 grads so the optimiser (bound to the
        # masters) doesn't orphan them. Fires before gradient clipping / the optimiser step.
        masters = getattr(self, "_fp32_masters", None)
        if not masters:
            return
        inv_scale = 1.0 / float(getattr(self, "_loss_scale", 1.0))
        pdict = dict(self.named_parameters())
        for name, master in masters.items():
            g = pdict[name].grad
            if g is None:
                continue
            gf = g.detach().to(torch.float32) * inv_scale
            master.grad = gf.clone() if master.grad is None else master.grad.add_(gf)
            pdict[name].grad = None

    def validation_step(self, batch: TrainingInputData, batch_idx):
        with torch.no_grad():
            metrics = self.evaluate_forward(batch)
            return self.process_metrics(metrics, "val")
    
    def test_step(self, batch: TrainingInputData, batch_idx):
        with torch.no_grad():
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

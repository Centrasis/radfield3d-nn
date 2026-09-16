from radfield3dnn.rftypes import TrainingInputData, RadiationFieldChannel, RadiationField, AirKermaField
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from radfield3dnn.datasets.channel_join import ChannelsJoin
import torch


class ValidateGroundTruth(Callback):
    channels_join = ChannelsJoin()

    MAX_ENERGY_EV = 1.5e+5
    BIN_WIDTH_EV = MAX_ENERGY_EV / 32

    def on_train_batch_start(self, trainer: Trainer, pl_module: BaseNeuralRadFieldModel, batch: TrainingInputData, batch_idx: int) -> None:
            ValidateGroundTruth.channels_join = ValidateGroundTruth.channels_join.to(pl_module.device)
            gt: RadiationFieldChannel | AirKermaField = ValidateGroundTruth.channels_join.forward(batch.ground_truth)

            def _drop_singleton_channel(t: torch.Tensor, ref_ndim: int) -> torch.Tensor:
                # Align a (B,1,D,H,W) value map to the (B,D,H,W) shape of the bin-summed spectrum.
                if t.ndim == ref_ndim + 1 and t.shape[1] == 1:
                    return t.squeeze(1)
                return t

            if isinstance(gt, RadiationFieldChannel):
                spectrum = gt.spectrum
                spectrum_sums = spectrum.sum(dim=1)                       # drop the energy-bin axis
                zeros_sum_spectra = (spectrum_sums < 1e-6)
                zero_counts_map = _drop_singleton_channel(gt.flux, spectrum_sums.ndim) < 1e-6
                assert zeros_sum_spectra.shape == zero_counts_map.shape, \
                    f"GT validation: spectra-sum {tuple(zeros_sum_spectra.shape)} vs flux {tuple(zero_counts_map.shape)} shape mismatch."
                assert torch.all(zeros_sum_spectra == zero_counts_map), \
                    "Ground truth validation failed: mismatch between zero spectra sum and zero flux places!"
                # .all() over the boolean tensor — `assert <multi-element tensor>` would raise the
                # ambiguous-truth RuntimeError instead of a clean AssertionError.
                nonzero = spectrum_sums[~zeros_sum_spectra]
                assert bool(((nonzero - 1.0).abs() <= 1e-6).all()), \
                    "Ground truth validation failed: non-normalized spectra found where flux is non-zero!"
            elif isinstance(gt, AirKermaField) and isinstance(batch.original_ground_truth, (RadiationField, RadiationFieldChannel)):
                spectrum = ValidateGroundTruth.channels_join.forward(batch.original_ground_truth).spectrum
                spectrum_sums = spectrum.sum(dim=1)
                zero_counts_map = spectrum_sums < 1e-6
                zero_airkerma_map = _drop_singleton_channel(gt.air_kerma, spectrum_sums.ndim) < 1e-6
                assert zero_counts_map.shape == zero_airkerma_map.shape, \
                    f"GT validation: spectra-sum {tuple(zero_counts_map.shape)} vs airkerma {tuple(zero_airkerma_map.shape)} shape mismatch."
                assert torch.all(zero_counts_map == zero_airkerma_map), \
                    "Ground truth validation failed: mismatch between zero spectra sum and zero airkerma places!"

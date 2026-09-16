from radfield3dnn import TrainingInputData, RadiationFieldChannel, RadiationField, AirKermaField
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

            if isinstance(gt, RadiationFieldChannel):
                spectrum = gt.spectrum
                spectrum_sums = spectrum.sum(dim=1)
                zero_counts_map = gt.flux < 1e-6
                zeros_sum_spectra = (spectrum_sums < 1e-6)
                same_places = (zeros_sum_spectra == zero_counts_map)
                assert torch.all(same_places), "Ground truth validation failed: mismatch between zero spectra sum and zero flux places!"
                assert torch.isclose(spectrum_sums[~zeros_sum_spectra], 1.0, atol=1e-6), "Ground truth validation failed: non-normalized spectra found where flux is non-zero!"
            elif isinstance(gt, AirKermaField) and isinstance(batch.original_ground_truth, (RadiationField, RadiationFieldChannel)):
                spectrum = ValidateGroundTruth.channels_join.forward(batch.original_ground_truth).spectrum
                spectrum_sums = spectrum.sum(dim=1)
                zero_counts_map = spectrum_sums < 1e-6
                zero_airkerma_map = gt.air_kerma < 1e-6
                same_places = (zero_counts_map == zero_airkerma_map)
                assert torch.all(same_places), "Ground truth validation failed: mismatch between zero spectra sum and zero airkerma places!"

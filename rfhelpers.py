

from rftypes import RadiationField, TrainingInputData, AirKermaField, RadiationFieldChannel
import torch
from models.base import BaseNeuralRadFieldModel
from normalizations.base import Normalizer
from normalizations.linear import LinearNormalizer
from datasets.channel_join import ChannelsJoin
import time


class InferenceHelper:
    channels_join = ChannelsJoin()
    linear_normalizer = LinearNormalizer((0, 1), always_normalize=True)

    @staticmethod
    def inference_step(batch: TrainingInputData, pl_module: BaseNeuralRadFieldModel, voxel_resolution: tuple[int, int, int], spectra_bins: int = 32) -> AirKermaField | RadiationFieldChannel:
        gt = batch.original_ground_truth if batch.original_ground_truth is not None else batch.ground_truth

        with torch.no_grad():
            batch = TrainingInputData(
                input=batch.input,
                ground_truth=pl_module._normalizer.forward(gt),
                original_ground_truth=batch.original_ground_truth
            )

            pred_field: RadiationField | AirKermaField | RadiationFieldChannel = pl_module.forward2volume_from_training_input(batch, voxel_resolution, spectra_bins=spectra_bins)
            if isinstance(pred_field, RadiationField):
                if pred_field.direct_beam is not None:
                    pred_field: RadiationFieldChannel = InferenceHelper.channels_join.forward(pred_field)
                    pred_field = pl_module._normalizer.forward(pred_field)  # Ensure prediction is normalized
                else:
                    pred_field = pred_field.scatter_field
            pred_field: RadiationFieldChannel | AirKermaField = pl_module._normalizer.inverse(pred_field)
            return pred_field

    @staticmethod
    def timed_inference_step(batch: TrainingInputData, pl_module: BaseNeuralRadFieldModel, voxel_resolution: tuple[int, int, int], spectra_bins: int = 32) -> tuple[AirKermaField | RadiationFieldChannel, float]:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        pred_field = InferenceHelper.inference_step(batch, pl_module, voxel_resolution, spectra_bins=spectra_bins)
        end_evt.record()
        end_evt.synchronize()
        duration_ms = start_evt.elapsed_time(end_evt)
        return pred_field, duration_ms

    @staticmethod
    def generate_gt_and_pred_for_validation(batch: TrainingInputData, pl_module: BaseNeuralRadFieldModel, voxel_resolution: tuple[int, int, int], spectra_bins: int = 32) -> tuple[AirKermaField | RadiationFieldChannel, AirKermaField | RadiationFieldChannel]:
        """
        Generate ground truth and prediction for validation purposes.
        Args:
            batch (TrainingInputData): The input batch containing ground truth data.
            pl_module (BaseNeuralRadFieldModel): The model used for prediction.
            voxel_resolution (tuple[int, int, int]): The desired voxel resolution for the output volume.
            spectra_bins (int, optional): Number of spectral bins. Defaults to 32.
        Returns:
            tuple[AirKermaField | RadiationFieldChannel, AirKermaField | RadiationFieldChannel]: A tuple containing the ground truth and predicted fields. Fields are normalized to [0,1] scale.
        """
        if InferenceHelper.linear_normalizer.device != pl_module.device:
            InferenceHelper.linear_normalizer = InferenceHelper.linear_normalizer.to(pl_module.device)
            InferenceHelper.channels_join = InferenceHelper.channels_join.to(pl_module.device)

        with torch.no_grad():
            pred_field = InferenceHelper.inference_step(batch, pl_module, voxel_resolution, spectra_bins=spectra_bins)
            gt = InferenceHelper.channels_join.forward(batch.ground_truth) if isinstance(batch.ground_truth, RadiationField) else batch.ground_truth
            pred_field = InferenceHelper.linear_normalizer.forward(pred_field)
            gt = InferenceHelper.linear_normalizer.forward(gt)

            return gt, pred_field

    @staticmethod
    def extract_gt(batch: TrainingInputData) -> RadiationFieldChannel | AirKermaField:
        gt = batch.original_ground_truth if batch.original_ground_truth is not None else batch.ground_truth
        if isinstance(gt, RadiationField):
            gt = InferenceHelper.channels_join.forward(gt)
        elif not isinstance(gt, RadiationFieldChannel) or isinstance(gt, AirKermaField):
            raise ValueError("Ground truth must be RadiationField or RadiationFieldChannel or AirKermaField for metrics plotting.")
        return gt

    @staticmethod
    def extract_fluence_or_airkerma(field: RadiationFieldChannel | RadiationField | AirKermaField) -> torch.Tensor:
        if isinstance(field, RadiationField):
            field = InferenceHelper.channels_join.forward(field)
            return field.flux
        elif isinstance(field, RadiationFieldChannel):
            return field.flux
        elif isinstance(field, AirKermaField):
            return field.air_kerma
        else:
            raise ValueError("Field must be RadiationField, RadiationFieldChannel or AirKermaField to extract flux or air kerma.")

    @staticmethod
    def try_extract_spectrum(field: RadiationFieldChannel | RadiationField | AirKermaField) -> torch.Tensor | None:
        if isinstance(field, RadiationField):
            field = InferenceHelper.channels_join.forward(field)
            return field.spectrum
        elif isinstance(field, RadiationFieldChannel):
            return field.spectrum
        elif isinstance(field, AirKermaField):
            return None
        else:
            raise ValueError("Field must be RadiationField, RadiationFieldChannel or AirKermaField to extract spectrum.")

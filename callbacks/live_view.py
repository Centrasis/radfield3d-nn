from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
import torch

from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.rfhelpers import InferenceHelper
from radfield3dnn.rftypes import TrainingInputData
from visualizers.live_reconstruction import LiveReconstructionView


class LiveReconstructionCallback(Callback):
    """Renders ground truth vs. prediction of validation field ``field_index`` after each
    validation pass into a page served on 127.0.0.1:``port`` (see LiveReconstructionView)."""

    def __init__(self, out_paths: list[str], spectra_bins: int = 32, voxel_resolution: tuple = (64, 64, 64),
                 field_index: int = 0, refresh_s: int = 20, open_browser: bool = True, port: int = 8765):
        super().__init__()
        self.voxel_resolution = torch.tensor(voxel_resolution, dtype=torch.int32)
        self.spectra_bins = spectra_bins
        self.field_index = field_index
        self.view = LiveReconstructionView(out_paths, refresh_s=refresh_s, open_browser=open_browser, port=port)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: BaseNeuralRadFieldModel, outputs,
                                batch: TrainingInputData, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx != self.field_index:
            return
        gt, pred_field = InferenceHelper.generate_gt_and_pred_for_validation(
            batch, pl_module, voxel_resolution=self.voxel_resolution, spectra_bins=self.spectra_bins)
        with torch.no_grad():
            gt_flux = InferenceHelper.extract_flux_or_airkerma(gt)
            pred_flux = InferenceHelper.extract_flux_or_airkerma(pred_field).detach()
        gt_vol = gt_flux.reshape(-1, *gt_flux.shape[-3:])[0]
        pred_vol = pred_flux.reshape(-1, *pred_flux.shape[-3:])[0]
        # The framework batch may carry a geometry channel; passing it (geometry=..., slice_center=...) is left to the wiring.
        self.view.render(gt_vol, pred_vol, geometry=None, slice_center=None, title=f"epoch {trainer.current_epoch} — validation field {batch_idx}")

    def on_fit_end(self, trainer: Trainer, pl_module: BaseNeuralRadFieldModel) -> None:
        self.view.close()

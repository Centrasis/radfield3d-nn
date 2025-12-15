from models.base import BaseNeuralRadFieldModel
from datasets.dataloader import RadiationFieldDataModule
import lightning.pytorch as pl
from loggers.logger import LoggerBase


class Task:
    def run_task(self, trainer: pl.Trainer, model: BaseNeuralRadFieldModel, datamodule: RadiationFieldDataModule):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int) -> list[pl.Callback]:
        return []

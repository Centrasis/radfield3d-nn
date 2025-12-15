import lightning.pytorch as pl
from sys import platform
from tasks.base import Task
from callbacks.metrics_plotter import ErrorFieldPlotter
from models.base import BaseNeuralRadFieldModel
from datasets.dataloader import RadiationFieldDataModule
from rich import print
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from loggers.logger import LoggerBase
from callbacks.plotter import ValidationPlotter
from callbacks.warmup_early_stopping import WarmupEarlyStopping


class TrainTask(Task):
    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int) -> list[pl.Callback]:
        return [
            WarmupEarlyStopping(monitor="val_loss", patience=max(epochs // 5, 3), mode="min", warmup_epochs=epochs // 3),
            ModelCheckpoint(
                dirpath=os.path.join(logs_path, "models"),
                filename=f"{model_name}-" + '{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                save_last=True,
                save_top_k=1,
                monitor="val_loss",
            ),
            ValidationPlotter(
                logger,
                mu_tr_path=mu_tr_file,
                max_energy_eV=1.5e+5,
                voxel_size=voxel_size_m if voxel_size_m > 0.0 else 0.01,
                plot_spectra=True,
                voxel_resolution=voxel_resolution if voxel_resolution is not None else (50, 50, 50)
            ),
            ErrorFieldPlotter(
                logger=logger,
                spectra_bins=32,
                mu_tr_file=mu_tr_file,
                max_energy_eV=1.5e+5,
                voxel_resolution=voxel_resolution if voxel_resolution is not None else (50, 50, 50)
            )
        ]

    def run_task(self, trainer: pl.Trainer, model: BaseNeuralRadFieldModel, datamodule: RadiationFieldDataModule):
        print("[green]Starting training task...")
        trainer.fit(model, datamodule=datamodule)
        print("[green]Final test!")
        trainer.test(model, datamodule=datamodule)

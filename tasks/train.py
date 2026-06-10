import lightning.pytorch as pl
from tasks.base import Task
from callbacks.metrics_plotter import ErrorFieldPlotter, SpectrumErrorFieldPlotter
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.datasets.dataloader import RadiationFieldDataModule
from rich import print
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from loggers.logger import LoggerBase
from callbacks.plotter import ValidationPlotter
from callbacks.warmup_early_stopping import WarmupEarlyStopping
from callbacks.package_export import PackageExportCallback
from lightning.pytorch.tuner import Tuner


class TrainTask(Task):
    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int, dataset_path: str) -> list[pl.Callback]:
        return [
            WarmupEarlyStopping(monitor="val_raw_loss", patience=max(epochs // 5, 3), mode="min", warmup_epochs=epochs // 3),
            ModelCheckpoint(
                dirpath=os.path.join(logs_path, "models"),
                filename=f"{model_name}-" + '{epoch}-{val_raw_loss:.2f}',
                save_last=True,
                save_top_k=1,
                monitor="val_raw_loss",
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
            ),
            SpectrumErrorFieldPlotter(
                logger=logger,
                spectra_bins=32,
                voxel_resolution=voxel_resolution if voxel_resolution is not None else (50, 50, 50)
            ),
            PackageExportCallback(
                out_dir=os.path.join(logs_path, "models"),
                model_name=model_name,
                dataset_path=dataset_path,
                max_energy_eV=1.5e+5,
                spectra_bins=32,
            ),
        ]

    def run_task(self, trainer: pl.Trainer, model: BaseNeuralRadFieldModel, datamodule: RadiationFieldDataModule):
        # The LR finder is run only for models that support it. Fused fp16 cpp
        # models (PBRFNetCPP) opt out (`use_lr_finder=False`): the finder's
        # high-LR sweep overflows their fp16 fused weights to NaN, and their
        # `configure_optimizers` clamps the LR to `max_lr` anyway, so the sweep
        # is both harmful and pointless — they run at their configured LR.
        if getattr(model, "use_lr_finder", True):
            print(f"[blue]Search learning rate with batch size {datamodule.batch_size}...")
            lr_trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_steps=250,
                precision=trainer.precision,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=True,
                num_sanity_val_steps=trainer.num_sanity_val_steps
            )
            lr_tuner = Tuner(lr_trainer)
            lr_result = lr_tuner.lr_find(
                model,
                datamodule=datamodule,
                min_lr=1e-4,    # used by original NeRF paper as initial lr (5e-4)
                max_lr=1e-2,
                num_training=250
            )
            suggested_lr = lr_result.suggestion()
            if suggested_lr is None:
                print(f"[yellow]LR finder returned no suggestion; keeping current lr {model._lr}.")
            else:
                print(f"[green]LR finder suggestion: {suggested_lr}")
                model._lr = float(suggested_lr)
        else:
            print(f"[yellow]Skipping LR finder for fixed-LR model ({type(model).__name__}); using lr={model._lr}.")

        print("[green]Starting training task...")
        # Optional resume: RF_RESUME_CKPT=<path/last.ckpt> restores weights + optimizer +
        # scheduler + epoch, continuing a run that was interrupted (e.g. by a DataLoader stall).
        resume_ckpt = os.environ.get("RF_RESUME_CKPT") or None
        if resume_ckpt:
            print(f"[green]Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
        print("[green]Final test!")
        trainer.test(model, datamodule=datamodule)

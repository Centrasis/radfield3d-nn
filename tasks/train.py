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
    def __init__(self, early_stopping: bool = True, checkpoint_monitor: str = None):
        # `training: early_stopping: false` runs the full epoch budget: under a cosine schedule
        # the accuracy still climbs while the LR decays, and a patience stop cuts that tail.
        self.early_stopping = bool(early_stopping)
        # `training: checkpoint_monitor:` — the validation metric that selects the checkpoint and
        # drives early stopping. The default whole-field air-kerma accuracy is the wrong choice on a
        # dataset whose bulk is Monte-Carlo noise: it scores the unresolved voxels, so it peaks in
        # the first epochs and then FALLS as the model commits to the resolved signal, leaving the
        # trained weights unsaved. Name a metric that tracks quality there instead.
        self.checkpoint_monitor = checkpoint_monitor

    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int, dataset_path: str) -> list[pl.Callback]:
        # Early stop + checkpoint selection monitor the AIR-KERMA SCATTER ACCURACY, not the raw
        # loss: the SMAPEBalanced loss decouples from the physical metric (the loss plateaus/rises
        # while accuracy keeps climbing), so a val_raw_loss monitor stops runs early and picks
        # non-best checkpoints (e.g. the roi_full run was cut at epoch 54 = warmup + patience).
        # On a JOINED dataset the scatter-isolation metric does not exist; monitor the whole-field
        # GLOBAL air-kerma accuracy. NOT top90: top90 is beam-core-biased and decouples from the
        # SMAPEBalanced loss (peaks in the first epochs, then decays while global accuracy and
        # spectrum accuracy keep improving — ds04-analytical run wrm56i5b: top90 best 0.711@ep10
        # but global 0.42@ep10 vs 0.69@ep47).
        from radfield3dnn.datasets import list_dataset_field_names, load_dataset_field
        _probe = list_dataset_field_names(dataset_path)
        _joined = bool(_probe) and not all(
            c in load_dataset_field(dataset_path, _probe[0]).get_channel_names()
            for c in ("scatter_field", "direct_beam"))
        monitor = self.checkpoint_monitor or ("val_global_airkerma_accuracy" if _joined else "val_airkerma_accuracy_scatter")
        if self.checkpoint_monitor:
            print(f"[yellow]Checkpoint/early-stop monitor overridden: {monitor}")
        early_stop = [WarmupEarlyStopping(monitor=monitor, patience=max(epochs // 5, 3), mode="max", warmup_epochs=epochs // 3)] \
            if self.early_stopping else []
        return early_stop + [
            ModelCheckpoint(
                dirpath=os.path.join(logs_path, "models"),
                filename=f"{model_name}-{{epoch}}-{{{monitor}:.3f}}",
                save_top_k=1,
                monitor=monitor,
                mode="max",
            ),
            # The final epoch's weights, every epoch. Lightning's save_last writes only on the steps
            # where the monitored checkpoint also saves, so it would hold the best epoch instead.
            ModelCheckpoint(
                dirpath=os.path.join(logs_path, "models"),
                filename="last",
                save_top_k=1,
                monitor=None,
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
        # The LR finder is run only for models that support it (`use_lr_finder`).
        # On resume the checkpoint restores the optimizer (incl. its LR), so the
        # finder's 250-step sweep would be pure overhead and overwrite it.
        if getattr(model, "use_lr_finder", True) and not os.environ.get("RF_RESUME_CKPT"):
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
        # Test (and the on_test_end RF3M export) on the BEST checkpoint, not the last-epoch
        # weights — otherwise the shipped model is whatever epoch training happened to end on.
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

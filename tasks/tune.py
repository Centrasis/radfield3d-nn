import lightning.pytorch as pl
from sys import platform

import torch
from tasks.base import Task
from lightning.pytorch.tuner import Tuner
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.datasets.dataloader import RadiationFieldDataModule
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDatasetWithGeometry
from rich import print
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from loggers.logger import LoggerBase, TrainingSettings
from callbacks.plotter import ValidationPlotter
from callbacks.warmup_early_stopping import WarmupEarlyStopping
from radfield3dnn.models import ModelConstructor
from lightning.pytorch.callbacks import ModelSummary

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
import copy
import json
import gc
from contextlib import suppress
import multiprocessing as mp
from RadFiled3D.pytorch.datasets.processing import DataProcessing
import traceback

from callbacks.metrics_plotter import MetricsPlotter
from radfield3dnn.metrics.airkerma_accuracy import AirkermaAccuracy, AirkermaSphereAccuracy, AirkermaScatterAccuracy, AirkermaAccuracyEnergyWeighted
from radfield3dnn.metrics.ssim import AirkermaSSIM
from radfield3dnn.metrics import HistogramOverlapAccuracy


class HyperparameterTuningTask(Task):
    def __init__(self, model_config: str, experiment_name: str, n_trials: int):
        super().__init__()
        with open(model_config, "r") as f:
            self.base_model_config = json.load(f)
        self.logger: LoggerBase = None
        self.trainer: pl.Trainer = None
        self.model_name: str = ""
        self.datamodule: RadiationFieldDataModule = None
        self.hyper_parameter_space = self.base_model_config["hyperparameter_space"]
        self.study = None
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.trainer_callbacks = []
        # NEW: plain config holders
        self._trainer_cfg: dict = {}
        self._datamodule_cfg: dict = {}
        self._logger_cls = None
        self._logger_init_kwargs: dict = {}

    @staticmethod
    def create_processing_from_config(name: str, norm_cfg: dict):
        for cls in DataProcessing.__subclasses__():
            if cls.__name__ == name:
                try:
                    return cls.create_from_config(norm_cfg)
                except Exception as e:
                    raise ValueError(f"Error creating DataProcessing ({name}) from config: {e}")
        raise ValueError(f"Unsupported DataProcessing class in config: {name}")

    def run_task(self, trainer: pl.Trainer, model: BaseNeuralRadFieldModel, datamodule: RadiationFieldDataModule):
        self.max_inner_batch_size = model.max_inner_batch_size

        # Propagate the general-purpose run settings to the per-trial trainers so
        # tuning matches single-run training: weight EMA (stability) and the
        # LR-finder toggle. The per-trial trainers are rebuilt from primitives
        # below, so capture these as primitives here before `trainer` is dropped.
        _ema_cb = next((c for c in getattr(trainer, "callbacks", []) if c.__class__.__name__ == "WeightEMA"), None)
        self._trainer_cfg = {
            # ensure primitives only
            "accelerator": "cuda" if torch.cuda.is_available() else "gpu",
            "precision": str(trainer.precision) if getattr(trainer, "precision", None) is not None else "32-true",
            "max_epochs": int(trainer.max_epochs),
            "weight_ema_decay": float(_ema_cb.decay) if _ema_cb is not None else None,
            "use_lr_finder": bool(getattr(model, "use_lr_finder", True)),
        }
        self._datamodule_cfg = {
            "batch_size": datamodule.batch_size,
            "cpu_count": datamodule.cpu_count,
            "dataset_path": datamodule.dataloader_builder.dataset_path,
            "data_processings": [(aug.get_name(), aug.get_parameters()) for aug in datamodule.data_processings],
        }
        self._logger_cls = self.logger.__class__ if self.logger else None
        if hasattr(self.logger, "export_init_kwargs"):
            self._logger_init_kwargs = self.logger.export_init_kwargs()
        # Drop heavy objects (will be rebuilt per trial inside objective)
        self.trainer = None
        self.datamodule = None

        # Ensure a persistent storage so subprocess results aggregate into the same study
        optuna_path = os.path.join(self.logs_path, "optuna_studies")
        os.makedirs(optuna_path, exist_ok=True)
        storage_url = f"sqlite:///{os.path.join(optuna_path, 'optuna_study.db').replace(os.sep, '/')}"
        self.study = optuna.create_study(
            direction='minimize',
            study_name=self.experiment_name,
            storage=storage_url,
            load_if_exists=True
        )

        # Determine next available trial ID offset and respect missing ids in between
        files_pesent = len([fname for fname in os.listdir(optuna_path) if fname.startswith("optuna_model_") and fname.endswith(".json")])
        trial_id_offset = 0
        checked_files = 0
        while checked_files < files_pesent:
            path_to_check = os.path.join(optuna_path, f"optuna_model_{trial_id_offset}.json")
            if os.path.exists(path_to_check):
                checked_files += 1
            trial_id_offset += 1
            if trial_id_offset > 1000:
                raise RuntimeError("Too many missing trial IDs; aborting to prevent infinite loop.")
            
        if trial_id_offset > 0:
            print(f"[yellow]Resuming from trial ID offset {trial_id_offset} due to existing trials in storage.[/yellow]")

        for i in range(self.n_trials):
            p = mp.Process(
                target=_run_single_trial_subprocess,
                args=(
                    trial_id_offset + i,
                    storage_url,
                    self.experiment_name,
                    self.base_model_config,
                    self.hyper_parameter_space,
                    self._trainer_cfg,
                    self._datamodule_cfg,
                    self._logger_cls,
                    self._logger_init_kwargs,
                    self.model_name,
                    self.logs_path,
                    self.max_inner_batch_size,
                    self.voxel_size_m,
                    self.voxel_resolution,
                    self.mu_tr_file
                )
            )
            p.start()
            p.join()
        # Reload the study from storage to get aggregated results
        self.study = optuna.load_study(study_name=self.experiment_name, storage=storage_url)

        print("[green]Best trial:")
        trial = self.study.best_trial
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int) -> list[pl.Callback]:
        self.model_name = model_name
        self.logger = logger
        self.logs_path = logs_path
        self.mu_tr_file = mu_tr_file
        self.voxel_resolution = voxel_resolution
        self.voxel_size_m = voxel_size_m
        return []

    @staticmethod
    def create_metrics_plotter_cb(model: BaseNeuralRadFieldModel, mu_tr_file: str, voxel_resolution: tuple[float, float, float], voxel_size_m: float) -> MetricsPlotter:
        return MetricsPlotter(
            spectra_bins=32,
            metrics={
                'global_airkerma_accuracy': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5
                ),
                'top90_airkerma_accuracy': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    importance_threshold=0.1
                ),
                'airkerma_ssim': AirkermaSSIM(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    reduction='mean'
                ),
                'airkerma_ssim_gradient': AirkermaSSIM(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    reduction='mean',
                    ssim_type='gradient'
                ),
                'airkerma_onsphere_accuracy_radius25cm': AirkermaSphereAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    sphere_radius_m=0.25,
                    voxel_size_m=voxel_size_m if voxel_size_m > 0.0 else 0.01
                ),
                'airkerma_accuracy_scatter': AirkermaScatterAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5
                ),
                'spectrum_accuracy': HistogramOverlapAccuracy(),
                'top95_energy_weighted_airkerma_accuracy': AirkermaAccuracyEnergyWeighted(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    importance_threshold=0.05
                ),
                'global_airkerma_gamma_index_3percent_per_4cm': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    metric_type='gpr',
                    voxel_size_m=voxel_size_m if voxel_size_m > 0.0 else 0.01,
                    rel_dose_diff=0.03,
                    dist_crit_mm=40.0
                ),
                'global_airkerma_gamma_index_10percent_per_4cm': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    metric_type='gpr',
                    voxel_size_m=voxel_size_m if voxel_size_m > 0.0 else 0.01,
                    rel_dose_diff=0.1,
                    dist_crit_mm=40.0
                ),
                'global_airkerma_gamma_index_10percent_per_6cm': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    metric_type='gpr',
                    voxel_size_m=voxel_size_m if voxel_size_m > 0.0 else 0.01,
                    rel_dose_diff=0.1,
                    dist_crit_mm=60.0
                ),
                'global_airkerma_gamma_index_3percent_per_6cm': AirkermaAccuracy(
                    mu_tr_file=mu_tr_file,
                    spectra_bins=32,
                    max_energy_eV=1.5e+5,
                    metric_type='gpr',
                    voxel_size_m=voxel_size_m if voxel_size_m > 0.0 else 0.01,
                    rel_dose_diff=0.03,
                    dist_crit_mm=60.0
                )
            },
            voxel_resolution=voxel_resolution if voxel_resolution is not None else (50, 50, 50)
        )

    @staticmethod
    def append_results_csv(trial_id: int, results: dict, csv_file: str):
        df = pd.DataFrame([results])
        df.insert(0, 'trial_id', trial_id)
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)

    def _attach_trainer_callback(self, callback: pl.Callback) -> bool:
        add_callback = getattr(self.trainer, "add_callback", None)
        has_added_callback = False
        if callable(add_callback):
            add_callback(callback)
            has_added_callback = True
        existing_callbacks = getattr(self.trainer, "callbacks", None)
        if isinstance(existing_callbacks, list):
            self.trainer_callbacks = existing_callbacks
            if not has_added_callback:
                if callback not in existing_callbacks:
                    existing_callbacks.append(callback)
            has_added_callback = True
        if isinstance(existing_callbacks, tuple):
            try:
                callback_list = list(existing_callbacks)
                self.trainer_callbacks = callback_list
                if not has_added_callback:
                    if callback not in callback_list:
                        callback_list.append(callback)
                    setattr(self.trainer, "callbacks", callback_list)
                has_added_callback = True
            except AttributeError:
                pass
        connector = getattr(self.trainer, "_callback_connector", None)
        if connector and hasattr(connector, "_callbacks"):
            self.trainer_callbacks = connector._callbacks
            if not has_added_callback:
                if callback not in connector._callbacks:
                    connector._callbacks.append(callback)
            has_added_callback = True
        if not has_added_callback:
            print("[yellow]Unable to register pruning callback on trainer; proceeding without it.")
        return has_added_callback

    def _detach_trainer_callback(self, callback: pl.Callback):
        remove_callback = getattr(self.trainer, "remove_callback", None)
        if callable(remove_callback):
            remove_callback(callback)
            return
        existing_callbacks = getattr(self.trainer, "callbacks", None)
        if isinstance(existing_callbacks, list) and callback in existing_callbacks:
            existing_callbacks.remove(callback)
            return
        connector = getattr(self.trainer, "_callback_connector", None)
        if connector and hasattr(connector, "_callbacks") and callback in connector._callbacks:
            connector._callbacks.remove(callback)

    def _cleanup_after_trial(self, trainers: list[pl.Trainer]):
        for trainer in trainers:
            if trainer is None:
                continue
            callbacks = getattr(trainer, "callbacks", None)
            if isinstance(callbacks, list):
                callbacks.clear()
            teardown_hook = getattr(trainer, "_call_teardown_hook", None)
            if callable(teardown_hook):
                with suppress(Exception):
                    teardown_hook()
            strategy = getattr(trainer, "strategy", None)
            if strategy is not None:
                strategy_teardown = getattr(strategy, "teardown", None)
                if callable(strategy_teardown):
                    with suppress(Exception):
                        strategy_teardown()
        builder = getattr(self.datamodule, "dataloader_builder", None)
        if builder is not None:
            for attr_name in ("close", "close_all", "shutdown"):
                closer = getattr(builder, attr_name, None)
                if callable(closer):
                    with suppress(Exception):
                        closer()
        teardown_dm = getattr(self.datamodule, "teardown", None)
        if callable(teardown_dm):
            for stage in ("fit", "validate", "test", None):
                with suppress(Exception):
                    teardown_dm(stage)
        for cache_attr in ("_train_dataloader", "_val_dataloader", "_test_dataloader", "_predict_dataloader"):
            if hasattr(self.datamodule, cache_attr):
                setattr(self.datamodule, cache_attr, None)
        gc.collect()

    @staticmethod
    def create_datamodule_from_config(datamodule_cfg: dict) -> RadiationFieldDataModule:
        try:
            cfg = {
                "zip_directory": datamodule_cfg["dataset_path"],
                "dataset_cls": RadField3DDatasetWithGeometry,
                "batch_size": datamodule_cfg["batch_size"],
                "num_workers": datamodule_cfg["cpu_count"],
                "data_processings": [
                    HyperparameterTuningTask.create_processing_from_config(name, params)
                    for name, params in datamodule_cfg["data_processings"]
                ]
            }
            dm_kwargs = {k: v for k, v in cfg.items()
                            if k in RadiationFieldDataModule.__init__.__code__.co_varnames}
            return RadiationFieldDataModule(**dm_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct datamodule from plain config: {e}")

# Subprocess worker: runs exactly one trial in a fresh forked process
def _run_single_trial_subprocess(
    external_trial_id: int,
    storage_url: str,
    study_name: str,
    base_model_config: dict,
    hyper_space: dict,
    trainer_cfg: dict,
    datamodule_cfg: dict,
    logger_cls,
    logger_init_kwargs: dict,
    model_name: str,
    logs_path: str,
    max_inner_batch_size: int,
    voxel_size_m: float,
    voxel_resolution: tuple[float],
    mu_tr_file: str
):
    try:
        # attach pruner to the study in the subprocess so pruning actually works
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        study = optuna.create_study(
            direction='minimize',
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True,
            pruner=pruner
        )

        def objective(trial: optuna.Trial):
            # Hyperparameter suggestions
            parameter_suggestions = {}
            for pname, pvals in hyper_space.items():
                if isinstance(pvals, (list, tuple)):
                    parameter_suggestions[pname] = trial.suggest_categorical(pname, list(pvals))
                else:
                    raise ValueError(f"Hyperparameter space for {pname} must be list/tuple.")
            # Build model
            model_config = copy.deepcopy(base_model_config)
            model_config["parameters"].update(parameter_suggestions)
            model_cls = ModelConstructor.create_model_from_dict(model_config)
            model = model_cls()
            model.max_inner_batch_size = max_inner_batch_size
            datamodule = HyperparameterTuningTask.create_datamodule_from_config(datamodule_cfg)
            # Prepare data
            datamodule.is_prepared = False
            datamodule.prepare_data()
            # Logger reconstruction
            local_logger = None
            if logger_cls:
                try:
                    local_logger = logger_cls(**logger_init_kwargs)
                    local_logger.setup_experiment(
                        local_logger.experiment_name,
                        TrainingSettings(
                            batch_size=datamodule_cfg["batch_size"],
                            num_workers=datamodule_cfg["cpu_count"],
                            epochs=trainer_cfg["max_epochs"],
                            model_name=model_name,
                            dataset_path=datamodule_cfg["dataset_path"],
                            dataset_loading_mode="layerwise",
                            hyper_parameters=parameter_suggestions,
                            data_augmentations=datamodule_cfg["data_processings"]  # fixed key
                        )
                    )
                except Exception:
                    local_logger = None
            # LR finder — respect the same toggle as single-run training. The
            # finder picks a different LR per seed/trial (a variance source); when
            # disabled, the trial uses the model's configured LR for reproducibility.
            if trainer_cfg.get("use_lr_finder", True) and getattr(model, "use_lr_finder", True):
                lr_trainer = pl.Trainer(
                    accelerator="cuda" if torch.cuda.is_available() else "gpu",
                    devices=1,
                    max_steps=250,
                    precision=trainer_cfg["precision"],
                    logger=False,
                    enable_progress_bar=False,
                    enable_checkpointing=True,
                    num_sanity_val_steps=0 if platform == "win32" else 2
                )
                lr_tuner = Tuner(lr_trainer)
                lr_result = lr_tuner.lr_find(
                    model,
                    datamodule=datamodule,
                    min_lr=1e-4,
                    max_lr=1e-2,
                    num_training=250
                )
                _sug = lr_result.suggestion()
                if _sug is not None:
                    model._lr = float(_sug)
            # Main trainer
            pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_raw_loss")
            metrics_cb = HyperparameterTuningTask.create_metrics_plotter_cb(
                model=model,
                mu_tr_file=mu_tr_file,
                voxel_resolution=voxel_resolution,
                voxel_size_m=voxel_size_m
            )

            print(f"[blue] Start training of trial {external_trial_id} with parameters: {parameter_suggestions} and learning rate: {model._lr}[/blue]")

            _tune_callbacks = [pruning_cb, metrics_cb]
            if trainer_cfg.get("weight_ema_decay"):
                # Same stability EMA as single-run training (evaluate smoothed weights).
                from callbacks.ema import WeightEMA
                _tune_callbacks.append(WeightEMA(decay=float(trainer_cfg["weight_ema_decay"])))
            tune_trainer = pl.Trainer(
                accelerator="cuda" if torch.cuda.is_available() else "gpu",
                devices=1,
                max_epochs=trainer_cfg["max_epochs"],
                precision=trainer_cfg["precision"],
                logger=(local_logger.get_lightning_callback() if local_logger else False),
                enable_progress_bar=False,
                enable_checkpointing=False,
                callbacks=_tune_callbacks,
                num_sanity_val_steps=0 if platform == "win32" else 2
            )
            tune_trainer.fit(model, datamodule=datamodule)
            callback_metrics = getattr(tune_trainer, "callback_metrics", {})
            # Optimise the raw (positive, quality-meaningful) loss, not the
            # DB-MTL log-space `val_loss` which can be negative (findings/O6).
            val_raw_loss = float(callback_metrics.get("val_raw_loss", torch.tensor(float("inf"))))
            test_metrics = tune_trainer.test(model, datamodule=datamodule)
            results = {}
            if "val_raw_loss" in callback_metrics:
                results["val_raw_loss"] = val_raw_loss
            if test_metrics:
                results.update(test_metrics[0])
            # Persist using external_trial_id
            optuna_path = os.path.join(logs_path, "optuna_studies")
            os.makedirs(optuna_path, exist_ok=True)
            with open(os.path.join(optuna_path, f"optuna_model_{external_trial_id}.json"), "w") as f:
                f.write(json.dumps({
                    "parameters": model.get_custom_parameters(),
                    "model_name": model_name,
                    "metrics": results
                }, indent=4))
            HyperparameterTuningTask.append_results_csv(
                external_trial_id,
                results,
                os.path.join(optuna_path, "optuna_study_results.csv")
            )
            torch.save(model.state_dict(), os.path.join(optuna_path, f"optuna_model_{external_trial_id}.pt"))
            print(f"[blue] Wrote result of external trial {external_trial_id} to {optuna_path}.")
            if local_logger:
                local_logger.finalize_logging()
            return val_raw_loss

        study.optimize(objective, n_trials=1)

    except Exception as e:
        print(f"[red]Optuna subprocess failed (external trial {external_trial_id}): {e}")
        print(traceback.format_exc())  # added stack trace

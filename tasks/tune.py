import lightning.pytorch as pl
from sys import platform

import torch
from tasks.base import Task
from lightning.pytorch.tuner import Tuner
from radfield3dnn.models.base import BaseNeuralRadFieldModel
from radfield3dnn.datasets.dataloader import RadiationFieldDataModule
from radfield3dnn.datasets import construct_datamodule
from rich import print
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from loggers.logger import LoggerBase, TrainingSettings
from callbacks.plotter import ValidationPlotter
from callbacks.warmup_early_stopping import WarmupEarlyStopping
from radfield3dnn.models import ModelConstructor
from lightning.pytorch.callbacks import ModelSummary
from radfield3dnn.datasets.mc_floor_cut import MCFloorCut
from radfield3dnn.preprocessing.augmentations import ROIbasedSampler
from radfield3dnn.preprocessing.normalizations import TranslationNormalization

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
import copy
import json
import gc
from contextlib import suppress
import multiprocessing as mp
import traceback

from callbacks.metrics_plotter import MetricsPlotter
from radfield3dnn.metrics.airkerma_accuracy import AirkermaAccuracy, AirkermaSphereAccuracy, AirkermaScatterAccuracy, AirkermaAccuracyEnergyWeighted
from radfield3dnn.metrics.ssim import AirkermaSSIM
from radfield3dnn.metrics import HistogramOverlapAccuracy
from radfield3dnn.metrics.factory import build_airkerma_metrics


def encode_search_space(hyper_space: dict) -> dict:
    """``hyperparameter_space`` -> the choices Optuna actually sees, per parameter.

    Plain values (int/float/str/bool) pass through. STRUCTURED choices (dicts/lists — e.g. whole
    ``conditioning_params`` or ``location_encoding_params`` blocks) are JSON-encoded: Optuna's
    sqlite storage and distribution-compatibility checks on study resume want hashable primitive
    choices, and the JSON string doubles as a readable parameter value in the study UI/CSV.
    This is the single encoding — both the per-trial draw and the grid sampler go through it.
    """
    encoded = {}
    for pname, pvals in hyper_space.items():
        if not isinstance(pvals, (list, tuple)):
            raise ValueError(f"Hyperparameter space for {pname} must be list/tuple.")
        if any(isinstance(v, (dict, list)) for v in pvals):
            encoded[pname] = [json.dumps(v, sort_keys=True) for v in pvals]
        else:
            encoded[pname] = list(pvals)
    return encoded


def suggest_parameters(trial, hyper_space: dict) -> dict:
    """Draw one suggestion per entry of ``hyperparameter_space``, decoding the JSON-encoded
    structured blocks back into dicts (see encode_search_space)."""
    suggestions = {}
    for pname, choices in encode_search_space(hyper_space).items():
        drawn = trial.suggest_categorical(pname, choices)
        original = hyper_space[pname]
        suggestions[pname] = json.loads(drawn) if any(
            isinstance(v, (dict, list)) for v in original) else drawn
    return suggestions


def apply_suggestions(parameters: dict, suggestions: dict) -> dict:
    """Apply drawn suggestions onto a (deep-copied) ``parameters`` dict.

    A plain key replaces the top-level parameter. A DOTTED key ("conditioning_params.
    translation_encoding") replaces a nested block — that is what lets orthogonal sub-blocks be
    separate search axes instead of enumerating their cross product by hand. Intermediate dicts
    are copied (never mutated: the base config is reused across trials) and must already exist —
    a typo'd path fails loudly instead of silently creating a parameter no model reads.
    """
    out = copy.deepcopy(parameters)
    for key, value in suggestions.items():
        if "." not in key:
            out[key] = value
            continue
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"hyperparameter path {key!r}: {part!r} is not an existing dict "
                               f"in the base parameters — fix the path or the base config.")
            node = node[part]
        node[parts[-1]] = value
    return out


class HyperparameterTuningTask(Task):
    def __init__(self, model_config: str, experiment_name: str, n_trials: int, sampler: str = "tpe"):
        super().__init__()
        with open(model_config, "r") as f:
            self.base_model_config = json.load(f)
        self.logger: LoggerBase = None
        self.trainer: pl.Trainer = None
        self.model_name: str = ""
        self.datamodule: RadiationFieldDataModule = None
        if "hyperparameter_space" not in self.base_model_config:
            raise ValueError(
                f"--task tune needs a 'hyperparameter_space' block in the model JSON, but "
                f"{model_config} has none. It maps each searched parameter to its list of "
                f"candidate values (a dotted key like 'conditioning_params.translation_encoding' "
                f"replaces a nested sub-block); see configs/tpbrfnet_ds04.model.json.")
        self.hyper_parameter_space = self.base_model_config["hyperparameter_space"]
        self.study = None
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        # "tpe" (Optuna's default) explores a large/continuous space but samples WITH replacement:
        # over a small exhaustive grid it repeats combinations and leaves others unmeasured. "grid"
        # runs each combination exactly once — the right sampler for an ablation.
        if sampler not in ("tpe", "grid"):
            raise ValueError(f"tune.sampler must be 'tpe' or 'grid', got {sampler!r}.")
        self.sampler = sampler
        grid_size = 1
        for choices in encode_search_space(self.hyper_parameter_space).values():
            grid_size *= len(choices)
        self.grid_size = grid_size
        if sampler == "grid" and n_trials < grid_size:
            print(f"[yellow]tune.sampler=grid: n_trials={n_trials} < {grid_size} grid points — "
                  f"{grid_size - n_trials} combination(s) will not be measured.")
        self.trainer_callbacks = []
        # NEW: plain config holders
        self._trainer_cfg: dict = {}
        self._datamodule_spec: dict = {}
        self.joined_dataset: bool = False
        self._logger_cls = None
        self._logger_init_kwargs: dict = {}
        self._logger_run_name: str = ""

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
        # The datamodule hands back the construct_datamodule call that built it, so each trial
        # subprocess rebuilds an IDENTICAL one — same dataset decorators (patient translation,
        # geometry mask), same beam-parameter normalization, same processing order. A trial that
        # trains on a differently-shaped batch than the single-run pipeline is not a tuning result.
        self._datamodule_spec = datamodule.construction_spec()
        # A Voxelwise run overrides batch_size AFTER construct_datamodule (run_network_task.py),
        # so take the live value, not the one the spec was built with.
        self._datamodule_spec["batch_size"] = datamodule.batch_size
        self._logger_cls = self.logger.__class__ if self.logger else None
        if self.logger is not None:
            self._logger_init_kwargs = self.logger.export_init_kwargs()
            self._logger_run_name = self.logger.experiment_name
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

        consecutive_failures = 0
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
                    self._datamodule_spec,
                    self._logger_cls,
                    self._logger_init_kwargs,
                    self._logger_run_name,
                    self.model_name,
                    self.logs_path,
                    self.max_inner_batch_size,
                    self.voxel_size_m,
                    self.voxel_resolution,
                    self.mu_tr_file,
                    self.joined_dataset,
                    self.sampler,
                    self.hyper_parameter_space,
                )
            )
            p.start()
            p.join()
            # A trial that dies leaves NO study entry, so a broken setup would otherwise run the
            # whole budget of empty trials and only fail at `best_trial`. Two in a row is a setup
            # error (bad config, missing dependency), not an unlucky sample.
            if p.exitcode == 0:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            print(f"[red]Trial {trial_id_offset + i} exited with code {p.exitcode} "
                  f"({consecutive_failures} in a row).")
            if consecutive_failures >= 2:
                raise RuntimeError(
                    f"Two consecutive trials failed before reporting a value — the tuning setup "
                    f"itself is broken, not a single sample. The trial's traceback is above.")
        # Reload the study from storage to get aggregated results
        self.study = optuna.load_study(study_name=self.experiment_name, storage=storage_url)
        if not self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            raise RuntimeError(f"No trial completed in study {self.experiment_name!r}; nothing to report.")

        print("[green]Best trial:")
        trial = self.study.best_trial
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    def get_trainer_callbacks(self, model_name: str, logs_path: str, logger: LoggerBase, mu_tr_file: str, voxel_resolution: tuple[int, int, int], voxel_size_m: float, epochs: int, dataset_path: str) -> list[pl.Callback]:
        self.model_name = model_name
        self.logger = logger
        self.logs_path = logs_path
        self.mu_tr_file = mu_tr_file
        self.voxel_resolution = voxel_resolution
        self.voxel_size_m = voxel_size_m
        # A joined dataset carries one 'radiation' channel, so the scatter/beam-ISOLATION metrics
        # have no separate direct_beam to isolate. Probed once here (same check as the train task)
        # and passed to every trial: metrics/factory.py is the single source of the metric set.
        import glob as _glob
        from radfiled3d.store import FieldStore as _FieldStore
        _probe = sorted(_glob.glob(os.path.join(dataset_path, "fields", "*.rf3")))
        self.joined_dataset = bool(_probe) and not all(
            c in _FieldStore.load(_probe[0]).get_channel_names() for c in ("scatter_field", "direct_beam"))
        if self.joined_dataset:
            print("[yellow]Joined dataset (single 'radiation' channel) — scatter/beam-isolation metrics disabled.")
        return []

    @staticmethod
    def create_metrics_plotter_cb(model: BaseNeuralRadFieldModel, mu_tr_file: str, voxel_resolution: tuple[float, float, float], voxel_size_m: float, joined: bool = False) -> MetricsPlotter:
        return MetricsPlotter(
            spectra_bins=32,
            metrics=build_airkerma_metrics(mu_tr_file, voxel_size_m, joined=joined),
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

# Subprocess worker: runs exactly one trial in a fresh forked process
def _run_single_trial_subprocess(
    external_trial_id: int,
    storage_url: str,
    study_name: str,
    base_model_config: dict,
    hyper_space: dict,
    trainer_cfg: dict,
    datamodule_spec: dict,
    logger_cls,
    logger_init_kwargs: dict,
    logger_run_name: str,
    model_name: str,
    logs_path: str,
    max_inner_batch_size: int,
    voxel_size_m: float,
    voxel_resolution: tuple[float],
    mu_tr_file: str,
    joined_dataset: bool = False,
    sampler: str = "tpe",
    full_hyper_space: dict = None,
):
    try:
        # attach pruner to the study in the subprocess so pruning actually works
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
        # GridSampler reads the study's completed trials from the shared storage, so one trial per
        # subprocess still walks the grid exactly once.
        sampler_obj = (optuna.samplers.GridSampler(encode_search_space(full_hyper_space or hyper_space))
                       if sampler == "grid" else None)
        study = optuna.create_study(
            direction='minimize',
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler_obj,
        )

        def objective(trial: optuna.Trial):
            # Hyperparameter suggestions (dict-valued blocks JSON-round-trip through Optuna;
            # dotted keys replace nested sub-blocks)
            parameter_suggestions = suggest_parameters(trial, hyper_space)
            # Build model
            model_config = copy.deepcopy(base_model_config)
            model_config["parameters"] = apply_suggestions(model_config["parameters"], parameter_suggestions)
            model_cls = ModelConstructor.create_model_from_dict(model_config)
            model = model_cls()
            model.max_inner_batch_size = max_inner_batch_size
            # Replay the parent's construct_datamodule call verbatim (its recorded spec already has
            # the translation preflight resolved, so the 11k-field metadata scan does not repeat).
            datamodule = construct_datamodule(**datamodule_spec)
            # Logger reconstruction
            local_logger = None
            if logger_cls:
                try:
                    local_logger = logger_cls(**logger_init_kwargs)
                    # One tracker run PER TRIAL, named by its trial id — otherwise 16 ablation
                    # trials are indistinguishable in the project.
                    local_logger.setup_experiment(
                        f"{logger_run_name}-trial{external_trial_id}",
                        TrainingSettings(
                            batch_size=datamodule_spec["batch_size"],
                            num_workers=datamodule_spec["num_workers"],
                            epochs=trainer_cfg["max_epochs"],
                            model_name=model_name,
                            dataset_path=datamodule_spec["dataset_path"],
                            dataset_loading_mode="layerwise",
                            hyper_parameters=parameter_suggestions,
                            data_augmentations=[(aug.get_name(), aug.get_parameters())
                                                for aug in datamodule.data_processings],
                        )
                    )
                except Exception as e:
                    # Not fatal — the trial's metrics still reach optuna_study_results.csv — but a
                    # silent None means a whole tuning study logs nowhere. Say so.
                    print(f"[red]Trial {external_trial_id}: experiment tracker unavailable "
                          f"({type(e).__name__}: {e}); this trial logs to the optuna CSV only.")
                    print(traceback.format_exc())
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
                voxel_size_m=voxel_size_m,
                joined=joined_dataset,
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
                gradient_clip_val=1.0,   # same as single-run training (run_network_task.py)
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
        # Exit NONZERO: the parent counts consecutive failures and aborts the study rather than
        # burning the whole trial budget on a setup error that repeats every time.
        print(f"[red]Optuna subprocess failed (external trial {external_trial_id}): {e}")
        print(traceback.format_exc())
        raise SystemExit(1)

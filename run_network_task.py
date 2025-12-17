import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary, RichProgressBar, DeviceStatsMonitor
import shutil
from radfield3dnn.models import ModelConstructor
from lightning.pytorch.callbacks import GradientAccumulationScheduler
import argparse
from callbacks.validate_gt import ValidateGroundTruth
from callbacks.metrics_plotter import MetricsPlotter
from radfield3dnn.metrics.airkerma_accuracy import AirkermaAccuracy, AirkermaRelDifferencesStdDev, AirkermaSphereAccuracy, AirkermaScatterAccuracy, AirkermaAccuracyEnergyWeighted
from radfield3dnn.metrics.ssim import AirkermaSSIM
import json
from radfield3dnn.normalizations import NormalizerConstructor

from radfield3dnn.preprocessing.airkerma import AirkermaProcessing
from joblib import Parallel, delayed
from multiprocessing import Manager
import time
from rich import print
from radfield3dnn.datasets import DatasetType, OriginalGroundTruthPreservation, construct_datamodule, get_dataset_dimensions_and_voxel_size

from rich.progress import Progress
from rich.progress import BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
from radfield3dnn.datasets.channel_join import ChannelsJoin

from radfield3dnn.metrics import HistogramOverlapAccuracy
import multiprocessing as mp
from sys import platform

from loggers.logger import LoggerBase, TrainingSettings
from loggers.mlflow import MLFlowLogger
from loggers.wandb import WandBLogger

from tasks.base import Task


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Use 'spawn' to avoid issues with CUDA and multiprocessing on Windows
    mp.freeze_support()

    parser = argparse.ArgumentParser(add_help=True, description="Train a model on the radiation field data.")
    parser.add_argument("--model_config", type=str, required=False, help="The model configuration file.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.", required=False)
    parser.add_argument("--effective_batch_size", type=int, default=None, help="The virtual batch size for gradient accumulation. Default: No gradient accumulation.", required=False)
    parser.add_argument("--dataset_path", type=str, required=True, help="The dataset to use.")
    parser.add_argument("--dataset_type", type=str, required=False, default=None, help="The dataset type to use. (Voxelwise, Layerwise)")
    parser.add_argument("--epochs", type=int, default=25, help="The number of epochs to train.", required=False)
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers to use.", required=False)
    parser.add_argument("--normalization", type=str, default=f"{NormalizerConstructor.get_available_normalizers()[0]}", help=f"The normalization to use. Must be one of {NormalizerConstructor.get_available_normalizers()}", required=False)
    parser.add_argument("--offline", action="store_true", help="Run in offline mode.", required=False, default=False)
    parser.add_argument("--cache_dataset", action="store_true", help="Copy the dataset to the cache directory to speed up training.", required=False, default=False)
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="The directory to use for caching the dataset. Default: ./.cache", required=False)
    parser.add_argument("--logs_path", type=str, help="The path to save the logs.", required=True)
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training.", required=False, default=False)
    parser.add_argument("--augmentations", action="store_true", help="Use data augmentation.", required=False, default=False)
    parser.add_argument("--join_channels", action="store_true", help="Join the channels of the radiation field.", required=False, default=False)
    parser.add_argument("--mu_tr_file", type=str, required=False, help="The file containing the mass energy absorption coefficients for the Airkerma metric.", default=None)
    parser.add_argument("--enforce_voxel_resolution", type=int, nargs=3, required=False, default=None, help="Enforce a specific voxel resolution for the dataset. Format: x y z")
    parser.add_argument("--logger", type=str, default="wandb", help="The logger to use. Must be one of 'wandb' or 'mlflow'.", required=False)
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode, skipping batch size maximization and reducing overall system ressources usage.", required=False, default=False)
    parser.add_argument("--max_inner_batch_size", type=int, default=None, help="The maximum inner batch size (Used by voxel- or patch-wise models). Default: Search automatically.", required=False)
    parser.add_argument("--compile_model", action="store_true", help="Whether to compile the model, if supported.", required=False, default=False)
    parser.add_argument("--use_geometry", action="store_true", help="Use geometry dataset (only for Layerwise datasets).", required=False, default=False)
    parser.add_argument("--use_beam_parameters", action="store_true", help="Use beam parameters normalization.", required=False, default=False)
    parser.add_argument("--use_airkerma", action="store_true", help="Use airkerma field in dataset and model.", required=False, default=False)
    parser.add_argument("--validate_gt", action="store_true", help="Validate ground truth data at the start of each training batch.", required=False, default=False)
    parser.add_argument("--task", type=str, default="train", help="The task to run. Currently 'train' and 'tune' are supported.", required=False)
    parser.add_argument("--n_trials", type=int, default=50, help="The number of trials for hyperparameter tuning (only for 'tune' task).", required=False)
    args = parser.parse_args()

    model_config = args.model_config
    model_name = json.load(open(model_config, "r"))["model_name"]
    dataset_path = args.dataset_path
    dataset_base_name = os.path.basename(dataset_path)
    dataset_base_name = os.path.splitext(dataset_base_name)[0]

    TASK_NAME = args.task.lower()
    if TASK_NAME == "train":
        from tasks.train import TrainTask
        NETWORK_TASK: Task = TrainTask()
    elif TASK_NAME == "tune":
        from tasks.tune import  HyperparameterTuningTask
        NETWORK_TASK: Task = HyperparameterTuningTask(
            model_config=args.model_config,
            experiment_name=f"tune-{model_name}-{dataset_base_name}",
            n_trials=args.n_trials
        )
    else:
        raise ValueError(f"Task {TASK_NAME} not found.")

    # enforce separate cwd for each training run
    new_cwd = os.path.join(os.getcwd(), f"{model_name}_{dataset_base_name}")
    os.makedirs(new_cwd, exist_ok=True)
    os.chdir(new_cwd)
    
    batch_size = args.batch_size
    effective_batch_size = args.effective_batch_size
    num_workers = args.num_workers
    mixed_precision = args.mixed_precision
    dataset_type = args.dataset_type
    if not dataset_type in ["Voxelwise", "Layerwise", None]:
        raise ValueError(f"Dataset type {dataset_type} not found.")
    epochs: int = args.epochs
    OFFLINE_MODE = args.offline
    TEST_MODE = args.test_mode
    CACHE_DATASET = args.cache_dataset
    AUGMENTATIONS = args.augmentations
    normalization = args.normalization.lower()
    VOXEL_RESOLUTION = args.enforce_voxel_resolution
    MAX_INNER_BATCH_SIZE = args.max_inner_batch_size
    SHOULD_TRY_COMPILE_MODEL = args.compile_model
    USE_GEOMETRY = args.use_geometry
    USE_BEAM_PARAMETERS = args.use_beam_parameters
    USE_AIRKERMA = args.use_airkerma
    SHOULD_VALIDATE_GT = args.validate_gt

    mu_tr_file = args.mu_tr_file
    if mu_tr_file is not None and not os.path.isabs(mu_tr_file):
        mu_tr_file = os.path.join(os.path.dirname(model_config), mu_tr_file)
    if mu_tr_file is not None and not os.path.exists(mu_tr_file):
        raise FileNotFoundError(f"Mass energy absorption coefficients file {mu_tr_file} not found.")

    normalizer = NormalizerConstructor.construct_by_name(normalization)
    
    # Clean up old learning rate finder files
    print("[yellow]Cleaning up old learning rate finder files...")
    for f in os.listdir("."):
        if f.startswith(".lr_find_"):
            os.remove(os.path.join(".", f))
    
    if CACHE_DATASET:
        if not os.path.isabs(args.cache_dir):
            cache_path = os.path.dirname(__file__)
            cache_path = os.path.abspath(cache_path)
            cache_path = os.path.join(cache_path, args.cache_dir)
        else:
            cache_path = args.cache_dir
        print(f"[yellow]Using cache directory: {cache_path}")
        print("[yellow]Checking cached dataset...")

        files_to_copy = []
        files_relative_path = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                files_to_copy.append(file_path)
                rel_path = file_path.removeprefix(dataset_path).removeprefix("\\").removeprefix("/")
                files_relative_path.append(rel_path)

        if os.path.exists(cache_path):
            existing_files: list[str] = []
            existing_relative_files: list[str] = []
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    existing_files.append(file_path)
                    rel_path = file_path.removeprefix(cache_path).removeprefix("\\").removeprefix("/")
                    existing_relative_files.append(rel_path)

            # check if all files are already in the cache
            all_files_exist = True
            for file in files_relative_path:
                if file not in existing_relative_files:
                    all_files_exist = False
                    break
            
            if all_files_exist:
                print("[green]All files are already in the cache!")
            else:
                print("[yellow]Clearing cache...")
                shutil.rmtree(cache_path)

        if not os.path.exists(cache_path):
            def copy_file(file: str, src_base: str, dest_base: str, progress_dict: dict):
                dest_path = os.path.join(dest_base, file)
                src_path = os.path.join(src_base, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)
                with progress_dict["lock"]:
                    progress_dict["completed"] += 1

            print("[yellow]Caching dataset...")
            os.makedirs(cache_path)
            with Manager() as manager:
                progress_dict = manager.dict()
                progress_dict["completed"] = 0
                progress_dict["lock"] = manager.Lock()
                with Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("[progress.completed]{task.completed}/{task.total}")
                ) as progress:
                    task = progress.add_task("[cyan]Copying files...", total=len(files_relative_path))

                    def update_progress():
                        while progress_dict["completed"] < len(files_relative_path):
                            with progress_dict["lock"]:
                                progress.update(task, completed=progress_dict["completed"])
                            time.sleep(0.1)

                    from threading import Thread
                    updater_thread = Thread(target=update_progress, daemon=True)
                    updater_thread.start()

                    Parallel(n_jobs=os.cpu_count() // 2)(
                        delayed(copy_file)(file, dataset_path, cache_path, progress_dict) for file in files_relative_path
                    )
                    with progress_dict["lock"]:
                        progress.update(task, completed=progress_dict["completed"])
                print("[green]Cached dataset!")
        dataset_path = cache_path

    print("CUDA available:", torch.cuda.is_available())
    torch.set_float32_matmul_precision('high')
    #torch.autograd.set_detect_anomaly(True)

    experiment_name = f"{model_name}-{dataset_base_name}"
    logs_path = os.path.join(args.logs_path, experiment_name)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    if AUGMENTATIONS:
        dataprocessings = [
            OriginalGroundTruthPreservation(),
            #LimitedAugmentation(
            #    GaussianFluenceNoise(1e-2, repeats_per_field=1.1, error_scaled_noise=False),
            #    end_epoch=epochs//2,
            #    start_epoch=0
            #),
            #LimitedAugmentation(
            #    GaussianFluenceSmoothing(
            #        kernel_size=3,
            #        sigma=0.75,
            #        p=0.75,
            #        dataset_multiplier=1.2,
            #        random_strength=True
            #    ),
            #    end_epoch=epochs//2
            #),
            #LimitedAugmentation(
            #    SmoothingSpectra(kernel_size=3, sigma=1.0, p=0.75, dataset_multiplier=1.0),
            #    end_epoch=epochs//2
            #)
        ]
        print("[yellow]Using augmentations!")
    else:
        dataprocessings = [OriginalGroundTruthPreservation()]

    if args.join_channels:
        print("[yellow]Joining channels!")
        dataprocessings.append(ChannelsJoin())

    if USE_AIRKERMA:
        if mu_tr_file is None:
            raise ValueError("Mass energy absorption coefficients file must be provided when using airkerma.")
        print("[yellow]Using Airkerma processing!")
        airkerma_processor = AirkermaProcessing(mu_tr_file=mu_tr_file, bins=32, max_energy_eV=1.5e+5)
        dataprocessings.append(airkerma_processor)

    # create model and dataset
    model_cls = ModelConstructor.create_model_from_config(model_config, normalizer=normalizer)
    model = model_cls()
    if dataset_type is None:
        dataset_type = ModelConstructor.get_dataset_type_for_model(model_name)
    dataset_cls = None
    dataset_type_str = dataset_type
    if "Voxelwise" == dataset_type:
        dataset_type = DatasetType.Voxelwise
    elif "Layerwise" == dataset_type:
        dataset_type = DatasetType.Layerwise
    else:
        raise ValueError(f"Dataset type {dataset_type} not found.")

    datamodule = construct_datamodule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_geometry=USE_GEOMETRY,
        use_beam_parameters=USE_BEAM_PARAMETERS,
        dataprocessings=dataprocessings,
        voxel_resolution=VOXEL_RESOLUTION
    )
    FIELD_DIMENSIONS, VOXEL_SIZE_M = get_dataset_dimensions_and_voxel_size(datamodule)

    # create training logger
    logger: LoggerBase = None
    if args.logger.lower() == "wandb":
        logger = WandBLogger(project_name='radiation-field-estimator', logs_dir=os.path.join(logs_path, "wandb"), offline=OFFLINE_MODE)
    elif args.logger.lower() == "mlflow":
        logger = MLFlowLogger(project_name='radiation-field-estimator', logs_dir=os.path.join(logs_path, "mlflow"))
    else:
        raise ValueError(f"Logger {args.logger} not found.")

    logger.setup_experiment(
        experiment_name,
        TrainingSettings(
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            model_name=model_name,
            dataset_path=dataset_path,
            dataset_loading_mode=dataset_type_str,
            hyper_parameters=json.load(open(model_config, "r"))["parameters"],
            data_augmentations=[(aug.get_name(), aug.get_parameters()) for aug in dataprocessings]
        )
    )

    metrics_plotter = MetricsPlotter(
        spectra_bins=32,
        metrics={
            'global_airkerma_accuracy': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5
            ),
            'top50_airkerma_accuracy': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.5
            ),
            'top50_airkerma_accuracy_ncc': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.5,
                metric_type='ncc'
            ),
            'top90_airkerma_accuracy': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.1,
            ),
            'top90_airkerma_accuracy_ncc': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.1,
                metric_type='ncc'
            ),
            'airkerma_ssim': AirkermaSSIM(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                reduction='mean',
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
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
            ),
            'airkerma_onsphere_accuracy_radius25cm_ncc': AirkermaSphereAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                sphere_radius_m=0.25,
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                metric_type='ncc'
            ),
            'top50_airkerma_stddev': AirkermaRelDifferencesStdDev(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.5
            ),
            'top90_airkerma_stddev': AirkermaRelDifferencesStdDev(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                importance_threshold=0.1
            ),
            'airkerma_accuracy_scatter': AirkermaScatterAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5
            ),
            'airkerma_accuracy_scatter_ncc': AirkermaScatterAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                metric_type='ncc'
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
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                rel_dose_diff=0.03,
                dist_crit_mm=40.0
            ),
            'global_airkerma_gamma_index_10percent_per_4cm': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                metric_type='gpr',
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                rel_dose_diff=0.1,
                dist_crit_mm=40.0
            ),
            'global_airkerma_gamma_index_5percent_per_4cm': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                metric_type='gpr',
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                rel_dose_diff=0.05,
                dist_crit_mm=40.0
            ),
            'global_airkerma_gamma_index_10percent_per_6cm': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                metric_type='gpr',
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                rel_dose_diff=0.1,
                dist_crit_mm=60.0
            ),
            'global_airkerma_gamma_index_3percent_per_6cm': AirkermaAccuracy(
                mu_tr_file=mu_tr_file,
                spectra_bins=32,
                max_energy_eV=1.5e+5,
                metric_type='gpr',
                voxel_size_m=VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01,
                rel_dose_diff=0.03,
                dist_crit_mm=60.0
            )
        },
        voxel_resolution=VOXEL_RESOLUTION if VOXEL_RESOLUTION is not None else (50, 50, 50)
    )

    if effective_batch_size is not None and effective_batch_size < batch_size:
        raise ValueError(f"Virtual batch size {effective_batch_size} must be greater than or equal to batch size {batch_size}.")
    gradient_accumulation = effective_batch_size // batch_size if effective_batch_size is not None else None

    if TEST_MODE:
        print("[yellow]Running in test mode! using small batch size of 4096!")
        model.max_inner_batch_size = 4096 if MAX_INNER_BATCH_SIZE is None else MAX_INNER_BATCH_SIZE
    elif MAX_INNER_BATCH_SIZE is None:
        model._search_optimal_batch_size()
    else:
        model.max_inner_batch_size = MAX_INNER_BATCH_SIZE
        print(f"[yellow]Override maximum inner batch size and set it to: {model.max_inner_batch_size}")

    model._normalizer = normalizer

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        precision="16-mixed" if mixed_precision else "32-true",    # Mixed or Full precision training
        logger=logger.get_lightning_callback(),
        enable_checkpointing=(TASK_NAME == "train"),
        gradient_clip_val=1.0,
        callbacks=[
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
            RichProgressBar(),
            ModelSummary(),
            metrics_plotter
        ] + NETWORK_TASK.get_trainer_callbacks(
            logger=logger,
            epochs=epochs,
            logs_path=logs_path,
            model_name=model_name,
            mu_tr_file=mu_tr_file,
            voxel_resolution=VOXEL_RESOLUTION,
            voxel_size_m=VOXEL_SIZE_M
        ) + [
            GradientAccumulationScheduler(scheduling={0: gradient_accumulation})
        ] if gradient_accumulation is not None else [] + [
            ValidateGroundTruth()
        ] if SHOULD_VALIDATE_GT else []
    )
    logger.log_model(model)

    # Check if the model is running on Linux and compile it if so
    if SHOULD_TRY_COMPILE_MODEL and (platform == "linux" or platform == "linux2"):
        try:
            model = torch.compile(model, mode="default")
            print("[green]Model compiled successfully!")
        except Exception as e:
            print(f"[red]Failed to compile model: {e}. Continuing without compilation.")

    if DatasetType.Voxelwise == dataset_type:
        batch_size = model.max_inner_batch_size // 8
        datamodule.batch_size = batch_size
        print(f"[yellow]Overriding batch size to {batch_size} for Voxelwise dataset.")

    NETWORK_TASK.run_task(trainer, model, datamodule)
    logger.finalize_logging()

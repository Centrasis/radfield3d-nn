import os
import json
import argparse
import multiprocessing as mp
from sys import platform

import torch
# Use the file_system sharing strategy for DataLoader worker tensors. The default
# file_descriptor strategy passes one fd per shared tensor and, across many runs /
# large layerwise batches, exhausts fds — surfacing as "ConnectionResetError: [Errno
# 104]" in multiprocessing.resource_sharer + "leaked semaphore" warnings at worker
# spawn. file_system uses /dev/shm-backed names instead and is robust to that.
torch.multiprocessing.set_sharing_strategy("file_system")
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, RichProgressBar,
    DeviceStatsMonitor, GradientAccumulationScheduler,
)
from rich import print

from radfield3dnn.models import ModelConstructor
from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from radfield3dnn.datasets import DatasetType, OriginalGroundTruthPreservation, construct_datamodule, get_dataset_dimensions_and_voxel_size
from radfield3dnn.datasets.channel_join import ChannelsJoin
from radfield3dnn.metrics.airkerma_accuracy import (
    AirkermaAccuracy, AirkermaRelDifferencesStdDev,
    AirkermaSphereAccuracy, AirkermaScatterAccuracy, AirkermaAccuracyEnergyWeighted,
    AirkermaSupervoxelScatterAccuracy, AirkermaBeamAccuracy,
)
from radfield3dnn.metrics.ssim import AirkermaSSIM
from radfield3dnn.metrics import HistogramOverlapAccuracy
from radfield3dnn.metrics.factory import build_airkerma_metrics
from radfield3dnn.preprocessing.airkerma import AirkermaProcessing

from callbacks.validate_gt import ValidateGroundTruth
from callbacks.metrics_plotter import MetricsPlotter
from loggers.logger import LoggerBase, TrainingSettings
from loggers.mlflow import MLFlowLogger
from loggers.wandb import WandBLogger
from tasks.base import Task


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_logger_config(train_cfg: dict) -> dict:
    """Normalize the experiment-tracking config into {type, project_name, run_name, offline,
    mlflow_tracking_uri}.

    Canonical form is a SUB-SECTION under ``training``::

        training:
          logger:
            type: mlflow                     # mlflow | wandb
            mlflow_tracking_uri: /data/mlflow   # mlflow only
            project_name: radiation-field-estimator
            run_name: tpbrfnet-fourier-ds04
            offline: false

    The legacy flat form (``training: logger: wandb`` plus sibling ``project_name`` /
    ``run_name`` / ``offline`` / ``mlflow_tracking_uri`` keys) is still accepted so existing
    configs keep running; a flat key is only consulted when the section omits it."""
    section = train_cfg.get("logger", None)
    if isinstance(section, dict):
        cfg = dict(section)
    else:
        # str -> the legacy `logger: wandb` scalar; None -> logger block absent entirely.
        cfg = {"type": section} if section is not None else {}
    pick = lambda key, default=None: cfg.get(key, train_cfg.get(key, default))
    return {
        "type": str(pick("type", "wandb")).lower(),
        "project_name": pick("project_name", "radiation-field-estimator"),
        "run_name": pick("run_name", None),
        "offline": bool(pick("offline", False)),
        "mlflow_tracking_uri": pick("mlflow_tracking_uri", None),
    }


def collect_cacheable_files(dataset_path: str, min_age_s: float, now: float = None) -> tuple[list[str], int]:
    """Walk the dataset and return (relative paths to cache, number skipped as too young).

    Guard for datasets that are still GROWING: a field whose file was modified less than
    ``min_age_s`` seconds ago may still be open in a running simulation writer — copying it would
    poison the cache with a half-written .rf3. Such files are skipped for this run (0 disables)."""
    import time as _time
    if now is None:
        now = _time.time()
    files_rel, skipped_young = [], 0
    for root, _, files in os.walk(dataset_path):
        for f in files:
            full = os.path.join(root, f)
            if min_age_s > 0 and (now - os.path.getmtime(full)) < min_age_s:
                skipped_young += 1
                continue
            files_rel.append(full.removeprefix(dataset_path).lstrip("/\\"))
    return files_rel, skipped_young


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()

    # Hang diagnosis on clusters: `kill -USR1 <pid>` dumps every thread's stack to stderr
    # without killing the process — if a run ever wedges again, this names the blocked frame.
    try:
        import faulthandler, signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Train a radfield3d neural network model.")
    parser.add_argument("config", type=str, help="YAML training configuration file.")
    parser.add_argument("--task", type=str, default="train", choices=["train", "tune"],
                        help="Task to run: 'train' or 'tune'.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to the dataset. Optional if the YAML sets `dataset: path:`; "
                             "the CLI value wins when both are given.")
    parser.add_argument("--logs_path", type=str, required=True, help="Path to save logs and checkpoints.")
    parser.add_argument("--mu_tr_file", type=str, default=None,
                        help="Mass energy absorption coefficients file for Airkerma metric.")
    parser.add_argument("--seed", type=int, default=torch.randint(0, 2**31 - 1, (1,)).item(),
                        help="Random seed. Defaults to a fresh random seed each run (persisted to the run config).")
    args = parser.parse_args()

    # Apply the global seed to every RNG so the run is reproducible and follows this single seed.
    pl.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentations", {})
    tune_cfg = cfg.get("tune", {})

    # ── Dataset path ──────────────────────────────────────────────────────────
    # The YAML `dataset: path:` makes the config self-contained; --dataset_path overrides it
    # (same dataset config, different machine/mount).
    dataset_path = args.dataset_path or ds_cfg.get("path")
    if not dataset_path:
        raise SystemExit("Config error: no dataset given — set `dataset: path:` in the YAML "
                         "or pass --dataset_path.")
    # The run chdirs into its logs working dir below, so pin relative paths NOW: dataset_path
    # against the invocation cwd, definition_file against the YAML's own directory (so the config
    # folder stays relocatable as a unit).
    dataset_path = os.path.abspath(dataset_path)
    dataset_definition_file = ds_cfg.get("definition_file")
    if dataset_definition_file and not os.path.isabs(dataset_definition_file):
        dataset_definition_file = os.path.join(os.path.dirname(os.path.abspath(args.config)), dataset_definition_file)

    # ── Model config ──────────────────────────────────────────────────────────
    model_config = train_cfg["model_config"]
    with open(model_config) as f:
        raw_model_cfg = json.load(f)
    model_name = raw_model_cfg["model_name"]

    # ── voxel_supersampling compatibility gate ────────────────────────────────
    # K>1 replicates every kept voxel K times in forward2volume and mean-reduces the
    # predictions; that path only exists in the pure-Python whole-field pipeline.
    _tp = raw_model_cfg.get("parameters", {}).get("training_params", {}) or {}
    _k = int(_tp.get("voxel_supersampling", 1))
    if "randomize_voxel_location_in_training" in _tp:
        raise SystemExit("Config error: training_params.randomize_voxel_location_in_training was "
                         "removed — use voxel_supersampling (1 = old False; K>1 = K-sample voxel-mean).")
    if _k > 1:
        if model_name.endswith("CPP"):
            raise SystemExit(f"Config error: voxel_supersampling={_k} is not supported for fused "
                             f"tcnn models ({model_name}); the fp16 fused path is deprecated and "
                             f"untested with mean-reduction. Use the pure-Python model or K=1.")
        if cfg.get("dataset", {}).get("type", "Layerwise") == "Voxelwise":
            raise SystemExit(f"Config error: voxel_supersampling={_k} requires the Layerwise "
                             f"(whole-field) pipeline; dataset.type=Voxelwise bypasses forward2volume.")
        if _k > 8:
            print(f"[yellow]voxel_supersampling={_k}: step cost scales ~linearly with K; >8 is rarely worth it.")
        print(f"[green]Voxel supersampling ON: K={_k} in-voxel samples per voxel, per-voxel mean supervision.")

    # ── Task setup ────────────────────────────────────────────────────────────
    dataset_base = os.path.splitext(os.path.basename(dataset_path))[0]
    # Run name: defaults to "<model>-<dataset>", but `training: logger: run_name:` overrides it so
    # ablation runs can be named by their VERSION (e.g. "concat-d4") in a dedicated project_name.
    logger_cfg = resolve_logger_config(train_cfg)
    experiment_name = logger_cfg["run_name"] or f"{model_name}-{dataset_base}"

    if args.task == "train":
        from tasks.train import TrainTask
        NETWORK_TASK: Task = TrainTask()
    else:
        from tasks.tune import HyperparameterTuningTask
        NETWORK_TASK: Task = HyperparameterTuningTask(
            model_config=model_config,
            experiment_name=f"tune-{experiment_name}",
            n_trials=tune_cfg.get("n_trials", 50),
        )

    # ── Working directory per run ─────────────────────────────────────────────
    # Lives UNDER the logs folder (never the repo), so a run only writes into --logs_path.
    new_cwd = os.path.join(os.path.abspath(args.logs_path), f"{model_name}_{dataset_base}")
    os.makedirs(new_cwd, exist_ok=True)
    os.chdir(new_cwd)

    # ── Dataset caching ───────────────────────────────────────────────────────
    # (dataset_path already resolved from --dataset_path / `dataset: path:` above)
    if ds_cfg.get("cache", False):
        import shutil, time
        from joblib import Parallel, delayed
        from multiprocessing import Manager
        from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
        from threading import Thread

        cache_dir = ds_cfg.get("cache_dir", "./.cache")
        cache_path = cache_dir if os.path.isabs(cache_dir) else os.path.abspath(cache_dir)

        # `dataset: cache_min_age_s` (default 60, 0 = off): don't copy fields younger than this —
        # they may still be being written by a running simulation (see collect_cacheable_files).
        min_age_s = float(ds_cfg.get("cache_min_age_s", 60))
        files_rel, skipped_young = collect_cacheable_files(dataset_path, min_age_s)
        if skipped_young:
            print(f"[yellow]Dataset cache: skipped {skipped_young} file(s) modified less than "
                  f"{min_age_s:.0f}s ago (possibly still under simulation); they are excluded "
                  f"from this run.")

        if os.path.exists(cache_path):
            existing = set()
            for root, _, files in os.walk(cache_path):
                for f in files:
                    existing.add(os.path.join(root, f).removeprefix(cache_path).lstrip("/\\"))
            if not all(f in existing for f in files_rel):
                shutil.rmtree(cache_path)

        if not os.path.exists(cache_path):
            print("[yellow]Caching dataset…")
            os.makedirs(cache_path)

            def _copy(rel, src, dst, prog):
                dest = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(os.path.join(src, rel), dest)
                with prog["lock"]:
                    prog["done"] += 1

            with Manager() as mgr:
                prog = mgr.dict(done=0, lock=mgr.Lock())
                with Progress(TextColumn("{task.description}"), BarColumn(),
                               TimeElapsedColumn(), TimeRemainingColumn()) as bar:
                    t = bar.add_task("[cyan]Copying…", total=len(files_rel))
                    Thread(target=lambda: [bar.update(t, completed=prog["done"]) or
                                           __import__("time").sleep(0.1)
                                           for _ in iter(lambda: prog["done"] < len(files_rel), False)],
                           daemon=True).start()
                    Parallel(n_jobs=os.cpu_count() // 2)(
                        delayed(_copy)(f, dataset_path, cache_path, prog) for f in files_rel
                    )
        dataset_path = cache_path
        print("[green]Dataset cached.")

    # ── Normalizer (from model JSON — the source of truth; ModelConstructor injects it) ──
    norm_name = raw_model_cfg.get("parameters", {}).get("normalizer", "linear0_1")

    # ── Data processings ──────────────────────────────────────────────────────
    # MC floor noise removal runs AFTER OriginalGroundTruthPreservation snapshots the GT, so
    # the cut is TRAINING-ONLY: the loss sees the cleaned field (noise floor removed) while
    # the air-kerma metric scores against the ORIGINAL uncut GT.
    #
    # Why the order matters: the air-kerma accuracy is SMAPE-based and the `scatter` metric
    # uses importance_threshold=0, i.e. it scores EVERY voxel. If MCFloorCut zeroes a voxel
    # in the GT the metric sees, then SMAPE(small_pred, 0) = 2 (worst case) for the implicit
    # MLP's small-but-nonzero prediction there — turning every cut voxel into an accuracy-0
    # landmine and collapsing scatter accuracy. Snapshotting the uncut GT first keeps the
    # metric comparable to the no-MC-floor baseline.
    dataprocessings = [OriginalGroundTruthPreservation()]

    # Multi-resolution resampling (training only) — MUST run before any -inf-masking stage
    # (floor cut / ROI sampling). Teaches the scale axis for IPE encoders (auto_region_width).
    mr_cfg = aug_cfg.get("multi_resolution", {})
    if mr_cfg.get("enabled", False):
        from radfield3dnn.preprocessing.augmentations.multi_resolution import MultiResolutionResampling
        _scales = tuple(mr_cfg.get("scales", (0.5, 0.75, 1.0, 1.5, 2.0)))
        dataprocessings.append(MultiResolutionResampling(scales=_scales))
        print(f"[green]Multi-resolution resampling ON: scales {_scales} "
              f"(area-pool down = exact labels; trilinear up = approximate labels).")
    mc_floor = aug_cfg.get("mc_floor_cut", None)
    if mc_floor:
        from radfield3dnn.datasets.mc_floor_cut import MCFloorCut
        # Either a single scalar (both channels) or a per-channel mapping
        # {scatter: <rel>, direct: <rel>} — scatter is diffuse/low-DR (cut gently,
        # keeps its spatial info) while the direct beam is sharp/high-DR (cut harder
        # to strip the MC leakage floor). See radfield3dnn/datasets/mc_floor_cut.py.
        if isinstance(mc_floor, dict) and (mc_floor.get("mask", False) or str(mc_floor.get("mode", "")).lower() == "neginf"):
            # MASKING mode: set the shared FLOOR ROI to -inf (not 0), TRAINING-ONLY, join-safe.
            # Pairs with the ROIbasedSampler (floor_as_zero) which re-injects a few floor zeros.
            from radfield3dnn.roi import BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
            b_rel = float(mc_floor.get("beam_rel", BEAM_REL_DEFAULT))
            s_lo = float(mc_floor.get("scatter_lo", SCATTER_LO_DEFAULT))
            dataprocessings.append(MCFloorCut(as_neginf=True, beam_rel=b_rel, scatter_lo=s_lo))
            print(f"[green]MC floor cut (training target only, -inf MASK): floor ROI (NOT beam & joined < {s_lo:.0e}·joined_max) → -inf in both channels; validation sees the full field.")
        elif isinstance(mc_floor, dict) and mc_floor.get("use_error", False):
            et = float(mc_floor.get("error_threshold", 0.5))
            dataprocessings.append(MCFloorCut(use_error=True, error_threshold=et))
            print(f"[green]MC floor cut (training target only, ERROR-based): zeroing per-channel voxels with MC error >= {et} (data-adaptive; joined keeps the union of confident voxels).")
        elif isinstance(mc_floor, dict):
            s_rel = float(mc_floor.get("scatter", 1e-4))
            d_rel = float(mc_floor.get("direct", 1e-2))
            dataprocessings.append(MCFloorCut(scatter_rel=s_rel, direct_rel=d_rel))
            print(f"[green]MC floor cut (training target only): scatter < {s_rel:.0e}, direct < {d_rel:.0e} of per-channel per-field peak.")
        else:
            dataprocessings.append(MCFloorCut(rel_threshold=float(mc_floor)))
            print(f"[green]MC floor cut enabled (training target only): zeroing GT voxels < {float(mc_floor):.0e} of per-field peak.")
    epochs = train_cfg.get("epochs", 25)

    if aug_cfg.get("enabled", False):
        from radfield3dnn.preprocessing.augmentations.noise import GaussianFluenceNoise
        from radfield3dnn.preprocessing.augmentations.smoothing import GaussianFluenceSmoothing
        from radfield3dnn.preprocessing.augmentations.augmentation_limit import LimitedAugmentation
        dataprocessings += [
            LimitedAugmentation(GaussianFluenceNoise(1e-2, repeats_per_field=1.1, error_scaled_noise=False),
                                end_epoch=epochs // 2),
            LimitedAugmentation(GaussianFluenceSmoothing(kernel_size=3, sigma=0.75, p=0.75,
                                                          dataset_multiplier=1.2, random_strength=True),
                                end_epoch=epochs // 2),
        ]

    if aug_cfg.get("join_channels", False):
        # Join scatter + direct into a single flux target.
        dataprocessings.append(ChannelsJoin())

    if aug_cfg.get("smooth_spectra", False):
        from radfield3dnn.preprocessing.augmentations.smooth_spectra import SmoothingSpectra
        from radfield3dnn.preprocessing.augmentations.augmentation_limit import LimitedAugmentation
        dataprocessings.append(LimitedAugmentation(
            SmoothingSpectra(kernel_size=3, sigma=1.0, p=0.75, dataset_multiplier=1.0),
            end_epoch=epochs,
        ))

    is_cfg = aug_cfg.get("importance_sampling", {})
    if is_cfg.get("enabled", False):
        from radfield3dnn.preprocessing.augmentations.augmentation_limit import LimitedAugmentation
        is_end_epoch = is_cfg.get("end_epoch", epochs // 2)
        # `method`: "error" (default) = ErrorbasedImportanceSampler (drop high-MC-error voxels);
        #           "roi" = ROIbasedSampler (keep all beam, sample scatter relative to beam count,
        #                   sample a capped floor — matches the air-kerma scatter ROI + TwoROIGammaLoss);
        #           "roi_full" = FullScatterROISampler (keep ALL beam + ALL scatter, sip a few % of floor).
        method = str(is_cfg.get("method", "error")).lower()
        if method == "roi":
            from radfield3dnn.preprocessing.augmentations.roi_sampling import ROIbasedSampler
            from radfield3dnn.roi import BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
            sampler = ROIbasedSampler(
                beam_rel=is_cfg.get("beam_rel", BEAM_REL_DEFAULT),
                scatter_lo=is_cfg.get("scatter_lo", SCATTER_LO_DEFAULT),
                beam_keep_ratio=is_cfg.get("beam_keep_ratio", 1.0),
                scatter_ratio=is_cfg.get("scatter_ratio", 2.0),
                floor_ratio=is_cfg.get("floor_ratio", 1.0),
                field_multiplier=is_cfg.get("field_multiplier", 3.0),
                floor_as_zero=is_cfg.get("floor_as_zero", True),
                scatter_ratio_end=is_cfg.get("scatter_ratio_end", None),
                schedule_switch=is_cfg.get("schedule_switch", 0.8),
            )
            _sr_end = is_cfg.get("scatter_ratio_end", None)
            _sched = "" if _sr_end is None else (f" → {_sr_end}×beam scatter for the last "
                     f"{(1.0 - is_cfg.get('schedule_switch', 0.8)) * 100:.0f}% of epochs (fine-tune)")
            print(f"[green]ROI-based voxel sampling: keep {is_cfg.get('beam_keep_ratio',1.0)}·beam "
                  f"(>= {is_cfg.get('beam_rel',BEAM_REL_DEFAULT):.0e}·direct_max) + "
                  f"{is_cfg.get('scatter_ratio',2.0)}×beam scatter + {is_cfg.get('floor_ratio',1.0)}×beam floor"
                  f"{' (floor→0)' if is_cfg.get('floor_as_zero', True) else ''}{_sched}; "
                  f"field ×{is_cfg.get('field_multiplier',3.0)}.")
        elif method == "roi_full":
            from radfield3dnn.preprocessing.augmentations.roi_sampling import FullScatterROISampler
            from radfield3dnn.roi import BEAM_REL_DEFAULT
            # coerce: YAML parses "1e-4" (no decimal point) as a string, so float() it here.
            _scatter_lo = float(is_cfg.get("scatter_lo", 1e-4))
            _floor_keep = float(is_cfg.get("floor_keep_ratio", 0.03))
            _field_mult = float(is_cfg.get("field_multiplier", 2.0))
            _floor_val = float(is_cfg.get("floor_value", 1e-8))
            sampler = FullScatterROISampler(
                beam_rel=float(is_cfg.get("beam_rel", BEAM_REL_DEFAULT)),
                scatter_lo=_scatter_lo,
                floor_keep_ratio=_floor_keep,
                field_multiplier=_field_mult,
                floor_as_zero=is_cfg.get("floor_as_zero", True),
                floor_value=_floor_val,
            )
            print(f"[green]ROI-full voxel sampling: keep ALL beam + ALL scatter "
                  f"(>= {_scatter_lo:.0e}·joined_max) + {_floor_keep*100:.0f}% of the floor"
                  f"{f' (floor→{_floor_val:.0e})' if is_cfg.get('floor_as_zero', True) else ''}; field ×{_field_mult}.")
        else:
            # ErrorbasedImportanceSampler: drop unreliable high-MC-error voxels as a WARMUP, then
            # switch off for fine-tuning (the background is a real target). `max_drop_chance` anneals
            # from its start value to `max_drop_chance_end` across the active window.
            from radfield3dnn.preprocessing.augmentations.importance_sampling import ErrorbasedImportanceSampler
            sampler = ErrorbasedImportanceSampler(
                max_drop_chance=is_cfg.get("max_drop_chance", 0.9),
                max_drop_chance_end=is_cfg.get("max_drop_chance_end", 0.3),
                high_fluence_keep_threshold=is_cfg.get("keep_flux_threshold", 0.8),
            )
        dataprocessings.append(LimitedAugmentation(sampler, end_epoch=is_end_epoch))

    mu_tr_file = args.mu_tr_file
    if mu_tr_file and not os.path.isabs(mu_tr_file):
        mu_tr_file = os.path.join(os.path.dirname(model_config), mu_tr_file)
    if mu_tr_file and not os.path.exists(mu_tr_file):
        raise FileNotFoundError(f"mu_tr_file not found: {mu_tr_file}")

    if ds_cfg.get("use_airkerma", False):
        if not mu_tr_file:
            raise ValueError("--mu_tr_file required when use_airkerma=true")
        dataprocessings.append(AirkermaProcessing(mu_tr_file=mu_tr_file, bins=32, max_energy_eV=1.5e+5))

    # ── Geometry-covered voxel exclusion (DEFAULT ON) ─────────────────────────
    # Voxels inside the phantom are excluded from scoring by the simulation — their GT holds no
    # statistics, so they are dropped (-inf) from the TRAINING target. Appended AFTER the samplers
    # so a floor-zero-injected geometry voxel still ends dropped, not "confidently zero".
    # `augmentations: exclude_geometry_voxels: false` disables; `geometry_threshold` tunes the
    # density cutoff (0.0 = any occupancy counts, also matches the binary mask).
    exclude_geometry = aug_cfg.get("exclude_geometry_voxels", True)
    if exclude_geometry:
        from radfield3dnn.preprocessing.augmentations.geometry_exclusion import GeometryVoxelExclusion
        dataprocessings.append(GeometryVoxelExclusion(threshold=float(aug_cfg.get("geometry_threshold", 0.0))))
        print("[green]Geometry-covered voxels are excluded from the training target "
              "(augmentations: exclude_geometry_voxels).")

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision('high')

    import tempfile
    # Inject the resolved normalizer back into the config for model construction
    full_cfg = dict(raw_model_cfg)
    full_cfg.setdefault("parameters", {})["normalizer"] = norm_name  # keep as string for ModelConstructor

    precision = train_cfg.get("precision", "fp32")
    if precision == "fp16":
        full_cfg["parameters"]["precision"] = "fp16"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(full_cfg, tmp)
        tmp_path = tmp.name

    model_cls = ModelConstructor.create_model_from_config(tmp_path)
    os.unlink(tmp_path)
    model = model_cls().cuda()
    # Reproducibility: `training: lr_finder: false` skips the per-seed LR sweep
    # and trains at the model's configured LR (a fixed LR removes a variance source).
    if not train_cfg.get("lr_finder", True):
        model._use_lr_finder = False
    # MTL ablation knob: `training: mtl_balancing: false` combines the flux/spectrum task losses
    # by a plain equal-weight sum (no DB-MTL); default true keeps full DB-MTL balancing. Lets the
    # same config be scored with and without MTL loss weighting.
    model._use_mtl = train_cfg.get("mtl_balancing", True)
    if not model._use_mtl:
        print("[yellow]MTL loss weighting DISABLED (equal-weight task-loss sum).[/yellow]")
    # Step-2 knob: `training: mtl_gradient_balancing: false` keeps DB-MTL's loss-SCALE balancing
    # (step 1, scale-invariant, single backward — the right tool for a large task-magnitude gap,
    # e.g. SMAPEBalanced flux ~1.0 vs HistogramLoss spectrum ~0.025 = 40x) while skipping the
    # per-task trunk-gradient balancing (step 2, N_tasks extra full backwards on refresh steps —
    # also a peak-memory risk on full-volume training).
    model._mtl_gradient_balancing = train_cfg.get("mtl_gradient_balancing", True)
    if model._use_mtl and not model._mtl_gradient_balancing:
        print("[yellow]DB-MTL step 1 only (loss-scale balancing); gradient-magnitude balancing off.[/yellow]")
    # Fixed spectrum-task multiplier (`training: spectrum_loss_weight`, default 1.0) — the safe
    # (non-adaptive) fix for the flux/spectrum magnitude gap with self-weighted flux losses.
    model._spectrum_loss_weight = float(train_cfg.get("spectrum_loss_weight", 1.0))
    if model._spectrum_loss_weight != 1.0:
        print(f"[yellow]Fixed spectrum loss weight: x{model._spectrum_loss_weight}.[/yellow]")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_type_str = ds_cfg.get("type", None)
    if dataset_type_str is None:
        dataset_type_str = ModelConstructor.get_dataset_type_for_model(model_name)

    voxel_resolution = ds_cfg.get("voxel_resolution", None)
    if voxel_resolution is not None:
        voxel_resolution = tuple(voxel_resolution)

    datamodule = construct_datamodule(
        dataset_path=dataset_path,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        use_geometry=ds_cfg.get("use_geometry", False),
        use_beam_parameters=ds_cfg.get("use_beam_parameters", False),
        dataprocessings=dataprocessings,
        voxel_resolution=voxel_resolution,
        prefetch_to_device=train_cfg.get("prefetch_to_device", True),
        max_fields=ds_cfg.get("max_fields", None),
        cache_to_ram=ds_cfg.get("cache_to_ram", False),
        cache_ram_gb=ds_cfg.get("cache_ram_gb", None),
        use_translation=ds_cfg.get("use_translation", False),
        # `dataset: definition_file:` — path to the dataset definition JSON the dataset was
        # generated with; source of the beam-parameter + patient-translation normalization ranges.
        # (resolved above relative to the YAML config's directory)
        dataset_definition=dataset_definition_file,
        # `dataset: scan_translation_metadata:` (default true) — check every field for the
        # patient_translation entry before training; `skip_fields_without_translation:` trains on
        # the fields that have it instead of failing.
        scan_translation_metadata=ds_cfg.get("scan_translation_metadata", True),
        skip_fields_without_translation=ds_cfg.get("skip_fields_without_translation", False),
        # attach the binary geometry-occupancy mask so GeometryVoxelExclusion has its channel
        # (no-op for fields without a geometry channel; use_geometry attaches it anyway)
        attach_geometry_mask=exclude_geometry and not ds_cfg.get("use_geometry", False),
    )
    _, VOXEL_SIZE_M = get_dataset_dimensions_and_voxel_size(datamodule)

    # Flux-head output-bias init is decided by the flux activation itself (its `init_bias`
    # property), inside the model's apply_weights_init — no external/data-adaptive override.

    dataset_type = DatasetType.Voxelwise if dataset_type_str == "Voxelwise" else DatasetType.Layerwise

    # ── Logger ────────────────────────────────────────────────────────────────
    logs_path = os.path.join(args.logs_path, experiment_name)
    os.makedirs(logs_path, exist_ok=True)

    # (logger_cfg resolved above from the `training: logger:` sub-section; see
    # resolve_logger_config for the accepted forms.)
    logger_name = logger_cfg["type"]
    # Experiment-tracking project. Override per-run via `training: logger: project_name:` to keep
    # separate runs (e.g. ablations / loss changes) in their own project.
    project_name = logger_cfg["project_name"]
    if logger_name == "wandb":
        logger: LoggerBase = WandBLogger(
            project_name=project_name,
            logs_dir=os.path.join(logs_path, "wandb"),
            offline=logger_cfg["offline"],
        )
    elif logger_name == "mlflow":
        # `training: logger: mlflow_tracking_uri:` — where MLflow tracks to: a local/UNC
        # directory, a file:// URI, or an http(s)://... tracking server. Default:
        # <logs_path>/<run>/mlflow.
        logger: LoggerBase = MLFlowLogger(
            project_name=project_name,
            logs_dir=logger_cfg["mlflow_tracking_uri"] or os.path.join(logs_path, "mlflow"),
        )
    else:
        raise ValueError(f"Unknown logger type: {logger_name!r} (training: logger: type:). "
                         f"Valid: 'mlflow', 'wandb'.")

    logger.setup_experiment(experiment_name, TrainingSettings(
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        epochs=epochs,
        model_name=model_name,
        dataset_path=dataset_path,
        dataset_loading_mode=dataset_type_str,
        hyper_parameters=raw_model_cfg.get("parameters", {}),
        data_augmentations=[(aug.get_name(), aug.get_parameters()) for aug in dataprocessings],
    ))

    # ── Metrics plotter ───────────────────────────────────────────────────────
    voxel_res_for_plot = tuple(voxel_resolution) if voxel_resolution else (50, 50, 50)
    vx = VOXEL_SIZE_M if VOXEL_SIZE_M > 0.0 else 0.01
    # Joined datasets (single 'radiation' channel) can't compute the scatter/beam-isolation metrics.
    import glob as _glob
    from RadFiled3D.utils import FieldStore as _FieldStore
    _probe_files = sorted(_glob.glob(os.path.join(dataset_path, "fields", "*.rf3")))
    _joined = bool(_probe_files) and not all(
        c in _FieldStore.load(_probe_files[0]).get_channel_names() for c in ("scatter_field", "direct_beam"))
    if _joined:
        print("[yellow]Joined dataset (single 'radiation' channel) — scatter/beam-isolation metrics disabled.")
    metrics_plotter = MetricsPlotter(
        spectra_bins=32,
        metrics=build_airkerma_metrics(mu_tr_file, vx, joined=_joined),
        voxel_resolution=voxel_res_for_plot,
    )

    # ── Batch size / compile ───────────────────────────────────────────────────
    test_mode = train_cfg.get("test_mode", False)
    max_inner = train_cfg.get("max_inner_batch_size", None)

    if test_mode:
        model.max_inner_batch_size = max_inner or 4096
    elif max_inner is None:
        model._search_optimal_batch_size()
    else:
        print(f"[yellow] Override max_inner_batch_size to {max_inner}")
        model.max_inner_batch_size = max_inner

    if train_cfg.get("compile_model", False) and platform in ("linux", "linux2"):
        try:
            model = torch.compile(model, mode="default")
            print("[green]Model compiled.")
        except Exception as e:
            print(f"[red]Compile failed: {e}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    batch_size = train_cfg.get("batch_size", 32)
    effective_batch_size = train_cfg.get("effective_batch_size", None)
    mixed_precision = train_cfg.get("mixed_precision", False)

    if effective_batch_size and effective_batch_size < batch_size:
        raise ValueError(f"effective_batch_size ({effective_batch_size}) < batch_size ({batch_size})")
    grad_accum = effective_batch_size // batch_size if effective_batch_size else None

    # ── VRAM preflight ────────────────────────────────────────────────────────
    # A whole-field step keeps the activations of EVERY voxel of every field in the batch alive for
    # the backward pass, so the requirement scales with batch_size x voxels. Estimate it up front:
    # an OOM here surfaces deep inside Lightning and is then MASKED by a teardown error
    # ("Please call `iter(combined_loader)` first."), which says nothing about the real cause.
    if hasattr(model, "estimate_step_activation_bytes") and torch.cuda.is_available():
        _preflight_counts = voxel_resolution if voxel_resolution is not None else None
        if _preflight_counts is None:
            try:
                _dim, _vx = get_dataset_dimensions_and_voxel_size(datamodule)
                _preflight_counts = tuple(int(round(d / _vx)) for d in _dim)
            except Exception:
                _preflight_counts = None
        if _preflight_counts is not None:
            try:
                need = model.estimate_step_activation_bytes(batch_size, _preflight_counts)
                total = torch.cuda.get_device_properties(0).total_memory
                voxels = 1
                for _c in _preflight_counts:
                    voxels *= int(_c)
                print(f"[blue]VRAM preflight: batch_size {batch_size} x {voxels:,} voxels "
                      f"≈ {need / 2**30:.1f} GiB retained activations vs {total / 2**30:.1f} GiB on GPU 0.[/blue]")
                if need > 0.8 * total:
                    fits = max(1, int(0.8 * total / max(need / max(batch_size, 1), 1)))
                    print(f"[red]VRAM preflight: this step almost certainly runs out of memory.[/red]\n"
                          f"[yellow]  Lower `training: batch_size` to ~{fits} (and set "
                          f"`effective_batch_size: {batch_size}` to keep the same effective batch via "
                          f"gradient accumulation), reduce `dataset: voxel_resolution`, or enable "
                          f"`augmentations: importance_sampling` — it drops most voxels before the "
                          f"forward. NOTE: `max_inner_batch_size` does NOT help; every chunk's "
                          f"activations are retained until backward.[/yellow]")
            except Exception as _e:
                print(f"[yellow]VRAM preflight skipped: {_e}[/yellow]")

    callbacks = [
        LearningRateMonitor("epoch"),
        DeviceStatsMonitor(),
        RichProgressBar(),
        ModelSummary(),
        metrics_plotter,
    ] + NETWORK_TASK.get_trainer_callbacks(
        logger=logger, epochs=epochs, logs_path=logs_path,
        model_name=model_name, mu_tr_file=mu_tr_file,
        voxel_resolution=voxel_resolution, voxel_size_m=VOXEL_SIZE_M,
        dataset_path=dataset_path,
    )

    # Listed per-step debug view of training (inputs → outputs → per-ROI loss terms → DB-MTL
    # weights), printed and appended to <logs>/debug_probe.log. `training: debug_probe: true`.
    if train_cfg.get("debug_probe", False):
        from callbacks.debug_probe import TrainingDebugProbe
        callbacks.append(TrainingDebugProbe(
            every_n_steps=int(train_cfg.get("debug_probe_every", 50)),
            log_path=os.path.join(logs_path, "debug_probe.log"),
        ))
        print(f"[green]TrainingDebugProbe enabled (every {train_cfg.get('debug_probe_every', 50)} steps).")

    if grad_accum:
        callbacks.append(GradientAccumulationScheduler(scheduling={0: grad_accum}))
    if train_cfg.get("weight_ema", False):
        # Evaluate an EMA of the recent weights instead of the noisy last-step
        # weights — reduces seed-to-seed variance (see callbacks/ema.py).
        from callbacks.ema import WeightEMA
        callbacks.append(WeightEMA(decay=float(train_cfg.get("weight_ema_decay", 0.999))))
    if train_cfg.get("validate_gt", False):
        callbacks.append(ValidateGroundTruth())

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=50,
        accelerator="gpu",
        devices=1,
        # Optional fast-iteration caps (default 1.0 = full). Useful for quickly
        # smoke-testing a new model end-to-end before a full run.
        limit_train_batches=train_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
        # Validation runs the full-volume assembly + the heavy HTML plotters, so on long runs
        # validating every epoch dominates wall-clock (and can wedge the online media upload).
        # Default 1 (every epoch); set check_val_every_n_epoch in the YAML to throttle it.
        check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
        num_sanity_val_steps=0,
        precision="16-mixed" if mixed_precision else "32-true",
        profiler=os.environ.get("RF_PROFILER", "simple") or None,  # ON by default; set RF_PROFILER=advanced for op-level, or "" to disable
        logger=logger.get_lightning_callback(),
        enable_checkpointing=(args.task == "train"),
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    logger.log_model(model)

    if dataset_type == DatasetType.Voxelwise:
        datamodule.batch_size = model.max_inner_batch_size // 8

    NETWORK_TASK.run_task(trainer, model, datamodule)
    logger.finalize_logging()

    # ── Deterministic shutdown ────────────────────────────────────────────────
    # The dataloaders run persistent workers wrapped in the CUDA-stream prefetcher, which
    # Lightning's teardown cannot see — left alone, their cleanup at interpreter exit can
    # deadlock against the collapsing CUDA context and the run never terminates (observed as a
    # dead-but-running SLURM step after the final test table). Shut everything down explicitly,
    # then hard-exit as a last resort: everything worth persisting (checkpoints, RF3M package,
    # logger run) is already flushed at this point.
    print("[green]Run complete — shutting down dataloaders and CUDA.[/green]")
    try:
        datamodule.shutdown_dataloaders()
    except Exception as _e:
        print(f"[yellow]Dataloader shutdown: {_e}[/yellow]")
    try:
        import gc
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as _e:
        print(f"[yellow]CUDA cleanup: {_e}[/yellow]")
    if train_cfg.get("hard_exit", True):
        # `training: hard_exit: false` opts out (e.g. when profiling interpreter shutdown).
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(0)

# RadField3D-NN

[![Build](https://github.com/Centrasis/radfield3d-nn/actions/workflows/build.yml/badge.svg)](https://github.com/Centrasis/radfield3d-nn/actions/workflows/build.yml)
[![Tests](https://github.com/Centrasis/radfield3d-nn/actions/workflows/tests.yml/badge.svg)](https://github.com/Centrasis/radfield3d-nn/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/radfield3d-nn.svg)](https://pypi.org/project/radfield3d-nn/)

Neural networks that predict spatially-resolved X-ray **flux** and **spectrum** fields (and a
derived **air-kerma** metric) from beam parameters, trained on
[RadFiled3D](https://github.com/Centrasis/RadFiled3D) `.rf3` datasets. Pure Python behind one
PyTorch-Lightning entry point; trained models export to a self-contained `.rf3m` package that the
[`radfiled3d-nn`](https://github.com/Centrasis/radfiled3d-nn) runtime loads and runs with no
PyTorch dependency.

## Models

| Model | Type | Description |
|---|---|---|
| `PBRFNet` | Pure Python | Parametric Beam Radiation Field Network — per-voxel implicit field, queried `(xyz, beam-params) → (flux, spectrum)` |
| `TPBRFNet` | Pure Python | `PBRFNet` + patient translation (vec3 from field metadata; x/y encoded, z is constant) |
| `SPERFNet` | Pure Python | Spectral Enhanced Radiation Field Network (PBRFNet's parent; no parametric beam distance) |
| `SRBFNet` | Pure Python | Static Rotatable Beam Field Network (base of the lineage) |
| `FieldScatterUNet` | Pure Python | Field-wise 3D U-Net: predicts the whole scatter volume in one pass from the direct beam |

The shipped/deployable targets are **`PBRFNet`** / **`TPBRFNet`** (per-voxel) and
**`FieldScatterUNet`** (field-wise). The former tiny-cuda-nn fused models (`PBRFNetCPP`,
`SPERFNetCPP`) are **gone**: only ONNX was ever deployable, they did not support the current
training features, and keeping a second implementation of the same network alive cost more than it
returned. Their checkpoints no longer load.

## Installation

```bash
pip install -e .
```

Nothing is compiled. `radfiled3d-nn` (the `.rf3m` container and the inference runtime) comes in as
an ordinary dependency; the optional `HashGridEncoding` additionally needs `tinycudann` on the
machine.

## Usage

```
python run_network_task.py [OPTIONS] CONFIG_YAML

Options:
  --task {train,tune}   Task to run (default: train)
  --dataset_path PATH   Path to the RadFiled3D dataset (required)
  --logs_path PATH      Directory for logs and checkpoints (required)
  --mu_tr_file FILE     Mass energy-absorption coefficient table (for the air-kerma metrics)
  --seed INT            Random seed (default: a fresh random seed, persisted into the run config)
```

```bash
python run_network_task.py --task train \
    --dataset_path /data/dataset --logs_path /logs \
    --mu_tr_file /data/mu_tr.txt --seed 42 config.yaml
```

## Configuration

Configuration is split across three places:

1. **5 CLI args** — `--task`, `--dataset_path`, `--logs_path`, `--mu_tr_file`, `--seed`.
2. **Training YAML** (positional `CONFIG_YAML`) — `training` / `dataset` / `augmentations` / `tune`
   blocks (below).
3. **Model JSON** (`training.model_config`) — architecture + normalizer (below).

### Training YAML

Every key is optional; the default is shown. `training.model_config` is the only required field.

```yaml
training:
  model_config: path/to/model.json   # REQUIRED — the model JSON (architecture + normalizer)
  epochs: 25                         # number of training epochs
  batch_size: 32                     # fields per optimizer step (whole fields, not voxels)
  effective_batch_size: null         # if set, gradient-accumulate up to this many fields/step
  max_inner_batch_size: null         # voxel-chunk size for full-volume assembly (transient VRAM only)
  num_workers: 4                     # dataloader workers (keep low; /dev/shm is small)
  precision: fp32                    # fp32 | fp16 — pure-Python model compute precision
  mixed_precision: false             # AMP: fp32 master weights, fp16 compute
  compile_model: false               # torch.compile the model
  prefetch_to_device: false          # prefetch batches onto the GPU
  check_val_every_n_epoch: 1         # validation cadence
  early_stopping: true               # patience stop on the monitored air-kerma accuracy (after a
                                     #   warmup of epochs/3); false runs the whole epoch budget
  limit_train_batches: null          # cap train batches/epoch (debug)
  limit_val_batches: null            # cap val batches/epoch (debug)
  lr_finder: false                   # run the Lightning LR finder before training
  weight_ema: false                  # keep an exponential-moving-average copy of the weights
  weight_ema_decay: 0.999            # EMA decay when weight_ema is on
  mtl_balancing: true                # DB-MTL balancing of the flux vs spectrum tasks
  mtl_gradient_balancing: false      # extra per-task gradient-magnitude balancing (costly)
  spectrum_loss_weight: null         # fixed weight on the spectrum task (overrides MTL if set)
  validate_gt: false                 # sanity-check the ground truth at startup
  hard_exit: true                    # os._exit(0) after the run's own shutdown — guards against
                                     #   interpreter-exit deadlocks (persistent dataloader workers +
                                     #   CUDA teardown) leaving a dead-but-running cluster job
  test_mode: false                   # short smoke run
  debug_probe: false                 # log a per-step LOSS/region breakdown to <logs>/debug_probe.log
  debug_probe_every: 50              # debug-probe interval (steps)
  logger:                            # experiment tracking (sub-section of `training`)
    type: wandb                      #   wandb | mlflow
    mlflow_tracking_uri: null        #   mlflow only: local/UNC dir, file:// URI, or http(s)://
                                     #     tracking-server URL; default <logs_path>/<run>/mlflow
    project_name: radiation-field-estimator  # tracking project (separate ablations)
    run_name: null                   #   run name (defaults to "<model>-<dataset>")
    offline: false                   #   wandb only: log offline (no network)

dataset:
  path: null                         # dataset directory; --dataset_path on the CLI overrides it
                                     #   (one of the two is required)
  type: Layerwise                    # Layerwise | Voxelwise (mostly a batch-size switch; see note)
  voxel_resolution: null             # [x, y, z] inference grid, or null to use the field's own
  use_beam_parameters: false         # reshape 3D origin -> 1D source distance (PBRFNet needs TRUE)
  use_geometry: false                # load the phantom geometry channel (binary occupancy mask)
  use_translation: false             # yield TranslationalInput: vec3 patient translation from field metadata (TPBRFNet needs TRUE)
  scan_translation_metadata: true    # with use_translation: check EVERY field for the patient_translation
                                     #   entry before training and report all offenders (metadata-only read)
  skip_fields_without_translation: false  # train on the fields that have it instead of failing
  definition_file: null              # dataset definition JSON the dataset was GENERATED with; source of the
                                     #   beam-parameter ranges (replaces statistics.json) and of the patient-
                                     #   translation [0,1] normalization ranges (required for use_translation)
  use_airkerma: false                # train directly on the air-kerma field
  max_fields: null                   # cap the number of fields loaded (fast iteration)
  cache: false                       # cache decoded fields to disk
  cache_dir: ./.cache                # disk cache location
  cache_min_age_s: 60                # don't cache files modified less than this many seconds ago
                                     #   (they may still be written by a running simulation; 0 = off)
  cache_to_ram: false                # cache decoded fields in RAM
  cache_ram_gb: null                 # RAM cache budget (GB)

augmentations:
  enabled: false                     # Gaussian fluence noise + smoothing (first half of training)
  exclude_geometry_voxels: true      # drop geometry-covered voxels from the TRAINING target: the
                                     #   simulation excludes them from scoring, so their GT holds no
                                     #   statistics (mask read from the field's geometry channel;
                                     #   fields without one are unaffected)
  geometry_threshold: 0.0            # covered = geometry/density > this (0.0 also fits binary masks)
  smooth_spectra: false              # 3D Gaussian smoothing over the spatial domain
  join_channels: false               # join scatter + direct into one flux target
  mc_floor_cut:                      # remove the Monte-Carlo noise floor from the TRAINING target
    mask: true                       #   MASK mode: set the floor ROI to -inf (not 0), join-safe
    # beam_rel: 0.05                  #   floor = NOT beam AND joined < scatter_lo*joined_max
    # scatter_lo: 5.0e-5              #   (a scalar, or {scatter, direct}, instead zeroes per-channel)
  importance_sampling:
    enabled: false
    method: error                    # error (ErrorbasedImportanceSampler) | roi (ROIbasedSampler)
    # --- method: error ---
    max_drop_chance: 0.9             # max probability of dropping a low-information voxel
    keep_flux_threshold: 0.8         # keep voxels above this fraction of peak flux
    # --- method: roi ---
    beam_rel: 0.05                   # beam = direct >= 0.05*direct_max
    scatter_lo: 5.0e-5               # scatter floor = joined >= 5e-5*joined_max
    beam_keep_ratio: 1.0             # fraction of beam voxels kept
    scatter_ratio: 2.0               # scatter voxels sampled per kept beam voxel
    floor_ratio: 1.0                 # floor voxels sampled per kept beam voxel
    floor_as_zero: true              # re-inject the sampled floor as a genuine 0
    field_multiplier: 3.0            # repeat each field xN per epoch (fresh scatter subset each time)

tune:
  n_trials: 50                       # Optuna trials when --task tune
  sampler: tpe                       # tpe | grid — grid runs every combination of the model
                                     #   JSON's hyperparameter_space exactly once (an ablation);
                                     #   tpe samples with replacement (a large/continuous space)
```

> **VRAM / batch_size:** a whole-field step predicts every voxel of every field in the batch, and
> all of those activations stay alive until the backward pass — the requirement scales with
> `batch_size x voxels`, roughly `2 KB x (2 + trunk_depth + 3) x d_model / 192` per voxel. At
> `voxel_resolution: [64, 64, 64]` with `d_model: 192, trunk_depth: 5` that is ~2 GiB per field, so
> `batch_size: 32` needs ~67 GiB. Keep the per-step batch small and restore the effective batch
> with `effective_batch_size` (gradient accumulation), reduce the resolution, or enable
> `augmentations: importance_sampling` (it drops most voxels before the forward).
> **`max_inner_batch_size` does not help here** — it bounds the transient chunk size, not the
> retained graph. The run prints a VRAM preflight estimate at startup.
>
> **Logger block:** the legacy flat form (`training: logger: wandb` with sibling `project_name` /
> `run_name` / `offline` / `mlflow_tracking_uri` keys) is still accepted; a flat key is only read
> when the `logger:` sub-section does not define it.
>
> **Training-only:** `mc_floor_cut` and the importance samplers apply only during training;
> validation/test always see the whole, unmasked field, so reported accuracy measures whole-field
> generalisation.
>
> **`use_beam_parameters`:** PBRFNet / TPBRFNet require it `true` (their beam encoder expects a 1D
> source distance). Per-voxel models that consume the 3D origin need it `false`.
>
> **Feature composition:** `use_geometry`, `use_translation` and `use_beam_parameters` combine
> FREELY. The network input is one class with every possible member (`rftypes.FieldInput`); each
> optional member is initialized by its input decorator (`datasets/decorators.py` — geometry,
> translation) stacked over the plain dataset, while the beam-parameter members come from the base
> input and the flag only adds the normalization. A decorator never overwrites a member another
> source already set — it warns and keeps the existing value.

### Model JSON

`{"model_name", "parameters": {...}}`. The **normalizer lives here**, not the CLI. Example (the
reference PBRFNet recipe):

```json
{
  "model_name": "PBRFNet",
  "parameters": {
    "normalizer": "linear0_1",
    "d_model": 192,
    "out_spectra_dim": 32,
    "trunk_depth": 5,
    "flux_head_hidden": 1,
    "flux_activation": "sigmoid",
    "flux_loss": "SMAPEBalanced",
    "spectrum_loss": "HistogramLoss",
    "location_encoding_params":  {"type": "sinusoidal", "pos_enc_dim": 14, "append_input": true},
    "direction_encoding_params": {"type": "spherical_harmonics", "degree": 4, "append_input": true},
    "spectra_encoding_params":   {"type": "simple", "in_spectra_dim": 32, "encoded_spectra_dims": 32},
    "conditioning_params":       {"type": "Concat", "use_beam_shape": false},
    "training_params":           {"learning_rate": 1.0e-3, "max_lr": 5.0e-4}
  }
}
```

| Parameter | Meaning |
|---|---|
| `normalizer` | Target transform: `linear0_1` (physical [0,1]) or `asinh` (bounded HDR tonemap) |
| `d_model` | Trunk / hidden width |
| `out_spectra_dim` | Predicted spectrum histogram bins |
| `trunk_depth` | Number of trunk MLP layers (≥2) |
| `flux_head_hidden` | SiLU-separated hidden layers in the flux head (0 = single Linear) |
| `flux_activation` | `sigmoid` (default recipe) \| `softplus` \| `exp` (log-space head, (0,1] codomain) \| `clamp` \| `softclip` \| `none` |
| `flux_loss` | `SMAPEBalanced` \| `SMAPEBalancedLogRatio` (same regions, non-saturating log-ratio core) — or the dict form `{"type": "SMAPEBalancedLogRatio", "eps": 1e-7, ...}` to set the core's floor `eps` (below it relative errors stop mattering; 1e-4 by name) and the region knobs \| `L1Plain` \| `L1Loss` \| `TwoROIGammaLoss` \| `RawNeRF` \| `StructuralSimilarity3DLoss` |
| `spectrum_loss` | `HistogramLoss` |
| `location_encoding_params` | Position encoder — `sinusoidal` or `hashgrid` (`type` + kwargs) |
| `direction_encoding_params` | Direction encoder — `spherical_harmonics` (`degree`) or `rff` |
| `spectra_encoding_params` | Spectrum encoder — `simple` (bottleneck) or `projector` (raw) |
| `conditioning_params` | Beam→trunk fusion `type`: `None`, `FiLM`, `ResFiLM`, `Gated`, `Concat`, `Attention`, `TokenAttention` (+ `use_beam_shape`) |
| `conditioning_params.beam_geometry` | In-graph beam/patient geometry from the existing inputs: `definition_file` (physical ranges; persisted as `physics`), `beam_frame` / `patient_frame` (per-point encoders or `false`), `ray_features`, `direct_prior`. Needs rectangle collimation; see `configs/tpbrfnet_ds04_geometry.json` |
| `conditioning_params.physics` (PhysPBRFNet only) | Ray-marched physics probe over a canonical patient occupancy carried in the model: `patient_occupancy_file` (npz from `scripts/make_patient_occupancy.py`), `emitters`, `direct_samples`, `emitter_samples`, `token_dim`, `learn_density`, `mu_per_m`. Needs `beam_geometry` with a translation and `direct_prior`; same inputs as TPBRFNet. See `configs/physpbrfnet_ds04.json` |
| `training_params` | `learning_rate`, `max_lr`, voxel-sampling flags |
| `precision` | `fp32` \| `fp16` (pure-Python models) |

`FieldScatterUNet` takes a different parameter set (`depth`, `cond_dim`, `out_dims`,
`use_analytic_direct`, …); see `radfield3dnn/models/field_unet.py`.

## Deployment — RF3M packages

Training writes a self-contained **`.rf3m`** package: the model's exported ONNX graph(s) bound to
its validity domain (the valid beam-parameter ranges and the physical meaning, in metric units, of
the normalised I/O), lightweight provenance, and the test metrics — everything a deployment needs
to run and interpret the model without the training stack.

The container, the inference runtime and the graphics interop are the separate
**[`radfiled3d-nn`](https://github.com/Centrasis/radfiled3d-nn)** module (`pip install
radfiled3d-nn`, `from radfiled3d import nn`). This repo writes packages and never parses or executes
one. What it contributes is the half only the training stack can: the graphs, and the declaration
of what each of their tensors means.

### Writing a package

`PackageExportCallback` does this automatically at the end of the test stage; the same thing by
hand:

```python
from radfield3dnn.deploy import ModelPackager

ModelPackager(model, datamodule, trainer.callback_metrics,
              dataset_path="/data/DS04-DynGeom").save("model.rf3m")
```

The model declares its own tensors through `deploy_interface()` — the *meaning* of each graph
tensor (semantic, unit); tracing decides which of them exist. Input shapes are read
back off the traced graph, ranges come from the dataset's `statistics.json`, and the package is
read back after writing, so a declaration that does not match the graphs never reaches disk.
Tensors are named with the deploy module's own semantic vocabulary (`beam_direction`,
`source_distance`, `tube_spectrum`, …) so every package in the corpus binds the same way.

### Reading one back

```python
from radfiled3d.nn import deploy, inference

meta = deploy.ModelStore.load_metadata("model.rf3m")      # header only: no graph decoded, no session, no GPU
print(meta["tensors"])                        # what to bind, and what it means

session = inference.Session(deploy.ModelStore.load("model.rf3m"), backend="cuda", device=0)
session.set_voxel_grid([64, 64, 64])          # CLEARS every binding — bind after, never before
session.bind_input("position", positions)     # caller-owned, bound by pointer; no copy
session.bind_input("beam_direction", d)        # METRIC values: the network normalizes its own inputs
session.bind_output("flux", flux)             # nothing is allocated for you
session.infer()                               # names EVERY missing binding at once
```

A bound buffer must outlive its binding — the session holds a raw pointer into it. Backends are
compile-time options of that module: one the installed wheel lacks raises a `ValueError` naming its
CMake option (`RFNN_WITH_CUDA`, …) rather than falling back to something slower. The default wheel
is the CPU/ONNX build.

The runtime executes the package's **`trunk`** graph. A model that factors into a `beam_encoder`
and a `trunk` ships both, and a consumer runs the encoder itself and binds the resulting `latent`
as an ordinary trunk input — the container records nowhere how graphs compose.

One container format exists (`FormatVersion::V1 = 1`); the writer always emits the current one and
this repo never names a version. Historical packages were migrated with that module's
`rf3m convert`.

## Tests

```bash
# Pure-Python tests (no GPU; self-skip if RadFiled3D is absent)
python -m pytest tests/test_asinh_normalizer.py tests/test_metric_loss_alignment.py -v

# Full suite (some tests need a GPU)
python -m pytest tests/ -v
```

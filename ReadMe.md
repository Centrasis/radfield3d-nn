# RadField3D-NN

Neural networks that predict spatially-resolved X-ray **flux** and **spectrum** fields (and a
derived **air-kerma** metric) from beam parameters, trained on
[RadFiled3D](https://github.com/Centrasis/RadFiled3D) `.rf3` datasets. Pure-Python models and
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)-backed C++ models coexist behind one
PyTorch-Lightning entry point; trained models export to a self-contained `.rf3m` package that runs
through an ONNX-Runtime deploy runtime with no PyTorch dependency.

## Models

| Model | Type | Description |
|---|---|---|
| `PBRFNet` | Pure Python | Parametric Beam Radiation Field Network — per-voxel implicit field, queried `(xyz, beam-params) → (flux, spectrum)` |
| `SPERFNet` | Pure Python | Spectral Enhanced Radiation Field Network (PBRFNet's parent; no parametric beam distance) |
| `SRBFNet` | Pure Python | Static Rotatable Beam Field Network (base of the lineage) |
| `FieldScatterUNet` | Pure Python | Field-wise 3D U-Net: predicts the whole scatter volume in one pass from the direct beam |
| `PBRFNetCPP` | C++ / tcnn | PBRFNet with a fused tiny-cuda-nn encoder (built only with `RFNN_WITH_TCNN`) |
| `SPERFNetCPP` | C++ / tcnn | Distance-less `PBRFNetCPP` variant for fixed-distance datasets |

The shipped/deployable targets are **`PBRFNet`** (per-voxel) and **`FieldScatterUNet`** (field-wise).

## Installation

```bash
pip install -e .                              # pure-Python install (tcnn off by default)
RFNN_WITH_TCNN=1 pip install -e .             # also build the tcnn C++ models (needs CUDA + GPU)
CMAKE_CUDA_ARCHITECTURES=89 RFNN_WITH_TCNN=1 pip install -e .   # target a specific CUDA arch
```

With tcnn enabled, CMake fetches tiny-cuda-nn from GitHub on first build.

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
  max_inner_batch_size: null         # voxel-chunk size for full-volume assembly (caps VRAM)
  num_workers: 4                     # dataloader workers (keep low; /dev/shm is small)
  precision: fp32                    # fp32 | fp16 — pure-Python model compute precision
  mixed_precision: false             # AMP: fp32 master weights, fp16 compute
  compile_model: false               # torch.compile the model
  prefetch_to_device: false          # prefetch batches onto the GPU
  check_val_every_n_epoch: 1         # validation cadence
  limit_train_batches: null          # cap train batches/epoch (debug)
  limit_val_batches: null            # cap val batches/epoch (debug)
  lr_finder: false                   # run the Lightning LR finder before training
  weight_ema: false                  # keep an exponential-moving-average copy of the weights
  weight_ema_decay: 0.999            # EMA decay when weight_ema is on
  mtl_balancing: true                # DB-MTL balancing of the flux vs spectrum tasks
  mtl_gradient_balancing: false      # extra per-task gradient-magnitude balancing (costly)
  spectrum_loss_weight: null         # fixed weight on the spectrum task (overrides MTL if set)
  validate_gt: false                 # sanity-check the ground truth at startup
  test_mode: false                   # short smoke run
  debug_probe: false                 # log a per-step LOSS/region breakdown to <logs>/debug_probe.log
  debug_probe_every: 50              # debug-probe interval (steps)
  logger: wandb                      # wandb | mlflow
  project_name: radiation-field-estimator  # experiment-tracking project (separate ablations)
  run_name: null                     # run name (defaults to "<model>-<dataset>")
  offline: false                     # log offline (no network)

dataset:
  type: Layerwise                    # Layerwise | Voxelwise (mostly a batch-size switch; see note)
  voxel_resolution: null             # [x, y, z] inference grid, or null to use the field's own
  use_beam_parameters: false         # reshape 3D origin -> 1D source distance (PBRFNet needs TRUE)
  use_geometry: false                # load the phantom density channel (analytic direct beam shadow)
  use_airkerma: false                # train directly on the air-kerma field
  max_fields: null                   # cap the number of fields loaded (fast iteration)
  cache: false                       # cache decoded fields to disk
  cache_dir: ./.cache                # disk cache location
  cache_to_ram: false                # cache decoded fields in RAM
  cache_ram_gb: null                 # RAM cache budget (GB)

augmentations:
  enabled: false                     # Gaussian fluence noise + smoothing (first half of training)
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
```

> **Training-only:** `mc_floor_cut` and the importance samplers apply only during training;
> validation/test always see the whole, unmasked field, so reported accuracy measures whole-field
> generalisation.
>
> **`use_beam_parameters`:** PBRFNet / `*CPP` require it `true` (their beam encoder expects a 1D
> source distance). Per-voxel models that consume the 3D origin need it `false`.

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
| `flux_activation` | `clamp` \| `softclip` \| `sigmoid` (sigmoid is the HDR `linear0_1` head) |
| `flux_loss` | `SMAPEBalanced` \| `L1Plain` \| `L1Loss` \| `TwoROIGammaLoss` \| `RawNeRF` \| `StructuralSimilarity3DLoss` |
| `spectrum_loss` | `HistogramLoss` |
| `location_encoding_params` | Position encoder — `sinusoidal` or `hashgrid` (`type` + kwargs) |
| `direction_encoding_params` | Direction encoder — `spherical_harmonics` (`degree`) or `rff` |
| `spectra_encoding_params` | Spectrum encoder — `simple` (bottleneck) or `projector` (raw) |
| `conditioning_params` | Beam→trunk fusion `type`: `None`, `FiLM`, `ResFiLM`, `Gated`, `Concat`, `Attention`, `TokenAttention` (+ `use_beam_shape`) |
| `training_params` | `learning_rate`, `max_lr`, voxel-sampling flags |
| `precision` | `fp32` \| `fp16` (pure-Python models) |

`FieldScatterUNet` takes a different parameter set (`depth`, `cond_dim`, `out_dims`,
`use_analytic_direct`, …); see `radfield3dnn/models/field_unet.py`.

## Deployment — RF3M packages

Training writes a self-contained **`.rf3m`** package: the model's exported ONNX graph(s) bound to
its validity domain (the valid beam-parameter ranges and the physical meaning, in metric units, of
the normalised I/O), lightweight provenance, and the test metrics — everything a deployment needs
to run and interpret the model without the training stack. Loading runs through the C++
`rfnn_deploy` runtime (ONNX Runtime, no PyTorch / CUDA):

```python
from radfield3dnn.deploy import load_rf3m
pred = load_rf3m("PBRFNet.rf3m")              # -> Voxel|VolumeFieldPredictor (runnable)
out  = pred.predict_volume(beam, (48, 48, 48))  # dict(flux=np[D,H,W], spectrum=np[D,H,W,bins])
```

Stored ONNX graphs always use a **dynamic batch axis**, so a package runs at any batch / voxel
count. Per-voxel models (PBRFNet) export two graphs — a `beam_encoder` (beam parameters → latent)
and a `trunk` (position + latent → flux/spectrum) — so the runtime encodes the beam once and reuses
the latent across every voxel. Field-wise models export a single `trunk`.

### RF3M binary format (little-endian)

The byte layout is owned by `rfnn::io::V1::ModelStore` (the reference implementation is
`src/RadField3DNN/model_io.cpp`); its `rfnn_deploy` binding is `ModelStore.load(path)`:

```
[4]    magic "RF3M"
[u32]  version (== 2)
[str]  dataset_name              # str = [u32 length][UTF-8 bytes]
[str]  software_version
[str]  physics
ModelDomain:
  [i32]  spectrum_bins
  [f32]  spectrum_max_energy_ev
  [u32]  beam_parameter_count
  per beam parameter:
    [str] name                   # "direction" | "distance" | "opening_angle" | "spectrum"
    [i32] slot_count             # length in the input vector
    [f32] range_min
    [f32] range_max
    [str] unit                   # "", "m", "deg", "eV"
[u32]  metric_count
  per metric: [str] name [f32] value
[u32]  graph_count
  per graph:
    [str] name                   # "beam_encoder" | "trunk"
    [u64] byte_length
    [byte * byte_length]         # raw ONNX model bytes
```

The predicted spatial resolution / voxel geometry is **not** stored — it is chosen at inference and
may vary, so it is not a property of the model. A field box is 1.0 m, so voxel size = box /
resolution.

## Tests

```bash
# Pure-Python tests (no GPU; self-skip if RadFiled3D is absent)
python -m pytest tests/test_asinh_normalizer.py tests/test_metric_loss_alignment.py -v

# Full suite (GPU + the built tcnn extension for tests/test_pbrfnet_cpp.py, tests/test_nn.py)
python -m pytest tests/ -v
```

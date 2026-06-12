# RadField3D-NN

Neural networks for spatially-resolved X-ray flux and spectrum prediction, built on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [RadFiled3D](https://github.com/Centrasis/RadFiled3D) datasets.

## Models

| Model | Type | Description |
|---|---|---|
| `SRBFNet` | Pure Python | Static Rotatable Beam Field Network |
| `SPERFNet` | Pure Python | Spectral Enhanced Radiation Field Network |
| `PBRFNet` | Pure Python | Parametric Beam Radiation Field Network |
| `SPERFNetCPP` | C++/TCNN | Distance-less SPERFNet with TCNN encoder |
| `PBRFNetCPP` | C++/TCNN | PBRFNet with TCNN hash-grid encoder |
| `Beam2ScatterUNet` | Pure Python | 3D U-Net mapping direct beam to scatter field |

## Installation

```bash
pip install -e .
```

Requires CUDA toolkit and a GPU. On first build, CMake fetches tiny-cuda-nn from GitHub automatically.

To target a specific CUDA architecture:
```bash
CMAKE_CUDA_ARCHITECTURES=89 pip install -e .
```

## Usage

```
python run_network_task.py [OPTIONS] CONFIG_YAML

Options:
  --task {train,tune}   Task to run (default: train)
  --dataset_path PATH   Path to the RadFiled3D dataset (required)
  --logs_path PATH      Directory for logs and checkpoints (required)
  --mu_tr_file FILE     Mass energy absorption coefficients file (for Airkerma metrics)
  --seed INT            Random seed (default: random)
```

## Configuration

All training settings are in a YAML file. The model architecture is specified in a JSON file (referenced from the YAML).

### YAML Config

```yaml
training:
  model_config: path/to/model.json  # JSON model config
  epochs: 100
  batch_size: 32
  effective_batch_size: null         # null = no gradient accumulation
  num_workers: 4
  precision: fp32                    # fp32 | fp16
  mixed_precision: false             # AMP (fp32 weights, fp16 compute)
  flux_offset: 0.5
  compile_model: false
  test_mode: false
  max_inner_batch_size: null
  validate_gt: false
  logger: wandb                      # wandb | mlflow
  project_name: radiation-field-estimator  # experiment-tracking project (override to keep ablations separate)
  offline: false

dataset:
  type: Layerwise                    # Layerwise | Voxelwise
  voxel_resolution: null             # [x, y, z] or null
  cache: false
  cache_dir: ./.cache
  use_geometry: false
  use_beam_parameters: false
  use_airkerma: false

augmentations:
  enabled: false
  smooth_spectra: false
  join_channels: false
  importance_sampling:
    enabled: false
    method: error            # "error" (default) or "roi" — selects the voxel sampler
    # --- method: error (ErrorbasedImportanceSampler) ---
    max_drop_chance: 0.9
    keep_flux_threshold: 0.8
    # --- method: roi (ROIbasedSampler — keep beam, sample scatter/floor by the metric ROIs) ---
    beam_rel: 0.05           # beam = direct >= 0.05*direct_max (matches the scatter metric + loss)
    scatter_lo: 5.0e-5       # scatter floor = joined >= 5e-5*joined_max
    beam_keep_ratio: 1.0     # fraction of beam voxels kept (0..1, default 1 = keep all)
    scatter_ratio: 2.0       # scatter voxels sampled per kept beam voxel (0..inf)
    floor_ratio: 1.0         # floor voxels sampled per kept beam voxel (capped by what exists)
    field_multiplier: 3.0    # repeat each field ×N/epoch; beam always kept, fresh scatter subset each repeat

tune:
  n_trials: 50
```

### Model JSON Config

The model JSON specifies the architecture and **normalizer**:

```json
{
  "model_name": "PBRFNetCPP",
  "parameters": {
    "normalizer": "asinh_split",
    "d_model": 256,
    "spectra_bins": 32,
    "flux_activation": "clamp",
    "flux_clamp_min": 0.0,
    "flux_clamp_max": 1.0
  }
}
```

Available normalizers: `linear0_1`, `linear-1_1`, `log_scale`, `asinh`, `asinh_split`, `asinh_auto`

(`asinh_auto` scans the dataset on startup to pick per-channel σ values automatically.)

### Example: Single Training Run

```bash
python run_network_task.py \
    --task train \
    --dataset_path /data/DS03 \
    --logs_path /logs \
    --mu_tr_file /data/mu_tr.json \
    --seed 42 \
    config.yaml
```

### Example: Hyperparameter Tuning

```bash
python run_network_task.py \
    --task tune \
    --dataset_path /data/DS03 \
    --logs_path /logs \
    --seed 42 \
    config.yaml
```

Set `tune.n_trials` in the YAML to control the number of Optuna trials.

## Running Tests

```bash
# Pure Python tests (no GPU required)
python -m pytest tests/test_asinh_normalizer.py tests/test_logscale_normalizer.py tests/test_channels_split_relative.py -v

# Full test suite (GPU required for C++ extension tests)
python -m pytest tests/ -v
```

## Architecture Notes

- **Beam2ScatterUNet**: 4-channel input (direct-beam flux + 3D coordinate grid). Uses FiLM conditioning on input spectra. Fixed from prior version which had wrong final-layer initialization and single-channel input.
- **Pure Python models** (SRBFNet, SPERFNet, PBRFNet): fp32/fp16 precision switchable via `precision` in model JSON.
- **C++ models** (SPERFNetCPP, PBRFNetCPP): Use tiny-cuda-nn hash-grid encoder, always fp16 internally.

# RadField3D-NN

[![Build](https://github.com/Centrasis/radfield3d-nn/actions/workflows/build.yml/badge.svg)](https://github.com/Centrasis/radfield3d-nn/actions/workflows/build.yml)
[![Tests](https://github.com/Centrasis/radfield3d-nn/actions/workflows/tests.yml/badge.svg)](https://github.com/Centrasis/radfield3d-nn/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/radfield3d-nn.svg)](https://pypi.org/project/radfield3d-nn/)

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
| `TPBRFNet` | Pure Python | `PBRFNet` + patient translation (vec3 from field metadata; x/y encoded, z is constant) |
| `SPERFNet` | Pure Python | Spectral Enhanced Radiation Field Network (PBRFNet's parent; no parametric beam distance) |
| `SRBFNet` | Pure Python | Static Rotatable Beam Field Network (base of the lineage) |
| `FieldScatterUNet` | Pure Python | Field-wise 3D U-Net: predicts the whole scatter volume in one pass from the direct beam |
| ~~`PBRFNetCPP`~~ | **DEPRECATED** — C++ / tcnn | PBRFNet with a fused tiny-cuda-nn encoder (built only with `RFNN_WITH_TCNN`) |
| ~~`SPERFNetCPP`~~ | **DEPRECATED** — C++ / tcnn | Distance-less `PBRFNetCPP` variant for fixed-distance datasets |

The shipped/deployable targets are **`PBRFNet`** / **`TPBRFNet`** (per-voxel) and
**`FieldScatterUNet`** (field-wise).

> **The `*CPP` (tiny-cuda-nn) models are deprecated — do not extend them, do not start new work on
> them.** They are a dead deployment path: only ONNX is deployable (the RF3M header's `backend` word
> reserves `1` for tcnn and the runtime **rejects** it), and they do not support the current training
> features (`voxel_supersampling` is gated off for them in `run_network_task.py`). They remain in the
> tree to keep existing checkpoints loadable; new models are pure-Python and export to ONNX.

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
  use_translation: false             # derive the 1D patient translation from the geometry density layer (TPBRFNet needs TRUE)
  translation_axis: z                # axis the patient moves along (density center-of-mass, meters from box center)
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
import numpy as np
import torch

from radfield3dnn.deploy import ModelInput, ModelOutput, Unit, load_rf3m, read_metadata

# What is in the package — header only, no inference session, no GPU.
meta = read_metadata("PBRFNet.rf3m")
print(meta.interface.is_voxelwise, meta.domain.spectrum_bins, meta.metrics)

pred = load_rf3m("PBRFNet.rf3m", backend="cuda")   # "cpu" | "cuda" | "tensorrt"
print(pred)                                        # backend actually in use, grid, I/O flags

pred.set_voxel_grid((32, 32, 32))                  # the grid this run fills; free to change per call

# The CALLER owns every buffer. Bind one per input the model declares — ask it, do not assume.
# Bound buffers reach the graph VERBATIM, so they hold NORMALISED values.
spectrum  = torch.ones(pred.input_spectrum_bins, device="cuda")   # tube histogram (150 bins here)
direction = torch.tensor([0.0, 0.0, -1.0], device="cuda")
distance  = torch.zeros(1, device="cuda")

pred.bind_global_parameter(ModelInput.TUBE_SPECTRUM, spectrum)
pred.bind_global_parameter(ModelInput.BEAM_DIRECTION, direction)
pred.bind_global_parameter(ModelInput.SOURCE_DISTANCE, distance)

# Metric values go through the converter that knows this model's domain. normalize() hands them back,
# so a device buffer is filled by your own framework; write() puts them straight into a bound HOST
# buffer. (A source 0.8 m away -> the [0,1] range the network was trained on.)
norm = pred.parameter_normalizer()
distance.copy_(torch.from_numpy(norm.normalize(ModelInput.SOURCE_DISTANCE, [0.8], Unit.Metres)))

# Outputs are written STRAIGHT into your memory — nothing is allocated, nothing is copied.
flux = torch.empty((32, 32, 32), dtype=torch.float32, device="cuda")
spec = torch.empty((pred.spectrum_bins, 32, 32, 32), dtype=torch.float32, device="cuda")
pred.bind_output_layer(ModelOutput.FLUX, flux)
pred.bind_output_layer(ModelOutput.SPECTRUM, spec)

pred.predict_volume()                              # no arguments — it runs into what was bound

# A bound input is re-read every run: edit it in place and the next inference sees it, no rebinding.
spectrum.mul_(2.0)
pred.predict_volume()

# Refresh only part of the grid. Every voxel not named keeps the value it already had.
roi = np.array([[i, j, k] for i in range(4, 8) for j in range(4, 8) for k in range(4, 8)],
               dtype=np.int32)
pred.predict_voxels(roi)
```

`bind_*` accepts a **numpy array or a torch tensor** (CPU or CUDA) and binds its memory directly — no
copy, no allocation, no result to unpack. It is **strict by default**: a buffer whose device or
precision the session cannot consume is *refused*, with an error naming exactly what mismatched and
what a conversion would cost. `convert_buffer=True` accepts that cost deliberately: the runtime keeps
a staged copy, which for an input **snapshots it at bind time** (later in-place edits are no longer
seen) and, on a GPU session, is what uploads a host buffer each run.

The model holds a raw pointer **into** your array, so a bound buffer must outlive the binding — the
Python bindings keep a reference for you, so binding a temporary is safe.

There is no Vulkan backend: ONNX Runtime has no Vulkan execution provider. A Vulkan consumer instead
registers its own buffer as the prediction target (`rfnn::vk::VulkanBufferTarget`, C++): the buffer's
exported memory is bound as the model's output, so inference writes Vulkan-visible memory directly —
no export copy, no host round-trip.

`print(pred)` reports the backend it is *actually* running on (a `"cuda"` request falls back to CPU
when the CUDA runtime is missing), the grid, and the model's declared input/output flags.

The allocating convenience path is still there when you do not care about the memory:
`pred.predict_volume(beam, (32, 32, 32))` returns fresh numpy arrays and normalises the
`BeamParameters` for you.

Stored ONNX graphs always use a **dynamic batch axis**, so a package runs at any batch / voxel
count. Per-voxel models (PBRFNet) export two graphs — a `beam_encoder` (beam parameters → latent)
and a `trunk` (position + latent → flux/spectrum) — so the runtime encodes the beam once and reuses
the latent across every voxel. Field-wise models export a single `trunk`.

### Reusing an exported model

The runtime is **C++ first** — the deployed consumer is a C++ renderer — and the *same* API is bound
to Python, so a pretrained package can be driven from either side with identical semantics. There is
one implementation; Python calls it, it does not re-implement it.

**The design.** A package declares its own interface, so loading is not a guessing game. `load()`
reads the declaration, checks the domain can actually shape everything it promises, and returns the
executable class that implements it — `VoxelFieldPredictor` when `Position` is set, otherwise
`VolumeFieldPredictor`. The caller never branches on the model's *name*; it asks the interface what
the model takes.

The C++ flow is the same API, without the numpy/torch marshalling — a buffer is a `MemoryRef`
(pointer + capacity + device + dtype):

```cpp
#include <radfield3d-nn/field_predictors.h>
#include <radfield3d-nn/model_io.h>

using namespace radfield3dnn;
using rfnn::io::V1::ModelStore;

// 1 — LOAD, on the backend you want. The backend decides which memory the model can bind with no
// copy: a CUDA session binds CUDA buffers, a CPU session binds host buffers. Ask for the one your
// buffers already live on — mixing them is refused (see bind_* below).
auto pred = ModelStore::load("PBRFNet.rf3m", Backend::Cpu);   // Backend::Cpu | Cuda | TensorRT
//   by name, the same vocabulary Python uses:
//     ModelStore::load("PBRFNet.rf3m", backend_from_string("cuda"));
//   full control (fp16 kernels, TensorRT engine cache):
//     ModelStore::load("PBRFNet.rf3m", ExecutionOptions{.use_gpu = true, .fp16 = true});

// A GPU request is best-effort: ONNX Runtime falls back to CPU when the CUDA runtime is missing.
// Ask what actually registered before assuming your device buffers can be bound.
const SessionMemory sm = pred->session_memory();
std::printf("running on %s (%s)\n", sm.provider.c_str(), to_string(sm.device).c_str());

// 2 — SET UP. Ask the model what it consumes; do not assume.
const ModelInterface& iface = pred->interface();
if (iface.takes(ModelInput::PatientTranslation3D)) { /* this model wants a patient translation */ }

pred->set_voxel_grid({32, 32, 32});     // the grid this run fills; free to change per call

// The CALLER owns every buffer. Values reach the graph VERBATIM, so they must be NORMALISED —
// parameter_normalizer() converts metric ones using the model's own domain.
std::vector<float> spectrum(pred->input_spectrum_bins(), 1.0f);   // tube histogram (150 bins)
std::vector<float> direction{0.0f, 0.0f, -1.0f};
std::vector<float> distance(1, 0.0f);

auto host = [](std::vector<float>& v) {
    return MemoryRef{v.data(), v.size() * sizeof(float), DeviceKind::Cpu, ElementType::F32, 0};
};
pred->bind_global_parameter(ModelInput::TubeSpectrum,  host(spectrum));
pred->bind_global_parameter(ModelInput::BeamDirection, host(direction));
pred->bind_global_parameter(ModelInput::SourceDistance, host(distance));

// A source 0.8 m away -> the [0,1] range the network trained on, written into the bound buffer.
pred->parameter_normalizer().write(ModelInput::SourceDistance, std::vector<float>{0.8f}.data(), 1,
                                   Unit::Metres);

// 3 — BIND YOUR OUTPUT MEMORY and run. The model writes straight into it and allocates nothing.
std::vector<float> flux(pred->required_elements(ModelOutput::Flux));
std::vector<float> spec(pred->required_elements(ModelOutput::Spectrum));
pred->bind_output_layer(ModelOutput::Flux,     host(flux));
pred->bind_output_layer(ModelOutput::Spectrum, host(spec));

pred->predict_volume();                 // no arguments — it runs into what was bound

// A bound input is re-read every run: edit it in place and the next inference sees it.
spectrum[0] = 2.0f;
pred->predict_volume();

// Per-voxel models can refresh ONLY part of the grid; every voxel not named keeps its value.
if (pred->type() == PredictorType::VoxelField) {
    auto* voxel = static_cast<VoxelFieldPredictor*>(pred.get());
    std::vector<std::array<int, 3>> roi;
    for (int i = 4; i < 8; ++i)
        for (int j = 4; j < 8; ++j)
            for (int k = 4; k < 8; ++k) roi.push_back({i, j, k});
    voxel->predict_voxels(roi);
}
```

On `Backend::Cuda` every bound buffer must be CUDA memory — the host vectors above are **refused**,
with a message naming the mismatch and its cost. Bind a device pointer instead:

```cpp
float* d_flux = nullptr;
cudaMalloc(&d_flux, n * sizeof(float));
pred->bind_output_layer(ModelOutput::Flux,
                        MemoryRef{d_flux, n * sizeof(float), DeviceKind::Cuda,
                                  ElementType::F32, /*device_id=*/0});
```

The small metric inputs are the exception worth knowing: bind a **host** buffer for them with
`convert_buffer = true`, and the runtime keeps a staged copy it uploads on every run — which is what
lets `parameter_normalizer().write()` fill them with metric values even on a CUDA session. (The other
way round, `normalize()` simply *returns* the converted values, so you can upload them yourself.)

The allocating path remains for convenience — `predict_volume(beam, dims)` returns its own vectors and
normalises the `BeamParameters` for you; `predict_into_field()` writes into a RadFiled3D field, and
`predict_to_device()` leaves the result in device memory for the CUDA→Vulkan path.

### Saving a trained model from Python

`PackageExportCallback` writes the package at the end of `test`, so a normal training run produces
the `.rf3m` with no extra step. Done directly, it is:

```python
from radfield3dnn.deploy import ModelPackager
ModelPackager(model, datamodule, test_metrics).save("PBRFNet.rf3m")
```

The packager exports the ONNX graphs, gathers the domain from the dataset, asks the **model** for its
interface (`model.deploy_interface()` — the model knows its own I/O; nothing is configured by hand),
and hands all of it to the C++ writer. It then **reads the package back** and fails loudly if the
result is not what was declared — in particular if a per-voxel model did not come back as a
`VoxelFieldPredictor`, or a field-wise CNN as a `VolumeFieldPredictor`. A package that cannot be
loaded is never left on disk.

> **Not yet deployable:** a model whose location encoder is *resolution-aware* (`ipe` with
> `auto_region_width`) needs a `region_state` input the runtime does not bind yet. Such a package is
> refused at load with an explicit message rather than mis-binding silently.

### GPU integration — dev dependencies

`scripts/setup_gpu_dev_env.sh` (Linux, run with sudo) / `scripts/setup_gpu_dev_env.ps1` (Windows)
install what the optional GPU pieces need to BUILD; per CMake option:

| CMake option | Enables | Needs |
|---|---|---|
| `RFNN_CUDA_VULKAN_INTEROP` | `rfnn::vk::VulkanBufferTarget`, `cuda_vk` image export, `predict_to_device` | CUDA toolkit (cudart headers + lib) |
| `RFNN_BACKEND_VULKAN` | `rfnn::vk` compute helpers (air-kerma combine, visibility culling) | Vulkan SDK + `glslangValidator` (Linux: `libvulkan-dev glslang-tools`) |
| `backend "tensorrt"` (runtime) | TensorRT execution provider | TensorRT libs (`libnvinfer` / `nvinfer.dll`) on the library path |

Registering a Vulkan buffer is per-OS only at the export step: Linux exports an opaque FD
(`vkGetMemoryFdKHR` → `register_buffer`), Windows an OPAQUE_WIN32 handle
(`vkGetMemoryWin32HandleKHR` → `register_buffer_win32`; the handle stays caller-owned, unlike the
FD, which is consumed). Everything downstream is identical.

### Inference requirements (GPU)

The deploy runtime needs no PyTorch. It runs on the **CPU** execution provider out of the box, but the
**CUDA** execution provider is ~20× faster (≈20 ms vs ≈400 ms per inference) and is the intended path.
Using the CUDA EP requires, installed on the system:

- an **NVIDIA driver + GPU** capable of **CUDA 13**,
- the **CUDA 13 runtime** (`libcudart.so.13`, `libcublas.so.13`, `libcufft.so.12`), and
- **cuDNN 9 for CUDA 13** (`libcudnn.so.9`) — package `cudnn9-cuda-13` from NVIDIA's CUDA apt repo. Use
  an LTS repo that carries cuDNN (e.g. `ubuntu2404`, distro-agnostic; the repo path uses `x86_64`, not
  `amd64`). Brand-new non-LTS repos may ship only the base CUDA toolkit (no cuDNN).

If the CUDA EP cannot be loaded, the runtime falls back to the CPU EP (correct, but slow).

### How a model is stored

A package answers three questions, in this order, and each one gates the next:

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ 1. INTERFACE  — what does it consume and produce?                          │
  │      inputs : Position | TubeSpectrum | BeamDirection | SourceDistance …   │
  │      outputs: Flux | Spectrum | …                                          │
  │    Bit flags, not a network name. Two architectures with the same I/O are  │
  │    interchangeable to a consumer.                                          │
  │                                                                            │
  │      Position SET ──► VoxelFieldPredictor   (queried at points)            │
  │      Position CLEAR ► VolumeFieldPredictor  (emits the volume in one shot) │
  └───────────────┬────────────────────────────────────────────────────────────┘
                  │ validate()  — every flag must find its resolution below
  ┌───────────────▼────────────────────────────────────────────────────────────┐
  │ 2. DOMAIN — at what RESOLUTION, and in what physical units?                │
  │      spectrum_bins, angular segments, collimation kind, field box (m),     │
  │      and the valid [min,max] range of every beam parameter.                │
  │    Flags say WHICH quantities; the domain says how WIDE they are. A        │
  │    package that promises Spectrum but records no bin count is refused —    │
  │    on write AND on read.                                                   │
  └───────────────┬────────────────────────────────────────────────────────────┘
                  │
  ┌───────────────▼────────────────────────────────────────────────────────────┐
  │ 3. GRAPHS — the weights, as ONNX                                           │
  │      "beam_encoder" : beam parameters → latent    (once per field)         │
  │      "trunk"        : (position, latent) → flux, spectrum   (per query)    │
  │    Per-voxel models ship both, so the beam is encoded ONCE and the latent  │
  │    reused across every voxel. Field-wise models ship a single "trunk".     │
  └────────────────────────────────────────────────────────────────────────────┘
```

The predicted spatial **resolution is not stored** — the consumer picks it at inference (the C++
runtime sizes the grid to available GPU memory), so it is not a property of the model. Graphs are
exported with a **dynamic batch axis** for exactly this reason. A field box is 1.0 m, so
voxel size = box / resolution.

### RF3M binary format (little-endian)

The byte layout has exactly ONE implementation, `src/RadField3DNN/model_io.cpp`
(`rfnn::io::V1::ModelStore`); Python reaches it through the `rfnn_deploy` bindings and never
serialises anything itself. `str` = `[u32 length][UTF-8 bytes]`.

```
[4]    magic "RF3M"
[u32]  backend                   # ModelBackend: 1 = tiny-cuda-nn (never shipped, rejected), 2 = ONNX
[u32]  version                   # schema version within that backend (== 1)
ModelInterface:
  [u64]  interface_id            # (inputs << 32) | outputs — the bit flags above
  [u8]   resolution_aware        # the location encoder must be configured per queried grid
  [i32]  region_state_dims       # width of the trunk's region_state input (0 = absent)
  [f32]  region_width_frame      # region_width = frame / N for an N³ query grid
[str]  dataset_name
[str]  software_version
[str]  physics
ModelDomain:
  [i32]  spectrum_bins           # OUTPUT spectrum histogram bins
  [f32]  spectrum_max_energy_ev  # bin i spans [i, i+1)·max/bins eV
  [f32 × 3] field_dimensions_m   # metric box the normalised [0,1]³ positions map into
  [i32]  angular_phi_segments    # AngularFlux resolution (0 = no angular output)
  [i32]  angular_theta_segments
  [u8]   collimation             # CollimationType: 0 none, 1 rectangle, 2 cone, 3 ellipsoid
  [u32]  beam_parameter_count
  per beam parameter:
    [str] name                   # "direction" | "distance" | "spectrum" | …
    [u8]  range_type             # 0 MinMax, 1 Spectrum, 2 Map
    [u32] payload_length         # bytes to the next entry (skippable without parsing)
    [payload]                    #   MinMax  : [f32 min][f32 max][str unit]
                                 #   Spectrum: [f32 min][f32 max][f32 bin_width][str unit]
[u32]  metric_count
  per metric: [str] name [f32] value
[u32]  graph_count
  per graph:
    [str] name                   # "beam_encoder" | "trunk" | "encoding_config"
    [u64] byte_length
    [byte × byte_length]         # raw ONNX model bytes
```

Packages written before the interface entered the header are migrated once with
`scripts/convert_rf3m.py <dir> --write` (it derives each package's interface from its own graphs,
keeps a `.bak`, and verifies the rewrite by loading it back).

## Tests

```bash
# Pure-Python tests (no GPU; self-skip if RadFiled3D is absent)
python -m pytest tests/test_asinh_normalizer.py tests/test_metric_loss_alignment.py -v

# Full suite (GPU + the built tcnn extension for tests/test_pbrfnet_cpp.py, tests/test_nn.py)
python -m pytest tests/ -v
```

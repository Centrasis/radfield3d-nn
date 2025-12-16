# RadField3D Neural Networks

This framework is designed to train neural models using spatial radiation fields stored and loaded by [RadFiled3D](https://github.com/Centrasis/RadFiled3D). It supports direct per volume and per voxel predictors. The framework leverages PyTorch and PyTorch Lightning for model training.

## Requirements

All required modules are listed in the `requirements.txt` file. Ensure you have all dependencies installed before running the training scripts.
Note: install pytorch for your existing CUDA version prior to installing the `requirements.txt`. If not, the CPU variant of pytorch gets installed.

## Usage

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python -m run_network_task --task train --model_config </path/to/config.json> --epochs <n> --dataset_path </path/to/dataset_folder or .zip> --dataset_type <Layerwise/Voxelwise> --batch_size <n> --num_workers <n> <--join_channels |> --effective_batch_size <n> --normalization <linear0_1 | linear-1_1 | log_1e+3> --logs_path </path/to/store/logs> --mu_tr_file <path/to/used/mu_tr.txt> --enforce_voxel_resolution <W H D> --logger <mlflow | wandb> <--max_inner_batch_size <n> |> <--use_beam_parameters |> <--use_beam_parameters |> <--validate_gt |>
    ```

4. To tune hyperparameters of a model:
    ```bash
    python -m run_network_task --task tune --model_config </path/to/config.json> --epochs <n> --dataset_path </path/to/dataset_folder or .zip> --dataset_type <Layerwise/Voxelwise> --batch_size <n> --num_workers <n> <--join_channels |> --effective_batch_size <n> --normalization <linear0_1 | linear-1_1 | log_1e+3> --logs_path </path/to/store/logs> --mu_tr_file <path/to/used/mu_tr.txt> --enforce_voxel_resolution <W H D> --logger <mlflow | wandb> <--max_inner_batch_size <n> |> <--use_beam_parameters |> <--use_beam_parameters |> <--validate_gt |>
    ```

For a short description of each parameter please just call:
```bash
    python -m run_network_task --help
```

### Optional dependencies
- [WandB](https://wandb.ai/site/): Optional cloud-based logger.
- [mlflow](https://mlflow.org/): Optional local logger.
- [tcnn](https://github.com/NVlabs/tiny-cuda-nn): Highly optimized CUDA implementation of fully connected neural networks and encodings, like hashgrid or spherical harmonics.

## Datasets
- Datasets are located on Zenodo:
    - **[DS-01](https://xyz)**: Fixed H-100 cone beam; fixed distance
    - **[DS-02](https://xyz)**: Dynamic C-Arm spectra cone beam; fixed distance
    - **[DS-03](https://xyz)**: Dynamic C-Arm spectra rectangular beam; dynamic distance

## Getting started
### Using models
In order to load models, place the models configuration json together with the weights file, sharing the same basename in a folder. Just load the weights file to let the module search for the configuration to create the matching model.

### Adding models
Inherit from ``BaseNeuralRadFieldModel`` and set the ``__model_name__`` class attribute with a matching name. Make sure, that the file of the new model was imported before importing ``radfield3dnn.models`` to allow the model factory to access the new model defininition.
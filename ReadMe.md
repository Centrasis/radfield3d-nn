# Neural Model Training Framework

This framework is designed to train neural models using data loaded with RadFiled3D. It supports various neural network architectures and provides tools for monitoring and logging the training process. The framework leverages PyTorch and PyTorch Lightning for model training and includes internal modules like `ptbDataLab` and `NeuralDashboard` as submodules.

## Requirements

All required modules are listed in the `requirements.txt` file. Ensure you have all dependencies installed before running the training scripts.

## Submodules

- **ptbDataLab**: Added as a submodule to the root directory.
- **NeuralDashboard**: Added as a submodule to the root directory.

## Usage

1. Clone the repository and initialize submodules:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    git submodule update --init --recursive
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python train.py
    ```

4. To render from a trained model, use:
    ```bash
    python render_from_model.py --model_path <path-to-model-checkpoint> --model_name <model-name>
    ```

## Additional Information

- **Training Monitor**: Utilizes `NeuralDashboard` for monitoring training metrics and hardware usage.
- **Data Modules**: Supports `RadiationFieldDataModule` and `SpectraField3DDataModule` for loading training data.

For more detailed information, refer to the individual script files and their respective documentation.

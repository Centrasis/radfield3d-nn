# Neural Model Training Framework

## This is the code used to produce the results for the Paper: "Learning-based estimation of spatially resolved scatter radiation fields in interventional radiology".
## Please refer to the main branch for a clean installable and modular version of this code with several bug fixes worked in.

This framework is designed to train neural models using data loaded with RadFiled3D. It supports various neural network architectures and provides tools for monitoring and logging the training process. The framework leverages PyTorch and PyTorch Lightning for model training and includes internal modules like `ptbDataLab` and `NeuralDashboard` as submodules.

## Requirements

All required modules are listed in the `requirements.txt` file. Ensure you have all dependencies installed before running the training scripts.
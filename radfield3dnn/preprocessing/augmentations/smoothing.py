import torch
import torch.nn.functional as F
import random
from radfield3dnn.rftypes import TrainingInputData, RadiationFieldChannel, RadiationField
from RadFiled3D.pytorch.datasets.processing import DataProcessing


class GaussianSmoothing(DataProcessing):
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, strength: float = 0.5, p: float = 1.0, dataset_multiplier: float = 1.0, random_strength: bool = False):
        """
        Apply Gaussian smoothing to a tensor with a given strength.
        :param kernel_size: Size of the Gaussian kernel (must be odd).
        :param sigma: Standard deviation of the Gaussian kernel.
        :param strength: Strength of the smoothing (0.0 = no smoothing, 1.0 = full smoothing).
        :param p: Probability of applying the augmentation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.strength = strength
        self.p = p
        self._dataset_multiplier = dataset_multiplier
        self.random_strength = random_strength

    def apply_gaussian_smoothing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing to a tensor of shape (C, H, W, D) or (N, C, H, W, D).
        """
        strength = self.strength if not self.random_strength else random.uniform(0.0, 1.0)
        if strength <= 1e-6:
            return tensor

        if len(tensor.shape) == 4:
            tensor = tensor.unsqueeze(1)

        # Keep a reference for blending
        original = tensor

        channels = tensor.shape[1]
        
        # Create a Gaussian kernel
        kernel = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / self.sigma).pow(2))
        kernel = kernel / kernel.sum()
        kernel_3d = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
        kernel_3d = kernel_3d.to(tensor.device).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1, 1)

        # Apply Gaussian smoothing
        padding = self.kernel_size // 2
        smoothed_tensor = F.conv3d(tensor, kernel_3d, padding=padding, groups=channels)

        # Blend with original according to strength
        smoothed_tensor = original * (1.0 - strength) + smoothed_tensor * strength

        # Keep previous squeeze behavior
        smoothed_tensor = smoothed_tensor.squeeze(0)
        return smoothed_tensor

    def dataset_multiplier(self) -> float:
        return self._dataset_multiplier


class GaussianFluenceSmoothing(GaussianSmoothing):
    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply gaussian smoothing to input tensor of shape (N, C, H, W, D) in a random way for each sample.
        """
        if self.training and torch.rand(1).item() < self.p:
            scatter_field = x.ground_truth.scatter_field.flux
            direct_beam = x.ground_truth.direct_beam.flux

            scatter_field = self.apply_gaussian_smoothing(scatter_field)
            direct_beam = self.apply_gaussian_smoothing(direct_beam)
            
            x = TrainingInputData(
                input=x.input,
                ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=x.ground_truth.scatter_field.spectrum,
                        flux=scatter_field,
                        error=x.ground_truth.scatter_field.error
                    ),
                    direct_beam=RadiationFieldChannel(
                        spectrum=x.ground_truth.direct_beam.spectrum,
                        flux=direct_beam,
                        error=x.ground_truth.direct_beam.error
                    )
                ),
                original_ground_truth=x.original_ground_truth # keep original_ground_truth unchanged
            )
        return x


class GaussianSpectraSmoothing(GaussianSmoothing):
    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply gaussian smoothing to input tensor of shape (N, C, H, W, D) in a random way for each sample.
        """
        if self.training:
            scatter_field = x.ground_truth.scatter_field.spectrum
            direct_beam = x.ground_truth.direct_beam.spectrum

            scatter_field = self.apply_gaussian_smoothing(scatter_field)
            direct_beam = self.apply_gaussian_smoothing(direct_beam)
            
            x = TrainingInputData(
                input=x.input,
                ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=scatter_field,
                        flux=x.ground_truth.scatter_field.flux,
                        error=x.ground_truth.scatter_field.error
                    ),
                    direct_beam=RadiationFieldChannel(
                        spectrum=direct_beam,
                        flux=x.ground_truth.direct_beam.flux,
                        error=x.ground_truth.direct_beam.error
                    )
                )
            )
        return x

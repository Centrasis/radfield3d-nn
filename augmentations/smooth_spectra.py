import torch
from rftypes import TrainingInputData, RadiationFieldChannel, RadiationField
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from torch import nn


class SmoothingSpectra(DataProcessing):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, p: float = 1.0, dataset_multiplier: float = 1.0):
        """
        Smooth the spectra in the RadiationFieldChannel using a Gaussian kernel.
        :param kernel_size: Size of the Gaussian kernel (must be odd).
        :param sigma: Standard deviation of the Gaussian kernel.
        :param p: Probability of applying the augmentation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self._dataset_multiplier = dataset_multiplier

    def smooth_spectrum(self, spectrum: torch.Tensor, error: torch.Tensor = None) -> torch.Tensor:
        assert len(spectrum.shape) in [4, 5], f"Expected spectrum shape to be 4D or 5D, got {spectrum.shape}"

        has_batch_dim = len(spectrum.shape) == 5
        if error is not None:
            if error.ndim == spectrum.ndim - 1:
                error = error.unsqueeze(1) # Add channel dimension if missing
            if not has_batch_dim:
                assert len(error.shape) == len(spectrum.shape), "Error tensor shape must match spectrum spatial dimensions"
            else:
                assert len(error.shape) == len(spectrum.shape) and spectrum.shape[0] == error.shape[0], "Error tensor shape must match spectrum spatial dimensions"

        # calculate mean spectrum from all adjacient voxels
        
        if len(spectrum.shape) == 4:
            spectrum = spectrum.unsqueeze(0)
            if error is not None:
                error = error.unsqueeze(0)

        if error is not None:
            epsilon = 1e-8
            weights = 1.0 / (error.pow(2) + epsilon)
            weights = weights / weights.max()
            spectrum = spectrum * weights

        kernel_1d = torch.ones(self.kernel_size, device=spectrum.device)
        kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d).clone()
        center_index = self.kernel_size // 2
        kernel_3d[center_index, center_index, center_index] = 2.0

        kernel_3d /= kernel_3d.sum()
        num_channels = spectrum.shape[1]
        kernel = kernel_3d.expand(num_channels, 1, -1, -1, -1)
        smoothed_spectrum = nn.functional.conv3d(spectrum, kernel, padding=self.kernel_size // 2, groups=num_channels)

        # apply Gaussian smoothing to each voxel that spectrum sum is not zero and that the adjacent voxels sum is not zero
        spectrum_sum = torch.sum(spectrum, dim=1, keepdim=True)
        spectrum_sum = spectrum_sum.expand_as(smoothed_spectrum)
        smoothed_spectrum_sum = torch.sum(smoothed_spectrum, dim=1, keepdim=True)
        smoothed_spectrum_sum = smoothed_spectrum_sum.expand_as(smoothed_spectrum)
        smoothed_spectrum[smoothed_spectrum_sum != 0.0] = smoothed_spectrum[smoothed_spectrum_sum != 0.0] / smoothed_spectrum_sum[smoothed_spectrum_sum != 0.0]
        mask = (spectrum_sum > 0) & (smoothed_spectrum_sum > 0)
        smoothed_spectrum = torch.where(mask, smoothed_spectrum, spectrum)
        if not has_batch_dim:
            smoothed_spectrum = smoothed_spectrum.squeeze(0)

        smoothed_spectrum[smoothed_spectrum_sum == 0.0] = 1.0 / smoothed_spectrum.size(1)  # set to uniform if original spectrum was zero
        return smoothed_spectrum

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if torch.rand(1).item() > self.p:
            return x

        return TrainingInputData(
            input=x.input,
            ground_truth=RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=self.smooth_spectrum(x.ground_truth.scatter_field.spectrum, x.ground_truth.scatter_field.error),
                    flux=x.ground_truth.scatter_field.flux,
                    error=x.ground_truth.scatter_field.error
                ) if x.ground_truth.scatter_field is not None else None,
                direct_beam=RadiationFieldChannel(
                    spectrum=self.smooth_spectrum(x.ground_truth.direct_beam.spectrum, x.ground_truth.direct_beam.error),
                    flux=x.ground_truth.direct_beam.flux,
                    error=x.ground_truth.direct_beam.error
                ) if x.ground_truth.direct_beam is not None else None
            ),
            original_ground_truth=x.original_ground_truth # keep original_ground_truth unchanged
        )

    def dataset_multiplier(self) -> float:
        return self._dataset_multiplier
    
    def get_parameters(self) -> dict[str, float]:
        return {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma
        }

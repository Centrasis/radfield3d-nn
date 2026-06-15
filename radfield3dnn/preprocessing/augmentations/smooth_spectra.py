import torch
from radfield3dnn.rftypes import (
    TrainingInputData,
    RadiationFieldChannel,
    RadiationField,
    AirKermaField,
    rf3RadiationField,
    rf3TrainingInputData,
)
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from torch import nn
from typing import Union


class SmoothingSpectra(DataProcessing):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, p: float = 1.0, dataset_multiplier: float = 1.0):
        """
        Spatially smooth the per-voxel spectra by blending each voxel's spectrum with its
        spatial neighbours' spectra (Gaussian-weighted), pooling their photon statistics.

        Type-dispatched ``forward``:
          * ``RadiationField``  (scatter + direct beam): smooth each
            channel's spectrum independently.
          * ``RadiationFieldChannel`` (single channel, e.g. after
            ``ChannelsJoin``): smooth the channel's spectrum.
          * ``AirKermaField`` / anything without a spectrum: passthrough.

        :param kernel_size: Size of the Gaussian kernel (must be odd).
        :param sigma: Standard deviation of the Gaussian kernel.
        :param p: Probability of applying the augmentation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self._dataset_multiplier = dataset_multiplier

    def smooth_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Spatially smooth the per-voxel spectra by blending each voxel's histogram with its
        spatial neighbours spectra — pooling neighbouring photon statistics to denoise the
        per-voxel spectrum estimate. Each energy bin is convolved over the (D, H, W) axes with a
        depthwise 3D Gaussian kernel, then the result is renormalised per voxel so it stays a
        unit-sum histogram. The energy-bin shape of any single voxel is never blurred.

        Accepts ``(C, D, H, W)`` or ``(B, C, D, H, W)`` and returns the same shape.
        """
        assert spectrum.ndim in (4, 5), \
            f"Expected spectrum shape to be 4D or 5D, got {tuple(spectrum.shape)}"
        has_batch_dim = spectrum.ndim == 5
        if not has_batch_dim:
            spectrum = spectrum.unsqueeze(0)  # (1, C, D, H, W)
        B, C, D, H, W = spectrum.shape

        # 3D Gaussian kernel over the spatial axes, sum(k) = 1; applied depthwise (one per energy bin).
        half = self.kernel_size // 2
        coords = torch.arange(self.kernel_size, dtype=spectrum.dtype, device=spectrum.device) - half
        g = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        k3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        k3d = k3d / k3d.sum()
        kernel = k3d.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1, 1)

        # Mark voxels that are non-finite (e.g. ``-inf`` from the importance sampler) so we don't
        # poison the conv with them; we'll restore the original at those positions at the end.
        voxel_finite = torch.isfinite(spectrum).all(dim=1, keepdim=True)  # (B, 1, D, H, W)
        spectrum_clean = torch.where(
            voxel_finite.expand_as(spectrum),
            spectrum,
            torch.zeros_like(spectrum),
        )

        # Depthwise 3D convolution: each voxel's spectrum becomes a Gaussian-weighted average of its
        # spatial neighbours' spectra (masked neighbours contribute 0).
        smoothed = nn.functional.conv3d(spectrum_clean, kernel, padding=half, groups=C)

        # Renormalise per voxel so the histogram still sums to 1. Empty voxels (no neighbouring
        # counts) get a uniform fallback so HistogramLoss downstream doesn't divide by zero.
        s = smoothed.sum(dim=1, keepdim=True)
        nonempty = s > 1e-8
        smoothed = torch.where(
            nonempty.expand_as(smoothed),
            smoothed / s.clamp(min=1e-8),
            torch.full_like(smoothed, 1.0 / C),
        )

        # Restore original at non-finite voxels so downstream loss/metric masking still sees them as
        # invalid (HistogramLoss masks non-finite; ChannelsJoin's assertion fires before this stage).
        if not voxel_finite.all():
            smoothed = torch.where(
                voxel_finite.expand_as(smoothed),
                smoothed,
                spectrum,
            )

        if not has_batch_dim:
            smoothed = smoothed.squeeze(0)
        return smoothed

    def _smooth_channel(self, ch: RadiationFieldChannel) -> RadiationFieldChannel:
        """Smooth the spectrum of a single ``RadiationFieldChannel``.
        Channels without a spectrum (rare; e.g. flux-only intermediates)
        pass through unchanged."""
        if ch is None or ch.spectrum is None:
            return ch
        return RadiationFieldChannel(
            spectrum=self.smooth_spectrum(ch.spectrum),
            flux=ch.flux,
            error=ch.error,
        )

    def _smooth_ground_truth(self, gt: Union[RadiationField, rf3RadiationField, RadiationFieldChannel, AirKermaField]):
        if isinstance(gt, (RadiationField, rf3RadiationField)):
            # Two-channel: smooth each channel independently. `geometry`
            # is carried through; some rf3 variants may not expose it.
            return RadiationField(
                scatter_field=self._smooth_channel(gt.scatter_field),
                direct_beam=self._smooth_channel(gt.direct_beam),
                geometry=getattr(gt, "geometry", None),
            )
        if isinstance(gt, RadiationFieldChannel):
            # Single-channel (e.g. post-`ChannelsJoin`): smooth in-place.
            return self._smooth_channel(gt)
        # AirKermaField or anything else without a spectrum → passthrough.
        return gt

    def forward(self, x: Union[TrainingInputData, RadiationField, RadiationFieldChannel, AirKermaField]):
        if torch.rand(1).item() > self.p:
            return x

        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self._smooth_ground_truth(x.ground_truth),
                # original_ground_truth carries the un-augmented reference
                # so downstream consumers (loss-with-original, plotters)
                # see the raw target; never smooth it.
                original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None,
            )
        # Bare ground-truth payload — smooth directly.
        return self._smooth_ground_truth(x)

    def dataset_multiplier(self) -> float:
        return self._dataset_multiplier
    
    def get_parameters(self) -> dict[str, float]:
        return {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma
        }

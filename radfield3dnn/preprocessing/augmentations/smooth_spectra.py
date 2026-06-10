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
        Smooth the spectra in the ground truth using a Gaussian kernel.

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

    def smooth_spectrum(self, spectrum: torch.Tensor, error: torch.Tensor = None) -> torch.Tensor:
        """1D Gaussian smoothing over the energy-bin axis of a per-voxel
        histogram.

        Reduces bin-to-bin Monte Carlo discretisation noise within each
        voxel's spectrum *without* mixing across voxels, so the
        position-dependent spectrum shape the network is trying to learn
        is preserved.

        Accepts ``(C, D, H, W)`` or ``(B, C, D, H, W)``; C is the
        histogram bin axis. Returns the same shape, unit-summed per
        voxel (boundary bins lose Gaussian-tail mass to the zero-pad,
        which the renormalise step fixes).

        The previous implementation applied a 3D box filter across the
        *spatial* axes per bin AND multiplied by ``1/error²`` weights
        before the conv. On DS03 (where MC error anti-correlates with
        flux at ρ ≈ −0.6) that effectively copied beam-core spectra
        into scatter-halo voxels and destroyed the position→spectrum
        mapping the spectrum head is trying to learn — observable as a
        flat ``val_scatter_spectrum_loss`` plateau at ~0.218 in run
        ``dg1k61rm``. The new implementation does the operation the
        name and docstring originally promised: a 1D Gaussian over
        the bin axis.

        ``error`` is accepted for signature compatibility with the old
        callers but ignored. A correct error-weighted variant would
        use ``conv(spec * w) / conv(w)`` (proper inverse-variance
        average), not the destructive ``spec * w`` pre-multiply the old
        code used — that's a separate change to consider if you want
        denoising sensitive to MC variance.
        """
        assert spectrum.ndim in (4, 5), \
            f"Expected spectrum shape to be 4D or 5D, got {tuple(spectrum.shape)}"
        has_batch_dim = spectrum.ndim == 5
        if not has_batch_dim:
            spectrum = spectrum.unsqueeze(0)  # (1, C, D, H, W)
        B, C, D, H, W = spectrum.shape

        # 1D Gaussian kernel over the bin axis, ∑k = 1.
        half = self.kernel_size // 2
        coords = torch.arange(self.kernel_size, dtype=spectrum.dtype, device=spectrum.device) - half
        k = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        k = k / k.sum()

        # Mark voxels that are non-finite (e.g. ``-inf`` from
        # ErrorbasedImportanceSampler) so we don't poison the conv with
        # them; we'll restore the original at those positions at the end.
        voxel_finite = torch.isfinite(spectrum).all(dim=1, keepdim=True)  # (B, 1, D, H, W)
        spectrum_clean = torch.where(
            voxel_finite.expand_as(spectrum),
            spectrum,
            torch.zeros_like(spectrum),
        )

        # conv1d wants (N, C_in, L). We want to slide over the *bin*
        # axis, so move bins to the end and merge (B, D, H, W) into the
        # batch dim. Kernel is (out=1, in=1, kernel_size).
        flat = spectrum_clean.permute(0, 2, 3, 4, 1).reshape(B * D * H * W, 1, C)
        kernel = k.view(1, 1, -1)
        smoothed_flat = nn.functional.conv1d(flat, kernel, padding=half)
        smoothed = smoothed_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # Renormalise per voxel so the histogram still sums to 1
        # (boundary bins lose mass to the zero-pad). Empty voxels
        # (original sum ~ 0) get a uniform fallback so HistogramLoss
        # downstream doesn't divide by zero.
        s = smoothed.sum(dim=1, keepdim=True)
        nonempty = s > 1e-8
        smoothed = torch.where(
            nonempty.expand_as(smoothed),
            smoothed / s.clamp(min=1e-8),
            torch.full_like(smoothed, 1.0 / C),
        )

        # Restore original at non-finite voxels so downstream loss/metric
        # masking still sees them as invalid (HistogramLoss masks
        # non-finite, ChannelsJoin's assertion fires before this stage).
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
            spectrum=self.smooth_spectrum(ch.spectrum, ch.error),
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

import torch
from radfiled3d.pytorch.datasets.processing import DataProcessing

from radfield3dnn.roi import BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
from radfield3dnn.rftypes import (
    RadiationField, RadiationFieldChannel, TrainingInputData, rf3RadiationField,
    rf3TrainingInputData,
)


class EmptyVoxelSpectrumFill(DataProcessing):
    """Give signal-free voxels the field's mean scatter spectrum instead of their own.

    A voxel below ``threshold_rel`` of the field's peak flux carries no usable spectral
    information: in Monte-Carlo data it holds a handful of counts, so its 32-bin histogram is
    noise; in the analytical mockup it is a ray-traced value whose spectrum is meaningless at that
    magnitude. Either way the network is asked to reproduce a per-voxel histogram that encodes
    nothing, which spends spectrum-head capacity on noise.

    The replacement is the FLUX-WEIGHTED mean spectrum of that field's scatter ROI (above the
    empty cut, below the beam cut) — the physically right prior for a voxel whose photons got
    there by scattering. Weighting by flux keeps the mean dominated by voxels that actually
    carry signal.

    Per FIELD, never per batch: two fields in a batch have different tube spectra and different
    peak fluxes, so a batch-global mean would mix them.

    Spectra are per-voxel normalised to sum 1, and the weighted mean of normalised histograms is
    itself normalised — the invariant every downstream consumer (HistogramLoss, the air-kerma
    metric) relies on holds without renormalising.
    """

    def __init__(self, threshold_rel: float = SCATTER_LO_DEFAULT,
                 beam_rel: float = BEAM_REL_DEFAULT):
        super().__init__()
        self.threshold_rel = float(threshold_rel)
        self.beam_rel = float(beam_rel)

    @staticmethod
    def _spatial_amax(x: torch.Tensor) -> torch.Tensor:
        """Per-field peak over the trailing spatial dims, keepdim — matches roi.py's convention
        so this cut and the ROI sampler's agree on what 'the field's maximum' means."""
        dims = tuple(range(x.ndim - 3, x.ndim))
        return x.amax(dim=dims, keepdim=True)

    def _fill_channel(self, ch: RadiationFieldChannel) -> RadiationFieldChannel:
        if ch is None or ch.flux is None or ch.spectrum is None:
            return ch
        flux, spectrum = ch.flux, ch.spectrum
        peak = self._spatial_amax(flux).clamp(min=torch.finfo(flux.dtype).tiny)
        empty = flux < self.threshold_rel * peak
        scatter = (~empty) & (flux < self.beam_rel * peak)

        # Flux-weighted mean spectrum over the scatter ROI, per field. Sum over the spatial dims
        # only, so the bin axis (dim 1) survives.
        dims = tuple(range(flux.ndim - 3, flux.ndim))
        w = torch.where(scatter, flux, torch.zeros_like(flux))
        wsum = w.sum(dim=dims, keepdim=True)
        mean_spectrum = (spectrum * w).sum(dim=dims, keepdim=True) / wsum.clamp(min=torch.finfo(flux.dtype).tiny)

        # A field with no scatter ROI at all (everything beam or everything empty) has no prior to
        # apply — leave it untouched rather than writing zeros.
        has_scatter = wsum > 0
        filled = torch.where(empty & has_scatter, mean_spectrum, spectrum)
        return RadiationFieldChannel(spectrum=filled, flux=flux, error=ch.error)

    def _fill(self, gt):
        if isinstance(gt, (RadiationField, rf3RadiationField)):
            return RadiationField(
                scatter_field=self._fill_channel(gt.scatter_field),
                direct_beam=self._fill_channel(gt.direct_beam),
                geometry=getattr(gt, "geometry", None),
            )
        if isinstance(gt, RadiationFieldChannel):
            return self._fill_channel(gt)
        return gt

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if not isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            raise ValueError("EmptyVoxelSpectrumFill expects TrainingInputData")
        with torch.no_grad():
            new_gt = self._fill(x.ground_truth)
        return TrainingInputData(
            input=x.input,
            ground_truth=new_gt,
            # the scored reference stays the field as simulated
            original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None,
        )

    def get_parameters(self) -> dict[str, float]:
        return {"threshold_rel": self.threshold_rel, "beam_rel": self.beam_rel}

    @staticmethod
    def create_from_config(config: dict) -> "EmptyVoxelSpectrumFill":
        return EmptyVoxelSpectrumFill(
            threshold_rel=config.get("threshold_rel", SCATTER_LO_DEFAULT),
            beam_rel=config.get("beam_rel", BEAM_REL_DEFAULT),
        )

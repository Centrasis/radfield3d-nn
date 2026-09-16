from .base import MetricBase
from torch import Tensor
import torch
from typing import Union, Literal


class HistogramOverlapAccuracy(MetricBase):
    def __init__(self, epsilon: float = 1e-8, weight_with_error: bool = False, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean'):
        super().__init__(layer_name='spectrum', reduction=reduction, weight_with_error=weight_with_error, eps=epsilon)

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        # PER-VOXEL histogram intersection-over-union: reduce over the BIN axis only, keep every
        # voxel. The previous implementation flattened via boolean indexing first, which collapsed
        # `sum(dim=())` over EVERYTHING — one global flux-weighted scalar that could not see
        # per-voxel spectral shape errors at all (swapping the spectra of two equal-flux voxels
        # left it unchanged).
        bin_axis = 1 if target.ndim >= 3 else 0   # (B, bins, ...) batched, (bins, ...) unbatched
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        # a voxel counts only when its WHOLE histogram is valid — partially masked histograms are
        # excluded voxels (ROI wrappers / geometry exclusion mark them via -inf bins)
        voxel_valid = valid.all(dim=bin_axis)
        if not voxel_valid.any():
            return torch.zeros(0, device=target.device, dtype=target.dtype)

        t = torch.where(valid, target, torch.zeros_like(target)) + self.eps
        p = torch.where(valid, prediction, torch.zeros_like(prediction)) + self.eps
        minimum = torch.min(p, t)
        intersection = minimum.sum(dim=bin_axis)
        union = (p + t - minimum).sum(dim=bin_axis)
        iou = intersection / union
        # excluded voxels -> -inf; MetricBase.forward's finite-filter drops them before reduction
        iou = torch.where(voxel_valid, iou, torch.full_like(iou, -torch.inf))
        return iou.view(-1) if self.weight_with_error else self.reduction_fn(iou.view(-1)[voxel_valid.view(-1)])


class SpectrumROIAccuracy(MetricBase):
    """Histogram-overlap spectrum accuracy restricted to a dose ROI — the spectrum is only physically
    defined where there is radiation, so scoring it over the near-empty floor dilutes the number.
    ``roi='scatter'`` scores the shared scatter ROI (matches AirkermaScatterAccuracy(use_roi=True) /
    radfield3dnn.roi); ``roi='top90'`` scores the top-90% air-kerma voxels (matches
    top90_airkerma_accuracy, importance_threshold=0.1). Out-of-ROI voxels get spectrum -inf so the
    wrapped HistogramOverlapAccuracy drops them — the value is the SAME functional as
    spectrum_accuracy, just over the ROI."""

    def __init__(self, roi: Literal['scatter', 'top90'], mu_tr_file: str = None, spectra_bins: int = 32,
                 max_energy_eV: float = 1.5e+5, beam_rel: float = None, scatter_lo: float = None,
                 top_threshold: float = 0.1):
        super().__init__(layer_name='spectrum')
        from radfield3dnn.roi import BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
        assert roi in ('scatter', 'top90'), f"Unknown roi {roi!r}"
        self.roi = roi
        self.beam_rel = float(beam_rel if beam_rel is not None else BEAM_REL_DEFAULT)
        self.scatter_lo = float(scatter_lo if scatter_lo is not None else SCATTER_LO_DEFAULT)
        self.top_threshold = float(top_threshold)
        self._iou = HistogramOverlapAccuracy()
        if roi == 'top90':
            from radfield3dnn.preprocessing.airkerma import Airkerma
            self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)

    @staticmethod
    def _squeeze_trailing(x: Tensor) -> Tensor:
        return x.squeeze(-1) if x.shape[-1] == 1 else x

    def _roi_keep_mask(self, target, input) -> Tensor:
        from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel
        from radfield3dnn.roi import compute_roi_masks, _spatial_amax
        if self.roi == 'scatter':
            gt = input.original_ground_truth if input.original_ground_truth is not None else input.ground_truth
            assert isinstance(gt, RadiationField) and gt.direct_beam is not None and gt.scatter_field is not None, \
                "SpectrumROIAccuracy(roi='scatter') needs the two-channel ground truth (direct_beam + scatter_field)."
            xgt = self._squeeze_trailing(gt.direct_beam.flux)
            sgt = self._squeeze_trailing(gt.scatter_field.flux)
            _, scatter, _ = compute_roi_masks(xgt, sgt + xgt, self.beam_rel, self.scatter_lo)
            return scatter
        # top90: keep voxels whose GT air-kerma is >= top_threshold * per-field max.
        ak = self._squeeze_trailing(self.airkerma.forward(target.spectrum, target.flux))
        return ak >= self.top_threshold * _spatial_amax(ak)

    def forward(self, target, prediction, input=None) -> Tensor:
        from radfield3dnn.rftypes import RadiationFieldChannel
        if isinstance(prediction, RadiationFieldChannel) and prediction.spectrum is None:
            return None
        keep = self._roi_keep_mask(target, input)
        non_roi = ~keep

        def masked(field):
            spec = field.spectrum
            nr = non_roi
            # insert the missing BIN axis where it lives: position 1 (after batch) for a batched
            # (B, bins, D, H, W) spectrum, position 0 for an unbatched (bins, D, H, W) one.
            # (The previous trailing unsqueeze built (B, D, H, W, 1) and expanded onto the wrong
            # axis / failed outright.)
            if nr.ndim == spec.ndim - 1:
                nr = nr.unsqueeze(1 if spec.ndim >= 5 else 0)
            assert nr.ndim == spec.ndim, f"ROI mask rank {nr.ndim} does not match spectrum rank {spec.ndim}"
            spec = torch.where(nr.expand_as(spec), torch.full_like(spec, float('-inf')), spec)
            return field._replace(spectrum=spec)

        return self._iou.forward(masked(target), masked(prediction), input)

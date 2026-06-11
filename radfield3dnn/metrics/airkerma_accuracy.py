from radfield3dnn.rftypes import AirKermaField, RadiationFieldChannel, TrainingInputData, RadiationField
from radfield3dnn.preprocessing.airkerma import Airkerma
from .smape import SMAPEAccuracy, EnergyWeightedSMAPEAccuracy
from .base import MetricBase
from .ncc import NCCAccuracy
from typing import Union, Literal
from torch import Tensor
import torch
from .log_rmse import LogRMSEAccuracy
from .gpr import GammaPassingRate


class AirkermaAccuracy(MetricBase):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, importance_threshold: float = 0.0, keep_dim: bool = False, metric_type: Union[Literal['smape'], Literal['log_rmse'], Literal['ncc'], Literal['gpr']] = 'smape', voxel_size_m: float = 0.01, rel_dose_diff: float = 0.03, dist_crit_mm: float = 3.0):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=5e-9, importance_threshold=importance_threshold, keep_dim=keep_dim)
        elif metric_type == 'log_rmse':
            self.metric = LogRMSEAccuracy(layer_name=None, weight_with_error=weight_with_error, importance_threshold=importance_threshold, keep_dim=keep_dim)
        elif metric_type == 'ncc':
            self.metric = NCCAccuracy(layer_name=None, reduction='mean', weight_with_error=weight_with_error, importance_threshold=importance_threshold)
        elif metric_type == 'gpr':
            self.metric = GammaPassingRate(layer_name=None, reduction='mean', weight_with_error=weight_with_error, keep_dim=keep_dim, voxel_size_m=voxel_size_m, rel_dose_diff=rel_dose_diff, dist_crit_mm=dist_crit_mm)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        return self.metric._calc_metric(target, prediction)

    def forward(self, target: Union[RadiationFieldChannel, Tensor, AirKermaField], prediction: Union[RadiationFieldChannel, Tensor, AirKermaField], input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None

        # Compute air kerma without eps clamping; only enforce non-negativity on flux
        if isinstance(target, RadiationFieldChannel):
            target_airkerma = self.airkerma.forward(target.spectrum, target.flux)
        elif isinstance(target, AirKermaField):
            target_airkerma = target.air_kerma
        else:
            target_airkerma = target
        
        if isinstance(prediction, RadiationFieldChannel):
            prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.flux)
        elif isinstance(prediction, AirKermaField):
            prediction_airkerma = prediction.air_kerma
        else:
            prediction_airkerma = prediction

        return self.metric.forward(target_airkerma, prediction_airkerma, input)
    

class AirkermaAccuracyEnergyWeighted(EnergyWeightedSMAPEAccuracy):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, importance_threshold: float = 0.0, keep_dim: bool = False):
        super().__init__(layer_name=None, weight_with_error=weight_with_error, importance_threshold=importance_threshold, keep_dim=keep_dim, clamp=True, zero_eps=5e-9)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)

    def forward(self, target: Union[RadiationFieldChannel, Tensor, AirKermaField], prediction: Union[RadiationFieldChannel, Tensor, AirKermaField], input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None

        # Compute air kerma without eps clamping; only enforce non-negativity on flux
        if isinstance(target, RadiationFieldChannel):
            target_airkerma = self.airkerma.forward(target.spectrum, target.flux)
        elif isinstance(target, AirKermaField):
            target_airkerma = target.air_kerma
        else:
            target_airkerma = target
        
        if isinstance(prediction, RadiationFieldChannel):
            prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.flux)
        elif isinstance(prediction, AirKermaField):
            prediction_airkerma = prediction.air_kerma
        else:
            prediction_airkerma = prediction

        # Delegate to SMAPEAccuracy: will ignore trivial denominators via its eps and return accuracy in [0,1] (clamp=True)
        return super().forward(target_airkerma, prediction_airkerma, input)


class AirkermaRelDifferencesStdDev(MetricBase):
    """
    Mean standard deviation of relative air kerma errors over the spatial volume.
    Relative error: (prediction - target) / target (target clamped to eps).
    Returns a single scalar (mean over batch).
    """
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, zero_eps: float = 5e-9, weight_with_error: bool = False, importance_threshold: float = 0.0, metric_type: Union[Literal['smape'], Literal['log_rmse'], Literal['ncc']] = 'smape'):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=zero_eps, importance_threshold=importance_threshold, keep_dim=True)
        elif metric_type == 'log_rmse':
            self.metric = LogRMSEAccuracy(layer_name=None, weight_with_error=weight_with_error, importance_threshold=importance_threshold, keep_dim=True)
        elif metric_type == 'ncc':
            self.metric = NCCAccuracy(layer_name=None, reduction='mean', weight_with_error=weight_with_error, importance_threshold=importance_threshold)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
    def _calc_metric(self, target, prediction):
        return self.metric._calc_metric(target, prediction)

    def forward(self, target: Union[RadiationFieldChannel, Tensor, AirKermaField], prediction: Union[RadiationFieldChannel, Tensor, AirKermaField], input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None

        # Compute air kerma without eps clamping (only clamp negatives to 0 for flux)
        if isinstance(target, RadiationFieldChannel):
            target_airkerma = self.airkerma.forward(target.spectrum, target.flux)
        elif isinstance(target, AirKermaField):
            target_airkerma = target.air_kerma
        else:
            target_airkerma = target
        
        if isinstance(prediction, RadiationFieldChannel):
            prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.flux)
        elif isinstance(prediction, AirKermaField):
            prediction_airkerma = prediction.air_kerma
        else:
            prediction_airkerma = prediction

        # True relative errors; clamp only denominator
        errors = self.metric._calc_metric(target_airkerma, prediction_airkerma)
        valid_mask = torch.isfinite(errors)
        errors = errors[valid_mask]

        # Reduce: std over spatial+channel dims (exclude batch dim=0), then mean over batch
        std_per_sample = errors.std(unbiased=False)
        std_per_sample = torch.nan_to_num(std_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        result = std_per_sample.mean(dim=0)
        return result


class AirkermaSphereAccuracy(MetricBase):
    """
    Accuracy metric for airkerma on the surface of a sphere around the center of the volume.
    All voxels overlapping the surface of the sphere are considered.
    """
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, sphere_radius_m: float, voxel_size_m: float, weight_with_error: bool = False, metric_type: Union[Literal['smape'], Literal['log_rmse'], Literal['ncc']] = 'smape', importance_threshold: float = 0.0, keep_dim: bool = False):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        self.sphere_radius_m = sphere_radius_m
        self.voxel_size_m = voxel_size_m
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=5e-9, importance_threshold=importance_threshold, keep_dim=keep_dim)
        elif metric_type == 'log_rmse':
            self.metric = LogRMSEAccuracy(layer_name=None, weight_with_error=weight_with_error, importance_threshold=importance_threshold, keep_dim=keep_dim)
        elif metric_type == 'ncc':
            self.metric = NCCAccuracy(layer_name=None, reduction='mean', weight_with_error=weight_with_error, importance_threshold=importance_threshold)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        return self.metric._calc_metric(target, prediction)

    def forward(self, target: Union[RadiationFieldChannel, Tensor], prediction: Union[RadiationFieldChannel, Tensor], input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None

        # Compute air kerma without eps clamping; only enforce non-negativity on flux
        if isinstance(target, RadiationFieldChannel):
            target_airkerma = self.airkerma.forward(target.spectrum, target.flux)
        elif isinstance(target, AirKermaField):
            target_airkerma = target.air_kerma
        else:
            target_airkerma = target
        
        if isinstance(prediction, RadiationFieldChannel):
            prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.flux)
        elif isinstance(prediction, AirKermaField):
            prediction_airkerma = prediction.air_kerma
        else:
            prediction_airkerma = prediction

        B, C, D, H, W = target_airkerma.shape
        device = target_airkerma.device
        center = torch.tensor([D / 2, H / 2, W / 2], device=device).view(1, 3)
        grid_d = torch.arange(D, device=device).view(1, D, 1, 1).expand(B, D, H, W)
        grid_h = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, D, H, W)
        grid_w = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, D, H, W)
        grid = torch.stack((grid_d, grid_h, grid_w), dim=-1).float()
        distances = torch.sqrt(torch.sum((grid - center) ** 2, dim=-1)) * self.voxel_size_m
        sphere_mask = (distances >= (self.sphere_radius_m - (self.voxel_size_m / 2))) & (distances <= (self.sphere_radius_m + (self.voxel_size_m / 2)))
        sphere_mask = sphere_mask.unsqueeze(1)
        target_airkerma = target_airkerma[sphere_mask]
        prediction_airkerma = prediction_airkerma[sphere_mask]
        
        accuracy = self.metric._calc_metric(target_airkerma, prediction_airkerma)
        return accuracy.mean()


class AirkermaSupervoxelScatterAccuracy(AirkermaAccuracy):
    """Scatter accuracy on SUPERVOXEL-aggregated air-kerma (sv³ voxels summed = dose in a volume).

    The per-voxel scatter metrics are MC-noise-bounded: the bulk's relative noise is ~400% median,
    so even a PERFECT model caps at ≈0.66 per-voxel. Aggregating sv³ voxels divides the noise by
    ~sv^{3/2} (sv=8 → ÷22, bulk rel-noise → ~20%, perfect-model ceiling ≈0.92), which is where the
    >0.85–0.9 accuracy goal (10–15% uncertainty in the occupied region) is physically measurable —
    and it is the clinically meaningful quantity (dose accumulated in a shoulder-sized volume),
    the same philosophy as the gamma pass-rate's distance tolerance.

    Mask: supervoxels whose direct-beam content exceeds ``max_relative_flux`` of the direct max are
    excluded (the beam is scored by top90/gamma); everything else is scored.
    """

    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, supervoxel: int = 8,
                 max_relative_flux: float = 5e-2, weight_with_error: bool = False):
        super().__init__(mu_tr_file, spectra_bins, max_energy_eV, weight_with_error,
                         importance_threshold=0.0, keep_dim=False, metric_type='smape')
        self.sv = int(supervoxel)
        self.max_relative_flux = float(max_relative_flux)

    def _pool(self, vol: Tensor) -> Tensor:
        """Sum-pool the spatial dims by sv (crop the remainder)."""
        v = vol
        while v.dim() < 5:
            v = v.unsqueeze(0)
        D, H, W = v.shape[-3:]
        d, h, w = D // self.sv, H // self.sv, W // self.sv
        v = v[..., :d * self.sv, :h * self.sv, :w * self.sv]
        return torch.nn.functional.avg_pool3d(v, self.sv) * (self.sv ** 3)

    def forward(self, target, prediction, input: TrainingInputData = None) -> Tensor:
        # air-kerma volumes of GT and prediction
        t_ak = self.airkerma.forward(target.spectrum, target.flux) if isinstance(target, RadiationFieldChannel) \
            else (target.air_kerma if isinstance(target, AirKermaField) else target)
        p_ak = self.airkerma.forward(prediction.spectrum, prediction.flux) if isinstance(prediction, RadiationFieldChannel) \
            else (prediction.air_kerma if isinstance(prediction, AirKermaField) else prediction)
        t_sv, p_sv = self._pool(t_ak), self._pool(p_ak)

        # beam exclusion from the (unjoined) direct channel, pooled with the same kernel
        xgt = None
        if input is not None:
            src = input.original_ground_truth if getattr(input, "original_ground_truth", None) is not None else input.ground_truth
            if isinstance(src, RadiationField) and src.direct_beam is not None:
                xgt = src.direct_beam.flux
        if xgt is not None:
            x_sv = self._pool(xgt)
            beam_sv = x_sv > x_sv.max() * self.max_relative_flux
            t_sv = t_sv.masked_fill(beam_sv, -torch.inf)
            p_sv = p_sv.masked_fill(beam_sv, -torch.inf)

        return self.metric.forward(t_sv, p_sv, input)


class AirkermaScatterAccuracy(AirkermaAccuracy):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, keep_dim: bool = False, max_relative_flux: float = 5e-2, min_relative_flux: float = 5e-3, metric_type: Union[Literal['smape'], Literal['log_rmse'], Literal['ncc']] = 'smape',
                 use_error: bool = True, error_threshold: float = 0.5):
        """Scatter-field air-kerma accuracy (beam excluded). Two low-signal masking modes:

        * ``use_error=True`` (DEFAULT — THE scatter-volume definition, the same volume the
          loss-normalizer study + MC-floor analysis visualize): voxels whose **per-voxel MC
          statistical error** marks the GT as noise-dominated (``error >= error_threshold``;
          the DS03 error layer is ~binary {0,1}) are excluded. This scores the REAL scatter
          field — ~86–91% of DS03 voxels — without rewarding/punishing the unfittable MC noise.
        * ``use_error=False`` (LEGACY — the published bright-ring definition): voxels below
          ``min_relative_flux`` of the total-flux max are excluded — only the bright-scatter
          ring (~3% of DS03 voxels) scores. Kept for comparability with the published 0.84.

        The beam exclusion (direct > ``max_relative_flux``·direct_max) applies in both modes.
        """
        super().__init__(mu_tr_file, spectra_bins, max_energy_eV, weight_with_error, importance_threshold=0.0, keep_dim=keep_dim, metric_type=metric_type)
        self.max_relative_flux = max_relative_flux
        self.min_relative_flux = min_relative_flux
        self.use_error = bool(use_error)
        self.error_threshold = float(error_threshold)

    def forward(self, target: Union[RadiationFieldChannel, AirKermaField, Tensor], prediction: Union[RadiationFieldChannel, AirKermaField, Tensor], input: TrainingInputData = None) -> Tensor:
        assert (isinstance(input.ground_truth, RadiationField) and input.ground_truth.direct_beam is not None) or (input.original_ground_truth is not None and input.original_ground_truth.direct_beam is not None), "Input TrainingInputData must contain direct_beam for scatter field accuracy."

        xgt_ch = input.original_ground_truth.direct_beam if input.original_ground_truth is not None and input.original_ground_truth.direct_beam is not None else input.ground_truth.direct_beam
        xgt = xgt_ch.flux if isinstance(xgt_ch, RadiationFieldChannel) else xgt_ch
        sgt_ch = input.original_ground_truth.scatter_field if input.original_ground_truth is not None and input.original_ground_truth.scatter_field is not None else input.ground_truth.scatter_field
        sgt = sgt_ch.flux if isinstance(sgt_ch, RadiationFieldChannel) else sgt_ch
        fgt = sgt + xgt

        beam_mask = xgt > xgt.max() * self.max_relative_flux  # ignore areas with > max_relative_flux of max primary flux

        scatter_error = sgt_ch.error if isinstance(sgt_ch, RadiationFieldChannel) else None
        if self.use_error and scatter_error is not None:
            # Noise-aware mode: exclude voxels where the MC statistical error marks the GT itself
            # as noise-dominated; everything else (the real, reliable scatter field) is scored.
            noise_mask = scatter_error >= self.error_threshold
            if noise_mask.shape != fgt.shape and noise_mask.numel() == fgt.numel():
                noise_mask = noise_mask.reshape(fgt.shape)
            beam_mask = beam_mask | noise_mask
        else:
            if self.use_error:
                print("[yellow]AirkermaScatterAccuracy(use_error=True) but no scatter error layer present — falling back to the flux-threshold mask.[/yellow]")
            low_flux_mask_gt = fgt < fgt.max() * self.min_relative_flux  # ignore areas with < min_relative_flux of max total flux
            if isinstance(prediction, RadiationFieldChannel):
                low_flux_mask = prediction.flux < prediction.flux.max() * self.min_relative_flux  # ignore areas with < min_relative_flux of max predicted flux
            elif isinstance(prediction, AirKermaField):
                low_flux_mask = prediction.air_kerma < prediction.air_kerma.max() * self.min_relative_flux  # ignore areas with < min_relative_flux of max predicted air kerma
            else:
                low_flux_mask = prediction < prediction.max() * self.min_relative_flux  # ignore areas with < min_relative_flux of max predicted value
            beam_mask = beam_mask | (low_flux_mask & low_flux_mask_gt)  # combine masks

        if isinstance(target, RadiationFieldChannel):
            target.flux[beam_mask] = -torch.inf
        elif isinstance(target, AirKermaField):
            target.air_kerma[beam_mask] = -torch.inf
        else:
            target[beam_mask] = -torch.inf


        if isinstance(prediction, RadiationFieldChannel):
            prediction.flux[beam_mask] = -torch.inf
        elif isinstance(prediction, AirKermaField):
            prediction.air_kerma[beam_mask] = -torch.inf
        else:
            prediction[beam_mask] = -torch.inf

        return super().forward(target, prediction, input)

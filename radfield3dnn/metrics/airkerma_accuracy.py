from radfield3dnn import AirKermaField, RadiationFieldChannel, TrainingInputData, RadiationField
from radfield3dnn.preprocessing.airkerma import Airkerma
from .smape import SMAPEAccuracy, EnergyWeightedSMAPEAccuracy
from .base import MetricBase
from typing import Union, Literal
from torch import Tensor
import torch
from .gpr import GammaPassingRate


class AirkermaAccuracy(MetricBase):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, importance_threshold: float = 0.0, keep_dim: bool = False, metric_type: Union[Literal['smape'], Literal['gpr']] = 'smape', voxel_size_m: float = 0.01, rel_dose_diff: float = 0.03, dist_crit_mm: float = 3.0):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=5e-9, importance_threshold=importance_threshold, keep_dim=keep_dim)
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
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, zero_eps: float = 5e-9, weight_with_error: bool = False, importance_threshold: float = 0.0, metric_type: Union[Literal['smape']] = 'smape'):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=zero_eps, importance_threshold=importance_threshold, keep_dim=True)
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
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, sphere_radius_m: float, voxel_size_m: float, weight_with_error: bool = False, metric_type: Union[Literal['smape']] = 'smape', importance_threshold: float = 0.0, keep_dim: bool = False):
        super().__init__(layer_name=None, weight_with_error=weight_with_error)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        self.sphere_radius_m = sphere_radius_m
        self.voxel_size_m = voxel_size_m
        if metric_type == 'smape':
            self.metric = SMAPEAccuracy(layer_name=None, clamp=True, weight_with_error=weight_with_error, zero_eps=5e-9, importance_threshold=importance_threshold, keep_dim=keep_dim)
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


class AirkermaScatterAccuracy(AirkermaAccuracy):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, keep_dim: bool = False, max_relative_flux: float = 5e-2, min_relative_flux: float = 5e-3, metric_type: Union[Literal['smape']] = 'smape'):
        super().__init__(mu_tr_file, spectra_bins, max_energy_eV, weight_with_error, importance_threshold=0.0, keep_dim=keep_dim, metric_type=metric_type)
        self.max_relative_flux = max_relative_flux
        self.min_relative_flux = min_relative_flux

    def forward(self, target: Union[RadiationFieldChannel, AirKermaField, Tensor], prediction: Union[RadiationFieldChannel, AirKermaField, Tensor], input: TrainingInputData = None) -> Tensor:
        assert (isinstance(input.ground_truth, RadiationField) and input.ground_truth.direct_beam is not None) or (input.original_ground_truth is not None and input.original_ground_truth.direct_beam is not None), "Input TrainingInputData must contain direct_beam for scatter field accuracy."

        xgt = input.original_ground_truth.direct_beam if input.original_ground_truth is not None and input.original_ground_truth.direct_beam is not None else input.ground_truth.direct_beam
        xgt = xgt.flux if isinstance(xgt, RadiationFieldChannel) else xgt
        sgt = input.original_ground_truth.scatter_field if input.original_ground_truth is not None and input.original_ground_truth.scatter_field is not None else input.ground_truth.scatter_field
        sgt = sgt.flux if isinstance(sgt, RadiationFieldChannel) else sgt
        fgt = sgt + xgt

        beam_mask = xgt > xgt.max() * self.max_relative_flux  # ignore areas with > max_relative_flux of max primary flux
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

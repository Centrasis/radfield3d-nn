"""Single source for the validation/test air-kerma metric set, shared by training and tuning."""
from radfield3dnn.metrics.airkerma_accuracy import (
    AirkermaAccuracy, AirkermaScatterAccuracy, AirkermaBeamAccuracy,
    AirkermaSphereAccuracy, AirkermaSupervoxelScatterAccuracy, AirkermaAccuracyEnergyWeighted,
)
from radfield3dnn.metrics.ssim import AirkermaSSIM
from radfield3dnn.metrics import HistogramOverlapAccuracy


def build_airkerma_metrics(mu_tr_file: str, voxel_size_m: float, spectra_bins: int = 32,
                           max_energy_eV: float = 1.5e+5) -> dict:
    vx = voxel_size_m if voxel_size_m and voxel_size_m > 0.0 else 0.01

    def ak(**kw):
        return AirkermaAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, **kw)

    return {
        'global_airkerma_accuracy': ak(),
        'top90_airkerma_accuracy': ak(importance_threshold=0.1),
        'airkerma_accuracy_roi': ak(importance_threshold=0.01),
        'airkerma_ssim': AirkermaSSIM(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, reduction='mean'),
        'airkerma_ssim_gradient': AirkermaSSIM(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, reduction='mean', ssim_type='gradient'),
        'airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_roi=True, scatter_lo=5e-5),
        'noiseaware_airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_error=True, error_threshold=0.5),
        'legacy_airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_error=False),
        'airkerma_accuracy_scatter_sv8': AirkermaSupervoxelScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, supervoxel=8),
        'airkerma_accuracy_beam': AirkermaBeamAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV),
        'airkerma_onsphere_accuracy_radius25cm': AirkermaSphereAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, sphere_radius_m=0.25, voxel_size_m=vx),
        'top95_energy_weighted_airkerma_accuracy': AirkermaAccuracyEnergyWeighted(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, importance_threshold=0.05),
        'spectrum_accuracy': HistogramOverlapAccuracy(),
        'global_airkerma_gamma_3pct_4cm_cut1pct': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=40.0, dose_threshold=0.01),
        'global_airkerma_gamma_3pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=20.0),
        'global_airkerma_gamma_3pct_4cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=40.0),
        'global_airkerma_gamma_3pct_6cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=60.0),
        'global_airkerma_gamma_10pct_4cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=40.0),
        'global_airkerma_gamma_10pct_6cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=60.0),
    }

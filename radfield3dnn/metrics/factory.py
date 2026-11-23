"""Single source for the validation/test air-kerma metric set, shared by training and tuning."""
from radfield3dnn.metrics.airkerma_accuracy import (
    AirkermaAccuracy, AirkermaScatterAccuracy, AirkermaBeamAccuracy,
    AirkermaSphereAccuracy, AirkermaSupervoxelScatterAccuracy, AirkermaAccuracyEnergyWeighted,
)
from radfield3dnn.metrics.patient_shadow import PatientShadowAccuracy
from radfield3dnn.metrics.ssim import AirkermaSSIM
from radfield3dnn.metrics import HistogramOverlapAccuracy, SpectrumROIAccuracy


def build_airkerma_metrics(mu_tr_file: str, voxel_size_m: float, spectra_bins: int = 32,
                           max_energy_eV: float = 1.5e+5, joined: bool = False,
                           definition_file: str = None) -> dict:
    """Validation/test air-kerma metric set. On a JOINED dataset (single 'radiation' channel,
    no separate direct_beam) the scatter/beam-isolation metrics cannot be computed — they need the
    direct beam to separate scatter from beam — so they are omitted; the whole-field air-kerma,
    spectrum, SSIM and gamma metrics still apply."""
    vx = voxel_size_m if voxel_size_m and voxel_size_m > 0.0 else 0.01

    def ak(**kw):
        return AirkermaAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, **kw)

    metrics = {
        'global_airkerma_accuracy': ak(),
        'top90_airkerma_accuracy': ak(importance_threshold=0.1),
        'airkerma_accuracy_roi': ak(importance_threshold=0.01),
        'airkerma_ssim': AirkermaSSIM(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, reduction='mean'),
        'airkerma_ssim_gradient': AirkermaSSIM(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, reduction='mean', ssim_type='gradient'),
        'airkerma_onsphere_accuracy_radius25cm': AirkermaSphereAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, sphere_radius_m=0.25, voxel_size_m=vx),
        'top95_energy_weighted_airkerma_accuracy': AirkermaAccuracyEnergyWeighted(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, importance_threshold=0.05),
        'spectrum_accuracy': HistogramOverlapAccuracy(),
        # Does the field follow the PATIENT? Every entry above is dominated by the bright beam and
        # scores a model that ignores the body almost as well as one that does not. These two both
        # return None when the dataset carries no occupancy map or the phantom is out of the beam,
        # so they simply do not appear rather than reporting a zero nothing measured.
        'patient_shadow_attenuation': PatientShadowAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, mode='attenuation'),
        'patient_shadow_boundary': PatientShadowAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, mode='boundary'),
        'spectrum_accuracy_top90': SpectrumROIAccuracy(roi='top90', mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, top_threshold=0.1),
        'global_airkerma_gamma_3pct_4cm_cut1pct': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=40.0, dose_threshold=0.01),
        'global_airkerma_gamma_3pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=20.0),
        # At a 2 cm criterion on a 2 cm grid the distance search spans exactly one voxel, so the
        # six axial candidates sit at gamma >= 1 and only the voxel itself can pass: these read as
        # pointwise dose tests at 5 % and 10 % of the field peak.
        'global_airkerma_gamma_5pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.05, dist_crit_mm=20.0),
        'global_airkerma_gamma_10pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=20.0),
        # Local-normalised gamma: the dose criterion is a percentage of each voxel's OWN air kerma,
        # so it stays meaningful across the six decades of a field, where the global-normalised
        # entries above measure against the field peak and pass almost any prediction below a few
        # percent of it.
        # The cutoff sits at the Monte-Carlo noise floor, below which the reference is noise or
        # exactly zero and a local denominator would demand unbounded precision on it. Located from
        # DS03's own per-voxel error channel: 0 % of voxels are noise-dominated above 1e-2 of the
        # field peak, 21 % in 1e-3..1e-2, 87 % in 1e-4..1e-3 and 100 % below — and the excluded
        # region is ~1.5 % of a field, so the cut costs almost no coverage.
        'local_airkerma_gamma_3pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=20.0, dose_threshold=1e-4, gamma_normalization='local'),
        'local_airkerma_gamma_5pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.05, dist_crit_mm=20.0, dose_threshold=1e-4, gamma_normalization='local'),
        'local_airkerma_gamma_10pct_2cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=20.0, dose_threshold=1e-4, gamma_normalization='local'),
        'local_airkerma_gamma_10pct_4cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=40.0, dose_threshold=1e-4, gamma_normalization='local'),
        'global_airkerma_gamma_3pct_4cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=40.0),
        'global_airkerma_gamma_3pct_6cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.03, dist_crit_mm=60.0),
        'global_airkerma_gamma_10pct_4cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=40.0),
        'global_airkerma_gamma_10pct_6cm': ak(metric_type='gpr', voxel_size_m=vx, rel_dose_diff=0.1, dist_crit_mm=60.0),
    }
    if definition_file:
        # The analytic beam has to be the one the fields were generated with, so this entry appears
        # only when the training config named a dataset definition (`dataset.definition_file`).
        metrics['patient_shadow_onset'] = PatientShadowAccuracy(
            mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV,
            mode='onset', definition_file=definition_file)
    if not joined:
        # scatter/beam-isolation metrics require the separate direct_beam + scatter_field channels
        metrics.update({
            'airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_roi=True, scatter_lo=5e-5),
            'noiseaware_airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_error=True, error_threshold=0.5),
            'legacy_airkerma_accuracy_scatter': AirkermaScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, use_error=False),
            'airkerma_accuracy_scatter_sv8': AirkermaSupervoxelScatterAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, supervoxel=8),
            'airkerma_accuracy_beam': AirkermaBeamAccuracy(mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV),
            'spectrum_accuracy_scatter': SpectrumROIAccuracy(roi='scatter', mu_tr_file=mu_tr_file, spectra_bins=spectra_bins, max_energy_eV=max_energy_eV, scatter_lo=5e-5),
        })
    return metrics

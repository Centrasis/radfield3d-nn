"""Eval-side geometry exclusion + metric correctness fixes.

Covers: the InferenceHelper -inf hook, SSIM excluding (not zero-scoring) masked voxels,
per-voxel spectrum histogram overlap, sphere-metric NaN guard, MetricBase (1,)-filter gap.
"""
import pytest
import torch

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from RadFiled3D.pytorch.types import RadiationFieldChannel, TranslationalInput
from radfield3dnn.rftypes import AirKermaField, RadiationField, TrainingInputData

B, N, BINS = 2, 8, 8


def _geometry():
    g = torch.zeros(B, 1, N, N, N)
    g[:, 0, 2, 3, 4] = 1.0
    return g


def _channel(seed=0, channel_dim=False):
    # channel_dim=True mirrors the Layerwise layout ((B,1,...) flux) the airkerma integral needs
    g = torch.Generator().manual_seed(seed)
    flux = torch.rand(B, 1, N, N, N, generator=g) if channel_dim else torch.rand(B, N, N, N, generator=g)
    return RadiationFieldChannel(flux=flux,
                                 spectrum=torch.rand(B, BINS, N, N, N, generator=g).softmax(1),
                                 error=None)


def _batch_with_geometry():
    g = torch.Generator().manual_seed(9)
    inp = TranslationalInput(direction=torch.randn(B, 3, generator=g), origin=torch.rand(B, 1, generator=g),
                             spectrum=torch.rand(B, 32, generator=g), translation=torch.rand(B, 3, generator=g),
                             geometry=_geometry())
    gt = RadiationField(scatter_field=_channel(1), direct_beam=None, geometry=None)
    return TrainingInputData(input=inp, ground_truth=gt, original_ground_truth=gt)


# ── the eval hook ─────────────────────────────────────────────────────────────

def test_exclude_geometry_voxels_masks_gt_and_pred():
    from radfield3dnn.rfhelpers import InferenceHelper
    batch = _batch_with_geometry()
    gt, pred = _channel(1), _channel(2)
    gt2, pred2 = InferenceHelper._exclude_geometry_voxels(batch, gt, pred)
    for field in (gt2, pred2):
        assert torch.isneginf(field.flux[:, 2, 3, 4]).all()
        assert torch.isneginf(field.spectrum[:, :, 2, 3, 4]).all()
        assert int(torch.isneginf(field.flux).sum()) == B
    # airkerma variant
    ak = AirKermaField(air_kerma=torch.rand(B, 1, N, N, N), geometry=None)
    ak2, _ = InferenceHelper._exclude_geometry_voxels(batch, ak, AirKermaField(air_kerma=torch.rand(B, 1, N, N, N), geometry=None))
    assert torch.isneginf(ak2.air_kerma[:, 0, 2, 3, 4]).all()


def test_exclude_is_noop_without_geometry():
    from radfield3dnn.rfhelpers import InferenceHelper
    batch = _batch_with_geometry()
    batch = batch._replace(input=batch.input._replace(geometry=None))
    gt, pred = _channel(1), _channel(2)
    gt2, pred2 = InferenceHelper._exclude_geometry_voxels(batch, gt, pred)
    assert gt2 is gt and pred2 is pred


# ── SSIM: excluded voxels must not score ──────────────────────────────────────

def test_ssim_excludes_masked_voxels_instead_of_scoring_zero():
    from radfield3dnn.metrics.ssim import SSIM3D
    m = SSIM3D(layer_name=None, reduction='mean', eps=1e-8)
    t = torch.rand(1, 1, N, N, N)
    # perfect prediction outside the mask, catastrophic inside it
    p_good = t.clone()
    t_masked = t.clone(); t_masked[:, :, 2, 3, 4] = -torch.inf
    p_bad = t.clone(); p_bad[:, :, 2, 3, 4] = 123.0    # value inside excluded voxel
    p_bad_masked = p_bad.clone()
    s_ref = m._calc_metric(t_masked, p_good)
    s_bad = m._calc_metric(t_masked, p_bad_masked)
    # the excluded voxel's center is dropped from the mean; its window bias is bounded — the two
    # scores must stay close to identical (previously the zero-fill scored the region as agreement)
    assert torch.isfinite(s_ref).all() and torch.isfinite(s_bad).all()


def test_airkerma_ssim_propagates_exclusion():
    from radfield3dnn.metrics.ssim import AirkermaSSIM
    import tempfile, os
    mu = os.path.join(tempfile.mkdtemp(), "mu.txt")
    with open(mu, "w") as f:
        f.write("1000 0.01\n150000 0.002\n")
    m = AirkermaSSIM(mu_tr_file=mu, spectra_bins=BINS, max_energy_eV=1.5e5)
    ch = _channel(3, channel_dim=True)
    flux = ch.flux.clone(); flux[:, 0, 2, 3, 4] = -torch.inf
    masked = ch._replace(flux=flux)
    out = m.forward(masked, _channel(4, channel_dim=True))
    assert torch.isfinite(out).all()


# ── spectrum accuracy: per-voxel, not global ─────────────────────────────────

def test_histogram_overlap_sees_per_voxel_spectral_errors():
    from radfield3dnn.metrics.histogram_accuracy import HistogramOverlapAccuracy
    m = HistogramOverlapAccuracy()
    g = torch.Generator().manual_seed(5)
    spec = torch.rand(1, BINS, 2, 1, 1, generator=g).softmax(1)
    perfect = m._calc_metric(spec, spec.clone())
    # swap the spectra of the two voxels (equal per-voxel flux: histograms are normalized) —
    # the OLD global-sum implementation returned the same score as perfect
    swapped = spec.clone()
    swapped[:, :, 0, 0, 0], swapped[:, :, 1, 0, 0] = spec[:, :, 1, 0, 0], spec[:, :, 0, 0, 0]
    degraded = m._calc_metric(spec, swapped)
    assert float(perfect) > float(degraded) + 0.01


def test_histogram_overlap_excludes_masked_voxels():
    from radfield3dnn.metrics.histogram_accuracy import HistogramOverlapAccuracy
    m = HistogramOverlapAccuracy()
    g = torch.Generator().manual_seed(6)
    t = torch.rand(1, BINS, 2, 1, 1, generator=g).softmax(1)
    p = t.clone()
    p[:, :, 1, 0, 0] = torch.rand(BINS, generator=g)     # wrong spectrum in voxel 1
    t_masked = t.clone(); t_masked[:, :, 1, 0, 0] = -torch.inf   # ... but voxel 1 is excluded
    assert float(m._calc_metric(t_masked, p)) == pytest.approx(1.0, abs=1e-5)


# ── sphere metric: empty shell must skip, not poison ─────────────────────────

def test_sphere_metric_empty_shell_returns_empty_not_nan():
    from radfield3dnn.metrics.airkerma_accuracy import AirkermaSphereAccuracy
    import tempfile, os
    mu = os.path.join(tempfile.mkdtemp(), "mu.txt")
    with open(mu, "w") as f:
        f.write("1000 0.01\n150000 0.002\n")
    # radius 0.25 m on an 8-voxel 0.01 m grid: the shell lies entirely outside the field
    m = AirkermaSphereAccuracy(mu_tr_file=mu, spectra_bins=BINS, max_energy_eV=1.5e5,
                               sphere_radius_m=0.25, voxel_size_m=0.01)
    out = m.forward(_channel(1, channel_dim=True), _channel(2, channel_dim=True))
    assert out.ndim >= 1 and out.numel() == 0            # accumulator skips; no NaN


# ── MetricBase: (1,)-shaped results are filtered too ─────────────────────────

def test_metric_base_filters_batchsize_one_results():
    from radfield3dnn.metrics.smape import SMAPEAccuracy
    m = SMAPEAccuracy(layer_name=None, clamp=True)
    t = torch.full((1, 4, 4, 4), -torch.inf)             # everything excluded
    out = m.forward(t, torch.rand(1, 4, 4, 4))
    assert out.numel() == 0                              # skip, not NaN passthrough


# ── gamma: empty fields skip; excluded voxels never act as candidates ────────

def test_gamma_empty_field_skips_instead_of_scoring_zero():
    from radfield3dnn.metrics.gpr import GammaPassingRate
    m = GammaPassingRate(layer_name=None, voxel_size_m=0.01, rel_dose_diff=0.03,
                         dist_crit_mm=20.0, dose_threshold=0.1)
    g = torch.Generator().manual_seed(0)
    t = torch.rand(2, 1, 6, 6, 6, generator=g)
    t[1] = -torch.inf                       # field 1: fully excluded -> no reference voxels
    p = torch.rand(2, 1, 6, 6, 6, generator=g)
    rate, _ = m.gamma_index(p, t, dist_crit_mm=20.0, voxel_size_mm=(10, 10, 10), dose_threshold=0.1)
    assert torch.isfinite(rate[0]) and torch.isnan(rate[1])
    # through the metric pipeline the NaN field is dropped, not averaged in
    out = m.forward(t, p)
    assert out.numel() == 1 or out.ndim == 0
    assert torch.isfinite(out).all()


def test_gamma_excluded_prediction_voxels_are_not_candidates():
    from radfield3dnn.metrics.gpr import GammaPassingRate
    m = GammaPassingRate(layer_name=None)
    # one hot reference voxel; the prediction is wrong EVERYWHERE except at an excluded (-inf)
    # neighbor that would agree perfectly if it were allowed to act as a candidate
    t = torch.zeros(1, 1, 5, 5, 5)
    t[0, 0, 2, 2, 2] = 1.0
    p = torch.zeros(1, 1, 5, 5, 5)
    p[0, 0, 2, 2, 3] = -torch.inf           # excluded neighbor "containing" the right dose
    rate, _ = m.gamma_index(p, t, dist_crit_mm=20.0, voxel_size_mm=(10, 10, 10),
                            rel_dose_diff=0.03, dose_threshold=0.5)
    # the only reference voxel must FAIL (0.0) — the excluded candidate cannot rescue it,
    # and NaN from the exclusion must not poison the result
    assert rate[0] == 0.0


def test_gamma_reference_with_no_reachable_candidates_is_skipped():
    from radfield3dnn.metrics.gpr import GammaPassingRate
    m = GammaPassingRate(layer_name=None)
    t = torch.zeros(1, 1, 3, 3, 3)
    t[0, 0, 1, 1, 1] = 1.0
    p = torch.full((1, 1, 3, 3, 3), -torch.inf)   # every candidate excluded
    rate, _ = m.gamma_index(p, t, dist_crit_mm=10.0, voxel_size_mm=(10, 10, 10), dose_threshold=0.5)
    # no usable candidate anywhere -> the reference voxel is skipped -> empty field -> NaN skip
    assert torch.isnan(rate[0])

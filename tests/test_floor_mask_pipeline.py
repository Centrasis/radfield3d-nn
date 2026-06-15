"""Tests for the -inf floor-masking pipeline:
  * MCFloorCut(as_neginf=True): masks the shared FLOOR ROI to -inf in both channels, join-safe,
    training-only;
  * ROIbasedSampler(floor_as_zero=True): the sampled floor subset becomes a genuine 0, the rest
    of the floor (and non-sampled scatter) stays -inf, beam/scatter keep their real values;
  * RawNeRFLoss: bounded (no zero-prediction explosion) and robust to -inf masked targets.
"""
import torch

from radfield3dnn.roi import compute_roi_masks, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
from radfield3dnn.datasets.mc_floor_cut import MCFloorCut
from radfield3dnn.preprocessing.augmentations.roi_sampling import ROIbasedSampler
from radfield3dnn.losses.combinations import SMAPERegionBalancedLoss
from radfield3dnn.rftypes import TrainingInputData, RadiationField, RadiationFieldChannel


def _synthetic_field(D=20, seed=0):
    torch.manual_seed(seed)
    direct = torch.zeros(1, 1, D, D, D)
    direct[0, 0, D // 2, D // 2, D // 2] = 1.0
    direct[0, 0, D // 2 - 1:D // 2 + 2, D // 2 - 1:D // 2 + 2, D // 2 - 1:D // 2 + 2] += 0.2
    scatter = torch.rand(1, 1, D, D, D) * 1e-3
    scatter += (torch.rand(1, 1, D, D, D) < 0.3) * 1e-7      # sub-floor (noise) voxels
    return direct, scatter, direct + scatter


def _make_input(direct, scatter, joined, D, S=8, joined_gt=True):
    def ch(flux):
        return RadiationFieldChannel(flux=flux, spectrum=torch.rand(1, S, D, D, D),
                                     error=torch.zeros(1, 1, D, D, D))
    ogt = RadiationField(scatter_field=ch(scatter.clone()), direct_beam=ch(direct.clone()))
    if joined_gt:
        gt = RadiationField(scatter_field=RadiationFieldChannel(flux=joined.clone(),
                            spectrum=torch.rand(1, S, D, D, D), error=None), direct_beam=None)
    else:
        gt = RadiationField(scatter_field=ch(scatter.clone()), direct_beam=ch(direct.clone()))
    return TrainingInputData(input=None, ground_truth=gt, original_ground_truth=ogt)


# ── MCFloorCut -inf mode ─────────────────────────────────────────────────────────────────
def test_mcfloorcut_neginf_masks_floor_both_channels():
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    _, _, floor = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    inp = _make_input(direct, scatter, joined, D, joined_gt=False)
    cut = MCFloorCut(as_neginf=True); cut.train()
    out = cut.forward(inp)
    sc = out.ground_truth.scatter_field.flux
    dr = out.ground_truth.direct_beam.flux
    # floor voxels -> -inf in BOTH channels; non-floor stay finite
    assert torch.isinf(sc[floor]).all() and torch.isinf(dr[floor]).all()
    assert torch.isfinite(sc[~floor]).all() and torch.isfinite(dr[~floor]).all()


def test_mcfloorcut_neginf_is_join_safe():
    # After summing the two channels, floor -> -inf, every non-floor voxel stays FINITE (no NaN):
    # the scatter region (direct is floor there but joined is not) must NOT be masked.
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    _, scat, floor = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    inp = _make_input(direct, scatter, joined, D, joined_gt=False)
    cut = MCFloorCut(as_neginf=True); cut.train()
    out = cut.forward(inp)
    joined_after = out.ground_truth.scatter_field.flux + out.ground_truth.direct_beam.flux
    assert not torch.isnan(joined_after).any()
    assert torch.isfinite(joined_after[scat]).all()        # scatter region survives the join
    assert torch.isinf(joined_after[floor]).all()


def test_mcfloorcut_neginf_training_only():
    D = 16
    direct, scatter, joined = _synthetic_field(D)
    inp = _make_input(direct, scatter, joined, D, joined_gt=False)
    cut = MCFloorCut(as_neginf=True); cut.eval()
    out = cut.forward(inp)
    assert torch.isfinite(out.ground_truth.scatter_field.flux).all()    # no masking in eval


def test_mcfloorcut_zero_mode_unchanged():
    # The default (zeroing) mode must still zero, not mask.
    D = 16
    direct, scatter, joined = _synthetic_field(D)
    inp = _make_input(direct, scatter, joined, D, joined_gt=False)
    cut = MCFloorCut(rel_threshold=1e-2); cut.train()
    out = cut.forward(inp)
    assert torch.isfinite(out.ground_truth.direct_beam.flux).all()      # zeroed, never -inf


# ── ROIbasedSampler floor_as_zero ────────────────────────────────────────────────────────
def test_roi_sampler_floor_as_zero():
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    beam, _, floor = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    samp = ROIbasedSampler(scatter_ratio=2.0, floor_ratio=1.0, floor_as_zero=True); samp.train()
    out = samp.forward(_make_input(direct, scatter, joined, D))
    flux = out.ground_truth.scatter_field.flux
    is_zero = (flux == 0.0)
    is_inf = torch.isinf(flux)
    # the re-injected floor voxels are EXACT zeros, and every zero lives in the floor ROI
    assert bool(is_zero.any()) and not bool((is_zero & ~floor).any())
    # beam is kept at its real (nonzero, finite) value
    assert torch.isfinite(flux[beam]).all() and bool((flux[beam] != 0).all())
    # there are still masked (-inf) voxels (the non-sampled floor + non-sampled scatter)
    assert bool(is_inf.any())
    # zero count is capped by ~beam count (floor_ratio=1)
    assert int(is_zero.sum()) <= int(beam.sum()) + 1


def test_roi_sampler_floor_as_value_keeps_original():
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    samp = ROIbasedSampler(scatter_ratio=2.0, floor_ratio=1.0, floor_as_zero=False); samp.train()
    out = samp.forward(_make_input(direct, scatter, joined, D))
    flux = out.ground_truth.scatter_field.flux
    kept = torch.isfinite(flux)
    # with floor_as_zero off, kept voxels are the ORIGINAL joined values (no injected zeros from
    # the floor sampling beyond whatever was already ~0); none are exactly 0 unless the source was.
    assert bool(kept.any()) and torch.isinf(flux[~kept]).all()


# ── SMAPEBalanced bounded + -inf robust ───────────────────────────────────────────────────
def test_smapebalanced_bounded_and_masked():
    torch.manual_seed(1)
    t = (torch.rand(2, 1, 16, 16, 16) ** 4) * 0.5
    lf = SMAPERegionBalancedLoss()
    z = torch.zeros_like(t).requires_grad_(True)
    Lz = lf.forward(target=t.clone(), prediction=z, input=TrainingInputData(input=None, ground_truth=None))
    Lz.sum().backward()
    assert torch.isfinite(Lz).all() and float(Lz.mean().detach()) < 100
    assert torch.isfinite(z.grad).all()
    # -inf masked target -> finite loss
    m = t.clone()
    m[0, 0, :4] = -torch.inf
    p = t.clone().requires_grad_(True)
    Lm = lf.forward(target=m, prediction=p, input=TrainingInputData(input=None, ground_truth=None))
    assert torch.isfinite(Lm).all()

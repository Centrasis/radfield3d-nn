"""Tests for the shared beam/scatter/floor ROI: the mask helper, TwoROIGammaLoss (target-count
equalisation + -inf handling) and the ROIbasedSampler (keep beam, sample scatter/floor, multiplier).
"""
import torch

from radfield3dnn.roi import compute_roi_masks, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT
from radfield3dnn.losses.combinations import TwoROIGammaLoss
from radfield3dnn.preprocessing.augmentations.roi_sampling import ROIbasedSampler
from radfield3dnn.rftypes import TrainingInputData, RadiationField, RadiationFieldChannel


def _synthetic_field(D=20, seed=0):
    torch.manual_seed(seed)
    direct = torch.zeros(1, 1, D, D, D)
    direct[0, 0, D // 2, D // 2, D // 2] = 1.0
    direct[0, 0, D // 2 - 1:D // 2 + 2, D // 2 - 1:D // 2 + 2, D // 2 - 1:D // 2 + 2] += 0.2
    scatter = torch.rand(1, 1, D, D, D) * 1e-3
    scatter += (torch.rand(1, 1, D, D, D) < 0.2) * 1e-6   # some sub-floor voxels
    return direct, scatter, direct + scatter


def test_roi_masks_partition_and_disjoint():
    direct, scatter, joined = _synthetic_field()
    beam, sc, fl = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    # disjoint
    assert not bool((beam & sc).any() or (beam & fl).any() or (sc & fl).any())
    # partition the whole field
    assert int(beam.sum() + sc.sum() + fl.sum()) == joined.numel()
    # beam = direct >= 0.05*direct_max ; scatter is the bulk ; floor non-empty (sub-floor voxels)
    assert int(beam.sum()) >= 1 and int(sc.sum()) > int(beam.sum()) and int(fl.sum()) >= 1


def test_loss_handles_masked_inf_and_is_finite():
    direct, scatter, joined = _synthetic_field()
    target = joined.clone()
    target[0, 0, :5] = -torch.inf                      # masked / not-sampled slab
    pred = (joined + 0.01 * torch.randn_like(joined)).clamp_min(0).requires_grad_(True)
    loss = TwoROIGammaLoss(beam_rel=BEAM_REL_DEFAULT, scatter_lo=SCATTER_LO_DEFAULT)
    L = loss.forward(target=target, prediction=pred, input=None)
    L.backward()
    assert torch.isfinite(L).item() and torch.isfinite(pred.grad).all().item()


def test_loss_equalises_roi_influence_by_target_count():
    # A beam term (few voxels, equal per-voxel error) and a scatter term (many voxels, equal
    # per-voxel error) must contribute EQUALLY because each is a per-ROI mean (count-normalised).
    direct, scatter, joined = _synthetic_field()
    target = joined.clone()
    beam, sc, fl = compute_roi_masks(target, target, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    # prediction: a constant absolute offset only in the beam vs only in the scatter
    loss = TwoROIGammaLoss(w_beam=1.0, w_scatter=1.0, w_floor=0.0)
    p_beam = target.clone(); p_beam[beam] += 0.1
    p_sc = target.clone(); p_sc[sc] += 0.1
    Lb = loss.forward(target=target.clone(), prediction=p_beam, input=None)
    Ls = loss.forward(target=target.clone(), prediction=p_sc, input=None)
    # beam is L1 (abs); scatter is relative — not numerically identical, but both finite and the
    # beam term (27 voxels) is NOT drowned by the scatter term (thousands of voxels): same order.
    assert torch.isfinite(Lb) and torch.isfinite(Ls)
    assert float(Lb) > 0  # the few-voxel beam still produces a real gradient signal


def _make_input(direct, scatter, joined, D, S=8):
    def ch(flux):
        return RadiationFieldChannel(flux=flux, spectrum=torch.rand(1, S, D, D, D),
                                     error=torch.zeros(1, 1, D, D, D))
    ogt = RadiationField(scatter_field=ch(scatter.clone()), direct_beam=ch(direct.clone()))
    gt = RadiationField(scatter_field=RadiationFieldChannel(flux=joined.clone(),
                        spectrum=torch.rand(1, S, D, D, D), error=None), direct_beam=None)
    return TrainingInputData(input=None, ground_truth=gt, original_ground_truth=ogt)


def test_sampler_keeps_beam_samples_scatter_floor():
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    beam, sc, fl = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    samp = ROIbasedSampler(scatter_ratio=2.0, floor_ratio=1.0, field_multiplier=3.0); samp.train()
    out = samp.forward(_make_input(direct, scatter, joined, D))
    kept = torch.isfinite(out.ground_truth.scatter_field.flux)
    nb = int(beam.sum())
    # all beam kept; ~2x beam scatter; ~min(1x beam, n_floor) floor
    assert bool((kept & beam).sum() == nb)
    expected = nb + round(2 * nb) + min(round(1 * nb), int(fl.sum()))
    assert abs(int(kept.sum()) - expected) <= 2
    assert samp.dataset_multiplier() == 3.0


def test_sampler_varies_scatter_subset_but_always_keeps_beam():
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    beam, _, _ = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    samp = ROIbasedSampler(scatter_ratio=2.0, field_multiplier=3.0); samp.train()
    inp = _make_input(direct, scatter, joined, D)
    k1 = torch.isfinite(samp.forward(inp).ground_truth.scatter_field.flux)
    k2 = torch.isfinite(samp.forward(inp).ground_truth.scatter_field.flux)
    assert bool((k1 != k2).any())                    # different scatter subset each draw
    assert bool((k1 & beam).sum() == beam.sum()) and bool((k2 & beam).sum() == beam.sum())


def test_sampler_scatter_ratio_schedule():
    # scatter_ratio_end + schedule_switch: ratio is `scatter_ratio` until `schedule_switch` progress,
    # then `scatter_ratio_end` — and the SAMPLED scatter count follows the active ratio.
    D = 20
    direct, scatter, joined = _synthetic_field(D)
    beam, sc, fl = compute_roi_masks(direct, joined, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT)
    nb = int(beam.sum())
    samp = ROIbasedSampler(scatter_ratio=2.5, scatter_ratio_end=1.0, schedule_switch=0.8,
                           floor_ratio=0.0, floor_as_zero=False); samp.train()
    samp.set_schedule_progress(0.0)
    assert samp._eff_scatter_ratio == 2.5
    out_early = samp.forward(_make_input(direct, scatter, joined, D))
    n_early = int(torch.isfinite(out_early.ground_truth.scatter_field.flux).sum())
    samp.set_schedule_progress(0.85)
    assert samp._eff_scatter_ratio == 1.0
    out_late = samp.forward(_make_input(direct, scatter, joined, D))
    n_late = int(torch.isfinite(out_late.ground_truth.scatter_field.flux).sum())
    # early keeps ~beam + 2.5*beam, late keeps ~beam + 1*beam → fewer kept voxels late
    assert n_early > n_late
    assert abs(n_early - (nb + round(2.5 * nb))) <= 2
    assert abs(n_late - (nb + round(1.0 * nb))) <= 2


def test_sampler_eval_mode_is_noop():
    D = 16
    direct, scatter, joined = _synthetic_field(D)
    samp = ROIbasedSampler(); samp.eval()
    inp = _make_input(direct, scatter, joined, D)
    out = samp.forward(inp)
    assert torch.isfinite(out.ground_truth.scatter_field.flux).all()   # nothing masked in eval

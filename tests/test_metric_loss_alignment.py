"""Tests for the noise-aware scatter metric and the metric-targeted losses (pipeline-audit P0).

CPU-only, no GPU/RadFiled3D needed: synthetic fields with a known beam / ring / bulk / noise layout.
"""
import math
import torch
import pytest

from radfield3dnn.losses.combinations import SMAPERegionBalancedLoss
from radfield3dnn.rftypes import TrainingInputData, RadiationField, RadiationFieldChannel


def _synth_field(B=2, D=8, seed=0):
    """Joined-style field in normalized [0,1] units with a known region layout.

    Returns (target, error, masks) where masks = (beam, ring, bulk, noise) boolean tensors.
    Layout per item: beam = first 4 voxels (t=0.5), ring = next 8 (t=1e-2), noise = next 16
    (t=1e-4, err=1.0), bulk = rest (t=1e-4, err≈0.5 — reliable scatter + leakage direct).
    """
    g = torch.Generator().manual_seed(seed)
    n = D ** 3
    t = torch.full((B, n), 1e-4)
    err = torch.full((B, n), 0.5)
    beam = torch.zeros(B, n, dtype=torch.bool); beam[:, :4] = True
    ring = torch.zeros(B, n, dtype=torch.bool); ring[:, 4:12] = True
    noise = torch.zeros(B, n, dtype=torch.bool); noise[:, 12:28] = True
    t[beam] = 0.5
    t[ring] = 1e-2
    err[noise] = 1.0
    bulk = ~(beam | ring | noise)
    return t.view(B, D, D, D), err.view(B, D, D, D), (
        beam.view(B, D, D, D), ring.view(B, D, D, D), bulk.view(B, D, D, D), noise.view(B, D, D, D))


def _input_with_error(err):
    gt = RadiationFieldChannel(flux=None, spectrum=None, error=err)
    return TrainingInputData(input=None, ground_truth=gt)


class TestSMAPERegionBalanced:
    def test_perfect_prediction_is_zero(self):
        t, err, _ = _synth_field()
        loss = SMAPERegionBalancedLoss()
        out = loss(t, t.clone(), _input_with_error(err))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_noise_voxels_hinged_not_fitted(self):
        """Noise voxels must NOT be fitted to their (noise) GT — but they must NOT be free either:
        predictions above the hinge threshold get pushed down (the corner-hallucination fix)."""
        t, err, (beam, ring, bulk, noise) = _synth_field()
        loss = SMAPERegionBalancedLoss()
        # (a) prediction BELOW the hinge in the noise region -> zero gradient there (noise not fitted)
        p = t.clone()
        p[noise] = 1e-4                       # well below hinge (ring_rel*max = 5e-3*0.5)
        p = p.requires_grad_(True)
        loss(t, p, _input_with_error(err)).sum().backward()
        assert p.grad[noise].abs().sum() == 0.0, "below-hinge noise voxels must be free (no noise fitting)"
        # (b) prediction ABOVE the hinge (hallucinated radiation) -> pushed DOWN
        p2 = t.clone()
        p2[noise] = 0.2                       # hallucinated blob in the noise region
        p2 = p2.requires_grad_(True)
        loss(t, p2, _input_with_error(err)).sum().backward()
        g_noise = p2.grad[noise]
        assert g_noise.abs().sum() > 0, "above-hinge noise voxels must receive gradient"
        assert (g_noise > 0).float().mean() > 0.99, "gradient must push hallucinated values DOWN"
        # the scored regions still receive gradient
        p3 = (t + 0.1 * t.clamp(min=1e-4)).requires_grad_(True)
        loss(t, p3, _input_with_error(err)).sum().backward()
        for name, m in (("beam", beam), ("ring", ring), ("bulk", bulk)):
            assert p3.grad[m].abs().sum() > 0, f"{name} region should receive gradient"

    def test_region_gradient_mass_balanced(self):
        """Each region's gradient mass in RELATIVE-error currency (|∂L/∂p|·t — what SMAPE-accuracy
        and the gamma 3% criterion respond to) must be ~equal for beam and ring despite the wildly
        different voxel counts; the bulk is intentionally eps-damped (the MC-noise guard) but must
        stay within ~10x, not the 25-1000x of a plain mean."""
        t, err, (beam, ring, bulk, noise) = _synth_field()
        # uniform multiplicative error in every region -> equal per-voxel relative error
        p = (t * 1.2).requires_grad_(True)
        loss = SMAPERegionBalancedLoss()
        loss(t, p, _input_with_error(err)).sum().backward()
        g_rel = (p.grad.abs() * t)                      # chain |dL/dp| into relative-error units
        masses = torch.tensor([g_rel[beam].sum(), g_rel[ring].sum(), g_rel[bulk].sum()])
        assert (masses > 0).all()
        assert masses[0] / masses[1] < 1.5 and masses[1] / masses[0] < 1.5, \
            f"beam/ring relative gradient mass should be balanced, got {masses.tolist()}"
        assert masses.max() / masses[2] < 10.0, \
            f"bulk may be eps-damped but not starved, got {masses.tolist()}"

    def test_voxel_count_imbalance_would_be_25x_unbalanced(self):
        """Sanity contrast: a plain mean WOULD give the bulk ~30x the beam's gradient mass."""
        t, err, (beam, ring, bulk, noise) = _synth_field()
        p = (t * 1.2).requires_grad_(True)
        smape = (2 * (p - t).abs()) / (p.abs() + t.abs() + 1e-3)
        smape.mean().backward()
        g = p.grad.abs()
        assert g[bulk].sum() / g[beam].sum() > 10, "plain mean is bulk-dominated (the bug)"

    def test_finite_on_zeros_and_nans(self):
        t, err, _ = _synth_field()
        t[0, 0, 0, 0] = float("nan")
        p = torch.zeros_like(t)
        out = SMAPERegionBalancedLoss()(t, p, _input_with_error(err))
        assert torch.isfinite(out).all()

    def test_scale_invariance(self):
        """SMAPE core is scale-invariant -> per-field max normalization does not change the loss."""
        t, err, _ = _synth_field()
        p = t * 1.3
        l1 = SMAPERegionBalancedLoss(eps=0.0)(t, p, _input_with_error(err))
        l2 = SMAPERegionBalancedLoss(eps=0.0)(t * 7.0, p * 7.0, _input_with_error(err))
        assert torch.allclose(l1, l2, rtol=1e-5)

    def test_row_mode_fallback(self):
        out = SMAPERegionBalancedLoss()(torch.rand(64), torch.rand(64), _input_with_error(None) if False else TrainingInputData(input=None, ground_truth=None))
        assert out.shape == (64,) and torch.isfinite(out).all()


class TestGhostBeamSuppression:
    """Regression for the ep153 failure: under the SMAPE core a misplaced (ghost/rotated) beam —
    p ≈ 0.3 where t ≈ 1e-4 — receives near-zero gradient (over-prediction saturation, observed
    bright-IoU 0.29 / 63% ghost mass). The logratio core must push ghosts down at full strength."""

    def _ghost_setup(self):
        t, err, (beam, ring, bulk, noise) = _synth_field()
        p = t.clone()
        ghost = torch.zeros_like(beam); ghost[:, 2, 2, 2] = True   # one bulk voxel per item
        p[ghost] = 0.3                                             # misplaced bright structure
        return t, err, p.requires_grad_(True), ghost

    def test_smape_core_cannot_erase_ghosts(self):
        t, err, p, ghost = self._ghost_setup()
        SMAPERegionBalancedLoss(core="smape")(t, p, _input_with_error(err)).sum().backward()
        g_smape = p.grad[ghost].abs().max()
        assert g_smape < 1e-4, f"documents the pathology: SMAPE ghost gradient should be tiny, got {float(g_smape)}"

    def test_logratio_core_erases_ghosts(self):
        t, err, p, ghost = self._ghost_setup()
        loss = SMAPERegionBalancedLoss(core="logratio")
        loss(t, p, _input_with_error(err)).sum().backward()
        g = p.grad[ghost]
        assert (g > 0).all(), "ghost gradient must push DOWN"
        # compare against the SMAPE core's gradient on the identical setup
        t2, err2, p2, ghost2 = self._ghost_setup()
        SMAPERegionBalancedLoss(core="smape")(t2, p2, _input_with_error(err2)).sum().backward()
        ratio = float(g.abs().max() / p2.grad[ghost2].abs().max().clamp(min=1e-12))
        assert ratio > 100, f"logratio ghost suppression should be >>100x SMAPE's, got {ratio:.1f}x"

    def test_logratio_still_balanced_and_finite(self):
        t, err, (beam, ring, bulk, noise) = _synth_field()
        p = (t * 1.3).requires_grad_(True)
        loss = SMAPERegionBalancedLoss(core="logratio")
        out = loss(t, p, _input_with_error(err))
        assert torch.isfinite(out).all()
        out.sum().backward()
        for name, m in (("beam", beam), ("ring", ring), ("bulk", bulk)):
            assert p.grad[m].abs().sum() > 0, f"{name} region should receive gradient"
        # perfect prediction -> zero loss
        z = SMAPERegionBalancedLoss(core="logratio")(t, t.clone(), _input_with_error(err))
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)


class TestSupervoxelMetric:
    def test_perfect_is_one_and_noise_is_damped(self):
        from radfield3dnn.metrics.airkerma_accuracy import AirkermaSupervoxelScatterAccuracy
        import os
        mu = "/mnt/data/const/mu_tr/emuen_rho_air_1keV-1MeV.txt"
        if not os.path.exists(mu):
            import pytest; pytest.skip("mu_tr table not available")
        m = AirkermaSupervoxelScatterAccuracy(mu_tr_file=mu, spectra_bins=8, max_energy_eV=1.5e5, supervoxel=8)
        g = torch.Generator().manual_seed(0)
        flux = torch.rand(1, 1, 48, 48, 48, generator=g) * 1e-3 + 1e-4
        spec = torch.rand(1, 8, 48, 48, 48, generator=g); spec = spec / spec.sum(1, keepdim=True)
        from radfield3dnn.rftypes import RadiationFieldChannel
        gt = RadiationFieldChannel(flux=flux, spectrum=spec, error=None)
        pr = RadiationFieldChannel(flux=flux.clone(), spectrum=spec.clone(), error=None)
        acc_perfect = float(m.forward(gt, pr, input=None))
        assert acc_perfect > 0.999, f"perfect prediction must score ~1.0, got {acc_perfect}"
        # 100% per-voxel multiplicative noise -> aggregated accuracy must far exceed per-voxel
        noisy = RadiationFieldChannel(flux=flux * (1 + torch.randn(flux.shape, generator=g).clamp(-0.9, 3)),
                                      spectrum=spec.clone(), error=None)
        acc_sv = float(m.forward(gt, noisy, input=None))
        assert acc_sv > 0.85, f"sv8 aggregation should damp 100% voxel noise to >0.85, got {acc_sv}"


class TestMTLBalancer:
    def test_loss_floor_bounds_amplification(self):
        """The log transform's 1/L gradient amplification must be bounded by loss_floor — a solved
        task (tiny loss) must NOT receive an exploding share of the trunk gradient (the step-1-only
        path previously had no guard down to eps=1e-8 -> up to 1e8x)."""
        from radfield3dnn.losses.mtl.mtl import MultiTaskLossBalancer
        mtl = MultiTaskLossBalancer(loss_floor=1e-3)
        w = torch.tensor([1.0], requires_grad=True)
        flux = (w * 1.0).sum()                # L ~ 1.0
        solved = (w * 1e-7).sum()             # solved task, L = 1e-7 << floor
        surrogate = mtl.combine({"flux": flux, "solved": solved}, None)
        surrogate.backward()
        # gradient through log(clamp(L, floor)): flux contributes 1/1.0 = 1; the solved task is
        # below the floor -> clamp kills its gradient entirely (it releases the trunk).
        assert torch.isfinite(w.grad).all()
        assert abs(float(w.grad) - 1.0) < 1e-4, f"solved task should contribute 0, flux 1/L=1; got {float(w.grad)}"

    def test_scale_balancing_equalizes_40x_gap(self):
        """The observed 40x flux/spectrum magnitude gap: under log-space balancing both tasks
        contribute the SAME relative gradient (1/L_i * dL_i), unlike the flat sum."""
        from radfield3dnn.losses.mtl.mtl import MultiTaskLossBalancer
        mtl = MultiTaskLossBalancer(loss_floor=1e-3)
        a = torch.tensor([1.0], requires_grad=True)
        b = torch.tensor([1.0], requires_grad=True)
        flux = (a * 1.0).sum()                # L_f = 1.0 (SMAPEBalanced scale)
        spec = (b * 0.025).sum()              # L_s = 0.025 (HistogramLoss scale, 40x smaller)
        mtl.combine({"flux": flux, "spec": spec}, None).backward()
        # d/da log(a*1.0) = 1.0 ; d/db log(b*0.025) = 1.0  -> equal relative push
        assert abs(float(a.grad) - 1.0) < 1e-5 and abs(float(b.grad) - 1.0) < 1e-5, \
            f"log balancing must equalize: got {float(a.grad)} vs {float(b.grad)}"


class TestNoiseAwareScatterMetricMask:
    """The masking logic of AirkermaScatterAccuracy(use_error=True), tested standalone
    (the full metric needs the mu_tr file; here we verify the mask construction contract)."""

    def test_error_mask_supersedes_flux_threshold(self):
        # mirror of the metric's masking code path
        D = 8
        direct = torch.zeros(D, D, D); direct[:2] = 1.0          # beam slab
        scatter = torch.full((D, D, D), 1e-4)                    # diffuse, reliable
        err = torch.zeros(D, D, D); err[5] = 1.0                 # one noise slab
        beam_mask = direct > direct.max() * 5e-2
        noise_mask = err >= 0.5
        scored = ~(beam_mask | noise_mask)
        frac = scored.float().mean()
        # beam slab (2/8) + noise slab (1/8) excluded -> 5/8 scored
        assert abs(float(frac) - 5 / 8) < 1e-6
        # the diffuse bulk IS scored (the whole point vs the legacy 5e-3 flux threshold,
        # under which scatter=1e-4 << 5e-3*max would be excluded entirely)
        legacy_low = (scatter + direct) < (scatter + direct).max() * 5e-3
        assert legacy_low[~beam_mask].float().mean() > 0.9  # legacy would drop >90% of non-beam

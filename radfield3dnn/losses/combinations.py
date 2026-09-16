from .std import WassersteinLossWeighted, L1LossWeighted, reduce_per_sample
from .base import Loss
from torch import Tensor
from radfield3dnn.rftypes import TrainingInputData
import torch
from torch import nn
from torch.nn import functional as F


class HistogramLoss(Loss):
    """Loss for comparing spectral histograms: Wasserstein + L1."""

    def __init__(self, bin_dim: int = -1, weight_with_error: bool = False,
                 penalize_out_of_range: bool = False, calc_moments: bool = False,
                 ws_weight: float = 0.7, l1_weight: float = 0.3):
        super().__init__()
        self.weight_with_error = weight_with_error
        self.penalize_out_of_range = penalize_out_of_range
        self.l1_loss = L1LossWeighted(weight_with_error)
        self.wasserstein_loss = WassersteinLossWeighted(bin_dim, weight_with_error)
        self.bin_dim = bin_dim
        self.calc_moments = calc_moments
        # W:L1 split for the non-moments path. A pure EMD spectrum loss is (1.0, 0.0); routing it
        # through this class keeps the -inf/ROI-mask-safe bin handling that the bare
        # WassersteinLossWeighted lacks.
        self.ws_weight = float(ws_weight)
        self.l1_weight = float(l1_weight)

    def compute_moments(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.arange(dist.size(self.bin_dim), dtype=dist.dtype, device=dist.device)
        if self.bin_dim < 0:
            self.bin_dim += dist.dim()
        dims = [1] * dist.dim()
        dims[self.bin_dim] = dist.size(self.bin_dim)
        x = x.view(*dims)
        dims = [dist.size(i) for i in range(dist.dim())]
        dims[self.bin_dim] = 1
        x = x.repeat(*dims)
        mean = torch.sum(dist * x, dim=self.bin_dim, keepdim=True)
        var = torch.sum(dist * (x - mean) ** 2, dim=self.bin_dim, keepdim=True)
        return mean, var

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        hist_size = target.size(self.bin_dim)
        mask = ~(torch.isfinite(target) & torch.isfinite(prediction))

        # Bin axis for the Wasserstein term. When masked values force a reshape to (N, hist_size)
        # the bin axis becomes the last one; pass it per call rather than mutating the shared
        # sub-module's `.dim`.
        ws_dim = self.bin_dim
        if mask.any():
            assert not self.weight_with_error, "HistogramLoss does not support weighting with error when there are masked values."
            ws_dim = -1
            # Drop the masked (-inf) voxels and reshape to (n_valid_voxels, hist_size). The bin axis
            # must be moved to LAST before the boolean-mask flatten so each voxel's histogram stays
            # contiguous; otherwise the reshape interleaves different voxels' bins.
            bd = self.bin_dim if self.bin_dim >= 0 else self.bin_dim + target.dim()
            perm = [d for d in range(target.dim()) if d != bd] + [bd]
            mask_p = mask.permute(*perm)
            target = target.permute(*perm)[~mask_p].view(-1, hist_size)
            prediction = prediction.permute(*perm)[~mask_p].view(-1, hist_size)

        target_sum = torch.clamp(torch.sum(target, dim=ws_dim, keepdim=True), min=1e-8)
        target = target / target_sum
        pred_sum = torch.clamp(torch.sum(prediction, dim=ws_dim, keepdim=True), min=1e-8)
        prediction = prediction / pred_sum

        ws = self.wasserstein_loss(target, prediction, input, dim=ws_dim)
        l1 = self.l1_loss(target, prediction, input)

        if self.penalize_out_of_range:
            target_max_index = (target > 1e-8).nonzero(as_tuple=True)[0].max().item() if (target > 1e-8).any() else 0
            prediction_max_index = (prediction > 1e-8).nonzero(as_tuple=True)[0].max().item() if (prediction > 1e-8).any() else 0
            distance = abs(target_max_index - prediction_max_index)
            if distance > 0:
                ws *= 1.0 + distance * 0.1

        if self.calc_moments:
            bin_count = prediction.size(self.bin_dim) if len(prediction.shape) > self.bin_dim else 1
            pred_mean, pred_var = self.compute_moments(prediction)
            target_mean, target_var = self.compute_moments(target)
            moments_loss = (nn.functional.l1_loss(pred_mean, target_mean) / bin_count) + \
                           (nn.functional.l1_loss(pred_var, target_var) / bin_count)
            moments_loss = torch.clamp(moments_loss, max=1.0, min=0.0)
            return ws * 0.33 + l1 * 0.33 + moments_loss * 0.34
        else:
            return ws * self.ws_weight + l1 * self.l1_weight


class SMAPERegionBalancedLoss(Loss):
    """Metric-targeted, region-balanced SMAPE loss.

    Trains the SAME functional the evaluation scores (per-voxel SMAPE = 2|p−t|/(|p|+|t|+eps)) and
    rebalances the gradient across the three regions the metrics actually score, instead of letting
    voxel counts decide (a typical field is bulk:ring:beam ≈ 92% : 3% : <0.1%):

      * beam  — t ≥ beam_rel · max(t)   (drives top90 + the GPR high-dose criterion)
      * ring  — ring_rel ≤ t < beam_rel (the bright-ring scatter metric region)
      * bulk  — t < ring_rel AND statistically reliable (drives the noise-aware scatter metric)
      * noise — MC-noise voxels (joined error ≥ err_threshold) get a ONE-SIDED HINGE instead of a
        fit: zero cost while the prediction stays below hinge_rel · max(t), SMAPE-style cost above.
        The hinge pins the noise region down ("we don't know the exact value, but we know it is
        small") without fitting the MC noise itself.

    Each region's per-voxel cost is averaged separately and the region means are averaged, so every
    region receives the same total gradient mass per field. SMAPE is scale-invariant, so under
    LinearNormalizer(0,1) this optimizes physical relative accuracy directly.
    """

    def __init__(self, eps: float = 1e-4, beam_rel: float = 5e-2, ring_rel: float = 5e-3,
                 err_threshold: float = 0.75, hinge_rel: float = None, core: str = "smape"):
        super().__init__()
        # eps must sit at/below the reliable-bulk median (~6e-5 normalized) so the bulk denominator
        # is not eps-dominated; the reliability mask + hinge already guard noise amplification.
        self.eps = float(eps)
        # Per-voxel core:
        #   "smape"    — 2|p−t|/(|p|+|t|+eps): the metric's own functional, but saturates on the
        #                over-prediction side (misplaced bright structure is nearly free to keep).
        #   "logratio" — |log10((p+eps)/(t+eps))| clamped to 6 decades: non-saturating relative
        #                error; gradient ~ 1/(p+eps) pushes ghosts down at full strength while
        #                keeping strong under-prediction gradients. Monotone in relative error.
        assert core in ("smape", "logratio"), f"Unknown core {core!r}"
        self.core = core
        self.beam_rel = float(beam_rel)
        self.ring_rel = float(ring_rel)
        self.err_threshold = float(err_threshold)
        # Hinge threshold for the noise region (relative to per-field max). Defaults to ring_rel:
        # a noise voxel predicting above the bright-ring threshold is unambiguously wrong.
        self.hinge_rel = float(hinge_rel) if hinge_rel is not None else float(ring_rel)

    def _gt_error(self, input: TrainingInputData, like: Tensor):
        gt = getattr(input, "ground_truth", None) if input is not None else None
        err = None
        if gt is not None:
            if hasattr(gt, "scatter_field") and gt.scatter_field is not None and getattr(gt.scatter_field, "error", None) is not None:
                err = gt.scatter_field.error
                if hasattr(gt, "direct_beam") and gt.direct_beam is not None and getattr(gt.direct_beam, "error", None) is not None:
                    err = (err + gt.direct_beam.error) / 2.0
            elif getattr(gt, "error", None) is not None:
                err = gt.error
        if err is None:
            return None
        if err.shape != like.shape and err.numel() == like.numel():
            err = err.reshape(like.shape)
        elif err.shape != like.shape:
            return None
        return err

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        if prediction.shape != target.shape:
            if prediction.ndim == target.ndim + 1 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            elif target.ndim == prediction.ndim + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        t = target.masked_fill(~valid, 0.0)
        p = prediction.masked_fill(~valid, 0.0)

        if self.core == "logratio":
            smape = (torch.log10(p.abs() + self.eps) - torch.log10(t.abs() + self.eps)).abs().clamp(max=6.0)
        else:
            smape = (2.0 * (p - t).abs()) / (p.abs() + t.abs() + self.eps)   # per-voxel metric core, [0,2]
        smape = torch.nan_to_num(smape, nan=0.0, posinf=0.0, neginf=0.0)

        if smape.ndim <= 1:                      # row mode (no spatial structure): no region balancing
            return reduce_per_sample(smape, valid)

        reduce_dims = tuple(range(1, smape.ndim))
        tmax = t.amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
        rel = t / tmax

        err = self._gt_error(input, t)
        reliable = (err < self.err_threshold) if err is not None else torch.ones_like(valid)

        beam = valid & (rel >= self.beam_rel)
        ring = valid & (rel >= self.ring_rel) & (rel < self.beam_rel)
        bulk = valid & (rel < self.ring_rel) & reliable
        noise = valid & (rel < self.ring_rel) & ~reliable

        # One-sided hinge for the noise region: free below tau = hinge_rel * max(t), cost above.
        tau = self.hinge_rel * tmax
        if self.core == "logratio":
            hinge = (torch.log10(p.abs() + self.eps) - torch.log10(tau + self.eps)).clamp(min=0.0, max=6.0)
        else:
            over = (p - tau).clamp(min=0.0)
            hinge = (2.0 * over) / (p.abs() + tau + self.eps)
        hinge = torch.nan_to_num(hinge, nan=0.0, posinf=0.0, neginf=0.0)

        def region_mean(values, mask):
            cnt = mask.to(values.dtype).sum(dim=reduce_dims)
            s = (values * mask.to(values.dtype)).sum(dim=reduce_dims)
            return s / cnt.clamp(min=1.0), (cnt > 0)

        means, present = zip(*(region_mean(v, m) for v, m in
                               ((smape, beam), (smape, ring), (smape, bulk), (hinge, noise))))
        means = torch.stack(means, dim=0)                    # [4, B]
        present = torch.stack(present, dim=0).to(smape.dtype)
        # average over the regions PRESENT in each field (equal gradient mass per present region)
        return (means * present).sum(dim=0) / present.sum(dim=0).clamp(min=1.0)


class TwoROIGammaLoss(nn.Module):
    """Loss over the shared beam / scatter / floor ROIs (radfield3dnn.roi.compute_roi_masks),
    matched 1:1 to the air-kerma scatter metric and the ROI voxel sampler.

    The three ROIs are derived FROM THE TARGET (the joined flux the model is trained on; in the
    beam, joined ≈ direct so the beam ROI agrees with the metric's direct-channel definition):
      beam    = target >= beam_rel  * max(target)
      scatter = NOT beam AND target >= scatter_lo * max(target)
      floor   = the rest (the MC-noise floor; light smoothness glue)

    Each term is the MEAN over its ROI's voxels, so a small beam and a large scatter region
    contribute EQUALLY (per-ROI influence equalised by the target voxel count). Non-sampled voxels
    carry -inf (the ROI-sampler / masking convention shared with the other losses) and are excluded.

    beam term = L1 (absolute, matches the beam air-kerma SMAPE); scatter/floor = bounded symmetric
    SMAPE |p−t|/(|p|+|t|+eps). Optional 1-voxel-DTA soft gamma (w_gamma>0). Expects pred/target as
    full volumes (B, X, Y, Z) in linear, normalised units (needed for the spatial shifts).
    """
    def __init__(self, beam_rel=0.05, scatter_lo=5e-5,
                 w_beam=1.0, w_scatter=1.0, w_floor=0.05, w_gamma=0.0,
                 gamma_crit_beam=0.03, gamma_crit_scatter=0.10,
                 eps_scatter=5e-5, softmin_tau=0.1):
        super().__init__()
        self.beam_rel, self.scatter_lo = beam_rel, scatter_lo
        self.w = dict(beam=w_beam, scatter=w_scatter, floor=w_floor, gamma=w_gamma)
        self.cb, self.cs = gamma_crit_beam, gamma_crit_scatter
        self.eps, self.tau = eps_scatter, softmin_tau
        self.stage3 = False

    @staticmethod
    def _shifted_stack(x):
        """All 27 shifts within ±1 voxel (includes identity). x: (B,X,Y,Z)."""
        xp = F.pad(x.unsqueeze(1), (1,1,1,1,1,1), mode='replicate').squeeze(1)
        return torch.stack([xp[:, 1+dx:xp.shape[1]-1+dx,
                                  1+dy:xp.shape[2]-1+dy,
                                  1+dz:xp.shape[3]-1+dz]
                            for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)],
                           dim=0)                                   # (27,B,X,Y,Z)

    def _soft_gamma(self, pred, target, mask, crit, ref):
        """Soft 1-voxel-DTA gamma: per ref voxel, soft-min over shifted
        predictions of |pred_shift - target| / (crit*ref); hinge at 1."""
        if not mask.any():
            return pred.new_zeros(())
        shifts = self._shifted_stack(pred)                          # (27,B,...)
        g = (shifts - target.unsqueeze(0)).abs() / (crit * ref + 1e-12)
        g = -self.tau * torch.logsumexp(-g / self.tau, dim=0)       # soft-min
        return F.relu(g[mask] - 1.0).mean()                         # only failures

    def forward(self, target: Tensor, prediction: Tensor, input: TrainingInputData) -> Tensor:
        from radfield3dnn.roi import compute_roi_masks
        finite = torch.isfinite(target)   # -inf = not-sampled / masked -> excluded from every ROI
        # ROIs FROM THE TARGET (joined flux). Pass it as both 'direct' and 'joined': the joined-max
        # beam ≈ the metric's direct-channel beam. Zero-fill the -inf voxels so the per-field max
        # stays finite; they are re-excluded via `finite` below.
        safe = torch.where(finite, target, torch.zeros_like(target))
        m_beam, m_scatter, m_floor = compute_roi_masks(safe, safe, self.beam_rel, self.scatter_lo)
        m_beam, m_scatter, m_floor = m_beam & finite, m_scatter & finite, m_floor & finite

        loss = prediction.new_zeros(())
        # Per-term breakdown for the training debug probe (callbacks/debug_probe.py); floats only.
        self.last_terms: dict[str, float] = {}
        if self.w['beam'] > 0 and m_beam.any():
            # absolute L1 — matches the beam air-kerma SMAPE
            term = (prediction[m_beam] - target[m_beam]).abs().mean()
            self.last_terms['beam'] = float(term.detach())
            self.last_terms['n_beam'] = int(m_beam.sum())
            loss = loss + self.w['beam'] * term
        for name, m in (('scatter', m_scatter), ('floor', m_floor)):
            if self.w[name] > 0 and m.any():
                # bounded symmetric SMAPE core (≤1 per voxel) — the metric's own functional
                p, t = prediction[m], target[m]
                rel = (p - t).abs() / (p.abs() + t.abs() + self.eps)
                term = rel.mean()   # normalises by THIS ROI's target voxel count -> equal influence
                self.last_terms[name] = float(term.detach())
                self.last_terms[f'n_{name}'] = int(m.sum())
                loss = loss + self.w[name] * term
        if self.stage3 and self.w['gamma'] > 0:
            tmax = safe.detach().amax()
            loss = loss + self.w['gamma'] * (
                self._soft_gamma(prediction, target, m_beam,    self.cb, tmax)
              + self._soft_gamma(prediction, target, m_scatter, self.cs,
                                 target.detach().clamp_min(self.scatter_lo)))
        return loss

"""Comparing two von Mises-Fisher mixtures on the sphere, in closed form.

A voxel's directional ground truth is a mixture ``P(x) = sum_l w_l vMF(x; mu_l, kappa_l)`` over the
unit sphere. Comparing a prediction with it needs a divergence that is symmetric, zero only when the
two agree, and — the binding constraint — EXACT AT ANY CONCENTRATION.

**Quadrature is not an option here, and the data is what rules it out.** Half the directional weight
in DS04-Angio-ArtisQ sits in lobes narrower than 5 degrees and 38 % in lobes narrower than 4; a
uniform sphere grid that resolves those needs tens of thousands of points PER VOXEL, against
262,144 voxels a field. A Jensen-Shannon estimate on any affordable grid would simply miss the
lobes it is meant to be scoring, and would report agreement between two models that both missed.

What makes a closed form possible is one identity::

    integral over S^2 of exp(a . x) dS = 4 pi sinh(|a|) / |a| = 1 / C(|a|)

with ``C(k) = k / (4 pi sinh k)`` the vMF normaliser. Two vMF densities therefore overlap by

    integral f1 f2 dS = C(k1) C(k2) / C(|k1 mu1 + k2 mu2|)

exactly, at any concentration, and a mixture is the weighted sum of its L*L pairs — sixteen terms
for four lobes. Everything below is that identity, carried in logs because kappa reaches 7e4 here
and ``sinh`` overflows a float32 above about 89.
"""
import math

import torch

LOG_4PI = math.log(4.0 * math.pi)


def log_vmf_normaliser(kappa: torch.Tensor) -> torch.Tensor:
    """``log C(kappa)``, the log of the vMF normalising constant on S^2.

    ``log sinh k = k + log1p(-exp(-2k)) - log 2`` keeps it finite for large kappa, where sinh itself
    overflows. For small kappa the same expression tends to ``-log(4 pi)`` — the uniform density —
    which is why kappa is floored rather than allowed to reach zero: an unused lobe slot and a
    uniform target (``datasets.lobes.fill_uniform``) both carry kappa = 0, and would otherwise take
    the whole field to NaN. At the floor the density is uniform to within 1e-9.
    """
    k = kappa.clamp_min(1e-4)
    log_sinh = k + torch.log1p(-torch.exp(-2.0 * k)) - math.log(2.0)
    return torch.log(k) - LOG_4PI - log_sinh


def log_overlap(weight_a, mean_a, kappa_a, weight_b, mean_b, kappa_b) -> torch.Tensor:
    """``log integral P Q dS`` for two mixtures, over every pair of lobes.

    Shapes ``(..., L)`` for the weights and concentrations and ``(..., L, 3)`` for the unit means.
    The means must be unit-norm; a zero mean is only safe because its weight is zero, and the log
    weight then drives that pair's term to -inf rather than its direction mattering.
    """
    a = kappa_a.unsqueeze(-1) * mean_a                     # (..., L, 3)
    b = kappa_b.unsqueeze(-1) * mean_b                     # (..., M, 3)
    resultant = (a.unsqueeze(-2) + b.unsqueeze(-3)).norm(dim=-1)        # (..., L, M)

    log_pair = (log_vmf_normaliser(kappa_a).unsqueeze(-1)
                + log_vmf_normaliser(kappa_b).unsqueeze(-2)
                - log_vmf_normaliser(resultant))
    log_w = (torch.log(weight_a.clamp_min(1e-12)).unsqueeze(-1)
             + torch.log(weight_b.clamp_min(1e-12)).unsqueeze(-2))
    return torch.logsumexp((log_w + log_pair).flatten(-2), dim=-1)


def directional_similarity(pred, target, eps: float = 1e-12) -> torch.Tensor:
    """Correlation of the two densities: ``<P,Q> / sqrt(<P,P><Q,Q>)``, in ``(0, 1]``.

    One where the mixtures agree and falling towards zero as they separate, with no threshold
    anywhere and no sensitivity to how sharp the lobes are — which is what disqualifies IoU, whose
    answer depends entirely on where a density is cut into a region.

    `pred` and `target` are ``(weight, mean, kappa)`` triples.
    """
    pw, pm, pk = pred
    tw, tm, tk = target
    log_pq = log_overlap(pw, pm, pk, tw, tm, tk)
    log_pp = log_overlap(pw, pm, pk, pw, pm, pk)
    log_qq = log_overlap(tw, tm, tk, tw, tm, tk)
    return torch.exp(log_pq - 0.5 * (log_pp + log_qq)).clamp(0.0, 1.0)


def cauchy_schwarz_divergence(pred, target) -> torch.Tensor:
    """``-log`` of the similarity above: zero when the mixtures agree, growing without bound.

    The Cauchy-Schwarz divergence, which is to a mixture what Jensen-Shannon is to a pair of simple
    densities — symmetric, zero only on agreement — and unlike JS it has a closed form here. That is
    the whole reason to prefer it: see the module docstring for why quadrature cannot be used.
    """
    return -torch.log(directional_similarity(pred, target).clamp_min(1e-12))


def dominant_angle_error_deg(pred, target) -> torch.Tensor:
    """Angle between the two mixtures' HEAVIEST lobes, in degrees: where is the flux mostly going?

    The interpretable companion to `directional_similarity`, which says how well the whole
    distributions agree but not in a unit anyone can picture. Undefined where a mixture has two
    lobes of near-equal weight — the argmax flips between them — so read it alongside the
    similarity, never instead of it.
    """
    pw, pm, _ = pred
    tw, tm, _ = target
    pick = lambda w, m: torch.gather(m, -2, w.argmax(-1, keepdim=True).unsqueeze(-1).expand(*w.shape[:-1], 1, 3)).squeeze(-2)
    cos = (pick(pw, pm) * pick(tw, tm)).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))

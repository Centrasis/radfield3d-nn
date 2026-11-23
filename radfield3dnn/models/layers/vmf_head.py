"""A per-point von Mises-Fisher mixture head: where the flux at a point is GOING.

Emits ``L`` lobes of ``(weight, mean, kappa)`` from the trunk's features. Each output is constrained
to its domain by construction rather than by a penalty, because a loss term that merely discourages
an invalid mixture still lets one through — and the divergence it feeds is undefined on one:

  weight  softmax over the lobes      a mixture: non-negative, summing to one
  mean    L2-normalised 3-vector      a direction on the unit sphere
  kappa   exp of a SIGMOID-BOUNDED log  concentration across ~7 decades

**Why kappa is predicted in the log, and bounded softly.** The data spans kappa from 0.02 to 7e4 —
half the directional weight sits in lobes narrower than 5 degrees, the rest much broader. A
linear output cannot cover that, and a hard clamp on the log zeroes the gradient of a lobe the
instant it touches a bound, freezing it there. A sigmoid over ``[log k_min, log k_max]`` still
guarantees the bound but lets the gradient DECAY as a lobe approaches it rather than cut off at a
threshold — so a lobe pushed toward a bound can come back. It does not keep a gradient everywhere:
in float32 the sigmoid saturates to exactly 1 near a logit of 17, and the gradient there is zero.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOBE_PARAMS = 5  # weight logit, mean x/y/z, kappa logit


class VMFLobeHead(nn.Module):
    def __init__(self, d_in: int, lobes: int = 4, hidden: int = 128,
                 kappa_bounds: tuple[float, float] = (1e-2, 2e5), kappa_init: float = 100.0):
        super().__init__()
        assert lobes >= 1, "a mixture needs at least one lobe"
        assert 0 < kappa_bounds[0] < kappa_init < kappa_bounds[1], \
            f"kappa_init {kappa_init} must lie strictly inside {kappa_bounds}"
        self.lobes = int(lobes)
        self._log_k_min = math.log(kappa_bounds[0])
        self._log_k_max = math.log(kappa_bounds[1])
        self._kappa_init = float(kappa_init)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, self.lobes * LOBE_PARAMS))

    def initialize(self):
        """Start every lobe at the same, typical concentration, with equal weights.

        The kappa bias is set so the initial concentration is `kappa_init` — about the median of the
        data's mass — instead of the midpoint of a 7-decade log range, which would start every lobe
        near kappa 50 by accident and spend the first epochs only moving it. The output weights are
        small so the means start near the bias directions and the lobes are not born identical.
        """
        out = self.net[-1]
        nn.init.normal_(out.weight, std=1e-3)
        bias = torch.zeros(self.lobes, LOBE_PARAMS)
        fraction = (math.log(self._kappa_init) - self._log_k_min) / (self._log_k_max - self._log_k_min)
        bias[:, 4] = math.log(fraction / (1.0 - fraction))
        # Distinct initial means, spread over the sphere, so the lobes do not start as one.
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for l in range(self.lobes):
            z = 1.0 - 2.0 * (l + 0.5) / self.lobes
            r = math.sqrt(max(0.0, 1.0 - z * z))
            bias[l, 1:4] = torch.tensor([r * math.cos(golden * l), r * math.sin(golden * l), z])
        with torch.no_grad():
            out.bias.copy_(bias.flatten())

    def forward(self, features: torch.Tensor):
        """``(..., d_in)`` -> weight ``(..., L)``, mean ``(..., L, 3)``, kappa ``(..., L)``."""
        raw = self.net(features).unflatten(-1, (self.lobes, LOBE_PARAMS)).float()
        weight = torch.softmax(raw[..., 0], dim=-1)
        mean = F.normalize(raw[..., 1:4], dim=-1, eps=1e-6)
        log_kappa = self._log_k_min + (self._log_k_max - self._log_k_min) * torch.sigmoid(raw[..., 4])
        return weight, mean, torch.exp(log_kappa)

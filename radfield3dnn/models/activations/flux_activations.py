import torch
from torch import nn
from torch import Tensor


# Fraction of the codomain ABOVE the floor at which a LINEAR-normalizer flux head starts.
# linear0_1 flux targets are crushed near 0, so starting the head at the codomain MIDPOINT (0.5)
# puts most voxels far above their target → the bulk L1 gradient sweeps the whole field DOWN and
# overshoots into the all-zero basin. Starting just above the floor (~1% of range) instead means
# the bulk is already ~correct (≈0 gradient) and the only live signal is the bright beam/scatter
# voxels pulling UP.
LINEAR_INIT_FRACTION = 0.01


def flux_head_init_bias(activation, normalizer) -> float:
    """Pre-activation bias for the flux output head, chosen by the NORMALIZER family:

      * LINEAR normalizer (linear0_1): the target bulk sits at the codomain floor, so
        start the head just above it (``LINEAR_INIT_FRACTION`` of the range) — see the constant.
      * LOG-like normalizer (asinh tonemap): the transform already spreads the dynamic
        range, so the codomain MIDPOINT (each activation's ``init_bias``, the max-gradient point) is
        the right, well-conditioned start.
    """
    # Duck-typed normalizer family detection (avoids importing the normalizer classes here).
    is_log_like = hasattr(normalizer, "log_max") or any(
        k in type(normalizer).__name__.lower() for k in ("log", "asinh", "tonemap"))
    if is_log_like or not hasattr(activation, "bias_for_output"):
        return float(getattr(activation, "init_bias", 0.0))
    lo, hi = getattr(normalizer, "range", (0.0, 1.0))
    target_output = lo + (hi - lo) * LINEAR_INIT_FRACTION
    return float(activation.bias_for_output(target_output))


class GradientConservingClamping(nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    @property
    def init_bias(self) -> float:
        # Clamp is the identity inside its range, so the pre-activation bias that places the
        # initial output at the codomain midpoint (max distance from both saturating floors)
        # is the midpoint itself.
        return 0.5 * (self.min_value + self.max_value)

    def bias_for_output(self, y: float) -> float:
        # Identity inside the range → the pre-activation producing output y is y itself (clamped).
        return float(min(max(y, self.min_value), self.max_value))

    def forward(self, x: Tensor) -> Tensor:
        clamped_x = torch.clamp(x, min=self.min_value, max=self.max_value)
        return x + (clamped_x - x).detach()


class SoftClip(nn.Module):
    """Smooth, differentiable analogue of GradientConservingClamping([0,1]).

    y = 0.5 * (tanh(k * x) + 1)  maps R -> (0, 1).

    At x=0 the output is 0.5 (codomain midpoint), gradient is k/2; outside the
    linear regime the gradient decays smoothly to 0 but never vanishes
    exactly — so a prediction that overshoots into the (-inf, 0) or
    (1, inf) zones still receives a recovery gradient. This avoids the
    GradientConservingClamping failure mode where, once the network
    pushes a sample to z << 0, the clamped output is 0, the gradient is
    the identity, the next batch with mostly-zero targets pulls z further
    negative and the prediction is permanently stuck at 0.

    `k` is the slope at z=0 (times 2). Larger k -> sharper transition,
    closer to a hard clamp; the default k=1 follows the smooth-bounded
    density activation used in Mip-NeRF / NeRF-W. NOTE: this is NOT
    related to the "SoftCLIP" cross-modal contrastive paper (Gao et al.
    ECCV 2024); the name here refers to the generic DSP / DL meaning of
    "soft clipping" (any smooth saturating bounded activation).
    """

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = float(k)

    @property
    def init_bias(self) -> float:
        # tanh is centered at 0: output 0.5 (codomain midpoint), maximum gradient k/2.
        return 0.0

    def bias_for_output(self, y: float) -> float:
        # y = 0.5*(tanh(kx)+1) → x = atanh(2y-1)/k. Clamp y off the asymptotes.
        import math
        y = min(max(y, 1e-6), 1.0 - 1e-6)
        return float(math.atanh(2.0 * y - 1.0) / self.k)

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (torch.tanh(self.k * x) + 1.0)


class LogitSigmoid(nn.Module):
    """Logistic flux activation: the network emits a *logit* mapped to (0, 1) by sigmoid.

    y = sigmoid(z),  z = the network output, clamped (gradient-conserving) to
    ``[-logit_range, logit_range]`` (default ±30).

    Rationale (HDR, linear0_1 targets): the scatter band is crushed to ~1e-3 of the peak. A
    hard ``GradientConservingClamping([0,1])`` floors negatives at exactly 0 with identity
    gradient → the "predict-0-forever" lock-in. Sigmoid instead spans the full dynamic range
    *inside* (0,1): the network must **split its logits across ~[-30, 30]** — sigmoid(-30)≈9e-14
    (background) … sigmoid(0)=0.5 … sigmoid(30)≈1 (peak) — i.e. ~13 decades, so it can represent
    both the crushed scatter and the beam peak in linear-normalised space, with a nonzero
    recovery gradient everywhere (no lock-in). The ±30 gradient-conserving clamp keeps the logit
    finite (sigmoid saturates to exact 0/1 in fp beyond that, killing the tail gradient) while
    still passing an identity gradient back if the network overshoots, so it can recover.

    Pairs with a (0,1)-codomain normalizer (``LinearNormalizer(0,1)``). The output-layer bias
    is zero-initialized (``init_bias``), starting the head at sigmoid(0)=0.5 — the maximum-
    gradient point.
    """

    def __init__(self, logit_range: float = 30.0):
        super().__init__()
        self.logit_range = float(logit_range)

    @property
    def init_bias(self) -> float:
        # sigmoid(0) = 0.5: codomain midpoint and maximum-gradient point of the logistic.
        return 0.0

    def bias_for_output(self, y: float) -> float:
        # y = sigmoid(z) → z = logit(y) = log(y/(1-y)); clamped to the grad-conserving logit range.
        import math
        y = min(max(y, 1e-12), 1.0 - 1e-12)
        z = math.log(y / (1.0 - y))
        return float(min(max(z, -self.logit_range), self.logit_range))

    def forward(self, x: Tensor) -> Tensor:
        z = x + (torch.clamp(x, -self.logit_range, self.logit_range) - x).detach()
        return torch.sigmoid(z)


class ArcTan(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (4.0 / torch.pi) * torch.atan(x) # Scaled to [-1, 1] for x in [-1, 1]


class SymmetricSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(x) - 1.0  # Scaled to [-1, 1] for x in [-inf, inf]


class SmoothedTanh(nn.Module):
    def __init__(self, scale: float = 1.2):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x / self.scale) * self.scale  # Scaled tanh for smoother transitions

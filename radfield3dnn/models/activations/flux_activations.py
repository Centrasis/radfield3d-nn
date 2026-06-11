import torch
from torch import nn
from torch import Tensor


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

    def forward(self, x: Tensor) -> Tensor:
        clamped_x = torch.clamp(x, min=self.min_value, max=self.max_value)
        return x + (clamped_x - x).detach()


class SoftClip(nn.Module):
    """Smooth, differentiable analogue of GradientConservingClamping([0,1]).

    y = 0.5 * (tanh(k * x) + 1)  maps R -> (0, 1).

    At x=0 the output is 0.5 (codomain midpoint, same starting point as the
    historic 0.5 offset + linear-clamp), gradient is k/2; outside the
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
    gradient point — matching the published ``_init_decoders`` zero-bias.
    """

    def __init__(self, logit_range: float = 30.0):
        super().__init__()
        self.logit_range = float(logit_range)

    @property
    def init_bias(self) -> float:
        # sigmoid(0) = 0.5: codomain midpoint and maximum-gradient point of the logistic.
        return 0.0

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

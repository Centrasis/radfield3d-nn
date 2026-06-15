from torch.nn import Module
from torch import Tensor
import torch
from torch import nn


# Positivity step used before sum-normalisation. The choice matters when the
# upstream layer has NO biases (e.g. tcnn CutlassMLP / the C++ spectrum
# decoder): ReLU on zero-mean random logits zeros ~half the bins AND kills
# their gradient — those bins can never become active unless something else
# drags their pre-activation across 0, so at init the spectrum sits near
# "half-uniform" with no way out. Softplus is always > 0, gradient ∈ (0,1)
# everywhere, softplus(0)=log2; it gives the same uniform-by-default behaviour
# but every bin keeps a live gradient, so the optimiser can shape the output
# toward the peaked target without needing biases or a learned shift.
_POSITIVITY = {
    "relu":     nn.ReLU,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}


class HistogramNormalize(Module):
    def __init__(self, dim: int = 0, enforce_positivity: bool = True,
                 positivity: str | None = None):
        """``positivity`` overrides the legacy ``enforce_positivity`` flag:

        * ``"relu"``      — ReLU (legacy default, matches pure-Python PBRFNet).
        * ``"softplus"``  — recommended for bias-less upstream layers
                            (tcnn CutlassMLP / PBRFNetCPP spectrum head):
                            no dead-bin gradient at init.
        * ``"identity"``  — pass-through (assumes input already non-negative).

        Defaults to ReLU when ``enforce_positivity=True``.
        """
        super().__init__()
        self.dim = dim
        if positivity is None:
            positivity = "relu" if enforce_positivity else "identity"
        if positivity not in _POSITIVITY:
            raise ValueError(
                f"positivity must be one of {sorted(_POSITIVITY)}, got {positivity!r}"
            )
        self.positivity = _POSITIVITY[positivity]()
        self._positivity_kind = positivity

    def forward(self, hists: Tensor) -> Tensor:
        hists = self.positivity(hists)
        # Branch-free nan/inf -> 0 cleanup (a data-dependent `if ...any()` would block
        # torch.export / ONNX).
        hists = torch.where(torch.isfinite(hists), hists, torch.zeros_like(hists))
        sum = torch.sum(hists, dim=self.dim, keepdim=True)
        sum = torch.clamp(sum, min=1e-8)
        return hists / sum

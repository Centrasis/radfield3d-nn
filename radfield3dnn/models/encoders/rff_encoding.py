import math
import torch
from torch import Tensor, nn

from .base import EncodingBase


class RandomFourierFeatures(EncodingBase):
    """Gaussian Random Fourier Features positional encoding (Tancik et al., NeurIPS 2020,
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains").

    ``γ(x) = [ sin(2π B x), cos(2π B x) ]`` with a FIXED random matrix ``B ∈ R^{d_input × num_features}``,
    ``B_ij ~ N(0, σ²)``. The bandwidth ``σ`` directly sets the frequency content the MLP can represent:

      * SMALL σ → smooth, low-frequency encoding — the right match for the diffuse scatter field (the
        opposite end from a hashgrid, which biases toward high frequency).
      * LARGE σ → high-frequency detail (and aliasing/noise if too large).

    Unlike a multi-resolution hashgrid, RFF is a **continuous, locality-preserving** function of position
    (no hash collisions) and has only ``d_input·num_features`` fixed (non-learned) parameters in ``B``.
    """

    def __init__(self, num_features: int = 96, d_input: int = 3, sigma: float = 4.0,
                 append_input: bool = True, dim: int = -1, seed: int = 0):
        super().__init__()
        self.num_features = int(num_features)
        self.d_input = int(d_input)
        self.sigma = float(sigma)
        self.append_input = bool(append_input)
        self.dim = dim
        g = torch.Generator().manual_seed(int(seed))
        B = torch.randn(self.d_input, self.num_features, generator=g) * self.sigma
        self.register_buffer("B", B)  # fixed; casts with the module (fp16/fp32)

    def calc_encoded_dim(self) -> int:
        return 2 * self.num_features + (self.d_input if self.append_input else 0)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.d_input, f"Input last dim should be {self.d_input}, got {x.shape[-1]}"
        proj = 2.0 * math.pi * (x @ self.B.to(x.dtype))          # [..., num_features]
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=self.dim)
        if self.append_input:
            enc = torch.cat([enc, x], dim=self.dim)
        return enc

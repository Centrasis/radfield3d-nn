import torch
from torch import nn, Tensor
from .base import EncodingBase
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class SinusoidalFrequencyEncoding(EncodingBase):
    # `use_tcnn` defaults to False: the pure-PyTorch path is ONNX-exportable (no CUDA custom op) and
    # is numerically EXACT to tcnn's "Frequency" encoding — verified element-wise. tcnn computes, per
    # input dim and frequency i, the pair (sin(2^i·π·x), cos(2^i·π·x)) interleaved, dim-major.
    def __init__(self, pos_enc_dim: int, d_input: int, append_input: bool = False, dim: int = -1, use_tcnn: bool = False):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.d_input = d_input
        self.append_input = append_input
        self.dim = dim
        self._use_tcnn = bool(use_tcnn and tcnn is not None)
        if use_tcnn and tcnn is None:
            raise ImportError("tinycudann is not installed, cannot use tcnn encoding.")
        if self._use_tcnn:
            self.encoding = tcnn.Encoding(
                n_input_dims=d_input,
                encoding_config={"otype": "Frequency", "n_frequencies": pos_enc_dim},
            )
        else:
            self.register_buffer(
                "_freqs", (2.0 ** torch.arange(pos_enc_dim, dtype=torch.float32)) * torch.pi,
                persistent=False)

    def calc_encoded_dim(self) -> int:
        return 2 * self.pos_enc_dim * self.d_input + (self.d_input if self.append_input else 0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_input)
        assert x.shape[-1] == self.d_input, f"Input tensor last dim should be {self.d_input}, got {x.shape[-1]}"
        if self._use_tcnn:
            enc = self.encoding(x.contiguous()).to(x.dtype).contiguous()
            if self.append_input:
                enc = torch.cat([enc, x], dim=-1)
            return enc

        # Pure-PyTorch, tcnn-matched: per dim & frequency -> (sin, cos) interleaved, dim-major.
        ang = x.unsqueeze(-1) * self._freqs.to(x.dtype)                      # (..., d_input, P)
        enc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)         # (..., d_input, P, 2)
        enc = enc.reshape(*x.shape[:-1], 2 * self.pos_enc_dim * self.d_input)
        if self.append_input:
            enc = torch.cat([enc, x], dim=-1)
        return enc


class AngularSinusoidalFrequencyEncoding(SinusoidalFrequencyEncoding):
    def __init__(self, pos_enc_dim, append_input = False, dim = -1):
        super().__init__(pos_enc_dim, 2, append_input, dim)

    def forward(self, x):
        assert x.shape[-1] == 3, f"Input tensor last dim should be 3, got {x.shape[-1]}"
        x = AngularSinusoidalFrequencyEncoding.map_direction_vector2spherical_coords(x)
        return super().forward(x)
    
    @staticmethod
    def map_direction_vector2spherical_coords(direction: Tensor) -> Tensor:
        """Convert direction vectors (x,y,z) to spherical coordinates (theta, phi).
        This version uses torch.atan2 for better numerical stability.
        Args:
            direction: Tensor of shape (..., 3) containing cartesian direction vectors
        Returns:
            Tensor of shape (..., 2) containing spherical coordinates (theta, phi)
        """
        # Normalize the vectors
        direction = F.normalize(direction, dim=-1, p=2)
        
        # Extract x, y, z components
        x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
        
        # Convert to spherical coordinates
        # theta: angle from z-axis (0 to π)
        # phi: angle in xy-plane from x-axis (0 to 2π)
        theta = torch.acos(torch.clamp(z, -1.0 + 1e-8, 1.0 - 1e-8))
        phi = torch.atan2(y, x)
        
        # Normalize to [0, 1] range for more stable training
        theta = theta / torch.pi
        phi = (phi + torch.pi) / (2 * torch.pi)
        
        return torch.stack([theta, phi], dim=-1)

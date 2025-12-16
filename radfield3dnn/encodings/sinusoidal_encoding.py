import torch
from torch import nn, Tensor
from .base import EncodingBase
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class SinusoidalFrequencyEncoding(EncodingBase):
    def __init__(self, pos_enc_dim: int, d_input: int, append_input: bool = False, dim: int = -1, use_tcnn: bool = True):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.d_input = d_input
        self.append_input = append_input
        self.dim = dim
        if tcnn is not None and use_tcnn:
            self.encoding = tcnn.Encoding(
                n_input_dims=d_input,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": pos_enc_dim
                }
            )
        elif tcnn is None and use_tcnn:
            raise ImportError("tinycudann is not installed, cannot use tcnn encoding.")
        else:
            self.frequency_factors = 2 ** torch.linspace(0.0, pos_enc_dim - 1.0, steps=pos_enc_dim)[None, None, :]  # (1, 1, pos_enc_dim)

    def calc_encoded_dim(self) -> int:
        return 2 * self.pos_enc_dim * self.d_input + (self.d_input if self.append_input else 0)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_input)
        assert x.shape[-1] == self.d_input, f"Input tensor last dim should be {self.d_input}, got {x.shape[-1]}"

        if tcnn is None:
            if self.frequency_factors.device != x.device:
                self.frequency_factors = self.frequency_factors.to(x.device)

            orig_shape = x.shape[:-1]
            x_flat = x.reshape(-1, self.d_input)  # (N, d_input)

            # (N, d_input, pos_enc_dim)
            freqs = x_flat.unsqueeze(-1) * self.frequency_factors  # (N, d_input, pos_enc_dim)
            sin = torch.sin(freqs)
            cos = torch.cos(freqs)
            # Concatenate sin and cos along last dim for each input dim
            enc = torch.cat([sin, cos], dim=-1)  # (N, d_input, 2*pos_enc_dim)
            enc = enc.reshape(x_flat.shape[0], -1)  # (N, d_input*2*pos_enc_dim)

            if self.append_input:
                enc = torch.cat([enc, x_flat], dim=-1)

            enc = enc.reshape(*orig_shape, -1)
        else:
            enc = self.encoding(x.contiguous()).to(x.dtype).contiguous()
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

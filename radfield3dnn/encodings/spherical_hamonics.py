from torch import Tensor
from torch.amp import autocast
import torch
from .base import EncodingBase
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class SphericalHarmonics(EncodingBase):
    def __init__(self, degree: int = 4, append_input: bool = True):
        super().__init__()
        if tcnn is None:
            raise ImportError("tinycudann (tcnn) is required for SphericalHarmonics.")
        self.append_input = append_input
        self.degree = degree
        if self.append_input:
            self._sht = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": degree
                        },
                        {
                            "otype": "Identity"
                        }
                    ]
                }
            )
        else:
            self._sht = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": degree,
                    "n_dims_to_encode": 3
                }
            )

    def calc_encoded_dim(self) -> int:
        return self._sht.n_output_dims
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._sht.to(*args, **kwargs)
        return self
    
    def ensure_device(self, device):
        if self._sht is not None and self._sht.params.device != device:
            self._sht.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to compute spherical harmonics transform.
        The input can be either:
        - A tensor of shape (batch_size, 3) representing Cartesian coordinates.
        - A tensor of shape (batch_size, 2) representing spherical coordinates.
        The output will be a tensor of shape (batch_size, l_max + 1, m_max + 1) where l_max is the maximum degree and m_max is the maximum order of the spherical harmonics.
        If the input is in Cartesian coordinates, it will be converted to spherical coordinates before applying the spherical harmonics transform.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3) being the normalized direction vector to encode.
        Returns:
            Tensor: Output tensor of shape (batch_size, l_max + 1, m_max + 1) containing the spherical harmonics coefficients.
        """
        assert len(x.shape) == 2 and x.shape[1] == 3, f"Expected input shape to be (batch_size, 3), got {x.shape}"
        # normalize input to unit sphere
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        self.ensure_device(x.device)
        x = x.contiguous()  # tcnn requires contiguous input
        with autocast(device_type=x.device.type, enabled=False):
            out: Tensor = self._sht(x)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out.contiguous()

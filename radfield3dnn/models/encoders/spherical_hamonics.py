from torch import Tensor
from torch.amp import autocast
import torch
import torch.nn.functional as F
from .base import EncodingBase
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


# Real spherical-harmonics basis up to degree 4 (l = 0..3 → 16 coeffs)
def _sh_components(d: Tensor, degree: int) -> Tensor:
    x = d[..., 0]; y = d[..., 1]; z = d[..., 2]
    x2 = x * x; y2 = y * y; z2 = z * z
    out = [torch.full_like(x, 0.28209479177387814)]
    if degree >= 2:
        out += [-0.48860251190291987 * y,
                 0.48860251190291987 * z,
                -0.48860251190291987 * x]
    if degree >= 3:
        out += [1.0925484305920792 * x * y,
               -1.0925484305920792 * y * z,
                0.94617469575755997 * z2 - 0.31539156525251999,
               -1.0925484305920792 * x * z,
                0.54627421529603959 * (x2 - y2)]
    if degree >= 4:
        out += [-0.59004358992664352 * y * (3.0 * x2 - y2),
                 2.8906114426405538 * x * y * z,
                -0.45704579946446572 * y * (5.0 * z2 - 1.0),
                 0.37317633259011546 * z * (5.0 * z2 - 3.0),
                -0.45704579946446572 * x * (5.0 * z2 - 1.0),
                 1.4453057213202769 * z * (x2 - y2),
                -0.59004358992664352 * x * (x2 - 3.0 * y2)]
    return torch.stack(out, dim=-1)


class TorchSphericalHarmonics(EncodingBase):
    """Pure-PyTorch spherical-harmonics direction encoding — a drop-in, ONNX-exportable replacement
    for the tiny-cuda-nn SphericalHarmonics encoding (numerically matched; no CUDA custom op)."""

    MAX_DEGREE = 4

    def __init__(self, degree: int = 4, append_input: bool = True):
        super().__init__()
        if not (1 <= degree <= self.MAX_DEGREE):
            raise ValueError(f"TorchSphericalHarmonics supports degree 1..{self.MAX_DEGREE}, got {degree}")
        self.degree = degree
        # NOTE: `append_input` is accepted for interface compatibility but, like the original tcnn
        # encoding, has NO effect: tcnn's Composite gave the Identity nested no `n_dims_to_encode`,
        # so once SphericalHarmonics consumed all 3 input dims the Identity received 0 dims and
        # appended nothing. The output is therefore degree² SH coefficients only — matched here.
        self.append_input = append_input

    def calc_encoded_dim(self) -> int:
        return self.degree * self.degree

    def ensure_device(self, device):  # stateless math — nothing to move
        pass

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2 and x.shape[1] == 3, f"Expected (batch_size, 3), got {x.shape}"
        x = F.normalize(x, p=2, dim=-1)        # unit direction (matches the original wrapper)
        d = 2.0 * x - 1.0                      # tcnn's internal [0,1]->[-1,1] remap
        return _sh_components(d, self.degree).to(x.dtype)


class TcnnSphericalHarmonics(EncodingBase):
    """The original tiny-cuda-nn SphericalHarmonics encoding (kept for the parity test)."""

    def __init__(self, degree: int = 4, append_input: bool = True, dtype=None):
        super().__init__()
        if tcnn is None:
            raise ImportError("tinycudann (tcnn) is required for TcnnSphericalHarmonics.")
        self.append_input = append_input
        self.degree = degree
        cfg = {"otype": "SphericalHarmonics", "degree": degree, "n_dims_to_encode": 3}
        # (append_input is a no-op in tcnn: the Identity nested gets 0 leftover dims; see Torch impl.)
        kw = {} if dtype is None else {"dtype": dtype}   # dtype=float32 isolates the math from fp16
        self._sht = tcnn.Encoding(n_input_dims=3, encoding_config=cfg, **kw)

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
        assert len(x.shape) == 2 and x.shape[1] == 3, f"Expected (batch_size, 3), got {x.shape}"
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        self.ensure_device(x.device)
        x = x.contiguous()
        with autocast(device_type=x.device.type, enabled=False):
            out: Tensor = self._sht(x)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out.contiguous()


# The model uses the pure-PyTorch implementation (ONNX-exportable, CUDA-custom-op-free). The tcnn
# encoding remains available as TcnnSphericalHarmonics for benchmarking / parity tests.
SphericalHarmonics = TorchSphericalHarmonics

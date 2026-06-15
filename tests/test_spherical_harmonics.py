"""Parity + timing of the pure-PyTorch SphericalHarmonics vs the tiny-cuda-nn encoding.

Run directly for the timing report:  python tests/test_spherical_harmonics.py
Self-skips when CUDA / tiny-cuda-nn are unavailable.
"""
import time
import pytest
import torch

from radfield3dnn.models.encoders.spherical_hamonics import (
    TorchSphericalHarmonics, TcnnSphericalHarmonics,
)

try:
    import tinycudann as _tcnn  # noqa: F401
    _HAVE_TCNN = torch.cuda.is_available()
except ImportError:
    _HAVE_TCNN = False

_skip = pytest.mark.skipif(not _HAVE_TCNN, reason="needs CUDA + tiny-cuda-nn")


@_skip
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_matches_tcnn_fp32(degree):
    """Exact-math parity: compare against an fp32 tcnn encoding (isolates the formula from fp16)."""
    torch.manual_seed(0)
    dirs = torch.randn(4096, 3, device="cuda")
    py = TorchSphericalHarmonics(degree=degree)
    tc = TcnnSphericalHarmonics(degree=degree, dtype=torch.float32)
    assert py.encoded_dims == tc.encoded_dims == degree * degree
    with torch.no_grad():
        max_diff = (py(dirs).float() - tc(dirs).float()).abs().max().item()
    assert max_diff < 1e-4, f"degree={degree} max diff {max_diff}"


@_skip
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_matches_tcnn_fp16(degree):
    """Default tcnn output is fp16; the torch impl is within fp16 storage precision of it."""
    torch.manual_seed(0)
    dirs = torch.randn(4096, 3, device="cuda")
    with torch.no_grad():
        max_diff = (TorchSphericalHarmonics(degree=degree)(dirs).float()
                    - TcnnSphericalHarmonics(degree=degree)(dirs).float()).abs().max().item()
    assert max_diff < 2e-2, f"degree={degree} max diff {max_diff}"


def _bench(module, dirs, iters=100):
    with torch.no_grad():
        for _ in range(10):           # warmup
            module(dirs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            module(dirs)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def main():
    if not _HAVE_TCNN:
        print("CUDA + tiny-cuda-nn unavailable — skipping.")
        return
    dirs = torch.randn(1_000_000, 3, device="cuda")
    for degree in (4,):
        py = TorchSphericalHarmonics(degree=degree, append_input=True)
        tc = TcnnSphericalHarmonics(degree=degree, append_input=True)
        with torch.no_grad():
            md = (py(dirs).float() - tc(dirs).float()).abs().max().item()
        t_py = _bench(py, dirs)
        t_tc = _bench(tc, dirs)
        print(f"degree={degree}  N=1e6  max|Δ|={md:.2e}  "
              f"torch={t_py:.3f} ms  tcnn={t_tc:.3f} ms  (torch/tcnn={t_py/t_tc:.2f}x)")


if __name__ == "__main__":
    main()

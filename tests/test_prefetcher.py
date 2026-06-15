"""Parity tests for CudaStreamPrefetcher.

Confirms the side-stream prefetcher (datasets/prefetcher.py) is behaviour-neutral: the sequence of
GPU-resident, preprocessed batches it yields is identical — value for value — to the synchronous
move-to-device + run-processings path it replaces. Skips when no CUDA device is present.
"""
import collections
import pytest
import torch
from torch import nn

from radfield3dnn.datasets.prefetcher import CudaStreamPrefetcher

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="prefetcher needs a CUDA device")

# A small nested namedtuple so we also exercise apply_to_collection's structure preservation.
Sample = collections.namedtuple("Sample", ["x", "y"])


class AddBias(nn.Module):
    """A deterministic, model-independent batch transform with a device-resident buffer (so .to()
    actually has to move state, like the real normalizers/augmentations)."""
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("bias", torch.full((4,), float(scale)))

    def forward(self, batch: Sample) -> Sample:
        return Sample(x=batch.x * 2.0 + self.bias, y=batch.y + self.bias)


def _make_batches(n=6, b=3):
    g = torch.Generator().manual_seed(0)
    return [Sample(x=torch.randn(b, 4, generator=g), y=torch.randn(b, 4, generator=g)) for _ in range(n)]


def _reference(batches, device, scales):
    procs = [AddBias(s).to(device) for s in scales]
    out = []
    for raw in batches:
        batch = Sample(*(t.to(device) for t in raw))
        for p in procs:
            batch = p(batch)
        out.append(Sample(*(t.clone() for t in batch)))
    return out


@cuda
def test_prefetcher_matches_synchronous_path():
    device = torch.device("cuda")
    batches = _make_batches()
    scales = [0.5, -1.0]

    expected = _reference(batches, device, scales)

    procs = [AddBias(s) for s in scales]  # prefetcher uploads these to device itself
    pf = CudaStreamPrefetcher(list(batches), device, procs)
    got = [Sample(*(t.clone() for t in batch)) for batch in pf]

    torch.cuda.synchronize()
    assert len(got) == len(expected)
    for i, (a, b) in enumerate(zip(got, expected)):
        assert a.x.device.type == "cuda" and a.y.device.type == "cuda"
        assert torch.equal(a.x, b.x), f"x mismatch at batch {i}"
        assert torch.equal(a.y, b.y), f"y mismatch at batch {i}"


@cuda
def test_prefetcher_reiterable_and_length():
    device = torch.device("cuda")
    batches = _make_batches(n=4)
    pf = CudaStreamPrefetcher(list(batches), device, [AddBias(1.0)])
    assert len(pf) == 4
    first = [b.x.clone() for b in pf]
    second = [b.x.clone() for b in pf]  # second pass must reproduce the first
    torch.cuda.synchronize()
    assert len(first) == len(second) == 4
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@cuda
def test_prefetcher_empty_processings_is_pure_upload():
    device = torch.device("cuda")
    batches = _make_batches(n=3)
    pf = CudaStreamPrefetcher(list(batches), device, [])
    got = list(pf)
    torch.cuda.synchronize()
    for raw, batch in zip(batches, got):
        assert torch.equal(batch.x, raw.x.to(device))
        assert torch.equal(batch.y, raw.y.to(device))

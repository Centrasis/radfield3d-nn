"""Tests for AsinhTonemapNormalizer.

Validates:
* Endpoint round-trip (0 → 0, 1 → 1 exact).
* Smoothness: no sentinel discontinuity.
* fp32 round-trip accuracy on a log-spaced probe.
"""

import math
import torch
import pytest

from radfield3dnn.preprocessing.normalizations import (
    NormalizerConstructor,
    AsinhTonemapNormalizer,
)


def _probe(n: int = 4000) -> torch.Tensor:
    # Log-spaced over the per-field-normalised input range plus the endpoints.
    e = torch.linspace(-6.0, 0.0, n, dtype=torch.float64)
    x = torch.pow(10.0, e).to(torch.float32)
    extra = torch.tensor([0.0, 1.0], dtype=torch.float32)
    return torch.cat([extra, x])


def test_constructor_by_name():
    n = NormalizerConstructor.construct_by_name("asinh")
    assert isinstance(n, AsinhTonemapNormalizer)
    assert n.sigma == pytest.approx(3e-3)


def test_endpoints_exact():
    n = AsinhTonemapNormalizer(sigma=1e-3)
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y = n.apply_transformation(x, None)
    assert y[0].item() == pytest.approx(0.0, abs=1e-7), "zero must map to zero (no sentinel)"
    assert y[1].item() == pytest.approx(1.0, abs=1e-6), "one must map to one"


def test_codomain_bounded():
    """Every nonneg input maps into [0, 1]."""
    n = AsinhTonemapNormalizer(sigma=3e-3)
    x = _probe()
    y = n.apply_transformation(x, None)
    assert (y >= 0.0).all() and (y <= 1.0 + 1e-6).all()


def test_monotonic():
    n = AsinhTonemapNormalizer(sigma=3e-3)
    x = torch.linspace(0.0, 1.0, 2048, dtype=torch.float32)
    y = n.apply_transformation(x, None)
    diffs = y[1:] - y[:-1]
    assert (diffs >= -1e-7).all(), "tonemap must be monotonic non-decreasing"


def test_roundtrip_fp32():
    n = AsinhTonemapNormalizer(sigma=3e-3)
    x = _probe()
    rt = n.apply_inverse_transformation(n.apply_transformation(x, None), None)
    # Endpoints are exact by construction.
    assert rt[0].item() == 0.0
    assert rt[1].item() == pytest.approx(1.0, abs=1e-6)
    # Non-endpoints: relative error small in fp32 over 6 decades.
    nz = x > 1e-7
    rel = (rt[nz] - x[nz]).abs() / x[nz].abs()
    assert rel.max().item() < 1e-3, f"fp32 round-trip rel err {rel.max().item()}"


def test_no_sentinel_discontinuity():
    """asinh is smooth at zero — there is no discontinuity to learn around."""
    n = AsinhTonemapNormalizer(sigma=1e-3)
    x = torch.tensor([0.0, 1e-9, 1e-7, 1e-5], dtype=torch.float32)
    y = n.apply_transformation(x, None)
    # All four should be tiny (≤ ~0.05) and strictly ordered.
    assert (y[1:] >= y[:-1]).all()
    assert y.max().item() < 0.05


def test_per_element_error_budget_bounded():
    """In tonemap space the max absolute residual between any pair of inputs in [0, 1]
    is ≤ 1 (the L1 budget that makes the L1+SSIM loss fp16-safe)."""
    n = AsinhTonemapNormalizer(sigma=3e-3)
    x_max = n.apply_transformation(torch.tensor([1.0]), None).item()
    x_min = n.apply_transformation(torch.tensor([0.0]), None).item()
    assert (x_max - x_min) <= 1.0 + 1e-6

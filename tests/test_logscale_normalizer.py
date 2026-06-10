"""Tests for LogScaleNormalizer: real log-domain output with a true-zero
sentinel below the log10(x_min) floor.

Companion to test_logdecade_normalizer.py. The key claim this normalizer
makes that log_decade does NOT is that *true zero round-trips to true
zero*: log_decade maps 0 → range[0] → 1e-8 (not 0), which loses occluded
voxels. log_scale reserves a sentinel value (default -9) below the log
range and inverts anything below the half-way threshold back to 0.
"""

import math
import torch
import pytest

from radfield3dnn.preprocessing.normalizations import (
    NormalizerConstructor,
    LogScaleNormalizer,
)


def _probe(n: int = 4000) -> torch.Tensor:
    """Log-spaced probe over 1e-8 .. 1.0 plus the exact endpoints."""
    e = torch.linspace(-8.0, 0.0, n, dtype=torch.float64)
    x = torch.pow(10.0, e).to(torch.float32)
    extra = torch.tensor([1e-8, 1e-7, 1e-6, 1.0], dtype=torch.float32)
    return torch.cat([x, extra])


def test_constructor_by_name():
    n = NormalizerConstructor.construct_by_name("log_scale")
    assert isinstance(n, LogScaleNormalizer)
    assert n.x_min == 1e-8 and n.x_max == 1.0
    assert n.zero_floor == -9.0
    assert n.get_type() == "log_scale"


def test_endpoints_and_zero_sentinel():
    n = LogScaleNormalizer()
    x = torch.tensor([0.0, 1e-8, 1e-4, 1.0], dtype=torch.float32)
    y = n.apply_transformation(x, None)
    assert y[0].item() == pytest.approx(-9.0, abs=1e-6), "zero must map to zero_floor"
    assert y[1].item() == pytest.approx(-8.0, abs=1e-4)
    assert y[2].item() == pytest.approx(-4.0, abs=1e-4)
    assert y[3].item() == pytest.approx(0.0, abs=1e-6)


def test_zero_roundtrip_is_exact():
    n = LogScaleNormalizer()
    x = torch.tensor([0.0], dtype=torch.float32)
    rt = n.apply_inverse_transformation(n.apply_transformation(x, None), None)
    assert rt.item() == 0.0, "true zero must round-trip bitwise to zero"


def test_positive_roundtrip_in_fp32():
    n = LogScaleNormalizer()
    x = _probe()
    rt = n.apply_inverse_transformation(n.apply_transformation(x, None), None)
    rel = (rt - x).abs() / x.abs()
    assert rel.max().item() < 1e-4, f"fp32 round-trip not exact: max rel {rel.max().item()}"


def test_mixed_zero_and_positive_roundtrip():
    n = LogScaleNormalizer()
    x = torch.tensor([0.0, 1e-8, 0.0, 3e-5, 1.0, 0.0], dtype=torch.float32)
    rt = n.apply_inverse_transformation(n.apply_transformation(x, None), None)
    assert (rt[x == 0.0] == 0.0).all(), "zeros must be preserved bitwise"
    nz = x != 0.0
    rel = (rt[nz] - x[nz]).abs() / x[nz].abs()
    assert rel.max().item() < 1e-4


def test_inverse_threshold_behavior():
    """Anything below the midpoint between zero_floor and log10(x_min) (-8.5
    by default) inverts to 0; anything at or above log10(x_min) inverts to
    a positive value."""
    n = LogScaleNormalizer()
    assert n.inverse_zero_threshold == pytest.approx(-8.5, abs=1e-12)
    y = torch.tensor([-9.0, -8.6, -8.0, 0.0], dtype=torch.float32)
    x = n.apply_inverse_transformation(y, None)
    assert x[0].item() == 0.0
    assert x[1].item() == 0.0
    assert x[2].item() == pytest.approx(1e-8, rel=1e-4)
    assert x[3].item() == pytest.approx(1.0, rel=1e-4)


def test_monotonic_on_positive_inputs():
    n = LogScaleNormalizer()
    x = torch.logspace(-8, 0, 256).to(torch.float32)
    y = n.apply_transformation(x, None)
    diffs = y[1:] - y[:-1]
    assert (diffs > 0).all(), "forward must be strictly monotonic on positives"


def test_negative_input_raises():
    n = LogScaleNormalizer()
    with pytest.raises(ValueError):
        n.apply_transformation(torch.tensor([-1.0], dtype=torch.float32), None)


def test_clone_preserves_parameters():
    n = LogScaleNormalizer(x_min=1e-6, x_max=10.0, zero_floor=-7.5)
    c = n.clone()
    assert isinstance(c, LogScaleNormalizer)
    assert c.x_min == 1e-6 and c.x_max == 10.0 and c.zero_floor == -7.5
    assert c is not n


def test_clamps_input_above_x_max():
    n = LogScaleNormalizer()
    x = torch.tensor([10.0], dtype=torch.float32)  # above x_max=1.0
    y = n.apply_transformation(x, None)
    assert y.item() == pytest.approx(0.0, abs=1e-6), "above-x_max clamps to log10(x_max)=0"
    rt = n.apply_inverse_transformation(y, None)
    assert rt.item() == pytest.approx(1.0, abs=1e-4)


def test_clamps_input_below_x_min_but_nonzero():
    """Values in (0, x_min) clamp up to x_min — they are NOT routed through
    the zero sentinel. Only an exact 0 triggers the sentinel."""
    n = LogScaleNormalizer()
    x = torch.tensor([1e-12], dtype=torch.float32)  # below x_min but >0
    y = n.apply_transformation(x, None)
    assert y.item() == pytest.approx(-8.0, abs=1e-4), "(0, x_min) clamps to log10(x_min)"

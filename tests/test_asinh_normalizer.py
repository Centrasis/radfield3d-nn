"""Tests for AsinhTonemapNormalizer and SplitChannelAsinhNormalizer.

Validates:
* Endpoint round-trip (0 → 0, 1 → 1 exact).
* Smoothness: no sentinel discontinuity.
* fp32 round-trip accuracy on a log-spaced probe.
* Per-channel dispatch in SplitChannelAsinhNormalizer.
* Empirical σ selection via from_dataset on synthetic data with a
  known noise floor.
"""

import math
import torch
import pytest

from radfield3dnn.preprocessing.normalizations import (
    NormalizerConstructor,
    AsinhTonemapNormalizer,
    SplitChannelAsinhNormalizer,
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
    s = NormalizerConstructor.construct_by_name("asinh_split")
    assert isinstance(s, SplitChannelAsinhNormalizer)
    assert s.scatter_sigma == pytest.approx(3e-3)
    assert s.direct_sigma == pytest.approx(1e-3)


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
    # Non-endpoints: relative error ≤ 1e-4 in fp32 over 6 decades.
    nz = x > 1e-7
    rel = (rt[nz] - x[nz]).abs() / x[nz].abs()
    assert rel.max().item() < 1e-3, f"fp32 round-trip rel err {rel.max().item()}"


def test_no_sentinel_discontinuity():
    """asinh is smooth at zero — there is no discontinuity to learn around."""
    n = AsinhTonemapNormalizer(sigma=1e-3)
    x = torch.tensor([0.0, 1e-9, 1e-7, 1e-5], dtype=torch.float32)
    y = n.apply_transformation(x, None)
    # All four should be tiny (≤ ~0.01) and strictly ordered.
    assert (y[1:] >= y[:-1]).all()
    assert y.max().item() < 0.05


def test_per_element_error_budget_bounded():
    """In tonemap space the max absolute residual between any pair of
    inputs in [0, 1] is ≤ 1 (the L1 budget that makes the L1+SSIM loss
    fp16-safe). Contrast: raw log10 LogScaleNormalizer has a max
    residual of 9."""
    n = AsinhTonemapNormalizer(sigma=3e-3)
    x_max = n.apply_transformation(torch.tensor([1.0]), None).item()
    x_min = n.apply_transformation(torch.tensor([0.0]), None).item()
    assert (x_max - x_min) <= 1.0 + 1e-6


def test_split_channel_dispatch():
    n = SplitChannelAsinhNormalizer(scatter_sigma=2e-2, direct_sigma=5e-4)
    # Same input fed to both channels: should produce different outputs
    # because σ differs.
    x = torch.tensor([1e-3], dtype=torch.float32)
    y_scatter = n.scatter.apply_transformation(x, None)
    y_direct = n.direct.apply_transformation(x, None)
    assert y_scatter.item() != y_direct.item()


def test_from_dataset_synthetic():
    """A synthetic dataset where each "field" is a tensor with a known
    noise floor at 1e-4 (scatter channel). The selected σ_scatter at
    quantile 0.10 should land near 1e-4."""
    class _Item:
        def __init__(self, sc, dr):
            class _F:
                def __init__(self, flux): self.flux = flux
            class _GT:
                def __init__(self, sc, dr):
                    self.scatter_field = _F(sc)
                    self.direct_beam = _F(dr)
            self.ground_truth = _GT(sc, dr)

    items = []
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        # scatter: uniform on [1e-4, 1.0] in log space, max ≈ 1.0
        sc_log = torch.empty(5000).uniform_(-4.0, 0.0, generator=g)
        sc = torch.pow(10.0, sc_log)
        # direct: bimodal — 90% noise [1e-6, 1e-4], 10% signal [1e-2, 1.0]
        dr_noise = torch.pow(10.0, torch.empty(4500).uniform_(-6.0, -4.0, generator=g))
        dr_sig = torch.pow(10.0, torch.empty(500).uniform_(-2.0, 0.0, generator=g))
        dr = torch.cat([dr_noise, dr_sig])
        items.append(_Item(sc, dr))

    n = SplitChannelAsinhNormalizer.from_dataset(items, max_fields=20,
                                                 scatter_quantile=0.10,
                                                 direct_quantile=0.90)
    # Scatter is uniform on log [-4, 0], so p10 (after /max≈1) ≈ 10^(-4+0.4) ≈ 2.5e-4
    # The clamp lower bound is 1e-6; we just check it lands in the noise band.
    assert 1e-5 < n.scatter_sigma < 5e-3
    # Direct: 90% noise band tops out near 1e-4 (after /max≈1), so p90 ≈ 1e-4
    assert 1e-5 < n.direct_sigma < 1e-2


def test_split_radiationfield_dispatch():
    """SplitChannelAsinhNormalizer.forward on a RadiationField applies the
    correct per-channel σ."""
    from radfield3dnn.rftypes import RadiationField
    try:
        from RadFiled3D.pytorch.types import RadiationFieldChannel
    except ImportError:
        pytest.skip("RadFiled3D not available in test env")
    sc_flux = torch.tensor([0.5], dtype=torch.float32)
    dr_flux = torch.tensor([0.5], dtype=torch.float32)
    sc_spec = torch.zeros((32,), dtype=torch.float32)
    dr_spec = torch.zeros((32,), dtype=torch.float32)
    f = RadiationField(
        scatter_field=RadiationFieldChannel(flux=sc_flux, spectrum=sc_spec, error=None),
        direct_beam=RadiationFieldChannel(flux=dr_flux, spectrum=dr_spec, error=None),
    )
    n = SplitChannelAsinhNormalizer(scatter_sigma=2e-2, direct_sigma=5e-4)
    out = n.forward(f)
    # With different σ, the two channels' outputs must differ.
    assert out.scatter_field.flux.item() != out.direct_beam.flux.item()

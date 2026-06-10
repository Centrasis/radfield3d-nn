"""Tests for ChannelsSplitRelative.

Covers:
* Per-volume max normalisation: outputs in [0, 1] with the maxes equal to 1.
* Joined-spectrum invariant: scatter slot carries the flux-weighted joined
  spectrum, direct slot carries a zero placeholder.
* Scaling metadata carried on the geometry slot.
* Per-field scatter_max / direct_max ratio recovery.
* Batched inputs (B, D, H, W) per-volume independence.
* Empty-volume safety (max == 0 doesn't divide by zero).
"""
import torch
import pytest

from radfield3dnn.datasets.channel_split_relative import ChannelsSplitRelative
from radfield3dnn.rftypes import RadiationField


# Re-export the RadiationFieldChannel from the RF3D namespace so the tests
# build the same object the real pipeline uses.
try:
    from RadFiled3D.pytorch.types import RadiationFieldChannel
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)


def _mk_field(sc_flux: torch.Tensor, dr_flux: torch.Tensor,
              spec_bins: int = 8) -> RadiationField:
    sc_spec = torch.zeros((spec_bins,) + sc_flux.shape, dtype=torch.float32)
    sc_spec[0] = 1.0   # peak in bin 0
    dr_spec = torch.zeros((spec_bins,) + dr_flux.shape, dtype=torch.float32)
    dr_spec[-1] = 1.0  # peak in last bin
    return RadiationField(
        scatter_field=RadiationFieldChannel(flux=sc_flux, spectrum=sc_spec, error=None),
        direct_beam=RadiationFieldChannel(flux=dr_flux, spectrum=dr_spec, error=None),
    )


def test_per_volume_max_normalisation():
    sc = torch.tensor([[[0.0, 0.5], [1.0, 2.0]]], dtype=torch.float32)  # max 2.0
    dr = torch.tensor([[[0.0, 0.0], [0.1, 0.4]]], dtype=torch.float32)  # max 0.4
    f = _mk_field(sc, dr)
    op = ChannelsSplitRelative()
    out = op.forward(f)
    assert out.scatter_field.flux.max().item() == pytest.approx(1.0)
    assert out.direct_beam.flux.max().item() == pytest.approx(1.0)
    # Min should still be 0.
    assert out.scatter_field.flux.min().item() == 0.0
    assert out.direct_beam.flux.min().item() == 0.0


def test_codomain_bounded():
    sc = torch.rand((3, 4, 5)) * 3.0
    dr = torch.rand((3, 4, 5)) * 0.7
    out = ChannelsSplitRelative().forward(_mk_field(sc, dr))
    assert (out.scatter_field.flux >= 0).all() and (out.scatter_field.flux <= 1).all()
    assert (out.direct_beam.flux >= 0).all() and (out.direct_beam.flux <= 1).all()


def test_scaling_metadata_carried():
    sc = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    dr = torch.tensor([[[0.3, 0.6]]], dtype=torch.float32)
    out = ChannelsSplitRelative().forward(_mk_field(sc, dr))
    scaling = ChannelsSplitRelative.extract_scaling(out)
    assert scaling is not None
    # Layout [scatter_max, direct_max].
    assert scaling[0].item() == pytest.approx(2.0)
    assert scaling[1].item() == pytest.approx(0.6)
    ratio = ChannelsSplitRelative.compute_max_ratio(scaling)
    assert ratio.item() == pytest.approx(2.0 / 0.6)


def test_joined_spectrum_on_scatter_slot():
    sc = torch.tensor([[[1.0]]], dtype=torch.float32)
    dr = torch.tensor([[[3.0]]], dtype=torch.float32)
    out = ChannelsSplitRelative().forward(_mk_field(sc, dr, spec_bins=4))
    # joined spectrum is on the scatter slot; direct slot is zero placeholder.
    assert out.scatter_field.spectrum.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert out.direct_beam.spectrum.sum().item() == 0.0
    # Joined favours direct (flux 3 vs 1) → last-bin > bin 0.
    sc_only_bin = out.scatter_field.spectrum[0, 0, 0, 0].item()
    dr_only_bin = out.scatter_field.spectrum[-1, 0, 0, 0].item()
    assert dr_only_bin > sc_only_bin


def test_batched_per_volume_independence():
    """Each volume in a batch normalises by its own max — not the
    batch-global max."""
    sc = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
    sc[0, 0, 0, 0] = 0.5   # volume 0 max = 0.5
    sc[1, 0, 0, 0] = 5.0   # volume 1 max = 5.0
    dr = torch.full((2, 1, 1, 1), 1.0, dtype=torch.float32)
    spec = torch.zeros((2, 4, 1, 1, 1), dtype=torch.float32)
    spec[:, 0] = 1.0
    f = RadiationField(
        scatter_field=RadiationFieldChannel(flux=sc, spectrum=spec, error=None),
        direct_beam=RadiationFieldChannel(flux=dr, spectrum=spec.clone(), error=None),
    )
    out = ChannelsSplitRelative().forward(f)
    # Both volumes' scatter normalise to 1.0 regardless of absolute max.
    assert out.scatter_field.flux[0].item() == pytest.approx(1.0)
    assert out.scatter_field.flux[1].item() == pytest.approx(1.0)


def test_empty_volume_safe():
    sc = torch.zeros((2, 2, 2), dtype=torch.float32)
    dr = torch.zeros((2, 2, 2), dtype=torch.float32)
    out = ChannelsSplitRelative().forward(_mk_field(sc, dr))
    assert torch.isfinite(out.scatter_field.flux).all()
    assert torch.isfinite(out.direct_beam.flux).all()
    scaling = ChannelsSplitRelative.extract_scaling(out)
    assert scaling[0].item() == 0.0 and scaling[1].item() == 0.0

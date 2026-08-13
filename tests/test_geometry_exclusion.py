"""GeometryVoxelExclusion: geometry-covered voxels leave the training target as -inf.

The simulation excludes phantom-covered voxels from scoring, so their GT is not a statistic; the
processing drops them with the repo's -inf sentinel, composing with the ROI samplers.
"""
import pytest
import torch

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from RadFiled3D.pytorch.types import RadiationFieldChannel, TranslationalInput
from radfield3dnn.preprocessing.augmentations.geometry_exclusion import GeometryVoxelExclusion
from radfield3dnn.rftypes import RadiationField, TrainingInputData

VC = (4, 4, 4)
BINS = 8


def _batch(geometry, B=2):
    g = torch.Generator().manual_seed(0)
    inp = TranslationalInput(
        direction=torch.randn(B, 3, generator=g),
        origin=torch.rand(B, 1, generator=g),
        spectrum=torch.rand(B, 32, generator=g),
        translation=torch.rand(B, 3, generator=g),
        geometry=geometry,
    )
    gt = RadiationField(
        scatter_field=RadiationFieldChannel(flux=torch.rand(B, *VC, generator=g),
                                            spectrum=torch.rand(B, BINS, *VC, generator=g), error=None),
        direct_beam=RadiationFieldChannel(flux=torch.rand(B, *VC, generator=g), spectrum=None, error=None),
        geometry=None)
    return TrainingInputData(input=inp, ground_truth=gt, original_ground_truth=gt)


def _mask(B=2):
    m = torch.zeros(B, 1, *VC)
    m[:, 0, 1, 2, 3] = 1.0
    m[0, 0, 0, 0, 0] = 1.0
    return m


def test_covered_voxels_dropped_in_all_channels():
    excl = GeometryVoxelExclusion().train(True)
    out = excl(_batch(_mask()))
    flux = out.ground_truth.scatter_field.flux
    assert torch.isneginf(flux[0, 0, 0, 0]) and torch.isneginf(flux[:, 1, 2, 3]).all()
    assert not torch.isneginf(flux[1, 0, 0, 0])                      # only sample 0 covered there
    assert torch.isneginf(out.ground_truth.scatter_field.spectrum[:, :, 1, 2, 3]).all()
    assert torch.isneginf(out.ground_truth.direct_beam.flux[:, 1, 2, 3]).all()
    # exactly the covered voxels, nothing else
    assert int(torch.isneginf(flux).sum()) == 3
    # the input (incl. the geometry channel itself) is untouched
    assert torch.equal(out.input.geometry, _mask())


def test_binary_and_density_geometry_both_work():
    dense = _mask() * 1.85  # raw density values instead of a binary mask
    out = GeometryVoxelExclusion(threshold=0.0).train(True)(_batch(dense))
    assert torch.isneginf(out.ground_truth.scatter_field.flux[:, 1, 2, 3]).all()


def test_geometry_without_channel_dim_works():
    out = GeometryVoxelExclusion().train(True)(_batch(_mask().squeeze(1)))
    assert torch.isneginf(out.ground_truth.scatter_field.flux[:, 1, 2, 3]).all()


def test_noop_cases():
    excl = GeometryVoxelExclusion()
    x = _batch(None)
    assert excl.train(True)(x) is x                       # no geometry channel
    x = _batch(_mask())
    assert excl.train(False)(x) is x                      # eval: whole field is scored
    mismatched = _batch(torch.ones(2, 1, 8, 8, 8))        # wrong grid -> refuse to mask
    out = excl.train(True)(mismatched)
    assert not torch.isneginf(out.ground_truth.scatter_field.flux).any()


def test_floor_injected_geometry_voxel_stays_dropped():
    # Order contract with the ROI sampler: run AFTER floor_as_zero injection, so a geometry voxel
    # the sampler re-injected as "zero dose" ends -inf, not confidently 0.
    from radfield3dnn.preprocessing.augmentations.roi_sampling import ROIbasedSampler
    x = _batch(torch.ones(2, 1, *VC))                     # EVERY voxel covered (extreme case)
    sampler = ROIbasedSampler(floor_as_zero=True).train(True)
    excl = GeometryVoxelExclusion().train(True)
    out = excl(sampler(x))
    assert torch.isneginf(out.ground_truth.scatter_field.flux).all()


def test_real_layerwise_shapes_with_channel_dims():
    # EXACTLY the shapes from the prefetcher crash: geometry (B, 1, 64^3), flux (B, 1, 64^3),
    # spectrum (B, 32, 64^3) — the flux-shaped mask must NOT be unsqueezed onto the spectrum
    # (that produced a 6-dim expand RuntimeError); each tensor gets its own broadcast.
    B, N, BINS_ = 4, 6, 32   # 6^3 instead of 64^3: same rank/semantics, fast
    g = torch.Generator().manual_seed(0)
    geometry = torch.zeros(B, 1, N, N, N)
    geometry[:, 0, 1, 2, 3] = 1.0
    inp = TranslationalInput(direction=torch.randn(B, 3, generator=g), origin=torch.rand(B, 1, generator=g),
                             spectrum=torch.rand(B, 32, generator=g), translation=torch.rand(B, 3, generator=g),
                             geometry=geometry)
    gt = RadiationField(
        scatter_field=RadiationFieldChannel(flux=torch.rand(B, 1, N, N, N, generator=g),
                                            spectrum=torch.rand(B, BINS_, N, N, N, generator=g), error=None),
        direct_beam=None, geometry=None)
    x = TrainingInputData(input=inp, ground_truth=gt, original_ground_truth=gt)
    out = GeometryVoxelExclusion().train(True)(x)
    flux, spec = out.ground_truth.scatter_field.flux, out.ground_truth.scatter_field.spectrum
    assert torch.isneginf(flux[:, 0, 1, 2, 3]).all() and int(torch.isneginf(flux).sum()) == B
    assert torch.isneginf(spec[:, :, 1, 2, 3]).all() and int(torch.isneginf(spec).sum()) == B * BINS_


# (dataset composition is covered in tests/test_input_decorators.py)
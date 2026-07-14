"""voxel_supersampling (K in-voxel samples, per-voxel mean supervision) + the IPE encoder.

CPU-only. Covers: K-mean assembly mechanics (constant + linear fields), eval-path invariance,
importance-mask compatibility (per-voxel supervision needs NO neighbor voxels), gradient flow,
the removed-jitter-flag rejection, and the IntegratedSinusoidalEncoding damping math.
"""
import math
import pytest
import torch
import torch.nn.functional as F

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
from radfield3dnn.rftypes import DirectionalInput


def _kwargs(k):
    return dict(
        d_model=32,
        location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 4, "append_input": True},
        direction_encoding_params={"type": "spherical_harmonics", "degree": 2, "append_input": True},
        spectra_encoding_params={"type": "simple", "in_spectra_dim": 8, "encoded_spectra_dims": 8},
        out_spectra_dim=8, conditioning_params={"type": "Concat", "use_beam_shape": False},
        normalizer=LinearNormalizer((0.0, 1.0)), flux_loss="SMAPEBalanced",
        flux_activation="sigmoid", trunk_depth=2,
        training_params={"voxel_supersampling": k, "voxels_centered_around_origin": False},
    )


def _model(k):
    torch.manual_seed(0)
    m = PBRFNet(**_kwargs(k))
    m.max_inner_batch_size = 512
    return m


def _input(B=1):
    g = torch.Generator().manual_seed(3)
    return DirectionalInput(
        direction=F.normalize(torch.randn(B, 3, generator=g), dim=-1),
        origin=torch.rand(B, 1, generator=g) * 0.4 + 0.35,
        spectrum=torch.ones(B, 8) / 8,
    )


def _patch_positional_flux(model, fn):
    """Make the core model output flux = fn(position) deterministically (spectrum untouched)."""
    core = model.get_core_model()
    orig = core.forward

    def patched(batch, global_parameters=None, region_state=None):
        out = orig(batch, global_parameters=global_parameters, region_state=region_state)
        flux = fn(batch.position)
        return out._replace(scatter_field=out.scatter_field._replace(flux=flux))
    core.forward = patched
    return model


def test_k_mean_of_constant_field_is_exact():
    # position-independent output -> the K-mean must equal the constant exactly
    m = _patch_positional_flux(_model(4), lambda p: torch.full((p.shape[0],), 0.25))
    m.train()
    out = m.forward2volume(_input(), torch.tensor([4, 4, 4]), spectra_bins=8)
    assert torch.allclose(out.scatter_field.flux, torch.tensor(0.25), atol=1e-6)


def test_k_mean_of_linear_field_matches_cell_mean():
    # flux = x-coordinate: E[flux over cell i] = lower_corner_i + w/2. With K samples the MC mean
    # must be an unbiased, low-variance estimate of that cell mean (tolerance ~ w/sqrt(12K)).
    N, K = 4, 64
    torch.manual_seed(7)
    m = _patch_positional_flux(_model(K), lambda p: p[:, 0])
    m.train()
    out = m.forward2volume(_input(), torch.tensor([N, N, N]), spectra_bins=8)
    flux = out.scatter_field.flux.reshape(N, N, N)
    w = 1.0 / N
    expected = (torch.arange(N) * w + w / 2)
    got = flux.mean(dim=(1, 2))  # x-profile
    assert torch.allclose(got, expected, atol=3 * w / math.sqrt(12 * K))


def test_eval_path_ignores_k():
    # in eval() the volume must be the single node-grid sample: identical for K=1 and K=8
    torch.manual_seed(0); m1 = _model(1)
    torch.manual_seed(0); m8 = _model(8)
    m1.eval(); m8.eval()
    x = _input()
    with torch.no_grad():
        a = m1.forward2volume(x, torch.tensor([4, 4, 4]), spectra_bins=8)
        b = m8.forward2volume(x, torch.tensor([4, 4, 4]), spectra_bins=8)
    assert torch.equal(a.scatter_field.flux, b.scatter_field.flux)


def test_mask_compatibility_no_neighbors_needed():
    # per-voxel supervision: dropped voxels -> -inf sentinel; kept voxels averaged from exactly K
    # in-voxel samples. Neighboring voxels are NOT required (unlike trilinear-target blending).
    N = 4
    m = _patch_positional_flux(_model(4), lambda p: torch.full((p.shape[0],), 0.5))
    m.train()
    mask = torch.zeros(1, N, N, N, dtype=torch.bool)
    mask[0, 0] = True  # drop the whole x=0 slab
    out = m.forward2volume(_input(), torch.tensor([N, N, N]), spectra_bins=8, mask=mask)
    flux = out.scatter_field.flux.reshape(N, N, N)
    assert torch.isneginf(flux[0]).all()
    assert torch.allclose(flux[1:], torch.tensor(0.5), atol=1e-6)


def test_gradients_flow_through_k_mean():
    m = _model(4)
    m.train()
    out = m.forward2volume(_input(), torch.tensor([3, 3, 3]), spectra_bins=8)
    out.scatter_field.flux.sum().backward()
    grads = [p.grad for p in m.get_core_model().parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_removed_jitter_flag_rejected():
    kw = _kwargs(1)
    kw["training_params"] = {"randomize_voxel_location_in_training": True}
    with pytest.raises(ValueError, match="voxel_supersampling"):
        PBRFNet(**kw)


# ---------------------------------------------------------------- IPE encoder ----

def test_ipe_registered_and_point_query_matches_sinusoidal():
    from radfield3dnn.models.encoders.factory import build_encoding
    sin = build_encoding({"type": "sinusoidal", "pos_enc_dim": 6, "append_input": True})
    ipe = build_encoding({"type": "ipe", "pos_enc_dim": 6, "append_input": True})
    assert ipe.encoded_dims == sin.encoded_dims
    x = torch.rand(17, 3) * 2 - 1
    assert torch.allclose(ipe(x), sin(x), atol=1e-6)  # region_width=None -> exact point encoding


def test_ipe_damping_matches_numerical_box_average():
    # E[enc(x)] over a uniform box == damped enc(mu), per channel (exact, sinc form)
    from radfield3dnn.models.encoders.ipe_encoding import IntegratedSinusoidalEncoding
    w = 1.0 / 16
    enc = IntegratedSinusoidalEncoding(pos_enc_dim=5, d_input=3, append_input=True)
    enc.region_width = w
    mu = torch.tensor([[0.3, -0.2, 0.55]])
    analytic = enc(mu)
    g = (torch.arange(48) + 0.5) / 48 - 0.5  # midpoint quadrature per axis
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    offsets = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3) * w
    enc.region_width = None
    numeric = enc(mu + offsets).mean(dim=0, keepdim=True)
    assert torch.allclose(analytic, numeric, atol=1e-4)


def test_ipe_append_input_undamped_and_high_freqs_killed():
    from radfield3dnn.models.encoders.ipe_encoding import IntegratedSinusoidalEncoding
    enc = IntegratedSinusoidalEncoding(pos_enc_dim=8, d_input=3, append_input=True)
    enc.region_width = 2.0 / 64  # one voxel in the [-1,1] frame
    x = torch.rand(4, 3)
    out = enc(x)
    assert torch.allclose(out[:, -3:], x)  # appended mean undamped
    # level 6 damping = sinc(2^6*pi/64) = sinc(pi) = 0 exactly
    damp = enc.compute_region_state(torch.tensor(2.0 / 64))
    assert abs(float(damp[6])) < 1e-6
    assert float(damp[0]) > 0.999


# ------------------------------------------------- multi-resolution resampling ----

def _mr_batch(N=8, bins=4):
    g = torch.Generator().manual_seed(5)
    flux = torch.rand(1, 1, N, N, N, generator=g)
    spec = torch.rand(1, bins, N, N, N, generator=g)
    spec = spec / spec.sum(dim=1, keepdim=True)
    from radfield3dnn.rftypes import RadiationFieldChannel, TrainingInputData, DirectionalInput
    gt = RadiationFieldChannel(spectrum=spec, flux=flux, error=torch.rand(1, 1, N, N, N, generator=g))
    inp = DirectionalInput(direction=torch.randn(1, 3, generator=g), origin=torch.rand(1, 1, generator=g),
                           spectrum=torch.rand(1, bins, generator=g))
    return TrainingInputData(input=inp, ground_truth=gt)


def test_multires_downsample_is_exact_voxel_mean():
    from radfield3dnn.preprocessing.augmentations.multi_resolution import MultiResolutionResampling
    torch.manual_seed(0)
    mr = MultiResolutionResampling(scales=(0.5,)); mr.train()
    x = _mr_batch(N=8)
    out = mr(x)
    flux, spec = out.ground_truth.flux, out.ground_truth.spectrum
    assert flux.shape[-1] == 4
    manual = x.ground_truth.flux.reshape(1, 1, 4, 2, 4, 2, 4, 2).mean(dim=(3, 5, 7))
    assert torch.allclose(flux, manual, atol=1e-6)          # area pool == exact voxel mean
    assert torch.allclose(spec.sum(dim=1), torch.ones_like(spec.sum(dim=1)), atol=1e-5)  # renormalized
    # flux-weighted merge: brightest source voxel dominates the pooled spectrum
    w = x.ground_truth.flux[0, 0, :2, :2, :2].reshape(-1)
    s = x.ground_truth.spectrum[0, :, :2, :2, :2].reshape(4, -1)
    expected = (s * w).sum(dim=1) / w.sum()
    expected = expected / expected.sum()
    assert torch.allclose(spec[0, :, 0, 0, 0], expected, atol=1e-5)


def test_multires_upsample_and_eval_noop_and_inf_guard():
    from radfield3dnn.preprocessing.augmentations.multi_resolution import MultiResolutionResampling
    mr = MultiResolutionResampling(scales=(2.0,)); mr.train()
    x = _mr_batch(N=4)
    out = mr(x)
    assert out.ground_truth.flux.shape[-1] == 8              # trilinear up
    mr.eval()
    assert mr(x).ground_truth.flux.shape[-1] == 4            # eval no-op
    mr.train()
    bad = x._replace(ground_truth=x.ground_truth._replace(
        flux=x.ground_truth.flux.clone().index_fill_(2, torch.tensor([0]), float("-inf"))))
    with pytest.raises(AssertionError, match="BEFORE"):
        mr(bad)


def test_ipe_auto_region_width_follows_grid():
    torch.manual_seed(0)
    kw = _kwargs(1)
    kw["location_encoding_params"] = {"type": "ipe", "pos_enc_dim": 4, "append_input": True,
                                      "auto_region_width": True}
    m = PBRFNet(**kw); m.max_inner_batch_size = 512; m.eval()
    enc = m.get_core_model().positional_location_encoding
    with torch.no_grad():
        m.forward2volume(_input(), torch.tensor([4, 4, 4]), spectra_bins=8)
        assert abs(enc.region_width - 1.0 / 4) < 1e-9   # voxels_centered False -> 1/N frame
        m.forward2volume(_input(), torch.tensor([8, 8, 8]), spectra_bins=8)
        assert abs(enc.region_width - 1.0 / 8) < 1e-9


def test_region_hook_is_generic_and_point_encoders_ignore_it():
    # the model announces its query geometry to ANY encoder; point encoders must no-op
    m = _model(1)                                   # sinusoidal location encoder
    m.set_query_region(0.25)                        # must not raise, must not alter behaviour
    m.eval()
    with torch.no_grad():
        out = m.forward2volume(_input(), torch.tensor([4, 4, 4]), spectra_bins=8)
    assert torch.isfinite(out.scatter_field.flux).all()


def test_external_single_voxel_query_requires_and_honours_region():
    # querying the model DIRECTLY (not via forward2volume) must still use a correctly configured
    # encoder: unconfigured auto-IPE fails loudly; after set_query_region it encodes with damping.
    from radfield3dnn.models.encoders.ipe_encoding import IntegratedSinusoidalEncoding
    enc = IntegratedSinusoidalEncoding(pos_enc_dim=4, d_input=3, append_input=True,
                                       auto_region_width=True)
    x = torch.rand(5, 3)
    with pytest.raises(RuntimeError, match="set_query_region"):
        enc(x)
    enc.set_query_region(1.0 / 32)
    damped = enc(x)
    point = IntegratedSinusoidalEncoding(pos_enc_dim=4, d_input=3, append_input=True)  # region None
    assert not torch.allclose(damped, point(x))     # damping actually applied

    # and through the model's public API, for a single-voxel PositionalInput forward
    torch.manual_seed(0)
    kw = _kwargs(1)
    kw["location_encoding_params"] = {"type": "ipe", "pos_enc_dim": 4, "append_input": True,
                                      "auto_region_width": True}
    m = PBRFNet(**kw); m.eval()
    from radfield3dnn.rftypes import PositionalInput
    xi = _input()
    single = PositionalInput(direction=xi.direction, origin=xi.origin, spectrum=xi.spectrum,
                             position=torch.rand(1, 3))
    with pytest.raises(RuntimeError, match="set_query_region"):
        m.get_core_model()(single)
    m.set_query_region(m.voxel_width_in_encoder_frame(torch.tensor([64, 64, 64])))
    with torch.no_grad():
        out = m.get_core_model()(single)
    assert torch.isfinite(out.scatter_field.flux).all()

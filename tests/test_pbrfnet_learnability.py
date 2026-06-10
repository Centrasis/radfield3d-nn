"""PBRFNet flux-path bug-hunt: gradient flow + learnability guarantees.

These tests pin down that the PBRFNet implementation itself is sound — i.e. when
the real-data accuracy flatlines it is a *recipe* problem (normalizer / loss /
init crush), NOT a broken forward/backward path. They are the "model can learn
and the path is clean according to the gradients" guard the project requires
before trusting any experiment sweep.

What is asserted (log-space recipe = LogScaleNormalizer + bias at the log mean):
* ``test_flux_output_sits_at_bias_not_clamp_floor`` — at init the flux head
  emits ~bias (here -3), proving no predict-0 / clamp-floor lock-in.
* ``test_flux_gradient_flow`` — every learnable module on the flux path receives
  a finite, non-zero gradient (flux head, trunk blocks, FiLM conditioners, beam
  encoder). The fixed sinusoidal positional encoding correctly has none.
* ``test_flux_overfits_single_structured_field`` — the model drives a structured
  HDR beam-blob target from L1~2.6/corr~0 to L1<0.15/corr>0.99: the path *can*
  represent and learn spatial flux structure.

If any of these fail, the PBRFNet path has regressed and ALL experiment results
must be invalidated and re-run on the fixed network.
"""
import torch
import torch.nn.functional as F
import pytest

try:
    from RadFiled3D.pytorch.types import PositionalInput
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.preprocessing.normalizations.logscale import LogScaleNormalizer


LOG_MEAN_BIAS = -3.0  # the log-space prior the auto-bias-init lands on for DS03


def _build_model():
    """PBRFNet in the log-space recipe (matches the PBRFNet-logflux config)."""
    torch.manual_seed(0)
    norm = LogScaleNormalizer(x_min=1e-6, x_max=1.0, zero_floor=-7.0)
    m = PBRFNet(
        d_model=192,
        location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 12, "append_input": True},
        direction_encoding_params={"type": "spherical_harmonics", "degree": 4, "append_input": True},
        spectra_encoding_params={"type": "simple", "in_spectra_dim": 32, "encoded_spectra_dims": 32},
        out_spectra_dim=32, conditioning_params={"type": "FiLM", "use_beam_shape": False},
        normalizer=norm, flux_loss="FluxLoss", flux_activation="clamp",
    )
    core = m.get_core_model()
    flux_lins = [mm for mm in core.flux_decoder.modules() if isinstance(mm, torch.nn.Linear)]
    torch.nn.init.constant_(flux_lins[-1].bias, LOG_MEAN_BIAS)
    return m, core


def _synthetic_batch(B=4096, seed=1, single_beam=False):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(B, 3, generator=g) - 0.5
    direction = F.normalize(torch.randn(B, 3, generator=g), dim=-1)
    origin = torch.full((B, 1), 0.7) if single_beam else torch.rand(B, 1, generator=g) * 0.5 + 0.5
    spectrum = torch.rand(B, 32, generator=g)
    spectrum = spectrum / spectrum.sum(-1, keepdim=True)
    batch = PositionalInput(direction=direction, origin=origin, spectrum=spectrum,
                            position=pos, geometry=None, beam_shape_type=None,
                            beam_shape_parameters=None)
    return batch, pos


def _beam_blob_target(pos, sigma=0.12):
    """HDR-like log-space target: a Gaussian beam blob from -6 background to ~0 peak."""
    r = (pos[:, :2] ** 2).sum(-1).sqrt()
    return (-6.0 + 6.0 * torch.exp(-(r ** 2) / (2 * sigma ** 2))).detach()


def test_flux_output_sits_at_bias_not_clamp_floor():
    """At init the flux head must emit ~bias, not be pinned at the clamp floor
    (the predict-0 lock-in that starves the crushed scatter band of gradient)."""
    m, core = _build_model()
    batch, _ = _synthetic_batch()
    with torch.no_grad():
        flux = m(batch).scatter_field.flux
    assert torch.isfinite(flux).all()
    # mean within 0.2 of the bias prior, and clearly off both clamp boundaries.
    assert abs(flux.mean().item() - LOG_MEAN_BIAS) < 0.2
    assert flux.min().item() > core.flux_activation.min_value + 1.0
    assert flux.max().item() < core.flux_activation.max_value - 1.0


def test_flux_gradient_flow():
    """Every learnable module on the flux path gets a finite non-zero gradient."""
    m, core = _build_model()
    batch, pos = _synthetic_batch()
    target = _beam_blob_target(pos)
    loss = F.l1_loss(m(batch).scatter_field.flux, target)
    loss.backward()

    def grad_sum(mod):
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in gs), "non-finite gradient"
        return sum(g.abs().sum().item() for g in gs)

    # learnable modules on the flux path must all receive gradient
    for name in ("flux_decoder", "block1", "block2",
                 "beam_encoder", "beam_conditioner1", "beam_conditioner2"):
        mod = getattr(core, name)
        assert grad_sum(mod) > 0.0, f"{name} received zero gradient"

    # the sinusoidal positional encoding is parameter-free → correctly no grad
    assert all(p.grad is None for p in core.positional_location_encoding.parameters()) \
        or len(list(core.positional_location_encoding.parameters())) == 0


def test_flux_overfits_single_structured_field():
    """The flux path can represent + learn spatial HDR structure: a single field
    is driven to near-perfect correlation. Proves the architecture, not a recipe,
    is what limits real-data accuracy."""
    m, core = _build_model()
    batch, pos = _synthetic_batch(B=8192, single_beam=True)
    target = _beam_blob_target(pos)

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    init_loss = None
    for it in range(400):
        opt.zero_grad()
        pred = m(batch).scatter_field.flux
        loss = F.l1_loss(pred, target)
        loss.backward()
        opt.step()
        if init_loss is None:
            init_loss = loss.item()

    with torch.no_grad():
        pred = m(batch).scatter_field.flux
        final_loss = F.l1_loss(pred, target).item()
        corr = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()

    assert init_loss > 1.0, "target should start far from the -3 prior"
    assert final_loss < 0.15, f"failed to overfit: final L1={final_loss:.3f}"
    assert corr > 0.99, f"failed to learn structure: corr={corr:.3f}"
    # learned the full HDR span, not a collapsed constant
    assert pred.max().item() - pred.min().item() > 4.0


def test_sigmoid_overfits_crushed_linear_field_no_lockin():
    """The `sigmoid` flux activation represents a CRUSHED linear0_1 HDR field (background ~1e-3,
    peak ~1.0) without the clamp predict-0 lock-in: from a logit(center) init the network learns
    both the tiny background AND the peak. This guards the re-added sigmoid HDR path."""
    import math
    torch.manual_seed(0)
    from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
    m = PBRFNet(
        d_model=192,
        location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 12, "append_input": True},
        direction_encoding_params={"type": "spherical_harmonics", "degree": 4, "append_input": True},
        spectra_encoding_params={"type": "simple", "in_spectra_dim": 32, "encoded_spectra_dims": 32},
        out_spectra_dim=32, conditioning_params={"type": "FiLM", "use_beam_shape": False},
        normalizer=LinearNormalizer((0.0, 1.0)), flux_loss="FluxLoss",
        flux_activation="sigmoid",
    )
    core = m.get_core_model()
    from radfield3dnn.models.activations.flux_activations import LogitSigmoid
    assert isinstance(core.flux_activation, LogitSigmoid)

    batch, pos = _synthetic_batch(B=8192, single_beam=True)
    r = (pos[:, :2] ** 2).sum(-1).sqrt()
    target = (1e-3 + (1.0 - 1e-3) * torch.exp(-(r ** 2) / (2 * 0.12 ** 2))).detach()  # crushed linear HDR

    flux_lins = [mm for mm in core.flux_decoder.modules() if isinstance(mm, torch.nn.Linear)]
    torch.nn.init.constant_(flux_lins[-1].bias, math.log(1e-3 / (1 - 1e-3)))  # logit(center)

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = (m(batch).scatter_field.flux - target).abs().mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = m(batch).scatter_field.flux
        corr = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()
    assert corr > 0.99, f"sigmoid failed to learn crushed HDR: corr={corr:.3f}"
    assert pred.min().item() < 5e-3, "sigmoid lost the crushed background (lock-in?)"
    assert pred.max().item() > 0.9, "sigmoid failed to reach the peak"

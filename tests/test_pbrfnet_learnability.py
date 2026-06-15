"""PBRFNet flux-path learnability guarantees.

These tests pin down that the PBRFNet implementation is sound: every learnable module on
the flux path receives a finite non-zero gradient, and the network can represent + learn
spatial HDR flux structure. They guard the shipped recipe (linear0_1 normalizer + sigmoid
flux head), where the target spans from a crushed background up to a sharp peak.
"""
import math
import torch
import torch.nn.functional as F
import pytest

try:
    from RadFiled3D.pytorch.types import PositionalInput
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.models.nerf import PBRFNet
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
from radfield3dnn.models.activations.flux_activations import LogitSigmoid


def _build_model():
    """PBRFNet in the shipped recipe: linear0_1 normalizer + sigmoid flux head."""
    torch.manual_seed(0)
    m = PBRFNet(
        d_model=192,
        location_encoding_params={"type": "sinusoidal", "pos_enc_dim": 12, "append_input": True},
        direction_encoding_params={"type": "spherical_harmonics", "degree": 4, "append_input": True},
        spectra_encoding_params={"type": "simple", "in_spectra_dim": 32, "encoded_spectra_dims": 32},
        out_spectra_dim=32, conditioning_params={"type": "FiLM", "use_beam_shape": False},
        normalizer=LinearNormalizer((0.0, 1.0)), flux_loss="SMAPEBalanced",
        flux_activation="sigmoid",
    )
    return m, m.get_core_model()


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


def _blob_target(pos, sigma=0.2):
    """Smooth Gaussian beam-blob target in [0, 1]: 0.05 background up to a ~1.0 peak."""
    r = (pos[:, :2] ** 2).sum(-1).sqrt()
    return (0.05 + 0.95 * torch.exp(-(r ** 2) / (2 * sigma ** 2))).detach()


def test_flux_output_is_bounded_and_unlocked():
    """At init the sigmoid flux head emits values inside (0, 1), not pinned at a boundary."""
    m, core = _build_model()
    assert isinstance(core.flux_activation, LogitSigmoid)
    batch, _ = _synthetic_batch()
    with torch.no_grad():
        flux = m(batch).scatter_field.flux
    assert torch.isfinite(flux).all()
    assert flux.min().item() >= 0.0 and flux.max().item() <= 1.0
    assert flux.max().item() - flux.min().item() > 0.0, "head is not stuck at a constant boundary"


def test_flux_gradient_flow():
    """Every learnable module on the flux path gets a finite non-zero gradient."""
    m, core = _build_model()
    batch, pos = _synthetic_batch()
    target = _blob_target(pos)
    loss = F.l1_loss(m(batch).scatter_field.flux, target)
    loss.backward()

    def grad_sum(mod):
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in gs), "non-finite gradient"
        return sum(g.abs().sum().item() for g in gs)

    for name in ("flux_decoder", "block1", "block2",
                 "beam_encoder", "beam_conditioner1", "beam_conditioner2"):
        mod = getattr(core, name)
        assert grad_sum(mod) > 0.0, f"{name} received zero gradient"

    # the sinusoidal positional encoding is parameter-free → correctly no grad
    assert all(p.grad is None for p in core.positional_location_encoding.parameters()) \
        or len(list(core.positional_location_encoding.parameters())) == 0


def test_sigmoid_overfits_structured_field():
    """The sigmoid flux head can represent and learn spatial HDR structure: a single field is
    driven to near-perfect correlation, covering the full [background, peak] span — proving the
    flux path is not pinned at a constant / boundary."""
    m, core = _build_model()
    batch, pos = _synthetic_batch(B=8192, single_beam=True)
    target = _blob_target(pos)

    flux_lins = [mm for mm in core.flux_decoder.modules() if isinstance(mm, torch.nn.Linear)]
    torch.nn.init.constant_(flux_lins[-1].bias, math.log(0.05 / (1 - 0.05)))  # logit(background)

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = (m(batch).scatter_field.flux - target).abs().mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = m(batch).scatter_field.flux
        corr = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()
    assert corr > 0.95, f"sigmoid failed to learn structure: corr={corr:.3f}"
    assert pred.max().item() > 0.8, "sigmoid failed to reach the peak"
    assert pred.max().item() - pred.min().item() > 0.5, "collapsed to a near-constant field"

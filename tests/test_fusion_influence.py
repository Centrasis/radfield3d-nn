"""Specialized fusion-influence test.

A voxel's value depends on its location (relative to the implicit geometry) AND the beam parameters.
A fusion is the module that lets the beam (cond) steer the location feature (x). This test measures,
per fusion module *in isolation* (no surrounding model):

  * conditioning AUTHORITY  — Jacobian RMS  ‖∂y/∂cond‖ / ‖∂y/∂x‖ : how hard the beam can move the
    output vs how hard the location can. The unit-level analogue of the model's beam/spatial 0.012.
  * realised SPAN          — var(y across conds, x fixed) / var(y across x, cond fixed).
  * lightweight COST        — parameter count (+ latency in the __main__ benchmark, needs a free GPU).

The pytest asserts the FusionBase contract and that ConcatLinear's additive conditioning grants the
beam at least as much authority as FiLM's per-channel affine (the design hypothesis), at <= FiLM params.

Run the full ranked benchmark (params + fp32/fp16 latency) with a FREE GPU:
    python -m tests.test_fusion_influence
"""
import torch
import torch.nn as nn
import pytest

from radfield3dnn.models.layers import (
    FiLM, ResidualFiLM, GatedFusion, ConcatLinear, CrossAttentionFusion,
    TokenCrossAttentionFusion, FusionBase,
)


def _build(d, dc, act=nn.SiLU):
    # TokenCrossAttention is measured in two regimes: as ONE token over the whole cond vector (the
    # apples-to-apples drop-in) and as a SET of 4 equal tokens (its intended per-parameter regime,
    # where attention conditioning is meant to win — Rebain et al. 2022). Both consume cond width dc.
    n_tok = 4 if dc % 4 == 0 else 1
    return {
        "FiLM":          FiLM(dc, d, non_linearity=act),
        "ResidualFiLM":  ResidualFiLM(dc, d, non_linearity=act),
        "GatedFusion":   GatedFusion(d, dc, hidden=d),
        "ConcatLinear":  ConcatLinear(dc, d, non_linearity=act),
        "CrossAttention": CrossAttentionFusion(dc, d, n_heads=4, n_tokens=4),
        "TokenCrossAttn1": TokenCrossAttentionFusion(dc, d, n_heads=4),
        "TokenCrossAttn4": TokenCrossAttentionFusion([dc // n_tok] * n_tok, d, n_heads=4),
    }


@torch.no_grad()
def _span(fusion, d, dc, device, n=512, gen=None):
    """Output variance across cond (x fixed) vs across x (cond fixed)."""
    g = gen or torch.Generator(device=device).manual_seed(0)
    x0 = torch.randn(1, d, generator=g, device=device).expand(n, d).contiguous()
    conds = torch.randn(n, dc, generator=g, device=device)
    y_across_cond = fusion(x0, conds)                       # vary beam, fix location
    c0 = torch.randn(1, dc, generator=g, device=device).expand(n, dc).contiguous()
    xs = torch.randn(n, d, generator=g, device=device)
    y_across_x = fusion(xs, c0)                             # vary location, fix beam
    return float(y_across_cond.var(0).mean()), float(y_across_x.var(0).mean())


def _authority(fusion, d, dc, device, n=512, gen=None):
    """Jacobian RMS ratio ‖∂y/∂cond‖ / ‖∂y/∂x‖ over random (x, cond)."""
    g = gen or torch.Generator(device=device).manual_seed(1)
    x = torch.randn(n, d, generator=g, device=device, requires_grad=True)
    cond = torch.randn(n, dc, generator=g, device=device, requires_grad=True)
    y = fusion(x, cond)
    gx, gc = torch.autograd.grad(y.sum(), [x, cond])
    rms = lambda t: float(t.pow(2).mean().sqrt())
    sx = rms(gx)
    return rms(gc) / max(sx, 1e-12), sx


def _randomize_(fusion):
    """Re-init every Linear to a common scheme so 'authority' reflects the fusion's CAPACITY (a
    trained-like state), not its special init. Several fusions zero-init the cond pathway for an
    identity start (ConcatLinear, ResidualFiLM α=0), which would read as zero authority at init —
    the wrong quantity for "how strongly CAN the beam move a voxel"."""
    with torch.no_grad():
        for m in fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                pass
        # ResidualFiLM gates its modulation by a learnable α (init 0) — open it to measure capacity.
        if hasattr(fusion, "alpha"):
            nn.init.constant_(fusion.alpha, 1.0)
    return fusion


def measure(d=192, dc=192, device="cpu", capacity=True):
    rows = {}
    for name, f in _build(d, dc).items():
        f = f.to(device)
        if capacity:
            _randomize_(f)
        auth, _ = _authority(f, d, dc, device)
        vc, vx = _span(f, d, dc, device)
        rows[name] = {
            "params": sum(p.numel() for p in f.parameters()),
            "authority_jac": auth,
            "span_beam": vc, "span_loc": vx,
            "span_ratio": vc / max(vx, 1e-12),
        }
    return rows


# ─────────────────────────── pytest (CPU, fast) ───────────────────────────

def test_fusion_contract_same_dim():
    """Every fusion is a FusionBase and returns dim(y)==dim(x)."""
    d, dc = 64, 48
    x = torch.randn(32, d); cond = torch.randn(32, dc)
    for name, f in _build(d, dc).items():
        assert isinstance(f, FusionBase), f"{name} not a FusionBase"
        y = f(x, cond)
        assert y.shape == (32, d), f"{name} broke the same-dim contract: {tuple(y.shape)}"


def test_concatlinear_identity_at_init():
    """ConcatLinear (no norm/act) starts as an exact identity in x when cond=0 — beam off at init."""
    d, dc = 64, 48
    f = ConcatLinear(dc, d, non_linearity=None, norm="none")
    x = torch.randn(16, d)
    y = f(x, torch.zeros(16, dc))
    assert torch.allclose(y, x, atol=1e-6)


def test_concatlinear_beam_authority_ge_film():
    """The design hypothesis, measured by the user's quantity — REALISED beam-vs-location influence on
    the voxel (span_ratio = var(y across beams) / var(y across location)).

    ConcatLinear's full-rank additive conditioning should give the beam at least as much realised
    authority over a voxel as FiLM's per-channel affine, at <= FiLM's parameter cost. FiLM's bounded
    affine is the weakest conditioner here (the unit-level echo of the model's 0.012 beam/spatial)."""
    rows = measure(d=128, dc=128, device="cpu")
    assert rows["ConcatLinear"]["params"] <= rows["FiLM"]["params"]
    assert rows["ConcatLinear"]["span_ratio"] >= rows["FiLM"]["span_ratio"]
    # every fusion gives the beam a finite, non-trivial voice
    for name, r in rows.items():
        assert r["span_ratio"] > 0.0 and r["span_ratio"] == r["span_ratio"], name
        assert r["authority_jac"] == r["authority_jac"], name  # finite


# ─────────────────────────── GPU benchmark (free GPU) ───────────────────────────

def _latency(fusion, d, dc, device, dtype, B=131072, iters=50):
    fusion = fusion.to(device=device, dtype=dtype)
    x = torch.randn(B, d, device=device, dtype=dtype)
    cond = torch.randn(B, dc, device=device, dtype=dtype)
    import time
    with torch.no_grad():
        for _ in range(5):
            fusion(x, cond)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fusion(x, cond)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms for B voxels


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = dc = 192
    rows = measure(d, dc, device=dev)
    print(f"\nFusion influence + cost  (d={d}, cond={dc}, device={dev})")
    print(f"{'fusion':14s} {'params':>9s} {'authority':>10s} {'span_beam':>11s} {'span_loc':>10s} {'span_ratio':>11s}", end="")
    if dev == "cuda":
        print(f" {'fp32 ms':>9s} {'fp16 ms':>9s}")
    else:
        print()
    for name in _build(d, dc).keys():
        r = rows[name]; line = (f"{name:14s} {r['params']:>9d} {r['authority_jac']:>10.4f} "
                                f"{r['span_beam']:>11.4e} {r['span_loc']:>10.4e} {r['span_ratio']:>11.4f}")
        if dev == "cuda":
            f = _build(d, dc)[name]
            try: l32 = _latency(f, d, dc, dev, torch.float32)
            except Exception: l32 = float('nan')
            try: l16 = _latency(_build(d, dc)[name], d, dc, dev, torch.float16)
            except Exception: l16 = float('nan')
            line += f" {l32:>9.3f} {l16:>9.3f}"
        print(line)
    print("\nauthority = ‖∂y/∂cond‖/‖∂y/∂x‖ (beam vs location power); higher = stronger conditioning.")


if __name__ == "__main__":
    main()

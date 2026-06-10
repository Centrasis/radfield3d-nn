"""AnalyticDirectBeam — a reusable, ONNX-exportable torch module that reconstructs
the AIR direct-beam field analytically (no Monte Carlo, no simulated direct).

Shared by the field-wise model (FieldScatterUNet) and the per-voxel MLP scatter-only
variant: both learn ONLY the scatter and obtain the direct beam from this module.

Physics (validated vs the DS03 GT direct, corr(log) ~0.96):
  direct(p) = C · (1/r²) · air_transmission(spectrum, r) · in_beam(p) · ¬shadow(p)
  - source = origin·field − 0.5·field (field-relative origin → centred metres),
  - in_beam: diverging rectangle; local axis (0,0,−1) + rect in local X/Y mapped by
    the MINIMAL rotation to the beam direction (RadField3DSimulation convention),
  - air_transmission: exp(−μ̄_air·r), μ̄ = Σ_E w(E)·μ_air(E) (spectrum-weighted mean,
    μ per-bin precomputed as a buffer → torch-only),
  - shadow: ray-march source→voxel through the `density` channel via torch.gather,
  - C = voxel_area·|O|²/rect_area → physical counts/primary (drop-in for sim direct).

Every op is standard ONNX (MatMul/Gemm, Sin? no; Cross→Mul/Sub, Exp, Reciprocal,
Less, And, Gather, Where, Floor, Cast), so it lowers cleanly to ONNX Runtime C++.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class AnalyticDirectBeam(nn.Module):
    # NIST dry-air total mass attenuation (cm^2/g) anchor points (log-log interp).
    _AIR_E = (10.0, 15, 20, 30, 40, 50, 60, 80)
    _AIR_MU = (5.120, 1.614, 0.7779, 0.3538, 0.2485, 0.2080, 0.1875, 0.1662)
    _RHO_AIR = 1.205e-3  # g/cm^3

    def __init__(self, voxel_size_m: float = 0.02, max_spectrum_bins: int = 256,
                 bin_width_keV: float = 1.0, n_shadow: int = 24, field_size_m: float = 1.0,
                 buildup: float = 1.46, penumbra_frac: float = 0.08):
        super().__init__()
        self.vd = float(voxel_size_m)
        self.n_shadow = int(n_shadow)
        self.field = float(field_size_m)
        self.buildup = float(buildup)
        self.penumbra_frac = float(penumbra_frac)
        # μ_air per 1-keV spectrum bin, precomputed (torch-only at runtime)
        E = (np.arange(max_spectrum_bins) + 0.5) * bin_width_keV
        mu = np.exp(np.interp(np.log(np.clip(E, self._AIR_E[0], self._AIR_E[-1])),
                              np.log(self._AIR_E), np.log(self._AIR_MU))) * self._RHO_AIR
        self.register_buffer("air_mu_bins", torch.tensor(mu, dtype=torch.float32), persistent=False)
        self._grid = None

    def voxel_size_for(self, dims) -> float:
        """The PHYSICAL voxel size for a grid of `dims` covering the fixed field box
        (`field_size_m`). Hardcoding 0.02 only matched 50³; at any other resolution the
        beam geometry must use field/dims or the whole field is mis-sized."""
        return self.field / float(dims[0])

    def _voxel_grid(self, dims, device, dtype, vd) -> Tensor:
        if self._grid is None or self._grid.shape[:3] != tuple(dims):
            cc = [(torch.arange(n, device=device, dtype=dtype) + 0.5) * vd - 0.5 * n * vd for n in dims]
            X, Y, Z = torch.meshgrid(cc[0], cc[1], cc[2], indexing="ij")
            self._grid = torch.stack([X, Y, Z], -1)
        return self._grid.to(device=device, dtype=dtype)

    def forward(self, direction: Tensor, origin: Tensor, spectrum: Tensor,
                rect: Tensor, density: Tensor = None, dims=None, query_points: Tensor = None,
                voxel_size: float = None) -> Tensor:
        """direction(B,3) origin(B,3 field-relative) spectrum(B,S) rect(B,2 m).
        Either over the full voxel grid (→ direct(B,1,D,H,W); needs `density` or
        `dims`) OR at explicit `query_points` (B,N,3) centred-metres → direct(B,N)
        (per-voxel "fetch by xyz" path; shadow skipped). Physical counts/primary.
        `voxel_size` overrides the grid spacing (required in point mode for the
        physical-scale constant C; defaults to field/dims in grid mode)."""
        dt = direction.dtype
        dev = direction.device
        B = direction.shape[0]

        if query_points is not None:
            P = query_points.to(dt)                          # (B,N,3) per-sample points
            N = P.shape[1]
            point_mode = True
            vd = float(voxel_size) if voxel_size is not None else self.vd
        else:
            dims = tuple(density.shape[-3:]) if density is not None else tuple(dims)
            N = dims[0] * dims[1] * dims[2]
            point_mode = False
            vd = float(voxel_size) if voxel_size is not None else self.voxel_size_for(dims)
            P = self._voxel_grid(dims, dev, dt, vd).reshape(-1, 3)

        O = origin.to(dt) * self.field - 0.5 * self.field
        D = direction.to(dt)
        D /= D.norm(dim=1, keepdim=True)
        w = spectrum.to(dt)
        w /= w.sum(dim=1, keepdim=True).clamp_min(1e-30)
        mu_eff = (w * self.air_mu_bins[: w.shape[1]].to(dev, dt)[None]).sum(1)
        # rect axes via minimal rotation (Rodrigues, (1-c)/s² = 1/(1+c))
        up = torch.tensor([0., 0., -1.], device=dev, dtype=dt).expand(B, 3)
        v = torch.cross(up, D, dim=1); cdot = (up * D).sum(1)
        k = 1.0 / (1.0 + cdot).clamp_min(1e-6)
        Vx = torch.zeros(B, 3, 3, device=dev, dtype=dt)
        Vx[:, 0, 1] = -v[:, 2]; Vx[:, 0, 2] = v[:, 1]; Vx[:, 1, 0] = v[:, 2]
        Vx[:, 1, 2] = -v[:, 0]; Vx[:, 2, 0] = -v[:, 1]; Vx[:, 2, 1] = v[:, 0]
        R = torch.eye(3, device=dev, dtype=dt)[None] + Vx + torch.bmm(Vx, Vx) * k[:, None, None]
        e1 = R[:, :, 0]; e2 = R[:, :, 1]
        rv = (P - O[:, None, :]) if point_mode else (P[None] - O[:, None, :])
        r = rv.norm(dim=2)
        d_along = (rv * D[:, None, :]).sum(2)
        lat = rv - d_along[..., None] * D[:, None, :]
        l1 = (lat * e1[:, None, :]).sum(2); l2 = (lat * e2[:, None, :]).sum(2)
        dref = O.norm(dim=1)
        # half-widths at this depth (diverging cone); the MC 50% edge sits at the
        # geometric half-width (measured ratio 1.03), so place the sigmoid 50% point
        # exactly there and soften over penumbra_frac·hw (10→90% ≈ 4.4·sigmoid-width).
        hw = (rect[:, 0:1] / 2) * d_along / dref[:, None]
        hh = (rect[:, 1:2] / 2) * d_along / dref[:, None]
        w1 = (self.penumbra_frac * hw / 4.4).clamp_min(1e-6)
        w2 = (self.penumbra_frac * hh / 4.4).clamp_min(1e-6)
        soft = torch.sigmoid((hw - l1.abs()) / w1) * torch.sigmoid((hh - l2.abs()) / w2)
        # drop the sigmoid tail to keep the field sparse (and the MC floor honest)
        in_beam = torch.where(soft > 1e-2, soft, torch.zeros_like(soft)) * (d_along > 0).to(dt)
        air = torch.exp(-mu_eff[:, None] * r * 100.0)
        C = self.buildup * (vd * vd) * (dref * dref) / (rect[:, 0] * rect[:, 1]).clamp_min(1e-9)
        # counts/primary is physically <= 1: the 1/r² point-source model diverges near
        # the source (which can sit inside the 1 m field box for some geometries), so
        # floor r at sqrt(C) — the radius where C/r² = 1 — to cap the direct at 1.
        r_min = C.sqrt()[:, None]
        direct = C[:, None] * (1.0 / torch.maximum(r, r_min) ** 2) * air * in_beam
        # phantom shadow via gather (skipped when no density channel is available)
        if density is not None:
            dens = density.reshape(B, N)
            t = torch.linspace(0.02, 0.95, self.n_shadow, device=dev, dtype=dt)
            samp = O[:, None, None, :] + (P[None, :, None, :] - O[:, None, None, :]) * t[None, None, :, None]
            gd = torch.tensor(dims, device=dev, dtype=dt)
            vi = ((samp + 0.5 * gd * vd) / vd).floor().to(torch.int64)
            vi = vi.clamp(torch.zeros(3, device=dev, dtype=torch.int64),
                          torch.tensor([d - 1 for d in dims], device=dev, dtype=torch.int64))
            flat = (vi[..., 0] * dims[1] + vi[..., 1]) * dims[2] + vi[..., 2]
            sampled = torch.gather(dens, 1, flat.reshape(B, -1)).reshape(B, N, self.n_shadow)
            shadow = (sampled > 0).any(2).to(dt)
            direct = direct * (1.0 - shadow)
        if point_mode:
            return direct                                    # (B,N)
        return direct.reshape(B, 1, dims[0], dims[1], dims[2])

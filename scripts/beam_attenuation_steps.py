#!/usr/bin/env python
"""Where does a model MISS the attenuation step the simulation shows?

Everything about the beam comes out of the dataset, not out of a guess. Each field's dynamic
metadata carries the collimated rectangle at the isocentre (``xray_tube_field_rect_dimensions_m``),
its shape, the patient translation and the tube spectrum; ``radiation_origin`` is the focal spot
relative to the field centre in metres, so ``radiation_direction == -origin/|origin|`` and the
source-to-isocentre distance is ``|origin|``. The rectangle's ROLL is not recorded and is recovered
from the field itself: in the plane across the beam the direct beam's second moments are the
rectangle's, and their principal angle is the collimator's.

The model is then driven with exactly that configuration and compared against the simulated field
RAY BY RAY. Along a ray the transmission

    T(t) = flux(t) r(t)^2 / (flux(r_ref) r_ref^2)

removes the inverse square and the beam's own profile, because the reference is a point of the same
ray. Comparing the model's T against the simulation's T therefore needs no attenuation coefficient
and no Beer-Lambert model: whatever the real material does, both fields are asked the same question,
and Monte-Carlo scatter — which fills a shadow in by most of a decade and which no analytic
reference reproduces — is present in the reference rather than charged to the model.

Reported per ray that crosses the body:

  step        how far the transmission falls between entering the phantom and leaving it, in decades
  missing     the simulation's step minus the model's; positive means the model let through what
              the geometry stopped
  onset       how far past the phantom's surface each field has lost half a decade, in cm; the
              offset says whether a step that IS there sits at the surface or is smeared past it

The null is the stretch of each ray BEFORE the phantom. A ray that misses the body entirely would
be the obvious control, but when the body is in the beam almost none do; the air between the focal
spot and the patient is on every ray, carries no step, and is where the two fields should already
agree. Its spread is the model's ordinary error, and no step smaller than it is a finding.
"""
import argparse
import glob
import os

import numpy as np
import torch
from radfiled3d.store import FieldStore
from radfiled3d.nn import deploy, inference

TRANSLATION_SPAN = {"x": (-0.25, 0.25), "y": (-0.5, 0.5)}


def read_field(path):
    """The simulated field and the complete beam configuration that produced it."""
    field = FieldStore.load(path)
    meta = FieldStore.load_metadata(path)          # held: the voxel views point into it
    tube = meta.get_header().simulation.tube
    origin = np.array([tube.radiation_origin.x, tube.radiation_origin.y, tube.radiation_origin.z])
    direction = np.array([tube.radiation_direction.x, tube.radiation_direction.y,
                          tube.radiation_direction.z])
    rect = meta.get_dynamic_metadata("xray_tube_field_rect_dimensions_m").get_data()
    translation = meta.get_dynamic_metadata("patient_translation").get_data()
    flux = field.get_channel("radiation").get_layer_as_ndarray("flux")[..., 0].astype(np.float64)
    return dict(
        flux=flux,
        density=field.get_channel("geometry").get_layer_as_ndarray("density")[..., 0].astype(np.float64),
        origin=origin, direction=direction, sid_m=float(np.linalg.norm(origin)),
        rect_m=np.array([rect.x, rect.y]),
        shape=float(meta.get_dynamic_metadata("xray_field_shape").get_data()),
        translation_m=np.array([translation.x, translation.y, translation.z]),
        # COPIED, not wrapped: the voxel accessors hand back views into `meta`, which is
        # released when this returns, and a dangling spectrum reads as noise that changes
        # from field to field.
        spectrum=np.array(meta.get_dynamic_metadata("tube_spectrum").get_data(), np.float32),
        side=flux.shape[0], voxel_m=float(field.get_field_dimensions().x) / flux.shape[0])


class Model:
    """The package, queried on the same grid the dataset uses."""

    def __init__(self, package, side, device=0):
        self.session = inference.Session(deploy.ModelStore.load(package), backend="cuda", device=device)
        self.session.set_voxel_grid([side] * 3)
        self.shapes = {t.name: tuple(t.shape)
                       for t in deploy.ModelStore.load_metadata(package).tensors if t.role == "input"}
        sid = [t for t in deploy.ModelStore.load_metadata(package).tensors
               if t.name == "source_distance"][0].range
        self.sid_lo, self.sid_hi = float(sid[0]), float(sid[1])
        idx = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1) / side
        # RadFiled3D's voxel order is x fastest, then y, then z.
        self.positions = torch.from_numpy(
            np.transpose(idx, (2, 1, 0, 3)).reshape(-1, 3).astype(np.float32).copy()).cuda()
        self.side = side
        self._bound = {}
        # Allocated ONCE and rebound every call: the session keeps a raw pointer into whatever is
        # bound, so a buffer that is freed between calls leaves the session pointing at memory the
        # allocator has since handed to something else.
        self.flux = torch.zeros(side ** 3, device="cuda")
        self.spectrum = torch.zeros(side ** 3 * 32, device="cuda")

    def _bind(self, name, value):
        tensor = torch.from_numpy(np.asarray(value, np.float32).reshape(self.shapes[name])).cuda()
        self._bound[name] = tensor              # the session holds a raw pointer; keep it alive
        self.session.bind_input(name, tensor)

    def __call__(self, gt):
        self.session.bind_input("position", self.positions)
        self._bind("beam_direction", gt["direction"])
        self._bind("source_distance", [(gt["sid_m"] - self.sid_lo) / (self.sid_hi - self.sid_lo)])
        self._bind("beam_collimation", gt["rect_m"] / (gt["voxel_m"] * gt["side"]))
        self._bind("beam_collimation_kind", [gt["shape"]])
        self._bind("tube_spectrum", gt["spectrum"] / max(gt["spectrum"].sum(), 1e-12))
        self._bind("patient_translation", [
            (gt["translation_m"][a] - TRANSLATION_SPAN[k][0]) /
            (TRANSLATION_SPAN[k][1] - TRANSLATION_SPAN[k][0]) if k else 0.0
            for a, k in enumerate(("x", "y", None))])
        self.session.bind_output("flux", self.flux)
        self.session.bind_output("spectrum", self.spectrum)
        self.session.infer()
        torch.cuda.synchronize()
        return self.flux.cpu().numpy().astype(np.float64).reshape(
            self.side, self.side, self.side).transpose(2, 1, 0)


def collimator_roll(gt, e1, e2):
    """The rectangle's angle in the plane across the beam, from the direct beam's own moments."""
    n = gt["side"]
    source = np.full(3, n / 2.0) + gt["origin"] / gt["voxel_m"]
    idx = np.stack(np.meshgrid(*[np.arange(n) + 0.5] * 3, indexing="ij"), -1)
    v = idx - source
    axial = v @ gt["direction"]
    q1, q2 = v @ e1, v @ e2
    sid = gt["sid_m"] / gt["voxel_m"]
    reach = 0.5 * np.hypot(*gt["rect_m"]) / gt["voxel_m"] / sid
    window = (axial > 0.2 * sid) & (np.hypot(q1, q2) < 1.6 * reach * axial) & (gt["density"] <= 0)
    w = np.where(window, gt["flux"] * axial ** 2, 0.0)      # inverse square out: every depth counts alike
    w = np.where(w >= 0.25 * w.max(), w, 0.0)               # the direct beam, not its scattered halo
    if w.sum() <= 0:
        return 0.0
    n1 = np.divide(q1, axial, out=np.zeros_like(q1), where=axial > 0)
    n2 = np.divide(q2, axial, out=np.zeros_like(q2), where=axial > 0)
    m11 = (w * n1 * n1).sum() / w.sum()
    m22 = (w * n2 * n2).sum() / w.sum()
    m12 = (w * n1 * n2).sum() / w.sum()
    return 0.5 * float(np.arctan2(2 * m12, m11 - m22))


def beam_rays(gt, rays_across, core):
    """Directions filling the collimated rectangle, and the focal spot, in voxel units."""
    d = gt["direction"]
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    phi = collimator_roll(gt, e1, e2)
    a1 = e1 * np.cos(phi) + e2 * np.sin(phi)
    a2 = -e1 * np.sin(phi) + e2 * np.cos(phi)
    # The longer side goes with the principal axis the moments found.
    half = 0.5 * np.array([max(gt["rect_m"]), min(gt["rect_m"])]) / gt["sid_m"] * core
    g = (np.arange(rays_across) + 0.5) / rays_across * 2 - 1
    u, v = np.meshgrid(g, g, indexing="ij")
    dirs = d[None, :] + (u.ravel() * half[0])[:, None] * a1 + (v.ravel() * half[1])[:, None] * a2
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    source = np.full(3, gt["side"] / 2.0) + gt["origin"] / gt["voxel_m"]
    return source, dirs, np.degrees(phi)


def march(volume, source, dirs, radii, side):
    pts = source[None, None, :] + dirs[:, None, :] * radii[None, :, None]
    c = np.clip(pts.astype(int), 0, side - 1)
    return volume[c[..., 0], c[..., 1], c[..., 2]]


def analyse(gt, predicted, rays_across=28, core=0.75, ref_fraction=0.3):
    n = gt["side"]
    source, dirs, roll = beam_rays(gt, rays_across, core)
    sid = gt["sid_m"] / gt["voxel_m"]
    r_ref = ref_fraction * sid
    radii = np.arange(r_ref, 1.2 * n, 0.25)                 # quarter-voxel steps, to the box corner
    inside = np.all((source[None, None, :] + dirs[:, None, :] * radii[None, :, None] >= 0) &
                    (source[None, None, :] + dirs[:, None, :] * radii[None, :, None] < n), axis=2)
    occ = march(gt["density"], source, dirs, radii, n) > 0
    occ &= inside
    fg = march(gt["flux"], source, dirs, radii, n)
    fn = march(predicted, source, dirs, radii, n)
    r2 = radii[None, :] ** 2

    def transmission(f):
        ref = f[:, 0:1] * r_ref ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where((f > 0) & (ref > 0), f * r2 / np.where(ref > 0, ref, 1), np.nan))

    Tg, Tn = transmission(fg), transmission(fn)
    crossed = occ.any(1)
    rows = []
    for i in np.where(crossed)[0]:
        where = np.where(occ[i])[0]
        enter, leave = where[0], where[-1]
        if enter < 2 or leave + 2 >= len(radii) or not inside[i, leave + 2]:
            continue
        before, after = np.nanmedian(Tg[i, max(0, enter - 4):enter]), np.nanmedian(Tg[i, leave + 1:leave + 5])
        bn, an = np.nanmedian(Tn[i, max(0, enter - 4):enter]), np.nanmedian(Tn[i, leave + 1:leave + 5])
        if not np.isfinite([before, after, bn, an]).all():
            continue
        step_gt, step_nn = before - after, bn - an
        chord_cm = occ[i].sum() * 0.25 * gt["voxel_m"] * 100

        def onset(T, base):
            """Where the ray has lost half a decade, in cm past the phantom's surface.

            An ABSOLUTE threshold rather than a share of each field's own step: a model whose step
            is half as deep reaches the half-way point of ITS step sooner, which would read as an
            earlier onset for what is really a shallower one.
            """
            seg = T[i, enter:leave + 1]
            hit = np.where(np.isfinite(seg) & (seg <= base - 0.5))[0]
            return np.nan if not len(hit) else hit[0] * 0.25 * gt["voxel_m"] * 100

        rows.append((step_gt, step_nn, chord_cm,
                     onset(Tg, before), onset(Tn, bn)))
    # The null: air, on every ray, between the reference and whatever it first meets.
    null = []
    for i in range(len(dirs)):
        stop = np.where(occ[i])[0]
        stop = stop[0] if len(stop) else np.where(inside[i])[0][-1] if inside[i].any() else 0
        if stop < 6:
            continue
        d = Tn[i, 2:stop] - Tg[i, 2:stop]
        null.extend(d[np.isfinite(d)])
    return np.array(rows), np.array(null), roll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", help="the .rf3m to test")
    ap.add_argument("dataset", help="dataset directory holding fields/*.rf3")
    ap.add_argument("--fields", type=int, default=24, help="how many fields to measure")
    ap.add_argument("--stride", type=int, default=7, help="step through the field list")
    ap.add_argument("--rays", type=int, default=28, help="rays across the collimated rectangle")
    ap.add_argument("--core", type=float, default=0.75,
                    help="share of the rectangle analysed; the rim is penumbra")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset, "fields", "**", "*.rf3"), recursive=True))
    if not files:
        raise SystemExit(f"no fields under {args.dataset}")
    model = None
    print(f"{'field':>6} {'sid':>5} {'rect cm':>10} {'roll':>5} {'rays':>5} "
          f"{'null p05..p95':>15} {'step sim':>9} {'step nn':>8} {'missing':>8} {'onset cm':>9}")
    missing_all, onset_all, step_all = [], [], []
    for path in files[:args.fields * args.stride:args.stride]:
        gt = read_field(path)
        if model is None:
            model = Model(args.package, gt["side"])
        rows, null, roll = analyse(gt, model(gt), args.rays, args.core)
        if len(rows) < 20 or len(null) < 40:
            # Usually because every ray leaves the field box while still inside the body: the
            # phantom is longer than the 1.28 m box, so a beam along it never sees an exit to
            # measure the step against.
            print(f"{os.path.basename(path)[:6]:>6} {gt['sid_m']:5.2f} "
                  f"{gt['rect_m'][0]*100:4.1f}x{gt['rect_m'][1]*100:<4.1f} {roll:5.0f} {len(rows):>5}"
                  f"   no ray both enters and leaves the body inside the box")
            continue
        lo, hi = np.percentile(null, [5, 95])
        deep = rows[:, 0] > 0.3                        # the simulation actually shows a step here
        missing = rows[deep, 0] - rows[deep, 1]
        onset = rows[deep, 4] - rows[deep, 3]
        missing_all.extend(missing)
        step_all.extend(rows[deep, 0])
        onset_all.extend(onset[np.isfinite(onset)])
        print(f"{os.path.basename(path)[:6]:>6} {gt['sid_m']:5.2f} "
              f"{gt['rect_m'][0]*100:4.1f}x{gt['rect_m'][1]*100:<4.1f} {roll:5.0f} {deep.sum():>5} "
              f"{lo:+6.2f}..{hi:+6.2f} {np.median(rows[deep,0]):9.2f} "
              f"{np.median(rows[deep,1]):8.2f} {np.median(missing):+8.2f} "
              f"{np.nanmedian(onset):+9.1f}")

    if missing_all:
        m = np.array(missing_all)
        print(f"\n{len(m)} rays cross the phantom with a step the simulation resolves "
              f"(median {np.median(step_all):.2f} decades).")
        ratio = 10 ** np.median(m)
        print(f"missing attenuation: median {np.median(m):+.2f} dec "
              f"({np.percentile(m,25):+.2f} .. {np.percentile(m,75):+.2f} interquartile) — behind the "
              f"body the model holds {ratio:.2f}x the flux the simulation does "
              f"({'too bright' if ratio > 1 else 'too dark'}).")
        print(f"of those rays, {100*(m > 0.3).mean():.0f} % miss more than a third of a decade "
              f"and {100*(m < -0.3).mean():.0f} % overshoot it by as much in the other direction.")
        if onset_all:
            o = np.array(onset_all)
            print(f"onset offset: median {np.median(o):+.1f} cm "
                  f"({np.percentile(o,25):+.1f} .. {np.percentile(o,75):+.1f}) — positive means the "
                  f"model's drop begins deeper into the body than the simulation's.")


if __name__ == "__main__":
    main()

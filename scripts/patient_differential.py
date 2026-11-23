#!/usr/bin/env python
"""Move the patient out of the beam and subtract: does the flux that disappears sit where the body is?

The measurement in `beam_attenuation_steps.py` needs a reference on each ray, which forces a choice
about where that reference sits and how far it can be trusted. This one needs none. The model is
queried TWICE with an identical beam -- once with the patient where the field puts it, once with the
patient carried clear of the beam -- and the two fields are divided:

    D(v) = log10( flux_clear(v) / flux_blocked(v) )

Everything the two queries share divides out exactly: the inverse square, the beam profile, the
penumbra, the spectrum, and every systematic error the model makes that does not depend on the
patient. What survives is the model's own answer to "what does this body remove from this beam",
which is the one thing the translation input exists to carry.

The geometry says where that should be: a voxel is SHADOWED when it lies inside the collimated cone
and its ray back to the focal spot crosses the phantom. So the question becomes a detection problem
with no free scale at all --

  AUC     how well D ranks shadowed voxels above lit ones. 0.5 is a field that ignores the patient,
          1.0 one whose response falls exactly on the geometric shadow.
  Dice    overlap of {D > 0.3 decades} with the geometric shadow: the same question as a region
          rather than a ranking, and sensitive to a shadow of the right shape but the wrong size.
  depth   median D inside the shadow. AUC and Dice are blind to it -- a model that removes 0.05
          decades in exactly the right voxels scores perfectly on both -- so it is reported beside
          them, against the ~1.9 decades the Monte-Carlo simulation actually removes.

The phantom comes from the dataset's own geometry channel in the PATIENT frame
(`scripts/make_patient_occupancy.py`), shifted by the translation, which is how the field's own
geometry is built and what makes "shadowed" a statement about the data rather than about a marker.
"""
import argparse
import glob
import os

import numpy as np
import torch
from radfiled3d.store import FieldStore
from radfiled3d.nn import deploy, inference

TRANSLATION_SPAN = {"x": (-0.25, 0.25), "y": (-0.5, 0.5)}


def read_beam(path):
    """The beam configuration of one field. Its flux is not needed; only the beam is borrowed."""
    meta = FieldStore.load_metadata(path)             # held: the voxel views point into it
    tube = meta.get_header().simulation.tube
    origin = np.array([tube.radiation_origin.x, tube.radiation_origin.y, tube.radiation_origin.z])
    rect = meta.get_dynamic_metadata("xray_tube_field_rect_dimensions_m").get_data()
    tr = meta.get_dynamic_metadata("patient_translation").get_data()
    return dict(origin=origin,
                direction=np.array([tube.radiation_direction.x, tube.radiation_direction.y,
                                    tube.radiation_direction.z]),
                sid_m=float(np.linalg.norm(origin)),
                rect_m=np.array([rect.x, rect.y]),
                shape=float(meta.get_dynamic_metadata("xray_field_shape").get_data()),
                translation_m=np.array([tr.x, tr.y, tr.z]),
                spectrum=np.array(meta.get_dynamic_metadata("tube_spectrum").get_data(), np.float32))


class Model:
    def __init__(self, package, side, device=0):
        self.session = inference.Session(deploy.ModelStore.load(package), backend="cuda", device=device)
        self.session.set_voxel_grid([side] * 3)
        tensors = deploy.ModelStore.load_metadata(package).tensors
        self.shapes = {t.name: tuple(t.shape) for t in tensors if t.role == "input"}
        sid = [t for t in tensors if t.name == "source_distance"][0].range
        self.sid_lo, self.sid_hi = float(sid[0]), float(sid[1])
        idx = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1) / side
        self.positions = torch.from_numpy(
            np.transpose(idx, (2, 1, 0, 3)).reshape(-1, 3).astype(np.float32).copy()).cuda()
        self.side = side
        self._bound = {}
        # Allocated once: the session keeps a raw pointer into whatever is bound.
        self.flux = torch.zeros(side ** 3, device="cuda")
        self.spectrum = torch.zeros(side ** 3 * 32, device="cuda")

    def _bind(self, name, value):
        t = torch.from_numpy(np.asarray(value, np.float32).reshape(self.shapes[name])).cuda()
        self._bound[name] = t
        self.session.bind_input(name, t)

    def __call__(self, beam, translation_m, field_m):
        self.session.bind_input("position", self.positions)
        self._bind("beam_direction", beam["direction"])
        self._bind("source_distance", [(beam["sid_m"] - self.sid_lo) / (self.sid_hi - self.sid_lo)])
        self._bind("beam_collimation", beam["rect_m"] / field_m)
        self._bind("beam_collimation_kind", [beam["shape"]])
        self._bind("tube_spectrum", beam["spectrum"] / max(beam["spectrum"].sum(), 1e-12))
        self._bind("patient_translation",
                   [(translation_m[0] - TRANSLATION_SPAN["x"][0]) /
                    (TRANSLATION_SPAN["x"][1] - TRANSLATION_SPAN["x"][0]),
                    (translation_m[1] - TRANSLATION_SPAN["y"][0]) /
                    (TRANSLATION_SPAN["y"][1] - TRANSLATION_SPAN["y"][0]), 0.0])
        self.session.bind_output("flux", self.flux)
        self.session.bind_output("spectrum", self.spectrum)
        self.session.infer()
        torch.cuda.synchronize()
        return self.flux.cpu().numpy().astype(np.float64).reshape(
            self.side, self.side, self.side).transpose(2, 1, 0)


class Geometry:
    """The phantom in its own frame, and the shadow it throws for a given beam and translation."""

    def __init__(self, occupancy_npz, side, voxel_m):
        blob = np.load(occupancy_npz)
        occ = blob["occupancy"] > 0.5
        if occ.shape[0] != side:
            raise SystemExit(f"occupancy is {occ.shape[0]}^3, the query grid is {side}^3")
        # Exported at a pose that is not exactly zero; fold that out so a translation of zero means
        # the phantom sits where the dataset's zero-translation field puts it.
        self.occ = occ
        self.base = np.asarray(blob["source_translation_m"], float)
        self.side = side
        self.voxel_m = voxel_m
        self.idx = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1)

    def occupancy_at(self, translation_m):
        shift = np.rint((np.asarray(translation_m[:3], float) - self.base) / self.voxel_m).astype(int)
        out = np.zeros_like(self.occ)
        src = [slice(max(0, -s), self.side - max(0, s)) for s in shift]
        dst = [slice(max(0, s), self.side - max(0, -s)) for s in shift]
        out[tuple(dst)] = self.occ[tuple(src)]
        return out

    def cone(self, beam, core):
        source = np.full(3, self.side / 2.0) + beam["origin"] / self.voxel_m
        v = self.idx - source
        axial = v @ beam["direction"]
        off = np.linalg.norm(v - axial[..., None] * beam["direction"], axis=-1)
        half = 0.5 * min(beam["rect_m"]) / self.voxel_m * core / (beam["sid_m"] / self.voxel_m)
        return (axial > 1e-3) & (off <= half * axial), source, axial

    def shadow(self, beam, translation_m, core=0.85, steps=192):
        """Inside the cone, and with material between the focal spot and the voxel."""
        cone, source, _ = self.cone(beam, core)
        occ = self.occupancy_at(translation_m)
        pts = self.idx[cone]
        t = np.linspace(0.0, 1.0, steps)[None, :, None]
        seg = source[None, None, :] + (pts - source)[:, None, :] * t
        c = np.clip(seg.astype(int), 0, self.side - 1)
        crossed = occ[c[..., 0], c[..., 1], c[..., 2]].any(1)
        out = np.zeros_like(cone)
        out[cone] = crossed
        return cone, out


def clear_translation(geo, beam, blocked_at, core, steps=64):
    """A patient pose that leaves the beam unobstructed, and the share of the cone it still shadows.

    The objective is the share of the cone left IN SHADOW, not the number of body voxels that sit
    inside it. Those are very different questions: the cone is narrow at the focal spot, so a dozen
    voxels of body near the apex shadow the whole volume downstream of them, while the same dozen
    out at the far face shadow almost nothing. Picking the pose by voxel count chooses exactly the
    former, and the "clear" field then comes back DARKER than the blocked one -- an inverted
    measurement that reads as a model pathology and is not one.
    """
    cone, source, _ = geo.cone(beam, core)
    pts = geo.idx[cone]
    t = np.linspace(0.0, 1.0, steps)[None, :, None]
    seg = source[None, None, :] + (pts - source)[:, None, :] * t     # the beam does not move
    best = None
    for tx in np.linspace(*TRANSLATION_SPAN["x"], 9):
        for ty in np.linspace(*TRANSLATION_SPAN["y"], 17):
            if np.hypot(tx - blocked_at[0], ty - blocked_at[1]) < 0.15:
                continue
            # Shifting the phantom and shifting the ray samples the other way are the same thing,
            # and the second keeps one occupancy array instead of a hundred.
            shift = np.rint((np.array([tx, ty, 0.0]) - geo.base) / geo.voxel_m).astype(int)
            q = seg - shift
            inside = np.all((q >= 0) & (q < geo.side), axis=2)
            c = np.clip(q.astype(int), 0, geo.side - 1)
            frac = float((geo.occ[c[..., 0], c[..., 1], c[..., 2]] & inside).any(1).mean())
            if best is None or frac < best[0]:
                best = (frac, np.array([tx, ty, 0.0]))
            if frac == 0.0:
                return best[1], 0.0
    return best[1], best[0]


def roc_auc(score, label):
    """Rank-based, so no threshold is chosen anywhere in this measurement."""
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return np.nan
    return (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("dataset", help="dataset directory, for the beams to borrow")
    ap.add_argument("occupancy", help="patient occupancy npz from make_patient_occupancy.py --grid 64")
    ap.add_argument("--fields", type=int, default=16)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--core", type=float, default=0.85)
    ap.add_argument("--dice-threshold", type=float, default=0.3,
                    help="decades of removed flux that count as shadow")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset, "fields", "**", "*.rf3"), recursive=True))
    if not files:
        raise SystemExit(f"no fields under {args.dataset}")
    field_m = args.side * 0.02
    geo = Geometry(args.occupancy, args.side, field_m / args.side)
    model = Model(args.package, args.side)

    print(f"{'field':>6} {'SID':>5} {'collimated':>11} {'cone':>6} {'shadow':>7} {'clear':>6} "
          f"{'left%':>6} {'AUC':>6} {'Dice':>6} {'depth':>7} {'elsewhere':>10}")
    auc_all, dice_all, depth_all, lit_all = [], [], [], []
    for path in files[:args.fields * args.stride:args.stride]:
        beam = read_beam(path)
        cone, shadow = geo.shadow(beam, beam["translation_m"], args.core)
        # The classes are lopsided by geometry, not by choice: the focal spot sits inside the field
        # box and the cone's volume grows as the cube of the distance, so most in-beam voxels lie
        # downstream of the body. A few dozen unshadowed ones are enough to rank against.
        if shadow.sum() < 200 or (cone & ~shadow).sum() < 40:
            print(f"{os.path.basename(path)[:6]:>6} {beam['sid_m']:5.2f} "
                  f"{beam['rect_m'][0]*100:4.1f}x{beam['rect_m'][1]*100:<4.1f} {int(cone.sum()):>6} "
                  f"{int(shadow.sum()):>7}   the body does not divide this beam")
            continue
        clear, still_shadowed = clear_translation(geo, beam, beam["translation_m"], args.core)
        if still_shadowed > 0.10:
            print(f"{os.path.basename(path)[:6]:>6} {beam['sid_m']:5.2f} "
                  f"{beam['rect_m'][0]*100:4.1f}x{beam['rect_m'][1]*100:<4.1f} {int(cone.sum()):>6} "
                  f"{int(shadow.sum()):>7}   no pose clears this beam "
                  f"({100*still_shadowed:.0f} % of the cone still shadowed)")
            continue
        blocked = model(beam, beam["translation_m"], field_m)
        lit = model(beam, clear, field_m)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.log10(np.where((blocked > 0) & (lit > 0), lit / np.maximum(blocked, 1e-30), np.nan))
        use = cone & np.isfinite(D)
        score, label = D[use], shadow[use]
        auc = roc_auc(score, label)
        hit = D[use] > args.dice_threshold
        dice = 2 * (hit & label).sum() / max(hit.sum() + label.sum(), 1)
        depth = np.median(score[label])
        lit_med = np.median(score[~label])
        auc_all.append(auc); dice_all.append(dice); depth_all.append(depth); lit_all.append(lit_med)
        print(f"{os.path.basename(path)[:6]:>6} {beam['sid_m']:5.2f} "
              f"{beam['rect_m'][0]*100:4.1f}x{beam['rect_m'][1]*100:<4.1f} {int(cone.sum()):>6} "
              f"{int(shadow.sum()):>7} {int((cone & ~shadow).sum()):>6} {100*still_shadowed:5.1f} "
              f"{auc:6.3f} {dice:6.3f} {depth:+7.2f} {lit_med:+10.2f}")

    if auc_all:
        a, d, p, l = map(np.array, (auc_all, dice_all, depth_all, lit_all))
        print(f"\n{len(a)} fields.  AUC {np.median(a):.3f} ({np.percentile(a,25):.3f}"
              f"..{np.percentile(a,75):.3f})   Dice {np.median(d):.3f}"
              f" at >{args.dice_threshold} dec removed")
        print(f"removed flux: {np.median(p):+.2f} dec inside the geometric shadow, "
              f"{np.median(l):+.2f} dec outside it, both inside the beam.")
        print("AUC 0.5 is a field that ignores the patient; 1.0 one whose response falls exactly on "
              "the geometric shadow. Depth is what neither AUC nor Dice can see.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the canonical patient occupancy PhysPBRFNet carries (conditioning_params.physics.
patient_occupancy_file): the geometry channel of the dataset field whose patient translation is
closest to zero, shifted back by that translation into the patient frame, max-pooled to a coarse
grid over the field box. Output npz: occupancy (M, M, M) float in [0, 1], field_size_m, grid,
source_field, source_translation_m."""
import argparse
import glob
import os
import numpy as np
import torch
from scipy import ndimage
from radfiled3d.store import FieldStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset directory (with fields/*.rf3)")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--scan", type=int, default=1000, help="fields to scan for the smallest translation")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dataset, "fields", "*.rf3")))[:args.scan]
    best, best_t = None, None
    for f in files:
        md = FieldStore.load_metadata(f)
        if "patient_translation" not in md.get_dynamic_metadata_keys():
            continue
        v = md.get_dynamic_metadata("patient_translation").get_data()
        t = np.array([v.x, v.y, v.z])
        if best is None or np.linalg.norm(t) < np.linalg.norm(best_t):
            best, best_t = f, t
    if best is None:
        raise SystemExit("no field with a patient_translation entry")
    field = FieldStore.load(best)
    occ = field.get_channel("geometry").get_layer_as_ndarray("density")
    occ = (occ[..., 0] if occ.ndim == 4 else occ) > 0
    n = occ.shape[0]
    field_size = float(field.get_field_dimensions().x)
    vx = field_size / n
    # world occupancy → patient frame: shift by −t (voxels), nearest neighbour
    shifted = ndimage.shift(occ.astype(np.float32), shift=-best_t / vx, order=0, mode="constant", cval=0.0) > 0.5
    coarse = torch.nn.functional.adaptive_max_pool3d(torch.from_numpy(shifted.astype(np.float32))[None, None], args.grid)[0, 0].numpy()
    np.savez(args.out, occupancy=coarse, field_size_m=field_size, grid=args.grid, source_field=os.path.basename(best), source_translation_m=best_t)
    print(f"{best}: translation {best_t.round(4).tolist()} m, occupied {int(occ.sum())} voxels at {n}^3 → {int((coarse > 0).sum())} at {args.grid}^3; wrote {args.out}")


if __name__ == "__main__":
    main()

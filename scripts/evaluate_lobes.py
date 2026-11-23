#!/usr/bin/env python
"""Score a TDPBRFNet (or, for flux and spectrum only, a TPBRFNet) on held-out fields and write an
HTML report of WHERE it errs.

For flux, spectrum and direction alike: the ground truth, the prediction, and a per-voxel error map
on the same slice, so an error is something you can point at rather than a single number.

    flux       log10(pred / truth), signed — the colour says too bright or too dark, and a decade is
               a decade whether the voxel holds 1e-8 or 1
    spectrum   1 - histogram intersection of the two normalised spectra, in [0, 1]
    direction  1 - directional_similarity of the two vMF mixtures, in [0, 1], plus the angle between
               their heaviest lobes

Everything is compared in PHYSICAL space (the model's normalizer inverted), as the training metrics
are. Voxels a quantity is undefined in are left out of that quantity's map and aggregate: a voxel
with no flux has no spectrum to compare, and one with no lobes has no direction. A model without a
directional head (TPBRFNet) is scored on flux and spectrum with the very same code, and its direction
quantities are NaN, so the two models' reports compare cell for cell.
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(args):
    from radfield3dnn.datasets import construct_datamodule
    from radfield3dnn.models import ModelConstructor
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]
    model_cfg = json.load(open(cfg["training"]["model_config"]))
    model_cls = ModelConstructor.create_model_from_dict(model_cfg)
    # Score EXACTLY the run's held-out fields, from its own record. The split goes into the
    # datamodule's constructor, which builds its datasets there; assigning file lists afterwards
    # changes nothing it iterates.
    split = json.load(open(args.split))
    overlap = set(split["test"]) & set(split["train"])
    assert not overlap, f"split.json is corrupt: {len(overlap)} fields are in both train and test"
    relative = lambda files: [os.path.relpath(f, ds["path"]) if os.path.isabs(f) else f for f in files]
    dm = construct_datamodule(
        dataset_path=ds["path"], batch_size=1, num_workers=0, use_geometry=False,
        use_beam_parameters=ds.get("use_beam_parameters", True), use_translation=ds.get("use_translation", True),
        dataset_definition=ds.get("definition_file"), scan_translation_metadata=False, use_lobes=model_cls.learns_lobes,
        translation_output_range=tuple(ds.get("translation_output_range", (0.0, 1.0))), prefetch_to_device=False,
        split={k: relative(split[k]) for k in ("train", "val", "test")})
    model = model_cls()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.max_inner_batch_size = int(cfg["training"].get("max_inner_batch_size", 32768))
    # a model trained with voxel_supersampling learned each voxel's cell mean: score it on that
    model.set_eval_cell_samples(int(getattr(args, "cell_samples", 1)))
    return dm, model.cuda().eval(), cfg


@torch.no_grad()
def full_lobes(model, x, shape, chunk=65536):
    """The lobe head over EVERY voxel of one field, chunked."""
    voxels = int(np.prod(shape))
    weight, mean, kappa = [], [], []
    for start in range(0, voxels, chunk):
        idx = torch.arange(start, min(start + chunk, voxels), device="cuda").unsqueeze(0)
        w, m, k = model.predict_lobes(x, idx, shape)
        weight.append(w[0]); mean.append(m[0]); kappa.append(k[0])
    return torch.cat(weight), torch.cat(mean), torch.cat(kappa)


@torch.no_grad()
def beam_regions(geometry, batch, shape):
    """Per-voxel masks from the analytic beam: ``exit`` is inside the collimated rectangle and past
    the isocentre, where a patient in the beam leaves its shadow; ``entry`` is inside it and before."""
    axes = [torch.linspace(0.0, 1.0, n, device="cuda") for n in shape]
    position = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    x = batch.input
    ray = geometry(position, x.direction[:1].cuda(), x.origin[:1].cuda(), x.beam_shape_parameters[:1].cuda())
    depth, l1, l2 = ray.beam_frame.unbind(-1)
    inside = (l1.abs() < 1.0) & (l2.abs() < 1.0)
    return {"entry": (inside & (depth < 0)).cpu(), "exit": (inside & (depth > 0)).cpu()}


ROI_FLOOR = 1e-6   # of the field's peak flux: the training floor cut (augmentations.mc_floor_cut.scatter_lo)


def score_field(model, batch, max_energy_keV, geometry, airkerma=None):
    from radfield3dnn.rfhelpers import InferenceHelper
    from radfield3dnn.losses.vmf import directional_similarity, dominant_angle_error_deg
    shape = tuple(batch.ground_truth.flux.shape[-3:])
    V = int(np.prod(shape))
    gt, pred = InferenceHelper.generate_gt_and_pred_for_validation(
        batch, model, torch.tensor(shape), spectra_bins=model.out_spectra_dim)
    gt_ch = gt.scatter_field if hasattr(gt, "scatter_field") else gt
    pr_ch = pred.scatter_field if hasattr(pred, "scatter_field") else pred
    f_gt = gt_ch.flux.reshape(-1).float().cpu()
    f_pr = pr_ch.flux.reshape(-1).float().cpu()
    s_gt = gt_ch.spectrum.reshape(gt_ch.spectrum.shape[-4], -1).T.float().cpu()      # (V, bins)
    s_pr = pr_ch.spectrum.reshape(pr_ch.spectrum.shape[-4], -1).T.float().cpu()

    # flux: signed decades where both are positive
    lit = (f_gt > 0) & (f_pr > 0)
    flux_err = torch.full((V,), float("nan"))
    flux_err[lit] = torch.log10(f_pr[lit] / f_gt[lit])

    # spectrum: 1 - histogram intersection of the normalised spectra
    def unit(s):
        tot = s.clamp_min(0).sum(-1, keepdim=True)
        return torch.where(tot > 0, s.clamp_min(0) / tot.clamp_min(1e-30), torch.zeros_like(s)), tot.squeeze(-1) > 0
    p_gt, has_gt = unit(s_gt)
    p_pr, has_pr = unit(s_pr)
    spec_ok = has_gt & has_pr
    spec_err = torch.full((V,), float("nan"))
    spec_err[spec_ok] = 1.0 - torch.minimum(p_gt[spec_ok], p_pr[spec_ok]).sum(-1)
    centres = (torch.arange(s_gt.shape[-1]) + 0.5) * (max_energy_keV / s_gt.shape[-1])
    mean_e = lambda p: (p * centres).sum(-1)

    # direction: only a model with a lobe head has one
    ang_err = torch.full((V,), float("nan"))
    dir_err = torch.full((V,), float("nan"))
    if model.learns_lobes:
        from radfield3dnn.models.nerf_translation_direction import TDPBRFNet
        from radfield3dnn.datasets.lobes import uniform_voxels
        tw, tm, tk = TDPBRFNet.flat_lobes(batch.lobes.cuda())
        target = (tw[0], tm[0], tk[0])
        prediction = full_lobes(model, batch.input, shape)
        # every voxel has a target (an uncrossed one is uniform); a uniform one has no heaviest lobe
        directed = ~uniform_voxels(tw[0], tk[0])
        sim = directional_similarity(prediction, target)
        ang = dominant_angle_error_deg(prediction, target)
        dir_err = (1.0 - sim).cpu()
        ang_err[directed.cpu()] = ang[directed].cpu()
        dir_rgb = lambda w, m: torch.gather(m, -2, w.argmax(-1, keepdim=True).unsqueeze(-1).expand(*w.shape[:-1], 1, 3)).squeeze(-2).abs().cpu()
        dir_gt, dir_pred = dir_rgb(*target[:2]), dir_rgb(*prediction[:2])
    else:
        directed = None
        dir_gt = dir_pred = torch.full((V, 3), float("nan"))

    maps = dict(
        flux_gt=f_gt, flux_pred=f_pr, flux_err=flux_err,
        energy_gt=torch.where(has_gt, mean_e(p_gt), torch.tensor(float("nan"))),
        energy_pred=torch.where(has_pr, mean_e(p_pr), torch.tensor(float("nan"))),
        spec_err=spec_err,
        dir_gt=dir_gt, dir_pred=dir_pred,
        dir_err=dir_err, ang_err=ang_err,
    )
    finite = lambda t: t[torch.isfinite(t)]
    # a statistic of nothing is NaN, which is what a model without a directional head reports
    stat = lambda t, fn: float(fn(t)) if t.numel() else float("nan")
    abs_dec = finite(flux_err).abs()
    # the entry/exit regions need the dataset definition's beam; without one the region keys are absent
    regions = beam_regions(geometry, batch, shape) if geometry is not None else {}
    region_flux = {}
    for name, mask in regions.items():
        e = finite(flux_err[mask])
        region_flux[f"flux_{name}_beam_signed_decades"] = float(e.median()) if e.numel() else float("nan")
        region_flux[f"flux_{name}_beam_abs_decades"] = float(e.abs().median()) if e.numel() else float("nan")
    agg = dict(
        flux_median_abs_decades=float(abs_dec.median()),
        flux_within_2x=float((abs_dec <= np.log10(2)).float().mean()),
        flux_within_10pct=float((abs_dec <= np.log10(1.1)).float().mean()),
        spectrum_intersection=float(1.0 - finite(spec_err).mean()),
        spectrum_mean_energy_abs_keV=float(finite(maps["energy_pred"] - maps["energy_gt"]).abs().median()),
        direction_similarity=stat(finite(dir_err), lambda t: 1.0 - t.mean()),
        direction_similarity_median=stat(finite(dir_err), lambda t: 1.0 - t.median()),
        direction_dominant_angle_median_deg=stat(finite(ang_err), torch.median),
        direction_dominant_within_15deg=stat(finite(ang_err), lambda t: (t <= 15.0).float().mean()),
        uniform_fraction=float((~directed).float().mean()) if directed is not None else float("nan"),
        **region_flux,
    )
    # the flux loss's regions (SMAPERegionBalancedLoss: beam >= 5e-2, ring 5e-3..5e-2, bulk below, of
    # the field's peak), each with the loss's own per-voxel term (|log10 ratio|, clamped at 6 decades)
    # and the accuracy's (1 - SMAPE/2, which saturates beyond a factor of ten)
    rel = f_gt / f_gt.max()
    smape_acc = 1.0 - (f_pr - f_gt).abs() / (f_pr.abs() + f_gt.abs()).clamp_min(5e-9)
    for name, m in (("beam", rel >= 5e-2), ("ring", (rel >= 5e-3) & (rel < 5e-2)), ("bulk", rel < 5e-3)):
        e = finite(flux_err[m]).abs().clamp(max=6.0)
        agg[f"flux_{name}_fraction"] = float(m.float().mean())
        agg[f"flux_{name}_mean_abs_decades"] = float(e.mean()) if e.numel() else float("nan")
        agg[f"flux_{name}_smape_accuracy"] = float(smape_acc[m].mean()) if m.any() else float("nan")
    # air kerma from fluence, spectrum and mu_tr; scored where the metric's scatter region is: the
    # voxels between 0.5 % and 5 % of the field's peak FLUX
    if airkerma is not None:
        ak_gt = airkerma.forward(gt_ch.spectrum, gt_ch.flux).reshape(-1).float().cpu()
        ak_pr = airkerma.forward(pr_ch.spectrum, pr_ch.flux).reshape(-1).float().cpu()
        ak_acc = 1.0 - (ak_pr - ak_gt).abs() / (ak_pr.abs() + ak_gt.abs()).clamp_min(5e-9)
        scatter = (rel >= 5e-3) & (rel < 5e-2)
        agg["airkerma_scatter_smape_accuracy"] = float(ak_acc[scatter].mean()) if scatter.any() else float("nan")
        agg["airkerma_global_smape_accuracy"] = float(ak_acc.mean())
    # the same quantities over the trained region only: voxels at or above ROI_FLOOR of the peak
    roi = f_gt >= ROI_FLOOR * f_gt.max()
    abs_roi = finite(flux_err[roi]).abs()
    agg.update(
        roi_fraction=float(roi.float().mean()),
        flux_median_abs_decades_roi=float(abs_roi.median()),
        flux_within_2x_roi=float((abs_roi <= np.log10(2)).float().mean()),
        spectrum_intersection_roi=float(1.0 - finite(spec_err[roi]).mean()),
        direction_similarity_roi=stat(finite(dir_err[roi]), lambda t: 1.0 - t.mean()),
        direction_dominant_angle_median_deg_roi=stat(finite(ang_err[roi]), torch.median),
    )
    return {k: v.reshape(*shape, *v.shape[1:]) for k, v in maps.items()}, agg


def figure(maps, axis, index, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    def sl(t):
        a = t.numpy()
        return np.take(a, index, axis=axis)

    rows = [
        ("flux", "flux_gt", "flux_pred", "flux_err", "log10 flux", "log10(pred / truth)"),
        ("spectrum", "energy_gt", "energy_pred", "spec_err", "mean energy (keV)", "1 - spectral intersection"),
        ("direction", "dir_gt", "dir_pred", "dir_err", "dominant direction (|xyz| as RGB)", "1 - directional similarity"),
    ]
    if not torch.isfinite(maps["dir_err"]).any():
        rows = rows[:2]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.6, 4.0 * len(rows)))
    for r, (name, g, p, e, vlabel, elabel) in enumerate(rows):
        a_g, a_p, a_e = sl(maps[g]), sl(maps[p]), sl(maps[e])
        if name == "flux":
            a_g, a_p = np.log10(np.maximum(a_g, 1e-12)), np.log10(np.maximum(a_p, 1e-12))
            lo, hi = np.nanpercentile(np.concatenate([a_g.ravel(), a_p.ravel()]), [1, 99.9])
            kw = dict(cmap="inferno", vmin=lo, vmax=hi)
        elif name == "spectrum":
            lo, hi = np.nanpercentile(np.concatenate([a_g.ravel(), a_p.ravel()]), [1, 99])
            kw = dict(cmap="viridis", vmin=lo, vmax=hi)
        else:
            kw = {}
        for c, (arr, lab) in enumerate(((a_g, "truth"), (a_p, "prediction"))):
            ax = axes[r, c]
            im = ax.imshow(np.clip(arr, 0, 1) if name == "direction" else arr, origin="lower", **kw)
            ax.set_title(f"{name} — {lab}", fontsize=10, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            if name != "direction":
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(vlabel, fontsize=8)
        ax = axes[r, 2]
        if name == "flux":
            im = ax.imshow(a_e, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5))
        else:
            im = ax.imshow(a_e, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name} — error per voxel", fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(elabel, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True,
                    help="the run's split.json. REQUIRED: the split cannot be re-derived from the seed, "
                         "and a re-derived one overlaps training and flatters the model")
    ap.add_argument("--out", required=True, help="HTML file to write")
    ap.add_argument("--render", type=int, default=4, help="fields to render (all test fields are scored)")
    ap.add_argument("--json", help="also write the aggregate metrics here")
    ap.add_argument("--mu_tr_file", help="mass energy-absorption table; adds the air-kerma scatter accuracy")
    ap.add_argument("--cell_samples", type=int, default=1,
                    help="flux and spectrum per voxel as the mean over this many points of its cell (a cube: 8 = 2x2x2), "
                         "as a model trained with voxel_supersampling was fit; 1 queries the node grid. The lobes are "
                         "always taken at the node grid, where the direction loss trains them")
    args = ap.parse_args()

    dm, model, cfg = load(args)
    definition = json.load(open(cfg["dataset"]["definition_file"]))
    max_energy_keV = float(definition["Metaparameters"]["MaxEnergy"]) / 1000.0
    from radfield3dnn.models.layers.beam_geometry import BeamPatientGeometry
    geometry = BeamPatientGeometry.from_dataset_definition(definition, position_frame_extent=1.0,
                                                           with_translation=False).cuda()
    test_files = dm._dataloader_builder.test_files
    assert [str(f) for f in dm.test_dataset.file_paths] == [str(f) for f in test_files], \
        "the test dataset does not iterate the split's fields in order; per-field names would be wrong"

    airkerma = None
    if args.mu_tr_file:
        from radfield3dnn.preprocessing.airkerma import Airkerma
        airkerma = Airkerma(Airkerma.load_mu_tr_table(args.mu_tr_file), model.out_spectra_dim,
                            float(definition["Metaparameters"]["MaxEnergy"]))
    per_field, renders = [], []
    for i, batch in enumerate(dm.test_dataloader()):
        batch = dm.on_after_batch_transfer(dm.transfer_batch_to_device(batch, torch.device("cuda"), 0), 0)
        maps, agg = score_field(model, batch, max_energy_keV, geometry, airkerma)
        per_field.append({"field": str(test_files[i]), **agg})
        print(f"field {i:>3}: flux |dec| {agg['flux_median_abs_decades']:.3f}  "
              f"spectrum {agg['spectrum_intersection']:.3f}  direction {agg['direction_similarity']:.3f}  "
              f"angle {agg['direction_dominant_angle_median_deg']:.1f} deg")
        if i < args.render:
            n = maps["flux_gt"].shape[0]
            for axis, axname in ((2, "z"), (1, "y")):
                renders.append((f"test field {i} — central {axname} slice", figure(maps, axis, n // 2,
                                f"Test field {i}, central {axname} slice")))

    keys = [k for k in per_field[0] if k != "field"]
    summary = {k: float(np.nanmean([f[k] for f in per_field])) for k in keys}
    if args.json:
        json.dump({"summary": summary, "fields": per_field}, open(args.json, "w"), indent=2)
    write_html(args.out, summary, per_field, renders, args, cfg)
    print(f"\n{len(per_field)} test fields -> {args.out}")
    for k, v in summary.items():
        print(f"  {k:<40s} {v:.4f}")


def write_html(path, summary, per_field, renders, args, cfg):
    rows = "".join(
        f"<tr><td>{i}</td><td>{f['flux_median_abs_decades']:.3f}</td><td>{100*f['flux_within_2x']:.1f}%</td>"
        f"<td>{f['spectrum_intersection']:.3f}</td><td>{f['direction_similarity']:.3f}</td>"
        f"<td>{f['direction_dominant_angle_median_deg']:.1f}°</td></tr>" for i, f in enumerate(per_field))
    imgs = "".join(f"<figure><figcaption>{t}</figcaption><img src='data:image/png;base64,{b}'></figure>"
                   for t, b in renders)
    s = summary
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Lobe Head Evaluation</title>
<style>
:root{{--ink:#15191d;--mute:#6a7480;--rule:#dce0e4;--paper:#f6f7f8;--card:#fff}}
body{{background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:28px 18px 80px}}
h1{{font-size:2rem;margin:0 0 6px}} h2{{margin-top:40px}} .mute{{color:var(--mute)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}}
.grid div{{background:var(--card);padding:14px 16px}} .k{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--mute)}}
.v{{font-size:1.8rem;font-variant-numeric:tabular-nums}} table{{border-collapse:collapse;width:100%;background:var(--card);font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:right}} th:first-child,td:first-child{{text-align:left}}
figure{{margin:24px 0;background:var(--card);border:1px solid var(--rule);padding:12px}} img{{width:100%;height:auto}}
figcaption{{color:var(--mute);font-size:13px;margin-bottom:8px}}
@media (prefers-color-scheme: dark){{:root{{--ink:#e7ebef;--mute:#8d97a2;--rule:#2a3138;--paper:#0f1216;--card:#171c21}}}}
</style></head><body>
<h1>Lobe head evaluation</h1>
<p class="mute">{len(per_field)} held-out fields · checkpoint <code>{os.path.basename(args.checkpoint)}</code> · config <code>{os.path.basename(args.config)}</code></p>
<div class="grid">
<div><div class="k">flux, median error</div><div class="v">{s['flux_median_abs_decades']:.3f}</div><div class="mute">decades</div></div>
<div><div class="k">flux within 2×</div><div class="v">{100*s['flux_within_2x']:.1f}%</div><div class="mute">of lit voxels</div></div>
<div><div class="k">spectrum</div><div class="v">{s['spectrum_intersection']:.3f}</div><div class="mute">histogram intersection</div></div>
<div><div class="k">direction</div><div class="v">{s['direction_similarity']:.3f}</div><div class="mute">vMF similarity, 1 = identical</div></div>
<div><div class="k">dominant lobe</div><div class="v">{s['direction_dominant_angle_median_deg']:.1f}°</div><div class="mute">median angle; {100*s['direction_dominant_within_15deg']:.0f}% within 15°</div></div>
</div>
<h2>Where the errors are</h2>
<p class="mute">Rows: flux, spectrum, direction. Columns: truth, prediction, error per voxel. Flux error is signed
(red too bright, blue too dark); spectrum and direction errors run 0 (agree) to 1. Direction is shown as the heaviest
lobe's axis with |x|,|y|,|z| as R,G,B. Blank voxels are where a quantity is undefined.</p>
{imgs}
<h2>Per field</h2>
<table><tr><th>field</th><th>flux |dec|</th><th>flux ≤2×</th><th>spectrum</th><th>direction</th><th>angle</th></tr>{rows}</table>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()

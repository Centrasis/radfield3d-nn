"""fp16 representability probe for the HDR radiation field.

The pure-Python PBRFNet trains fine in fp32 but, with fp16 weights, the flux
head converges to ~0 air-kerma accuracy while the spectrum head still learns —
i.e. fp16 *trains* but cannot reach a good flux solution. This asks the prior
question the user posed: **can fp16 weights even represent the field at all?**

It trains a field→field **autoencoder** that reuses PBRFNet's flux/spectrum
**decoder-head architecture** (`Linear→SiLU→Linear` flux head, `Linear→SiLU→
Linear` spectrum head + `HistogramNormalize`, `GradientConservingClamping` flux
activation on the log-scale codomain), in both fp32 and fp16, on real voxels.
Autoencoding removes the conditioning→field learning difficulty, so it isolates
representation capacity:

* fp16 reconstruction ≈ fp32  → fp16 *can* represent the HDR field; the training
  failure is optimisation (loss scaling / dynamics), not the architecture.
* fp16 reconstruction ≫ worse → fp16 *weights* cannot hold the HDR flux mapping;
  an alternative HDR-ready architecture is needed.

Run: `python -m radfield3dnn.models.autoencoder <dataset_dir> [n_fields]`
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping
from radfield3dnn.preprocessing.normalizations.logscale import LogScaleNormalizer


class FieldAutoencoder(nn.Module):
    """Autoencoder over (log-flux, spectrum) voxels using PBRFNet's head blocks."""

    def __init__(self, d_model: int = 192, bins: int = 32, flux_offset: float = -4.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1 + bins, d_model), nn.SiLU(True),
            nn.Linear(d_model, d_model), nn.SiLU(True),
        )
        # identical structure to PBRFNet backbone heads
        self.flux_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(True), nn.Linear(d_model, 1),
        )
        self.spectra_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(True), nn.Linear(d_model // 2, bins),
        )
        self.spectra_activation = HistogramNormalize(dim=-1)
        self.flux_activation = GradientConservingClamping(-9.0, 0.0)
        self._flux_offset = flux_offset

    def forward(self, flux: torch.Tensor, spectrum: torch.Tensor):
        h = self.encoder(torch.cat([flux, spectrum], dim=-1))
        f = self.flux_activation(self.flux_decoder(h) + self._flux_offset)
        s = self.spectra_activation(self.spectra_decoder(h))
        return f, s


def _load_voxels(dataset_dir: str, n_fields: int):
    """Return (flux[N,1], spectrum[N,bins]) log-scale-normalised voxels."""
    from RadFiled3D.pytorch.radiationfieldloader import DataLoaderBuilder
    from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset
    builder = DataLoaderBuilder(dataset_dir, train_ratio=1.0, val_ratio=0.0,
                                test_ratio=0.0, dataset_class=RadField3DDataset)
    ds = builder.build_train_dataset()
    norm = LogScaleNormalizer()
    fluxes, specs = [], []
    for i in range(min(n_fields, len(ds))):
        s = ds[i].ground_truth.scatter_field
        flux = norm.forward(s.flux.float()).reshape(-1, 1)              # (V,1) log-scale
        spec = s.spectrum.float()
        spec = spec.reshape(spec.shape[0], -1).transpose(0, 1)         # (V, bins)
        spec = spec / spec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fluxes.append(flux)
        specs.append(spec)
    return torch.cat(fluxes), torch.cat(specs)


def _train(precision: str, flux, spectrum, steps=3000, d_model=192):
    dev = "cuda"
    model = FieldAutoencoder(d_model=d_model, bins=spectrum.shape[1]).to(dev)
    f32_flux, f32_spec = flux.to(dev), spectrum.to(dev)
    if precision == "fp16":
        model = model.half()
        masters = {n: p.detach().float().clone().requires_grad_(True) for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(list(masters.values()), lr=1e-3, betas=(0.9, 0.99), eps=1e-15)
        in_flux, in_spec = f32_flux.half(), f32_spec.half()
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-15)
        in_flux, in_spec = f32_flux, f32_spec
    N = in_flux.shape[0]
    for step in range(steps):
        if precision == "fp16":
            with torch.no_grad():
                pd = dict(model.named_parameters())
                for n, m in masters.items():
                    pd[n].copy_(m.half())
        idx = torch.randint(0, N, (8192,), device=dev)
        pf, ps = model(in_flux[idx], in_spec[idx])
        loss = (pf.float() - f32_flux[idx]).abs().mean() + (ps.float() - f32_spec[idx]).abs().mean()
        opt.zero_grad()
        loss.backward()
        if precision == "fp16":
            pd = dict(model.named_parameters())
            for n, m in masters.items():
                if pd[n].grad is not None:
                    m.grad = pd[n].grad.detach().float()
                    pd[n].grad = None
        opt.step()
    # eval: physical-space flux reconstruction (what air-kerma cares about)
    with torch.no_grad():
        if precision == "fp16":
            pd = dict(model.named_parameters())
            for n, m in masters.items():
                pd[n].copy_(m.half())
        pf, ps = model(in_flux, in_spec)
        pf = pf.float().reshape(-1, 1)
        flux_log_l1 = (pf - f32_flux).abs().mean().item()
        phys_pred = torch.pow(10.0, pf).clamp(0, 1)
        phys_gt = torch.pow(10.0, f32_flux).clamp(0, 1)
        # relative physical-flux error on the high-flux voxels (top 5%)
        thr = torch.quantile(phys_gt, 0.95)
        hi = phys_gt >= thr
        rel = ((phys_pred[hi] - phys_gt[hi]).abs() / (phys_gt[hi] + 1e-6)).mean().item()
        spec_overlap = torch.minimum(ps.float(), f32_spec).sum(-1).mean().item()
    return flux_log_l1, rel, spec_overlap


def main():
    dataset_dir = sys.argv[1]
    n_fields = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"loading {n_fields} fields from {dataset_dir} ...")
    flux, spectrum = _load_voxels(dataset_dir, n_fields)
    print(f"voxels={flux.shape[0]}  flux-range=[{flux.min():.2f},{flux.max():.2f}]")
    for prec in ("fp32", "fp16"):
        log_l1, hi_rel, spec = _train(prec, flux, spectrum)
        print(f"[{prec}] flux_log_L1={log_l1:.4f}  top5%_phys_rel_err={hi_rel:.3f}  spec_overlap={spec:.3f}")


if __name__ == "__main__":
    main()

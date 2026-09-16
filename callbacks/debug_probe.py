"""TrainingDebugProbe — a listed per-step debug view of training.

Every ``every_n_steps`` it prints (and appends to ``<logs>/debug_probe.log``) exactly what flowed
through the model this step:

  INPUT   the model inputs after the full data pipeline (position grid, direction, origin/distance,
          tube spectrum) — shapes, ranges, norms, NaN/Inf counts;
  TARGET  the normalised GT flux the loss saw — finite fraction (= what the voxel sampler kept),
          value range/quantiles, per-ROI voxel counts (beam/scatter/floor, radfield3dnn.roi);
  OUTPUT  the model's flux/spectrum — range, quantiles, fraction pinned at the activation
          floor/ceiling (lock-in detector), spectrum entropy;
  LOSS    the per-task losses, the flux loss's per-ROI term breakdown (TwoROIGammaLoss.last_terms)
          and the DB-MTL weights of this step.

Capture happens in BaseNeuralRadFieldModel.calculate_metrics (the `_debug_capture` seam), so the
probe shows the tensors EXACTLY as the loss consumed them. Enable via YAML:

    training:
      debug_probe: true
      debug_probe_every: 50
"""
import math
import os

import lightning.pytorch as pl
import torch


def _stats(t: torch.Tensor) -> str:
    f = t[torch.isfinite(t)]
    if f.numel() == 0:
        return "ALL NON-FINITE"
    q = torch.quantile(f.float().flatten()[: 2_000_000], torch.tensor([0.5, 0.99], device=f.device))
    return (f"[{f.min():.3g}, {f.max():.3g}] μ={f.mean():.3g} q50={q[0]:.3g} q99={q[1]:.3g}"
            + (f"  nonfinite={t.numel()-f.numel()}" if f.numel() != t.numel() else ""))


class TrainingDebugProbe(pl.Callback):
    def __init__(self, every_n_steps: int = 50, log_path: str | None = None):
        self.every = max(1, int(every_n_steps))
        self.log_path = log_path
        self._fh = None

    def _emit(self, text: str):
        print(text)
        if self.log_path:
            if self._fh is None:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._fh = open(self.log_path, "a", buffering=1)
            self._fh.write(text + "\n")

    def on_fit_start(self, trainer, pl_module):
        pl_module._debug_probe_enabled = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every != 0:
            return
        cap = getattr(pl_module, "_debug_capture", None)
        if not cap:
            return
        lines = [f"── debug probe · step {trainer.global_step} · epoch {trainer.current_epoch} " + "─" * 30]

        # INPUT — the beam parameters after the full pipeline (normalisations applied)
        inp = cap.get("batch_input")
        if inp is not None:
            for name in ("position", "direction", "origin", "spectrum"):
                t = getattr(inp, name, None)
                if isinstance(t, torch.Tensor):
                    lines.append(f" INPUT   {name:<9s} {str(tuple(t.shape)):>16s}  {_stats(t)}")

        # TARGET — what the loss compared against (post-normaliser, post-sampler)
        tgt = cap["target_flux"]
        finite = torch.isfinite(tgt)
        ffrac = float(finite.float().mean())
        lines.append(f" TARGET  flux      {str(tuple(tgt.shape)):>16s}  finite {ffrac*100:.1f}%  {_stats(tgt)}")
        try:
            from radfield3dnn.roi import compute_roi_masks
            safe = torch.where(finite, tgt, torch.zeros_like(tgt))
            b, s, fl = compute_roi_masks(safe, safe)
            lines.append(f"         ROI counts (finite): beam {int((b & finite).sum())} | "
                         f"scatter {int((s & finite).sum())} | floor {int((fl & finite).sum())}")
        except Exception:
            pass

        # OUTPUT — what the model produced
        pred = cap["pred_flux"]
        lines.append(f" OUTPUT  flux      {str(tuple(pred.shape)):>16s}  {_stats(pred)}")
        lo, hi = float(pred.min()), float(pred.max())
        at_floor = float((pred <= lo + 1e-9).float().mean())
        at_ceil = float((pred >= 0.999).float().mean())
        lines.append(f"         @min {at_floor*100:.2f}%  @≥0.999 {at_ceil*100:.2f}%   (lock-in detector)")
        spec = cap.get("pred_spectrum")
        if spec is not None:
            p = spec.clamp_min(1e-12)
            ent = float((-(p * p.log()).sum(dim=1)).mean()) if spec.ndim >= 2 else float("nan")
            lines.append(f"         spectrum  {str(tuple(spec.shape)):>16s}  entropy μ={ent:.3f} (max {math.log(spec.shape[1]):.3f})")

        # LOSS — per-task + flux per-ROI terms + DB-MTL weights
        terms = cap.get("flux_loss_terms") or {}
        tstr = "  ".join(f"{k}={v:.4g}" if not k.startswith("n_") else f"{k}={v}" for k, v in terms.items())
        lines.append(f" LOSS    flux {cap['flux_loss']:.4f}" + (f"   ({tstr})" if tstr else ""))
        if cap.get("spectrum_loss") is not None:
            lines.append(f"         spectrum {cap['spectrum_loss']:.4f}")
        mtl = getattr(pl_module, "_mtl", None)
        if mtl is not None and getattr(mtl, "last_weights", None):
            w = "  ".join(f"{k}={v:.3g}" for k, v in mtl.last_weights.items())
            lines.append(f"         dbmtl weights: {w}")

        self._emit("\n".join(lines))

    def teardown(self, trainer, pl_module, stage):
        if self._fh:
            self._fh.close()
            self._fh = None

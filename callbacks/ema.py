"""Weight EMA (exponential moving average) for less-noisy, more reproducible models.

SGD on this problem follows a noisy trajectory: the final-epoch weights land at a
seed-dependent point on a flat, noisy basin, which is a major source of the
~10% seed-to-seed variance in the air-kerma metrics. Evaluating an **exponential
moving average of the recent weights** instead of the last-step weights averages
that noise out — it consistently lands nearer the basin's centre, lowering
variance and usually nudging the metrics up (the standard Polyak/EMA trick, the
same idea behind SWA).

The callback keeps a shadow copy of the model parameters, updates it after every
optimizer step, and swaps it in for **validation and test** (restoring the live
training weights afterwards) so both the monitored metrics and the final test
reflect the smoothed weights. Buffers (e.g. the non-finite-loss counter) are left
untouched; only `named_parameters` are averaged, which is correct for the
batchnorm-free MLP heads of the SRBF/SPERF/PBRF NeRF models.
"""

from __future__ import annotations
import torch
from lightning.pytorch.callbacks import Callback


class WeightEMA(Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        assert 0.0 < decay < 1.0, "EMA decay must be in (0, 1)."
        self.decay = float(decay)
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    # ── build / update the shadow ────────────────────────────────────────────
    def on_train_start(self, trainer, pl_module):
        if not self._shadow:  # fresh run (not a resume that already restored it)
            # Keep the shadow in fp32. In a fp16 model the EMA increment
            # (1-decay)*param ≈ 1e-5 underflows fp16, so an fp16 shadow never
            # moves from its init and eval runs a near-init model (huge train↔val
            # gap). An fp32 shadow accumulates the small increments correctly.
            self._shadow = {n: p.detach().float().clone()
                            for n, p in pl_module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        d = self.decay
        for n, p in pl_module.named_parameters():
            s = self._shadow.get(n)
            if s is not None:
                s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    # ── swap EMA in for eval, restore live weights afterwards ────────────────
    @torch.no_grad()
    def _swap_in(self, pl_module):
        if not self._shadow:
            return
        self._backup = {}
        for n, p in pl_module.named_parameters():
            s = self._shadow.get(n)
            if s is not None:
                self._backup[n] = p.detach().clone()
                p.copy_(s)

    @torch.no_grad()
    def _swap_out(self, pl_module):
        if not self._backup:
            return
        for n, p in pl_module.named_parameters():
            b = self._backup.get(n)
            if b is not None:
                p.copy_(b)
        self._backup = {}

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._swap_in(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_exception(self, trainer, pl_module, exception):
        # try/finally semantics across the start/end hook pair: if validation/test raises between
        # _swap_in and _swap_out, restore the live training weights so the run never silently
        # continues from the EMA shadow. _swap_out is a no-op when nothing was swapped in.
        self._swap_out(pl_module)

    # ── persist the shadow with the checkpoint ───────────────────────────────
    def state_dict(self):
        return {"decay": self.decay, "shadow": self._shadow}

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        self._shadow = state.get("shadow", {})

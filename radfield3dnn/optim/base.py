"""Optimizer/scheduler *behaviour* interface.

A behaviour encapsulates everything a ``LightningModule.configure_optimizers`` would build for a
model — the optimizer, its parameter groups, and the LR schedule — and returns that result. One
class per behaviour, all sharing this interface, so a model just holds a behaviour instance and
delegates ``configure_optimizers`` to it.

This base also owns the **fp16 → fp32 master-weight** wiring (the setup half). In fp16 the optimizer
must optimize fp32 master copies of the weights so Adam's state lives in fp32 — sub-fp16-ULP updates
accumulate across steps instead of rounding to zero (pure-fp16 Adam leaves the model near init). The
matching *runtime* hooks (sync masters → fp16 before forward, transfer fp16 grads → fp32 masters
after backward, loss scaling) live on ``BaseNeuralRadFieldModel`` and read the state set up here.
"""
import torch


class OptimizerBehaviour:
    """Interface for an optimizer + scheduler behaviour. Subclasses implement :meth:`build`."""

    def configure(self, model):
        """Wire the fp16 masters (if the model is fp16), then build and return the
        ``configure_optimizers()`` result. This is what the model's ``configure_optimizers`` calls."""
        self.setup_fp16_masters(model)
        return self.build(model)

    def build(self, model):
        """Return the Lightning ``configure_optimizers()`` output (``([optimizer], [scheduler_cfg])``)
        for ``model``. Use :meth:`optimizer_target` to bind each parameter so the fp16 path optimizes
        the fp32 masters."""
        raise NotImplementedError("OptimizerBehaviour subclasses must implement build().")

    # ── fp16 fp32-master-weight wiring (setup half) ───────────────────────────
    @staticmethod
    def is_fp16(model) -> bool:
        return getattr(model, "_precision", "fp32") == "fp16"

    @classmethod
    def setup_fp16_masters(cls, model) -> bool:
        """fp16 only: create fp32 master copies of every trainable parameter plus the loss scale on
        the model. The fp16 weights become a cast view that the model's training hooks sync from / to
        the masters. Built here (at configure-optimizers time, after the model is on its device).
        No-op in fp32. Returns whether masters were created."""
        if not cls.is_fp16(model):
            return False
        model._fp32_masters = {
            name: p.detach().float().clone().requires_grad_(True)
            for name, p in model.named_parameters() if p.requires_grad
        }
        # Loss scale: keeps the largest grads under fp16's 65504 ceiling while lifting the HDR-tail
        # gradients above the ~6e-8 underflow floor.
        model._loss_scale = 256.0
        return True

    @staticmethod
    def optimizer_target(model, name: str, param: torch.Tensor) -> torch.Tensor:
        """The tensor the optimizer should bind for a parameter: its fp32 master (fp16 path) or the
        parameter itself (fp32 path)."""
        masters = getattr(model, "_fp32_masters", None)
        return masters[name] if masters is not None else param

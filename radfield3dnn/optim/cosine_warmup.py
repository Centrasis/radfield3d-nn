"""``CosineWithWarmup`` optimizer behaviour.

AdamW with component-specific LR / weight-decay groups (hash/positional encodings, MLP, no-decay)
and a step-wise **linear warmup** followed by **cosine** decay, composed with ``SequentialLR``.
This is the original ``RFNetBase.configure_optimizers`` body, lifted verbatim into a behaviour so the
schedule is encapsulated and swappable.
"""
import math
import torch
import torch.nn as nn

from .base import OptimizerBehaviour


def _accumulate_grad_batches(trainer) -> int:
    try:
        for cb in getattr(trainer, "callbacks", []):
            if cb.__class__.__name__ == "GradientAccumulationScheduler":
                sched = getattr(cb, "scheduling", None)
                if isinstance(sched, dict) and sched:
                    keys = sorted(int(k) for k in sched.keys())
                    return int(sched.get(0, sched[keys[0]]))
    except Exception:
        pass
    return 1


def build_warmup_cosine_schedule(optimizer, trainer, effective_lr, *, warmup_lr: float = 1e-5,
                                 default_warmup_steps: int = 200, eta_min: float = 5e-6):
    """Step-interval linear-warmup -> cosine-annealing schedule, in optimizer steps. Shared by
    CosineWithWarmup and PBRFNetTCNN."""
    total_opt_steps = int(trainer.estimated_stepping_batches)
    max_epochs = int(max(trainer.max_epochs, 1))
    acc_batches = max(1, _accumulate_grad_batches(trainer))
    if not torch.isfinite(torch.tensor(total_opt_steps)) or total_opt_steps <= 0:
        total_opt_steps = default_warmup_steps
        max_epochs = 1
    total_opt_steps /= acc_batches
    warmup_budget = int(max(default_warmup_steps / acc_batches, 1))
    steps_per_epoch = int(math.ceil(total_opt_steps / max_epochs))
    warmup_epochs = int(max(warmup_budget / steps_per_epoch, 1))
    warmup_steps = int(min(warmup_epochs * steps_per_epoch, max(1, total_opt_steps - 1)))
    cosine_steps = int(max(1, total_opt_steps - warmup_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_lr / effective_lr, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=eta_min)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return [optimizer], [{
        "scheduler": schedule, "interval": "step", "monitor": "train_loss", "name": "warmup+cosine",
    }]


class CosineWithWarmup(OptimizerBehaviour):
    """Linear warmup → cosine-annealing schedule (step interval), AdamW with per-component groups."""

    def build(self, model):
        # fp16 weight storage diverges above ~2e-3 (fp16 flux ceiling); clamp lr to the same 5e-4
        # ceiling nerf_cpp.PBRFNetCPP uses so an aggressive lr_find pick can't blow up the run.
        if self.is_fp16(model):
            max_lr = getattr(model, "_max_lr", 5e-4)
            effective_lr = min(max(float(model._lr), 1e-5), float(max_lr))
        else:
            effective_lr = max(float(model._lr), 1e-5)

        # LayerNorm params get no weight decay.
        ln_param_ids = set()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    ln_param_ids.add(id(p))

        # Separate parameters by component. In fp16 the optimizer binds the fp32 *masters*
        # (optimizer_target) so Adam state stays fp32; in fp32 it binds the params directly.
        encoding_params, mlp_params, no_decay = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            target = self.optimizer_target(model, name, param)
            if 'positional_location_encoding.encoding.params' in name or 'positional_direction_encoding.encoding.params' in name:
                encoding_params.append(target)
            elif ("_normalizer" in name) or ("_normalizer.m" in name) or (name.endswith(".bias")) or (id(param) in ln_param_ids):
                no_decay.append(target)
            else:
                mlp_params.append(target)

        assert len(encoding_params) + len(mlp_params) + len(no_decay) == len(list(model.parameters())), "Parameter separation error"

        # eps: InstantNGP uses 1e-15 (far below the 1e-8 default) so the Adam preconditioner stays
        # aggressive on the tiny HDR-flux-tail gradients. Usable in both precisions because Adam state
        # lives in fp32 (the fp16 path optimises the fp32 masters).
        adam_eps = 1e-15
        optimizer = torch.optim.AdamW([
                {'params': encoding_params, 'lr': 1e-2, 'initial_lr': 1e-2, "weight_decay": 0.0, "eps": adam_eps},
                {'params': mlp_params, 'lr': effective_lr, 'initial_lr': effective_lr, 'weight_decay': 1e-4, "eps": adam_eps},
                {'params': no_decay, 'lr': effective_lr, 'initial_lr': effective_lr, 'weight_decay': 0.0, "eps": adam_eps},
            ],
            betas=(0.9, 0.99)
        )

        return build_warmup_cosine_schedule(optimizer, model.trainer, effective_lr)

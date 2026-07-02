"""Multi-task loss balancing, encapsulated.

Implements DB-MTL (Lin et al. 2023, "Dual-Balancing for Multi-Task Learning",
arXiv:2308.12029) as a small, model-agnostic component so the balancing logic
lives in one place instead of being threaded through `BaseNeuralRadFieldModel`.

DB-MTL has two balancing steps:

1. **Loss-scale balancing** — log-transform every task loss so tasks with very
   different raw magnitudes (e.g. flux ~0.5 vs normalised-histogram spectrum
   ~0.15) become directly comparable. This step alone is *scale-invariant* and
   needs no model internals, so it is the black-box-safe fallback for fused
   models (e.g. the tcnn-fused `PBRFNetCPP`) that cannot expose a shared
   representation.

2. **Gradient-magnitude balancing** — rescale each task's gradient w.r.t. a
   shared representation to the largest task-gradient norm so no task dominates
   the shared update by sheer gradient size.

Plain DB-MTL recomputes the step-2 norms from a single minibatch every step.
On this problem that is far too noisy: the per-task trunk-gradient norms swing
by orders of magnitude between minibatches, and once an *easy* task (the
spectrum head) is solved its trunk gradient collapses toward zero, so the raw
``max_norm / norm_i`` weight explodes (observed up to 1000×) and the trunk is
forced to keep serving the solved task while starving the hard task. Two guards
fix this without changing the method:

* the norms are **EMA-smoothed** across steps (variance reduction), and
* the resulting weights are **clamped** to ``max_weight`` (prevents a vanished
  gradient from producing a runaway weight).

The combined surrogate ``Σ_i w_i · log L_i`` lives in log space and its *value*
is not meaningful (it can be negative). Callers should use it only as a gradient
carrier — see `BaseNeuralRadFieldModel.process_metrics`, which wraps it in a
straight-through estimator so the reported/monitored loss stays positive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiTaskLossBalancer(nn.Module):
    def __init__(self, ema_momentum: float = 0.9, max_weight: float = 10.0, eps: float = 1e-8,
                 update_every: int = 16, loss_floor: float = 1e-3):
        """
        Args:
            ema_momentum: smoothing for the per-task gradient norms in [0, 1).
                Higher = smoother/slower. 0 disables smoothing (plain DB-MTL).
            max_weight: upper bound on a task's balancing weight. Caps the
                amplification of tasks whose shared gradient has vanished.
            eps: numerical floor for logs and norm divisions.
            update_every: recompute the (expensive) per-task shared-gradient norms —
                and hence the balancing weights — only every N training steps, reusing
                the **cached** weights in between. The norms are EMA-smoothed and drift
                slowly, so this keeps full DB-MTL balancing at ~1/N the cost (the per-
                task `autograd.grad` was N_tasks extra full-network backwards/step).
            loss_floor: GRADIENT guard for the log transform. ``∇ log L = (1/L)·∇L``
                amplifies a task's gradient by 1/L, unbounded as the task is solved.
                Clamping the loss at ``loss_floor`` before the log bounds the
                amplification at 1/loss_floor and — because clamp kills the gradient
                below the floor — stops pushing a task once it is solved past the floor
                (correct behaviour: a solved task should release
                the shared trunk, not dominate it).
        """
        super().__init__()
        self.ema_momentum = ema_momentum
        self.max_weight = max_weight
        self.eps = eps
        self.loss_floor = float(loss_floor)
        self.update_every = int(update_every)
        self._calls = 0                       # training-step counter (grad-enabled calls)
        self._cached_weights: dict[str, float] | None = None  # last computed balancing weights
        # EMA state of the per-task gradient norms (name -> scalar tensor).
        # Plain dict (re-warms in a few steps); not part of the checkpoint.
        self._ema_norms: dict[str, Tensor] = {}
        # Most recent weights / gradient norms, for logging by the caller.
        self.last_weights: dict[str, float] = {}
        self.last_gradnorms: dict[str, float] = {}
        # B-hardening telemetry: tasks whose trunk-gradient norm overflowed to
        # inf/NaN this step (the fp16-overflow guard fired). Logged by the caller
        # (`{stage}_dbmtl_overflow_{task}`) so a triggered guard is VISIBLE rather
        # than silently masking a real divergence. `overflow_count` is cumulative.
        self.last_overflowed: dict[str, bool] = {}
        self.overflow_count: int = 0

    def _log_losses(self, task_losses: dict[str, Tensor]) -> dict[str, Tensor]:
        # loss_floor (not eps) is the clamp: bounds the 1/L gradient amplification of the log
        # transform at 1/loss_floor and zeroes the push on tasks solved below the floor.
        return {n: torch.log(l.mean().clamp_min(self.loss_floor)) for n, l in task_losses.items()}

    def combine(self, task_losses: dict[str, Tensor], balance_parameters: list[nn.Parameter] | None,
                balance_limits: list[int | None] | None = None) -> Tensor:
        """Return the DB-MTL surrogate (a gradient carrier, value not meaningful).

        Args:
            task_losses: name -> per-sample (or scalar) task loss tensor.
            balance_parameters: shared-representation parameters for step-2
                gradient-magnitude balancing. Pass ``None`` (black-box models,
                a single task, or eval with no grad) to use loss-scale balancing
                only (step 1) — a single backward, no per-task autograd.
            balance_limits: optional list parallel to ``balance_parameters``; each
                entry is an element count N or ``None``. When N is given, only the
                first N elements of that parameter's gradient enter the per-task
                norm — used by the fused PBRFNetCPP to restrict the norm to the
                shared TRUNK slice of its single fused-weights tensor
                (``weights[:head_offset]``), excluding the task-specific heads that
                live in the same blob. ``None`` uses the whole gradient.
        """
        logs = self._log_losses(task_losses)
        self.last_weights = {n: 1.0 for n in logs}
        self.last_gradnorms = {}
        self.last_overflowed = {n: False for n in logs}

        # Step 1 only: loss-scale balancing.
        if balance_parameters is None or len(logs) == 1 or not torch.is_grad_enabled():
            return torch.stack(list(logs.values())).sum()

        # CACHED step 2: the per-task `autograd.grad(loss, trunk)` below is N_tasks
        # extra FULL-network backwards/step. The resulting weights drift slowly (EMA-
        # smoothed), so recompute them only every `update_every` steps and reuse the
        # cache otherwise → one backward on the ~(N-1)/N of steps that reuse it.
        self._calls += 1
        if self._cached_weights is not None and (self._calls % self.update_every != 0):
            self.last_weights = dict(self._cached_weights)
            total = torch.zeros((), device=next(iter(logs.values())).device)
            for n, lg in logs.items():
                total = total + self._cached_weights[n] * lg     # detached scalar × diff log-loss
            return total

        if balance_limits is None:
            balance_limits = [None] * len(balance_parameters)

        def _sq_trunk(grads):
            # Sum of squares over each param's gradient, restricted to its trunk
            # slice [:limit] when a limit is given (heads excluded).
            total = 0.0
            for g, lim in zip(grads, balance_limits):
                if g is None:
                    continue
                gd = g.detach()
                if lim is not None:
                    gd = gd.reshape(-1)[:lim]
                total = total + (gd ** 2).sum()
            return total

        # Step 2: gradient-magnitude balancing w.r.t. the shared representation.
        norms: dict[str, Tensor] = {}
        for n, lg in logs.items():
            if not lg.requires_grad:
                # Constant/placeholder task loss (e.g. the two-head model's
                # flux-only direct head emits a zero direct-spectrum, or a channel
                # fully removed by importance sampling). `autograd.grad` would raise
                # on it; it contributes no gradient, so keep weight 1.0 and skip it.
                continue
            grads = torch.autograd.grad(lg, balance_parameters, retain_graph=True, allow_unused=True)
            sq = _sq_trunk(grads)
            if isinstance(sq, Tensor):
                norm = (sq + self.eps).sqrt()
            else:  # every grad was None (task detached from these params)
                norm = torch.full((), self.eps ** 0.5, device=lg.device)
            # ── B: fp16-overflow hardening ────────────────────────────────────
            # A task's gradient can overflow to inf/NaN in the fp16-fused cpp
            # path (a starved task whose 1/L blows up). An inf/NaN norm would
            # poison max_norm and turn EVERY task weight into NaN. Reuse the
            # smoothed EMA when we have one, else drop this task to a neutral
            # weight (skip) for this step — keeps training finite instead of
            # diverging.
            if not torch.isfinite(norm):
                self.last_overflowed[n] = True
                self.overflow_count += 1
                # Surface it: a triggered guard means a real fp16 overflow, not a
                # benign event. Warn (throttled to avoid per-step spam) so it is
                # never silently masked; the caller also logs it to wandb.
                if self.overflow_count <= 5 or self.overflow_count % 100 == 0:
                    import warnings
                    warnings.warn(
                        f"[DB-MTL] trunk-gradient norm for task '{n}' overflowed to "
                        f"{float(norm)} (fp16 overflow); falling back to EMA/neutral "
                        f"weight. overflow_count={self.overflow_count}.", RuntimeWarning)
                prev = self._ema_norms.get(n)
                if prev is None or not torch.isfinite(prev):
                    self.last_gradnorms[n] = float('inf')
                    continue
                norm = prev
            self.last_gradnorms[n] = float(norm)
            # EMA-smooth the norm used for weighting.
            prev = self._ema_norms.get(n)
            smoothed = norm if prev is None else self.ema_momentum * prev + (1.0 - self.ema_momentum) * norm
            self._ema_norms[n] = smoothed.detach()
            norms[n] = smoothed

        # No task had a usable gradient (all constant) — fall back to loss-scale.
        if not norms:
            return torch.stack(list(logs.values())).sum()

        max_norm = torch.stack(list(norms.values())).max()
        # B: if even the max is non-finite (all norms overflowed), fall back to
        # plain loss-scale balancing rather than emit NaN weights.
        if not torch.isfinite(max_norm):
            return torch.stack(list(logs.values())).sum()
        total = torch.zeros((), device=max_norm.device)
        for n, lg in logs.items():
            if n in norms:
                w = (max_norm / norms[n].clamp_min(self.eps)).clamp(max=self.max_weight).detach()
                if not torch.isfinite(w):
                    w = torch.ones((), device=max_norm.device)  # B: never weight by NaN
            else:
                w = torch.ones((), device=max_norm.device)  # constant/skipped task: neutral weight
            self.last_weights[n] = float(w)
            total = total + w * lg
        self._cached_weights = dict(self.last_weights)   # reuse for the next update_every-1 steps
        return total

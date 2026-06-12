import math
import torch
from .nerf import ModuleBuilder
from .nerf import RFNetBase
from radfield3dnn.rftypes import PositionalInput, RadiationField, RadiationFieldChannel
from torch import nn, Tensor
from typing import Union
try:
    # The tiny-cuda-nn native module. OPTIONAL: tcnn is deactivated by default (a plain
    # `pip install` is pure-Python), so this import may be absent. The cpp-backed model classes
    # below are still defined and self-register; only CONSTRUCTING one touches `rfnn` and then
    # raises the clear error below. Build the module with `RFNN_WITH_TCNN=1 pip install -e .`.
    import radfield3dnn.radfield3dnn as rfnn
except ImportError:
    class _TcnnUnavailable:
        def __getattr__(self, name):
            raise ImportError(
                "radfield3dnn.radfield3dnn (the tiny-cuda-nn native module) is not built — the "
                "cpp-backed models (PBRFNetCPP, SPERFNetCPP, …) are deactivated. Reinstall with "
                "`RFNN_WITH_TCNN=1 pip install -e .` to enable them.")
    rfnn = _TcnnUnavailable()
from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize


class _TcnnModule(nn.Module):
    """Pure-Python nn.Module wrapper around a C++ tcnn facade.

    The C++ facades (`rfnn.BaseRadiationPredictionModel`,
    `rfnn.PBRFBeamEncoder`) are bound via `torch::python::bind_module`, so they
    *are* torch.nn.Module objects. Registered directly, they show up in
    `LightningModule.named_modules()`, and Lightning's `_ModuleMode` restores
    train/eval state after every validation by assigning `mod.training = ...`
    on each — but the C++ binding exposes `training` as a read-only property,
    so this throws `AttributeError: property '' ... has no setter` at the first
    validation teardown.

    This mirrors tiny-cuda-nn's own `tinycudann.modules.Module`: keep the C++
    object as a *non-submodule* attribute and expose its trainable tensor as a
    normal Python `nn.Parameter`. The wrapper is plain Python, so `.training`
    is settable (Lightning is happy); the parameter is the *same* tensor the
    C++ bridge handed tcnn a raw pointer to (no copy), so optimizer updates
    flow straight back into tcnn's storage and the weights still appear in
    `state_dict()` — checkpointable here and loadable later for detached,
    Python-free C++ execution.
    """

    # Per-cpp-class-name dynamic-subclass cache. Lightning's ModelSummary (and
    # nn.Module.__repr__) read `type(layer).__name__`, so overriding only
    # `_get_name` is not enough — we must retype the instance to a subclass
    # whose Python name matches the wrapped C++ module. Cached so repeated
    # constructions reuse the same class (state_dict / pickling stable).
    _wrapper_class_cache: dict[str, type] = {}

    def __init__(self, cpp: nn.Module):
        super().__init__()
        # object.__setattr__ bypasses nn.Module.__setattr__, so `cpp` is NOT
        # registered as a child module and never reaches named_modules().
        object.__setattr__(self, "_cpp", cpp)
        # `cpp.weights` is the exact requires_grad fp16 tensor the C++ autograd
        # Function writes `.grad` to — but it is a bare Tensor, not an
        # nn.Parameter, so register_parameter() rejects it and wrapping it in
        # a new Parameter would create a different object that never receives
        # that grad. Insert the object itself into `_parameters` so the
        # autograd graph keeps working through this leaf (grad flows fp16),
        # and it still serialises via state_dict.
        self._parameters["weights"] = cpp.weights
        # ─── fp32 master weights (InstantNGP-style) ────────────────────────
        # The optimizer operates on a fp32 *master* copy of the weights,
        # while the C++ fused-MLP kernel keeps consuming fp16. The forward
        # pre-hook syncs `fp32_master → cpp.weights` so the kernel always
        # sees the latest cast. The post-backward hook (`on_after_backward`
        # on the LightningModule) transfers fp16 grads back to fp32 master
        # grads. Mirrors `tiny-cuda-nn/include/tiny-cuda-nn/optimizers/adam.h`'s
        # `<T, PARAMS_T>` split (fp32 state, fp16 params): updates smaller
        # than fp16 ULP (~1e-3 near 1.0) accumulate in fp32 across many
        # steps until they cross a fp16 ULP boundary, then bump the fp16
        # weight by one ULP. Without this, ~56% of Adam's small-magnitude
        # updates round to 0 in fp16 state (documented in
        # pbrfnetcpp-nan-status.md 2026-05-18). Memory cost: one extra fp32
        # tensor of the same numel — for PBRFNetCPP that's +1.22 MB per
        # `_TcnnModule`, negligible vs the fused fwd_ctx / batch buffers.
        master = cpp.weights.detach().clone().to(torch.float32)
        # `register_buffer` with `persistent=True` puts it in state_dict so
        # checkpoint round-trips preserve the fp32 precision (the fp16 view
        # alone would lose ~10 mantissa bits per save/load cycle).
        self.register_buffer("fp32_master", master, persistent=True)
        # `requires_grad=True` on a buffer is legal and required so torch.optim
        # accepts it as a trainable param — the buffer is NOT in `.parameters()`
        # (only the fp16 weights are), so the optimizer must be built by
        # explicitly listing the masters (see `PBRFNetCPP.configure_optimizers`).
        self.fp32_master.requires_grad_(True)
        # Sync fp32 master → fp16 cpp.weights at the start of every forward,
        # done INLINE in `_TcnnModule.forward` below — not via a forward
        # pre-hook. `PBRFNetCPP.forward` calls `self.model.forward(...)` and
        # `self.encoder.forward(...)` directly (bypassing `__call__`), which
        # would skip pre-hooks entirely. The inline sync runs regardless of
        # how forward is invoked.
        cpp_name = cpp._get_name() or type(cpp).__name__
        self._name = cpp_name
        # Retype this instance so Lightning's ModelSummary (which uses
        # `type(layer).__name__`) shows e.g. "PBRFBeamEncoder" instead of
        # "_TcnnModule". A subclass keeps isinstance(_TcnnModule) and inherits
        # all behaviour; the cached class is created once per cpp-class name.
        cls = _TcnnModule._wrapper_class_cache.get(cpp_name)
        if cls is None:
            cls = type(cpp_name, (_TcnnModule,), {})
            _TcnnModule._wrapper_class_cache[cpp_name] = cls
        self.__class__ = cls

    def _sync_master_to_fp16(self):
        # Copy fp32 master → fp16 cpp.weights in-place so tcnn's raw pointer
        # (captured at construction) stays valid and the JIT-fused kernel
        # reads the freshly-Adam-updated weights. Runs once per forward call,
        # before the C++ kernel reads `cpp.weights`.
        with torch.no_grad():
            self._parameters["weights"].copy_(self.fp32_master.to(torch.float16))

    def _apply(self, fn, recurse=True):
        # The wrapped C++ tcnn module captured a raw pointer to `weights`'
        # CUDA storage exactly once, at construction
        # (bridge.inl/utils.h: set_params(weights.data_ptr())); there is no
        # re-bind API. nn.Module._apply — invoked by .cpu()/.cuda()/.to() and,
        # decisively, by Lightning's Tuner.lr_find checkpoint teardown and the
        # Trainer's device moves around fit — swaps this tensor's underlying
        # storage. tcnn then keeps reading the old, now-freed CUDA buffer:
        # Python sees finite `weights` while tcnn emits NaN/constant output,
        # which is exactly the "fine untrained, NaN after the first epoch"
        # failure (lr_find runs before fit, so the desync happens before
        # training even starts). tcnn is CUDA-only and its parameter buffer is
        # fixed for the module's lifetime, so device/dtype casts MUST be a
        # no-op here — that keeps tcnn's pointer valid forever. Checkpoints
        # still round-trip: load_state_dict copies into `weights` in place
        # (same storage), and state_dict() serialises the live CUDA tensor.
        return self

    def forward(self, *args, **kwargs):
        return self._cpp.forward(*args, **kwargs)

    def _get_name(self) -> str:
        return self._name


class PBRFNetCPP(RFNetBase):
    __model_name__ = "PBRFNetCPP"

    @property
    def use_lr_finder(self) -> bool:
        """Fused fp16 model: skip the LR finder (its high-LR sweep NaNs the
        fp16 fused weights; the optimizer clamps LR to max_lr anyway)."""
        return False

    @property
    def mtl_gradient_balancing(self) -> bool:
        """Enable DB-MTL gradient-magnitude balancing (step 2) for the fused model.

        Previously disabled on the assumption that a per-task ``autograd.grad``
        through the fused C++ autograd Function is "not re-entrant". That was
        WRONG: the bridge backward supports ``retain_graph`` and multiple
        per-task ``autograd.grad`` calls work fine (verified 2026-06-05). With
        balancing off, the flux and spectrum tasks share the trunk at fixed 1.0
        weights; the spectrum task (easy, beam-driven) dominates the shared
        update and starves the hard spatial flux task — the cpp capped ~0.26
        while a DB-MTL-balanced pure-Python twin reaches ~0.86 (see
        claude-notes/summary.md).

        The balance target is the fused ``model``/``encoder`` weights
        (`_shared_parameters`); `output_head_markers` is empty so the whole fused
        blob is used — trunk-dominated, so it approximates the shared-trunk
        gradient. Cost: 2 extra (per-task) backward passes per step.
        """
        return True

    def __init__(self, location_encoding_dims=10, in_spectra_dim=32, d_model=256, flux_loss="L1Loss", spectrum_loss="HistogramLoss", learning_rate: float=1e-3, randomize_voxel_location_in_training: bool = True, voxels_centered_around_origin: bool = True, normalizer=None, seed: int = 1337, max_lr: float = 1e-3, flux_offset: float = 0.5, flux_activation: str = "clamp", location_encoding_kind: str = "frequency", flux_clamp_min: float = 0.0, flux_clamp_max: float = 1.0, trunk_hidden_layers: int = 1, beam_fusion: str = "film"):
        # Single-head model: one joined flux head + a joined spectrum head.
        # flux_clamp_min/max: codomain of the hard-clamp flux activation.
        # Defaults (0, 1) reproduce historic behavior. For real log-space
        # training paired with the LogScaleNormalizer, use
        #     normalizer="log_scale", flux_activation="clamp",
        #     flux_clamp_min=-9.0, flux_clamp_max=0.0, flux_offset=-4.5
        # The clamp range [-9, 0] matches LogScaleNormalizer's full output
        # domain (zero sentinel at -9, log10 of [1e-8, 1.0] in [-8, 0]). With
        # this setup, prefer flux_loss="L1Loss" — the log step is now in the
        # data, so an additional log inside the loss would double-log.
        # Lightning's save_hyperparameters(ignore=[... "normalizer"]) in
        # BaseNeuralRadFieldModel drops `normalizer` from the checkpoint
        # hparams, so LightningModule.load_from_checkpoint reconstructs us with
        # normalizer=None and the base ctor's isinstance(Normalizer) assert
        # fails -> stored models could not be loaded back. Coerce here: a name
        # string is constructed (base already does this too) and None falls
        # back to the project default ("log_decade") so headless / checkpoint
        # reload works.
        #
        # log_decade -> LogDecadeNormalizer((0,1), x_min=1e-8, x_max=1.0): a
        # fixed base-10 *decade* map giving every decade an equal slice of
        # (0,1). The HDR flux field spans ~8 decades with ~99% near-zero
        # voxels; under FP16 a linear map sends 1e-8 -> 1e-8 which underflows
        # to 0 (Sigmoid can't emit it -> predicts ~0 everywhere, SSIM~0,
        # NCC<0 — observed), and log_1e+5's log1p(x*1e5) is ~linear for
        # x<<1e-5 so it crushes the 1e-8..1e-7 tail into the FP16 floor. The
        # decade map keeps FP16's ~constant relative mantissa step ->
        # ~constant relative error (~1%) across the whole range incl. ~1.0.
        # Subclasses LinearNormalizer with range (0,1), so the Sigmoid flux
        # head below already matches its output range.
        if normalizer is None:
            # LogDecadeNormalizer was removed in the clean repo; `log_scale`
            # (LogScaleNormalizer, log10 of [1e-8, 1] with a zero sentinel at
            # -9) is the supported HDR default and pairs with this model's
            # log-space flux activation (see the clamp/offset note above).
            normalizer = NormalizerConstructor.construct_by_name("log_scale")
        elif isinstance(normalizer, str):
            normalizer = NormalizerConstructor.construct_by_name(normalizer)
        super().__init__(
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            normalizer=normalizer
        )
        # The set seed in time before tcnn seeds the networks
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._flux_offset = float(flux_offset)
        # 0=clamp, 1=softclip — see SoftClip docstring in
        # python/radfield3dnn/activations/flux_activations.py.
        if flux_activation not in ("clamp", "softclip"):
            raise ValueError(f"flux_activation must be 'clamp' or 'softclip', got {flux_activation!r}")
        self._flux_activation_kind = flux_activation
        self._flux_clamp_min = float(flux_clamp_min)
        self._flux_clamp_max = float(flux_clamp_max)
        if self._flux_clamp_min >= self._flux_clamp_max:
            raise ValueError(
                f"flux_clamp_min ({flux_clamp_min}) must be strictly less than "
                f"flux_clamp_max ({flux_clamp_max}).")
        # Softclip is hard-coded to [0, 1] (0.5*(tanh(z)+1)); a custom clamp
        # range only makes sense for the gradient-conserving clamp branch.
        if flux_activation == "softclip" and (
            self._flux_clamp_min != 0.0 or self._flux_clamp_max != 1.0
        ):
            raise ValueError(
                "softclip flux_activation only produces [0, 1]; non-default "
                "flux_clamp_min/max requires flux_activation='clamp'.")
        # String selector for the LocationEncoding kind ("frequency" or
        # "hashgrid"); BaseRadiationPredictionModelPyImpl parses and converts
        # to the C++ enum, so we just forward the string verbatim. Validated
        # locally to give a Python-level error before crossing the binding.
        if location_encoding_kind.lower() not in ("frequency", "fourier", "sinusoidal", "hashgrid", "hash", "hash_grid"):
            raise ValueError(
                f"location_encoding_kind must be 'frequency' or 'hashgrid', got {location_encoding_kind!r}")
        self._location_encoding_kind = location_encoding_kind
        # Trunk depth (n_hidden_layers of mlp_block + mlp_post). 1 = historic
        # default. 0 makes the trunk a single Linear each → 4 Linears on the
        # flux path; pairs well with hashgrid (the encoding already provides
        # high-frequency features, so a deep trunk is wasted and harder to
        # train in fp16).
        self._trunk_hidden_layers = int(trunk_hidden_layers)
        # Beam-encoding fusion: "film" (affine) or "gated" (bounded hardsigmoid
        # gate). "gated" is the intended pairing with hashgrid, whose
        # discontinuous features an affine FiLM conditions poorly.
        if beam_fusion not in ("film", "gated"):
            raise ValueError(f"beam_fusion must be 'film' or 'gated', got {beam_fusion!r}")
        self._beam_fusion = beam_fusion
        self.model = _TcnnModule(rfnn.BaseRadiationPredictionModel(
            d_model=d_model,
            location_encoding_dim=location_encoding_dims,
            flux_offset=self._flux_offset,
            flux_activation=0 if flux_activation == "clamp" else 1,
            location_encoding_kind=location_encoding_kind,
            flux_clamp_min=self._flux_clamp_min,
            flux_clamp_max=self._flux_clamp_max,
            trunk_hidden_layers=self._trunk_hidden_layers,
            beam_fusion=beam_fusion,
        ))
        self.encoder = _TcnnModule(rfnn.PBRFBeamEncoder(
            spectrum_dim=in_spectra_dim,
            d_model=d_model
        ))
        # Preinit to None
        self.model.weights.grad = None
        self.encoder.weights.grad = None
        # Head/trunk split for DB-MTL. The fused `model.weights` blob holds the
        # shared trunk (weights[:trunk_len]) followed by the spectrum+flux heads
        # (weights[trunk_len:]). `trunk_weights()`/`head_weights()` are
        # storage-sharing PyTorch views exposed by the C++ model. DB-MTL restricts
        # its per-task gradient norm to the trunk slice (see
        # `_shared_parameter_limits`) so it mirrors the pure-Python head-exclusion
        # instead of up-weighting the (head-dominated) spectrum task and NaN-ing.
        try:
            self._trunk_param_len = int(self.model._cpp.trunk_weights().numel())
        except Exception:
            self._trunk_param_len = None
        self._max_lr = float(max_lr)
        # NOTE: fp16 loss scaling is NOT done here. The C++ autograd bridge
        # (`GenericModelFunction::backward`, src/binding_utils.cu) already
        # multiplies dL/dy by LOSS_SCALE=128 before the fp16 backward and
        # divides both the weight- and input-gradient by 128 before returning,
        # so the fp32 masters / Adam already see true-magnitude gradients with
        # no underflow and no compounding across the stacked encoder→model
        # bridges. Adding a second Python-side scale would only risk fp16
        # overflow in dL/y for zero benefit (Adam is scale-invariant).

        self.location_encoding_dims = location_encoding_dims
        self.d_model = d_model
        self.in_spectra_dim = in_spectra_dim

        self.flux_loss_name = flux_loss
        self.spectrum_loss_name = spectrum_loss

        self._flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)

    def get_core_model(self) -> nn.Module:
        return self

    # ── Head / trunk views (storage-sharing) for DB-MTL + inspection ──────────
    @property
    def trunk_weights(self) -> Tensor:
        """View of the shared-trunk slice of the fused weights (weights[:trunk_len])."""
        return self.model._cpp.trunk_weights()

    @property
    def head_weights(self) -> Tensor:
        """View of the output-head slice of the fused weights (weights[trunk_len:])."""
        return self.model._cpp.head_weights()

    @property
    def trunk_grad(self) -> Union[Tensor, None]:
        g = self.model.weights.grad
        return None if (g is None or self._trunk_param_len is None) else g[:self._trunk_param_len]

    @property
    def head_grad(self) -> Union[Tensor, None]:
        g = self.model.weights.grad
        return None if (g is None or self._trunk_param_len is None) else g[self._trunk_param_len:]

    def _shared_parameter_limits(self, shared_params):
        """Restrict the DB-MTL per-task gradient norm to the shared TRUNK slice of
        the fused `model.weights` (excludes the spectrum/flux heads that live in
        the same blob); `encoder.weights` is wholly shared, so it keeps None."""
        model_w = self.model.weights
        return [self._trunk_param_len if (self._trunk_param_len is not None and p is model_w) else None
                for p in shared_params]

    def on_after_backward(self):
        # InstantNGP-style fp32 master weights: after each backward (including
        # every micro-batch when accumulate_grad_batches > 1), the C++ autograd
        # Function has accumulated fp16 gradients into each `_TcnnModule`'s
        # `weights.grad`. Transfer them (cast to fp32) into the corresponding
        # `fp32_master.grad`, accumulating across micro-batches in fp32 for
        # higher precision, and then clear the fp16 grad so a subsequent
        # backward does not double-count.
        #
        # Fires BEFORE Lightning's `configure_gradient_clipping` and
        # `on_before_optimizer_step`, so the fp32 master grad is correctly
        # populated when gradient clipping (which operates on the optimizer's
        # param_groups → fp32 masters) runs.
        for m in self.modules():
            if isinstance(m, _TcnnModule):
                fp16w = m._parameters["weights"]
                if fp16w.grad is None:
                    continue
                # The C++ bridge already returns true-magnitude (un-scaled)
                # gradients (it divides by LOSS_SCALE before returning), so no
                # further unscaling is needed here.
                grad_fp32 = fp16w.grad.detach().to(torch.float32)
                if m.fp32_master.grad is None:
                    m.fp32_master.grad = grad_fp32.clone()
                else:
                    # Accumulate (not copy) so micro-batch grads sum in fp32.
                    m.fp32_master.grad.add_(grad_fp32)
                # Clear so the next backward starts at zero. (The optimizer
                # is bound to fp32_master, not to fp16 weights, so leaving
                # the fp16 grad set would orphan it on the next step.)
                fp16w.grad = None

    def configure_optimizers(self):
        # AdamW(lr~1e-3-clamped, betas=(0.9, 0.99), eps=1e-15, wd=1e-6) on the
        # fp32 masters, with warmup(~1 epoch)+cosine decay (see lr_peak note and
        # the schedule block below).
        #
        # Reference: NVlabs/instant-ngp `configs/nerf/base.json` + their
        # tiny-cuda-nn `optimizers/adam.h` (templated as <STATE_T=fp32,
        # PARAMS_T=fp16> — fp32 master with fp16 cast on every step).
        # torch-ngp `main_nerf.py:80-87` is the verbatim Python port.
        #
        # Critical: this recipe is ONLY safe because `_TcnnModule` now keeps a
        # **fp32 master** (`fp32_master` buffer, requires_grad=True) alongside
        # the fp16 `weights`. The optimizer operates on the fp32 master, so:
        #   • Adam state (exp_avg, exp_avg_sq) is fp32 → eps=1e-15 is
        #     representable (fp32 ULP near 1.0 ≈ 1.2e-7).
        #   • Sub-fp16-ULP updates (e.g. 5e-6) accumulate in fp32 master
        #     across many steps until they cross a fp16 ULP boundary, then
        #     bump the fp16 weight by one ULP. Without the master, ~56% of
        #     Adam's small updates round to 0 (documented in
        #     pbrfnetcpp-nan-status.md 2026-05-18).
        #   • fp32-master → fp16-weight sync happens in `_TcnnModule`'s
        #     forward pre-hook, just before the JIT-fused kernel reads the
        #     pointer.
        #   • fp16-weight → fp32-master grad transfer happens in
        #     `on_after_backward` below, in fp32 (accumulates across micro-
        #     batches when `accumulate_grad_batches > 1`).
        #
        # lr_peak (default `max_lr=1e-3`): the fp16-fused weights take a ~10×
        # smaller *effective* step per nominal lr than the pure-Python model, so
        # the old `max_lr=1e-4` starved flux learning — flux plateaued at ~0.02
        # (fourier) / ~0.08 (hashgrid) while spectrum (beam-driven, not spatial)
        # learned fine and masked it. Raising to 1e-3 unsticks flux with NO NaN
        # (the old "~1.2e-4 overflow ceiling" does not hold with the current
        # RadFiled3D / kernel); 3e-3 oscillates. Root-cause runs 2026-06-05,
        # wandb `radfield-flux-debug` E4/E5/E7 — see
        # claude-notes/flux-plateau-experiments.md.
        effective_lr = min(max(float(self._lr), 1e-5), self._max_lr)

        # The optimizer operates on the fp32 masters, NOT on the fp16
        # parameters returned by `self.parameters()`. fp16 grads accumulate on
        # the fp16 weights via the autograd Function in the C++ bridge; the
        # `on_after_backward` hook transfers those grads (cast to fp32) into
        # `fp32_master.grad`, then clears the fp16 grad to keep things in sync.
        masters = []
        for m in self.modules():
            if isinstance(m, _TcnnModule):
                # requires_grad was set in _TcnnModule.__init__ — assert it
                # so a future state_dict restore that resets the flag is loud.
                assert m.fp32_master.requires_grad, "_TcnnModule.fp32_master must have requires_grad=True"
                masters.append(m.fp32_master)

        optimizer = torch.optim.AdamW(
            [{'params': masters,
              'lr': effective_lr, 'initial_lr': effective_lr,
              'weight_decay': 1e-6,        # NGP l2_reg (configs/nerf/base.json)
              'eps': 1e-15}],              # NGP exact — works because state is fp32
            betas=(0.9, 0.99)              # NGP exact (vs PyTorch default 0.999)
        )

        # Post-step hook: AFTER each optimizer.step() updates the fp32 masters,
        # in-place sync them into the corresponding fp16 cpp.weights so the
        # NEXT forward call's JIT-fused kernel reads the freshly-updated
        # weights. The hook is on the optimizer (not on _TcnnModule.forward)
        # because the latter would in-place modify cpp.weights mid-graph in
        # multi-forward-then-backward patterns (e.g. chunked-forward grad
        # accumulation tests), which trips autograd's version-counter check
        # (`one of the variables needed for gradient computation has been
        # modified by an inplace operation`).
        wrapper = self
        def _sync_masters_to_fp16(_opt, _args, _kwargs):
            for sub in wrapper.modules():
                if isinstance(sub, _TcnnModule):
                    with torch.no_grad():
                        sub._parameters["weights"].copy_(
                            sub.fp32_master.to(torch.float16)
                        )
        optimizer.register_step_post_hook(_sync_masters_to_fp16)

        def get_accumulate_grad_batches(trainer) -> int:
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

        total_opt_steps = int(self.trainer.estimated_stepping_batches)
        acc_batches = max(1, get_accumulate_grad_batches(self.trainer))
        if not torch.isfinite(torch.tensor(total_opt_steps)) or total_opt_steps <= 0:
            total_opt_steps = 1000  # avoid zero division
        total_opt_steps = int(max(1, total_opt_steps / acc_batches))

        # Step-wise warmup (~1 epoch) followed by cosine decay — the same
        # schedule the pure-Python RFNetBase uses, which holds lr near the peak
        # far longer than the old NGP power-decay (0.1^(step/T), already halved
        # by ~ep6). With the higher peak lr this gives a smooth monotonic flux
        # climb (E7); power-decay + low lr was the prior plateau.
        max_epochs = int(max(self.trainer.max_epochs, 1))
        steps_per_epoch = int(math.ceil(total_opt_steps / max_epochs))
        warmup_steps = int(min(max(steps_per_epoch, 1), max(1, total_opt_steps - 1)))
        cosine_steps = int(max(1, total_opt_steps - warmup_steps))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-5 / effective_lr, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=5e-6
        )
        schedule = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return [optimizer], [{
            "scheduler": schedule,
            "interval": "step",
            "monitor": "train_loss",
            "name": "warmup+cosine",
        }]

    def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
        B = batch.direction.size(0)
        granularity = 256
        g = B % granularity
        direction = batch.direction
        spectrum  = resample_histogram_bilinear(batch.spectrum, self.in_spectra_dim)
        distance  = batch.origin
        if g != 0:
            padded_B = (B + granularity - 1) // granularity * granularity
            pad_size = padded_B - B
            direction = nn.functional.pad(direction, (0, 0, 0, pad_size)) # Pad batch dimension
            spectrum  = nn.functional.pad(spectrum, (0, 0, 0, pad_size)) # Pad batch dimension
            distance  = nn.functional.pad(distance, (0, 0, 0, pad_size)) # Pad batch dimension
        return self.encoder.forward(direction, distance, spectrum)[:B]

    def forward(self, batch: PositionalInput, global_parameters: Union[Tensor, None, list] = None) -> RadiationField:
        params_enc = self.encode_additional_parameters(batch) if global_parameters is None else global_parameters
        B = batch.direction.size(0)
        granularity = 256
        g = B % granularity
        params = params_enc
        location = batch.position
        if g != 0:
            padded_B = (B + granularity - 1) // granularity * granularity
            pad_size = padded_B - B
            params = nn.functional.pad(params, (0, 0, 0, pad_size)) # Pad batch dimension
            location = nn.functional.pad(location, (0, 0, 0, pad_size)) # Pad batch dimension

        flux, spectrum = self.model.forward(location, params)
        f_nans = torch.isnan(flux).sum()
        s_nans = torch.isnan(spectrum).sum()
        if f_nans > 0 or s_nans > 0:
            # Pinpoint the ORIGIN: is the corruption already in the inputs /
            # weights (-> upstream optimizer/grad problem) or only in the
            # flux head's output (-> localized to that projection slice)?
            def st(name, t):
                tf = t.detach().float()
                fin = torch.isfinite(tf)
                return (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
                        f"finite={int(fin.sum())}/{tf.numel()} "
                        f"nan={int(torch.isnan(tf).sum())} inf={int(torch.isinf(tf).sum())} "
                        f"min={tf[fin].min().item() if fin.any() else 'n/a'} "
                        f"max={tf[fin].max().item() if fin.any() else 'n/a'}")
            report = "\n  ".join([
                f"training={self.training} B={B} pad={'yes' if g != 0 else 'no'}",
                st("location", location), st("params", params),
                st("model.weights", self.model.weights),
                st("encoder.weights", self.encoder.weights),
                st("flux(out)", flux),
                st("spectrum(out)", spectrum),
            ])
            raise Exception(
                f"Flux NaNs: {f_nans}; Spec NaNs: {s_nans}\n  {report}")
        # Single-head output: one joined per-volume-relative flux + joined
        # spectrum, carried on the scatter slot. ``direct_beam`` is None.
        flux = flux[:B].squeeze(-1).float()
        spec = spectrum[:B].float()
        return RadiationField(
            scatter_field=RadiationFieldChannel(
                flux=flux,
                spectrum=spec,
            ),
            direct_beam=None,
        )

    def get_custom_parameters(self):
        return {
            "location_encoding_dims": self.location_encoding_dims,
            "d_model": self.d_model,
            "in_spectra_dim": self.in_spectra_dim,
            "flux_loss": self.flux_loss_name,
            "spectrum_loss": self.spectrum_loss_name,
            "randomize_voxel_location_in_training": self.randomize_voxel_location_in_training,
            "voxels_centered_around_origin": self.voxels_centered_around_origin,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None,
            "seed": self.seed,
            "max_lr": self._max_lr,
            "flux_offset": self._flux_offset,
            "flux_activation": self._flux_activation_kind,
            "flux_clamp_min": self._flux_clamp_min,
            "flux_clamp_max": self._flux_clamp_max,
            "location_encoding_kind": self._location_encoding_kind,
            "trunk_hidden_layers": self._trunk_hidden_layers,
            "beam_fusion": self._beam_fusion,
        }


class SPERFNetCPP(PBRFNetCPP):
    """Distance-less variant of PBRFNetCPP, mirroring the pure-Python SPERFNet
    (radfield3dnn/models/nerf.py): the beam encoder takes only direction and
    spectrum, no beam distance. Used to test PBRFNetCPP's network on the
    simpler fixed-distance DS02 dataset (where the beam distance is constant
    and would only add a uninformative input dimension).

    The trunk and decoder/activation pipeline (BaseRadiationPredictionModel
    with the in-kernel Sigmoid + Softplus+sum-norm bake-in) are unchanged;
    only the beam encoder is swapped (rfnn.SPERFBeamEncoder instead of
    rfnn.PBRFBeamEncoder) and `encode_additional_parameters` no longer passes
    distance.
    """
    __model_name__ = "SPERFNetCPP"

    def __init__(self, location_encoding_dims=10, in_spectra_dim=32, d_model=256,
                 flux_loss="L1Loss", spectrum_loss="HistogramLoss",
                 learning_rate: float = 1e-3,
                 randomize_voxel_location_in_training: bool = True,
                 voxels_centered_around_origin: bool = True,
                 normalizer=None, seed: int = 1337, max_lr: float = 1e-4,
                 flux_offset: float = 0.5, flux_activation: str = "clamp",
                 location_encoding_kind: str = "frequency",
                 flux_clamp_min: float = 0.0, flux_clamp_max: float = 1.0,
                 trunk_hidden_layers: int = 1, beam_fusion: str = "film"):
        super().__init__(
            location_encoding_dims=location_encoding_dims,
            in_spectra_dim=in_spectra_dim,
            d_model=d_model,
            flux_loss=flux_loss,
            spectrum_loss=spectrum_loss,
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            normalizer=normalizer,
            seed=seed,
            max_lr=max_lr,
            flux_offset=flux_offset,
            flux_activation=flux_activation,
            location_encoding_kind=location_encoding_kind,
            flux_clamp_min=flux_clamp_min,
            flux_clamp_max=flux_clamp_max,
            trunk_hidden_layers=trunk_hidden_layers,
            beam_fusion=beam_fusion,
        )
        # Replace the PBRFBeamEncoder created by the base ctor with the
        # distance-less SPERFBeamEncoder. Re-seed the global torch RNG so the
        # tcnn pcg32 init draw is deterministic with respect to `seed` (same
        # contract the base ctor honours for its own encoder/model).
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.encoder = _TcnnModule(rfnn.SPERFBeamEncoder(
            spectrum_dim=in_spectra_dim,
            d_model=d_model,
        ))
        # See PBRFNetCPP.__init__: prevent AccumulateGrad from silently
        # overwriting the pre-allocated .grad on chunked-forward backwards.
        self.encoder.weights.grad = None

    def encode_additional_parameters(self, batch: PositionalInput) -> Tensor:
        # Same batch-granularity padding as PBRFNetCPP, but the encoder takes
        # only direction + spectrum — batch.origin (distance) is intentionally
        # ignored for the simpler fixed-distance DS02 dataset.
        B = batch.direction.size(0)
        granularity = 256
        g = B % granularity
        direction = batch.direction
        spectrum  = resample_histogram_bilinear(batch.spectrum, self.in_spectra_dim)
        if g != 0:
            padded_B = (B + granularity - 1) // granularity * granularity
            pad_size = padded_B - B
            direction = nn.functional.pad(direction, (0, 0, 0, pad_size))
            spectrum  = nn.functional.pad(spectrum,  (0, 0, 0, pad_size))
        return self.encoder.forward(direction, spectrum)[:B]

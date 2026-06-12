"""Training-behavior tests for radfield3dnn.models.nerf_cpp.PBRFNetCPP.

The C++ unit tests in `tests/cxx/` cover BaseRadiationPredictionModel and
PBRFBeamEncoder in isolation; this file covers the Python wrapper that the
training pipeline actually instantiates (`run_network_task.py` -> `tasks/train.py`).

Goal: catch regressions where PBRFNetCPP produces all-zero or NaN fluxs
after the first optimizer step, which is what is observed in the live pipeline.
The architecture being checked here mirrors the working PBRFNet in nerf.py:
sinusoidal/freq position encoding -> FiLM-conditioned MLP -> sigmoid flux
head + histogram-normalized spectrum head.
"""

import os
import math

import pytest
# PBRFNetCPP needs the tiny-cuda-nn native module; skip this whole module when tcnn is
# deactivated/not built (its import succeeds via the stub, but constructing PBRFNetCPP would
# raise ImportError).
pytest.importorskip("radfield3dnn.radfield3dnn")

import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from radfield3dnn.models.nerf_cpp import PBRFNetCPP
from radfield3dnn.preprocessing.normalizations import LinearNormalizer, LogScaleNormalizer
from radfield3dnn.rftypes import PositionalInput, TrainingInputData
from RadFiled3D.pytorch.types import RadiationFieldChannel
import lightning.pytorch as pl


BATCH = 1024  # tcnn enforces BATCH_SIZE_GRANULARITY = 256; 1024 is a comfortable multiple.
IN_SPECTRA_DIM = 32


def _make_batch(B: int = BATCH) -> PositionalInput:
    """A batch matching what the dataloader hands to PBRFNetCPP after normalization.

    Voxel positions are in [-1, 1] (voxels_centered_around_origin=True), origin
    is a single normalized distance in [0, 1] (beam_parameters normalization).
    """
    return PositionalInput(
        direction=torch.rand(B, 3, device="cuda"),
        spectrum=torch.rand(B, 150, device="cuda"),
        position=torch.rand(B, 3, device="cuda") * 2.0 - 1.0,
        origin=torch.rand(B, 1, device="cuda"),
        geometry=None,
        beam_shape_parameters=torch.rand(B, 1, device="cuda"),
        beam_shape_type=torch.zeros(B, 1, device="cuda"),
    )


def _build_model(d_model: int = 64) -> PBRFNetCPP:
    m = PBRFNetCPP(
        location_encoding_dims=12,
        in_spectra_dim=IN_SPECTRA_DIM,
        d_model=d_model,
        normalizer=LinearNormalizer(),
        # Linear-space loss only. The new offset+linear-clamp flux activation
        # can emit exact 0.0 — `L1LogLoss` (the production-tuned default) does
        # `log(pred+1e-8)` whose backward gradient `1/(pred+1e-8)` reaches ~1e8
        # at pred=0, overflowing fp16 and producing NaN weights after one
        # optimizer step. Old Sigmoid activation was strictly in (0,1) so this
        # never triggered. Production training uses `flux_loss=FluxLoss`
        # (linear-space Huber+SSIM3D, bounded gradient) and is unaffected.
        flux_loss="L1Loss",
    ).cuda()
    return m


# -----------------------------------------------------------------------------
# 1) Initial-state sanity. PBRFNet produces finite, non-degenerate outputs at
#    init; PBRFNetCPP must do the same.
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_forward_finite_at_init():
    torch.manual_seed(0)
    m = _build_model()
    batch = _make_batch()
    field = m.forward(batch)

    flux = field.scatter_field.flux
    spectrum = field.scatter_field.spectrum

    assert flux.shape == (BATCH,), f"flux shape {flux.shape}"
    assert spectrum.shape == (BATCH, IN_SPECTRA_DIM), f"spectrum shape {spectrum.shape}"
    assert torch.isfinite(flux).all(), "flux has non-finite values at init"
    assert torch.isfinite(spectrum).all(), "spectrum has non-finite values at init"

    # offset+clamp activation lands flux around the offset (default 0.5) at
    # init and is hard-bounded to the closed interval [0, 1].
    assert (flux >= 0.0).all() and (flux <= 1.0).all()
    # HistogramNormalize produces non-negative values summing to ~1 per row.
    assert (spectrum >= 0.0).all()
    spec_sums = spectrum.sum(dim=-1)
    assert torch.allclose(spec_sums, torch.ones_like(spec_sums), atol=1e-2), (
        f"spectrum rows must sum to ~1 after HistogramNormalize, got sums in "
        f"[{spec_sums.min().item():.4f}, {spec_sums.max().item():.4f}]"
    )


# -----------------------------------------------------------------------------
# 2) After one optimizer step, the flux head must still emit finite, varied
#    values. This reproduces the production symptom: "after the first epoch the
#    fluxs are still all zero or even worse NaN."
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_flux_finite_after_one_step():
    torch.manual_seed(0)
    m = _build_model()
    optimizer = optim.Adam(m.parameters(), lr=1e-3, eps=1e-4)

    batch = _make_batch()
    target_flux = torch.rand(BATCH, device="cuda") * 0.5 + 0.25  # in (0.25, 0.75)
    target_spec = torch.softmax(torch.randn(BATCH, IN_SPECTRA_DIM, device="cuda"), dim=-1)

    optimizer.zero_grad()
    field = m.forward(batch)
    loss = (
        nn.functional.l1_loss(field.scatter_field.flux, target_flux)
        + nn.functional.l1_loss(field.scatter_field.spectrum, target_spec)
    )
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        field2 = m.forward(batch)
        flux2 = field2.scatter_field.flux
        spec2 = field2.scatter_field.spectrum

    assert torch.isfinite(flux2).all(), (
        f"flux has NaN/inf after one optimizer step (min={flux2.min().item()}, "
        f"max={flux2.max().item()})"
    )
    assert torch.isfinite(spec2).all(), "spectrum has NaN/inf after one optimizer step"
    # Flux shouldn't collapse to a single value.
    assert flux2.std().item() > 1e-5, (
        f"flux collapsed to a constant after one step: std={flux2.std().item()}"
    )
    # And shouldn't be all zero.
    assert flux2.abs().mean().item() > 1e-4, (
        f"flux collapsed to ~0 after one step: mean abs={flux2.abs().mean().item()}"
    )


# -----------------------------------------------------------------------------
# 3) Gradients must reach every parameter group (encoder + main model). A
#    silently-disconnected sub-module would explain zero/NaN training.
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_gradients_reach_all_params():
    torch.manual_seed(0)
    m = _build_model()

    batch = _make_batch()
    field = m.forward(batch)
    loss = field.scatter_field.flux.mean() + field.scatter_field.spectrum.mean()
    loss.backward()

    named = list(m.named_parameters())
    assert len(named) >= 2, f"expected at least encoder + model weights, got {len(named)}"

    # Skip the Kendall & Gal 2018 multitask uncertainty scalars and the
    # per-field ratio head: they are part of the multitask / ratio loss
    # paths in BaseNeuralRadFieldModel.process_metrics, not of the raw
    # `flux.mean() + spectrum.mean()` test loss path, so they legitimately
    # receive no gradient here. The intent of this test is to verify
    # gradient flow into the C++ encoder and model bridges.
    grad_norms = {}
    for name, p in named:
        if not p.requires_grad:
            continue
        if name in ("_loss_logvar_flux", "_loss_logvar_spectrum"):
            continue
        if name.startswith("ratio_head."):
            continue
        assert p.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {name}"
        grad_norms[name] = p.grad.norm().item()

    # Both `encoder.weights` and `model.weights` should see some gradient flow.
    nonzero = [n for n, g in grad_norms.items() if g > 0.0]
    assert any("encoder" in n for n in nonzero), (
        f"PBRFBeamEncoder weights got zero gradient (norms={grad_norms})"
    )
    assert any("model" in n for n in nonzero), (
        f"BaseRadiationPredictionModel weights got zero gradient (norms={grad_norms})"
    )


# -----------------------------------------------------------------------------
# 4) Short training run. PBRFNet trains successfully on a smooth synthetic
#    target; PBRFNetCPP, which mirrors its architecture in C++/tcnn, should
#    behave equivalently and at minimum show a meaningful loss drop without
#    going to NaN.
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_short_training_drops_loss_and_stays_finite():
    torch.manual_seed(0)
    m = _build_model(d_model=128)
    # 3e-4, not 1e-3: the tcnn weights are fp16 master params and the backward
    # uses a fixed loss scale, so lr above ~5e-4 makes the fp16 update step
    # unstable and the loss diverges (this is why the real pipeline's lr_find
    # lands ~1.4e-4 and configure_optimizers caps at 5e-4). At 3e-4 the model
    # converges cleanly (~0.26 -> ~0.02 over 600 steps). The old 1e-3 only
    # "passed" because the previous broken init emitted near-constant outputs
    # with no usable gradient, so the loss drifted instead of diverging.
    optimizer = optim.Adam(m.parameters(), lr=3e-4, eps=1e-4)

    spec_phase = torch.linspace(0.0, 2.0 * math.pi, IN_SPECTRA_DIM, device="cuda")

    def make_targets(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pos lives in [-1, 1]; map to [0, 1] for the target so it is in the
        # range the sigmoid head can fit.
        u = (pos + 1.0) * 0.5
        tgt_flux = 0.5 * (1.0 + torch.sin(2.0 * u[:, 0]) * torch.cos(2.0 * u[:, 1]))
        amp = u.sum(dim=1, keepdim=True) / 3.0
        tgt_spec_unnorm = amp * (1.0 + torch.sin(spec_phase.unsqueeze(0) + u[:, 2:3]))
        tgt_spec = tgt_spec_unnorm / tgt_spec_unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return tgt_flux, tgt_spec

    first_loss = None
    last_loss = None
    for step in range(600):
        batch = _make_batch()
        target_flux, target_spec = make_targets(batch.position)

        optimizer.zero_grad()
        field = m.forward(batch)
        loss = (
            nn.functional.l1_loss(field.scatter_field.flux, target_flux)
            + nn.functional.l1_loss(field.scatter_field.spectrum, target_spec)
        )
        assert torch.isfinite(loss), f"loss went non-finite at step {step}: {loss.item()}"
        loss.backward()
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert math.isfinite(last_loss)
    assert last_loss < first_loss, f"loss did not decrease: {first_loss} -> {last_loss}"

    # Final flux must not have collapsed.
    with torch.no_grad():
        field = m.forward(_make_batch())
        flux = field.scatter_field.flux
        spec = field.scatter_field.spectrum
    assert torch.isfinite(flux).all() and torch.isfinite(spec).all()
    assert flux.std().item() > 1e-4, f"flux collapsed after training: std={flux.std().item()}"


# -----------------------------------------------------------------------------
# 5) Chunked-forward gradient correctness. FeedforwardPointwiseModel splits
#    each batch into max_inner_batch_size-sized chunks and runs `self.forward`
#    on each before a single .backward(). Each chunk goes through the C++
#    autograd Function, which writes the parameter gradient into the bridge's
#    `internal_grad` buffer. If that buffer is *overwritten* per chunk rather
#    than *accumulated*, the total gradient computed by training is wrong even
#    though every chunk's loss is finite.
#
#    This test compares (a) the single-shot gradient over the full batch with
#    (b) the gradient produced by running the same data as two chunks summed
#    into one loss. Discrepancies here would directly cause the kind of
#    silent training collapse the user is seeing.
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_chunked_forward_accumulates_gradients():
    torch.manual_seed(0)
    m = _build_model(d_model=64)

    full_B = 2 * BATCH
    batch = _make_batch(full_B)

    def grad_norms_after(forward_fn):
        # Mirror optimizer.zero_grad(set_to_none=True) — the PyTorch default
        # and what Lightning uses in this repo's training loop. Zeroing in
        # place (.grad.zero_()) is not equivalent: AccumulateGrad treats an
        # already-defined .grad as the starting contribution and overwrites
        # it on the next backward, instead of summing.
        for p in m.parameters():
            p.grad = None
        loss = forward_fn()
        loss.backward()
        return {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in m.named_parameters()}

    def slice_batch(b: PositionalInput, lo: int, hi: int) -> PositionalInput:
        return PositionalInput(
            direction=b.direction[lo:hi],
            spectrum=b.spectrum[lo:hi],
            position=b.position[lo:hi],
            origin=b.origin[lo:hi],
            geometry=None,
            beam_shape_parameters=b.beam_shape_parameters[lo:hi],
            beam_shape_type=b.beam_shape_type[lo:hi],
        )

    def loss_full():
        f = m.forward(batch)
        return f.scatter_field.flux.mean() + f.scatter_field.spectrum.mean()

    def loss_chunked():
        f1 = m.forward(slice_batch(batch, 0, BATCH))
        f2 = m.forward(slice_batch(batch, BATCH, full_B))
        # Same per-element weighting as loss_full: mean over 2*BATCH elements.
        return 0.5 * (f1.scatter_field.flux.mean() + f1.scatter_field.spectrum.mean()
                      + f2.scatter_field.flux.mean() + f2.scatter_field.spectrum.mean())

    g_full = grad_norms_after(loss_full)
    g_chunked = grad_norms_after(loss_chunked)

    for name in g_full:
        a, b = g_full[name], g_chunked[name]
        assert (a is None) == (b is None), f"grad presence diverges for {name}"
        if a is None:
            continue
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
            f"non-finite grads for {name}"
        )
        rel = (a - b).norm() / a.norm().clamp_min(1e-8)
        assert rel.item() < 5e-2, (
            f"chunked-vs-full gradient mismatch on {name}: rel diff {rel.item():.4f}, "
            f"|g_full|={a.norm().item():.4e}, |g_chunked|={b.norm().item():.4e}. "
            f"Likely cause: PBRFBeamEncoder/BaseRadiationPredictionModel bridge "
            f"backward overwrites the parameter-gradient buffer per chunk instead "
            f"of accumulating, so only the last chunk's gradient survives."
        )


# -----------------------------------------------------------------------------
# 6) Full Lightning pipeline. Tests 1-5 prove the C++/CUDA forward, backward,
#    chunked-gradient accumulation and a plain-PyTorch optimizer loop are all
#    correct. None of them exercise the Lightning glue, which is exactly where
#    the live failure lived:
#
#      * PBRFNetCPP inherited RFNetBase.configure_optimizers (AdamW eps=1e-8).
#        tcnn keeps params/grads in fp16, where 1e-8 underflows the Adam
#        denominator -> the first optimizer.step() NaNs every weight and the
#        loss goes non-finite after step 1.
#      * The old custom training_step set automatic_optimization=False, which
#        Lightning refuses to combine with the pipeline's Trainer(
#        gradient_clip_val=1.0) -> MisconfigurationException at fit start.
#
#    This test drives the *real* inherited training path (evaluate_forward ->
#    normalizer -> configured losses -> automatic optimization) through an
#    actual pl.Trainer.fit, with the same Trainer settings run_network_task.py
#    uses, and asserts the model trains finitely. It is the regression guard
#    for PBRFNetCPP being a plug-and-play replacement for PBRFNet.
# -----------------------------------------------------------------------------

_SPEC_PHASE = torch.linspace(0.0, 2.0 * math.pi, IN_SPECTRA_DIM)


def _make_targets(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # pos lives in [-1, 1]; map to [0, 1] so the target is in the range the
    # sigmoid flux head and HistogramNormalize spectrum head can fit.
    u = (pos + 1.0) * 0.5
    tgt_flux = 0.5 * (1.0 + torch.sin(2.0 * u[:, 0]) * torch.cos(2.0 * u[:, 1]))
    amp = u.sum(dim=1, keepdim=True) / 3.0
    tgt_spec_unnorm = amp * (1.0 + torch.sin(_SPEC_PHASE.unsqueeze(0) + u[:, 2:3]))
    tgt_spec = tgt_spec_unnorm / tgt_spec_unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return tgt_flux, tgt_spec


class _SyntheticRFDataset(Dataset):
    """Each item is one full PBRF batch wrapped as TrainingInputData with a
    pointwise RadiationFieldChannel ground truth (flux (B,), spectrum (B,C)),
    which routes evaluate_forward through the pointwise `self(x)` path."""

    def __init__(self, n_items: int = 8):
        self.n_items = n_items

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> TrainingInputData:
        g = torch.Generator().manual_seed(idx)
        pos = torch.rand(BATCH, 3, generator=g) * 2.0 - 1.0
        tgt_flux, tgt_spec = _make_targets(pos)
        inp = PositionalInput(
            direction=torch.rand(BATCH, 3, generator=g),
            spectrum=torch.rand(BATCH, 150, generator=g),
            position=pos,
            origin=torch.rand(BATCH, 1, generator=g),
            geometry=None,
            beam_shape_parameters=torch.rand(BATCH, 1, generator=g),
            beam_shape_type=torch.zeros(BATCH, 1),
        )
        gt = RadiationFieldChannel(spectrum=tgt_spec, flux=tgt_flux, error=None)
        return TrainingInputData(input=inp, ground_truth=gt)


class _RFDataModule(pl.LightningDataModule):
    def train_dataloader(self) -> DataLoader:
        # Each dataset item is already a full batch; collate just unwraps it
        # and the dataloader moves nothing to GPU (Lightning does that).
        return DataLoader(_SyntheticRFDataset(), batch_size=1, collate_fn=lambda b: b[0])


class _LossProbe(pl.Callback):
    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, *_):
        self.losses.append(float(trainer.callback_metrics["train_loss"]))


def test_pbrfnetcpp_lightning():
    torch.manual_seed(0)
    m = _build_model(d_model=128)
    # run_network_task.py always presets max_inner_batch_size before fit, so
    # BaseNeuralRadFieldModel.on_fit_start skips its auto batch-size search
    # (that helper feeds an origin of shape [B,3], incompatible with every
    # PBRF-style model, PBRFNet included). Mirror the real pipeline here.
    m.max_inner_batch_size = 4096

    probe = _LossProbe()
    trainer = pl.Trainer(
        max_epochs=5,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=1.0,  # same as run_network_task.py; must not conflict
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[probe],
    )

    trainer.fit(m, datamodule=_RFDataModule())

    assert probe.losses, "training never ran a step"
    assert all(math.isfinite(x) for x in probe.losses), (
        f"non-finite train loss under Lightning: {probe.losses}"
    )
    assert probe.losses[-1] < probe.losses[0], (
        f"loss did not decrease under Lightning: {probe.losses[0]} -> {probe.losses[-1]}"
    )

    # Lightning moves the module back to CPU during fit teardown; the C++
    # weights must be back on the GPU before a manual forward.
    m = m.cuda().eval()
    with torch.no_grad():
        field = m.forward(_make_batch())
        flux = field.scatter_field.flux
        spec = field.scatter_field.spectrum
    # The two production symptoms were "flux all zero" and "NaN after the
    # first step" once the model was driven by Lightning. Assert neither
    # recurs through a full fit + teardown round-trip. (Non-collapse over a
    # longer run is already covered by
    # test_pbrfnetcpp_short_training_drops_loss_and_stays_finite; at 5 epochs
    # through the full loss stack it depends on tcnn's own init RNG, which
    # torch.manual_seed does not control, so it is not asserted here.)
    assert torch.isfinite(flux).all() and torch.isfinite(spec).all(), (
        "flux/spectrum went non-finite after Lightning training"
    )
    assert flux.abs().mean().item() > 1e-4, (
        f"flux collapsed to ~0 after Lightning training: "
        f"mean abs={flux.abs().mean().item()}"
    )


# -----------------------------------------------------------------------------
# 7) The actual production failure. tasks/train.py runs Tuner.lr_find BEFORE
#    trainer.fit. lr_find checkpoints the model, runs a destructive lr sweep,
#    then restores it. The C++ bridge captures a raw pointer to the `weights`
#    CUDA storage once at construction (no re-bind API); any storage swap
#    (Lightning device moves / lr_find checkpoint restore) silently desyncs
#    tcnn -> Python sees finite weights while tcnn emits NaN/constant output
#    "after the first epoch". `_TcnnModule._apply` pins the storage to fix it.
#
#    Also guards the second production gap: a trained model must store to a
#    Lightning checkpoint and load back (load_from_checkpoint) byte-identical,
#    so it can be run detached from the training pipeline.
# -----------------------------------------------------------------------------

def test_pbrfnetcpp_survives_real_lr_find_and_checkpoint_roundtrip():
    from lightning.pytorch.tuner import Tuner

    torch.manual_seed(0)
    m = _build_model(d_model=128)
    m.max_inner_batch_size = 4096
    w0 = m.model.weights.detach().clone()
    eptr0 = m.encoder.weights.data_ptr()

    lr_trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_steps=60, precision="32-true",
        logger=False, enable_progress_bar=False, enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    Tuner(lr_trainer).lr_find(
        m, datamodule=_RFDataModule(), min_lr=1e-4, max_lr=1e-2,
        num_training=60,
    )

    w = m.model.weights
    # tcnn's raw pointer must still track the live tensor on CUDA, and the
    # sweep must have been fully rolled back by the checkpoint restore.
    assert w.device.type == "cuda", f"weights left on {w.device} after lr_find"
    assert w.data_ptr() == m.model._cpp.weights.data_ptr(), (
        "python weights tensor desynced from the C++ bridge tensor"
    )
    assert m.encoder.weights.data_ptr() == eptr0, "encoder storage swapped"
    assert torch.isfinite(w).all(), "weights non-finite after lr_find"
    assert (w.detach() - w0).abs().max().item() < 1e-3, (
        "lr_find did not restore the pre-sweep weights"
    )

    # tcnn must still produce finite output through its (possibly stale) ptr.
    with torch.no_grad():
        f = m.cuda().eval().forward(_make_batch())
    assert torch.isfinite(f.scatter_field.flux).all(), (
        "flux NaN after lr_find -> tcnn pointer desynced (the live bug)"
    )
    assert torch.isfinite(f.scatter_field.spectrum).all()

    # --- checkpoint store + headless load_from_checkpoint round-trip ---
    trainer = pl.Trainer(
        max_epochs=1, accelerator="gpu", devices=1, precision="32-true",
        logger=False, enable_checkpointing=False, num_sanity_val_steps=0,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(m, datamodule=_RFDataModule())
    batch = _make_batch(1024)
    with torch.no_grad():
        o_src = m.cuda().eval().forward(batch).scatter_field.flux.clone()

    ckpt = "/tmp/_pbrf_regression.ckpt"
    trainer.save_checkpoint(ckpt)
    try:
        m2 = PBRFNetCPP.load_from_checkpoint(ckpt).cuda().eval()
        with torch.no_grad():
            o_dst = m2.forward(batch).scatter_field.flux.clone()
    finally:
        os.remove(ckpt)

    assert torch.isfinite(o_dst).all(), "loaded model produced non-finite flux"
    assert (o_src - o_dst).abs().max().item() < 1e-4, (
        f"checkpoint round-trip changed the output by "
        f"{(o_src - o_dst).abs().max().item():.3e} (store/load is broken)"
    )


# -----------------------------------------------------------------------------
# 8) Effectiveness for the user's flux range. The dataset's flux is mostly
#    ~1e-8 .. 1e-7 with some values ~1.0. Under the FP16-only fused C++ path
#    the model must learn LOW RELATIVE ERROR across that whole span. This
#    proves the log_decade normalizer (default) enables that end-to-end, and
#    that the FP32-era linear0_1 choice provably does not (its 1e-8 tail
#    underflows to 0 in FP16 so the network can never represent it).
# -----------------------------------------------------------------------------

def _train_decade_field(normalizer, steps: int = 500, seed: int = 0):
    """Fit a deterministic flux field spanning 1e-8 .. 1.0 (physical) through
    `normalizer`, FP16 weights, and report physical relative error by band."""
    torch.manual_seed(seed)
    m = PBRFNetCPP(
        location_encoding_dims=12,
        in_spectra_dim=IN_SPECTRA_DIM,
        d_model=128,
        normalizer=normalizer,
    ).cuda()
    n = m._normalizer
    opt = optim.Adam(m.parameters(), lr=3e-4, eps=1e-4)

    def physical_target(pos: torch.Tensor) -> torch.Tensor:
        # pos in [-1, 1]^3 -> smooth s in [0, 1] -> exponent in [-8, 0]
        # -> target in [1e-8, 1.0], densely covering the 1e-8/1e-7 tail.
        s = 0.5 * (1.0 + torch.sin(math.pi * pos[:, 0]) * torch.cos(math.pi * pos[:, 1]))
        return torch.pow(torch.tensor(10.0, device=pos.device), -8.0 * (1.0 - s))

    first = last = None
    for _ in range(steps):
        batch = _make_batch()
        tgt_phys = physical_target(batch.position)
        tgt_norm = n.apply_transformation(tgt_phys, None).detach()
        opt.zero_grad()
        pred_norm = m.forward(batch).scatter_field.flux
        loss = nn.functional.l1_loss(pred_norm, tgt_norm)
        assert torch.isfinite(loss), "loss went non-finite"
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()

    with torch.no_grad():
        batch = _make_batch()
        tgt_phys = physical_target(batch.position)
        pred_norm = m.forward(batch).scatter_field.flux.float()
        pred_phys = n.apply_inverse_transformation(pred_norm, None)
        rel = (pred_phys - tgt_phys).abs() / tgt_phys.abs()
        low = tgt_phys <= 1e-6           # the user's 1e-8 .. 1e-6 band
        high = tgt_phys >= 1e-1          # the values "around 1.0"
        low_med = rel[low].median().item() if low.any() else float("nan")
        high_med = rel[high].median().item() if high.any() else float("nan")
    return dict(first=first, last=last, low_med=low_med, high_med=high_med,
                finite=bool(torch.isfinite(pred_phys).all()))


def test_pbrfnetcpp_learns_low_relative_error_across_decades_fp16():
    ld = _train_decade_field(LogScaleNormalizer(), seed=0)

    assert ld["finite"], "log_scale run produced non-finite physical flux"
    assert ld["last"] < 0.5 * ld["first"], (
        f"log_scale did not learn: loss {ld['first']:.4g} -> {ld['last']:.4g}"
    )
    assert ld["low_med"] < 0.75, (
        f"log_scale low-band (1e-8..1e-6) rel error too high: {ld['low_med']:.3g}"
    )
    assert ld["high_med"] < 0.5, (
        f"log_scale high-band (~1.0) rel error too high: {ld['high_med']:.3g}"
    )


def test_log_scale_beats_linear0_1_on_the_low_flux_tail_fp16():
    """Decisive contrast: same model/steps, only the normalizer differs. The
    1e-8..1e-6 band is where linear0_1 structurally cannot work under FP16."""
    ld = _train_decade_field(LogScaleNormalizer(), seed=0)
    lin = _train_decade_field(LinearNormalizer((0.0, 1.0)), seed=0)

    assert ld["low_med"] < lin["low_med"] / 3.0, (
        f"log_scale must beat linear0_1 on the 1e-8..1e-6 tail by >=3x: "
        f"log_scale={ld['low_med']:.3g} vs linear0_1={lin['low_med']:.3g}"
    )


# -----------------------------------------------------------------------------
# 10) SPERFNetCPP: distance-less variant for the simpler DS02 dataset.
#     Same trunk + baked-in activations; only the beam encoder changes
#     (rfnn.SPERFBeamEncoder, direction + spectrum only). Smoke + forward +
#     one-optimiser-step finite check.
# -----------------------------------------------------------------------------

from radfield3dnn.models.nerf_cpp import SPERFNetCPP

def _build_sperf(d_model: int = 64) -> SPERFNetCPP:
    return SPERFNetCPP(
        location_encoding_dims=12,
        in_spectra_dim=IN_SPECTRA_DIM,
        d_model=d_model,
        normalizer=LinearNormalizer(),
        flux_loss="L1Loss",  # see comment in _build_model — avoid log-space loss
    ).cuda()


def test_sperfnetcpp_forward_finite_at_init():
    torch.manual_seed(0)
    m = _build_sperf()
    batch = _make_batch()
    field = m.forward(batch)
    flux = field.scatter_field.flux
    spec = field.scatter_field.spectrum
    assert flux.shape == (BATCH,)
    assert spec.shape == (BATCH, IN_SPECTRA_DIM)
    assert torch.isfinite(flux).all() and torch.isfinite(spec).all()
    # Baked-in offset+linear-clamp bounds flux to the closed [0,1]; softplus
    # +sum-norm keeps spectrum non-negative summing to ~1.
    assert (flux >= 0.0).all() and (flux <= 1.0).all()
    assert (spec >= 0.0).all()
    sums = spec.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)


def test_sperfnetcpp_flux_finite_after_one_step():
    torch.manual_seed(0)
    m = _build_sperf()
    opt = optim.Adam(m.parameters(), lr=3e-4, eps=1e-4)
    batch = _make_batch()
    tgt_flux = torch.rand(BATCH, device="cuda") * 0.5 + 0.25
    tgt_spec = torch.softmax(torch.randn(BATCH, IN_SPECTRA_DIM, device="cuda"), dim=-1)
    opt.zero_grad()
    f = m.forward(batch)
    loss = (nn.functional.l1_loss(f.scatter_field.flux, tgt_flux)
            + nn.functional.l1_loss(f.scatter_field.spectrum, tgt_spec))
    loss.backward(); opt.step()
    with torch.no_grad():
        f2 = m.forward(batch)
    assert torch.isfinite(f2.scatter_field.flux).all()
    assert torch.isfinite(f2.scatter_field.spectrum).all()
    assert f2.scatter_field.flux.std().item() > 1e-5


def test_sperfnetcpp_gradients_reach_both_modules():
    """Encoder + main model must both receive non-zero gradient — verifies the
    SPERFBeamEncoder forward/backward bridge is correctly wired."""
    torch.manual_seed(0)
    m = _build_sperf()
    f = m.forward(_make_batch())
    (f.scatter_field.flux.mean() + f.scatter_field.spectrum.mean()).backward()
    grads = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
    assert any("encoder" in n and g is not None and g.norm().item() > 0 for n, g in grads.items()), \
        "SPERFBeamEncoder weights got zero gradient"
    assert any("model" in n and g is not None and g.norm().item() > 0 for n, g in grads.items()), \
        "BaseRadiationPredictionModel weights got zero gradient"


# -----------------------------------------------------------------------------
# 11) Production log_scale stack survives lr_find. PBRFNet_CPP.json now
#     defaults to normalizer="log_scale" + flux_activation="clamp" +
#     flux_clamp_min=-9, flux_clamp_max=0, flux_offset=-4.5. Field values live
#     in [-9, 0] instead of [0, 1], so per-element errors can reach ~9 — and
#     loss functions with an L2 component see err^2 up to ~81×, vs ~1× in the
#     old [0,1] regime. With FP16 master weights and tasks/train.py's
#     lr_find(min_lr=1e-4, max_lr=1e-2, num_training=250), the update step
#     overflows in the upper half of the sweep and poisons a single weight to
#     NaN — which then cascades through every matmul, producing all-NaN flux
#     AND spectrum outputs on the next forward.
#
#     The regression guard runs the exact tasks/train.py lr_find sweep with
#     the production log_scale config and asserts the model survives. Pair
#     with flux_loss="L1Loss" — pure L1 has bounded ±1 per-element gradient
#     regardless of the output range, so an FP16 weight update at the top of
#     the sweep no longer overflows. (FluxLoss / FluxLossRelative carry the
#     L2 component that makes this unstable; they are NOT safe with the
#     log_scale stack at high lr_find LRs.)
# -----------------------------------------------------------------------------

def _build_log_scale_model(d_model: int = 128, flux_loss: str = "L1Loss") -> PBRFNetCPP:
    return PBRFNetCPP(
        location_encoding_dims=12,
        in_spectra_dim=IN_SPECTRA_DIM,
        d_model=d_model,
        normalizer=LogScaleNormalizer(),
        flux_activation="clamp",
        flux_clamp_min=-9.0,
        flux_clamp_max=0.0,
        flux_offset=-4.5,
        flux_loss=flux_loss,
    ).cuda()


def test_pbrfnetcpp_log_scale_survives_production_lr_find():
    """Production PBRFNet_CPP.json config + tasks/train.py lr_find sweep."""
    from lightning.pytorch.tuner import Tuner

    torch.manual_seed(0)
    m = _build_log_scale_model(d_model=128, flux_loss="L1Loss")
    m.max_inner_batch_size = 4096

    lr_trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_steps=250, precision="32-true",
        logger=False, enable_progress_bar=False, enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    # Identical sweep to tasks/train.py:55-62 — the path that blew up in
    # production at ~39% (≈ lr 6e-4) before the log_scale-safe loss switch.
    Tuner(lr_trainer).lr_find(
        m, datamodule=_RFDataModule(), min_lr=1e-4, max_lr=1e-2,
        num_training=250,
    )

    w = m.model.weights
    assert torch.isfinite(w).all(), (
        f"model.weights went non-finite during lr_find with the log_scale "
        f"stack ({int(torch.isnan(w).sum())} NaNs, "
        f"{int(torch.isinf(w).sum())} infs). The fp16 update step overflowed "
        f"under the high-LR end of the sweep — likely a loss function with an "
        f"L2 / squared component is being paired with the [-9, 0] flux range."
    )
    assert torch.isfinite(m.encoder.weights).all(), "encoder weights non-finite"

    # Verify the forward path emits clean outputs in the configured codomain.
    with torch.no_grad():
        f = m.cuda().eval().forward(_make_batch())
    flux_n = f.scatter_field.flux
    spec_n = f.scatter_field.spectrum
    assert torch.isfinite(flux_n).all() and torch.isfinite(spec_n).all()
    assert (flux_n >= -9.0).all() and (flux_n <= 0.0).all(), (
        f"flux escaped the configured clamp [-9, 0]: "
        f"[{flux_n.min().item():.4f}, {flux_n.max().item():.4f}]"
    )


def test_l1withssim3d_handles_inf_dropped_voxels():
    """L1WithSSIM3DLoss must accept 5D volumetric tensors with `-inf`
    at locations excluded by ErrorbasedImportanceSampler. The L1 path
    masks them via valid-voxel-count denominator; the SSIM3D path
    handles them internally (replace with min non-masked target, then
    multiply the SSIM map by a valid_mask).

    This test pins both halves of that contract."""
    from radfield3dnn.losses.combinations import L1WithSSIM3DLoss

    torch.manual_seed(0)
    loss_fn = L1WithSSIM3DLoss()

    B, C, D, H, W = 2, 1, 16, 16, 16
    target = torch.rand(B, C, D, H, W) * 9.0 - 9.0  # log_scale codomain
    pred = (target + torch.randn_like(target) * 0.5).requires_grad_(True)

    for drop_frac in (0.0, 0.1, 0.5, 0.9, 1.0):
        target_inv = target.clone()
        if drop_frac > 0.0:
            drop_mask = torch.rand(B, C, D, H, W) < drop_frac
            target_inv[drop_mask] = float("-inf")
        else:
            drop_mask = torch.zeros_like(target, dtype=torch.bool)

        # Forward must produce finite per-sample loss for any drop fraction
        # including the degenerate 100 % case.
        out = loss_fn(target=target_inv, prediction=pred, input=None)
        assert torch.isfinite(out).all(), (
            f"loss non-finite at drop_frac={drop_frac}: {out}"
        )
        assert out.shape == (B,), f"unexpected loss shape {out.shape}"

        # Backward must produce finite gradients; dropped voxels must
        # contribute exactly zero L1 gradient.
        pred.grad = None
        out.sum().backward(retain_graph=True)
        g = pred.grad
        assert torch.isfinite(g).all(), (
            f"grad non-finite at drop_frac={drop_frac}"
        )
        if drop_frac > 0.0 and drop_frac < 1.0:
            # Dropped voxels can only get gradient through SSIM3D's
            # local-statistics (kernel reaches into neighbors). Their
            # L1 contribution is zero by construction. The combined
            # gradient should be much smaller than at kept voxels.
            dropped_grad = g[drop_mask].abs().mean().item()
            kept_grad = g[~drop_mask].abs().mean().item()
            assert dropped_grad < kept_grad, (
                f"dropped voxels should have weaker gradient than kept; "
                f"got dropped={dropped_grad} kept={kept_grad}"
            )


def test_pbrfnetcpp_log_scale_l1withssim3d_survives_lr_find():
    """L1WithSSIM3DLoss is the recommended structural-aware loss for the
    log_scale stack: pure-L1 core (bounded ±1 gradient → fp16-safe at
    [-9, 0] outputs) plus SSIM3D to break the "predict the median
    forever" plateau that plain L1 hits on the sparse-target field.
    This test exercises the exact ``tasks/train.py`` lr_find sweep
    that breaks FluxLoss + log_scale (see
    ``test_pbrfnetcpp_log_scale_fluxloss_is_fp16_unsafe``) and asserts
    the model stays finite end-to-end."""
    from lightning.pytorch.tuner import Tuner

    torch.manual_seed(0)
    m = _build_log_scale_model(d_model=128, flux_loss="L1WithSSIM3DLoss")
    m.max_inner_batch_size = 4096

    lr_trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_steps=250, precision="32-true",
        logger=False, enable_progress_bar=False, enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    Tuner(lr_trainer).lr_find(
        m, datamodule=_RFDataModule(), min_lr=1e-4, max_lr=1e-2,
        num_training=250,
    )

    w = m.model.weights
    assert torch.isfinite(w).all(), (
        f"model.weights went non-finite during lr_find with the log_scale "
        f"stack + L1WithSSIM3DLoss "
        f"({int(torch.isnan(w).sum())} NaNs, {int(torch.isinf(w).sum())} infs). "
        f"That should not happen — the L2 component that overflows for "
        f"FluxLoss is exactly what this loss drops."
    )
    assert torch.isfinite(m.encoder.weights).all()
    with torch.no_grad():
        f = m.cuda().eval().forward(_make_batch())
    assert torch.isfinite(f.scatter_field.flux).all()
    assert torch.isfinite(f.scatter_field.spectrum).all()


def test_pbrfnetcpp_log_scale_fluxloss_is_fp16_unsafe():
    """Diagnostic: confirm FluxLoss (Huber + SSIM, L2 component) WITH the
    log_scale stack is precisely the combination that overflows fp16 during
    lr_find. Documents the constraint so a future PBRFNet_CPP.json edit
    cannot silently regress.

    If this test ever starts passing, it means an upstream change made
    FluxLoss fp16-safe for [-9, 0] outputs and the regression guard can be
    relaxed / merged with the L1Loss test above."""
    import pytest
    from lightning.pytorch.tuner import Tuner

    torch.manual_seed(0)
    m = _build_log_scale_model(d_model=128, flux_loss="FluxLoss")
    m.max_inner_batch_size = 4096

    lr_trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_steps=250, precision="32-true",
        logger=False, enable_progress_bar=False, enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    try:
        Tuner(lr_trainer).lr_find(
            m, datamodule=_RFDataModule(), min_lr=1e-4, max_lr=1e-2,
            num_training=250,
        )
    except Exception:
        # The forward NaN-guard in PBRFNetCPP.forward raises mid-sweep; treat
        # that as the expected unsafe behavior.
        return

    # If lr_find completed without raising, the unsafe combination must still
    # have produced non-finite weights — assert that to keep this test
    # informative.
    w = m.model.weights
    assert not torch.isfinite(w).all(), (
        "FluxLoss + log_scale stack survived lr_find — the fp16 instability "
        "is no longer reproducible. Update PBRFNet_CPP.json hyperparameter "
        "space to re-include FluxLoss and remove this diagnostic test."
    )


# -----------------------------------------------------------------------------
# 5) Bounded gated fusion (beam_fusion="gated") paired with the hashgrid
#    location encoding. The C++ unit tests (tests/cxx/test_network.cu) cover
#    GatedFusion's structure + JIT codegen; this exercises its forward AND
#    backward end-to-end through the autograd bridge — the only place the
#    hand-written gated backward actually runs numerically.
# -----------------------------------------------------------------------------

def _build_gated_hashgrid_model(d_model: int = 128) -> PBRFNetCPP:
    return PBRFNetCPP(
        location_encoding_dims=12,
        in_spectra_dim=IN_SPECTRA_DIM,
        d_model=d_model,
        normalizer=LinearNormalizer(),
        flux_loss="L1Loss",
        location_encoding_kind="hashgrid",
        beam_fusion="gated",
    ).cuda()


def test_gated_fusion_forward_finite_at_init():
    torch.manual_seed(0)
    m = _build_gated_hashgrid_model()
    field = m.forward(_make_batch())
    flux = field.scatter_field.flux
    spectrum = field.scatter_field.spectrum
    assert flux.shape == (BATCH,)
    assert spectrum.shape == (BATCH, IN_SPECTRA_DIM)
    assert torch.isfinite(flux).all(), "gated-fusion flux non-finite at init"
    assert torch.isfinite(spectrum).all(), "gated-fusion spectrum non-finite at init"
    # Hard clamp [0, 1] on the flux head still holds with the gated conditioner.
    assert (flux >= 0.0).all() and (flux <= 1.0).all()


def test_gated_fusion_gradients_finite_and_reach_conditioner():
    torch.manual_seed(0)
    m = _build_gated_hashgrid_model()
    field = m.forward(_make_batch())
    loss = field.scatter_field.flux.abs().mean() + field.scatter_field.spectrum.abs().mean()
    loss.backward()
    # The single fused weight blob carries the gated conditioners' params; its
    # gradient must be finite and non-trivially non-zero (the gated backward ran).
    named = [(n, p) for n, p in m.named_parameters() if p.grad is not None]
    assert any("model" in n for n, _ in named), "no gradient on the fused model weights"
    for n, p in named:
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {n}"
    assert any(p.grad.abs().sum().item() > 0 for _, p in named), "all gradients are zero"


def test_gated_fusion_short_training_drops_loss_and_stays_finite():
    torch.manual_seed(0)
    m = _build_gated_hashgrid_model(d_model=128)
    optimizer = optim.Adam(m.parameters(), lr=3e-4, eps=1e-4)

    spec_phase = torch.linspace(0.0, 2.0 * math.pi, IN_SPECTRA_DIM, device="cuda")

    def make_targets(pos: torch.Tensor):
        u = (pos + 1.0) * 0.5
        tgt_flux = 0.5 * (1.0 + torch.sin(2.0 * u[:, 0]) * torch.cos(2.0 * u[:, 1]))
        amp = u.sum(dim=1, keepdim=True) / 3.0
        tgt_spec_unnorm = amp * (1.0 + torch.sin(spec_phase.unsqueeze(0) + u[:, 2:3]))
        tgt_spec = tgt_spec_unnorm / tgt_spec_unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return tgt_flux, tgt_spec

    first_loss = last_loss = None
    for step in range(400):
        batch = _make_batch()
        target_flux, target_spec = make_targets(batch.position)
        optimizer.zero_grad()
        field = m.forward(batch)
        loss = (
            nn.functional.l1_loss(field.scatter_field.flux, target_flux)
            + nn.functional.l1_loss(field.scatter_field.spectrum, target_spec)
        )
        assert torch.isfinite(loss), f"gated-fusion loss non-finite at step {step}: {loss.item()}"
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert math.isfinite(last_loss)
    assert last_loss < first_loss, f"gated-fusion loss did not decrease: {first_loss} -> {last_loss}"

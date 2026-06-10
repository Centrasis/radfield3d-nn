"""XAttnSirenField — a from-scratch per-voxel radiation-field predictor.

A NEW method (not derived from any existing model in this repo): each query
coordinate is decoded by a **modulated SIREN**, conditioned on the beam parameters
via **cross-attention** over a set of beam-context tokens.

Why this and not the existing point models:
  - Conditioning by cross-attention instead of FiLM/concat — "Attention Beats
    Concatenation for Conditioning Neural Fields" (Rebain et al., arXiv:2209.10684).
  - A SIREN (sinusoidal) decoder instead of a Fourier-features + SiLU MLP — sine
    activations directly represent the high-frequency, high-dynamic-range scatter
    field (Sitzmann et al. 2020; Coordinate-Aware Modulation, arXiv:2311.14993).

Deployment: every op is a standard ONNX operator (Gemm, Sin, MatMul, Softmax,
Add, LayerNormalization), so the `_core` graph exports to ONNX Runtime C++ /
TensorRT. `export_onnx()` + an `onnxruntime` parity check is the acceptance gate.
Trained in fp32; ONNX export may be cast to fp16 for real-time inference.

Only the framework *plumbing* is reused (the `FeedforwardPointwiseModel`
per-voxel volume assembly, the losses, the normalizer, DB-MTL) — the network is
entirely new.
"""
from __future__ import annotations

import math
from typing import Literal, Union

import torch
import torch.nn as nn
from torch import Tensor

from .feedforward import FeedforwardPointwiseModel
from .base import ModuleBuilder
from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel, PositionalInput
from radfield3dnn.metrics.types import TrainingMetrics, ChannelMetrics
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear


class _SineLayer(nn.Module):
    """A single SIREN layer with shift-modulation: y = sin(w0 * (Wx + b) + m).

    SIREN init (Sitzmann 2020): first layer weights ~U(-1/in, 1/in); hidden
    layers ~U(-sqrt(6/in)/w0, +...). All ops (Linear→Gemm, Sin) are ONNX-standard.
    """

    def __init__(self, in_features: int, out_features: int, w0: float, is_first: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = float(w0)
        with torch.no_grad():
            if is_first:
                b = 1.0 / in_features
            else:
                b = math.sqrt(6.0 / in_features) / self.w0
            self.linear.weight.uniform_(-b, b)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, mod: Tensor | None = None) -> Tensor:
        pre = self.linear(x)
        if mod is not None:
            pre = pre + mod
        return torch.sin(self.w0 * pre)


class XAttnSirenField(FeedforwardPointwiseModel):
    """Per-voxel field: cross-attention-conditioned modulated SIREN."""

    __model_name__ = "XAttnSirenField"

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_spectrum_tokens: int = 8,
        siren_depth: int = 5,
        siren_w0: float = 30.0,
        siren_hidden_w0: float = 1.0,
        in_spectra_dim: int = 32,
        out_spectra_dim: int = 32,
        flux_loss: str = "L1MagWeighted",
        spectrum_loss: str = "HistogramLoss",
        flux_offset: float = -4.5,
        flux_activation: Literal["clamp"] = "clamp",
        flux_clamp_min: float = -9.0,
        flux_clamp_max: float = 0.0,
        normalizer=None,
        precision: Literal["fp32", "fp16"] = "fp32",
        learning_rate: float = 1e-3,
        max_lr: float = 1e-3,
        randomize_voxel_location_in_training: bool = False,
        voxels_centered_around_origin: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            randomize_voxel_location_in_training=randomize_voxel_location_in_training,
            voxels_centered_around_origin=voxels_centered_around_origin,
            normalizer=normalizer,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_spectrum_tokens = n_spectrum_tokens
        self.in_spectra_dim = in_spectra_dim
        self.out_spectra_dim = out_spectra_dim
        self.siren_depth = siren_depth
        self.siren_w0 = siren_w0
        self.siren_hidden_w0 = siren_hidden_w0
        self._precision = precision
        self._max_lr = float(max_lr)
        self._flux_offset = float(flux_offset)
        self._flux_clamp_min = float(flux_clamp_min)
        self._flux_clamp_max = float(flux_clamp_max)
        self.flux_loss_name = flux_loss
        self.spectrum_loss_name = spectrum_loss

        self._flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)

        # ── Beam-context tokens (K = n_spectrum_tokens + direction + distance) ──
        self.spectrum_to_tokens = nn.Linear(in_spectra_dim, n_spectrum_tokens * d_model)
        self.direction_token = nn.Linear(3, d_model)
        self.distance_token = nn.Linear(1, d_model)
        self.n_ctx = n_spectrum_tokens + 2
        self.ctx_pos_embed = nn.Parameter(torch.randn(1, self.n_ctx, d_model) * 0.02)
        self.ctx_norm = nn.LayerNorm(d_model)

        # ── Query encoder (SIREN input layer over the xyz coordinate) ──
        self.query_in = _SineLayer(3, d_model, w0=siren_w0, is_first=True)

        # ── Cross-attention (manual MHA; standard ONNX ops) ──
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.cond_norm = nn.LayerNorm(d_model)

        # ── Modulated-SIREN decoder; per-layer shift from the conditioning c ──
        self.siren_layers = nn.ModuleList([
            _SineLayer(d_model, d_model, w0=siren_hidden_w0, is_first=False)
            for _ in range(siren_depth)
        ])
        self.mod_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(siren_depth)
        ])
        for m in self.mod_layers:  # identity-ish modulation at init
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        # ── Heads ──
        self.flux_head = nn.Linear(d_model, d_model)
        self.flux_out = nn.Linear(d_model, 1)
        self.spectrum_head = nn.Linear(d_model, out_spectra_dim)
        # Per-voxel multiplicative correction on the analytic direct beam (used by the
        # scatter+analytic variant). corr = 1 + tanh(·) ∈ [0,2]; zero-init → corr≡1 at
        # start (analytic unchanged), then learns the systematic core build-up/penumbra
        # residual the static analytic can't capture → breaks the top90 ceiling.
        self.direct_corr_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.direct_corr_head.weight); nn.init.zeros_(self.direct_corr_head.bias)
        self.spectra_activation = HistogramNormalize(dim=-1)
        if isinstance(self._normalizer, LinearNormalizer) and self._normalizer.range == (0.0, 1.0):
            self.flux_activation = GradientConservingClamping(0.0, 1.0)
        elif isinstance(self._normalizer, LinearNormalizer) and self._normalizer.range == (-1.0, 1.0):
            self.flux_activation = GradientConservingClamping(-1.0, 1.0)
        else:
            self.flux_activation = GradientConservingClamping(flux_clamp_min, flux_clamp_max)

        if precision == "fp16":
            self.half()

    @property
    def _compute_dtype(self):
        return torch.float16 if self._precision == "fp16" else torch.float32

    # ───────────────────────── core (ONNX-exportable) ─────────────────────────
    def _core(self, position: Tensor, direction: Tensor, distance: Tensor, spectrum: Tensor):
        """Tensor→tensor graph. position(B,3) direction(B,3) distance(B,1)
        spectrum(B,in_spectra_dim) → (flux_raw(B,1), spectrum_logits(B,out))."""
        B = position.shape[0]
        # context tokens
        spec_tok = self.spectrum_to_tokens(spectrum).reshape(B, self.n_spectrum_tokens, self.d_model)
        dir_tok = self.direction_token(direction).unsqueeze(1)
        dist_tok = self.distance_token(distance).unsqueeze(1)
        ctx = torch.cat([spec_tok, dir_tok, dist_tok], dim=1)          # (B,K,D)
        ctx = self.ctx_norm(ctx + self.ctx_pos_embed)

        # query from coordinate (SIREN first activation)
        q0 = self.query_in(position)                                   # (B,D)

        # multi-head cross-attention: query attends to context
        q = self.q_proj(q0).reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)   # (B,H,1,hd)
        k = self.k_proj(ctx).reshape(B, self.n_ctx, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,K,hd)
        v = self.v_proj(ctx).reshape(B, self.n_ctx, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)         # (B,H,1,K)
        attn = torch.softmax(scores, dim=-1)
        ctx_vec = torch.matmul(attn, v)                                # (B,H,1,hd)
        ctx_vec = ctx_vec.transpose(1, 2).reshape(B, self.d_model)     # (B,D)
        c = self.cond_norm(q0 + self.attn_out(ctx_vec))               # conditioning latent (B,D)

        # modulated-SIREN decoder over the coordinate features
        h = q0
        for layer, mod in zip(self.siren_layers, self.mod_layers):
            h = layer(h, mod(c))
        h = h + c  # final residual with the conditioning latent

        flux_raw = self.flux_out(torch.sin(self.flux_head(h)))         # (B,1)
        spectrum_logits = self.spectrum_head(h)                        # (B,out)
        direct_corr = 1.0 + torch.tanh(self.direct_corr_head(h))       # (B,1) ∈ [0,2]
        return flux_raw, spectrum_logits, direct_corr

    def forward(self, x: PositionalInput, global_parameters: Tensor | None = None) -> RadiationField:
        dtype = self._compute_dtype
        spectrum = resample_histogram_bilinear(x.spectrum.to(torch.float32), self.in_spectra_dim).to(dtype)
        position = x.position.to(dtype)
        direction = x.direction.to(dtype)
        # distance_token is Linear(1, …): feed the scalar source→isocentre distance
        # (origin is normalised to [0,1] with the field centre at 0.5), not the raw 3D
        # origin (which crashed as mat (N×3)·(1×D)).
        distance = (x.origin.to(dtype) - 0.5).norm(dim=-1, keepdim=True)
        flux_raw, spectrum_logits, direct_corr = self._core(position, direction, distance, spectrum)
        flux = self.flux_activation(flux_raw.squeeze(-1) + self._flux_offset)
        spectrum_out = self.spectra_activation(spectrum_logits)
        # ONLY the scatter+analytic variant exposes the per-voxel direct-beam correction
        # (in the direct_beam flux slot, spectrum=None). The plain model returns
        # direct_beam=None so it stays a clean single-channel (scatter-only) predictor —
        # otherwise InferenceHelper would mis-read it as two-head and emit a None-spectrum
        # joined field, which made the air-kerma metrics return None.
        direct = RadiationFieldChannel(flux=direct_corr.squeeze(-1), spectrum=None, error=None) \
            if getattr(self, "_expose_corr", False) else None
        return RadiationField(
            scatter_field=RadiationFieldChannel(flux=flux, spectrum=spectrum_out, error=None),
            direct_beam=direct,
        )

    def get_core_model(self) -> nn.Module:
        return self

    def configure_optimizers(self):
        # AdamW with a no-decay split (LayerNorm affine, biases, and the learned
        # token positional embedding are exempt from weight decay), step-wise
        # warmup (~1 epoch) then cosine decay. Standard modern recipe.
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "pos_embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        lr = min(max(float(self._lr), 1e-5), self._max_lr)
        opt = torch.optim.AdamW([
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.99), eps=1e-8)

        total = int(self.trainer.estimated_stepping_batches) if self.trainer is not None else 1000
        total = max(total, 1)
        max_epochs = int(max(self.trainer.max_epochs, 1)) if self.trainer is not None else 100
        steps_per_epoch = max(1, total // max_epochs)
        warmup = min(steps_per_epoch, max(1, total - 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - warmup), eta_min=lr * 1e-2),
            ],
            milestones=[warmup],
        )
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def export_onnx(self, path: str, opset: int = 17, fp16: bool = False):
        """Export the per-voxel `_core` graph to ONNX (ORT-C++ runnable)."""
        self.eval()
        dev = next(self.parameters()).device
        dt = torch.float16 if fp16 else torch.float32
        mdl = self.half() if fp16 else self.float()
        B = 8
        dummy = (
            torch.rand(B, 3, device=dev, dtype=dt),
            torch.randn(B, 3, device=dev, dtype=dt),
            torch.rand(B, 1, device=dev, dtype=dt),
            torch.rand(B, self.in_spectra_dim, device=dev, dtype=dt),
        )
        torch.onnx.export(
            _CoreWrapper(self), dummy, path, opset_version=opset,
            input_names=["position", "direction", "distance", "spectrum"],
            output_names=["flux_raw", "spectrum_logits"],
            dynamic_axes={k: {0: "N"} for k in ["position", "direction", "distance", "spectrum", "flux_raw", "spectrum_logits"]},
        )
        return path

    def get_custom_parameters(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_spectrum_tokens": self.n_spectrum_tokens,
            "siren_depth": self.siren_depth,
            "siren_w0": self.siren_w0,
            "siren_hidden_w0": self.siren_hidden_w0,
            "in_spectra_dim": self.in_spectra_dim,
            "out_spectra_dim": self.out_spectra_dim,
            "flux_loss": self.flux_loss_name,
            "spectrum_loss": self.spectrum_loss_name,
            "flux_offset": self._flux_offset,
            "flux_activation": "clamp",
            "flux_clamp_min": self._flux_clamp_min,
            "flux_clamp_max": self._flux_clamp_max,
            "precision": self._precision,
            "max_lr": self._max_lr,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None,
        }


class _CoreWrapper(nn.Module):
    """Thin module exposing `_core` as a flat tensor->tensor graph for ONNX."""
    def __init__(self, model: XAttnSirenField):
        super().__init__()
        self.model = model

    def forward(self, position, direction, distance, spectrum):
        return self.model._core(position, direction, distance, spectrum)


class XAttnSirenScatter(XAttnSirenField):
    """Per-voxel MLP that predicts SCATTER ONLY and obtains the direct beam from the
    shared, ONNX-exportable AnalyticDirectBeam (added in forward2volume / fetched by
    xyz). Scored on the JOINED field so it is directly comparable to the field-wise
    FieldScatterUNet. Run with join_channels=false + use_geometry=true."""
    __model_name__ = "XAttnSirenScatter"

    def __init__(self, *args, beam_spectrum_bins: int = 64, train_voxel_samples: int = 4096,
                 scale_head_ckpt: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        from radfield3dnn.models.layers.analytic_direct import AnalyticDirectBeam
        from radfield3dnn.models.layers.scatter_scale import ScatterScaleHead
        self.adb = AnalyticDirectBeam(voxel_size_m=0.02)
        # (a) ρ MLP: beam params → log10(Σscatter/Σdirect). DECOUPLED — pre-trained
        # standalone (6% err) and FROZEN here, because folding it into the joint
        # DB-MTL loss starves it (it reached only 65% err trained jointly).
        self.scale_head = ScatterScaleHead(n_spectrum_bins=beam_spectrum_bins)
        self.beam_spectrum_bins = int(beam_spectrum_bins)
        self.train_voxel_samples = int(train_voxel_samples)
        self._expose_corr = True   # this variant uses the direct-correction head

        self._scale_head_frozen = False
        if scale_head_ckpt:
            self.scale_head.load_state_dict(torch.load(scale_head_ckpt, map_location="cpu"))
            for p in self.scale_head.parameters():
                p.requires_grad_(False)
            self._scale_head_frozen = True
        # the JOINED field is normalised to max flux = 1.0 (per-field max-norm).
        self._joined_norm = LinearNormalizer(range=(0.0, 1.0), always_normalize=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._scale_head_frozen:
            self.scale_head.eval()   # keep frozen BN running-stats; no train-mode updates
        return self

    @staticmethod
    def _max_norm(channel: RadiationFieldChannel) -> RadiationFieldChannel:
        """Per-field max-normalise the flux so the joined field has max flux = 1.0
        (the deployment convention). The spectrum (a per-voxel histogram) is unchanged."""
        sp = tuple(range(1, channel.flux.dim()))
        m = channel.flux.amax(dim=sp, keepdim=True).clamp_min(1e-12)
        return channel._replace(flux=channel.flux / m)

    def _direct_spectrum(self, inp, dims, dtype, device, B):
        """The direct beam carries the (tube) spectrum, broadcast over the beam volume
        — deployable (no simulated direct needed)."""
        d = resample_histogram_bilinear(inp.spectrum.to(torch.float32), self.out_spectra_dim).to(dtype)
        d = (d / d.sum(-1, keepdim=True).clamp_min(1e-30)).to(device)
        return d.reshape(B, self.out_spectra_dim, 1, 1, 1).expand(B, self.out_spectra_dim, *dims)

    def deployment_joined_physical(self, batch) -> RadiationFieldChannel:
        """The deployment field: (analytic air direct × learned per-voxel correction)
        + ρ-scaled predicted scatter, joined in PHYSICAL space (no simulated direct).
        Single source of truth used by BOTH loss and accuracy paths; caller max-norms
        to max flux = 1.0."""
        inp = batch.input
        gt0 = batch.ground_truth
        sf0 = gt0.scatter_field if hasattr(gt0, "scatter_field") else gt0
        dims = tuple(sf0.flux.shape[-3:])
        dt = sf0.flux.dtype; dev = sf0.flux.device; B = sf0.flux.shape[0]
        direct_phys = self.adb(inp.direction, inp.origin, inp.spectrum,
                               inp.beam_shape_parameters, inp.geometry, dims=dims).to(dt).reshape(sf0.flux.shape)
        dspec = self._direct_spectrum(inp, dims, dt, dev, B)
        pred = self.forward2volume_from_training_input(self._normalizer.forward(batch))
        # (b) per-voxel correction on the analytic direct (rides in direct_beam.flux)
        corr = pred.direct_beam.flux.reshape(direct_phys.shape) if pred.direct_beam is not None else 1.0
        direct_corrected = direct_phys * corr
        self.scale_head.eval()
        log_ratio = self.scale_head(inp.direction, inp.origin, inp.spectrum, inp.beam_shape_parameters)
        scatter_phys = self.scale_head.rescale_scatter(pred.scatter_field.flux, direct_corrected, log_ratio)
        return self._channels_join.join_channels(RadiationField(
            scatter_field=RadiationFieldChannel(flux=scatter_phys, spectrum=pred.scatter_field.spectrum, error=None),
            direct_beam=RadiationFieldChannel(flux=direct_corrected, spectrum=dspec, error=None)))

    def _sample_voxels(self, gt0, dev):
        """Sample K voxels per field: HALF uniform (for the diffuse scatter field) and
        HALF importance-sampled ∝ direct-beam dose (multinomial), so the high-dose beam
        core — the top90 region the direct-correction must fit — is actually trained.
        Returns flat indices, [0,1] positions (idx/(dim-1)), centred-metre coords."""
        sf0 = gt0.scatter_field if hasattr(gt0, "scatter_field") else gt0
        B = sf0.flux.shape[0]; D, H, W = sf0.flux.shape[-3:]; N = D * H * W
        K = self.train_voxel_samples
        Ku = K // 2
        uni = torch.randint(0, N, (B, K - Ku), device=dev)
        prob = gt0.direct_beam.flux.reshape(B, N).clamp_min(0).float() + 1e-12
        imp = torch.multinomial(prob, Ku, replacement=True)                        # ∝ dose
        flat = torch.cat([uni, imp], dim=1)                                        # (B,K)
        i = flat // (H * W); j = (flat // W) % H; k = flat % W
        ijk = torch.stack([i, j, k], -1).float()                                  # (B,K,3)
        denom = torch.tensor([D - 1, H - 1, W - 1], device=dev).float().clamp_min(1)
        pos = ijk / denom                                                          # [0,1]
        vd = self.adb.voxel_size_for((D, H, W))   # resolution-aware (field/dims), NOT hardcoded 0.02
        gd = torch.tensor([D, H, W], device=dev).float()
        cm = (ijk + 0.5) * vd - 0.5 * gd * vd                                      # centred metres
        return flat, pos, cm, (B, K, D, H, W), vd

    def evaluate_forward(self, batch):
        # Efficient SUBSAMPLED training: K random voxels per field (not the full
        # volume → no 44-min/epoch). Per voxel the net predicts the normalised scatter
        # shape AND the direct correction; ρ is frozen (pre-trained). Losses: pure
        # scatter (flux+spectrum) + direct-correction (corr·analytic ≈ GT direct).
        inp = batch.input; gt0 = batch.ground_truth
        sf0 = gt0.scatter_field if hasattr(gt0, "scatter_field") else gt0
        db0 = gt0.direct_beam
        dev = sf0.flux.device; dt = sf0.flux.dtype
        flat, pos, cm, (B, K, D, H, W), vd = self._sample_voxels(gt0, dev)
        N = D * H * W; S = sf0.spectrum.shape[1]

        # gather GT at sampled voxels
        sca_flux = sf0.flux.reshape(B, N).gather(1, flat)                          # (B,K)
        sca_spec = sf0.spectrum.reshape(B, S, N).gather(2, flat[:, None, :].expand(B, S, K))  # (B,S,K)
        dir_flux = db0.flux.reshape(B, N).gather(1, flat)                          # (B,K)
        sca_max = sf0.flux.reshape(B, N).amax(1, keepdim=True).clamp_min(1e-12)
        gt_sca_n = (sca_flux / sca_max).reshape(B * K)                             # normalised shape target
        gt_spec = sca_spec.permute(0, 2, 1).reshape(B * K, S)

        # per-voxel forward (fast)
        rep = lambda t: t.repeat_interleave(K, 0)
        pin = PositionalInput(position=pos.reshape(B * K, 3).to(dt), direction=rep(inp.direction),
                              origin=rep(inp.origin), spectrum=rep(inp.spectrum), geometry=None,
                              beam_shape_type=None, beam_shape_parameters=rep(inp.beam_shape_parameters))
        pred = self(pin)
        corr = pred.direct_beam.flux.reshape(B, K)
        # analytic direct at the sampled voxels (point mode, build-up+penumbra, no shadow)
        analytic = self.adb(inp.direction, inp.origin, inp.spectrum, inp.beam_shape_parameters,
                            query_points=cm, voxel_size=vd).to(dt)                 # (B,K)
        # (b) direct-correction loss: corr·analytic ≈ GT direct, only in the beam
        m = (analytic > 1e-9)
        corr_tgt = (dir_flux / analytic.clamp_min(1e-9)).clamp(0.0, 2.0)
        loss_corr = (((corr - corr_tgt) ** 2) * m).sum() / m.sum().clamp_min(1)
        self._extra_task_losses["direct_corr"] = loss_corr.reshape(1)

        # pure-scatter loss (clean signal), reusing the framework loss fns
        gt_field = RadiationField(scatter_field=RadiationFieldChannel(flux=gt_sca_n, spectrum=gt_spec, error=None), direct_beam=None)
        pred_field = RadiationField(scatter_field=RadiationFieldChannel(flux=pred.scatter_field.flux, spectrum=pred.scatter_field.spectrum, error=None), direct_beam=None)
        return self.calculate_metrics(pred_field, gt_field, batch)

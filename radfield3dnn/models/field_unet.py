"""FieldScatterUNet — a FIELD-WISE scatter predictor.

Predicts the ENTIRE 3D scatter field in one forward (vs the per-voxel coordinate
query of PBRFNet), so it can exploit spatial correlation between voxels. Field-to-field:
  input  = direct-beam flux volume (1ch) + normalised xyz coordinate grid (3ch)
  cond   = beam spectrum  (encoded → FiLM)
  output = scatter spectrum volume (out_spectra_bins ch) + scatter flux volume (1ch)

Architecture:
  • residual Conv3d blocks (Conv→GroupNorm→SiLU ×2 + projection skip),
  • attention-gated skip connections (Attention U-Net, Oktay et al. 2018) — focuses
    the decoder on the sparse high-flux region (the field is ~87% near-zero),
  • FiLM conditioning from the beam spectrum at the bottleneck and every decoder
    stage (gamma = 1+tanh, identity-start).

Deployment: every op is standard ONNX (Conv3d, ConvTranspose3d, GroupNorm, Sigmoid,
Tanh, Add, Mul, Pad, Slice) → ONNX Runtime C++ / TensorRT, via `export_onnx()`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseNeuralRadFieldModel, ModuleBuilder
from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel, TrainingInputData
from radfield3dnn.models.activations.HistogramNormalize import HistogramNormalize
from radfield3dnn.models.activations.flux_activations import GradientConservingClamping
from radfield3dnn.models.encoders.spectra_encoder import SimpleSpectraEncoder
from radfield3dnn.models.layers.analytic_direct import AnalyticDirectBeam
from radfield3dnn.preprocessing.normalizations import NormalizerConstructor
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = _gn(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        return self.act(h + self.skip(x))


class _FiLM3D(nn.Module):
    """Spatial FiLM: gamma=1+tanh(.), beta from the spectrum latent; identity start."""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        self.channels = channels

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gb = self.to_gb(cond)                                  # (B, 2C)
        gamma, beta = gb[:, :self.channels], gb[:, self.channels:]
        gamma = 1.0 + torch.tanh(gamma)
        shape = (x.shape[0], self.channels, 1, 1, 1)
        return x * gamma.view(shape) + beta.view(shape)


class _AttentionGate(nn.Module):
    """Oktay attention gate: gate the skip `x` by the coarser signal `g`."""
    def __init__(self, ch_x: int, ch_g: int, ch_int: int):
        super().__init__()
        self.theta_x = nn.Conv3d(ch_x, ch_int, 1)
        self.phi_g = nn.Conv3d(ch_g, ch_int, 1)
        self.psi = nn.Conv3d(ch_int, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        a = self.act(self.theta_x(x) + self.phi_g(g))
        alpha = torch.sigmoid(self.psi(a))                     # (B,1,...)
        return x * alpha


class FieldScatterUNet(BaseNeuralRadFieldModel):
    __model_name__ = "FieldScatterUNet"

    def __init__(
        self,
        d_model: int = 32,
        depth: int = 3,
        in_spectra_dim: int = 32,
        out_spectra_bins: int = 32,
        cond_dim: int = 64,
        out_dims: tuple = (48, 48, 48),
        flux_loss: str = "SMAPEBalanced",
        spectrum_loss: str = "HistogramLoss",
        flux_clamp_min: float = 0.0,
        flux_clamp_max: float = 1.0,
        normalizer=None,
        learning_rate: float = 1e-3,
        max_lr: float = 1e-3,
        use_analytic_direct: bool = True,
    ):
        if normalizer is None:
            normalizer = NormalizerConstructor.construct_by_name("linear0_1")
        elif isinstance(normalizer, str):
            normalizer = NormalizerConstructor.construct_by_name(normalizer)
        super().__init__(normalizer=normalizer, learning_rate=learning_rate)
        self.d_model = d_model
        self.depth = depth
        self.in_spectra_dim = in_spectra_dim
        self.out_spectra_bins = out_spectra_bins
        self.cond_dim = cond_dim
        self.out_dims = tuple(out_dims)
        self._max_lr = float(max_lr)
        self._flux_clamp_min = float(flux_clamp_min)
        self._flux_clamp_max = float(flux_clamp_max)
        self.use_analytic_direct = bool(use_analytic_direct)
        self.adb = AnalyticDirectBeam(voxel_size_m=0.02)
        self.flux_loss_name = flux_loss
        self.spectrum_loss_name = spectrum_loss
        self._flux_loss_fn = ModuleBuilder.ConstructLoss_fn(flux_loss)
        self._spectrum_loss_fn = ModuleBuilder.ConstructLoss_fn(spectrum_loss)

        self.spectra_encoder = SimpleSpectraEncoder(in_spectra_dim, cond_dim)

        chans = [d_model * (2 ** i) for i in range(depth + 1)]   # e.g. 32,64,128,256
        # encoder: skip_i = c_i @ res/2^i (saved BEFORE down); down preserves channels,
        # enc maps c_i -> c_{i+1} after downsampling. bottleneck = c_depth.
        self.stem = _ResBlock3D(4, chans[0])
        self.downs = nn.ModuleList([nn.Conv3d(chans[i], chans[i], 2, stride=2) for i in range(depth)])
        self.enc = nn.ModuleList([_ResBlock3D(chans[i], chans[i + 1]) for i in range(depth)])
        # bottleneck (FiLM-conditioned)
        self.bott = _ResBlock3D(chans[depth], chans[depth])
        self.bott_film = _FiLM3D(chans[depth], cond_dim)
        # decoder
        self.ups = nn.ModuleList([nn.ConvTranspose3d(chans[depth - i], chans[depth - i - 1], 2, stride=2) for i in range(depth)])
        self.gates = nn.ModuleList([_AttentionGate(chans[depth - i - 1], chans[depth - i - 1], chans[depth - i - 1]) for i in range(depth)])
        self.dec = nn.ModuleList([_ResBlock3D(2 * chans[depth - i - 1], chans[depth - i - 1]) for i in range(depth)])
        self.dec_films = nn.ModuleList([_FiLM3D(chans[depth - i - 1], cond_dim) for i in range(depth)])
        # heads
        self.head = nn.Conv3d(chans[0], out_spectra_bins + 1, 1)
        self.spectra_activation = HistogramNormalize(dim=1)
        if isinstance(self._normalizer, LinearNormalizer) and self._normalizer.range == (0.0, 1.0):
            self.flux_activation = GradientConservingClamping(0.0, 1.0)
        elif isinstance(self._normalizer, LinearNormalizer) and self._normalizer.range == (-1.0, 1.0):
            self.flux_activation = GradientConservingClamping(-1.0, 1.0)
        else:
            self.flux_activation = GradientConservingClamping(flux_clamp_min, flux_clamp_max)

    @staticmethod
    def _next_pow2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    def _analytic_direct(self, batch: TrainingInputData) -> Tensor:
        """Pull the raw beam geometry from the batch and call the shared,
        ONNX-exportable AnalyticDirectBeam module."""
        inp = batch.input
        return self.adb(inp.direction, inp.origin, inp.spectrum,
                        inp.beam_shape_parameters, inp.geometry)


    def _core(self, x: Tensor, cond: Tensor) -> Tensor:
        """(B,4,D,H,W) volume + (B,cond_dim) spectrum latent → (B,out_bins+1,D,H,W)."""
        h = self.stem(x)
        skips = []
        for down, enc in zip(self.downs, self.enc):
            skips.append(h)        # skip_i = c_i at this resolution
            h = enc(down(h))       # downsample, then c_i -> c_{i+1}
        h = self.bott_film(self.bott(h), cond)
        for up, gate, dec, film, skip in zip(self.ups, self.gates, self.dec, self.dec_films, reversed(skips)):
            h = up(h)
            g = gate(skip, h)
            h = dec(torch.cat([h, g], dim=1))
            h = film(h, cond)
        return self.head(h)

    def forward(self, batch: TrainingInputData) -> RadiationField:
        gt = batch.ground_truth
        direct = gt.direct_beam.flux if (isinstance(gt, RadiationField) and gt.direct_beam is not None) else gt.scatter_field.flux
        dev = direct.device
        spatial = direct.shape[2:]
        pad = [self._next_pow2(d) for d in spatial]
        diff = [pad[i] - spatial[i] for i in range(3)]
        pl = [d // 2 for d in diff]
        pr = [diff[i] - pl[i] for i in range(3)]
        flux_in = F.pad(direct, (pl[2], pr[2], pl[1], pr[1], pl[0], pr[0]))
        vmap = self.generate_voxelmap3d(torch.tensor(pad, device=dev), None, dev)
        vmap = (vmap * 2.0 - 1.0).unsqueeze(0).expand(direct.shape[0], -1, -1, -1, -1).permute(0, 4, 1, 2, 3)
        x = torch.cat([flux_in, vmap], dim=1)

        cond = self.spectra_encoder(batch.input.spectrum)
        out = self._core(x, cond)
        out = out[:, :, pl[0]:pad[0] - pr[0], pl[1]:pad[1] - pr[1], pl[2]:pad[2] - pr[2]]
        flux = self.flux_activation(out[:, self.out_spectra_bins]).unsqueeze(1)
        spectrum = self.spectra_activation(out[:, :self.out_spectra_bins])
        return RadiationField(
            scatter_field=RadiationFieldChannel(spectrum=spectrum, flux=flux, error=None),
            direct_beam=None,
        )

    def forward2volume_from_training_input(self, batch: TrainingInputData, voxel_counts=None, spectra_bins: int = 32) -> RadiationField:
        gt = batch.ground_truth
        gt_flux = gt.scatter_field.flux + (gt.direct_beam.flux if gt.direct_beam is not None else 0.0)
        mask = ~torch.isfinite(gt_flux)
        pred = self(batch)
        if mask.any():
            inf_f = torch.full_like(pred.scatter_field.flux, -torch.inf)
            inf_s = torch.full_like(pred.scatter_field.spectrum, -torch.inf)
            pred = RadiationField(
                scatter_field=RadiationFieldChannel(
                    spectrum=torch.where(mask.expand_as(pred.scatter_field.spectrum), inf_s, pred.scatter_field.spectrum),
                    flux=torch.where(mask, inf_f, pred.scatter_field.flux), error=None),
                direct_beam=None)
        return pred

    def forward2volume(self, x, voxel_counts, spectra_bins: int = 32, mask=None):
        return self(x)

    def evaluate_forward(self, batch: TrainingInputData):
        # Score on the FULL JOINED field (direct + scatter). The net predicts SCATTER
        # only; the joined total is reconstructed as (direct + predicted_scatter) in
        # PHYSICAL space, then re-normalised. When use_analytic_direct is set, the direct
        # beam is computed from the RAW batch beam geometry + density channel by the
        # internal AnalyticDirectBeam instead of using the simulated direct — so the model
        # is self-contained and deployable (no simulated direct needed at inference).
        if self.use_analytic_direct and isinstance(batch.ground_truth, RadiationField) \
                and batch.ground_truth.direct_beam is not None:
            adb = self._analytic_direct(batch).to(batch.ground_truth.direct_beam.flux.dtype)
            db = batch.ground_truth.direct_beam._replace(flux=adb.reshape(batch.ground_truth.direct_beam.flux.shape))
            batch = batch._replace(ground_truth=batch.ground_truth._replace(direct_beam=db))

        batch = self._normalizer.forward(batch)
        gt = batch.ground_truth
        pred = self.forward2volume_from_training_input(batch)   # scatter (normalised)

        # joined prediction: pred-scatter + direct, joined in physical space
        pred_full = RadiationField(scatter_field=pred.scatter_field, direct_beam=gt.direct_beam)
        pred_full = self._normalizer.inverse(pred_full)
        pred_joined = self._normalizer.forward(self._channels_join(pred_full))

        # joined GT
        gt_joined = self._normalizer.forward(self._channels_join(self._normalizer.inverse(gt)))

        pred_field = RadiationField(scatter_field=pred_joined, direct_beam=None)
        return self.calculate_metrics(pred_field, gt_joined, batch)

    def _search_optimal_batch_size(self):
        self.max_inner_batch_size = 8

    def get_core_model(self) -> nn.Module:
        return self

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or "norm" in n.lower() or n.endswith(".bias")) else decay).append(p)
        lr = min(max(float(self._lr), 1e-5), self._max_lr)
        opt = torch.optim.AdamW([
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.99))
        total = int(self.trainer.estimated_stepping_batches) if self.trainer is not None else 1000
        max_ep = int(max(self.trainer.max_epochs, 1)) if self.trainer is not None else 100
        warm = max(1, total // max_ep)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warm),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - warm), eta_min=lr * 1e-2)],
            milestones=[warm])
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def export_onnx(self, path: str):
        self.eval()
        dev = next(self.parameters()).device
        D = self._next_pow2(self.out_dims[0])
        dummy = (torch.rand(1, 4, D, D, D, device=dev), torch.rand(1, self.cond_dim, device=dev))
        # Dynamic batch axis so a stored model accepts any number of fields at inference.
        batch = torch.export.Dim("batch")
        torch.onnx.export(_CoreWrap(self).eval(), dummy,
                          input_names=["volume", "cond"], output_names=["out"],
                          dynamic_shapes=({0: batch}, {0: batch}), dynamo=True).save(path)
        return path

    def get_custom_parameters(self) -> dict:
        return {
            "d_model": self.d_model, "depth": self.depth,
            "in_spectra_dim": self.in_spectra_dim, "out_spectra_bins": self.out_spectra_bins,
            "cond_dim": self.cond_dim, "out_dims": list(self.out_dims),
            "flux_loss": self.flux_loss_name, "spectrum_loss": self.spectrum_loss_name,
            "flux_clamp_min": self._flux_clamp_min,
            "flux_clamp_max": self._flux_clamp_max, "max_lr": self._max_lr,
            "normalizer": self._normalizer.get_type() if self._normalizer is not None else None,
        }


class _CoreWrap(nn.Module):
    def __init__(self, m: FieldScatterUNet):
        super().__init__()
        self.m = m

    def forward(self, volume, cond):
        return self.m._core(volume, cond)

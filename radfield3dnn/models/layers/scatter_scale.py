"""ScatterScaleHead — a small, decoupled MLP that predicts the physical magnitude of
the scatter field relative to the analytic direct beam, from the beam parameters alone.

Rationale: the main network learns only the NORMALISED scatter SHAPE (max 1, clean
low-dynamic-range signal). To recombine that shape with the analytic direct into a
physical joined field, we need one scalar per field — the scatter/direct magnitude
coupling. Predicting the *integrated* flux ratio ρ = Σscatter/Σdirect is robust
(small-MLP val R²≈0.95, ~6% error on DS03), whereas the peak ratio is MC-noisy
(R²≈0.68). The join then rescales the shape so Σscatter = ρ·Σdirect, adds the direct,
and renormalises to max flux 1.0.

Every op is standard ONNX (Gemm, SiLU=Sigmoid·Mul, BatchNorm folds to Gemm at eval),
so it lowers cleanly to ONNX Runtime C++ alongside AnalyticDirectBeam.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScatterScaleHead(nn.Module):
    def __init__(self, n_spectrum_bins: int = 64, hidden: int = 64):
        super().__init__()
        # The raw tube spectrum bin-count varies (smoothing/resampling), so resample
        # it to a FIXED size internally → the head is robust to the input binning.
        self.n_spectrum_bins = int(n_spectrum_bins)
        # features: direction(3) + origin(3) + |origin| source-iso dist(1) + rect(2) + spectrum
        in_dim = 3 + 3 + 1 + 2 + self.n_spectrum_bins
        self.in_norm = nn.BatchNorm1d(in_dim)            # per-feature standardisation
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _features(self, direction: Tensor, origin: Tensor, spectrum: Tensor, rect: Tensor) -> Tensor:
        dist = origin.norm(dim=1, keepdim=True)
        spec = F.interpolate(spectrum.unsqueeze(1), size=self.n_spectrum_bins,
                             mode="linear", align_corners=False).squeeze(1)
        return torch.cat([direction, origin, dist, rect, spec], dim=1)

    def forward(self, direction: Tensor, origin: Tensor, spectrum: Tensor, rect: Tensor) -> Tensor:
        """Returns log10(ρ) where ρ = Σscatter/Σdirect, shape (B,). Use 10**out as the ratio."""
        x = self._features(direction, origin, spectrum, rect)
        return self.net(self.in_norm(x)).squeeze(1)

    @staticmethod
    def rescale_scatter(scatter_shape: Tensor, direct: Tensor, log_ratio: Tensor) -> Tensor:
        """Scale a (normalised) scatter SHAPE so that Σscatter = ρ·Σdirect per sample,
        with ρ = 10**log_ratio. scatter_shape/direct: (B,1,D,H,W) or (B,N). Returns the
        physically-scaled scatter (same shape) to be added to `direct` before the final
        max-normalisation of the joined field."""
        dims = tuple(range(1, scatter_shape.dim()))
        s_sum = scatter_shape.clamp_min(0).sum(dims, keepdim=True).clamp_min(1e-30)
        d_sum = direct.clamp_min(0).sum(dims, keepdim=True)
        ratio = torch.pow(10.0, log_ratio).reshape((-1,) + (1,) * (scatter_shape.dim() - 1))
        return scatter_shape * (ratio * d_sum / s_sum)

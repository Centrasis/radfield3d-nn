from torch import Tensor, nn
import torch
from radfield3dnn.utils.mean_sampling import resample_histogram_bilinear


class SimpleSpectraEncoder(nn.Module):
    def __init__(self, in_spectra_dim: int, encoded_spectra_dims: int):
        super().__init__()
        self.in_spectra_dim = in_spectra_dim
        self.encoded_dims = encoded_spectra_dims
        self.encoder = nn.Sequential(
            nn.Linear(in_spectra_dim, encoded_spectra_dims),
            nn.LayerNorm(encoded_spectra_dims),
            nn.SiLU(),
            nn.Linear(encoded_spectra_dims, encoded_spectra_dims),
            nn.SiLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_rebin = resample_histogram_bilinear(x, self.in_spectra_dim)
        x_enc = self.encoder(x_rebin)
        return x_enc


class SpectraProjector(nn.Module):
    def __init__(self, in_spectra_dim: int, out_spectra_dim: int):
        super().__init__()
        self.in_spectra_dim = in_spectra_dim
        self.out_spectra_dim = out_spectra_dim
        self.encoded_dims = out_spectra_dim
        self.spectra_conv_project = nn.Linear(in_spectra_dim, out_spectra_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_rebin = resample_histogram_bilinear(x, self.in_spectra_dim)
        x_enc = self.spectra_conv_project(x_rebin)  # [B, D]
        return x_enc


class SpectraEncoder(nn.Module):
    def __init__(self, in_spectra_dim: int, encoded_spectra_dims: int):
        super().__init__()
        self.in_spectra_dim = in_spectra_dim
        self.encoded_dims = encoded_spectra_dims
        self.activation = nn.SiLU()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.Conv1d(16, 32, kernel_size=1, bias=False), 
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2, groups=32, bias=False),
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.SiLU()
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(1)  # area/shape
        self.pool_max = nn.AdaptiveMaxPool1d(1)  # peak presence
        self.global_project = nn.Linear(64 * 2, encoded_spectra_dims)
        # replace MLP with a Conv1d that collapses the n-bins spectrum to encoded_spectra_dims
        self.spectra_conv_project = nn.Conv1d(
            1, encoded_spectra_dims, kernel_size=in_spectra_dim, bias=True
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * encoded_spectra_dims, encoded_spectra_dims),
            nn.LayerNorm(encoded_spectra_dims),
            nn.SiLU(),
            nn.Linear(encoded_spectra_dims, encoded_spectra_dims),
            nn.SiLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_rebin = resample_histogram_bilinear(x, self.in_spectra_dim)
        x_enc = self.spectra_conv_project(x_rebin.unsqueeze(1)).squeeze(-1)  # [B, D]
        x_conv = self.conv(x.unsqueeze(1))  # [B, 64, bins]
        g_avg = self.pool_avg(x_conv)  # [B, 64, 1]
        g_max = self.pool_max(x_conv)  # [B, 64, 1]
        x_global = torch.cat([g_avg, g_max], dim=1).squeeze(-1)  # [B, 128]
        x_global = self.global_project(x_global)  # [B, D]
        x = self.fuse(torch.cat([x_enc, x_global], dim=-1))
        return x

from .base import MetricBase
import torch
from torch import Tensor
from rftypes import TrainingInputData, RadiationField, RadiationFieldChannel
from typing import Union, Literal
import math


class GammaPassingRate(MetricBase):
    def __init__(self, layer_name: str = None, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean', weight_with_error: bool = False, keep_dim: bool = False, voxel_size_m: float = 0.01, rel_dose_diff: float = 0.03, dist_crit_mm: float = 3.0, dose_threshold: float = 1e-5):
        super().__init__(layer_name=layer_name, reduction=reduction, weight_with_error=weight_with_error, eps=1e-9)
        self.keep_dim = keep_dim
        self.voxel_size_mm = (voxel_size_m * 1000.0, voxel_size_m * 1000.0, voxel_size_m * 1000.0)
        self.rel_dose_diff = rel_dose_diff
        self.dist_crit_mm = dist_crit_mm
        self.dose_threshold = dose_threshold

    @staticmethod
    @torch.no_grad()
    def gamma_index(pred: Tensor, tgt: Tensor, rel_dose_diff=0.03, dist_crit_mm=3.0, voxel_size_mm=(1.0, 1.0, 1.0), dose_threshold=1e-5):
        B, C, D, H, W = tgt.shape
        device = tgt.device

        # normalization (global, per-batch, per-channel) and threshold mask
        max_tgt = tgt.amax(dim=(2, 3, 4), keepdim=True)  # shape: (B,C,1,1,1)
        mask = tgt >= (dose_threshold * max_tgt)

        # denominator for dose difference (global-relative gamma)
        dose_den = torch.clamp(rel_dose_diff * max_tgt, min=1e-12)

        # compute search offsets in voxels (ceil to avoid missing candidates), then keep only inside sphere
        dz_max = int(math.ceil(dist_crit_mm / voxel_size_mm[0]))
        dy_max = int(math.ceil(dist_crit_mm / voxel_size_mm[1]))
        dx_max = int(math.ceil(dist_crit_mm / voxel_size_mm[2]))

        zz, yy, xx = torch.meshgrid(
            torch.arange(-dz_max, dz_max + 1, device=device),
            torch.arange(-dy_max, dy_max + 1, device=device),
            torch.arange(-dx_max, dx_max + 1, device=device),
            indexing='ij'
        )
        offsets = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        # squared distances to reduce sqrt calls in the loop
        distances_mm_sq = (
            (offsets[:, 0] * voxel_size_mm[0]).float().pow(2)
            + (offsets[:, 1] * voxel_size_mm[1]).float().pow(2)
            + (offsets[:, 2] * voxel_size_mm[2]).float().pow(2)
        )
        keep = distances_mm_sq <= (dist_crit_mm * dist_crit_mm)
        offsets = offsets[keep]
        distances_mm_sq = distances_mm_sq[keep]

        # track minimum gamma^2
        gamma_map_sq = torch.full_like(tgt, torch.inf, device=device, dtype=torch.float32)

        for idx, (dz_t, dy_t, dx_t) in enumerate(offsets):
            dz, dy, dx = int(dz_t.item()), int(dy_t.item()), int(dx_t.item())
            shifted = torch.full_like(pred, torch.inf)

            z_dst_start, z_dst_end = max(0, -dz), min(D, D - dz)
            y_dst_start, y_dst_end = max(0, -dy), min(H, H - dy)
            x_dst_start, x_dst_end = max(0, -dx), min(W, W - dx)

            # shift pred by offsets
            if z_dst_start < z_dst_end and y_dst_start < y_dst_end and x_dst_start < x_dst_end:
                shifted[:, :, z_dst_start:z_dst_end, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = pred[:, :, z_dst_start + dz:z_dst_end + dz, y_dst_start + dy:y_dst_end + dy, x_dst_start + dx:x_dst_end + dx]

            dose_diff_sq = ((shifted - tgt) / dose_den).pow(2)
            dist_term_sq = distances_mm_sq[idx] / (dist_crit_mm * dist_crit_mm)
            gamma_candidate_sq = dose_diff_sq + dist_term_sq

            gamma_map_sq = torch.minimum(gamma_map_sq, gamma_candidate_sq)

        gamma_map = torch.sqrt(gamma_map_sq)

        valid = mask & torch.isfinite(gamma_map)
        pass_map = (gamma_map <= 1.0) & valid

        valid_count = valid.view(B, -1).sum(dim=1)
        pass_count = pass_map.view(B, -1).sum(dim=1)
        # avoid division by zero if no valid voxels
        pass_rate = torch.where(valid_count > 0, pass_count.float() / valid_count.float(), torch.zeros_like(pass_count, dtype=torch.float32))
        return pass_rate, pass_map

    @torch.no_grad()
    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        grate, gmap = self.gamma_index(
            prediction,
            target,
            voxel_size_mm=self.voxel_size_mm,
            dist_crit_mm=self.dist_crit_mm,
            rel_dose_diff=self.rel_dose_diff,
            dose_threshold=self.dose_threshold,
        )

        if self.keep_dim:
            return gmap
        else:
            return grate

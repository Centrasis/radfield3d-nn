from .base import MetricBase
from torch import Tensor
from typing import Union, Literal
import torch
from radfield3dnn.preprocessing.airkerma import Airkerma
from radfield3dnn.rftypes import AirKermaField, RadiationFieldChannel, TrainingInputData
import torch.nn.functional as F


class SSIM3D(MetricBase):
    """
    Structural Similarity Index (SSIM) for 3D data.
    """

    @staticmethod
    def make_gaussian_kernel3d(window_size: int, sigma: float, device=None) -> Tensor:
        """
        Creates a 3D Gaussian kernel.
        """
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()  # 1D normalize
        g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        g3d = g3d / g3d.sum()  # normalize
        return g3d.view(1, 1, window_size, window_size, window_size)

    @staticmethod
    def ssim3d(target: Tensor, prediction: Tensor, window_size: int = 7, max_val: float = 1.0, eps: float = 1e-8, kernel_type: Union[Literal['gaussian', 'uniform']] = 'uniform', valid_mask: Tensor = None) -> Tensor:
        assert prediction.shape == target.shape, "Prediction and target must have the same shape."
        B, C, D, H, W = prediction.shape
        if kernel_type == 'gaussian':
            kernel = SSIM3D.make_gaussian_kernel3d(window_size, 1.5, device=prediction.device)
        elif kernel_type == 'uniform':
            kernel = torch.ones((1, 1, window_size, window_size, window_size), device=prediction.device) / (window_size ** 3)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        mu_x = F.conv3d(prediction, kernel, padding=window_size // 2, groups=1)
        mu_y = F.conv3d(target, kernel, padding=window_size // 2, groups=1)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x2 = F.conv3d(prediction * prediction, kernel, padding=window_size // 2, groups=1) - mu_x2
        sigma_y2 = F.conv3d(target * target, kernel, padding=window_size // 2, groups=1) - mu_y2
        sigma_xy = F.conv3d(prediction * target, kernel, padding=window_size // 2, groups=1) - mu_xy

        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)

        if valid_mask is not None:
            # EXCLUDED voxels (e.g. geometry-covered: not scored by the simulation) must not
            # contribute to the score. Windows overlapping them are still slightly biased by the
            # zero-filled values inside the conv, but their CENTERS are dropped from the mean —
            # the previous behaviour scored them as "both zero -> perfect" instead.
            valid = valid_mask.to(ssim_map.dtype)
            ssim = (ssim_map * valid).sum(dim=[1, 2, 3, 4]) / valid.sum(dim=[1, 2, 3, 4]).clamp_min(1.0)
            return ssim
        ssim = ssim_map.mean(dim=[1, 2, 3, 4])
        return ssim

    @staticmethod
    def _sanitize_pair(target: Tensor, prediction: Tensor):
        """Return (target, prediction, valid_mask): non-finite voxels in EITHER side are the
        exclusion set; both sides get them zero-filled (out-of-place) so the convs stay finite,
        and the mask keeps them out of the final mean."""
        valid = torch.isfinite(target) & torch.isfinite(prediction)
        if not valid.all():
            target = torch.where(valid, target, torch.zeros_like(target))
            prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
            return target, prediction, valid
        return target, prediction, None

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        # sanitize on clones — in-place mutation here leaks into metrics evaluated after
        # this one on the same tensors (same class of bug as the scatter-metric -inf leak)
        target, prediction, valid = SSIM3D._sanitize_pair(target, prediction)

        target = target / (target.max() + self.eps)
        prediction = prediction / (prediction.max() + self.eps)
        return SSIM3D.ssim3d(target, prediction, window_size=7, max_val=1.0, eps=self.eps.item(), valid_mask=valid)


class MultiLevelSSIM(MetricBase):
    def __init__(self, levels: int = 3, weight_with_error: bool = False, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean'):
        super().__init__(layer_name=None, reduction=reduction, weight_with_error=weight_with_error, eps=1e-8)
        self.levels = levels

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        ssim_total = torch.zeros((target.size(0),), device=target.device, dtype=target.dtype)
        for level in range(self.levels):
            factor = 2 ** level
            if factor > 1:
                target_level = F.avg_pool3d(target, kernel_size=factor, stride=factor)
                prediction_level = F.avg_pool3d(prediction, kernel_size=factor, stride=factor)
            else:
                target_level = target
                prediction_level = prediction
            ssim_level = SSIM3D.ssim3d(target_level, prediction_level, window_size=7, max_val=1.0, eps=self.eps.item())
            ssim_total += ssim_level
        ssim_avg = ssim_total / self.levels
        return ssim_avg


class GradientSSIM3D(SSIM3D):
    def __init__(
        self,
        window_size: int = 7,
        kernel_type: Literal['gaussian', 'uniform'] = 'uniform',
        spacing=(1.0, 1.0, 1.0),
        weight_with_error: bool = False,
        reduction: Literal['mean', 'median', 'none'] = 'mean',
        eps: float = 1e-8,
    ):
        super().__init__(layer_name=None, reduction=reduction, weight_with_error=weight_with_error, eps=eps)
        self.window_size = int(window_size)
        self.kernel_type = kernel_type
        self.spacing = spacing

    @staticmethod
    def gradient_mag3d(x: Tensor, spacing=(1.0, 1.0, 1.0)) -> Tensor:
        """
        Compute 3D gradient magnitude using torch.gradient.
        Accepts [B, C, D, H, W] or [B, D, H, W]; returns same rank as input (5D).
        """
        # Promote 4D -> 5D (assume missing channel dim)
        squeezed = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeezed = True
        assert x.dim() == 5, "gradient_mag3d expects a 4D or 5D tensor."

        # Gradients along depth, height, width
        gz, gy, gx = torch.gradient(x, spacing=spacing, dim=(2, 3, 4), edge_order=1)
        gmag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-12)

        return gmag.squeeze(1) if squeezed else gmag

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        # sanitize like SSIM3D (both sides, keep the exclusion mask)
        target, prediction, valid = SSIM3D._sanitize_pair(target, prediction)

        # normalize inputs to [0,1]
        target = target / (target.max() + self.eps)
        prediction = prediction / (prediction.max() + self.eps)

        # gradient magnitude maps
        g_t = GradientSSIM3D.gradient_mag3d(target, spacing=self.spacing)
        g_p = GradientSSIM3D.gradient_mag3d(prediction, spacing=self.spacing)

        # scale gradients to [0,1]
        g_t = g_t / (g_t.max() + self.eps)
        g_p = g_p / (g_p.max() + self.eps)

        return SSIM3D.ssim3d(
            g_t, g_p,
            window_size=self.window_size,
            max_val=1.0,
            eps=self.eps.item() if isinstance(self.eps, torch.Tensor) else float(self.eps),
            kernel_type=self.kernel_type,
            valid_mask=valid,
        )


class AirkermaSSIM(MetricBase):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, reduction: Literal['mean', 'median', 'none'] = 'mean', ssim_type: Literal['single', 'multi', 'gradient'] = 'single', ssim_levels: int = None):
        super().__init__(layer_name=None, reduction=reduction, weight_with_error=weight_with_error, eps=1e-8)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)
        assert ssim_levels is not None or ssim_type != 'multi', "For multi-level SSIM, ssim_levels must be specified."
        if ssim_type == 'single':
            self.ssim = SSIM3D(layer_name=None, reduction=reduction, weight_with_error=weight_with_error, eps=1e-8)
        elif ssim_type == 'multi':
            self.ssim = MultiLevelSSIM(levels=ssim_levels, reduction=reduction, weight_with_error=weight_with_error)
        elif ssim_type == 'gradient':
            self.ssim = GradientSSIM3D(reduction=reduction, weight_with_error=weight_with_error)
        else:
            raise ValueError(f"Unknown ssim_type: {ssim_type}")

    def forward(self, target: Union[RadiationFieldChannel, AirKermaField, Tensor], prediction: Union[RadiationFieldChannel, Tensor], input: TrainingInputData = None) -> Tensor:
        if isinstance(prediction, RadiationFieldChannel) and (prediction.spectrum is None or prediction.flux is None):
            return None
        
        # Convert to air-kerma while PRESERVING exclusion: the air-kerma integral needs finite
        # inputs, so non-finite voxels are zero-filled for the compute (out-of-place; never mutate
        # the caller's tensors) and then re-marked -inf in the RESULT — the inner SSIM builds its
        # validity mask from that and drops them from the mean instead of scoring them as zero.
        def _ak_with_exclusion(field):
            if isinstance(field, RadiationFieldChannel):
                flux, spectrum = field.flux, field.spectrum
                invalid = ~torch.isfinite(flux)
                if invalid.any():
                    flux = torch.where(invalid, torch.zeros_like(flux), flux)
                    inv_s = invalid.expand_as(spectrum) if invalid.dim() == spectrum.dim() else invalid.unsqueeze(1).expand_as(spectrum)
                    spectrum = torch.where(inv_s, torch.full_like(spectrum, 1.0 / spectrum.size(1)), spectrum)
                ak = self.airkerma.forward(spectrum, flux)
                if invalid.any():
                    inv_ak = invalid if invalid.dim() == ak.dim() else invalid.unsqueeze(1).expand_as(ak)
                    ak = torch.where(inv_ak, torch.full_like(ak, -torch.inf), ak)
                return ak
            ak = field.air_kerma if isinstance(field, AirKermaField) else field
            return ak

        prediction_airkerma = _ak_with_exclusion(prediction)
        target_airkerma = _ak_with_exclusion(target)

        return self.ssim.forward(target_airkerma, prediction_airkerma, input)

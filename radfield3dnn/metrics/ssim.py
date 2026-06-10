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
    def ssim3d(target: Tensor, prediction: Tensor, window_size: int = 7, max_val: float = 1.0, eps: float = 1e-8, kernel_type: Union[Literal['gaussian', 'uniform']] = 'uniform') -> Tensor:
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

        ssim = ssim_map.mean(dim=[1, 2, 3, 4])
        return ssim

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        mask = torch.isinf(prediction) & (prediction < 0)
        prediction[mask] = 0.0

        target = target / (target.max() + self.eps)
        prediction = prediction / (prediction.max() + self.eps)
        return SSIM3D.ssim3d(target, prediction, window_size=7, max_val=1.0, eps=self.eps.item())


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
        # sanitize like SSIM3D
        mask = torch.isinf(prediction) & (prediction < 0)
        if mask.any():
            prediction = prediction.clone()
            prediction[mask] = 0.0

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
        
        if isinstance(prediction, RadiationFieldChannel):
            invalid_mask = ~torch.isfinite(prediction.flux)
            if invalid_mask.any():
                prediction.flux[invalid_mask] = 0.0
                invalid_mask = invalid_mask.expand_as(prediction.spectrum)
                prediction.spectrum[invalid_mask] = 1.0 / prediction.spectrum.size(1)  # set to uniform if invalid
        elif isinstance(prediction, AirKermaField):
            invalid_mask = ~torch.isfinite(prediction.air_kerma)
            if invalid_mask.any():
                prediction.air_kerma[invalid_mask] = 0.0
        else:
            invalid_mask = ~torch.isfinite(prediction)
            if invalid_mask.any():
                prediction[invalid_mask] = 0.0

        if isinstance(target, RadiationFieldChannel):
            target_airkerma = self.airkerma.forward(target.spectrum, target.flux)
        elif isinstance(target, AirKermaField):
            target_airkerma = target.air_kerma
        else:
            target_airkerma = target

        invalid_mask = ~torch.isfinite(target_airkerma)
        if invalid_mask.any():
            target_airkerma[invalid_mask] = 0.0

        if isinstance(prediction, RadiationFieldChannel):
            prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.flux)
        elif isinstance(prediction, AirKermaField):
            prediction_airkerma = prediction.air_kerma
        else:
            prediction_airkerma = prediction
        return self.ssim.forward(target_airkerma, prediction_airkerma, input)

from torch.nn import functional as F
from torch.nn import Module
import torch
from .base import Loss
from rftypes import TrainingInputData


class Sobel3D(Module):
    def __init__(self):
        super().__init__()
        d = torch.tensor([-1, 0, 1], dtype=torch.float32)
        s = torch.tensor([1, 2, 1], dtype=torch.float32)
        
        kx = d[:, None, None] * s[None, :, None] * s[None, None, :]
        ky = s[:, None, None] * d[None, :, None] * s[None, None, :]
        kz = s[:, None, None] * s[None, :, None] * d[None, None, :]

        self.register_buffer('kx', kx.unsqueeze(0).unsqueeze(0) / 32.0)
        self.register_buffer('ky', ky.unsqueeze(0).unsqueeze(0) / 32.0)
        self.register_buffer('kz', kz.unsqueeze(0).unsqueeze(0) / 32.0)
        
    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.kx = self.kx.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.ky = self.ky.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.kz = self.kz.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D Sobel filter to input tensor x.
        Expects input shape [B, C, D, H, W] and returns gradient magnitude of same shape.
        """
        orig_shape = x.shape
        if x.dim() == 4:
            x = x.unsqueeze(1)
        _, C, _, _, _ = x.shape
        kx = self.kx.repeat(C, 1, 1, 1, 1)
        ky = self.ky.repeat(C, 1, 1, 1, 1)
        kz = self.kz.repeat(C, 1, 1, 1, 1)

        pad = (1,1,1,1,1,1)
        x = F.pad(x, pad, mode='replicate')

        grad_x = F.conv3d(x, kx, groups=C)
        grad_y = F.conv3d(x, ky, groups=C)
        grad_z = F.conv3d(x, kz, groups=C)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-12)
        if orig_shape == x.shape:
            return grad_mag
        else:
            return grad_mag.squeeze(1)


class SobelLoss(Loss):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel3D()

    def forward(self, target: torch.Tensor, prediction: torch.Tensor, input: TrainingInputData) -> torch.Tensor:
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        invalid_mask = ~valid_mask
        if (invalid_mask.any()):
            target[invalid_mask] = 0.0
            prediction[invalid_mask] = 0.0

        sobel_loss = torch.abs(self.sobel(prediction) - self.sobel(target))

        if (invalid_mask.any()):
            sobel_loss = sobel_loss[valid_mask]

        return torch.mean(sobel_loss)


class GradientL1Loss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, prediction: torch.Tensor, input: TrainingInputData) -> torch.Tensor:
        valid_mask = torch.isfinite(target) & torch.isfinite(prediction)
        invalid_mask = ~valid_mask
        if (invalid_mask.any()):
            target[invalid_mask] = 0.0
            prediction[invalid_mask] = 0.0

        gtz, gty, gtx = torch.gradient(target, dim=(2,3,4), spacing=(1.0, 1.0, 1.0), edge_order=1)
        gpz, gpy, gpx = torch.gradient(prediction, dim=(2,3,4), spacing=(1.0, 1.0, 1.0), edge_order=1)
        gt = torch.sqrt(gtx ** 2 + gty ** 2 + gtz ** 2 + 1e-12)
        pt = torch.sqrt(gpx ** 2 + gpy ** 2 + gpz ** 2 + 1e-12)

        if (invalid_mask.any()):
            pt = pt[valid_mask]
            gt = gt[valid_mask]

        grad_l1_loss = F.l1_loss(pt, gt, reduction='mean')
        return grad_l1_loss

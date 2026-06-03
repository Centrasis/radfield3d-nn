import torch
from rftypes import TrainingInputData, RadiationFieldChannel, RadiationField, PositionalInput
from torch import nn
from activations.HistogramNormalize import HistogramNormalize
from RadFiled3D.pytorch.datasets.processing import DataProcessing
import torch.nn.functional as F


class GaussianFluenceNoise(DataProcessing):
    def __init__(self, strength: float = 0.05, eps: float = 1e-8, repeats_per_field: float = 1.0, error_scaled_noise: bool = True):
        """
        Apply gaussian noise to the flux field.
        :param strength: The relative strength of the noise, applied as a percentage of the flux value.
        :param eps: A small value to enable a minimal noise amount for any voxel.
        :param repeats_per_field: How many times this augmentation should be applied per field.
        :param error_scaled_noise: If True, the noise will be scaled by the error of the flux field.
        """
        super().__init__()
        # register as buffers so they persist & move with .to(), but are not trained
        self.strength = torch.tensor(float(strength), dtype=torch.float32, requires_grad=False)
        self.eps = torch.tensor(float(eps), dtype=torch.float32, requires_grad=False)
        self.repeats_per_field = repeats_per_field
        self.error_scaled_noise = error_scaled_noise

    def set_strength(self, value: float):
        with torch.no_grad():
            self.strength.fill_(float(value))
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.strength = self.strength.to(*args, **kwargs)
        self.eps = self.eps.to(*args, **kwargs)
        return self

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply multiplicative gaussian jitter in linear space to (N, C, H, W, D).
        """
        if self.training:
            with torch.no_grad():
                if self.eps.device != x.ground_truth.scatter_field.flux.device:
                    self.eps = self.eps.to(x.ground_truth.scatter_field.flux.device)
                    self.strength = self.strength.to(x.ground_truth.scatter_field.flux.device)

                strength = torch.clamp(self.strength, min=0.0, max=1.0)

                base_sf = x.ground_truth.scatter_field.flux
                base_xb = x.ground_truth.direct_beam.flux

                def apply_jitter(field, err):
                    # Relative jitter ~ N(0, strength), optionally scaled by error
                    rel = torch.randn_like(field) * strength
                    if self.error_scaled_noise and err is not None:
                        err_clean = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
                        scale = torch.clamp(1.0 + err_clean, min=0.0, max=10.0)
                        rel = rel * scale
                    # Clip relative jitter to a safe range to avoid extreme factors
                    max_rel_value = min(0.5, 3.0 * float(strength.item()))
                    rel = torch.clamp(rel, min=-max_rel_value, max=max_rel_value)
                    out = field * (1.0 + rel)
                    # Sanitize and enforce positivity
                    out = torch.nan_to_num(out, nan=float(self.eps.item()), posinf=float(self.eps.item()), neginf=float(self.eps.item()))
                    out = torch.clamp(out, min=self.eps)
                    return out

                scatter_field = apply_jitter(base_sf, x.ground_truth.scatter_field.error)
                direct_beam = apply_jitter(base_xb, x.ground_truth.direct_beam.error)

                # Preserve total flux per sample/channel (helps with smoothing stability)
                reduce_dims = tuple(range(2, base_sf.ndim))
                sf_base_sum = base_sf.sum(dim=reduce_dims, keepdim=True) + self.eps
                sf_new_sum = scatter_field.sum(dim=reduce_dims, keepdim=True) + self.eps
                scatter_field = scatter_field * (sf_base_sum / sf_new_sum)

                xb_base_sum = base_xb.sum(dim=reduce_dims, keepdim=True) + self.eps
                xb_new_sum = direct_beam.sum(dim=reduce_dims, keepdim=True) + self.eps
                direct_beam = direct_beam * (xb_base_sum / xb_new_sum)

            x = TrainingInputData(
                input=x.input,
                ground_truth=RadiationField(
                    scatter_field=RadiationFieldChannel(
                        spectrum=x.ground_truth.scatter_field.spectrum,
                        flux=scatter_field,
                        error=x.ground_truth.scatter_field.error
                    ),
                    direct_beam=RadiationFieldChannel(
                        spectrum=x.ground_truth.direct_beam.spectrum,
                        flux=direct_beam,
                        error=x.ground_truth.direct_beam.error
                    )
                ),
                original_ground_truth=x.original_ground_truth # keep original_ground_truth unchanged
            )
        return x

    def get_parameters(self) -> dict[str, float]:
        return {
            "strength": float(self.strength.item()),
            "eps": float(self.eps.item()),
            "repeats_per_field": self.repeats_per_field,
            "error_scaled_noise": self.error_scaled_noise
        }

    def dataset_multiplier(self) -> float:
        return self.repeats_per_field


class GaussianSpectrumNoise(DataProcessing):
    def __init__(self, strength: float = 0.05, eps: float = 1e-9, repeats_per_field: float = 1.0, error_scaled_noise: bool = True):
        """
        Apply gaussian noise to the spectrum field.
        :param strength: The relative strength of the noise, applied as a percentage of the spectrum value.
        :param eps: A small value to enable a minimal noise amount for any voxel.
        :param repeats_per_field: How many times this augmentation should be applied per field.
        :param error_scaled_noise: If True, the noise will be scaled by the error of
        the spectrum field.
        """
        super().__init__()
        self.strength = torch.tensor(strength, dtype=torch.float32, requires_grad=False)
        self.eps = torch.tensor(eps, dtype=torch.float32, requires_grad=False)
        self.repeats_per_field = repeats_per_field
        self.error_scaled_noise = error_scaled_noise
        self.normalizer = HistogramNormalize(dim=1, enforce_positivity=True)

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if self.training:
            with torch.no_grad():
                if self.eps.device != x.ground_truth.scatter_field.spectrum.device:
                    self.eps = self.eps.to(x.ground_truth.scatter_field.spectrum.device)
                    self.strength = self.strength.to(x.ground_truth.scatter_field.spectrum.device)

                scatter_field = x.ground_truth.scatter_field.spectrum
                direct_beam = x.ground_truth.direct_beam.spectrum

                noise_scatter = torch.randn_like(scatter_field, device=scatter_field.device) * self.strength
                if self.error_scaled_noise and x.ground_truth.scatter_field.error is not None:
                    noise_scatter = noise_scatter * x.ground_truth.scatter_field.error
                noise_scatter = torch.nan_to_num(noise_scatter, nan=0.0, posinf=0.0, neginf=0.0)
                scatter_field = scatter_field + torch.clamp(scatter_field, min=self.eps) * noise_scatter

                noise_beam = torch.randn_like(direct_beam, device=scatter_field.device) * self.strength
                if self.error_scaled_noise and x.ground_truth.direct_beam.error is not None:
                    noise_beam = noise_beam * x.ground_truth.direct_beam.error
                noise_beam = torch.nan_to_num(noise_beam, nan=0.0, posinf=0.0, neginf=0.0)
                direct_beam = direct_beam + torch.clamp(direct_beam, min=self.eps) * noise_beam
                
                scatter_field = self.normalizer(scatter_field)
                direct_beam = self.normalizer(direct_beam)
                
                x = TrainingInputData(
                    input=x.input,
                    ground_truth=RadiationField(
                        scatter_field=RadiationFieldChannel(
                            spectrum=scatter_field.detach(),
                            flux=x.ground_truth.scatter_field.flux,
                            error=x.ground_truth.scatter_field.error# * (1.0 + noise_scatter.mean(dim=1, keepdim=True))
                        ),
                        direct_beam=RadiationFieldChannel(
                            spectrum=direct_beam.detach(),
                            flux=x.ground_truth.direct_beam.flux,
                            error=x.ground_truth.direct_beam.error# * (1.0 + noise_beam.mean(dim=1, keepdim=True))
                        )
                    )
                )
        return x
    
    def get_parameters(self) -> dict[str, float]:
        return {
            "strength": self.strength.item(),
            "eps": self.eps.item(),
            "repeats_per_field": self.repeats_per_field
        }
    
    def dataset_multiplier(self) -> float:
        return self.repeats_per_field

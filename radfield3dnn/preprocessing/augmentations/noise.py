import torch
from radfield3dnn.rftypes import (
    TrainingInputData, RadiationFieldChannel, RadiationField,
    rf3RadiationField, rf3TrainingInputData,
)
from radfiled3d.pytorch.datasets.processing import DataProcessing


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
        # PLAIN FLOATS, deliberately not tensors. The datamodule moves every processing to the GPU
        # (on_after_batch_transfer) and the datasets alias that same list, so a processing holding
        # a device tensor gets pickled to each spawn-started DataLoader worker as CUDA IPC memory
        # — which fails outright where IPC is unavailable ("pidfd_getfd: Operation not permitted",
        # worker exits, run dies at the first epoch). The jitter needs no persistent state: it is
        # built per call on the input's own device.
        self.strength = float(strength)
        self.eps = float(eps)
        self.repeats_per_field = repeats_per_field
        self.error_scaled_noise = error_scaled_noise

    def set_strength(self, value: float):
        self.strength = float(value)
        return self

    def _jitter_flux(self, field: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """Multiplicative gaussian jitter in linear space, with optional error scaling and per-channel
        total-flux preservation."""
        strength = min(max(self.strength, 0.0), 1.0)
        rel = torch.randn_like(field) * strength
        if self.error_scaled_noise and err is not None:
            err_clean = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
            rel = rel * torch.clamp(1.0 + err_clean, min=0.0, max=10.0)
        max_rel_value = min(0.5, 3.0 * strength)
        rel = torch.clamp(rel, min=-max_rel_value, max=max_rel_value)
        out = field * (1.0 + rel)
        out = torch.nan_to_num(out, nan=self.eps, posinf=self.eps, neginf=self.eps)
        out = torch.clamp(out, min=self.eps)
        # Preserve total flux per sample/channel (helps smoothing/stability).
        reduce_dims = tuple(range(2, field.ndim)) if field.ndim >= 3 else tuple(range(field.ndim))
        base_sum = field.sum(dim=reduce_dims, keepdim=True) + self.eps
        new_sum = out.sum(dim=reduce_dims, keepdim=True) + self.eps
        return out * (base_sum / new_sum)

    def _jitter_channel(self, ch: RadiationFieldChannel) -> RadiationFieldChannel:
        if ch is None or ch.flux is None:
            return ch
        return RadiationFieldChannel(spectrum=ch.spectrum, flux=self._jitter_flux(ch.flux, ch.error), error=ch.error)

    def _jitter_ground_truth(self, gt):
        # Type-dispatched: two-channel RadiationField (scatter + direct) OR a single RadiationFieldChannel
        # (e.g. after ChannelsJoin). AirKermaField / anything without flux passes through.
        if isinstance(gt, (RadiationField, rf3RadiationField)):
            return RadiationField(
                scatter_field=self._jitter_channel(gt.scatter_field),
                direct_beam=self._jitter_channel(gt.direct_beam),
                geometry=getattr(gt, "geometry", None),
            )
        if isinstance(gt, RadiationFieldChannel):
            return self._jitter_channel(gt)
        return gt

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """Apply multiplicative gaussian jitter in linear space to the flux channel(s)."""
        if not self.training:
            return x
        with torch.no_grad():
            new_gt = self._jitter_ground_truth(x.ground_truth)
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=new_gt,
                original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None,  # never jitter the reference
            )
        return new_gt

    def get_parameters(self) -> dict[str, float]:
        return {
            "strength": float(self.strength),
            "eps": float(self.eps),
            "repeats_per_field": self.repeats_per_field,
            "error_scaled_noise": self.error_scaled_noise
        }

    def dataset_multiplier(self) -> float:
        return self.repeats_per_field

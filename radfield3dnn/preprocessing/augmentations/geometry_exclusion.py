"""Exclude geometry-covered voxels from the training target.

Voxels inside the phantom/geometry are EXCLUDED FROM SCORING by the simulation, so whatever their
ground-truth flux/spectrum holds is not a statistic — training on it teaches the model noise (and a
floor-sampler may even re-inject such a voxel as a confident "zero dose"). This processing masks
every geometry-covered voxel to the repo's -inf drop sentinel, which removes it from the loss, from
forward2volume's computed voxels AND from anything a sampler kept — it therefore composes with the
ROI/error samplers regardless of order, with one caveat: it must run AFTER any sampler that
REWRITES target values (floor_as_zero injection), so an injected geometry voxel ends dropped, not
zeroed. run_network_task appends it after the samplers accordingly.

The mask comes from the input's geometry channel (density layer, or the binary occupancy mask the
datasets can attach): covered = geometry > threshold. Batches without a geometry channel pass
through untouched, so the processing is safe to enable by default.

TRAINING-ONLY by design: validation/test deliberately score the whole, unmasked field (repo
convention), and the eval paths do not interpret the -inf sentinel.
"""
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing

from radfield3dnn.rftypes import (AirKermaField, RadiationField, RadiationFieldChannel,
                                  TrainingInputData, rf3RadiationField)


class GeometryVoxelExclusion(DataProcessing):
    def __init__(self, threshold: float = 0.0):
        """threshold: a voxel counts as geometry-covered when its geometry-channel value (density,
        or 1.0 in a binary occupancy mask) is strictly greater than this. 0.0 covers both."""
        super().__init__()
        self.threshold = float(threshold)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _covered_mask(self, geometry: torch.Tensor, flux: torch.Tensor) -> torch.Tensor | None:
        """Boolean mask shaped like ``flux`` (True = geometry-covered), or None if shapes cannot
        be reconciled (mismatched grid — better to train on everything than to mask wrongly)."""
        g = geometry
        # geometry arrives as (B, 1, D, W, H) (dataset adds a channel dim) or already (B, D, W, H)
        while g.dim() > flux.dim() and g.shape[1] == 1:
            g = g.squeeze(1)
        if g.dim() == flux.dim() - 1 and flux.shape[1] == 1:      # flux carries a channel dim
            g = g.unsqueeze(1)
        if g.shape != flux.shape:
            return None
        return g > self.threshold

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if not self.training:
            return x
        geometry = getattr(x.input, "geometry", None)
        if geometry is None:
            return x

        gt = x.ground_truth
        ref_flux = None
        if isinstance(gt, (RadiationField, rf3RadiationField)):
            ref_flux = gt.scatter_field.flux if gt.scatter_field is not None else \
                (gt.direct_beam.flux if gt.direct_beam is not None else None)
        elif isinstance(gt, RadiationFieldChannel):
            ref_flux = gt.flux
        elif isinstance(gt, AirKermaField):
            ref_flux = gt.air_kerma
        if ref_flux is None:
            return x

        covered = self._covered_mask(geometry, ref_flux)
        if covered is None or not covered.any():
            return x

        def _drop(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, torch.full_like(t, -torch.inf), t).contiguous()

        def _mask_channel(ch: RadiationFieldChannel) -> RadiationFieldChannel:
            if ch is None:
                return None
            spec = _drop(ch.spectrum, covered.unsqueeze(1).expand_as(ch.spectrum)) if ch.spectrum is not None else None
            return ch._replace(flux=_drop(ch.flux, covered.expand_as(ch.flux)), spectrum=spec)

        if isinstance(gt, (RadiationField, rf3RadiationField)):
            new_gt = gt._replace(scatter_field=_mask_channel(gt.scatter_field),
                                 direct_beam=_mask_channel(gt.direct_beam))
        elif isinstance(gt, RadiationFieldChannel):
            new_gt = _mask_channel(gt)
        else:  # AirKermaField
            new_gt = gt._replace(air_kerma=_drop(gt.air_kerma, covered.expand_as(gt.air_kerma)))

        return x._replace(ground_truth=new_gt)

    @staticmethod
    def create_from_config(config: dict) -> "GeometryVoxelExclusion":
        return GeometryVoxelExclusion(threshold=config.get("threshold", 0.0))

    def get_parameters(self) -> dict:
        return {"threshold": self.threshold}

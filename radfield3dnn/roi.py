"""Shared region-of-interest (ROI) definition for radiation fields.

ONE definition of beam / scatter / floor, used by the metric (AirkermaScatterAccuracy),
the loss (TwoROIGammaLoss) and the voxel sampler (ROIbasedSampler) so all three score and
train on exactly the same regions:

  * beam    = direct_beam >= ``beam_rel`` * direct_max         (≈0.55% of voxels at 0.05)
  * scatter = NOT beam AND joined >= ``scatter_lo`` * joined_max (≈80% at 5e-5; 3.8% MC-noise,
              perfect-model SMAPE ceiling ≈0.985)
  * floor   = NOT beam AND joined <  ``scatter_lo`` * joined_max (≈19.5%, ~60% MC-noise — the
              genuine noise floor; 1e-4 was rejected because it dumps the clean [5e-5,1e-4]
              decade into the floor)

The beam is defined on the DIRECT channel (sharp, high dynamic range) exactly like the air-kerma metric's
beam exclusion; scatter/floor are split on the JOINED flux. Per-field maxima (reduced over the
trailing spatial dims) so a batch of fields is handled at once.
"""
from torch import Tensor

BEAM_REL_DEFAULT = 0.05     # beam = direct >= 0.05 * direct_max  (matches the metric's max_relative_flux)
SCATTER_LO_DEFAULT = 5e-5   # scatter floor = joined >= 5e-5 * joined_max


def _spatial_amax(x: Tensor) -> Tensor:
    """Per-field max over the trailing spatial dims (last 3 for (...,D,H,W), keepdim).
    Falls back to a global max for <3-D inputs."""
    if x.ndim >= 3:
        dims = tuple(range(x.ndim - 3, x.ndim))
        return x.amax(dim=dims, keepdim=True)
    return x.amax()


def compute_roi_masks(direct_flux: Tensor, joined_flux: Tensor,
                      beam_rel: float = BEAM_REL_DEFAULT,
                      scatter_lo: float = SCATTER_LO_DEFAULT) -> tuple[Tensor, Tensor, Tensor]:
    """Return (beam, scatter, floor) boolean masks broadcasting over ``joined_flux``'s shape.

    direct_flux / joined_flux: matching shapes (…, D, H, W); maxima are per-field.
    """
    dmax = _spatial_amax(direct_flux)
    jmax = _spatial_amax(joined_flux)
    beam = direct_flux >= beam_rel * dmax
    above_floor = joined_flux >= scatter_lo * jmax
    scatter = (~beam) & above_floor
    floor = (~beam) & (~above_floor)
    return beam, scatter, floor

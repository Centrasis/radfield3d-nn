"""Per-channel per-field relative normalisation for the two-head PBRFNetCPP.

Sibling to ``ChannelsJoin``. Where ``ChannelsJoin`` sums scatter and direct
into a single channel + flux-weighted spectrum, ``ChannelsSplitRelative``
keeps both flux channels but normalises each by its own per-volume max,
producing unitless shape fields in ``[0, 1]``. The joined spectrum is
preserved at the scatter slot of the output ``RadiationField`` so the
spectrum head still trains against a single histogram per voxel (the
user's design constraint: spectrum stays joined, only flux is split).

The per-volume maxima used for normalisation are stored on the returned
``RadiationField``'s ``geometry`` slot — a lightweight tensor pair
``[scatter_max, direct_max]`` is the cheapest carrier that survives the
existing dataloader collation. Loss and inference helpers read it back
to (a) recover physical flux at inference and (b) compute the per-field
``scatter_max / direct_max`` ratio that the model's ratio head is
trained against (architecture-change.md §0c).
"""
from typing import Union

import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing

from radfield3dnn.rftypes import (
    RadiationField, TrainingInputData, RadiationFieldChannel,
    rf3RadiationField, rf3TrainingInputData,
)


# Marker key for the per-volume scaling metadata. Stored as a tensor of
# shape (2,) = [scatter_max, direct_max] on the geometry slot — the only
# place a per-volume scalar pair fits without changing the
# RadiationField named-tuple contract.
SCALING_META_NDIM = 1


class ChannelsSplitRelative(DataProcessing):
    """Replace raw scatter / direct flux with their per-volume-max relative
    fields. Both output flux channels are in ``[0, 1]``.

    The spectrum on each output channel is the joined flux-weighted
    convex combination (same formula as ``ChannelsJoin``) so the
    downstream single-spectrum-head training contract is unchanged.

    Empty volumes (max == 0) keep the original flux untouched and store
    a scaling pair of ``[0, 0]`` — the loss and inference helpers treat
    that as "no signal" and produce a sensible no-op.
    """

    def __init__(self, eps: float = 1e-8, normalize_per_channel: bool = True):
        super().__init__()
        self.eps = float(eps)
        # When False, keep both flux channels RAW (no per-channel-max division) so the scatter:direct
        # magnitude relation is preserved in the targets — used by the implicit relation-preservation
        # stack (linear_joint normalizer + ChannelMaxBalancedLoss). The joined spectrum and the
        # per-field maxes are still computed/stored exactly as before.
        self.normalize_per_channel = bool(normalize_per_channel)

    def _join_spectrum(self, field) -> torch.Tensor:
        sc = field.scatter_field
        db = field.direct_beam
        scatter_flux = sc.flux
        beam_flux = db.flux
        total_flux = scatter_flux + beam_flux

        spec_ndim = sc.spectrum.ndim
        flux_ndim = total_flux.ndim
        if spec_ndim > flux_ndim:
            assert spec_ndim - flux_ndim == 1, (
                f"Flux/spectrum dim mismatch: flux={total_flux.shape} "
                f"spectrum={sc.spectrum.shape}")
            bin_axis = 0 if spec_ndim == 4 else 1
            total_b   = total_flux.unsqueeze(bin_axis)
            scatter_b = scatter_flux.unsqueeze(bin_axis)
            beam_b    = beam_flux.unsqueeze(bin_axis)
        else:
            total_b, scatter_b, beam_b = total_flux, scatter_flux, beam_flux

        eps = self.eps
        ratio_beam    = (beam_b    + eps) / (total_b + eps)
        ratio_scatter = (scatter_b + eps) / (total_b + eps)
        spectrum = ratio_scatter * sc.spectrum + ratio_beam * db.spectrum

        if spec_ndim == 1:
            spectrum_sum = torch.clamp(torch.sum(spectrum), min=eps)
        elif spec_ndim == 4:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=0, keepdim=True), min=eps)
        else:
            spectrum_sum = torch.clamp(torch.sum(spectrum, dim=1, keepdim=True), min=eps)
        spectrum = spectrum / spectrum_sum

        empty_mask = (total_b <= 0)
        if empty_mask.any():
            spectrum = torch.where(
                empty_mask.expand_as(spectrum),
                torch.zeros_like(spectrum),
                spectrum,
            )
        return spectrum

    def split_channels(self, field) -> RadiationField:
        if field.scatter_field is None or field.direct_beam is None:
            # Single-channel fields pass through unchanged — the caller is
            # already using the joined-flux contract.
            return field

        sc = field.scatter_field
        db = field.direct_beam

        assert torch.isfinite(sc.flux).all() and torch.isfinite(db.flux).all(), (
            "ChannelsSplitRelative: non-finite values in scatter/direct flux — "
            "dataset is invalid.")

        joined_spectrum = self._join_spectrum(field)

        # Per-volume max over the spatial dims, preserving any leading
        # batch dim. Flux is either (D, H, W) (single-volume) or
        # (B, D, H, W) (batched). Reduction over all but the leading dim
        # captures the per-volume scalar in either case.
        def _per_volume_max(t: torch.Tensor) -> torch.Tensor:
            if t.ndim <= 3:
                # No leading batch: produce a 0-D tensor.
                return torch.amax(t)
            reduce_dims = tuple(range(1, t.ndim))
            return torch.amax(t, dim=reduce_dims)

        sc_max = _per_volume_max(sc.flux)
        dr_max = _per_volume_max(db.flux)

        # Avoid divide-by-zero on empty fields; flag with max==0 so the
        # loss knows to skip.
        sc_safe = torch.where(sc_max > 0, sc_max, torch.ones_like(sc_max))
        dr_safe = torch.where(dr_max > 0, dr_max, torch.ones_like(dr_max))

        if not self.normalize_per_channel:
            # Raw mode: keep physical flux (relation preserved in the targets). Maxes are still
            # stored on geometry and the joined spectrum is still computed below.
            sc_rel = sc.flux
            dr_rel = db.flux
        else:
            if sc.flux.ndim <= 3:
                sc_rel = sc.flux / sc_safe
                dr_rel = db.flux / dr_safe
            else:
                sc_rel = sc.flux / sc_safe.view((-1,) + (1,) * (sc.flux.ndim - 1))
                dr_rel = db.flux / dr_safe.view((-1,) + (1,) * (db.flux.ndim - 1))

            sc_rel = torch.clamp(sc_rel, min=0.0, max=1.0)
            dr_rel = torch.clamp(dr_rel, min=0.0, max=1.0)

        # Stack the per-volume maxes into a single tensor on the geometry
        # slot. Layout: (2,) for a single field, (B, 2) when batched.
        if sc_max.ndim == 0:
            scaling = torch.stack([sc_max, dr_max], dim=0)
        else:
            scaling = torch.stack([sc_max, dr_max], dim=-1)

        sc_out = RadiationFieldChannel(
            flux=sc_rel,
            spectrum=joined_spectrum,
            error=sc.error,
        )
        dr_out = RadiationFieldChannel(
            flux=dr_rel,
            # Spectrum lives on the scatter slot per the design constraint;
            # use a zero placeholder on direct so collation is consistent.
            spectrum=torch.zeros_like(joined_spectrum),
            error=db.error,
        )
        return RadiationField(
            scatter_field=sc_out,
            direct_beam=dr_out,
            geometry=scaling,
        )

    def forward(self, x: Union[TrainingInputData, RadiationField]) -> Union[TrainingInputData, RadiationField]:
        if isinstance(x, (TrainingInputData, rf3TrainingInputData)):
            return TrainingInputData(
                input=x.input,
                ground_truth=self.forward(x.ground_truth),
                original_ground_truth=x.original_ground_truth if isinstance(x, TrainingInputData) else None,
            )
        if isinstance(x, (RadiationField, rf3RadiationField)):
            return self.split_channels(x)
        if isinstance(x, RadiationFieldChannel):
            return x
        raise TypeError(
            f"Unsupported type: {type(x)}. Expected TrainingInputData or RadiationField.")

    @classmethod
    def create_from_config(cls, config: dict) -> "ChannelsSplitRelative":
        return ChannelsSplitRelative()

    @staticmethod
    def extract_scaling(field: RadiationField) -> Union[torch.Tensor, None]:
        """Recover the ``[scatter_max, direct_max]`` tensor from a
        post-split ``RadiationField``. Returns ``None`` if the field
        wasn't split (no scaling metadata)."""
        if not hasattr(field, "geometry") or field.geometry is None:
            return None
        return field.geometry

    @staticmethod
    def compute_max_ratio(scaling: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute the per-field ``scatter_max / direct_max`` ratio that
        the model's ratio head is trained against.

        Always returns a positive value: when ``direct_max == 0`` (empty
        beam) the ratio falls back to ``scatter_max / eps`` (typically
        ~1.0), well outside any plausible signal range — callers should
        mask such samples out of the ratio loss."""
        if scaling.ndim == 1:
            sc_max = scaling[0]
            dr_max = scaling[1]
        else:
            sc_max = scaling[..., 0]
            dr_max = scaling[..., 1]
        return sc_max / torch.clamp(dr_max, min=eps)

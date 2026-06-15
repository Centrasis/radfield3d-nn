"""MCFloorCut — remove the Monte-Carlo low-dose noise floor from the GT.

A simulated field has a near-zero MC noise floor in ~99% of voxels (the direct beam
is nonzero in nearly all voxels purely from MC scatter/leakage). Training on that
floor wastes capacity on noise and lets the diffuse background dominate the loss, which
suppresses the high-dose structure the air-kerma metric rewards. Zeroing every voxel
below a small fraction of the per-field peak removes the floor while keeping the bulk of
the physical flux (e.g. rel=1e-3 keeps ~99% of the flux in a small fraction of voxels).

The two channels have very different dynamic range, so the threshold is PER-CHANNEL:
the **scatter** field is diffuse and low-DR (peak ~2e-3, ~100% of voxels within 1e-4 of
peak) — its low values still carry spatial information, so it is cut only very gently
(default 1e-4 ≈ keep ~all voxels). The **direct beam** is sharp and high-DR (peak ~0.5,
but ~99% of its flux lives in ~5-20% of voxels; the other ~80% is MC leakage floor) — so
it is cut harder (default 1e-5 cleans the leakage while keeping ~99% of the beam flux).
Each threshold is relative to that channel's own per-field peak. The spectrum is zeroed
where the flux is, so air-kerma there is exactly zero (matching how the metric masks low
flux).
"""
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel, rf3RadiationField
from radfield3dnn.roi import compute_roi_masks, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT


class MCFloorCut(DataProcessing):
    def __init__(self, rel_threshold: float = 1e-3,
                 scatter_rel: float = None, direct_rel: float = None,
                 use_error: bool = False, error_threshold: float = 0.5,
                 as_neginf: bool = False,
                 beam_rel: float = BEAM_REL_DEFAULT, scatter_lo: float = SCATTER_LO_DEFAULT):
        super().__init__()
        # Per-channel value thresholds; fall back to the single rel_threshold when not given.
        self.scatter_rel = float(scatter_rel if scatter_rel is not None else rel_threshold)
        self.direct_rel = float(direct_rel if direct_rel is not None else rel_threshold)
        # MASKING mode (``as_neginf``): instead of ZEROING the noise floor per channel, mask the
        # shared FLOOR ROI (radfield3dnn.roi: NOT beam AND joined < scatter_lo·joined_max) to -inf
        # in BOTH channels — the same sentinel the ROIbasedSampler/losses use, so the loss simply
        # excludes those voxels (a zero floor still trains the net toward zero and feeds the
        # all-zero collapse; -inf removes that pressure entirely). It is JOIN-SAFE: it masks on the
        # joined floor, so a scatter-region voxel whose direct channel is floor is NOT masked (its
        # joined value is above the floor) — unlike per-channel zeroing, where -inf in one channel
        # would poison the joined voxel after ChannelsJoin. It is TRAINING-ONLY (validation sees the
        # whole, unmasked field) and matches the ROI floor so the sampler can re-inject a few floor
        # voxels as genuine zeros.
        self.as_neginf = bool(as_neginf)
        self.beam_rel = float(beam_rel)
        self.scatter_lo = float(scatter_lo)
        # Error-based mode: zero each channel where its per-voxel MC error flags the voxel
        # as noise-dominated (error >= error_threshold; the error field is ~binary {0,1}).
        # This is data-adaptive — the dropped fraction is whatever is genuinely noise-dominated
        # per field. After ChannelsJoin a voxel only vanishes if it is noise in BOTH channels,
        # so the joined target keeps most voxels automatically while stripping the MC floor where
        # it truly dominates.
        self.use_error = bool(use_error)
        self.error_threshold = float(error_threshold)

    def _cut(self, ch: RadiationFieldChannel, rel: float) -> RadiationFieldChannel:
        if ch is None or ch.flux is None:
            return ch
        flux = ch.flux
        sp = (-3, -2, -1)  # spatial dims; works for (C,D,H,W) and (B,C,D,H,W)
        if self.use_error and getattr(ch, "error", None) is not None:
            keep = (ch.error < self.error_threshold).to(flux.dtype)  # keep confident voxels
        else:
            thr = flux.amax(dim=sp, keepdim=True) * rel
            keep = (flux >= thr).to(flux.dtype)
        new_spec = None
        if ch.spectrum is not None:
            # keep is (…,1,D,H,W); spectrum is (…,S,D,H,W) → broadcast over the bin axis
            new_spec = ch.spectrum * keep
        return ch._replace(flux=flux * keep, spectrum=new_spec)

    def _mask_floor_neginf(self, gt):
        """Mask the shared FLOOR ROI to -inf in both channels (join-safe, see __init__)."""
        scatter = gt.scatter_field
        direct = gt.direct_beam
        sc_flux = scatter.flux if scatter is not None else None
        dr_flux = direct.flux if direct is not None else None
        if sc_flux is None and dr_flux is None:
            return gt
        joined = (sc_flux if sc_flux is not None else 0) + (dr_flux if dr_flux is not None else 0)
        direct_for_beam = dr_flux if dr_flux is not None else joined  # no split → beam from joined
        _, _, floor = compute_roi_masks(direct_for_beam, joined, self.beam_rel, self.scatter_lo)
        if not bool(floor.any()):
            return gt

        def _mask(ch):
            if ch is None or ch.flux is None:
                return ch
            neg = torch.full_like(ch.flux, -torch.inf)
            new_flux = torch.where(floor, neg, ch.flux)
            new_spec = ch.spectrum
            if ch.spectrum is not None:
                fm = floor.expand_as(ch.spectrum)
                new_spec = torch.where(fm, torch.full_like(ch.spectrum, -torch.inf), ch.spectrum)
            return ch._replace(flux=new_flux, spectrum=new_spec)

        return gt._replace(scatter_field=_mask(scatter), direct_beam=_mask(direct))

    def forward(self, x):
        gt = x.ground_truth
        if not isinstance(gt, (RadiationField, rf3RadiationField)):
            return x
        if self.as_neginf:
            # Masking mode is TRAINING-ONLY: validation/test must see the whole, unmasked field
            # to check whether the model predicts the full volume despite the sparse training mask.
            if not self.training:
                return x
            return x._replace(ground_truth=self._mask_floor_neginf(gt))
        return x._replace(ground_truth=gt._replace(
            scatter_field=self._cut(gt.scatter_field, self.scatter_rel),
            direct_beam=self._cut(gt.direct_beam, self.direct_rel),
        ))

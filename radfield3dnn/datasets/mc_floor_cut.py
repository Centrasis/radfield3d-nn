"""MCFloorCut — remove the Monte-Carlo low-dose noise floor from the GT.

A simulated field has a near-zero MC noise floor in ~99% of voxels (the direct beam
is nonzero in 98.6% of DS03 voxels purely from MC scatter/leakage). Training on that
floor wastes capacity on noise and lets the diffuse background dominate the loss, which
suppresses the high-dose structure the air-kerma metric rewards. Zeroing every voxel
below a small fraction of the per-field peak removes the floor while keeping >99% of the
physical flux (DS03: rel=1e-3 keeps 99.0% of the flux in 8.5% of voxels).

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
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import RadiationField, RadiationFieldChannel, rf3RadiationField


class MCFloorCut(DataProcessing):
    def __init__(self, rel_threshold: float = 1e-3,
                 scatter_rel: float = None, direct_rel: float = None,
                 use_error: bool = False, error_threshold: float = 0.5):
        super().__init__()
        # Per-channel value thresholds; fall back to the single rel_threshold when not given.
        self.scatter_rel = float(scatter_rel if scatter_rel is not None else rel_threshold)
        self.direct_rel = float(direct_rel if direct_rel is not None else rel_threshold)
        # Error-based mode: zero each channel where its per-voxel MC error flags the voxel
        # as noise-dominated (error >= error_threshold; the DS03 error field is ~binary {0,1}).
        # This is data-adaptive — the dropped fraction is whatever is genuinely noise-dominated
        # per field (scatter ~8.8%, direct ~94.6% leakage floor). After ChannelsJoin a voxel
        # only vanishes if it is noise in BOTH channels (~8% on DS03), so the joined target keeps
        # >=~80% of voxels automatically while stripping the MC floor where it truly dominates.
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

    def forward(self, x):
        gt = x.ground_truth
        if not isinstance(gt, (RadiationField, rf3RadiationField)):
            return x
        return x._replace(ground_truth=gt._replace(
            scatter_field=self._cut(gt.scatter_field, self.scatter_rel),
            direct_beam=self._cut(gt.direct_beam, self.direct_rel),
        ))

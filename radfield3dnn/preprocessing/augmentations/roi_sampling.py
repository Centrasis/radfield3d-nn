"""ROI-based voxel sampler — the structural twin of ErrorbasedImportanceSampler, but it samples
by the shared beam / scatter / floor ROIs (radfield3dnn.roi) instead of the MC-error layer.

Per field it:
  * KEEPS the beam in full (beam = direct >= beam_rel * direct_max; ``beam_keep_ratio`` < 1 keeps a
    random fraction instead — default 1.0 = keep all);
  * randomly samples the SCATTER ROI, ``scatter_ratio`` voxels per kept beam voxel (0..inf,
    default 2 → twice as many scatter voxels as beam voxels);
  * randomly samples the FLOOR, ``floor_ratio`` voxels per kept beam voxel, CAPPED by however many
    floor voxels exist (the floor may be empty — then nothing is taken).

Everything else is masked out by setting flux (and spectrum) to -inf — the same convention the
losses use, so TwoROIGammaLoss/AirkermaScatterAccuracy see exactly the sampled voxels.

Because the beam is kept deterministically but the scatter/floor subsets are drawn fresh on every
call, repeating each field within an epoch (``dataset_multiplier`` > 1) makes the model see
DIFFERENT parts of the scatter ROI each repeat while always re-seeing the beam.
"""
import torch
from RadFiled3D.pytorch.datasets.processing import DataProcessing

from radfield3dnn.rftypes import AirKermaField, TrainingInputData, RadiationFieldChannel, RadiationField
from radfield3dnn.roi import compute_roi_masks, BEAM_REL_DEFAULT, SCATTER_LO_DEFAULT


class ROIbasedSampler(DataProcessing):
    def __init__(self, beam_rel: float = BEAM_REL_DEFAULT, scatter_lo: float = SCATTER_LO_DEFAULT,
                 beam_keep_ratio: float = 1.0, scatter_ratio: float = 2.0, floor_ratio: float = 1.0,
                 field_multiplier: float = 3.0, floor_as_zero: bool = True,
                 scatter_ratio_end: float = None, schedule_switch: float = 0.8):
        """
        :param beam_rel:       beam = direct >= beam_rel * direct_max (matches the metric/loss).
        :param scatter_lo:     scatter floor = joined >= scatter_lo * joined_max.
        :param beam_keep_ratio: fraction of beam voxels to keep [0..1], default 1.0 (keep all).
        :param scatter_ratio:  scatter voxels to sample per KEPT beam voxel [0..inf), default 2.0.
        :param floor_ratio:    floor voxels to sample per kept beam voxel [0..inf), capped by the
                               number of floor voxels that exist (may be 0), default 1.0.
        :param field_multiplier: how many times each field is repeated per epoch (>1). Each repeat
                               re-keeps the beam but draws a fresh random scatter/floor subset.
        :param floor_as_zero:  if True (default), the sampled FLOOR voxels are set to a genuine 0
                               (not their noisy MC value), so the network is shown "a bit of zero"
                               and does not hallucinate signal in the background; the rest of the
                               floor (and non-sampled scatter) stays -inf/masked. If False the
                               sampled floor keeps its original (noisy) target value.
        :param scatter_ratio_end: if set, the scatter_ratio is SCHEDULED — ``scatter_ratio`` for the
                               first ``schedule_switch`` fraction of training, then this value for
                               the rest (e.g. start 2.5 → end 1.0 over the last 20% for fine-tuning).
                               Driven by LimitedAugmentation.set_schedule_progress. None = constant.
        :param schedule_switch: training-progress fraction [0..1] at which scatter_ratio switches to
                               scatter_ratio_end (default 0.8 = last 20% of epochs).
        """
        super().__init__()
        assert 0.0 <= beam_keep_ratio <= 1.0, "beam_keep_ratio must be in [0, 1]"
        assert scatter_ratio >= 0.0 and floor_ratio >= 0.0, "ratios must be >= 0"
        self.beam_rel = float(beam_rel)
        self.scatter_lo = float(scatter_lo)
        self.beam_keep_ratio = float(beam_keep_ratio)
        self.scatter_ratio = float(scatter_ratio)
        self.floor_ratio = float(floor_ratio)
        self.field_multiplier = float(field_multiplier)
        self.floor_as_zero = bool(floor_as_zero)
        self.scatter_ratio_end = float(scatter_ratio_end) if scatter_ratio_end is not None else None
        self.schedule_switch = float(schedule_switch)
        self._progress = 0.0   # 0→1 over the active training window (set by LimitedAugmentation)

    def set_schedule_progress(self, p: float):
        """Hook driven by LimitedAugmentation: fraction of the active window elapsed (0→1)."""
        self._progress = float(p)

    @property
    def _eff_scatter_ratio(self) -> float:
        """Current scatter_ratio honouring the optional start→end schedule."""
        if self.scatter_ratio_end is None:
            return self.scatter_ratio
        return self.scatter_ratio_end if self._progress >= self.schedule_switch else self.scatter_ratio

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    @staticmethod
    def _subsample(mask: torch.Tensor, n_keep: int) -> torch.Tensor:
        """Return a boolean mask with at most ``n_keep`` True entries chosen uniformly at random
        from the True entries of ``mask`` (a fresh draw every call)."""
        idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0 or n_keep <= 0:
            return torch.zeros_like(mask)
        if n_keep >= idx.numel():
            return mask.clone()
        perm = torch.randperm(idx.numel(), device=mask.device)[:n_keep]
        out = torch.zeros(mask.numel(), dtype=torch.bool, device=mask.device)
        out[idx[perm]] = True
        return out.view_as(mask)

    def _keep_masks(self, direct: torch.Tensor, joined: torch.Tensor):
        """Return (keep_real, keep_floor) boolean masks: keep_real = all (or beam_keep_ratio of)
        beam + a random scatter_ratio×beam scatter subset (targets kept at their real value);
        keep_floor = a random floor_ratio×beam floor subset (capped; set to 0 when floor_as_zero).
        Everything in NEITHER mask is dropped (-inf)."""
        beam, scatter, floor = compute_roi_masks(direct, joined, self.beam_rel, self.scatter_lo)

        keep_beam = beam if self.beam_keep_ratio >= 1.0 else \
            self._subsample(beam, int(round(self.beam_keep_ratio * int(beam.sum()))))
        n_beam = int(keep_beam.sum())

        keep_scatter = self._subsample(scatter, int(round(self._eff_scatter_ratio * n_beam)))
        keep_floor = self._subsample(floor, int(round(self.floor_ratio * n_beam)))  # capped inside
        return keep_beam | keep_scatter, keep_floor

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if not self.training:
            return x

        # Derive the ROIs from the preserved ORIGINAL GT (uncut, with the SEPARATE direct channel) — the
        # mask is still applied only to ground_truth below. ground_truth is joined/floor-cut by this point,
        # so deriving from it would compute the beam ROI from joined flux (no direct channel) and shrink
        # the scatter region — under-sampling the low-flux scatter.
        gt = x.original_ground_truth if x.original_ground_truth is not None else x.ground_truth
        if isinstance(gt, RadiationField):
            scatter_flux = gt.scatter_field.flux if gt.scatter_field is not None else None
            direct_flux = gt.direct_beam.flux if gt.direct_beam is not None else None
            if direct_flux is None and scatter_flux is None:
                return x
            joined = (scatter_flux if scatter_flux is not None else 0) + \
                     (direct_flux if direct_flux is not None else 0)
            direct = direct_flux if direct_flux is not None else joined  # no split -> beam from joined
        elif isinstance(gt, RadiationFieldChannel):
            joined = direct = gt.flux
        else:
            return x  # AirKermaField etc. — ROI sampling not defined

        keep_real, keep_floor = self._keep_masks(direct, joined)
        # zero_mask: floor voxels re-injected as genuine 0 (only when floor_as_zero); otherwise the
        # sampled floor keeps its real value and just joins keep_real.
        if self.floor_as_zero:
            zero_mask = keep_floor
        else:
            keep_real = keep_real | keep_floor
            zero_mask = torch.zeros_like(keep_real)
        drop_mask = ~(keep_real | zero_mask)   # neither kept-real nor zero-injected → -inf
        if not drop_mask.any() and not zero_mask.any():
            return x

        def _apply(t: torch.Tensor, dm: torch.Tensor, zm: torch.Tensor) -> torch.Tensor:
            # zero-inject first (floor → 0), then mask the dropped voxels to -inf.
            out = torch.where(zm, torch.zeros_like(t), t)
            out = torch.where(dm, torch.full_like(t, -torch.inf), out)
            return out.contiguous()

        def _mask_channel(ch: RadiationFieldChannel) -> RadiationFieldChannel:
            if ch is None:
                return None
            spec = None
            if ch.spectrum is not None:
                dm_s = drop_mask.expand_as(ch.spectrum)
                zm_s = zero_mask.expand_as(ch.spectrum)
                spec = _apply(ch.spectrum, dm_s, zm_s)
            return ch._replace(flux=_apply(ch.flux, drop_mask, zero_mask), spectrum=spec)

        tgt = x.ground_truth
        if isinstance(tgt, RadiationField):
            new_gt = tgt._replace(scatter_field=_mask_channel(tgt.scatter_field),
                                  direct_beam=_mask_channel(tgt.direct_beam))
        elif isinstance(tgt, RadiationFieldChannel):
            new_gt = _mask_channel(tgt)
        elif isinstance(tgt, AirKermaField):
            new_gt = tgt._replace(air_kerma=_apply(tgt.air_kerma, drop_mask, zero_mask))
        else:
            raise TypeError("Unsupported ground truth type for ROI sampling.")

        return x._replace(ground_truth=new_gt)

    def dataset_multiplier(self) -> float:
        return self.field_multiplier

    def get_parameters(self) -> dict[str, float]:
        return {
            "beam_rel": self.beam_rel,
            "scatter_lo": self.scatter_lo,
            "beam_keep_ratio": self.beam_keep_ratio,
            "scatter_ratio": self.scatter_ratio,
            "floor_ratio": self.floor_ratio,
            "field_multiplier": self.field_multiplier,
            "floor_as_zero": self.floor_as_zero,
            "scatter_ratio_end": self.scatter_ratio_end if self.scatter_ratio_end is not None else self.scatter_ratio,
            "schedule_switch": self.schedule_switch,
        }

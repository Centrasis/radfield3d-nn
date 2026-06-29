from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import AirKermaField, TrainingInputData, RadiationFieldChannel, RadiationField, PositionalInput
import torch



class ErrorbasedImportanceSampler(DataProcessing):
    def __init__(self, max_drop_chance: float = 0.99, high_fluence_keep_threshold: float = 0.8,
                 max_drop_chance_end: float | None = None):
        """
        Drop voxels with low error probabilitys with a certain chance in (N, C, H, W, D).
        Invalid voxels are marked with -inf flux.
        :param max_drop_chance: Drop chance at the START of the schedule [0.0, 1.0).
        :param high_fluence_keep_threshold: Keep all voxels with a relative flux above this threshold [0.0, 1.0)
        :param max_drop_chance_end: Drop chance at the END of the schedule
            [0.0, 1.0). When set, the effective drop chance is annealed linearly
            from ``max_drop_chance`` to ``max_drop_chance_end`` as the wrapping
            ``LimitedAugmentation`` advances the schedule (high→low denoises
            aggressively early, then reintroduces voxels before the sampler is
            switched off for fine-tuning — removes the need to hand-tune a single
            drop rate). When None the rate is constant.
        """
        super().__init__()
        assert 0.0 <= max_drop_chance < 1.0, "max_drop_chance must be in [0.0, 1.0)"
        assert 0.0 <= high_fluence_keep_threshold < 1.0, "high_fluence_keep_threshold must be in [0.0, 1.0)"
        assert max_drop_chance_end is None or 0.0 <= max_drop_chance_end < 1.0, "max_drop_chance_end must be in [0.0, 1.0)"
        self.max_drop_chance = max_drop_chance
        self.high_fluence_keep_threshold = high_fluence_keep_threshold
        self.max_drop_chance_end = max_drop_chance if max_drop_chance_end is None else max_drop_chance_end
        # Schedule progress in [0, 1], driven externally by LimitedAugmentation.
        self._progress = 0.0

    def set_schedule_progress(self, progress: float) -> None:
        """Generic hook called by LimitedAugmentation with the fraction of the
        augmentation's active window elapsed (0 at start_epoch, 1 at end_epoch)."""
        self._progress = float(min(1.0, max(0.0, progress)))

    @property
    def effective_max_drop_chance(self) -> float:
        return self.max_drop_chance + (self.max_drop_chance_end - self.max_drop_chance) * self._progress

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Drop voxels with low error probabilitys with a certain chance in (N, C, H, W, D).
        Invalid voxels are marked with -inf flux.
        """
        if not self.training:
            return x

        # Derive the drop mask from the preserved ORIGINAL GT (uncut, separate channels with the per-
        # channel error); the mask is applied to ground_truth below. ground_truth is joined/floor-cut by
        # this point, so deriving from it would lose the per-channel error/fluence the sampler needs.
        gt = x.original_ground_truth if x.original_ground_truth is not None else x.ground_truth

        assert (isinstance(gt, RadiationField) and (gt.scatter_field.error is not None or (gt.direct_beam is not None and gt.direct_beam.error is not None))) or (isinstance(gt, RadiationFieldChannel) and gt.error is not None), "Error field is missing"
        if isinstance(gt, RadiationField):
            error_field = gt.scatter_field.error
            fluence_field = gt.scatter_field.flux
            spectra_field = gt.scatter_field.spectrum
            if gt.direct_beam is not None:
                if error_field is None:
                    error_field = gt.direct_beam.error
                else:
                    # Out-of-place: error_field aliases gt.scatter_field.error;
                    # in-place ops here would corrupt the (possibly cached) GT.
                    error_field = (error_field + gt.direct_beam.error) / 2.0
                if fluence_field is None:
                    fluence_field = gt.direct_beam.flux
                else:
                    fluence_field = (fluence_field + gt.direct_beam.flux) / 2.0
        else:
            error_field = gt.error
            fluence_field = gt.flux
            spectra_field = gt.spectrum

        # Out-of-place shift/scale — never mutate the GT tensors in place.
        error_field = error_field - error_field.min()  # shift to min = 0
        if error_field.max() > 0:
            error_field = error_field / error_field.max()  # normalize to [0 = no error, 1 = max error]
        error_field = torch.where((fluence_field >= fluence_field.max() * self.high_fluence_keep_threshold), torch.tensor(0.0, device=error_field.device), error_field)  # never drop high flux voxels
        error_field = 1.0 - error_field  # invert to get drop probability
        error_field = torch.clamp(error_field, min=1.0 - self.effective_max_drop_chance, max=1.0)
        rand_field = torch.rand_like(error_field)
        drop_mask = (rand_field > error_field)
        if drop_mask.any():
            if drop_mask.all():
                # avoid dropping all voxels
                idx = torch.randint(0, drop_mask.numel(), (1,), device=drop_mask.device)
                drop_mask = drop_mask.view(-1)
                drop_mask[idx] = False
                drop_mask = drop_mask.view_as(error_field)
            drop_mask_spectra = drop_mask.expand_as(spectra_field) if spectra_field is not None else None
            
            if isinstance(x.ground_truth, RadiationField) or isinstance(x.ground_truth, RadiationFieldChannel):
                fluence_inf = torch.full_like(x.ground_truth.scatter_field.flux, -torch.inf) if isinstance(x.ground_truth, RadiationField) else torch.full_like(x.ground_truth.flux, -torch.inf)
                spectra_inf = fluence_inf.expand_as(spectra_field) if spectra_field is not None else None
                x = TrainingInputData(
                    input=x.input,
                    ground_truth=RadiationField(
                        scatter_field=RadiationFieldChannel(
                            flux=torch.where(drop_mask, fluence_inf, x.ground_truth.scatter_field.flux).contiguous(),
                            spectrum=torch.where(drop_mask_spectra, spectra_inf, x.ground_truth.scatter_field.spectrum).contiguous() if drop_mask_spectra is not None else None,
                            error=x.ground_truth.scatter_field.error
                        ) if x.ground_truth.scatter_field is not None else None,
                        direct_beam=RadiationFieldChannel(
                            flux=torch.where(drop_mask, fluence_inf, x.ground_truth.direct_beam.flux).contiguous(),
                            spectrum=torch.where(drop_mask_spectra, spectra_inf, x.ground_truth.direct_beam.spectrum).contiguous() if drop_mask_spectra is not None else None,
                            error=x.ground_truth.direct_beam.error,
                        ) if x.ground_truth.direct_beam is not None else None,
                        geometry=x.ground_truth.geometry
                    ) if isinstance(x.ground_truth, RadiationField) else RadiationFieldChannel(
                        flux=torch.where(drop_mask, fluence_inf, x.ground_truth.flux).contiguous(),
                        spectrum=torch.where(drop_mask_spectra, spectra_inf, x.ground_truth.spectrum).contiguous() if drop_mask_spectra is not None else None,
                        error=x.ground_truth.error
                    ),
                    original_ground_truth=x.original_ground_truth
                )
            elif isinstance(x.ground_truth, AirKermaField):
                x = TrainingInputData(
                    input=x.input,
                    ground_truth=AirKermaField(
                        air_kerma=torch.where(drop_mask, torch.tensor(float('-inf'), device=x.ground_truth.air_kerma.device, dtype=x.ground_truth.air_kerma.dtype), x.ground_truth.air_kerma).contiguous(),
                        geometry=x.ground_truth.geometry
                    ),
                    original_ground_truth=x.original_ground_truth
                )
            else:
                raise TypeError("Unsupported ground truth type for importance sampling.")
        return x
    
    def dataset_multiplier(self) -> float:
        return 1.5
    
    def get_parameters(self) -> dict[str, float]:
        return {
            "max_drop_chance": self.max_drop_chance,
            "max_drop_chance_end": self.max_drop_chance_end,
            "high_fluence_keep_threshold": self.high_fluence_keep_threshold
        }

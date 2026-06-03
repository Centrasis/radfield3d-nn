from RadFiled3D.pytorch.datasets.processing import DataProcessing
from rftypes import AirKermaField, TrainingInputData, RadiationFieldChannel, RadiationField, PositionalInput
import torch



class ErrorbasedImportanceSampler(DataProcessing):
    def __init__(self, max_drop_chance: float = 0.99, high_fluence_keep_threshold: float = 0.8):
        """
        Drop voxels with low error probabilitys with a certain chance in (N, C, H, W, D).
        Invalid voxels are marked with -inf flux.
        :param max_drop_chance: Maximum chance to drop a voxel [0.0, 1.0)
        :param high_fluence_keep_threshold: Keep all voxels with a relative flux above this threshold [0.0, 1.0)
        """
        super().__init__()
        assert 0.0 <= max_drop_chance < 1.0, "max_drop_chance must be in [0.0, 1.0)"
        assert 0.0 <= high_fluence_keep_threshold < 1.0, "high_fluence_keep_threshold must be in [0.0, 1.0)"
        self.max_drop_chance = max_drop_chance
        self.high_fluence_keep_threshold = high_fluence_keep_threshold

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
                    error_field += gt.direct_beam.error
                    error_field /= 2.0
                if fluence_field is None:
                    fluence_field = gt.direct_beam.flux
                else:
                    fluence_field += gt.direct_beam.flux
                    fluence_field /= 2.0
        else:
            error_field = gt.error
            fluence_field = gt.flux
            spectra_field = gt.spectrum

        error_field -= error_field.min()  # shift to min = 0
        if error_field.max() > 0:
            error_field /= error_field.max()  # normalize to [0 = no error, 1 = max error]
        error_field = torch.where((fluence_field >= fluence_field.max() * self.high_fluence_keep_threshold), torch.tensor(0.0, device=error_field.device), error_field)  # never drop high flux voxels
        error_field = 1.0 - error_field  # invert to get drop probability
        error_field = torch.clamp(error_field, min=1.0 - self.max_drop_chance, max=1.0)
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
            "high_fluence_keep_threshold": self.high_fluence_keep_threshold
        }

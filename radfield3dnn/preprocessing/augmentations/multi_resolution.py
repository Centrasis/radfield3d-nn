import random
import torch
import torch.nn.functional as F
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import TrainingInputData, RadiationField, rf3RadiationField, RadiationFieldChannel, AirKermaField


class MultiResolutionResampling(DataProcessing):
    """Randomly re-grid the TRAINING ground truth so the network sees multiple voxel scales.

    Purpose: teach the scale axis for IPE-encoded models (an integrated encoding trained at a
    single region width never learns how the field changes with scale). Per training batch one
    scale factor s is drawn from ``scales``:

      * s < 1 (coarser): ``area`` pooling — EXACT label, voxel averages aggregate exactly.
      * s > 1 (finer):   trilinear interpolation — APPROXIMATE label (the true sub-voxel field is
        unknown); it biases the fine-scale target toward trilinear smoothness, which is the
        honest best available without finer simulations.

    Spectra are pooled FLUX-WEIGHTED (pool(flux·spec)/pool(flux)) and renormalized per voxel, so
    a bright voxel dominates the merged spectrum exactly as it would in a real coarser tally.
    Eval/validation is untouched (module in eval() is a no-op), and the module must run BEFORE
    any -inf-masking stage (floor cut / ROI sampling): resampling a masked field would smear the
    sentinels. Pair with an IPE location encoder using ``auto_region_width: true`` so the damping
    follows the resampled grid automatically.
    """

    def __init__(self, scales=(0.5, 0.75, 1.0, 1.5, 2.0)):
        super().__init__()
        assert all(s > 0 for s in scales) and any(abs(s - 1.0) > 1e-9 for s in scales), \
            f"scales must be positive and include at least one non-1 factor, got {scales}"
        self.scales = tuple(float(s) for s in scales)

    def _resample_channel(self, ch: RadiationFieldChannel, scale: float) -> RadiationFieldChannel:
        flux = ch.flux            # (B, 1, x, y, z)
        assert torch.isfinite(flux).all(), \
            "MultiResolutionResampling must run BEFORE -inf-masking stages (floor cut / ROI sampling)."
        mode = "area" if scale < 1.0 else "trilinear"
        kw = {} if mode == "area" else {"align_corners": False}
        new_flux = F.interpolate(flux, scale_factor=scale, mode=mode, **kw)
        new_spec = None
        if ch.spectrum is not None:
            weighted = F.interpolate(ch.spectrum * flux, scale_factor=scale, mode=mode, **kw)
            new_spec = weighted / new_flux.clamp_min(1e-12)
            new_spec = new_spec / new_spec.sum(dim=1, keepdim=True).clamp_min(1e-12)
            new_spec = new_spec * (new_flux > 0)
        new_err = None
        if ch.error is not None:
            new_err = F.interpolate(ch.error, scale_factor=scale, mode=mode, **kw)
        return ch._replace(flux=new_flux, spectrum=new_spec, error=new_err)

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        if not self.training:
            return x
        scale = random.choice(self.scales)
        if abs(scale - 1.0) < 1e-9:
            return x
        gt = x.ground_truth
        if isinstance(gt, (RadiationField, rf3RadiationField)):
            new_gt = gt._replace(
                scatter_field=self._resample_channel(gt.scatter_field, scale) if gt.scatter_field is not None else None,
                direct_beam=self._resample_channel(gt.direct_beam, scale) if gt.direct_beam is not None else None,
            )
        elif isinstance(gt, RadiationFieldChannel):
            new_gt = self._resample_channel(gt, scale)
        elif isinstance(gt, AirKermaField):
            new_gt = gt._replace(air_kerma=F.interpolate(
                gt.air_kerma, scale_factor=scale, mode="area" if scale < 1.0 else "trilinear",
                **({} if scale < 1.0 else {"align_corners": False})))
        else:
            raise TypeError("Unsupported ground truth type for multi-resolution resampling.")
        # The ROI sampler derives its beam/scatter masks from original_ground_truth (the clean
        # pre-masking snapshot) — re-grid it identically so masks match the resampled target.
        new_orig = x.original_ground_truth
        if new_orig is not None:
            if isinstance(new_orig, (RadiationField, rf3RadiationField)):
                new_orig = new_orig._replace(
                    scatter_field=self._resample_channel(new_orig.scatter_field, scale) if new_orig.scatter_field is not None else None,
                    direct_beam=self._resample_channel(new_orig.direct_beam, scale) if new_orig.direct_beam is not None else None)
            elif isinstance(new_orig, RadiationFieldChannel):
                new_orig = self._resample_channel(new_orig, scale)
        return x._replace(ground_truth=new_gt, original_ground_truth=new_orig)

    def get_parameters(self):
        return {"scales": list(self.scales)}

    @classmethod
    def create_from_config(cls, config: dict) -> "MultiResolutionResampling":
        return cls(scales=tuple(config.get("scales", (0.5, 0.75, 1.0, 1.5, 2.0))))

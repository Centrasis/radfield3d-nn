from .base import MetricBase
from torch import Tensor, nn
from typing import Union, Literal
import torch
from ptbdatalab.ptbDataLab.processings.airkerma import Airkerma
from rftypes import RadiationFieldChannel, TrainingInputData


class PSNR(MetricBase):
    def __init__(self, layer_name: Union[Literal['fluence'], Literal['spectrum'], Literal['error'], None] = None, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean', weight_with_error: bool = False, eps: float = 1e-8):
        super().__init__(layer_name=layer_name, reduction=reduction, weight_with_error=weight_with_error, eps=eps)

    def _calc_metric(self, target: Tensor, prediction: Tensor) -> Tensor:
        target = target / (target.max() + self.eps)
        prediction = prediction / (prediction.max() + self.eps)
        mse = nn.functional.mse_loss(prediction, target, reduction='none')
        psnr = 10 * torch.log10(1 / (mse + self.eps))
        return psnr


class AirkermaPSNR(PSNR):
    def __init__(self, mu_tr_file: str, spectra_bins: int, max_energy_eV: float, weight_with_error: bool = False, importance_threshold: float = 0.0, reduction: Union[Literal['mean'], Literal['median'], Literal['none']] = 'mean'):
        super().__init__(layer_name=None, reduction=reduction, weight_with_error=weight_with_error, eps=1e-8)
        self.airkerma = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), spectra_bins, max_energy_eV)

    def forward(self, target: Union[RadiationFieldChannel, Tensor], prediction: Union[RadiationFieldChannel, Tensor], input: TrainingInputData = None) -> Tensor:
        if prediction.spectrum is None or prediction.fluence is None:
            return None
        target_airkerma = self.airkerma.forward(target.spectrum, target.fluence)
        prediction_airkerma = self.airkerma.forward(prediction.spectrum, prediction.fluence)
        return super().forward(target_airkerma, prediction_airkerma, input)

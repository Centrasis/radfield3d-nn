from typing import NamedTuple, Union
from torch import Tensor


class ChannelMetrics(NamedTuple):
    fluence_loss: Tensor
    spectrum_loss: Tensor


class TrainingMetrics(NamedTuple):
    scatter_field: Union[ChannelMetrics, None] = None
    xray_beam: Union[ChannelMetrics, None] = None
    airkerma_field: Union[Tensor, None] = None

    def __add__(self, other: 'TrainingMetrics') -> 'TrainingMetrics':
        return TrainingMetrics(
            scatter_field=ChannelMetrics(
                fluence_loss=self.scatter_field.fluence_loss + other.scatter_field.fluence_loss,
                spectrum_loss=self.scatter_field.spectrum_loss + other.scatter_field.spectrum_loss
            ) if self.scatter_field is not None and other.scatter_field is not None else None,
            xray_beam=ChannelMetrics(
                fluence_loss=self.xray_beam.fluence_loss + other.xray_beam.fluence_loss,
                spectrum_loss=self.xray_beam.spectrum_loss + other.xray_beam.spectrum_loss
            ) if self.xray_beam is not None and other.xray_beam is not None else None,
            airkerma_field=(self.airkerma_field + other.airkerma_field) if self.airkerma_field is not None and other.airkerma_field is not None else None
        )

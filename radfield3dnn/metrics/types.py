from typing import NamedTuple, Union
from torch import Tensor


class ChannelMetrics(NamedTuple):
    flux_loss: Tensor
    spectrum_loss: Tensor


class TrainingMetrics(NamedTuple):
    scatter_field: Union[ChannelMetrics, None] = None
    direct_beam: Union[ChannelMetrics, None] = None
    airkerma_field: Union[Tensor, None] = None

    def __add__(self, other: 'TrainingMetrics') -> 'TrainingMetrics':
        return TrainingMetrics(
            scatter_field=ChannelMetrics(
                flux_loss=self.scatter_field.flux_loss + other.scatter_field.flux_loss,
                spectrum_loss=self.scatter_field.spectrum_loss + other.scatter_field.spectrum_loss
            ) if self.scatter_field is not None and other.scatter_field is not None else None,
            direct_beam=ChannelMetrics(
                flux_loss=self.direct_beam.flux_loss + other.direct_beam.flux_loss,
                spectrum_loss=self.direct_beam.spectrum_loss + other.direct_beam.spectrum_loss
            ) if self.direct_beam is not None and other.direct_beam is not None else None,
            airkerma_field=(self.airkerma_field + other.airkerma_field) if self.airkerma_field is not None and other.airkerma_field is not None else None
        )

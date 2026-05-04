import plotly.graph_objects as go
from radfield3dnn import RadiationFieldChannel, RadiationField, AirKermaField
import torch
from radfield3dnn.preprocessing.airkerma import Airkerma
from radfield3dnn.datasets.channel_join import ChannelsJoin


class AirkermaPlotter(object):
    def __init__(self, mutr_path: str, max_energy_eV: float = 1.5e+5, voxel_size: float = 0.025, normalize_airkerma: bool = True) -> None:
        super().__init__()
        self.airkerma_calc = Airkerma(
            mu_tr=Airkerma.load_mu_tr_table(mutr_path),
            max_energy_eV=max_energy_eV
        )
        self.max_energy_eV = max_energy_eV
        self.channels_join = ChannelsJoin()
        self.voxel_size = voxel_size
        self.normalize_airkerma = normalize_airkerma

    def create_airkerma_figure(self, title: str = "Airkerma Volume Plot") -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Axis in m',
                yaxis_title='Y Axis in m',
                zaxis_title='Z Axis in m'
            )
        )
        return fig

    def get_airkerma(self, field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor) -> torch.Tensor:
        airkerma = None
        with torch.no_grad():
            if isinstance(field, torch.Tensor):
                airkerma = field.detach()
            else:
                field: RadiationFieldChannel | AirKermaField = self.channels_join.join_channels(field) if isinstance(field, RadiationField) else field
                if isinstance(field, RadiationFieldChannel):
                    airkerma = self.airkerma_calc.calc_airkerma(field.spectrum, field.flux).detach()
                elif isinstance(field, AirKermaField):
                    airkerma = field.air_kerma.detach()
                else:
                    raise ValueError("Invalid field type provided for airkerma volume plotting.")
        return airkerma

    def add_airkerma_to_figure(self, fig: go.Figure, field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, name: str = None, batch_idx: int = None) -> None:
        airkerma = self.get_airkerma(field).detach() if not isinstance(field, torch.Tensor) else field.detach()
        if batch_idx is not None:
            airkerma = airkerma[batch_idx]
        airkerma = airkerma.view(*airkerma.shape[-3:]) if airkerma.dim() >= 4 else airkerma
        assert len(airkerma.shape) == 3, "Airkerma volume must be a 3D tensor or a batch of 3D tensors!"
        if self.normalize_airkerma:
            airkerma = airkerma / airkerma.max()
        airkerma = airkerma.cpu().numpy()
        xs=[i * self.voxel_size for i in range(airkerma.shape[0]) for j in range(airkerma.shape[1]) for k in range(airkerma.shape[2])] 
        ys=[j * self.voxel_size for i in range(airkerma.shape[0]) for j in range(airkerma.shape[1]) for k in range(airkerma.shape[2])]
        zs=[k * self.voxel_size for i in range(airkerma.shape[0]) for j in range(airkerma.shape[1]) for k in range(airkerma.shape[2])]
        fig.add_trace(
            go.Volume(
                x=xs,
                y=ys,
                z=zs,
                value=airkerma.flatten(),
                isomin=0,
                isomax=1,
                opacity=0.1, # needs to be small to see through all surfaces
                surface_count=20, # needs to be a large number for good volume rendering
                colorscale='Viridis',
                colorbar=dict(title=name if name is not None else 'Airkerma in Gy')
            )
        )

    def plot_airkerma_volume(self, field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, title: str = "Airkerma Volume Plot", batch_idx: int = None) -> go.Figure:
        fig = self.create_airkerma_figure(title=title)
        with torch.no_grad():
            airkerma = self.get_airkerma(field) if not isinstance(field, torch.Tensor) else field
            batch_size = airkerma.shape[0] if airkerma.dim() == 5 else None
            if batch_size is None:
                airkerma_volume = airkerma.view(*airkerma.shape[-3:])
                airkerma_volumes = [airkerma_volume]
            else:
                airkerma_volumes: list[torch.Tensor] = []
                for bi in range(batch_size):
                    if batch_idx is not None and bi != batch_idx:
                        continue
                    airkerma_volume = airkerma[bi].view(*airkerma.shape[-3:])
                    airkerma_volumes.append(airkerma_volume)

            for b_idx, airkerma in enumerate(airkerma_volumes):
                self.add_airkerma_to_figure(
                    fig,
                    airkerma,
                    name=f'Airkerma in Gy (Batch: {b_idx})' if batch_size is not None else 'Airkerma in Gy'
                )
            return fig

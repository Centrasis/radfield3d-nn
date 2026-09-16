import plotly.graph_objects as go
from plotly.subplots import make_subplots
from radfield3dnn.rftypes import RadiationFieldChannel, RadiationField, AirKermaField
import torch
from radfield3dnn.preprocessing.airkerma import Airkerma
from radfield3dnn.datasets.channel_join import ChannelsJoin


class SlicedAirkermaPlotter(object):
    def __init__(self, mutr_path: str, max_energy_eV: float = 1.5e+5, voxel_size: float = 0.025, normalize_airkerma: bool = True, normalize_slice: bool = True) -> None:
        super().__init__()
        self.airkerma_calc = Airkerma(
            mu_tr=Airkerma.load_mu_tr_table(mutr_path),
            max_energy_eV=max_energy_eV
        )
        self.max_energy_eV = max_energy_eV
        self.channels_join = ChannelsJoin()
        self.voxel_size = voxel_size
        self.normalize_airkerma = normalize_airkerma
        self.normalize_slice = normalize_slice

    def create_airkerma_figure(self, title: str = "Airkerma Sliced Plot") -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title='X Axis in m',
            yaxis_title='Y Axis in m'
        )
        return fig
    
    def create_airkerma_subplot_figure(self, rows: int, cols: int, subplot_titles: list[str] = None, title: str = "Airkerma Sliced Subplot") -> go.Figure:
        if subplot_titles is None:
            subplot_titles = [f"Subplot {i+1}" for i in range(rows * cols)]
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03,
            horizontal_spacing=0.01,
            row_heights=[1.0/rows for _ in range(rows)]
        )
        fig.update_layout(
            title=title,
            xaxis_title='X Axis in m',
            yaxis_title='Y Axis in m'
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
    
    def get_2D_slice(self, airkerma: torch.Tensor, X: float = None, Y: float = None, Z: float = None) -> torch.Tensor:
        assert (X is not None) + (Y is not None) + (Z is not None) == 1, "Exactly one of X, Y, or Z slice positions must be specified!"
        # Clamp normalized position to [0, 1] and map to valid index [0, size-1]
        if X is not None:
            size = airkerma.shape[0]
            pos = max(0.0, min(1.0, float(X)))
            slice_idx = int(round(pos * (size - 1)))
            slice_data = airkerma[slice_idx, :, :]
        elif Y is not None:
            size = airkerma.shape[1]
            pos = max(0.0, min(1.0, float(Y)))
            slice_idx = int(round(pos * (size - 1)))
            slice_data = airkerma[:, slice_idx, :]
        else:  # Z is not None
            size = airkerma.shape[2]
            pos = max(0.0, min(1.0, float(Z)))
            slice_idx = int(round(pos * (size - 1)))
            slice_data = airkerma[:, :, slice_idx]

        # Clean and normalize the slice (not the whole volume)
        slice_data = slice_data.contiguous()
        slice_data = torch.nan_to_num(slice_data, nan=0.0, posinf=0.0, neginf=0.0)
        slice_data = torch.clamp_min(slice_data, 0.0)
        if self.normalize_slice:
            denom = slice_data.max()
            if torch.isfinite(denom) and float(denom) > 0.0:
                slice_data = slice_data / denom
        return slice_data

    def add_airkerma_slice_to_figure(self, fig: go.Figure, field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, X: float = None, Y: float = None, Z: float = None, name: str = None, row: int = None, col: int = None) -> None:
        assert (X is not None) + (Y is not None) + (Z is not None) == 1, "Exactly one of X, Y, or Z slice positions must be specified!"
        assert (row is None and col is None) or (row is not None and col is not None), "Both row and col must be specified for subplot placement!"
        airkerma = self.get_airkerma(field).detach() if not isinstance(field, torch.Tensor) else field.detach()
        airkerma = airkerma.view(*airkerma.shape[-3:]) if airkerma.dim() >= 4 else airkerma
        assert len(airkerma.shape) == 3, "Airkerma volume must be a 3D tensor or a batch of 3D tensors!"
        if self.normalize_airkerma:
            maxv = torch.nan_to_num(airkerma, nan=0.0, posinf=0.0, neginf=0.0).max()
            if float(maxv) > 0.0:
                airkerma = airkerma / maxv
        slice_data = self.get_2D_slice(airkerma, X=X, Y=Y, Z=Z).cpu().numpy()

        # Heatmap expects z with shape (n_rows, n_cols) == (len(y), len(x))
        n_rows, n_cols = slice_data.shape
        x_coords = [j * self.voxel_size for j in range(n_cols)]
        y_coords = [i * self.voxel_size for i in range(n_rows)]

        fig.add_trace(go.Heatmap(
            z=slice_data,
            x=x_coords,
            y=y_coords,
            colorscale='Viridis',
            zmin=0.0,
            zmax=1.0,
            name=name
        ), row=row, col=col)
        # Apply equal aspect to all subplots (handles multi-subplot figures)
        self._enforce_equal_aspect_all_subplots(fig)

    def plot_airkerma_slices(self, field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, X: float = None, Y: float = None, Z: float = None, title: str = "Airkerma Sliced Plot", name: str = None, row: int = None, col: int = None) -> go.Figure:
        fig = self.create_airkerma_figure(title=title)
        airkerma = self.get_airkerma(field).detach() if not isinstance(field, torch.Tensor) else field.detach()
        if len(airkerma.shape) == 5:
            for i in range(airkerma.shape[0]):
                self.add_airkerma_slice_to_figure(fig, airkerma[i], X=X, Y=Y, Z=Z, name=f"{name} {i}" if name is not None else None, row=row, col=col)
        elif len(airkerma.shape) == 4 and airkerma.shape[0] == 1:
            self.add_airkerma_slice_to_figure(fig, airkerma[0], X=X, Y=Y, Z=Z, name=name, row=row, col=col)
        else:
            self.add_airkerma_slice_to_figure(fig, airkerma, X=X, Y=Y, Z=Z, name=name, row=row, col=col)
        return fig

    def create_blended_rgb(self, pred_img: torch.Tensor, target_img: torch.Tensor):
        """Create RGB image where red=prediction, blue=target, purple=overlap"""
        device = pred_img.device
        dtype = pred_img.dtype

        # Direct channel mapping after per-slice normalization
        rgb = torch.zeros((*pred_img.shape, 3), device=device, dtype=dtype)
        rgb[..., 0] = pred_img.clamp_min(0.0)  # red
        rgb[..., 2] = target_img.clamp_min(0.0)  # blue

        maxv = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).max()
        if maxv > 0:
            rgb = rgb / maxv
        return (rgb.clamp(0, 1) * 255).to(torch.uint8)
    
    def add_blended_slices(self, fig: go.Figure, pred_field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, gt_field: RadiationFieldChannel | RadiationField | AirKermaField | torch.Tensor, X: float = None, Y: float = None, Z: float = None, name1: str = "Field 1", name2: str = "Field 2", row: int = None, col: int = None) -> go.Figure:
        airkerma1 = self.get_airkerma(pred_field).detach() if not isinstance(pred_field, torch.Tensor) else pred_field.detach()
        airkerma2 = self.get_airkerma(gt_field).detach() if not isinstance(gt_field, torch.Tensor) else gt_field.detach()
        airkerma1 = airkerma1.view(*airkerma1.shape[-3:]) if airkerma1.dim() >= 4 else airkerma1
        airkerma2 = airkerma2.view(*airkerma2.shape[-3:]) if airkerma2.dim() >= 4 else airkerma2
        assert len(airkerma1.shape) == 3 and len(airkerma2.shape) == 3, "Airkerma volumes must be 3D tensors or batches of 3D tensors!"
        assert (row is None and col is None) or (row is not None and col is not None), "Both row and col must be specified for subplot placement!"
        if self.normalize_airkerma:
            max1 = torch.nan_to_num(airkerma1, nan=0.0, posinf=0.0, neginf=0.0).max()
            max2 = torch.nan_to_num(airkerma2, nan=0.0, posinf=0.0, neginf=0.0).max()
            if float(max1) > 0.0:
                airkerma1 = airkerma1 / max1
            if float(max2) > 0.0:
                airkerma2 = airkerma2 / max2

        slice_data1 = self.get_2D_slice(airkerma1, X=X, Y=Y, Z=Z)
        slice_data2 = self.get_2D_slice(airkerma2, X=X, Y=Y, Z=Z)
        blended_slice = self.create_blended_rgb(slice_data1, slice_data2).cpu().numpy()

        fig.add_trace(go.Image(
            z=blended_slice,
            colormodel="rgb",
            x0=0,
            y0=0,
            dx=self.voxel_size,
            dy=self.voxel_size,
            name=f"{name1} (Red) / {name2} (Blue)"
        ), row=row, col=col)
        # Apply equal aspect to all subplots (handles multi-subplot figures)
        self._enforce_equal_aspect_all_subplots(fig)

    def _enforce_equal_aspect_all_subplots(self, fig: go.Figure) -> None:
        # Anchor each y-axis to its corresponding x-axis (yaxis{n} -> xaxis{n})
        layout_dict = fig.layout.to_plotly_json()
        for key in list(layout_dict.keys()):
            if key.startswith("xaxis"):
                suffix = key[5:]  # '' for first, '2', '3', ...
                xattr = f"xaxis{suffix}"
                yattr = f"yaxis{suffix}"
                xaxis = getattr(fig.layout, xattr, None)
                yaxis = getattr(fig.layout, yattr, None)
                if xaxis is not None and yaxis is not None:
                    xaxis.constrain = "domain"
                    yaxis.scaleanchor = f"x{suffix}" if suffix else "x"
                    yaxis.scaleratio = 1

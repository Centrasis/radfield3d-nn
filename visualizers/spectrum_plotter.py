import plotly.graph_objects as go
from plotly.subplots import make_subplots
from radfield3dnn import RadiationFieldChannel, RadiationField
from radfield3dnn.rfhelpers import InferenceHelper
import torch
from typing import Literal
from dataclasses import dataclass


@dataclass
class SpectrumDescriptor:
    batch_idx: int
    xyz: tuple[int, int, int]
    trace_name: str | None = None


class SpectrumPlotter(object):
    def __init__(self, bins: int = 32, max_energy_eV: float = 1.5e+5, bin_width_ev: float = None, used_unit: Literal["eV", "keV", "MeV"] = "keV", bar_opacity: float = 1.0) -> None:
        super().__init__()
        self.bins = bins
        self.max_energy_eV = max_energy_eV
        self.bin_width_eV = bin_width_ev if bin_width_ev is not None else max_energy_eV / bins
        self.energy_bins = torch.linspace(0, max_energy_eV, bins)
        self.bar_opacity = bar_opacity
        if used_unit == "MeV":
            self.energy_bins = self.energy_bins / 1e+6
        elif used_unit == "keV":
            self.energy_bins = self.energy_bins / 1e+3
        elif used_unit == "eV":
            self.energy_bins = self.energy_bins
        else:
            raise ValueError("Invalid unit specified. Choose from 'eV', 'keV', or 'MeV'.")
        self.used_unit = used_unit

    def plot_spectra(self, field: RadiationFieldChannel | RadiationField | list[RadiationField] | list[RadiationFieldChannel], descriptor: list[SpectrumDescriptor] | SpectrumDescriptor, title: str = "Spectrum Plot", row: int = None, col: int = None) -> go.Figure:
        if isinstance(field, (RadiationFieldChannel, RadiationField)):
            fields = [field]
        elif isinstance(field, list):
            fields = field
        else:
            raise ValueError("Invalid field type provided for spectrum plotting.")

        fig = self.create_spectrum_figure(title=title)  # Initialize empty figure
        for i, field in enumerate(fields):
            ds = descriptor[i] if isinstance(descriptor, list) else descriptor
            if ds.trace_name is not None and len(fields) > 1:
                ds.trace_name = ds.trace_name if ds.trace_name is not None else f"Trace {i+1}"
            self.add_spectrum_to_figure(fig, field, descriptor=ds, row=row, col=col)
        return fig

    def add_spectrum_to_figure(self, fig: go.Figure, field: RadiationFieldChannel | RadiationField, descriptor: SpectrumDescriptor, row: int = None, col: int = None) -> None:
        spectrum = InferenceHelper.try_extract_spectrum(field)
        spectrum = spectrum[descriptor.batch_idx, :, descriptor.xyz[0], descriptor.xyz[1], descriptor.xyz[2]] if spectrum.dim() == 5 else spectrum[:, descriptor.xyz[0], descriptor.xyz[1], descriptor.xyz[2]]
        spectrum = spectrum.detach().view(self.bins).cpu()
        assert spectrum is not None, "Provided field does not contain a spectrum!"
        assert spectrum.shape[-1] == self.bins, f"Spectrum bin count mismatch! Expected {self.bins}, got {spectrum.shape[-1]}."
        assert (row is None and col is None) or (row is not None and col is not None), "Both row and col must be specified for subplot placement!"
        fig.add_trace(go.Bar(x=self.energy_bins.numpy(), y=spectrum.numpy(), name=descriptor.trace_name, opacity=self.bar_opacity), row=row, col=col)

    def create_spectrum_figure(self, title: str = "Spectrum Plot") -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title=f'Energy in {self.used_unit}', yaxis_title='Intensity', barmode='overlay')
        return fig

    def create_spectrum_subplots(self, rows: int, cols: int, subtitles: list[str] = None, title: str = "Spectrum Plots") -> go.Figure:
        if subtitles is None:
            subtitles = [f"Spectrum {i+1}" for i in range(rows * cols)]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subtitles)
        fig.update_layout(title=title, xaxis_title=f'Energy in {self.used_unit}', yaxis_title='Intensity', barmode='overlay')
        return fig

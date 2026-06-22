from RadFiled3D.pytorch.datasets.processing import DataProcessing
from radfield3dnn.rftypes import TrainingInputData, AirKermaField, RadiationField, RadiationFieldChannel
from torch.nn import Module
from torch import Tensor
from radfield3dnn.utils.mean_sampling import resample_histogram_means
import torch
import numpy as np


class Airkerma(Module):
    def __init__(self, mu_tr: Tensor, bins: int = 32, max_energy_eV: float = 1.5e+5):
        """
        Initializes the Airkerma class with the given parameters.
        :param mu_tr: The mass energy absorption coefficients for the given spectra. Shape: (n, 1) with (energy edge value, mu_tr value)
        :param spectra_bin_edges: The bin edges of the spectra. Shape: (num_bins + 1,) with num_bins being the number of bins in the spectra.
        :return: None
        """
        super().__init__()
        # Use bins+1 edges, evenly spaced from 0 to max_energy_eV (inclusive)
        self._spectra_bin_edges = torch.linspace(0.0, max_energy_eV, steps=bins+1, dtype=torch.float32)
        # Resample mu_tr to these histogram bins
        self._mu_tr: Tensor = resample_histogram_means(mu_tr, self._spectra_bin_edges, True)

    @staticmethod
    def load_mu_tr_table(mu_tr_file: str, energy_unit: str = "eV") -> Tensor:
        # Explicitly detect a header line — a non-numeric header ("energy mu_tr") OR a single-number
        # count line whose column count differs from the data rows — instead of a bare try/except
        # (which would also swallow KeyboardInterrupt/MemoryError and surface real parse errors only
        # as the confusing second attempt).
        def _numeric_cols(line: str) -> int:
            parts = line.replace(",", " ").split()
            if not parts:
                return -1
            try:
                [float(p) for p in parts]
                return len(parts)
            except ValueError:
                return -1   # non-numeric line

        with open(mu_tr_file, "r") as f:
            first, second = f.readline(), f.readline()
        c0, c1 = _numeric_cols(first), _numeric_cols(second)
        # First line is data only if it is numeric and its column count matches the data rows.
        skiprows = 0 if (c0 >= 1 and (c1 < 1 or c0 == c1)) else 1
        mu_tr = np.loadtxt(mu_tr_file, skiprows=skiprows)
        if energy_unit == "eV":
            # No need to convert as Geant4 outputs in 1.0 = 1MeV
            pass
        elif energy_unit == "keV":
            # Convert keV -> eV
            mu_tr[:, 0] *= 1e3
        elif energy_unit == "MeV":
            # Convert MeV -> eV
            mu_tr[:, 0] *= 1e6
        else:
            raise ValueError("Unknown energy unit: " + energy_unit)
        return torch.tensor(mu_tr, dtype=torch.float32, requires_grad=False)

    def calc_airkerma(self, spectra: Tensor | None, fluxes: Tensor) -> Tensor:
        """
        Calculates the air kerma from the spectra and fluxes.
        :param spectra: The spectra to calculate the air kerma from. Shape: (batch_size, num_bins, x, y, z) or None. If None, a uniform spectrum is assumed.
        :param fluxes: The fluxes to calculate the air kerma from. Shape: (batch_size, x, y, z) or (batch_size, 1, x, y, z)
        :return: The air kerma. Shape: (batch_size, 1, x, y, z)
        """
        if self._mu_tr.device != fluxes.device:
            self._mu_tr = self._mu_tr.to(fluxes.device)
            self._spectra_bin_edges = self._spectra_bin_edges.to(fluxes.device)

        if spectra is None:
            spectra = torch.ones((fluxes.shape[0], self._spectra_bin_edges.shape[0]-1, *fluxes.shape[2:]), device=fluxes.device, dtype=torch.float32)
            spectra /= spectra.shape[1]  # Uniform distribution if no spectra given
        assert (spectra.dim() == 5 and spectra.size(1) > 2) or (spectra.dim() == 4 and spectra.size(0) > 2), "spectra must be shaped (B, num_bins, X, Y, Z) with num_bins > 2 or (num_bins, X, Y, Z) with num_bins > 2"
        assert fluxes.dim() in [4, 5], "fluxes must be shaped (B, X, Y, Z) or (B, 1, X, Y, Z)"
        assert fluxes.dim() == spectra.dim(), "spectra and fluxes must have the same number of dimensions"
        # Normalize spectra along the energy bins
        spectra_integral = torch.sum(spectra, dim=1, keepdim=True)
        spectra_integral = torch.nan_to_num(spectra_integral, nan=0.0, posinf=0.0, neginf=0.0)
        spectra_integral_low = spectra_integral < 1e-8
        if torch.any(spectra_integral_low):
            spectra_integral = torch.clamp(spectra_integral, min=1e-8)
            spectra = spectra / spectra_integral
            spectra[spectra_integral_low.expand_as(spectra)] = 1.0 / spectra.shape[1]
        else:
            spectra = spectra / spectra_integral

        # μ_tr per bin (already resampled to our histogram bins)
        mu_tr_vals = self._mu_tr[:-1, 1].view(1, -1, 1, 1, 1)

        # Ensure fluxes shape is (B,1,X,Y,Z)
        if fluxes.dim() == 4:
            fluxes = fluxes.unsqueeze(1)
        elif not (fluxes.dim() == 5 and fluxes.shape[1] == 1):
            raise ValueError("fluxes must be shaped (B,X,Y,Z) or (B,1,X,Y,Z)")

        # Bin centers and widths from edges (edges length = bins+1)
        bin_edges = self._spectra_bin_edges
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = (bin_edges[1:] - bin_edges[:-1])

        bin_centers = bin_centers.view(1, -1, 1, 1, 1)
        bin_widths = bin_widths.view(1, -1, 1, 1, 1)

        # Discrete integral: sum over bins of Φ_norm(E_i) * μ_tr(E_i) * E_i * ΔE, then scale by flux
        y = spectra * mu_tr_vals * bin_centers
        kerma = (y * bin_widths).sum(dim=1, keepdim=True) * fluxes

        return kerma

    def forward(self, spectra: Tensor, fluxes: Tensor) -> Tensor:
        """
        Calculates the air kerma from the spectra and fluxes.
        :param spectra: The spectra to calculate the air kerma from. Shape: (batch_size, num_bins, x, y, z)
        :param fluxes: The fluxes to calculate the air kerma from. Shape: (batch_size, 1, x, y, z) or (batch_size, x, y, z)
        :return: The air kerma. Shape: (batch_size, 1, x, y, z)
        """
        return self.calc_airkerma(spectra, fluxes)


class AirkermaProcessing(DataProcessing):
    def __init__(self, mu_tr_file: str, bins: int = 32, max_energy_eV: float = 1.5e+5):
        super().__init__()
        self.airkerma_module = Airkerma(Airkerma.load_mu_tr_table(mu_tr_file), bins, max_energy_eV)

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply data processing to the input data.
        """
        assert isinstance(x.ground_truth, RadiationFieldChannel), "AirkermaProcessing only supports RadiationFieldChannel as ground truth."
        air_kerma = self.airkerma_module.forward(
            spectra=x.ground_truth.spectrum if x.ground_truth.spectrum is not None else None,
            fluxes=x.ground_truth.flux
        )

        return TrainingInputData(
            input=x.input,
            ground_truth=AirKermaField(
                air_kerma=air_kerma,
                geometry=x.original_ground_truth.geometry if x.original_ground_truth is not None and isinstance(x.original_ground_truth, RadiationField) else None
            ),
            original_ground_truth=x.original_ground_truth
        )

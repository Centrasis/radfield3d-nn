from rftypes import AirKermaField, TrainingInputData, RadiationFieldChannel, RadiationField
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset
from datasets.channel_join import ChannelsJoin
from datasets import OriginalGroundTruthPreservation
import argparse
from augmentations.smooth_spectra import SmoothingSpectra
from datasets.dataloader import RadiationFieldDataModule
from normalizations.linear import LinearNormalizer
from normalizations.lognormalizer import LogNormalizer
from metrics.airkerma_accuracy import AirkermaAccuracy, AirkermaScatterAccuracy, Airkerma
from processings.airkerma import AirkermaProcessing
import plotly.graph_objects as go
import torch
from rich.progress import track
from utils.visualizers.spectrum_plotter import SpectrumPlotter, SpectrumDescriptor
from utils.visualizers.volumetric_plotter import AirkermaPlotter
from utils.visualizers.sliced_plotter import SlicedAirkermaPlotter
import os
from RadFiled3D.RadFiled3D import FieldStore, CartesianRadiationField
from rich import print
from torch import Tensor


def generate_scatter_only_mask(input: TrainingInputData, prediction: torch.Tensor):
    xgt = input.original_ground_truth.direct_beam if input.original_ground_truth is not None and input.original_ground_truth.direct_beam is not None else input.ground_truth.direct_beam
    xgt = xgt.flux if isinstance(xgt, RadiationFieldChannel) else xgt
    sgt = input.original_ground_truth.scatter_field if input.original_ground_truth is not None and input.original_ground_truth.scatter_field is not None else input.ground_truth.scatter_field
    sgt = sgt.flux if isinstance(sgt, RadiationFieldChannel) else sgt
    fgt = sgt + xgt

    max_relative_fluence = 5e-2
    min_relative_fluence = 5e-3

    beam_mask = xgt > xgt.max() * max_relative_fluence  # ignore areas with > max_relative_fluence of max primary flux
    low_fluence_mask_gt = fgt < fgt.max() * min_relative_fluence  # ignore areas with < min_relative_fluence of max total flux
    beam_mask = beam_mask | low_fluence_mask_gt  # combine masks

    prediction_fluence = prediction.clone()
    prediction_fluence[beam_mask] = -torch.inf
    return prediction_fluence


def show_figure(fig: go.Figure, title="Volume", store: bool = True):
    fig.show()
    if store:
        fig.write_image(os.path.join(os.getcwd(), f"{title.replace(' ', '_')}.pdf"))


def plot_spectra_volume(spectra: torch.Tensor, title="Spectra Volume", resolution: float = 0.05, normalize: bool = True, max_energy_eV: float = 1.5e+5):
    if spectra.dim() == 5:
        bins = spectra.shape[1]
    elif spectra.dim() == 4:
        bins = spectra.shape[0]
    elif spectra.dim() == 2:
        bins = spectra.shape[1]
    else:
        raise ValueError("Spectra tensor must be 4D, 5D or 2D.")
    
    spectra = spectra.detach().reshape(-1, bins)
    fig = go.Figure()
    bin_width = max_energy_eV / bins
    spectra_to_draw = max(1, int(spectra.shape[0] * resolution))
    sample_indices = torch.randperm(spectra.shape[0], device=spectra.device)[:spectra_to_draw]
    trace_opacity = max(0.12, min(0.75, 1.5 / max(1, spectra_to_draw ** 0.5)))
    x_positions = ((torch.arange(bins, device=spectra.device, dtype=spectra.dtype) + 0.5) * bin_width / 1000).cpu().numpy()
    for idx in sample_indices.cpu().tolist():
        spectrum = spectra[idx]
        if normalize:
            spectrum = spectrum / spectrum.sum().clamp_min(1e-12)
        fig.add_trace(go.Bar(
            x=x_positions,
            y=spectrum.cpu().numpy(),
            marker=dict(color=f"rgba(33, 158, 188, {trace_opacity})"),
            hovertemplate=f"Spectrum {idx}<extra></extra>",
            showlegend=False
        ))
    fig.update_layout(barmode="overlay", title=title, xaxis_title='Energy in keV', yaxis_title='Intensity', font=dict(size=16))
    return fig


def plot_spectra_volume_stdev(spectra: torch.Tensor, title="Spectra Volume", normalize: bool = True, max_energy_eV: float = 1.5e+5):
    if spectra.dim() == 5:
        bins = spectra.shape[1]
    elif spectra.dim() == 4:
        bins = spectra.shape[0]
    elif spectra.dim() == 2:
        bins = spectra.shape[1]
    else:
        raise ValueError("Spectra tensor must be 4D, 5D or 2D.")
    
    spectra = spectra.detach().reshape(-1, bins)
    fig = go.Figure()
    bin_width = max_energy_eV / bins
    x_positions = ((torch.arange(bins, device=spectra.device, dtype=spectra.dtype) + 0.5) * bin_width / 1000).cpu().numpy()
    if normalize:
        spectra = spectra / spectra.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean_spectrum = spectra.mean(dim=0)
    std_spectrum = spectra.std(dim=0)
    fig.add_trace(go.Bar(
        x=x_positions,
        y=mean_spectrum.cpu().numpy(),
        error_y=dict(
            type='data',
            array=std_spectrum.cpu().numpy(),
            visible=True
        ),
        marker=dict(color="rgba(33, 158, 188, 0.75)"),
        hovertemplate="Mean Spectrum<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(barmode="overlay", title=title, xaxis_title='Energy in keV', yaxis_title='Intensity', font=dict(size=16))
    return fig


def plot_mean_spectrum(spectra: torch.Tensor):
    if spectra.dim() == 5:
        bins = spectra.shape[1]
    elif spectra.dim() == 4:
        bins = spectra.shape[0]
    elif spectra.dim() == 2:
        bins = spectra.shape[1]
    else:
        raise ValueError("Spectra tensor must be 4D, 5D or 2D.")
    
    spectra = spectra.reshape(-1, bins)

    mean_spectrum = spectra.mean(dim=0)

    # calc and print stdev for each bin
    std_spectrum = spectra.std(dim=0)
    for i in range(bins):
        print(f"Bin {i}: Mean={mean_spectrum[i].item():.4f}, Std={std_spectrum[i].item():.4f}")

    bin_width = 1.5e+5 / bins
    x_positions = ((torch.arange(bins, device=spectra.device, dtype=spectra.dtype) + 0.5) * bin_width / 1000).cpu().numpy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_positions,
        y=mean_spectrum.cpu().numpy(),
        marker=dict(color="rgba(33, 158, 188, 0.75)"),
        hovertemplate="Mean Spectrum<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(title="Mean Spectrum", xaxis_title='Energy in keV', yaxis_title='Intensity', font=dict(size=16))
    return fig


def plot_field_volume(flux, title="Fluence Volume"):
    fig = go.Figure(data=go.Volume(
        x=[i for i in range(flux.shape[0]) for j in range(flux.shape[1]) for k in range(flux.shape[2])], 
        y=[j for i in range(flux.shape[0]) for j in range(flux.shape[1]) for k in range(flux.shape[2])],
        z=[k for i in range(flux.shape[0]) for j in range(flux.shape[1]) for k in range(flux.shape[2])],
        value=flux.flatten(),
        isomin=0,
        isomax=1,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        colorscale='Viridis',
        colorbar=dict(title=title)
    ))
    fig.update_layout(title=title, font=dict(size=16))
    fig.show()


def generate_sphere_mask(target, vx_size = 0.02, sphere_radius_m = 0.25):
    B, C, D, H, W = target.shape
    device = target.device
    center = torch.tensor([D / 2, H / 2, W / 2], device=device).view(1, 3)
    grid_d = torch.arange(D, device=device).view(1, D, 1, 1).expand(B, D, H, W)
    grid_h = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, D, H, W)
    grid_w = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, D, H, W)
    grid = torch.stack((grid_d, grid_h, grid_w), dim=-1).float()
    distances = torch.sqrt(torch.sum((grid - center) ** 2, dim=-1)) * vx_size
    sphere_mask = (distances >= (sphere_radius_m - (vx_size / 2))) & (distances <= (sphere_radius_m + (vx_size / 2)))
    return sphere_mask.view(B, 1, D, H, W)


def calc_gini_coeff2(values: Tensor) -> Tensor:
    sums_of_diffs = torch.empty_like(values)
    for i in range(len(sums_of_diffs)):
        sums_of_diffs[i] = (values - values[i]).abs().sum()
    sums_of_sums_of_diffs = sums_of_diffs.sum()
    guk = sums_of_sums_of_diffs / (2 * len(values) * values.sum())
    return guk

def calc_gini_coeff(values: Tensor) -> Tensor:
    sorted_values = torch.sort(values)[0]
    n = len(sorted_values)
    indices = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    gini = (2 * indices * sorted_values).sum() / (n * sorted_values.sum()) - (n + 1) / n
    return gini


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Analyze a RadField3D dataset and compute statistics.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--mu_tr_path", type=str, required=True, help="Path to the mu_tr file.")
    parser.add_argument("--plot_metrics", action="store_true", help="Whether to plot metrics volumes.")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    mu_tr_path = args.mu_tr_path

    normalizer = LinearNormalizer((0, 1)).eval()
    airkerma = Airkerma(mu_tr=Airkerma.load_mu_tr_table(mu_tr_path), bins=32, max_energy_eV=1.5e+5)
    join_channels = ChannelsJoin()
    spec_plotter = SpectrumPlotter(bins=32, max_energy_eV=1.5e+5, used_unit="keV", bar_opacity=0.6)
    airkerma_plotter = AirkermaPlotter(mutr_path=mu_tr_path, max_energy_eV=1.5e+5, voxel_size=0.02)
    sliced_plotter = SlicedAirkermaPlotter(mutr_path=mu_tr_path, max_energy_eV=1.5e+5, voxel_size=0.02)

    random_radfields = []

    datamodule = RadiationFieldDataModule(
        zip_directory=dataset_path,
        dataset_cls=RadField3DDataset,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        batch_size=1,
        num_workers=4,
        data_processings=[
           OriginalGroundTruthPreservation(),
           ChannelsJoin()
        ]
    )

    airkerma_processing = AirkermaProcessing(
        mu_tr_file=mu_tr_path,
        bins=32,
        max_energy_eV=1.5e+5
    )

    top90 = AirkermaAccuracy(
        mu_tr_file=mu_tr_path,
        spectra_bins=32,
        max_energy_eV=1.5e+5,
        importance_threshold=0.1,
        
        keep_dim=True
    )

    top_50 = AirkermaAccuracy(
        mu_tr_file=mu_tr_path,
        spectra_bins=32,
        max_energy_eV=1.5e+5,
        importance_threshold=0.5,
        
        keep_dim=True
    )

    scatter_only = AirkermaScatterAccuracy(
        mu_tr_file=mu_tr_path,
        spectra_bins=32,
        max_energy_eV=1.5e+5,
        
        keep_dim=True,
    )

    datamodule.prepare_data()

    global_top_50_map = None
    global_top_90_map = None
    global_scatter_map = None
    global_sphere_map = None
    global_no_spectra_map = None
    MAX_ENERGY_EV = 1.5e+5
    FLUENCE_DISTRIBUTION_BINS = int(1e+6)
    BIN_WIDTH_EV = MAX_ENERGY_EV / 32
    ENERGIES_keV = [((i + 0.5) * BIN_WIDTH_EV) / 1000 for i in range(32)]
    empty_spectra_rel_counts = []

    spec_fig = spec_plotter.create_spectrum_figure(title="Random Voxel Spectrum")
    scatter_spec_fig = spec_plotter.create_spectrum_figure(title="Scatter-only Spectrum")
    beam_spec_fig = spec_plotter.create_spectrum_figure(title="Primary Beam Spectrum")
    scatter_spec_one_field_fig = spec_plotter.create_spectrum_figure(title="Scatter-only One Field Spectrum")
    zero_reference_ak_map = None
    fluence_distributions = torch.zeros((FLUENCE_DISTRIBUTION_BINS, 2), dtype=torch.float32)

    linear_normalizer = LinearNormalizer((0, 1)).eval()
    log_normalizer = LogNormalizer((0, 1), input_scale=1.0).eval()

    random_spectra_scatter = []
    random_spectra_beam = []

    ds_parameters = {
        "directions": {
            "x": [],
            "y": [],
            "z": []
        },
        "tube_distances_m": [],
        "tube_voltages_kVp": [],
        "tube_spectrum_means_keV": [],
        "tube_opening_angles_deg": [],
        "empty_voxels_fractions": [],
        "min_max_fluences": [],
    }

    test_file = None
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "fields")):
        test_files = [os.path.join(dataset_path, "fields", f) for f in os.listdir(os.path.join(dataset_path, "fields")) if f.endswith(".rf3")]
        test_file = test_files[0]
    else:
        raise ValueError("Dataset path is invalid or does not contain 'fields' directory.")
    test_field: CartesianRadiationField = FieldStore.load(test_file)
    field_dim = test_field.get_field_dimensions()
    field_dimensions = torch.tensor([field_dim.x, field_dim.y, field_dim.z], dtype=torch.float32) # in meters

    count = 0
    for train_in in track(datamodule.test_dataloader(), description="Processing dataset..."):
        count += 1
        train_in: TrainingInputData = datamodule.on_after_batch_transfer(train_in, 0)

        norm_flu = linear_normalizer.forward(train_in.ground_truth.flux)
        try:
            #log_flu = log_normalizer.forward(train_in.ground_truth.flux)
            log_flu = torch.zeros_like(norm_flu)
            log_gt0 = norm_flu > 0.0
            log_flu[log_gt0] = torch.log(norm_flu[log_gt0]).abs() # values are between 0..1 so log -> negative
            log_flu /= log_flu.max()
        except Exception as e:
            print(f"Error occurred while normalizing flux: {e}")
            continue
        fluence_distributions[:, 0] += torch.histogram(norm_flu, bins=FLUENCE_DISTRIBUTION_BINS, range=(0, 1)).hist
        fluence_distributions[:, 1] += torch.histogram(log_flu, bins=FLUENCE_DISTRIBUTION_BINS, range=(0, 1)).hist

        ds_parameters["directions"]["x"].append(train_in.input.direction[0][0].item())
        ds_parameters["directions"]["y"].append(train_in.input.direction[0][1].item())
        ds_parameters["directions"]["z"].append(train_in.input.direction[0][2].item())
        origin = train_in.input.origin # normalized origin relative to field_dimensions
        tube_position_m = (origin * field_dimensions) - (field_dimensions / 2)
        ds_parameters["tube_distances_m"].append(torch.linalg.norm(tube_position_m, dim=1).item())
        ds_parameters["min_max_fluences"] += [train_in.ground_truth.flux.min().item(), train_in.ground_truth.flux.max().item()]
        energies_keV = torch.arange(0, train_in.input.spectrum.size(1), device=train_in.input.spectrum.device, dtype=train_in.input.spectrum.dtype) * (MAX_ENERGY_EV / train_in.input.spectrum.size(1)) / 1000
        # extract the highest index from spectrum where the value is > 0
        highest_index = (train_in.input.spectrum[0] > 0).nonzero(as_tuple=True)[0].max().item()
        ds_parameters["tube_voltages_kVp"].append(energies_keV[highest_index - 1].item())
        ds_parameters["tube_spectrum_means_keV"].append(train_in.input.spectrum[0].dot(energies_keV) / train_in.input.spectrum[0].sum().clamp_min(1e-12).item())
        ds_parameters["tube_opening_angles_deg"].append(train_in.input.beam_shape_parameters[0][0].item())
        ds_parameters["empty_voxels_fractions"].append((norm_flu < 1e-3).float().sum().item() / norm_flu.numel())

        # plot spectrum
        spec_plotter.add_spectrum_to_figure(spec_fig, train_in.ground_truth, SpectrumDescriptor(
            batch_idx=0,
            xyz=(
                torch.randint(0, train_in.ground_truth.spectrum.shape[2], (1,)).item(),
                torch.randint(0, train_in.ground_truth.spectrum.shape[3], (1,)).item(),
                torch.randint(0, train_in.ground_truth.spectrum.shape[4], (1,)).item()
            ),
            trace_name=f"S:{len(empty_spectra_rel_counts)}"
        ))

        if 0.95 < torch.rand(1).item() and len(random_radfields) < 3:
            random_radfields.append(train_in.ground_truth)

        spectra_volume = train_in.ground_truth.spectrum.clone()
        spectrum_sums = spectra_volume.sum(dim=1)
        train_in = airkerma_processing.forward(train_in)
        train_in = normalizer(train_in)

        empty_spectra_rel_counts.append((spectrum_sums < 1e-6).float().sum().item() / spectrum_sums.numel())
        zero_airkerma_map = train_in.ground_truth.air_kerma < 1e-6
        zeros_sum_spectra = (spectrum_sums < 1e-6)
        same_places = zeros_sum_spectra == zero_airkerma_map
        zeros_sum_spectra &= (~zero_airkerma_map.squeeze(1))
        if not torch.all(same_places):
            error_count = (~same_places).sum().item()
            print(f"Mismatch between zero spectra sum and zero airkerma places! Error count: {error_count}")

        top_50_map = top_50._calc_metric(train_in.ground_truth.air_kerma, torch.zeros_like(train_in.ground_truth.air_kerma)).view(*train_in.ground_truth.air_kerma.shape[-3:])
        top_90_map = top90._calc_metric(train_in.ground_truth.air_kerma, torch.zeros_like(train_in.ground_truth.air_kerma)).view(*train_in.ground_truth.air_kerma.shape[-3:])
        scatter_tensor = generate_scatter_only_mask(train_in, train_in.ground_truth.air_kerma)
        scatter_map = scatter_tensor.view(*train_in.ground_truth.air_kerma.shape[-3:])
        sphere_mask = generate_sphere_mask(train_in.ground_truth.air_kerma, vx_size=0.02, sphere_radius_m=0.25).view(*train_in.ground_truth.air_kerma.shape[-3:])

        batch_size, bins, depth, height, width = spectra_volume.shape
        scatter_mask = torch.isfinite(scatter_tensor)
        while scatter_mask.dim() > 4:
            scatter_mask = scatter_mask.squeeze(1)
        if scatter_mask.dim() == 3:
            scatter_mask = scatter_mask.unsqueeze(0)
        scatter_mask = scatter_mask.reshape(train_in.ground_truth.air_kerma.shape[0], depth, height, width)
        airkerma_volume = train_in.ground_truth.air_kerma
        while airkerma_volume.dim() > 4:
            airkerma_volume = airkerma_volume.squeeze(1)
        if airkerma_volume.dim() == 3:
            airkerma_volume = airkerma_volume.unsqueeze(0)
        airkerma_volume = airkerma_volume.reshape(batch_size, depth, height, width)
        ak_thresholds = 0.05 * airkerma_volume.view(batch_size, -1).max(dim=1).values.view(batch_size, 1, 1, 1)
        beam_mask = (~scatter_mask) & (airkerma_volume >= ak_thresholds)
        spectra_by_voxel = spectra_volume.permute(0, 2, 3, 4, 1).reshape(-1, bins)

        scatter_spectra_vol = spectra_by_voxel[scatter_mask.reshape(-1)]
        beam_spectra_vol = spectra_by_voxel[beam_mask.reshape(-1)]

        if False:  # Plot some spectra volumes
            print(f"Plotting first field spectra scatter Volume of Shape: {spectra_volume.shape}")
            
            if spectra_by_voxel.shape[0] > 0:
                rand_idx = torch.randint(spectra_by_voxel.shape[0], (1,), device=spectra_by_voxel.device)
                rand_spectrum = spectra_by_voxel[rand_idx].unsqueeze(0)
                show_figure(plot_spectra_volume(rand_spectrum.squeeze(0), title="Random Spectra Global", resolution=1.0), store=False)

            if scatter_spectra_vol.shape[0] > 0:
                rand_idx = torch.randint(scatter_spectra_vol.shape[0], (1,), device=scatter_spectra_vol.device)
                rand_spectrum = scatter_spectra_vol[rand_idx].unsqueeze(0)
                show_figure(plot_spectra_volume(rand_spectrum.squeeze(0), title="Random Spectra Scatter", resolution=1.0), store=False)
                scatter_spec_fig = plot_spectra_volume(scatter_spectra_vol, title="Scatter-only Spectra Volume")
                show_figure(scatter_spec_fig, title="Scatter-only One Field Spectrum")
                show_figure(plot_mean_spectrum(scatter_spectra_vol), title="Mean Scatter-only Spectrum")

            if beam_spectra_vol.shape[0] > 0:
                rand_idx = torch.randint(beam_spectra_vol.shape[0], (1,), device=beam_spectra_vol.device)
                rand_spectrum = beam_spectra_vol[rand_idx].unsqueeze(0)
                show_figure(plot_spectra_volume(rand_spectrum.squeeze(0), title="Random Spectra Beam", resolution=1.0), store=False)
                beam_spec_fig = plot_spectra_volume(beam_spectra_vol, title="Primary Beam Spectra Volume")
                show_figure(beam_spec_fig, title="Primary Beam One Field Spectrum")
                show_figure(plot_mean_spectrum(beam_spectra_vol), title="Mean Primary Beam Spectrum")
        
        rand_idx = torch.randint(scatter_spectra_vol.shape[0], (1,), device=scatter_spectra_vol.device)
        random_spectra_scatter.append(scatter_spectra_vol[rand_idx].squeeze(0).cpu())
        rand_idx = torch.randint(beam_spectra_vol.shape[0], (1,), device=beam_spectra_vol.device)
        random_spectra_beam.append(beam_spectra_vol[rand_idx].squeeze(0).cpu())

        if global_top_90_map is None:
            global_top_90_map = torch.zeros_like(top_90_map)
            global_top_50_map = torch.zeros_like(top_90_map)
            global_scatter_map = torch.zeros_like(top_90_map)
            global_sphere_map = torch.zeros_like(top_90_map)
            global_no_spectra_map = torch.zeros_like(top_90_map)
            zero_reference_ak_map = torch.zeros_like(top_90_map)

        global_no_spectra_map += zeros_sum_spectra.view_as(global_no_spectra_map).float()

        top50_mask = torch.isneginf(top_50_map)
        top_50_map[top50_mask] = 0.0
        top_50_map[~top50_mask] = 1.0

        top90_mask = torch.isneginf(top_90_map)
        top_90_map[top90_mask] = 0.0
        top_90_map[~top90_mask] = 1.0

        scatter_mask = torch.isneginf(scatter_map)
        scatter_map[scatter_mask] = 0.0
        scatter_map[~scatter_mask] = 1.0

        airkerma_normalized = train_in.ground_truth.air_kerma.clone().view(*top_50_map.shape)
        airkerma_normalized /= airkerma_normalized.max()
        field_airkerma_scatter = train_in.ground_truth.air_kerma.clone().view(*scatter_mask.shape)
        field_airkerma_scatter[scatter_mask] = 0.0
        field_airkerma_scatter /= field_airkerma_scatter.max()
        field_airkerma_scatter = torch.log1p(field_airkerma_scatter)
        field_airkerma_scatter /= field_airkerma_scatter.max()

        global_top_50_map += top_50_map
        global_top_90_map += top_90_map
        global_scatter_map += scatter_map
        global_sphere_map[sphere_mask] += 1.0

        zero_ref = top90._calc_metric(train_in.ground_truth.air_kerma, torch.zeros_like(train_in.ground_truth.air_kerma)).view(*train_in.ground_truth.air_kerma.shape[-3:])
        zero_ref_mask = torch.isneginf(zero_ref)
        zero_reference_ak_map[~zero_ref_mask] += zero_ref[~zero_ref_mask]

    print("Statistics of dataset parameters:")
    statistics = {}
    for key, values in ds_parameters.items():
        if isinstance(values, dict):
            for subkey, subvalues in values.items():
                tensor_values = torch.tensor(subvalues)
                if key not in statistics:
                    statistics[key] = {}

                statistics[key].update({
                    subkey: {
                        "Mean": tensor_values.mean().item(),
                        "Std": tensor_values.std().item(),
                        "Min": tensor_values.min().item(),
                        "Max": tensor_values.max().item(),
                        "IQR": (tensor_values.quantile(0.75) - tensor_values.quantile(0.25)).item(),
                        "Dynamic Range (db)": 10 * torch.log10((tensor_values.max() / tensor_values.min().clamp_min(1e-12))).item(),
                    }
                })
                print(f"{key}.{subkey}: Mean={statistics[key][subkey]['Mean']:.4f}, Std={statistics[key][subkey]['Std']:.4f}, Min={statistics[key][subkey]['Min']:.4f}, Max={statistics[key][subkey]['Max']:.4f}, IQR={statistics[key][subkey]['IQR']:.4f}, Dynamic Range (db)={statistics[key][subkey]['Dynamic Range (db)']:.4f}")
        else:
            tensor_values = torch.tensor(values)
            statistics[key] = {
                "Mean": tensor_values.mean().item(),
                "Std": tensor_values.std().item(),
                "Min": tensor_values.min().item(),
                "Max": tensor_values.max().item(),
                "IQR": (tensor_values.quantile(0.75) - tensor_values.quantile(0.25)).item(),
                "Dynamic Range (db)": 10 * torch.log10((tensor_values.max() / tensor_values.min().clamp_min(1e-12))).item(),
            }
            print(f"{key}: Mean={statistics[key]['Mean']:.4f}, Std={statistics[key]['Std']:.4f}, Min={statistics[key]['Min']:.4f}, Max={statistics[key]['Max']:.4f}, IQR={statistics[key]['IQR']:.4f}, Dynamic Range (db)={statistics[key]['Dynamic Range (db)']:.4f}")

    statistics["fluences"] = {
        "GUK": calc_gini_coeff(fluence_distributions[:, 0] / count).item()
    }
    print(f"Fluence GUK: {statistics['fluences']['GUK']}")

    statistics_path = os.path.join(dataset_path, "statistics.json")
    import json
    with open(statistics_path, "w") as f:
        json.dump(statistics, f, indent=4)
    print(f"Saved dataset statistics to {statistics_path}")

    if not args.plot_metrics:
        exit(0)

    global_top_90_map /= global_top_90_map.max()
    global_top_50_map /= global_top_50_map.max()
    global_scatter_map /= global_scatter_map.max()
    global_sphere_map /= global_sphere_map.max()
    global_no_spectra_map /= global_no_spectra_map.max()
    zero_reference_ak_map /= count
    fluence_distributions[:, 0] /= count
    fluence_distributions[:, 1] /= count
    print(f"Mean error against zero reference airkerma: {(zero_reference_ak_map.mean().item())}")

    print(f"Plotting spectra across files")
    random_spectra_scatter = torch.stack(random_spectra_scatter, dim=0)
    random_spectra_beam = torch.stack(random_spectra_beam, dim=0)
    show_figure(
        plot_spectra_volume(
            random_spectra_scatter,
            title="Random Scatter-only Spectra over Test Set",
            resolution=1.0
        ), title="Random Scatter-only Spectra over Test Set", store=True
    )
    show_figure(
        plot_spectra_volume(
            random_spectra_beam,
            title="Random Primary Beam Spectra over Test Set",
            resolution=1.0
        ), title="Random Primary Beam Spectra over Test Set", store=True
    )
    show_figure(
        plot_spectra_volume_stdev(
            random_spectra_scatter,
            title="Random Scatter-only Spectra over Test Set with StdDev",
            normalize=True
        ), title="Random Scatter-only Spectra over Test Set with StdDev", store=True
    )
    show_figure(
        plot_spectra_volume_stdev(
            random_spectra_beam,
            title="Random Primary Beam Spectra over Test Set with StdDev",
            normalize=True
        ), title="Random Primary Beam Spectra over Test Set with StdDev", store=True
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
        x=[i / 100 for i in range(100)],
        y=fluence_distributions[:, 0],
        name='Linear Normalized Fluence Distribution',
        marker_color='blue',
        opacity=0.75
    ))
    fig.add_trace(
        go.Bar(
        x=[i / 100 for i in range(100)],
        y=fluence_distributions[:, 1],
        name='Log Normalized Fluence Distribution',
        marker_color='red',
        opacity=0.75
    ))
    fig.update_layout(
        title='Fluence Distributions over Test Set',
        xaxis_title='Normalized Fluence',
        yaxis_title='Counts',
        font=dict(size=16),
    )
    show_figure(fig, title="Fluence Distributions over Test Set")

    top_50_map = global_top_50_map
    top_90_map = global_top_90_map
    scatter_map = global_scatter_map
    sphere_map = global_sphere_map
    global_no_spectra_map = global_no_spectra_map

    spec_fig.update_layout(
        title='Random Voxel Spectrum',
        xaxis_title='Energy in keV',
        yaxis_title='Intensity',
        font=dict(size=16),
    )
    show_figure(spec_fig, title="Random Voxel Spectrum")

    random_radfields = RadiationFieldChannel(
        spectrum=torch.stack([rf.spectrum.squeeze(0) for rf in random_radfields], dim=0),
        flux=torch.stack([rf.flux.squeeze(0) for rf in random_radfields], dim=0),
    )
    fig = airkerma_plotter.plot_airkerma_volume(random_radfields, title="Random Ground Truth Airkerma Volumes")
    show_figure(fig, title="Random Ground Truth Airkerma Volumes")

    fig = sliced_plotter.plot_airkerma_slices(random_radfields, X=0.5, title="Random Ground Truth Airkerma Slices YZ at X=0.5m")
    show_figure(fig, title="Random Ground Truth Airkerma Slices YZ at X=0.5m")

    fig = sliced_plotter.create_airkerma_subplot_figure(rows=3, cols=2, title="Random Ground Truth Airkerma Slices Subplots")
    for i in range(2):
        sliced_plotter.add_airkerma_slice_to_figure(
            fig,
            RadiationFieldChannel(
                spectrum=random_radfields.spectrum[i].unsqueeze(0),
                flux=random_radfields.flux[i].unsqueeze(0)
            ),
            X=0.5,
            name="Random Slice YZ at X=0.5m",
            row=1,
            col=i+1
        )
        sliced_plotter.add_airkerma_slice_to_figure(
            fig,
            RadiationFieldChannel(
                spectrum=random_radfields.spectrum[i].unsqueeze(0),
                flux=random_radfields.flux[i].unsqueeze(0)
            ),
            Y=0.5,
            name="Random Slice XZ at Y=0.5m",
            row=2,
            col=i+1
        )
        sliced_plotter.add_airkerma_slice_to_figure(
            fig,
            RadiationFieldChannel(
                spectrum=random_radfields.spectrum[i].unsqueeze(0),
                flux=random_radfields.flux[i].unsqueeze(0)
            ),
            Z=0.5,
            name="Random Slice YZ at X=0.5m",
            row=3,
            col=i+1
        )
    show_figure(fig, title="Random Ground Truth Airkerma Slices Subplots")

    fig = sliced_plotter.create_airkerma_figure(title="Blended Random Airkerma Slices")
    sliced_plotter.add_blended_slices(
        fig,
        pred_field=RadiationFieldChannel(
            spectrum=random_radfields.spectrum[0],
            flux=random_radfields.flux[0]
        ),
        gt_field=RadiationFieldChannel(
            spectrum=random_radfields.spectrum[1],
            flux=random_radfields.flux[1]
        ),
        X=0.5
    )
    show_figure(fig, title="Blended Random Airkerma Slices")

    fig = airkerma_plotter.plot_airkerma_volume(top_50_map, title="Top 50% Airkerma Accuracy Volume")
    show_figure(fig, title="Top 50% Airkerma Accuracy Volume")

    fig = airkerma_plotter.plot_airkerma_volume(top_90_map, title="Top 90% Airkerma Accuracy Volume")
    show_figure(fig, title="Top 90% Airkerma Accuracy Volume")

    fig = airkerma_plotter.plot_airkerma_volume(scatter_map, title="Scatter-only Airkerma Accuracy Volume")
    show_figure(fig, title="Scatter-only Airkerma Accuracy Volume")

    fig = go.Figure(
        data=go.Heatmap(
            z=scatter_map[:, scatter_map.shape[1]//2, :],
            colorscale='Viridis',
            colorbar=dict(title='Scatter-only Map')
        )
    )
    show_figure(fig, title="Scatter-only Map XZ-plane")
    fig = go.Figure(
        data=go.Heatmap(
            z=scatter_map[scatter_map.shape[0]//2, :, :],
            colorscale='Viridis',
            colorbar=dict(title='Scatter-only Map')
        )
    )
    show_figure(fig, title="Scatter-only Map YZ-plane")
    fig = go.Figure(
        data=go.Heatmap(
            z=scatter_map[:, :, scatter_map.shape[2]//2],
            colorscale='Viridis',
            colorbar=dict(title='Scatter-only Map')
        )
    )
    show_figure(fig, title="Scatter-only Map XY-plane")

    fig = airkerma_plotter.plot_airkerma_volume(zero_reference_ak_map, title="Zero Reference Airkerma Volume")
    show_figure(fig, title="Zero Reference Airkerma Volume")

    #scatter_map_xz = scatter_map[:, scatter_map.shape[1]//2, :]
    #fig = go.Figure(data=go.Heatmap(
    #    z=scatter_map_xz,
    #    colorscale='Viridis',
    #    colorbar=dict(title='Scatter-only Airkerma Accuracy')
    #))
    #fig.update_layout(title='Scatter-only Airkerma Accuracy XZ Slice')
    #fig.show()

    #fig = go.Figure(go.Volume(
    #    x=[i for i in range(sphere_map.shape[0]) for j in range(sphere_map.shape[1]) for k in range(sphere_map.shape[2])],
    #    y=[j for i in range(sphere_map.shape[0]) for j in range(sphere_map.shape[1]) for k in range(sphere_map.shape[2])],
    #    z=[k for i in range(sphere_map.shape[0]) for j in range(sphere_map.shape[1]) for k in range(sphere_map.shape[2])],
    #    value=sphere_map.flatten(),
    #    isomin=0,
    #    isomax=1,
    #    opacity=0.1, # needs to be small to see through all surfaces
    #    surface_count=20, # needs to be a large number for good volume rendering
    #    colorscale='Viridis',
    #    colorbar=dict(title='Sphere Count')
    #))
    #fig.update_layout(title='Sphere Count Volume')
    #fig.show()

    fig = go.Figure(go.Volume(
        x=[i for i in range(global_no_spectra_map.shape[0]) for j in range(global_no_spectra_map.shape[1]) for k in range(global_no_spectra_map.shape[2])],
        y=[j for i in range(global_no_spectra_map.shape[0]) for j in range(global_no_spectra_map.shape[1]) for k in range(global_no_spectra_map.shape[2])],
        z=[k for i in range(global_no_spectra_map.shape[0]) for j in range(global_no_spectra_map.shape[1]) for k in range(global_no_spectra_map.shape[2])],
        value=global_no_spectra_map.flatten(),
        isomin=0,
        isomax=1,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        colorscale='Viridis',
        colorbar=dict(title='No Spectra Count')
    ))
    fig.update_layout(font=dict(size=16), title='No Spectra Count Volume')
    show_figure(fig, title="No Spectra Count Volume")

    fig = go.Figure(data=go.Histogram(
        x=empty_spectra_rel_counts,
        nbinsx=50,
        marker_color='blue',
        opacity=0.75
    ))
    fig.update_layout(
        title='Relative Count of Empty Spectra Voxels per Volume',
        xaxis_title='Relative Count of Empty Spectra Voxels',
        yaxis_title='Number of Volumes',
        font=dict(size=16),
    )
    show_figure(fig, title="Relative Count of Empty Spectra Voxels per Volume")

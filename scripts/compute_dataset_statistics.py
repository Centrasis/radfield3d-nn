"""
Compute the dataset-level statistics that are stored in ``<dataset>/statistics.json``.

This is a self-contained re-implementation of the statistics block of the original
``radfield3d-modeltrain-framework/tools/analyze_dataset.py`` (the plotting / air-kerma
parts of that tool are intentionally dropped — they were never part of statistics.json,
and they required an external ``mu_tr`` table).

Metrics (all aggregated over every ``*.rf3`` field in ``<dataset>/fields``):

    directions.{x,y,z}        tube radiation direction components (from metadata)
    tube_distances_m          ||radiation_origin|| in meters (tube distance to field center)
    tube_voltages_kVp         energy (keV) of the highest non-empty tube-spectrum bin (peak kVp)
    tube_spectrum_means_keV   mean energy of the (normalised) tube spectrum, in keV
    tube_opening_angles_deg   first beam-shape parameter (rect.x for RECTANGLE fields)
    empty_voxels_fractions    fraction of voxels with linearly-normalised flux < 1e-3
    min_max_fluences          per-field min and max of the channel-joined flux
    fluences.GUK              Gini coefficient of the aggregated normalised-flux histogram

Each scalar metric is summarised with Mean / Std / Min / Max / IQR / Dynamic Range (db),
matching the original tool exactly.

Usage:
    python scripts/compute_dataset_statistics.py --dataset_path /path/to/dataset
    # writes <dataset>/statistics.json (use --dry-run to print without writing)
"""
import argparse
import json
import os

import torch
from rich import print
from rich.progress import track

from radfiled3d.store import FieldStore
from radfiled3d import CartesianRadiationField
from radfiled3d.pytorch.datasets.radfield3d import RadField3DDataset

from radfield3dnn.datasets import is_zipped_dataset, list_dataset_field_names, load_dataset_field, dataset_statistics_path
from radfield3dnn.datasets.channel_join import ChannelsJoin
from radfield3dnn.preprocessing.normalizations.linear import LinearNormalizer


MAX_ENERGY_EV = 1.5e+5
FLUENCE_DISTRIBUTION_BINS = int(1e+6)


def calc_gini_coeff(values: torch.Tensor) -> torch.Tensor:
    """Gini coefficient (GUK) of a 1-D, non-negative distribution."""
    sorted_values = torch.sort(values)[0]
    n = len(sorted_values)
    indices = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    return (2 * indices * sorted_values).sum() / (n * sorted_values.sum()) - (n + 1) / n


def summarise(values: torch.Tensor) -> dict:
    """Mean/Std/Min/Max/IQR/Dynamic-Range summary, matching analyze_dataset.py."""
    return {
        "Mean": values.mean().item(),
        "Std": values.std().item(),
        "Min": values.min().item(),
        "Max": values.max().item(),
        "IQR": (values.quantile(0.75) - values.quantile(0.25)).item(),
        "Dynamic Range (db)": (10 * torch.log10(values.max() / values.min().clamp_min(1e-12))).item(),
    }


def compute_statistics(dataset_path: str, max_fields: int = None, exclude_files: set = None) -> dict:
    zipped = is_zipped_dataset(dataset_path)
    file_paths = list_dataset_field_names(dataset_path)
    if exclude_files:
        before = len(file_paths)
        # the lister returns absolute paths for a directory dataset; the exclusion list names fields dataset-relative
        rel = lambda f: f if zipped else os.path.relpath(f, dataset_path)
        file_paths = [f for f in file_paths if rel(f) not in exclude_files]
        print(f"[yellow]Excluded {before - len(file_paths)} fields listed as unusable[/yellow]")
    if not file_paths:
        raise ValueError(f"No .rf3 fields found in {dataset_path}")
    if max_fields is not None and max_fields > 0:
        step = max(1, len(file_paths) // max_fields)
        file_paths = file_paths[::step][:max_fields]
        print(f"[yellow]Sampling every {step}th field -> {len(file_paths)} of the dataset[/yellow]")
    print(f"Found [bold]{len(file_paths)}[/bold] fields in {dataset_path}")

    # Field dimensions (meters) are constant across the dataset -> read once.
    test_field: CartesianRadiationField = load_dataset_field(dataset_path, file_paths[0])
    fd = test_field.get_field_dimensions()
    field_dimensions = torch.tensor([fd.x, fd.y, fd.z], dtype=torch.float32)

    # Per-file accessor: a dataset whose generator writes some metadata only for some fields has
    # more than one byte layout, and the shared accessor would misread every field but the first's.
    from radfield3dnn.datasets.lobes import PerFileAccessorDataset
    dataset = (PerFileAccessorDataset(zip_file=dataset_path, file_paths=file_paths) if zipped
               else PerFileAccessorDataset(file_paths=file_paths))
    join = ChannelsJoin()
    linear_normalizer = LinearNormalizer((0, 1)).eval()

    ds_parameters = {
        "directions": {"x": [], "y": [], "z": []},
        "tube_distances_m": [],
        "tube_voltages_kVp": [],
        "tube_spectrum_means_keV": [],
        "tube_opening_angles_deg": [],
        "empty_voxels_fractions": [],
        "min_max_fluences": [],
    }
    fluence_distributions = torch.zeros(FLUENCE_DISTRIBUTION_BINS, dtype=torch.float32)
    count = 0

    for idx in track(range(len(dataset)), description="Processing dataset..."):
        train_in = join(dataset[idx])
        count += 1

        flux = train_in.ground_truth.flux
        norm_flu = linear_normalizer.forward(flux)
        fluence_distributions += torch.histogram(
            norm_flu, bins=FLUENCE_DISTRIBUTION_BINS, range=(0.0, 1.0)
        ).hist

        direction = train_in.input.direction
        ds_parameters["directions"]["x"].append(direction[0].item())
        ds_parameters["directions"]["y"].append(direction[1].item())
        ds_parameters["directions"]["z"].append(direction[2].item())

        # origin is normalised to [0,1] relative to field center -> recover meters.
        tube_position_m = (train_in.input.origin * field_dimensions) - (field_dimensions / 2)
        ds_parameters["tube_distances_m"].append(torch.linalg.norm(tube_position_m).item())

        ds_parameters["min_max_fluences"] += [flux.min().item(), flux.max().item()]

        spectrum = train_in.input.spectrum
        energies_keV = torch.arange(0, spectrum.size(0), dtype=spectrum.dtype) * (MAX_ENERGY_EV / spectrum.size(0)) / 1000.0
        highest_index = (spectrum > 0).nonzero(as_tuple=True)[0].max().item()
        ds_parameters["tube_voltages_kVp"].append(energies_keV[highest_index - 1].item())
        ds_parameters["tube_spectrum_means_keV"].append(
            (spectrum.dot(energies_keV) / spectrum.sum().clamp_min(1e-12)).item()
        )

        ds_parameters["tube_opening_angles_deg"].append(train_in.input.beam_shape_parameters[0].item())
        ds_parameters["empty_voxels_fractions"].append((norm_flu < 1e-3).float().sum().item() / norm_flu.numel())

    statistics = {}
    for key, values in ds_parameters.items():
        if isinstance(values, dict):
            statistics[key] = {sub: summarise(torch.tensor(subvals)) for sub, subvals in values.items()}
        else:
            statistics[key] = summarise(torch.tensor(values))

    statistics["fluences"] = {"GUK": calc_gini_coeff(fluence_distributions / count).item()}
    return statistics


def main():
    parser = argparse.ArgumentParser(
        prog="compute_dataset_statistics",
        description="Compute and write statistics.json for a RadField3D dataset (directory or zip).",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset directory (containing a 'fields' folder) or dataset .zip.")
    parser.add_argument("--max_fields", type=int, default=None, help="Sample at most this many fields, spread evenly over the dataset.")
    parser.add_argument("--exclude_files", type=str, default=None, help="JSON file with a 'fields' list of dataset-relative field names to skip.")
    parser.add_argument("--dry-run", action="store_true", help="Print the statistics without writing statistics.json.")
    args = parser.parse_args()

    exclude = None
    if args.exclude_files:
        exclude = set(json.load(open(args.exclude_files))["fields"])
    statistics = compute_statistics(args.dataset_path, max_fields=args.max_fields, exclude_files=exclude)
    print(statistics)

    if args.dry_run:
        print("[yellow]--dry-run: not writing statistics.json[/yellow]")
        return
    out_path = dataset_statistics_path(args.dataset_path)
    with open(out_path, "w") as f:
        json.dump(statistics, f, indent=4)
    print(f"[green]Wrote statistics to {out_path}[/green]")


if __name__ == "__main__":
    main()

from pathlib import Path
from rich import print
from rich.progress import track
import plotly.graph_objects as go
import os
import argparse
import sys
import numpy as np
import multiprocessing as mp
    

def extract_old_field_metadata(file_path: str) -> dict:
    before_modules = list(sys.modules.keys())
    import RadFiled3D_old.RadFiled3D as Old_RadFiled3D

    metadata = Old_RadFiled3D.FieldStore.load_metadata(file_path)
    header = metadata.get_header()
    dyn_keys = metadata.get_dynamic_metadata_keys()
    if "tube_spectrum" in dyn_keys:
        spectrum: Old_RadFiled3D.HistogramVoxel = metadata.get_dynamic_metadata("tube_spectrum")
        spectrum_bins = spectrum.get_bins()
        spectrum_bin_width = spectrum.get_histogram_bin_width()
        spectrum = spectrum.get_histogram().copy()
    else:
        spectrum = None

    spec_data = {
        "bins": spectrum_bins,
        "bin_width": spectrum_bin_width,
        "spectrum": spectrum
    } if spectrum is not None else None


    data = {
        "software": {
            "name": header.software.name,
            "version": header.software.version,
            "commit": header.software.commit,
            "doi": header.software.doi,
            "repository": header.software.repository
        },
        "simulation": {
            "geometry": header.simulation.geometry,
            "primary_particle_count": header.simulation.primary_particle_count,
            "physics_list": header.simulation.physics_list,
            "tube": {
                "max_energy_eV": header.simulation.tube.max_energy_eV,
                "radiation_direction": (header.simulation.tube.radiation_direction.x, header.simulation.tube.radiation_direction.y, header.simulation.tube.radiation_direction.z),
                "radiation_origin": (header.simulation.tube.radiation_origin.x, header.simulation.tube.radiation_origin.y, header.simulation.tube.radiation_origin.z),
                "tube_id": header.simulation.tube.tube_id,
                "spectrum": spec_data
            }
        }
    }
    for module_name in list(sys.modules.keys()):
        if module_name not in before_modules:
            
            del sys.modules[module_name]

    return data

def extract_old_field_data(file_path: str) -> dict:
    before_modules = list(sys.modules.keys())
    import RadFiled3D_old.RadFiled3D as Old_RadFiled3D

    def get_dtype_from_nptype(dtype, shape) -> Old_RadFiled3D.DType:
        if dtype == "float32":
            if len(shape) == 3 or (len(shape) == 4 and shape[-1] == 1):
                return Old_RadFiled3D.DType.FLOAT32
            elif len(shape) == 4 and shape[-1] == 2:
                return Old_RadFiled3D.DType.VEC2
            elif len(shape) == 4 and shape[-1] == 3:
                return Old_RadFiled3D.DType.VEC3
            elif len(shape) == 4 and shape[-1] == 4:
                return Old_RadFiled3D.DType.VEC4
            elif len(shape) == 4 and shape[-1] > 4:
                return Old_RadFiled3D.DType.HISTOGRAM
            raise ValueError(f"Unknown shape for dtype {dtype}: {shape}")
        elif dtype == "float64":
            return Old_RadFiled3D.DType.FLOAT64
        elif dtype == "int32":
            return Old_RadFiled3D.DType.INT32
        elif dtype == "uint64":
            return Old_RadFiled3D.DType.UINT64
        elif dtype == "int8":
            return Old_RadFiled3D.DType.BYTE
        elif dtype == "bool":
            return Old_RadFiled3D.DType.BYTE
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    
    field: Old_RadFiled3D.CartesianRadiationField = Old_RadFiled3D.FieldStore.load(file_path)
    if field.get_typename() != "CartesianRadiationField":
        raise ValueError(f"Unexpected field type: {field.get_typename()}")
    
    field_dimensions = field.get_field_dimensions()
    voxel_dimensions = field.get_voxel_dimensions()
    existing_channels = field.get_channel_names()

    data = {
        "field_dimensions": (field_dimensions.x, field_dimensions.y, field_dimensions.z),
        "voxel_dimensions": (voxel_dimensions.x, voxel_dimensions.y, voxel_dimensions.z),
        "channels": {}
    }
                    
    for channel_name in existing_channels:
        original_channel = field.get_channel(channel_name)
        existing_layers = original_channel.get_layers()
        data["channels"][channel_name] = {}

        for layer_name in existing_layers:
            layer_data = field.get_channel(channel_name).get_layer_as_ndarray(layer_name)

            dtype = get_dtype_from_nptype(layer_data.dtype.name, layer_data.shape)
            if dtype == Old_RadFiled3D.DType.HISTOGRAM:
                a_vx: Old_RadFiled3D.HistogramVoxel = original_channel.get_voxel_flat(layer_name, 0)
                bin_width = a_vx.get_histogram_bin_width()
            else:
                bin_width = None

            data["channels"][channel_name][layer_name] = {
                "dtype": dtype.name,
                "unit": original_channel.get_layer_unit(layer_name),
                "data": layer_data.copy(),
                "bin_width": bin_width
            }

    for module_name in list(sys.modules.keys()):
        if module_name not in before_modules:
            
            del sys.modules[module_name]

    return data


def write_new_field_to(metadata_original: dict, field_data: dict, file_path: str, field_shape: str = None, shape_parameters: list = None):
    before_modules = list(sys.modules.keys())
    from RadFiled3D.RadFiled3D import CartesianRadiationField, DType, HistogramVoxel, vec3, FieldShape, vec2
    from RadFiled3D.utils import FieldStore
    from RadFiled3D.metadata.v1 import Metadata

    if field_shape == "cone":
        field_shape = FieldShape.CONE
    elif field_shape == "rectangle":
        field_shape = FieldShape.RECTANGLE
    elif field_shape == "ellipsis":
        field_shape = FieldShape.ELLIPSIS
    elif field_shape is not None:
        raise ValueError(f"Unknown field shape: {field_shape}")

    def get_dtype_from_name(name: str) -> DType:
        if name == DType.FLOAT32.name:
            return DType.FLOAT32
        elif name == DType.FLOAT64.name:
            return DType.FLOAT64
        elif name == DType.INT32.name:
            return DType.INT32
        elif name == DType.UINT64.name:
            return DType.UINT64
        elif name == DType.BYTE.name:
            return DType.BYTE
        elif name == DType.VEC2.name:
            return DType.VEC2
        elif name == DType.VEC3.name:
            return DType.VEC3
        elif name == DType.VEC4.name:
            return DType.VEC4
        elif name == DType.HISTOGRAM.name:
            return DType.HISTOGRAM
        else:
            raise ValueError(f"Unknown dtype name: {name}")

    new_field = CartesianRadiationField(
        vec3(
            field_data["field_dimensions"][0],
            field_data["field_dimensions"][1],
            field_data["field_dimensions"][2]
        ),
        vec3(
            field_data["voxel_dimensions"][0],
            field_data["voxel_dimensions"][1],
            field_data["voxel_dimensions"][2]
        )
    )
    metadata: Metadata = FieldStore.load_metadata(file_path)

    for channel_name, channel_data in field_data["channels"].items():
        new_channel = new_field.add_channel(channel_name)
        for layer_name, layer_data in channel_data.items():
            dtype = get_dtype_from_name(layer_data["dtype"])
            if dtype != DType.HISTOGRAM:
                assert layer_data["bin_width"] is None, "Bin width must be None for non-histogram layer"
                new_channel.add_layer(
                    layer_name,
                    layer_data["unit"],
                    dtype
                )
            else:
                assert layer_data["bin_width"] is not None, "Bin width must be provided for histogram layer"
                a_vx: HistogramVoxel = new_channel.add_histogram_layer(
                    layer_name,
                    layer_data["data"].shape[-1],
                    layer_data["bin_width"],
                    layer_data["unit"]
                )
            new_channel.get_layer_as_ndarray(layer_name)[:] = layer_data["data"]

    metadata.software.name = metadata_original["software"]["name"]
    metadata.software.version = metadata_original["software"]["version"]
    metadata.software.commit = metadata_original["software"]["commit"]
    metadata.software.doi = metadata_original["software"]["doi"]
    metadata.software.repository = metadata_original["software"]["repository"]

    metadata.simulation.geometry = metadata_original["simulation"]["geometry"]
    metadata.simulation.primary_particle_count = metadata_original["simulation"]["primary_particle_count"]
    metadata.simulation.physics_list = metadata_original["simulation"]["physics_list"]
    metadata.simulation.tube.max_energy_eV = metadata_original["simulation"]["tube"]["max_energy_eV"]
    metadata.simulation.tube.radiation_direction = vec3(
        metadata_original["simulation"]["tube"]["radiation_direction"][0],
        metadata_original["simulation"]["tube"]["radiation_direction"][1],
        metadata_original["simulation"]["tube"]["radiation_direction"][2]
    )
    metadata.simulation.tube.radiation_origin = vec3(
        metadata_original["simulation"]["tube"]["radiation_origin"][0],
        metadata_original["simulation"]["tube"]["radiation_origin"][1],
        metadata_original["simulation"]["tube"]["radiation_origin"][2]
    )
    metadata.simulation.tube.tube_id = metadata_original["simulation"]["tube"]["tube_id"]

    if field_shape is not None:
        metadata.simulation.tube.field_shape = field_shape
    if field_shape == FieldShape.CONE:
        metadata.simulation.tube.opening_angle_deg = shape_parameters[0]
    elif field_shape == FieldShape.RECTANGLE:
        metadata.simulation.tube.field_rect_dimensions_m = vec2(shape_parameters[0], shape_parameters[1])
    elif field_shape == FieldShape.ELLIPSIS:
        metadata.simulation.tube.field_ellipsis_opening_angles_deg = vec2(shape_parameters[0], shape_parameters[1])
    else:
        raise ValueError(f"Unknown field shape: {field_shape}")
    
    if metadata_original["simulation"]["tube"]["spectrum"] is not None:
        bins = metadata.simulation.tube.spectrum.shape[0]
        bin_width = abs(metadata.simulation.tube.spectrum[0, 0] - metadata.simulation.tube.spectrum[1, 0])
        assert abs(bin_width - metadata_original["simulation"]["tube"]["spectrum"]["bin_width"]) < 1e-6, "Spectrum bin width does not match"
        assert bins == metadata_original["simulation"]["tube"]["spectrum"]["bins"], "Spectrum bins do not match"
        spectrum = np.zeros((bins, 2), dtype=np.float32)
        spectrum[:, 0] = np.arange(bins) * bin_width
        metadata.simulation.tube.max_energy_eV = bins * bin_width
        spectrum[:, 1] = metadata_original["simulation"]["tube"]["spectrum"]["spectrum"]
        spectrum[:, 1] /= np.sum(spectrum[:, 1])  # normalize
        metadata.simulation.tube.spectrum = spectrum
    
    os.remove(file_path)
    FieldStore.store(new_field, metadata=metadata, file=file_path)

    for module_name in list(sys.modules.keys()):
        if module_name not in before_modules:
            
            del sys.modules[module_name]


def plot_spectrum_from_new_field(file_path: str):
    from RadFiled3D.utils import FieldStore

    metadata = FieldStore.load_metadata(file_path)
    if metadata.simulation.tube.spectrum is not None:
        spectrum = metadata.simulation.tube.spectrum
        bin_width = abs(spectrum[0, 0] - spectrum[1, 0])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=spectrum[:, 0], y=spectrum[:, 1], width=bin_width, name="Spectrum"))
        fig.update_layout(
            title="X-ray Tube Spectrum",
            xaxis_title="Energy (eV)",
            yaxis_title="Counts",
            bargap=0.1,
        )
        fig.show()
    else:
        print("No spectrum found in the field metadata.")

    for module_name in list(sys.modules.keys()):
        if 'RadFiled3D' in module_name:
            del sys.modules[module_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--field_shape", type=str, help="Field shape write to the field (cone, rectangle, ellipsis). Default is not redefine field shape.", default=None, required=False)
    parser.add_argument("--shape_parameters", type=float, nargs='+', help="Shape parameters for the field shape like opening angle in degree or field dimensions in meters.", required=False)
    parser.add_argument("--show_spectra_plot", action="store_true", help="Show spectra plot for each field.")
    args = parser.parse_args()

    path = args.dataset_path
    field_shape = args.field_shape
    if field_shape is not None:
        if isinstance(field_shape, list) or isinstance(field_shape, tuple):
            assert len(field_shape) == 1, "Field shape must be a single string."
            field_shape = field_shape[0]
    
    shape_parameters = args.shape_parameters if args.shape_parameters is not None else None
    assert field_shape is None or shape_parameters is not None, "If field shape is defined, shape parameters must be defined too."
    field_names = set()

    IS_ZIP = path.endswith(".zip")
    if IS_ZIP:
        raise ValueError("This script does not support zip files. Please extract the dataset first.")
    file_paths = [str(f) for f in (Path(path)).rglob("*.rf3")]

    for file_path in track(file_paths, description="[yellow]converting dataset..."):
        with mp.Pool(1) as pool:
            field = pool.apply(extract_old_field_data, args=(file_path,))
        with mp.Pool(1) as pool:
            metadata_original = pool.apply(extract_old_field_metadata, args=(file_path,))

        # load spectrum and plot it
        if metadata_original["simulation"]["tube"]["spectrum"] is not None:
            spectrum = metadata_original["simulation"]["tube"]["spectrum"]["spectrum"]
            spectrum_bins = metadata_original["simulation"]["tube"]["spectrum"]["bins"]
            spectrum_bin_width = metadata_original["simulation"]["tube"]["spectrum"]["bin_width"]
            if args.show_spectra_plot:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=np.arange(spectrum_bins) * spectrum_bin_width, y=spectrum, width=spectrum_bin_width, name="Spectrum"))
                fig.update_layout(
                    title="X-ray Tube Spectrum",
                    xaxis_title="Energy (eV)",
                    yaxis_title="Counts",
                    bargap=0.1,
                )
                fig.show()

        with mp.Pool(1) as pool:
            pool.apply(write_new_field_to, args=(metadata_original, field, file_path, field_shape, shape_parameters))
        if args.show_spectra_plot:
            with mp.Pool(1) as pool:
                pool.apply(plot_spectrum_from_new_field, args=(file_path,))
            while True:
                key_pressed = input("Press q to continue to next field, or ctrl+c to abort: ")
                if key_pressed == "q":
                    break

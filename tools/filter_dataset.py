from RadFiled3D.RadFiled3D import CartesianRadiationField, DType, HistogramVoxel, vec3
from RadFiled3D.RadFiled3D import FieldStore
from RadFiled3D.pytorch.helpers import RadiationFieldHelper
from RadFiled3D.metadata.v1 import Metadata, RadiationFieldMetadataHeaderV1
from pathlib import Path
import uuid
from rich import print
from rich.progress import track
from datasets.crop_dataset import CropDataset
import os
import argparse


def get_dtype_from_nptype(dtype, shape) -> DType:
    if dtype == "float32":
        if len(shape) == 3 or (len(shape) == 4 and shape[-1] == 1):
            return DType.FLOAT32
        elif len(shape) == 4 and shape[-1] == 2:
            return DType.VEC2
        elif len(shape) == 4 and shape[-1] == 3:
            return DType.VEC3
        elif len(shape) == 4 and shape[-1] == 4:
            return DType.VEC4
        elif len(shape) == 4 and shape[-1] > 4:
            return DType.HISTOGRAM
        raise ValueError(f"Unknown shape for dtype {dtype}: {shape}")
    elif dtype == "float64":
        return DType.FLOAT64
    elif dtype == "int32":
        return DType.INT32
    elif dtype == "uint64":
        return DType.UINT64
    elif dtype == "int8":
        return DType.BYTE
    elif dtype == "bool":
        return DType.BYTE
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--channels_to_keep", type=str, nargs='+', default=None, help="List of channels to keep")
    parser.add_argument("--layers_to_keep", type=str, nargs='+', default=None, help="List of layers to keep")
    parser.add_argument("--enforce_voxel_resolution", type=int, nargs=3, default=None, help="Voxel resolution to enforce (x, y, z)")
    parser.add_argument("--randomize_names", action="store_true", help="Randomize the name of the output files")
    parser.add_argument("--enforce_channels_and_layers", action="store_true", help="Enforce that all channels and layers to keep are present")
    parser.add_argument("--remove_erroneous", action="store_true", help="Remove erroneous files")
    args = parser.parse_args()

    path = args.dataset_path
    channels_to_keep = args.channels_to_keep
    layers_to_keep = args.layers_to_keep
    enforce_voxel_resolution = args.enforce_voxel_resolution
    randomize_names = args.randomize_names
    enforce_channels_and_layers = args.enforce_channels_and_layers
    remove_erroneous = args.remove_erroneous
    field_names = set()

    crop = CropDataset(enforce_voxel_resolution)

    IS_ZIP = path.endswith(".zip")
    if IS_ZIP:
        raise ValueError("This script does not support zip files. Please extract the dataset first.")
    file_paths = [str(f) for f in (Path(path)).rglob("*.rf3")]

    if randomize_names:
        for file_path in file_paths:
            name = os.path.basename(file_path)
            name = os.path.splitext(name)[0]
            field_names.add(name)

    accessor = FieldStore.construct_field_accessor(file_paths[0])
    for file_path in track(file_paths, description="[yellow]Filtering fields..."):
        try:
            field: CartesianRadiationField = FieldStore.load(file_path)
            if field.get_typename() != "CartesianRadiationField":
                raise ValueError(f"Unexpected field type: {field.get_typename()}")
            metadata_original: Metadata = FieldStore.load_metadata(file_path)

            field_dimensions = field.get_field_dimensions()
            if enforce_voxel_resolution is not None:
                field_dimensions = vec3(
                    enforce_voxel_resolution[0] * field.get_voxel_dimensions().x,
                    enforce_voxel_resolution[1] * field.get_voxel_dimensions().y,
                    enforce_voxel_resolution[2] * field.get_voxel_dimensions().z
                )

            new_field = CartesianRadiationField(
                field_dimensions,
                field.get_voxel_dimensions()
            )
            at_least_one_channel = False
            at_least_one_layer = False
            existing_channels = field.get_channel_names()
            if enforce_channels_and_layers:
                for channel_name in channels_to_keep:
                    if channel_name not in existing_channels:
                        raise ValueError(f"Channel '{channel_name}' is not present in the field.")
                    
            not_found_layers = layers_to_keep.copy()
            for channel_name in existing_channels:
                if channel_name not in channels_to_keep:
                    continue

                at_least_one_channel = True

                original_channel = field.get_channel(channel_name)
                new_channel = new_field.add_channel(channel_name)
                existing_layers = original_channel.get_layers()

                for layer_name in existing_layers:
                    if layer_name in not_found_layers:
                        not_found_layers.remove(layer_name)

                    at_least_one_layer = True

                    layer_data = RadiationFieldHelper.load_tensor_from_field(field, channel_name, layer_name)
                    layer_data = crop.crop_tensor(layer_data)
                    layer_data = layer_data.permute(1, 2, 3, 0)
                    layer_data = layer_data.numpy()

                    dtype = get_dtype_from_nptype(layer_data.dtype.name, layer_data.shape)

                    if dtype != DType.HISTOGRAM:
                        new_channel.add_layer(
                            layer_name,
                            original_channel.get_layer_unit(layer_name),
                            dtype
                        )
                    else:
                        a_vx: HistogramVoxel = original_channel.get_voxel_flat(layer_name, 0)
                        bin_width = a_vx.get_histogram_bin_width()
                        bins = a_vx.get_bins()
                        new_channel.add_histogram_layer(
                            layer_name,
                            bins,
                            bin_width,
                            original_channel.get_layer_unit(layer_name)
                        )
                    new_channel.get_layer_as_ndarray(layer_name)[:] = layer_data

            if enforce_channels_and_layers and (len(not_found_layers) > 0):
                raise ValueError(f"Layers not found in channel '{channel_name}': {not_found_layers}")

            if at_least_one_channel and at_least_one_layer:
                os.remove(file_path)
                if randomize_names:
                    random_string = str(uuid.uuid4()).replace('-', '')
                    while random_string in field_names:
                        random_string = str(uuid.uuid4()).replace('-', '')
                    file_path = os.path.join(os.path.dirname(file_path), f"{random_string}.rf3")
                    field_names.add(random_string)
                FieldStore.store(new_field, metadata=metadata_original, file=file_path)
            else:
                print(f"[red]Skipping file {file_path} because it has no channels or layers to keep.[/red]")
        except Exception as e:
            print(f"[red]Error processing file {file_path}: {e}[/red]")
            if remove_erroneous:
                print(f"[red]Removing erroneous file {file_path}[/red]")
                os.remove(file_path)

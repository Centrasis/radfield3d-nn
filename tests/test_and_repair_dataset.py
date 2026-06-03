from RadFiled3D.pytorch.datasets import RadField3DDataset
from RadFiled3D.RadFiled3D import FieldStore
from pathlib import Path
import zipfile
from rich import print
from rich.progress import track
import sys
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and repair dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()
    path = args.dataset_path

    IS_ZIP = path.endswith(".zip")
    file_paths = []
    if IS_ZIP:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
    else:
        file_paths = [str(f) for f in (Path(path)).rglob("*.rf3")]

    durations_layer_load = []
    layer_name = None
    channel_name = None
    ds = RadField3DDataset(file_paths=file_paths, zip_file=path if IS_ZIP else None)
    try:
        for (field, metadata) in track(ds, description="[yellow]Loading fields..."):
            pass
        print("[green]Dataset was good!")
        sys.exit(0)
    except Exception as e:
        if IS_ZIP:
            raise ValueError(f"Error loading fields from zip file '{path}': {e}")
        else:
            print(f"Error loading fields! try fixing them...")

    accessor = FieldStore.construct_field_accessor(file_paths[0])
    for file_path in track(file_paths, description="[yellow]Fixing fields..."):
        try:
            field = accessor.access_field(file_path)
        except Exception as e:
            print(f"[yellow]Attempt to repair {file_path}:")
            try:
                field = FieldStore.load(file_path)
                metadata = FieldStore.load_metadata(file_path)
                os.remove(file_path)
                FieldStore.store(field, metadata=metadata, file=file_path)
                try:
                    field = accessor.access_field(file_path)
                    print(f"[green]Successfully repaired {file_path}!")
                except Exception as e:
                    print(f"[red]Failed to access repaired field {file_path}! Field was broken even after rewrite: {e}")
            except Exception as e:
                print(f"[red]Failed to repair {file_path}: {e}")

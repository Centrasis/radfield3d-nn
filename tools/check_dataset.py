from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDatasetWithGeometry
from pathlib import Path
import zipfile
from rich import print
from rich.progress import track
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    args = parser.parse_args()

    path = args.dataset_path
    IS_ZIP = path.endswith(".zip")
    file_paths = []
    if IS_ZIP:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
    else:
        file_paths = [str(f) for f in (Path(path)).rglob("*.rf3")]

    insufficient_geom_count = 0
    file_count = 0
    insufficient_files = []
    ds = RadField3DDatasetWithGeometry(file_paths=file_paths, zip_file=path if IS_ZIP else None, create_binary_geometry_mask=True)
    for (field, metadata) in track(ds, description="[yellow]Checking fields..."):
        geo_sum = torch.sum(field.geometry.squeeze(0), dim=(0, 1, 2))
        if geo_sum <= 50:
            print(f"[yellow]Field has low geometry sum: {geo_sum}.")
            insufficient_geom_count += 1
            insufficient_files.append(file_paths[file_count])
        file_count += 1

    print(f"[green]Dataset was good! {insufficient_geom_count} fields had low geometry.")
    print("[yellow]Insufficient files:")
    for file in insufficient_files:
        print(f" - {file}")

    ds = RadField3DDatasetWithGeometry(file_paths=insufficient_files, zip_file=path if IS_ZIP else None, create_binary_geometry_mask=True)
    all_insufficient_files = True
    for (field, metadata) in track(ds, description="[yellow]Validating findings..."):
        geo_sum = torch.sum(field.geometry.squeeze(0), dim=(0, 1, 2))
        if geo_sum > 50:
            print(f"[red]A field was incorrectly marked as insufficient geometry.")
            all_insufficient_files = False
        elif geo_sum >= 1:
            print(f"[yellow]Geom was not empty but very small: {geo_sum}.")

    if all_insufficient_files:
        rm_fields = input("Remove all insufficient files? (y/n)")
        if rm_fields.lower() == 'y':
            import os
            for path in insufficient_files:
                os.remove(path)
            print("[green]All insufficient files have been removed.")
    else:
        print("[red]Some files were incorrectly marked as insufficient geometry. Please check the dataset manually.")

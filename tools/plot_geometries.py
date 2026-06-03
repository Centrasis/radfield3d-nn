from RadFiled3D.RadFiled3D import FieldStore
from RadFiled3D.metadata.v1 import Metadata
from RadFiled3D.pytorch.helpers import RadiationFieldHelper
import os
import argparse
import torch
import numpy as np
from rich import print
from plotly import graph_objects as go
from plotly.subplots import make_subplots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize dataset geometry")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory which contains the subdirectories: 'fields', 'spectra' and 'geom_desc'")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    
    dirs = os.listdir(dataset_path)
    if "fields" not in dirs:
        raise ValueError(f"Dataset directory '{dataset_path}' must contain 'fields' subdirectories.")

    field_files = [os.path.join(dataset_path, "fields", f) for f in os.listdir(os.path.join(dataset_path, "fields")) if f.endswith(".rf3")]
    print(f"Found {len(field_files)} voxelized geometry files in '{os.path.join(dataset_path, 'fields')}'.")

    for field_file in field_files:
        field = FieldStore.load(field_file)

        if not field.has_channel("geometry"):
            print(f"[red]Warning:[/red] Field file '{field_file}' does not contain geometry channel.")
            continue

        geom_tensor = RadiationFieldHelper.load_tensor_from_field(field, "geometry", "density").to(torch.float32).squeeze(0)
        geom_tensor[geom_tensor > 0.0] = 1.0  # Convert to binary tensor

        if not (geom_tensor == 1.0).any():
            print(f"[red]Warning:[/red] Geometry tensor in '{field_file}' is empty.")
            continue

        geom_tensor_content_xz = torch.sum(geom_tensor, dim=1)
        geom_tensor_content_xy = torch.sum(geom_tensor, dim=2)
        geom_tensor_content_yz = torch.sum(geom_tensor, dim=0)

        # Get shapes for aspect ratio
        x_dim, y_dim, z_dim = geom_tensor.shape
        aspect_xz = x_dim / z_dim
        aspect_xy = x_dim / y_dim
        aspect_yz = y_dim / z_dim

        fig = make_subplots(rows=3, cols=1, subplot_titles=("XZ Plane", "XY Plane", "YZ Plane"))

        fig.add_trace(
            go.Heatmap(z=geom_tensor_content_xz.numpy(), colorscale='Viridis', showscale=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=geom_tensor_content_xy.numpy(), colorscale='Viridis', showscale=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Heatmap(z=geom_tensor_content_yz.numpy(), colorscale='Viridis', showscale=False),
            row=3, col=1
        )

        fig.update_layout(
            title=f"Voxelized Geometry from {os.path.basename(field_file)}",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            width=800,
            height=800 * 3
        )
        fig.show()

        while True:
            user_input = input("Press Enter to continue to the next geometry or type 'exit' to quit: ")
            if user_input.lower() == 'exit':
                break
            else:
                break

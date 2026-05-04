import json
from pathlib import Path
import torch
from RadFiled3D.pytorch.datasets.radfield3d import RadField3DDataset, RadField3DDatasetWithGeometry
from RadFiled3D.utils import FieldStore
from RadFiled3D.RadFiled3D import CartesianRadiationField
from enum import Enum
from RadFiled3D.pytorch.datasets.processing import DataProcessing
from .crop_dataset import CropDataset
from radfield3dnn import TrainingInputData, rf3TrainingInputData, RadiationField, rf3RadiationField, RadiationFieldChannel, AirKermaField
import os
from radfield3dnn.normalizations.beam_parameters import BeamParametersNormalization
from rich import print
from .dataloader import RadiationFieldDataModule


class DatasetType(Enum):
    Voxelwise = 0
    Layerwise = 1


class OriginalGroundTruthPreservation(DataProcessing):
    def clone_channel(self, channel: RadiationFieldChannel) -> RadiationFieldChannel:
        return RadiationFieldChannel(
            flux=channel.flux.clone() if channel.flux is not None else None,
            spectrum=channel.spectrum.clone() if channel.spectrum is not None else None,
            error=channel.error.clone() if channel.error is not None else None
        )

    def clone_radfield(self, field: RadiationField) -> RadiationField:
        return RadiationField(
            scatter_field=self.clone_channel(field.scatter_field) if field.scatter_field is not None else None,
            direct_beam=self.clone_channel(field.direct_beam) if field.direct_beam is not None else None,
            geometry=field.geometry.clone() if "geometry" in field._fields and field.geometry is not None else None
        )
    
    def clone(self, x: RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor) -> RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor:
        if isinstance(x, (RadiationField, rf3RadiationField)):
            return self.clone_radfield(x)
        elif isinstance(x, RadiationFieldChannel):
            return self.clone_channel(x)
        elif isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, AirKermaField):
            return AirKermaField(
                air_kerma=x.air_kerma.clone(),
                geometry=x.geometry.clone() if x.geometry is not None else None
            )
        else:
            raise ValueError("Input must be of type RadiationField, RadiationFieldChannel, AirKermaField, or torch.Tensor")

    def forward(self, x: TrainingInputData | RadiationField | RadiationFieldChannel | AirKermaField | torch.Tensor) -> TrainingInputData:
        if isinstance(x, rf3TrainingInputData):
            return TrainingInputData(
                input=x.input,
                ground_truth=x.ground_truth,
                original_ground_truth=self.clone(x.ground_truth)
            )
        elif isinstance(x, TrainingInputData):
            return TrainingInputData(
                input=x.input,
                ground_truth=x.ground_truth,
                original_ground_truth=self.clone(x.ground_truth)
            )
        else:
            raise ValueError("Input must be of type TrainingInputData")

    @classmethod
    def create_from_config(cls, config: dict) -> "OriginalGroundTruthPreservation":
        return OriginalGroundTruthPreservation()


def get_dataset_dimensions_and_voxel_size(dataset: str | RadiationFieldDataModule) -> tuple[tuple[int, int, int], float]:
    if isinstance(dataset, str):
        dataset_path = dataset
        datamodule = RadiationFieldDataModule(
            Path(dataset_path),
            batch_size=1,
            num_workers=0,
            dataset_cls=RadField3DDataset,
            val_ratio=0.0,
            test_ratio=1.0,
            train_ratio=0.0
        )
        datamodule.prepare_data()
        dataset = datamodule
    elif not isinstance(dataset, RadiationFieldDataModule):
        raise TypeError(f"dataset must be a string path or a RadiationFieldDataModule instance, but got {type(dataset)}")
    
    test_dl = dataset.test_dataloader()
    test_ds = test_dl.dataset
    test_files = test_ds.file_paths
    test_field: CartesianRadiationField = FieldStore.load(test_files[0])
    field_dim = test_field.get_field_dimensions()
    vx_size_x = field_dim.x / test_field.get_voxel_counts().x
    vx_size_y = field_dim.y / test_field.get_voxel_counts().y
    vx_size_z = field_dim.z / test_field.get_voxel_counts().z
    assert vx_size_x == vx_size_y and vx_size_x == vx_size_z, f"Voxels are not isotropic! Voxel sizes: {vx_size_x}, {vx_size_y}, {vx_size_z}"
    return (field_dim.x, field_dim.y, field_dim.z), vx_size_x


def construct_datamodule(dataset_path: str, batch_size: int, num_workers: int, use_geometry: bool, use_beam_parameters: bool, dataprocessings: list[DataProcessing] = None, voxel_resolution: tuple[int, int, int] = None) -> RadiationFieldDataModule:
    if dataprocessings is None:
        dataprocessings = []
    dataset_cls = RadField3DDataset
    if use_geometry:
        print("[yellow]Using geometry dataset with voxelized geometries!")
        def create_geom_ds(file_paths: list[str] = None, zip_file: str = None, data_processings: list["DataProcessing"] = None):
            return RadField3DDatasetWithGeometry(file_paths=file_paths, zip_file=zip_file, data_processings=data_processings, create_binary_geometry_mask=True)
        dataset_cls = create_geom_ds

    stats = {}
    if os.path.exists(os.path.join(dataset_path, "statistics.json")):
        statistics_path = os.path.join(dataset_path, "statistics.json")
        with open(statistics_path, "r") as f:
            stats = json.load(f)
        print(f"[green]Loaded dataset statistics from {statistics_path}[/green]")
    else:
        print(f"[yellow]No dataset statistics found at {os.path.join(dataset_path, 'statistics.json')}[/yellow]")

    field_dim, vx_size = get_dataset_dimensions_and_voxel_size(dataset_path)
    vx_counts = (int(field_dim[0] / vx_size), int(field_dim[1] / vx_size), int(field_dim[2] / vx_size))
    assert abs(field_dim[0] - vx_size * vx_counts[0]) < 1e-8 and abs(field_dim[1] - vx_size * vx_counts[1]) < 1e-8 and abs(field_dim[2] - vx_size * vx_counts[2]) < 1e-8, f"Voxel dimensions do not match calculated voxel size! {field_dim} vs. {vx_size}"
    print(f"[blue]Dataset field dimensions: {field_dim}, voxel size: {vx_size} m, voxel counts: {vx_counts}[/blue]")

    if voxel_resolution is not None:
        print(f"[blue]Testing dataset with voxel resolution {voxel_resolution}[/blue]")
        assert vx_counts[0] >= voxel_resolution[0] and vx_counts[1] >= voxel_resolution[1] and vx_counts[2] >= voxel_resolution[2], f"Voxel resolution of dataset {vx_counts} does not match enforced resolution {voxel_resolution}"
        
        dataprocessings.append(
            CropDataset(voxel_resolution)
        )
        print(f"[green]Voxel resolution of dataset matches enforced resolution {voxel_resolution}!")

    if use_beam_parameters:
        print("[yellow]Using beam parameters normalization!")
        beam_normalizer = BeamParametersNormalization(
            opening_angle_range_deg=(
                stats["tube_opening_angles_deg"]["Min"],
                stats["tube_opening_angles_deg"]["Max"]
            ),
            size_per_voxel_m=vx_size,
            is_origin_centered=False,
            distance_range_m=(
                stats["tube_distances_m"]["Min"],
                stats["tube_distances_m"]["Max"]
            ),
            half_field_size=(field_dim[0]/2, field_dim[1]/2, field_dim[2]/2)
        )
        dataprocessings.append(beam_normalizer)

    datamodule = RadiationFieldDataModule(
        Path(dataset_path),
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_cls=dataset_cls,
        data_processings=dataprocessings
    )
    datamodule.prepare_data()

    return datamodule

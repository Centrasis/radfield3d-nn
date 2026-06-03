from pathlib import Path
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn, MofNCompleteColumn
import time
import os
from RadFiled3D.RadFiled3D import FieldStore
import torch
from RadFiled3D.RadFiled3D import CartesianRadiationField
from RadFiled3D.pytorch import DataLoaderBuilder
    

class RandomVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, files: list):
        self.files = files
        self.accessor = FieldStore.construct_field_accessor(files[0])
        self.voxel_count = self.accessor.get_voxel_count()
        field: CartesianRadiationField = self.accessor.access_field(files[0])
        self.voxel_counts = field.get_voxel_counts()

    def __getitem__(self, index):
        voxel_idx = index % self.voxel_count
        file_idx = index // self.voxel_count
        vx_xyz = [
            voxel_idx % self.voxel_counts.x,
            (voxel_idx // self.voxel_counts.x) % self.voxel_counts.y,
            voxel_idx // (self.voxel_counts.x * self.voxel_counts.y)
        ]
        buffer = open(self.files[file_idx], "rb").read()
        vx = self.accessor.access_voxel_flat_from_buffer(buffer, "scatter_field", "spectrum", voxel_idx)
        vx_np = torch.from_numpy(vx.get_histogram())
        vx_xyz = torch.tensor(vx_xyz, dtype=torch.float32)
        return vx_xyz, vx_np
    
    def __len__(self):
        return self.voxel_count * len(self.files)


if __name__ == "__main__":
    path = str(Path("C:/Users/lehner04/Documents/Datasets") / "Cylinder-Multi-Spectra.zip")

    base_path = Path("C:/Users/lehner04/Documents/Datasets") / "C100-Sequence-2"
    #files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".rf3")]

    #accessor = FieldStore.construct_field_accessor(files[0])

    PROGRESSBAR = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
        )

    #for batch_size in (1, 2, 4, 8, 16, 32, 64, 128):
    for batch_size in (64,):
        #datamodule = RadiationFieldDataModule(
        #    #str(Path("C:/Users/lehner04/Documents/Datasets") / "Cylinder-Multi-Spectra.zip"),
        #    str(Path("C:/Users/lehner04/Documents/Datasets") / "C100-Sequence-2/fields"),
        #    batch_size=batch_size,
        #    num_workers=0
        #)

        #datamodule.prepare_data()
        #datamodule.setup("fit")

        #dl = datamodule.train_dataloader()
        #dl = datamodule.train_dataset
        #ds = RandomDataset(100, 1000000)
        builder = DataLoaderBuilder(base_path, dataset_class=VoxelwiseDataset, on_dataset_created=lambda x: x.set_channel_and_layer("scatter_field", "spectrum"))
        dl = builder.build_train_dataloader(batch_size=batch_size, worker_count=8)

        i = 0
        start = time.time()
        
        with PROGRESSBAR as progress:
            task = progress.add_task(f"Loading in batches({batch_size})...", total=len(dl))
            for y, x in dl:
                progress.advance(task)
        
        end = time.time()
        print("Loaded with batch size: ", batch_size)
        print("Time per step: ", (end - start) / 100, " seconds")
        print("Time per voxel:", ((end - start) / 100) / batch_size, "seconds")

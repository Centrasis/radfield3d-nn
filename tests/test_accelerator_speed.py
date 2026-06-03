from pathlib import Path
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn, MofNCompleteColumn
import time
import os
from RadFiled3D.RadFiled3D import FieldStore, CartesianFieldAccessor


if __name__ == "__main__":
    base_path = Path("C:/Users/lehner04/Documents/Datasets") / "H-100-Alderson-2_5m" / "fields"
    files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".rf3")]

    PROGRESSBAR = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
        )

    print("Testing speed for whole fields...")
    with PROGRESSBAR as progress:
        task = progress.add_task("Loading fields with out accessor...", total=len(files))
        start_time_no_accessor = time.time()
        for file in files:
            field = FieldStore.load(file)
            progress.advance(task)
        stop_time_no_accessor = time.time()

        task = progress.add_task("Loading fields with accessor...", total=len(files))
        accessor = FieldStore.construct_field_accessor(files[0])
        start_time_accessor = time.time()
        for file in files:
            field = accessor.access_field(file)
            progress.advance(task)
        stop_time_accessor = time.time()

    avg_load_time_no_accessor_field = (stop_time_no_accessor - start_time_no_accessor) / len(files)
    avg_load_time_accessor_field = (stop_time_accessor - start_time_accessor) / len(files)

    with PROGRESSBAR as progress:
        #task = progress.add_task("Loading layer with out accessor...", total=len(files))
        #start_time_no_accessor = time.time()
        #for file in files:
        #    layer = FieldStore.load_single_grid_layer(file, "scatter_field", "spectrum")
        #    progress.advance(task)
        #stop_time_no_accessor = time.time()
        
        task = progress.add_task("Loading layer with accessor...", total=len(files))
        accessor: CartesianFieldAccessor = FieldStore.construct_field_accessor(files[0])
        start_time_accessor = time.time()
        for file in files:
            layer = accessor.access_layer(file, "scatter_field", "spectrum")
            progress.advance(task)
        stop_time_accessor = time.time()

    avg_load_time_no_accessor_layer = avg_load_time_no_accessor_field # (stop_time_no_accessor - start_time_no_accessor) / len(files)
    avg_load_time_accessor_layer = (stop_time_accessor - start_time_accessor) / len(files)

    with PROGRESSBAR as progress:
        #task = progress.add_task("Loading voxels with out accessor...", total=len(files))
        #start_time_no_accessor = time.time()
        #for file in files:
        #    layer = FieldStore.load_single_grid_layer(file, "scatter_field", "spectrum")
        #    voxel = layer.get_voxel(0, 0, 0)
        #    progress.advance(task)
        #stop_time_no_accessor = time.time()

        task = progress.add_task("Loading voxels with accessor...", total=len(files))
        accessor = FieldStore.construct_field_accessor(files[0])
        start_time_accessor = time.time()
        for file in files:
            voxel = accessor.access_voxel_flat(file, "scatter_field", "spectrum", 0)
            progress.advance(task)
        stop_time_accessor = time.time()
    
    avg_load_time_no_accessor_voxel = avg_load_time_no_accessor_field # (stop_time_no_accessor - start_time_no_accessor) / len(files)
    avg_load_time_accessor_voxel = (stop_time_accessor - start_time_accessor) / len(files)
    
    print(f"Average load time for whole fields without accessor:\t{avg_load_time_no_accessor_field} s")
    print(f"Average load time for whole fields with accessor:\t{avg_load_time_accessor_field} s")
    print(f"Average load time for layers without accessor:\t\t{avg_load_time_no_accessor_layer} s")
    print(f"Average load time for layers with accessor:\t\t{avg_load_time_accessor_layer} s")
    print(f"Average load time for voxels without accessor:\t\t{avg_load_time_no_accessor_voxel} s")
    print(f"Average load time for voxels with accessor:\t\t{avg_load_time_accessor_voxel} s")

    print("\n")
    print(f"Speedup for whole fields:\t{(avg_load_time_no_accessor_field / avg_load_time_accessor_field - 1) * 100}%")
    print(f"Speedup for layers:\t\t{(avg_load_time_no_accessor_layer / avg_load_time_accessor_layer - 1) * 100}%")
    print(f"Speedup for voxels:\t\t{(avg_load_time_no_accessor_voxel / avg_load_time_accessor_voxel - 1) * 100}%")

from RadFiled3D.RadFiled3D import FieldStore
from pathlib import Path
import time
from rich import print
from rich.progress import track
import os
import psutil
import gc


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes


if __name__ == "__main__":
    path = str(Path(os.getcwd()) / "RF_100.0keV_0_0_Cylinder.rf3")
    durations_layer_load = []
    begin_memory_usage = get_memory_usage() / (1024 * 1024)
    start = time.time()
    
    for _ in track(range(500)):
        durations_layer_load.append(time.time() - start)
        with open(path, "rb") as file:
            buffer = file.read()
            field = FieldStore.load_single_grid_layer_from_buffer(buffer, "scatter_field", "spectrum")
        start = time.time()

    print(f"Average time to load a layer: {sum(durations_layer_load) / len(durations_layer_load)}")
    print(f"Memory usage before loading: {begin_memory_usage} MB")
    print(f"Memory usage before GC: {(get_memory_usage() / (1024 * 1024) - begin_memory_usage)} MB")
    gc.collect()
    print(f"Memory usage after GC: {(get_memory_usage() / (1024 * 1024) - begin_memory_usage)} MB")

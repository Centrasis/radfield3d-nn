from RadFiled3D.pytorch import CartesianFieldSingleLayerDataset, DatasetBuilder
from pathlib import Path
import time
from rich import print
from rich.progress import track
from RadFiled3D.pytorch.helpers import RadiationFieldHelper
import h5py  # Add HDF5 support
import pickle


def save_to_hdf5(dataset, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for i, (field, metadata) in enumerate(track(dataset)):
            tensor = RadiationFieldHelper.load_tensor_from_layer(field)
            hdf5_file.create_dataset(f'field_{i}', data=tensor.numpy())
            hdf5_file.create_dataset(f'metadata_{i}', data=pickle.dumps(metadata))

def load_voxel(hdf5_path, field_index, x, y, z):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        dataset = hdf5_file[f'field_{field_index}']
        return dataset[x, y, z]
    
def load_metadata(hdf5_path, field_index):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        metadata = pickle.loads(hdf5_file[f'metadata_{field_index}'])
        return metadata

if __name__ == "__main__":
    path = str(Path("C:/Users/lehner04/Documents/Datasets") / "Cylinder-Multi-Spectra.zip")

    dataset_builder = DatasetBuilder(path, dataset_class=CartesianFieldSingleLayerDataset)
    dataset: CartesianFieldSingleLayerDataset = dataset_builder.build_test_dataset()
    dataset.set_channel_and_layer("scatter_field", "spectrum")
    hdf5_path = "C:/Users/lehner04/Documents/Datasets/dataset.h5"
    save_to_hdf5(dataset, hdf5_path)  # Save dataset to HDF5

    # Example of loading a voxel
    voxel = load_voxel(hdf5_path, field_index=0, x=10, y=20, z=30)
    print(voxel)
    metadata = load_metadata(hdf5_path, field_index=0)
    print(metadata)
